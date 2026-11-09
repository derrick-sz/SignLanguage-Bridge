"""Bounded, persistent JSONL transport for the authenticated local T2G model.

This module never imports Torch. A separate, server-owned runtime holds the
model, while requests are serialized and stale/broken workers fail closed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import unicodedata
import uuid

PROTOCOL = 'signbridge-t2g-jsonl-v1'
MODEL_REVISION = 'a883bc2d13a5d7f85e64591b32bae4cb48950b97'
MAX_INPUT_CHARACTERS = 200
MAX_GLOSSES = 32
NUM_BEAMS = 4
MAX_NEW_TOKENS = 150
MAX_PROTOCOL_BYTES = 32768
ASSET_FILES = {
    'model.safetensors': (1444104872, '93ef6db7b706a6d0f6906fbd1bcfad8b46878918b8cb21fc600193f3630aa2be'),
    'config.json': (1333, '617f7eb0b3a92a6234e24ca8882ca72825d3f0f1734ec2e487a983d1bae42081'),
    'generation_config.json': (200, 'b8335439acd504970f49c9fbb48670c218e227f90f593103fe3aa3fd3d052a8f'),
    'sentencepiece.bpe.model': (324576, '338a6df5c468c6ab568a441bc8a38684c79e8ccad080720bd1490a24fcb53881'),
    'signspark_t2g.json': (104, 'e6396fe2956fd2089e3c6e8e2f5db3274d309d207dea389247769c44d828deff'),
    'special_tokens_map.json': (641, '5b5c1bd328a83c2d2364b2f0615d003677468b206cd5a6566dd6da5e51bd76c2'),
    'tokenizer_config.json': (5853, 'dc1d1f8195dca6734a5907d04edfa72b1cbab667b1fde0a2ea68918b5d44aede'),
}
_UNAUTHENTICATED_INPUTS = ('added_tokens.json', 'tokenizer.json', 'adapter_config.json',
                           'model.safetensors.index.json', 'pytorch_model.bin',
                           'pytorch_model.bin.index.json', 'adapter_model.safetensors')


class PlannerUnavailableError(RuntimeError):
    """The authenticated planner is not ready, has exited or is invalid."""


class PlannerBusyError(RuntimeError):
    """A request is already using this single persistent model."""


def normalize_text(text):
    if not isinstance(text, str) or not 1 <= len(text) <= MAX_INPUT_CHARACTERS:
        raise ValueError('请输入1–200字。')
    normalized = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').strip()
    if not normalized or any(unicodedata.category(c) in ('Cc', 'Cs') for c in normalized):
        raise ValueError('输入包含无效控制字符。')
    return normalized


def validate_glosses(values):
    if (not isinstance(values, list) or not 1 <= len(values) <= MAX_GLOSSES
            or any(not isinstance(value, str) or not 1 <= len(value) <= MAX_INPUT_CHARACTERS
                   or value != value.strip() or any(c.isspace() or unicodedata.category(c) in ('Cc', 'Cs') for c in value)
                   for value in values)):
        raise ValueError('文字规划结果为空、超长或格式无效。')
    return list(values)


def planner_identity(device):
    size, digest = ASSET_FILES['model.safetensors']
    return {'model_revision': MODEL_REVISION, 'model_sha256': digest, 'model_bytes': size,
            'dataset': 'CSL-Daily', 'language': 'zh_CN', 'device': device,
            'prompt_version': 'signspark-retrieval-chinese-v1', 'num_beams': NUM_BEAMS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'warmup_profile': 'fixed-hello-v1',
            'assets_sha256': {name: value[1] for name, value in ASSET_FILES.items()}}


def asset_fingerprint(model_dir):
    return {name: (info.st_size, info.st_mtime_ns, info.st_ino)
            for name in ASSET_FILES for info in [(Path(model_dir) / name).stat()]}


def verify_assets(model_dir):
    """Read-only SHA/length verification; return metadata for change detection."""
    if any((Path(model_dir) / name).exists() for name in _UNAUTHENTICATED_INPUTS):
        raise ValueError('Unexpected T2G asset identity input')
    before = asset_fingerprint(model_dir)
    for name, (size, expected) in ASSET_FILES.items():
        if before[name][0] != size:
            raise ValueError('T2G asset identity mismatch')
        digest = hashlib.sha256()
        with (Path(model_dir) / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('T2G asset identity mismatch')
    if asset_fingerprint(model_dir) != before:
        raise ValueError('T2G asset identity changed during verification')
    return before


def _timeout(value):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 < value <= 3600):
        raise ValueError('Planner timeouts must be finite and positive')
    return float(value)


class TextGlossPlanner:
    """Warm the owned worker asynchronously and reject concurrent requests.

    Config: python, script, model_dir, device ('cpu', 'cuda' or 'cuda:N').
    Optional startup_timeout (120s), request_timeout (30s), stderr_log.
    The release identity and beam/token bounds are fixed, not client inputs.
    """
    def __init__(self, config):
        self._config = dict(config)
        self._device = self._config.get('device', 'cpu')
        if not isinstance(self._device, str) or not re.fullmatch(r'cpu|cuda(?::\d+)?', self._device):
            raise ValueError('Unsupported planner device')
        self._startup_timeout = _timeout(self._config.get('startup_timeout', 120))
        self._request_timeout = _timeout(self._config.get('request_timeout', 30))
        self._identity = planner_identity(self._device)
        self._state = 'starting'
        self._error = None
        self._closed = False
        self._process = None
        self._stderr = None
        self._fingerprint = None
        self._startup_metrics = None
        self._state_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._messages = queue.Queue(maxsize=4)
        self._starter = threading.Thread(target=self._start, daemon=True, name='text-gloss-start')
        self._starter.start()

    def _stop_process(self):
        with self._stop_lock:
            process = self._process
            if process is not None and process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            if process is not None:
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError:
                            pass

    def _fail(self, code):
        with self._state_lock:
            self._state = 'failed'
            self._error = 'closed' if self._closed else code
        self._stop_process()

    def _push(self, message):
        try:
            self._messages.put_nowait(message)
        except queue.Full:
            self._fail('protocol_error')

    def _read_stdout(self, process):
        try:
            while True:
                line = process.stdout.readline(MAX_PROTOCOL_BYTES + 1)
                if not line:
                    self._push(None)
                    return
                if len(line) > MAX_PROTOCOL_BYTES or not line.endswith(b'\n'):
                    self._push(None)
                    return
                try:
                    message = json.loads(line.decode('utf-8'))
                except (UnicodeError, ValueError):
                    self._push(None)
                    return
                self._push(message)
        except (OSError, ValueError):
            self._push(None)

    def _start(self):
        try:
            self._fingerprint = verify_assets(self._config['model_dir'])
            python = Path(self._config['python'])
            script = Path(self._config['script'])
            if not python.is_file() or not script.is_file():
                raise ValueError('Worker files missing')
            options = {'shell': False, 'stdin': subprocess.PIPE, 'stdout': subprocess.PIPE,
                       'stderr': subprocess.DEVNULL,
                       'cwd': str(Path(__file__).resolve().parents[2])}
            if self._config.get('stderr_log'):
                log = Path(self._config['stderr_log'])
                log.parent.mkdir(parents=True, exist_ok=True)
                self._stderr = log.open('ab', buffering=0)
                options['stderr'] = self._stderr
            if os.name == 'nt':
                options['creationflags'] = subprocess.CREATE_NO_WINDOW
            else:
                options['start_new_session'] = True
            environment = os.environ.copy()
            environment.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1',
                               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
            argv = [str(python), '-u', str(script), '--model-dir', str(self._config['model_dir']),
                    '--device', self._device, '--identity-json',
                    json.dumps(self._identity, ensure_ascii=False, separators=(',', ':'))]
            with self._state_lock:
                if self._closed:
                    raise ValueError('Planner closed during startup')
                self._process = subprocess.Popen(argv, env=environment, **options)
            threading.Thread(target=self._read_stdout, args=(self._process,), daemon=True,
                             name='text-gloss-stdout').start()
            message = self._messages.get(timeout=self._startup_timeout)
            if (not isinstance(message, dict) or message.get('type') != 'ready'
                    or message.get('protocol') != PROTOCOL
                    or message.get('model_identity') != self._identity):
                raise ValueError('Worker startup protocol invalid')
            metrics = message.get('startup_metrics')
            if (not isinstance(metrics, dict) or set(metrics) != {'load_ms', 'warmup_ms'}
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in metrics.values())):
                raise ValueError('Worker startup timing invalid')
            with self._state_lock:
                if self._closed:
                    return
                if self._process.poll() is not None or self._state == 'failed':
                    raise ValueError('Worker exited during startup')
                self._startup_metrics = metrics
                self._state = 'ready'
        except Exception:
            self._fail('startup_failed')

    def status(self):
        process = self._process
        if self._state == 'ready' and process is not None and process.poll() is not None:
            self._fail('worker_exited')
        with self._state_lock:
            return {'state': self._state, 'model_identity': deepcopy(self._identity) if self._state == 'ready' else None,
                    'error': self._error, 'startup_metrics': deepcopy(self._startup_metrics)}

    def plan(self, text):
        normalized = normalize_text(text)
        if not self._request_lock.acquire(blocking=False):
            raise PlannerBusyError('文字规划器正忙，请稍后重试。')
        try:
            if self.status()['state'] != 'ready':
                raise PlannerUnavailableError('文字规划环境暂时不可用。')
            try:
                if asset_fingerprint(self._config['model_dir']) != self._fingerprint:
                    raise ValueError('Worker asset files changed')
                identity = uuid.uuid4().hex
                request = json.dumps({'id': identity, 'text': normalized}, ensure_ascii=False).encode('utf-8') + b'\n'
                self._process.stdin.write(request)
                self._process.stdin.flush()
                message = self._messages.get(timeout=self._request_timeout)
                if not isinstance(message, dict) or message.get('id') != identity:
                    raise ValueError('Worker reply identity invalid')
                if message.get('type') == 'error':
                    if message.get('error') in {'invalid_input', 'input_limit', 'output_limit', 'empty_output', 'unknown_token'}:
                        raise _RequestRejected()
                    raise ValueError('Worker inference failed')
                if message.get('type') != 'result' or message.get('model_identity') != self._identity:
                    raise ValueError('Worker reply model identity invalid')
                glosses = validate_glosses(message.get('glosses'))
                elapsed = message.get('elapsed_ms')
                if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('Worker timing invalid')
                if self.status()['state'] != 'ready':
                    raise ValueError('Worker no longer available')
                return {'glosses': glosses, 'model_identity': deepcopy(self._identity), 'elapsed_ms': elapsed}
            except _RequestRejected:
                raise ValueError('当前文字无法生成完整且有界的手语词序，请缩短或调整输入。') from None
            except Exception:
                self._fail('request_failed')
                raise PlannerUnavailableError('文字规划环境暂时不可用。') from None
        finally:
            self._request_lock.release()

    def close(self):
        with self._state_lock:
            self._closed = True
        self._fail('closed')


class _RequestRejected(Exception):
    pass
