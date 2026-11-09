"""One isolated GPU worker with bounded jobs and verified immutable publication.

Injected runners accept (config, text, frames, run_path, stop_event, progress)
and return a process exit code. They must finish before another runner starts.
The run path must not exist on entry, just as required by the authenticated CLI.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import unicodedata
import uuid

from .registry import MotionRegistry
from .signspark_generation import MODEL_FILES, MODEL_REVISION, SOURCE_COMMIT


class GenerationBusyError(RuntimeError):
    """The bounded GPU queue is full."""


class GenerationUnavailableError(RuntimeError):
    """The configured local runtime is unavailable."""


_TERMINAL = frozenset(('complete', 'failed', 'cancelled'))
_PHASES = frozenset(('preparing', 'body', 'hand', 'face', 'packaging'))
_PATH_KEYS = ('python', 'worker_script', 'source', 'text_assets', 'models',
              'base_archive', 'source_presentation', 'smplx_model')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def _phase(run):
    try:
        receipt = json.loads((run/'generation-manifest.json').read_text(encoding='utf-8'))
        if receipt.get('state') == 'complete' or all(s in receipt.get('streams', {}) for s in ('body','hand','face')):
            return 'packaging'
        stream = receipt.get('active_step', {}).get('stream')
        return stream if stream in ('body', 'hand', 'face') else 'preparing'
    except (OSError, ValueError, AttributeError):
        return 'preparing'


def _kill_tree(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       shell=False, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def cli_runner(config, text, frames, run, stop, progress):
    """Launch only server-owned paths; user text is one unambiguous argv value."""
    argv = [str(config['python']), str(config['worker_script']), '--text='+text,
            '--frames', str(frames), '--steps', '50', '--seed', '123', '--output', str(run)]
    for key in _PATH_KEYS[2:]:
        argv.extend(('--'+key.replace('_','-'), str(config[key])))
    if config.get('source_manifest'):
        argv.extend(('--source-manifest', str(config['source_manifest'])))
    if config.get('readonly_checkpoint_mmap'):
        argv.append('--readonly-checkpoint-mmap')
    if config.get('cpu_offload'):
        argv.append('--cpu-offload')
    options = dict(shell=False, stdin=subprocess.DEVNULL)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    else:
        options['start_new_session'] = True
    with (run.parent/'worker.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            while process.poll() is None:
                progress(_phase(run))
                if stop.wait(.25):
                    _kill_tree(process)
                    break
            progress(_phase(run))
            return process.wait()
        finally:
            if process.poll() is None:
                _kill_tree(process)


class GenerationService:
    """Thread-safe serial scheduling; no corpus matching or target motion input."""

    def __init__(self, config, runner=None):
        self._config = dict(config)
        self._runner = runner or cli_runner
        self._injected = runner is not None
        self._jobs = OrderedDict()
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._thread = threading.Thread(target=self._work, name='sign-generation', daemon=True)
        self._thread.start()

    def _available(self):
        if self._closed or not self._config.get('output_root'):
            raise GenerationUnavailableError('本地生成服务尚未就绪。')
        if not self._injected and any(not self._config.get(k) or not Path(self._config[k]).exists() for k in _PATH_KEYS):
            raise GenerationUnavailableError('本地生成运行环境或资产尚未就绪。')

    def _public(self, job):
        return deepcopy(job['public'])

    def _update(self, job, **values):
        job['public'].update(values, updated_at=_now())
        _json_write(job['directory']/'job-state.json', job['public'])

    def submit(self, text, seconds=6):
        if (not isinstance(text, str) or not 1 <= len(text) <= 200
                or type(seconds) is not int or seconds not in (4,6,8,10)):
            raise ValueError('请输入1–200字并选择4、6、8或10秒。')
        normalized = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').strip()
        if not normalized or any(unicodedata.category(c) in ('Cc','Cs') for c in normalized):
            raise ValueError('输入包含无效控制字符。')
        with self._condition:
            self._available()
            if sum(j['public']['state'] not in _TERMINAL for j in self._jobs.values()) >= 8:
                raise GenerationBusyError('生成队列已满，请稍后重试。')
            # Evict only settled jobs. A cancelled runner can still be unwinding.
            while len(self._jobs) >= 32:
                old = next((key for key,j in self._jobs.items()
                            if j['public']['state'] in _TERMINAL and not j['active']), None)
                if old is None:
                    raise GenerationBusyError('生成队列已满，请稍后重试。')
                del self._jobs[old]
            identity = uuid.uuid4().hex
            directory = Path(self._config['output_root']).resolve()/identity
            try:
                directory.mkdir(parents=True, exist_ok=False)
                now = _now()
                public = dict(job_id=identity, text=text, generation_text=normalized, seconds=seconds,
                              frames=seconds*30, state='queued', phase='preparing', created_at=now,
                              updated_at=now, language_review='NOT_RUN', quality_warning=None,
                              motion_url=None, error=None)
                job = dict(public=public, directory=directory, stop=threading.Event(),
                           registry=None, clip_id=None, active=False)
                _json_write(directory/'job-state.json', public)
            except OSError as exc:
                raise GenerationUnavailableError('生成输出目录暂时不可用。') from exc
            self._jobs[identity] = job
            self._condition.notify()
            return self._public(job)

    def get(self, job_id):
        with self._condition:
            return self._public(self._jobs[job_id])

    def cancel(self, job_id):
        with self._condition:
            job = self._jobs[job_id]
            if job['public']['state'] not in _TERMINAL:
                job['stop'].set()
                self._update(job, state='cancelled', motion_url=None)
            return self._public(job)

    def motion(self, job_id):
        with self._condition:
            job = self._jobs[job_id]
            if job['public']['state'] != 'complete':
                raise ValueError('生成结果尚不可播放。')
            registry, identity = job['registry'], job['clip_id']
        motion = registry.motion(identity)
        motion['sha256'] = next(a['sha256'] for a in registry.catalog()['assets'] if a['id'] == identity)
        motion['archive_sha256'] = registry.sha256
        return motion

    def _validate(self, job, run):
        public = job['public']
        request = dict(text=public['generation_text'], frames=public['frames'], steps=50, seed=123)
        receipt = json.loads((run/'generation-manifest.json').read_text(encoding='utf-8'))
        requests = receipt.get('requests')
        if (receipt.get('state') != 'complete' or receipt.get('requests') != [request]
                or not isinstance(requests, list) or len(requests) != 1
                or any(type(requests[0].get(k)) is not int for k in ('frames','steps','seed'))
                or 'target_motion' not in receipt or receipt['target_motion'] is not None
                or type(receipt.get('observed_keyframes')) is not int or receipt['observed_keyframes'] != 0):
            raise ValueError('Generation receipt identity differs')
        digest = receipt.get('candidate_archive_sha256')
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('Missing archive digest')
        registry = MotionRegistry(run/'candidate-motion.zip', digest)
        source = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        identity = 'generated-csl-'+source[:16]
        asset = next(a for a in registry.catalog()['assets'] if a['id'] == identity)
        clip = registry.motion(identity)['clip']
        if (asset['group'] != 'generated' or clip['text'] != request['text']
                or clip['frames'] != request['frames'] or clip['fps'] != 30
                or clip.get('source_kind') != 'pretrained_text_conditioned_generation'
                or clip.get('source_sha256') != source
                or clip.get('source_model_sha256') != {s:v[1] for s,v in MODEL_FILES.items()}
                or clip.get('model_revision') != MODEL_REVISION
                or clip.get('generation_source_commit') != SOURCE_COMMIT
                or 'target_motion' not in clip or clip['target_motion'] is not None
                or type(clip.get('observed_keyframes')) is not int or clip['observed_keyframes'] != 0
                or type(clip.get('explicit_output_frames')) is not int
                or clip['explicit_output_frames'] != request['frames']):
            raise ValueError('Generated clip provenance differs')
        return registry, identity, bool(asset.get('quality_warning') or clip.get('quality_warning'))

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or any(
                    j['public']['state'] == 'queued' for j in self._jobs.values()))
                if self._closed:
                    return
                job = next(j for j in self._jobs.values() if j['public']['state'] == 'queued')
                job['active'] = True
            try:
                with self._condition:
                    if job['stop'].is_set():
                        continue
                    self._update(job, state='running')
                def progress(phase):
                    if phase in _PHASES:
                        with self._condition:
                            if job['public']['state'] == 'running' and phase != job['public']['phase']:
                                self._update(job, phase=phase)
                run = job['directory']/'run'
                code = self._runner(dict(self._config), job['public']['generation_text'],
                                    job['public']['frames'], run, job['stop'], progress)
                if code != 0:
                    raise ValueError('Worker exited unsuccessfully')
                if job['stop'].is_set():
                    continue
                registry, identity, warning = self._validate(job, run)
                with self._condition:
                    if not job['stop'].is_set() and job['public']['state'] == 'running':
                        job['registry'], job['clip_id'] = registry, identity
                        self._update(job, state='complete', phase='packaging', quality_warning=warning,
                                     motion_url='/v1/animation/generation/jobs/'+job['public']['job_id']+'/motion')
            except Exception as exc:
                try:
                    with (job['directory']/'service-error.log').open('a', encoding='utf-8') as log:
                        log.write(type(exc).__name__+': '+str(exc)+'\n')
                except OSError:
                    pass
                with self._condition:
                    if not job['stop'].is_set():
                        # Never expose worker paths, text in tracebacks, or subprocess argv.
                        message='生成失败，本次没有可播放的结果，请重试。'
                        try:
                            failed=json.loads((job['directory']/'run/generation-manifest.json').read_text(encoding='utf-8'))
                            reason=failed.get('error','')
                            if any(term in reason for term in ('MemoryError','OutOfMemory','out of memory','Cannot allocate memory')):
                                message='系统可用内存额度不足，本次生成未完成。请释放内存后重试。'
                        except (OSError,ValueError,AttributeError,TypeError):
                            pass
                        job['public'].update(state='failed', error=message, motion_url=None)
                        try:
                            self._update(job)
                        except OSError:
                            pass
            finally:
                with self._condition:
                    job['active'] = False
                    self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job['public']['state'] not in _TERMINAL:
                    job['stop'].set()
                    try:
                        self._update(job, state='cancelled', motion_url=None)
                    except OSError:
                        job['public']['state'] = 'cancelled'
            self._condition.notify_all()
        self._thread.join(timeout=10)
