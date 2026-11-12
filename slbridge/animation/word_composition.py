"""Source-preserving word retrieval; no full-sentence lookup or nearest-word fallback."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
import unicodedata
import zipfile

import numpy as np

from .gesture import _quaternions, _rotvec, _slerp
from .registry import _skeleton
from .word_library_build import SOURCE_IDENTITIES, is_complete_expression

PROFILE = 'retrieved-words-preserved-slerp6-v1'
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
MAX_WORDS, MAX_FRAMES, TRANSITION_FRAMES = 32, 1800, 6
MAX_CANDIDATES = 8
# A narrow presence guard, not a semantic translation score. Only literal
# place/time terms with an exact library entry are checked; paraphrases can
# therefore require review. Other omissions remain a human-review limitation.
_ANCHORS = ('医院', '学校', '银行', '机场', '车站', '公园', '超市', '公司',
            '厕所', '饭店', '图书馆', '北京', '上海', '成都', '今天', '明天', '昨天', '后天')


class WordLibraryError(RuntimeError):
    """An authenticated source is missing, changed or malformed."""


class CompositionUnavailableError(RuntimeError):
    pass


class CompositionBusyError(RuntimeError):
    pass


def _digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return _digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode('utf-8'))


def _source_release(source):
    pair = (source.get('lmdb_sha256'), source.get('annotations_sha256'))
    for split, expected in SOURCE_IDENTITIES.items():
        if pair[0] == expected[0] or pair[1] == expected[1]:
            if pair != expected or source.get('split') != split:
                raise ValueError('Official word data cannot be assigned to a different split')


def normalize_text(text):
    if not isinstance(text, str) or not 1 <= len(text) <= 200:
        raise ValueError('Expected 1–200 characters')
    normalized = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').strip()
    if not normalized or any(unicodedata.category(c) in ('Cc', 'Cf', 'Cs') for c in normalized):
        raise ValueError('Unsupported empty text or control character')
    return normalized


class WordLibrary:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.manifest_path = self.root / 'manifest.json'
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        try:
            raw = self.manifest_path.read_bytes()
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError('Word index too large')
            manifest = json.loads(raw)
            if (manifest.get('schema') != 'signbridge-word-library-v1'
                    or not _ID.fullmatch(manifest.get('bundle_id', ''))
                    or not isinstance(manifest.get('source'), dict)
                    or not isinstance(manifest.get('units'), list)
                    or not 1 <= len(manifest['units']) <= 20000):
                raise ValueError('Invalid word index')
            self._manifest_sha = _digest_bytes(raw)
            self.bundle_id = manifest['bundle_id']
            self.source = deepcopy(manifest['source'])
            self.is_standalone = manifest.get('standalone_expressions', False)
            if (type(self.is_standalone) is not bool or self.is_standalone !=
                    (self.source.get('selection_scope') == 'whole-single-gloss-utterance-v1')):
                raise ValueError('Invalid standalone expression scope')
            self.usage_role = manifest.get('usage_role', 'train-only-library')
            self.evaluation_eligible = manifest.get('evaluation_eligible')
            origins = self.source.get('sources')
            self._origins = {}
            if origins is not None:
                if (not isinstance(origins, list) or not 1 <= len(origins) <= 16
                        or self.usage_role != 'product-demo' or self.evaluation_eligible is not False
                        or self.source.get('usage_role') != 'product-demo'
                        or self.source.get('evaluation_eligible') is not False):
                    raise ValueError('Mixed-source word library must be product-only')
                for origin in origins:
                    key = origin.get('library_bundle_id')
                    if (not isinstance(key, str) or not _ID.fullmatch(key) or key in self._origins
                            or origin.get('split') not in ('train','dev','test')
                            or any(not _SHA.fullmatch(origin.get(k,'')) for k in
                                   ('manifest_sha256','lmdb_sha256','annotations_sha256'))):
                        raise ValueError('Invalid word source binding')
                    _source_release(origin)
                    self._origins[key] = origin
            elif self.source.get('split') != 'train':
                raise ValueError('Non-train assets require explicit product source bindings')
            else:
                _source_release(self.source)
            self.parents, self.rest = _skeleton(manifest['skeleton'])
            self.parents, self.rest = deepcopy(self.parents), deepcopy(self.rest)
            self._words = {}
            ids = set()
            for unit in manifest['units']:
                label, identity = unit.get('label'), unit.get('id')
                frames, source = unit.get('frames'), unit.get('source')
                if (not isinstance(label, str) or not label.strip() or len(label) > 100
                        or not isinstance(identity, str) or not _ID.fullmatch(identity) or identity in ids
                        or type(frames) is not int or not 1 <= frames <= 3000
                        or unit.get('fps') != 30 or not _SHA.fullmatch(unit.get('sha256', ''))
                        or not isinstance(source, dict) or source.get('split') not in ('train','dev','test')
                        or not isinstance(source.get('video_file'), str) or not source['video_file']
                        or type(source.get('start')) is not int or type(source.get('end')) is not int
                        or source['start'] < 0 or source['end'] - source['start'] != frames):
                    raise ValueError('Invalid word source metadata')
                if self._origins:
                    origin = self._origins.get(source.get('library_bundle_id'))
                    if (origin is None or source.get('split') != origin['split']
                            or source.get('library_manifest_sha256') != origin['manifest_sha256']
                            or any(source.get(k) != origin[k] for k in ('lmdb_sha256','annotations_sha256'))):
                        raise ValueError('Word unit source identity differs from its library')
                elif source.get('split') != 'train' or 'library_bundle_id' in source:
                    raise ValueError('Unbound non-train word source')
                complete = is_complete_expression(unit)
                if self.is_standalone and not complete:
                    raise ValueError('Standalone library contains a context fragment')
                self._path(unit['path'])
                ids.add(identity)
                choices = self._words.setdefault(label, [])
                if len(choices) >= MAX_CANDIDATES:
                    raise ValueError('More than eight candidates per word')
                choices.append(deepcopy(unit))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise WordLibraryError('Word library is unavailable or invalid') from exc

    def _path(self, name):
        if not isinstance(name, str) or not name or '\\' in name or ':' in name:
            raise ValueError('Unsafe word path')
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.npz':
            raise ValueError('Unsafe word path')
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Word path escaped library')
        return path

    def _verify_index(self):
        try:
            if _digest_bytes(self.manifest_path.read_bytes()) != self._manifest_sha:
                raise ValueError('Index changed')
        except (OSError, ValueError) as exc:
            raise WordLibraryError('Word index changed; reload the library') from exc

    def _load(self, unit):
        try:
            path = self._path(unit['path'])
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError('Word asset too large')
            raw = path.read_bytes()
            if _digest_bytes(raw) != unit['sha256']:
                raise ValueError('Word asset changed')
            # Verify current bytes even on cache hits: mtime is not identity.
            if unit['id'] in self._cache:
                self._cache.move_to_end(unit['id'])
                return self._cache[unit['id']]
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 8 * 1024 * 1024:
                    raise ValueError('Expanded word asset too large')
            with np.load(BytesIO(raw), allow_pickle=False) as saved:
                if set(saved.files) != {'poses', 'expression'}:
                    raise ValueError('Unexpected word fields')
                poses, expression = saved['poses'], saved['expression']
            if (poses.shape != (unit['frames'], 55, 3) or poses.dtype != np.float32
                    or expression.shape != (unit['frames'], 50) or expression.dtype != np.float32
                    or not np.isfinite(poses).all() or not np.isfinite(expression).all()):
                raise ValueError('Invalid word parameters')
            poses.setflags(write=False)
            self._cache[unit['id']] = poses
            while len(self._cache) > 64:
                self._cache.popitem(last=False)
            return poses
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise WordLibraryError('A word motion is unavailable or changed') from exc

    def status(self):
        return {'bundle_id': self.bundle_id, 'word_count': len(self._words),
                'unit_count': sum(map(len, self._words.values())), 'language_review': 'NOT_RUN',
                'usage_role': self.usage_role, 'evaluation_eligible': self.evaluation_eligible,
                'max_candidates': MAX_CANDIDATES}

    def has_word(self, label):
        return label in self._words

    def compose(self, text, glosses):
        text = normalize_text(text)
        if (not isinstance(glosses, list) or not 1 <= len(glosses) <= MAX_WORDS
                or any(not isinstance(g, str) or not g.strip() or len(g) > 100 for g in glosses)):
            raise ValueError('Expected 1–32 explicit glosses')
        with self._lock:
            self._verify_index()
            result = {'input_text': text, 'generation_text': text, 'glosses': list(glosses),
                      'language_review': 'NOT_RUN', 'bundle_id': self.bundle_id}
            missing = list(dict.fromkeys(g for g in glosses if g not in self._words))
            if missing:
                return {**result, 'status': 'blocked', 'missing_glosses': missing}
            pieces, items, transitions = [], [], []
            count = 0
            for gloss in glosses:
                candidates = [(unit, self._load(unit)) for unit in self._words[gloss]]
                complete = [pair for pair in candidates if pair[0].get('complete_expression')]
                if complete:
                    candidates = complete
                if pieces:
                    left = _quaternions(pieces[-1][-1])
                    def score(pair):
                        dot = np.sum(left * _quaternions(pair[1][0]), axis=-1)
                        return (float(np.mean(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))), pair[0]['id'])
                    unit, poses = min(candidates, key=score)
                else:
                    unit, poses = candidates[0]
                proposed = count + len(poses) + (TRANSITION_FRAMES if pieces else 0)
                if proposed > MAX_FRAMES:
                    raise ValueError('Combined motion exceeds 1800 frames')
                if pieces:
                    right = _quaternions(poses[0])
                    bridge = []
                    for index in range(1, TRANSITION_FRAMES + 1):
                        t = index / (TRANSITION_FRAMES + 1)
                        bridge.append(_rotvec(_slerp(left, right, t * t * (3 - 2 * t))))
                    pieces.append(np.asarray(bridge))
                    transitions.append({'output_frame_range': [count, count + TRANSITION_FRAMES],
                                        'frames': TRANSITION_FRAMES, 'method': 'shortest-arc-slerp-smoothstep'})
                    count += TRANSITION_FRAMES
                items.append({'unit_id': unit['id'], 'gloss': gloss, 'sha256': unit['sha256'],
                              'source': deepcopy(unit['source']),
                              'source_frame_range': [unit['source']['start'], unit['source']['end']],
                              'output_frame_range': [count, count + len(poses)]})
                if unit.get('complete_expression'):
                    items[-1]['selection_kind'] = 'whole_native_expression'
                count += len(poses)
                pieces.append(poses)
            self._verify_index()
            poses = np.concatenate(pieces)
            composition = {'items': items, 'transitions': transitions, 'bundle_id': self.bundle_id,
                           'source': deepcopy(self.source), 'source_frames_preserved': True,
                           'frame_range_convention': 'zero-based half-open', 'processing_version': PROFILE}
            clip = {'id': 'words-' + _digest([text, glosses, items, self.bundle_id, PROFILE])[:24],
                    'text': text, 'aliases': [], 'frames': count, 'fps': 30.0, 'poses': poses.tolist(),
                    'source_kind': 'retrieved_word_composition', 'composition': composition,
                    'experimental': True, 'human_approved': False, 'language_review': 'NOT_RUN',
                    'motion_review': 'NOT_RUN', 'composition_review': 'NOT_RUN', 'training_ready': False,
                    'unmapped_target_channels': ['expression', 'jaw', 'spatial_inflection'],
                    'quality_warning': True}
            motion = {'clip': clip, 'parents': deepcopy(self.parents), 'restJoints': deepcopy(self.rest),
                      'sha256': _digest(clip), 'processing_version': PROFILE}
            return {**result, 'status': 'ready', 'missing_glosses': [], 'motion': motion}


class CompositionService:
    def __init__(self, library, planner, *, standalone_library=None):
        if standalone_library is not None and (not standalone_library.is_standalone
                or standalone_library.parents != library.parents or standalone_library.rest != library.rest):
            raise ValueError('Expected a standalone library with the same source skeleton')
        self.library, self.planner = library, planner
        self.standalone_library = standalone_library
        self._lock = threading.Lock()

    def status(self):
        return {'enabled': True, 'planner': self.planner.status(), **self.library.status(),
                'max_characters': 200, 'max_words': MAX_WORDS,
                'standalone_expressions': {'enabled': self.standalone_library is not None,
                    **(self.standalone_library.status() if self.standalone_library is not None else {})}}

    def compose(self, text):
        normalized = normalize_text(text)
        if not self._lock.acquire(blocking=False):
            raise CompositionBusyError('Another composition request is running')
        try:
            started = time.perf_counter()
            exact = normalized.rstrip('。！？.!?')
            if self.standalone_library is not None and self.standalone_library.has_word(exact):
                result = self.standalone_library.compose(normalized, [exact])
                elapsed = (time.perf_counter() - started) * 1000
                return {**result, 'input_text': text, 'generation_text': normalized,
                        'selection_mode': 'exact_native_utterance', 'model_identity': None,
                        'timings_ms': {'planning_ms': 0, 'composition_ms': elapsed, 'total_ms': elapsed}}
            try:
                plan = self.planner.plan(normalized)
            except RuntimeError as exc:
                if type(exc).__name__ == 'PlannerBusyError':
                    raise CompositionBusyError('Text planner is busy') from exc
                raise CompositionUnavailableError('Text planner is unavailable') from exc
            planned = time.perf_counter()
            if not isinstance(plan, dict) or 'glosses' not in plan:
                raise CompositionUnavailableError('Invalid text plan')
            glosses = plan['glosses']
            if (not isinstance(glosses, list) or not 1 <= len(glosses) <= MAX_WORDS
                    or any(not isinstance(g, str) or not g.strip() for g in glosses)):
                raise CompositionUnavailableError('Invalid text plan glosses')
            joined = ''.join(glosses)
            omitted = [term for term in _ANCHORS
                       if term in normalized and term in self.library._words and term not in joined]
            if omitted:
                result = {'status': 'needs_review', 'glosses': list(glosses), 'omitted_terms': omitted,
                          'missing_glosses': [], 'language_review': 'NOT_RUN',
                          'bundle_id': self.library.bundle_id,
                          'coverage_check': 'literal-known-place-time-presence-v1'}
            else:
                result = self.library.compose(normalized, glosses)
            finished = time.perf_counter()
            return {**result, 'input_text': text, 'generation_text': normalized,
                    'model_identity': plan.get('model_identity'),
                    'timings_ms': {'planning_ms': (planned - started) * 1000,
                                   'composition_ms': (finished - planned) * 1000,
                                   'total_ms': (finished - started) * 1000}}
        finally:
            self._lock.release()

    def close(self):
        self.planner.close()
