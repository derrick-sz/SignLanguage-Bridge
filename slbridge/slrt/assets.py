"""Asset identities and configuration; independent of Torch and the UI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

CHECKPOINT_SHA256 = 'e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d'
VOCAB_SHA256 = '0c4984f0dd3e7897b8c8d9e6e99a848b60814a5d9cf7a49baf7cc9b9cd124c76'
BACKEND_ID = 'slrt_online_csl'
# v2: frame-level RGB+heatmap cache inside the batch preparation path.  The
# numeric protocol is unchanged (each frame is prepared by the same code); the
# identity is bumped so runs/caches recorded under v1 are never mixed with the
# new execution path.  tests/test_slrt_frame_cache.py proves bit-identical
# tensors and identical Gloss.
PROTOCOL = 'slrt-csl-rgb320-crop270-hm170-sigma8-v2'
PROTOCOL_V1 = 'slrt-csl-rgb320-crop270-hm170-sigma8-v1'


class SLRTResourceError(ValueError):
    """Missing dependencies/assets or an incompatible trained model."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, description: str, expected: str | None = None) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise SLRTResourceError(f'{description} missing: {path}')
    if expected and sha256(path) != expected:
        raise SLRTResourceError(f'{description} SHA256 mismatch: {path}')
    return path


def load_vocabulary(path: Path, *, expected_classes: int = 2001) -> tuple[str, ...]:
    require_file(path, 'Gloss vocabulary')
    try:
        vocab = json.loads(Path(path).read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as e:
        raise SLRTResourceError('Gloss vocabulary must be an ordered UTF-8 JSON list') from e
    if (not isinstance(vocab, list) or len(vocab) != expected_classes or
            any(not isinstance(t, str) or not t.strip() for t in vocab) or len(set(vocab)) != len(vocab)):
        raise SLRTResourceError(f'Gloss vocabulary must contain {expected_classes} unique nonempty classes')
    if vocab[0] != '<blank>' or vocab.count('<blank>') != 1:
        raise SLRTResourceError('SLRT-CSL requires exactly one <blank> at index 0')
    return tuple(vocab)


@dataclass(frozen=True)
class SLRTConfig:
    resource_dir: str = 'models/external/EMNLP24_Online/CSL-Daily'
    checkpoint: str | None = None
    vocabulary: str | None = None
    checkpoint_sha256: str = CHECKPOINT_SHA256
    vocabulary_sha256: str = VOCAB_SHA256
    model_id: str = 'slrt-emnlp24-csl-daily'
    sign_language: str = 'csl'
    device: str = 'auto'
    device_id: int = 0
    cpu_threads: int = 4
    window_size: int = 16
    stride: int = 1
    vote_size: int = 7
    pose_backend: str = 'hrnet'
    pose_device: str = 'auto'
    detector_config: str | None = None
    detector_checkpoint: str | None = None
    hrnet_config: str | None = None
    hrnet_checkpoint: str | None = None
    rtmpose_cache_dir: str = 'models/external/uni_sign/rtmlib-cache'
    max_gap_seconds: float = .5
    queue_size: int = 32
    max_gloss_history: int = 512

    def __post_init__(self):
        if self.sign_language != 'csl':
            raise ValueError('This trained SLRT checkpoint supports CSL only')
        if self.device not in {'cpu', 'cuda', 'auto'} or self.pose_device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError('device must be auto, cpu or cuda')
        if type(self.window_size) is not int or self.window_size < 16 or self.window_size > 64 or self.window_size % 8:
            raise ValueError('window_size must be a multiple of 8 in [16,64]; only 16 is the trained protocol')
        for name in ('stride', 'cpu_threads', 'queue_size', 'max_gloss_history'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.stride > self.window_size or type(self.vote_size) is not int or self.vote_size not in {1, 3, 5, 7, 9, 11, 13}:
            raise ValueError('stride must not exceed window; vote_size must be an odd integer in [1,13]')
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError('device_id must be a nonnegative integer')
        if self.pose_backend not in {'hrnet', 'hrnet-npz', 'rtmpose-experimental'}:
            raise ValueError('pose_backend must be hrnet, hrnet-npz or rtmpose-experimental')
        if type(self.max_gap_seconds) not in (int, float) or not math.isfinite(self.max_gap_seconds) or not 0 < self.max_gap_seconds <= 10:
            raise ValueError('max_gap_seconds must be in (0,10]')
        for name in ('checkpoint_sha256', 'vocabulary_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{name} must be an explicit lowercase SHA256')
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError('model_id must be nonempty')

    @property
    def checkpoint_path(self):
        return Path(self.checkpoint) if self.checkpoint else Path(self.resource_dir) / 'ckpts/cslr_best.ckpt'

    @property
    def vocabulary_path(self):
        return Path(self.vocabulary) if self.vocabulary else Path(self.resource_dir) / 'meta/csl_iso_with_blank.vocab'

    def to_dict(self):
        return {k:str(v) if isinstance(v,Path) else v for k,v in asdict(self).items()}

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path).resolve()
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            raise ValueError('SLRT config must be a JSON object')
        raw = dict(raw)
        if raw.pop('backend', BACKEND_ID) != BACKEND_ID:
            raise ValueError('Configuration backend must be slrt_online_csl')
        allowed = {f.name for f in fields(cls)}
        if set(raw) - allowed:
            raise ValueError(f'Unknown SLRT config fields: {sorted(set(raw)-allowed)}')
        for name in ('resource_dir', 'checkpoint', 'vocabulary', 'detector_config', 'detector_checkpoint', 'hrnet_config', 'hrnet_checkpoint', 'rtmpose_cache_dir'):
            if raw.get(name) and not Path(raw[name]).is_absolute():
                raw[name] = str((path.parent / raw[name]).resolve())
        return cls(**raw)
