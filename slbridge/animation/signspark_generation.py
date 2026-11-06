"""Bounded source-free requests and strict, verified pretrained-weight loading."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import zipfile

import numpy as np

PADDED_FRAMES = 304  # trained temporal UNet shape; actual output is explicitly cropped
STREAM_WIDTHS = {'body': 60, 'hand': 90, 'face': 56}
SOURCE_COMMIT = '916402d0ecd6322fb4a9e5f513229a94a57bf4b6'
MODEL_REVISION = 'b907f25f9bb9f5a0adcaf131a23a155dc471d9a8'
MODEL_FILES = {
    'hand': (5581699849, '5bb8e407ba2c106a22c3b0c16fd1b0fc0f412a58d235143325fdb67efe5916c2'),
    'body': (5580532361, 'bf95bb02802b1c46f40afb4424f8f521be6c8641ad8acfe406a666b463e3a380'),
    'face': (5580376713, '841e6ce5c929b142ad8ea3cc152fa67f57e7a8841954bf068df42a5d986ead53'),
}


@dataclass(frozen=True)
class GenerationRequest:
    text: str
    frames: int = 128
    steps: int = 50
    seed: int = 123

    def __post_init__(self):
        if (not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 200
                or any(ord(c) < 32 for c in self.text)
                or type(self.frames) is not int or not 16 <= self.frames <= PADDED_FRAMES
                or type(self.steps) is not int or not 1 <= self.steps <= 100
                or type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError('Invalid text, explicit frame count, steps or seed')


def source_free_condition(request, stream):
    """Accept no target/data argument: observations are finite zeros with mask off."""
    if not isinstance(request, GenerationRequest) or stream not in STREAM_WIDTHS:
        raise ValueError('Expected validated request and known stream')
    width = STREAM_WIDTHS[stream]
    valid = np.zeros((1, 1, PADDED_FRAMES), dtype=bool)
    valid[:, :, :request.frames] = True
    return {
        'obs_x0': np.zeros((1, width, PADDED_FRAMES), dtype=np.float32),
        'obs_mask': np.zeros((1, width, PADDED_FRAMES), dtype=bool),
        'y': {'mask': valid, 'lengths': np.array([request.frames], dtype=np.int64),
              'text': ['<Chinese> '+request.text.strip()]},
    }


def verified_file(path, expected_sha256, expected_bytes):
    path = Path(path)
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Expected trusted SHA256 identity')
    if path.stat().st_size != expected_bytes:
        raise ValueError('Pretrained checkpoint length mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while raw := f.read(8*1024*1024):
            digest.update(raw)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Pretrained checkpoint SHA256 mismatch')
    return path


def load_strict_checkpoint(model, path, *, expected_sha256, expected_bytes, assign: bool = False):
    """Verify before mapping; strict assignment can avoid a full CPU weight copy.

    Assignment is intended before optimizer creation or device moves. This
    worker has no optimizer. Keep the checkpoint immutable while mapped state
    is alive; nonpersistent buffers must retain real values from construction.
    """
    path = verified_file(path, expected_sha256, expected_bytes)
    import torch  # CPU animation service can import request validation without torch
    if not assign and any(value.is_meta for value in model.parameters()):
        raise ValueError('Meta parameters require assign=True for checkpoint materialization')
    mapped = zipfile.is_zipfile(path)
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=mapped)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError('Expected the author EMA tensor state dictionary')
    result = model.load_state_dict(state, strict=True, assign=assign)
    remaining=[kind+':'+name for kind,values in (('parameter',model.named_parameters()),('buffer',model.named_buffers()))
               for name,value in values if value.is_meta]
    if remaining:
        raise ValueError('Unmaterialized meta parameter/buffer after strict load: '+', '.join(remaining[:8]))
    return {'sha256': expected_sha256, 'bytes': expected_bytes,
            'assign': assign, 'mmap': mapped,
            'tensors': len(state), 'missing_keys': list(result.missing_keys),
            'unexpected_keys': list(result.unexpected_keys)}
