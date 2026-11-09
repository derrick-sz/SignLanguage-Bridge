"""Bounded source-free requests and strict, verified pretrained-weight loading."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re
import sys
import warnings
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


def _load_readonly_mapped_checkpoint(path, torch):
    """Use the reviewed 2.9.1 weights-only reader with an OS read-only mapping.

    Unlike Windows' native torch mmap (private/copy-on-write), mode='r' does
    not reserve a potential writable copy of the whole checkpoint. Torch has
    no read-only tensor flag: consumers MUST NOT write into these CPU tensors.
    Assignment followed by a device move only reads the mapped bytes. Storage
    views retain the NumPy mapping through from_numpy's storage owner.
    """
    if str(torch.__version__).split('+', 1)[0] != '2.9.1':
        raise RuntimeError('readonly_mmap requires reviewed PyTorch 2.9.1')
    serialization = torch.serialization
    try:
        parameters = inspect.signature(serialization._load).parameters
        compatible = (tuple(parameters) == (
            'zip_file', 'map_location', 'pickle_module', 'pickle_file',
            'overall_storage', 'pickle_load_args')
            and parameters['overall_storage'].default is None
            and parameters['pickle_load_args'].kind == inspect.Parameter.VAR_KEYWORD
            and callable(serialization._open_zipfile_reader)
            and callable(serialization._is_torchscript_zip)
            and callable(serialization._weights_only_unpickler.Unpickler))
    except (AttributeError, TypeError, ValueError):
        compatible = False
    if not compatible:
        raise RuntimeError('Unsupported PyTorch readonly_mmap private reader interface')
    with path.open('rb') as checkpoint, serialization._open_zipfile_reader(checkpoint) as reader:
        if serialization._is_torchscript_zip(reader):
            raise ValueError('readonly_mmap requires a tensor state dictionary, not TorchScript')
        # _load byteswaps storages in place on foreign-endian input. Reject it
        # before exposing any read-only address to that operation.
        if reader.has_record('byteorder'):
            byteorder = reader.get_record('byteorder')
        else:
            endianness = serialization.get_default_load_endianness()
            if endianness == serialization.LoadEndianness.NATIVE:
                byteorder = sys.byteorder.encode('ascii')
            elif endianness == serialization.LoadEndianness.BIG:
                byteorder = b'big'
            else:
                byteorder = b'little'
        if byteorder != sys.byteorder.encode('ascii'):
            raise ValueError('readonly_mmap requires native checkpoint byteorder')
        mapped_bytes = np.memmap(path, dtype=np.uint8, mode='r')
        # This intentional bridge is safe only for reads. Suppress this exact
        # warning locally; never mark the mapping writable or use shared RW.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given NumPy array is not writable',
                                    category=UserWarning)
            storage = torch.from_numpy(mapped_bytes).untyped_storage()
        return serialization._load(reader, 'cpu', serialization._weights_only_unpickler,
                                   overall_storage=storage, encoding='utf-8')


def load_strict_checkpoint(model, path, *, expected_sha256, expected_bytes, assign: bool = False,
                           readonly_mmap: bool = False):
    """Verify before mapping; strict assignment can avoid a full CPU weight copy.

    Assignment is intended before optimizer creation or device moves. This
    worker has no optimizer. Keep the checkpoint immutable while mapped state
    is alive; nonpersistent buffers must retain real values from construction.
    readonly_mmap opts into a pinned, OS read-only mapping for inference. With
    assign=True, never mutate CPU parameters/buffers backed by the checkpoint;
    move them to the inference device before any operation requiring writes.
    """
    path = verified_file(path, expected_sha256, expected_bytes)
    import torch  # CPU animation service can import request validation without torch
    if not assign and any(value.is_meta for value in model.parameters()):
        raise ValueError('Meta parameters require assign=True for checkpoint materialization')
    mapped = zipfile.is_zipfile(path)
    if readonly_mmap:
        if not mapped:
            raise ValueError('readonly_mmap requires a torch.save ZIP checkpoint')
        state = _load_readonly_mapped_checkpoint(path, torch)
    else:
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=mapped)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError('Expected the author EMA tensor state dictionary')
    result = model.load_state_dict(state, strict=True, assign=assign)
    remaining=[kind+':'+name for kind,values in (('parameter',model.named_parameters()),('buffer',model.named_buffers()))
               for name,value in values if value.is_meta]
    if remaining:
        raise ValueError('Unmaterialized meta parameter/buffer after strict load: '+', '.join(remaining[:8]))
    return {'sha256': expected_sha256, 'bytes': expected_bytes,
            'assign': assign, 'mmap': mapped, 'readonly_mmap': readonly_mmap,
            'tensors': len(state), 'missing_keys': list(result.missing_keys),
            'unexpected_keys': list(result.unexpected_keys)}


def prepare_cpu_offload(model, texts: list[str], device='cuda'):
    """Cache exact CPU text conditioning before installing inference offload.

    Call only after verified strict loading, while every tensor is real CPU
    data. The original encoder reads the immutable FP32 checkpoint once per
    unique prompt. Only these small embeddings move eagerly to the execution
    device; Accelerate stages individual layers on demand. Cached text must
    match the complete upstream prompt, including its language prefix.

    Buffers remain real on the execution device, including nonpersistent
    position/RoPE buffers. Offloaded parameters intentionally become meta
    placeholders between forwards; never call model.to(...) after this setup.
    """
    if (not isinstance(texts, list) or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)):
        raise ValueError('Expected a nonempty list of explicit text prompts')
    import torch
    from accelerate import cpu_offload
    from types import MethodType

    tensors = [*model.parameters(), *model.buffers()]
    if any(value.is_meta or value.device.type != 'cpu' for value in tensors):
        raise ValueError('CPU offload preparation requires real CPU parameters and buffers')
    if (any(value.dtype != torch.float32 for value in model.parameters())
            or any(value.is_floating_point() and value.dtype != torch.float32 for value in model.buffers())):
        raise ValueError('CPU offload preparation requires FP32 weights and floating buffers')
    if not callable(getattr(model, 'encode_text', None)):
        raise ValueError('Expected the original model text encoder')
    execution_device = torch.device(device)
    model.eval().requires_grad_(False)
    original_encode_text = model.encode_text
    cache = {}
    embedding_width = None
    with torch.inference_mode():
        for text in dict.fromkeys(texts):
            embedding = original_encode_text([text])
            if (not torch.is_tensor(embedding) or embedding.device.type != 'cpu'
                    or embedding.is_meta or embedding.dtype != torch.float32
                    or embedding.ndim != 2 or embedding.shape[0] != 1
                    or embedding.shape[1] == 0 or not torch.isfinite(embedding).all().item()
                    or (embedding_width is not None and embedding.shape[1] != embedding_width)):
                raise ValueError('Expected a finite, real CPU FP32 text embedding with shape [1, width]')
            embedding_width = embedding.shape[1]
            cache[text] = embedding.detach().to(execution_device)

    def cached_encode_text(self, raw_text):
        if (not isinstance(raw_text, list) or not raw_text
                or any(not isinstance(text, str) or text not in cache for text in raw_text)):
            raise ValueError('Text conditioning requires known cached text prompts')
        if len(raw_text) == 1:
            return cache[raw_text[0]]
        return torch.cat([cache[text] for text in raw_text], dim=0)

    model.encode_text = MethodType(cached_encode_text, model)
    cpu_offload(model, execution_device=execution_device, offload_buffers=False)
    return {'execution_device': str(execution_device), 'offload_buffers': False,
            'cached_texts': len(cache), 'embedding_dtype': 'float32',
            'embedding_width': embedding_width, 'strategy': 'accelerate.cpu_offload'}
