"""One reusable Torch network; heavy imports remain optional."""
from __future__ import annotations

import gc
import sys
from time import perf_counter

from .assets import SLRTConfig, SLRTResourceError, load_vocabulary, require_file
from .preprocess import FramePreparationCache


class SLRTRuntime:
    def __init__(self, config: SLRTConfig, *, frame_cache_enabled: bool = True):
        if not isinstance(frame_cache_enabled, bool):
            raise ValueError('frame_cache_enabled must be a boolean')
        started = perf_counter()
        self.config = config
        self._closed = False
        # Optional per-frame preparation cache (stride < window removes ~16x
        # duplicate work); values are bit-identical to the batched path.
        self._frame_cache = FramePreparationCache() if frame_cache_enabled else None
        require_file(config.checkpoint_path, 'SLRT checkpoint', config.checkpoint_sha256)
        require_file(config.vocabulary_path, 'SLRT vocabulary', config.vocabulary_sha256)
        self.vocabulary = load_vocabulary(config.vocabulary_path)
        try:
            if sys.version_info < (3,11):
                import typing
                # Same compatibility boundary as the existing Uni-Sign runtime.
                # Qt may have installed typing_extensions.Self on Python 3.10.
                typing.Self = typing.TypeVar('Self')
            import torch
            from .network import SLRTNetwork
        except (ImportError, OSError) as e:
            raise SLRTResourceError('SLRT needs compatible Torch/torchvision; use requirements-slrt.txt in a separate environment') from e
        self.torch = torch
        if config.device == 'cuda' and not torch.cuda.is_available():
            raise SLRTResourceError('CUDA explicitly requested but Torch CUDA is unavailable')
        self.device = 'cuda' if config.device == 'auto' and torch.cuda.is_available() else 'cpu' if config.device == 'auto' else config.device
        self.torch_device = f'cuda:{config.device_id}' if self.device == 'cuda' else 'cpu'
        if self.device == 'cuda' and config.device_id >= torch.cuda.device_count():
            raise SLRTResourceError(f'CUDA device {config.device_id} is unavailable')
        # CPU tensor conversion also runs in CUDA mode. Bound those workers so
        # they do not compete with ORT and capture for every small window.
        torch.set_num_threads(config.cpu_threads)
        try:
            raw = torch.load(config.checkpoint_path, map_location='cpu', weights_only=True)
            state = raw['model_state']
            self.model = SLRTNetwork(len(self.vocabulary))
            self.model.load_state_dict(state, strict=True)
        except (KeyError, TypeError, RuntimeError, ValueError) as e:
            raise SLRTResourceError(f'SLRT checkpoint network/classes mismatch: {e}') from e
        self.state_keys = len(state)
        self.model.to(self.torch_device).eval().requires_grad_(False)
        del raw, state
        self.load_ms = (perf_counter() - started) * 1000

    @property
    def info(self):
        t = self.torch
        return dict(backend='pytorch_slrt', device=self.torch_device, dtype='float32',
                    torch_version=t.__version__, state_keys=self.state_keys, strict_load=True,
                    classes=len(self.vocabulary), blank_id=0, load_ms=self.load_ms,
                    checkpoint_sha256=self.config.checkpoint_sha256, vocabulary_sha256=self.config.vocabulary_sha256,
                    gpu_name=t.cuda.get_device_name(self.config.device_id) if self.device == 'cuda' else None,
                    gpu_peak_allocated_mb=t.cuda.max_memory_allocated(self.config.device_id) / 1024**2 if self.device == 'cuda' else None,
                    gpu_peak_reserved_mb=t.cuda.max_memory_reserved(self.config.device_id) / 1024**2 if self.device == 'cuda' else None)

    def synchronize(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize(self.config.device_id)

    @property
    def frame_cache_info(self) -> dict:
        """Cache counters for diagnostics; not part of the textual result contract."""
        if self._frame_cache is None:
            return dict(enabled=False, limit=0, size=0, hits=0, misses=0)
        return self._frame_cache.info()

    def clear_frame_cache(self) -> None:
        if self._frame_cache is not None:
            self._frame_cache.clear()

    def predict(self, bgr, keypoints):
        from .preprocess import prepare_inputs
        if self._closed:
            raise RuntimeError('SLRT runtime is closed')
        self.synchronize()
        start = perf_counter()
        with self.torch.inference_mode():
            if self._frame_cache is not None:
                rgb, hm = self._frame_cache.prepare(bgr, keypoints, device=self.torch_device)
            else:
                rgb, hm = prepare_inputs(bgr, keypoints, device=self.torch_device)
            self.synchronize()
            preprocessed = perf_counter()
            probabilities = self.model(rgb, hm)
            self.synchronize()
            predicted = perf_counter()
            probabilities = probabilities[0].cpu().numpy()
        return probabilities, dict(preprocess_ms=(preprocessed-start)*1000,
                                   inference_ms=(predicted-preprocessed)*1000,
                                   window_total_ms=(perf_counter()-start)*1000)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.model = None
        self._frame_cache = None
        gc.collect()
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()
