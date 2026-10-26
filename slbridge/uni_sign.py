"""Optional Uni-Sign pose-only SLT backend.

The checkpoint is a CSL-Daily pose-only translation model. It is deliberately
kept behind a registered adapter and never replaces the ONNX CTC contract. The
runtime uses the pinned upstream source locally; upstream source is not vendored
into this repository.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import gc
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from ._io import read_json
from .contracts import RecognitionResult, TextPrediction
from .model_adapters import (
    InputRepresentation,
    ModelCapabilities,
    ModelInputContract,
    get_input_representation,
    register_input_representation,
)
from .runtime_contracts import InferenceRequest, _identifier
from .runtime_options import RuntimeOptions
from .uni_sign_assets import (
    MT5_FILES,
    MODEL_LICENSE,
    SOURCE_LICENSE_STATUS,
    UNISIGN_REVISION,
    UNISIGN_SOURCE_COMMIT,
    all_checks_ok,
    verify_checkpoint,
    verify_mt5_dir,
)
from .uni_sign_preprocess import (
    RTMPOSE_KEYPOINT_COUNT,
    PoseWindow,
    deterministic_frame_indices,
    normalize_rtmpose_pixels,
    orient_pose_window,
    prepare_pose_window,
)


REPRESENTATION_ID = "uni_sign_rgb_v1"
BACKEND_ID = "uni_sign_pose_slt"
SCORE_TYPE = "uncalibrated_text"


def _register_representation() -> None:
    try:
        get_input_representation(REPRESENTATION_ID)
    except ValueError:
        register_input_representation(InputRepresentation(
            REPRESENTATION_ID, (None, None, None, 3), dtype="uint8", layout="T,H,W,C"))


_register_representation()


@dataclass(frozen=True)
class UniSignConfig:
    bundle_dir: Path
    checkpoint: Path
    checkpoint_bytes: int
    checkpoint_sha256: str
    mt5_dir: Path
    source_dir: Path
    source_commit: str
    source_zip_sha256: str
    model_revision: str
    mt5_revision: str
    run_id: str
    display_name: str
    runtime: dict[str, Any]
    license: dict[str, Any]

    @classmethod
    def from_bundle(cls, path: str | Path) -> "UniSignConfig":
        bundle_dir = Path(path).resolve()
        raw = read_json(bundle_dir / "manifest.json")
        if not isinstance(raw, dict):
            raise ValueError("Uni-Sign manifest must be a JSON object")
        required = {
            "schema_version", "run_id", "display_name", "backend", "task_type",
            "sign_language", "capabilities", "checkpoint", "checkpoint_bytes",
            "checkpoint_sha256", "model_repo", "model_revision", "mt5_dir",
            "mt5_repo", "mt5_revision", "mt5_model_file_sha256", "source_dir",
            "source_repo", "source_commit", "source_zip_sha256", "runtime", "license",
            "provenance",
        }
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing or unknown:
            raise ValueError(f"Invalid Uni-Sign manifest fields: missing={missing}, unknown={unknown}")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("Unsupported Uni-Sign manifest schema")
        if raw["backend"] != BACKEND_ID or raw["task_type"] != "text" or raw["sign_language"] != "csl":
            raise ValueError("Uni-Sign manifest must declare the text/csl/uni_sign_pose_slt backend")
        if raw["capabilities"] != ["clip"]:
            raise ValueError("Uni-Sign manifest capabilities must be ['clip']")
        _identifier(raw["run_id"], "run_id")
        if not isinstance(raw["display_name"], str) or not raw["display_name"].strip():
            raise ValueError("Uni-Sign manifest requires a display_name")
        checkpoint = (bundle_dir / raw["checkpoint"]).resolve()
        mt5_dir = (bundle_dir / raw["mt5_dir"]).resolve()
        source_dir = (bundle_dir / raw["source_dir"]).resolve()
        if type(raw["checkpoint_bytes"]) is not int or raw["checkpoint_bytes"] <= 0:
            raise ValueError("checkpoint_bytes must be a positive integer")
        for name in ("checkpoint_sha256", "mt5_model_file_sha256", "source_zip_sha256"):
            value = raw[name]
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if raw["model_revision"] != UNISIGN_REVISION:
            raise ValueError("Uni-Sign manifest does not use the pinned model revision")
        if raw["source_commit"] != UNISIGN_SOURCE_COMMIT:
            raise ValueError("Uni-Sign manifest does not use the pinned source commit")
        if raw["mt5_model_file_sha256"] != MT5_FILES["pytorch_model.bin"][1]:
            raise ValueError("Uni-Sign manifest does not use the pinned mT5 model file")
        if not isinstance(raw["runtime"], dict) or not isinstance(raw["license"], dict):
            raise ValueError("Uni-Sign runtime/license metadata must be objects")
        return cls(
            bundle_dir=bundle_dir,
            checkpoint=checkpoint,
            checkpoint_bytes=raw["checkpoint_bytes"],
            checkpoint_sha256=raw["checkpoint_sha256"],
            mt5_dir=mt5_dir,
            source_dir=source_dir,
            source_commit=raw["source_commit"],
            source_zip_sha256=raw["source_zip_sha256"],
            model_revision=raw["model_revision"],
            mt5_revision=raw["mt5_revision"],
            run_id=raw["run_id"],
            display_name=raw["display_name"],
            runtime=dict(raw["runtime"]),
            license=dict(raw["license"]),
        )


class TextStabilizer:
    """Annotate repeated-window text without pretending it is human-confirmed."""

    def __init__(self, confirm_after: int = 2, *, history_size: int = 8):
        if type(confirm_after) is not int or confirm_after < 2:
            raise ValueError("confirm_after must be an integer >= 2")
        if type(history_size) is not int or history_size < confirm_after:
            raise ValueError("history_size must be at least confirm_after")
        self.confirm_after = confirm_after
        self.history_size = history_size
        self.reset()

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return " ".join(text.split())

    def reset(self) -> None:
        self._text = ""
        self._streak = 0
        self._history: list[str] = []
        self._revision = 0

    def update(self, text: str, *, at: float | None = None) -> dict[str, Any]:
        normalized = self.normalize(text)
        previous = self._text
        changed = normalized != previous
        if normalized:
            if changed:
                self._text = normalized
                self._streak = 1
                self._revision += 1
            else:
                self._streak += 1
            self._history.append(normalized)
            if len(self._history) > self.history_size:
                self._history = self._history[-self.history_size:]
        else:
            self._text = ""
            self._streak = 0
        state = "empty" if not normalized else (
            "confirmed" if self._streak >= self.confirm_after else "temporary")
        return {
            "state": state,
            "normalized_text": normalized,
            "repeat_count": self._streak,
            "changed": bool(changed),
            "duplicate": bool(normalized and not changed),
            "previous_text": previous or None,
            "revision": self._revision,
            "history": list(self._history),
            "at_monotonic": None if at is None else float(at),
        }


class RtmposeExtractor:
    """RTMPose Wholebody wrapper with explicit CUDA DLL and cache handling."""

    def __init__(self, *, mode: str, backend: str, device: str, cache_dir: Path,
                 dll_dirs: tuple[str, ...] = (), strict_cuda: bool = False,
                 warmup_image: Path | None = None, detect_stride: int = 1):
        if type(detect_stride) is not int or not 1 <= detect_stride <= 30:
            raise ValueError("detect_stride must be an integer in [1, 30]")
        self._detect_stride = detect_stride
        self._frames_seen = 0
        self._last_boxes = None
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)
        self._dll_handles: list[Any] = []
        self._requested_device = device
        self.cache_dir = cache_dir
        self.warmup_image = Path(warmup_image) if warmup_image is not None else None
        self.fallback_reason = None
        self.preload_failures: list[str] = []
        self.warmup_error = None
        if device == "cuda":
            self._configure_windows_cuda(dll_dirs)
        from rtmlib import Wholebody

        self._wholebody = Wholebody(to_openpose=False, mode=mode, backend=backend, device=device)
        self.mode = mode
        self.backend = backend
        det_providers = tuple(getattr(self._wholebody.det_model.session, "get_providers", lambda: ())())
        pose_providers = tuple(getattr(self._wholebody.pose_model.session, "get_providers", lambda: ())())
        cuda_active = ("CUDAExecutionProvider" in det_providers
                       and "CUDAExecutionProvider" in pose_providers)
        if device == "cuda" and not cuda_active:
            self.fallback_reason = (
                "RTMPose requested CUDA but ONNX Runtime created a CPU session; "
                "basic inference remains available, but pose GPU acceleration is not proven"
            )
            if strict_cuda:
                self.close()
                raise RuntimeError(self.fallback_reason)
        self.device = "cuda" if cuda_active else "cpu"
        self.det_providers = det_providers
        self.pose_providers = pose_providers

    def _configure_windows_cuda(self, dll_dirs: tuple[str, ...]) -> None:
        if os.name != "nt":
            return
        candidates = [Path(path).resolve() for path in dll_dirs]
        nvidia_root = Path(sys.prefix) / "Lib/site-packages/nvidia"
        if nvidia_root.is_dir():
            candidates.extend(sorted({path.parent.resolve()
                                      for path in nvidia_root.glob("*/bin/*.dll")}))
        candidates = list(dict.fromkeys(path for path in candidates if path.is_dir()))
        # Repeated extractor construction (multi-stream sessions) must not grow
        # PATH without bound; Windows caps the environment at 32767 characters.
        known = {entry.casefold() for entry in os.environ.get("PATH", "").split(os.pathsep)}
        for directory in candidates:
            try:
                self._dll_handles.append(os.add_dll_directory(str(directory)))
                if str(directory).casefold() not in known:
                    known.add(str(directory).casefold())
                    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            except OSError as error:
                self.preload_failures.append(f"{directory}: {error}")
        try:
            import onnxruntime as ort
            for directory in candidates:
                if hasattr(ort, "preload_dlls"):
                    try:
                        ort.preload_dlls(directory=str(directory))
                    except Exception as error:
                        self.preload_failures.append(f"preload {directory}: {type(error).__name__}: {error}")
            if hasattr(os, "add_dll_directory"):
                from .onnx_backend import _preload_cudnn_engines
                _preload_cudnn_engines()
        except Exception as error:
            self.preload_failures.append(f"ONNX Runtime CUDA preload: {type(error).__name__}: {error}")

    def warmup(self) -> None:
        import numpy as np
        try:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            if self.warmup_image is not None and self.warmup_image.is_file():
                import cv2
                decoded = cv2.imread(str(self.warmup_image))
                if decoded is not None:
                    image = decoded
            self._wholebody(image)
        except Exception as error:
            self.warmup_error = f"{type(error).__name__}: {error}"

    def extract(self, frame: Any):
        if self._detect_stride <= 1:
            return self._wholebody(frame)
        # Reuse the last person box for `detect_stride - 1` frames. This trades a
        # small amount of box staleness for a large detection saving on static,
        # single-person footage; it is opt-in and must be validated downstream.
        self._frames_seen += 1
        if self._last_boxes is None or (self._frames_seen - 1) % self._detect_stride == 0:
            self._last_boxes = self._wholebody.det_model(frame)
        keypoints, scores = self._wholebody.pose_model(frame, bboxes=self._last_boxes)
        return keypoints, scores

    def reset_detection(self) -> None:
        """Forget the reused box; the next frame runs detection again."""
        self._last_boxes = None
        self._frames_seen = 0

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {
            "pose_mode": self.mode,
            "pose_backend": self.backend,
            "pose_detect_stride": self._detect_stride,
            "pose_requested_device": self._requested_device,
            "pose_device": self.device,
            "pose_detection_providers": list(self.det_providers),
            "pose_estimation_providers": list(self.pose_providers),
            "pose_fallback_reason": self.fallback_reason,
            "pose_preload_failures": list(self.preload_failures),
            "pose_warmup_error": self.warmup_error,
            "pose_cache_dir": str(self.cache_dir),
            "pose_warmup_image": str(self.warmup_image) if self.warmup_image is not None else None,
        }

    def close(self) -> None:
        self._wholebody = None
        for handle in reversed(self._dll_handles):
            try:
                handle.close()
            except Exception:
                pass
        self._dll_handles.clear()

class TorchUniSignRuntime:
    """Load and execute the pinned PyTorch model in an isolated import scope."""

    def __init__(self, config: UniSignConfig, *, device: str, dtype: str):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Uni-Sign device must be cpu or cuda")
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("Uni-Sign dtype must be float32 or bfloat16")
        self.config = config
        self.device_name = device
        self.dtype_name = dtype
        self._torch = None
        self._torch_dtype = None
        self._model = None
        self._tokenizer = None
        self.load_ms = 0.0
        self.load_details: dict[str, Any] = {}

    def _load_upstream_class(self):
        source_dir = self.config.source_dir
        models_file = source_dir / "models.py"
        if not models_file.is_file():
            raise FileNotFoundError(f"Uni-Sign source is missing: {models_file}")
        saved: dict[str, Any] = {}
        names = {"config", "models", "datasets", "utils", "stgcn_layers", "deformable_attention_2d"}
        for name in list(sys.modules):
            if name in names or name.startswith("stgcn_layers."):
                saved[name] = sys.modules.pop(name)
        sys.path.insert(0, str(source_dir))
        module_name = f"_slbridge_upstream_uni_sign_{uuid4().hex}"
        try:
            import config as upstream_config
            upstream_config.mt5_path = str(self.config.mt5_dir)
            spec = importlib.util.spec_from_file_location(module_name, models_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load upstream model module from {models_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module.Uni_Sign
        finally:
            try:
                sys.path.remove(str(source_dir))
            except ValueError:
                pass
            for name in list(sys.modules):
                if name == module_name or name in names or name.startswith("stgcn_layers."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)

    def load(self) -> None:
        import typing
        if sys.version_info < (3, 11):
            # torch 2.9 torchvision/torch._dynamo annotates Union[..., Self].
            # Python 3.10 treats the typing_extensions Self special form as
            # invalid inside Union; a TypeVar keeps the annotation evaluable.
            typing.Self = typing.TypeVar("Self")
        started = time.perf_counter()
        try:
            import torch
        except Exception as error:  # pragma: no cover - exercised in the isolated env
            raise RuntimeError("Uni-Sign requires PyTorch in its isolated environment") from error
        self._torch = torch
        self._torch_dtype = torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float32
        if self.device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available to PyTorch")
            torch.cuda.synchronize()

        UniSign = self._load_upstream_class()
        args = SimpleNamespace(dataset="CSL_Daily", task="SLT", rgb_support=False,
                               hidden_dim=256, label_smoothing=0.2)
        model = UniSign(args=args)
        checkpoint = torch.load(self.config.checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError("Uni-Sign checkpoint does not contain a model state dict")
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        loaded = model.load_state_dict(state, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(
                "Uni-Sign checkpoint does not match the pinned model definition: "
                f"missing={loaded.missing_keys[:8]}, unexpected={loaded.unexpected_keys[:8]}"
            )
        model.eval()
        model = model.to(self.device_name)
        model = model.to(self._torch_dtype)
        if self.device_name == "cuda":
            torch.cuda.synchronize()
        self._model = model
        self._tokenizer = model.mt5_tokenizer
        self.load_ms = (time.perf_counter() - started) * 1000
        self.load_details = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "load_ms": self.load_ms,
            "device": self.device_name,
            "dtype": self.dtype_name,
        }
        if self.device_name == "cuda":
            self.load_details.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            })
            torch.cuda.reset_peak_memory_stats()

    def infer(self, pose: PoseWindow, *, num_beams: int, max_new_tokens: int) -> tuple[str, dict[str, float]]:
        if self._model is None or self._torch is None or self._tokenizer is None:
            self.load()
        torch = self._torch
        if self.device_name == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        source: dict[str, Any] = {}
        for name, value in pose.parts.items():
            tensor = torch.from_numpy(value.copy()).unsqueeze(0).to(self.device_name)
            source[name] = tensor.to(self._torch_dtype)
        source["attention_mask"] = torch.ones((1, len(pose.sampled_indices)),
                                              dtype=torch.long, device=self.device_name)
        source["src_length_batch"] = torch.tensor([len(pose.sampled_indices)],
                                                  dtype=torch.long, device=self.device_name)
        # The upstream forward computes a training loss even for online inference.
        # A non-data sentinel keeps that unused label path non-empty; generation
        # consumes only the visual embeddings and never this sentinel.
        target = {"gt_sentence": ["."], "gt_gloss": [""]}
        with torch.inference_mode():
            stack_out = self._model(source, target)
            if self.device_name == "cuda":
                torch.cuda.synchronize()
            forward_done = time.perf_counter()
            generated = self._model.generate(stack_out, max_new_tokens=max_new_tokens, num_beams=num_beams)
            if self.device_name == "cuda":
                torch.cuda.synchronize()
            decode_started = time.perf_counter()
            text = self._tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)[0].strip()
            decode_done = time.perf_counter()
        timings = {
            "model_forward_ms": (forward_done - started) * 1000,
            "generation_ms": (decode_started - forward_done) * 1000,
            "decode_ms": (decode_done - decode_started) * 1000,
            "model_total_ms": (decode_done - started) * 1000,
        }
        return text, timings

    @property
    def runtime_info(self) -> dict[str, Any]:
        info = {
            "backend": "pytorch_uni_sign",
            "providers": [f"PyTorch {self.device_name.upper()}"],
            "device": self.device_name,
            "dtype": self.dtype_name,
            "load_ms": self.load_ms,
            **self.load_details,
        }
        if self.device_name == "cuda" and self._torch is not None:
            info.update({
                "gpu_peak_allocated_mb": self._torch.cuda.max_memory_allocated() / 1024**2,
                "gpu_peak_reserved_mb": self._torch.cuda.max_memory_reserved() / 1024**2,
            })
        return info

    def close(self) -> None:
        torch = self._torch
        self._model = None
        self._tokenizer = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class UniSignBackend:
    """One loaded Uni-Sign instance with a bounded, reuse-oriented pose cache."""

    def __init__(self, config: UniSignConfig, options: RuntimeOptions, *,
                 runtime_factory: Callable[..., Any] | None = None,
                 extractor_factory: Callable[..., Any] | None = None):
        if not isinstance(config, UniSignConfig):
            raise ValueError("UniSignConfig is required")
        if not isinstance(options, RuntimeOptions):
            raise ValueError("RuntimeOptions is required")
        self.config = config
        self.options = options
        self._runtime_factory = runtime_factory or TorchUniSignRuntime
        self._extractor_factory = extractor_factory
        self._lock = threading.RLock()
        self._runtime = None
        self._extractor = None
        self._loaded = False
        self._closed = False
        self._load_ms = None
        self._model_warmup_ms = None
        self._cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
        self._cache_limit = 512
        self._source_id = None
        self._last_window_first = None
        self._stabilizer = TextStabilizer(int(config.runtime.get("stabilize_after", 2)))
        self._resolve_runtime_settings()
        self.bundle = SimpleNamespace(
            path=config.bundle_dir,
            manifest=SimpleNamespace(
                run_id=config.run_id, backend=BACKEND_ID, task_type="text",
                sign_language="csl", capabilities=("clip",),
                input_spec=SimpleNamespace(representation_id=REPRESENTATION_ID),
            ),
        )
        self.capabilities = ModelCapabilities(
            model_id=config.run_id,
            model_version=config.checkpoint_sha256,
            sign_language="csl",
            task_type="text",
            inputs=(ModelInputContract(REPRESENTATION_ID, (None, None, None, 3), "uint8",
                                       f"uni_sign_rtmpose133_v1:{config.source_commit}",
                                       ("file", "captured_clip")),),
            vocabulary=(),
            runtime="pytorch",
            devices=("cpu", "cuda"),
            modes=("clip",),
            score_type=SCORE_TYPE,
        )

    def _resolve_runtime_settings(self) -> None:
        runtime = self.config.runtime
        requested = self.options.device
        configured = str(runtime.get("device", "auto"))
        device = configured if requested == "auto" else requested
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported Uni-Sign device: {device}")
        configured_dtype = str(runtime.get("dtype", "auto"))
        if configured_dtype == "auto":
            dtype = "bfloat16" if device == "cuda" else "float32"
        else:
            dtype = configured_dtype
        if dtype == "bfloat16" and device != "cuda":
            dtype = "float32"
        self.device = device
        self.dtype = dtype
        self.max_frames = int(runtime.get("max_frames", 256))
        self.num_beams = int(runtime.get("num_beams", 4))
        self.max_new_tokens = int(runtime.get("max_new_tokens", 100))
        self.pose_mode = str(runtime.get("pose_mode", "lightweight"))
        self.pose_normalization = str(
            runtime.get("pose_normalization", "isotropic_short_side")
        )
        self.pose_orientation = str(runtime.get("pose_orientation", "preserve"))
        self.pose_backend = str(runtime.get("pose_backend", "onnxruntime"))
        self.pose_device = device if runtime.get("pose_device", "auto") == "auto" else str(runtime.get("pose_device"))
        self.max_video_frames = int(runtime.get("max_video_frames", 600))
        sample_only = runtime.get("pose_sample_only", False)
        if not isinstance(sample_only, bool):
            raise ValueError("pose_sample_only must be a boolean")
        self.pose_sample_only = sample_only
        detect_stride = runtime.get("pose_detect_stride", 1)
        if type(detect_stride) is not int or not 1 <= detect_stride <= 30:
            raise ValueError("pose_detect_stride must be an integer in [1, 30]")
        self.pose_detect_stride = detect_stride
        cache_value = Path(runtime.get("rtmlib_cache_dir", "../rtmlib-cache"))
        self.rtmlib_cache_dir = ((self.config.bundle_dir / cache_value).resolve()
                                 if not cache_value.is_absolute() else cache_value)
        if not 2 <= self.max_frames <= 1000 or not 1 <= self.num_beams <= 10:
            raise ValueError("Invalid Uni-Sign frame/beam settings")
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError("Invalid Uni-Sign max_new_tokens")
        if self.pose_normalization not in {"isotropic_short_side", "wh"}:
            raise ValueError(
                "pose_normalization must be isotropic_short_side or wh"
            )
        if self.pose_orientation not in {"preserve", "mirror"}:
            raise ValueError("pose_orientation must be preserve or mirror")
        if self.pose_backend != "onnxruntime":
            raise ValueError("Only the pinned onnxruntime pose backend is supported")
        if self.max_video_frames < self.max_frames or self.max_video_frames > 6000:
            raise ValueError("max_video_frames must fit the model window and be <= 6000")

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded and not self._closed

    @property
    def runtime_info(self) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime.runtime_info if self._runtime is not None else {}
            extractor = self._extractor.runtime_info if self._extractor is not None else {}
            info = {
                "backend": "pytorch_uni_sign",
                "providers": runtime.get("providers", [f"PyTorch {self.device.upper()}"]),
                "device": self.device,
                "dtype": self.dtype,
                "load_ms": self._load_ms,
                "model_warmup_ms": self._model_warmup_ms,
                "checkpoint": str(self.config.checkpoint),
                "checkpoint_sha256": self.config.checkpoint_sha256,
                "mt5_dir": str(self.config.mt5_dir),
                "source_dir": str(self.config.source_dir),
                "source_commit": self.config.source_commit,
                "model_revision": self.config.model_revision,
                "license": MODEL_LICENSE,
                "source_license_status": SOURCE_LICENSE_STATUS,
                "pose_normalization": self.pose_normalization,
                "pose_orientation": self.pose_orientation,
                "pose_mode": self.pose_mode,
                "pose_sample_only": self.pose_sample_only,
                "pose_detect_stride": self.pose_detect_stride,
                "max_frames": self.max_frames,
                "streaming": "sliding_window_over_complete_windows",
            }
            info.update(extractor)
            info.update(runtime)
            return info

    def get_model_info(self) -> dict[str, Any]:
        try:
            manifest = read_json(self.config.bundle_dir / "manifest.json")
        except OSError:
            manifest = {}
        return {
            "run_id": self.config.run_id,
            "display_name": self.config.display_name,
            "backend": BACKEND_ID,
            "task_type": "text",
            "sign_language": "csl",
            "loaded": self.loaded,
            "runtime": self.runtime_info,
            "license": self.config.license,
            "provenance": manifest.get("provenance", {}),
        }

    def load_model(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Uni-Sign backend is closed")
            if self._loaded:
                return self.get_model_info()
            checkpoint_check = verify_checkpoint(self.config.checkpoint)
            if not checkpoint_check.ok:
                raise ValueError(
                    f"Uni-Sign checkpoint is incomplete or corrupt: {checkpoint_check.reason}"
                )
            mt5_checks = verify_mt5_dir(self.config.mt5_dir)
            if not all_checks_ok(mt5_checks):
                failed = [name for name, check in mt5_checks.items() if not check.ok]
                raise ValueError(f"mT5 files are missing or corrupt: {failed}")
            started = time.perf_counter()
            runtime = None
            extractor = None
            try:
                # On Windows, importing PyTorch before ONNX Runtime keeps each
                # runtime's bundled CUDA/cuDNN DLL family isolated. The reverse
                # order makes torch's cudnn_cnn64_9.dll fail with WinError 127.
                runtime = self._runtime_factory(self.config, device=self.device, dtype=self.dtype)
                runtime.load()
                warmup_image = self.config.source_dir / "demo/rtmlib-main/demo.jpg"
                if self._extractor_factory is None:
                    extractor = RtmposeExtractor(
                        mode=self.pose_mode, backend=self.pose_backend,
                        device=self.pose_device, cache_dir=self.rtmlib_cache_dir,
                        dll_dirs=self.options.cuda_dll_dirs,
                        strict_cuda=self.options.device == "cuda",
                        warmup_image=warmup_image,
                        detect_stride=self.pose_detect_stride)
                else:
                    extractor = self._extractor_factory(
                        mode=self.pose_mode, backend=self.pose_backend,
                        device=self.pose_device, cache_dir=self.rtmlib_cache_dir,
                        dll_dirs=self.options.cuda_dll_dirs,
                        strict_cuda=self.options.device == "cuda",
                        warmup_image=warmup_image,
                        detect_stride=self.pose_detect_stride)
                extractor_warmup = getattr(extractor, "warmup", None)
                if callable(extractor_warmup):
                    extractor_warmup()
                warmup_started = time.perf_counter()
                runtime.infer(self._warmup_pose(), num_beams=self.num_beams, max_new_tokens=8)
                self._model_warmup_ms = (time.perf_counter() - warmup_started) * 1000
            except BaseException:
                for resource in (runtime, extractor):
                    close = getattr(resource, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            pass
                raise
            self._runtime = runtime
            self._extractor = extractor
            self._loaded = True
            self._load_ms = (time.perf_counter() - started) * 1000
            return self.get_model_info()

    warmup = load_model

    def _warmup_pose(self):
        import numpy as np
        frames = min(48, self.max_frames)
        points = np.zeros((frames, 1, RTMPOSE_KEYPOINT_COUNT, 2), dtype=np.float32)
        scores = np.ones((frames, 1, RTMPOSE_KEYPOINT_COUNT), dtype=np.float32)
        for index in range(frames):
            phase = index / max(1, frames - 1)
            points[index, 0, 0] = [0.3 + 0.05 * phase, 0.3]
            for offset, joint in enumerate(range(3, 11)):
                points[index, 0, joint] = [0.25 + 0.04 * offset + 0.02 * phase, 0.35 + 0.015 * offset]
            points[index, 0, 91:112] = [0.4 + 0.01 * phase, 0.45]
            points[index, 0, 112:133] = [0.55 + 0.01 * phase, 0.45]
            points[index, 0, 23:40:2] = [0.45 + 0.005 * phase, 0.25]
            points[index, 0, 83:91] = [0.45 + 0.005 * phase, 0.3]
            points[index, 0, 53] = [0.45, 0.25]
        return prepare_pose_window(points, scores, max_frames=self.max_frames)

    def unload_model(self) -> None:
        with self._lock:
            runtime, extractor = self._runtime, self._extractor
            self._runtime = self._extractor = None
            self._loaded = False
            self._cache.clear()
            self._stabilizer.reset()
        for resource in (extractor, runtime):
            close = getattr(resource, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        gc.collect()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.unload_model()

    def _clear_for_source(self, source_id: str, first_index: int) -> None:
        if (source_id != self._source_id or self._last_window_first is None
                or first_index < self._last_window_first):
            self._cache.clear()
            self._source_id = source_id
            reset = getattr(self._extractor, "reset_detection", None)
            if callable(reset):
                reset()
        self._last_window_first = first_index

    def _first_person_pose(self, raw_points: Any, raw_scores: Any, *, width: int, height: int) -> tuple[Any, Any]:
        import numpy as np

        points = np.asarray(raw_points, dtype=np.float32)
        scores = np.asarray(raw_scores, dtype=np.float32)
        if points.ndim == 2 and points.shape == (RTMPOSE_KEYPOINT_COUNT, 2):
            points = points[None, ...]
        elif points.ndim == 3 and points.shape[-2:] == (RTMPOSE_KEYPOINT_COUNT, 2):
            points = points[:1]
        else:
            points = np.zeros((1, RTMPOSE_KEYPOINT_COUNT, 2), dtype=np.float32)
        if scores.ndim == 1 and scores.shape == (RTMPOSE_KEYPOINT_COUNT,):
            scores = scores[None, ...]
        elif scores.ndim == 2 and scores.shape[-1] == RTMPOSE_KEYPOINT_COUNT:
            scores = scores[:1]
        else:
            scores = np.zeros((1, RTMPOSE_KEYPOINT_COUNT), dtype=np.float32)
        points = normalize_rtmpose_pixels(
            points, width, height, mode=self.pose_normalization)
        return points[0], scores[0]

    def _extract_cached(self, frame: Any, *, source_id: str, frame_index: int) -> tuple[Any, Any, bool]:
        if self._extractor is None:
            raise RuntimeError("Uni-Sign pose extractor is not loaded")
        height, width = frame.shape[:2]
        key = (source_id, int(frame_index), int(height), int(width))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached[0], cached[1], True
        raw_points, raw_scores = self._extractor.extract(frame)
        point, score = self._first_person_pose(raw_points, raw_scores, width=width, height=height)
        value = (point, score)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return value[0], value[1], False

    def _prepare_frames(self, frames: list[tuple[Any, int]], *, source_id: str) -> tuple[PoseWindow, dict[str, Any]]:
        if len(frames) < 2:
            raise ValueError("Uni-Sign requires at least two video frames")
        self._clear_for_source(source_id, int(frames[0][1]))
        new_frames = hit_frames = 0
        points, scores = [], []
        extract_started = time.perf_counter()
        for image, index in frames:
            point, score, hit = self._extract_cached(image, source_id=source_id, frame_index=index)
            hit_frames += int(hit)
            new_frames += int(not hit)
            points.append(point)
            scores.append(score)
        extract_ms = (time.perf_counter() - extract_started) * 1000
        preprocess_started = time.perf_counter()
        pose = prepare_pose_window(points, scores, max_frames=self.max_frames)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        metrics = {
            "pose_extract_ms": extract_ms,
            "pose_preprocess_ms": preprocess_ms,
            "pose_ms": extract_ms + preprocess_ms,
            "pose_cache_hit_frames": hit_frames,
            "pose_new_frames": new_frames,
            "window_frames": len(frames),
            "sampled_frames": len(pose.sampled_indices),
        }
        return pose, metrics

    def _video_pose_window(self, path: str | Path) -> tuple[PoseWindow, dict[str, Any], tuple[float, float]]:
        import cv2

        video = Path(path).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Video does not exist: {video}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video}")
        points, scores = [], []
        new_frames = hit_frames = 0
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Video has no valid FPS")
            if reported > self.max_video_frames:
                raise ValueError(f"Video has {reported} frames; limit is {self.max_video_frames}")
            source_id = f"file:{video}"
            self._clear_for_source(source_id, 0)
            # When the clip is longer than the model window, only the frames that
            # survive ``prepare_pose_window`` are worth extracting. The plan comes
            # from the container's frame count; skipped frames are grabbed without
            # being decoded to BGR, so both decode and RTMPose cost drop.
            sampled_indices = None
            if self.pose_sample_only and reported >= 2:
                planned = deterministic_frame_indices(reported, max_frames=self.max_frames)
                if len(planned) < reported:
                    sampled_indices = planned
            index = 0
            extract_started = time.perf_counter()
            if sampled_indices is None:
                while index < self.max_video_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    point, score, hit = self._extract_cached(frame, source_id=source_id, frame_index=index)
                    hit_frames += int(hit)
                    new_frames += int(not hit)
                    points.append(point)
                    scores.append(score)
                    index += 1
            else:
                wanted = set(sampled_indices)
                while index < self.max_video_frames:
                    if not capture.grab():
                        break
                    if index in wanted:
                        ok, frame = capture.retrieve()
                        if not ok:
                            break
                        point, score, hit = self._extract_cached(frame, source_id=source_id, frame_index=index)
                        hit_frames += int(hit)
                        new_frames += int(not hit)
                        points.append(point)
                        scores.append(score)
                    index += 1
            if index >= self.max_video_frames:
                ok, _ = capture.read()
                if ok:
                    raise ValueError(f"Video exceeds {self.max_video_frames} frames")
            extract_ms = (time.perf_counter() - extract_started) * 1000
            if index < 2:
                raise ValueError("Video contains fewer than two decodable frames")
            preprocess_started = time.perf_counter()
            if sampled_indices is None:
                pose = prepare_pose_window(points, scores, max_frames=self.max_frames)
            else:
                # A container that over-reports its frame count ends the grab loop
                # early; indices are ascending, so the prefix stays correct.
                effective = tuple(sampled_indices[:len(points)])
                if len(effective) < 2:
                    raise ValueError("Video contains fewer than two decodable frames")
                window = prepare_pose_window(points, scores, max_frames=len(effective))
                pose = PoseWindow(
                    body=window.body, left=window.left, right=window.right,
                    face_all=window.face_all, sampled_indices=effective,
                    quality=dict(window.quality,
                                 source_frames=index,
                                 sampled_indices=list(effective),
                                 pose_sample_only=True,
                                 skipped_frames=index - len(effective)),
                )
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            metrics = {
                "pose_extract_ms": extract_ms,
                "pose_preprocess_ms": preprocess_ms,
                "pose_ms": extract_ms + preprocess_ms,
                "pose_cache_hit_frames": hit_frames,
                "pose_new_frames": new_frames,
                "window_frames": index,
                "sampled_frames": len(pose.sampled_indices),
                "pose_sample_only": sampled_indices is not None,
                "pose_skipped_frames": (index - len(pose.sampled_indices)
                                        if sampled_indices is not None else 0),
            }
            return pose, metrics, (0.0, (index - 1) / fps)
        finally:
            capture.release()

    def _clip_pose_window(self, clip: Any) -> tuple[PoseWindow, dict[str, Any], tuple[float, float]]:
        frames = list(getattr(clip, "frames", ()))
        if len(frames) < 2:
            raise ValueError("Captured clip contains fewer than two frames")
        indexed = [(frame.image, int(frame.frame_index)) for frame in frames]
        pose, metrics = self._prepare_frames(indexed, source_id=str(clip.source_id))
        start = float(frames[0].timestamp)
        end = float(frames[-1].timestamp)
        return pose, metrics, (start, end)

    def _make_result(self, *, request: InferenceRequest, pose: PoseWindow, text: str,
                     timings: dict[str, float], metrics: dict[str, Any],
                     time_range: tuple[float, float], refused_reason: str | None = None) -> RecognitionResult:
        stability = self._stabilizer.update(text or "", at=time.monotonic())
        runtime_info = self._runtime.runtime_info if self._runtime is not None else {}
        duration = max(0.0, float(time_range[1]) - float(time_range[0]))
        total_ms = float(timings.get("adapter_total_ms", 0.0))
        quality = {
            **pose.quality,
            **metrics,
            **{key: float(value) for key, value in timings.items()},
            "window_duration_seconds": duration,
            "processing_fps": (int(metrics.get("window_frames", 0)) / (total_ms / 1000.0)
                               if total_ms > 0 else None),
            "real_time_factor": (total_ms / 1000.0 / duration if duration > 0 else None),
            "stability_state": stability["state"],
            "stable_repeat_count": stability["repeat_count"],
            "text_changed": stability["changed"],
            "text_duplicate": stability["duplicate"],
            "text_previous": stability["previous_text"],
            "text_normalized": stability["normalized_text"],
            "text_revision": stability["revision"],
            "runtime_device": self.device,
            "runtime_dtype": self.dtype,
            "license": MODEL_LICENSE,
            "streaming": False,
            "continuous_candidate": request.request_id.startswith("continuous-"),
        }
        for key in ("gpu_peak_allocated_mb", "gpu_peak_reserved_mb"):
            if key in runtime_info:
                quality[key] = float(runtime_info[key])
        if refused_reason:
            return RecognitionResult(
                status="rejected", sample_id=request.request_id, model_id=self.config.run_id,
                sign_language="csl", prediction=None, quality=quality, reason=refused_reason,
                task_type="text", score_type=SCORE_TYPE, time_range=time_range)
        if not text or not text.strip():
            quality["empty_text_output"] = True
            return RecognitionResult(
                status="rejected", sample_id=request.request_id, model_id=self.config.run_id,
                sign_language="csl", prediction=None, quality=quality, reason="empty_text_output",
                task_type="text", score_type=SCORE_TYPE, time_range=time_range)
        return RecognitionResult(
            status="recognized", sample_id=request.request_id, model_id=self.config.run_id,
            sign_language="csl", prediction=TextPrediction(text=text.strip(), language="zh", score=None),
            quality=quality, task_type="text", score_type=SCORE_TYPE, time_range=time_range)

    def recognize_clip(self, clip: Any, request: InferenceRequest) -> RecognitionResult:
        request.check_active()
        request_started = time.perf_counter()
        with self._lock:
            if not self._loaded:
                self.load_model()
            pose, metrics, time_range = self._clip_pose_window(clip)
            pose = orient_pose_window(pose, self.pose_orientation)
            if pose.quality["valid_body_fraction"] < 0.1:
                return self._make_result(request=request, pose=pose, text="", timings={}, metrics=metrics,
                                         time_range=time_range, refused_reason="insufficient_body_frames")
            request.check_active()
            runtime = self._runtime
            text, timings = runtime.infer(pose, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
            timings["adapter_total_ms"] = (time.perf_counter() - request_started) * 1000
            request.check_active()
            return self._make_result(request=request, pose=pose, text=text, timings=timings,
                                     metrics=metrics, time_range=time_range)

    def recognize_file(self, path: str | Path, request: InferenceRequest) -> RecognitionResult:
        request.check_active()
        request_started = time.perf_counter()
        with self._lock:
            if not self._loaded:
                self.load_model()
            self._stabilizer.reset()
            pose, metrics, time_range = self._video_pose_window(path)
            pose = orient_pose_window(pose, self.pose_orientation)
            if pose.quality["valid_body_fraction"] < 0.1:
                return self._make_result(request=request, pose=pose, text="", timings={}, metrics=metrics,
                                         time_range=time_range, refused_reason="insufficient_body_frames")
            text, timings = self._runtime.infer(pose, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
            timings["adapter_total_ms"] = (time.perf_counter() - request_started) * 1000
            request.check_active()
            return self._make_result(request=request, pose=pose, text=text, timings=timings,
                                     metrics=metrics, time_range=time_range)


class UniSignAdapter:
    """Typed runtime adapter; construction validates files but defers GPU loading."""

    def __init__(self, config: UniSignConfig, options: RuntimeOptions, *,
                 runtime_factory: Callable[..., Any] | None = None,
                 extractor_factory: Callable[..., Any] | None = None):
        self._backend = UniSignBackend(config, options, runtime_factory=runtime_factory,
                                       extractor_factory=extractor_factory)
        self.capabilities = self._backend.capabilities

    @property
    def recognizer(self) -> UniSignBackend:
        return self._backend

    @property
    def runtime_info(self) -> dict[str, Any]:
        return self._backend.runtime_info

    def warmup(self) -> None:
        self._backend.load_model()

    def infer(self, source: Any, request: InferenceRequest, *, input_kind: str) -> tuple[RecognitionResult, dict[str, float]]:
        if input_kind == "file":
            result = self._backend.recognize_file(source, request)
        elif input_kind == "captured_clip":
            result = self._backend.recognize_clip(source, request)
        else:
            raise ValueError("Uni-Sign supports file and captured_clip inputs only")
        timing = {
            "pose_ms": float(result.quality.get("pose_ms", 0.)),
            "inference_ms": float(result.quality.get("model_total_ms", 0.)),
            "total_ms": float(result.quality.get("adapter_total_ms", 0.)),
        }
        return result, timing

    def close(self) -> None:
        self._backend.close()


def load_uni_sign_adapter(path: str | Path, *, options: RuntimeOptions | None = None) -> UniSignAdapter:
    config = UniSignConfig.from_bundle(path)
    return UniSignAdapter(config, options or RuntimeOptions())


def unisign_bundle_available(path: str | Path) -> bool:
    try:
        UniSignConfig.from_bundle(path)
    except (OSError, ValueError):
        return False
    return (Path(path) / "manifest.json").is_file()












