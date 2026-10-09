"""Opt-in native runner for the audited, externally supplied CE-CSL TFNet.

Upstream code and weights are not bundled. ``source_dir`` is an explicit trust
boundary: these five audited Python source files execute in this process. No
training entrypoint, dataset loader, tracked bytecode, or ImageNet download is
used. Importing this adapter itself does not import PyTorch or upstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Mapping

import numpy as np


TFNET_SOURCE_COMMIT = "ee7cb6a253a711e8f3699a1e8e0943179268060c"
TFNET_CHECKPOINT_SHA256 = "160d35a42d9f8f52a090d0131586df1631d2fcf8c9f41ee94729cae1fb6b0386"
TFNET_CLASSES = 3516
TFNET_HIDDEN_SIZE = 1024
_SOURCE_MODULES = ("Module", "BiLSTM", "Transformer", "SEN", "Net")
_SOURCE_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), *args], capture_output=True,
        text=True, encoding="utf-8", check=False,
    )
    if result.returncode:
        raise ValueError(f"Cannot verify trusted TFNet source: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_tfnet_source(source_dir: str | Path) -> dict[str, Any]:
    """Require the fixed Git commit and unchanged audited model source files."""
    source = Path(source_dir).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("TFNet source_dir must be a directory")
    if _git(source, "rev-parse", "HEAD") != TFNET_SOURCE_COMMIT:
        raise ValueError(f"TFNet source must be pinned to {TFNET_SOURCE_COMMIT}")
    files = [f"{name}.py" for name in _SOURCE_MODULES]
    # Git handles checkout newline conversion. Other files (including the
    # upstream case-colliding tracked pycache) are outside this execution path.
    changed = _git(source, "diff", "--name-only", TFNET_SOURCE_COMMIT, "--", *files)
    if changed:
        raise ValueError(f"Audited TFNet source was modified: {changed}")
    return {
        "source_dir": str(source), "source_commit": TFNET_SOURCE_COMMIT,
        "source_files_sha256": {name: _sha256(source / name) for name in files},
    }


def _load_source_model(source_dir: Path, device: Any) -> Any:
    """Execute only audited source text, temporarily resolving its bare imports."""
    with _SOURCE_LOCK:
        namespace = "_slbridge_tfnet_" + hashlib.sha256(str(source_dir).encode()).hexdigest()[:16]
        previous = {name: sys.modules.get(name) for name in _SOURCE_MODULES}
        loaded: dict[str, ModuleType] = {}
        try:
            for name in _SOURCE_MODULES:
                filename = source_dir / f"{name}.py"
                module = ModuleType(f"{namespace}.{name}")
                module.__file__ = str(filename)
                module.__package__ = namespace
                sys.modules[module.__name__] = module
                sys.modules[name] = module
                loaded[name] = module
                # Reading/compiling avoids importing stale or tracked pyc and
                # does not write __pycache__ into the external source checkout.
                exec(compile(filename.read_bytes(), str(filename), "exec"), module.__dict__)
            upstream_module = loaded["Module"]
            original = upstream_module.resnet34MAM
            try:
                upstream_module.resnet34MAM = lambda **kwargs: upstream_module.ResNet34MAM(
                    upstream_module.BasicBlock, [3, 4, 6, 3], **kwargs,
                )
                model = loaded["Net"].moduleNet(
                    hiddenSize=TFNET_HIDDEN_SIZE, wordSetNum=TFNET_CLASSES,
                    moduleChoice="TFNet", device=device, dataSetName="CE-CSL", isFlag=False,
                )
            finally:
                upstream_module.resnet34MAM = original
            return model
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


def _load_checkpoint(path: Path, torch: Any) -> Mapping[str, Any]:
    """Weights-only loading with the audited float64 NumPy scalar allowlist.

    The upstream training metadata uses numpy.core.multiarray.scalar, np.dtype
    and the concrete float64 dtype class. No arbitrary globals or unsafe pickle
    retry is permitted. This allowlist is scoped to this load call.
    """
    numpy_core = "numpy._core.multiarray" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core.multiarray"
    scalar = importlib.import_module(numpy_core).scalar
    # NumPy 2 renamed this same scalar constructor. The old checkpoint name
    # remains explicit; the current spelling supports equivalent safe metadata.
    allowed = [(scalar, "numpy.core.multiarray.scalar"), (scalar, "numpy._core.multiarray.scalar"),
               np.dtype, type(np.dtype(np.float64))]
    try:
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"TFNet weights-only checkpoint load failed: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("TFNet checkpoint must be a mapping")
    if not isinstance(checkpoint.get("moduleNet_state_dict"), Mapping):
        raise ValueError("TFNet checkpoint has no moduleNet_state_dict")
    return checkpoint


def validate_tfnet_input(values: Any, lengths: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate already preprocessed normalized BLCHW values and CPU lengths."""
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise ValueError("TFNet values must have dtype float32")
    if array.ndim != 5 or array.shape[0] < 1 or array.shape[2:] != (3, 224, 224):
        raise ValueError("TFNet values must have shape B,L,3,224,224")
    batch, padded_length = array.shape[:2]
    if padded_length < 16 or padded_length % 4:
        raise ValueError("TFNet padded length must be >=16 and aligned to four")
    if not np.isfinite(array).all() or array.min() < -1.000001 or array.max() > 1.000001:
        raise ValueError("TFNet values must be finite and normalized to [-1,1]")
    if lengths is None:
        effective = np.full(batch, padded_length, dtype=np.int64)
    else:
        effective = np.asarray(lengths)
        if effective.shape != (batch,) or effective.dtype.kind not in "iu":
            raise ValueError("TFNet lengths must be an integer vector of size B")
        effective = effective.astype(np.int64, copy=False)
    if np.any(effective < 16) or np.any(effective > padded_length) or np.any(effective % 4):
        raise ValueError("TFNet effective lengths must be aligned and within padded storage")
    if np.any(effective[:-1] < effective[1:]):
        raise ValueError("TFNet batch lengths must be sorted in descending order")
    # PreparedRGB deliberately freezes its storage. torch.from_numpy cannot
    # safely wrap read-only arrays; preserve the zero-copy path for writable
    # contiguous inputs and copy only immutable/noncontiguous storage.
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=np.float32, order="C", copy=True)
    return array, np.ascontiguousarray(effective)


@dataclass(frozen=True)
class TFNetOutput:
    scores: np.ndarray
    lengths: np.ndarray
    elapsed_seconds: float

    @property
    def logits(self) -> np.ndarray:
        return self.scores


class TFNetNativeRunner:
    """Strictly loaded upstream model exposing fused raw scores, never decoding."""

    def __init__(
        self, checkpoint_path: str | Path, source_dir: str | Path, *, device: str = "cpu",
        expected_checkpoint_sha256: str = TFNET_CHECKPOINT_SHA256,
    ) -> None:
        if expected_checkpoint_sha256 != TFNET_CHECKPOINT_SHA256:
            raise ValueError("This adapter accepts only the audited TFNet checkpoint identity")
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        source = verify_tfnet_source(source_dir)
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != expected_checkpoint_sha256:
            raise ValueError("TFNet checkpoint SHA-256 differs from the audited identity")
        try:
            import torch
        except ImportError as error:
            raise ValueError("Native TFNet inference requires optional PyTorch dependencies") from error
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Native TFNet supports explicit cpu or cuda devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but PyTorch reports no available CUDA device")
        self._lock = threading.RLock()
        self.model = _load_source_model(Path(source["source_dir"]), self.device)
        checkpoint = _load_checkpoint(checkpoint_path, torch)
        state = checkpoint["moduleNet_state_dict"]
        expected = self.model.state_dict()
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        if missing or unexpected:
            raise ValueError(f"TFNet state keys differ; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError(f"TFNet state tensor shape/dtype differs: {key}")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.provenance = {
            **source, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": actual_hash,
            "strict_state_tensor_count": len(state), "classes": TFNET_CLASSES,
            "device": str(self.device), "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "imagenet_download": False,
            "weights_only": True, "eval_mode": True, "isTrain": False,
            "precision": {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                          "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                          "dtype": "float32", "captured_at": "model_load",
                          "policy": "caller_controlled; flags may be changed after this snapshot"},
        }

    def predict(self, values: Any, lengths: Any = None) -> TFNetOutput:
        array, effective = validate_tfnet_input(values, lengths)
        torch = self._torch
        with self._lock, torch.inference_mode():
            if self.model is None:
                raise RuntimeError("TFNet runner is closed")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            tensor = torch.from_numpy(array).to(self.device)
            # Upstream collation passes a list of one-element CPU tensors.
            # TemporalConv.update_lgt preserves those dimensions before cat;
            # passing a flat tensor would produce zero-dimensional entries.
            length_tensors = [torch.from_numpy(effective[index:index + 1].copy())
                              for index in range(len(effective))]
            self.model.eval()
            outputs = self.model(tensor, length_tensors, isTrain=False)
            scores = outputs[0].detach().float().cpu().numpy().copy()
            output_lengths = outputs[5].detach().cpu().numpy().astype(np.int64, copy=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
        expected_lengths = (effective - 12) // 4
        if (scores.shape != (int(expected_lengths.max()), array.shape[0], TFNET_CLASSES)
                or not np.array_equal(output_lengths, expected_lengths) or not np.isfinite(scores).all()):
            raise ValueError("TFNet fused output violated expected TBV/length/finite contract")
        return TFNetOutput(scores=scores, lengths=output_lengths, elapsed_seconds=elapsed)

    def close(self) -> None:
        with self._lock:
            self.model = None
