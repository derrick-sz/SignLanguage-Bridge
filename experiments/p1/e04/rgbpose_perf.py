"""task-34 -- RGB+pose production path performance vs pose-only video path A.

Lead config update (2026-10-27): the pinned ``csl_daily_rgb_pose_slt.pth`` has an
all-zero ``fusion_gate`` (zero-init + Tanh -> ReLU), so the RGB branch has **zero
effect** on the output (``fused_post = gcn``).  The sensible production form is
therefore ``hand_conf_layout="legacy"`` -- the model still defines the RGB branch
(``enable_rgb=True``) but no hand crops are computed.  A second arm with
``hand_conf_layout="squeeze"`` (real hand crops) quantifies the wasted
preprocessing cost that the zero gate discards.

Three arms, same pilot samples, same order, single process, rotating order to
avoid GPU-clock drift:

    a                existing pose-only video path (unisign_pose_slt, stage1)
    rgbpose_legacy   RGB+pose adapter, hand_conf_layout="legacy"  (production form)
    rgbpose_squeeze  RGB+pose adapter, hand_conf_layout="squeeze" (waste control)

Both RGB+pose arms run through the production shared ``ModelAdapter`` registry with
``pose_source="rtmpose"`` (same RTMPose frontend as A), ``enable_rgb=True`` and
``fallback_to_pose_only=False`` (measurement must never silently fall back to A).

Per run: normalised segments ``video_decode_ms / pose_extract_ms / hand_crop_ms /
model_forward_ms / text_decode_ms / preprocess_total_ms / e2e_ms``, RTF (e2e /
container duration), torch peak/reserved VRAM, whole-device nvidia-smi sampling,
predictions (for legacy-vs-squeeze equivalence), failures/OOM retries.

Phases: ``cold`` (3 fresh subprocesses per arm), ``paired`` (20 samples x 2 rounds,
rotating arm order), ``continuous`` (2 minutes rotating arms).

Usage::

    .conda-env\\python.exe experiments/p1/e04/rgbpose_perf.py check
    .conda-env\\python.exe experiments/p1/e04/rgbpose_perf.py run \
        --run-dir experiments/p1/e04/rgbpose-perf-<run_id> \
        --samples 20 --rounds 2 --continuous-minutes 2
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1.e04 import perf_common as common  # noqa: E402

A_BUNDLE = ROOT / "models" / "external" / "uni_sign" / "model_bundle"
A_BACKEND_ID = "uni_sign_pose_slt"
RGB_POSE_BACKEND_ID = "unisign_rgb_pose_slt"
RGB_POSE_ENTRY = "slbridge.uni_sign_rgb_pose"
RGB_POSE_BUNDLE_CANDIDATES = (
    ROOT / "models" / "external" / "uni_sign_rgb_pose" / "model_bundle",
    ROOT / "models" / "external" / "uni_sign_rgb_pose_bundle",
    ROOT / "models" / "external" / "uni_sign_rgb_pose" / "bundle",
    ROOT / "models" / "external" / "uni_sign" / "rgb_pose_model_bundle",
)
RETRYABLE = ("out of memory", "cuda error", "memory allocation failure", "acceleratorerror")
RETRY_SLEEP_SECONDS = 20.0
SEGMENT_KEYS = ("video_decode_ms", "pose_extract_ms", "hand_crop_ms",
                "model_forward_ms", "text_decode_ms", "preprocess_total_ms", "e2e_ms")

# Gate-zero checkpoint: production form carries the RGB branch definition but skips
# hand crops; squeeze is the control that pays for crops the zero gate discards.
BASE_RGB_POSE_CONFIG: dict[str, Any] = {
    "pose_source": "rtmpose",
    "enable_rgb": True,
    "fallback_to_pose_only": False,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "rgbpose_legacy": {
        "hand_conf_layout": "legacy",
        "role": "production form: zero-gate RGB branch, no hand crops",
    },
    "rgbpose_squeeze": {
        "hand_conf_layout": "squeeze",
        "role": "waste control: real hand crops discarded by the zero gate",
    },
}
STAGE2_BUNDLE_DEFAULT = (ROOT / "experiments" / "p1" / "e01" / "runs"
                         / "stage2-pose-subset-pilot50-20261027" / "pose_bundle")
PATH_CHOICES: tuple[str, ...] = ("a", "b_stage2", "rgbpose_legacy", "rgbpose_squeeze")
PATHS: tuple[str, ...] = ("a", "b_stage2")  # active arms; main() rebinds from --arms
PAIR_DEFS: tuple[tuple[str, str], ...] = (
    ("a", "b_stage2"),
    ("b_stage2", "rgbpose_legacy"),
    ("a", "rgbpose_legacy"),
    ("rgbpose_legacy", "rgbpose_squeeze"),
    ("a", "rgbpose_squeeze"),
    ("b_stage2", "rgbpose_squeeze"),
)


def active_pair_defs() -> list[tuple[str, str]]:
    return [(first, second) for first, second in PAIR_DEFS
            if first in PATHS and second in PATHS]
PATH_LABELS = {
    "a": "unisign_pose_slt (A, pose-only video, stage1)",
    "b_stage2": "unisign_pose_slt + stage2 pose subset (B, product candidate)",
    "rgbpose_legacy": "unisign_rgb_pose_slt legacy (no hand crops, production form)",
    "rgbpose_squeeze": "unisign_rgb_pose_slt squeeze (real hand crops, waste control)",
}
# Frozen 1086 run: model-side cross-check only (official_archive preprocess is not
# comparable to the rtmpose production mode).
FROZEN_MODEL_REFERENCE = {
    "model_total_mean_ms": 376.4,
    "model_total_p50_ms": 369.8,
    "peak_allocated_mb": 1265.5,
    "peak_reserved_mb": 2700.0,
    "load_seconds": 5.4497,
    "note": "experiments/p1/e01/runs/unisign-rgb-pose-1086-20261027 (model side only)",
}
OVERRIDES: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# readiness / adapter resolution
# ---------------------------------------------------------------------------
def _rgb_pose_module():
    try:
        return importlib.import_module(RGB_POSE_ENTRY)
    except Exception:  # noqa: BLE001 - readiness reports instead of raising
        return None


def _bundle_manifest_backend(path: Path) -> str | None:
    manifest = Path(path) / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        payload = common.read_json(manifest)
    except Exception:  # noqa: BLE001
        return None
    return payload.get("backend") if isinstance(payload, dict) else None


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown RGB+pose variant {variant!r}")
    return {**BASE_RGB_POSE_CONFIG, **VARIANTS[variant]}


def discover_rgb_pose_entry() -> dict[str, Any]:
    module = _rgb_pose_module()
    symbols = sorted(name for name in dir(module)
                     if name.startswith("load_") or name.startswith("UniSign")) if module else []
    explicit = OVERRIDES.get("rgbpose_entry")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(RGB_POSE_BUNDLE_CANDIDATES)
    external = ROOT / "models" / "external"
    if external.is_dir():
        candidates.extend(path.parent for path in external.glob("*/**/manifest.json"))
    bundles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).resolve()) if Path(candidate).exists() else str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        backend = _bundle_manifest_backend(Path(candidate))
        if backend is not None:
            bundles.append({"path": str(candidate), "backend": backend})
    try:
        from slbridge.model_adapters import model_adapter_ids

        adapter_ids = model_adapter_ids()
        registry_error = None
    except Exception as error:  # noqa: BLE001
        adapter_ids = ()
        registry_error = f"{type(error).__name__}: {error}"
    return {
        "module": RGB_POSE_ENTRY,
        "module_present": module is not None,
        "module_symbols": symbols,
        "factory_candidates": [name for name in symbols if name.startswith("load_")],
        "registered_backend_ids": [name for name in adapter_ids
                                   if name == RGB_POSE_BACKEND_ID or "rgb" in name.lower()],
        "adapter_ids": tuple(adapter_ids),
        "registry_error": registry_error,
        "bundles": bundles,
        "base_config": dict(BASE_RGB_POSE_CONFIG),
        "variants": {name: variant_config(name) for name in VARIANTS},
    }


def readiness() -> dict[str, Any]:
    entry = discover_rgb_pose_entry()
    pilot_ready = 0
    try:
        pilot_ready = len(common.select_pilot_samples(20))
    except Exception:  # noqa: BLE001
        pilot_ready = 0
    a_ok = (A_BUNDLE / "manifest.json").is_file()
    stage2 = stage2_bundle()
    stage2_manifest = common.read_json(stage2 / "manifest.json") if (stage2 / "manifest.json").is_file() else {}
    checkpoint = Path(stage2_manifest.get("checkpoint", "")) if stage2_manifest else Path()
    stage2_ok = bool(stage2_manifest.get("backend") == A_BACKEND_ID and checkpoint.is_file())
    rgb_bundle_found = any(bundle.get("backend") == RGB_POSE_BACKEND_ID
                           for bundle in entry["bundles"])
    rgb_ready = bool(entry["module_present"] or entry["registered_backend_ids"] or rgb_bundle_found)
    return {
        "checked_at": common.utc_now(),
        "a_bundle": str(A_BUNDLE),
        "a_bundle_ok": a_ok,
        "a_registered": A_BACKEND_ID in entry["adapter_ids"],
        "rgb_pose": entry,
        "stage2_bundle": str(stage2),
        "stage2_bundle_ok": stage2_ok,
        "stage2_checkpoint": str(checkpoint) if stage2_manifest else None,
        "stage2_checkpoint_sha256": stage2_manifest.get("checkpoint_sha256"),
        "pilot_samples_available": pilot_ready,
        "ready": bool(a_ok and stage2_ok and pilot_ready >= 20),
        "frozen_model_reference": FROZEN_MODEL_REFERENCE,
    }


def build_runtime_options(device: str = "cuda", config: dict[str, Any] | None = None):
    import dataclasses

    from slbridge.runtime_options import RuntimeOptions

    config = config or BASE_RGB_POSE_CONFIG
    names = {field.name for field in dataclasses.fields(RuntimeOptions)}
    kwargs = {name: value for name, value in config.items() if name in names}
    return RuntimeOptions(device=device, **kwargs)


def _adapter_with_requested_config(adapter: Any, config: dict[str, Any]):
    """Apply measurement switches (hand_conf_layout / pose_source / fallback) to the
    production adapter config without touching the bundle manifest."""
    from dataclasses import fields, replace

    from slbridge.uni_sign_rgb_pose import UniSignRGBPoseAdapter, UniSignRGBPoseConfig

    base = getattr(adapter, "config", None)
    if not isinstance(base, UniSignRGBPoseConfig):
        return adapter
    names = {field.name for field in fields(base)}
    changes = {key: value for key, value in config.items() if key in names}
    target = replace(base, **changes)
    if target == base:
        return adapter
    try:
        adapter.close()
    except Exception:  # noqa: BLE001
        pass
    return UniSignRGBPoseAdapter(target)


def load_rgb_pose_adapter(config: dict[str, Any] | None = None):
    """Construct the RGB+pose adapter through production entry points.

    Returns (adapter, resolution).  Never falls back to path A: a failure here is
    reported as BLOCKED evidence instead of silently measuring A.
    """
    config = dict(config or BASE_RGB_POSE_CONFIG)
    entry = discover_rgb_pose_entry()
    if entry["registry_error"]:
        raise RuntimeError(f"adapter registry unavailable: {entry['registry_error']}")
    options = build_runtime_options(config=config)
    errors: list[str] = []
    from slbridge.model_adapters import load_model_adapter

    rgb_bundles = [bundle for bundle in entry["bundles"]
                   if bundle.get("backend") == RGB_POSE_BACKEND_ID]
    for bundle in rgb_bundles:
        try:
            adapter = load_model_adapter(Path(bundle["path"]), options=options)
            configured = _adapter_with_requested_config(adapter, config)
            return configured, {"route": "load_model_adapter", "bundle": bundle,
                                "config": config,
                                "config_overridden": configured is not adapter}
        except Exception as error:  # noqa: BLE001 - factory fallback
            errors.append(f"load_model_adapter({bundle['path']}): {type(error).__name__}: {error}")

    module = _rgb_pose_module()
    if module is not None:
        import inspect

        factory = None
        for name in ("load_unisign_rgb_pose_adapter", "load_uni_sign_rgb_pose_adapter",
                     "load_rgb_pose_adapter"):
            candidate = getattr(module, name, None)
            if callable(candidate):
                factory = (name, candidate)
                break
        if factory is not None:
            name, function = factory
            signature = inspect.signature(function)
            path_argument = None
            if rgb_bundles:
                path_argument = rgb_bundles[0]["path"]
            elif (ROOT / "models/external/uni_sign/csl_daily_rgb_pose_slt.pth").is_file():
                path_argument = str(ROOT / "models/external/uni_sign/csl_daily_rgb_pose_slt.pth")
            kwargs: dict[str, Any] = {"options": options}
            for config_name in ("config", "settings", "adapter_config"):
                if config_name in signature.parameters:
                    kwargs[config_name] = dict(config)
            for switch in config:
                if switch in signature.parameters:
                    kwargs[switch] = config[switch]
            attempts = []
            if path_argument is not None:
                attempts.append((path_argument,))
            attempts.append(())
            for positional in attempts:
                try:
                    adapter = function(*positional, **kwargs)
                    return adapter, {"route": f"factory:{name}",
                                     "path": positional[0] if positional else None,
                                     "kwargs": sorted(kwargs), "config": config}
                except TypeError as error:
                    errors.append(f"{name}{positional}: TypeError: {error}")
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{name}{positional}: {type(error).__name__}: {error}")
    raise RuntimeError(
        "RGB+pose adapter cannot be constructed yet; resolution errors: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _request(request_id: str):
    from slbridge.runtime_contracts import InferenceRequest

    return InferenceRequest(request_id=request_id,
                            deadline_monotonic=time.monotonic() + 300.0)


def _torch():
    import torch

    return torch


def _vram_snapshot() -> dict[str, Any]:
    torch = _torch()
    snapshot: dict[str, Any] = {}
    if torch.cuda.is_available():
        snapshot = {
            "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            "torch_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "scope": "process_torch_allocator",
        }
        free, total = torch.cuda.mem_get_info()
        snapshot["device_free_mb_at_read"] = round(free / 1024**2, 1)
        snapshot["device_total_mb"] = round(total / 1024**2, 1)
    return snapshot


def _reset_peak() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _number(quality: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = quality.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _round(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return None


def _parse_quality(quality: dict[str, Any], *, extract_ms: float | None = None) -> dict[str, Any]:
    quality = quality or {}
    pose_extract = _number(quality, "pose_extract_ms", "pose_ms")
    pose_preprocess = _number(quality, "pose_preprocess_ms")
    video_decode = _number(quality, "video_decode_ms")
    if video_decode is None and pose_extract is not None and extract_ms is not None:
        video_decode = max(0.0, pose_extract - extract_ms)
    hand_crop = _number(quality, "hand_crop_ms", "rgb_crop_ms", "handcrop_ms",
                        "support_rgb_ms", "rgb_preprocess_ms", "crop_ms")
    forward = _number(quality, "model_forward_ms", "forward_ms")
    generation = _number(quality, "generation_ms")
    decode = _number(quality, "decode_ms")
    text_decode = _number(quality, "text_decode_ms")
    if text_decode is None and generation is not None:
        text_decode = generation + (decode or 0.0)
    model_total = _number(quality, "model_total_ms")
    adapter_total = _number(quality, "adapter_total_ms")
    preprocess_total = _number(quality, "preprocess_total_ms", "preprocess_ms_total",
                               "preprocess_ms")
    if preprocess_total is None and adapter_total is not None and model_total is not None:
        preprocess_total = max(0.0, adapter_total - model_total)
    return {
        "video_decode_ms": _round(video_decode),
        "pose_extract_ms": _round(pose_extract),
        "pose_preprocess_ms": _round(pose_preprocess),
        "hand_crop_ms": _round(hand_crop),
        "model_forward_ms": _round(forward),
        "text_decode_ms": _round(text_decode),
        "preprocess_total_ms": _round(preprocess_total),
        "model_total_ms": _round(model_total),
        "adapter_total_ms": _round(adapter_total),
    }


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def _infer_record(runner: Any, sample: dict[str, Any], repetition: str,
                  extract_ms: float | None = None, extract_frames: int | None = None) -> dict[str, Any]:
    _reset_peak()
    request = _request(f"perf-{runner.path}-{sample['sample_id']}-{repetition}")
    started = perf_counter()
    result, adapter_timing = runner.adapter.infer(sample["trimmed"], request, input_kind="file")
    wall_ms = (perf_counter() - started) * 1000
    quality = result.quality or {}
    segments = _parse_quality(quality, extract_ms=extract_ms)
    adapter_timing = {key: _round(value) for key, value in (adapter_timing or {}).items()}
    if segments.get("adapter_total_ms") is None and adapter_timing.get("total_ms") is not None:
        segments["adapter_total_ms"] = adapter_timing["total_ms"]
    text = (result.prediction.text
            if getattr(result.prediction, "text", None) is not None else "")
    return {
        "status": result.status,
        "reason": result.reason,
        "text": text,
        "segments": segments,
        "e2e_wall_ms": _round(wall_ms),
        "adapter_timing": {key: _round(value) for key, value in (adapter_timing or {}).items()},
        "raw_quality": _json_safe({key: value for key, value in quality.items()
                                   if isinstance(value, (int, float, str, bool, type(None)))}),
        "vram": _vram_snapshot(),
        "extra": {"extractor_calls": extract_frames},
    }


class PoseOnlyRunner:
    """Existing pose-only video runtime; bundle/checkpoint switch only.

    ``a`` uses the pinned stage1 ``csl_daily_pose_only_slt.pth`` bundle; ``b_stage2``
    uses the analyst-built run-local bundle for ``csl_daily_stage2_pose_subset.pth``
    (same backend id, same runtime, same RTMPose frontend -> apples-to-apples).
    """

    def __init__(self, path_key: str, bundle: Path) -> None:
        from slbridge.model_adapters import load_model_adapter
        from slbridge.runtime_options import RuntimeOptions

        self.path = path_key
        self.bundle = Path(bundle)
        self.label = PATH_LABELS[path_key]
        self.adapter = load_model_adapter(self.bundle, options=RuntimeOptions(device="cuda"))
        self.load_seconds: float | None = None
        self.resolution = {"route": "load_model_adapter", "bundle": str(self.bundle)}
        self._extract_ms = 0.0
        self._extract_frames = 0

    def load(self) -> None:
        started = perf_counter()
        self.adapter.warmup()
        self.load_seconds = perf_counter() - started
        self._instrument()

    def _instrument(self) -> None:
        backend = self.adapter.recognizer
        original = backend._extract_cached  # noqa: SLF001 - read-only instrumentation

        def timed_extract(frame, *, source_id, frame_index):
            began = perf_counter()
            try:
                return original(frame, source_id=source_id, frame_index=frame_index)
            finally:
                self._extract_ms += (perf_counter() - began) * 1000
                self._extract_frames += 1

        backend._extract_cached = timed_extract  # noqa: SLF001

    def infer(self, sample: dict[str, Any], repetition: str) -> dict[str, Any]:
        self._extract_ms = 0.0
        self._extract_frames = 0
        record = _infer_record(self, sample, repetition)
        quality = record.get("raw_quality") or {}
        pose_extract = quality.get("pose_extract_ms")
        if isinstance(pose_extract, (int, float)):
            record["segments"]["video_decode_ms"] = _round(
                max(0.0, float(pose_extract) - self._extract_ms))
        record["extra"]["extractor_calls"] = self._extract_frames
        return record

    @property
    def runtime_info(self) -> dict[str, Any]:
        return self.adapter.runtime_info

    def close(self) -> None:
        self.adapter.close()


class RGBPoseRunner:
    label = "unisign_rgb_pose_slt"

    def __init__(self, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.path = variant
        self.label = PATH_LABELS[variant]
        self.config = variant_config(variant)
        self.adapter, self.resolution = load_rgb_pose_adapter(self.config)
        for method in ("warmup", "infer", "close"):
            if not callable(getattr(self.adapter, method, None)):
                raise RuntimeError(f"RGB+pose adapter lacks required method {method!r}")
        self.load_seconds: float | None = None
        self._pose_ms = 0.0
        self._crop_ms = 0.0
        self._extractor = None
        self._instrumented = False

    def load(self) -> None:
        started = perf_counter()
        self.adapter.warmup()
        self.load_seconds = perf_counter() - started
        self._instrument()

    def _instrument(self) -> None:
        """Time RTMPose extraction and hand cropping from this script only.

        The production adapter reports a single ``preprocess_ms``; the task-34
        report needs ``pose_extract`` (same RTMPose frontend as A) and
        ``hand_crop`` separately.  Only wrappers are installed here, no library
        code is modified.
        """
        ensure = getattr(self.adapter, "_ensure_extractor", None)
        if callable(ensure):
            try:
                extractor = ensure()
            except Exception:  # noqa: BLE001 - surface later as adapter failure
                extractor = None
            if extractor is not None:
                self._extractor = extractor
                original_extract = extractor.extract
                runner = self

                def timed_extract(*args, **kwargs):
                    began = perf_counter()
                    try:
                        return original_extract(*args, **kwargs)
                    finally:
                        runner._pose_ms += (perf_counter() - began) * 1000

                extractor.extract = timed_extract
        try:
            import slbridge.uni_sign_rgb_pose as rgb_module
        except Exception:  # noqa: BLE001
            rgb_module = None
        if rgb_module is not None and not getattr(rgb_module, "_perf_crop_patched", False):
            original_build = rgb_module.build_support_rgb

            def timed_build(*args, **kwargs):
                timer = _ACTIVE_CROP_TIMER.get("runner")
                began = perf_counter()
                try:
                    return original_build(*args, **kwargs)
                finally:
                    if timer is not None:
                        timer._crop_ms += (perf_counter() - began) * 1000

            rgb_module.build_support_rgb = timed_build
            rgb_module._perf_crop_patched = True
        self._instrumented = True

    def infer(self, sample: dict[str, Any], repetition: str) -> dict[str, Any]:
        self._pose_ms = 0.0
        self._crop_ms = 0.0
        _ACTIVE_CROP_TIMER["runner"] = self
        try:
            record = _infer_record(self, sample, repetition)
        finally:
            _ACTIVE_CROP_TIMER["runner"] = None
        segments = record["segments"]
        if self._extractor is not None:
            segments["pose_extract_ms"] = _round(self._pose_ms)
        segments["hand_crop_ms"] = _round(self._crop_ms)
        preprocess = segments.get("preprocess_total_ms")
        if preprocess is not None:
            used = (segments.get("pose_extract_ms") or 0.0) + (segments.get("hand_crop_ms") or 0.0)
            segments["video_decode_ms"] = _round(max(0.0, preprocess - used))
        return record

    @property
    def runtime_info(self) -> dict[str, Any]:
        return self.adapter.runtime_info

    def close(self) -> None:
        self.adapter.close()


# Module-level holder so the single hand-crop wrapper times the active runner.
_ACTIVE_CROP_TIMER: dict[str, Any] = {"runner": None}


def stage2_bundle() -> Path:
    configured = OVERRIDES.get("stage2_bundle")
    return Path(configured) if configured else STAGE2_BUNDLE_DEFAULT


def make_runner(path: str):
    if path == "a":
        return PoseOnlyRunner("a", A_BUNDLE)
    if path == "b_stage2":
        return PoseOnlyRunner("b_stage2", stage2_bundle())
    if path in VARIANTS:
        return RGBPoseRunner(path)
    raise ValueError(f"unknown path {path!r}")


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------
def run_trace_row(*, run_id: str, phase: str, path: str, sample: dict[str, Any],
                  repetition: int, record: dict[str, Any], attempt: int = 1,
                  error: str | None = None) -> dict[str, Any]:
    video_seconds = sample.get("video_seconds")
    e2e = record.get("e2e_wall_ms")
    segments = record.get("segments") or {}
    rtf = (e2e / 1000.0 / video_seconds) if (e2e is not None and video_seconds) else None
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "phase": phase,
        "path": path,
        "variant": VARIANTS.get(path, {}).get("hand_conf_layout"),
        "sample_id": sample.get("sample_id"),
        "signer_id": sample.get("signer_id"),
        "repetition": repetition,
        "video_seconds": _round(video_seconds, 6),
        "fps": _round(sample.get("fps"), 3),
        "frames": sample.get("frames"),
        "rtf": _round(rtf, 4),
        "e2e_ms": e2e,
        "status": "ok" if record.get("status") == "recognized" else str(record.get("status")),
        "reason": record.get("reason"),
        "error": error,
        "attempts": attempt,
        **{key: segments.get(key) for key in SEGMENT_KEYS if key != "e2e_ms"},
        "adapter_timing": record.get("adapter_timing") or {},
        "vram": record.get("vram") or {},
        "raw_quality": record.get("raw_quality") or {},
        "extra": record.get("extra") or {},
        "text": record.get("text") or "",
        "text_length": len(record.get("text") or ""),
    }


def _run_once(runner: Any, sample: dict[str, Any], repetition: str) -> tuple[dict[str, Any] | None, int, str | None]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return runner.infer(sample, repetition), attempt, None
        except BaseException as error:  # noqa: BLE001 - recorded as evidence
            if attempt <= 2 and any(marker in str(error).lower() for marker in RETRYABLE):
                print(f"[retry] {runner.path}/{sample['sample_id']} allocation failure; "
                      f"retry {attempt}/2 after {RETRY_SLEEP_SECONDS:.0f}s", flush=True)
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            return None, attempt, f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------
def cmd_cold(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    samples = common.attach_fps(common.select_pilot_samples(min(4, args.samples)))
    results: dict[str, list[dict[str, Any]]] = {path: [] for path in PATHS}
    for path in PATHS:
        for attempt in range(args.cold_repeats):
            out_path = run_dir / f"cold-{path}-{attempt}.json"
            log_path = run_dir / f"cold-{path}-{attempt}.log"
            command = [sys.executable, str(Path(__file__).resolve()), "cold-child",
                       "--path", path, "--out", str(out_path),
                       "--sample-json", json.dumps(samples[0], ensure_ascii=False),
                       "--pose-source", BASE_RGB_POSE_CONFIG["pose_source"],
                       "--stage2-bundle", str(stage2_bundle())]
            if OVERRIDES.get("rgbpose_entry"):
                command += ["--rgbpose-entry", OVERRIDES["rgbpose_entry"]]
            if BASE_RGB_POSE_CONFIG["fallback_to_pose_only"]:
                command.append("--allow-fallback")
            started = perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                         stderr=subprocess.STDOUT, env=_child_env())
            wall = perf_counter() - started
            payload = common.read_json(out_path) if out_path.is_file() else {"status": "missing"}
            row = {
                "path": path,
                "spawn": attempt,
                "exit_code": process.returncode,
                "process_wall_seconds": round(wall, 3),
                "load_seconds": _round(payload.get("load_seconds"), 3),
                "first_inference_ms": _round(payload.get("first_inference_ms")),
                "vram_after_load": payload.get("vram_after_load"),
                "resolution": payload.get("resolution"),
                "runtime_info": payload.get("runtime_info"),
                "status": payload.get("status", "missing"),
                "error": payload.get("error"),
            }
            results[path].append(row)
            common.append_jsonl(run_dir / "cold.jsonl", [row])
            print(json.dumps({"event": "cold", **{key: row[key] for key in
                                                  ("path", "spawn", "load_seconds",
                                                   "first_inference_ms", "status")}},
                             ensure_ascii=False), flush=True)
    summary = {
        "phase": "cold",
        "samples": [samples[0]["sample_id"]],
        "per_path": {path: {"attempts": len(rows),
                            "load_seconds": _stats([r["load_seconds"] for r in rows
                                                    if r["load_seconds"] is not None]),
                            "process_wall_seconds": _stats([r["process_wall_seconds"] for r in rows]),
                            }
                     for path, rows in results.items()},
        "rows": results,
    }
    common.write_json(run_dir / "cold-summary.json", summary)
    return 0


def cmd_cold_child(args: argparse.Namespace) -> int:
    sample = json.loads(args.sample_json)
    payload: dict[str, Any] = {"status": "error", "error": None}
    runner = None
    try:
        runner = make_runner(args.path)
        runner.load()
        payload["load_seconds"] = runner.load_seconds
        payload["resolution"] = _json_safe(getattr(runner, "resolution", None))
        payload["config"] = _json_safe(getattr(runner, "config", BASE_RGB_POSE_CONFIG))
        payload["runtime_info"] = _json_safe(runner.runtime_info)
        payload["vram_after_load"] = _vram_snapshot()
        record, attempts, error = _run_once(runner, sample, "cold")
        payload.update({"status": "ok" if record is not None else "failed",
                        "error": error, "attempts": attempts,
                        "first_inference_ms": record.get("e2e_wall_ms") if record else None,
                        "segments": record.get("segments") if record else None})
    except BaseException as error:  # noqa: BLE001
        payload.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                pass
    common.write_json(Path(args.out), payload)
    return 0 if payload["status"] == "ok" else 1


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    run_id = args.run_id or f"rgbpose-perf-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    ready = readiness()
    if not ready["ready"]:
        payload = {"status": "BLOCKED", "reason": "RGB+pose adapter not runnable yet",
                   "readiness": ready}
        common.write_json(run_dir / "blocked.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    baseline = common.nvidia_smi()
    if baseline is None:
        raise SystemExit("nvidia-smi unavailable")
    device = baseline["devices"][0]
    if device["memory_used_mb"] > args.max_foreign_vram_mb and not args.allow_shared_gpu:
        raise SystemExit(f"GPU busy: {device['memory_used_mb']:.0f} MB used "
                         f"(limit {args.max_foreign_vram_mb:.0f} MB)")
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    manifest = common.base_manifest(run_id, "E04-rgbpose-perf", purpose=(
        "RGB+pose zero-gate production form (legacy, no hand crops) vs squeeze "
        "hand-crop waste control vs pose-only video path A: paired e2e/RTF/VRAM"))
    manifest.update({
        "model_assets": common.model_assets_manifest(),
        "environment": common.environment_fingerprint(),
        "gpu_baseline": baseline,
        "ram": common.memory_snapshot(),
        "samples": [sample["sample_id"] for sample in samples],
        "signers": sorted({sample["signer_id"] for sample in samples}),
        "readiness": ready,
        "variants": {path: variant_config(path) for path in VARIANTS if path in PATHS},
        "path_labels": {path: PATH_LABELS[path] for path in PATHS},
        "arms": list(PATHS),
        "stage2_bundle": str(stage2_bundle()),
        "frozen_model_reference": FROZEN_MODEL_REFERENCE,
        "protocol": {"decode": "production ModelAdapter.infer(path, input_kind='file')",
                     "pose_source": BASE_RGB_POSE_CONFIG["pose_source"],
                     "order": "rotating A/legacy/squeeze per sample inside each round",
                     "rounds": args.rounds, "samples": len(samples),
                     "continuous_minutes": args.continuous_minutes,
                     "gate_note": ("pinned checkpoint fusion_gate all-zero; legacy carries "
                                   "the RGB branch without hand crops, squeeze pays for "
                                   "crops the gate discards")},
        "commands": [common.command_line()],
    })
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "baseline", "gpu": device, "samples": len(samples)},
                     ensure_ascii=False), flush=True)

    sampler = common.DeviceSampler(interval_seconds=1.0).start()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    load_info: dict[str, Any] = {}
    runners: dict[str, Any] = {}
    try:
        for path in PATHS:
            runner = make_runner(path)
            started = perf_counter()
            runner.load()
            elapsed = perf_counter() - started
            runners[path] = runner
            load_info[path] = {"load_seconds": round(elapsed, 3),
                               "runner_load_seconds": _round(runner.load_seconds),
                               "resolution": _json_safe(getattr(runner, "resolution", None)),
                               "config": _json_safe(getattr(runner, "config", None)),
                               "runtime_info": _json_safe(runner.runtime_info),
                               "vram_after_load": _vram_snapshot()}
            print(json.dumps({"event": "loaded", "path": path,
                              "load_seconds": round(elapsed, 3),
                              "vram": load_info[path]["vram_after_load"]},
                             ensure_ascii=False), flush=True)
        common.write_json(run_dir / "load.json", load_info)
        manifest["load"] = load_info
        common.write_json(run_dir / "manifest.json", manifest)
        for path, runner in runners.items():
            record, attempts, error = _run_once(runner, samples[0], "warmup")
            print(json.dumps({"event": "warmup", "path": path, "ok": record is not None,
                              "error": error}, ensure_ascii=False), flush=True)
        # ---- paired rotating rounds ------------------------------------
        for round_index in range(args.rounds):
            for position, sample in enumerate(samples):
                offset = (round_index + position) % len(PATHS)
                order = PATHS[offset:] + PATHS[:offset]
                for path in order:
                    record, attempts, error = _run_once(runners[path], sample,
                                                        f"r{round_index}-p{position}")
                    phase = f"round{round_index + 1}"
                    if record is None:
                        failures.append({"phase": phase, "path": path,
                                         "sample_id": sample["sample_id"], "error": error})
                        rows.append(run_trace_row(run_id=run_id, phase=phase, path=path,
                                                  sample=sample, repetition=position,
                                                  record={}, attempt=attempts, error=error))
                        continue
                    rows.append(run_trace_row(run_id=run_id, phase=phase, path=path,
                                              sample=sample, repetition=position,
                                              record=record, attempt=attempts))
            print(json.dumps({"event": "round_done", "round": round_index + 1,
                              "rows": len(rows)}, ensure_ascii=False), flush=True)
        # ---- continuous load -------------------------------------------
        deadline = perf_counter() + args.continuous_minutes * 60.0
        index = 0
        while perf_counter() < deadline:
            sample = samples[index % len(samples)]
            path = PATHS[index % len(PATHS)]
            record, attempts, error = _run_once(runners[path], sample, f"continuous-{index}")
            if record is None:
                failures.append({"phase": "continuous", "path": path,
                                 "sample_id": sample["sample_id"], "error": error})
            else:
                rows.append(run_trace_row(run_id=run_id, phase="continuous", path=path,
                                          sample=sample, repetition=index,
                                          record=record, attempt=attempts))
            index += 1
        print(json.dumps({"event": "continuous_done", "iterations": index},
                         ensure_ascii=False), flush=True)
    finally:
        device_samples = sampler.stop()
        for runner in runners.values():
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                pass
    if device_samples:
        common.write_jsonl(run_dir / "device_samples.jsonl", device_samples)
    common.write_jsonl(run_dir / "trace.jsonl", rows)
    metrics = _summarise(run_id=run_id, rows=rows, failures=failures,
                         load_info=load_info, samples=samples, device_samples=device_samples)
    metrics["gpu_baseline"] = baseline
    metrics["ram"] = common.memory_snapshot()
    metrics["frozen_model_reference"] = FROZEN_MODEL_REFERENCE
    common.write_json(run_dir / "metrics.json", metrics)
    manifest.update({"run_status": "PASS" if not failures else "PARTIAL",
                     "finished_at": common.utc_now(),
                     "load": load_info,
                     "failures": failures,
                     "comparison": metrics["comparison"],
                     "output_files": sorted(path.name for path in run_dir.iterdir() if path.is_file())})
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "done", "run_dir": str(run_dir),
                      "comparison": metrics["comparison"]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict[str, Any]:
    return common.stats(values)


def _vram_peak(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak_allocated = [row["vram"].get("torch_peak_allocated_mb") for row in rows if row.get("vram")]
    peak_reserved = [row["vram"].get("torch_peak_reserved_mb") for row in rows if row.get("vram")]
    allocated = [row["vram"].get("torch_allocated_mb") for row in rows if row.get("vram")]

    def maximum(values: list[Any]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return max(clean) if clean else None
    return {"peak_allocated_mb": maximum(peak_allocated),
            "peak_reserved_mb": maximum(peak_reserved),
            "resident_allocated_mb_p50": common.percentile(
                [float(value) for value in allocated if value is not None], 50)}


def _summarise(*, run_id: str, rows: list[dict[str, Any]], failures: list[dict[str, Any]],
               load_info: dict[str, Any], samples: list[dict[str, Any]],
               device_samples: list[dict[str, Any]]) -> dict[str, Any]:
    def ok_rows(path: str, phases: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        return [row for row in rows if row["path"] == path and row["status"] == "ok"
                and (phases is None or row["phase"] in phases)]

    def summarise_path(path: str) -> dict[str, Any]:
        data = ok_rows(path)
        payload: dict[str, Any] = {
            "label": PATH_LABELS[path],
            "variant": VARIANTS.get(path, {}).get("hand_conf_layout"),
            "runs": len(data),
            "failures": len([item for item in failures if item["path"] == path]),
        }
        for key in SEGMENT_KEYS:
            payload[key] = _stats([row[key] for row in data if row.get(key) is not None])
        payload["rtf"] = _stats([row["rtf"] for row in data if row.get("rtf") is not None])
        payload["vram"] = _vram_peak(data)
        payload["phases"] = {}
        for phase in sorted({row["phase"] for row in data}):
            subset = [row for row in data if row["phase"] == phase]
            payload["phases"][phase] = {
                "runs": len(subset),
                "e2e_ms": _stats([row["e2e_ms"] for row in subset if row.get("e2e_ms") is not None]),
                "peak_allocated_mb": max([(row.get("vram") or {}).get("torch_peak_allocated_mb") or 0
                                          for row in subset], default=None),
            }
        return payload

    per_path = {path: summarise_path(path) for path in PATHS}
    phases = sorted({row["phase"] for row in rows})

    def paired(first: str, second: str, phase: str) -> dict[str, Any]:
        first_by_sample = {row["sample_id"]: row for row in ok_rows(first, (phase,))}
        second_by_sample = {row["sample_id"]: row for row in ok_rows(second, (phase,))}
        shared = sorted(set(first_by_sample) & set(second_by_sample))
        e2e = [second_by_sample[s]["e2e_ms"] - first_by_sample[s]["e2e_ms"] for s in shared]
        model = [second_by_sample[s].get("model_forward_ms") - first_by_sample[s].get("model_forward_ms")
                 for s in shared
                 if second_by_sample[s].get("model_forward_ms") is not None
                 and first_by_sample[s].get("model_forward_ms") is not None]
        preprocess = [second_by_sample[s].get("preprocess_total_ms") - first_by_sample[s].get("preprocess_total_ms")
                      for s in shared
                      if second_by_sample[s].get("preprocess_total_ms") is not None
                      and first_by_sample[s].get("preprocess_total_ms") is not None]
        identical = sum((first_by_sample[s].get("text") or "") == (second_by_sample[s].get("text") or "")
                        for s in shared)
        return {"pairs": len(shared),
                "delta_e2e_ms": _stats(e2e),
                "delta_model_forward_ms": _stats(model),
                "delta_preprocess_total_ms": _stats(preprocess),
                "second_slower_e2e_count": sum(1 for value in e2e if value > 0),
                "identical_predictions": identical}

    pair_payload: dict[str, Any] = {}
    for first, second in active_pair_defs():
        pair_payload[f"{second}_minus_{first}"] = {
            "first": first,
            "second": second,
            "phases": {phase: paired(first, second, phase) for phase in phases},
        }

    def delta_p50(second: str, first: str) -> float | None:
        payload = pair_payload.get(f"{second}_minus_{first}") or {}
        values = [(payload.get("phases") or {}).get(phase, {}).get("delta_e2e_ms", {}).get("p50")
                  for phase in ("round1", "round2")]
        clean = [float(value) for value in values if value is not None]
        return round(statistics.fmean(clean), 2) if clean else None

    def p50(path: str, key: str) -> float | None:
        return (per_path[path].get(key) or {}).get("p50")

    def delta_p50(second: str, first: str) -> float | None:
        payload = pair_payload.get(f"{second}_minus_{first}") or {}
        values = [(payload.get("phases") or {}).get(phase, {}).get("delta_e2e_ms", {}).get("p50")
                  for phase in ("round1", "round2")]
        clean = [float(value) for value in values if value is not None]
        return round(statistics.fmean(clean), 2) if clean else None

    def pair_identical(second: str, first: str) -> dict[str, Any]:
        payload = pair_payload.get(f"{second}_minus_{first}") or {}
        pairs = shared = 0
        for phase_payload in (payload.get("phases") or {}).values():
            identical = phase_payload.get("identical_predictions")
            count = phase_payload.get("pairs")
            if isinstance(identical, int) and isinstance(count, int):
                pairs += identical
                shared += count
        return {"identical": pairs, "pairs": shared}

    def p50(path: str, key: str) -> float | None:
        return (per_path[path].get(key) or {}).get("p50")

    comparison = {
        "arms": list(PATHS),
        "e2e_p50_ms": {path: p50(path, "e2e_ms") for path in PATHS},
        "e2e_p95_ms": {path: (per_path[path].get("e2e_ms") or {}).get("p95") for path in PATHS},
        "e2e_mean_ms": {path: (per_path[path].get("e2e_ms") or {}).get("mean") for path in PATHS},
        "model_forward_p50_ms": {path: p50(path, "model_forward_ms") for path in PATHS},
        "preprocess_total_p50_ms": {path: p50(path, "preprocess_total_ms") for path in PATHS},
        "rtf_p50": {path: p50(path, "rtf") for path in PATHS},
        "peak_allocated_mb": {path: (per_path[path].get("vram") or {}).get("peak_allocated_mb")
                              for path in PATHS},
        "peak_reserved_mb": {path: (per_path[path].get("vram") or {}).get("peak_reserved_mb")
                             for path in PATHS},
        "paired_delta_e2e_p50_ms": {f"{second}_minus_{first}": delta_p50(second, first)
                                    for first, second in active_pair_defs()},
        "paired_delta_e2e_mean_ms": {
            f"{second}_minus_{first}": (
                round((per_path[second].get("e2e_ms") or {}).get("mean", 0.0)
                      - (per_path[first].get("e2e_ms") or {}).get("mean", 0.0), 2)
                if per_path[second].get("e2e_ms") and per_path[first].get("e2e_ms") else None)
            for first, second in active_pair_defs()},
        "identical_predictions": {f"{second}_vs_{first}": pair_identical(second, first)
                                  for first, second in active_pair_defs()},
        "cold_load_seconds": {path: load_info.get(path, {}).get("load_seconds") for path in PATHS},
        "stage2_bundle": str(stage2_bundle()),
        "core_question_b_vs_a": {
            "delta_e2e_p50_ms": delta_p50("b_stage2", "a"),
            "delta_e2e_mean_ms": (
                round((per_path["b_stage2"].get("e2e_ms") or {}).get("mean", 0.0)
                      - (per_path["a"].get("e2e_ms") or {}).get("mean", 0.0), 2)
                if "b_stage2" in per_path and "a" in per_path else None),
            "delta_model_forward_p50_ms": (
                round((p50("b_stage2", "model_forward_ms") or 0.0)
                      - (p50("a", "model_forward_ms") or 0.0), 2)
                if p50("b_stage2", "model_forward_ms") is not None
                and p50("a", "model_forward_ms") is not None else None),
            "delta_peak_allocated_mb": (
                round(((per_path["b_stage2"].get("vram") or {}).get("peak_allocated_mb") or 0.0)
                      - ((per_path["a"].get("vram") or {}).get("peak_allocated_mb") or 0.0), 1)
                if "b_stage2" in per_path and "a" in per_path else None),
        },
        "tiny_note": ("paired rotating-round deltas are the primary number; absolute values "
                      "are not clean-room SLA; C/D rgb-pose arms are optional and BLOCKED "
                      "until the rtmpose adapter defect is fixed"),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "paths": per_path,
        "pairs": pair_payload,
        "comparison": comparison,
        "failures": failures,
        "load": load_info,
        "samples": [{"sample_id": sample["sample_id"], "video_seconds": sample.get("video_seconds")}
                    for sample in samples],
        "device_peak": {},
    }
    if device_samples:
        used = [item["memory_used_mb"] for item in device_samples]
        util = [item["utilization_gpu_pct"] for item in device_samples]
        result["device_peak"] = {
            "samples": len(device_samples),
            "peak_memory_used_mb": max(used) if used else None,
            "mean_memory_used_mb": round(statistics.fmean(used), 1) if used else None,
            "mean_utilization_gpu_pct": round(statistics.fmean(util), 1) if util else None,
            "max_utilization_gpu_pct": max(util) if util else None,
            "scope": "whole_device_including_other_processes",
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_rgb_pose_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--pose-source", choices=("rtmpose", "official_archive"),
                            default="rtmpose")
        target.add_argument("--rgbpose-entry", default=None)
        target.add_argument("--allow-fallback", action="store_true")
        target.add_argument("--stage2-bundle", default=None,
                            help="run-local pose bundle for the stage2 subset checkpoint (B arm)")
        target.add_argument("--arms", default="a,b_stage2",
                            help="comma separated active arms among " + ",".join(PATH_CHOICES))

    check = sub.add_parser("check")
    add_rgb_pose_flags(check)
    run = sub.add_parser("run")
    add_rgb_pose_flags(run)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--samples", type=int, default=20)
    run.add_argument("--rounds", type=int, default=2)
    run.add_argument("--continuous-minutes", type=float, default=2.0)
    run.add_argument("--max-foreign-vram-mb", type=float, default=4500.0)
    run.add_argument("--allow-shared-gpu", action="store_true")
    cold = sub.add_parser("cold")
    add_rgb_pose_flags(cold)
    cold.add_argument("--run-dir", type=Path, required=True)
    cold.add_argument("--samples", type=int, default=20)
    cold.add_argument("--cold-repeats", type=int, default=3)
    cold_child = sub.add_parser("cold-child")
    add_rgb_pose_flags(cold_child)
    cold_child.add_argument("--path", choices=PATH_CHOICES, required=True)
    cold_child.add_argument("--out", type=Path, required=True)
    cold_child.add_argument("--sample-json", required=True)
    return parser


def _apply_overrides(args: argparse.Namespace) -> None:
    pose_source = getattr(args, "pose_source", None)
    if pose_source:
        BASE_RGB_POSE_CONFIG["pose_source"] = pose_source
    BASE_RGB_POSE_CONFIG["enable_rgb"] = True
    BASE_RGB_POSE_CONFIG["fallback_to_pose_only"] = bool(getattr(args, "allow_fallback", False))
    entry = getattr(args, "rgbpose_entry", None)
    if entry:
        OVERRIDES["rgbpose_entry"] = entry
    stage2 = getattr(args, "stage2_bundle", None)
    if stage2:
        OVERRIDES["stage2_bundle"] = stage2
    requested = getattr(args, "arms", None)
    if requested:
        arms = tuple(name.strip() for name in requested.split(",") if name.strip())
        unknown = [name for name in arms if name not in PATH_CHOICES]
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}")
        global PATHS
        PATHS = arms


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_overrides(args)
    if args.command == "check":
        print(json.dumps(readiness(), ensure_ascii=False, indent=2))
        return 0
    handlers = {"run": cmd_run, "cold": cmd_cold, "cold-child": cmd_cold_child}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

