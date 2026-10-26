"""E04 -- GPU performance, VRAM and serial-vs-parallel benchmark for P-1.

What this script measures (one process at a time unless a phase is explicitly
``parallel``; the single GPU is otherwise shared, which is never a valid
throughput comparison):

* cold start, 3 fresh processes per model: process wall time, model load time,
  first-inference time (CUDA warm-up included), peak VRAM;
* warm repeats (default 30 timed inferences) after untimed warm-up;
* continuous load (default 2 minutes) to expose thermal/allocation drift;
* segmented timings ``video_decode_ms`` / ``pose_ms`` / ``forward_ms`` /
  ``text_decode_ms`` / ``queue_ms`` / ``e2e_ms``; CPU wall clock uses
  ``time.perf_counter`` and every GPU segment is bounded by
  ``torch.cuda.synchronize`` inside the runtime (recorded per row);
* torch peak/resident VRAM plus whole-device ``nvidia-smi`` sampling
  (the sampler cannot separate our process from other processes -- rows say so);
* serial (Uni-Sign video then SLRT) vs two-process parallel on the same batch.

Models (all through the pinned local assets, no source changes):

* ``unisign_pose``  -- official released CSL-Daily pose pkl -> mT5;
* ``unisign_video`` -- trimmed mp4 -> RTMPose wholebody 133 -> mT5;
* ``slrt_hrnet_npz`` -- trimmed mp4 + HRNet NPZ -> SLRT S3D 16-frame windows.

The script refuses to start a GPU phase when the device already holds more than
``--max-foreign-vram-mb`` (default 3000 MB) unless ``--allow-shared-gpu`` is
passed; shared-GPU rows are tagged so the report cannot silently quote them as
exclusive numbers.

Usage (after the Lead confirms the GPU is free)::

    .conda-env\\python.exe experiments/p1/e04/benchmark_p1.py all \
        --run-id e04-gpu-<stamp> --cold-repeats 3 --warm-repeats 30 --load-minutes 2

Individual phases are also available: ``cold``, ``warm``, ``load``, ``serial``,
``parallel``, ``summarize``.  ``child`` is an internal worker used for the cold
and parallel phases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1.e04 import perf_common as common  # noqa: E402

MODELS = common.MODEL_IDS
DEFAULT_SAMPLES = 5
DEVICE_INTERVAL_SECONDS = 1.0
START_BARRIER_TIMEOUT_SECONDS = 300.0

FAILURE_MARKERS = (
    "out of memory",
    "OutOfMemory",
    "CUDA error",
    "Memory allocation failure",
    "_ArrayMemoryError",
    "Insufficient memory",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _now() -> float:
    return time.perf_counter()


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _sample_trace_row(
    *, run_id: str, phase: str, model: str, sample: dict[str, Any], repetition: int,
    timings: dict[str, Any], status: str = "ok", error: str | None = None,
    text: str | None = None, gloss: list[str] | None = None, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_seconds = sample.get("video_seconds")
    e2e_ms = timings.get("e2e_ms")
    rtf = (e2e_ms / 1000.0 / video_seconds) if (e2e_ms is not None and video_seconds) else None
    row = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "phase": phase,
        "model": model,
        "model_display": common.MODEL_META[model]["display"],
        "sample_id": sample.get("sample_id"),
        "signer_id": sample.get("signer_id"),
        "repetition": repetition,
        "status": status,
        "error": error,
        "video_seconds": _round(video_seconds, 6),
        "fps": _round(sample.get("fps"), 3),
        "frames": sample.get("frames"),
        "rtf": _round(rtf, 4),
        "timing_method": {
            "cpu": "time.perf_counter",
            "gpu": "torch.cuda.synchronize inside runtime around each GPU segment",
        },
    }
    for key in common.SEGMENT_KEYS:
        row[key] = _round(timings.get(key))
    row["timings_raw"] = {
        key: _round(value)
        for key, value in timings.items()
        if key not in common.SEGMENT_KEYS and isinstance(value, (int, float))
    }
    row["vram"] = timings.get("vram") or {}
    row["input"] = timings.get("input") or {}
    if text is not None:
        row["output_text"] = text
    if gloss is not None:
        row["gloss_sequence"] = gloss
    if extra:
        row.update(extra)
    return row


def _is_failure_marker(error: str | None) -> bool:
    if not error:
        return False
    return any(marker.lower() in error.lower() for marker in FAILURE_MARKERS)


def _normalize_status(value: Any) -> str:
    """Map the various runtime status words onto ok / empty / other.

    ``RecognitionResult.status`` uses ``recognized``/``rejected``; the SLRT
    stream uses ``final``/``partial``.  Only ``ok`` counts as a usable sample.
    """
    if value is None or value in ("ok", "recognized", "final"):
        return "ok"
    if value in ("empty", "rejected", "partial"):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# model runners
# ---------------------------------------------------------------------------
class BaseRunner:
    model_id = "base"

    def __init__(self) -> None:
        self.load_seconds: float | None = None
        self.load_details: dict[str, Any] = {}
        self._torch = None

    # -- lifecycle -----------------------------------------------------
    def load(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def preload(self, samples: list[dict[str, Any]]) -> None:
        return None

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    # -- measurement ---------------------------------------------------
    @property
    def resident_vram_mb(self) -> float | None:
        if self._torch is None:
            return None
        try:
            return float(self._torch.cuda.memory_allocated() / 1024**2)
        except Exception:  # noqa: BLE001
            return None

    def reset_vram_peak(self) -> None:
        if self._torch is not None:
            try:
                self._torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass

    def vram_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "torch_resident_allocated_mb": _round(self.resident_vram_mb, 1),
            "scope": "process_torch_allocator_peak, no other process",
        }
        if self._torch is not None:
            try:
                snapshot["torch_peak_allocated_mb"] = _round(
                    self._torch.cuda.max_memory_allocated() / 1024**2, 1)
                snapshot["torch_peak_reserved_mb"] = _round(
                    self._torch.cuda.max_memory_reserved() / 1024**2, 1)
                free, total = self._torch.cuda.mem_get_info()
                snapshot["device_free_mb_at_read"] = _round(free / 1024**2, 1)
                snapshot["device_total_mb"] = _round(total / 1024**2, 1)
            except Exception:  # noqa: BLE001
                pass
        return snapshot

    def runtime_info(self) -> dict[str, Any]:
        return {}

    def infer(self, sample: dict[str, Any], repetition: int) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class UniSignPoseRunner(BaseRunner):
    """Official released pose pkl -> pinned mT5 (model-only latency)."""

    model_id = "unisign_pose"

    def __init__(self, *, num_beams: int = 4, max_new_tokens: int = 100, max_frames: int | None = None):
        super().__init__()
        from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig

        self.config = UniSignConfig.from_bundle(common.UNISIGN_BUNDLE)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.max_frames = int(max_frames or self.config.runtime.get("max_frames", 256))
        self.runtime = TorchUniSignRuntime(self.config, device="cuda", dtype="bfloat16")
        self._cache: dict[str, tuple[Any, Any, dict[str, Any]]] = {}

    def load(self) -> None:
        from slbridge.uni_sign_preprocess import prepare_pose_window

        started = _now()
        self.runtime.load()
        self._torch = self.runtime._torch  # noqa: SLF001 - same package, documented
        # First forward is part of cold start; it also reveals CUDA warm-up cost.
        self.load_seconds = _now() - started
        self.load_details = {
            "model_load_ms_runtime": _round(self.runtime.load_ms),
            "dtype": "bfloat16",
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "max_frames": self.max_frames,
            "parameter_count": self.runtime.load_details.get("parameter_count"),
            "cold_first_forward_ms": None,
        }
        warmup = np.zeros((8, 1, 133, 2), dtype=np.float32)
        warmup[:, 0, :, :] = 0.4
        warmup_scores = np.ones((8, 1, 133), dtype=np.float32)
        tiny = prepare_pose_window(warmup, warmup_scores, max_frames=8)
        first_started = _now()
        self.runtime.infer(tiny, num_beams=self.num_beams, max_new_tokens=8)
        self.load_details["cold_first_forward_ms"] = _round((_now() - first_started) * 1000)

    def preload(self, samples: list[dict[str, Any]]) -> None:
        for sample in samples:
            keypoints, scores, info = common.load_pose_pkl(sample["pkl"])
            self._cache[sample["sample_id"]] = (keypoints, scores, info)

    def infer(self, sample: dict[str, Any], repetition: int) -> dict[str, Any]:
        from slbridge.uni_sign_preprocess import prepare_pose_window

        keypoints, scores, info = self._cache[sample["sample_id"]]
        self.reset_vram_peak()
        pose_started = _now()
        pose = prepare_pose_window(keypoints, scores, max_frames=self.max_frames)
        pose_done = _now()
        text, timings = self.runtime.infer(
            pose, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
        e2e_done = _now()
        return {
            "video_decode_ms": None,  # pose comes from disk; no video decode in this path
            "pose_ms": (pose_done - pose_started) * 1000,
            "forward_ms": timings.get("model_forward_ms"),
            "text_decode_ms": (timings.get("generation_ms") or 0.0) + (timings.get("decode_ms") or 0.0),
            "queue_ms": 0.0,  # synchronous single-caller path, no queueing
            "e2e_ms": (e2e_done - pose_started) * 1000,
            "raw": {**{key: _round(value) for key, value in timings.items()},
                    "pose_prepare_raw_ms": _round((pose_done - pose_started) * 1000)},
            "input": {
                "pose_frames_total": info.get("frames_total"),
                "pose_frames_used": info.get("frames_used"),
                "pose_crop": [info.get("start"), info.get("end")],
                "sampled_frames": pose.quality.get("sampled_frames"),
                "body_scale": _round(pose.quality.get("body_scale")),
                "valid_body_fraction": _round(pose.quality.get("valid_body_fraction")),
                "pose_sha256": common.sha256_bytes(
                    keypoints.tobytes(), scores.tobytes(), str(keypoints.shape).encode()),
            },
            "vram": self.vram_snapshot(),
            "text": text,
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "TorchUniSignRuntime",
            "model_total_ms": None,
            "capabilities": ["pose_pkl"],
            "load": self.load_details,
            "runtime": self.runtime.runtime_info,
        }

    def close(self) -> None:
        self.runtime.close()


class UniSignVideoRunner(BaseRunner):
    """Trimmed mp4 -> RTMPose wholebody -> pinned mT5 (video end-to-end path).

    ``_video_pose_window`` interleaves OpenCV decode with RTMPose per frame, so
    decode and pose are separated here by wrapping the per-frame
    ``_extract_cached`` call: ``video_decode_ms = pose_extract_ms - sum(extract)``.
    The wrapper only times the existing call; source code is untouched.
    """

    model_id = "unisign_video"

    def __init__(self, *, num_beams: int = 4, max_new_tokens: int = 100):
        super().__init__()
        from slbridge.runtime_options import RuntimeOptions
        from slbridge.uni_sign import UniSignBackend, UniSignConfig

        self.config = UniSignConfig.from_bundle(common.UNISIGN_BUNDLE)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.backend = UniSignBackend(self.config, RuntimeOptions(device="cuda"))
        self._extract_ms = 0.0
        self._extract_frames = 0

    def load(self) -> None:
        started = _now()
        self.backend.load_model()
        self.load_seconds = _now() - started
        runtime = getattr(self.backend, "_runtime", None)
        self._torch = getattr(runtime, "_torch", None)
        self.load_details = {
            "model_load_ms_backend": _round(self.backend._load_ms),  # noqa: SLF001
            "model_warmup_ms": _round(self.backend._model_warmup_ms),  # noqa: SLF001
            "max_frames": self.backend.max_frames,
            "num_beams": self.num_beams,
            "pose_sample_only": self.backend.pose_sample_only,
            "pose_detect_stride": self.backend.pose_detect_stride,
            "pose_normalization": self.backend.pose_normalization,
            "pose_orientation": self.backend.pose_orientation,
        }
        original = self.backend._extract_cached  # noqa: SLF001 - instrumentation only

        def timed_extract(frame, *, source_id, frame_index):
            began = _now()
            try:
                return original(frame, source_id=source_id, frame_index=frame_index)
            finally:
                self._extract_ms += (_now() - began) * 1000
                self._extract_frames += 1

        self.backend._extract_cached = timed_extract  # noqa: SLF001

    def infer(self, sample: dict[str, Any], repetition: int) -> dict[str, Any]:
        from slbridge.runtime_contracts import InferenceRequest

        self._extract_ms = 0.0
        self._extract_frames = 0
        self.reset_vram_peak()
        request = InferenceRequest(
            request_id=f"e04-video-{sample['sample_id']}-{repetition}",
            deadline_monotonic=time.monotonic() + 300.0,
        )
        started = _now()
        result = self.backend.recognize_file(sample["trimmed"], request)
        wall_ms = (_now() - started) * 1000
        quality = result.quality or {}
        extract_ms = self._extract_ms
        pose_extract_ms = quality.get("pose_extract_ms") or extract_ms
        pose_preprocess_ms = quality.get("pose_preprocess_ms") or 0.0
        decode_ms = max(0.0, float(pose_extract_ms) - extract_ms)
        text = (result.prediction.text
                if result.prediction is not None and result.task_type == "text" else None)
        return {
            "video_decode_ms": decode_ms,
            "pose_ms": extract_ms + float(pose_preprocess_ms),
            "forward_ms": quality.get("model_forward_ms"),
            "text_decode_ms": (quality.get("generation_ms") or 0.0) + (quality.get("decode_ms") or 0.0),
            "queue_ms": 0.0,
            "e2e_ms": wall_ms,
            "raw": {
                "pose_extract_ms": _round(pose_extract_ms),
                "pose_instrumented_extract_ms": _round(extract_ms),
                "pose_preprocess_ms": _round(pose_preprocess_ms),
                "adapter_total_ms": _round(quality.get("adapter_total_ms")),
                "model_total_ms": _round(quality.get("model_total_ms")),
                "window_duration_seconds": _round(quality.get("window_duration_seconds")),
                "real_time_factor": _round(quality.get("real_time_factor")),
            },
            "input": {
                "window_frames": quality.get("window_frames"),
                "sampled_frames": quality.get("sampled_frames"),
                "source_frames": quality.get("source_frames"),
                "valid_body_fraction": _round(quality.get("valid_body_fraction")),
                "body_scale": _round(quality.get("body_scale")),
                "extractor_calls": self._extract_frames,
                "pose_cache_hit_frames": quality.get("pose_cache_hit_frames"),
            },
            "vram": self.vram_snapshot(),
            "text": text,
            "status": result.status,
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "UniSignBackend.recognize_file",
            "load": self.load_details,
            "runtime": self.backend.runtime_info,
        }

    def close(self) -> None:
        self.backend.close()


class SLRTHrnetNpzRunner(BaseRunner):
    """In-process SLRT backend over the precomputed HRNet NPZ.

    ``recognize_video`` already splits decode from window processing; the
    per-window pose lookup (``hrnet-npz``) is an array index, so ``pose_ms`` is
    near zero and is reported as measured, not omitted.
    """

    model_id = "slrt_hrnet_npz"

    def __init__(self) -> None:
        super().__init__()
        from slbridge.slrt.assets import SLRTConfig
        from slbridge.slrt.backend import SLRTBackend

        self.config = SLRTConfig.from_file(common.SLRT_CONFIG_HRNET_NPZ)
        self.backend = SLRTBackend(self.config)
        self._torch = self.backend.runtime.torch

    def load(self) -> None:
        # The backend constructor already loads the strict 938-key checkpoint.
        self.load_seconds = (self.backend.runtime.load_ms or 0.0) / 1000.0
        self.load_details = {**self.backend.runtime.info}
        self.load_details["window_size"] = self.config.window_size
        self.load_details["stride"] = self.config.stride
        self.load_details["vote_size"] = self.config.vote_size

    def infer(self, sample: dict[str, Any], repetition: int) -> dict[str, Any]:
        self.reset_vram_peak()
        started = _now()
        result = self.backend.recognize_video(sample["trimmed"], pose_npz=sample["npz"])
        wall_ms = (_now() - started) * 1000
        timings = result.get("timings") or {}

        def total(name: str) -> float | None:
            block = timings.get(name) or {}
            return block.get("total")

        return {
            "video_decode_ms": result.get("video_decode_ms"),
            "pose_ms": total("pose_ms"),
            "forward_ms": total("inference_ms"),
            "text_decode_ms": None,  # no text head in this SLRT checkpoint
            "queue_ms": 0.0,
            "e2e_ms": wall_ms,
            "raw": {
                "frame_preprocess_ms": _round(total("frame_preprocess_ms")),
                "preprocess_ms": _round(total("preprocess_ms")),
                "window_total_ms": _round(total("window_total_ms")),
                "wall_seconds_backend": _round(result.get("wall_seconds")),
                "processing_fps": _round(result.get("processing_fps")),
                "first_stable_wall_latency_ms": _round(result.get("first_stable_wall_latency_ms")),
                "windows_processed": result.get("windows_processed"),
                "frames_received": result.get("frames_received"),
                "dropped_frames": result.get("dropped_frames"),
                "quality_rejected_frames": result.get("quality_rejected_frames"),
            },
            "input": {
                "frames_received": result.get("frames_received"),
                "windows_processed": result.get("windows_processed"),
                "pose_backend": "hrnet-npz",
                "video_sha256": (result.get("source") or {}).get("video_sha256"),
            },
            "vram": self.vram_snapshot(),
            "gloss": result.get("gloss_sequence") or [],
            "status": result.get("status"),
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "SLRTBackend.recognize_video",
            "load": self.load_details,
            "runtime": self.backend.runtime.info,
        }

    def close(self) -> None:
        self.backend.close()


def build_runner(model: str) -> BaseRunner:
    if model == "unisign_pose":
        return UniSignPoseRunner()
    if model == "unisign_video":
        return UniSignVideoRunner()
    if model == "slrt_hrnet_npz":
        return SLRTHrnetNpzRunner()
    raise SystemExit(f"unknown model: {model}")


# Errors that mean "the host/GPU could not allocate right now", not "the model
# is broken".  Pitfall §7: retry after resource contention before recording a
# failure.  On this 32 GB Windows host, Uni-Sign video load can hit
# ``CUDA error: out of memory`` during the mT5 warm-up although >13 GB VRAM is
# free, because the failure is allocation/commit pressure, not a model defect.
RETRYABLE_MARKERS = (
    "out of memory",
    "cuda error",
    "memory allocation failure",
    "acceleratorerror",
    "insufficient memory",
)
LOAD_RETRIES = 2
RETRY_SLEEP_SECONDS = 25.0


def _is_retryable_failure(error: Any) -> bool:
    if not error:
        return False
    text = str(error).lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def _release_cuda_after_failure() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort cleanup only
        pass


def build_runner_with_retry(model: str, *, retries: int = LOAD_RETRIES,
                            reason: str = "") -> tuple[BaseRunner, int]:
    """Build + load a runner, retrying only allocation-type failures."""
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        runner: BaseRunner | None = None
        try:
            runner = build_runner(model)
            runner.load()
            return runner, attempt
        except BaseException as error:  # noqa: BLE001 - recorded as evidence
            last_error = error
            if runner is not None:
                try:
                    runner.close()
                except Exception:  # noqa: BLE001
                    pass
            retryable = _is_retryable_failure(error)
            if attempt < retries and retryable:
                print(f"[retry] {reason}{model} load failed with "
                      f"{type(error).__name__}: {str(error)[:160]}; retry "
                      f"{attempt + 1}/{retries} after {RETRY_SLEEP_SECONDS:.0f}s",
                      flush=True)
                _release_cuda_after_failure()
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            raise
    raise last_error if last_error is not None else RuntimeError("runner load failed")


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------
def gpu_guard(phase: str, *, max_foreign_vram_mb: float, allow_shared: bool) -> dict[str, Any]:
    baseline = common.nvidia_smi()
    if baseline is None:
        raise SystemExit("nvidia-smi is unavailable; E04 needs a real GPU")
    device = baseline["devices"][0]
    if device["memory_used_mb"] > max_foreign_vram_mb and not allow_shared:
        raise SystemExit(
            f"GPU is busy for phase {phase!r}: {device['memory_used_mb']:.0f} MB used "
            f"(limit {max_foreign_vram_mb:.0f} MB). Wait for the GPU owner, or pass "
            "--allow-shared-gpu and mark the rows as shared."
        )
    return {
        "phase": phase,
        "gpu_baseline": baseline,
        "ram": common.memory_snapshot(),
        "shared_gpu_detected": bool(device["memory_used_mb"] > max_foreign_vram_mb),
    }


# ---------------------------------------------------------------------------
# phase implementations
# ---------------------------------------------------------------------------
def _phase_context(run_dir: Path, phase: str, model: str) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "phase": phase,
        "model": model,
        "trace_path": str(run_dir / "perf_trace.jsonl"),
    }


def cmd_cold(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    guard = gpu_guard("cold", max_foreign_vram_mb=args.max_foreign_vram_mb,
                      allow_shared=args.allow_shared_gpu)
    samples = common.attach_fps(common.select_pilot_samples(1))
    results = []
    for attempt in range(args.cold_repeats):
        out_path = run_dir / f"cold-{args.model}-{attempt}.json"
        log_path = run_dir / f"cold-{args.model}-{attempt}.log"
        command = [
            sys.executable, str(Path(__file__).resolve()), "cold_child",
            "--model", args.model, "--out", str(out_path),
            "--sample-json", json.dumps(samples[0], ensure_ascii=False),
            "--allow-shared-gpu", str(bool(args.allow_shared_gpu)),
        ]
        spawns = 0
        while True:
            spawns += 1
            started = _now()
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                      stderr=subprocess.STDOUT, env=_child_env())
            wall = _now() - started
            payload = common.read_json(out_path) if out_path.is_file() else {"status": "missing"}
            error = payload.get("error")
            if (payload.get("status") != "ok" and spawns <= args.cold_retries
                    and _is_retryable_failure(error)):
                print(f"[retry] cold {args.model} attempt {attempt} spawn {spawns} failed "
                      f"with allocation error; retrying after {RETRY_SLEEP_SECONDS:.0f}s",
                      flush=True)
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            break
        row = _sample_trace_row(
            run_id=args.run_id, phase="cold", model=args.model, sample=samples[0],
            repetition=attempt, timings=payload.get("timings") or {}, status=payload.get("status", "missing"),
            error=payload.get("error"), text=payload.get("text"), gloss=payload.get("gloss"),
            extra={
                "process_exit_code": proc.returncode,
                "process_wall_seconds": _round(wall, 3),
                "model_load_seconds": _round(payload.get("load_seconds"), 3),
                "first_inference_ms": _round((payload.get("timings") or {}).get("e2e_ms")),
                "cold_start_scope": "fresh process: imports + asset verification + model load + one inference",
                "spawn_attempts": spawns,
                "retryable_failure": bool(_is_retryable_failure(error)),
                "shared_gpu": guard["shared_gpu_detected"],
            },
        )
        common.append_jsonl(run_dir / "perf_trace.jsonl", [row])
        results.append(row)
    successful_loads = [row["model_load_seconds"] for row in results
                        if row.get("status") == "ok"]
    summary = {
        "phase": "cold",
        "model": args.model,
        "repeats": args.cold_repeats,
        "successful": len(successful_loads),
        "failed": len(results) - len(successful_loads),
        "gpu_guard": guard,
        "rows": results,
        "model_load_seconds": common.stats(successful_loads),
        "process_wall_seconds": common.stats(
            [row["process_wall_seconds"] for row in results if row.get("status") == "ok"]),
        "note": ("cold rows count fresh-process attempts; allocation-type failures are "
                 "retried (spawn_attempts > 1) and reported separately from model failures"),
    }
    common.write_json(run_dir / f"phase-cold-{args.model}.json", summary)
    print(json.dumps({"phase": "cold", "model": args.model,
                      "successful": summary["successful"],
                      "failed": summary["failed"],
                      "model_load_seconds": summary["model_load_seconds"]}, ensure_ascii=False))
    return 0


def cmd_cold_child(args: argparse.Namespace) -> int:
    sample = json.loads(args.sample_json)
    payload: dict[str, Any] = {"status": "error", "error": None}
    runner = None
    process_started = _now()
    try:
        runner = build_runner(args.model)
        runner.load()
        runner.preload([sample])
        payload["load_seconds"] = runner.load_seconds
        payload["runtime_info"] = runner.runtime_info()
        payload["vram_after_load"] = runner.vram_snapshot()
        timings = runner.infer(sample, 0)
        payload.update({"status": "ok", "timings": timings, "text": timings.get("text"),
                        "gloss": timings.get("gloss")})
        payload["vram_after_inference"] = runner.vram_snapshot()
    except BaseException as error:  # noqa: BLE001 - child must report failures
        payload.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-4000:],
            "oom": _is_failure_marker(str(error)),
        })
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                pass
    payload["process_seconds"] = _round(_now() - process_started, 3)
    common.write_json(args.out, payload)
    return 0 if payload["status"] == "ok" else 1


def _run_warm_loop(
    *, runner: BaseRunner, samples: list[dict[str, Any]], run_id: str, phase: str,
    repetitions: int, warmup: int, run_dir: Path, model: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shared warm/load loop: warm-up untimed, then timed inferences."""
    sampler = common.DeviceSampler(interval_seconds=DEVICE_INTERVAL_SECONDS).start()
    failures = 0
    oom = 0
    rows: list[dict[str, Any]] = []
    started = _now()
    timed = 0
    phase_seconds = 0.0
    try:
        for index in range(warmup):
            runner.infer(samples[index % len(samples)], -1 - index)
        runner.reset_vram_peak()
        iteration = 0
        while True:
            if deadline is not None and _now() >= deadline:
                break
            if deadline is None and timed >= repetitions:
                break
            sample = samples[iteration % len(samples)]
            iteration += 1
            try:
                timings = runner.infer(sample, timed)
                row = _sample_trace_row(
                    run_id=run_id, phase=phase, model=model, sample=sample, repetition=timed,
                    timings=timings, status=_normalize_status(timings.get("status")),
                    text=timings.get("text"), gloss=timings.get("gloss"),
                )
                rows.append(row)
            except BaseException as error:  # noqa: BLE001 - failures are data
                failures += 1
                oom += int(_is_failure_marker(str(error)))
                rows.append(_sample_trace_row(
                    run_id=run_id, phase=phase, model=model, sample=sample, repetition=timed,
                    timings={}, status="failed", error=f"{type(error).__name__}: {error}",
                ))
            timed += 1
        phase_seconds = _now() - started
    finally:
        device_samples = sampler.stop()
    summary = {
        "phase": phase,
        "model": model,
        "warmup_iterations": warmup,
        "timed_iterations": timed,
        "failures": failures,
        "oom_failures": oom,
        "phase_seconds": _round(phase_seconds, 2),
        "throughput_items_per_second": _round(timed / phase_seconds if phase_seconds else None, 4),
        "resident_vram_after": runner.vram_snapshot(),
        "runtime_info": runner.runtime_info(),
        "device_sampling": {
            "interval_seconds": DEVICE_INTERVAL_SECONDS,
            "scope": "whole_device_including_other_processes",
            "peak": None,
        },
    }
    if device_samples:
        used = [s["memory_used_mb"] for s in device_samples]
        util = [s["utilization_gpu_pct"] for s in device_samples]
        summary["device_sampling"]["peak"] = {
            "samples": len(device_samples),
            "peak_memory_used_mb": max(used),
            "mean_memory_used_mb": round(statistics.fmean(used), 1),
            "mean_utilization_gpu_pct": round(statistics.fmean(util), 1),
            "max_utilization_gpu_pct": max(util),
        }
        common.write_jsonl(run_dir / f"device-{phase}-{model}.jsonl", device_samples)
    return rows, summary


def cmd_warm(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    guard = gpu_guard("warm", max_foreign_vram_mb=args.max_foreign_vram_mb,
                      allow_shared=args.allow_shared_gpu)
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    runner, load_retries = build_runner_with_retry(args.model, reason="warm: ")
    runner.preload(samples)
    rows, summary = _run_warm_loop(
        runner=runner, samples=samples, run_id=args.run_id, phase="warm",
        repetitions=args.warm_repeats, warmup=args.warmup, run_dir=run_dir, model=args.model)
    common.append_jsonl(run_dir / "perf_trace.jsonl", rows)
    summary.update({"gpu_guard": guard, "load_seconds": runner.load_seconds,
                    "load_retries": load_retries,
                    "samples": [s["sample_id"] for s in samples]})
    common.write_json(run_dir / f"phase-warm-{args.model}.json", summary)
    runner.close()
    print(json.dumps({"phase": "warm", "model": args.model,
                      "timed_iterations": summary["timed_iterations"],
                      "failures": summary["failures"]}, ensure_ascii=False))
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    guard = gpu_guard("load", max_foreign_vram_mb=args.max_foreign_vram_mb,
                      allow_shared=args.allow_shared_gpu)
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    runner, load_retries = build_runner_with_retry(args.model, reason="load: ")
    runner.preload(samples)
    deadline = _now() + args.load_minutes * 60.0
    rows, summary = _run_warm_loop(
        runner=runner, samples=samples, run_id=args.run_id, phase="load",
        repetitions=0, warmup=args.warmup, run_dir=run_dir, model=args.model,
        deadline=deadline)
    common.append_jsonl(run_dir / "perf_trace.jsonl", rows)
    summary.update({"gpu_guard": guard, "load_seconds": runner.load_seconds,
                    "load_retries": load_retries,
                    "requested_load_minutes": args.load_minutes,
                    "actual_load_seconds": summary["phase_seconds"],
                    "samples": [s["sample_id"] for s in samples]})
    common.write_json(run_dir / f"phase-load-{args.model}.json", summary)
    runner.close()
    print(json.dumps({"phase": "load", "model": args.model,
                      "actual_load_seconds": summary["actual_load_seconds"],
                      "timed_iterations": summary["timed_iterations"]}, ensure_ascii=False))
    return 0


def _wait_for_file(path: Path, timeout: float) -> None:
    started = _now()
    while not path.is_file():
        if _now() - started > timeout:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _shared_gpu_flags(args: argparse.Namespace) -> list[str]:
    """Only pass boolean flags when true; ``str(False)`` is a truthy argv value."""
    flags = ["--max-foreign-vram-mb", str(args.max_foreign_vram_mb)]
    if args.allow_shared_gpu:
        flags.append("--allow-shared-gpu")
    return flags


def _spawn_child(args: argparse.Namespace, *, model: str, out_jsonl: Path, summary: Path,
                 ready: Path, go: Path, samples: list[dict[str, Any]],
                 log_path: Path) -> subprocess.Popen:
    command = [
        sys.executable, str(Path(__file__).resolve()), "child",
        "--model", model,
        "--samples-json", json.dumps(samples, ensure_ascii=False),
        "--out-jsonl", str(out_jsonl),
        "--summary", str(summary),
        "--ready-file", str(ready),
        "--go-file", str(go),
        "--warmup", str(args.warmup),
        "--run-id", args.run_id,
        *_shared_gpu_flags(args),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log_handle,
                               stderr=subprocess.STDOUT, env=_child_env())
    process._e04_log = log_handle  # type: ignore[attr-defined]
    return process


def cmd_child(args: argparse.Namespace) -> int:
    """Batch worker used by the serial and parallel phases."""
    samples = json.loads(args.samples_json)
    runner = None
    payload: dict[str, Any] = {"status": "error", "error": None, "model": args.model}
    ready = Path(args.ready_file)
    go = Path(args.go_file)
    try:
        runner, load_retries = build_runner_with_retry(args.model, reason="child: ")
        runner.preload(samples)
        for index in range(args.warmup):
            runner.infer(samples[index % len(samples)], -1 - index)
        runner.reset_vram_peak()
        ready.write_text(json.dumps({"pid": os.getpid(), "model": args.model,
                                     "ready_at": common.utc_now()}), encoding="utf-8")
        _wait_for_file(go, START_BARRIER_TIMEOUT_SECONDS)
        batch_started = _now()
        rows = []
        for index, sample in enumerate(samples):
            try:
                timings = runner.infer(sample, index)
                rows.append(_sample_trace_row(
                    run_id=args.run_id, phase="child", model=args.model, sample=sample,
                    repetition=index, timings=timings,
                    status=_normalize_status(timings.get("status")),
                    text=timings.get("text"), gloss=timings.get("gloss")))
            except BaseException as error:  # noqa: BLE001
                rows.append(_sample_trace_row(
                    run_id=args.run_id, phase="child", model=args.model, sample=sample,
                    repetition=index, timings={}, status="failed",
                    error=f"{type(error).__name__}: {error}"))
        batch_seconds = _now() - batch_started
        common.write_jsonl(Path(args.out_jsonl), rows)
        payload.update({
            "status": "ok",
            "batch_seconds": _round(batch_seconds, 3),
            "samples": len(samples),
            "failures": sum(1 for row in rows if row["status"] == "failed"),
            "resident_vram_after": runner.vram_snapshot(),
            "load_seconds": runner.load_seconds,
            "load_retries": load_retries,
            "runtime_info": runner.runtime_info(),
        })
    except BaseException as error:  # noqa: BLE001
        payload.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc()[-4000:]})
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                pass
    common.write_json(Path(args.summary), payload)
    return 0 if payload["status"] == "ok" else 1


def _collect_child(process: subprocess.Popen, rows_path: Path, summary_path: Path) -> dict[str, Any]:
    try:
        process.wait()
    finally:
        handle = getattr(process, "_e04_log", None)
        if handle is not None:
            handle.close()
    rows = common.read_jsonl(rows_path)
    summary = common.read_json(summary_path) if summary_path.is_file() else {"status": "missing"}
    summary["process_exit_code"] = process.returncode
    return {"rows": rows, "summary": summary}


def cmd_serial(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    guard = gpu_guard("serial", max_foreign_vram_mb=args.max_foreign_vram_mb,
                      allow_shared=args.allow_shared_gpu)
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    results = {}
    started = _now()
    for model in ("unisign_video", "slrt_hrnet_npz"):
        out_jsonl = run_dir / f"serial-{model}.jsonl"
        summary_path = run_dir / f"serial-{model}-summary.json"
        ready = run_dir / f"serial-{model}.ready"
        go = run_dir / f"serial-{model}.go"
        go.write_text("go", encoding="utf-8")
        log_path = run_dir / f"serial-{model}.log"
        process = _spawn_child(args, model=model, out_jsonl=out_jsonl, summary=summary_path,
                               ready=ready, go=go, samples=samples, log_path=log_path)
        collected = _collect_child(process, out_jsonl, summary_path)
        rows = [dict(row, phase="serial") for row in collected["rows"]]
        common.append_jsonl(run_dir / "perf_trace.jsonl", rows)
        results[model] = dict(collected["summary"],
                              batch_seconds_from_trace=_round(
                                  sum(row["e2e_ms"] for row in rows) / 1000.0, 3))
    total = _now() - started
    summary = {
        "phase": "serial",
        "models": list(results),
        "samples": [s["sample_id"] for s in samples],
        "gpu_guard": guard,
        "wall_seconds_including_load": _round(total, 2),
        "sum_batch_seconds": _round(sum(item.get("batch_seconds") or 0 for item in results.values()), 3),
        "per_model": results,
        "note": "Models run one after another on the same GPU; batch_seconds excludes load/warm-up.",
    }
    common.write_json(run_dir / "phase-serial.json", summary)
    print(json.dumps({"phase": "serial", "sum_batch_seconds": summary["sum_batch_seconds"]},
                     ensure_ascii=False))
    return 0


def cmd_parallel(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    guard = gpu_guard("parallel", max_foreign_vram_mb=args.max_foreign_vram_mb,
                      allow_shared=args.allow_shared_gpu)
    ram = common.memory_snapshot()
    available = ram.get("available_gb")
    if (available is not None and available < args.min_available_ram_gb
            and not args.allow_low_ram):
        summary = {
            "phase": "parallel",
            "status": "BLOCKED_LOW_RAM",
            "reason": (f"available system RAM {available} GB < gate "
                       f"{args.min_available_ram_gb} GB; a two-process run would risk "
                       "OOM that is indistinguishable from a model failure"),
            "gpu_guard": guard,
            "ram": ram,
            "created_at": common.utc_now(),
        }
        common.write_json(run_dir / "phase-parallel.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
        return 2
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    stamp = str(int(time.time() * 1000))
    go = run_dir / f"parallel-{stamp}.go"
    sampler = common.DeviceSampler(interval_seconds=0.5).start()
    children: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    started = _now()
    barrier_seconds = compute_seconds = 0.0
    device_samples: list[dict[str, Any]] = []
    try:
        for model in ("unisign_video", "slrt_hrnet_npz"):
            out_jsonl = run_dir / f"parallel-{model}.jsonl"
            summary_path = run_dir / f"parallel-{model}-summary.json"
            ready = run_dir / f"parallel-{model}.ready"
            if ready.is_file():
                ready.unlink()
            children[model] = {
                "process": _spawn_child(args, model=model, out_jsonl=out_jsonl,
                                        summary=summary_path, ready=ready, go=go,
                                        samples=samples,
                                        log_path=run_dir / f"parallel-{model}.log"),
                "rows": out_jsonl,
                "summary": summary_path,
                "ready": ready,
            }
        for model, child in children.items():
            _wait_for_file(child["ready"], START_BARRIER_TIMEOUT_SECONDS)
        barrier_seconds = _now() - started
        go.write_text(json.dumps({"at": common.utc_now()}), encoding="utf-8")
        compute_started = _now()
        for model, child in children.items():
            results[model] = _collect_child(child["process"], child["rows"], child["summary"])
        compute_seconds = _now() - compute_started
    except BaseException:
        for child in children.values():
            process = child["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
            handle = getattr(process, "_e04_log", None)
            if handle is not None:
                handle.close()
        raise
    finally:
        device_samples = sampler.stop()
    merged = []
    for model, collected in results.items():
        merged.extend(dict(row, phase="parallel") for row in collected["rows"])
    common.append_jsonl(run_dir / "perf_trace.jsonl", merged)
    if device_samples:
        common.write_jsonl(run_dir / "device-parallel.jsonl", device_samples)
    used = [sample["memory_used_mb"] for sample in device_samples]
    util = [sample["utilization_gpu_pct"] for sample in device_samples]
    summary = {
        "phase": "parallel",
        "models": list(results),
        "samples": [s["sample_id"] for s in samples],
        "gpu_guard": guard,
        "barrier_wait_seconds": _round(barrier_seconds, 2),
        "compute_wall_seconds": _round(compute_seconds, 2),
        "child_batch_seconds": {model: collected["summary"].get("batch_seconds")
                                for model, collected in results.items()},
        "per_model": {model: collected["summary"] for model, collected in results.items()},
        "device_sampling": {
            "interval_seconds": 0.5,
            "samples": len(device_samples),
            "peak_memory_used_mb": max(used) if used else None,
            "mean_memory_used_mb": round(statistics.fmean(used), 1) if used else None,
            "mean_utilization_gpu_pct": round(statistics.fmean(util), 1) if util else None,
            "max_utilization_gpu_pct": max(util) if util else None,
            "scope": "whole_device_including_both_children",
        },
        "note": ("Both processes load their model before the start barrier; only the "
                 "batch window is compared with serial. Two processes on one GPU do not "
                 "add compute throughput automatically."),
    }
    common.write_json(run_dir / "phase-parallel.json", summary)
    print(json.dumps({"phase": "parallel", "compute_wall_seconds": summary["compute_wall_seconds"],
                      "peak_device_memory_used_mb": summary["device_sampling"]["peak_memory_used_mb"]},
                     ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "run_id", "mode", "model", "samples", "successes", "failures", "oom_failures",
    "wall_seconds", "batch_seconds", "p50_e2e_ms", "p95_e2e_ms", "p50_forward_ms",
    "p95_forward_ms", "p50_pose_ms", "p95_pose_ms", "peak_torch_vram_mb",
    "peak_device_memory_used_mb", "throughput_items_per_second", "mean_rtf",
    "speedup_vs_serial", "notes",
]


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    segment_stats = {}
    for key in common.SEGMENT_KEYS + ("rtf",):
        segment_stats[key] = common.stats([row.get(key) for row in ok])
    torch_peaks = [row.get("vram", {}).get("torch_peak_allocated_mb") for row in ok]
    resident = [row.get("vram", {}).get("torch_resident_allocated_mb") for row in ok]
    timestamps = [row.get("created_at") for row in rows if row.get("created_at")]
    return {
        "attempted": len(rows),
        "successes": len(ok),
        "failures": len(failed),
        "failure_rate": round(len(failed) / len(rows), 4) if rows else None,
        "oom_failures": sum(1 for row in failed if _is_failure_marker(row.get("error"))),
        "segments": segment_stats,
        "peak_torch_vram_mb": max([v for v in torch_peaks if v is not None], default=None),
        "resident_vram_mb": statistics.fmean([v for v in resident if v is not None])
        if any(v is not None for v in resident) else None,
        "first_at": min(timestamps) if timestamps else None,
        "last_at": max(timestamps) if timestamps else None,
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    trace_path = run_dir / "perf_trace.jsonl"
    rows = common.read_jsonl(trace_path)
    if not rows:
        raise SystemExit(f"no trace rows in {trace_path}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["phase"], row["model"]), []).append(row)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "run_id": rows[0]["run_id"],
        "created_at": common.utc_now(),
        "trace_rows": len(rows),
        "groups": {},
    }
    for (phase, model), group_rows in sorted(groups.items()):
        metrics["groups"][f"{phase}/{model}"] = _group_stats(group_rows)

    # phase-level wall numbers from the phase summaries
    for name in ("phase-warm-unisign_pose", "phase-warm-unisign_video",
                 "phase-warm-slrt_hrnet_npz", "phase-load-unisign_pose",
                 "phase-load-unisign_video", "phase-load-slrt_hrnet_npz",
                 "phase-serial", "phase-parallel"):
        path = run_dir / f"{name}.json"
        if path.is_file():
            payload = common.read_json(path)
            payload.pop("device_samples", None)
            payload.pop("rows", None)
            metrics.setdefault("phase_summaries", {})[name] = payload

    # serial vs parallel CSV
    serial = common.read_json(run_dir / "phase-serial.json") if (run_dir / "phase-serial.json").is_file() else {}
    parallel = common.read_json(run_dir / "phase-parallel.json") if (run_dir / "phase-parallel.json").is_file() else {}
    serial_batches = serial.get("per_model", {})
    parallel_batches = parallel.get("per_model", {})
    serial_sum = serial.get("sum_batch_seconds")
    parallel_wall = parallel.get("compute_wall_seconds")
    csv_rows = []
    for mode, source in (("serial", serial_batches), ("parallel", parallel_batches)):
        for model, block in source.items():
            model_stats = metrics["groups"].get(f"{mode}/{model}", {})
            csv_rows.append({
                "run_id": metrics["run_id"],
                "mode": mode,
                "model": model,
                "samples": block.get("samples"),
                "successes": model_stats.get("successes"),
                "failures": model_stats.get("failures"),
                "oom_failures": model_stats.get("oom_failures"),
                "wall_seconds": parallel_wall if mode == "parallel" else serial.get("wall_seconds_including_load"),
                "batch_seconds": block.get("batch_seconds"),
                "p50_e2e_ms": _round((model_stats.get("segments") or {}).get("e2e_ms", {}).get("p50")),
                "p95_e2e_ms": _round((model_stats.get("segments") or {}).get("e2e_ms", {}).get("p95")),
                "p50_forward_ms": _round((model_stats.get("segments") or {}).get("forward_ms", {}).get("p50")),
                "p95_forward_ms": _round((model_stats.get("segments") or {}).get("forward_ms", {}).get("p95")),
                "p50_pose_ms": _round((model_stats.get("segments") or {}).get("pose_ms", {}).get("p50")),
                "p95_pose_ms": _round((model_stats.get("segments") or {}).get("pose_ms", {}).get("p95")),
                "peak_torch_vram_mb": model_stats.get("peak_torch_vram_mb"),
                "peak_device_memory_used_mb": (parallel.get("device_sampling", {}) or {}).get("peak_memory_used_mb")
                if mode == "parallel" else None,
                "throughput_items_per_second": _round(
                    (block.get("samples") or 0) / block["batch_seconds"], 4)
                if block.get("batch_seconds") else None,
                "mean_rtf": _round((model_stats.get("segments") or {}).get("rtf", {}).get("mean"), 4),
                "speedup_vs_serial": None,
                "notes": "load/warm-up excluded from batch_seconds",
            })
    for row in csv_rows:
        if row["mode"] == "parallel" and row["samples"]:
            serial_one = (serial_batches.get(row["model"]) or {}).get("batch_seconds")
            if serial_one and row.get("batch_seconds"):
                row["speedup_vs_serial"] = _round(serial_one / row["batch_seconds"], 4)
    import csv as csv_module

    csv_path = run_dir / "parallel_vs_serial.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
    metrics["serial_vs_parallel"] = {
        "serial": serial,
        "parallel": parallel,
        "serial_sum_batch_seconds": serial_sum,
        "parallel_compute_wall_seconds": parallel_wall,
        "serial_sum_over_parallel_wall": _round(serial_sum / parallel_wall, 4)
        if serial_sum and parallel_wall else None,
        "interpretation": (
            "serial_sum_over_parallel_wall > 1 means the parallel compute window was "
            "shorter than the sum of the two serial batches (overlap paid off) -- but "
            "inspect per-model batch seconds: each model is usually slower under "
            "contention, so per-sample latency/RTF degrades even when total wall drops. "
            "< 1 means contention made the parallel window longer than serial."),
    }
    common.write_json(run_dir / "metrics.json", metrics)

    # human-readable markdown summary
    lines = [f"# E04 measured summary ({metrics['run_id']})", ""]
    lines.append(common.md_table(
        ["phase/model", "attempts", "ok", "fail", "P50 e2e ms", "P95 e2e ms",
         "P50 fwd ms", "P50 pose ms", "mean RTF", "peak torch VRAM MB"],
        [
            [name, block["attempted"], block["successes"], block["failures"],
             _round(block["segments"]["e2e_ms"]["p50"]),
             _round(block["segments"]["e2e_ms"]["p95"]),
             _round(block["segments"]["forward_ms"]["p50"]),
             _round(block["segments"]["pose_ms"]["p50"]),
             _round(block["segments"]["rtf"]["mean"], 4),
             block["peak_torch_vram_mb"]]
            for name, block in sorted(metrics["groups"].items())
        ]))
    lines.append("")
    lines.append(f"- serial sum of batch seconds: {serial_sum}")
    lines.append(f"- parallel compute wall seconds: {parallel_wall}")
    lines.append("- serial-sum / parallel-wall: "
                 f"{metrics['serial_vs_parallel']['serial_sum_over_parallel_wall']}")
    (run_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# all-in-one orchestration
# ---------------------------------------------------------------------------
def _call_self(argv: list[str]) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), *argv]
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=str(ROOT), env=_child_env())


def cmd_all(args: argparse.Namespace) -> int:
    run_dir = common.ensure_dir(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        manifest = common.base_manifest(args.run_id, "E04", purpose=(
            "Uni-Sign(pose/video) and SLRT(hrnet-npz) cold/warm/continuous-load "
            "segmented latency, VRAM, serial vs parallel on one GPU"))
        common.write_json(manifest_path, manifest)
    external_check = common.nvidia_smi()
    if external_check is None:
        raise SystemExit("nvidia-smi unavailable; refusing to run E04")
    device = external_check["devices"][0]
    if device["memory_used_mb"] > args.max_foreign_vram_mb and not args.allow_shared_gpu:
        raise SystemExit(
            f"GPU is busy: {device['memory_used_mb']:.0f} MB used by other processes "
            f"(limit {args.max_foreign_vram_mb:.0f} MB). Wait for the owner or pass "
            "--allow-shared-gpu."
        )
    common.update_manifest(
        run_dir, gpu_baseline_at_start=external_check,
        model_assets=common.model_assets_manifest(),
        environment=common.environment_fingerprint(),
        planned={
            "cold_repeats": args.cold_repeats,
            "warm_repeats": args.warm_repeats,
            "warmup_iterations": args.warmup,
            "load_minutes": args.load_minutes,
            "samples_per_phase": args.samples,
            "parallel_batch_samples": args.samples,
            "models": list(MODELS),
        },
        commands_to_reproduce=(
            f"{sys.executable} {Path(__file__).resolve()} all --run-id {args.run_id} "
            f"--run-dir {run_dir} --cold-repeats {args.cold_repeats} "
            f"--warm-repeats {args.warm_repeats} --load-minutes {args.load_minutes}"),
    )
    exit_codes: dict[str, int] = {}
    common_flags = [
        "--run-id", args.run_id, "--run-dir", str(run_dir),
        "--warmup", str(args.warmup), *_shared_gpu_flags(args),
    ]
    for model in MODELS:
        phase_flags = {
            "cold": ["--cold-repeats", str(args.cold_repeats),
                     "--cold-retries", str(args.cold_retries)],
            "warm": ["--warm-repeats", str(args.warm_repeats),
                     "--samples", str(args.samples)],
            "load": ["--load-minutes", str(args.load_minutes),
                     "--samples", str(args.samples)],
        }
        for phase in ("cold", "warm", "load"):
            code = _call_self([
                phase, "--model", model, *common_flags, *phase_flags[phase],
            ])
            exit_codes[f"{phase}/{model}"] = code
            if code != 0:
                print(f"[warn] phase {phase}/{model} exited {code}; continuing to keep evidence",
                      flush=True)
    for phase in ("serial", "parallel"):
        extra: list[str] = []
        if phase == "parallel":
            extra = ["--min-available-ram-gb", str(args.min_available_ram_gb)]
            if args.allow_low_ram:
                extra.append("--allow-low-ram")
        code = _call_self([
            phase, *common_flags,
            "--samples", str(max(2, min(args.samples, 5))), "--warmup", "1", *extra,
        ])
        exit_codes[phase] = code
    code = _call_self(["summarize", "--run-id", args.run_id, "--run-dir", str(run_dir)])
    exit_codes["summarize"] = code
    failures = [name for name, value in exit_codes.items() if value != 0]
    common.update_manifest(
        run_dir,
        run_status="PASS" if not failures else "PARTIAL",
        finished_at=common.utc_now(),
        phase_exit_codes=exit_codes,
        failures=failures,
        output_files=sorted(path.name for path in run_dir.iterdir() if path.is_file()),
    )
    print(json.dumps({"run_dir": str(run_dir), "exit_codes": exit_codes, "failures": failures},
                     ensure_ascii=False))
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", required=True)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--max-foreign-vram-mb", type=float, default=3000.0,
                       help="refuse to start if whole-device memory_used exceeds this")
        p.add_argument("--allow-shared-gpu", nargs="?", const=True, default=False)
        p.add_argument("--warmup", type=int, default=2)

    p_cold = sub.add_parser("cold")
    add_common(p_cold)
    p_cold.add_argument("--model", choices=MODELS, required=True)
    p_cold.add_argument("--cold-repeats", type=int, default=3)
    p_cold.add_argument("--cold-retries", type=int, default=2,
                        help="extra fresh-process spawns for allocation-type load failures")

    p_cold_child = sub.add_parser("cold_child")
    p_cold_child.add_argument("--model", choices=MODELS, required=True)
    p_cold_child.add_argument("--out", type=Path, required=True)
    p_cold_child.add_argument("--sample-json", required=True)
    p_cold_child.add_argument("--allow-shared-gpu", default="False")

    p_warm = sub.add_parser("warm")
    add_common(p_warm)
    p_warm.add_argument("--model", choices=MODELS, required=True)
    p_warm.add_argument("--warm-repeats", type=int, default=30)
    p_warm.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)

    p_load = sub.add_parser("load")
    add_common(p_load)
    p_load.add_argument("--model", choices=MODELS, required=True)
    p_load.add_argument("--load-minutes", type=float, default=2.0)
    p_load.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)

    p_child = sub.add_parser("child")
    p_child.add_argument("--run-id", required=True)
    p_child.add_argument("--model", choices=MODELS, required=True)
    p_child.add_argument("--samples-json", required=True)
    p_child.add_argument("--out-jsonl", type=Path, required=True)
    p_child.add_argument("--summary", type=Path, required=True)
    p_child.add_argument("--ready-file", required=True)
    p_child.add_argument("--go-file", required=True)
    p_child.add_argument("--warmup", type=int, default=1)
    p_child.add_argument("--max-foreign-vram-mb", type=float, default=3000.0)
    p_child.add_argument("--allow-shared-gpu", default="False")

    p_serial = sub.add_parser("serial")
    add_common(p_serial)
    p_serial.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)

    p_parallel = sub.add_parser("parallel")
    add_common(p_parallel)
    p_parallel.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    p_parallel.add_argument("--min-available-ram-gb", type=float, default=5.0)
    p_parallel.add_argument("--allow-low-ram", nargs="?", const=True, default=False)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--run-id", required=True)
    p_sum.add_argument("--run-dir", type=Path, required=True)

    p_all = sub.add_parser("all")
    p_all.add_argument("--run-id", required=True)
    p_all.add_argument("--run-dir", type=Path)
    p_all.add_argument("--cold-repeats", type=int, default=3)
    p_all.add_argument("--warm-repeats", type=int, default=30)
    p_all.add_argument("--load-minutes", type=float, default=2.0)
    p_all.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    p_all.add_argument("--warmup", type=int, default=2)
    p_all.add_argument("--max-foreign-vram-mb", type=float, default=3000.0)
    p_all.add_argument("--allow-shared-gpu", nargs="?", const=True, default=False)
    p_all.add_argument("--min-available-ram-gb", type=float, default=5.0)
    p_all.add_argument("--allow-low-ram", nargs="?", const=True, default=False)
    p_all.add_argument("--cold-retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "run_dir", None) is None:
        run_id = getattr(args, "run_id", None)
        if run_id is not None:
            args.run_dir = common.RUNS_DIR / run_id
    handlers = {
        "cold": cmd_cold,
        "cold_child": cmd_cold_child,
        "warm": cmd_warm,
        "load": cmd_load,
        "child": cmd_child,
        "serial": cmd_serial,
        "parallel": cmd_parallel,
        "summarize": cmd_summarize,
        "all": cmd_all,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
