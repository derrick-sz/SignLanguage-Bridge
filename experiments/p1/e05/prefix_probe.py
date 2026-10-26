"""E05 -- observed-prefix stress test and true-online feasibility for P-1.

What this probe does
--------------------
``已观测前缀`` means the model receives only the first N% of the *observed*
pose frames of a sentence clip.  No future frame is copied, extrapolated or
padded; if a model cannot accept short input it is reported as a failure.  This
is **not** a live stream: it is a frozen-prefix offline replay that measures how
early the sentence translation becomes stable.

Primary path (pose): the released CSL-Daily pose archive (``start``/``end``
cropped as upstream ``S2T_Dataset.load_pose`` does) is truncated to
12.5%/25%/50%/75%/100% and fed to the pinned Uni-Sign mT5 checkpoint
(beam=4, max_new_tokens=100, no padding of future frames).

Secondary path (video): the same fractions of the trimmed sentence mp4 are
decoded and run through the pinned video end-to-end runtime (RTMPose wholebody
133 -> mT5), either as in-memory ``CapturedClip`` (default) or as a physically
truncated mp4 (``--video-mode file``).

Metrics (all defined for *observed prefixes*, not for word-level accuracy):
``first_match_prefix``   first fraction whose text equals the 100% text;
``first_stable_prefix``  first fraction after which the text never changes again
                         (the final text's first appearance that survives);
rewrite rate             share of adjacent fraction pairs whose text changes;
empty rate               share of fractions with empty output;
prefix-vs-100% similarity character edit similarity of each prefix output to
                         the 100% output (stability, not accuracy);
history consistency      normalized text at 100% vs the authors' released
                         ``test_tmp_pres.txt`` snapshot (same checkpoint family,
                         different frame sampling; not a quality claim).

Timings reuse the runtime's own ``torch.cuda.synchronize`` boundaries, so
``forward_ms`` / ``text_decode_ms`` are real GPU-synchronized wall segments;
``e2e_ms`` here covers pose-window preparation + model forward + generation.

Usage (after the Lead confirms the GPU is free)::

    .conda-env\\python.exe experiments/p1/e05/prefix_probe.py run \
        --run-id e05-prefix-<stamp> --samples 10 --phases pose,video
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
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

DEFAULT_FRACTIONS = "0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0"
DEFAULT_VIDEO_FRACTIONS = "0.25,0.5,0.75,1.0"
DELIVERABLE_DIR = ROOT / "experiments" / "p1" / "e05"
FAILURE_MARKERS = (
    "out of memory", "OutOfMemory", "CUDA error", "Memory allocation failure",
    "_ArrayMemoryError", "Insufficient memory",
)


def _now() -> float:
    return time.perf_counter()


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


def _parse_fractions(text: str) -> list[float]:
    fractions = sorted({float(item) for item in text.split(",") if item.strip()})
    if not fractions or fractions[-1] > 1.0 or fractions[0] <= 0.0:
        raise SystemExit(f"fractions must be in (0, 1], got {fractions}")
    if 1.0 not in fractions:
        fractions.append(1.0)
    return fractions


def _prefix_frames(total_frames: int, fraction: float) -> int:
    """Observed prefix length; never padded, never longer than observed."""
    if total_frames < 2:
        return int(total_frames)
    if fraction >= 1.0:
        return int(total_frames)
    return max(2, min(int(total_frames), int(round(total_frames * fraction))))


def _fraction_label(fraction: float) -> str:
    return f"{round(fraction * 100, 1):g}%"


def require_gpu_idle(*, max_foreign_vram_mb: float, allow_shared: bool) -> dict[str, Any]:
    baseline = common.nvidia_smi()
    if baseline is None:
        raise SystemExit("nvidia-smi is unavailable; E05 needs the real GPU")
    device = baseline["devices"][0]
    if device["memory_used_mb"] > max_foreign_vram_mb and not allow_shared:
        raise SystemExit(
            f"GPU is busy: {device['memory_used_mb']:.0f} MB used (limit "
            f"{max_foreign_vram_mb:.0f} MB). Wait for the GPU owner or pass "
            "--allow-shared-gpu.")
    return {"gpu_baseline": baseline,
            "ram": common.memory_snapshot(),
            "shared_gpu_detected": bool(device["memory_used_mb"] > max_foreign_vram_mb)}


# ---------------------------------------------------------------------------
# pose-path runner (model-only; pose comes from the released pkl)
# ---------------------------------------------------------------------------
class PosePrefixRunner:
    def __init__(self, *, num_beams: int = 4, max_new_tokens: int = 100,
                 max_frames: int = 256, seed: int = 0) -> None:
        from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig

        self.config = UniSignConfig.from_bundle(common.UNISIGN_BUNDLE)
        self.runtime = TorchUniSignRuntime(self.config, device="cuda", dtype="bfloat16")
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.seed = seed
        self.load_seconds: float | None = None
        self.torch = None

    def load(self) -> None:
        from slbridge.uni_sign_preprocess import prepare_pose_window

        started = _now()
        self.runtime.load()
        self.torch = self.runtime._torch  # noqa: SLF001 - same package
        self.load_seconds = _now() - started
        warmup = np.zeros((8, 1, 133, 2), dtype=np.float32)
        warmup[:, 0, :, :] = 0.4
        scores = np.ones((8, 1, 133), dtype=np.float32)
        self.runtime.infer(prepare_pose_window(warmup, scores, max_frames=8),
                           num_beams=2, max_new_tokens=4)

    def _reset_peak(self) -> None:
        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()

    def infer(self, keypoints: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
        from slbridge.uni_sign_preprocess import prepare_pose_window

        if self.torch is not None:
            self.torch.manual_seed(self.seed)
        self._reset_peak()
        pose_started = _now()
        pose = prepare_pose_window(keypoints, scores, max_frames=self.max_frames)
        pose_done = _now()
        text, timings = self.runtime.infer(
            pose, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
        e2e_done = _now()
        vram: dict[str, Any] = {}
        if self.torch is not None:
            vram = {
                "torch_peak_allocated_mb": _round(
                    self.torch.cuda.max_memory_allocated() / 1024**2, 1),
                "torch_peak_reserved_mb": _round(
                    self.torch.cuda.max_memory_reserved() / 1024**2, 1),
                "scope": "process_torch_allocator_peak",
            }
        return {
            "text": text,
            "pose_ms": (pose_done - pose_started) * 1000,
            "forward_ms": timings.get("model_forward_ms"),
            "text_decode_ms": (timings.get("generation_ms") or 0.0) + (timings.get("decode_ms") or 0.0),
            "e2e_ms": (e2e_done - pose_started) * 1000,
            "timings_raw": {key: _round(value) for key, value in timings.items()},
            "sampled_frames": pose.quality.get("sampled_frames"),
            "body_scale": _round(pose.quality.get("body_scale"), 5),
            "valid_body_fraction": _round(pose.quality.get("valid_body_fraction"), 4),
            "vram": vram,
        }

    def close(self) -> None:
        self.runtime.close()


# ---------------------------------------------------------------------------
# video-path runner
# ---------------------------------------------------------------------------
class VideoPrefixRunner:
    """Pinned video end-to-end runtime; frames are decoded only up to the prefix."""

    def __init__(self) -> None:
        from slbridge.runtime_options import RuntimeOptions
        from slbridge.uni_sign import UniSignBackend, UniSignConfig

        self.config = UniSignConfig.from_bundle(common.UNISIGN_BUNDLE)
        self.backend = UniSignBackend(self.config, RuntimeOptions(device="cuda"))
        self.load_seconds: float | None = None
        self.torch = None
        self._extract_ms = 0.0

    def load(self) -> None:
        started = _now()
        self.backend.load_model()
        self.load_seconds = _now() - started
        runtime = getattr(self.backend, "_runtime", None)
        self.torch = getattr(runtime, "_torch", None)
        original = self.backend._extract_cached  # noqa: SLF001 - instrumentation only
        counter = {"ms": 0.0}

        def timed_extract(frame, *, source_id, frame_index):
            began = _now()
            try:
                return original(frame, source_id=source_id, frame_index=frame_index)
            finally:
                counter["ms"] += (_now() - began) * 1000

        self.backend._extract_cached = timed_extract  # noqa: SLF001
        self._extract_counter = counter

    def _reset_peak(self) -> None:
        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()

    def infer_clip(self, sample: dict[str, Any], prefix_frames: int) -> dict[str, Any]:
        import cv2

        from slbridge.runtime_contracts import InferenceRequest
        from slbridge.sources import CapturedClip, FramePacket

        capture = cv2.VideoCapture(str(sample["trimmed"]))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {sample['trimmed']}")
        frames = []
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            for index in range(prefix_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = index / fps if fps > 0 else float(index)
                frames.append(FramePacket(image=frame, timestamp=float(timestamp),
                                          frame_index=index,
                                          source_id=f"file:{sample['sample_id']}"))
        finally:
            capture.release()
        if len(frames) < 2:
            raise ValueError(f"prefix decoded {len(frames)} frames; Uni-Sign needs >=2")
        clip = CapturedClip(tuple(frames), source_id=f"file:{sample['sample_id']}")
        return self._infer(sample, clip=clip, prefix_frames=len(frames),
                           request_id=f"e05-video-clip-{sample['sample_id']}-{prefix_frames}")

    def infer_file(self, sample: dict[str, Any], prefix_frames: int,
                   video_path: Path) -> dict[str, Any]:
        import cv2

        capture = cv2.VideoCapture(str(sample["trimmed"]))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {sample['trimmed']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        writer = None
        written = 0
        try:
            for index in range(prefix_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps or 30.0,
                                             (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot create {video_path}")
                writer.write(frame)
                written += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        if written < 2:
            raise ValueError(f"prefix wrote {written} frames; Uni-Sign needs >=2")
        return self._infer(sample, video_path=video_path, prefix_frames=written,
                           request_id=f"e05-video-file-{sample['sample_id']}-{written}")

    def _infer(self, sample: dict[str, Any], *, clip: Any = None, video_path: Path | None = None,
               prefix_frames: int, request_id: str) -> dict[str, Any]:
        from slbridge.runtime_contracts import InferenceRequest

        self._extract_counter["ms"] = 0.0
        self._reset_peak()
        request = InferenceRequest(request_id=request_id,
                                   deadline_monotonic=time.monotonic() + 300.0)
        began = _now()
        if clip is not None:
            result = self.backend.recognize_clip(clip, request)
        else:
            result = self.backend.recognize_file(video_path, request)
        wall_ms = (_now() - began) * 1000
        quality = result.quality or {}
        text = (result.prediction.text
                if result.prediction is not None and result.task_type == "text" else None)
        return {
            "text": text,
            "status": result.status,
            "prefix_frames_decoded": prefix_frames,
            "video_decode_ms": max(0.0, (quality.get("pose_extract_ms") or 0.0)
                                   - self._extract_counter["ms"]),
            "pose_ms": self._extract_counter["ms"] + float(quality.get("pose_preprocess_ms") or 0.0),
            "forward_ms": quality.get("model_forward_ms"),
            "text_decode_ms": (quality.get("generation_ms") or 0.0) + (quality.get("decode_ms") or 0.0),
            "e2e_ms": wall_ms,
            "timings_raw": {
                "pose_extract_ms": _round(quality.get("pose_extract_ms")),
                "pose_preprocess_ms": _round(quality.get("pose_preprocess_ms")),
                "model_total_ms": _round(quality.get("model_total_ms")),
                "adapter_total_ms": _round(quality.get("adapter_total_ms")),
            },
            "window_frames": quality.get("window_frames"),
            "valid_body_fraction": _round(quality.get("valid_body_fraction"), 4),
            "body_scale": _round(quality.get("body_scale"), 5),
            "vram": {
                "torch_peak_allocated_mb": _round(
                    self.torch.cuda.max_memory_allocated() / 1024**2, 1) if self.torch else None,
                "scope": "process_torch_allocator_peak",
            },
        }

    def close(self) -> None:
        self.backend.close()


# ---------------------------------------------------------------------------
# per-sample prefix analysis
# ---------------------------------------------------------------------------
def analyse_sample(records: list[dict[str, Any]], fractions: list[float]) -> dict[str, Any]:
    """Derive first-match/first-stable/rewrite metrics from one sample's rows."""
    by_fraction = {round(row["fraction"], 6): row for row in records}
    texts = [common.normalize_text(by_fraction.get(round(f, 6), {}).get("text")) for f in fractions]
    final_text = texts[-1]
    first_match = None
    for fraction, text in zip(fractions, texts):
        if text == final_text:
            first_match = fraction
            break
    first_stable = None
    for index in range(len(fractions)):
        if all(text == final_text for text in texts[index:]):
            first_stable = fractions[index]
            break
    rewrite_flags = [texts[index] != texts[index + 1] for index in range(len(texts) - 1)]
    similarities = [common.char_edit_similarity(text, final_text) if final_text else None
                    for text in texts]
    return {
        "sample_id": records[0]["sample_id"] if records else None,
        "fractions": fractions,
        "texts": texts,
        "final_text": final_text,
        "final_text_empty": not bool(final_text),
        "first_match_fraction": first_match,
        "first_match_prefix_pct": _round(first_match * 100 if first_match is not None else None, 1),
        "first_stable_fraction": first_stable,
        "first_stable_prefix_pct": _round(first_stable * 100 if first_stable is not None else None, 1),
        "rewrite_count_adjacent": sum(rewrite_flags),
        "rewrite_pairs": len(rewrite_flags),
        "rewrite_rate_adjacent": (sum(rewrite_flags) / len(rewrite_flags)) if rewrite_flags else None,
        "empty_count": sum(1 for text in texts if not text),
        "empty_rate": (sum(1 for text in texts if not text) / len(texts)) if texts else None,
        "similarity_to_100": [round(value, 4) if value is not None else None
                              for value in similarities],
        "similarity_to_100_mean": (round(statistics.fmean(
            [value for value in similarities if value is not None]), 4)
            if any(value is not None for value in similarities) else None),
    }


def _aggregate(analyses: list[dict[str, Any]], fractions: list[float]) -> dict[str, Any]:
    stable = [item["first_stable_fraction"] for item in analyses
              if item["first_stable_fraction"] is not None]
    match = [item["first_match_fraction"] for item in analyses
             if item["first_match_fraction"] is not None]
    rewrite = [item["rewrite_rate_adjacent"] for item in analyses
               if item["rewrite_rate_adjacent"] is not None]
    empty = [item["empty_rate"] for item in analyses if item["empty_rate"] is not None]
    return {
        "samples": len(analyses),
        "first_stable_prefix_pct_counts": {
            _fraction_label(fraction): sum(1 for value in stable if abs(value - fraction) < 1e-9)
            for fraction in fractions
        },
        "first_match_prefix_pct_counts": {
            _fraction_label(fraction): sum(1 for value in match if abs(value - fraction) < 1e-9)
            for fraction in fractions
        },
        "first_stable_prefix_pct_mean": _round(statistics.fmean(
            [value * 100 for value in stable]), 2) if stable else None,
        "first_match_prefix_pct_mean": _round(statistics.fmean(
            [value * 100 for value in match]), 2) if match else None,
        "first_stable_prefix_pct_median": _round(statistics.median(
            [value * 100 for value in stable]), 2) if stable else None,
        "rewrite_rate_adjacent_mean": _round(statistics.fmean(rewrite), 4) if rewrite else None,
        "empty_rate_mean": _round(statistics.fmean(empty), 4) if empty else None,
        "final_text_empty_count": sum(1 for item in analyses if item["final_text_empty"]),
        "note": ("first_stable_prefix is measured on frozen observed prefixes; it is the "
                 "fraction after which the text never changes again. It is not a word-level "
                 "accuracy or a live-stream latency."),
    }


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------
def run_pose_phase(args: argparse.Namespace, run_dir: Path, fractions: list[float]) -> dict[str, Any]:
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    runner = PosePrefixRunner(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                              max_frames=args.max_frames, seed=args.seed)
    runner.load()
    upstream = common.parse_upstream_predictions() if args.use_historian else {}
    rows: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    failures = 0
    errors: list[dict[str, Any]] = []
    for sample in samples:
        keypoints, scores, info = common.load_pose_pkl(sample["pkl"])
        total_frames = int(keypoints.shape[0])
        sample_rows: list[dict[str, Any]] = []
        for fraction in fractions:
            frames = _prefix_frames(total_frames, fraction)
            prefix_kp = keypoints[:frames]
            prefix_scores = scores[:frames]
            input_hash = common.sha256_bytes(
                prefix_kp.tobytes(), prefix_scores.tobytes(),
                str(prefix_kp.shape).encode("utf-8"))
            base = {
                "schema_version": 1,
                "run_id": args.run_id,
                "created_at": common.utc_now(),
                "mode": "pose",
                "sample_id": sample["sample_id"],
                "signer_id": sample["signer_id"],
                "fraction": fraction,
                "fraction_pct": _round(fraction * 100, 1),
                "prefix_frames": frames,
                "total_frames": total_frames,
                "crop_start": info.get("start"),
                "crop_end": info.get("end"),
                "prefix_seconds": _round(frames / sample["fps"], 3) if sample.get("fps") else None,
                "video_seconds_full": _round(sample.get("video_seconds"), 3),
                "input_sha256": input_hash,
                "input_rule": "first N frames of the start/end-cropped released pkl; no padding",
                "model": "unisign_pose",
                "checkpoint_sha256": runner.config.checkpoint_sha256,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "shared_gpu": bool(args.allow_shared_gpu),
            }
            if frames >= total_frames and total_frames < 2:
                row = dict(base, status="failed", error="clip has fewer than 2 frames")
                rows.append(row)
                sample_rows.append(row)
                continue
            try:
                result = runner.infer(prefix_kp, prefix_scores)
                row = dict(base, status="ok", text=result["text"],
                           text_normalized=common.normalize_text(result["text"]),
                           **{key: _round(result[key]) for key in
                              ("pose_ms", "forward_ms", "text_decode_ms", "e2e_ms")},
                           sampled_frames=result["sampled_frames"],
                           pose_input_frames=frames,
                           body_scale=result["body_scale"],
                           valid_body_fraction=result["valid_body_fraction"],
                           vram=result["vram"],
                           timings_raw=result["timings_raw"])
            except BaseException as error:  # noqa: BLE001 - failures are evidence
                failures += 1
                row = dict(base, status="failed",
                           error=f"{type(error).__name__}: {error}")
                errors.append({"sample_id": sample["sample_id"],
                               "fraction": fraction,
                               "error": str(error)[:500]})
            rows.append(row)
            sample_rows.append(row)
        complete = [row for row in sample_rows if row.get("status") == "ok"]
        if len(complete) == len(fractions):
            analysis = analyse_sample(complete, fractions)
            if upstream and sample["sample_id"] in upstream:
                analysis["history_upstream_text"] = upstream[sample["sample_id"]]
                analysis["history_identical"] = (
                    common.normalize_text(analysis["final_text"]) == upstream[sample["sample_id"]])
                analysis["history_char_similarity"] = round(common.char_edit_similarity(
                    analysis["final_text"], upstream[sample["sample_id"]]), 4)
            analyses.append(analysis)
    # determinism check: same 100% input twice with the same seed
    determinism = {"checked": False, "identical": None, "texts": []}
    if args.determinism_check and samples:
        keypoints, scores, _ = common.load_pose_pkl(samples[0]["pkl"])
        repeat_texts = []
        for _ in range(2):
            result = runner.infer(keypoints, scores)
            repeat_texts.append(result["text"])
        determinism = {"checked": True, "sample_id": samples[0]["sample_id"],
                       "texts": repeat_texts,
                       "identical": repeat_texts[0].strip() == repeat_texts[1].strip(),
                       "seed": args.seed, "num_beams": args.num_beams}
    runner.close()
    aggregate = _aggregate(analyses, fractions)
    history_rows = [item for item in analyses if "history_identical" in item]
    aggregate.update({
        "history_upstream_available": bool(upstream),
        "history_identical_count": sum(1 for item in history_rows if item["history_identical"]),
        "history_compared": len(history_rows),
        "history_char_similarity_mean": _round(statistics.fmean(
            [item["history_char_similarity"] for item in history_rows]), 4) if history_rows else None,
        "history_scope": ("authors' released prediction snapshot vs our 100% prefix; same "
                          "checkpoint family but deterministic linspace frame sampling "
                          "instead of upstream random.sample, so not bit-identical"),
    })
    summary = {
        "phase": "pose",
        "load_seconds": runner.load_seconds,
        "fractions": fractions,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "rows": len(rows),
        "failures": failures,
        "errors": errors,
        "per_sample": analyses,
        "aggregate": aggregate,
        "determinism": determinism,
        "runtime_info": {
            "device": "cuda", "dtype": "bfloat16",
            "checkpoint_sha256": runner.config.checkpoint_sha256,
            "max_frames": args.max_frames,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    common.append_jsonl(run_dir / "prefix_probe.jsonl", rows)
    return summary


def _video_prefix_frame_count(sample: dict[str, Any]) -> int:
    """Frames actually decodable from the trimmed clip (declared count may lie)."""
    import cv2

    capture = cv2.VideoCapture(str(sample["trimmed"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {sample['trimmed']}")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise RuntimeError(f"no frames decoded from {sample['trimmed']}")
    return count if not declared or abs(declared - count) > 2 else declared


def run_video_phase(args: argparse.Namespace, run_dir: Path, fractions: list[float],
                    pose_summary: dict[str, Any] | None) -> dict[str, Any]:
    samples = common.attach_fps(common.select_pilot_samples(args.samples)[: args.video_samples])
    runner = VideoPrefixRunner()
    runner.load()
    modes = ["clip"] if args.video_mode == "clip" else (
        ["file"] if args.video_mode == "file" else ["clip", "file"])
    video_dir = run_dir / "prefix_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures = 0
    for sample in samples:
        total = _video_prefix_frame_count(sample)
        for fraction in fractions:
            frames = _prefix_frames(total, fraction)
            for mode in modes:
                base = {
                    "schema_version": 1,
                    "run_id": args.run_id,
                    "created_at": common.utc_now(),
                    "mode": f"video_{mode}",
                    "sample_id": sample["sample_id"],
                    "signer_id": sample["signer_id"],
                    "fraction": fraction,
                    "fraction_pct": _round(fraction * 100, 1),
                    "prefix_frames": frames,
                    "total_frames": total,
                    "prefix_seconds": _round(frames / sample["fps"], 3) if sample.get("fps") else None,
                    "video_seconds_full": _round(sample.get("video_seconds"), 3),
                    "model": "unisign_video",
                    "input_rule": ("first N decoded frames of the trimmed mp4, re-encoded for "
                                   "file mode; no future frames, no padding"),
                    "checkpoint_sha256": runner.config.checkpoint_sha256,
                    "shared_gpu": bool(args.allow_shared_gpu),
                }
                try:
                    if mode == "clip":
                        result = runner.infer_clip(sample, frames)
                    else:
                        path = video_dir / f"{sample['sample_id']}_{int(round(fraction * 1000))}.mp4"
                        result = runner.infer_file(sample, frames, path)
                    base.update({
                        "status": "ok" if result.get("status") == "recognized" else str(result.get("status")),
                        "text": result["text"],
                        "text_normalized": common.normalize_text(result["text"]),
                        "prefix_frames_decoded": result["prefix_frames_decoded"],
                        **{key: _round(result[key]) for key in
                           ("video_decode_ms", "pose_ms", "forward_ms", "text_decode_ms", "e2e_ms")},
                        "window_frames": result["window_frames"],
                        "valid_body_fraction": result["valid_body_fraction"],
                        "body_scale": result["body_scale"],
                        "vram": result["vram"],
                        "timings_raw": result["timings_raw"],
                    })
                except BaseException as error:  # noqa: BLE001
                    failures += 1
                    base.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                                 "traceback": traceback.format_exc()[-2000:]})
                rows.append(base)
    runner.close()
    common.append_jsonl(run_dir / "prefix_probe.jsonl", rows)

    # compare video prefix output with the pose-path prefix output (frontend differs)
    pose_by_key: dict[tuple[str, float], str] = {}
    if pose_summary is not None:
        for item in pose_summary.get("per_sample", []):
            for fraction, text in zip(item["fractions"], item["texts"]):
                pose_by_key[(item["sample_id"], round(fraction, 6))] = text
    comparisons = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["sample_id"], round(row["fraction"], 6))
        if key in pose_by_key:
            pose_text = pose_by_key[key]
            comparisons.append({
                "sample_id": row["sample_id"],
                "fraction": row["fraction"],
                "mode": row["mode"],
                "identical": common.normalize_text(row.get("text")) == common.normalize_text(pose_text),
                "char_similarity": round(common.char_edit_similarity(row.get("text"), pose_text), 4),
                "video_text": common.normalize_text(row.get("text")),
                "pose_text": pose_text,
            })
    aggregate = {
        "samples": len(samples),
        "rows": len(rows),
        "failures": failures,
        "successes": sum(1 for row in rows if row.get("status") == "ok"),
        "empty_outputs": sum(1 for row in rows if row.get("status") == "ok"
                              and not common.normalize_text(row.get("text"))),
        "by_fraction": {
            _fraction_label(fraction): {
                "attempts": sum(1 for row in rows if abs(row["fraction"] - fraction) < 1e-9),
                "successes": sum(1 for row in rows if abs(row["fraction"] - fraction) < 1e-9
                                 and row.get("status") == "ok"),
                "p50_e2e_ms": _round(common.percentile(
                    [row["e2e_ms"] for row in rows
                     if abs(row["fraction"] - fraction) < 1e-9 and row.get("e2e_ms")], 50)),
                "p50_pose_ms": _round(common.percentile(
                    [row["pose_ms"] for row in rows
                     if abs(row["fraction"] - fraction) < 1e-9 and row.get("pose_ms")], 50)),
                "p50_forward_ms": _round(common.percentile(
                    [row["forward_ms"] for row in rows
                     if abs(row["fraction"] - fraction) < 1e-9 and row.get("forward_ms")], 50)),
            }
            for fraction in fractions
        },
        "pose_vs_video": {
            "compared": len(comparisons),
            "identical": sum(1 for item in comparisons if item["identical"]),
            "mean_char_similarity": _round(statistics.fmean(
                [item["char_similarity"] for item in comparisons]), 4) if comparisons else None,
            "note": ("Different frontends (official released pose vs local RTMPose) and "
                     "different truncation handling; disagreement is expected and is not "
                     "a correctness label."),
            "details": comparisons,
        },
        "feasibility": ("video end-to-end accepts observed prefixes down to "
                        f"{min(row['prefix_frames'] for row in rows)} frames; no padding used"),
    }
    return {"phase": "video", "load_seconds": runner.load_seconds, "aggregate": aggregate,
            "modes": modes, "fractions": fractions,
            "sample_ids": [sample["sample_id"] for sample in samples]}


# ---------------------------------------------------------------------------
# summary rendering
# ---------------------------------------------------------------------------
def render_markdown(run_id: str, summary: dict[str, Any]) -> str:
    pose = summary.get("pose") or {}
    aggregate = pose.get("aggregate") or {}
    lines = [f"# E05 observed-prefix probe summary ({run_id})", ""]
    lines.append("> 已观测前缀 = 只给前 N% 的已观测姿态/视频帧，不填充未来帧；"
                 "这是冻结前缀离线回放，不是实时流。")
    lines.append("")
    if pose:
        lines.append("## Pose path (official released pkl -> pinned mT5, beam=4)")
        lines.append("")
        lines.append(f"- samples: {len(pose.get('sample_ids', []))}, rows: {pose.get('rows')}, "
                     f"failures: {pose.get('failures')}, load: {_round(pose.get('load_seconds'), 2)} s")
        lines.append(f"- first_stable_prefix mean: {aggregate.get('first_stable_prefix_pct_mean')}%, "
                     f"median: {aggregate.get('first_stable_prefix_pct_median')}%")
        lines.append(f"- first_match_prefix mean: {aggregate.get('first_match_prefix_pct_mean')}%")
        lines.append(f"- adjacent rewrite rate (mean over samples): "
                     f"{aggregate.get('rewrite_rate_adjacent_mean')}")
        lines.append(f"- empty rate (mean): {aggregate.get('empty_rate_mean')}; "
                     f"final-text empty: {aggregate.get('final_text_empty_count')}")
        history = aggregate.get("history_compared") or 0
        if history:
            lines.append(f"- history consistency vs authors' snapshot: "
                         f"{aggregate.get('history_identical_count')}/{history} identical, "
                         f"mean char similarity {aggregate.get('history_char_similarity_mean')}")
        determinism = pose.get("determinism") or {}
        if determinism.get("checked"):
            lines.append(f"- determinism (same 100% input twice, seed={determinism.get('seed')}): "
                         f"identical={determinism.get('identical')}")
        lines.append("")
        lines.append("| sample | " + " | ".join(
            _fraction_label(fraction) for fraction in pose.get("fractions", [])) + " | stable% |")
        lines.append("|" + "|".join(["---"] * (len(pose.get("fractions", [])) + 2)) + "|")
        for item in pose.get("per_sample", []):
            cells = [text if text else "∅" for text in item["texts"]]
            lines.append("| " + item["sample_id"] + " | " + " | ".join(cells) + " | "
                         + str(item.get("first_stable_prefix_pct")) + " |")
        lines.append("")
    video = summary.get("video") or {}
    if video:
        van = video.get("aggregate") or {}
        lines.append("## Video path (trimmed mp4 -> local RTMPose -> mT5)")
        lines.append("")
        lines.append(f"- modes: {video.get('modes')}, samples: {van.get('samples')}, "
                     f"successes: {van.get('successes')}/{van.get('rows')}, "
                     f"failures: {van.get('failures')}, empty: {van.get('empty_outputs')}")
        lines.append(f"- feasibility: {van.get('feasibility')}")
        for label, block in (van.get("by_fraction") or {}).items():
            lines.append(f"- {label}: ok {block['successes']}/{block['attempts']}, "
                         f"P50 e2e {block['p50_e2e_ms']} ms, pose {block['p50_pose_ms']} ms, "
                         f"forward {block['p50_forward_ms']} ms")
        compare = van.get("pose_vs_video") or {}
        if compare.get("compared"):
            lines.append(f"- pose-vs-video prefix agreement: {compare.get('identical')}"
                         f"/{compare.get('compared')} identical, mean char similarity "
                         f"{compare.get('mean_char_similarity')} (frontends differ)")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--deliverable-dir", type=Path, default=DELIVERABLE_DIR)
    run.add_argument("--samples", type=int, default=10,
                     help="pilot samples for the pose path (one per signer when <=10)")
    run.add_argument("--video-samples", type=int, default=10)
    run.add_argument("--fractions", default=DEFAULT_FRACTIONS)
    run.add_argument("--video-fractions", default=DEFAULT_VIDEO_FRACTIONS)
    run.add_argument("--phases", default="pose,video",
                     help="comma separated subset of pose,video")
    run.add_argument("--video-mode", choices=("clip", "file", "both"), default="clip")
    run.add_argument("--num-beams", type=int, default=4)
    run.add_argument("--max-new-tokens", type=int, default=100)
    run.add_argument("--max-frames", type=int, default=256)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-history-check", dest="use_historian", action="store_false")
    run.add_argument("--no-determinism-check", dest="determinism_check", action="store_false")
    run.add_argument("--max-foreign-vram-mb", type=float, default=3000.0)
    run.add_argument("--allow-shared-gpu", nargs="?", const=True, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error(f"unknown command {args.command}")
    run_id = args.run_id
    run_dir = Path(args.run_dir) if args.run_dir else common.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    deliverable_dir = Path(args.deliverable_dir)
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    fractions = _parse_fractions(args.fractions)
    video_fractions = _parse_fractions(args.video_fractions)
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    if any(phase not in ("pose", "video") for phase in phases):
        raise SystemExit(f"unknown phase in {phases}")

    guard = require_gpu_idle(max_foreign_vram_mb=args.max_foreign_vram_mb,
                             allow_shared=args.allow_shared_gpu)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        common.write_json(manifest_path, common.base_manifest(
            run_id, "E05",
            purpose=("frozen observed-prefix stress test (12.5/25/50/75/100%) on "
                     "Uni-Sign pose and video paths; first-stable prefix, rewrite "
                     "rate, future-frame audit")))
    common.update_manifest(
        run_dir,
        model_assets=common.model_assets_manifest(),
        environment=common.environment_fingerprint(),
        planned={"fractions": fractions, "video_fractions": video_fractions,
                 "pose_samples": args.samples, "video_samples": args.video_samples,
                 "phases": phases, "num_beams": args.num_beams,
                 "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                 "video_mode": args.video_mode},
        gpu_guard=guard,
    )
    started = _now()
    summary: dict[str, Any] = {"run_id": run_id, "created_at": common.utc_now(),
                               "gpu_guard": guard, "fractions": fractions,
                               "video_fractions": video_fractions}
    status = "PASS"
    if "pose" in phases:
        print("[e05] pose prefix phase ...", flush=True)
        summary["pose"] = run_pose_phase(args, run_dir, fractions)
        print(json.dumps({"phase": "pose", "aggregate": summary["pose"]["aggregate"]},
                         ensure_ascii=False, indent=2), flush=True)
    if "video" in phases:
        print("[e05] video prefix phase ...", flush=True)
        summary["video"] = run_video_phase(args, run_dir, video_fractions,
                                           summary.get("pose"))
        print(json.dumps({"phase": "video", "aggregate": {
            key: summary["video"]["aggregate"][key]
            for key in ("samples", "rows", "successes", "failures", "empty_outputs")}},
            ensure_ascii=False), flush=True)
    summary["elapsed_seconds"] = round(_now() - started, 2)
    pose_failures = int((summary.get("pose") or {}).get("failures") or 0)
    video_failures = int(((summary.get("video") or {}).get("aggregate") or {}).get("failures") or 0)
    summary["failures"] = {"pose": pose_failures, "video": video_failures}
    status = "PASS" if pose_failures == 0 and video_failures == 0 else "PARTIAL"
    common.write_json(run_dir / "prefix_summary.json", summary)
    markdown = render_markdown(run_id, summary)
    (run_dir / "prefix_summary.md").write_text(markdown, encoding="utf-8")
    (deliverable_dir / "prefix_probe_summary.md").write_text(markdown, encoding="utf-8")
    # JSONL deliverable copy (raw rows live in the run directory as well)
    rows = common.read_jsonl(run_dir / "prefix_probe.jsonl")
    common.write_jsonl(deliverable_dir / "prefix_probe.jsonl", rows)
    common.update_manifest(
        run_dir,
        run_status=status,
        finished_at=common.utc_now(),
        output_files=sorted(path.name for path in run_dir.iterdir() if path.is_file()),
        deliverables={"summary_md": str(deliverable_dir / "prefix_probe_summary.md"),
                      "jsonl": str(deliverable_dir / "prefix_probe.jsonl")},
    )
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
