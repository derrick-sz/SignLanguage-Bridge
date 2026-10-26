"""Sliding-window Uni-Sign experiment with pose reuse and latency metrics.

This is an offline or blocking-camera experiment. It is not a native streaming
model and it does not rewrite the CSL-Daily model into a causal encoder.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slbridge.runtime_contracts import InferenceRequest  # noqa: E402
from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.sources import CapturedClip, FramePacket  # noqa: E402
from slbridge.uni_sign import load_uni_sign_adapter  # noqa: E402


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return np.ascontiguousarray(frame)
    height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
    resized = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def open_input(args):
    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        return capture, f"video:{args.video.resolve()}", fps, False
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    return capture, f"camera:{args.camera}", 30.0, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uni-Sign sliding-window latency experiment")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=48)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.window_frames <= 256:
        parser.error("--window-frames must be in [2, 256]")
    if not 1 <= args.stride_frames <= args.window_frames:
        parser.error("--stride-frames must be in [1, window-frames]")
    if not 0 < args.max_seconds <= 120:
        parser.error("--max-seconds must be in (0, 120]")
    if not 0 < args.timeout_seconds <= 120:
        parser.error("--timeout-seconds must be in (0, 120]")

    capture, source_id, nominal_fps, is_camera = open_input(args)
    options = RuntimeOptions(device=args.device, device_id=args.device_id)
    adapter = load_uni_sign_adapter(args.bundle, options=options)
    rows = []
    texts = []
    states = []
    buffer: deque[FramePacket] = deque(maxlen=args.window_frames)
    load_started = time.perf_counter()
    stream_started = None
    window_end_wall = None
    decoded_frames = 0
    skipped_windows = 0
    truncated = False
    try:
        adapter.warmup()
        load_seconds = time.perf_counter() - load_started
        stream_started = time.perf_counter()
        while True:
            ok, frame = capture.read()
            now = time.perf_counter()
            if not ok:
                break
            elapsed = now - stream_started if is_camera else decoded_frames / nominal_fps
            if elapsed > args.max_seconds:
                truncated = True
                break
            frame = resize_frame(frame, args.max_width)
            timestamp = decoded_frames / nominal_fps if not is_camera else elapsed
            packet = FramePacket(image=frame, timestamp=float(timestamp),
                                 frame_index=decoded_frames, source_id=source_id)
            buffer.append(packet)
            decoded_frames += 1
            window_end_wall = now
            if len(buffer) < args.window_frames:
                continue
            if (decoded_frames - args.window_frames) % args.stride_frames:
                continue
            clip = CapturedClip(tuple(buffer), source_id=source_id)
            sequence = len(rows) + 1
            request = InferenceRequest(f"benchmark-uni-sign-{sequence}",
                                       deadline_monotonic=time.monotonic() + args.timeout_seconds)
            started = time.perf_counter()
            try:
                result, timing = adapter.infer(clip, request, input_kind="captured_clip")
            except TimeoutError:
                skipped_windows += 1
                continue
            returned = time.perf_counter()
            text = (result.prediction.text if result.prediction is not None and result.task_type == "text"
                    else None)
            rows.append({
                "window": sequence,
                "frame_index_start": buffer[0].frame_index,
                "frame_index_end": buffer[-1].frame_index,
                "video_start_seconds": buffer[0].timestamp,
                "video_end_seconds": buffer[-1].timestamp,
                "window_seconds": buffer[-1].timestamp - buffer[0].timestamp,
                "processing_ms": (returned - started) * 1000,
                "window_end_to_result_ms": (returned - window_end_wall) * 1000,
                "window_end_wall_from_stream_ms": (window_end_wall - stream_started) * 1000,
                "result_wall_from_stream_ms": (returned - stream_started) * 1000,
                "text": text,
                "status": result.status,
                "reason": result.reason,
                "stability_state": result.quality.get("stability_state"),
                "repeat_count": result.quality.get("stable_repeat_count"),
                "pose_ms": result.quality.get("pose_ms"),
                "pose_cache_hit_frames": result.quality.get("pose_cache_hit_frames"),
                "pose_new_frames": result.quality.get("pose_new_frames"),
                "model_total_ms": result.quality.get("model_total_ms"),
                "window_duration_seconds": result.quality.get("window_duration_seconds"),
                "real_time_factor": result.quality.get("real_time_factor"),
                "gpu_peak_allocated_mb": result.quality.get("gpu_peak_allocated_mb"),
                "result": result.to_dict(),
            })
            texts.append(text)
            states.append(result.quality.get("stability_state"))
    finally:
        capture.release()
        runtime = adapter.runtime_info
        adapter.close()
    elapsed_total = time.perf_counter() - stream_started
    output_load_seconds = load_seconds
    processing = [row["processing_ms"] for row in rows]
    end_to_result = [row["window_end_to_result_ms"] for row in rows]
    first_output_ms = rows[0]["result_wall_from_stream_ms"] if rows else None
    stable_rows = [row for row in rows if row["stability_state"] == "confirmed"]
    confirmed_end_to_result = [row["window_end_to_result_ms"] for row in stable_rows]
    confirmed_wall = [row["result_wall_from_stream_ms"] for row in stable_rows]
    stable_output_ms = stable_rows[0]["result_wall_from_stream_ms"] if stable_rows else None
    change_count = sum(a != b for a, b in zip(texts, texts[1:]))
    duplicate_count = sum(a is not None and a == b for a, b in zip(texts, texts[1:]))
    withdrawn_count = sum(
        previous_state == "confirmed" and current_text is not None and previous_text is not None and current_text != previous_text
        for previous_state, current_text, previous_text in zip(states, texts[1:], texts[:-1])
    )
    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "source": {"id": source_id, "camera": is_camera, "nominal_fps": nominal_fps,
                   "max_width": args.max_width, "decoded_frames": decoded_frames,
                   "max_seconds": args.max_seconds, "truncated": truncated},
        "config": {"window_frames": args.window_frames, "stride_frames": args.stride_frames,
                   "device": args.device, "bundle": str(args.bundle.resolve())},
        "load_seconds": output_load_seconds,
        "runtime": runtime,
        "windows": rows,
        "metrics": {
            "window_count": len(rows),
            "skipped_windows": skipped_windows,
            "processing_ms_p50": percentile(processing, 50),
            "processing_ms_p95": percentile(processing, 95),
            "window_end_to_result_ms_p50": percentile(end_to_result, 50),
            "window_end_to_result_ms_p95": percentile(end_to_result, 95),
            "first_output_ms": first_output_ms,
            "first_stable_output_ms": stable_output_ms,
            "confirmed_window_count": len(stable_rows),
            "confirmed_window_end_to_result_ms_p50": percentile(confirmed_end_to_result, 50),
            "confirmed_window_end_to_result_ms_p95": percentile(confirmed_end_to_result, 95),
            "confirmed_result_wall_from_stream_ms_p50": percentile(confirmed_wall, 50),
            "confirmed_result_wall_from_stream_ms_p95": percentile(confirmed_wall, 95),
            "text_change_count": change_count,
            "text_duplicate_count": duplicate_count,
            "withdrawn_confirmed_count": withdrawn_count,
            "empty_count": sum(text is None or not text.strip() for text in texts),
            "omission": None,
            "omission_reason": "No same-domain CSL-Daily reference set was available.",
            "actual_processing_fps": decoded_frames / elapsed_total if elapsed_total > 0 else None,
            "stream_rtf": elapsed_total / max(0.001, decoded_frames / nominal_fps),
            "gpu_peak_allocated_mb": runtime.get("gpu_peak_allocated_mb"),
            "gpu_peak_reserved_mb": runtime.get("gpu_peak_reserved_mb"),
            "p95_sample_count": len(rows),
            "p95_boundary": "Descriptive only for this clip; not a population P95.",
            "claim": "sliding-window over complete model calls; not native streaming",
        },
    }
    output_path = (args.output or ROOT / "artifacts/unisign-benchmark" /
                   f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **output["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





