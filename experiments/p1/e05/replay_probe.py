"""task-8 (G2 2-C) -- SLRT ``replay`` first-final wall-clock probe.

What this measures
------------------
SLRT's replay mode feeds a real mp4 on its **original timeline** (``video_packets``
+ ``SLRTCaptureSession`` paces frames by source timestamps), then runs the native
online sliding-window state machine behind it.  This probe records the *wall
clock* of the whole path -- capture queue -> window inference -> decoder vote ->
event publication -- and separates ``available_time`` (source timestamp of the
latest required input) from actual **wall delivery**, which the protocol says
must never be conflated.

Per video it reports
  * replay wall vs video duration (RTF), windows/s, dropped frames (queue and stream)
  * first partial wall, first final wall, first *stable* final wall (first final
    after the last retraction), first-stable result field for cross-checking
  * partial caption rewrite rate (how many partial captions were later contradicted
    by the final sequence), retraction count
  * head-of-window -> delivery latency (window inference start -> first event
    published for that window) P50/P95; frame arrival -> worker dequeue wait
  * window inference duration P50/P95 and per-video GPU peak alloc

Negative / stress clips (not accuracy samples)
  * ``black``  -- all-black frames at the same size/FPS (no person): expects the
    quality gate to reject every frame, 0 windows, empty output;
  * ``mirror`` -- horizontally flipped real clip with the **pixel-consistent**
    RTMPose frontend: tests whether mirroring is gated at all;
  * ``concat`` -- two different pilot clips glued with no pause: tests cross-
    sentence behaviour (the stream must not invent a sentence boundary).
  Negatives use ``rtmpose-experimental`` because a mirrored/black clip cannot be
  paired with the official HRNet NPZ (its video SHA256 is bound to the original
  pixels).  Positives use the requested ``hrnet-npz`` backend.

This is *replay at original timeline* evidence, not a live camera acceptance:
decode pacing, camera capture jitter, exposure and operator behaviour are not
covered.  No source file under ``slbridge/`` is modified; wall-time recording is
done by wrapping the existing objects from this script.

Usage::

    .conda-env\\python.exe experiments/p1/e05/replay_probe.py run \\
        --run-dir experiments/p1/e05/replay-probe-<stamp> --samples 10 --negatives 2 --concat
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1.e04 import perf_common as common  # noqa: E402

FAILURE_MARKERS = (
    "out of memory", "cuda error", "memory allocation failure",
    "acceleratorerror", "insufficient memory",
)


def _now() -> float:
    return perf_counter()


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


def _percentiles(values: list[float], qs=(50, 95)) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {f"p{q}": None for q in qs} | {"count": 0}
    ordered = sorted(clean)
    out: dict[str, float | None] = {}
    for q in qs:
        position = (len(ordered) - 1) * (q / 100.0)
        low, high = int(position), min(int(position) + 1, len(ordered) - 1)
        fraction = position - low
        out[f"p{q}"] = round(ordered[low] * (1 - fraction) + ordered[high] * fraction, 3)
    out["count"] = len(ordered)
    return out


# ---------------------------------------------------------------------------
# negative / stress clip builders (only written under the probe run directory)
# ---------------------------------------------------------------------------
def _video_meta(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    meta = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "declared_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if not math.isfinite(meta["fps"]) or meta["fps"] <= 0:
        meta["fps"] = 30.0
    return meta


def _write_frames(path: Path, frames: list[np.ndarray], fps: float) -> int:
    import cv2

    if not frames:
        raise ValueError("no frames to write")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                import cv2 as _cv2

                frame = _cv2.resize(frame, (width, height), interpolation=_cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return len(frames)


def build_black_clip(src: Path, dst: Path) -> dict[str, Any]:
    import cv2

    meta = _video_meta(src)
    capture = cv2.VideoCapture(str(src))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(np.zeros_like(frame))
    finally:
        capture.release()
    count = _write_frames(dst, frames, meta["fps"])
    return {"kind": "black", "source": str(src), "path": str(dst), "frames": count,
            "fps": meta["fps"], "size": [meta["width"], meta["height"]]}


def build_mirror_clip(src: Path, dst: Path) -> dict[str, Any]:
    import cv2

    meta = _video_meta(src)
    capture = cv2.VideoCapture(str(src))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.flip(frame, 1))
    finally:
        capture.release()
    count = _write_frames(dst, frames, meta["fps"])
    return {"kind": "mirror", "source": str(src), "path": str(dst), "frames": count,
            "fps": meta["fps"], "size": [meta["width"], meta["height"]],
            "note": "pixels mirrored consistently; RTMPose frontend extracts pose from the mirrored pixels"}


def build_concat_clip(src_a: Path, src_b: Path, dst: Path) -> dict[str, Any]:
    import cv2

    meta_a, meta_b = _video_meta(src_a), _video_meta(src_b)
    fps = meta_a["fps"] if abs(meta_a["fps"] - meta_b["fps"]) < 0.1 else min(meta_a["fps"], meta_b["fps"])
    frames: list[np.ndarray] = []
    for src in (src_a, src_b):
        capture = cv2.VideoCapture(str(src))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
    count = _write_frames(dst, frames, fps)
    return {"kind": "concat", "sources": [str(src_a), str(src_b)], "path": str(dst),
            "frames": count, "fps": fps,
            "note": "two sentence clips glued with no pause; used to observe cross-sentence behaviour"}


# ---------------------------------------------------------------------------
# replay instrumentation (wraps existing objects; slbridge is not modified)
# ---------------------------------------------------------------------------
class ReplayRecorder:
    def __init__(self) -> None:
        self._patched_runtime_ids: set[int] = set()
        self.reset()

    def reset(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.window_sink_rows: list[dict[str, Any]] = []
        self.predicts: list[dict[str, Any]] = []
        self.feeds: list[dict[str, Any]] = []
        self.arrivals: list[dict[str, Any]] = []
        self.stream = None

    def event_sink(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event, wall_recv=_now()))

    def window_sink(self, row: dict[str, Any]) -> None:
        self.window_sink_rows.append(dict(row, wall_recv=_now()))

    def patch_backend(self, backend: Any) -> None:
        """Wrap backend.new_stream so the created stream/runtime are timed."""
        original_new_stream = backend.new_stream
        state: dict[str, Any] = {"stream": None}

        def new_stream(*args: Any, **kwargs: Any):
            stream = original_new_stream(*args, **kwargs)
            state["stream"] = stream
            runtime = stream.runtime
            original_predict = runtime.predict

            def predict(bgr, keypoints):
                # The runtime is shared across sessions, so resolve the *current*
                # stream from the holder instead of capturing one session's stream.
                current = state["stream"]
                anchor = int(getattr(current, "_next_anchor", -1)) if current is not None else -1
                frames = getattr(current, "_frames", None) if current is not None else None
                last_packet = getattr(current, "_last_packet", None) if current is not None else None
                available = (frames[-1].packet.timestamp if frames
                             else (last_packet.timestamp if last_packet is not None else None))
                started = _now()
                probabilities, timings = original_predict(bgr, keypoints)
                finished = _now()
                self.predicts.append({
                    "anchor_frame": anchor,
                    "available_time": available,
                    "wall_start": started,
                    "wall_end": finished,
                    "model_ms": {key: float(value) for key, value in timings.items()},
                })
                return probabilities, timings

            # The runtime object is shared by every session of one backend; wrap
            # it only once, otherwise wrappers nest and duplicate every window.
            if id(runtime) not in self._patched_runtime_ids:
                runtime.predict = predict  # instance-level wrapper; library untouched
                self._patched_runtime_ids.add(id(runtime))

            original_feed = stream.feed

            def feed(packet):
                wall = _now()
                self.feeds.append({"frame_index": int(packet.frame_index),
                                   "timestamp": float(packet.timestamp),
                                   "wall_feed": wall})
                return original_feed(packet)

            stream.feed = feed
            self.stream = stream
            return stream

        backend.new_stream = new_stream

    def patch_session(self, session: Any) -> None:
        frame_queue = session._frames
        original_put = frame_queue.put_nowait

        def put_nowait(item):
            epoch, packet = item
            self.arrivals.append({"frame_index": int(packet.frame_index),
                                  "timestamp": float(packet.timestamp),
                                  "wall_arrive": _now()})
            return original_put(item)

        frame_queue.put_nowait = put_nowait


def _events_in_window(events: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [event for event in events if start - 1e-6 <= event["wall_recv"] <= end + 2e-3]


def _match_events_to_predicts(recorder: ReplayRecorder) -> list[dict[str, Any]]:
    """Attach the first published event of each window inference to that window."""
    rows = []
    used: set[int] = set()
    for index, predict in enumerate(recorder.predicts):
        start, end = predict["wall_start"], predict["wall_end"]
        matched = []
        for event_index, event in enumerate(recorder.events):
            if event_index in used:
                continue
            if start - 1e-6 <= event["wall_recv"] <= end + 2e-3:
                matched.append((event_index, event))
                used.add(event_index)
        first = min(matched, key=lambda pair: pair[1]["wall_recv"])[1] if matched else None
        rows.append({
            "predict_index": index,
            "anchor_frame": predict["anchor_frame"],
            "available_time": _round(predict["available_time"], 4),
            "wall_start_ms": None,  # filled relative to session start later
            "wall_end_ms": None,
            "inference_ms": _round((end - start) * 1000, 3),
            "model_ms": {key: _round(value) for key, value in predict["model_ms"].items()},
            "events_emitted": [{"status": event["status"], "gloss": event.get("gloss"),
                                "wall_recv": event["wall_recv"]} for _, event in matched],
            "first_event_wall": first["wall_recv"] if first else None,
            "head_to_delivery_ms": _round((first["wall_recv"] - start) * 1000, 3) if first else None,
        })
    return rows


def analyse_session(
    *, recorder: ReplayRecorder, session: Any, video: dict[str, Any], mode: str,
    started_at: float, wall_seconds: float, result: dict[str, Any],
    vram: dict[str, Any] | None,
) -> dict[str, Any]:
    events = recorder.events
    predicts = recorder.predicts

    def rel(wall: float | None) -> float | None:
        return _round((wall - started_at) * 1000, 3) if wall is not None else None

    # --- events ------------------------------------------------------------
    partials = [e for e in events if e.get("status") == "partial"]
    finals = [e for e in events if e.get("status") == "final"]
    retracted = [e for e in events if e.get("status") == "retracted"]
    for event in events:
        event["wall_ms"] = rel(event["wall_recv"])
    first_partial_ms = partials[0]["wall_ms"] if partials else None
    first_final_ms = finals[0]["wall_ms"] if finals else None
    last_retraction_ms = retracted[-1]["wall_ms"] if retracted else None
    stable_finals = [e for e in finals
                     if last_retraction_ms is None or e["wall_ms"] > last_retraction_ms]
    first_stable_final_ms = stable_finals[0]["wall_ms"] if stable_finals else None

    # --- partial caption rewrite analysis ---------------------------------
    final_tokens = tuple(result.get("gloss_sequence") or [])
    finals_so_far: list[str] = []
    dead_end = 0
    partial_states = 0
    changes = 0
    previous_caption: tuple[str, ...] | None = None
    for event in events:
        status = event.get("status")
        if status == "final":
            if event.get("gloss"):
                finals_so_far.append(event["gloss"])
        elif status == "partial":
            partial_states += 1
            token = event.get("gloss")
            caption = tuple(finals_so_far) + ((token,) if token else ())
            if previous_caption is not None and caption != previous_caption:
                changes += 1
            if not caption or caption[: len(final_tokens)] != final_tokens[: len(caption)]:
                dead_end += 1
            previous_caption = caption
    # --- timing ------------------------------------------------------------
    predict_rows = _match_events_to_predicts(recorder)
    for row in predict_rows:
        row["wall_start_ms"] = rel(row["wall_start"]) if "wall_start" in row else None
    # (wall_start/end were popped by _round only for inference; recompute explicitly)
    for row, predict in zip(predict_rows, predicts):
        row["window_start_ms"] = rel(predict["wall_start"])
        row["window_end_ms"] = rel(predict["wall_end"])
        row["available_to_start_ms"] = (
            _round(row["window_start_ms"] - (predict["available_time"] or 0) * 1000, 3)
            if row["window_start_ms"] is not None and predict["available_time"] is not None
            else None)
    arrivals = recorder.arrivals
    feeds = recorder.feeds
    arrival_by_index = {item["frame_index"]: item["wall_arrive"] for item in arrivals}
    right_context = int(getattr(getattr(session, "config", None), "window_size", 16)) // 2
    queue_waits = []
    for feed_row in feeds:
        arrival = arrival_by_index.get(feed_row["frame_index"])
        if arrival is not None:
            queue_waits.append((feed_row["wall_feed"] - arrival) * 1000)
    head_to_delivery = [row["head_to_delivery_ms"] for row in predict_rows
                        if row["head_to_delivery_ms"] is not None]
    head_ready_to_delivery = []
    ready_to_inference_start = []
    for row, predict in zip(predict_rows, predicts):
        enabling_index = row["anchor_frame"] + right_context
        enabling_arrival = arrival_by_index.get(enabling_index)
        row["enabling_frame_index"] = enabling_index
        row["enabling_frame_arrival_wall_ms"] = rel(enabling_arrival)
        if enabling_arrival is not None:
            ready_to_inference_start.append((predict["wall_start"] - enabling_arrival) * 1000)
            if row["first_event_wall"] is not None:
                head_ready_to_delivery.append((row["first_event_wall"] - enabling_arrival) * 1000)
            else:
                head_ready_to_delivery.append(None)
        else:
            ready_to_inference_start.append(None)
            head_ready_to_delivery.append(None)
    inference_ms = [row["inference_ms"] for row in predict_rows]
    timestamps = [item["timestamp"] for item in feeds]
    video_duration = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else None
    first_frame_ms = rel(arrivals[0]["wall_arrive"]) if arrivals else None

    window_events = []
    for row, predict in zip(predict_rows, predicts):
        window_events.append({
            "schema_version": 1,
            "video_id": video["video_id"],
            "mode": mode,
            "window_index": row["predict_index"],
            "anchor_frame": row["anchor_frame"],
            "available_time": row["available_time"],
            "enabling_frame_index": row.get("enabling_frame_index"),
            "enabling_frame_arrival_wall_ms": row.get("enabling_frame_arrival_wall_ms"),
            "window_start_wall_ms": rel(predict["wall_start"]),
            "window_end_wall_ms": rel(predict["wall_end"]),
            "ready_to_inference_start_ms": _round(
                (predict["wall_start"] - arrival_by_index[row["enabling_frame_index"]]) * 1000, 3)
            if row.get("enabling_frame_index") in arrival_by_index else None,
            "inference_ms": row["inference_ms"],
            "model_ms": row["model_ms"],
            "events_emitted": [{"status": item["status"], "gloss": item["gloss"],
                                "wall_ms": rel(item["wall_recv"])} for item in row["events_emitted"]],
            "first_event_wall_ms": rel(row["first_event_wall"]),
            "head_to_delivery_ms": row["head_to_delivery_ms"],
            "enabling_frame_to_delivery_ms": _round(
                (row["first_event_wall"] - arrival_by_index[row["enabling_frame_index"]]) * 1000, 3)
            if row.get("enabling_frame_index") in arrival_by_index and row["first_event_wall"] is not None
            else None,
        })

    timings = result.get("timings") or {}
    def block(name: str) -> dict[str, Any]:
        value = timings.get(name) or {}
        return {"total_ms": _round(value.get("total")), "count": value.get("count"),
                "p50_ms": _round(value.get("p50")), "p95_ms": _round(value.get("p95"))}

    metrics = {
        "video_id": video["video_id"],
        "mode": mode,
        "kind": video.get("kind", "positive"),
        "video_path": video["video_path"],
        "pose_npz": video.get("pose_npz"),
        "frontend": video.get("frontend", "hrnet-npz"),
        "source_frames": len(feeds),
        "video_duration_seconds": _round(video_duration, 4),
        "replay_wall_seconds": _round(wall_seconds, 4),
        "rtf_wall_over_video": _round(wall_seconds / video_duration, 4) if video_duration else None,
        "status": result.get("status"),
        "error": result.get("error") or session.error,
        "gloss_sequence": result.get("gloss_sequence") or [],
        "windows_processed": result.get("windows_processed"),
        "windows_per_second": _round((result.get("windows_processed") or 0) / wall_seconds, 4)
        if wall_seconds else None,
        "frames_received": result.get("frames_received"),
        "frames_captured": result.get("frames_captured"),
        "quality_rejected_frames": result.get("quality_rejected_frames"),
        "dropped_frames_stream": result.get("dropped_frames"),
        "queue_dropped": result.get("queue_dropped"),
        "max_queue_depth": result.get("max_queue_depth"),
        "first_partial_wall_ms": first_partial_ms,
        "first_final_wall_ms": first_final_ms,
        "first_stable_final_wall_ms": first_stable_final_ms,
        "first_stable_result_field_ms": result.get("first_stable_wall_latency_ms"),
        "first_frame_arrival_wall_ms": first_frame_ms,
        "first_partial_after_playback_start_ms": _round(first_partial_ms - first_frame_ms, 3)
        if first_partial_ms is not None and first_frame_ms is not None else None,
        "first_stable_after_playback_start_ms": _round(first_stable_final_ms - first_frame_ms, 3)
        if first_stable_final_ms is not None and first_frame_ms is not None else None,
        "playback_start_to_session_start_ms": first_frame_ms,
        "partial_event_count": len(partials),
        "final_event_count": len(finals),
        "retraction_count": len(retracted),
        "retraction_reasons": sorted({str(e.get("reason")) for e in retracted}),
        "partial_caption_states": partial_states,
        "partial_caption_changes": changes,
        "partial_caption_dead_ends": dead_end,
        "partial_rewrite_rate": _round(dead_end / partial_states, 4) if partial_states else None,
        "caption_change_rate": _round(changes / partial_states, 4) if partial_states else None,
        "window_inference_ms": _percentiles(inference_ms),
        "head_to_delivery_ms": _percentiles(head_to_delivery),
        "frame_arrival_to_worker_ms": _percentiles(queue_waits),
        "ready_to_inference_start_ms": _percentiles(
            [value for value in ready_to_inference_start if value is not None]),
        "enabling_frame_to_delivery_ms": _percentiles(
            [value for value in head_ready_to_delivery if value is not None]),
        "model_blocks": {name: block(name) for name in
                         ("pose_ms", "preprocess_ms", "inference_ms", "window_total_ms")},
        "vram": vram or {},
        "events": [{"wall_ms": e.get("wall_ms"), "status": e.get("status"),
                    "gloss": e.get("gloss"), "token_id": e.get("token_id"),
                    "available_time": e.get("available_time"),
                    "evidence_time_range": e.get("evidence_time_range"),
                    "score": e.get("score"), "reason": e.get("reason")}
                   for e in events],
    }
    # stable-final definition cross-check: partial captions that equal the final
    # prefix are "not contradicted"; dead ends are the ones later rewritten.
    if not finals and (result.get("gloss_sequence") or []):
        metrics["note"] = "final events missing although gloss_sequence is non-empty"
    if result.get("error") or (session.error and "release" in str(session.error)):
        metrics["status"] = "error"
    return {"metrics": metrics, "window_events": window_events, "predicts": predict_rows}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_one(*, backend: Any, recorder: ReplayRecorder, video: dict[str, Any], mode: str,
            gpu_sampler: Any) -> dict[str, Any]:
    from slbridge.slrt.capture import SLRTCaptureSession

    recorder.reset()
    session = SLRTCaptureSession(
        backend,
        video=video["video_path"],
        pose_npz=video.get("pose_npz"),
        event_sink=recorder.event_sink,
        window_sink=recorder.window_sink,
    )
    recorder.patch_session(session)
    pool_before = backend.runtime.info.get("gpu_peak_allocated_mb")
    t0 = _now()
    session.start()
    session_started_at = session._started_at  # noqa: SLF001 - read-only wall anchor
    try:
        while session.is_running:
            time.sleep(0.02)
    finally:
        wall = _now() - session_started_at
        result = session.result()
        try:
            session.stop()
        except RuntimeError as error:  # release still in progress
            result.setdefault("errors", []).append(str(error))
    metrics = analyse_session(
        recorder=recorder, session=session, video=video, mode=mode,
        started_at=session_started_at, wall_seconds=wall, result=result,
        vram={"gpu_peak_allocated_mb_after": _round(backend.runtime.info.get("gpu_peak_allocated_mb")),
              "gpu_peak_allocated_mb_before": _round(pool_before),
              "scope": "process torch allocator peak"},
    )
    metrics["metrics"]["session_error"] = session.error
    metrics["metrics"]["capture_source_opened"] = result.get("capture_source_opened")
    metrics["metrics"]["source_release_confirmed"] = result.get("source_release_confirmed")
    metrics["metrics"]["result_source"] = (result.get("source") or {})
    return metrics


def make_manifest(run_dir: Path, *, run_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    manifest = common.base_manifest(run_id, "E05-task8-replay", purpose=(
        "SLRT replay (original timeline) wall-clock first-stable probe: real wall "
        "delivery of native partials, rewrite/retraction rates, queue-to-delivery "
        "latency and replay RTF"))
    manifest.update({
        "model_assets": {
            "slrt": common.model_assets_manifest().get("slrt"),
        },
        "environment": common.environment_fingerprint(),
        "planned": plan,
        "evidence_scope": ("replay at original timeline (video_packets paced by source "
                           "timestamps) -- semi-real, NOT live camera acceptance"),
        "no_source_modification": "slbridge/ untouched; wall times recorded by wrapper in this script",
        "commands": [common.command_line()],
    })
    common.write_json(run_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--samples", type=int, default=10)
    run.add_argument("--negatives", type=int, default=2, help="black + mirror clips")
    run.add_argument("--concat", action="store_true", help="also run one glued two-sentence clip")
    run.add_argument("--max-foreign-vram-mb", type=float, default=4500.0)
    run.add_argument("--allow-shared-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error(f"unknown command {args.command}")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"replay-probe-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"

    baseline = common.nvidia_smi()
    if baseline is None:
        raise SystemExit("nvidia-smi unavailable; replay probe needs the GPU")
    device = baseline["devices"][0]
    if device["memory_used_mb"] > args.max_foreign_vram_mb and not args.allow_shared_gpu:
        raise SystemExit(
            f"GPU busy: {device['memory_used_mb']:.0f} MB used (limit "
            f"{args.max_foreign_vram_mb:.0f} MB); pass --allow-shared-gpu to override")

    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    plan = {
        "samples": [sample["sample_id"] for sample in samples],
        "negatives_requested": args.negatives,
        "concat": bool(args.concat),
        "backend_positive": "hrnet-npz",
        "backend_negative": "rtmpose-experimental",
        "gpu_baseline": baseline,
        "ram": common.memory_snapshot(),
        "exclusive_check": "no other CUDA compute app is expected; desktop baseline recorded",
    }
    manifest = make_manifest(run_dir, run_id=run_id, plan=plan)
    print(json.dumps({"event": "baseline", "gpu": device, "ram": plan["ram"]}, ensure_ascii=False), flush=True)

    from dataclasses import replace

    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend

    positive_config = SLRTConfig.from_file(common.SLRT_CONFIG_HRNET_NPZ)
    negative_config = replace(SLRTConfig.from_file(common.SLRT_CONFIG_RTMPOSE),
                              pose_backend="rtmpose-experimental", pose_device="cuda")
    videos: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # ---- positives: official HRNet NPZ + trimmed mp4 ----------------------
    print("[replay] loading SLRT (hrnet-npz) ...", flush=True)
    positive = SLRTBackend(positive_config)
    positive_info = positive.runtime.info
    recorder = ReplayRecorder()
    recorder.patch_backend(positive)  # once per backend: runtime is reused by all sessions
    positive_time_blocks: list[dict[str, Any]] = []
    # Warm-up replay (first session pays frontend load + first CUDA kernels and
    # would otherwise show up as a "real-time failure" of the first sample).
    if samples:
        warm_video = {
            "video_id": f"warmup-{samples[0]['sample_id']}",
            "kind": "warmup",
            "video_path": samples[0]["trimmed"],
            "pose_npz": samples[0]["npz"],
            "frontend": "hrnet-npz",
        }
        try:
            warm_outcome = run_one(backend=positive, recorder=recorder, video=warm_video,
                                   mode="warmup", gpu_sampler=None)
            warm_metrics = warm_outcome["metrics"]
            videos.append(warm_metrics)
            wall_rows.extend(warm_outcome["window_events"])
            all_events.extend({"video_id": warm_video["video_id"], "mode": "warmup", **event}
                              for event in warm_metrics.pop("events"))
        except BaseException as error:  # noqa: BLE001 - warm-up failure is recorded only
            errors.append({"video_id": warm_video["video_id"],
                           "error": f"{type(error).__name__}: {error}"})
        print(json.dumps({"event": "warmup_done",
                          "wall_s": warm_metrics.get("replay_wall_seconds") if samples else None,
                          "queue_dropped": warm_metrics.get("queue_dropped") if samples else None},
                         ensure_ascii=False), flush=True)
    try:
        for sample in samples:
            video = {
                "video_id": sample["sample_id"],
                "kind": "positive",
                "video_path": sample["trimmed"],
                "pose_npz": sample["npz"],
                "frontend": "hrnet-npz",
                "video_seconds": sample.get("video_seconds"),
            }
            try:
                outcome = run_one(backend=positive, recorder=recorder, video=video,
                                  mode="positive", gpu_sampler=None)
            except BaseException as error:  # noqa: BLE001 - recorded as evidence
                errors.append({"video_id": video["video_id"], "error": f"{type(error).__name__}: {error}"})
                continue
            metrics = outcome["metrics"]
            metrics["video_seconds_from_pilot"] = _round(sample.get("video_seconds"), 4)
            videos.append(metrics)
            wall_rows.extend(outcome["window_events"])
            all_events.extend(
                {"video_id": video["video_id"], "mode": "positive", **event}
                for event in metrics.pop("events"))
            positive_time_blocks.append(metrics)
            print(json.dumps({"event": "replay_done", "video_id": video["video_id"],
                              "status": metrics["status"],
                              "wall_s": metrics["replay_wall_seconds"],
                              "rtf": metrics["rtf_wall_over_video"],
                              "first_stable_final_wall_ms": metrics["first_stable_final_wall_ms"],
                              "retractions": metrics["retraction_count"],
                              "drop_stream": metrics["dropped_frames_stream"],
                              "drop_queue": metrics["queue_dropped"],
                              "gloss": metrics["gloss_sequence"]},
                             ensure_ascii=False), flush=True)
    finally:
        positive.close()

    # ---- negatives: rtmpose frontend (black / mirror / concat) ------------
    if args.negatives or args.concat:
        print("[replay] loading SLRT (rtmpose-experimental) for negative clips ...", flush=True)
        negative_dir = run_dir / "negatives"
        negative_dir.mkdir(parents=True, exist_ok=True)
        negative_videos: list[dict[str, Any]] = []
        source_a = samples[0]["trimmed"]
        source_b = samples[min(1, len(samples) - 1)]["trimmed"]
        if args.negatives >= 1:
            negative_videos.append({**build_black_clip(Path(source_a), negative_dir / "black.mp4"),
                                    "video_id": "negative-black",
                                    "video_path": str(negative_dir / "black.mp4"),
                                    "pose_npz": None, "frontend": "rtmpose-experimental"})
        if args.negatives >= 2:
            negative_videos.append({**build_mirror_clip(Path(source_a), negative_dir / "mirror.mp4"),
                                    "video_id": "negative-mirror",
                                    "video_path": str(negative_dir / "mirror.mp4"),
                                    "pose_npz": None, "frontend": "rtmpose-experimental"})
        if args.concat:
            negative_videos.append({**build_concat_clip(Path(source_a), Path(source_b),
                                                        negative_dir / "concat.mp4"),
                                    "video_id": "stress-concat",
                                    "video_path": str(negative_dir / "concat.mp4"),
                                    "pose_npz": None, "frontend": "rtmpose-experimental"})
        negative = SLRTBackend(negative_config)
        negative_recorder = ReplayRecorder()
        negative_recorder.patch_backend(negative)  # once per backend
        try:
            for video in negative_videos:
                try:
                    outcome = run_one(backend=negative, recorder=negative_recorder,
                                      video=video, mode="negative", gpu_sampler=None)
                except BaseException as error:  # noqa: BLE001
                    errors.append({"video_id": video["video_id"],
                                   "error": f"{type(error).__name__}: {error}"})
                    continue
                metrics = outcome["metrics"]
                metrics["clip"] = {key: video[key] for key in
                                   ("kind", "frames", "fps", "source", "sources", "note", "size")
                                   if key in video}
                videos.append(metrics)
                wall_rows.extend(outcome["window_events"])
                all_events.extend({"video_id": video["video_id"], "mode": "negative", **event}
                                  for event in metrics.pop("events"))
                print(json.dumps({"event": "replay_done", "video_id": video["video_id"],
                                  "status": metrics["status"],
                                  "wall_s": metrics["replay_wall_seconds"],
                                  "windows": metrics["windows_processed"],
                                  "quality_rejected": metrics["quality_rejected_frames"],
                                  "retractions": metrics["retraction_count"],
                                  "gloss": metrics["gloss_sequence"]},
                                 ensure_ascii=False), flush=True)
        finally:
            negative.close()
        manifest["negative_clips"] = negative_videos

    # ---- write artifacts --------------------------------------------------
    common.write_jsonl(run_dir / "window_events.jsonl", wall_rows)
    common.write_jsonl(run_dir / "stream_events.jsonl", all_events)
    positives = [row for row in videos if row["mode"] == "positive"]
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        def values(key: str) -> list[float]:
            return [row[key] for row in rows if row.get(key) is not None]

        return {
            "n": len(rows),
            "successful": sum(1 for row in rows if row.get("status") == "final"
                              and not row.get("error")),
            "failed": sum(1 for row in rows if row.get("error") or row.get("status") == "error"),
            "empty_output": sum(1 for row in rows if not row.get("gloss_sequence")),
            "rtf": _percentiles(values("rtf_wall_over_video")),
            "replay_wall_seconds": _percentiles([row["replay_wall_seconds"] for row in rows]),
            "video_duration_seconds": _percentiles(values("video_duration_seconds")),
            "first_partial_wall_ms": _percentiles(values("first_partial_wall_ms")),
            "first_final_wall_ms": _percentiles(values("first_final_wall_ms")),
            "first_stable_final_wall_ms": _percentiles(values("first_stable_final_wall_ms")),
            "first_stable_after_playback_start_ms": _percentiles(
                values("first_stable_after_playback_start_ms")),
            "windows_per_second": _percentiles(values("windows_per_second")),
            "window_inference_ms": _percentiles(
                [row["window_inference_ms"]["p50"] for row in rows
                 if row["window_inference_ms"].get("p50") is not None]),
            "head_to_delivery_ms_p50": _percentiles(
                [row["head_to_delivery_ms"]["p50"] for row in rows
                 if row["head_to_delivery_ms"].get("p50") is not None]),
            "frame_arrival_to_worker_ms_p50": _percentiles(
                [row["frame_arrival_to_worker_ms"]["p50"] for row in rows
                 if row["frame_arrival_to_worker_ms"].get("p50") is not None]),
            "partial_rewrite_rate": _percentiles(values("partial_rewrite_rate")),
            "retraction_count_total": sum(row.get("retraction_count") or 0 for row in rows),
            "dropped_frames_total": sum(row.get("dropped_frames_stream") or 0 for row in rows),
            "queue_dropped_total": sum(row.get("queue_dropped") or 0 for row in rows),
            "quality_rejected_total": sum(row.get("quality_rejected_frames") or 0 for row in rows),
            "partial_event_total": sum(row.get("partial_event_count") or 0 for row in rows),
            "final_event_total": sum(row.get("final_event_count") or 0 for row in rows),
        }

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "scope": ("SLRT replay at original timeline; semi-real wall-clock evidence, "
                  "NOT live camera acceptance"),
        "gpu_baseline": baseline,
        "environment": manifest["environment"],
        "model": {"slrt_checkpoint_sha256": positive_info.get("checkpoint_sha256"),
                  "vocabulary_sha256": positive_info.get("vocabulary_sha256"),
                  "window_size": positive_config.window_size,
                  "stride": positive_config.stride,
                  "vote_size": positive_config.vote_size,
                  "lookahead_derivation": (
                      "window right context 8 frames + (vote_size//2)*stride 3 frames = "
                      "11 frames of observed lookahead before the centre vote is available")},
        "samples": plan["samples"],
        "videos": videos,
        "aggregate": {"warmup": aggregate([row for row in videos if row["mode"] == "warmup"]),
                      "positive": aggregate(positives),
                      "negative": aggregate([row for row in videos if row["mode"] == "negative"])},
        "errors": errors,
        "artifacts": {"window_events": "window_events.jsonl",
                      "stream_events": "stream_events.jsonl"},
    }
    common.write_json(run_dir / "summary.json", summary)
    manifest.update({
        "run_status": "PASS" if not errors else "PARTIAL",
        "finished_at": common.utc_now(),
        "videos_run": [row["video_id"] for row in videos],
        "errors": errors,
        "output_files": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
    })
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "done", "run_dir": str(run_dir),
                      "positive": summary["aggregate"]["positive"],
                      "negative": summary["aggregate"]["negative"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
