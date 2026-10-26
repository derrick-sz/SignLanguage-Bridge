"""Real RecognitionSession smoke test for Uni-Sign video streaming."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slbridge.continuous import ContinuousConfig  # noqa: E402
from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.session import RecognitionSession  # noqa: E402
from slbridge.sources import VideoSource  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real asynchronous RecognitionSession with Uni-Sign")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=12)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-wait", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.window_frames <= 600 or not 1 <= args.stride_frames <= args.window_frames:
        parser.error("invalid window/stride")
    if not 0 < args.fps <= 240 or not 0 < args.max_wait <= 600:
        parser.error("invalid fps/max-wait")
    window_seconds = args.window_frames / args.fps
    config = ContinuousConfig(
        minimum_seconds=window_seconds, window_seconds=8.0,
        minimum_frames=args.window_frames,
        update_interval_seconds=args.stride_frames / args.fps,
        max_result_age_seconds=max(3.0, 2 * window_seconds + 1.0),
        max_observation_gap_seconds=.5,
        max_frames=args.window_frames, max_width=640)
    session = RecognitionSession(options=RuntimeOptions(device=args.device))
    events = []
    started = time.perf_counter()
    error = None
    source = None
    try:
        session.load_model(args.bundle)
        deadline = time.perf_counter() + 90
        while session.busy or session.recognizer is None:
            if time.perf_counter() > deadline:
                raise TimeoutError("model loading did not finish")
            for event in session.poll():
                events.append({"kind": event.kind, "value": repr(event.value)})
            time.sleep(.02)
        source = VideoSource(args.video)
        session.open_source(source, kind='video', video_path=str(args.video))
        deadline = time.perf_counter() + 15
        while not session.source_active:
            if time.perf_counter() > deadline:
                raise TimeoutError("video source did not become active")
            for event in session.poll():
                events.append({"kind": event.kind, "value": repr(event.value)})
            time.sleep(.02)
        session.enable_continuous(config)
        deadline = time.perf_counter() + args.max_wait
        stream_done = False
        while time.perf_counter() < deadline:
            batch = session.poll()
            for event in batch:
                if event.kind == 'continuous_result':
                    update = event.value
                    result = update.response.result
                    events.append({
                        "kind": event.kind,
                        "start_time": update.start_time,
                        "end_time": update.end_time,
                        "age_seconds": update.age_seconds,
                        "text": result.prediction.text if result.prediction is not None else None,
                        "status": result.status,
                        "stability_state": result.quality.get("stability_state"),
                        "pose_cache_hit_frames": result.quality.get("pose_cache_hit_frames"),
                        "pose_new_frames": result.quality.get("pose_new_frames"),
                        "adapter_total_ms": result.quality.get("adapter_total_ms"),
                    })
                elif event.kind in {'continuous_status', 'continuous_clear', 'error', 'eof', 'source_closed'}:
                    events.append({"kind": event.kind, "value": str(event.value)})
                    if event.kind in {'eof', 'error'}:
                        stream_done = True
            if stream_done:
                break
            time.sleep(.02)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.disable_continuous()
        except Exception:
            pass
        session.close()
        wall_seconds = time.perf_counter() - started
    results = [event for event in events if event.get('kind') == 'continuous_result']
    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "video": str(args.video.resolve()),
        "bundle": str(args.bundle.resolve()),
        "config": {"window_frames": args.window_frames, "stride_frames": args.stride_frames,
                   "fps": args.fps, "device": args.device},
        "wall_seconds": wall_seconds,
        "error": error,
        "continuous_result_count": len(results),
        "confirmed_count": sum(item.get('stability_state') == 'confirmed' for item in results),
        "texts": [item.get('text') for item in results],
        "no_stable_output": not any(item.get('stability_state') == 'confirmed' for item in results),
        "events": events,
        "claim": "async capture + one bounded model worker; sliding windows are not native streaming",
    }
    output_path = (args.output or ROOT / "artifacts/unisign-session" /
                   f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "wall_seconds": wall_seconds,
                      "error": error, "continuous_result_count": len(results),
                      "confirmed_count": output["confirmed_count"],
                      "texts": output["texts"][-5:]}, ensure_ascii=False, indent=2))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

