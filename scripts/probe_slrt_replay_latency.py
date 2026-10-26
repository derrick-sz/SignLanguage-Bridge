"""Replay-mode wall-latency probe: feed test videos at original FPS like a live source.

Runs after the batch WER evaluation so the GPU is idle; measures first-final
wall latency, delivery pace and queue drops under real-time pacing. Frames are
queued at 512-square (capture path), so results are latency figures, not a
second accuracy measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter, sleep

from slbridge._io import configure_utf8_streams
from slbridge.slrt.assets import SLRTConfig, sha256
from slbridge.slrt.backend import SLRTBackend
from slbridge.slrt.capture import SLRTCaptureSession

LABEL_DIR = Path(r"D:\Zenodo\CE-CSL\CE-CSL\label")
VIDEO_DIR = Path(r"D:\Zenodo\CE-CSL\CE-CSL\video")


def pick_videos(per_translator=1):
    import csv
    with (LABEL_DIR / "test.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    by_translator = {}
    for row in rows:
        by_translator.setdefault(row[1], []).append(row[0])
    picked = []
    for name in sorted(by_translator):
        for number in sorted(by_translator[name])[:per_translator]:
            video = VIDEO_DIR / "test" / name / f"{number}.mp4"
            if video.is_file():
                picked.append(dict(sample_id=number, translator=name, video=video))
    return picked


def main():
    configure_utf8_streams()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-translator", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    videos = pick_videos(args.per_translator)
    backend = SLRTBackend(SLRTConfig(pose_backend="rtmpose-experimental"))
    records = []
    try:
        for item in videos:
            finals = []
            sink = lambda e: finals.append(dict(status=e["status"], gloss=e.get("gloss"),
                                                wall_elapsed_ms=e.get("wall_elapsed_ms")))
            session = SLRTCaptureSession(backend, video=item["video"], event_sink=sink)
            start = perf_counter()
            session.start()
            while session.is_running:
                sleep(.05)
            session.stop()
            wall = perf_counter() - start
            result = session.result()
            record = dict(sample_id=item["sample_id"], translator=item["translator"],
                          video_sha=sha256(item["video"]),
                          status=result.get("status"), error=result.get("error"),
                          gloss_sequence=result.get("gloss_sequence"),
                          tokens=result.get("tokens", []),
                          frames_captured=result.get("frames_captured"),
                          queue_dropped=result.get("queue_dropped"),
                          max_queue_depth=result.get("max_queue_depth"),
                          first_final_wall_latency_ms=result.get("first_stable_wall_latency_ms"),
                          final_wall_elapsed_ms=[e["wall_elapsed_ms"] for e in finals if e["status"] == "final"],
                          frames_received=result.get("frames_received"),
                          windows_processed=result.get("windows_processed"),
                          dropped_frames=result.get("dropped_frames"),
                          session_wall_seconds=wall)
            records.append(record)
            print(json.dumps({k: record[k] for k in ("sample_id", "translator", "status",
                                                     "first_final_wall_latency_ms", "queue_dropped",
                                                     "gloss_sequence")}, ensure_ascii=False), flush=True)
    finally:
        backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema_version=1, mode="replay_realtime_pacing",
                                           videos=records), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
