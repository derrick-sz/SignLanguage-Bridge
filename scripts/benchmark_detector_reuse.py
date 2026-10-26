"""Quantify cross-frame detection reuse before touching production code.

The detector (YOLOX-tiny @416x416) is roughly 60% of the per-frame pose cost.
The obvious optimisation is to re-run detection every K frames and reuse the last
box for the frames in between. That changes the pose input, so it must be
measured before it is adopted:

* how much wall-clock it saves, and
* how far the resulting canonical pose tensors drift from the per-frame
  detection baseline (mean / P95 / max absolute difference, and the share of
  keypoints within 2 px in pixel space).

Only after that evidence exists should a runtime flag be added.

Usage:
    .\\.conda-env\\python.exe scripts\\benchmark_detector_reuse.py --limit 8 --strides 2 3 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.uni_sign import RtmposeExtractor  # noqa: E402
from slbridge.uni_sign_preprocess import (  # noqa: E402
    deterministic_frame_indices,
    normalize_rtmpose_pixels,
    prepare_pose_window,
)

DEFAULT_VIDEO_ROOT = ROOT / "data/CSL-NEWS/archive_001"


def read_frames(capture, limit):
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def run(extractor, frames, width, height, *, stride, max_frames):
    """Return (seconds, pose window, keypoint pixel array, per-frame bbox count)."""
    det = extractor._wholebody.det_model
    pose_model = extractor._wholebody.pose_model
    started = time.perf_counter()
    boxes = None
    points, scores = [], []
    detections = 0
    for index, frame in enumerate(frames):
        if boxes is None or index % stride == 0:
            boxes = det(frame)
            detections += 1
        kp, sc = pose_model(frame, bboxes=boxes)
        points.append(np.asarray(kp, dtype=np.float32))
        scores.append(np.asarray(sc, dtype=np.float32))
    elapsed = time.perf_counter() - started

    points_arr = np.asarray(points, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    if points_arr.ndim == 3:
        points_arr = points_arr[:, None]
    if scores_arr.ndim == 2:
        scores_arr = scores_arr[:, None]
    pixels = points_arr[:, 0]
    normalised = normalize_rtmpose_pixels(pixels, width, height, mode="isotropic_short_side")
    window = prepare_pose_window(normalised, scores_arr[:, 0], max_frames=max_frames)
    return elapsed, window, pixels, detections


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-frame detection reuse benchmark")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=120,
                        help="Only the first N frames are used to keep the benchmark short")
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/detector-reuse-benchmark.json")
    args = parser.parse_args(argv)

    import cv2

    videos = sorted(path for path in args.video_root.iterdir() if path.suffix.lower() == ".mp4")
    videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"No videos under {args.video_root}")

    options = RuntimeOptions(device=args.device)
    extractor = RtmposeExtractor(mode="lightweight", backend="onnxruntime", device=args.device,
                                 cache_dir=ROOT / "models/external/uni_sign/rtmlib-cache",
                                 dll_dirs=options.cuda_dll_dirs,
                                 strict_cuda=args.device == "cuda",
                                 warmup_image=ROOT / "models/external/uni_sign_source/demo/rtmlib-main/demo.jpg")
    extractor.warmup()

    rows = []
    try:
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = read_frames(capture, args.frames_per_clip)
            capture.release()
            if len(frames) < 2:
                continue
            base_seconds, base_pose, base_pixels, base_det = run(
                extractor, frames, width, height, stride=1, max_frames=args.max_frames)
            row = {
                "video": video.name, "frames": len(frames),
                "baseline_seconds": round(base_seconds, 4),
                "baseline_detections": base_det,
                "strides": {},
            }
            for stride in args.strides:
                seconds, window, pixels, detections = run(
                    extractor, frames, width, height, stride=stride, max_frames=args.max_frames)
                diffs = []
                for part in ("body", "left", "right", "face_all"):
                    diffs.append(np.abs(getattr(base_pose, part) - getattr(window, part)))
                flat = np.concatenate([diff.ravel() for diff in diffs])
                pixel_diff = np.abs(base_pixels - pixels)
                row["strides"][str(stride)] = {
                    "seconds": round(seconds, 4),
                    "speedup": round(base_seconds / seconds, 3) if seconds else None,
                    "detections": detections,
                    "canonical_mean_abs_diff": round(float(flat.mean()), 5),
                    "canonical_p95_abs_diff": round(float(np.percentile(flat, 95)), 5),
                    "canonical_max_abs_diff": round(float(flat.max()), 5),
                    "keypoints_within_2px_fraction": round(float((pixel_diff <= 2.0).mean()), 4),
                    "keypoints_within_5px_fraction": round(float((pixel_diff <= 5.0).mean()), 4),
                }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        extractor.close()

    summary = {"schema_version": 1, "clips": len(rows),
               "frames_per_clip": args.frames_per_clip,
               "device": args.device, "rows": rows, "aggregate": {}}
    for stride in args.strides:
        key = str(stride)
        values = [row["strides"][key] for row in rows if key in row["strides"]]
        if not values:
            continue
        summary["aggregate"][key] = {
            "mean_speedup": round(statistics.mean(v["speedup"] for v in values), 3),
            "mean_canonical_p95_abs_diff": round(
                statistics.mean(v["canonical_p95_abs_diff"] for v in values), 5),
            "mean_keypoints_within_2px_fraction": round(
                statistics.mean(v["keypoints_within_2px_fraction"] for v in values), 4),
            "mean_detections": round(statistics.mean(v["detections"] for v in values), 1),
        }
    summary["protocol_note"] = ("Divergence is measured against per-frame detection on the same "
                                "frames. A speedup is only usable if the divergence is small AND a "
                                "downstream accuracy check on the same clips agrees.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=1), flush=True)
    print(json.dumps({"event": "written", "path": str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
