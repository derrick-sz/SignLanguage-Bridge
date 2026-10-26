"""Measure the cost of extracting poses on every frame vs only the sampled frames.

Context: ``UniSignBackend._video_pose_window`` decodes every frame and runs
RTMPose on all of them, then ``prepare_pose_window(max_frames=256)`` throws away
all but 256. For the local CSL-News clips 39% are longer than 256 frames, so
30-50% of the pose work on those clips is discarded.

This script does not change the runtime. It measures the same pipeline two ways
on real clips and verifies that the resulting canonical pose tensors are
identical, which is the precondition for adopting the optimisation:

* ``baseline``  - decode all frames, extract on all, then subsample to max_frames
                  (what the runtime does today);
* ``sampled``   - decide the deterministic sample indices up front, ``grab()``
                  the skipped frames without decoding them to BGR, extract only
                  the sampled frames, then build the window from exactly those.

Usage:
    .\\.conda-env\\python.exe scripts\\benchmark_pose_sampling.py --limit 12
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
    PoseWindow,
)

DEFAULT_VIDEO_ROOT = ROOT / "data/CSL-NEWS/archive_001"


def collect_frames(capture, max_video_frames):
    import cv2

    frames = []
    index = 0
    while index < max_video_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        index += 1
    return frames


def extract_all(extractor, frames):
    points, scores = [], []
    for frame in frames:
        pts, sc = extractor.extract(frame)
        points.append(pts)
        scores.append(sc)
    return points, scores


def extract_sampled(extractor, path, indices, max_video_frames):
    """grab() the skipped frames, retrieve() only the sampled ones."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    wanted = set(indices)
    points, scores = [], []
    index = 0
    try:
        while index < max_video_frames:
            ok = capture.grab()
            if not ok:
                break
            if index in wanted:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                pts, sc = extractor.extract(frame)
                points.append(pts)
                scores.append(sc)
            index += 1
    finally:
        capture.release()
    if len(points) != len(indices):
        raise ValueError(f"sampled {len(points)} frames, expected {len(indices)}")
    return points, scores


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pose-sampling cost benchmark")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-video-frames", type=int, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/pose-sampling-benchmark.json")
    args = parser.parse_args(argv)

    videos = sorted(path for path in args.video_root.iterdir() if path.suffix.lower() == ".mp4")
    videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"No videos under {args.video_root}")

    import cv2

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
            fps = float(capture.get(cv2.CAP_PROP_FPS))

            started = time.perf_counter()
            frames = collect_frames(capture, args.max_video_frames)
            capture.release()
            decode_only = time.perf_counter() - started

            started = time.perf_counter()
            points, scores = extract_all(extractor, frames)
            baseline_seconds = time.perf_counter() - started

            index = np.asarray(points, dtype=np.float32)
            score = np.asarray(scores, dtype=np.float32)
            if index.ndim == 3:
                index = index[:, None, :, :]
            if score.ndim == 2:
                score = score[:, None, :]
            index = index[:, 0]
            score = score[:, 0]
            normalised = normalize_rtmpose_pixels(index, width, height,
                                                  mode="isotropic_short_side")
            baseline_pose = prepare_pose_window(normalised, score, max_frames=args.max_frames)

            sampled_indices = deterministic_frame_indices(len(frames), max_frames=args.max_frames)
            started = time.perf_counter()
            sampled_points, sampled_scores = extract_sampled(
                extractor, video, sampled_indices, args.max_video_frames)
            sampled_seconds = time.perf_counter() - started
            sampled_index = np.asarray(sampled_points, dtype=np.float32)[:, 0]
            sampled_score = np.asarray(sampled_scores, dtype=np.float32)[:, 0]
            sampled_norm = normalize_rtmpose_pixels(sampled_index, width, height,
                                                    mode="isotropic_short_side")
            sampled_pose = prepare_pose_window(sampled_norm, sampled_score,
                                               max_frames=len(sampled_indices))
            # prepare_pose_window records indices into the array it was given; when
            # only the sampled frames are extracted those indices must be remapped
            # back to source frame numbers, otherwise downstream tooling loses the
            # frame mapping while the tensors themselves stay identical.
            sampled_pose = PoseWindow(
                body=sampled_pose.body, left=sampled_pose.left, right=sampled_pose.right,
                face_all=sampled_pose.face_all,
                sampled_indices=tuple(int(value) for value in sampled_indices),
                quality=dict(sampled_pose.quality,
                             sampled_indices=[int(value) for value in sampled_indices]))

            identical = all(
                np.array_equal(getattr(baseline_pose, part), getattr(sampled_pose, part))
                for part in ("body", "left", "right", "face_all")
            )
            rows.append({
                "video": video.name,
                "width": width, "height": height, "fps": fps,
                "frames": len(frames),
                "sampled_frames": len(sampled_indices),
                "discarded_frames": len(frames) - len(sampled_indices),
                "decode_only_seconds": round(decode_only, 4),
                "baseline_seconds": round(baseline_seconds, 4),
                "sampled_seconds": round(sampled_seconds, 4),
                "speedup": round(baseline_seconds / sampled_seconds, 3) if sampled_seconds else None,
                "pose_tensors_identical": bool(identical),
                "sampled_indices_match": list(sampled_pose.sampled_indices) == list(baseline_pose.sampled_indices),
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    finally:
        extractor.close()

    baseline_total = sum(row["baseline_seconds"] for row in rows)
    sampled_total = sum(row["sampled_seconds"] for row in rows)
    summary = {
        "schema_version": 1,
        "clips": len(rows),
        "max_frames": args.max_frames,
        "device": args.device,
        "mean_baseline_seconds": round(statistics.mean(row["baseline_seconds"] for row in rows), 4),
        "mean_sampled_seconds": round(statistics.mean(row["sampled_seconds"] for row in rows), 4),
        "total_baseline_seconds": round(baseline_total, 3),
        "total_sampled_seconds": round(sampled_total, 3),
        "overall_speedup": round(baseline_total / sampled_total, 3) if sampled_total else None,
        "all_pose_tensors_identical": all(row["pose_tensors_identical"] for row in rows),
        "clips_over_max_frames": sum(1 for row in rows if row["discarded_frames"] > 0),
        "protocol_note": ("Same extractor, same deterministic sample indices; only the "
                          "discarded frames are skipped. Identical tensors mean the "
                          "optimisation is behaviour-preserving for file inference."),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "clips", "mean_baseline_seconds", "mean_sampled_seconds", "overall_speedup",
        "all_pose_tensors_identical", "clips_over_max_frames")}, ensure_ascii=False, indent=1),
        flush=True)
    print(json.dumps({"event": "written", "path": str(args.output)}, ensure_ascii=False), flush=True)
    return 0 if summary["all_pose_tensors_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

