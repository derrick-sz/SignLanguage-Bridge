"""Build a deterministic CSL-Daily selection from the locally available videos.

``D:\\Zenodo\\csl-daliy`` holds *uncropped* sentence recordings.  The official
sentence-level annotations (``csl-daily.{train,dev,test}``, gzip-pickled lists)
name each sample and its ``num_frames`` sentence segment; the official keypoint
archive (``csl-daily-keypoints.pkl``) holds the matching ``[T, 133, 3]`` HRNet
poses in a 512-px reference space.

This script intersects the two so that every later evaluation runs on exactly the
samples whose video, annotation and (optionally) keypoints are all present.  It
writes a ``selection.json`` in the same shape the existing 52-sample pilot used:

    {"source": ..., "split": ..., "count": N, "samples": [
        {"sample": "S000020_P0004_T00", "num_frames": 52,
         "reference": "他今年四岁。", "gloss": "他 今天 年龄 4",
         "keypoints_frames": 52}, ...]}

The order is sorted by sample name so shards and reports are reproducible.
Nothing is downloaded or modified; only the selection file is written.

Usage (PowerShell, project root):

    .\\.conda-env\\python.exe scripts\\build_csldaily_video_selection.py `
        --split test `
        --output artifacts\\csldaily-video-test-627-20261026\\selection.json `
        --check-keypoints
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = Path(r"D:\Zenodo\csl-daliy")
DEFAULT_ANNOTATION_DIR = Path(r"F:\model_weigth\data\csl-daily")
DEFAULT_KEYPOINTS = DEFAULT_ANNOTATION_DIR / "csl-daily-keypoints.pkl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotations(path: Path):
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path}: expected a non-empty list of annotations")
    by_name = {}
    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"{path}: annotation without a usable name")
        if name in by_name:
            raise SystemExit(f"{path}: duplicate annotation for {name}")
        for key in ("num_frames", "gloss", "text"):
            if key not in record:
                raise SystemExit(f"{path}: {name} is missing {key!r}")
        by_name[name] = record
    return by_name


def load_keypoint_frames(path: Path, wanted: set[str]):
    with Path(path).open("rb") as handle:
        keypoints = pickle.load(handle)
    if not isinstance(keypoints, dict):
        raise SystemExit(f"{path}: expected a dict of sample -> pose")
    frames = {}
    for name in wanted:
        pose = keypoints.get(name)
        if pose is None:
            frames[name] = None
            continue
        array = np.asarray(pose["keypoints"])
        if array.ndim == 4:  # some releases keep a leading view axis
            array = array[:, 0]
        frames[name] = int(array.shape[0]) if array.ndim == 3 else None
    return frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--annotation", type=Path,
                        help="override the annotation file (default: <dir>/csl-daily.<split>)")
    parser.add_argument("--keypoints-pkl", type=Path, default=None,
                        help="official keypoint archive to cross-check (default: none)")
    parser.add_argument("--check-keypoints", action="store_true",
                        help=f"cross-check {DEFAULT_KEYPOINTS.name}")
    parser.add_argument("--require-keypoints", action="store_true",
                        help="drop samples without a matching pose (default: keep and report)")
    parser.add_argument("--expected-total", type=int, default=None,
                        help="fail unless the split annotation has this many samples")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    annotation_path = args.annotation or (args.annotation_dir / f"csl-daily.{args.split}")
    if not annotation_path.is_file():
        raise SystemExit(f"annotation not found: {annotation_path}")
    if not args.video_dir.is_dir():
        raise SystemExit(f"video directory not found: {args.video_dir}")

    annotations = load_annotations(annotation_path)
    if args.expected_total is not None and len(annotations) != args.expected_total:
        raise SystemExit(f"{annotation_path}: {len(annotations)} annotations, expected {args.expected_total}")

    videos = {path.stem for path in Path(args.video_dir).glob("*.mp4")}
    names = sorted(name for name in annotations if name in videos)
    missing_video = len(annotations) - len(names)

    keypoints_path = args.keypoints_pkl
    if args.check_keypoints and keypoints_path is None:
        keypoints_path = DEFAULT_KEYPOINTS
    frame_table = None
    if keypoints_path is not None:
        if not Path(keypoints_path).is_file():
            raise SystemExit(f"keypoints archive not found: {keypoints_path}")
        frame_table = load_keypoint_frames(keypoints_path, set(names))

    samples = []
    dropped = []
    for name in names:
        record = annotations[name]
        num_frames = int(record["num_frames"])
        entry = {
            "sample": name,
            "num_frames": num_frames,
            "reference": str(record["text"]),
            "gloss": str(record["gloss"]),
        }
        if frame_table is not None:
            frames = frame_table.get(name)
            entry["keypoints_frames"] = frames
            if frames is None or frames != num_frames:
                if args.require_keypoints:
                    dropped.append(name)
                    continue
        samples.append(entry)

    if not samples:
        raise SystemExit("no sample has both a video and an annotation")

    payload = {
        "source": f"{args.video_dir} ∩ csl-daily.{args.split}",
        "split": args.split,
        "count": len(samples),
        "video_dir": str(args.video_dir),
        "annotation": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "annotation_total": len(annotations),
        "missing_video": missing_video,
        "samples": samples,
    }
    if keypoints_path is not None:
        payload["keypoints_pkl"] = str(keypoints_path)
        payload["keypoints_pkl_bytes"] = Path(keypoints_path).stat().st_size
        payload["keypoints_missing"] = [name for name in names if frame_table.get(name) is None]
        payload["keypoints_frame_mismatch"] = [
            {"sample": name, "annotation": int(annotations[name]["num_frames"]),
             "keypoints": frame_table.get(name)}
            for name in names
            if frame_table.get(name) is not None
            and frame_table.get(name) != int(annotations[name]["num_frames"])
        ]
        payload["keypoints_dropped"] = sorted(dropped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    frames = [int(entry["num_frames"]) for entry in samples]
    stats = {
        "output": str(args.output.resolve()),
        "split": args.split,
        "annotations": len(annotations),
        "videos": len(videos),
        "selected": len(samples),
        "missing_video": missing_video,
        "frames_min": min(frames),
        "frames_mean": round(statistics.fmean(frames), 2),
        "frames_median": statistics.median(frames),
        "frames_max": max(frames),
        "gloss_tokens_mean": round(statistics.fmean(len(entry["gloss"].split()) for entry in samples), 2),
        "reference_chars_mean": round(statistics.fmean(len(entry["reference"]) for entry in samples), 2),
    }
    if frame_table is not None:
        stats["keypoints_missing"] = len(payload["keypoints_missing"])
        stats["keypoints_frame_mismatch"] = len(payload["keypoints_frame_mismatch"])
        stats["keypoints_dropped"] = len(payload["keypoints_dropped"])
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
