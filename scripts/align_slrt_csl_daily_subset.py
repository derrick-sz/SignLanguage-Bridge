"""Align the 52-sample CSL-Daily test subset to official HRNet keypoints for SLRT.

The videos in ``D:\\Zenodo\\csl-daliy`` are the raw, *uncropped* recordings
(1280/1080 square, 30/29.97 fps), while the annotations describe a sentence clip
(``num_frames``).  The official ``sentence_frames-512x512.zip`` is not available,
so this script derives an approximate sentence segment per video:

1. RTMPose over every source frame (63 SLRT points out of COCO-WholeBody133),
2. sliding a length-T window over the video and scoring the mean Euclidean
   distance to the official HRNet keypoints of the same sample (512-pixel
   reference space),
3. cropping ``[best_start, best_start + T)`` and writing a local HRNet NPZ so
   the existing SLRT CLI can consume it as ``--pose-backend hrnet-npz``.

This is a *local approximation*, not the official sentence-frame release and not
a reproduction of the paper's protocol.  No source, weight or data file is
modified; by default every output goes to ``artifacts/slrt-52-aligned-20261025``.

The selection, annotation file and output directory can be overridden, which is
what the larger local test sweep uses (``--selection``, ``--artifact-dir``,
``--annotation``).  ``--shard-index/--shard-count`` splits the pending list across
independent processes; each shard still writes into the same artifact tree, so the
cache checks keep the run resumable across restarts.

Stages are resumable and independent:

    python scripts/align_slrt_csl_daily_subset.py --stage align
    python scripts/align_slrt_csl_daily_subset.py --stage infer
    python scripts/align_slrt_csl_daily_subset.py --stage report
    python scripts/align_slrt_csl_daily_subset.py --stage selfcheck
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import pickle
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "slrt-52-aligned-20261025"
SELECTION = ROOT / "artifacts" / "unisign-csldaily-52-test-20261025" / "selection.json"
VIDEO_DIR = Path(r"D:\Zenodo\csl-daliy")
KEYPOINTS_PKL = Path(r"F:\model_weigth\data\csl-daily\csl-daily-keypoints.pkl")
ANNOTATION_PKL = Path(r"F:\model_weigth\data\csl-daily\csl-daily.test")
LOW_CONFIDENCE_PX = 25.0
COORDINATE_SYSTEM = "coco_wholebody_133_xyc_512_v1"
FRONTEND = "hrnet_w48_dark"
SOURCE_NOTE = "official csl-daily-keypoints.pkl + RTMPose alignment"
MIN_VALID_FRACTION = 0.5


# --------------------------------------------------------------------------- io
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------- inputs
def load_samples():
    """Selection order, annotations and official keypoints, cross-checked."""
    selection = read_json(SELECTION)
    names = [item["sample"] for item in selection["samples"]]
    declared = int(selection.get("count", len(names)))
    if not names or len(set(names)) != len(names) or declared != len(names):
        raise SystemExit(f"expected {declared} unique samples, got {len(names)}")
    with gzip.open(ANNOTATION_PKL, "rb") as handle:
        annotations = {item["name"]: item for item in pickle.load(handle)}
    with KEYPOINTS_PKL.open("rb") as handle:
        keypoints = pickle.load(handle)
    samples = []
    for item in selection["samples"]:
        name = item["sample"]
        if name not in annotations or name not in keypoints:
            raise SystemExit(f"{name}: missing annotation or keypoints")
        annotation = annotations[name]
        frames = int(annotation["num_frames"])
        kp = np.asarray(keypoints[name]["keypoints"])
        if kp.ndim != 3 or kp.shape[1:] != (133, 3) or kp.shape[0] != frames:
            raise SystemExit(f"{name}: keypoints {kp.shape} do not match num_frames={frames}")
        if item.get("num_frames") != frames or item.get("gloss") != annotation["gloss"]:
            raise SystemExit(f"{name}: selection.json disagrees with csl-daily.test")
        video = VIDEO_DIR / f"{name}.mp4"
        if not video.is_file():
            raise SystemExit(f"{name}: missing source video {video}")
        samples.append(dict(name=name, frames=frames, gloss=annotation["gloss"].split(),
                            text=annotation["text"], keypoints=kp, video=video))
    return samples


def repair_nonfinite(points):
    """Replace NaN/inf with interpolated (nearest at the ends) or zero values.

    Returns ``(repaired float32 array, detail list, point count, value count)``.
    Nothing is silently dropped: every repaired (frame, point) is reported.
    """
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (133, 3):
        raise ValueError(f"expected T x 133 x 3, got {x.shape}")
    flat = x.reshape(x.shape[0], -1)
    bad = ~np.isfinite(flat)
    if not bad.any():
        return x, [], 0, 0
    frames, columns = np.nonzero(bad)
    grouped = {}
    for frame, column in zip(frames.tolist(), columns.tolist()):
        point, channel = divmod(column, 3)
        grouped.setdefault((frame, point), []).append(channel)
    detail = [dict(frame=f, point=p, channels=sorted(c)) for (f, p), c in sorted(grouped.items())]
    filled = flat.copy()
    index = np.arange(flat.shape[0])
    for column in np.unique(columns).tolist():
        series = flat[:, column]
        good = np.isfinite(series)
        filled[:, column] = np.interp(index, index[good], series[good]) if good.any() else 0.0
    return filled.reshape(x.shape), detail, len(detail), int(bad.sum())


def prepare_official(keypoints):
    """Repair non-finite values and clip confidence into [0,1] for the local NPZ."""
    repaired, detail, point_count, value_count = repair_nonfinite(keypoints)
    confidence = repaired[..., 2]
    clipped = int(((confidence > 1) | (confidence < 0)).sum())
    repaired[..., 2] = np.clip(confidence, 0.0, 1.0)
    stats = dict(nonfinite_points=point_count, nonfinite_values=value_count, nonfinite_detail=detail,
                 confidence_clipped=clipped,
                 confidence_raw_min=float(np.nanmin(np.asarray(keypoints, dtype=np.float32)[..., 2])),
                 confidence_raw_max=float(np.nanmax(np.asarray(keypoints, dtype=np.float32)[..., 2])))
    return repaired, stats


# --------------------------------------------------------------- RTMPose pass
def rtmpose_cache_path(name: Path) -> Path:
    return ARTIFACT / "rtmpose" / f"{name}.npz"


def extract_rtmpose(sample, frontend, *, force=False):
    """Full-frame RTMPose extraction, cached; returns 63-point x/y/conf + validity."""
    import cv2

    from slbridge.slrt.preprocess import KEYPOINT_INDICES

    cache = rtmpose_cache_path(sample["name"])
    if cache.is_file() and not force:
        with np.load(cache, allow_pickle=False) as data:
            return (data["keypoints"].copy(), data["valid"].copy(),
                    [int(v) for v in data["missing_frames"]], int(data["source_frames"]))
    capture = cv2.VideoCapture(str(sample["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {sample['video']}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    points, valid, missing = [], [], []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            kp = frontend.extract(bgr, frame_index=index, timestamp=index / fps if fps > 0 else 0.0)
            if kp is None:
                missing.append(index)
                points.append(np.zeros((63, 3), dtype=np.float32))
                valid.append(False)
            else:
                points.append(np.asarray(kp, dtype=np.float32)[list(KEYPOINT_INDICES)])
                valid.append(True)
            index += 1
    finally:
        capture.release()
    if not points:
        raise RuntimeError(f"no frames decoded from {sample['video']}")
    arr = np.stack(points).astype(np.float32)
    val = np.asarray(valid, dtype=bool)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, keypoints=arr, valid=val, missing_frames=np.asarray(missing, dtype=np.int64),
                        source_frames=np.int64(index), fps=np.float64(fps), video_sha256=sha256(sample["video"]))
    return arr, val, missing, index


# ---------------------------------------------------------------- alignment
# Official HRNet output contains degenerate hand predictions (e.g. x = -30320,
# confidence 0.0-0.1) for hands it failed to see.  The official heatmap path
# crops those peaks away and is insensitive to them, but a raw 63-point mean
# distance is dominated by them (43-56 px on frames whose hands match to 5 px).
# Alignment therefore scores only points that both sides actually localise.
OFFICIAL_MIN_CONFIDENCE = 0.2
RTMPOSE_MIN_CONFIDENCE = 0.1
COORDINATE_MARGIN_PX = 64.0
MIN_VALID_POINT_FRACTION = 0.2
FILTER_RULE = (f"align only: official confidence >= {OFFICIAL_MIN_CONFIDENCE} and |x|,|y| within "
               f"[-{COORDINATE_MARGIN_PX:.0f}, 512+{COORDINATE_MARGIN_PX:.0f}], RTMPose confidence >= "
               f"{RTMPOSE_MIN_CONFIDENCE} and a detected frame; the NPZ keeps every official point unchanged")


def alignment_masks(rtm, official, valid):
    """Reliable (frame, point) pairs plus the counts excluded by each rule."""
    inside = ((official[..., 0] >= -COORDINATE_MARGIN_PX) & (official[..., 0] <= 512 + COORDINATE_MARGIN_PX) &
              (official[..., 1] >= -COORDINATE_MARGIN_PX) & (official[..., 1] <= 512 + COORDINATE_MARGIN_PX))
    confident = official[..., 2] >= OFFICIAL_MIN_CONFIDENCE
    official_ok = inside & confident
    counts = dict(official_points=int(official.shape[0] * official.shape[1]),
                  official_low_confidence=int((~confident).sum()),
                  official_out_of_range=int((~inside).sum()),
                  rtmpose_low_confidence=int((rtm[..., 2] < RTMPOSE_MIN_CONFIDENCE).sum()),
                  rtmpose_missing_frames=int((~valid).sum()))
    return official_ok, counts


def align_window(rtm, official, valid):
    """Length-T sliding window with the smallest mean per-point distance (512 px)."""
    frames, length = rtm.shape[0], official.shape[0]
    if frames < length:
        raise RuntimeError(f"video has {frames} frames, annotation needs {length}")
    starts = frames - length + 1
    rtm_xy = rtm[..., :2].astype(np.float64)
    official_xy = official[..., :2].astype(np.float64)
    official_ok, filter_counts = alignment_masks(rtm, official, valid)
    rtm_ok = (rtm[..., 2] >= RTMPOSE_MIN_CONFIDENCE) & valid[:, None]
    windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(rtm_xy, length, axis=0), -1, 1)
    rtm_ok_windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(rtm_ok.astype(np.float64), length, axis=0), -1, 1)
    min_valid = max(8, int(math.ceil(MIN_VALID_FRACTION * length)))
    min_pairs = max(64, int(math.ceil(MIN_VALID_POINT_FRACTION * length * official.shape[1])))
    scores = np.full(starts, np.inf)
    fractions = np.zeros(starts)
    reliable_fractions = np.zeros(starts)
    raw_scores = np.full(starts, np.inf)
    for begin in range(0, starts, 32):
        end = min(starts, begin + 32)
        block = windows[begin:end]
        distance = np.sqrt(((block - official_xy[None]) ** 2).sum(-1))
        mask = (rtm_ok_windows[begin:end] > 0) & official_ok[None]
        pairs = mask.astype(np.float64).sum(axis=2).sum(axis=1)
        total = (distance * mask).sum(axis=2).sum(axis=1)
        scaled = np.where(pairs > 0, total / np.maximum(pairs, 1), np.inf)
        frame_counts = (rtm_ok_windows[begin:end] > 0).sum(axis=2).sum(axis=1)
        scores[begin:end] = np.where((frame_counts >= min_valid) & (pairs >= min_pairs), scaled, np.inf)
        fractions[begin:end] = frame_counts / (length * official.shape[1])
        reliable_fractions[begin:end] = pairs / (length * official.shape[1])
        raw_scores[begin:end] = np.where(frame_counts >= min_valid, (distance * (rtm_ok_windows[begin:end] > 0)).sum(axis=2).sum(axis=1)
                                         / np.maximum((rtm_ok_windows[begin:end] > 0).sum(axis=2).sum(axis=1), 1), np.inf)
    if not np.isfinite(scores).any():
        raise RuntimeError("no window has enough reliable point correspondences")
    best = int(np.argmin(scores))
    center = (frames - length) // 2
    return dict(best_start=best, best_mean_px=float(scores[best]),
                first_mean_px=float(scores[0]) if np.isfinite(scores[0]) else None,
                center_mean_px=float(scores[center]) if np.isfinite(scores[center]) else None,
                all_points_best_mean_px=float(raw_scores[best]) if np.isfinite(raw_scores[best]) else None,
                best_valid_fraction=float(reliable_fractions[best]), candidate_starts=int(starts),
                min_valid_frames=min_valid, min_reliable_pairs=int(min_pairs), length=length,
                reliable_points_per_frame=int(round(official_ok.sum() / official.shape[0])),
                reliable_point_fraction=float(official_ok.mean()),
                filter_counts=filter_counts, filter_rule=FILTER_RULE,
                low_confidence=bool(scores[best] > LOW_CONFIDENCE_PX))


# --------------------------------------------------------------- video trim
def trim_video(sample, best_start, fps_source):
    import cv2

    out = ARTIFACT / "trimmed" / f"{sample['name']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    length = sample["frames"]
    capture = cv2.VideoCapture(str(sample["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {sample['video']}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open VideoWriter for {tmp}")
    written = 0
    first_frame = last_frame = None
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= best_start and written < length:
                if written == 0:
                    first_frame = frame.copy()
                writer.write(frame)
                last_frame = frame
                written += 1
            index += 1
        source_frames = index
    finally:
        capture.release()
        writer.release()
    if written != length:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"wrote {written} frames, expected {length}")
    tmp.replace(out)
    info = dict(path=str(out), fps=fps, width=width, height=height, source_fps=fps_source,
                frames=length, source_frames=source_frames, best_start=best_start,
                bytes=out.stat().st_size, sha256=sha256(out))
    # Independent re-open: declared count, exact decode count and decoded timestamps.
    from slbridge.slrt.backend import video_packets

    probe = cv2.VideoCapture(str(out))
    if not probe.isOpened():
        raise RuntimeError(f"trimmed video cannot be reopened: {out}")
    declared = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if declared != length:
        raise RuntimeError(f"declared frame count {declared} != {length} for {out}")
    metadata = {}
    timestamps = [packet.timestamp for packet in video_packets(out, metadata=metadata)]
    if len(timestamps) != length:
        raise RuntimeError(f"decoded {len(timestamps)} != {length} frames for {out}")
    decoded_fps = float(metadata["original_fps"])
    # The NPZ timeline is built from the fps the decoder reports for the trimmed
    # file, because that is what the SLRT backend uses for its POS_MSEC fallback.
    ideal = np.arange(length, dtype=np.float64) / decoded_fps
    info.update(decoded_frames=int(metadata["decoded_frames"]), decoded_fps=decoded_fps,
                declared_frame_count=int(metadata["declared_frame_count"]),
                timestamp_fallback_frames=int(metadata["timestamp_fallback_frames"]),
                timestamp_max_deviation=float(np.max(np.abs(np.asarray(timestamps) - ideal))))
    info["timestamps"] = np.asarray(timestamps, dtype=np.float64)
    info["timestamps_from_decoded_packets"] = bool(info["timestamp_max_deviation"] > 1e-3)
    return info


# -------------------------------------------------------------------- NPZ
def write_npz(sample, official, trim, alignment):
    length = sample["frames"]
    timestamps = (np.arange(length, dtype=np.float64) / trim["decoded_fps"]
                  if not trim["timestamps_from_decoded_packets"] else trim["timestamps"])
    metadata = {
        "schema_version": 1,
        "name": sample["name"],
        "video_path": trim["path"],
        "video_sha256": trim["sha256"],
        "frontend": FRONTEND,
        "coordinate_system": COORDINATE_SYSTEM,
        "source": SOURCE_NOTE,
        "source_keypoints": str(KEYPOINTS_PKL),
        "source_annotation": str(ANNOTATION_PKL),
        "source_video": str(sample["video"]),
        "source_video_frames": trim["source_frames"],
        "frames": int(length),
        "fps": float(trim["decoded_fps"]),
        "source_fps": float(trim["source_fps"]),
        "confidence_clipped": int(alignment["confidence_clipped"]),
        "nonfinite_points": int(alignment["nonfinite_points"]),
        "nonfinite_values": int(alignment["nonfinite_values"]),
        "nonfinite_detail": alignment["nonfinite_detail"],
        "confidence_raw_min": alignment["confidence_raw_min"],
        "confidence_raw_max": alignment["confidence_raw_max"],
        "alignment": {key: alignment[key] for key in
                      ("best_start", "best_mean_px", "first_mean_px", "center_mean_px",
                       "all_points_best_mean_px", "best_valid_fraction", "reliable_points_per_frame",
                       "reliable_point_fraction", "low_confidence", "filter_rule")},
        "alignment_filter_counts": alignment["filter_counts"],
        "alignment_method": "RTMPose full-frame 63-point sliding window, mean Euclidean distance in 512 px space",
        "timestamps_source": "decoded_pos_msec" if trim["timestamps_from_decoded_packets"] else "frame_index_over_fps",
        "coordinate_reference": "512x512 square resize of the full (uncropped) frame",
        "created_at": utc_now(),
        "created_by": "scripts/align_slrt_csl_daily_subset.py",
    }
    out = ARTIFACT / "pose" / f"{sample['name']}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keypoints=np.asarray(official, dtype=np.float32),
                        frame_indices=np.arange(length, dtype=np.int64),
                        timestamps=np.asarray(timestamps, dtype=np.float64),
                        metadata_json=json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    return out, metadata


# ------------------------------------------------------------------ stages
def stage_align(samples, args):
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.pose import RTMPoseExperimentalFrontend
    from slbridge.slrt.preprocess import KEYPOINT_INDICES, select_keypoints

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    frontend = RTMPoseExperimentalFrontend(SLRTConfig(pose_backend="rtmpose-experimental",
                                                     pose_device=args.device))
    print(json.dumps(dict(frontend=frontend.info.get("frontend"),
                          device=frontend.info.get("device", args.device))), flush=True)
    records = []
    try:
        for position, sample in enumerate(samples, 1):
            name = sample["name"]
            record_path = ARTIFACT / "alignment" / f"{name}.json"
            if record_path.is_file() and not args.force:
                record = read_json(record_path)
                if record.get("status") == "ok" and (ARTIFACT / "trimmed" / f"{name}.mp4").is_file() and \
                        (ARTIFACT / "pose" / f"{name}.npz").is_file():
                    print(f"[{position}/{len(samples)}] {name}: cached", flush=True)
                    records.append(record)
                    continue
            started = time.perf_counter()
            official_full, stats = prepare_official(sample["keypoints"])
            official = official_full[:, list(KEYPOINT_INDICES)]
            select_keypoints(official_full)  # same validation the local NPZ frontend applies
            rtm, valid, missing, source_frames = extract_rtmpose(sample, frontend, force=args.force)
            alignment = align_window(rtm, official, valid)
            alignment.update(stats)
            trim = trim_video(sample, alignment["best_start"], fps_source=frame_fps(sample))
            npz, metadata = write_npz(sample, official_full, trim, alignment)
            record = dict(status="ok", name=name, annotation_frames=int(sample["frames"]),
                          annotation_gloss=sample["gloss"], source_frames=int(source_frames),
                          rtmpose_missing_frames=missing, rtmpose_missing_count=len(missing),
                          **{k: v for k, v in alignment.items() if k not in ("nonfinite_detail",)},
                          nonfinite_detail=stats["nonfinite_detail"],
                          trim={k: v for k, v in trim.items() if k != "timestamps"},
                          npz=dict(path=str(npz), sha256=sha256(npz)),
                          elapsed_seconds=round(time.perf_counter() - started, 3))
            write_json(record_path, record)
            records.append(record)
            print(f"[{position}/{len(samples)}] {name}: start={alignment['best_start']} "
                  f"mean={alignment['best_mean_px']:.2f}px low_conf={alignment['low_confidence']} "
                  f"T={sample['frames']} N={source_frames} missing={len(missing)}", flush=True)
    finally:
        frontend.close()
    ok = sum(1 for record in records if record.get("status") == "ok")
    print(json.dumps(dict(stage="align", ok=ok, total=len(samples))), flush=True)
    return records


def frame_fps(sample):
    import cv2
    capture = cv2.VideoCapture(str(sample["video"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps


def stage_infer(samples, args):
    results = []
    for position, sample in enumerate(samples, 1):
        name = sample["name"]
        trimmed = ARTIFACT / "trimmed" / f"{name}.mp4"
        npz = ARTIFACT / "pose" / f"{name}.npz"
        result_path = ARTIFACT / "slrt" / f"{name}.json"
        events_path = ARTIFACT / "slrt" / f"{name}.events.jsonl"
        windows_path = ARTIFACT / "slrt" / f"{name}.windows.jsonl"
        log_dir = ARTIFACT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        record_path = ARTIFACT / "infer" / f"{name}.json"
        if record_path.is_file() and result_path.is_file() and not args.force:
            record = read_json(record_path)
            if record.get("exit_code") == 0:
                print(f"[{position}/{len(samples)}] {name}: cached", flush=True)
                results.append(record)
                continue
        if not trimmed.is_file() or not npz.is_file():
            record = dict(status="skipped_missing_alignment", name=name, exit_code=None,
                          error=f"missing {trimmed if not trimmed.is_file() else npz}")
        else:
            command = [sys.executable, "-m", "slbridge.slrt.cli", "video",
                       "--video", str(trimmed), "--pose-backend", "hrnet-npz", "--pose-npz", str(npz),
                       "--output", str(result_path), "--events-jsonl", str(events_path),
                       "--windows-jsonl", str(windows_path)]
            started = time.perf_counter()
            try:
                completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True,
                                           encoding="utf-8", errors="replace", timeout=args.timeout)
                exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as error:
                exit_code = None
                stdout = (error.stdout or "") if isinstance(error.stdout, str) else ""
                stderr = f"TimeoutExpired after {args.timeout}s"
            elapsed = time.perf_counter() - started
            (log_dir / f"{name}.stdout.log").write_text(stdout or "", encoding="utf-8")
            (log_dir / f"{name}.stderr.log").write_text(stderr or "", encoding="utf-8")
            parsed = read_json(result_path) if result_path.is_file() else None
            record = dict(name=name, exit_code=exit_code, elapsed_seconds=round(elapsed, 3),
                          result_path=str(result_path) if result_path.is_file() else None,
                          events_path=str(events_path) if events_path.is_file() else None,
                          windows_path=str(windows_path) if windows_path.is_file() else None,
                          status=(parsed or {}).get("status"),
                          gloss_sequence=(parsed or {}).get("gloss_sequence"),
                          parsed=bool(parsed), error=None, stderr_tail=(stderr or "").strip()[-2000:])
            if exit_code != 0:
                record["status"] = "error" if not parsed else parsed.get("status", "error")
                record["error"] = (record["stderr_tail"] or f"exit code {exit_code}")[:2000]
        write_json(record_path, record)
        results.append(record)
        print(f"[{position}/{len(samples)}] {name}: exit={record.get('exit_code')} "
              f"status={record.get('status')} gloss={record.get('gloss_sequence')} "
              f"{record.get('elapsed_seconds')}s", flush=True)
    return results


def wer_rows(samples, alignments, inferences):
    from slbridge.slrt.cli import _wer

    alignment_by_name = {record["name"]: record for record in alignments}
    rows = []
    for sample in samples:
        name = sample["name"]
        alignment = alignment_by_name.get(name, {})
        inference = next((item for item in inferences if item["name"] == name), {})
        predicted = inference.get("gloss_sequence") or []
        error = inference.get("error")
        status = inference.get("status")
        success = inference.get("exit_code") == 0 and status == "final" and bool(inference.get("parsed"))
        row = dict(sample=name, reference=sample["gloss"], predicted=list(predicted),
                   reference_tokens=len(sample["gloss"]), predicted_tokens=len(predicted),
                   frames=sample["frames"], best_start=alignment.get("best_start"),
                   best_mean_px=alignment.get("best_mean_px"),
                   first_mean_px=alignment.get("first_mean_px"),
                   center_mean_px=alignment.get("center_mean_px"),
                   all_points_best_mean_px=alignment.get("all_points_best_mean_px"),
                   reliable_point_fraction=alignment.get("reliable_point_fraction"),
                   filter_counts=alignment.get("filter_counts"),
                   low_confidence=alignment.get("low_confidence"),
                   rtmpose_missing=alignment.get("rtmpose_missing_count"),
                   nonfinite_points=alignment.get("nonfinite_points"),
                   confidence_clipped=alignment.get("confidence_clipped"),
                   exit_code=inference.get("exit_code"), status=status,
                   error=error, success=success, empty=success and not predicted)
        if success:
            metrics = _wer(sample["gloss"], predicted)
            row.update(wer=metrics["wer"], substitutions=metrics["substitutions"],
                       insertions=metrics["insertions"], deletions=metrics["deletions"],
                       edits=metrics["substitutions"] + metrics["insertions"] + metrics["deletions"])
        else:
            row.update(wer=None, substitutions=None, insertions=None, deletions=None, edits=None)
        rows.append(row)
    return rows


def aggregates(rows):
    attempted = len(rows)
    successful = [row for row in rows if row["success"]]
    failed = [row for row in rows if not row["success"]]
    empty = [row for row in successful if row["empty"]]
    nonempty = [row for row in successful if not row["empty"]]
    low = [row for row in rows if row["low_confidence"]]
    ref_tokens = sum(row["reference_tokens"] for row in successful)
    edits = sum(row["edits"] for row in successful)
    nonempty_ref = sum(row["reference_tokens"] for row in nonempty)
    nonempty_edits = sum(row["edits"] for row in nonempty)
    complete = len(failed) == 0
    low_success = [row for row in low if row["success"]]
    distances = [row["best_mean_px"] for row in rows if row["best_mean_px"] is not None]
    raw_distances = [row["all_points_best_mean_px"] for row in rows
                     if row.get("all_points_best_mean_px") is not None]
    filter_totals = {}
    for row in rows:
        for key, value in (row.get("filter_counts") or {}).items():
            filter_totals[key] = filter_totals.get(key, 0) + int(value)
    return dict(
        attempted=attempted,
        aligned=sum(1 for row in rows if row["best_start"] is not None),
        low_confidence=len(low),
        success=len(successful),
        empty=len(empty),
        errors=len(failed),
        missing=attempted - len(successful),
        complete=complete,
        empty_samples=[row["sample"] for row in empty],
        failed_samples=[dict(sample=row["sample"], exit_code=row["exit_code"],
                             status=row["status"], error=row["error"]) for row in failed],
        low_confidence_samples=[dict(sample=row["sample"], best_mean_px=round(row["best_mean_px"], 3),
                                     wer=row["wer"]) for row in low],
        corpus_wer=None if not complete else (edits / ref_tokens if ref_tokens else None),
        corpus_wer_scope=f"all {attempted} samples (only when every sample succeeded; empty predictions count as pure deletion)",
        corpus_wer_complete=complete,
        successful_subset_wer=edits / ref_tokens if ref_tokens else None,
        successful_subset_reference_tokens=ref_tokens,
        successful_subset_edits=edits,
        successful_subset_samples=len(successful),
        nonempty_subset_wer=nonempty_edits / nonempty_ref if nonempty_ref else None,
        nonempty_subset_reference_tokens=nonempty_ref,
        nonempty_subset_edits=nonempty_edits,
        nonempty_subset_samples=len(nonempty),
        missing_samples=attempted - len(successful),
        low_confidence_subset_wer=(sum(row["edits"] for row in low_success) /
                                   sum(row["reference_tokens"] for row in low_success))
        if low_success and sum(row["reference_tokens"] for row in low_success) else None,
        low_confidence_subset_samples=len(low_success),
        total_substitutions=sum(row["substitutions"] for row in successful),
        total_insertions=sum(row["insertions"] for row in successful),
        total_deletions=sum(row["deletions"] for row in successful),
        alignment_px=dict(count=len(distances), min=min(distances) if distances else None,
                          mean=statistics.fmean(distances) if distances else None,
                          median=statistics.median(distances) if distances else None,
                          max=max(distances) if distances else None),
        alignment_first_px_mean=statistics.fmean([row["first_mean_px"] for row in rows
                                                  if row.get("first_mean_px") is not None])
        if any(row.get("first_mean_px") is not None for row in rows) else None,
        alignment_center_px_mean=statistics.fmean([row["center_mean_px"] for row in rows
                                                   if row.get("center_mean_px") is not None])
        if any(row.get("center_mean_px") is not None for row in rows) else None,
        alignment_raw_all_points_px=dict(
            mean=statistics.fmean(raw_distances) if raw_distances else None,
            min=min(raw_distances) if raw_distances else None,
            max=max(raw_distances) if raw_distances else None,
            note="same windows scored with every RTMPose-confident official point (includes degenerate "
                 "official hand predictions); not used for best_start or low-confidence flags"),
        alignment_reliable_point_fraction=statistics.fmean(
            [row["reliable_point_fraction"] for row in rows if row.get("reliable_point_fraction") is not None])
        if any(row.get("reliable_point_fraction") is not None for row in rows) else None,
        alignment_filter_rule=FILTER_RULE,
        alignment_filter_totals=filter_totals,
        low_confidence_threshold_px=LOW_CONFIDENCE_PX,
        rtmpose_missing_frames=sum((row["rtmpose_missing"] or 0) for row in rows),
        nonfinite_points=sum((row["nonfinite_points"] or 0) for row in rows),
        confidence_clipped=sum((row["confidence_clipped"] or 0) for row in rows),
    )


def stage_report(samples, args):
    alignments = []
    for sample in samples:
        path = ARTIFACT / "alignment" / f"{sample['name']}.json"
        if path.is_file():
            record = read_json(path)
            if record.get("status") == "ok":
                alignments.append(record)
    inferences = []
    for sample in samples:
        path = ARTIFACT / "infer" / f"{sample['name']}.json"
        if path.is_file():
            record = read_json(path)
            parsed_path = ARTIFACT / "slrt" / f"{sample['name']}.json"
            if record.get("result_path") is None and parsed_path.is_file():
                record["result_path"] = str(parsed_path)
            if record.get("gloss_sequence") is None and parsed_path.is_file():
                parsed = read_json(parsed_path)
                record.setdefault("status", parsed.get("status"))
                record["gloss_sequence"] = parsed.get("gloss_sequence")
                record["parsed"] = True
            inferences.append(record)
    rows = wer_rows(samples, alignments, inferences)
    summary = dict(
        generated_at=utc_now(),
        scope=f"CSL-Daily {len(samples)}-sample subset of {ANNOTATION_PKL.name}, approximate sentence "
              "alignment; not the official sentence_frames-512x512 release and not a paper-protocol reproduction",
        inputs=dict(selection=str(SELECTION), selection_sha256=sha256(SELECTION),
                    videos=str(VIDEO_DIR), keypoints=str(KEYPOINTS_PKL),
                    keypoints_sha256=sha256(KEYPOINTS_PKL), annotation=str(ANNOTATION_PKL),
                    annotation_sha256=sha256(ANNOTATION_PKL),
                    script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__).resolve())),
        protocol=dict(pose_backend_for_inference="hrnet-npz (official keypoints, RTMPose-aligned crop)",
                      pose_frontend_declared=FRONTEND, coordinate_system=COORDINATE_SYSTEM,
                      alignment="mean Euclidean distance over the 63 SLRT points in 512 px space, "
                                "restricted to points both sides localise (see alignment_filter_rule); "
                                "exhaustive stride-1 length-T window (no coarse-then-refine shortcut)",
                      window=16, stride=1, vote_size=7, require_gloss=False,
                      slrt_cli="python -m slbridge.slrt.cli video --pose-backend hrnet-npz",
                      wer="slbridge.slrt.cli._wer token Levenshtein with S/D/I; no text normalization"),
        **aggregates(rows),
        per_sample=rows,
    )
    write_json(ARTIFACT / "summary.json", summary)
    write_comparison(rows, summary)
    print(json.dumps({key: summary[key] for key in
                      ("attempted", "aligned", "low_confidence", "success", "empty", "errors",
                       "corpus_wer", "successful_subset_wer", "nonempty_subset_wer")}, ensure_ascii=False, indent=2),
          flush=True)
    return summary


def write_comparison(rows, summary):
    columns = ["sample", "frames", "best_start", "best_mean_px", "low_confidence", "status", "wer",
               "substitutions", "insertions", "deletions", "reference_gloss", "predicted_gloss", "error"]
    csv_path = ARTIFACT / "comparison-aligned.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(
                sample=row["sample"], frames=row["frames"], best_start=row["best_start"],
                best_mean_px=None if row["best_mean_px"] is None else round(row["best_mean_px"], 3),
                low_confidence=row["low_confidence"], status=row["status"] if row["success"] else (row["status"] or "error"),
                wer=None if row["wer"] is None else round(row["wer"], 4),
                substitutions=row["substitutions"], insertions=row["insertions"],
                deletions=row["deletions"], reference_gloss=" ".join(row["reference"]),
                predicted_gloss=" ".join(row["predicted"]),
                error=(row["error"] or "").replace("\n", " ")[:300]))
    lines = [f"# SLRT CSL-Daily {len(rows)}-sample subset — approximate alignment pilot", "",
             f"Generated: {summary['generated_at']}", "",
             f"Scope: {summary['scope']}", "",
             "| metric | value |", "|---|---|",
             f"| attempted | {summary['attempted']} |",
             f"| aligned | {summary['aligned']} |",
             f"| low-confidence alignments (> {LOW_CONFIDENCE_PX:.0f} px) | {summary['low_confidence']} |",
             f"| SLRT success | {summary['success']} |",
             f"| empty Gloss output | {summary['empty']} |",
             f"| errors | {summary['errors']} |",
             f"| corpus WER (complete only) | {fmt(summary['corpus_wer'])} |",
             f"| successful-subset WER ({summary['successful_subset_samples']} samples) | "
             f"{fmt(summary['successful_subset_wer'])} |",
             f"| non-empty subset WER ({summary['nonempty_subset_samples']} samples) | "
             f"{fmt(summary['nonempty_subset_wer'])} |",
             f"| alignment distance, filtered (min/mean/median/max) | "
             f"{fmt(summary['alignment_px']['min'])} / {fmt(summary['alignment_px']['mean'])} / "
             f"{fmt(summary['alignment_px']['median'])} / {fmt(summary['alignment_px']['max'])} px |",
             f"| alignment distance, same windows with all points | "
             f"{fmt(summary['alignment_raw_all_points_px']['mean'])} px (degenerate official hand points included) |",
             "", "Protocol: RTMPose full-frame 63-point pose matching against official HRNet keypoints, "
             "exhaustive length-T window in 512 px space, mp4v crop at the original fps, SLRT CLI with "
             "pinned checkpoint, window 16 / stride 1 / vote 7, `--require-gloss` not used.", "",
             f"Alignment filter rule (scoring only; NPZ keeps every official point): {FILTER_RULE}", "",
             "| sample | ref gloss | predicted gloss | WER | S | D | I | best_start | frames | best_mean_px | low_conf | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| {sample} | {ref} | {pred} | {wer} | {s} | {d} | {i} | {start} | {frames} | {px} | {low} | {status} |".format(
            sample=row["sample"], ref=" ".join(row["reference"]), pred=" ".join(row["predicted"]) or "*(empty)*",
            wer=fmt(row["wer"]), s=dash(row["substitutions"]), d=dash(row["deletions"]),
            i=dash(row["insertions"]), start=dash(row["best_start"]), frames=row["frames"],
            px=dash(None if row["best_mean_px"] is None else round(row["best_mean_px"], 2)),
            low=row["low_confidence"],
            status=(row["status"] if row["success"] else (row["status"] or "error")) +
                   (f": {str(row['error']).replace('|', '/').replace(chr(10), ' ')[:120]}"
                    if row["error"] and not row["success"] else "")))
    lines += ["", "## Failed samples", ""]
    if summary["failed_samples"]:
        lines += [f"- `{item['sample']}`: exit={item['exit_code']} status={item['status']} error={item['error']}"
                  for item in summary["failed_samples"]]
    else:
        lines.append("- none")
    lines += ["", "## Empty Gloss outputs", "",
              ", ".join(f"`{name}`" for name in summary["empty_samples"]) or "- none", "",
              "## Low-confidence alignments", ""]
    if summary["low_confidence_samples"]:
        lines += [f"- `{item['sample']}`: {item['best_mean_px']} px, WER {fmt(item['wer'])}"
                  for item in summary["low_confidence_samples"]]
    else:
        lines.append("- none")
    lines += ["", "WER is token-level (space-separated Gloss) with no text normalization. "
                  f"`corpus WER` is reported only when all {len(rows)} samples produced a final result; "
                  "the successful-subset number covers fewer samples and is not a complete-corpus WER."]
    (ARTIFACT / "comparison-aligned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    return "null" if value is None else f"{value:.4f}"


def dash(value):
    return "—" if value is None else value


def stage_selfcheck(samples, args):
    import cv2

    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import video_packets
    from slbridge.slrt.pose import HRNetNPZFrontend

    rng = random.Random(20261025)
    chosen = rng.sample(samples, min(2, len(samples)))
    checks = []
    for sample in chosen:
        name = sample["name"]
        trimmed = ARTIFACT / "trimmed" / f"{name}.mp4"
        npz = ARTIFACT / "pose" / f"{name}.npz"
        result_path = ARTIFACT / "slrt" / f"{name}.json"
        entry = dict(sample=name, expected_frames=int(sample["frames"]))
        capture = cv2.VideoCapture(str(trimmed))
        entry["mp4_opened"] = bool(capture.isOpened())
        entry["mp4_declared_frames"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else None
        entry["mp4_fps"] = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else None
        decoded = 0
        if capture.isOpened():
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
        capture.release()
        entry["mp4_decoded_frames"] = decoded
        entry["mp4_frames_match"] = decoded == entry["expected_frames"] == entry["mp4_declared_frames"]
        # The crop must be the real source segment, not a re-timed or padded copy.
        alignment_record = read_json(ARTIFACT / "alignment" / f"{name}.json")
        best_start = int(alignment_record["best_start"])
        source = cv2.VideoCapture(str(VIDEO_DIR / f"{name}.mp4"))
        wanted = {best_start, best_start + entry["expected_frames"] - 1}
        source_frames = {}
        index = 0
        while source.isOpened():
            ok, frame = source.read()
            if not ok:
                break
            if index in wanted:
                source_frames[index] = frame
            index += 1
        source.release()
        trimmed_frames = []
        probe = cv2.VideoCapture(str(trimmed))
        while probe.isOpened():
            ok, frame = probe.read()
            if not ok:
                break
            trimmed_frames.append(frame)
        probe.release()
        if best_start in source_frames and len(trimmed_frames) == entry["expected_frames"]:
            entry["source_frame_check"] = dict(
                best_start=best_start,
                first_mean_absdiff=float(np.abs(trimmed_frames[0].astype(np.int16) -
                                                source_frames[best_start].astype(np.int16)).mean()),
                last_mean_absdiff=float(np.abs(trimmed_frames[-1].astype(np.int16) -
                                               source_frames[best_start + entry["expected_frames"] - 1]
                                               .astype(np.int16)).mean()))
        else:
            entry["source_frame_check"] = None
        metadata = {}
        packets = list(video_packets(trimmed, metadata=metadata))
        entry["packets_frames"] = len(packets)
        entry["packets_timestamp_fallback_frames"] = metadata.get("timestamp_fallback_frames")
        frontend = HRNetNPZFrontend(npz, trimmed)
        try:
            entry["npz_meta_frames"] = frontend.info["metadata"]["frames"]
            entry["npz_keypoints_shape"] = list(frontend.keypoints.shape)
            entry["npz_finite"] = bool(np.isfinite(frontend.keypoints).all())
            entry["npz_confidence_range"] = [float(frontend.keypoints[..., 2].min()),
                                             float(frontend.keypoints[..., 2].max())]
            probes = {}
            for index in (0, len(packets) // 2, len(packets) - 1):
                packet = packets[index]
                kp = frontend.extract(packet.image, frame_index=packet.frame_index, timestamp=packet.timestamp)
                probes[str(index)] = None if kp is None else dict(shape=list(kp.shape),
                                                                 confidence_max=float(kp[:, 2].max()))
            entry["npz_extract_probes"] = probes
        finally:
            frontend.close()
        entry["slrt_result_parsed"] = False
        if result_path.is_file():
            parsed = read_json(result_path)
            entry["slrt_result_parsed"] = True
            entry["slrt_result_keys"] = sorted(parsed)[:12]
            entry["slrt_status"] = parsed.get("status")
            entry["slrt_gloss"] = parsed.get("gloss_sequence")
            entry["slrt_frames_received"] = parsed.get("frames_received")
            entry["slrt_windows_processed"] = parsed.get("windows_processed")
            entry["slrt_pose_backend"] = (parsed.get("pose") or {}).get("frontend")
            entry["slrt_video_sha256"] = (parsed.get("source") or {}).get("video_sha256")
            entry["slrt_video_sha256_matches"] = entry["slrt_video_sha256"] == sha256(trimmed)
        checks.append(entry)
        print(json.dumps(entry, ensure_ascii=False, indent=2), flush=True)
    bulk = bulk_verify(samples)
    report = dict(generated_at=utc_now(),
                  checks=checks, bulk_npz_and_trim=bulk,
                  commands=dict(trim_inspect="cv2.VideoCapture + full decode; slbridge.slrt.backend.video_packets",
                                npz_frontend="slbridge.slrt.pose.HRNetNPZFrontend(npz, video).extract(...)",
                                slrt_result=f"json.loads({ARTIFACT / 'slrt'}/<name>.json)",
                                bulk="np.load metadata/array invariants + trimmed decode count for every sample"))
    write_json(ARTIFACT / "selfcheck.json", report)
    return report


def bulk_verify(samples):
    """Structural invariants for every sample: NPZ contents, SHA binding, decoded frames."""
    import cv2

    mismatches = []
    for sample in samples:
        name = sample["name"]
        trimmed = ARTIFACT / "trimmed" / f"{name}.mp4"
        npz = ARTIFACT / "pose" / f"{name}.npz"
        expected = int(sample["frames"])
        if not trimmed.is_file() or not npz.is_file():
            mismatches.append(dict(sample=name, reason="missing trimmed mp4 or npz"))
            continue
        capture = cv2.VideoCapture(str(trimmed))
        decoded = 0
        while capture.isOpened():
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
        capture.release()
        with np.load(npz, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"].item()))
            kp = data["keypoints"]
            stamps = data["timestamps"]
            indices = data["frame_indices"]
        problems = []
        if decoded != expected:
            problems.append(f"decoded {decoded} != {expected}")
        if kp.shape != (expected, 133, 3) or kp.dtype != np.float32:
            problems.append(f"keypoints {kp.shape}/{kp.dtype}")
        if not np.isfinite(kp).all():
            problems.append("non-finite keypoints")
        if float(kp[..., 2].min()) < 0 or float(kp[..., 2].max()) > 1:
            problems.append("confidence outside [0,1]")
        if not np.array_equal(indices, np.arange(expected)) or indices.dtype.kind not in "iu":
            problems.append("frame_indices")
        if len(stamps) != expected or (np.diff(stamps) <= 0).any():
            problems.append("timestamps not monotonic")
        if meta.get("video_sha256") != sha256(trimmed):
            problems.append("video_sha256 mismatch")
        if meta.get("frames") != expected or meta.get("fps") is None:
            problems.append("metadata frames/fps")
        if meta.get("frontend") != FRONTEND or meta.get("coordinate_system") != COORDINATE_SYSTEM:
            problems.append("metadata frontend/coordinate_system")
        for key in ("best_start", "best_mean_px", "low_confidence"):
            if key not in (meta.get("alignment") or {}):
                problems.append(f"alignment.{key} missing")
        for key in ("frames", "fps", "confidence_clipped", "nonfinite_points"):
            if key not in meta:
                problems.append(f"metadata.{key} missing")
        if problems:
            mismatches.append(dict(sample=name, reason="; ".join(problems)))
    result_paths = sorted((ARTIFACT / "slrt").glob("*.json"))
    out = dict(samples=len(samples), npz_and_trim_ok=len(samples) - len(mismatches),
               mismatches=mismatches, slrt_result_files=len(result_paths))
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("align", "infer", "report", "selfcheck", "all"), default="all")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these sample names")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N samples")
    parser.add_argument("--force", action="store_true", help="recompute existing outputs")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-sample SLRT CLI timeout (s)")
    parser.add_argument("--selection", type=Path, default=None,
                        help="override the selection.json (default keeps the 52-sample pilot)")
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="override the output artifact directory")
    parser.add_argument("--annotation", type=Path, default=None,
                        help="override the official annotation pickle (csl-daily.<split>)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="0-based shard index; samples are taken from their sorted position")
    parser.add_argument("--shard-count", type=int, default=1, help="number of concurrent shards")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not 0 <= args.shard_index < max(1, args.shard_count):
        raise SystemExit("shard-index must be in [0, shard-count)")
    global SELECTION, ARTIFACT, ANNOTATION_PKL
    if args.selection is not None:
        SELECTION = Path(args.selection).resolve()
    if args.artifact_dir is not None:
        ARTIFACT = Path(args.artifact_dir).resolve()
    if args.annotation is not None:
        ANNOTATION_PKL = Path(args.annotation).resolve()
    samples = load_samples()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {sample["name"] for sample in samples}
        if unknown:
            raise SystemExit(f"unknown samples: {sorted(unknown)}")
        samples = [sample for sample in samples if sample["name"] in wanted]
    if args.limit:
        samples = samples[:args.limit]
    if args.shard_count > 1:
        total = len(samples)
        samples = [sample for position, sample in enumerate(samples)
                   if position % args.shard_count == args.shard_index]
        print(json.dumps(dict(shard=args.shard_index, shard_count=args.shard_count,
                              shard_samples=len(samples), total_before_shard=total)), flush=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.stage in ("align", "all"):
        stage_align(samples, args)
    if args.stage in ("infer", "all"):
        stage_infer(samples, args)
    if args.stage in ("report", "all"):
        stage_report(samples, args)
    if args.stage in ("selfcheck", "all"):
        stage_selfcheck(samples, args)
    print(json.dumps(dict(stage=args.stage, samples=len(samples),
                          elapsed_seconds=round(time.perf_counter() - started, 1))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
