"""Run real video extraction and strict readback without a recognition model.

Run from the repository root with the desktop/MediaPipe environment:
    .venv-libras/Scripts/python.exe experiments/validate_keypoint_video.py

Every invocation creates a fresh artifact directory. This is an offline full-frame
acceptance check, not a live-camera benchmark or sign-recognition evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
from itertools import zip_longest
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from slbridge.keypoint_recording import KeypointRecording  # noqa: E402
from slbridge.sources import VideoSource  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _run_cli(arguments: list[str], prefix: Path) -> tuple[dict, float]:
    started = time.perf_counter()
    with prefix.with_suffix(".stdout.json").open("w", encoding="utf-8") as stdout:
        with prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                [sys.executable, "-m", "slbridge.keypoint_cli", *arguments],
                cwd=REPO, stdout=stdout, stderr=stderr, check=False,
            )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(
            f"keypoint CLI exited {result.returncode}; see {prefix.with_suffix('.stderr.log')}"
        )
    return json.loads(prefix.with_suffix(".stdout.json").read_text(encoding="utf-8")), elapsed


def validate(video: Path, output_base: Path) -> dict:
    video = video.resolve(strict=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    artifact_dir = output_base.resolve() / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    output = artifact_dir / "test-00001.slkp.jsonl"
    extraction, extraction_wall = _run_cli(
        ["extract", "--video", str(video), "--output", str(output),
         "--language", "csl", "--label", "acceptance sample; meaning unverified"],
        artifact_dir / "extract",
    )
    inspection, inspection_wall = _run_cli(["inspect", str(output)], artifact_dir / "inspect")
    if inspection != extraction:
        raise AssertionError("extract summary and a separate inspect process disagree")

    recording = KeypointRecording(output)
    durations: list[float] = []
    image_masks = Counter()
    normalized_masks = Counter()
    reset_reasons = Counter()
    frame_sizes = Counter()
    motion_valid_frames = 0
    source_frames = 0
    first_index = last_index = None
    first_time = last_time = None
    with VideoSource(video) as source:
        for source_packet, frame in zip_longest(iter(source.read, None), recording.iter_frames()):
            if source_packet is None or frame is None:
                raise AssertionError("decoded source frame count and recording frame count disagree")
            if (frame.frame_index != source_packet.frame_index
                    or frame.timestamp != source_packet.timestamp
                    or frame.source_id != source_packet.source_id):
                raise AssertionError("recording did not preserve source frame index, timestamp, or identity")
            source_frames += 1
            if first_index is None:
                first_index, first_time = frame.frame_index, frame.timestamp
            last_index, last_time = frame.frame_index, frame.timestamp
            frame_sizes[frame.image_size] += 1
            durations.append(1000 * (frame.extracted_monotonic - frame.received_monotonic))
            motion_valid_frames += int(frame.motion_valid)
            if frame.reset_reason is not None:
                reset_reasons[frame.reset_reason] += 1
            for group, start, stop in (("pose", 0, 33), ("left_hand", 33, 54), ("right_hand", 54, 75)):
                image_masks[group] += sum(frame.point_observed[start:stop])
                normalized_masks[group] += sum(frame.normalized_observed[start:stop])
            for index, observed in enumerate(frame.point_observed):
                if not observed and frame.image_xy[index] != (0.0, 0.0):
                    raise AssertionError("missing image point did not use zero plus a false mask")
            for index, observed in enumerate(frame.normalized_observed):
                if not observed and frame.normalized_xy[index] != (0.0, 0.0):
                    raise AssertionError("missing normalized point did not use zero plus a false mask")
        timestamp_mode = source.timestamp_mode

    if not durations or min(durations) < 0:
        raise AssertionError("extraction durations were missing or negative")
    source_span = last_time - first_time
    summary = recording.summary()
    if summary["metadata"]["annotations"]["review_status"] != "draft":
        raise AssertionError("collection labels must remain draft annotations")
    import cv2

    opencv_distributions = package_metadata.packages_distributions().get("cv2", [])
    evidence = {
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_directory": str(artifact_dir),
        "video": {"path": str(video), "sha256": _sha256(video), "bytes": video.stat().st_size,
                  "decoded_frames": source_frames, "first_index": first_index, "last_index": last_index,
                  "first_timestamp": first_time, "last_timestamp": last_time,
                  "timestamp_span_seconds": source_span, "timestamp_mode": timestamp_mode,
                  "image_sizes": {f"{width}x{height}": count for (width, height), count in frame_sizes.items()}},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "logical_cpu_count": os.cpu_count(),
                        "opencv_version": cv2.__version__,
                        "packages": {name: package_metadata.version(name)
                                     for name in ("mediapipe", "numpy", *opencv_distributions)}},
        "recording": summary,
        "wall_seconds": {"extract_cli": extraction_wall, "inspect_cli": inspection_wall},
        "packet_processing_ms": {"sum": sum(durations), "mean": sum(durations) / len(durations),
                                 "minimum": min(durations), "p50": _percentile(durations, .5),
                                 "p95": _percentile(durations, .95), "maximum": max(durations)},
        "mask_points": {"image_observed": dict(image_masks), "normalized_observed": dict(normalized_masks),
                        "total_possible": source_frames * 75},
        "reset_reasons": dict(reset_reasons),
        "motion_valid_frames": motion_valid_frames,
        "checks": {"native_mediapipe": True, "separate_extract_and_inspect_processes": True,
                   "summaries_equal": True, "all_decoded_frames_preserved": True,
                   "source_indices_and_timestamps_equal": True, "strict_reader_completed": True,
                   "missing_points_zero_with_false_masks": True,
                   "annotations_are_draft": summary["metadata"]["annotations"]["review_status"] == "draft"},
        "boundaries": ["offline full-frame video extraction, not live-camera throughput",
                       "no sign-recognition model, classification, translation, or trained temporal algorithm",
                       "one local video; hand detections and masks are not ground-truth accuracy",
                       "packet processing includes MediaPipe and packet conversion; CLI wall time includes startup and file I/O"],
    }
    result_path = artifact_dir / "acceptance.json"
    result_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"status": "passed", "result_path": str(result_path), "frames": source_frames,
            "extract_wall_seconds": extraction_wall, "inspect_wall_seconds": inspection_wall,
            "packet_mean_ms": evidence["packet_processing_ms"]["mean"],
            "packet_p95_ms": evidence["packet_processing_ms"]["p95"],
            "quality": summary["quality"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path,
                        default=Path("D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4"))
    parser.add_argument("--output-dir", type=Path, default=REPO / "artifacts/keypoint-video-cli-20261011")
    args = parser.parse_args()
    try:
        result = validate(args.video, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
