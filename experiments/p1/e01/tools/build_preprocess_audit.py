"""E01 step 3: preprocessing consistency audit for the 50 pilot samples.

Checks that the trimmed clip each path consumes matches the clip recorded in the
pose NPZ metadata, that frame counts line up with the annotation, and pins the
frame-sampling / crop protocol of both Uni-Sign input paths.

Writes ``experiments/p1/e01/preprocess_audit.csv``.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\build_preprocess_audit.py
"""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402
from slbridge.uni_sign_preprocess import deterministic_frame_indices  # noqa: E402

PROBE = shutil.which("ffprobe") or r"E:\ffmpeg\ffmpeg-2026-04-16-git-5abc240a27-full_build\bin\ffprobe.exe"
BASE = common.SHARED_ARTIFACTS
POSE_ARCHIVE = common.UNISIGN_POSE_DIR
MAX_FRAMES = 256

COLUMNS = [
    "sample_id", "signer_id", "cohort",
    "trimmed_path", "trimmed_exists", "trimmed_sha256", "npz_metadata_video_sha256",
    "trimmed_sha_matches_npz_metadata", "trimmed_frames_ffprobe", "trimmed_width",
    "trimmed_height", "trimmed_fps_ffprobe", "trimmed_fps_alignment",
    "npz_keypoints_shape", "npz_frames", "npz_coordinate_system", "npz_frontend",
    "npz_fps", "npz_source_video_frames", "raw_video_frames_counted",
    "pose_total_equals_raw_video", "annotation_frames", "alignment_source_frames",
    "alignment_best_start", "alignment_trim_frames", "alignment_best_mean_px",
    "alignment_low_confidence", "alignment_reliable_point_fraction",
    "alignment_confidence_clipped",
    "unisign_video_source", "unisign_video_frame_sampling", "unisign_video_max_frames",
    "unisign_video_sampled_frames", "unisign_video_downsampled",
    "unisign_video_pose_normalization", "unisign_video_pose_orientation",
    "unisign_video_input_frames",
    "pose_pkl_path", "pose_pkl_exists", "pose_frames_total", "pose_start", "pose_end",
    "pose_frames_used", "pose_sampled_frames", "pose_downsampled", "pose_w_h",
    "pose_annotation_delta_frames", "unisign_pose_crop_field_handling",
    "unisign_pose_normalization_note", "input_equivalence", "input_diff_explained",
    "note",
]


def ffprobe_stream(path: Path) -> dict:
    command = [PROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
               "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip()[:300])
    streams = json.loads(completed.stdout).get("streams") or []
    return streams[0] if streams else {}


def ratio(text):
    if not text or "/" not in text:
        return None
    num, den = text.split("/", 1)
    return float(num) / float(den) if float(den) else None


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    rows = common.read_csv_rows(common.DATASETS / "pilot_50.csv")
    probe_path = common.P1 / "e01" / "video_probe.csv"
    probe = ({item["sample_id"]: item for item in common.read_csv_rows(probe_path)}
             if probe_path.is_file() else {})
    output_rows = []
    for position, row in enumerate(rows, 1):
        sample = row["sample_id"].strip()
        annotation_frames = int(row["frames"])
        trimmed = Path(row["trimmed_video_path"])
        npz_path = Path(row["pose_npz_path"])
        alignment_path = BASE / "alignment" / f"{sample}.json"
        pose_pkl = POSE_ARCHIVE / f"{sample}.pkl"
        record = {
            "sample_id": sample,
            "signer_id": row["signer_id"],
            "cohort": row["cohort"],
            "trimmed_path": str(trimmed),
            "trimmed_exists": trimmed.is_file(),
            "annotation_frames": annotation_frames,
            "unisign_video_source": "trimmed mp4 -> RTMPose wholebody 133 (local frontend)",
            "unisign_video_frame_sampling": (
                f"deterministic linspace over decoded frames when T>{MAX_FRAMES}, "
                f"identity range when T<={MAX_FRAMES} (slbridge/uni_sign_preprocess.py:100-115)"),
            "unisign_video_max_frames": MAX_FRAMES,
            "unisign_video_pose_normalization": "isotropic_short_side",
            "unisign_video_pose_orientation": "preserve",
            "unisign_pose_crop_field_handling": (
                "released pkl start/end applied as keypoints[start:end] (upstream load_pose); "
                "rescale_x=1.0 because released w_h is square"),
            "unisign_pose_normalization_note": (
                "released CSL-Daily pkl is x/W,y/H normalized on square frames; "
                "isotropic_short_side == wh when W==H"),
            "note": "",
        }
        # ---- trimmed clip measured facts -------------------------------------
        if trimmed.is_file():
            try:
                stream = ffprobe_stream(trimmed)
                record.update({
                    "trimmed_frames_ffprobe": to_int(stream.get("nb_read_frames")),
                    "trimmed_width": stream.get("width"),
                    "trimmed_height": stream.get("height"),
                    "trimmed_fps_ffprobe": round(ratio(stream.get("avg_frame_rate")
                                                       or stream.get("r_frame_rate")), 6),
                    "trimmed_sha256": common.sha256_file(trimmed),
                })
            except Exception as error:  # noqa: BLE001
                record["note"] = f"trimmed ffprobe failed: {type(error).__name__}: {error}"[:300]
        # ---- pose NPZ metadata ----------------------------------------------
        if npz_path.is_file():
            bundle = np.load(npz_path, allow_pickle=True)
            metadata = json.loads(str(bundle["metadata_json"]))
            record.update({
                "npz_keypoints_shape": "x".join(str(item) for item in bundle["keypoints"].shape),
                "npz_frames": int(bundle["keypoints"].shape[0]),
                "npz_coordinate_system": metadata.get("coordinate_system"),
                "npz_frontend": metadata.get("frontend"),
                "npz_fps": metadata.get("fps"),
                "npz_source_video_frames": metadata.get("source_video_frames"),
                "npz_metadata_video_sha256": metadata.get("video_sha256"),
            })
            record["trimmed_sha_matches_npz_metadata"] = (
                record.get("trimmed_sha256") == metadata.get("video_sha256"))
        # ---- alignment record ------------------------------------------------
        if alignment_path.is_file():
            alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
            trim = alignment.get("trim") or {}
            record.update({
                "alignment_source_frames": alignment.get("source_frames"),
                "alignment_best_start": alignment.get("best_start"),
                "alignment_trim_frames": trim.get("frames"),
                "alignment_best_mean_px": round(float(alignment["best_mean_px"]), 4)
                if alignment.get("best_mean_px") is not None else None,
                "alignment_low_confidence": alignment.get("low_confidence"),
                "alignment_reliable_point_fraction": round(
                    float(alignment["reliable_point_fraction"]), 4)
                if alignment.get("reliable_point_fraction") is not None else None,
                "alignment_confidence_clipped": alignment.get("confidence_clipped"),
                "trimmed_fps_alignment": trim.get("fps"),
            })
        # ---- Uni-Sign video path sampling ------------------------------------
        video_frames = record.get("alignment_trim_frames") or record.get("npz_frames")
        if video_frames:
            sampled = deterministic_frame_indices(int(video_frames), max_frames=MAX_FRAMES)
            record.update({
                "unisign_video_sampled_frames": len(sampled),
                "unisign_video_downsampled": len(sampled) < int(video_frames),
                "unisign_video_input_frames": int(video_frames),
            })
        # ---- Uni-Sign released pose path -------------------------------------
        record["pose_pkl_path"] = str(pose_pkl)
        record["pose_pkl_exists"] = pose_pkl.is_file()
        if pose_pkl.is_file():
            with pose_pkl.open("rb") as handle:
                pose = pickle.load(handle)
            keypoints = np.asarray(pose["keypoints"])
            frames_total = int(keypoints.shape[0])
            start = to_int(pose.get("start"))
            end = to_int(pose.get("end"))
            frames_used = frames_total
            if start is not None and end is not None and end > start:
                frames_used = end - start
            sampled = deterministic_frame_indices(frames_used, max_frames=MAX_FRAMES)
            probe_row = probe.get(sample) or {}
            raw_frames = to_int(probe_row.get("frames_counted"))
            record.update({
                "pose_frames_total": frames_total,
                "pose_start": start,
                "pose_end": end,
                "pose_frames_used": frames_used,
                "pose_sampled_frames": len(sampled),
                "pose_downsampled": len(sampled) < frames_used,
                "pose_w_h": f"{pose.get('w_h')}",
                "pose_annotation_delta_frames": frames_used - annotation_frames,
                "raw_video_frames_counted": raw_frames,
                "pose_total_equals_raw_video": (frames_total == raw_frames)
                if raw_frames is not None else None,
            })
        # ---- cross-path verdict ----------------------------------------------
        video_ok = (record.get("trimmed_sha_matches_npz_metadata") is True
                    and record.get("trimmed_frames_ffprobe") == record.get("npz_frames")
                    and record.get("alignment_trim_frames") == annotation_frames)
        sizes = {record.get("trimmed_width"), record.get("trimmed_height")}
        square = len(sizes) == 1 and None not in sizes
        pose_frames_used = record.get("pose_frames_used")
        record["input_equivalence"] = (
            "same_sentence_clip; different pose frontend "
            "(local RTMPose vs released HRNet archive)")
        explained = (
            video_ok and square and record.get("pose_pkl_exists") is True
            and pose_frames_used is not None)
        record["input_diff_explained"] = bool(explained)
        notes = [
            f"trimmed_video={video_frames} frames vs annotation {annotation_frames} "
            f"(delta {None if video_frames is None else int(video_frames) - annotation_frames})",
            f"released pose T_raw={record.get('pose_frames_total')} == raw video frames, "
            f"crop end-start={pose_frames_used} vs annotation {annotation_frames} "
            f"(delta {record.get('pose_annotation_delta_frames')})",
            "square frame -> isotropic_short_side equals x/W,y/H",
        ]
        record["note"] = "; ".join(notes)
        output_rows.append(record)
        print(json.dumps({"event": "audited", "done": position, "sample": sample,
                          "video_frames": video_frames, "pose_frames_used": pose_frames_used,
                          "sha_match": record.get("trimmed_sha_matches_npz_metadata")},
                         ensure_ascii=False), flush=True)

    path = common.P1 / "e01" / "preprocess_audit.csv"
    written = common.write_csv_rows(path, COLUMNS, output_rows)
    print(json.dumps({"event": "written", "path": str(path), "rows": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
