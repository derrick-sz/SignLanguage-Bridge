"""E01 aggregate: build experiments/p1/e01/summary.json from the three run dirs.

Reads only the artifact/evidence files written by the E01 steps, cross-checks
them, and emits one machine-readable summary.  The narrative report
(``reproduction_report.md``) is written separately and cites these numbers.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\build_e01_summary.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402

RUN_IDS = {
    "U1_unisign_video": "e01-pilot50-u1-unisign-video-20261026",
    "U2_unisign_pose": "e01-pilot50-u2-unisign-pose-20261026",
    "S1_slrt_gloss": "e01-pilot50-s1-slrt-gloss-20261026",
}
HISTORY_1086 = {
    "U1_unisign_video": {
        "path": "artifacts/csldaily-video-test-627-20261026/unisign-video-1086-paperrefs/predictions.jsonl",
        "bleu4_13a": 24.81568280610249,
        "chrf": 21.79739979299852,
        "mean_char_edit_similarity": 0.4156886922997997,
        "baseline_n": 1086,
    },
    "U2_unisign_pose": {
        "path": "artifacts/cslnews-eval/csldaily-test-1086-pose-20261026/predictions.jsonl",
        "bleu4_13a": 25.58456044558977,
        "chrf": 22.428395124236005,
        "mean_char_edit_similarity": 0.42119971772304904,
        "baseline_n": 1086,
    },
    "S1_slrt_gloss": {
        "path": "artifacts/csldaily-video-test-627-20261026/slrt/<sample>.json",
        "corpus_wer": 0.3018,
        "baseline_n": 1086,
        "note": "historical 1086 SLRT corpus WER 0.3018 (S1,154/D805/I522); the artifact tree here "
                "holds the 1,086 per-sample Gloss files",
    },
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def numeric_column(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def main() -> int:
    e01 = common.P1 / "e01"
    pilot = common.read_csv_rows(common.DATASETS / "pilot_50.csv")
    probe = common.read_csv_rows(e01 / "video_probe.csv")
    audit = common.read_csv_rows(e01 / "preprocess_audit.csv")

    annotation_frames = [int(row["frames"]) for row in pilot]
    raw_frames = [int(row["frames_counted"]) for row in probe if row.get("frames_counted")]
    best_mean = numeric_column(audit, "alignment_best_mean_px")
    pose_delta = [int(row["pose_annotation_delta_frames"]) for row in audit
                  if row.get("pose_annotation_delta_frames") not in (None, "")]
    pose_delta_abs = [abs(value) for value in pose_delta]

    video_probe = {
        "rows": len(probe),
        "all_ok": all(row.get("status") == "ok" for row in probe),
        "probe_method": "ffprobe -count_frames (decoded frame count); fps from ffprobe avg_frame_rate",
        "raw_frames": {
            "min": min(raw_frames),
            "median": statistics.median(raw_frames),
            "max": max(raw_frames),
            "sum": sum(raw_frames),
        },
        "annotation_frames": {
            "min": min(annotation_frames),
            "median": statistics.median(annotation_frames),
            "max": max(annotation_frames),
            "equals_raw_frame_count": sum(
                1 for probe_row in probe
                if probe_row.get("annotation_equals_raw_frames") == "True"),
            "note": "pilot CSV 'frames' is the annotated sentence length, not the raw recording length",
        },
        "dims": sorted({(row.get("width"), row.get("height")) for row in probe}),
        "fps_values": sorted({row.get("fps") for row in probe}),
        "codec": sorted({row.get("codec_name") for row in probe}),
        "pix_fmt": sorted({row.get("pix_fmt") for row in probe}),
        "bytes_total": sum(int(row["bytes"]) for row in probe if row.get("bytes")),
        "sha256_recorded": sum(1 for row in probe if row.get("sha256")),
        "counted_equals_declared": sum(1 for row in probe
                                       if row.get("counted_equals_declared") == "True"),
        "csv": str(e01 / "video_probe.csv"),
    }

    preprocess_audit = {
        "rows": len(audit),
        "trimmed_sha256_matches_npz_metadata": sum(
            1 for row in audit if row.get("trimmed_sha_matches_npz_metadata") == "True"),
        "trimmed_frames_equal_npz_frames": sum(
            1 for row in audit if row.get("trimmed_frames_ffprobe") == row.get("npz_frames")),
        "trimmed_frames_equal_annotation": sum(
            1 for row in audit if row.get("alignment_trim_frames") == row.get("annotation_frames")),
        "ffprobe_equal_alignment_frames": sum(
            1 for row in audit if row.get("alignment_trim_frames") == row.get("trimmed_frames_ffprobe")),
        "low_confidence_samples": sum(
            1 for row in audit if row.get("alignment_low_confidence") == "True"),
        "best_mean_px": {
            "min": round(min(best_mean), 3),
            "mean": round(statistics.mean(best_mean), 3),
            "max": round(max(best_mean), 3),
        },
        "confidence_clipped_total": sum(
            int(row["alignment_confidence_clipped"]) for row in audit
            if row.get("alignment_confidence_clipped") not in (None, "")),
        "npz": {
            "coordinate_systems": sorted({row.get("npz_coordinate_system") for row in audit}),
            "frontends": sorted({row.get("npz_frontend") for row in audit}),
            "keypoint_shapes": sorted({row.get("npz_keypoints_shape") for row in audit}),
        },
        "frame_sampling": {
            "video_path": "deterministic linspace over decoded frames when T>256 "
                          "(slbridge/uni_sign_preprocess.py:100-115)",
            "pilot_clips_over_256": sum(
                1 for row in audit if row.get("unisign_video_downsampled") == "True"),
            "pose_pkls_over_256": sum(
                1 for row in audit if row.get("pose_downsampled") == "True"),
            "note": "all 50 pilot sentence clips are <=172 annotated frames; no downsampling in practice "
                    "except the one released pose pkl whose start/end crop is 303 frames",
        },
        "released_pose_crop": {
            "pkl_frames_total_equals_raw_video": sum(
                1 for row in audit if row.get("pose_total_equals_raw_video") == "True"),
            "crop_frames_minus_annotation": {
                "min": min(pose_delta),
                "median": statistics.median(pose_delta),
                "max": max(pose_delta),
                "nonzero": sum(1 for value in pose_delta if value),
                "abs_gt_5": sum(1 for value in pose_delta_abs if value > 5),
            },
            "note": "released pkl keeps full-video 133-point poses; the crop [start,end) is the "
                    "segment the model consumes and does not always equal the annotated sentence length",
        },
        "normalization": {
            "video_path": "isotropic_short_side (all clips are square 1280x1280 / 1080x1080, so this "
                          "equals x/W,y/H)",
            "pose_path": "x/W,y/H normalized released pose on square frames; rescale_x=1.0",
            "equivalence": "square frames -> both conventions are numerically identical; only the pose "
                           "frontend (local RTMPose vs released HRNet archive) and the crop/frame set differ",
        },
        "input_diff_explained": sum(
            1 for row in audit if row.get("input_diff_explained") == "True"),
        "csv": str(e01 / "preprocess_audit.csv"),
    }

    paths = {}
    for key, run_id in RUN_IDS.items():
        run_dir = common.RUNS / run_id
        manifest = load_json(run_dir / "manifest.json")
        comparison = load_json(run_dir / "comparison.json")
        metrics = load_json(run_dir / "metrics.json")
        paths[key] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": (manifest or {}).get("run_status"),
            "manifest": str(run_dir / "manifest.json"),
            "model_outputs": str(run_dir / "model_outputs.jsonl"),
            "metrics_json": str(run_dir / "metrics.json"),
            "eval_records": str(run_dir / "eval_records.jsonl"),
            "comparison_json": str(run_dir / "comparison.json"),
            "sample_counts": (manifest or {}).get("sample_counts"),
            "failure_samples": (manifest or {}).get("failure_samples"),
            "history_compare": (comparison or {}).get("match_rates"),
            "history_label": (comparison or {}).get("history_label"),
            "canonical_metrics_rerun": (comparison or {}).get("canonical_metrics_rerun_vs_gt1086"),
            "canonical_metrics_history_same_subset": (comparison or {}).get(
                "canonical_metrics_history_vs_gt1086"),
            "script_metrics_secondary_reference": (comparison or {}).get(
                "script_metrics_secondary_reference"),
            "script_paired_comparison_vs_upstream": (comparison or {}).get(
                "script_paired_comparison_vs_upstream"),
            "latency": (comparison or {}).get("latency_rerun") or (comparison or {}).get("latency_seconds"),
            "metric_scope": (metrics or {}).get("metric_scope"),
        }

    u1 = paths["U1_unisign_video"]["canonical_metrics_rerun"] or {}
    u2 = paths["U2_unisign_pose"]["canonical_metrics_rerun"] or {}
    s1 = paths["S1_slrt_gloss"]["canonical_metrics_rerun"] or {}
    video_vs_pose = {
        "video_minus_pose_bleu4": (
            round(float(u1["bleu4_13a"]) - float(u2["bleu4_13a"]), 4)
            if u1.get("bleu4_13a") is not None and u2.get("bleu4_13a") is not None else None),
        "video_minus_pose_chrf": (
            round(float(u1["chrf"]) - float(u2["chrf"]), 4)
            if u1.get("chrf") is not None and u2.get("chrf") is not None else None),
        "video_minus_pose_edit_similarity": (
            round(float(u1["mean_char_edit_similarity"]) - float(u2["mean_char_edit_similarity"]), 4)
            if u1.get("mean_char_edit_similarity") is not None
            and u2.get("mean_char_edit_similarity") is not None else None),
        "n": 50,
        "scope": "same 50 pilot samples, same checkpoint/decode, same canonical reference; the two "
                 "input paths differ only in pose frontend and frame set",
    }

    # two historical baselines cannot be compared as if they were the same measurement
    historical = {}
    for key, item in HISTORY_1086.items():
        entry = dict(item)
        entry["role"] = "imported"
        entry["comparable_to"] = (
            "same reference file and subset rules; not re-inferred by this task")
        historical[key] = entry

    summary = {
        "experiment_id": "E01",
        "title": "预处理一致性与推理可复现性（Pilot-50：U1 Uni-Sign 视频 / U2 Uni-Sign 官方 pose / S1 SLRT Gloss）",
        "created_at": common.utc_now(),
        "git": common.git_state(),
        "pilot": {
            "n": len(pilot),
            "signers": sorted({row["signer_id"] for row in pilot}),
            "signer_count": len({row["signer_id"] for row in pilot}),
            "cohorts": {cohort: sum(1 for row in pilot if row["cohort"] == cohort)
                        for cohort in sorted({row["cohort"] for row in pilot})},
            "annotation_frames": video_probe["annotation_frames"],
            "pilot_csv": str(common.DATASETS / "pilot_50.csv"),
            "pilot_csv_sha256": common.sha256_file(common.DATASETS / "pilot_50.csv"),
            "sample_of": "selection-1086.json (the 1,086 locally available CSL-Daily test clips)",
            "boundary": "Pilot-50 is a 50/1,086 subset of the local test coverage (1,086/1,176 of the "
                        "official test split); retrospective diagnostic only. Metrics here are subset "
                        "numbers and are not comparable to full-split paper numbers.",
        },
        "inputs": {"video_probe": video_probe, "preprocess_audit": preprocess_audit},
        "paths": paths,
        "comparisons": {
            "unisign_video_vs_pose_on_pilot50": video_vs_pose,
            "historical_baselines_imported_1086": historical,
        },
        "reference_variants": {
            "canonical_primary": "gt_1086.csv :: ref_sentence_upstream (upstream test_tmp_refs style: "
                                 "character-separated, half-width punctuation) -> used for all "
                                 "cross-run comparisons here",
            "official_annotation": "csl-daily.test text / gt_1086.gt_sentence_official (full-width "
                                   "punctuation, no spaces) -> only the frozen video script's own "
                                   "BLEU summary uses it",
            "warning": "the two variants differ by ~2.8-3.1 BLEU for identical predictions; never mix "
                       "them in one comparison",
            "observed_gap_pilot50": {
                "video_script_official_ref_bleu4": (
                    paths["U1_unisign_video"]["script_metrics_secondary_reference"] or {}).get("bleu4_13a"),
                "video_canonical_upstream_ref_bleu4": u1.get("bleu4_13a"),
                "delta": (
                    round(float(u1["bleu4_13a"])
                          - float((paths["U1_unisign_video"]["script_metrics_secondary_reference"]
                                   or {}).get("bleu4_13a")), 4)
                    if u1.get("bleu4_13a") is not None
                    and (paths["U1_unisign_video"]["script_metrics_secondary_reference"] or {}).get(
                        "bleu4_13a") is not None else None),
            },
        },
        "status": {
            "overall": "PASS" if all((paths[key]["status"] == "PASS") for key in RUN_IDS) else "FAIL",
            "per_path": {key: paths[key]["status"] for key in RUN_IDS},
            "evidence_gate": "PASS only where the run actually executed and saved evidence; "
                             "imported historical numbers are marked 'imported'",
        },
        "measured_vs_imported": {
            "measured": ["video_probe.csv", "preprocess_audit.csv", "U1 predictions + timings",
                         "U2 predictions + timings", "S1 Gloss + exit codes + timings",
                         "canonical metrics on the 50 subset"],
            "imported": ["historical 1086 Uni-Sign video predictions",
                         "historical 1086 Uni-Sign pose predictions",
                         "historical 1086 SLRT Gloss outputs",
                         "trimmed clips / NPZ poses / alignment records used as S1 input (reused, "
                         "not re-extracted)"],
            "not_run": ["S2 official crop/keypoint tooling re-check (optional branch)",
                        "repeatability repeats (same input twice) for the stochastic sampling path"],
        },
        "limitations": [
            "SLRT align stage was not re-run for S1: trimmed clips, HRNet NPZ poses and alignment "
            "records are the reviewed historical artifacts (byte-identical, verified by NPZ metadata "
            "video_sha256), so S1 measures inference reproducibility only.",
            "The released pose path consumes keypoints[start:end]; for 45/50 samples this differs from "
            "the annotated sentence length (median +3, max +140 frames). That is an imported-artifact "
            "property, identical in the historical run, and is recorded per sample in "
            "preprocess_audit.csv.",
            "Uni-Sign video path uses the local RTMPose frontend, not the official HRNet extractor.",
            "Pilot-50 has 10 signers; per-signer metrics are unstable at n=5.",
            "No legal/licence review was performed for the model weights or the dataset (CC-BY-NC-4.0 "
            "weights, CSL-Daily terms apply).",
        ],
    }
    path = common.write_json(e01 / "summary.json", summary)
    print(json.dumps({"event": "summary", "path": str(path),
                      "status": summary["status"],
                      "video_vs_pose": video_vs_pose}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
