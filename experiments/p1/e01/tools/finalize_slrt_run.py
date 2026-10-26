"""E01 finalize: turn the SLRT inference rerun into a P-1 run directory.

Reads the per-sample records the frozen driver
(``scripts/align_slrt_csl_daily_subset.py --stage infer``) wrote into the new
artifact tree and produces:

* ``model_outputs.jsonl``  GT-free per-sample Gloss sequences + exit codes,
* ``manifest.json``        provenance (git, hardware, weights, dataset snapshot,
                           command, counts, failures, status),
* ``metrics.json`` /
  ``eval_records.jsonl``   canonical CSLR corpus WER (``eval/evaluate.py --task CSLR``),
* ``comparison.json`` +
  ``experiments/p1/e01/compare_slrt.csv``  per-sample comparison against the
                           historical ``artifacts/.../slrt/<sample>.json`` Gloss.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\finalize_slrt_run.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402
from experiments.p1.eval import evaluate as evaluator  # noqa: E402
from experiments.p1.eval import metrics as metric_lib  # noqa: E402

RUN_ID = "e01-pilot50-s1-slrt-gloss-20261026"
MODEL_ID = "slrt-emnlp24-csldaily-twostream-slr"
HISTORY_ARTIFACT = common.SHARED_ARTIFACTS
COMMAND = (
    ".conda-env\\python.exe -X utf8 scripts\\align_slrt_csl_daily_subset.py --stage infer "
    "--selection experiments\\p1\\runs\\e01-pilot50-s1-slrt-gloss-20261026\\inputs\\selection-pilot50.json "
    "--annotation F:\\model_weigth\\data\\csl-daily\\csl-daily.test "
    "--artifact-dir experiments\\p1\\runs\\e01-pilot50-s1-slrt-gloss-20261026\\artifact "
    "--only <50 pilot ids> --force"
)



def sanitize_for_json(value):
    """Convert non-JSON values (e.g. sacreBLEU signature objects) to strings."""

    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def aggregate_hash(entries: dict[str, str]) -> str:
    payload = "\n".join(f"{key}:{entries[key]}" for key in sorted(entries))
    return common.sha256_text(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    run_dir = common.RUNS / args.run_id
    artifact = run_dir / "artifact"
    pilot_ids = [line.strip() for line in
                 (common.P1 / "e01" / "inputs" / "pilot_50_ids.txt").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    pilot_meta = {row["sample_id"].strip(): row for row in
                  common.read_csv_rows(common.DATASETS / "pilot_50.csv")}

    # ---- per-sample records from the frozen driver -------------------------
    model_rows = []
    for sample in pilot_ids:
        record_path = artifact / "infer" / f"{sample}.json"
        result_path = artifact / "slrt" / f"{sample}.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        gloss = result.get("gloss_sequence")
        if isinstance(gloss, list):
            gloss = [str(token) for token in gloss]
        exit_code = record.get("exit_code")
        status = str(result.get("status") or record.get("status") or "missing")
        success = exit_code == 0 and status == "final" and bool(record.get("parsed"))
        if success:
            output_status = "ok" if gloss else "empty"
        else:
            output_status = "error"
        row = {
            "sample_id": sample,
            "run_id": args.run_id,
            "model_id": MODEL_ID,
            "task": "CSLR",
            "modality": "trimmed_video+hrnet_npz_pose",
            "input_kind": "SLRT official-protocol sliding windows over trimmed clip + released HRNet pose NPZ",
            "attempt": 1,
            "status": output_status,
            "gloss_sequence": gloss or [],
            "prediction": " ".join(gloss or []),
            "empty_output": success and not gloss,
            "exit_code": exit_code,
            "pipeline_status": status,
            "elapsed_seconds": record.get("elapsed_seconds"),
            "error": record.get("error"),
            "frames_received": result.get("frames_received"),
            "windows_processed": result.get("windows_processed"),
            "quality_rejected_frames": result.get("quality_rejected_frames"),
            "dropped_frames": result.get("dropped_frames"),
            "video_decode_ms": result.get("video_decode_ms"),
            "wall_seconds": result.get("wall_seconds"),
            "processing_fps": result.get("processing_fps"),
            "config": result.get("config"),
            "pose": result.get("pose"),
            "run_record_path": str(record_path),
            "result_path": str(result_path),
        }
        model_rows.append(row)
    common.write_jsonl(run_dir / "model_outputs.jsonl", model_rows)
    common.write_jsonl(run_dir / "runtime_metrics.jsonl",
                       [{key: row.get(key) for key in
                         ("sample_id", "status", "exit_code", "elapsed_seconds",
                          "video_decode_ms", "wall_seconds", "processing_fps")}
                        for row in model_rows], guard=False)
    failures = [row["sample_id"] for row in model_rows if row["status"] == "error"]
    empties = [row["sample_id"] for row in model_rows if row["status"] == "empty"]
    successes = [row for row in model_rows if row["status"] in ("ok", "empty")]

    # ---- canonical evaluation (CSLR corpus WER) ----------------------------
    references_path = common.REFERENCES / "gt_1086.csv"
    payload, eval_records = evaluator.evaluate(
        run_dir / "model_outputs.jsonl", references_path, task="CSLR",
        bootstrap_iterations=0)
    common.write_json(run_dir / "metrics.json", sanitize_for_json(payload))
    common.write_jsonl(run_dir / "eval_records.jsonl", eval_records, guard=False)

    # ---- compare against the historical SLRT Gloss outputs -----------------
    comparison_rows = []
    exact = 0
    token_set_equal = 0
    compared = 0
    history_available = 0
    for sample in pilot_ids:
        history_path = HISTORY_ARTIFACT / "slrt" / f"{sample}.json"
        history_infer_path = HISTORY_ARTIFACT / "infer" / f"{sample}.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else None
        history_infer = (json.loads(history_infer_path.read_text(encoding="utf-8"))
                        if history_infer_path.is_file() else {})
        if history is not None:
            history_available += 1
        row = next(item for item in model_rows if item["sample_id"] == sample)
        history_gloss = [str(token) for token in (history or {}).get("gloss_sequence") or []]
        rerun_gloss = [str(token) for token in row.get("gloss_sequence") or []]
        history_ok = bool(history) and (history or {}).get("status") == "final"
        rerun_ok = row["status"] in ("ok", "empty")
        same = (history_gloss == rerun_gloss) if (history_ok and rerun_ok) else False
        if history_ok and rerun_ok:
            compared += 1
            exact += int(same)
            token_set_equal += int(sorted(history_gloss) == sorted(rerun_gloss))
        comparison_rows.append({
            "sample_id": sample,
            "signer_id": (pilot_meta.get(sample) or {}).get("signer_id") or "",
            "cohort": (pilot_meta.get(sample) or {}).get("cohort") or "",
            "history_status": (history or {}).get("status"),
            "history_exit_code": history_infer.get("exit_code"),
            "history_elapsed_seconds": history_infer.get("elapsed_seconds"),
            "rerun_status": row["pipeline_status"] if row["status"] != "empty" else "final",
            "rerun_output_status": row["status"],
            "history_gloss": " ".join(history_gloss),
            "rerun_gloss": " ".join(rerun_gloss),
            "gloss_exact_equal": same,
            "history_tokens": len(history_gloss),
            "rerun_tokens": len(rerun_gloss),
            "token_multiset_equal": sorted(history_gloss) == sorted(rerun_gloss)
            if (history_ok and rerun_ok) else False,
            "exit_code": row["exit_code"],
            "elapsed_seconds": row["elapsed_seconds"],
            "empty_output": row["empty_output"],
            "error": row["error"],
        })
    columns = ["sample_id", "signer_id", "cohort", "history_status", "history_exit_code",
               "history_elapsed_seconds", "rerun_status", "rerun_output_status",
               "history_gloss", "rerun_gloss", "gloss_exact_equal", "history_tokens",
               "rerun_tokens", "token_multiset_equal", "exit_code", "elapsed_seconds",
               "empty_output", "error"]
    compare_path = common.P1 / "e01" / "compare_slrt.csv"
    common.write_csv_rows(compare_path, columns, comparison_rows)

    # ---- manifest ----------------------------------------------------------
    checkpoint = common.SLRT_CHECKPOINT
    config_path = common.SLRT_CONFIG
    selection_path = run_dir / "inputs" / "selection-pilot50.json"
    trimmed_digests, pose_digests = {}, {}
    for sample in pilot_ids:
        trimmed = artifact / "trimmed" / f"{sample}.mp4"
        npz = artifact / "pose" / f"{sample}.npz"
        if trimmed.is_file():
            trimmed_digests[sample] = common.sha256_file(trimmed)
        if npz.is_file():
            pose_digests[sample] = common.sha256_file(npz)
    latencies = [row["elapsed_seconds"] for row in model_rows
                 if isinstance(row.get("elapsed_seconds"), (int, float))]
    history_latencies = [float(row["history_elapsed_seconds"]) for row in comparison_rows
                         if isinstance(row.get("history_elapsed_seconds"), (int, float))]
    manifest = {
        "run_id": args.run_id,
        "experiment_id": "E01",
        "run_status": "PASS" if (len(successes) == len(pilot_ids) and not failures) else "FAIL",
        "path": "slrt_gloss",
        "purpose": "E01 S1 SLRT Gloss reproduction on Pilot-50",
        "created_at": common.utc_now(),
        "operator": "reproducer",
        "code": common.git_state(),
        "host": {"host_alias": common.host_alias(), **common.environment_fingerprint()},
        "model": {
            "model_id": MODEL_ID,
            "task": "CSLR (CSL-Daily TwoStream-SLR, 2001 classes)",
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": common.sha256_file(checkpoint),
            "config": str(config_path),
            "config_sha256": common.sha256_file(config_path),
            "protocol": {
                "window_size": 16, "stride": 1, "vote_size": 7,
                "pose_backend": "hrnet-npz", "device": "cuda",
                "note": "CLI defaults as used by scripts/align_slrt_csl_daily_subset.py stage_infer",
            },
        },
        "input_protocol": {
            "kind": "slrt_gloss",
            "description": "trimmed mp4 + released-HRNet aligned NPZ -> official 16-frame sliding "
                           "windows, stride 1, 7-window majority vote -> Gloss",
            "align_stage_rerun": False,
            "align_stage_reuse": "trimmed/pose/alignment copied from "
                                 "artifacts/csldaily-video-test-627-20261026 (verified by NPZ metadata video_sha256)",
        },
        "dataset_snapshot": {
            "pilot_50_csv": str(common.DATASETS / "pilot_50.csv"),
            "pilot_50_csv_sha256": common.sha256_file(common.DATASETS / "pilot_50.csv"),
            "pilot_ids_sha256": common.sha256_file(common.P1 / "e01" / "inputs" / "pilot_50_ids.txt"),
            "sample_count": len(pilot_ids),
            "signers": sorted({(pilot_meta.get(row["sample_id"]) or {}).get("signer_id")
                                for row in model_rows} - {None, ""}),
            "filtered_selection": str(selection_path),
            "filtered_selection_sha256": common.sha256_file(selection_path),
            "trimmed_clip_sha256_aggregate": aggregate_hash(trimmed_digests),
            "pose_npz_sha256_aggregate": aggregate_hash(pose_digests),
            "annotation_pickle": r"F:\model_weigth\data\csl-daily\csl-daily.test",
            "official_keypoints_pickle": r"F:\model_weigth\data\csl-daily\csl-daily-keypoints.pkl",
            "reference_table": str(references_path),
            "reference_table_sha256": common.sha256_file(references_path),
            "reference_table_role": "EVAL ONLY (WER join, never fed to the model)",
        },
        "history_reference": {
            "label": "artifacts/csldaily-video-test-627-20261026/slrt/<sample>.json",
            "path": str(HISTORY_ARTIFACT / "slrt"),
            "samples_available": history_available,
            "role": "historical SLRT Gloss outputs imported for comparison, not re-inferred",
        },
        "command": COMMAND,
        "cwd": str(ROOT),
        "outputs": {
            "model_outputs": str(run_dir / "model_outputs.jsonl"),
            "runtime_metrics": str(run_dir / "runtime_metrics.jsonl"),
            "metrics": str(run_dir / "metrics.json"),
            "eval_records": str(run_dir / "eval_records.jsonl"),
            "artifact_dir": str(artifact),
            "comparison_csv": str(compare_path),
            "comparison": str(run_dir / "comparison.json"),
        },
        "sample_counts": {
            "planned": len(pilot_ids),
            "successful": len(successes),
            "failed": len(failures),
            "empty_outputs": len(empties),
            "corpus_wer_scope": "all 50 samples" if not failures else "successful subset only",
        },
        "failure_samples": [{"sample_id": row["sample_id"], "status": row["status"],
                             "exit_code": row["exit_code"], "error": row.get("error")}
                            for row in model_rows if row["status"] == "error"],
        "empty_output_samples": empties,
        "gt_policy": {
            "model_outputs_gt_free": True,
            "gt_used_only_in": ["metrics.json", "eval_records.jsonl"],
            "guarded_writer": "experiments.p1.common.write_jsonl(assert_gt_clean)",
        },
        "privacy_and_licensing": {
            "rights_checked": False,
            "no_raw_media_export": True,
            "no_credentials_in_logs": True,
        },
        "notes": [
            "single process, exclusive GPU; SLRT CLI per sample is a short-lived subprocess",
            "historical artifacts were only read; nothing under artifacts/ was modified",
            "align stage NOT rerun: input clips/pose/alignment are the reviewed historical ones",
        ],
    }
    common.write_json(run_dir / "manifest.json", manifest)

    wer_metrics = payload.get("metrics") or {}
    comparison = {
        "kind": "slrt_gloss",
        "run_id": args.run_id,
        "n_samples": len(pilot_ids),
        "history_available": history_available,
        "compared_n": compared,
        "rerun_successful": len(successes),
        "rerun_failed": len(failures),
        "rerun_empty": len(empties),
        "match_rates": {
            "gloss_sequence_exact": exact,
            "token_multiset_equal": token_set_equal,
            "denominator": compared,
            "exact_rate": round(exact / compared, 4) if compared else None,
        },
        "canonical_metrics_rerun_vs_gt1086": {
            key: wer_metrics.get(key) for key in
            ("n", "corpus_wer", "substitutions", "deletions", "insertions",
             "reference_tokens", "hypothesis_tokens")
        },
        "latency_seconds": {
            "n": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": round(common.percentile(latencies, 50), 3) if latencies else None,
            "p95": round(common.percentile(latencies, 95), 3) if latencies else None,
            "scope": "per-sample elapsed of the SLRT CLI subprocess incl. ~3.9 s weight load",
        },
        "latency_seconds_history": {
            "n": len(history_latencies),
            "mean": round(sum(history_latencies) / len(history_latencies), 3)
            if history_latencies else None,
            "p50": round(common.percentile(history_latencies, 50), 3)
            if history_latencies else None,
            "p95": round(common.percentile(history_latencies, 95), 3)
            if history_latencies else None,
            "scope": "historical infer/<sample>.json elapsed (may include GPU contention)",
        },
        "evaluation_scope": "Pilot-50 subset of the 1,086 local CSL-Daily test clips; corpus WER over "
                            "all 50 successful rows; reference = gt_1086.gt_gloss (EVAL ONLY)",
        "gloss_diff_samples": [
            {"sample_id": row["sample_id"],
             "history": row["history_gloss"],
             "rerun": row["rerun_gloss"]}
            for row in comparison_rows if not row["gloss_exact_equal"]
        ],
    }
    common.write_json(run_dir / "comparison.json", comparison)
    print(json.dumps({"event": "finalized", "run_id": args.run_id,
                      "match": comparison["match_rates"],
                      "wer": comparison["canonical_metrics_rerun_vs_gt1086"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
