"""E01 finalize: turn a frozen-script Uni-Sign run into a P-1 run directory.

For each of the two Uni-Sign paths (U1 video, U2 released pose) this:

1. moves the frozen script's own outputs into ``predictions-raw/`` (that script
   embeds the reference text by design, so it stays evaluation-side),
2. writes the deliverable GT-free ``model_outputs.jsonl``,
3. writes ``manifest.json`` (git, hardware, model hash, dataset snapshot, command,
   sample counts, failures, status),
4. runs the canonical evaluator (``experiments.p1.eval.evaluate``) against
   ``references/gt_1086.csv`` -> ``metrics.json`` + ``eval_records.jsonl``,
5. compares the rerun against the historical 1086 snapshot on the same 50
   samples (raw / NFKC match counts, per-sample diffs) -> ``comparison.json`` and
   ``experiments/p1/e01/compare_<kind>.csv``.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\finalize_unisign_run.py --kind video
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402
from experiments.p1.eval import evaluate as evaluator  # noqa: E402
from experiments.p1.eval import metrics as metric_lib  # noqa: E402

KINDS = {
    "video": {
        "run_id": "e01-pilot50-u1-unisign-video-20261026",
        "model_id": "unisign-csldaily-pose-only-slt",
        "modality": "video",
        "input": "trimmed mp4 -> local RTMPose wholebody 133 -> isotropic_short_side -> mT5",
        "history": "artifacts/csldaily-video-test-627-20261026/unisign-video-1086-paperrefs/predictions.jsonl",
        "history_label": "unisign-video-1086-paperrefs (imported_1086_video)",
        "compare_csv": "compare_unisign_video.csv",
        "command": (
            ".conda-env\\python.exe -X utf8 scripts\\evaluate_unisign_csldaily_video.py "
            "--selection artifacts\\csldaily-video-test-627-20261026\\selection-1086.json "
            "--artifact-dir artifacts\\csldaily-video-test-627-20261026 "
            "--output-dir experiments\\p1\\runs\\e01-pilot50-u1-unisign-video-20261026 "
            "--only-file experiments\\p1\\e01\\inputs\\pilot_50_ids.txt"
        ),
        "raw_files": ["protocol.json", "predictions.jsonl", "comparison.csv", "comparison.md",
                      "summary.json", "run.log"],
    },
    "pose": {
        "run_id": "e01-pilot50-u2-unisign-pose-20261026",
        "model_id": "unisign-csldaily-pose-only-slt",
        "modality": "released_pose",
        "input": "released csl_daily_pose_format pkl -> keypoints[start:end] -> isotropic_short_side -> mT5",
        "history": "artifacts/cslnews-eval/csldaily-test-1086-pose-20261026/predictions.jsonl",
        "history_label": "csldaily-test-1086-pose-20261026 (imported_1086_pose)",
        "compare_csv": "compare_unisign_pose.csv",
        "command": (
            ".conda-env\\python.exe -X utf8 scripts\\evaluate_unisign_csldaily_pose.py "
            "--pose-dir artifacts\\cslnews-eval\\csl-daily-pose\\pose-format "
            "--refs experiments\\p1\\runs\\e01-pilot50-u2-unisign-pose-20261026\\inputs\\refs-pilot50.txt "
            "--upstream-pres experiments\\p1\\runs\\e01-pilot50-u2-unisign-pose-20261026\\inputs\\pres-pilot50.txt "
            "--output-dir experiments\\p1\\runs\\e01-pilot50-u2-unisign-pose-20261026 "
            "--device cuda"
        ),
        "raw_files": ["protocol.json", "predictions.jsonl", "summary.json", "run.log"],
    },
}

TIMING_KEYS = ("pose_extract_ms", "pose_ms", "model_total_ms", "adapter_total_ms",
               "model_forward_ms", "generation_ms", "decode_ms", "e2e_seconds")
KEEP_KEYS = ("window_frames", "sampled_frames", "valid_body_fraction", "body_scale",
             "frames_total", "frames_used", "clip_sha256", "protocol_id", "rescale_x")


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def load_history(path: Path) -> dict[str, dict]:
    rows = {}
    for row in common.read_jsonl(path):
        sample = row.get("sample") or row.get("sample_id")
        if sample:
            rows[str(sample).strip()] = row
    return rows


def divergence(a: str, b: str) -> tuple[str, int | None]:
    """Classify why two prediction strings differ (no GT involved)."""
    if a == b:
        return "identical", None
    if nfkc(a) == nfkc(b):
        return "unicode_or_space_only", None
    stripped_a, stripped_b = "".join(a.split()), "".join(b.split())
    if stripped_a == stripped_b:
        return "whitespace_only", None
    position = None
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            position = index
            break
    if position is None:
        position = min(len(a), len(b))
    return "content_diff", position



def sanitize_for_json(value):
    """Convert non-JSON values (e.g. sacreBLEU signature objects) to strings."""

    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)def aggregate_hash(entries: dict[str, str]) -> str:
    payload = "\n".join(f"{key}:{entries[key]}" for key in sorted(entries))
    return common.sha256_text(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(KINDS), required=True)
    args = parser.parse_args(argv)
    spec = KINDS[args.kind]
    run_dir = common.RUNS / spec["run_id"]
    raw_dir = run_dir / "predictions-raw"
    raw_dir.mkdir(exist_ok=True)
    for name in spec["raw_files"]:
        source = run_dir / name
        if source.is_file():
            shutil.move(str(source), str(raw_dir / name))
    predictions_path = raw_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(f"missing predictions: {predictions_path}")
    raw_rows = common.read_jsonl(predictions_path)
    by_sample = {row["sample"]: row for row in raw_rows if row.get("sample")}
    print(json.dumps({"event": "raw", "rows": len(raw_rows), "unique": len(by_sample)},
                     ensure_ascii=False), flush=True)

    pilot_ids = [line.strip() for line in
                 (common.P1 / "e01" / "inputs" / "pilot_50_ids.txt").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    pilot_meta = {row["sample_id"].strip(): row for row in
                  common.read_csv_rows(common.DATASETS / "pilot_50.csv")}

    # ---- canonical GT-free model outputs -----------------------------------
    model_rows = []
    for sample in pilot_ids:
        row = by_sample.get(sample)
        if row is None:
            model_rows.append({"sample_id": sample, "run_id": spec["run_id"],
                               "model_id": spec["model_id"], "task": "CSLT",
                               "modality": spec["modality"], "input_kind": spec["input"],
                               "status": "missing_from_run", "prediction": "",
                               "error": "sample not present in script output", "attempt": 1})
            continue
        status = str(row.get("status") or "error")
        record = {
            "sample_id": sample,
            "run_id": spec["run_id"],
            "model_id": spec["model_id"],
            "task": "CSLT",
            "modality": spec["modality"],
            "input_kind": spec["input"],
            "attempt": 1,
            "status": "ok" if status == "ok" else ("empty" if status == "empty" else "error"),
            "prediction": row.get("prediction") or "",
            "error": (str(row["error"])[:1000] if row.get("error") else None),
            "created_at": row.get("created_at"),
        }
        for key in KEEP_KEYS:
            if key in row:
                record[key] = row.get(key)
        for key in TIMING_KEYS:
            if key in row:
                record[key] = row.get(key)
        model_rows.append(record)
    common.write_jsonl(run_dir / "model_outputs.jsonl", model_rows)
    runtime_rows = [{key: row.get(key) for key in ("sample_id", "status", *TIMING_KEYS)}
                    for row in model_rows]
    common.write_jsonl(run_dir / "runtime_metrics.jsonl", runtime_rows, guard=False)
    failures = [row["sample_id"] for row in model_rows if row["status"] != "ok"]
    empties = [row["sample_id"] for row in model_rows if row["status"] == "empty"]

    # ---- canonical evaluation (GT join only here) --------------------------
    references_path = common.REFERENCES / "gt_1086.csv"
    payload, eval_records = evaluator.evaluate(
        run_dir / "model_outputs.jsonl", references_path, task="CSLT",
        bootstrap_iterations=0)
    common.write_json(run_dir / "metrics.json", sanitize_for_json(payload))
    common.write_jsonl(run_dir / "eval_records.jsonl", eval_records, guard=False)
    history = load_history(ROOT / spec["history"])
    history_subset = {sample: history[sample] for sample in pilot_ids if sample in history}
    missing_history = sorted(set(pilot_ids) - set(history_subset))
    print(json.dumps({"event": "evaluated", "successful": payload["successful"],
                      "failed": payload["failed"], "bleu4": payload["metrics"]["bleu4_13a"],
                      "history_available": len(history_subset),
                      "history_missing": missing_history},
                     ensure_ascii=False), flush=True)

    # ---- same-subset historical metrics ------------------------------------
    pairs_history = []
    for sample in pilot_ids:
        ref = None
        for record in eval_records:
            if record["sample_id"] == sample:
                reference_text = record["reference"]
                break
        else:
            reference_text = None
        if reference_text is None:
            continue
        row = history_subset.get(sample)
        if row is None or str(row.get("status")) != "ok":
            continue
        pairs_history.append((reference_text, row.get("prediction") or ""))
    history_metrics = metric_lib.corpus_metrics_cslt(pairs_history) if pairs_history else None
    rerun_metrics = payload["metrics"]
    script_summary_path = raw_dir / "summary.json"
    script_summary = (json.loads(script_summary_path.read_text(encoding="utf-8"))
                      if script_summary_path.is_file() else {})
    script_metrics = script_summary.get("metrics")
    script_paired = script_summary.get("paired_comparison_vs_upstream")

    # ---- per-sample comparison --------------------------------------------
    comparison_rows = []
    raw_equal = nfkc_equal = 0
    for sample in pilot_ids:
        row = by_sample.get(sample) or {}
        hist = history_subset.get(sample) or {}
        rerun_text = row.get("prediction") or ""
        hist_text = hist.get("prediction") or ""
        hist_status = hist.get("status")
        rerun_status = row.get("status")
        if hist_status == "ok" and rerun_status == "ok":
            same_raw = rerun_text == hist_text
            same_nfkc = nfkc(rerun_text) == nfkc(hist_text)
            raw_equal += int(same_raw)
            nfkc_equal += int(same_nfkc)
            category, position = divergence(hist_text, rerun_text)
        else:
            same_raw = same_nfkc = False
            category = "missing_side"
            position = None
        hist_edit = hist.get("char_edit_similarity")
        rerun_edit = None
        for record in eval_records:
            if record["sample_id"] == sample:
                rerun_edit = record.get("char_edit_similarity")
                break
        comparison_rows.append({
            "sample_id": sample,
            "signer_id": (pilot_meta.get(sample) or {}).get("signer_id") or "",
            "cohort": (pilot_meta.get(sample) or {}).get("cohort") or "",
            "history_status": hist_status,
            "rerun_status": rerun_status,
            "history_prediction": hist_text,
            "rerun_prediction": rerun_text,
            "raw_equal": same_raw,
            "nfkc_equal": same_nfkc,
            "history_chars": len(hist_text),
            "rerun_chars": len(rerun_text),
            "history_char_edit_similarity": hist_edit,
            "rerun_char_edit_similarity": rerun_edit,
            "delta_char_edit_similarity": (
                round(float(rerun_edit) - float(hist_edit), 6)
                if (hist_edit is not None and rerun_edit is not None) else None),
            "divergence_class": category,
            "first_diff_index": position,
            "history_e2e_seconds": hist.get("e2e_seconds"),
            "rerun_e2e_seconds": row.get("e2e_seconds"),
            "rerun_pose_ms": row.get("pose_ms"),
            "rerun_model_total_ms": row.get("model_total_ms"),
        })
    columns = ["sample_id", "signer_id", "cohort", "history_status", "rerun_status",
               "history_prediction", "rerun_prediction", "raw_equal", "nfkc_equal",
               "history_chars", "rerun_chars", "history_char_edit_similarity",
               "rerun_char_edit_similarity", "delta_char_edit_similarity",
               "divergence_class", "first_diff_index", "history_e2e_seconds",
               "rerun_e2e_seconds", "rerun_pose_ms", "rerun_model_total_ms"]
    compare_path = common.P1 / "e01" / spec["compare_csv"]
    common.write_csv_rows(compare_path, columns, comparison_rows)

    # ---- manifest ----------------------------------------------------------
    bundle_manifest_path = common.UNISIGN_BUNDLE / "manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    selection_path = common.SHARED_ARTIFACTS / "selection-1086.json"
    dataset = {
        "pilot_50_csv": str(common.DATASETS / "pilot_50.csv"),
        "pilot_50_csv_sha256": common.sha256_file(common.DATASETS / "pilot_50.csv"),
        "pilot_ids": str(common.P1 / "e01" / "inputs" / "pilot_50_ids.txt"),
        "pilot_ids_sha256": common.sha256_file(common.P1 / "e01" / "inputs" / "pilot_50_ids.txt"),
        "sample_count": len(pilot_ids),
        "signers": sorted({(pilot_meta.get(row["sample_id"]) or {}).get("signer_id")
                           for row in comparison_rows} - {None, ""}),
        "reference_table": str(references_path),
        "reference_table_sha256": common.sha256_file(references_path),
        "reference_table_role": "EVAL ONLY (not fed to the model)",
    }
    if args.kind == "video":
        dataset.update({
            "artifact_dir": str(common.SHARED_ARTIFACTS),
            "selection_1086": str(selection_path),
            "selection_1086_sha256": common.sha256_file(selection_path),
            "trimmed_clip_sha256_aggregate": aggregate_hash(
                {row["sample"]: row["clip_sha256"] for row in raw_rows if row.get("clip_sha256")}),
            "preprocessed_input": "trimmed/<sample>.mp4 from the reviewed 1086 artifacts",
        })
    else:
        pose_dir = common.UNISIGN_POSE_DIR
        pose_digests = {}
        for sample in pilot_ids:
            path = pose_dir / f"{sample}.pkl"
            if path.is_file():
                pose_digests[sample] = common.sha256_file(path)
        dataset.update({
            "pose_dir": str(pose_dir),
            "pose_dir_file_count": len(list(pose_dir.glob("*.pkl"))),
            "pose_pkl_sha256_aggregate": aggregate_hash(pose_digests),
            "pose_pkls_scanned": len(pose_digests),
            "filtered_refs": str(run_dir / "inputs" / "refs-pilot50.txt"),
            "filtered_refs_sha256": common.sha256_file(run_dir / "inputs" / "refs-pilot50.txt"),
            "upstream_snapshot_pres": str(run_dir / "inputs" / "pres-pilot50.txt"),
            "upstream_snapshot_pres_sha256": common.sha256_file(run_dir / "inputs" / "pres-pilot50.txt"),
        })
    latency = {}
    for key in ("e2e_seconds", "pose_ms", "model_total_ms", "model_forward_ms", "generation_ms"):
        values = [row[key] for row in model_rows if isinstance(row.get(key), (int, float))]
        if values:
            latency[key] = {
                "n": len(values),
                "mean": round(sum(values) / len(values), 3),
                "p50": round(common.percentile(values, 50), 3),
                "p95": round(common.percentile(values, 95), 3),
            }
    manifest = {
        "run_id": spec["run_id"],
        "experiment_id": "E01",
        "run_status": "PASS" if (len(by_sample) == len(pilot_ids) and not failures) else "FAIL",
        "path": args.kind,
        "purpose": f"E01 {args.kind} reproduction on Pilot-50",
        "created_at": common.utc_now(),
        "operator": "reproducer",
        "code": common.git_state(),
        "host": {"host_alias": common.host_alias(), **common.environment_fingerprint()},
        "model": {
            "model_id": spec["model_id"],
            "task": "CSLT (CSL-Daily pose-only sign-language translation)",
            "checkpoint": str(common.UNISIGN_CHECKPOINT),
            "checkpoint_bytes": common.UNISIGN_CHECKPOINT.stat().st_size,
            "checkpoint_sha256": common.sha256_file(common.UNISIGN_CHECKPOINT),
            "bundle_manifest": str(bundle_manifest_path),
            "bundle_manifest_sha256": common.sha256_file(bundle_manifest_path),
            "mt5_model_file_sha256": bundle_manifest.get("mt5_model_file_sha256"),
            "source_commit": bundle_manifest.get("source_commit"),
            "precision": "bfloat16 (cuda)",
            "decode": {"num_beams": 4, "max_new_tokens": 100, "max_frames": 256},
            "license": bundle_manifest.get("license"),
        },
        "input_protocol": {
            "kind": args.kind,
            "description": spec["input"],
            "pose_normalization": "isotropic_short_side",
            "pose_orientation": "preserve",
            "frame_sampling": "deterministic linspace when T>256 (all pilot clips are <=256 frames)",
        },
        "dataset_snapshot": dataset,
        "history_reference": {
            "label": spec["history_label"],
            "path": spec["history"],
            "samples_available_in_snapshot": len(history_subset),
            "samples_missing_from_snapshot": missing_history,
            "role": "historical 1086 imported predictions; compared sample-by-sample, not re-inferred",
        },
        "command": spec["command"],
        "cwd": str(ROOT),
        "outputs": {
            "model_outputs": str(run_dir / "model_outputs.jsonl"),
            "runtime_metrics": str(run_dir / "runtime_metrics.jsonl"),
            "metrics": str(run_dir / "metrics.json"),
            "eval_records": str(run_dir / "eval_records.jsonl"),
            "predictions_raw_dir": str(raw_dir),
            "predictions_raw_note": "frozen evaluation script output; embeds the reference text, "
                                    "so it is kept in predictions-raw/ and never merged into model_outputs.jsonl",
            "comparison_csv": str(compare_path),
            "comparison": str(run_dir / "comparison.json"),
        },
        "sample_counts": {
            "planned": len(pilot_ids),
            "rows_written": len(by_sample),
            "successful": payload["successful"],
            "failed": len(failures),
            "empty_outputs": len(empties),
        },
        "failure_samples": [{"sample_id": row["sample_id"], "status": row["status"],
                             "error": row.get("error")}
                            for row in model_rows if row["status"] != "ok"],
        "gt_policy": {
            "model_outputs_gt_free": True,
            "gt_used_only_in": ["metrics.json", "eval_records.jsonl", "comparison metrics"],
            "guarded_writer": "experiments.p1.common.write_jsonl(assert_gt_clean)",
        },
        "privacy_and_licensing": {
            "rights_checked": False,
            "no_raw_media_export": True,
            "no_credentials_in_logs": True,
        },
        "notes": [
            "single process, exclusive GPU (other P-1 GPU work was blocked for this run)",
            "historical artifacts were only read; nothing under artifacts/ was modified",
        ],
    }
    common.write_json(run_dir / "manifest.json", manifest)

    comparison = {
        "kind": args.kind,
        "run_id": spec["run_id"],
        "history_label": spec["history_label"],
        "n_samples": len(pilot_ids),
        "n_history_available": len(history_subset),
        "history_missing": missing_history,
        "rerun_successful": payload["successful"],
        "rerun_failed": payload["failed"],
        "rerun_empty": len(empties),
        "match_rates": {
            "raw_exact": raw_equal,
            "nfkc_normalized": nfkc_equal,
            "denominator": len(pilot_ids),
            "raw_rate": round(raw_equal / len(pilot_ids), 4),
            "nfkc_rate": round(nfkc_equal / len(pilot_ids), 4),
        },
        "divergence_counts": {
            category: sum(1 for row in comparison_rows if row["divergence_class"] == category)
            for category in sorted({row["divergence_class"] for row in comparison_rows})
        },
        "canonical_metrics_rerun_vs_gt1086": {
            key: rerun_metrics.get(key) for key in
            ("n", "bleu4_13a", "chrf", "rouge_l_f1_char", "mean_char_edit_similarity",
             "exact_match_count")
        },
        "canonical_metrics_history_vs_gt1086": {
            key: history_metrics.get(key) for key in
            ("n", "bleu4_13a", "chrf", "rouge_l_f1_char", "mean_char_edit_similarity",
             "exact_match_count")
        } if history_metrics else None,
        "latency_rerun": latency,
        "evaluation_scope": "Pilot-50 subset of the 1,086 local CSL-Daily test clips, retrospective "
                            "diagnostic; canonical evaluator, upstream reference text "
                            "(gt_1086.ref_sentence_upstream)",
        "metric_reference_variants": {
            "canonical_primary": {
                "reference": "experiments/p1/references/gt_1086.csv :: ref_sentence_upstream",
                "style": "upstream test_tmp_refs.txt: character-separated, half-width punctuation",
                "role": "cross-run comparison + historical 1086 metric comparability",
            },
            "frozen_script_secondary": {
                "reference": "selection-1086.json :: reference (official csl-daily.test text)",
                "style": "official annotation text: no spaces, full-width punctuation",
                "role": "the frozen script's own summary metric; NOT comparable to the historical "
                        "'paperrefs' numbers (known ~2.8 BLEU reference-variant gap)",
            },
            "note": "the two reference variants are reported separately on purpose; never mix them",
        },
        "script_metrics_secondary_reference": script_metrics,
        "script_paired_comparison_vs_upstream": script_paired,
        "prediction_diff_samples": [
            {"sample_id": row["sample_id"],
             "history": row["history_prediction"],
             "rerun": row["rerun_prediction"],
             "divergence_class": row["divergence_class"],
             "first_diff_index": row["first_diff_index"]}
            for row in comparison_rows if not row["nfkc_equal"]
        ],
    }
    common.write_json(run_dir / "comparison.json", comparison)
    print(json.dumps({"event": "finalized", "run_id": spec["run_id"],
                      "match": comparison["match_rates"],
                      "metrics": comparison["canonical_metrics_rerun_vs_gt1086"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
