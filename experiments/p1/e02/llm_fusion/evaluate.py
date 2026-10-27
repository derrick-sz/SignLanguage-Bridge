"""Evaluation side for the G2 phase-2A LLM fusion ablation (EVAL ONLY).

This is the **only** module in `experiments/p1/e02/llm_fusion/` that reads the
ground-truth reference table `experiments/p1/references/gt_1086.csv`.  It joins
GT by ``sample_id`` to the GT-free `fusion_outputs.jsonl`, computes canonical
``p1-v1`` metrics (`experiments/p1/eval/metrics.py`) and signer-clustered paired
bootstrap deltas against the A baseline.

Outputs per run:
    runs/<run_id>/eval_records.jsonl   EVAL ONLY (contains GT-derived values; guard=False)
    runs/<run_id>/metrics.json

Usage:
    python -m experiments.p1.e02.llm_fusion.evaluate --run-id <run_id> [--iterations 1000]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.p1 import common
from experiments.p1.eval import metrics as M

ROOT = common.ROOT
FUSION_DIR = common.P1 / "e02" / "llm_fusion"
RUNS_DIR = FUSION_DIR / "runs"
GT_PATH = common.REFERENCES / "gt_1086.csv"
ABLATION_CSV = FUSION_DIR / "fusion_ablation_g2.csv"
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20261026

GROUP_DESCRIPTIONS = {
    "A": "Uni-Sign 原始译文（基准，不调 LLM）",
    "B": "SLRT -> G2T/SLT（BLOCKED：外部权重缺失；B0 只是确定性 join 退化基线）",
    "B0": "SLRT Gloss 确定性 join（无 G2T；词列，不成句）",
    "C": "Uni-Sign -> LLM（润色对照）",
    "D": "Uni-Sign + SLRT Gloss -> LLM",
    "E": "双模型 + 证据约束（槽位冲突提示 + 未校准 SLRT 分数）-> LLM",
    "F": "SLRT Gloss -> LLM 造句（v3_f：只给词汇序列，不给 Uni-Sign 译文）",
}

# GT-free candidate texts (A/A2/B0) may be loaded here when a run does not
# contain the corresponding group rows (e.g. group-F-only runs need A/B0 baselines
# for paired comparisons). Candidate files are model outputs, never GT.
E02_CANDIDATES = common.P1 / "e02" / "candidate_outputs.jsonl"

ABLATION_COLUMNS = [
    "run_id", "data_status", "prompt_version", "backend", "group", "description", "n",
    "bleu4_13a", "chrf", "mean_char_edit_similarity", "exact_match_count",
    "delta_bleu_vs_A", "delta_bleu_ci_lower", "delta_bleu_ci_upper",
    "delta_edit_vs_A", "delta_edit_ci_lower", "delta_edit_ci_upper",
    "backend_latency_p50_ms", "backend_latency_p95_ms",
    "tokens_prompt_sum", "tokens_completion_sum", "tokens_total_sum",
    "retries_total", "failed_calls", "note",
]


def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def load_candidate_texts(sample_ids: set[str] | None = None) -> dict[str, dict[str, str]]:
    """Load GT-free A/A2/B0 candidate model outputs for baseline injection.

    Group-F-only runs do not contain A/B0 rows (A has no LLM call; B0 is not a
    pipeline group), but paired comparisons need the same-sample A and B0 texts.
    These are model outputs from ``candidate_outputs.jsonl``, never GT.
    """

    by_sample: dict[str, dict[str, str]] = defaultdict(dict)
    for row in common.read_jsonl(E02_CANDIDATES):
        sample = row.get("sample_id")
        if sample_ids is not None and sample not in sample_ids:
            continue
        text = (row.get("output") or {}).get("text")
        by_sample[sample][row["candidate_id"]] = "" if text is None else str(text)
    return by_sample


def _pseudo_group_rows(group: str, texts: dict[str, str], samples: set[str]) -> list[dict]:
    rows: list[dict] = []
    for sample in sorted(samples):
        rows.append(
            {
                "run_id": None,
                "sample_id": sample,
                "group": group,
                "status": "ok" if texts[sample].strip() else "error",
                "output_text": texts[sample],
                "latency_ms": None,
                "tokens": {"in": None, "out": None, "total": None},
                "retries": None,
                "backend": "candidate_outputs (no LLM)",
            }
        )
    return rows


def edit_similarity(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def char_sep(text: str) -> str:
    return M.character_separated(text)


def _bleu(items: list[dict], hypothesis_key: str) -> float:
    import sacrebleu

    hyps = [item[f"{hypothesis_key}_char"] for item in items]
    refs = [item["reference_char"] for item in items]
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score


def paired_bleu_delta(items: list[dict]) -> float:
    return _bleu(items, "hyp_g") - _bleu(items, "hyp_a")


def paired_edit_delta(items: list[dict]) -> float:
    return statistics.fmean(item["sim_g"] for item in items) - statistics.fmean(item["sim_a"] for item in items)


def paired_group_bleu_delta(items: list[dict]) -> float:
    import sacrebleu

    refs = [item["reference_char"] for item in items]
    hyps_second = [item["hyp_second_char"] for item in items]
    hyps_first = [item["hyp_first_char"] for item in items]
    return (sacrebleu.corpus_bleu(hyps_second, [refs], tokenize="13a").score
            - sacrebleu.corpus_bleu(hyps_first, [refs], tokenize="13a").score)


def paired_group_edit_delta(items: list[dict]) -> float:
    return statistics.fmean(item["sim_second"] for item in items) - statistics.fmean(item["sim_first"] for item in items)


def evaluate_run(
    run_id: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    write: bool = True,
) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    outputs_path = run_dir / "fusion_outputs.jsonl"
    if not outputs_path.exists():
        raise SystemExit(f"fusion_outputs.jsonl not found in {run_dir}")
    manifest = common.read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    rows = common.read_jsonl(outputs_path)
    gt = load_gt()

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    # ---- baseline injection (GT-free candidate outputs, never overrides run rows) ----
    # Only group-F runs need injected baselines: F has no A rows (no LLM call for A)
    # and no B0 group rows. Legacy A/C/D/E runs keep their exact previous behaviour so
    # the existing self-test contract (eval_records == n_rows) stays valid.
    baseline_sources: dict[str, dict[str, Any]] = {}
    baseline_groups = ("A", "B0") if "F" in by_group else ("A",)
    run_samples = {row["sample_id"] for group_rows in by_group.values() for row in group_rows}
    candidate_texts = load_candidate_texts(run_samples)
    for baseline_group in baseline_groups:
        if baseline_group in by_group:
            baseline_sources[baseline_group] = {"source": "fusion_outputs rows", "injected": False}
            continue
        texts = {sample: mapping.get(baseline_group, "") for sample, mapping in candidate_texts.items()}
        samples = {sample for sample, text in texts.items() if sample in run_samples and text.strip()}
        if not samples:
            continue
        by_group[baseline_group] = _pseudo_group_rows(baseline_group, texts, samples)
        baseline_sources[baseline_group] = {
            "source": str(E02_CANDIDATES.relative_to(ROOT)),
            "file_sha256": common.sha256_file(E02_CANDIDATES),
            "injected": True,
            "note": "GT-free model output candidate; not an LLM call",
        }

    a_rows = {row["sample_id"]: row for row in by_group.get("A", [])}
    if not a_rows:
        raise SystemExit("A baseline rows missing; cannot compute paired deltas")

    group_metrics: dict[str, dict[str, Any]] = {}
    ok_rows_by_group: dict[str, list[dict]] = {}
    eval_records: list[dict] = []
    for group in sorted(by_group):
        group_rows = [row for row in by_group[group] if row.get("status") == "ok" and row.get("output_text", "").strip()]
        ok_rows_by_group[group] = group_rows
        pairs = [(gt[row["sample_id"]]["ref_sentence_upstream"], row["output_text"])
                 for row in group_rows if row["sample_id"] in gt]
        corpus = M.corpus_metrics_cslt(pairs) if pairs else {}
        # paired comparison vs A (same sample must be present in both groups)
        paired_records: list[dict] = []
        for row in group_rows:
            sample = row["sample_id"]
            if sample not in gt or sample not in a_rows:
                continue
            reference = gt[sample]["ref_sentence_upstream"]
            hypothesis_a = a_rows[sample].get("output_text", "")
            paired_records.append(
                {
                    "sample_id": sample,
                    "signer_id": gt[sample]["signer_id"],
                    "reference": reference,
                    "reference_char": char_sep(reference),
                    "hypothesis_a": hypothesis_a,
                    "hypothesis_g": row["output_text"],
                    "hyp_a_char": char_sep(hypothesis_a),
                    "hyp_g_char": char_sep(row["output_text"]),
                    "sim_a": edit_similarity(reference, hypothesis_a),
                    "sim_g": edit_similarity(reference, row["output_text"]),
                }
            )
            eval_records.append(
                {
                    "run_id": run_id,
                    "group": group,
                    "sample_id": sample,
                    "signer_id": gt[sample]["signer_id"],
                    "reference": reference,
                    "reference_official": gt[sample]["gt_sentence_official"],
                    "hypothesis_a": hypothesis_a,
                    "hypothesis_g": row["output_text"],
                    "sim_a": edit_similarity(reference, hypothesis_a),
                    "sim_g": edit_similarity(reference, row["output_text"]),
                    "output_status": row.get("status"),
                    "backend": row.get("backend"),
                }
            )
        delta_bleu = delta_edit = None
        if paired_records:
            if iterations > 0:
                delta_bleu = M.bootstrap_by_group(
                    paired_records, paired_bleu_delta, group_key="signer_id",
                    iterations=iterations, seed=seed,
                )
                delta_edit = M.bootstrap_by_group(
                    paired_records, paired_edit_delta, group_key="signer_id",
                    iterations=iterations, seed=seed,
                )
            else:
                delta_bleu = {"point": round(paired_bleu_delta(paired_records), 4),
                              "lower": None, "upper": None, "iterations": 0,
                              "method": "point_only_no_bootstrap"}
                delta_edit = {"point": round(paired_edit_delta(paired_records), 4),
                              "lower": None, "upper": None, "iterations": 0,
                              "method": "point_only_no_bootstrap"}
        latencies = [row["latency_ms"] for row in group_rows if isinstance(row.get("latency_ms"), (int, float))]
        tokens_in = [row["tokens"]["in"] for row in group_rows
                     if isinstance(row.get("tokens"), dict) and isinstance(row["tokens"].get("in"), int)]
        tokens_out = [row["tokens"]["out"] for row in group_rows
                      if isinstance(row.get("tokens"), dict) and isinstance(row["tokens"].get("out"), int)]
        tokens_total = [row["tokens"]["total"] for row in group_rows
                        if isinstance(row.get("tokens"), dict) and isinstance(row["tokens"].get("total"), int)]
        retries = [row.get("retries") for row in group_rows if isinstance(row.get("retries"), int)]
        failed_calls = sum(1 for row in by_group[group] if row.get("status") != "ok")
        group_metrics[group] = {
            "group": group,
            "description": GROUP_DESCRIPTIONS.get(group, ""),
            "n_rows": len(by_group[group]),
            "n_ok": len(group_rows),
            "metrics": corpus,
            "delta_bleu_vs_A": delta_bleu,
            "delta_edit_vs_A": delta_edit,
            "backend_latency_ms": {
                "p50": round(statistics.median(latencies), 3) if latencies else None,
                "p95": round(common.percentile(latencies, 95), 3) if latencies else None,
                "mean": round(statistics.fmean(latencies), 3) if latencies else None,
                "n": len(latencies),
            },
            "tokens": {
                "prompt_sum": sum(tokens_in) if tokens_in else None,
                "completion_sum": sum(tokens_out) if tokens_out else None,
                "total_sum": sum(tokens_total) if tokens_total else None,
                "n_calls_with_usage": len(tokens_out),
            },
            "tokens_out_sum": sum(tokens_out) if tokens_out else None,
            "retries_total": sum(retries) if retries else 0,
            "failed_calls": failed_calls,
            "calls": len(by_group[group]),
        }

    # ---- paired group comparisons (same samples; signer-clustered bootstrap) ----
    group_rows_by_sample: dict[str, dict[str, dict]] = {
        group: {row["sample_id"]: row for row in group_rows}
        for group, group_rows in ok_rows_by_group.items()
    }
    paired_group: dict[str, dict[str, Any]] = {}
    for first, second in (
        ("A", "C"), ("A", "D"), ("A", "E"), ("C", "D"), ("D", "E"), ("C", "E"),
        ("A", "F"), ("B0", "F"), ("A", "B0"),
    ):
        if first not in group_rows_by_sample or second not in group_rows_by_sample:
            continue
        common_samples = sorted(set(group_rows_by_sample[first]) & set(group_rows_by_sample[second]))
        records: list[dict] = []
        for sample in common_samples:
            if sample not in gt:
                continue
            reference = gt[sample]["ref_sentence_upstream"]
            hyp_first = group_rows_by_sample[first][sample].get("output_text", "")
            hyp_second = group_rows_by_sample[second][sample].get("output_text", "")
            records.append(
                {
                    "sample_id": sample,
                    "signer_id": gt[sample]["signer_id"],
                    "reference": reference,
                    "reference_char": char_sep(reference),
                    "hyp_first": hyp_first,
                    "hyp_second": hyp_second,
                    "hyp_first_char": char_sep(hyp_first),
                    "hyp_second_char": char_sep(hyp_second),
                    "sim_first": edit_similarity(reference, hyp_first),
                    "sim_second": edit_similarity(reference, hyp_second),
                }
            )
        if not records:
            continue
        if iterations > 0:
            delta_bleu_pair = M.bootstrap_by_group(
                records, paired_group_bleu_delta, group_key="signer_id",
                iterations=iterations, seed=seed,
            )
            delta_edit_pair = M.bootstrap_by_group(
                records, paired_group_edit_delta, group_key="signer_id",
                iterations=iterations, seed=seed,
            )
        else:
            delta_bleu_pair = {"point": round(paired_group_bleu_delta(records), 4),
                               "lower": None, "upper": None, "method": "point_only_no_bootstrap"}
            delta_edit_pair = {"point": round(paired_group_edit_delta(records), 4),
                               "lower": None, "upper": None, "method": "point_only_no_bootstrap"}
        paired_group[f"{first}_vs_{second}"] = {
            "first": first,
            "second": second,
            "n_pairs": len(records),
            "delta_bleu_second_minus_first": delta_bleu_pair,
            "delta_edit_second_minus_first": delta_edit_pair,
            "interpretation": f"{second} minus {first} on the same samples (positive = {second} better)",
        }

    metrics_payload = {
        "experiment": "G2-2A-LLM-FUSION",
        "run_id": run_id,
        "status": manifest.get("status", "UNKNOWN"),
        "data_status": manifest.get("data_status", "UNKNOWN"),
        "prompt_version": (manifest.get("prompt") or {}).get("version"),
        "prompt_sha256": ((manifest.get("prompt") or {}).get("files") or {}).get("json_sha256"),
        "prompt_md_sha256": ((manifest.get("prompt") or {}).get("files") or {}).get("md_sha256"),
        "backend": (manifest.get("backend") or {}).get("name", "unknown"),
        "provider": (manifest.get("llm_endpoint") or {}).get("provider"),
        "model": (manifest.get("llm_endpoint") or {}).get("model"),
        "model_version": (manifest.get("model") or {}).get("model_version"),
        "decoding": (manifest.get("model") or {}).get("decoding"),
        "labels": {
            "stub_backend": (manifest.get("backend") or {}).get("is_stub", False),
            "warning": "STUB_ONLY numbers validate the pipeline; they are NOT LLM model results"
            if (manifest.get("backend") or {}).get("is_stub") else "real backend",
        },
        "n_samples": manifest.get("sampling", {}).get("n_samples"),
        "groups": group_metrics,
        "baseline_sources": baseline_sources,
        "paired_group_comparisons": paired_group,
        "llm_endpoint": manifest.get("llm_endpoint"),
        "security": manifest.get("security"),
        "gt_join": {
            "path": str(GT_PATH.relative_to(ROOT)),
            "sha256": common.sha256_file(GT_PATH),
            "eval_only": True,
            "reference_field": "gt_1086.csv ref_sentence_upstream",
        },
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "method": "cluster_bootstrap_by_signer (paired: group vs A on the same samples)",
            "metric": "corpus BLEU-4 delta and mean char-edit-similarity delta",
        },
        "created_at": common.utc_now(),
    }
    if write:
        run_dir = RUNS_DIR / run_id
        common.write_jsonl(run_dir / "eval_records.jsonl", eval_records, guard=False)
        common.write_json(run_dir / "metrics.json", metrics_payload)
    return {"run_id": run_id, "manifest": manifest, "metrics": metrics_payload}


def collect_run_metrics(run_id: str) -> dict[str, Any] | None:
    path = RUNS_DIR / run_id / "metrics.json"
    if not path.exists():
        return None
    return common.read_json(path)


def write_ablation_csv(run_ids: list[str], *, path: Path = ABLATION_CSV) -> Path:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        payload = collect_run_metrics(run_id)
        if payload is None:
            continue
        for group, entry in sorted(payload["groups"].items()):
            corpus = entry.get("metrics") or {}
            delta_bleu = entry.get("delta_bleu_vs_A") or {}
            delta_edit = entry.get("delta_edit_vs_A") or {}
            prompt_version = payload.get("prompt_version")
            if not prompt_version and (RUNS_DIR / run_id / "manifest.json").exists():
                prompt_version = (common.read_json(RUNS_DIR / run_id / "manifest.json").get("prompt") or {}).get("version")
            rows.append(
                {
                    "run_id": run_id,
                    "data_status": payload.get("data_status"),
                    "prompt_version": prompt_version or "unknown",
                    "backend": payload.get("backend"),
                    "group": group,
                    "description": entry.get("description"),
                    "n": entry.get("n_ok"),
                    "bleu4_13a": corpus.get("bleu4_13a"),
                    "chrf": corpus.get("chrf"),
                    "mean_char_edit_similarity": corpus.get("mean_char_edit_similarity"),
                    "exact_match_count": corpus.get("exact_match_count"),
                    "delta_bleu_vs_A": delta_bleu.get("point"),
                    "delta_bleu_ci_lower": delta_bleu.get("lower"),
                    "delta_bleu_ci_upper": delta_bleu.get("upper"),
                    "delta_edit_vs_A": delta_edit.get("point"),
                    "delta_edit_ci_lower": delta_edit.get("lower"),
                    "delta_edit_ci_upper": delta_edit.get("upper"),
                    "backend_latency_p50_ms": (entry.get("backend_latency_ms") or {}).get("p50"),
                    "backend_latency_p95_ms": (entry.get("backend_latency_ms") or {}).get("p95"),
                    "tokens_prompt_sum": (entry.get("tokens") or {}).get("prompt_sum"),
                    "tokens_completion_sum": (entry.get("tokens") or {}).get("completion_sum"),
                    "tokens_total_sum": (entry.get("tokens") or {}).get("total_sum"),
                    "retries_total": entry.get("retries_total"),
                    "failed_calls": entry.get("failed_calls"),
                    "note": "STUB_ONLY pipeline validation; not an LLM result"
                    if payload.get("data_status", "").startswith("STUB_ONLY")
                    else ("REAL LLM run (exploratory); not final G2 evidence"
                          if payload.get("data_status", "").startswith("REAL")
                          else "real LLM"),
                }
            )
    # B (G2T) stays BLOCKED; record it explicitly so the ablation table is complete.
    rows.append(
        {
            "run_id": ",".join(run_ids),
            "data_status": "BLOCKED",
            "prompt_version": "n/a",
            "backend": "none",
            "group": "B",
            "description": GROUP_DESCRIPTIONS["B"],
            "n": None,
            "note": "BLOCKED: no CSL-Daily G2T/SLT weights (author did not release); B0 join is not a translation baseline",
        }
    )
    common.write_csv_rows(path, ABLATION_COLUMNS, rows)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a G2-2A fusion ablation run (EVAL ONLY)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS,
                        help="bootstrap iterations; 0 = point estimate only")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--ablation-csv", type=Path, default=ABLATION_CSV)
    args = parser.parse_args(argv)
    result = evaluate_run(args.run_id, iterations=args.iterations, seed=args.seed)
    write_ablation_csv([args.run_id], path=args.ablation_csv)
    print(json.dumps({
        "run_id": args.run_id,
        "data_status": result["metrics"]["data_status"],
        "backend": result["metrics"]["backend"],
        "groups": {
            group: {
                "n": entry["n_ok"],
                "bleu4_13a": (entry.get("metrics") or {}).get("bleu4_13a"),
                "delta_bleu_vs_A": (entry.get("delta_bleu_vs_A") or {}).get("point"),
            }
            for group, entry in result["metrics"]["groups"].items()
        },
        "ablation_csv": str(args.ablation_csv),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
