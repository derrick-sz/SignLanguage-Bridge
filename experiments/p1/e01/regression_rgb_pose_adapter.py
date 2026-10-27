"""RGB+pose adapter regression comparator (task-33).

Read-only comparison tool: it never runs a model and never edits the frozen
baseline.  The adapter run itself is produced elsewhere (normal protocol path
through ModelManager / load_model_adapter); this tool checks:

  1. per-sample exact text equality against the frozen predictions
     (task-33 requirement: Pilot-50 must be 50/50 character-identical);
  2. canonical p1-v1 metrics against the frozen metrics
     (BLEU-4 26.3026 / chrF 23.0005 / char edit 0.4358 for 1086, tolerance <= 0.01);
  3. a diff list with character-level evidence + a proposed cause category.

GT is read only here, for the metric recomputation (EVAL ONLY).  Adapter
predictions are GT-free by contract.

Usage:
    # pilot exact-match check (50/50 required)
    python -m experiments.p1.e01.regression_rgb_pose_adapter compare \
        --frozen experiments/p1/e01/runs/unisign-rgb-pose-pilot50-20261027/predictions_rgb_pose.jsonl \
        --candidate <adapter_run_dir>/predictions_rgb_pose.jsonl \
        --label pilot50 --out-json <adapter_run_dir>/regression_pilot50.json

    # 1086 exact-match + metric tolerance check
    python -m experiments.p1.e01.regression_rgb_pose_adapter compare \
        --frozen experiments/p1/e01/runs/unisign-rgb-pose-1086-20261027/predictions_rgb_pose.jsonl \
        --candidate <adapter_run_dir>/predictions_rgb_pose.jsonl \
        --frozen-metrics experiments/p1/e01/runs/unisign-rgb-pose-1086-20261027/metrics.json \
        --label full1086 --out-json <adapter_run_dir>/regression_1086.json

Exit code: 0 = PASS, 1 = FAIL (differences / metric drift).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1 import common
from experiments.p1.eval import metrics as M

GT_PATH = common.REFERENCES / "gt_1086.csv"
FROZEN_PILOT = common.P1 / "e01" / "runs" / "unisign-rgb-pose-pilot50-20261027" / "predictions_rgb_pose.jsonl"
FROZEN_1086 = common.P1 / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "predictions_rgb_pose.jsonl"
FROZEN_1086_METRICS = common.P1 / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "metrics.json"
FROZEN_PILOT_METRICS = common.P1 / "e01" / "runs" / "unisign-rgb-pose-pilot50-20261027" / "metrics.json"
FROZEN_1086_PROTOCOL = common.P1 / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "protocol.json"
BASELINE_METRICS = {"bleu4_13a": 26.3026, "chrf": 23.0005, "mean_char_edit_similarity": 0.4358}
METRIC_TOLERANCE = 0.01


def load_predictions(path: Path) -> dict[str, dict]:
    rows = [row for row in common.read_jsonl(path)]
    by_sample: dict[str, dict] = {}
    duplicates = []
    for row in rows:
        sample = row.get("sample_id")
        if sample in by_sample:
            duplicates.append(sample)
        by_sample[sample] = row
    if duplicates:
        raise SystemExit(f"duplicate sample ids in {path}: {duplicates[:10]}")
    return by_sample


def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def edit_similarity(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def corpus_metrics(texts: dict[str, str], gt: dict[str, dict]) -> dict:
    pairs = [(gt[sample]["ref_sentence_upstream"], text)
             for sample, text in sorted(texts.items()) if sample in gt and text.strip()]
    return M.corpus_metrics_cslt(pairs) if pairs else {}


def char_diff(frozen: str, candidate: str) -> dict:
    """Compact character-level evidence for one differing sample."""

    import difflib

    operations = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, frozen, candidate).get_opcodes():
        if tag == "equal":
            continue
        operations.append({"op": tag, "frozen": frozen[i1:i2], "candidate": candidate[j1:j2]})
    return {
        "len_frozen": len(frozen),
        "len_candidate": len(candidate),
        "ops": operations[:12],
        "ops_total": len(operations),
    }


def classify_diff(frozen: str, candidate: str) -> str:
    """Best-effort cause category; attribution is finalized from adapter logs."""

    if not candidate.strip():
        return "candidate_empty"
    if frozen.strip() == candidate.strip():
        return "whitespace_only"
    if frozen.replace(" ", "") == candidate.replace(" ", ""):
        return "space_or_tokenizer_join"
    if frozen.replace(" ", "").replace(",", "，").replace("?", "？") == candidate.replace(" ", "").replace(",", "，").replace("?", "？"):
        return "punctuation_normalization"
    if len(frozen) == len(candidate):
        return "same_length_char_difference"
    return "different_text_length"


def compare_pair(frozen_path: Path, candidate_path: Path, *, frozen_metrics_path: Path | None,
                 label: str, candidate_run_dir: Path | None = None) -> dict:
    frozen_path = Path(frozen_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    frozen_metrics_path = Path(frozen_metrics_path).resolve() if frozen_metrics_path else None
    frozen = load_predictions(frozen_path)
    candidate = load_predictions(candidate_path)
    frozen_ids = set(frozen)
    candidate_ids = set(candidate)
    missing = sorted(frozen_ids - candidate_ids)
    extra = sorted(candidate_ids - frozen_ids)
    common_ids = sorted(frozen_ids & candidate_ids)

    diffs = []
    exact = 0
    for sample in common_ids:
        frozen_text = str((frozen[sample].get("text") or ""))
        candidate_text = str((candidate[sample].get("text") or ""))
        if frozen_text == candidate_text:
            exact += 1
            continue
        diffs.append({
            "sample_id": sample,
            "cause_category": classify_diff(frozen_text, candidate_text),
            "frozen_text": frozen_text,
            "candidate_text": candidate_text,
            "char_diff": char_diff(frozen_text, candidate_text),
            "frozen_status": frozen[sample].get("status"),
            "candidate_status": candidate[sample].get("status"),
        })

    frozen_texts = {sample: str(frozen[sample].get("text") or "") for sample in common_ids}
    candidate_texts = {sample: str(candidate[sample].get("text") or "") for sample in common_ids}
    gt = load_gt()
    metrics_candidate = corpus_metrics(candidate_texts, gt)
    metrics_frozen = corpus_metrics(frozen_texts, gt)

    metric_comparison = {}
    if frozen_metrics_path is None:
        frozen_metrics_path = FROZEN_PILOT_METRICS if label.startswith("pilot") else FROZEN_1086_METRICS
    if frozen_metrics_path is not None and frozen_metrics_path.exists():
        frozen_payload = common.read_json(frozen_metrics_path)
        expected = ((frozen_payload.get("systems") or {}).get("rgb_pose") or {}).get("metrics") or {}
    else:
        expected = BASELINE_METRICS
    for key in ("bleu4_13a", "chrf", "mean_char_edit_similarity"):
        value = metrics_candidate.get(key)
        expected_value = expected.get(key, BASELINE_METRICS.get(key))
        delta = None if value is None or expected_value is None else round(value - expected_value, 4)
        metric_comparison[key] = {
            "baseline": expected_value,
            "adapter_run": value,
            "delta": delta,
            "within_tolerance": (delta is not None and abs(delta) <= METRIC_TOLERANCE),
        }

    exact_rate = round(exact / len(common_ids), 4) if common_ids else None
    expected_total = len(frozen_ids)
    exact_required = label.startswith("pilot")  # pilot: exact character equality required
    pass_exact = (exact == expected_total and not missing and not extra)
    pass_metrics = all(item["within_tolerance"] for item in metric_comparison.values())
    if exact_required:
        # Pilot-50 acceptance: 50/50 character-identical predictions (metrics are evidence only).
        status = "PASS" if pass_exact else "FAIL"
    else:
        # 1086 acceptance: canonical metrics within 0.01; exact matches are reported as extra evidence.
        status = "PASS" if (pass_metrics and not missing) else "FAIL"
    result = {
        "label": label,
        "status": status,
        "frozen_predictions": str(frozen_path.relative_to(ROOT)),
        "frozen_predictions_sha256": common.sha256_file(frozen_path),
        "candidate_predictions": str(candidate_path),
        "candidate_predictions_sha256": common.sha256_file(candidate_path) if candidate_path.exists() else None,
        "candidate_run_dir": str(candidate_run_dir) if candidate_run_dir else None,
        "n_frozen": len(frozen_ids),
        "n_candidate": len(candidate_ids),
        "n_common": len(common_ids),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing_sample_ids": missing[:20],
        "extra_sample_ids": extra[:20],
        "exact_match": exact,
        "exact_match_rate": exact_rate,
        "exact_required": exact_required,
        "metric_comparison": metric_comparison,
        "metrics_candidate_recomputed": metrics_candidate,
        "metrics_frozen_recomputed": metrics_frozen,
        "differences": diffs,
        "gt_eval_only": {"path": str(GT_PATH.relative_to(ROOT)), "eval_only": True},
        "created_at": common.utc_now(),
    }
    return result


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(result: dict, path: Path) -> None:
    lines = [f"# RGB+pose adapter 回归结果：{result['label']}", ""]
    lines.append(f"- 状态：**{'PASS' if result['status'] == 'PASS' else 'FAIL'}**")
    lines.append(f"- 冻结基线：`{result['frozen_predictions']}`（sha256 `{result['frozen_predictions_sha256']}`）")
    lines.append(f"- adapter 预测：`{result['candidate_predictions']}`（sha256 `{result['candidate_predictions_sha256']}`）")
    lines.append(f"- 样本：冻结 {result['n_frozen']} / adapter {result['n_candidate']} / 交集 {result['n_common']}；"
                 f"缺失 {result['n_missing']}，多余 {result['n_extra']}")
    lines.append(f"- 逐条逐字一致：**{result['exact_match']}/{result['n_common']}**（{_fmt(result['exact_match_rate'] * 100 if result['exact_match_rate'] is not None else None, 2)}%）")
    lines.append("")
    lines.append("## 指标对照（EVAL ONLY）")
    lines.append("")
    lines.append("| 指标 | 冻结基线 | adapter run | Δ | ≤0.01? |")
    lines.append("|---|---:|---:|---:|---|")
    for key, entry in result["metric_comparison"].items():
        lines.append(f"| {key} | {_fmt(entry['baseline'])} | {_fmt(entry['adapter_run'])} | {_fmt(entry['delta'])} | "
                     f"{'PASS' if entry['within_tolerance'] else 'FAIL'} |")
    lines.append("")
    if result["differences"]:
        lines.append("## 差异清单（前 20 条）")
        lines.append("")
        lines.append("| # | sample | 分类 | 冻结文本 | adapter 文本 |")
        lines.append("|---:|---|---|---|---|")
        for index, item in enumerate(result["differences"][:20], 1):
            lines.append(f"| {index} | `{item['sample_id']}` | {item['cause_category']} | {item['frozen_text']} | {item['candidate_text']} |")
        lines.append("")
        lines.append("分类统计：" + json.dumps(
            {category: sum(1 for item in result["differences"] if item["cause_category"] == category)
             for category in sorted({item["cause_category"] for item in result["differences"]})},
            ensure_ascii=False))
    else:
        lines.append("## 差异清单")
        lines.append("")
        lines.append("无差异：逐条逐字一致。")
    lines.append("")
    lines.append("## GT 纪律")
    lines.append("")
    lines.append("- adapter 预测为 GT-free；GT 仅在本工具中用于离线指标重算（EVAL ONLY）。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare an RGB+pose adapter run to the frozen E01 baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--frozen", type=Path, required=True)
    cmp_parser.add_argument("--candidate", type=Path, required=True)
    cmp_parser.add_argument("--frozen-metrics", type=Path, default=None)
    cmp_parser.add_argument("--label", default="pilot50")
    cmp_parser.add_argument("--candidate-run-dir", type=Path, default=None)
    cmp_parser.add_argument("--out-json", type=Path, default=None)
    cmp_parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "compare":
        result = compare_pair(args.frozen, args.candidate, frozen_metrics_path=args.frozen_metrics,
                              label=args.label, candidate_run_dir=args.candidate_run_dir)
        if args.out_json:
            common.write_json(args.out_json, result)
        if args.out_md:
            write_markdown(result, args.out_md)
        print(json.dumps({key: result[key] for key in (
            "label", "status", "n_frozen", "n_candidate", "n_common", "n_missing", "n_extra",
            "exact_match", "exact_match_rate", "metric_comparison", "differences_count") if key in result},
            ensure_ascii=False, indent=2, default=str))
        if result["differences"]:
            print(json.dumps({"first_differences": result["differences"][:5]}, ensure_ascii=False, indent=2, default=str))
        return 0 if result["status"] == "PASS" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
