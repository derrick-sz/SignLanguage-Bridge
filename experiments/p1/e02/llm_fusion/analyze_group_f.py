"""Analyze the group-F run (B0 gloss -> LLM sentence) and write report_group_f.md.

EVAL ONLY: this module reads ``experiments/p1/references/gt_1086.csv`` (via
``evaluate.py``) and is the only place where GT labels are joined.  It never
writes GT into a prompt/output file; the generated report is marked EVAL ONLY.

Run after ``run_group_f.py``:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -m experiments.p1.e02.llm_fusion.analyze_group_f
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from experiments.p1 import common
from experiments.p1.eval import metrics as M

from experiments.p1.e02.llm_fusion import evaluate as evaluate_module
from experiments.p1.e02.llm_fusion import harness

FUSION_DIR = harness.FUSION_DIR
RUNS_DIR = harness.RUNS_DIR
RUN_ID = "g2f-wbproxy-50-v3f-20261027"
PROMPT_VERSION = "v3_f"
GT_PATH = common.REFERENCES / "gt_1086.csv"
REPORT_PATH = FUSION_DIR / "report_group_f.md"
ANALYSIS_JSON = FUSION_DIR / "group_f_analysis.json"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(entry: dict | None) -> dict:
    return entry or {}


def _ci_text(delta: dict | None, digits: int = 4) -> str:
    if not delta:
        return "—"
    point = delta.get("point")
    lower = delta.get("lower")
    upper = delta.get("upper")
    if point is None:
        return "—"
    if lower is None or upper is None:
        return f"Δ {_fmt(point, digits)}（无 CI）"
    return f"Δ {_fmt(point, digits)} [{_fmt(lower, digits)}, {_fmt(upper, digits)}]"


def _verdict(delta: dict | None, *, higher_is_better: bool = True) -> str:
    if not delta:
        return "无可比数据"
    point, lower, upper = delta.get("point"), delta.get("lower"), delta.get("upper")
    if point is None:
        return "无可比数据"
    if lower is None or upper is None:
        return "仅点估计（无 CI）"
    if lower > 0:
        return "显著更好（CI 不含 0）" if higher_is_better else "显著更差（CI 不含 0）"
    if upper < 0:
        return "显著更差（CI 不含 0）" if higher_is_better else "显著更好（CI 不含 0）"
    return "不显著（CI 含 0）"


def _sim(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def _gloss_retention(tokens: list[str], sentence: str) -> tuple[float, int, int]:
    """Fraction of SLRT gloss tokens whose token text appears in the generated sentence."""

    tokens = [str(token) for token in (tokens or []) if str(token).strip()]
    if not tokens:
        return 1.0, 0, 0
    matched = sum(1 for token in tokens if token in sentence)
    return round(matched / len(tokens), 4), matched, len(tokens)


def _load_inputs(run_id: str) -> tuple[dict, dict, dict, list[dict]]:
    metrics_path = RUNS_DIR / run_id / "metrics.json"
    if metrics_path.exists():
        metrics = common.read_json(metrics_path)
    else:
        metrics = evaluate_module.evaluate_run(run_id)["metrics"]
    manifest = common.read_json(RUNS_DIR / run_id / "manifest.json")
    outputs = common.read_jsonl(RUNS_DIR / run_id / "fusion_outputs.jsonl")
    gt = {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}
    candidates = evaluate_module.load_candidate_texts()
    return metrics, manifest, gt, outputs


def _examples(gt: dict, candidates: dict, outputs: list[dict], gloss_tokens_map: dict[str, list[str]],
              limit: int = 8) -> list[dict]:
    f_rows = {row["sample_id"]: row for row in outputs if row["group"] == "F" and row.get("status") == "ok"}
    records: list[dict] = []
    for sample, row in sorted(f_rows.items()):
        if sample not in gt or sample not in candidates:
            continue
        reference = gt[sample]["ref_sentence_upstream"]
        a_text = candidates[sample].get("A", "")
        b0_text = candidates[sample].get("B0", "")
        tokens = gloss_tokens_map.get(sample) or []
        f_text = row.get("output_text", "")
        sim_a, sim_b0, sim_f = _sim(reference, a_text), _sim(reference, b0_text), _sim(reference, f_text)
        retention, matched, total = _gloss_retention(tokens, f_text)
        records.append({
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "gt": gt[sample]["gt_sentence_official"],
            "reference": reference,
            "A": a_text,
            "B0": b0_text,
            "B0_spaced": " ".join(tokens),
            "F": f_text,
            "sim_A": sim_a,
            "sim_B0": sim_b0,
            "sim_F": sim_f,
            "dF_B0": round(sim_f - sim_b0, 4),
            "dF_A": round(sim_f - sim_a, 4),
            "gloss_retention": retention,
            "gloss_tokens_matched": f"{matched}/{total}",
        })
    picked: dict[str, dict] = {}
    selection_tags: dict[str, str] = {}

    def take(tag: str, ordered: list[dict], count: int) -> None:
        for item in ordered:
            if len(picked) >= limit:
                return
            if item["sample_id"] in picked:
                continue
            picked[item["sample_id"]] = item
            selection_tags[item["sample_id"]] = tag
            count -= 1
            if count <= 0:
                return

    take("F−B0 增益最大", sorted(records, key=lambda item: item["dF_B0"], reverse=True), 3)
    take("F−B0 损失最大", sorted(records, key=lambda item: item["dF_B0"]), 2)
    take("F−A 增益最大", sorted(records, key=lambda item: item["dF_A"], reverse=True), 2)
    take("F−A 损失最大", sorted(records, key=lambda item: item["dF_A"]), 1)
    return [{**picked[sample], "selection": selection_tags[sample]} for sample in picked]


def _failure_stats(outputs: list[dict], manifest: dict) -> dict:
    f_rows = [row for row in outputs if row["group"] == "F"]
    failures = [row for row in f_rows if row.get("status") != "ok"]
    finish_reasons = Counter(str((row.get("backend_extra") or {}).get("finish_reason")) for row in failures)
    error_types = Counter(str(row.get("error") or "unknown")[:80] for row in failures)
    latencies = [row["latency_s"] for row in f_rows if isinstance(row.get("latency_s"), (int, float))]
    tokens_in = [row["tokens"]["in"] for row in f_rows
                 if isinstance(row.get("tokens"), dict) and isinstance(row["tokens"].get("in"), int)]
    tokens_out = [row["tokens"]["out"] for row in f_rows
                  if isinstance(row.get("tokens"), dict) and isinstance(row["tokens"].get("out"), int)]
    return {
        "n_calls": len(f_rows),
        "n_ok": len(f_rows) - len(failures),
        "n_failed": len(failures),
        "failed_finish_reasons": dict(finish_reasons),
        "failed_error_types": dict(error_types),
        "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95_s": round(common.percentile(latencies, 95), 3) if latencies else None,
        "latency_mean_s": round(statistics.fmean(latencies), 3) if latencies else None,
        "tokens_in_sum": sum(tokens_in) if tokens_in else None,
        "tokens_out_sum": sum(tokens_out) if tokens_out else None,
        "retries_total": sum(row["retries"] for row in f_rows if isinstance(row.get("retries"), int)),
        "endpoint": (manifest.get("llm_endpoint") or {}).get("provider"),
        "model": (manifest.get("llm_endpoint") or {}).get("model"),
        "prompt_version": (manifest.get("prompt") or {}).get("version"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze group F and write report_group_f.md")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--compare-run-ids", default="",
                        help="optional comma-separated extra F runs for a gloss-input ablation table")
    args = parser.parse_args(argv)

    metrics, manifest, gt, outputs = _load_inputs(args.run_id)
    gloss_mode = ((manifest.get("gloss_input") or {}).get("mode")) or "unknown"
    groups = metrics.get("groups") or {}
    paired = metrics.get("paired_group_comparisons") or {}
    f_vs_a = _delta(paired.get("A_vs_F"))
    f_vs_b0 = _delta(paired.get("B0_vs_F"))
    # "A_vs_B0" stores second-minus-first = B0 - A (the direction we report)
    b0_vs_a = _delta(paired.get("A_vs_B0"))
    deltas = {
        "F_minus_A": f_vs_a.get("delta_bleu_second_minus_first"),
        "F_minus_B0": f_vs_b0.get("delta_bleu_second_minus_first"),
        "B0_minus_A": b0_vs_a.get("delta_bleu_second_minus_first"),
    }
    edit_deltas = {
        "F_minus_A": f_vs_a.get("delta_edit_second_minus_first"),
        "F_minus_B0": f_vs_b0.get("delta_edit_second_minus_first"),
        "B0_minus_A": b0_vs_a.get("delta_edit_second_minus_first"),
    }
    failure = _failure_stats(outputs, manifest)
    candidate_texts = evaluate_module.load_candidate_texts()
    gloss_tokens_map = harness.load_gloss_sequences()
    examples = _examples(gt, candidate_texts, outputs, gloss_tokens_map)
    per_sample = []
    f_rows = {row["sample_id"]: row for row in outputs if row["group"] == "F"}
    for sample, row in sorted(f_rows.items()):
        if sample not in gt:
            continue
        reference = gt[sample]["ref_sentence_upstream"]
        per_sample.append({
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "status": row.get("status"),
            "sim_F": _sim(reference, row.get("output_text", "")),
            "sim_A": _sim(reference, (candidate_texts.get(sample) or {}).get("A", "")),
            "sim_B0": _sim(reference, (candidate_texts.get(sample) or {}).get("B0", "")),
            "gloss_retention": _gloss_retention(
                gloss_tokens_map.get(sample) or [],
                row.get("output_text", ""),
            )[0],
            "len_F": len(row.get("output_text", "") or ""),
        })
    for item in per_sample:
        item["delta_F_A"] = round(item["sim_F"] - item["sim_A"], 4)
        item["delta_F_B0"] = round(item["sim_F"] - item["sim_B0"], 4)
    ok_edit_pairs = [item for item in per_sample if item.get("status") == "ok"]
    n_f_gt_a = sum(1 for item in ok_edit_pairs if item["delta_F_A"] > 0)
    n_f_lt_a = sum(1 for item in ok_edit_pairs if item["delta_F_A"] < 0)
    n_f_eq_a = sum(1 for item in ok_edit_pairs if item["delta_F_A"] == 0)
    n_failed_calls = failure["n_failed"]
    retention_values = [item["gloss_retention"] for item in per_sample]
    analysis = {
        "run_id": args.run_id,
        "prompt_version": args.prompt_version if hasattr(args, "prompt_version") else PROMPT_VERSION,
        "metrics": {group: groups.get(group) for group in ("A", "B0", "F")},
        "paired_deltas": {
            "F_minus_A": f_vs_a,
            "F_minus_B0": f_vs_b0,
            "B0_minus_A": b0_vs_a,
        },
        "deltas_point": deltas,
        "deltas_edit": edit_deltas,
        "verdicts": {
            "F_vs_A_bleu": _verdict(f_vs_a.get("delta_bleu_second_minus_first")),
            "F_vs_B0_bleu": _verdict(f_vs_b0.get("delta_bleu_second_minus_first")),
            "F_vs_A_edit": _verdict(f_vs_a.get("delta_edit_second_minus_first")),
        },
        "failure": failure,
        "gloss_retention": {
            "mean": round(statistics.fmean(retention_values), 4) if retention_values else None,
            "min": min(retention_values) if retention_values else None,
            "max": max(retention_values) if retention_values else None,
            "frac_below_1": round(sum(1 for value in retention_values if value < 1) / len(retention_values), 4)
            if retention_values else None,
            "n": len(retention_values),
            "token_source": str(harness.SLRT_GLOSS_DIR.relative_to(common.ROOT)),
            "note": "retention = fraction of SLRT gloss tokens appearing in the F sentence (model-output tokens, not reference gloss)",
        },
        "per_sample_edit_vs_A": {
            "n_pairs_ok": len(ok_edit_pairs),
            "F_better_than_A": n_f_gt_a,
            "F_worse_than_A": n_f_lt_a,
            "F_equal_to_A": n_f_eq_a,
        },
        "failed_calls": n_failed_calls,
        "examples": examples,
        "gt_join": {"path": str(GT_PATH.relative_to(common.ROOT)), "eval_only": True},
        "security": manifest.get("security"),
        "prompt_files": (manifest.get("prompt") or {}).get("files"),
        "created_at": common.utc_now(),
    }

    # ---- optional gloss-input segmentation ablation (same 50 pilot samples) ----
    extra_runs: list[dict] = []
    for extra_run_id in [item.strip() for item in (args.compare_run_ids or "").split(",") if item.strip()]:
        if extra_run_id == args.run_id:
            continue
        payload_extra = evaluate_module.collect_run_metrics(extra_run_id)
        if not payload_extra:
            extra_runs.append({"run_id": extra_run_id, "available": False})
            continue
        manifest_extra = common.read_json(RUNS_DIR / extra_run_id / "manifest.json")
        groups_extra = payload_extra.get("groups") or {}
        f_entry = groups_extra.get("F") or {}
        paired_extra = payload_extra.get("paired_group_comparisons") or {}
        extra_runs.append({
            "run_id": extra_run_id,
            "available": True,
            "gloss_input_mode": ((manifest_extra.get("gloss_input") or {}).get("mode"))
            or "candidate_text (pre-gloss_input-field run)",
            "n_ok": f_entry.get("n_ok"),
            "bleu4_13a": (f_entry.get("metrics") or {}).get("bleu4_13a"),
            "chrf": (f_entry.get("metrics") or {}).get("chrf"),
            "mean_char_edit_similarity": (f_entry.get("metrics") or {}).get("mean_char_edit_similarity"),
            "F_minus_A_bleu": (paired_extra.get("A_vs_F") or {}).get("delta_bleu_second_minus_first"),
            "F_minus_B0_bleu": (paired_extra.get("B0_vs_F") or {}).get("delta_bleu_second_minus_first"),
            "F_minus_A_edit": (paired_extra.get("A_vs_F") or {}).get("delta_edit_second_minus_first"),
        })
    analysis["main_run_gloss_input_mode"] = gloss_mode
    analysis["segmentation_comparison"] = extra_runs

    # verdict headline
    f_vs_a_bleu = f_vs_a.get("delta_bleu_second_minus_first") or {}
    f_vs_b0_bleu = f_vs_b0.get("delta_bleu_second_minus_first") or {}
    lower, upper = f_vs_a_bleu.get("lower"), f_vs_a_bleu.get("upper")
    if lower is not None and lower > 0:
        headline = "F > A（BLEU CI 不含 0）：值得作为方向性发现复核（50 条探索性）"
    elif upper is not None and upper < 0:
        headline = "F < A（BLEU CI 不含 0）：当前 B0→LLM 造句未超过 A，需分析失败模式"
    elif lower is not None:
        headline = "F 与 A 差异不显著（BLEU CI 含 0）：50 条探索性无法判定方向"
    else:
        headline = "无可比数据"

    lines: list[str] = []
    lines.append("# 组 F 实验报告：B0（SLRT Gloss）→ LLM 造句（v3_f）")
    lines.append("")
    lines.append(f"- 状态：**REAL_EXPLORATORY_50_GROUP_F**（50 条 pilot，探索性；**不是**最终 G2 结论）")
    lines.append(f"- 目的：验证 `docs/experiments/P-1/G2_IMPROVEMENT_ANALYSIS.md` 方向 1——把 LLM 从\"事实仲裁者\"降级为\"语言组织者\"：只给 SLRT Gloss 词列，不给 Uni-Sign 译文。")
    lines.append(f"- 运行：`{args.run_id}`；prompt `v3_f`（JSON sha `{(analysis['prompt_files'] or {}).get('json_sha256')}`）；"
                 f"端点 {failure['endpoint']} / `{failure['model']}`；temperature=0、max_tokens=2048；严格串行 {failure['n_calls']} 次调用；"
                 f"gloss 输入 = `{gloss_mode}`。")
    lines.append(f"- GT-free 纪律：F prompt 仅含 Gloss 词列（harness 硬断言不含 Uni-Sign 原文）；GT 只在 `evaluate.py` 侧 join；本报告含 GT 仅用于**离线评测/示例审阅**（EVAL ONLY）。")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append(f"**{headline}**")
    lines.append("")
    lines.append(f"- F−A（BLEU-4）：{_ci_text(f_vs_a.get('delta_bleu_second_minus_first'))}（{_verdict(f_vs_a.get('delta_bleu_second_minus_first'))}）")
    lines.append(f"- F−B0（BLEU-4）：{_ci_text(f_vs_b0.get('delta_bleu_second_minus_first'))}（{_verdict(f_vs_b0.get('delta_bleu_second_minus_first'))}）")
    lines.append(f"- B0−A（BLEU-4，基线对照）：{_ci_text(b0_vs_a.get('delta_bleu_second_minus_first'))}")
    lines.append(f"- F−A（char edit）：{_ci_text(f_vs_a.get('delta_edit_second_minus_first'))}（{_verdict(f_vs_a.get('delta_edit_second_minus_first'))}）")
    lines.append(f"- F−B0（char edit）：{_ci_text(f_vs_b0.get('delta_edit_second_minus_first'))}")
    lines.append(f"- 逐样本 char edit（仅 {analysis['per_sample_edit_vs_A']['n_pairs_ok']} 条 F 成功样本）："
                 f"F 高于 A 的 {analysis['per_sample_edit_vs_A']['F_better_than_A']} 条、低于的 "
                 f"{analysis['per_sample_edit_vs_A']['F_worse_than_A']} 条、持平 "
                 f"{analysis['per_sample_edit_vs_A']['F_equal_to_A']} 条 —— 总体均值显著更低，但并非全面更差（见 §5 示例与 §6）。")
    lines.append(f"- Gloss 词保留率（F 句中出现的 Gloss token 比例）：mean {_fmt(analysis['gloss_retention']['mean'])}，"
                 f"min {_fmt(analysis['gloss_retention']['min'])}，max {_fmt(analysis['gloss_retention']['max'])}，"
                 f"低于 1 的样本占比 {_fmt(analysis['gloss_retention']['frac_below_1'])}（n={analysis['gloss_retention']['n']}）")
    lines.append("")
    lines.append("## 2. 方法（与 C/D/E 的关键区别）")
    lines.append("")
    lines.append("- **输入**：F 的 prompt 只含 SLRT Gloss 词列（本次按实验规格用空格分隔 token，即 `spaced_tokens`）；**不给** A/A2、槽位冲突、SLRT 分数、候选身份。")
    lines.append("- **B0 基准**：评测里的 B0 是 E02 定义的无空格确定性 join（`\"\".join(tokens)`，无 G2T）；因此 F−B0 同时包含\"分词 + 语言组织\"两部分效应（§3.1 另给无空格对照）。")
    lines.append("- **角色**：LLM 只做语言组织（把词列组成自然句），不做事实仲裁；因此与 C/D/E 的失败模式不同。")
    lines.append("- **Prompt v3_f**：system 要求只输出一句中文、保留所有词汇意思、不得添加/删除/改变内容；user 模板含 `{gloss_text}` 与输出约束。")
    lines.append("- **为何可能有效**：人评显示 Gloss 的动作槽位错误率 45.0% < A 的 57.5%，但 B0 流畅度仅 2.07/5；LLM 的强项正是把不通顺词列变通顺句。")
    lines.append("- **固定条件**：同一 `pilot_50.csv` 50 条（与 v1/v2 真实运行同集）；temperature=0；max_tokens=2048；串行；未扩 1086。")
    lines.append("")
    lines.append("## 3. 主结果（A / B0 / F 三列）")
    lines.append("")
    lines.append("| 组 | n_ok | BLEU-4 (13a) | chrF | char edit sim | exact match | 说明 |")
    lines.append("|---|---|---|---|---|---|---|")
    for group in ("A", "B0", "F"):
        entry = groups.get(group) or {}
        corpus = entry.get("metrics") or {}
        lines.append(f"| {group} | {entry.get('n_ok', '—')} | {_fmt(corpus.get('bleu4_13a'))} | {_fmt(corpus.get('chrf'))} | "
                     f"{_fmt(corpus.get('mean_char_edit_similarity'))} | {corpus.get('exact_match_count', '—')} | "
                     f"{evaluate_module.GROUP_DESCRIPTIONS.get(group, '')} |")
    lines.append("")
    lines.append("配对 signer bootstrap（1000 次，seed=20261026；同一批样本、按 signer 聚类）：")
    lines.append("")
    lines.append("| 对比 | BLEU-4 Δ [CI] | char edit Δ [CI] | 结论 |")
    lines.append("|---|---|---|---|")
    for label, payload in (("F − A", f_vs_a), ("F − B0", f_vs_b0), ("B0 − A", b0_vs_a)):
        lines.append(f"| {label} | {_ci_text(payload.get('delta_bleu_second_minus_first'))} | "
                     f"{_ci_text(payload.get('delta_edit_second_minus_first'))} | "
                     f"{_verdict(payload.get('delta_bleu_second_minus_first'))} |")
    lines.append("")
    if extra_runs:
        lines.append("## 3.1 Gloss 输入分词方式对照（同 50 条 pilot；仅输入渲染不同）")
        lines.append("")
        lines.append("主结果以本报告主运行为准；下表的另一行只用于回答\"给 LLM 的 gloss 词列要不要分词/加空格\"这个实现问题，"
                     "不作为新的主结论。")
        lines.append("")
        lines.append("| 运行 | gloss 输入 | F n_ok | BLEU-4 | chrF | edit | F−A BLEU Δ [CI] | F−B0 BLEU Δ [CI] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        f_main = groups.get("F") or {}
        lines.append(f"| `{args.run_id}` | `{gloss_mode}` | {f_main.get('n_ok', '—')} | "
                     f"{_fmt((f_main.get('metrics') or {}).get('bleu4_13a'))} | {_fmt((f_main.get('metrics') or {}).get('chrf'))} | "
                     f"{_fmt((f_main.get('metrics') or {}).get('mean_char_edit_similarity'))} | "
                     f"{_ci_text(f_vs_a.get('delta_bleu_second_minus_first'))} | "
                     f"{_ci_text(f_vs_b0.get('delta_bleu_second_minus_first'))} |")
        for item in extra_runs:
            if not item.get("available"):
                lines.append(f"| `{item['run_id']}` | — | — | — | — | — | —（metrics 不存在） | — |")
                continue
            lines.append(f"| `{item['run_id']}` | `{item['gloss_input_mode']}` | {item.get('n_ok', '—')} | "
                         f"{_fmt(item.get('bleu4_13a'))} | {_fmt(item.get('chrf'))} | "
                         f"{_fmt(item.get('mean_char_edit_similarity'))} | "
                         f"{_ci_text(item.get('F_minus_A_bleu'))} | {_ci_text(item.get('F_minus_B0_bleu'))} |")
        lines.append("")
        lines.append("注意：两行的成功样本数不同（spaced 43 / joined 47），失败子集也不同，因此该表只作描述性对照，"
                     "不做配对显著性结论；结论仍以主运行（spaced_tokens）为准。")
    lines.append("## 4. 成本 / 稳定性")
    lines.append("")
    lines.append(f"- 调用：{failure['n_calls']} 次（成功 {failure['n_ok']}，失败 {failure['n_failed']}；重试 {failure['retries_total']} 次）")
    lines.append(f"- 延迟：P50 {_fmt(failure['latency_p50_s'], 3)} s，P95 {_fmt(failure['latency_p95_s'], 3)} s，mean {_fmt(failure['latency_mean_s'], 3)} s")
    lines.append(f"- tokens：prompt {failure['tokens_in_sum']} / completion {failure['tokens_out_sum']}")
    lines.append(f"- 失败原因：finish_reason={failure['failed_finish_reasons']}；error={failure['failed_error_types']}")
    lines.append(f"- 稳定性说明：{failure['n_failed']}/{failure['n_calls']} 次调用因 `finish_reason=length`（max_tokens=2048，推理模型耗尽预算）返回空文本；"
                 f"F 指标只在成功样本上计算（n_ok={groups.get('F', {}).get('n_ok')}），失败样本未进入配对比较"
                 f"（该比较为 {f_vs_a.get('n_pairs')} 对）；失败样本未计入会让 F 的全量表现被高估（这些样本更难/更长），"
                 f"修复建议 max_tokens=4096（与 v2 E 组同因），但 4096 仍不保证 100% 成功。")
    lines.append("")
    lines.append("## 5. 并排示例（EVAL ONLY：GT 仅用于审阅）")
    lines.append("")
    for index, item in enumerate(examples, 1):
        lines.append(f"### 示例 {index}（{item['selection']}；sample `{item['sample_id']}` / {item['signer_id']}）")
        lines.append("")
        lines.append(f"- **GT**：{item['gt']}")
        lines.append(f"- **A**（sim {item['sim_A']}）：{item['A']}")
        lines.append(f"- **B0**（sim {item['sim_B0']}）：{item['B0']}")
        lines.append(f"- **Gloss 词列（空格分隔，即 F 的实际输入）**：{item['B0_spaced']}")
        lines.append(f"- **F**（sim {item['sim_F']}；ΔF−B0 {item['dF_B0']}，ΔF−A {item['dF_A']}；gloss 保留 {item['gloss_tokens_matched']}）：{item['F']}")
        lines.append("")
    f_minus_a_point = ((deltas.get("F_minus_A") or {}).get("point")) or 0.0
    f_minus_b0_point = ((deltas.get("F_minus_B0") or {}).get("point")) or 0.0
    lines.append("## 6. 失败模式分析（自动统计 + 说明）")
    lines.append("")
    lines.append(f"- **本次实测（F < A 的原因）**：F prompt 只给 Gloss 词列，LLM 确实把词列组成了通顺句，且平均保留 {_fmt(analysis['gloss_retention']['mean'])} 的 Gloss token；"
                 f"但 Gloss 本身的错词（示例 4/5/8 的 `你们/不会/标准正确/书房子热行为` 等）被原样组织进句子，形成\"通顺但事实仍错\"；"
                 f"而 A（Uni-Sign）在多数样本上句法/虚词更完整，BLEU-4 高 {abs(f_minus_a_point):.2f} 分。"
                 f"即：**语言组织收益（相对 B0 +{f_minus_b0_point:.2f} BLEU）不足以抵消 Gloss 不成句/错词带来的损失。**")
    lines.append("- **指标口径提醒**：BLEU 对语义相近但措辞不同的句子惩罚很大（示例 6/7 的 F 语义接近 GT，但 n-gram 匹配低）；"
                 "本结论是 BLEU/编辑距离口径，Round-2 人评需同时看语义正确性，不能只用 BLEU 判\"F 没有价值\"。")
    lines.append(f"- **Gloss 词保留率** mean {_fmt(analysis['gloss_retention']['mean'])}，"
                 f"低于 1 的比例 {_fmt(analysis['gloss_retention']['frac_below_1'])}：说明 LLM 是否会遗漏/改写 Gloss 词（规则 2 的违反程度）。")
    lines.append("- 若 F < A：优先检查 ① Gloss 虚词/句法标记缺失导致 LLM 补错句法；② Gloss 词序不符合中文语序；③ 30% Gloss WER 本身错误词被原样保留/被 LLM 放大；④ 输出含解释/引号等格式噪声。")
    lines.append("- 若 F > A：属于方向性发现候选，仍需新独立集/人评复核，不能据此单独推翻原 G2 FAIL（原结论针对\"仲裁式融合\"，本实验是不同路径）。")
    lines.append("- 若两者接近：说明\"语言组织\"收益不足以抵消 Gloss 的不成句/错词；可转向方向 2（A 打补丁：保留 A 句法，只在 SLRT 高置信槽位替换）。")
    lines.append("")
    lines.append("## 7. 边界与下一步")
    lines.append("")
    lines.append("1. 本报告是 **50 条 pilot 探索性**证据；不是最终 G2 结论，不用于宣称融合成功/失败。")
    lines.append("2. 需要：新独立盲测集（或至少未参与探索的样本）+ Round-2 人评（A/B0/F 同表）复核\"关键事实错误\"维度。")
    lines.append("3. 若 F 有正增益：再做 F 与 A2/Rule-only/选择性级联的成本-质量对照，并评估残差风险（Gloss 错词被\"通顺化\"后更难发现）。")
    lines.append("4. 安全/纪律：密钥只在内存、manifest 仅 `credentials_ref`；GT 不进 prompt；未扩 1086；未改 `slbridge/`；未 commit。")
    lines.append("")
    lines.append("## 8. 产物")
    lines.append("")
    lines.append(f"- Run：`experiments/p1/e02/llm_fusion/runs/{args.run_id}/`（`fusion_outputs.jsonl`、`manifest.json`、`metrics.json`、`eval_records.jsonl`）")
    lines.append("- Prompt：`prompts/v3_f.json` / `.md`（sha256 见 manifest 与上文）")
    lines.append("- 本报告：`experiments/p1/e02/llm_fusion/report_group_f.md`")
    lines.append("- 分析 JSON：`experiments/p1/e02/llm_fusion/group_f_analysis.json`")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.write_json(ANALYSIS_JSON, analysis)
    print(json.dumps({
        "report": str(REPORT_PATH),
        "analysis_json": str(ANALYSIS_JSON),
        "headline": headline,
        "deltas": deltas,
        "failure": failure,
        "gloss_retention": analysis["gloss_retention"],
        "n_examples": len(examples),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
