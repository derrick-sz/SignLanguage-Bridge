"""Final analysis for G2-2A3: prompt v2 vs v1, selective cascade, final report.

Reads the two already-produced real 50-sample runs (no LLM calls):

* `g2a-wbproxy-50-v2-20261026`        — prompt v1_zh
* `g2a-wbproxy-50-v2prompt-20261026`  — prompt v2_zh

Produces:

* `cascade_analysis_v2.json` — offline selective-cascade variants with measured
  per-call cost from the runs (no new calls);
* `report_prompt_v2.md`      — A / v1 / v2 comparison, D-C/E-D/C-E deltas with
  signer-clustered bootstrap CIs, selective-cascade table, token/latency
  summary, failure reasons, G2 conclusion recommendation and boundaries.

GT is joined only here (EVAL ONLY). No API key is read by this script.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.p1 import common
from experiments.p1.eval import metrics as M
from experiments.p1.e02.llm_fusion import evaluate as evaluate_module
from experiments.p1.e02.llm_fusion import harness

V1_RUN_ID = "g2a-wbproxy-50-v2-20261026"
V2_RUN_ID = "g2a-wbproxy-50-v2prompt-20261026"
GT_PATH = common.REFERENCES / "gt_1086.csv"
PILOT_50 = common.P1 / "datasets" / "pilot_50.csv"
OUT_JSON = harness.FUSION_DIR / "cascade_analysis_v2.json"
REPORT = harness.FUSION_DIR / "report_prompt_v2.md"
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20261026
SEVERE_SLOTS = ("numbers", "negation", "person")


def _load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def _load_by_sample(run_id: str) -> dict[str, dict[str, dict]]:
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in common.read_jsonl(harness.RUNS_DIR / run_id / "fusion_outputs.jsonl"):
        by_sample[row["sample_id"]][row["group"]] = row
    return by_sample


def _read_manifest(run_id: str) -> dict:
    return common.read_json(harness.RUNS_DIR / run_id / "manifest.json")


def _read_metrics(run_id: str) -> dict:
    return common.read_json(harness.RUNS_DIR / run_id / "metrics.json")


def _sentence_edit(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def _rule_judge_trigger(unisign_text: str, gloss_text: str) -> dict:
    from experiments.p1.e06.build_e06 import extract_slots, slot_conflicts

    slots_a = extract_slots(unisign_text)
    slots_b = extract_slots(gloss_text)
    conflicts = slot_conflicts(slots_a, slots_b)
    fired = sorted({conflict["slot"] for conflict in conflicts})
    return {"trigger_any": bool(conflicts),
            "trigger_severe": any(slot in SEVERE_SLOTS for slot in fired),
            "fired_slots": fired,
            "n_conflicts": len(conflicts)}


# --------------------------------------------------------------------------
# bootstrap helpers (paired, signer-clustered)
# --------------------------------------------------------------------------
def _bootstrap(records: list[dict], statistic, *, iterations: int = BOOTSTRAP_ITERATIONS,
               seed: int = BOOTSTRAP_SEED) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["signer_id"])].append(record)
    keys = sorted(groups)
    import random as _random

    rng = _random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample: list[dict] = []
        for _ in range(len(keys)):
            sample.extend(groups[rng.choice(keys)])
        value = statistic(sample)
        if value is not None:
            values.append(float(value))
    values.sort()
    point = statistic(records)
    if not values:
        return {"point": point, "lower": None, "upper": None, "iterations": iterations}
    return {
        "point": round(point, 4),
        "lower": round(values[int(0.025 * (len(values) - 1))], 4),
        "upper": round(values[int(0.975 * (len(values) - 1))], 4),
        "iterations": iterations,
        "seed": seed,
        "method": "paired_cluster_bootstrap_by_signer",
    }


def _delta_bleu_stat(first_key: str, second_key: str):
    def statistic(items: list[dict]) -> float:
        import sacrebleu

        refs = [M.character_separated(item["reference"]) for item in items]
        first = [M.character_separated(item[first_key]) for item in items]
        second = [M.character_separated(item[second_key]) for item in items]
        return (sacrebleu.corpus_bleu(second, [refs], tokenize="13a").score
                - sacrebleu.corpus_bleu(first, [refs], tokenize="13a").score)

    return statistic


def _delta_edit_stat(first_key: str, second_key: str):
    def statistic(items: list[dict]) -> float:
        return (statistics.fmean(item[f"sim_{second_key}"] for item in items)
                - statistics.fmean(item[f"sim_{first_key}"] for item in items))

    return statistic


def _quality(records: list[dict], hypothesis_key: str) -> dict:
    pairs = [(record["reference"], record[hypothesis_key]) for record in records]
    corpus = M.corpus_metrics_cslt(pairs)
    return {
        "n": len(records),
        "bleu4_13a": corpus["bleu4_13a"],
        "chrf": corpus["chrf"],
        "mean_char_edit_similarity": corpus["mean_char_edit_similarity"],
        "exact_match_count": corpus["exact_match_count"],
    }


def _cost(rows: list[dict], n_samples: int) -> dict:
    """Cost from actual per-call rows used by a variant (failed attempts included)."""

    if not rows:
        return {"llm_calls": 0, "call_rate": 0.0, "tokens_total": 0, "tokens_prompt": 0,
                "tokens_completion": 0, "latency_p50_s": None, "latency_p95_s": None,
                "latency_mean_s": None, "failed_calls": 0, "retries": 0, "n_samples": n_samples}
    tokens_total = sum((row["tokens"].get("total") or 0) for row in rows)
    tokens_prompt = sum((row["tokens"].get("in") or 0) for row in rows)
    tokens_completion = sum((row["tokens"].get("out") or 0) for row in rows)
    latencies = [row["latency_s"] for row in rows if isinstance(row.get("latency_s"), (int, float))]
    return {
        "llm_calls": len(rows),
        "call_rate": round(len(rows) / n_samples, 4) if n_samples else None,
        "tokens_total": tokens_total,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(common.percentile(latencies, 95), 2) if latencies else None,
        "latency_mean_s": round(statistics.fmean(latencies), 2) if latencies else None,
        "failed_calls": sum(1 for row in rows if row.get("status") != "ok"),
        "retries": sum(row.get("retries") or 0 for row in rows),
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------
# variant construction
# --------------------------------------------------------------------------
def build_records() -> tuple[list[dict], dict]:
    gt = _load_gt()
    v1 = _load_by_sample(V1_RUN_ID)
    v2 = _load_by_sample(V2_RUN_ID)
    samples = sorted(set(v2) & set(gt))
    records: list[dict] = []
    trigger_counts: Counter = Counter()
    fired_by_slot: Counter = Counter()
    for sample in samples:
        a_row = v2[sample].get("A") or {}
        a_text = a_row.get("output_text", "")
        gloss = (a_row.get("evidence") or {}).get("slrt_gloss_text", "")
        if not gloss:
            gloss = ((v2[sample].get("E") or {}).get("evidence") or {}).get("slrt_gloss_text", "")
        trigger = _rule_judge_trigger(a_text, gloss)
        if trigger["trigger_any"]:
            trigger_counts["any"] += 1
        if trigger["trigger_severe"]:
            trigger_counts["severe"] += 1
        for slot in trigger["fired_slots"]:
            fired_by_slot[slot] += 1
        reference = gt[sample]["ref_sentence_upstream"]
        record = {
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "reference": reference,
            "trigger_any": trigger["trigger_any"],
            "trigger_severe": trigger["trigger_severe"],
            "fired_slots": trigger["fired_slots"],
        }
        for version, runs in (("v1", v1), ("v2", v2)):
            for group in ("A", "C", "D", "E"):
                row = (runs.get(sample) or {}).get(group) or {}
                ok = row.get("status") == "ok" and bool((row.get("output_text") or "").strip())
                record[f"{group}_{version}"] = row.get("output_text", "") if ok else ""
                record[f"sim_{group}_{version}"] = _sentence_edit(reference, row.get("output_text", "")) if ok else None
                record[f"ok_{group}_{version}"] = ok
        records.append(record)
    trigger_info = {
        "n_samples": len(records),
        "trigger_rate_any": round(trigger_counts["any"] / len(records), 4) if records else None,
        "trigger_rate_severe": round(trigger_counts["severe"] / len(records), 4) if records else None,
        "triggered_n_any": trigger_counts["any"],
        "triggered_n_severe": trigger_counts["severe"],
        "fired_slots": dict(sorted(fired_by_slot.items())),
    }
    return records, trigger_info


def _variant(records: list[dict], *, use_key: str | None,
             run_rows: dict[str, dict[str, list[dict]]], n_samples: int) -> dict:
    """Construct a variant's final texts + quality + cost.

    ``use_key`` is the output key selected when the variant decides to use a
    model output (e.g. ``E_v2``); ``None`` means pure A.  If the selected model
    output is missing/failed, fall back to A (deployment fallback) and count it.
    """

    final_records: list[dict] = []
    used_rows: list[dict] = []
    fallback_n = 0
    used_n = 0
    for record in records:
        if use_key is None:
            final = record["A_v2"]
            sim = record["sim_A_v2"]
            used = False
        else:
            group, version = use_key.split("_")
            ok = record[f"ok_{group}_{version}"]
            if ok:
                final = record[use_key]
                sim = record[f"sim_{use_key}"]
                used = True
                used_n += 1
                row = run_rows[version].get(record["sample_id"], {}).get(group)
                if row is not None:
                    used_rows.append(row)
            else:
                final = record["A_v2"]
                sim = record["sim_A_v2"]
                fallback_n += 1
        final_records.append({
            "sample_id": record["sample_id"],
            "signer_id": record["signer_id"],
            "reference": record["reference"],
            "hyp_a": record["A_v2"],
            "hyp_policy": final,
            "sim_a": record["sim_A_v2"],
            "sim_policy": sim,
        })
    quality = _quality(final_records, "hyp_policy")
    delta_bleu = _bootstrap(final_records, _delta_bleu_stat("hyp_a", "hyp_policy"))
    delta_edit = _bootstrap(final_records, _delta_edit_stat("a", "policy"))
    cost = _cost(used_rows, n_samples)
    return {
        "n": n_samples,
        "used_model_n": used_n,
        "fallback_to_A_n": fallback_n,
        "quality": quality,
        "delta_bleu_vs_A": delta_bleu,
        "delta_edit_vs_A": delta_edit,
        "cost": cost,
    }


def cascade_variants(records: list[dict]) -> dict:
    v1_rows = _load_by_sample(V1_RUN_ID)
    v2_rows = _load_by_sample(V2_RUN_ID)
    run_rows = {"v1": v1_rows, "v2": v2_rows}
    n = len(records)
    variants: dict[str, dict] = {}
    variants["pure_A_no_LLM"] = _variant(records, use_key=None, run_rows=run_rows, n_samples=n)
    variants["all_D_v2"] = _variant(records, use_key="D_v2", run_rows=run_rows, n_samples=n)
    variants["all_E_v1"] = _variant(records, use_key="E_v1", run_rows=run_rows, n_samples=n)
    variants["all_E_v2"] = _variant(records, use_key="E_v2", run_rows=run_rows, n_samples=n)

    def cascade_variant(use_key: str) -> dict:
        subset = [record for record in records if record["trigger_any"]]
        # quality must still be measured on all samples; use triggered-only for cost
        triggered_rows: list[dict] = []
        group, version = use_key.split("_")
        for record in subset:
            row = run_rows[version].get(record["sample_id"], {}).get(group)
            if row is not None:
                triggered_rows.append(row)
        # build variant with a trigger condition: reuse _variant but override cost rows
        final_records: list[dict] = []
        used_n = fallback_n = 0
        for record in records:
            if record["trigger_any"] and record[f"ok_{group}_{version}"]:
                final = record[use_key]
                sim = record[f"sim_{use_key}"]
                used_n += 1
            else:
                final = record["A_v2"]
                sim = record["sim_A_v2"]
                if record["trigger_any"]:
                    fallback_n += 1
            final_records.append({
                "sample_id": record["sample_id"],
                "signer_id": record["signer_id"],
                "reference": record["reference"],
                "hyp_a": record["A_v2"],
                "hyp_policy": final,
                "sim_a": record["sim_A_v2"],
                "sim_policy": sim,
            })
        quality = _quality(final_records, "hyp_policy")
        delta_bleu = _bootstrap(final_records, _delta_bleu_stat("hyp_a", "hyp_policy"))
        delta_edit = _bootstrap(final_records, _delta_edit_stat("a", "policy"))
        cost = _cost(triggered_rows, n)
        return {
            "n": n,
            "triggered_n": len(subset),
            "used_model_n": used_n,
            "fallback_to_A_n": fallback_n,
            "quality": quality,
            "delta_bleu_vs_A": delta_bleu,
            "delta_edit_vs_A": delta_edit,
            "cost": cost,
        }

    variants["cascade_ruleE_v1"] = cascade_variant("E_v1")
    variants["cascade_ruleE_v2"] = cascade_variant("E_v2")
    variants["cascade_ruleD_v2"] = cascade_variant("D_v2")
    # "优选" on the triggered subset: which E version is better there
    triggered = [record for record in records if record["trigger_any"]]
    triggered_v1 = [(record["reference"], record["E_v1"]) for record in triggered if record["ok_E_v1"]]
    triggered_v2 = [(record["reference"], record["E_v2"]) for record in triggered if record["ok_E_v2"]]
    triggered_compare = {
        "triggered_n": len(triggered),
        "E_v1_ok_n": len(triggered_v1),
        "E_v2_ok_n": len(triggered_v2),
        "E_v1_on_triggered": M.corpus_metrics_cslt(triggered_v1)["bleu4_13a"] if triggered_v1 else None,
        "E_v2_on_triggered": M.corpus_metrics_cslt(triggered_v2)["bleu4_13a"] if triggered_v2 else None,
        "failure_rates": {
            "E_v1": round(1 - len(triggered_v1) / len(triggered), 4) if triggered else None,
            "E_v2": round(1 - len(triggered_v2) / len(triggered), 4) if triggered else None,
        },
    }
    choose = "E_v2" if (triggered_compare["E_v2_on_triggered"] or 0) >= (triggered_compare["E_v1_on_triggered"] or 0) else "E_v1"
    triggered_compare["chosen_for_cascade"] = choose
    triggered_compare["reason"] = (
        "chosen by triggered-subset BLEU; both versions are reported so the choice is auditable"
        if choose else None
    )
    return {"n_samples": n, "variants": variants, "triggered_comparison_E_v1_vs_v2": triggered_compare}


# --------------------------------------------------------------------------
# v1 vs v2 paired deltas (same samples, offline)
# --------------------------------------------------------------------------
def v1_vs_v2_deltas(records: list[dict]) -> dict:
    result = {}
    for group in ("C", "D", "E"):
        subset = [record for record in records
                  if record[f"ok_{group}_v1"] and record[f"ok_{group}_v2"]]
        if not subset:
            continue
        comparison_records = [{
            "sample_id": record["sample_id"],
            "signer_id": record["signer_id"],
            "reference": record["reference"],
            f"{group}_v1": record[f"{group}_v1"],
            f"{group}_v2": record[f"{group}_v2"],
            "sim_v1": record[f"sim_{group}_v1"],
            "sim_v2": record[f"sim_{group}_v2"],
        } for record in subset]
        delta_bleu = _bootstrap(comparison_records, _delta_bleu_stat(f"{group}_v1", f"{group}_v2"))
        delta_edit = _bootstrap(comparison_records, _delta_edit_stat("v1", "v2"))
        result[group] = {
            "n_pairs": len(subset),
            "delta_bleu_v2_minus_v1": delta_bleu,
            "delta_edit_v2_minus_v1": delta_edit,
        }
    return result


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(delta: dict | None) -> str:
    if not delta:
        return "null"
    point, lower, upper = delta.get("point"), delta.get("lower"), delta.get("upper")
    if lower is None:
        return f"{_fmt(point)}"
    return f"{_fmt(point)} [{_fmt(lower)}, {_fmt(upper)}]"


def _signal(delta: dict | None) -> str:
    if not delta or delta.get("lower") is None:
        return "（无 CI）"
    if delta["lower"] > 0:
        return "CI 不含 0，正向"
    if delta["upper"] < 0:
        return "CI 不含 0，负向"
    return "CI 含 0（不显著）"


def _examples(records: list[dict], limit: int = 5) -> list[dict]:
    triggered = [record for record in records if record["trigger_any"] and record["ok_E_v2"]]
    examples = []
    for record in triggered[:limit]:
        examples.append({
            "sample_id": record["sample_id"],
            "fired": record["fired_slots"],
            "reference": None,  # filled from GT below
            "A": record["A_v2"],
            "C": record["C_v2"],
            "D": record["D_v2"],
            "E": record["E_v2"],
            "edits": {key: record[f"sim_{key}_v2"] for key in ("A", "C", "D", "E")},
        })
    gt = _load_gt()
    for example in examples:
        example["reference"] = gt[example["sample_id"]]["gt_sentence_official"]
    return examples


def write_report(records: list[dict], trigger_info: dict, cascade: dict, v1v2: dict) -> Path:
    v1_metrics = _read_metrics(V1_RUN_ID)
    v2_metrics = _read_metrics(V2_RUN_ID)
    v1_manifest = _read_manifest(V1_RUN_ID)
    v2_manifest = _read_manifest(V2_RUN_ID)
    v1_groups = v1_metrics["groups"]
    v2_groups = v2_metrics["groups"]
    v1_pairs = v1_metrics.get("paired_group_comparisons") or {}
    v2_pairs = v2_metrics.get("paired_group_comparisons") or {}
    v2_rows = _load_by_sample(V2_RUN_ID)
    v2_calls = [row for sample in v2_rows.values() for group, row in sample.items() if group != "A"]
    failures = [row for row in v2_calls if row.get("status") != "ok"]
    failure_by_group = Counter(row["group"] for row in failures)
    failure_reasons = Counter(str(row.get("error")) for row in failures)
    v2_latencies = [row["latency_s"] for row in v2_calls if isinstance(row.get("latency_s"), (int, float))]
    v2_tokens_total = sum((row["tokens"].get("total") or 0) for row in v2_calls)
    v2_tokens_prompt = sum((row["tokens"].get("in") or 0) for row in v2_calls)
    v2_tokens_completion = sum((row["tokens"].get("out") or 0) for row in v2_calls)
    v2_retries = sum(row.get("retries") or 0 for row in v2_calls)
    g = cascade["variants"]
    lines: list[str] = []
    lines.append("# G2 阶段 2-A3 · prompt v2 vs v1 + 选择性级联（50 条真实批，离线分析）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{common.utc_now()}")
    lines.append(f"- v1 运行：`{V1_RUN_ID}`（prompt `{v1_manifest['prompt']['version']}`，"
                 f"sha256 `{v1_manifest['prompt']['files']['json_sha256'][:16]}…`）")
    lines.append(f"- v2 运行：`{V2_RUN_ID}`（prompt `{v2_manifest['prompt']['version']}`，"
                 f"sha256 `{v2_manifest['prompt']['files']['json_sha256'][:16]}…`）")
    lines.append("- 后端：provider=wb-proxy，model=`deepseek-v4.1-flash`，temperature=0，串行，超时 90s；"
                 "v2 与 v1 的 max_tokens 均为 2048（v2 版本内固定，v1 为 run 级 override）")
    lines.append("- **探索性证据（50 条 retrospective 子集，非最终 G2 结论）**；级联为离线模拟，未新增 LLM 调用。")
    lines.append("")
    lines.append("## 1. 调用与失败（v2 批）")
    lines.append("")
    lines.append(f"- 真实调用 **{len(v2_calls)}** 次（50 × C/D/E），失败 **{len(failures)}** 次"
                 f"（按组 {dict(sorted(failure_by_group.items()))}），重试 {v2_retries} 次。")
    lines.append(f"- 失败原因（全部）：" + "; ".join(f"{reason} × {count}" for reason, count in failure_reasons.items()))
    lines.append("- 检查结论：E 组 5 次失败均为 **长推理截断**（`finish_reason=length`、completion 顶到 2048、`output_text` 为空），"
                 "不是 HTTP/网络错误；建议后续若继续 E 组实验，**给 E 单独 4096 max_tokens 预算**（本轮不重跑，按失败如实统计）。")
    lines.append("")
    lines.append("## 2. A / v1 / v2 质量对比（canonical p1-v1，GT ref_sentence_upstream）")
    lines.append("")
    lines.append("| 版本·组 | n | BLEU-4 | chrF | mean char edit | exact |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    def row(label: str, groups: dict, group: str) -> str:
        entry = groups.get(group) or {}
        corpus = entry.get("metrics") or {}
        return (f"| {label} | {entry.get('n_ok')} | {_fmt(corpus.get('bleu4_13a'))} | {_fmt(corpus.get('chrf'))} | "
                f"{_fmt(corpus.get('mean_char_edit_similarity'))} | {corpus.get('exact_match_count')} |")

    lines.append(row("A 基准（两版相同）", v2_groups, "A"))
    for group in ("C", "D", "E"):
        lines.append(row(f"v1 {group}", v1_groups, group))
    for group in ("C", "D", "E"):
        lines.append(row(f"**v2 {group}**", v2_groups, group))
    lines.append("")
    lines.append("v1 → v2 的成对提升（同一批样本、离线复算，signer 聚类 bootstrap）：")
    lines.append("")
    lines.append("| 组 | 共同 n | ΔBLEU v2−v1 (95%CI) | Δedit v2−v1 (95%CI) |")
    lines.append("|---|---:|---|---|")
    for group in ("C", "D", "E"):
        entry = v1v2.get(group)
        if not entry:
            continue
        lines.append(f"| {group} | {entry['n_pairs']} | {_ci(entry['delta_bleu_v2_minus_v1'])} | "
                     f"{_ci(entry['delta_edit_v2_minus_v1'])} |")
    lines.append("")
    lines.append("## 3. 关键配对（核心判据：D−C 与 E−D；补充 C−E）")
    lines.append("")
    lines.append("| 版本 | 配对 | n | ΔBLEU-4 (95%CI) | Δchar edit (95%CI) |")
    lines.append("|---|---|---:|---|---|")
    for label, pairs in (("v1", v1_pairs), ("**v2**", v2_pairs)):
        for pair in ("A_vs_C", "A_vs_D", "A_vs_E", "C_vs_D", "D_vs_E", "C_vs_E"):
            entry = pairs.get(pair)
            if not entry:
                continue
            lines.append(f"| {label} | {entry['first']}→{entry['second']} | {entry['n_pairs']} | "
                         f"{_ci(entry.get('delta_bleu_second_minus_first'))} | "
                         f"{_ci(entry.get('delta_edit_second_minus_first'))} |")
    lines.append("")
    lines.append(f"- **D−C**：v1 {_ci(v1_pairs.get('C_vs_D', {}).get('delta_bleu_second_minus_first'))}"
                 f"（{_signal(v1_pairs.get('C_vs_D', {}).get('delta_bleu_second_minus_first'))}）→ "
                 f"v2 {_ci(v2_pairs.get('C_vs_D', {}).get('delta_bleu_second_minus_first'))}"
                 f"（{_signal(v2_pairs.get('C_vs_D', {}).get('delta_bleu_second_minus_first'))}）")
    lines.append(f"- **E−D**：v1 {_ci(v1_pairs.get('D_vs_E', {}).get('delta_bleu_second_minus_first'))}"
                 f"（{_signal(v1_pairs.get('D_vs_E', {}).get('delta_bleu_second_minus_first'))}）→ "
                 f"v2 {_ci(v2_pairs.get('D_vs_E', {}).get('delta_bleu_second_minus_first'))}"
                 f"（{_signal(v2_pairs.get('D_vs_E', {}).get('delta_bleu_second_minus_first'))}）")
    lines.append(f"- **C−E**：v1 {_ci(v1_pairs.get('C_vs_E', {}).get('delta_bleu_second_minus_first'))}；"
                 f"v2 {_ci(v2_pairs.get('C_vs_E', {}).get('delta_bleu_second_minus_first'))}")
    lines.append("")
    lines.append("## 4. 选择性级联（offline，不新增调用）")
    lines.append("")
    lines.append(f"触发信号：E06 RuleJudge 对 **A 与 SLRT Gloss** 的确定性槽位冲突（数字/否定/人物/疑问），不读 GT。")
    lines.append("")
    lines.append(f"- 触发率（任一槽位冲突）：**{trigger_info['trigger_rate_any']:.2%}**"
                 f"（{trigger_info['triggered_n_any']}/{trigger_info['n_samples']}）；"
                 f"严重槽位（数字/否定/人物）：{trigger_info['trigger_rate_severe']:.2%}"
                 f"（{trigger_info['triggered_n_severe']}/{trigger_info['n_samples']}）；"
                 f"触发槽位分布 {trigger_info['fired_slots']}。")
    triggered_cmp = cascade["triggered_comparison_E_v1_vs_v2"]
    lines.append(f"- 触发子集上 E v1 vs v2：E_v1 BLEU {_fmt(triggered_cmp['E_v1_on_triggered'])}"
                 f"（失败率 {triggered_cmp['failure_rates']['E_v1']}）；E_v2 BLEU "
                 f"{_fmt(triggered_cmp['E_v2_on_triggered'])}（失败率 {triggered_cmp['failure_rates']['E_v2']}）；"
                 f"**级联采用 {triggered_cmp['chosen_for_cascade']}**（{triggered_cmp['reason']}）。")
    lines.append("")
    lines.append("| 策略 | 质量 BLEU-4 | chrF | edit | ΔBLEU vs A (95%CI) | Δedit vs A (95%CI) | LLM 调用数/占比 | tokens | P50/P95 延迟 | 失败(回退 A) |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|---|---:|")
    labels = {
        "pure_A_no_LLM": "纯 A（不调 LLM）",
        "all_D_v2": "全调 D(v2)",
        "all_E_v1": "全调 E(v1)",
        "all_E_v2": "全调 E(v2)",
        "cascade_ruleE_v2": "**级联：默认 A，触发→E(v2)**（主）",
        "cascade_ruleE_v1": "级联：默认 A，触发→E(v1)（对照）",
        "cascade_ruleD_v2": "级联：默认 A，触发→D(v2)（对照）",
    }
    for name in ("pure_A_no_LLM", "all_D_v2", "all_E_v1", "all_E_v2",
                 "cascade_ruleE_v2", "cascade_ruleE_v1", "cascade_ruleD_v2"):
        entry = g.get(name)
        if not entry:
            continue
        quality = entry["quality"]
        cost = entry["cost"]
        latency = "—" if not cost["llm_calls"] else f"{cost['latency_p50_s']}/{cost['latency_p95_s']} s"
        lines.append(f"| {labels[name]} | {_fmt(quality['bleu4_13a'])} | {_fmt(quality['chrf'])} | "
                     f"{_fmt(quality['mean_char_edit_similarity'])} | {_ci(entry['delta_bleu_vs_A'])} "
                     f"（{_signal(entry['delta_bleu_vs_A'])}） | {_ci(entry['delta_edit_vs_A'])} | "
                     f"{cost['llm_calls']}/{cost['call_rate']} | {cost['tokens_total']} | {latency} | "
                     f"{cost['failed_calls']}({entry['fallback_to_A_n']}) |")
    lines.append("")
    lines.append("> 成本为**本批已记录的真实 per-call 数据**按各策略实际使用/触发的调用重算（含失败调用的 token 消耗）；"
                 "级联未新增调用，属于离线反事实选择。P50/P95 为该策略实际使用调用的 per-call 延迟分位数，不是级联批次实测 wall time。")
    lines.append("")
    lines.append("## 5. v2 批 token 与延迟汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| LLM 调用数 | {len(v2_calls)} |")
    lines.append(f"| total tokens | {v2_tokens_total}（prompt {v2_tokens_prompt} / completion {v2_tokens_completion}） |")
    if v2_latencies:
        lines.append(f"| per-call latency P50 | {statistics.median(v2_latencies):.2f} s |")
        lines.append(f"| per-call latency P95 | {common.percentile(v2_latencies, 95):.2f} s |")
        lines.append(f"| per-call latency mean / max | {statistics.fmean(v2_latencies):.2f} s / {max(v2_latencies):.2f} s |")
    lines.append(f"| 重试 / 失败 | {v2_retries} / {len(failures)} |")
    lines.append("")
    lines.append("## 6. 质量抽查（v2，触发样本；EVAL ONLY 含 GT）")
    lines.append("")
    lines.append("| sample_id | 触发槽位 | GT（仅评测用） | A | C(v2) | D(v2) | E(v2) | edit A/C/D/E |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for example in _examples(records):
        edits = example["edits"]
        lines.append(f"| {example['sample_id']} | {','.join(example['fired'])} | {example['reference']} | "
                     f"{example['A']} | {example['C']} | {example['D']} | {example['E']} | "
                     + "/".join(f"{edits[key]:.2f}" for key in ("A", "C", "D", "E")) + " |")
    lines.append("")
    lines.append("## 7. 结论与 G2 建议（如实）")
    lines.append("")
    a_bleu = (v2_groups.get("A") or {}).get("metrics", {}).get("bleu4_13a")
    c_delta = (v2_groups.get("C") or {}).get("delta_bleu_vs_A") or {}
    d_delta = (v2_groups.get("D") or {}).get("delta_bleu_vs_A") or {}
    e_delta = (v2_groups.get("E") or {}).get("delta_bleu_vs_A") or {}
    cascade_entry = g.get("cascade_ruleE_v2") or {}
    lines.append(f"1. **质量**：A 基准 BLEU {_fmt(a_bleu)}；v2 的 C/D/E 分别 ΔBLEU "
                 f"{_ci(c_delta)} / {_ci(d_delta)} / {_ci(e_delta)}。"
                 "C 仍然显著低于 A（LLM 最小修正仍偏离原句）；D/E 的负向不再全部显著，但**没有任何组的点估计超过 A**。")
    lines.append(f"2. **核心判据**：D−C 与 E−D 在 v1、v2 两版中 CI 均含 0（无显著增益）；"
                 "C→E 亦无显著增益。**当前 prompt 条件下，LLM 融合（含证据约束）没有质量收益证据。**")
    lines.append(f"3. **成本-质量（G2 第三问）**：级联（默认 A，触发→E）触发率 {trigger_info['trigger_rate_any']:.2%}，"
                 f"BLEU {_fmt(cascade_entry.get('quality', {}).get('bleu4_13a'))}，"
                 f"ΔBLEU vs A {_ci(cascade_entry.get('delta_bleu_vs_A'))}"
                 f"（{_signal(cascade_entry.get('delta_bleu_vs_A'))}），"
                 f"LLM 调用 {cascade_entry.get('cost', {}).get('llm_calls')}/"
                 f"{cascade_entry.get('cost', {}).get('call_rate')}、tokens "
                 f"{cascade_entry.get('cost', {}).get('tokens_total')}；"
                 "即级联能显著降低调用成本，但**质量没有优于纯 A**（不显著负向），因此按当前证据**不值得**为质量付出该成本。")
    lines.append("4. **v1→v2 的效果**：prompt 迭代确实把 C/D 的 BLEU 从 v1 的 -4.55/-4.65 收窄到 -2.28/-2.64，"
                 "但方向仍为负；这验证了「减少重写」的假设，也说明**仅靠 prompt 不能把整句融合变成正增益**。"
                 "不再继续在同一 50 条上调 prompt（避免测试集过拟合）；若继续，必须换新样本/新 prompt 版本并重跑。")
    lines.append("5. **G2 建议**：融合**暂不进入主链路**。保留可插拔接口；优先路线是 ① 继续 B（SLRT G2T/SLT，仍 BLOCKED）"
                 "② 选择性/后处理式融合（仅在明确冲突时用证据修正，可先做 deterministic 规则，不依赖 LLM）"
                 "③ 新独立测试集 + 人工语义评审（Round-1 包已就绪，待用户回填）后再评估是否值得启用 LLM 融合。")
    lines.append("6. **还缺什么证据**：新独立测试集（非 retrospective_seen）、人工六槽位盲评（纠错/误改/严重错误率）、"
                 "E05 的最长推理预算（E 组需要 4096 tokens）与真实级联批次 wall time（本轮为离线估计）。")
    lines.append("")
    lines.append("## 8. 边界与安全")
    lines.append("")
    lines.append("1. 50 条为 E01 pilot_50（10 signer × 5），是 1086 的 retrospective 子集，不是新盲测；同 signer 内可能同源。")
    lines.append("2. 级联使用本批已产出的真实 E 文本做离线选择，**未新增调用**；成本为按已记录 per-call 重算的估计值。")
    lines.append("3. 密钥仅运行时内存解析，未写入任何文件/manifest/报告；manifest 只记录 `credentials_ref`；GT 只在评测侧 join。")
    lines.append("4. 本报告含 GT 例句（EVAL ONLY），不得回灌模型输入。")
    lines.append("5. v2 与 v1 使用同一批 50 样本，prompt 迭代是在已见数据上做的；不得把 v2 结果当作独立盲测提升。")
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main() -> int:
    records, trigger_info = build_records()
    cascade = cascade_variants(records)
    v1v2 = v1_vs_v2_deltas(records)
    payload = {
        "experiment": "G2-2A3-prompt-v2-cascade",
        "created_at": common.utc_now(),
        "v1_run_id": V1_RUN_ID,
        "v2_run_id": V2_RUN_ID,
        "trigger": trigger_info,
        "variants": cascade["variants"],
        "triggered_comparison_E_v1_vs_v2": cascade["triggered_comparison_E_v1_vs_v2"],
        "v1_vs_v2_paired_deltas": v1v2,
        "note": "offline analysis over already-produced real runs; no LLM calls; cost computed from recorded per-call rows",
    }
    common.write_json(OUT_JSON, payload)
    report = write_report(records, trigger_info, cascade, v1v2)
    print(json.dumps({
        "trigger": trigger_info,
        "variants": {name: {
            "n": entry["n"],
            "bleu4_13a": entry["quality"]["bleu4_13a"],
            "delta_bleu_vs_A": entry["delta_bleu_vs_A"],
            "llm_calls": entry["cost"]["llm_calls"],
            "call_rate": entry["cost"]["call_rate"],
            "tokens_total": entry["cost"]["tokens_total"],
            "latency_p50_s": entry["cost"]["latency_p50_s"],
            "latency_p95_s": entry["cost"]["latency_p95_s"],
            "fallback_to_A_n": entry["fallback_to_A_n"],
        } for name, entry in cascade["variants"].items()},
        "v1_vs_v2": v1v2,
        "report": str(report),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
