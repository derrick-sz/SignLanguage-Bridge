"""G2 phase-2A3: prompt v2_zh real batch + selective-cascade offline analysis.

Runs the same 50 pilot samples through prompt v2_zh (C/D/E, real wb-proxy
deepseek-v4.1-flash, serial, max_tokens=2048 from the v2 spec), evaluates with the
canonical pipeline, and builds the offline selective-cascade policy
``A unless RuleJudge trigger -> E(v2)`` using the E06 slot-conflict judge
(no extra LLM calls).

Security: the API key is resolved at runtime in memory only; never printed or
written to any file/manifest/report.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict

from experiments.p1 import common
from experiments.p1.eval import metrics as M
from experiments.p1.e02.llm_fusion import evaluate as evaluate_module
from experiments.p1.e02.llm_fusion import harness
from experiments.p1.e02.llm_fusion.backends import get_backend

RUN_ID = "g2a-wbproxy-50-v2prompt-20261026"
V1_RUN_ID = "g2a-wbproxy-50-v2-20261026"
STUB_RUNS = ["g2a-stub-selftest-15", "g2a-stub-dryrun-1086"]
PROMPT_VERSION = "v2_zh"
PILOT_50 = common.P1 / "datasets" / "pilot_50.csv"
GT_PATH = common.REFERENCES / "gt_1086.csv"
REPORT = harness.FUSION_DIR / "report_prompt_v2.md"
ABLATION_CSV = harness.FUSION_DIR / "fusion_ablation_g2.csv"
BASE_URL = "http://127.0.0.1:8788/v1"
MODEL = "deepseek-v4.1-flash"
SEVERE_SLOTS = ("numbers", "negation", "person")
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20261026


def _backend_kwargs() -> dict:
    return {
        "base_url": BASE_URL,
        "model": MODEL,
        "credentials_ref": "WB_PROXY_API_KEY",
        "timeout_seconds": 90,
        "enabled": True,
    }


def run_batch() -> dict:
    backend = get_backend("openai_compat", **_backend_kwargs())
    description = backend.describe()
    print(json.dumps({
        "provider": description.get("provider"),
        "model": description.get("model"),
        "endpoint_probe": description.get("endpoint_probe"),
        "credentials_present": description.get("credentials_present"),
        "credentials_ref": description.get("credentials_ref"),
        "status": description.get("status"),
    }, ensure_ascii=False, indent=2))
    if not description.get("available"):
        raise SystemExit(f"real backend not available: {description}")
    sample_ids = [row["sample_id"] for row in common.read_csv_rows(PILOT_50)]
    return harness.run(
        backend_name="openai_compat",
        prompt_version=PROMPT_VERSION,
        groups=("A", "C", "D", "E"),
        sample_ids=sample_ids,
        run_id=RUN_ID,
        allow_real=True,
        overwrite=True,
        data_status="REAL_EXPLORATORY_50_PROMPT_V2",
        backend_kwargs=_backend_kwargs(),
        concurrency=1,
    )


# --------------------------------------------------------------------------
# helpers for the offline cascade analysis
# --------------------------------------------------------------------------
def _load_outputs(run_id: str) -> list[dict]:
    return common.read_jsonl(harness.RUNS_DIR / run_id / "fusion_outputs.jsonl")


def _load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def _sentence_edit(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def _rule_judge_trigger(unisign_text: str, gloss_text: str) -> dict:
    """Reuse the E06 RuleJudge slot extraction as the deterministic cascade trigger."""

    from experiments.p1.e06.build_e06 import extract_slots, slot_conflicts

    slots_a = extract_slots(unisign_text)
    slots_b = extract_slots(gloss_text)
    conflicts = slot_conflicts(slots_a, slots_b)
    fired = sorted({conflict["slot"] for conflict in conflicts})
    return {
        "trigger_any": bool(conflicts),
        "trigger_severe": any(slot in SEVERE_SLOTS for slot in fired),
        "fired_slots": fired,
        "n_conflicts": len(conflicts),
    }


def _paired_bootstrap(records: list[dict], statistic, *, iterations: int = BOOTSTRAP_ITERATIONS,
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
        return {"point": point, "lower": None, "upper": None}
    return {
        "point": round(point, 4),
        "lower": round(values[int(0.025 * (len(values) - 1))], 4),
        "upper": round(values[int(0.975 * (len(values) - 1))], 4),
        "iterations": iterations,
        "seed": seed,
        "method": "cluster_bootstrap_by_signer (paired)",
    }


def _bleu_reference(pairs: list[tuple[str, str]]) -> float:
    return M.corpus_metrics_cslt(pairs)["bleu4_13a"]


def cascade_analysis(run_id: str) -> dict:
    rows = _load_outputs(run_id)
    gt = _load_gt()
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["group"]] = row
    samples = [sample for sample in sorted(by_sample) if {"A", "E"} <= set(by_sample[sample])]
    records: list[dict] = []
    trigger_counts: Counter = Counter()
    fired_by_slot: Counter = Counter()
    for sample in samples:
        if sample not in gt:
            continue
        reference = gt[sample]["ref_sentence_upstream"]
        a_text = by_sample[sample]["A"].get("output_text", "")
        e_text = by_sample[sample]["E"].get("output_text", "")
        gloss_text = by_sample[sample]["E"].get("evidence", {}).get("slrt_gloss_text", "")
        trigger = _rule_judge_trigger(a_text, gloss_text)
        if trigger["trigger_any"]:
            trigger_counts["any"] += 1
        if trigger["trigger_severe"]:
            trigger_counts["severe"] += 1
        for slot in trigger["fired_slots"]:
            fired_by_slot[slot] += 1
        records.append({
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "reference": reference,
            "hyp_a": a_text,
            "hyp_e": e_text,
            "sim_a": _sentence_edit(reference, a_text),
            "sim_e": _sentence_edit(reference, e_text),
            "trigger_any": trigger["trigger_any"],
            "trigger_severe": trigger["trigger_severe"],
            "fired_slots": trigger["fired_slots"],
        })
    n = len(records)
    if n == 0:
        raise SystemExit("no paired A/E samples for cascade analysis")

    def policy_metrics(selector) -> dict:
        final_pairs = []
        records_policy = []
        for record in records:
            use_e = selector(record)
            hypothesis = record["hyp_e"] if use_e else record["hyp_a"]
            final_pairs.append((record["reference"], hypothesis))
            records_policy.append({
                **record,
                "hyp_policy": hypothesis,
                "sim_policy": record["sim_e"] if use_e else record["sim_a"],
                "used_e": bool(use_e),
            })
        corpus = M.corpus_metrics_cslt(final_pairs)

        def delta_bleu(items: list[dict]) -> float:
            import sacrebleu

            refs = [M.character_separated(item["reference"]) for item in items]
            hyps_policy = [M.character_separated(item["hyp_policy"]) for item in items]
            hyps_a = [M.character_separated(item["hyp_a"]) for item in items]
            return (sacrebleu.corpus_bleu(hyps_policy, [refs], tokenize="13a").score
                    - sacrebleu.corpus_bleu(hyps_a, [refs], tokenize="13a").score)

        def delta_edit(items: list[dict]) -> float:
            return statistics.fmean(item["sim_policy"] for item in items) - statistics.fmean(
                item["sim_a"] for item in items)

        def delta_vs_all_e(items: list[dict]) -> float:
            import sacrebleu

            refs = [M.character_separated(item["reference"]) for item in items]
            hyps_policy = [M.character_separated(item["hyp_policy"]) for item in items]
            hyps_e = [M.character_separated(item["hyp_e"]) for item in items]
            return (sacrebleu.corpus_bleu(hyps_policy, [refs], tokenize="13a").score
                    - sacrebleu.corpus_bleu(hyps_e, [refs], tokenize="13a").score)

        used = sum(1 for item in records_policy if item["used_e"])
        delta_bleu_ci = _paired_bootstrap(records_policy, delta_bleu)
        delta_edit_ci = _paired_bootstrap(records_policy, delta_edit)
        delta_vs_e_ci = _paired_bootstrap(records_policy, delta_vs_all_e)
        return {
            "n": n,
            "used_e_n": used,
            "policy_call_rate": round(used / n, 4),
            "bleu4_13a": corpus["bleu4_13a"],
            "chrf": corpus["chrf"],
            "mean_char_edit_similarity": corpus["mean_char_edit_similarity"],
            "exact_match_count": corpus["exact_match_count"],
            "delta_bleu_vs_A": delta_bleu_ci,
            "delta_edit_vs_A": delta_edit_ci,
            "delta_bleu_vs_all_E": delta_vs_e_ci,
        }

    policies = {
        "always_A": policy_metrics(lambda record: False),
        "always_E": policy_metrics(lambda record: True),
        "A_unless_rule_any": policy_metrics(lambda record: record["trigger_any"]),
        "A_unless_rule_severe": policy_metrics(lambda record: record["trigger_severe"]),
    }

    # measured E per-call cost (successful calls only) for estimated cascade cost
    e_rows = [row for row in rows if row["group"] == "E" and row.get("status") == "ok"]
    per_call_tokens = [row["tokens"]["total"] for row in e_rows if row.get("tokens", {}).get("total")]
    per_call_latency_s = [row["latency_s"] for row in e_rows if isinstance(row.get("latency_s"), (int, float))]
    mean_tokens = statistics.fmean(per_call_tokens) if per_call_tokens else None
    mean_latency_s = statistics.fmean(per_call_latency_s) if per_call_latency_s else None
    for name, entry in policies.items():
        used = entry["used_e_n"]
        entry["estimated_tokens"] = round(used * mean_tokens, 1) if mean_tokens else None
        entry["estimated_latency_s"] = round(used * mean_latency_s, 2) if mean_latency_s else None
        entry["savings_vs_always_E_calls"] = round(1 - used / n, 4)
    return {
        "n_samples": n,
        "trigger_rate_any": round(trigger_counts["any"] / n, 4),
        "trigger_rate_severe": round(trigger_counts["severe"] / n, 4),
        "triggered_n_any": trigger_counts["any"],
        "triggered_n_severe": trigger_counts["severe"],
        "fired_slots": dict(sorted(fired_by_slot.items())),
        "per_call_cost_E_measured": {
            "n_calls": len(e_rows),
            "mean_total_tokens": round(mean_tokens, 1) if mean_tokens else None,
            "mean_latency_s": round(mean_latency_s, 3) if mean_latency_s else None,
        },
        "policies": policies,
        "note": "cascade uses measured v2 E outputs for triggered samples and A otherwise; no new LLM calls; "
                "estimated tokens/latency = triggered_n x measured per-call mean (not a measured cascade batch)",
    }


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
    point = delta.get("point")
    lower, upper = delta.get("lower"), delta.get("upper")
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


def _groups_row(label: str, groups: dict, group: str) -> str:
    entry = groups.get(group, {})
    corpus = entry.get("metrics") or {}
    return (f"| {label} | {entry.get('n_ok')} | {_fmt(corpus.get('bleu4_13a'))} | {_fmt(corpus.get('chrf'))} | "
            f"{_fmt(corpus.get('mean_char_edit_similarity'))} | {corpus.get('exact_match_count')} | "
            f"{_ci(entry.get('delta_bleu_vs_A'))} | {_ci(entry.get('delta_edit_vs_A'))} |")


def _paired_rows(paired: dict, prefix: str) -> list[str]:
    lines = []
    for pair in ("A_vs_C", "A_vs_D", "A_vs_E", "C_vs_D", "D_vs_E", "C_vs_E"):
        entry = paired.get(pair)
        if not entry:
            continue
        lines.append(f"| {prefix} {entry['first']} → {entry['second']} | {entry['n_pairs']} | "
                     f"{_ci(entry.get('delta_bleu_second_minus_first'))} | "
                     f"{_ci(entry.get('delta_edit_second_minus_first'))} |")
    return lines


def _examples(run_id: str, cascade: dict, limit: int = 6) -> list[dict]:
    rows = _load_outputs(run_id)
    gt = _load_gt()
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["group"]] = row
    triggered = []
    for sample, groups in by_sample.items():
        if not {"A", "E"} <= set(groups) or sample not in gt:
            continue
        trigger = _rule_judge_trigger(groups["A"].get("output_text", ""),
                                      groups["E"].get("evidence", {}).get("slrt_gloss_text", ""))
        if not trigger["trigger_any"]:
            continue
        reference = gt[sample]["ref_sentence_upstream"]
        triggered.append({
            "sample_id": sample,
            "reference": gt[sample]["gt_sentence_official"],
            "A": groups["A"].get("output_text", ""),
            "C": groups.get("C", {}).get("output_text", ""),
            "D": groups.get("D", {}).get("output_text", ""),
            "E": groups["E"].get("output_text", ""),
            "E_reason": groups["E"].get("output_reason"),
            "fired": trigger["fired_slots"],
            "edits": {key: _sentence_edit(reference, groups[key].get("output_text", ""))
                      for key in ("A", "C", "D", "E") if key in groups},
            "delta_E_minus_A": _sentence_edit(reference, groups["E"].get("output_text", ""))
            - _sentence_edit(reference, groups["A"].get("output_text", "")),
        })
    triggered.sort(key=lambda item: item["sample_id"])
    # mix: largest E-A gains and largest E-A losses, then fill
    picked = {}
    for reverse in (True, False):
        for item in sorted(triggered, key=lambda item: item["delta_E_minus_A"], reverse=reverse):
            if len(picked) >= limit:
                break
            picked[item["sample_id"]] = item
    return list(picked.values())


def write_report(run_id: str, evaluation: dict, cascade: dict) -> __import__("pathlib").Path:
    metrics = evaluation["metrics"]
    groups = metrics["groups"]
    paired = metrics.get("paired_group_comparisons") or {}
    v1_metrics = common.read_json(harness.RUNS_DIR / V1_RUN_ID / "metrics.json")
    v1_groups = v1_metrics["groups"]
    v1_paired = v1_metrics.get("paired_group_comparisons") or {}
    manifest = common.read_json(harness.RUNS_DIR / run_id / "manifest.json")
    rows = _load_outputs(run_id)
    calls = [row for row in rows if row["group"] != "A"]
    failures = [row for row in calls if row.get("status") != "ok"]
    latencies = [row["latency_s"] for row in calls if isinstance(row.get("latency_s"), (int, float))]
    prompt_tokens = sum((row["tokens"].get("in") or 0) for row in calls)
    completion_tokens = sum((row["tokens"].get("out") or 0) for row in calls)
    total_tokens = sum((row["tokens"].get("total") or 0) for row in calls)
    retries = sum(row.get("retries") or 0 for row in calls)
    lines: list[str] = []
    lines.append("# G2 阶段 2-A3 · prompt v2_zh 50 条真实对照 + 选择性级联分析")
    lines.append("")
    lines.append(f"- run_id：`{run_id}`；生成时间（UTC）：{common.utc_now()}")
    lines.append("- **探索性证据（50 条 retrospective 子集）；非最终 G2 结论**（最终需新独立测试集 + 人工评审）")
    lines.append(f"- 后端：provider=wb-proxy，model=`{MODEL}`，endpoint=`{BASE_URL}/chat/completions`，"
                 "credentials_ref=`WB_PROXY_API_KEY@dsh-credentials`（密钥仅内存，未落盘）")
    lines.append(f"- prompt：**{manifest['prompt']['version']}**（json sha256 `{manifest['prompt']['files']['json_sha256'][:16]}…`，"
                 f"md sha256 `{(manifest['prompt']['files']['md_sha256'] or '')[:16]}…`）；"
                 f"decoding：temperature={manifest['model']['decoding']['temperature']}、"
                 f"max_tokens={manifest['model']['decoding']['max_tokens']}、top_p={manifest['model']['decoding']['top_p']}")
    lines.append("- 执行：严格串行（保护 DSH 代理）；超时 90s；失败重试 2 次指数退避；样本 = E01 `pilot_50.csv`（10 signer × 5）。")
    lines.append("")
    lines.append("## 0. v1 → v2 的 prompt 改动（基于 v1 证据，只改指令不改任务）")
    lines.append("")
    lines.append("- v1 观察到 C 显著低于 A（ΔBLEU -4.55，CI 不含 0）：无条件润色会改掉正确事实 / 扩写。")
    lines.append("- v2：C = 仅最小必要修正（否则原样输出、保留全部事实）；D = 以 A 原句为主、仅明确冲突槽位按 Gloss 修正；"
                 "E = D + 槽位对照 + 一行 `理由：`；系统提示增加「优先保留输入事实，不猜测/不扩写」。")
    lines.append("- v1_zh 已用于真实运行，保持不动；v2 为独立版本文件（见 `prompts/v2_zh.*`）。")
    lines.append("")
    lines.append("## 1. 调用与失败")
    lines.append("")
    lines.append(f"- 真实 LLM 调用 **{len(calls)}** 次（50 × C/D/E），成功 {len(calls) - len(failures)}，失败 **{len(failures)}**；"
                 f"重试合计 {retries} 次。")
    if failures:
        by_group = Counter(row["group"] for row in failures)
        lines.append(f"- 失败明细：按组 {dict(sorted(by_group.items()))}；原因："
                     + "; ".join(f"{row['group']}/{row['sample_id']}: {str(row.get('error'))[:80]}" for row in failures[:5]))
    lines.append("")
    lines.append("## 2. 三列对比：A / v1 / v2（canonical p1-v1，n=50 样本）")
    lines.append("")
    lines.append("| 版本·组 | n | BLEU-4 | chrF | mean char edit | exact | ΔBLEU vs A (95%CI) | Δedit vs A (95%CI) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    lines.append(_groups_row("A（基准，两版相同）", groups, "A"))
    for group in ("C", "D", "E"):
        lines.append(_groups_row(f"v1 {group}", v1_groups, group))
    for group in ("C", "D", "E"):
        lines.append(_groups_row(f"**v2 {group}**", groups, group))
    lines.append("")
    lines.append("## 3. 关键配对（核心判据）")
    lines.append("")
    lines.append("| 配对 | n | ΔBLEU-4 (95%CI) | Δchar edit (95%CI) |")
    lines.append("|---|---:|---|---|")
    lines.extend(_paired_rows(v1_paired, "v1"))
    lines.extend(_paired_rows(paired, "**v2**"))
    lines.append("")
    v1_dc = v1_paired.get("C_vs_D", {}).get("delta_bleu_second_minus_first")
    v2_dc = paired.get("C_vs_D", {}).get("delta_bleu_second_minus_first")
    v1_ed = v1_paired.get("D_vs_E", {}).get("delta_bleu_second_minus_first")
    v2_ed = paired.get("D_vs_E", {}).get("delta_bleu_second_minus_first")
    lines.append(f"- **D−C**：v1 {_ci(v1_dc)}（{_signal(v1_dc)}）→ v2 {_ci(v2_dc)}（{_signal(v2_dc)}）")
    lines.append(f"- **E−D**：v1 {_ci(v1_ed)}（{_signal(v1_ed)}）→ v2 {_ci(v2_ed)}（{_signal(v2_ed)}）")
    lines.append("")
    lines.append("## 4. 选择性级联（offline，不新增调用）")
    lines.append("")
    lines.append("触发信号：E06 RuleJudge 对 A 与 SLRT Gloss 的确定性槽位冲突检测（数字/否定/人物/疑问），不读 GT、不调模型。")
    lines.append("")
    lines.append(f"- 触发率（任一槽位冲突）：**{cascade['trigger_rate_any']:.4f}**（{cascade['triggered_n_any']}/{cascade['n_samples']}）；"
                 f"严重槽位（数字/否定/人物）：**{cascade['trigger_rate_severe']:.4f}**（{cascade['triggered_n_severe']}/{cascade['n_samples']}）。")
    lines.append(f"- 触发槽位分布：{cascade['fired_slots']}")
    lines.append(f"- 单次 E 调用实测成本（成功样本）：mean total tokens **{cascade['per_call_cost_E_measured']['mean_total_tokens']}**、"
                 f"mean latency **{cascade['per_call_cost_E_measured']['mean_latency_s']} s**。")
    lines.append("")
    lines.append("| 策略 | 有效 n | LLM 调用率 | 估计 tokens | 估计延迟 | BLEU-4 | chrF | edit | ΔBLEU vs A (95%CI) | ΔBLEU vs 全调 E (95%CI) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for name, entry in cascade["policies"].items():
        lines.append(f"| {name} | {entry['n']} | {entry['policy_call_rate']} | {entry['estimated_tokens']} | "
                     f"{entry['estimated_latency_s']} s | {_fmt(entry['bleu4_13a'])} | {_fmt(entry['chrf'])} | "
                     f"{_fmt(entry['mean_char_edit_similarity'])} | {_ci(entry['delta_bleu_vs_A'])} | "
                     f"{_ci(entry['delta_bleu_vs_all_E'])} |")
    lines.append("")
    lines.append("> 估计成本 = 触发数 × 实测单次 E 均值（**不是实测级联批次**）；级联使用本批已产出的真实 E 文本做离线选择，未新增调用。")
    lines.append("")
    lines.append("## 5. 成本（v2 真实批）")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| LLM 调用数 | {len(calls)} |")
    lines.append(f"| prompt tokens | {prompt_tokens} |")
    lines.append(f"| completion tokens | {completion_tokens} |")
    lines.append(f"| total tokens | {total_tokens} |")
    lines.append(f"| 平均 tokens/调用 | {(total_tokens / len(calls)):.1f} |" if calls else "| 平均 tokens/调用 | null |")
    if latencies:
        lines.append(f"| latency P50 | {statistics.median(latencies):.2f} s |")
        lines.append(f"| latency P95 | {common.percentile(latencies, 95):.2f} s |")
        lines.append(f"| latency mean | {statistics.fmean(latencies):.2f} s |")
    lines.append(f"| 重试/失败 | {retries} / {len(failures)} |")
    lines.append("")
    lines.append("## 6. 质量抽查（v2，触发样本；EVAL ONLY 含 GT）")
    lines.append("")
    lines.append("| sample_id | 触发槽位 | GT（仅评测用） | A | C(v2) | D(v2) | E(v2) | E 理由 | edit A/C/D/E |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for example in _examples(run_id, cascade):
        edits = example["edits"]
        lines.append(f"| {example['sample_id']} | {','.join(example['fired'])} | {example['reference']} | {example['A']} | "
                     f"{example['C']} | {example['D']} | {example['E']} | {example['E_reason'] or ''} | "
                     + "/".join(f"{edits[key]:.2f}" if key in edits else "-" for key in ("A", "C", "D", "E")) + " |")
    lines.append("")
    lines.append("## 7. 结论（必须如实）")
    lines.append("")
    # evidence-driven statements
    c_delta = groups.get("C", {}).get("delta_bleu_vs_A") or {}
    e_delta = groups.get("E", {}).get("delta_bleu_vs_A") or {}
    policy = cascade["policies"]["A_unless_rule_any"]
    lines.append(f"1. v2 的 C 相对 A：ΔBLEU {_ci(c_delta)}（{_signal(c_delta)}）。"
                 + ("'最小修正'仍未摆脱负向（LLM 依旧改写原句）→ 当前 prompt 条件下 LLM 润色无质量收益。"
                    if c_delta.get("upper") is not None and c_delta["upper"] < 0 else
                    "负向不再显著，但也没有正增益；需更大样本/人工评审判断。"))
    lines.append(f"2. v2 的 D−C：{_ci(v2_dc)}（{_signal(v2_dc)}）；E−D：{_ci(v2_ed)}（{_signal(v2_ed)}）。")
    if (v2_dc.get("lower") is not None and v2_dc["upper"] < 0):
        lines.append("   Gloss 证据在字面指标上仍为负向 → 用 gloss join 做事实修正并不改善字面一致性（gloss join 本身不是流畅译文）。")
    elif (v2_dc.get("lower") is not None and v2_dc["lower"] <= 0 <= v2_dc["upper"]
          and v2_ed.get("lower") is not None and v2_ed["lower"] <= 0 <= v2_ed["upper"]):
        lines.append("   Gloss 证据与证据约束均**无显著字面增益**：如实记录为「当前 prompt 条件下 LLM 融合无质量收益证据」；"
                     "不允许为了凑出正增益而继续在同一测试集上调 prompt。")
    lines.append(f"3. 选择性级联（A unless rule trigger → E(v2)）：触发率 {cascade['trigger_rate_any']:.2%}，"
                 f"BLEU {_fmt(policy['bleu4_13a'])}（Δ vs A {_ci(policy['delta_bleu_vs_A'])}，{_signal(policy['delta_bleu_vs_A'])}），"
                 f"估计成本较全调 E 节省 {policy['savings_vs_always_E_calls']:.2%} 调用。"
                 "即使级联只调用少部分样本，其质量仍由 E 的质量决定——E 无增益时级联也不会有增益（成本问题与质量问题是分开的两问）。")
    lines.append("4. **成本 vs 质量**：v2 每 50 条 C/D/E 的 total tokens 见 §5；若扩到 1086，按当前均值线性外推约 "
                 f"{(total_tokens / 50) * 1086 / 1000:.0f}k tokens（估算，未实测）。当前证据不支持扩量。")
    lines.append("5. **探索性边界**：50 条 retrospective_seen 子集、同一批样本上做了一次 v1→v2 prompt 迭代，"
                 "**不是独立盲测、不能声称泛化提升**；最终 G2 结论需新独立集 + 人工评审（Round-1 包）。")
    lines.append("6. 停止条件（预先声明）：若 v2 与 v1 均无 D−C/E−D 正增益且 C 仍劣于 A，则结论为"
                 "「以当前 LLM prompt 方案做整句融合无质量收益」，应转向 B（G2T/SLT，仍 BLOCKED）或选择性/后处理路线，而不是继续调 prompt。")
    lines.append("")
    lines.append("## 8. 边界与安全")
    lines.append("")
    lines.append("1. A/A2/B0 为 E02 冻结历史预测；C/D/E 为本轮真实 LLM 生成（wb-proxy/deepseek-v4.1-flash）。")
    lines.append("2. 密钥仅运行时内存解析，未打印/未写入任何文件/JSONL/manifest/日志/报告；manifest 仅 `credentials_ref`。")
    lines.append("3. GT 只在 `evaluate.py` join；prompt 输入经 gt_guard 与白名单分数列约束（0 泄漏）。")
    lines.append("4. 本报告含 GT 例句（EVAL ONLY），不得回灌模型输入。")
    lines.append("5. 级联为离线模拟（使用本批已产出的真实 E 文本），估算成本不是实测级联延迟。")
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main() -> int:
    result = run_batch()
    evaluation = evaluate_module.evaluate_run(RUN_ID, iterations=BOOTSTRAP_ITERATIONS,
                                               seed=evaluate_module.BOOTSTRAP_SEED, write=True)
    evaluate_module.write_ablation_csv(STUB_RUNS + [V1_RUN_ID, RUN_ID])
    # Canonical v2 analysis + report live in analyze_prompt_v2.py; this keeps a
    # batch rerun reproducible end-to-end (no extra LLM calls in the analysis).
    from experiments.p1.e02.llm_fusion import analyze_prompt_v2

    return analyze_prompt_v2.main()
    print(json.dumps({
        "run_id": RUN_ID,
        "status": metrics.get("status"),
        "data_status": metrics.get("data_status"),
        "n_samples": metrics.get("n_samples"),
        "groups": {group: {
            "n": entry.get("n_ok"),
            "bleu4_13a": (entry.get("metrics") or {}).get("bleu4_13a"),
            "chrf": (entry.get("metrics") or {}).get("chrf"),
            "mean_char_edit_similarity": (entry.get("metrics") or {}).get("mean_char_edit_similarity"),
        } for group, entry in metrics["groups"].items()},
        "pairs": {pair: {
            "n": entry["n_pairs"],
            "delta_bleu": entry["delta_bleu_second_minus_first"],
            "delta_edit": entry["delta_edit_second_minus_first"],
        } for pair, entry in (metrics.get("paired_group_comparisons") or {}).items()},
        "cascade": {
            "trigger_rate_any": cascade["trigger_rate_any"],
            "policies": {name: {
                "call_rate": entry["policy_call_rate"],
                "bleu4_13a": entry["bleu4_13a"],
                "delta_bleu_vs_A": entry["delta_bleu_vs_A"],
                "estimated_tokens": entry["estimated_tokens"],
                "estimated_latency_s": entry["estimated_latency_s"],
            } for name, entry in cascade["policies"].items()},
        },
        "report": str(report),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
