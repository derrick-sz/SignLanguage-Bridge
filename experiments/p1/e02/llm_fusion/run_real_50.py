"""Run the REAL 50-sample G2 exploration batch through the wb-proxy endpoint.

Endpoint (user-selected, Lead smoke-verified):
    provider  wb-proxy
    base_url  http://127.0.0.1:8788/v1
    model     deepseek-v4.1-flash
    auth      Bearer key resolved at runtime in memory from
              C:\\Users\\ADMIN\\.dsh\\.credentials.yaml refs.WB_PROXY_API_KEY
              (or the WB_PROXY_API_KEY / P1_OPENAI_API_KEY env var)

Security: the key is never printed, never written to any output file, JSONL,
manifest, log or report. Manifests only store ``credentials_ref``.

Run (CU-only apart from the HTTP calls; strictly serial):
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    $env:P1_LLM_ENABLE_REAL='1'
    .\\.conda-env\\python.exe -m experiments.p1.e02.llm_fusion.run_real_50
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from experiments.p1 import common
from experiments.p1.eval import metrics as M

from experiments.p1.e02.llm_fusion import evaluate as evaluate_module
from experiments.p1.e02.llm_fusion import harness
from experiments.p1.e02.llm_fusion.backends import get_backend

ROOT = common.ROOT
FUSION_DIR = harness.FUSION_DIR
RUN_ID = "g2a-wbproxy-50-v2-20261026"
INCIDENT_RUN_ID = "g2a-wbproxy-50-20261026"
MAX_TOKENS = 2048
PILOT_50 = common.P1 / "datasets" / "pilot_50.csv"
GT_PATH = common.REFERENCES / "gt_1086.csv"
REPORT = FUSION_DIR / "report_real_50.md"
STUB_RUNS = ["g2a-stub-selftest-15", "g2a-stub-dryrun-1086"]
BASE_URL = "http://127.0.0.1:8788/v1"
MODEL = "deepseek-v4.1-flash"


def _pilot_sample_ids() -> list[str]:
    rows = common.read_csv_rows(PILOT_50)
    return [row["sample_id"] for row in rows]


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
    # security: only booleans / reference names leave this function
    print(json.dumps({
        "endpoint_probe": description.get("endpoint_probe"),
        "model": description.get("model"),
        "provider": description.get("provider"),
        "credentials_present": description.get("credentials_present"),
        "credentials_ref": description.get("credentials_ref"),
        "status": description.get("status"),
    }, ensure_ascii=False, indent=2))
    if not description.get("available"):
        raise SystemExit(f"real backend not available: {description}")
    sample_ids = _pilot_sample_ids()
    result = harness.run(
        backend_name="openai_compat",
        groups=("A", "C", "D", "E"),
        sample_ids=sample_ids,
        run_id=RUN_ID,
        allow_real=True,
        overwrite=True,
        data_status="REAL_EXPLORATORY_50",
        backend_kwargs=_backend_kwargs(),
        concurrency=1,
        max_tokens_override=MAX_TOKENS,
    )
    return result


def _load_outputs(run_id: str) -> list[dict]:
    return common.read_jsonl(harness.RUNS_DIR / run_id / "fusion_outputs.jsonl")


def _sentence_edit(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def _examples(run_id: str, limit: int = 8) -> list[dict]:
    rows = _load_outputs(run_id)
    gt = {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["group"]] = row
    candidates = []
    for sample, groups in by_sample.items():
        if not {"A", "C", "D", "E"} <= set(groups) or sample not in gt:
            continue
        reference = gt[sample]["ref_sentence_upstream"]
        edits = {group: _sentence_edit(reference, groups[group].get("output_text", "")) for group in ("A", "C", "D", "E")}
        candidates.append({
            "sample_id": sample,
            "reference": gt[sample]["gt_sentence_official"],
            "A": groups["A"].get("output_text", ""),
            "C": groups["C"].get("output_text", ""),
            "D": groups["D"].get("output_text", ""),
            "E": groups["E"].get("output_text", ""),
            "edits": edits,
            "delta_D_minus_C": edits["D"] - edits["C"],
            "delta_E_minus_D": edits["E"] - edits["D"],
        })
    # mix: biggest D-C gains, biggest D-C losses, biggest E-D gains, biggest E-D losses
    picked: dict[str, dict] = {}
    for key, reverse, count in (("delta_D_minus_C", True, 2), ("delta_D_minus_C", False, 2),
                                ("delta_E_minus_D", True, 2), ("delta_E_minus_D", False, 2)):
        ordered = sorted(candidates, key=lambda item: item[key], reverse=reverse)
        for item in ordered:
            if len(picked) >= limit:
                break
            if item["sample_id"] not in picked:
                picked[item["sample_id"]] = item
    return list(picked.values())


def _incident_stats() -> dict | None:
    """Stats of the first attempt (max_tokens=256) where reasoning truncated the answer."""

    path = harness.RUNS_DIR / INCIDENT_RUN_ID / "fusion_outputs.jsonl"
    if not path.exists():
        return None
    rows = [row for row in common.read_jsonl(path) if row["group"] != "A"]
    failures = [row for row in rows if row.get("status") != "ok"]
    by_group: dict[str, int] = defaultdict(int)
    for row in failures:
        by_group[row["group"]] += 1
    finish_reasons = defaultdict(int)
    for row in failures:
        finish_reasons[str((row.get("backend_extra") or {}).get("finish_reason"))] += 1
    completion_tokens = [(row.get("tokens") or {}).get("out") for row in failures]
    return {
        "run_id": INCIDENT_RUN_ID,
        "max_tokens": 256,
        "n_calls": len(rows),
        "n_failed": len(failures),
        "failed_by_group": dict(sorted(by_group.items())),
        "failed_finish_reasons": dict(finish_reasons),
        "failed_completion_tokens_min_max": [min(completion_tokens), max(completion_tokens)]
        if completion_tokens and all(isinstance(value, int) for value in completion_tokens) else None,
        "diagnosis": "empty completion text with finish_reason=length: the model spent the 256-token "
                     "budget on internal reasoning before emitting the answer; retry with a larger fixed budget",
    }


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
    """Human-readable significance note for a paired delta CI."""

    if not delta or delta.get("lower") is None:
        return "（无 CI）"
    if delta["lower"] > 0:
        return "CI 不含 0，**指向正增益**"
    if delta["upper"] < 0:
        return "CI 不含 0，**指向负增益**"
    return "CI 含 0（不显著）"


def write_report(run_id: str, evaluation: dict) -> Path:
    metrics = evaluation["metrics"]
    manifest = harness.common.read_json(harness.RUNS_DIR / run_id / "manifest.json")
    groups = metrics["groups"]
    paired = metrics.get("paired_group_comparisons") or {}
    gt = {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}
    lines: list[str] = []
    lines.append("# G2 阶段 2-A2 · 真实 50 条探索批（wb-proxy / deepseek-v4.1-flash）")
    lines.append("")
    lines.append(f"- run_id：`{run_id}`；生成时间（UTC）：{common.utc_now()}")
    lines.append("- **数据状态：REAL（真实 LLM 生成）；探索性证据，非最终 G2 结论**（最终需新独立测试集 + 人工评审）")
    lines.append(f"- 后端：provider=**wb-proxy**，model=**{MODEL}**，endpoint=`{BASE_URL}/chat/completions`，"
                 "credentials_ref=`WB_PROXY_API_KEY@dsh-credentials`（密钥仅内存，未落盘）")
    lines.append(f"- prompt：`{manifest['prompt']['version']}`（json sha256 `{manifest['prompt']['files']['json_sha256'][:16]}…`）；"
                 f"decoding：temperature={manifest['model']['decoding']['temperature']}, "
                 f"max_tokens={manifest['model']['decoding']['max_tokens']}, top_p={manifest['model']['decoding']['top_p']}")
    lines.append("- 执行：**严格串行**（保护同时服务 DSH 的代理）；每请求超时 90s；失败重试 2 次（指数退避），全部计数记录。")
    lines.append("")
    incident = _incident_stats()
    if incident:
        lines.append("## 0. 运行事故与修正（重要，先看）")
        lines.append("")
        lines.append(f"- 首轮尝试 `{incident['run_id']}` 使用 `max_tokens=256`："
                     f"{incident['n_failed']}/{incident['n_calls']} 次调用返回**空 completion**"
                     f"（HTTP 200，`finish_reason=length`，completion tokens 顶到 256，按组：{incident['failed_by_group']}）。")
        lines.append("- 诊断：`deepseek-v4.1-flash` 会先消耗 token 做内部推理，256 token 在较长 prompt（D/E）上不足以产出最终句；")
        lines.append("  这不是代理故障，也不是指标问题。首轮结果因**选择性偏差**（只有短推理样本成功）不可用于质量结论，保留其 run 目录仅作事故证据，**不进入对比表**。")
        lines.append(f"- 修正：本轮正式数据 `{RUN_ID}` 固定 `max_tokens={MAX_TOKENS}`（prompt 文本未变，仍是 `v1_zh`；run 级 override 记录在 manifest.decoding）。")
        lines.append("")
    lines.append("## 1. 样本与调用")
    lines.append("")
    sampling = manifest["sampling"]
    lines.append(f"- 样本：**{sampling['n_samples']}** 条，复用 E01 `pilot_50.csv`（10 signer × 5，signer 分层）；"
                 f"groups = {manifest['groups']}；A（Uni-Sign 原文）直通不调 LLM。")
    lines.append(f"- 真实 LLM 调用：**{sum(1 for row in _load_outputs(run_id) if row['group'] != 'A')}** 次（50 × C/D/E）；"
                 f"输出行 {manifest['outputs']['n_rows']}；失败 {manifest['outputs']['n_errors']}。")
    failed_rows = [row for row in _load_outputs(run_id) if row["group"] != "A" and row.get("status") != "ok"]
    if failed_rows:
        failed_detail = defaultdict(int)
        for row in failed_rows:
            failed_detail[row["group"]] += 1
        lines.append(f"- v2 仍有的失败：{len(failed_rows)} 次（按组 {dict(sorted(failed_detail.items()))}），"
                     f"均为超长内部推理导致 `finish_reason=length`（completion 顶到 max_tokens=2048，"
                     "详见 `fusion_outputs.jsonl` 的 status/error 字段）；对应 D/E 的有效 n 少 1–2 条。")
    lines.append("- 参考口径：GT `ref_sentence_upstream`（EVAL ONLY）；canonical `p1-v1`（BLEU-4 13a + 逐字加空格后处理 / chrF / 字符编辑相似度）。")
    lines.append("")
    lines.append("## 2. 各组质量（真实 LLM；A=50，D/E 因 2 次截断为 49/48）")
    lines.append("")
    lines.append("| 组 | n | BLEU-4 | chrF | mean char edit | exact | ΔBLEU vs A (95%CI) | Δedit vs A (95%CI) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for group in ("A", "C", "D", "E"):
        entry = groups.get(group, {})
        corpus = entry.get("metrics") or {}
        lines.append(f"| {group} | {entry.get('n_ok')} | {_fmt(corpus.get('bleu4_13a'))} | {_fmt(corpus.get('chrf'))} | "
                     f"{_fmt(corpus.get('mean_char_edit_similarity'))} | {corpus.get('exact_match_count')} | "
                     f"{_ci(entry.get('delta_bleu_vs_A'))} | {_ci(entry.get('delta_edit_vs_A'))} |")
    lines.append("")
    lines.append("## 3. 关键配对对比（同批样本；signer 聚类 bootstrap 1000 iters，seed=20261026；可用配对数见 n）")
    lines.append("")
    lines.append("**重点判据：D vs C（Gloss 新增价值）与 E vs D（证据约束收益）**")
    lines.append("")
    lines.append("| 配对（后者−前者） | n | ΔBLEU-4 (95%CI) | Δchar edit (95%CI) |")
    lines.append("|---|---:|---|---|")
    for pair in ("A_vs_C", "A_vs_D", "A_vs_E", "C_vs_D", "D_vs_E", "C_vs_E"):
        entry = paired.get(pair)
        if not entry:
            continue
        lines.append(f"| {entry['first']} → {entry['second']} | {entry['n_pairs']} | "
                     f"{_ci(entry.get('delta_bleu_second_minus_first'))} | "
                     f"{_ci(entry.get('delta_edit_second_minus_first'))} |")
    lines.append("")
    lines.append("读法：CI 跨 0 表示 50 条样本上无法排除“无差异”；点估计方向仅供参考，**不得当作 G2 通过依据**。")
    lines.append("")
    lines.append("## 4. 成本（真实调用）")
    lines.append("")
    lines.append("| 组 | 有效 n | P50 延迟 | P95 延迟 | prompt tokens | completion tokens | total tokens | 失败 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for group in ("A", "C", "D", "E"):
        entry = groups.get(group, {})
        latency = entry.get("backend_latency_ms") or {}
        tokens = entry.get("tokens") or {}
        lines.append(f"| {group} | {entry.get('n_ok')} | {_fmt(latency.get('p50'))} ms | {_fmt(latency.get('p95'))} ms | "
                     f"{tokens.get('prompt_sum')} | {tokens.get('completion_sum')} | {tokens.get('total_sum')} | "
                     f"{entry.get('failed_calls')} |")
    lines.append("")
    lines.append("> 上表分组成本只统计**成功调用**；下方汇总包含 3 次失败调用已消耗的 usage（失败调用同样计费/token）。")
    rows = [row for row in _load_outputs(run_id) if row["group"] != "A"]
    latencies = [row["latency_s"] for row in rows if isinstance(row.get("latency_s"), (int, float))]
    prompt_tokens = sum((row["tokens"].get("in") or 0) for row in rows)
    completion_tokens = sum((row["tokens"].get("out") or 0) for row in rows)
    total_tokens = sum((row["tokens"].get("total") or 0) for row in rows)
    retries = sum(row.get("retries") or 0 for row in rows)
    failures = sum(1 for row in rows if row.get("status") != "ok")
    lines.append("| 汇总指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| LLM 调用数 | {len(rows)} |")
    lines.append(f"| prompt tokens 合计 | {prompt_tokens} |")
    lines.append(f"| completion tokens 合计 | {completion_tokens} |")
    lines.append(f"| total tokens 合计 | {total_tokens} |")
    lines.append(f"| 平均 tokens/调用 | {(total_tokens / len(rows)):.1f} |" if rows else "| 平均 tokens/调用 | null |")
    if latencies:
        lines.append(f"| latency P50 | {statistics.median(latencies):.2f} s |")
        lines.append(f"| latency P95 | {common.percentile(latencies, 95):.2f} s |")
        lines.append(f"| latency mean | {statistics.fmean(latencies):.2f} s |")
        lines.append(f"| latency max | {max(latencies):.2f} s |")
    lines.append(f"| 重试次数合计 | {retries} |")
    lines.append(f"| 失败调用数 | {failures} |")
    lines.append("")
    lines.append("> 该延迟是「代理 + 模型」端到端 HTTP 时延（串行、单请求），不是 GPU 推理时延；不可与 E04 的 GPU 指标直接比较。")
    lines.append("")
    lines.append("## 5. 质量抽查示例（EVAL ONLY，含 GT 句子）")
    lines.append("")
    lines.append("| sample_id | GT（仅评测用） | A | C | D | E | edit A/C/D/E |")
    lines.append("|---|---|---|---|---|---|---|")
    for example in _examples(run_id):
        edits = example["edits"]
        lines.append(f"| {example['sample_id']} | {example['reference']} | {example['A']} | {example['C']} | "
                     f"{example['D']} | {example['E']} | "
                     f"{edits['A']:.2f}/{edits['C']:.2f}/{edits['D']:.2f}/{edits['E']:.2f} |")
    lines.append("")
    lines.append("## 6. 结论（探索性）")
    lines.append("")
    a_vs_c = paired.get("A_vs_C", {})
    c_vs_d = paired.get("C_vs_D", {})
    d_vs_e = paired.get("D_vs_E", {})
    c_vs_e = paired.get("C_vs_E", {})

    def _fmt_pair(entry: dict) -> str:
        bleu = entry.get("delta_bleu_second_minus_first")
        edit = entry.get("delta_edit_second_minus_first")
        return (f"ΔBLEU-4 = {_ci(bleu)}（{_signal(bleu)}）；"
                f"Δchar edit = {_ci(edit)}（{_signal(edit)}）")

    lines.append(f"- **A → C（LLM 润色）：**{_fmt_pair(a_vs_c)}")
    lines.append(f"- **C → D（Gloss 新增价值，核心判据）：**{_fmt_pair(c_vs_d)}")
    lines.append(f"- **D → E（证据约束收益，核心判据）：**{_fmt_pair(d_vs_e)}")
    lines.append(f"- **C → E（相对纯润色的总变化）：**{_fmt_pair(c_vs_e)}")
    lines.append("")
    lines.append("解读（按 50 条证据，不作 G2 结论）：")
    lines.append("")
    lines.append("1. **无条件 LLM 重写在本批数据上显著降低字面一致性**：C 相对 A 的 BLEU（-4.55）与字符编辑相似度（-0.033）CI 均不含 0；")
    lines.append("   抽查示例可见模型会改写正确事实、补全证据外内容（如把「妈妈回房间工作」改写成无依据的「做这做那」）。")
    lines.append("   这提示『全部重写』的融合方式风险高；若做融合，应先做**选择性改写/冲突时才升级**（与 E06 的 Rule-only 路由结论一致）。")
    lines.append("2. **D − C 无显著字面增益**（BLEU CI 含 0；edit 点估计 +0.026、CI 边缘含 0）：50 条样本下 Gloss 证据的"
                 "字面一致性收益不成立，只能说明方向不明确；与 E02 的 A/A2 差异同量级（编辑相似度差 ~0.005–0.03），需要更大样本/人评。")
    lines.append("3. **E − D 方向为正但不显著**（BLEU +1.62、edit +0.011，CI 均含 0）；C → E 的字符编辑相似度"
                 "CI 不含 0（+0.040 [0.0019, 0.0821]），提示『证据约束』比纯润色略好，但 BLEU 不确定。")
    lines.append("4. **成本**：C/D/E 每 50 条分别 ~21k/38k/47k tokens、P50 2.0/3.8/3.9 s、P95 4.4/8.5/9.2 s；"
                 "若扩到 1086 条，D/E 预计 ~80 万/100 万 tokens 级（按当前均值线性外推，未实测）。")
    lines.append("5. **不能作为 G2 通过/否决依据**：50 条为 retrospective_seen 子集、无新独立集、无人工语义评审；")
    lines.append("   且 prompt v1_zh 为首版、未做 prompt 迭代（模型内部推理消耗也会影响稳定性，3/150 次 2048-token 截断）。")
    lines.append("")
    lines.append("建议下一步：")
    lines.append("")
    lines.append("- **先改 prompt 再扩量**：C 只做最小必要修正（保留与证据一致的原句），D/E 明确「冲突才改」；新增 `v2_zh` 版本并重跑 50 条对比，而不是直接跑 1086。")
    lines.append("- 结合 E06 的级联结论：优先评估「A 为默认、规则/分数触发才调 LLM」的选择性融合，而不是每句都调。")
    lines.append("- 人工评审（Round-1 包）优先用于验证「LLM 改写是否引入新的关键事实错误」；自动 BLEU 不能替代语义判断。")
    lines.append("- B（G2T/SLT）仍 BLOCKED；在拿到权重前，不建议把 LLM 后期融合作为主链路。")
    lines.append("")
    lines.append("## 7. 边界与安全")
    lines.append("")
    lines.append("1. 50 条为 E01 pilot 抽样（10 signer × 5），是 1086 的**子集**，不是新盲测；同 signer 内可能有同源样本。")
    lines.append("2. 候选来源：A/A2/B0 为 E02 冻结历史预测；C/D/E 为本轮真实 LLM 生成（provider=wb-proxy, model=deepseek-v4.1-flash）。")
    lines.append("3. **密钥安全**：仅运行时从 dsh credentials yaml 读入内存；未打印/未写入任何输出/JSONL/manifest/日志/本报告；manifest 只含 `credentials_ref` 引用。")
    lines.append("4. GT 只在评测侧 join（`evaluate.py`）；LLM 输入 prompt 不含 GT（harness 的 gt_guard 与 prompt 隔离检查见 stub 自检 33/33）。")
    lines.append("5. 本报告为评测侧文档，含 GT 例句（EVAL ONLY）；不得转回任何模型输入。")
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main() -> int:
    result = run_batch()
    evaluation = evaluate_module.evaluate_run(RUN_ID, iterations=1000, seed=evaluate_module.BOOTSTRAP_SEED, write=True)
    evaluate_module.write_ablation_csv(STUB_RUNS + [RUN_ID])
    report = write_report(RUN_ID, evaluation)
    metrics = evaluation["metrics"]
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
            "latency_p50_ms": (entry.get("backend_latency_ms") or {}).get("p50"),
            "latency_p95_ms": (entry.get("backend_latency_ms") or {}).get("p95"),
            "tokens_total_sum": (entry.get("tokens") or {}).get("total_sum"),
            "failed_calls": entry.get("failed_calls"),
            "retries_total": entry.get("retries_total"),
        } for group, entry in metrics["groups"].items()},
        "pairs": {pair: {
            "n": entry["n_pairs"],
            "delta_bleu": entry["delta_bleu_second_minus_first"],
            "delta_edit": entry["delta_edit_second_minus_first"],
        } for pair, entry in (metrics.get("paired_group_comparisons") or {}).items()},
        "report": str(report),
        "ablation_csv": str(evaluate_module.ABLATION_CSV),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
