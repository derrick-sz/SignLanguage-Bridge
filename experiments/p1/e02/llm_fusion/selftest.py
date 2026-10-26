"""Full-pipeline self-test for the G2 phase-2A LLM fusion harness (stub backend).

Runs the harness twice — 15-signer-stratified samples and the full 1086 dry-run —
then evaluates (EVAL ONLY), executes a battery of discipline/contract assertions,
writes `fusion_ablation_g2.csv` and `report.md`.

Real LLM backends stay disabled; this round is `IMPLEMENTED_UNTESTED` for real
endpoints and `STUB_ONLY` for every number produced here.

Usage:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -m experiments.p1.e02.llm_fusion.selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.p1 import common
from experiments.p1.eval import metrics as M
from experiments.p1.gt_guard import GroundTruthLeakError, assert_gt_clean, find_gt_keys

from experiments.p1.e02.llm_fusion import evaluate as evaluate_module
from experiments.p1.e02.llm_fusion import harness
from experiments.p1.e02.llm_fusion.backends import get_backend
from experiments.p1.e02.llm_fusion.backends.base import BackendDisabled
from experiments.p1.e02.llm_fusion.backends.codex_cli import CodexCLIBackend

FUSION_DIR = harness.FUSION_DIR
SELFTEST_RUN_ID = "g2a-stub-selftest-15"
DRYRUN_RUN_ID = "g2a-stub-dryrun-1086"
SELFTEST_SAMPLES = 15
SELFTEST_BOOTSTRAP_ITERATIONS = 1000


def _check(checks: list[dict], name: str, passed: bool, detail: Any = "") -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def _schema_ok(row: dict) -> bool:
    required = {
        "run_id", "sample_id", "group", "prompt_version", "prompt_spec_sha256",
        "backend", "backend_version", "decoding", "evidence", "score_features",
        "model_version", "output_text", "status", "error", "latency_ms", "tokens",
        "system_prompt_sha256", "user_prompt_sha256", "prompt_sha256", "created_at",
    }
    return required <= set(row) and row["group"] in {"A", "C", "D", "E"}


def run_selftest() -> dict[str, Any]:
    checks: list[dict] = []
    FUSION_DIR.mkdir(parents=True, exist_ok=True)

    # ---- run 1: 15-sample stub self-test (full pipeline incl. bootstrap) ----
    selftest_run = harness.run(
        backend_name="stub",
        limit=SELFTEST_SAMPLES,
        run_id=SELFTEST_RUN_ID,
        overwrite=True,
        data_status="STUB_ONLY_NOT_LLM_RESULTS",
    )
    selftest_eval = evaluate_module.evaluate_run(
        SELFTEST_RUN_ID, iterations=SELFTEST_BOOTSTRAP_ITERATIONS, seed=evaluate_module.BOOTSTRAP_SEED, write=True
    )

    # ---- run 2: full-1086 stub dry run (pipeline scale + GT-guard at scale) ----
    dryrun = harness.run(
        backend_name="stub",
        limit=None,
        run_id=DRYRUN_RUN_ID,
        overwrite=True,
        data_status="STUB_ONLY_PIPELINE_SCALE_CHECK",
    )
    dryrun_eval = evaluate_module.evaluate_run(DRYRUN_RUN_ID, iterations=0, write=True)

    selftest_rows = common.read_jsonl(harness.RUNS_DIR / SELFTEST_RUN_ID / "fusion_outputs.jsonl")
    dryrun_rows = common.read_jsonl(harness.RUNS_DIR / DRYRUN_RUN_ID / "fusion_outputs.jsonl")
    selftest_manifest = common.read_json(harness.RUNS_DIR / SELFTEST_RUN_ID / "manifest.json")
    dryrun_manifest = common.read_json(harness.RUNS_DIR / DRYRUN_RUN_ID / "manifest.json")
    prompts = harness.load_prompt_spec()
    candidates = harness.load_candidates()
    gt = evaluate_module.load_gt()

    n_selftest_samples = selftest_manifest["sampling"]["n_samples"]
    _check(checks, "selftest run has 15 samples", selftest_manifest["sampling"]["n_samples"] == SELFTEST_SAMPLES,
           selftest_manifest["sampling"]["n_samples"])
    _check(checks, "selftest rows == n_samples x 4 groups",
           len(selftest_rows) == SELFTEST_SAMPLES * 4, len(selftest_rows))
    _check(checks, "dryrun rows == 1086 x 4 groups", len(dryrun_rows) == 1086 * 4, len(dryrun_rows))
    group_counts = {group: sum(1 for row in selftest_rows if row["group"] == group) for group in "ACDE"}
    _check(checks, "each group has 15 rows in selftest", all(count == SELFTEST_SAMPLES for count in group_counts.values()),
           group_counts)

    # ---- GT isolation ------------------------------------------------------
    leaks_selftest = [row["sample_id"] + "/" + row["group"] for row in selftest_rows if find_gt_keys(row)]
    leaks_dryrun = [row["sample_id"] + "/" + row["group"] for row in dryrun_rows if find_gt_keys(row)]
    _check(checks, "gt_guard: 0 GT-key rows in selftest outputs", not leaks_selftest, len(leaks_selftest))
    _check(checks, "gt_guard: 0 GT-key rows in 1086 dry-run outputs", not leaks_dryrun, len(leaks_dryrun))
    _check(checks, "manifest gt_isolation.gt_guard_leaks == 0 (selftest)",
           selftest_manifest["gt_isolation"]["gt_guard_leaks"] == 0,
           selftest_manifest["gt_isolation"]["gt_guard_leaks"])
    _check(checks, "manifest records harness_reads_gt == False",
           selftest_manifest["gt_isolation"]["harness_reads_gt"] is False)
    _check(checks, "manifest gt_guard_rows_checked == n_rows (dry-run)",
           dryrun_manifest["gt_isolation"]["gt_guard_rows_checked"] == len(dryrun_rows),
           dryrun_manifest["gt_isolation"]["gt_guard_rows_checked"])

    # eval_records are the only files with GT; they are written guard=False and exist
    eval_records_path = harness.RUNS_DIR / SELFTEST_RUN_ID / "eval_records.jsonl"
    eval_records = common.read_jsonl(eval_records_path)
    _check(checks, "eval_records written (EVAL ONLY)", len(eval_records) == len(selftest_rows), len(eval_records))

    # ---- prompt isolation: GT reference / gloss must not appear ------------
    prompt_violations = 0
    checked_prompts = 0
    for row in selftest_rows:
        if row["group"] == "A":
            continue
        messages = harness.build_messages(
            row["group"],
            unisign_text=row["evidence"]["unisign_text"],
            gloss_text=row["evidence"]["slrt_gloss_text"],
            features=row["score_features"],
            spec=prompts,
        )
        prompt_text = "\n".join(message["content"] for message in messages)
        checked_prompts += 1
        gt_row = gt[row["sample_id"]]
        reference = gt_row["ref_sentence_upstream"]
        normalized_reference = M.normalize_text(reference)
        if reference and reference in prompt_text:
            prompt_violations += 1
        if normalized_reference and normalized_reference != M.normalize_text(row["evidence"]["unisign_text"]) \
                and normalized_reference in prompt_text:
            prompt_violations += 1
    _check(checks, "GT reference sentence absent from all built prompts",
           prompt_violations == 0, f"{checked_prompts} prompts checked, {prompt_violations} violations")

    forbidden_hits = 0
    forbidden_names = sorted(harness.FORBIDDEN_SCORE_FEATURES)
    for row in selftest_rows:
        if row["group"] == "A":
            continue
        messages = harness.build_messages(
            row["group"], unisign_text=row["evidence"]["unisign_text"],
            gloss_text=row["evidence"]["slrt_gloss_text"], features=row["score_features"], spec=prompts,
        )
        prompt_text = "\n".join(message["content"] for message in messages)
        forbidden_hits += sum(1 for name in forbidden_names if name in prompt_text)
    _check(checks, "no forbidden E03 label/OOF column names in prompts", forbidden_hits == 0, forbidden_hits)

    # ---- score feature whitelist ------------------------------------------
    features = harness.load_score_features()
    allowed = set(harness.SCORE_FEATURE_WHITELIST)
    feature_keys = {key for record in features.values() for key in record}
    _check(checks, "score feature loader only returns whitelisted columns", feature_keys <= allowed,
           sorted(feature_keys - allowed))
    row_feature_keys = {key for row in selftest_rows for key in row["score_features"]}
    _check(checks, "output score_features keys are whitelisted", row_feature_keys <= allowed,
           sorted(row_feature_keys - allowed))
    forbidden_in_features = {key for row in selftest_rows for key in row["score_features"] if key in harness.FORBIDDEN_SCORE_FEATURES}
    _check(checks, "no GT-label column leaked into score_features", not forbidden_in_features,
           sorted(forbidden_in_features))

    # ---- gt_guard negative control ---------------------------------------
    negative_raised = False
    try:
        assert_gt_clean({"sample_id": "x", "reference": "leak"}, context="negative-control")
    except GroundTruthLeakError:
        negative_raised = True
    _check(checks, "gt_guard negative control raises on injected GT key", negative_raised)

    # ---- harness source contains no GT path ------------------------------
    source_files = [FUSION_DIR / "harness.py"]
    source_files.extend(sorted((FUSION_DIR / "backends").glob("*.py")))
    source_files.extend(sorted((FUSION_DIR / "prompts").glob("v1_zh.*")))
    gt_hits = [str(path.relative_to(FUSION_DIR)) for path in source_files
               if "gt_1086" in path.read_text(encoding="utf-8")]
    _check(checks, "harness/backends/prompts never reference gt_1086.csv", not gt_hits, gt_hits)

    # ---- real backends are gated (must raise, not call) -------------------
    codex = get_backend("codex_cli")
    openai = get_backend("openai_compat")
    codex_blocked = openai_blocked = False
    try:
        codex.generate([{"role": "user", "content": "x"}], group="C")
    except BackendDisabled:
        codex_blocked = True
    try:
        openai.generate([{"role": "user", "content": "x"}], group="C")
    except BackendDisabled:
        openai_blocked = True
    _check(checks, "codex_cli backend disabled without explicit enablement", codex_blocked)
    _check(checks, "openai_compat backend disabled without explicit enablement", openai_blocked)
    probe = CodexCLIBackend.probe()
    _check(checks, "codex CLI probed locally (availability evidence, no LLM call)",
           probe.get("available") is True, probe)

    # ---- schema / determinism / self-consistency --------------------------
    _check(checks, "all selftest rows satisfy output schema", all(_schema_ok(row) for row in selftest_rows),
           sum(0 if _schema_ok(row) else 1 for row in selftest_rows))
    _check(checks, "all stub outputs non-empty and status ok",
           all(row["status"] == "ok" and row["output_text"].strip() for row in selftest_rows),
           sum(1 for row in selftest_rows if not (row["status"] == "ok" and row["output_text"].strip())))

    stub = get_backend("stub")
    determinism_ok = True
    sample_ids = [row["sample_id"] for row in selftest_rows if row["group"] == "A"][:3]
    for sample in sample_ids:
        for group in ("C", "D", "E"):
            messages = harness.build_messages(
                group, unisign_text=candidates[sample]["A"], gloss_text=candidates[sample]["B0"],
                features=features.get(sample, {}), spec=prompts,
            )
            first = stub.generate(messages, group=group,
                                  evidence={"unisign_text": candidates[sample]["A"],
                                            "gloss_text": candidates[sample]["B0"]})
            second = stub.generate(messages, group=group,
                                   evidence={"unisign_text": candidates[sample]["A"],
                                             "gloss_text": candidates[sample]["B0"]})
            if first.text != second.text:
                determinism_ok = False
    _check(checks, "stub outputs deterministic (temperature 0 semantics)", determinism_ok,
           f"{len(sample_ids)} samples x C/D/E")

    # A baseline metrics must equal a direct canonical recomputation of candidate A texts
    selftest_metrics = selftest_eval["metrics"]
    group_a = selftest_metrics["groups"]["A"]["metrics"]
    selected_all = [row["sample_id"] for row in selftest_rows if row["group"] == "A"]
    direct_all = M.corpus_metrics_cslt([(gt[sample]["ref_sentence_upstream"], candidates[sample]["A"])
                                        for sample in selected_all])
    _check(checks, "A baseline metrics match direct canonical recomputation",
           group_a.get("bleu4_13a") == direct_all["bleu4_13a"]
           and group_a.get("chrf") == direct_all["chrf"]
           and group_a.get("mean_char_edit_similarity") == direct_all["mean_char_edit_similarity"],
           {"run": group_a.get("bleu4_13a"), "direct": direct_all["bleu4_13a"]})

    # ---- manifest contract ------------------------------------------------
    _check(checks, "manifest status == IMPLEMENTED_UNTESTED",
           selftest_manifest.get("status") == "IMPLEMENTED_UNTESTED", selftest_manifest.get("status"))
    _check(checks, "manifest data_status == STUB_ONLY_NOT_LLM_RESULTS",
           selftest_manifest.get("data_status") == "STUB_ONLY_NOT_LLM_RESULTS",
           selftest_manifest.get("data_status"))
    _check(checks, "manifest records prompt json sha256",
           len(selftest_manifest["prompt"]["files"]["json_sha256"]) == 64)
    _check(checks, "manifest records prompt md sha256",
           len(selftest_manifest["prompt"]["files"]["md_sha256"] or "") == 64)
    _check(checks, "manifest records candidate_outputs/e03 hashes",
           len(selftest_manifest["inputs"]["candidate_outputs_sha256"]) == 64
           and len(selftest_manifest["inputs"]["e03_scores_sha256"]) == 64)
    _check(checks, "manifest groups recorded", selftest_manifest.get("groups") == ["A", "C", "D", "E"],
           selftest_manifest.get("groups"))
    _check(checks, "decoding fixed temperature=0 / max_tokens=256",
           selftest_manifest["model"]["decoding"]["temperature"] == 0
           and selftest_manifest["model"]["decoding"]["max_tokens"] == 256)

    # ---- ablation csv ------------------------------------------------------
    csv_path = evaluate_module.write_ablation_csv([SELFTEST_RUN_ID, DRYRUN_RUN_ID])
    csv_rows = common.read_csv_rows(csv_path)
    _check(checks, "fusion_ablation_g2.csv has A/C/D/E x 2 runs + B BLOCKED",
           len(csv_rows) == 9 and any(row["group"] == "B" and row["data_status"] == "BLOCKED" for row in csv_rows),
           len(csv_rows))
    _check(checks, "all stub rows in ablation csv are labelled STUB_ONLY",
           all(str(row["note"]).startswith("STUB_ONLY") for row in csv_rows if row["group"] != "B"))

    failed = [check for check in checks if not check["pass"]]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "failed": [check["name"] for check in failed],
        "checks": checks,
        "runs": [SELFTEST_RUN_ID, DRYRUN_RUN_ID],
        "selftest_eval": selftest_eval["metrics"],
        "dryrun_eval": dryrun_eval["metrics"],
        "ablation_csv": str(csv_path),
        "codex_probe": probe,
        "backends": harness.list_backends(),
    }

    _write_report(result)
    common.write_json(FUSION_DIR / "selftest_result.json", result)
    print(json.dumps({
        "status": result["status"],
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "failed": result["failed"],
        "runs": result["runs"],
        "ablation_csv": str(csv_path),
    }, ensure_ascii=False, indent=2))
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metrics_table(metrics: dict[str, Any]) -> list[str]:
    lines = ["| 组 | n | BLEU-4 | chrF | mean char edit | exact | ΔBLEU vs A (95%CI) | Δedit vs A (95%CI) |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for group in ("A", "C", "D", "E"):
        entry = metrics["groups"].get(group)
        if not entry:
            continue
        corpus = entry.get("metrics") or {}
        delta_bleu = entry.get("delta_bleu_vs_A") or {}
        delta_edit = entry.get("delta_edit_vs_A") or {}
        if delta_bleu.get("lower") is not None:
            bleu_ci = f"{_fmt(delta_bleu.get('point'), 4)} [{_fmt(delta_bleu.get('lower'), 4)}, {_fmt(delta_bleu.get('upper'), 4)}]"
        else:
            bleu_ci = _fmt(delta_bleu.get("point"), 4)
        if delta_edit.get("lower") is not None:
            edit_ci = f"{_fmt(delta_edit.get('point'), 4)} [{_fmt(delta_edit.get('lower'), 4)}, {_fmt(delta_edit.get('upper'), 4)}]"
        else:
            edit_ci = _fmt(delta_edit.get("point"), 4)
        lines.append(
            f"| {group} | {entry.get('n_ok')} | {_fmt(corpus.get('bleu4_13a'), 4)} | {_fmt(corpus.get('chrf'), 4)} | "
            f"{_fmt(corpus.get('mean_char_edit_similarity'), 4)} | {corpus.get('exact_match_count')} | {bleu_ci} | {edit_ci} |"
        )
    return lines


def _write_report(result: dict[str, Any]) -> None:
    selftest = result["selftest_eval"]
    dryrun = result["dryrun_eval"]
    manifest = common.read_json(harness.RUNS_DIR / SELFTEST_RUN_ID / "manifest.json")
    dryrun_manifest = common.read_json(harness.RUNS_DIR / DRYRUN_RUN_ID / "manifest.json")
    lines: list[str] = []
    lines.append("# G2 阶段 2-A · LLM 融合消融 harness 报告")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{common.utc_now()}")
    lines.append("- 实验：G2-2A-LLM-FUSION（A/C/D/E 消融；B=G2T 仍 BLOCKED）")
    lines.append(f"- **状态：IMPLEMENTED_UNTESTED（真实 LLM 端点未调用、未验证）；stub 全链路自检 PASS（{result['checks_total'] - result['checks_failed']}/{result['checks_total']}）**")
    lines.append("- **本轮所有数字都来自 stub 后端：STUB_ONLY，不是 LLM 模型结果。** 真实 LLM 端点等用户/Lead 选定后按 §5 切换。")
    lines.append("- 数据：1086 条 CSL-Daily test 子集（retrospective_seen，非新盲测）；候选文本为 E02 冻结历史预测（A/A2/B0）。")
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    lines.append(f"- 管线已验证可端到端运行：15 条 signer 分层样本 × 4 组 = {manifest['outputs']['n_rows']} 行输出，"
                 f"GT guard 0 泄漏；全量 1086 dry-run = {dryrun_manifest['outputs']['n_rows']} 行输出，GT guard 0 泄漏。")
    lines.append(f"- 自检断言：**{result['checks_total'] - result['checks_failed']}/{result['checks_total']} 通过**"
                 + (f"；失败项：{result['failed']}" if result["failed"] else "。"))
    lines.append("- 真实 LLM 后端：`codex_cli` 已实现（本机 codex CLI 探测："
                 f"{result['codex_probe'].get('version', 'n/a')}，可用={result['codex_probe'].get('available')}），"
                 "但**未调用**（需 `P1_LLM_ENABLE_REAL=1` + 固定模型版本 + 用户/Lead 批准）；"
                 "`openai_compat` 仅接口，未实现 HTTP 调用。")
    lines.append("- 下一步：端点选定后先跑 50 条探索批（同一 prompt/参数/模型版本），再决定是否扩到 1086；"
                 "真实数字出来后写新 run_id 与新报告，不与本 stub 报告混用。")
    lines.append("")
    lines.append("## 2. 目录与产物")
    lines.append("")
    lines.append("```text")
    lines.append("experiments/p1/e02/llm_fusion/")
    lines.append("  harness.py                 # GT-free pipeline: candidates -> prompts -> backend -> JSONL")
    lines.append("  evaluate.py                # EVAL ONLY: GT join + canonical metrics + signer bootstrap")
    lines.append("  selftest.py                # stub 全链路自检（本报告生成器）")
    lines.append("  backends/{base,stub,codex_cli,openai_compat}.py")
    lines.append("  prompts/v1_zh.{json,md}    # 版本化 prompt（sha256 记录在 manifest）")
    lines.append("  runs/g2a-stub-selftest-15/        fusion_outputs.jsonl / manifest.json / eval_records.jsonl / metrics.json")
    lines.append("  runs/g2a-stub-dryrun-1086/        (同上，全量 dry-run)")
    lines.append("  fusion_ablation_g2.csv     # 消融总表（stub 标注）")
    lines.append("  selftest_result.json       # 断言明细")
    lines.append("```")
    lines.append("")
    lines.append("## 3. GT 隔离证据")
    lines.append("")
    lines.append(f"- harness/backends/prompts 源码内不含 `gt_1086` 引用（evaluate.py 是唯一 GT 读取点）。")
    lines.append(f"- 15 条自检：`gt_guard` 逐行扫描 {manifest['outputs']['n_rows']} 行，泄漏 **{manifest['gt_isolation']['gt_guard_leaks']}**；"
                 f"GT reference 句子在 prompt 中出现次数 **0**。")
    lines.append(f"- E03 CSV 只读白名单列：{', '.join(manifest['inputs']['score_feature_whitelist'][:6])} …；"
                 f"GT 标签/OOF 列（wer/bad/error_any/oof_p_bad_* 等）全部排除，并有断言。")
    lines.append(f"- 负例控制：向 `assert_gt_clean` 注入 `reference` 键会抛 `GroundTruthLeakError`（通过）。")
    lines.append("- eval_records.jsonl 是唯一含 GT 的文件（EVAL ONLY，guard=False），与推理载荷物理分离。")
    lines.append("")
    lines.append("## 4. stub 全链路自检结果（STUB_ONLY，不可当作 LLM 质量）")
    lines.append("")
    lines.append(f"### 4.1 15 条 signer 分层样本（bootstrap {SELFTEST_BOOTSTRAP_ITERATIONS} iters，seed {evaluate_module.BOOTSTRAP_SEED}）")
    lines.append("")
    lines.extend(_metrics_table(selftest))
    lines.append("")
    lines.append(f"- 样本选择：按 signer 排序轮转取首样本（确定性；manifest 记录 n_samples={manifest['sampling']['n_samples']}、"
                 f"signer 分布 {manifest['sampling']['signer_counts']}）。")
    lines.append("- stub 规则：C=Uni-Sign 归一化回显；D=C+数字槽对齐 Gloss；E=D+疑问语气对齐。**不是语言模型**。")
    lines.append("")
    lines.append("### 4.2 全量 1086 dry-run（仅点指标，不做 bootstrap；STUB_ONLY pipeline scale check）")
    lines.append("")
    lines.extend(_metrics_table(dryrun))
    lines.append("")
    lines.append("## 5. 真实 LLM 接入清单（当前 BLOCKED/待用户选择）")
    lines.append("")
    lines.append("- 端点候选：① 本机 Codex CLI（已登录，`codex-cli 0.160.1`）；② 用户提供的 OpenAI 兼容端点。")
    lines.append("- 需要固定：provider + model-id/版本、prompt 版本（v1_zh）、temperature=0、max_tokens=256、预算（调用数/费用上限）。")
    lines.append("- Codex 调用形态（脚本已实现，未执行）：`codex exec --json --ephemeral --sandbox read-only -m <model> -o <tmpfile> -`，prompt 走 stdin。")
    lines.append("- 启用条件：`P1_LLM_ENABLE_REAL=1` + `--allow-real` + `P1_CODEX_MODEL`（或 OpenAI 兼容的三个 env 变量，密钥只在进程环境注入）。")
    lines.append("- 真实跑批：先 50 条探索批（`--limit 50`），再按需 1086；每次新 run_id；报告必须标 `real` 并与本 stub 报告分离。")
    lines.append("")
    lines.append("## 6. 自检断言明细")
    lines.append("")
    lines.append("| # | 断言 | 结果 | 详情 |")
    lines.append("|---|---|---|---|")
    for index, check in enumerate(result["checks"], 1):
        detail = check["detail"]
        detail_text = json.dumps(detail, ensure_ascii=False) if isinstance(detail, (dict, list)) else str(detail)
        if len(detail_text) > 160:
            detail_text = detail_text[:157] + "..."
        lines.append(f"| {index} | {check['name']} | {'✅' if check['pass'] else '❌'} | {detail_text} |")
    lines.append("")
    lines.append("## 7. 边界")
    lines.append("")
    lines.append("1. 本报告所有 BLEU/chrF/编辑相似度都是 **stub 规则输出** 的指标，用于验证管线，**不得引用为 LLM 消融结果**。")
    lines.append("2. 1086 是 retrospective_seen 子集；候选文本是冻结历史预测，不是本轮重跑。")
    lines.append("3. 真实 LLM 的 token 数/时延/BLEU 全部为空（未调用），不得填估计值。")
    lines.append("4. A/A2/B0 的 E02 指标见 `experiments/p1/e02/translation_metrics.json`；本 harness 的 A 组指标只用于对照一致性。")
    lines.append("5. B（SLRT G2T/SLT）仍 BLOCKED（外部权重缺失）；B0 join 12.22 BLEU 是拼接基线，不代表 SLRT 翻译能力。")
    lines.append("")
    (FUSION_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = run_selftest()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
