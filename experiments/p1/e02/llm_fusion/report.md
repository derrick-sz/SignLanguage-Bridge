# G2 阶段 2-A · LLM 融合消融 harness 报告

- 生成时间（UTC）：2026-10-26T15:23:11+00:00
- 实验：G2-2A-LLM-FUSION（A/C/D/E 消融；B=G2T 仍 BLOCKED）
- **状态：IMPLEMENTED_UNTESTED（真实 LLM 端点未调用、未验证）；stub 全链路自检 PASS（33/33）**
- **本轮所有数字都来自 stub 后端：STUB_ONLY，不是 LLM 模型结果。** 真实 LLM 端点等用户/Lead 选定后按 §5 切换。
- 数据：1086 条 CSL-Daily test 子集（retrospective_seen，非新盲测）；候选文本为 E02 冻结历史预测（A/A2/B0）。

## 1. 结论

- 管线已验证可端到端运行：15 条 signer 分层样本 × 4 组 = 60 行输出，GT guard 0 泄漏；全量 1086 dry-run = 4344 行输出，GT guard 0 泄漏。
- 自检断言：**33/33 通过**。
- 真实 LLM 后端：`codex_cli` 已实现（本机 codex CLI 探测：codex-cli 0.160.1，可用=True），但**未调用**（需 `P1_LLM_ENABLE_REAL=1` + 固定模型版本 + 用户/Lead 批准）；`openai_compat` 仅接口，未实现 HTTP 调用。
- 下一步：端点选定后先跑 50 条探索批（同一 prompt/参数/模型版本），再决定是否扩到 1086；真实数字出来后写新 run_id 与新报告，不与本 stub 报告混用。

## 2. 目录与产物

```text
experiments/p1/e02/llm_fusion/
  harness.py                 # GT-free pipeline: candidates -> prompts -> backend -> JSONL
  evaluate.py                # EVAL ONLY: GT join + canonical metrics + signer bootstrap
  selftest.py                # stub 全链路自检（本报告生成器）
  backends/{base,stub,codex_cli,openai_compat}.py
  prompts/v1_zh.{json,md}    # 版本化 prompt（sha256 记录在 manifest）
  runs/g2a-stub-selftest-15/        fusion_outputs.jsonl / manifest.json / eval_records.jsonl / metrics.json
  runs/g2a-stub-dryrun-1086/        (同上，全量 dry-run)
  fusion_ablation_g2.csv     # 消融总表（stub 标注）
  selftest_result.json       # 断言明细
```

## 3. GT 隔离证据

- harness/backends/prompts 源码内不含 `gt_1086` 引用（evaluate.py 是唯一 GT 读取点）。
- 15 条自检：`gt_guard` 逐行扫描 60 行，泄漏 **0**；GT reference 句子在 prompt 中出现次数 **0**。
- E03 CSV 只读白名单列：token_score_mean, token_score_min, token_score_median, token_score_p10, token_score_max, window_top_prob_mean …；GT 标签/OOF 列（wer/bad/error_any/oof_p_bad_* 等）全部排除，并有断言。
- 负例控制：向 `assert_gt_clean` 注入 `reference` 键会抛 `GroundTruthLeakError`（通过）。
- eval_records.jsonl 是唯一含 GT 的文件（EVAL ONLY，guard=False），与推理载荷物理分离。

## 4. stub 全链路自检结果（STUB_ONLY，不可当作 LLM 质量）

### 4.1 15 条 signer 分层样本（bootstrap 1000 iters，seed 20261026）

| 组 | n | BLEU-4 | chrF | mean char edit | exact | ΔBLEU vs A (95%CI) | Δedit vs A (95%CI) |
|---|---:|---:|---:|---:|---:|---|---|
| A | 15 | 23.6188 | 20.7146 | 0.4368 | 0 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| C | 15 | 23.6188 | 20.7146 | 0.4368 | 0 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| D | 15 | 23.6228 | 20.7356 | 0.4432 | 0 | 0.0039 [-0.0047, 0.0423] | 0.0064 [0.0000, 0.0179] |
| E | 15 | 23.6228 | 20.7356 | 0.4543 | 0 | 0.0039 [-0.0047, 0.0423] | 0.0175 [0.0000, 0.0432] |

- 样本选择：按 signer 排序轮转取首样本（确定性；manifest 记录 n_samples=15、signer 分布 {'P0000': 2, 'P0001': 2, 'P0002': 2, 'P0003': 2, 'P0004': 2, 'P0005': 1, 'P0006': 1, 'P0007': 1, 'P0008': 1, 'P0009': 1}）。
- stub 规则：C=Uni-Sign 归一化回显；D=C+数字槽对齐 Gloss；E=D+疑问语气对齐。**不是语言模型**。

### 4.2 全量 1086 dry-run（仅点指标，不做 bootstrap；STUB_ONLY pipeline scale check）

| 组 | n | BLEU-4 | chrF | mean char edit | exact | ΔBLEU vs A (95%CI) | Δedit vs A (95%CI) |
|---|---:|---:|---:|---:|---:|---|---|
| A | 1086 | 24.8157 | 21.7974 | 0.4157 | 15 | 0.0000 | 0.0000 |
| C | 1086 | 24.8157 | 21.7974 | 0.4157 | 15 | 0.0000 | 0.0000 |
| D | 1086 | 24.8097 | 21.7923 | 0.4155 | 15 | -0.0060 | -0.0001 |
| E | 1086 | 24.7290 | 21.7119 | 0.4141 | 15 | -0.0867 | -0.0016 |

## 5. 真实 LLM 接入清单（当前 BLOCKED/待用户选择）

- 端点候选：① 本机 Codex CLI（已登录，`codex-cli 0.160.1`）；② 用户提供的 OpenAI 兼容端点。
- 需要固定：provider + model-id/版本、prompt 版本（v1_zh）、temperature=0、max_tokens=256、预算（调用数/费用上限）。
- Codex 调用形态（脚本已实现，未执行）：`codex exec --json --ephemeral --sandbox read-only -m <model> -o <tmpfile> -`，prompt 走 stdin。
- 启用条件：`P1_LLM_ENABLE_REAL=1` + `--allow-real` + `P1_CODEX_MODEL`（或 OpenAI 兼容的三个 env 变量，密钥只在进程环境注入）。
- 真实跑批：先 50 条探索批（`--limit 50`），再按需 1086；每次新 run_id；报告必须标 `real` 并与本 stub 报告分离。

## 6. 自检断言明细

| # | 断言 | 结果 | 详情 |
|---|---|---|---|
| 1 | selftest run has 15 samples | ✅ | 15 |
| 2 | selftest rows == n_samples x 4 groups | ✅ | 60 |
| 3 | dryrun rows == 1086 x 4 groups | ✅ | 4344 |
| 4 | each group has 15 rows in selftest | ✅ | {"A": 15, "C": 15, "D": 15, "E": 15} |
| 5 | gt_guard: 0 GT-key rows in selftest outputs | ✅ | 0 |
| 6 | gt_guard: 0 GT-key rows in 1086 dry-run outputs | ✅ | 0 |
| 7 | manifest gt_isolation.gt_guard_leaks == 0 (selftest) | ✅ | 0 |
| 8 | manifest records harness_reads_gt == False | ✅ |  |
| 9 | manifest gt_guard_rows_checked == n_rows (dry-run) | ✅ | 4344 |
| 10 | eval_records written (EVAL ONLY) | ✅ | 60 |
| 11 | GT reference sentence absent from all built prompts | ✅ | 45 prompts checked, 0 violations |
| 12 | no forbidden E03 label/OOF column names in prompts | ✅ | 0 |
| 13 | score feature loader only returns whitelisted columns | ✅ | [] |
| 14 | output score_features keys are whitelisted | ✅ | [] |
| 15 | no GT-label column leaked into score_features | ✅ | [] |
| 16 | gt_guard negative control raises on injected GT key | ✅ |  |
| 17 | harness/backends/prompts never reference gt_1086.csv | ✅ | [] |
| 18 | codex_cli backend disabled without explicit enablement | ✅ |  |
| 19 | openai_compat backend disabled without explicit enablement | ✅ |  |
| 20 | codex CLI probed locally (availability evidence, no LLM call) | ✅ | {"available": true, "path": "D:\\npm-global\\codex.CMD", "version": "codex-cli 0.160.1", "returncode": 0} |
| 21 | all selftest rows satisfy output schema | ✅ | 0 |
| 22 | all stub outputs non-empty and status ok | ✅ | 0 |
| 23 | stub outputs deterministic (temperature 0 semantics) | ✅ | 3 samples x C/D/E |
| 24 | A baseline metrics match direct canonical recomputation | ✅ | {"run": 23.6188, "direct": 23.6188} |
| 25 | manifest status == IMPLEMENTED_UNTESTED | ✅ | IMPLEMENTED_UNTESTED |
| 26 | manifest data_status == STUB_ONLY_NOT_LLM_RESULTS | ✅ | STUB_ONLY_NOT_LLM_RESULTS |
| 27 | manifest records prompt json sha256 | ✅ |  |
| 28 | manifest records prompt md sha256 | ✅ |  |
| 29 | manifest records candidate_outputs/e03 hashes | ✅ |  |
| 30 | manifest groups recorded | ✅ | ["A", "C", "D", "E"] |
| 31 | decoding fixed temperature=0 / max_tokens=256 | ✅ |  |
| 32 | fusion_ablation_g2.csv has A/C/D/E x 2 runs + B BLOCKED | ✅ | 9 |
| 33 | all stub rows in ablation csv are labelled STUB_ONLY | ✅ |  |

## 7. 边界

1. 本报告所有 BLEU/chrF/编辑相似度都是 **stub 规则输出** 的指标，用于验证管线，**不得引用为 LLM 消融结果**。
2. 1086 是 retrospective_seen 子集；候选文本是冻结历史预测，不是本轮重跑。
3. 真实 LLM 的 token 数/时延/BLEU 全部为空（未调用），不得填估计值。
4. A/A2/B0 的 E02 指标见 `experiments/p1/e02/translation_metrics.json`；本 harness 的 A 组指标只用于对照一致性。
5. B（SLRT G2T/SLT）仍 BLOCKED（外部权重缺失）；B0 join 12.22 BLEU 是拼接基线，不代表 SLRT 翻译能力。

