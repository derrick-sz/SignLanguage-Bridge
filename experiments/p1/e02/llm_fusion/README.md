# G2 阶段 2-A · LLM 融合消融 harness（A/C/D/E）

`experiments/p1/e02/llm_fusion/` 实现 G2 核心对照实验的可审计管线。

- **真实批（已执行，2026-10-26）**：`openai_compat` 后端已实现真实 HTTP 调用，端点 = **wb-proxy**
  `http://127.0.0.1:8788/v1`、model = `deepseek-v4.1-flash`、密钥运行时只在内存解析（`WB_PROXY_API_KEY@dsh-credentials`）。
  50 条探索批（pilot_50：10 signer × 5）× C/D/E = 150 次真实调用：
  **`runs/g2a-wbproxy-50-v2-20261026/`**，报告见 [`report_real_50.md`](report_real_50.md)。
  首轮 `max_tokens=256` 因模型内部推理截断 73/150 次（事故记录见
  [`runs/g2a-wbproxy-50-20261026/incident_note.md`](runs/g2a-wbproxy-50-20261026/incident_note.md)），
  正式数据改用 `max_tokens=2048`（prompt 文本未变，仍 `v1_zh`）。
- **stub 自检**：`selftest.py` 在 CPU 上验证管线（状态 `IMPLEMENTED_UNTESTED` 仅针对未接入的 codex_cli 路径）；
  stub 数字只用于管线验证，见 [`report.md`](report.md) 与 `fusion_ablation_g2.csv` 的 STUB_ONLY 行。

## 1. 分组（用户 2026-10-26 批复字母）

| 组 | 方案 | 是否调 LLM | 输入证据 |
|---|---|---|---|
| A | Uni-Sign 原始译文（基准） | 否 | A 文本直通 |
| B | SLRT → G2T/SLT | — | **BLOCKED**（CSL-Daily G2T/SLT 权重未发布）；B0 只是确定性 join 退化基线 |
| C | Uni-Sign → LLM（润色对照） | 是 | 仅 Uni-Sign 译文 |
| D | Uni-Sign + SLRT Gloss → LLM | 是 | Uni-Sign 译文 + SLRT Gloss（确定性 join，即 E02 B0） |
| E | 双模型 + 证据约束 → LLM | 是 | D + E06 RuleJudge 槽位冲突提示（数字/否定/人物/疑问）+ E03 **未校准** SLRT 分数特征 |

核心判据（预先约定）：`C vs A` 衡量 LLM 润色收益；`D vs C` 衡量 Gloss 的新增价值；`E vs D` 衡量证据约束。
最终结论需要新独立测试集 + 人工评审（见 `experiments/p1/e06/human_review/`）。

**50 条真实批结果速览**（探索性，非 G2 结论）：A=31.23 BLEU / C=26.68 / D=26.95（n=49）/ E=28.63（n=48）；
C 显著低于 A（ΔBLEU -4.55，CI 不含 0）；D−C 与 E−D 的 BLEU/edit CI 均含 0（不显著），
C→E 的 edit CI 不含 0（+0.040）。详见 `report_real_50.md`。

## 2. 目录

```text
experiments/p1/e02/llm_fusion/
  harness.py                # GT-free：候选 -> prompt -> backend -> runs/<run_id>/fusion_outputs.jsonl
  evaluate.py               # EVAL ONLY：GT join + canonical p1-v1 指标 + signer 聚类配对 bootstrap
  selftest.py               # stub 全链路自检 + 断言 + fusion_ablation_g2.csv + report.md
  backends/
    base.py                 # BackendResult / LLMBackend 接口 / BackendDisabled
    stub.py                 # 确定性 stub（仅自测，不是模型）
    codex_cli.py            # 本机 codex CLI 封装（已实现，默认不调用）
    openai_compat.py        # OpenAI 兼容端点接口（仅接口，无 HTTP 调用）
  prompts/
    v1_zh.json / v1_zh.md   # 版本化 prompt（sha256 写进 manifest）
  runs/<run_id>/            # fusion_outputs.jsonl / manifest.json / eval_records.jsonl / metrics.json
  fusion_ablation_g2.csv    # 消融总表（stub 数字明确标注）
  report.md                 # 自检与接入报告
  selftest_result.json      # 断言明细
```

## 3. 运行

```powershell
$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'
cd E:\githubproject\SignLanguage-Bridge

# 15 条 signer 分层样本 stub 自检（含 bootstrap）+ 1086 全量 dry-run + 全部断言
.\.conda-env\python.exe -m experiments.p1.e02.llm_fusion.selftest

# 单独跑某次构建（stub）
.\.conda-env\python.exe -m experiments.p1.e02.llm_fusion.harness run `
    --backend stub --limit 15 --run-id my-stub-run --overwrite

# 单独评测某个 run（EVAL ONLY，GT join）
.\.conda-env\python.exe -m experiments.p1.e02.llm_fusion.evaluate --run-id my-stub-run
```

## 4. 切换真实 LLM（需用户/Lead 批准）

候选端点：① 本机 Codex CLI（已登录，`codex exec`）；② 用户提供的 OpenAI 兼容端点。

固定项：prompt 版本（当前 `v1_zh`）、temperature=0、max_tokens=256、模型 id/版本、调用预算。
启用开关：`P1_LLM_ENABLE_REAL=1` + CLI `--allow-real`。

- Codex：`P1_CODEX_MODEL=<model>`；调用形态
  `codex exec --json --ephemeral --sandbox read-only -m <model> -o <tmpfile> -`（prompt 走 stdin）。
- OpenAI 兼容：`P1_OPENAI_BASE_URL` / `P1_OPENAI_API_KEY` / `P1_OPENAI_MODEL`（密钥只在进程环境注入，不入库）；
  **HTTP 客户端尚未实现**，需补齐后调用。

真实跑批：先 `--limit 50` 探索批，再按需 1086；每次新 `run_id`；真实数字必须标 `real`，不得与 stub 数字混表。

## 5. GT 隔离（硬约束）

- `harness.py` / `backends/` / `prompts/` **不读取** `references/gt_1086.csv`；GT join 只在 `evaluate.py`。
- `experiments/p1/e03/per_sample_scores.csv` 含 GT 标签列，loader 按白名单只读 token/window 原始分数列
  （`harness.SCORE_FEATURE_WHITELIST`）；`wer/bad/error_any/oof_p_bad_*` 等全部禁止。
- 每行输出写入前过 `gt_guard.assert_gt_clean`；`selftest.py` 另有 0 泄漏断言、prompt 不含 GT 句子断言与负例控制。
- `eval_records.jsonl` 是唯一含 GT 的文件（EVAL ONLY，`guard=False`），与推理载荷物理分离。

## 6. 当前状态与边界

- 状态：`IMPLEMENTED_UNTESTED`（真实 LLM 未调用）；stub 自检 PASS（见 `selftest_result.json` / `report.md`）。
- **stub 输出是确定性规则回显，不是 LLM 结果；stub 指标只用于验证管线。**
- 1086 是 retrospective_seen 子集，候选为冻结历史预测；不是新盲测、不是本轮重跑。
- A2（官方 pose）未进入本轮 LLM 输入：A 与 A2 同属 Uni-Sign，双模型指 Uni-Sign + SLRT；A2 作为备选路径
  留给后续消融（如需要可加组 F/A2）。
