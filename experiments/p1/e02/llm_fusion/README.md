# G2 阶段 2-A · LLM 融合消融 harness（A/C/D/E）

`experiments/p1/e02/llm_fusion/` 实现 G2 核心对照实验的可审计管线。

- **真实批 v1（已执行，2026-10-26）**：50 条 pilot_50（10 signer × 5）× C/D/E = 150 次真实调用，
  `runs/g2a-wbproxy-50-v2-20261026/`，报告 [`report_real_50.md`](report_real_50.md)（prompt `v1_zh`）。
- **真实批 v2（已执行，2026-10-26/27）**：版本化 prompt **`v2_zh`**（C 最小必要修正 / D 以 A 为主仅冲突槽位修正 / E 一句译文+一行理由）
  同 50 条 pilot_50 样本重跑：**`runs/g2a-wbproxy-50-v2prompt-20261026/`**；
  报告 [`report_prompt_v2.md`](report_prompt_v2.md)，选择性级联离线分析 [`cascade_analysis_v2.json`](cascade_analysis_v2.json)。
  v2 把 C/D 的 ΔBLEU 从 v1 的 -4.55/-4.65 收窄到 -2.28/-2.64（仍为负），D−C 与 E−D 在两版中均无显著增益；
  级联（默认 A、RuleJudge 触发→E，触发率 28%）质量与 A 无显著差异、tokens 22,981（全调 E 61,393），但**质量不优于 A → 按当前证据不值得为质量付该成本**。
  边界：A/A2/B0 为 E02 冻结历史预测；C/D/E 为真实 LLM 生成；50 条为 retrospective 子集，非最终 G2 结论。
  首轮 v1 `max_tokens=256` 因模型内部推理截断 73/150 次（事故记录见
  [`runs/g2a-wbproxy-50-20261026/incident_note.md`](runs/g2a-wbproxy-50-20261026/incident_note.md)），
  v1 正式数据用 `max_tokens=2048`；v2 版本内固定 2048，E 组仍有 5 次 2048 长推理截断（建议后续 E 单独给 4096 预算，本轮未重跑）。
- **stub 自检**：`selftest.py` 在 CPU 上验证管线（`codex_cli` 路径仍 IMPLEMENTED_UNTESTED）；
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
  run_real_50.py            # v1_zh 真实 50 条批（wb-proxy/deepseek-v4.1-flash）
  run_prompt_v2.py          # v2_zh 真实 50 条批（批跑）；分析委托 analyze_prompt_v2.py
  analyze_prompt_v2.py      # 离线分析：v1 vs v2、选择性级联、report_prompt_v2.md
  backends/
    base.py                 # BackendResult / LLMBackend 接口 / BackendDisabled
    stub.py                 # 确定性 stub（仅自测，不是模型）
    codex_cli.py            # 本机 codex CLI 封装（已实现，默认不调用）
    openai_compat.py        # 真实 OpenAI 兼容 HTTP（urllib，wb-proxy；密钥仅内存）
  prompts/
    v1_zh.json / v1_zh.md   # 版本化 prompt（真实 v1 批使用，勿改）
    v2_zh.json / v2_zh.md   # 基于 v1 证据的最小修正版（C/D/E 指令）
  runs/<run_id>/            # fusion_outputs.jsonl / manifest.json / eval_records.jsonl / metrics.json
  fusion_ablation_g2.csv    # 消融总表（STUB_ONLY / REAL v1 / REAL v2 分列，含 prompt_version）
  report.md                 # stub 自检与接入报告
  report_real_50.md         # v1 真实 50 条报告（含 max_tokens 事故）
  report_prompt_v2.md       # v2 vs v1 + 选择性级联 + G2 建议
  cascade_analysis_v2.json  # 级联离线分析明细
  selftest_result.json      # stub 断言明细
```

## 3. 运行

```powershell
$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'
cd E:\githubproject\SignLanguage-Bridge

# stub 自检（15 条 + 1086 dry-run + 全部断言）
.\.conda-env\python.exe -m experiments.p1.e02.llm_fusion.selftest

# 真实 v2_zh 50 条批（需 P1_LLM_ENABLE_REAL=1；端点 wb-proxy、model deepseek-v4.1-flash）
$env:P1_LLM_ENABLE_REAL='1'
.\.conda-env\python.exe -m experiments.p1.e02.llm_fusion.run_prompt_v2

# 单独评测某个 run（EVAL ONLY，GT join）
.\.conda-env\python.exe -m experiments.p1.e02.llm_fusion.evaluate --run-id <run_id>
```

## 4. 真实 LLM 端点（用户已选定并用于实验）

- **wb-proxy**：`POST http://127.0.0.1:8788/v1/chat/completions`，model=`deepseek-v4.1-flash`，Bearer 认证。
- 密钥：运行时只在内存解析 `C:\Users\ADMIN\.dsh\.credentials.yaml` 的 `refs.WB_PROXY_API_KEY`
  （或环境变量 `WB_PROXY_API_KEY` / `P1_OPENAI_API_KEY`）；**不打印/不落盘**；manifest 只写
  `credentials_ref: WB_PROXY_API_KEY@dsh-credentials`。
- 启用：`P1_LLM_ENABLE_REAL=1` + `--allow-real`；参数 temperature=0、max_tokens=2048、串行。
- 备选：本机 Codex CLI 封装保留（`codex_cli`，默认不调用）。

## 5. GT 隔离（硬约束）

- `harness.py` / `backends/` / `prompts/` **不读取** `references/gt_1086.csv`；GT join 只在 `evaluate.py`。
- `experiments/p1/e03/per_sample_scores.csv` 含 GT 标签列，loader 按白名单只读 token/window 原始分数列
  （`harness.SCORE_FEATURE_WHITELIST`）；`wer/bad/error_any/oof_p_bad_*` 等全部禁止。
- 每行输出写入前过 `gt_guard.assert_gt_clean`；`selftest.py` 另有 0 泄漏断言、prompt 不含 GT 句子断言与负例控制。
- `eval_records.jsonl` 是唯一含 GT 的文件（EVAL ONLY，`guard=False`），与推理载荷物理分离。

## 6. 当前状态与边界

- 真实 LLM 批已执行（v1_zh 与 v2_zh，各 50 条；端点 wb-proxy/deepseek-v4.1-flash）。
- **G2 结论（探索性）**：v1/v2 两版中 D−C 与 E−D 均无显著字面增益；所有 LLM 组 BLEU 点估计均低于 A；
  选择性级联（默认 A、触发→E）质量与 A 无显著差异，但质量不优于 A → **按当前证据不值得为质量付出 LLM 调用成本**；
  融合暂不进主链路（最终仍需新独立集 + 人工评审）。
- **stub 输出是确定性规则回显，不是 LLM 结果；stub 指标只用于验证管线。**
- E 组若继续实验建议单独给 4096 max_tokens（本批 5/50 在 2048 处截断，未重跑）。
- 1086 是 retrospective_seen 子集，候选为冻结历史预测；不是新盲测、不是本轮重跑。
- A2（官方 pose）未进入本轮 LLM 输入：A 与 A2 同属 Uni-Sign，双模型指 Uni-Sign + SLRT；A2 作为备选路径
  留给后续消融（如需要可加组 F/A2）。
