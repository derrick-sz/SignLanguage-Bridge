# LLM 融合 prompt · v1_zh

本目录保存 **版本化、不可原地修改** 的 prompt 规格。机器可读版本是 [`v1_zh.json`](v1_zh.json)，
本文件只说明设计意图与变更规则。`harness.py` 只加载 JSON 文件；每次 run 的 manifest 会记录
JSON 与 MD 的 sha256。

## 版本规则

- 版本号形如 `vN_zh`；**已用于任何 run 的版本不得原地修改**，改 prompt 必须新建 `v2_zh.json` 并重新跑。
- manifest 记录：`prompt_version`、`prompts/v1_zh.json` sha256、`prompts/v1_zh.md` sha256。
- 后端/模型/温度/max_tokens 与 prompt 一起构成可审计配置；任何一项变化都要新 run_id。

## G2 分组（按用户 2026-10-26 批复字母）

| 组 | 方案 | 是否调用 LLM | 输入 |
|---|---|---|---|
| A | Uni-Sign 原始译文（基准） | 否 | 无（直通 candidate_outputs 的 A 文本） |
| C | Uni-Sign → LLM（润色对照） | 是 | 仅 Uni-Sign 译文 |
| D | Uni-Sign + SLRT Gloss → LLM | 是 | Uni-Sign 译文 + SLRT Gloss（确定性 join） |
| E | 双模型 + 证据约束 → LLM | 是 | 在 D 基础上加 E06 RuleJudge 槽位冲突提示 + E03 未校准 SLRT 分数特征 |
| B | SLRT → G2T/SLT | — | **BLOCKED（外部权重缺失）**；B0 只是确定性 join 退化基线，不能当翻译能力 |

## 固定 system prompt 要点（完整文本在 JSON）

1. 你是手语翻译助手，只输出一句自然中文，不解释、不加引号、不分行。
2. 不得虚构证据外内容（人物/数量/否定/时间/动作/对象）。
3. 证据冲突时以 Gloss 证据为事实依据，Uni-Sign 译文只用于语言组织。
4. 证据不足时保持原意。

## 解码参数（固定）

`temperature=0`、`max_tokens=256`、`top_p=1.0`、无 stop。真实后端若不能完全遵守，必须在
manifest 的 `decoding_effective` 里记录实际值，不得静默放宽。

## GT 隔离（硬约束）

- prompt 只由 `experiments/p1/e02/candidate_outputs.jsonl`（A/A2/B0 模型输出，GT-free）与
  **白名单 SLRT 分数列** 组装。
- **禁止**进入 prompt：`gt_gloss`、`ref_sentence_upstream`、`gt_sentence_official`、`wer`、`bad`、
  `error_any`、`uni_video_edit_sim_EVAL_ONLY`、`should_upgrade_proxy_EVAL_ONLY`、E03 的 OOF 概率列
  （`oof_p_bad_*`，由 GT 标签训练得到）等任何 GT 或 GT 派生值。
- `E03/per_sample_scores.csv` 含 GT 标签列；harness 的 loader 使用白名单，只读取
  token/window 原始分数列（见 `harness.py:SCORE_FEATURE_WHITELIST`），并有单测断言。
- 每次 run 的 `fusion_outputs.jsonl` 在写入前逐行过 `experiments/p1/gt_guard.assert_gt_clean`。
- 评测（GT join）在 `evaluate.py` 中单独进行，`harness.py` 与 `backends/` 不读取 GT 文件。

## 组 E 的槽位提示

复用 E06 RuleJudge 的确定性槽位抽取（`experiments/p1/e06/build_e06.py` 的 `extract_slots` /
`slot_conflicts`），对 `Uni-Sign 译文` 与 `SLRT Gloss` 逐槽位给出「Uni-Sign 值 / Gloss 值 / 一致或冲突」，
并在提示中明确「规则抽取，可能有误报」。槽位：数字/数量、否定极性、人物指代、疑问语气。
