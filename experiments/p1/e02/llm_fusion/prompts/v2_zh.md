# LLM 融合 prompt · v2_zh（基于 v1 证据的最小修正版）

机器可读版本是 [`v2_zh.json`](v2_zh.json)，本文件只记录设计意图与变更理由。v1_zh 已用于真实运行，
**保持不动**；v2 是独立版本文件。

## 为什么有 v2（v1 证据）

v1（50 条真实批，`report_real_50.md`）观察到：

- C（无条件润色）相对 A：ΔBLEU **-4.55**（CI 不含 0）、Δedit -0.033 → LLM 会改掉正确事实、补全证据外内容。
- D−C / E−D 的 BLEU/edit 增量 CI 均含 0（不显著）；人工抽检示例可见扩写与事实漂移。
- 根因判断：v1 要求"输出自然流畅译文"，模型倾向**重写整句**，语义漂移风险高于收益。

## v2 的改动（只改指令，不改任务）

| 组 | v1 | v2 |
|---|---|---|
| 系统 | 输出自然译文，冲突以 Gloss 为准 | 增加"**优先保留输入事实；不猜测/不扩写；只有证据明确支持才改事实**" |
| C | 润色成自然句子 | **最小必要修正**：仅语法/口语碎片/标点问题才改写；其余原样输出；保留全部事实 |
| D | 用 Gloss 核对 + 参考 Uni-Sign 组织 | **以 Uni-Sign 原句为主**；仅当 Gloss 与 A 在某个槽位**明确冲突**时修正该槽位；不扩写 |
| E | D + 槽位提示 | D + 槽位对比；输出"一句译文 + 一行 `理由：…`"（≤30 字），理由是处置说明而非分析过程 |

## 固定参数

`temperature=0`、`max_tokens=2048`、`top_p=1.0`（max_tokens 在 v2 版本内固定为 2048，避免 v1 的
推理截断事故）。manifest 记录 prompt json/md 的 sha256 与 effective decoding。

## 输出解析（E 组）

`output_format.E.mode = sentence_plus_reason`：

- 第一行非空文本 → `output_text`（参与评测）；
- 以 `理由：` 开头的后续行 → `output_reason`（仅审计，不参与指标）；
- 原始完整文本保存在 `output_raw_text`。

## GT 隔离（与 v1 相同）

prompt 只由 `candidate_outputs.jsonl`（A/B0 模型输出）与白名单 SLRT 原始分数列构成；
GT 只在 `evaluate.py` join；每次 run 逐行过 `gt_guard.assert_gt_clean`。

## 诚实性约束（预先声明）

- v2 是同一样本集上的 prompt 迭代；**不得**在测试参考上反复改 prompt 后声称"独立盲测提升"。
- 若 v2 仍无正增益，报告必须如实写"当前 prompt 条件下 LLM 融合无质量收益"，并说明证据与停止条件。
- 最终 G2 结论需要新独立测试集 + 人工评审。
