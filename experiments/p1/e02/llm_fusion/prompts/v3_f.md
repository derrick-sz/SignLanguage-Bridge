# Group F prompt v3_f（B0 gloss → 自然中文句；LLM 只做语言组织）

> 版本：`v3_f`（2026-10-27）｜状态：实验用
> 对应：`docs/experiments/P-1/G2_IMPROVEMENT_ANALYSIS.md` 方向 1（最有前景的未测路径）
> 与 C/D/E 的关键区别：**不给 Uni-Sign 译文**。LLM 不再做"事实仲裁"，只把 SLRT Gloss 词列组织成自然句。

## System

```text
你是手语翻译助手。输入是手语识别模型给出的词汇序列。请把这些词汇组成一句自然流畅的简体中文句子。规则：1 只输出一句中文译文；2 必须保留所有词汇的意思，不得添加、删除或改变任何内容；3 不得虚构词汇之外的信息；4 使用简体中文与中文标点。
```

## User template（group F）

```text
【任务】把下面这些手语识别出的词汇组成一句自然流畅的简体中文句子。保留所有词汇的意思，不添加不删除不改变。
【词汇序列】{gloss_text}
【输出】仅输出一句中文译文。
```

- `{gloss_text}` = E02 `candidate_outputs.jsonl` 中 **B0**（SLRT Gloss 确定性输出）的文本，GT-free。
- **不注入** A / A2、槽位冲突、分数、候选身份、任何 GT 或 EVAL_ONLY 字段。

## 解码

- `temperature=0`、`max_tokens=2048`、`top_p=1.0`；串行（concurrency=1）。
- 端点：wb-proxy `POST http://127.0.0.1:8788/v1/chat/completions`，model `deepseek-v4.1-flash`（与 v1/v2 真实运行相同）。
- 密钥运行时从 `refs.WB_PROXY_API_KEY` 读取，只在内存；不打印、不落盘（manifest 只记录 `credentials_ref`）。

## 实验设计（50 条 pilot）

- 同一 `pilot_50.csv` 的 50 条样本（与 v1/v2 真实运行同集）。
- 输出：GT-free `runs/g2f-wbproxy-50-v3f-*/fusion_outputs.jsonl` + `manifest.json`（prompt version/sha256、endpoint、decoding、样本数、失败数）。
- 评测（EVAL ONLY）：F 的 BLEU-4/chrF/edit + 与 **A、B0** 的配对 signer bootstrap（F−A、F−B0 各报 Δ 与 CI）。
- 报告：`report_group_f.md`，含 F vs A vs B0 三列、5–10 条 (GT/A/B0/F) 并排示例、失败/异常分析；无论结果好坏如实报告。

## 纪律

1. F 的 prompt 只能含 Gloss 词列；harness 对"Uni-Sign 原文出现在 prompt 中"做硬断言。
2. 不把参考译文/GT Gloss/WER/edit 标签/GT 派生分数写进 prompt。
3. 不扩 1086（本轮只 50 条 pilot）；不改 `slbridge/`；不 commit。
4. 结论是 50 条探索性证据，不单独构成 G2 最终结论；需新独立集/人评复核。
