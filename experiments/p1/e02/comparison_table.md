# E02 同任务翻译基线与互补性（imported 1086 快照）

- run_id：`e02-offline-1086-20261026T134508Z`
- 样本：CSL-Daily test 1086 条（retrospective_seen，非新盲测）
- 参考口径：上游 `test_tmp_refs.txt`（半角标点、逐字分隔）
- 状态：A/A2 为历史快照重算（IMPORTED_MEASURED）；B0 为确定性 Gloss join（DERIVED_BASELINE）；C–G/B1 BLOCKED

| 分支 | n | BLEU-4 | chrF | 平均字符编辑相似度 | 完全匹配 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| A — Uni-Sign RGB/video -> sentence (frozen 1086 snapshot) | 1086 | 24.8157 | 21.7974 | 0.4157 | 15 | IMPORTED_MEASURED |
| A2 — Uni-Sign released pose -> sentence (frozen 1086 snapshot) | 1086 | 25.5846 | 22.4284 | 0.4212 | 18 | IMPORTED_MEASURED |
| B0 — SLRT Gloss -> deterministic token join (fallback baseline, not a translator) | 1086 | 12.2183 | 13.8087 | 0.3484 | 0 | DERIVED_BASELINE |
| SLRT_gloss — SLRT Gloss (CSLR) on frozen 1086 snapshot | — | — | — | — | — | IMPORTED_MEASURED |
| ORACLE_A_A2 — offline GT oracle over A/A2 (upper bound, not deployable) | — | — | — | — | — | OFFLINE_ORACLE |
| B1 — SLRT Gloss -> mBART full-vocab fallback G2T | — | — | — | — | — | BLOCKED |
| C — SLRT TwoStream-SLT -> sentence | — | — | — | — | — | BLOCKED |
| D — SLRT Gloss -> LLM -> sentence | — | — | — | — | — | BLOCKED |
| E — Uni-Sign RGB -> LLM (polish control) | — | — | — | — | — | BLOCKED |
| F — Uni-Sign + SLRT gloss -> LLM | — | — | — | — | — | BLOCKED |
| G — dual model + evidence-constrained LLM | — | — | — | — | — | BLOCKED |

## A vs A2 配对（同一批 1086 样本）

- A2−A 平均字符编辑相似度差：**+0.0055**（95% CI -0.0013 … +0.0116）
- A2−A BLEU-4 差：**+0.7689**（95% CI +0.2299 … +1.4372）
- 逐样本胜负：A2 更好 241、A 更好 203、持平 642
- 纠正（A 差→A2 明显更好）：93；误改（A 好→A2 明显更差）：50

## 误差重叠（sim<0.5 视为差）

- 两者都差：661；仅 A 差：43；仅 A2 差：29；两者都好：353

## GT Oracle 上限（仅离线分析，不可部署）

- A/A2 逐样本 oracle：BLEU-4 **26.598**，chrF **23.2554**，平均编辑相似度 **0.443**（选择分布 {'A': 830, 'A2': 256}）

## Legacy vs Canonical

- Uni-Sign 视频 legacy BLEU-4 24.8156828061 → canonical 24.8157
- SLRT legacy corpus WER 0.3018248175 → canonical 0.3018
