# E02 实验报告 — 同任务翻译基线与互补收益（离线回放）

## A. 实验基本信息
- 实验编号：E02
- run_id：`e02-offline-1086-20261026T134508Z`
- 状态：**PASS（离线回放部分）**；B1/C/D/E/F/G **BLOCKED**
- 样本范围：CSL-Daily test **1086 / 1176**（retrospective_seen_1086，10 名 signer；缺 90 条偏长句视频）
- 数据来源：冻结历史快照（A=Uni-Sign 视频端到端、A2=官方 pose、SLRT=Gloss），**本轮没有重新推理**
- 指标定义：`p1-v1`（sacreBLEU 13a + 上游 mt5 逐字加空格后处理；chrF 原始字符串；WER token 级）
- 参考口径：上游 `test_tmp_refs.txt`（半角标点、空格分字）；官方标注（全角标点）仅诊断列
- Git SHA：cf291c2a81558faf6de0dc5a7243e730531e4367（dirty tree=True）

## B. 复现实验步骤
```bash
$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'
.\.conda-env\python.exe -m experiments.p1.e02.run_e02
```
- 输入快照与哈希见 `experiments/p1/runs/e02-offline-1086-20261026T134508Z/manifest.json`
- 数据台账：`experiments/p1/datasets/manifest_1086.csv`
- GT（eval-only）：`experiments/p1/references/gt_1086.csv`

## C. 结果（imported / derived，非本轮重新推理）
| 分支 | 样本数 | BLEU-4 | chrF | 平均字符编辑相似度 | 完全匹配 |
|---|---:|---:|---:|---:|---:|
| A Uni-Sign 视频 | 1086 | 24.8157 | 21.7974 | 0.4157 | 15 |
| A2 Uni-Sign 官方 pose | 1086 | 25.5846 | 22.4284 | 0.4212 | 18 |
| B0 Gloss 确定性 join（fallback baseline） | 1086 | 12.2183 | 13.8087 | 0.3484 | 0 |
| SLRT Gloss（CSLR, corpus WER） | 1086 | — | — | — | — |

- SLRT corpus WER：**0.3018**（S 1330 / D 717 / I 434）
- Legacy vs canonical：视频 legacy BLEU 24.8156828061 vs canonical 24.8157；SLRT legacy WER 0.3018248175 vs 0.3018
- Bootstrap（按 signer 聚类）与 A/A2 配对结果见 `translation_metrics.json` / `comparison_table.md`

## D. 错误分析
- A vs A2 误差并不完全重叠（两者都差 661 条，仅 A 差 43、仅 A2 差 29）→ 存在可融合空间，但需要无 GT 的可观测信号（E03/E06）。
- 未做人评；`human_review.status = NOT_RUN`。严重语义错误率必须由人评协议给出，本文不替代。

## E. 结论与工程选择
- 是否支持原假设：**INCONCLUSIVE（质量提升潜力存在，但融合未实测）**
- 可引用结论：A/A2 两路在同一 1086 条上分别为 BLEU 24.8157 / 25.5846；差 +0.7689；逐样本强弱交叉存在。
- C 分支需先取得 CSL-Daily SLT/G2T 权重；B1 需作者剪枝 mBART 或验证 full-vocab fallback；D–G 需固定 LLM 端点与预算。
- 对 V3.0：保留双模型接口；生产默认单模型，融合需等 E06 无 GT 路由证据。
- 下一步最小行动：E03（SLRT score 校准）+ E06（Rule-only 路由）。

## 边界
- 1086 是 retrospective_seen，不是新盲测；所有结论不能外推到陌生 signer/新采集条件。
- 历史快照的推理协议见对应报告（R17/R18）；本报告只重算指标。
