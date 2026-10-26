# Uni-Sign 完整 CSL-Daily 1,176 条 test 复跑

日期：**2026-10-25**。这是 Uni-Sign pose-only 模型在 **完整 CSL-Daily test（1,176 条）** 上的复跑；输入是作者发布的 `csl_daily_pose_format.zip`（内部 `pose-rtmpose-192-clean/`），不是视频 RGB 端到端，也不是 SLRT。

## 结果

| 指标 | 数值 |
|---|---:|
| attempted / successful | **1,176 / 1,176** |
| failed / empty | 0 / 0 |
| BLEU-4（SacreBLEU 13a，上游 mt5 空格修复） | **25.5504** |
| chrF | 22.3893 |
| char ROUGE-L F1 | 0.5342 |
| 平均字符编辑相似度 | 0.4204（中位 0.3810） |
| 完全匹配 | 19 / 1,176 |
| 预测 / 参考字符数 | 17,912 / 19,288（BP 0.9261） |

与论文 CSL-Daily pose-only BLEU **25.61** 差 0.06；与项目此前 R10 的 **25.55** 一致。1-gram/4-gram 精度 58.04% / 14.13%，说明这是长句翻译任务，不是逐字分类准确率。

## 与 52 条子集的关系

同一模型在 52 条视频子集上 BLEU **24.16**，低于完整 1,176 条的 25.55；原因是 52 条子集的参考句平均 **23.6 字**，完整 test 平均 **16.4 字**，长句更难。不能把 24.16 当模型整体水平，也不能把 25.55 当视频 RGB 端到端成绩。

## 产物

- 汇总：`artifacts/unisign-csldaily-full-test-20261025/output/summary.json`
- 逐条预测：`artifacts/unisign-csldaily-full-test-20261025/output/predictions.jsonl`（1,176 行）
- pose：`artifacts/unisign-csldaily-full-test-20261025/pose/`；参考：`refs.txt`
- 复跑脚本：`scripts/evaluate_unisign_csldaily_pose.py`
