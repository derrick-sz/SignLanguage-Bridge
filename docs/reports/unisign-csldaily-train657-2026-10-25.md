# Uni-Sign 657 条 train 拟合检查

日期：**2026-10-25**。对象：本地 Uni-Sign pose-only（`csl_daily_pose_only_slt.pth`）在 `D:\Zenodo\csl-daliy` 的 **657 条 train 视频**（与作者 `csl-daily.train` 交集）上的训练集拟合检查。输入为作者官方 `csl_daily_pose_format.zip`（`pose-rtmpose-192-clean`），不是视频 RGB 端到端，也不是 SLRT。

## 结果

| 指标 | 数值 |
|---|---:|
| attempted / successful | **657 / 657** |
| failed / empty | 0 / 0 |
| BLEU-4（SacreBLEU 13a） | **85.8240** |
| chrF | 79.6230 |
| char ROUGE-L F1 | 0.99979 |
| 平均字符编辑相似度 | 0.99968（中位 1.0） |
| 完全匹配 | **654 / 657** |
| 预测 / 参考字符数 | 15,808 / 15,812 |

对照：同一模型完整 test 1,176 条 BLEU **25.5504**；52 条 test 子集 BLEU **24.1558**。训练集 BLEU 85.8、exact 654/657 表明模型对训练样本接近记忆，属于**明显过拟合/拟合检查**，不能作为泛化准确率。

## 读法

- 657 是本地视频子集的 train 部分，不是完整 CSL-Daily 训练集（18,401 条）；也不是留出集。
- 高 BLEU/编辑相似度主要反映训练拟合，不是识别能力；与 test 25.55 的差距才是泛化差距。
- Uni-Sign 是 pose-only；此结果不证明 RGB 端到端，也不证明 SLRT 的 Gloss 识别能力。
- SLRT 的 657 train 拟合检查需要逐条对齐/裁剪原始视频并跑 657 次在线推理，按 52 条试点实测约 **6 小时**（align ~3h + infer ~3h）；本轮未启动全量，避免把长时间运行伪装成结果。

## 产物

- 汇总：`artifacts/unisign-csldaily-train-657-20261025/output/summary.json`
- 逐条预测：`artifacts/unisign-csldaily-train-657-20261025/output/predictions.jsonl`
- pose/refs：`artifacts/unisign-csldaily-train-657-20261025/{pose,refs.txt}`
- 复跑脚本：`scripts/evaluate_unisign_csldaily_pose.py`
