# Uni-Sign on 52 CSL-Daily test videos (subset)

日期：**2026-10-25**。对象：本地 `csl_daily_pose_only_slt.pth`（pinned，SHA256 `73bea87a…c1a6c`）在 `D:\Zenodo\csl-daliy` 视频子集对应的 **52 条 CSL-Daily test** 上的翻译结果。此报告不是完整 1,176 条 test 复现，也不是分类准确率。

## 输入与协议

- 视频子集：`D:\Zenodo\csl-daliy` 743 个 mp4 ∩ `csl-daily.test` = **52 条 test**（不随机、不挑选）。
- 姿态输入：作者发布的 **Uni-Sign CSL-Daily pose archive** `csl_daily_pose_format.zip`（4,594,281,295 B，内部目录 `pose-rtmpose-192-clean/`，20,654 个 pkl），不是把 SLRT/TwoStream 的 HRNet keypoints 直接转换。
- 参考文本：上游 `out/eval_csl_daily_pose_only/test_tmp_refs.txt`（逐字空格分隔）。
- 模型：`models/external/uni_sign/csl_daily_pose_only_slt.pth`；bundle 默认；bf16；beam=4；GPU。
- 评测脚本：`scripts/evaluate_unisign_csldaily_pose.py`；逐条结果 `output-official/predictions.jsonl`。

## 结果（n=52）

| 指标 | 数值 |
|---|---:|
| 成功 / 尝试 | 52 / 52 |
| BLEU-4 (SacreBLEU 13a，按上游 mt5 空格修复) | **24.1558** |
| chrF | 21.2167 |
| char ROUGE-L F1 | 0.4975 |
| 平均字符编辑相似度 | 0.3786（中位 0.3801） |
| 完全匹配 | 0 / 52 |

与上游对同 52 条的预测对照：**35/52 逐字完全相同**，平均字符编辑相似度 0.924；说明本地推理协议与上游基本一致，剩余差异来自子集/输入细节。该子集 BLEU 低于本项目此前 **完整 1,176 条** 的 25.55，不能把 24.16 当全量结果。

子集偏长句：这 52 条参考句平均 **23.6 字**（中位 21.5、最大 42），完整 test 平均 **16.4 字**（中位 16、P90 23）。长句的 4-gram 命中率和编辑相似度天然更低；52 条不是完整 test 的随机抽样。

## 表格

- 逐条对照：[comparison-official.md](../../artifacts/unisign-csldaily-52-test-20261025/comparison-official.md)
- CSV：[comparison-official.csv](../../artifacts/unisign-csldaily-52-test-20261025/comparison-official.csv)
- 原始预测：[predictions.jsonl](../../artifacts/unisign-csldaily-52-test-20261025/output-official/predictions.jsonl)
- 汇总：[summary.json](../../artifacts/unisign-csldaily-52-test-20261025/output-official/summary.json)

## 协议边界与被否决的输入

- 这不是完整 CSL-Daily test（1,176 条）复现；52 条是视频子集交集，不是随机抽样。
- BLEU/编辑相似度是翻译指标，不是“识别准确率”；也不能与 SLRT 的 Gloss WER 直接比较。
- 先把 `F:\model_weigth\data\csl-daily\csl-daily-keypoints.pkl`（HRNet 133 点、512 像素）转成 `[T,133,2]/512` + scores 后跑同一模型：50/52 成功、BLEU **15.16**，与上游同 52 条只有 **2/50** 逐字相同；该输入与作者 Uni-Sign pose archive 不等价，**不采用该结果**。正式结果使用上面的官方 pose archive。
