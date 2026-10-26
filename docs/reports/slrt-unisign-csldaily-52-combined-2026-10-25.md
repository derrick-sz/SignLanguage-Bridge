# SLRT vs Uni-Sign：CSL-Daily 52 条 test 对比

日期：**2026-10-25**。同一批 **52 条 CSL-Daily test**（`D:\Zenodo\csl-daliy` ∩ `csl-daily.test`）上：

- **Uni-Sign**：pose-only，输入作者官方 pose archive（`pose-rtmpose-192-clean`），输出中文句子；指标 BLEU-4 / chrF / 字符编辑相似度。
- **SLRT**：RGB+pose 在线 Gloss 模型，输入原始 mp4 经 RTMPose + 官方 keypoints 近似对齐裁剪，输出 Gloss；指标词级 WER。

两者任务和指标不同，**不能横向比较谁更准**；下表只是同批样本的并排对照。

## 汇总

| 项目 | Uni-Sign 52 | SLRT 52 | Uni-Sign 完整 1,176 |
|---|---:|---:|---:|
| 成功 | 52 / 52 | 52 / 52 | 1,176 / 1,176 |
| 主指标 | BLEU-4 **24.1558** | WER **0.3300** | BLEU-4 **25.5504** |
| chrF | 21.2167 | — | 22.3893 |
| char ROUGE-L F1 | 0.4975 | — | 0.5342 |
| 平均字符编辑相似度 | 0.3786 | — | 0.4204 |
| 完全匹配 | 0 / 52 | — | 19 / 1,176 |
| S / D / I | — | 94 / 37 / 65 | — |
| 对齐（过滤后） | — | mean 6.88 px、median 6.31 | — |

- Uni-Sign 52 的逐条对照：[comparison-official.md](../../artifacts/unisign-csldaily-52-test-20261025/comparison-official.md)
- SLRT 52 的逐条对照：[comparison-aligned.md](../../artifacts/slrt-52-aligned-20261025/comparison-aligned.md)
- **合并表（GT + 两模型预测）**：[comparison-both.md](../../artifacts/slrt-unisign-csldaily-52-combined-20261025/comparison-both.md) / [CSV](../../artifacts/slrt-unisign-csldaily-52-combined-20261025/comparison-both.csv) / [summary.json](../../artifacts/slrt-unisign-csldaily-52-combined-20261025/summary.json)

## 读法

- **Uni-Sign 52 的 24.16 BLEU 低于完整 1,176 条的 25.55**：52 条参考句平均 23.6 字，完整 test 平均 16.4 字，子集偏长句；不是协议错误。
- **SLRT 52 的 33.0% WER** 是近似对齐子集，不是官方 `sentence_frames-512x512.zip`；它显著好于 CE-CSL 跨域 89.18%，说明同域链路可用，但不能与论文 CSL-Daily 在线 29.3% 或 CSLR test 27.4% 直接对标。
- 52 条中有 1 条 Uni-Sign 预测与参考完全一致；SLRT 有 1 条 WER=0。逐条差异大，长句和专有词是主要错误来源。
- 同一模型在 52 条与完整 1,176 条上的一致性说明子集选择偏差是 BLEU 差异的主因之一；SLRT 完整 test 仍需完整视频/官方帧包，当前无法用同一协议跑 1,176 条。
