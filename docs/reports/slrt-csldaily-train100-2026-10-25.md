# SLRT 100 条 train 拟合检查

日期：**2026-10-26**。对象：本地 SLRT-CSL（pinned `cslr_best.ckpt`）在 `D:\Zenodo\csl-daliy` 的 **657 条 train 中分层抽取 100 条**上的拟合检查。抽样按 signer 比例、seed=0，不是随机全量；仍是**近似对齐裁剪**，不是官方 `sentence_frames-512x512` 协议。

## 抽样与协议

- 657 条 train（本地视频 ∩ `csl-daily.train`）按 signer 分层抽 100 条；9 个 signer，样本清单 `artifacts/unisign-csldaily-train-100-20261025/selection.json`。
- 原始 mp4 未裁剪，用 RTMPose 全帧 63 点 + 官方 keypoints 做过滤后的滑窗对齐，裁剪 `num_frames`，生成 `hrnet-npz`。
- SLRT：pinned checkpoint、window 16 / stride 1 / vote 7；`--require-gloss` 未使用。
- 实现：`scripts/align_slrt_csl_daily_train100.py`（由 52-test 脚本复制并改 selection/annotation/output；未改现有源码）。

## 结果

| 指标 | 数值 |
|---|---:|
| attempted / aligned / success | **100 / 100 / 100** |
| 空输出 / 错误 | 0 / 0 |
| **corpus WER（100 条完整）** | **0.20914（20.9%）** |
| S / D / I | 88 / 84 / 98 |
| 参考 token 数 | 1,291 |
| 对齐距离（过滤后） | mean **6.87 px**、median 6.26、max 16.88 |
| 低置信度对齐（>25 px） | **0** |
| 逐条 WER 均值 / 中位 | 0.238 / 0.167 |
| WER=0 / ≤0.1 / ≤0.2 / ≤0.3 / ≤0.5 | 12 / 31 / 65 / 78 / 94 条 |

最差样本 **S004220_P0000_T00 WER 3.25**（对齐 7.45 px、start=1）仍是对齐正常但模型大量插入/改写；`S004289_P0007_T00` WER 1.86（对齐 5.04 px）类似。说明这批 WER 主要不是裁错，而是模型过生成/难句。

## 拟合 vs 泛化

| 数据 | SLRT WER | Uni-Sign BLEU-4 |
|---|---:|---:|
| 100 条 train 子集（本报告） | **0.2091** | — |
| 657 条 train 子集（Uni-Sign，仅对照） | — | **85.8240**（exact 654/657） |
| 52 条 test 子集 | **0.3300** | **24.1558** |
| 完整 1,176 test | — | **25.5504** |

SLRT 在 train 上 WER 20.9%，低于 test 的 33.0%，存在拟合但不像 Uni-Sign 那样近乎记忆；两模型任务/指标不同，不能直接横向比较。

## 自检

100/100 的 trimmed/NPZ 结构不变量通过（帧数、shape、finite、confidence、indices、timestamps、video_sha256、metadata），100 个 SLRT JSON 均可解析；随机 2 条详检通过。自检记录：`artifacts/slrt-100-train-fit-20261025/selfcheck.json`。

## 产物

- 汇总：[summary.json](../../artifacts/slrt-100-train-fit-20261025/summary.json)
- 逐条表：[comparison-aligned.md](../../artifacts/slrt-100-train-fit-20261025/comparison-aligned.md) / [CSV](../../artifacts/slrt-100-train-fit-20261025/comparison-aligned.csv)
- 自检：[selfcheck.json](../../artifacts/slrt-100-train-fit-20261025/selfcheck.json)
- 脚本：[align_slrt_csl_daily_train100.py](../../scripts/align_slrt_csl_daily_train100.py)
- 抽样：[selection.json](../../artifacts/unisign-csldaily-train-100-20261025/selection.json)

## 边界

这是 **100 条 train 子集 + 近似对齐裁剪**；不得称完整 train（657/18,401）、官方 frame 协议或论文同协议复现。20.9% 只用于拟合检查，不能当泛化准确率。
