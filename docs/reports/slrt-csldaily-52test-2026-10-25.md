# SLRT CSL-Daily 52-test pilot：对齐裁剪与 Gloss WER

日期：**2026-10-25**。对象：本地 SLRT-CSL（pinned `cslr_best.ckpt`）在 `D:\Zenodo\csl-daliy` 视频子集对应的 **52 条 CSL-Daily test** 上的在线 Gloss 试点。此结果是**近似对齐的 52 条子集**，不是官方 `sentence_frames-512x512.zip` 协议，也不是完整 1,176 条 test。

## 为什么需要对齐

原始 mp4 是未裁剪长视频（例：`S000362_P0000_T00` 453 帧，标注 `num_frames=236`），不能直接喂 SLRT；否则多出的过渡/背景帧会引入额外 Gloss。官方 keypoints pkl 给出每条句子的 `[T,133,3]`（T=`num_frames`），可用它定位句段。

## 方法

- 52 条来自 `artifacts/unisign-csldaily-52-test-20261025/selection.json`（`D:\Zenodo\csl-daliy` ∩ `csl-daily.test`）。
- 用 RTMPose 前端（`rtmpose-experimental`，CUDA）对原始视频全帧提 133 点，取 `KEYPOINT_INDICES` 63 点（512 参考系）。
- 官方 HRNet keypoints 取同样 63 点；对齐仅使用两边都可靠的点：官方 confidence ≥0.2、坐标在 `[-64, 512+64]`、RTMPose confidence ≥0.1 且检测到人。
- **官方 pkl 有退化手点**：约 4.6% 的 133 点 confidence<0.2 或坐标越界（实测 x=-30320、y=32592、conf 0.0–0.1）。直接对全部 63 点求均值会被它们主导（全点 mean 19.02 px、最高 61.5 px），过滤后 mean 6.88 px；因此**只在对齐打分时过滤**，NPZ 仍保留全部 133 点（非有限值逐点插值、confidence clip 到 [0,1]，不筛点、不改官方数据）。
- 在原始视频上穷举所有长度 T 的窗口（stride 1），取过滤后均值欧氏距离最小的起点 `best_start`。
- 以 `best_start:best_start+T` 裁剪原始 mp4（保留原 FPS；30.0 与 29.97 两种，29.97 样本的 NPZ timestamps 与后端 packet 时间戳最大偏差 8.9e-16）；用官方 keypoints 裁剪段生成 `hrnet-npz`（NPZ 保留所有官方点；confidence 只做 [0,1] 裁剪并记录）。
- SLRT CLI：pinned checkpoint、`--pose-backend hrnet-npz`、window 16 / stride 1 / vote 7；不使用 `--require-gloss`。
- WER：`slbridge.slrt.cli._wer` 的 token Levenshtein（S/D/I，无文本归一化），参考 `csl-daily.test` 的 `gloss` 字段。

实现：`scripts/align_slrt_csl_daily_subset.py`（新增，未改现有源码）。

## 结果

| 指标 | 数值 |
|---|---:|
| attempted / aligned | 52 / 52 |
| low-confidence alignment（过滤后均值 >25 px） | **0** |
| SLRT success / empty / errors | 52 / 0 / 0 |
| **corpus WER（52 条，完整）** | **0.3300（33.0%）** |
| S / D / I | 94 / 37 / 65 |
| 参考 token 数 | 594 |
| 对齐距离（过滤后） | mean 6.88 px、median 6.31、max 19.17 |
| 同窗口全点距离（含退化手点） | mean 19.02 px |
| 逐条 WER 均值 / 中位 | 0.338 / 0.293 |
| WER ≤0.1 / ≤0.2 / ≤0.3 / ≤0.5 | 6 / 20 / 27 / 45 条 |
| 最好 / 最差 | `S001430_P0008_T00` 0.000；`S007350_P0000_T00` 2.143 |

最差样本 `S007350_P0000_T00` 的对齐距离仅 4.94 px，说明其高 WER 不是裁剪错位，而是模型在该句上大量插入/改写（预测明显长于参考，10 个插入）；`S005231_P0006_T00`（WER 0.909）也类似。对齐距离与 WER 的 Pearson 相关约 -0.13，本批 4–19 px 范围内 WER 不是对齐误差驱动。

## 自检

- 随机 2 条（`S002053_P0005_T00`、`S001996_P0004_T00`）：trimmed mp4 可打开、declared==decoded==T（320/243 帧）；首/末帧与源视频 `best_start` 对应帧 mean abs diff 2.1–2.7（mp4v 有损重编码）；NPZ 经 `HRNetNPZFrontend` 构造并抽取第 0/mid/last 帧均返回 `[133,3]`、全有限、confidence∈[0,1]；SLRT JSON 可解析、status=final、`video_sha256` 匹配。
- 批量：52/52 的 NPZ+trim 结构不变量通过（帧数、shape、finite、confidence 范围、indices、timestamps 单调、`video_sha256`、metadata 必填键），0 mismatch；52 个 result JSON 全部可解析；52 条均 exit=0、status=final、frames_received==T、quality_rejected=0、dropped=0。
- 计时：单 GPU 进程，align 878 s、infer 1006 s，整轮 1770 s。自检记录：`artifacts/slrt-52-aligned-20261025/selfcheck.json`。

## 读法

- **33.0% WER 是 52 条近似对齐子集**，不能直接与论文 CSL-Daily 在线 CSLR **29.3%** 或 CSLR test **27.4%**（不同口径）对比；它说明同域在线 Gloss 链路已经可用，且远好于此前 CE-CSL 跨域 **89.18%**。
- 对齐只是近似：官方 `sentence_frames-512x512.zip` 仍缺，52 条子集也非完整 test。要得到可对外引用的论文对照，需要官方帧包或带句子起止的 segment 元数据。
- 65 个插入是 WER 的主要非删除误差；若要提高，需要更准确的句段边界/帧采样，或分析模型过生成，而不是调整评测口径。
- 逐条结果：[comparison-aligned.md](../../artifacts/slrt-52-aligned-20261025/comparison-aligned.md)、[CSV](../../artifacts/slrt-52-aligned-20261025/comparison-aligned.csv)、[summary.json](../../artifacts/slrt-52-aligned-20261025/summary.json)。裁剪视频、NPZ、逐条 SLRT 输出在 `artifacts/slrt-52-aligned-20261025/{trimmed,pose,slrt}/`。
