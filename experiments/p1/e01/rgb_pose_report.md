# Uni-Sign RGB+pose 接入与 Pilot-50 评测（E01）

- 状态：**PILOT-50 探索性评测**（不是最终结论；不扩 1086）
- 运行：`unisign-rgb-pose-pilot50-20261027`；GPU 单进程串行；device=cuda；dtype=bfloat16；num_beams=4
- 权重：RGB+pose `csl_daily_rgb_pose_slt.pth`（SHA256 `c2ba5d7ae32f3fa40f7c6807b5004042be81fca192abe7d8ed7182c0d13470f6`）
- 输入协议：pose=官方 released pose archive（与 A2 同源）；RGB=本地近似句段裁剪视频的左右手 crops (112×112)；帧映射用本地 RTMPose 对齐元数据 `best_start`。**不是官方 sentence-crop 视频**（本地未发布），这是本实验的主要偏差。

## 1. 结论摘要

- **RGB+pose vs pose-only（同实现控制）**：ΔBLEU 1.3936 [-3.7945, 7.1409]（不显著（CI 含 0））；论文方向为 +0.75 BLEU（26.36 vs 25.61）。本实验点估计 1.3936，与论文方向一致；50 条样本的 CI 含 0（不显著，样本量不足）。
- RGB+pose BLEU：33.1731；pose-only 控制：31.7795；A2（官方 pose 历史）：31.7823；A（视频历史）：31.2308。

## 2. 主结果（canonical p1-v1，同一 50 条 pilot）

| 系统 | n | BLEU-4 | chrF | char edit | 说明 |
|---|---:|---:|---:|---:|---|
| A（Uni-Sign video） | 50 | 31.2308 | 27.1849 | 0.4831 | pilot-50 历史基线 |
| A2（Uni-Sign 官方 pose） | 50 | 31.7823 | 27.9108 | 0.4885 | pilot-50 历史基线 |
| pose-only 控制（本脚本） | 50 | 31.7795 | 27.9274 | 0.4885 | 同实现 pose 分支 |
| RGB+pose（本脚本） | 50 | 33.1731 | 28.1884 | 0.5020 | EfficientNet-B0 hand crops + PGF |

**配对 signer bootstrap（同一批样本，1000 次）：**

| 比较 | n | ΔBLEU [95%CI] | Δedit [95%CI] | 结论 |
|---|---:|---|---|---|
| RGB+pose − A | 50 | 1.9423 [-4.4029, 8.0787] | 0.0189 [-0.0269, 0.0761] | 不显著（CI 含 0） |
| RGB+pose − A2 | 50 | 1.3908 [-3.7980, 7.1415] | 0.0136 [-0.0406, 0.0791] | 不显著（CI 含 0） |
| RGB+pose − pose-only（控制） | 50 | 1.3936 [-3.7945, 7.1409] | 0.0136 [-0.0406, 0.0791] | 不显著（CI 含 0） |
| pose-only 控制 − A2 | 50 | -0.0028 [-0.0098, 0.0034] | 0.0000 [0.0000, 0.0000] | 不显著（CI 含 0） |

**逐样本 edit 相似度胜负（相对基线）：**

| 比较 | n | 改善 | 持平 | 恶化 |
|---|---:|---:|---:|---:|
| RGB+pose vs pose-only 控制 | 50 | 21 | 12 | 17 |
| RGB+pose vs A2 | 50 | 21 | 12 | 17 |
| pose-only 控制 vs A2 | 50 | 0 | 50 | 0 |

## 3. 资源 / 延迟

- GPU 基线（加载前）：16311 MiB, 2351 MiB, 4 %

| 系统 | 参数 | 加载 ms | 峰值显存 (allocated/reserved MB) | 模型 P50 | P95 | 预处理 P50 | 样本总 P50 |
|---|---:|---:|---|---:|---:|---:|---:|
| RGB+pose | 592.6M | 5246.5 | 1257.9/2694.0 | 360.9 ms | 469.3 ms | 9.8 ms | 370.7 ms |
| pose-only 控制 | 587.7M | 4186.4 | 1249.9/2686.0 | 334.8 ms | 414.3 ms | 10.1 ms | 345.9 ms |
- 历史 pose-only 延迟参考：277.6 ms（E01 旧协议，非本脚本同口径）。

## 4. 失败 / 偏差记录

- RGB 视频缺失样本：0/50；pose pkl 缺失：0/50
- 主要偏差：本地 trimmed 视频是近似句段裁剪（RTMPose↔官方 keypoints 对齐），不是官方 Uni-Sign sentence-crop；官方 `sentence_frames-512x512.zip` 本地仍缺。帧映射偏移见 predictions 中的 `alignment_offset`（部分样本可达 -20 帧），因此 RGB 手部 crop 可能不是视频中精确对应帧 → 若 BLEU 低于论文方向，不能确定是模型/权重问题还是输入偏差。
- 帧映射偏移（start−best_start）：样本数 50，min -127，median -2，max 2；偏移越大，RGB hand crop 与 pose 帧的对应越不可靠。
- 空输出样本数：0。
- 结构校验：RGB+pose 权重 1012 键与 pinned 模型定义完整匹配（missing/unexpected = 0），确认权重可加载；pose-only 控制在 pilot-50 上与 A2 历史值差 -0.0028 BLEU，说明本脚本 pose 分支实现与 A2 一致。

## 5. 能否推到 1086 / 建议

- 本 50 条 pilot 的 RGB+pose 点估计为正、与论文 +0.75 方向一致，但 CI 含 0；**是否扩 1086 取决于对输入偏差的容忍度**：若要扩，建议同一脚本、同一 pose archive、同一 trimmed 视频协议扩到 1086（本机 50 条模型侧约 1–2 分钟；1086 条约 15–30 分钟含控制组），并同时保留 pose-only 控制做配对 Δ。
- 若扩 1086 后 ΔCI 仍含 0，则维持 pose-only/A2 为默认；若 CI 不含 0 且方向保持，再评估将 RGB+pose 接入主链路与 A2 联合。当前不建议因 50 条结果改动生产默认。
- 本实验未改 `slbridge/` 生产路径、未扩 1086、未 commit；权重 CC BY-NC 4.0。
