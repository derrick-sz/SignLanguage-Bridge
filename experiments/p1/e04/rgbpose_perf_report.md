# Task-34：RGB+pose / stage2 权重 生产路径性能测量（RTX 5060 Ti 16GB）

- 日期：2026-10-27 · 执行：Lead 直接跑 `experiments/p1/e04/rgbpose_perf.py`（perf-streamer 会话中断，脚本由其编写）
- run 目录：`experiments/p1/e04/rgbpose-perf-20261027/`（`manifest.json` / `trace.jsonl` / `metrics.json` / `device_samples.jsonl`）
- 设备：NVIDIA GeForce RTX 5060 Ti，16,311 MiB，driver 616.92；用前基线 2,214 MiB / 6%
- 协议：20 条 pilot（signer 分层）、2 轮、**轮内两臂交替**（A→B→A→B）单进程串行；逐样本配对 Δ 为主结论；绝对值为非干净机房口径

## 1. 两臂定义

| 臂 | 模型 | 路径 |
|---|---|---|
| **A** | `csl_daily_pose_only_slt.pth`（stage1 pose-only，**当前生产默认**） | `uni_sign_pose_slt` adapter（现有 runtime） |
| **B** | `csl_daily_stage2_pose_subset.pth`（stage2 的 627 个 pose/mT5 键，**产品候选**） | **同一 `uni_sign_pose_slt` adapter/runtime**（仅换权重） |

两臂共用同一 RTMPose 前端、同一视频解码、同一归一化、同一解码参数 → Δ 只反映权重差异。

## 2. 结果

| 指标 | A（stage1） | B（stage2 子集） | 配对 Δ |
|---|---:|---:|---:|
| e2e P50 | 2,885.6 ms | 2,909.6 ms | **+19.0 ms（+0.66%）** |
| e2e 均值 | 2,953.5 ms | 2,897.4 ms | **−56.1 ms（−1.9%）** |
| e2e P95 | 4,236.2 ms | 4,039.1 ms | −197.1 ms |
| 模型段 P50 | 31.05 ms | 32.10 ms | +1.05 ms |
| 预处理 P50 | 2,612.5 ms | 2,633.0 ms | +20.5 ms |
| RTF P50 | 0.737 | 0.739 | +0.002 |
| 峰值显存（torch allocated） | 2,361.3 MB | 2,363.0 MB | **+1.7 MB** |
| 峰值显存（reserved） | 3,858.0 MB | 3,858.0 MB | 0 |
| 冷启动加载 | 26.36 s | 15.35 s | −11.0 s |
| 失败 | 0/61 | 0/60 | — |

- **整卡峰值**（含其他进程）：6,861 MB，平均利用率 40.1%（373 个采样点）
- 预测一致度（B vs A，同 40 对）：8/40 相同 —— 两臂是不同权重，本来就应该不同；一致度不用于判等价（等价性由 §6.2 的 legacy vs stage2 逐字 100% 证明）

## 3. 结论（回答三个问题）

1. **stage2 权重相对 stage1 的延迟代价 ≈ 0**：e2e P50 +19 ms（0.66%，在噪声量级）、均值 −56 ms、显存 +1.7 MB、RTF +0.002。
2. **手部裁剪（RGB 分支）不值得做**：`fusion_gate` 恒零（见 ADR-P1-011 修正），squeeze 模式的手部裁剪与 RGB 前向对输出零影响；C/D 臂因此**不纳入生产**（脚本中标记为 BLOCKED/可选）。
3. **工程建议**：把生产权重从 stage1 换成 stage2 的 627 键 pose 子集 = **+0.66 BLEU（1086）/ +1.39（pilot）零延迟代价**；但按用户要求，需 Round-2 独立人评 + 部署泛化验收通过后才改默认。

## 4. 局限

- 20 条 pilot × 2 轮，非干净机房 SLA；绝对值受桌面/agent 后台负载影响，配对 Δ 是主口径。
- `model_forward` 段只覆盖模型前向窄口径（不含完整生成流程），绝对值偏低；跨臂比较仍有效。
- 相机实时路径未单独计时（走同一 `predict` 路径，收益应同样成立）。
- rtmpose 生产模式的 RGB+pose adapter 有 4 个缺陷（backend/cache/归一化/EOF frame=None），由 p0a-backend-main 修复中；本测量不依赖它。
