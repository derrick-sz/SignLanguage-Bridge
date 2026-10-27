# Stage2 pose-subset 等价性验证（task-33 第三段）

- 目的：验证 gate 恒零的 RGB+pose checkpoint 与其 627 个 pose 键的纯 pose 架构**逐字等价**，从而可抛弃 RGB 分支、保留 stage2 收益且降低延迟。
- 输入：pilot 50 条，官方 pose archive（与冻结基线同源）；pose-only adapter runtime（`TorchUniSignRuntime`，同一 `uni_sign_pose_slt` 架构）。
- 子集 checkpoint：`models\external\uni_sign\csl_daily_stage2_pose_subset.pth`（627 键，1947137670 B，sha256 `951601a39fbbe19f…`；pose-only 键集 627，rgb_pose 键集 1012，缺失 0、形状不匹配 0）。
- run：`E:\githubproject\SignLanguage-Bridge\experiments\p1\e01\runs\stage2-pose-subset-pilot50-20261027`（GT-free predictions、stage2_pose_subset.json）；GPU 基线 `16311 MiB, 2002 MiB, 4 %`；load 11864.0 ms。

## 1. 逐条一致率

- 与冻结 legacy RGB+pose Pilot-50 预测逐字一致：**50/50（100.0%）**；差异 0 条。
- 差异清单为空。

## 2. 指标（EVAL ONLY，pilot-50）

| 系统 | BLEU-4 | chrF | char edit |
|---|---:|---:|---:|
| stage2 pose subset（纯 pose 架构） | 33.1731 | 28.1884 | 0.5020 |
| legacy RGB+pose（冻结，零 RGB） | 33.1731 | 28.1884 | 0.5020 |
| A2（官方 pose） | 31.7823 | 27.9108 | 0.4885 |
| pose 控制 | 31.7795 | 27.9274 | 0.4885 |

## 3. 结论

- **逐条 100% 一致**：stage2 pose 子集 + 纯 pose 架构与 legacy RGB+pose 完全等价。
- 因此 **RGB 分支可以完全去掉**：保留 stage2 的 pose/mT5 权重（+1.39 pilot / +0.66 1086 相对 A2/pose-only 的提升来自这里），延迟按 pose-only 路径（~278 ms 量级）而非 RGB+pose（~370 ms）。
- 但需注意：这不是论文 RGB+pose 的复现；真实 RGB+pose 仍需作者提供非零 fusion gate 权重或重新训练 gate。
