# S2（pose-only 子路径）与完整 RGB+Pose 的关系验证（2026-10-27）

> 用户问题（第 3 点）：S2 是 Stage2 权重的 Pose-only 子路径，需确认它与完整 RGB+Pose 的预测一致性，
> 避免把 Stage2 checkpoint 的提升直接归因于 RGB 分支，或误认为两种推理方式完全等价。

## 1. 结论先行

1. **在 1086 test 上，三条路径的预测逐字完全一致（1086/1086，100%）**：
   - 冻结实验（完整 RGB+Pose 模型，`csl_daily_rgb_pose_slt.pth`，官方 pose + 近似手部 crop）
   - 生产 adapter（`unisign_rgb_pose_slt`，`hand_conf_layout="legacy"` = 零 RGB 支持）
   - S2 子路径（`csl_daily_stage2_pose_subset.pth` 的 627 键 + **纯 pose 架构**）
2. 这不是"碰巧一致"，而是**数学必然**：该 checkpoint 的 `fusion_gate` 全零（zero-init 且权重为 0），
   上游公式 `fused_post = gate*fused + (1-gate)*gcn` ⇒ `gate≡0` ⇒ 输出恒等于 pose 分支。
   RGB 分支（手部 crop → EfficientNet → PGF → gate）对最终 logits **零贡献**。
3. 因此：
   - **不能**把 26.3026 的提升归因于 RGB 分支（它没有参与）；
   - 也**不能**说"S2 与完整 RGB+Pose 等价"——正确表述是：**该 checkpoint 的完整 RGB+Pose 推理本身等价于 pose-only**，
     S2 只是把这个等价关系显式化（去掉死代码路径）。
4. **真实 RGB+Pose（gate 非零）从未被验证**，也无法验证——作者未发布带非零 gate 的权重。

## 2. 证据

### 2.1 三向逐条比对（1086）

| 对比 | n 共同 | 逐字一致 | 差异 |
|---|---:|---:|---:|
| 冻结 vs adapter legacy | 1086 | **1086（100%）** | 0 |
| 冻结 vs S2 子路径 | 1086 | **1086（100%）** | 0 |
| 三向同时一致 | 1086 | **1086（100%）** | 0 |

产物：`experiments/p1/e01/rgbpose_vs_pose_subset_consistency.json`

### 2.2 mechanism 证据

| 证据 | 数值 |
|---|---|
| `fusion_gate.0.weight` | 0/131072 nonzero |
| `fusion_gate.2.weight` | 0/256 nonzero |
| `fusion_gate.*.bias` | 全零 |
| 真实手部 crop 注入（squeeze）后输出 | 与零 RGB **逐条相同**（pilot 50/50、1086 全量一致） |
| stage1 vs stage2 的 627 个 pose 键差异 | **41.6% 参数超 1e-6 容差**（gcn 0.29 / fusion_gcn 0.36 / mt5 0.16）→ 提升来自这里 |

### 2.3 视频输入（rtmpose）路径的一次抽查

- `S000020_P0004_T00`（52 帧 1280×1280）经 rtmpose 生产路径 → `她今年50岁。`，与冻结 official_archive 预测逐字一致。
- 但**采样口径不同**（A 路径是 256 帧确定性窗口滑动 + 帧缓存 + stride；RGB adapter 是全帧抽取后 linspace），
  单样本一致不足以宣称"视频输入下全局等价"——这是仍未消除的边界。

## 3. 对归因与验收的含义

| 主张 | 是否成立 |
|---|---|
| "RGB+pose 比 pose-only 好 0.66 BLEU" | ❌ 不成立——RGB 分支零贡献 |
| "stage2 权重比 stage1 好 0.66 BLEU" | ✅ 成立（1086：26.3026 vs 25.5846；edit 显著；9/10 signer 改善） |
| "S2 与完整 RGB+Pose 推理等价" | ✅ 成立（在该 checkpoint 上，1086 逐字一致） |
| "真实 RGB+Pose 能力已验证" | ❌ 不成立——需要非零 gate 权重，作者未发布 |

## 4. 仍未验证的边界

1. **视频输入（rtmpose）下的全局等价性**：只抽了 1 条样本；采样/窗口口径与 A 不同，长视频无保护。
2. **非官方 pose 来源**（摄像头实时抽取）下的行为：官方 pose archive 与 RTMPose 逐帧抽取的差异未系统评估。
3. **gate 非零的真实 RGB 模型**：不可得，无法验证手部外观信息是否真的有用。
