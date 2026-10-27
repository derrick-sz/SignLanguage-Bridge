# P1-RGBPOSE-ADAPTER：Uni-Sign RGB+pose 生产接入（工程化，非默认启用）

> 立项：2026-10-27（用户指令）· 状态：**IN PROGRESS**
> 目标：把已在 CSL-Daily 1086 上验证的 Uni-Sign RGB+pose 推理接入 SignLanguage-Bridge 后端，作为**候选**模型（不替换生产默认）。
> 唯一回归基线：冻结实验 `experiments/p1/e01/runs/unisign-rgb-pose-1086-20261027/`

## 1. 冻结基线（不可变）

| 项 | 值 |
|---|---|
| checkpoint | `models/external/uni_sign/csl_daily_rgb_pose_slt.pth`，SHA256 `c2ba5d7ae32f3fa40f7c6807b5004042be81fca192abe7d8ed7182c0d13470f6`，1,197,268,371 B |
| 上游源码 | `models/external/uni_sign_source`，commit `eed438bcb49e30405cd6ccdfcccca330c134e830` |
| pose 输入 | 官方 released CSL-Daily pose archive（normalized 133 keypoints，与 A2 同源） |
| RGB 输入 | 手部 crop 112×112（由 pose 133 → 手部 bbox → 视频帧裁剪），backbone EfficientNet-B0（权重在 ckpt 内） |
| 帧映射 | pose 帧索引 → 本地视频帧（本地 RTMPose 对齐 `best_start`） |
| 采样 | 帧：确定性 linspace（上游 random.sample）；手部：低置信加权采样 + 每样本固定种子 |
| 解码 | max_length 256、num_beams 4、max_new_tokens 100、bf16、cuda |
| **指标** | **BLEU-4 26.3026 / chrF 23.0005 / char edit 0.4358**（1086 test，canonical p1-v1） |
| 论文对照 | pose-only 25.61 / RGB+pose 26.36（Δ+0.75）；本地 25.6395 / 26.3026（Δ+0.66） |

## 2. 设计（单一推理真源）

```
experiments/p1/e01/eval_unisign_rgb_pose.py  ──┐
                                              ├─► slbridge/uni_sign_rgb_pose.py（新，唯一推理实现）
slbridge/model_adapters.py: unisign_rgb_pose_slt ┘     ├─ UniSignRGBPoseRuntime（模型加载/前向/decode）
                                                       ├─ hand crop（bbox_4hands / sample_hand_indices / build_support_rgb）
                                                       └─ official pose 加载 + 帧映射
```

- **回归脚本改为 import 该模块**，确保 adapter 与基线脚本共用同一实现；Pilot-50 回归必须逐字一致，任何差异只能来自集成层。
- `UniSignRGBPoseAdapter` 实现 `ModelAdapter` Protocol，注册 id `unisign_rgb_pose_slt`。
- 配置：`enable_rgb`（默认 True）、`fallback_to_pose_only`（默认 True，回退 `uni_sign_pose_slt` 并记录 `fallback_reason`）。
- 版本追踪写入 `RecognitionResult` metadata：`PROTOCOL_ID`、`checkpoint_sha256`、`source_commit`、`preprocess_version`、preprocess 参数哈希。
- 异常：加载失败/显存不足/输入缺失 → 明确错误码，不静默、不返回空文本冒充成功。
- GT-free：结果过 `gt_guard`。

## 3. 验收标准

| # | 验收项 | 判据 |
|---|---|---|
| 1 | Pilot-50 逐条回归（经 adapter 协议路径） | 与冻结 `predictions_rgb_pose.jsonl` **50/50 逐字一致**；差异必须逐条归因 |
| 2 | 1086 全量（经 adapter） | canonical 指标与基线差 **≤0.01**（BLEU 26.3026 / chrF 23.0005 / edit 0.4358） |
| 3 | 单元/集成测试 | 注册、能力、配置校验、回退、GT-free、确定性、手部 crop 几何；opt-in GPU 1 样本对照 |
| 4 | 双环境回归 | `.conda-env` 944+/14、`.venv-libras` 990+/14，零退化 |
| 5 | 性能（task-34） | RTX 5060 Ti：e2e/RTF/P50/P95/显存，与现有视频路径 A 配对比较；延迟/显存增量可接受性结论 |
| 6 | 产物完整 | 每 run 的 manifest / 预测 / 指标 / 性能日志；回归报告；ADR 更新 |

## 4. 状态门（不自动启用）

```
候选（当前） → Round-2 独立人评通过 → 部署泛化验收通过 → 才可讨论默认
```
- **在 Round-2 独立人评与部署泛化验收通过前，RGB+pose 保持候选状态，不自动启用为生产默认模型。**
- 生产默认仍为：视频→A（pose-only 视频端到端）/ 官方 pose→A2；RGB+pose 仅显式配置时启用。

## 5. 已知偏差（必须在报告中保留）

0. **【重大·2026-10-27 发现】该 checkpoint 的 RGB 分支数学上恒等消失，"+0.66 来自 stage2 pose 权重"**：
   - 官方 pose pkl 的 `scores` 形状是 `(T, 1, 133)`（多一个 singleton 维）；冻结脚本 `sample_hand_indices` 用 `confs[:, 91:112]` 索引 → 空 → 手部帧 `None` → RGB 支持为**全零占位**（`hand_frames_left/right = 0`，50/50 样本）
   - squeeze 修正后（`hand_conf_layout="squeeze"`）手部帧真实可用（left mean 10.3 / right mean 11.7），但 **输出与 legacy 逐条完全一致（50/50，Δ=0）**
   - 根因：`fusion_gate` 两个 Conv **zero-init** 且权重全零（`fusion_gate.0.weight` 0/131072、`fusion_gate.2.weight` 0/256、bias 全零），公式
     `fused_post = gate*fused + (1-gate)*gcn`，`gate ≡ 0` ⇒ **输出恒等于 pose 分支**（`models.py` L127-137/L192-194）
   - 因此 `csl_daily_rgb_pose_slt.pth` 实质是"**更好的 pose-only 模型**"：与 `csl_daily_pose_only_slt.pth` 的 627 个 pose/mT5 键 **41.6% 参数超 1e-6 容差**（gcn mean_rel 0.29 / fusion_gcn 0.36 / mt5 0.16）→ +0.66（1086）/ +1.39（pilot）来自 **stage2 的 pose 权重**，与 RGB 无关
   - **撤销"复现论文 26.36"的表述**：26.30≈26.36 是巧合，真实 RGB+pose 需作者提供**非零 fusion gate** 权重（或按正确协议重训/校准 gate）
   - 产品含义：生产形态应为 **零 RGB 支持 + 不做手部裁剪**（`hand_conf_layout="legacy"`），延迟接近 pose-only，同时拿到 stage2 权重收益；待 stage2-pose-subset 验证（用 pose-only 架构加载 627 键）后可彻底去掉 RGB 分支
1. 官方 Uni-Sign `sentence-crop` 视频（`sentence_frames-512x512.zip`）本地缺失 → RGB 用本地近似句段 trimmed 视频
2. 帧映射偏移实测 min −170 / median −1 / max +14（个别样本手部 crop 与 pose 帧对不齐）
3. 手部/帧采样改为固定种子（上游随机），保证可复现
4. 权重许可 CC BY-NC 4.0（禁商用）

## 6. 任务映射

| 任务 | 内容 | 状态 |
|---|---|---|
| task-32 | adapter 实现 + 测试（p0a-backend） | IN PROGRESS |
| task-33 | Pilot-50 逐条 + 1086 指标回归（analyst） | 待 task-32 |
| task-34 | 性能测量 vs 现有视频路径（perf-streamer） | 待 task-32 |
| — | ADR-P1-012 + 最终报告（Lead） | 待回归/性能完成 |
