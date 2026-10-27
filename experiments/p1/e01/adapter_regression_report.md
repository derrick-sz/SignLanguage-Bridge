# RGB+pose adapter 回归报告（task-33）

- 状态：**legacy Pilot-50 PASS（50/50 逐字）；corrected Pilot-50 已跑（真实手部帧可用，但 gate 全零 → 与 legacy 完全一致）；stage2 pose-subset 等价性验证 PASS（50/50 逐字，可去掉 RGB 分支）；1086 待排期。**
- 生成日期：2026-10-27
- 唯一基线：`experiments/p1/e01/runs/unisign-rgb-pose-pilot50-20261027/`（pilot）与 `.../unisign-rgb-pose-1086-20261027/`（1086）；冻结清单：`experiments/p1/e01/frozen_baseline_manifest.json`
- 四条硬结论（先读）：
  1. legacy 50/50 逐字一致只证明 **adapter 忠实复现冻结脚本**，不证明 RGB+pose 效果。
  2. **冻结基线的 RGB 分支实际是零输入**（官方 pose confs 形状 `(T,1,133)` 导致手部采样为空，见 §2）；26.3026 不是论文协议。
  3. **released `csl_daily_rgb_pose_slt.pth` 的 `fusion_gate` 全部为零**（上游代码 zero-init 后未训练），gate=0 → `fused_post = 0*fused + 1*pose`，**即使 corrected 提供真实手部 crop，RGB 分支对输出也完全没有影响**；corrected 与 legacy 逐条一致（§3/§4）。真实 RGB+pose 评测需作者提供非零 gate 权重或重新训练/校准 gate。
  4. **stage2 pose 子集（627 键）+ 纯 pose 架构与 legacy RGB+pose 逐字 50/50 完全一致**（§5）：RGB 分支可以完全去掉，保留 stage2 的 pose/mT5 收益（pilot +1.39 / 1086 +0.66 vs A2/pose-only）并把延迟拉回 pose-only 量级（~278 ms 级，而非 RGB+pose ~370 ms）。

## 1. legacy 回归（adapter 忠实复现冻结基线）

| 项 | 值 |
|---|---|
| 冻结基线预测 | `experiments/p1/e01/runs/unisign-rgb-pose-pilot50-20261027/predictions_rgb_pose.jsonl`；sha256 `d64bc2aa…121ab4`；50 行 |
| adapter 预测 | `experiments/p1/e01/runs/adapter-regression-pilot50-20261027/predictions_rgb_pose_adapter.jsonl`；sha256 `7154a81d…8819c2`；50 行 |
| adapter run 目录 | `experiments/p1/e01/runs/adapter-regression-pilot50-20261027/`（`manifest.json`、`regression_pilot.json/.md`、`regression_bundle/manifest.json`） |
| 加载路径 | `slbridge.model_adapters.load_model_adapter(<run-local bundle>, options=RuntimeOptions(device="cuda"))`（正常 registry 路径） |
| 回归配置 | `pose_source="official_archive"`；`pose_dir=artifacts/cslnews-eval/csl-daily-pose/pose-format`；`alignment_dir=artifacts/csldaily-video-test-627-20261026/pose`；`video_dir=.../trimmed`；`enable_rgb=True`；`fallback_to_pose_only=False`；cuda / bfloat16 / beams=4 / max_new_tokens=100 |
| runtime_info | protocol `unisign-rgb-pose-v1`；checkpoint `c2ba5d7a…3470f6`；source `eed438b…4e830`；model_loaded=true；fallback_pose_only_loaded=false；rgb_failures=[] |
| fallback 断言 | 50/50 行 `quality.fallback_reason == null` |
| 逐条逐字一致 | **50/50（100%）**；缺失 0、多余 0、差异清单 0 |
| canonical 指标（pilot，EVAL ONLY） | 冻结 33.1731 / 28.1884 / 0.5020；adapter 33.1731 / 28.1884 / 0.5020；Δ=0.0000/0.0000/0.0000（≤0.01 PASS） |
| 失败数 | 0（50/50 `recognized`） |
| GPU / 耗时 | 用前基线 `16311 MiB total / 2071 MiB used / 6%`；load 后 4846 MiB、推理后 2240 MiB；模型侧总耗时 20.3 s |
| 判定 | **PASS（仅 legacy 兼容性）** |

## 2. 缺陷一：冻结基线的 RGB 分支是零输入

1. 官方 pose pkl：`scores` = **(T,1,133)**，`keypoints` = (T,1,133,2)。
2. 冻结脚本（adapter legacy 默认路径）的 `sample_hand_indices(confs)` 用 `confs[:,91:112]` / `confs[good][:,key_ids]`，在 (T,1,133) 上是对 axis=1（长度 1）切片 → 空 → 左右手均 `None`。
3. `build_support_rgb` 因此返回 `left/right_sampled_indices=[-1]`、`left/right_hands = zeros(1,3,112,112)`；EfficientNet-B0 吃全零手图 → RGB 分支输出为常量（与图像无关）。
4. 对 Pilot-50 全量核验：`left=[-1]` **50/50**、`right=[-1]` **50/50**、**0/50 有任何真实手部帧**。
5. 上游正确写法是 `confs[:,0,91:112]`；squeeze 后 pilot 前 5 条每手可采到 10–11 帧 → 修正后真实手部 crop 可正常工作。

→ 冻结 26.3026 = “pose + 常量零 RGB 特征”的数字，**不是论文 26.36 协议**；与论文 26.36 的接近是巧合。ADR-P1-011 中“复现论文”的表述需要修正。

## 3. corrected Pilot-50（真实手部 crop，`hand_conf_layout="squeeze"`）

| 项 | 值 |
|---|---|
| run 目录 | `experiments/p1/e01/runs/adapter-regression-pilot50-corrected-20261027/`（`manifest.json`、`predictions_rgb_pose_adapter.jsonl`、`regression_bundle/manifest.json`、`corrected_analysis.json/.md`） |
| 配置 | 同 legacy，但 `hand_conf_layout="squeeze"`；bundle manifest sha 见 run 目录 |
| 手部帧统计（50 条） | **left：mean 10.32 / min 1 / max 18，0 帧样本 0；right：mean 11.66 / min 5 / max 17，0 帧样本 0；双手均 0 帧样本 0** |
| 每条 metadata 断言 | `hand_conf_layout_effective=="squeeze"`、`hand_frames_left/right>0`（50/50 通过） |
| 失败/回退 | 0 失败、0 fallback；模型侧总耗时 68.1 s；GPU 用前基线 2064 MiB used / 6% |
| 预测 sha256 | 见 `predictions_rgb_pose_adapter.jsonl` |

**corrected vs legacy / A2 / pose 控制（Pilot-50，EVAL ONLY）**

| 系统 | BLEU-4 | chrF | char edit |
|---|---:|---:|---:|
| A（Uni-Sign video） | 31.2308 | 27.1849 | 0.4831 |
| A2（官方 pose） | 31.7823 | 27.9108 | 0.4885 |
| pose 控制（本脚本） | 31.7795 | 27.9274 | 0.4885 |
| legacy RGB+pose（零 RGB） | 33.1731 | 28.1884 | 0.5020 |
| **corrected RGB+pose（真实手部）** | **33.1731** | **28.1884** | **0.5020** |

| 比较（配对 signer bootstrap，1000 次） | n | ΔBLEU [95%CI] | Δedit [95%CI] | 改善/持平/恶化 |
|---|---:|---|---|---|
| corrected − legacy | 50 | **0.0000 [0.0000, 0.0000]** | **0.0000 [0.0000, 0.0000]** | **0/50/0** |
| corrected − A2 | 50 | +1.3908 [−3.7980, +7.1415] | +0.0136 [−0.0406, +0.0791] | 21/12/17 |
| corrected − pose 控制 | 50 | +1.3936 [−3.7945, +7.1409] | +0.0136 [−0.0406, +0.0791] | 21/12/17 |
| corrected − A | 50 | +1.9423 [−4.4029, +8.0787] | +0.0189 [−0.0269, +0.0761] | 20/15/15 |

→ **corrected 与 legacy 逐条完全一致（50/50，0 字符差异）**；真实手部 crop 已进入预处理（帧数统计证明），但模型输出不变。原因见 §4。

## 4. 缺陷二（根因）：released checkpoint 的 fusion gate 全零

- 审计 `models/external/uni_sign/csl_daily_rgb_pose_slt.pth`（sha256 `c2ba5d7ae32f3fa4…`）：`fusion_gate` 共 4 个张量，**全部为零**：
  - `fusion_gate.0.weight` (256,512,1) nonzero=0；`fusion_gate.0.bias` (256,) nonzero=0
  - `fusion_gate.2.weight` (1,256,1) nonzero=0；`fusion_gate.2.bias` (1,) nonzero=0
- 上游 `models/external/uni_sign_source/models.py` 对 `fusion_gate` 的 Conv1d 权重/偏置做 **zero-init**；gate 未训练时保持全零。
- 前向公式：`fused_transposed_post = gate_score * fused_transposed + (1 - gate_score) * gcn_feat_selected`；gate_score=0 → `fused_post = gcn_feat_selected`，**RGB 分支（含真实手部 crop）对 pose 特征和最终文本没有任何影响**。
- 同族 checkpoint 交叉核对：`csl_stage2_weight.pth`（1012 键）同样 `fusion_gate` 全零；`csl_stage1_weight.pth`（627 键）无 fusion_gate。因此“rgb_pose 比 pose-only 高 +0.66”来自 stage2 的 pose/mT5 权重，不是 RGB 分支。

**影响**：
- **26.3026 不是论文 RGB+pose 协议的数字**；corrected 的 33.1731（pilot）也不是“真实 RGB+pose”数字——两者都是“stage2 pose 权重 + 死 RGB 分支”的输出。
- adapter 集成、corrected 预处理（真实手部帧）均已验证可用；**缺的是 checkpoint 的 fusion gate 权重**（或重新训练）。
- 若要真实评测 RGB+pose：需作者提供 fusion gate 非零的权重，或按正确协议重新训练/校准 gate（训练时 gate 不能冻结在零）。在此之前不应对外声称复现论文 26.36，也不应把 corrected 当作 RGB+pose 数字。

## 5. stage2 pose-subset 等价性验证（第三段）

领决定：既然 fusion gate 恒零，stage2 RGB+pose checkpoint 等价于“更好的 pose-only 模型”。验证方法：取 `csl_daily_rgb_pose_slt.pth` 中与 `csl_daily_pose_only_slt.pth` 完全一致的 **627 个 pose 键**（键名/形状 100% 匹配，0 缺失/0 形状不匹配），保存为 `models/external/uni_sign/csl_daily_stage2_pose_subset.pth`（`{"model": ...}` 包装；627 键，1,947,137,670 B，sha256 `951601a39fbbe19f…`），再用现有 **`uni_sign_pose_slt` adapter 的 pose-only 架构/runtime**（`TorchUniSignRuntime`，同一 bundle 配置；通过正常 `load_model_adapter` registry 校验）在官方 pose archive 输入上跑 Pilot-50。

| 项 | 值 |
|---|---|
| run 目录 | `experiments/p1/e01/runs/stage2-pose-subset-pilot50-20261027/`（`predictions_stage2_pose_subset.jsonl`、`stage2_pose_subset.json`、`pose_bundle/manifest.json`） |
| 报告 | `experiments/p1/e01/stage2_pose_subset_report.md` |
| 与 legacy RGB+pose 逐字一致 | **50/50（100.0%）**，差异 0 条 |
| 三系统指标（pilot，EVAL ONLY） | stage2-subset-pose **33.1731 / 28.1884 / 0.5020**；legacy RGB+pose **33.1731 / 28.1884 / 0.5020**；A2 31.7823 / 27.9108 / 0.4885；pose 控制 31.7795 / 27.9274 / 0.4885 |
| GPU / 耗时 | 用前基线 `16311 MiB total / 2002 MiB used / 4%`；load 11.9 s（50 条模型侧 ~20 s 量级） |
| 结论 | **可以完全去掉 RGB 分支**：stage2 pose 权重 + pose-only 架构 = legacy RGB+pose 输出，pilot 上保留 +1.39 BLEU（vs A2/pose-only），延迟按 pose-only 路径（~278 ms 级）而非 RGB+pose（~370 ms）。这不是论文 RGB+pose 复现（gate 仍为零），但作为产品路径可去掉无效的 RGB 分支、减少显存/预处理开销。 |

## 6. 1086 全量（待排期）

- **legacy 1086**：仍建议跑（验证 adapter 忠实复现冻结 1086），预期 1086/1086 逐字、指标 Δ≤0.01；命令：
  ```powershell
  .\.conda-env\python.exe -m experiments.p1.e01.run_adapter_regression \
      --bundle models/external/uni_sign_rgb_pose/model_bundle --mode full \
      --run-dir experiments/p1/e01/runs/adapter-regression-1086-20261027
  ```
- **corrected 1086**：在 fusion gate 仍全零的前提下**没有独立信息**（会与 legacy 完全一致）；建议 gate 修复后再跑，否则不消耗 GPU 排期。
- 若后续获得非零 gate 权重：再跑 corrected 1086，并与 legacy/A2/pose 控制做配对与覆盖率对照。

## 7. 产物 / GT / 安全 / 边界

| 产物 | 路径 |
|---|---|
| 比对工具 | `experiments/p1/e01/regression_rgb_pose_adapter.py`（逐字差异分类 + canonical 指标 + PASS/FAIL） |
| adapter run 工具 | `experiments/p1/e01/run_adapter_regression.py`（正常 registry path + 冻结配置校验 + corrected 手部断言） |
| corrected 分析 | `experiments/p1/e01/analyze_corrected_rgb_pose.py` → `runs/adapter-regression-pilot50-corrected-20261027/corrected_analysis.{json,md}` |
| 冻结清单 | `experiments/p1/e01/frozen_baseline_manifest.json` |
| legacy run | `experiments/p1/e01/runs/adapter-regression-pilot50-20261027/` |
| corrected run | `experiments/p1/e01/runs/adapter-regression-pilot50-corrected-20261027/` |
| stage2 subset run | `experiments/p1/e01/runs/stage2-pose-subset-pilot50-20261027/`；报告 `stage2_pose_subset_report.md`；子集 checkpoint `models/external/uni_sign/csl_daily_stage2_pose_subset.pth` |

- GT 只在评测侧 join（EVAL ONLY）；两份 predictions 均 GT-free（0 GT-key leaks）。
- 未改 `slbridge/` 生产代码（adapter 由 p0a 实现）；未改参考答案/评测协议；GPU 单进程串行、用前记基线；不 commit。
- 遗留输入偏差（仍需保留）：RGB 帧来自本地近似 trimmed 视频（官方 sentence-crop 缺失）；帧映射偏移 median −1、min/max 见 1086 protocol（−170/+14）。
