# RGB+pose adapter 回归报告（task-33，分两段）

- 状态：**第一段 legacy 回归 PASS（Pilot-50 50/50 逐字一致）；第二段 corrected 待 p0a 增加 `hand_conf_layout="squeeze"`；1086 待 Lead 调度。**
- 生成日期：2026-10-27
- 唯一基线：`experiments/p1/e01/runs/unisign-rgb-pose-pilot50-20261027/`（pilot）与 `.../unisign-rgb-pose-1086-20261027/`（1086）；冻结清单见 `experiments/p1/e01/frozen_baseline_manifest.json`
- 关键结论（必须先读）：**冻结基线的 RGB 分支实际是“全零占位输入”（手部帧从未被采样），因此 26.3026 不是论文 RGB+pose 协议的复现；legacy 回归的 50/50 一致只证明 adapter 忠实复现了冻结脚本（含该缺陷）。** corrected 段将给出真实手部 crop 的数字。

## 1. legacy 回归（adapter 忠实复现冻结基线）

| 项 | 值 |
|---|---|
| 冻结基线预测 | `experiments/p1/e01/runs/unisign-rgb-pose-pilot50-20261027/predictions_rgb_pose.jsonl`；sha256 `d64bc2aa…121ab4`；50 行 |
| adapter 预测 | `experiments/p1/e01/runs/adapter-regression-pilot50-20261027/predictions_rgb_pose_adapter.jsonl`；sha256 `7154a81d…8819c2`；50 行 |
| adapter run 目录 | `experiments/p1/e01/runs/adapter-regression-pilot50-20261027/`（`manifest.json`、`regression_pilot.json`、`regression_pilot.md`、`regression_bundle/manifest.json`） |
| 加载路径 | `slbridge.model_adapters.load_model_adapter(<run-local bundle>, options=RuntimeOptions(device="cuda"))`（正常 registry 路径；bundle backend=`unisign_rgb_pose_slt`） |
| 回归配置 | `pose_source="official_archive"`；`pose_dir=artifacts/cslnews-eval/csl-daily-pose/pose-format`；`alignment_dir=artifacts/csldaily-video-test-627-20261026/pose`；`video_dir=.../trimmed`；`enable_rgb=True`；`fallback_to_pose_only=False`；device=cuda，dtype=bfloat16，num_beams=4，max_new_tokens=100 |
| run-local regression bundle manifest sha256 | `0b19b5a4…904590c10` |
| runtime_info | protocol `unisign-rgb-pose-v1`；checkpoint `c2ba5d7a…3470f6`；source commit `eed438b…4e830`；model_loaded=true；fallback_pose_only_loaded=false；rgb_failures=[] |
| fallback 断言 | 50/50 行 `quality.fallback_reason == null`（无任何回退） |
| 逐条逐字一致 | **50/50（100%）**；缺失 0，多余 0，差异清单 0 |
| canonical 指标（pilot，EVAL ONLY） | 冻结 33.1731 / 28.1884 / 0.5020；adapter 33.1731 / 28.1884 / 0.5020；**Δ = 0.0000 / 0.0000 / 0.0000**（≤0.01 PASS） |
| 失败数 | 0（50/50 `recognized`） |
| GPU / 耗时 | 用前基线 `16311 MiB total / 2071 MiB used / 6%`；load 后 4846 MiB、推理后 2240 MiB；50 条模型侧总耗时 **20.3 s**（P50 见 predictions 的 `quality.model_total_ms`） |
| 判定 | **PASS（legacy 兼容性）** |

> 该 PASS 的含义仅限：adapter 与冻结脚本在同一配置/同一实现下逐字一致（含下述零 RGB 缺陷）。**不能据此声称“RGB+pose 复现论文 +0.75”。**

## 2. 关键协议缺陷：冻结基线的 RGB 分支是零输入

证据链（本机只读核验）：

1. 官方 pose pkl 的 `scores` 形状为 **(T, 1, 133)**（多一个 singleton 维），`keypoints` 为 (T, 1, 133, 2)。
2. 冻结脚本（以及 adapter 当前默认 legacy 路径）的 `sample_hand_indices(confs)` 使用 `confs[:, 91:112]` / `confs[good][:, key_ids]`；在 (T,1,133) 上这是对 **axis=1（长度 1）** 做切片 → 结果为空 → `filter_values > 0.3` 全 False → 左右手均返回 `None`。
3. `build_support_rgb` 因此返回 `left_sampled_indices=[-1]`、`right_sampled_indices=[-1]`、`left_hands/right_hands = zeros(1,3,112,112)`。模型照常前向：EfficientNet-B0 吃全零手图 → RGB 分支输出为**与图像无关的常量特征**，再经 fusion/门控进入 mT5。
4. 对 50 条 pilot 全量核验：`left_sampled_indices=[-1]` **50/50**，`right_sampled_indices=[-1]` **50/50**，**0/50 样本有任何真实手部帧**（`left_hands/right_hands` 全零）。
5. 上游参考实现的正确写法是 `confs[:, 0, 91:112]`（先用 singleton 维）；用 squeeze 后的 (T,133) 试采样：pilot 前 5 条每手可采到 **10–11 帧**（不同样本 10–15 帧量级），说明真实手部 crop 路径本身可用。

**影响**：
- 冻结基线 `26.3026` 是 **“pose + 常量零 RGB 特征”** 的数字，不是论文的“pose + 真实手部 crop”协议；与论文 26.36 的接近是**巧合**。
- 论文方向（pose 25.61 → RGB+pose 26.36）**尚未被本地真实复现**；ADR-P1-011 中“复现论文”的表述必须修正为“pose 部分复现（25.64 vs 25.61）；RGB 部分为 legacy 缺陷路径，真实 RGB+pose 待 corrected 评测”。
- 之前的 pose-only 控制 `25.6395` vs 论文 `25.61` 仍然成立（pose 分支无此缺陷）。

## 3. 第二段：corrected（真实手部 crop）——待 p0a 配置

- 需求：p0a 在 `UniSignRGBPoseConfig` 增加 `hand_conf_layout`（`"legacy"` 默认 = 现状零 RGB；`"squeeze"` = 先 squeeze 到 (T,133) 再采样），并让 bundle 可配置；legacy 默认保留以兼容冻结基线，corrected 显式开启。
- 计划（同一 50 条 Pilot-50、同一 GPU 串行、同 official_archive）：
  1. 跑 `hand_conf_layout="squeeze"`，记录每样本 **left/right 非零手部帧数**（mean/min/max、0 帧样本数）；
  2. 与 legacy 结果做配对 signer bootstrap：BLEU/chrF/edit 的 Δ+CI；逐样本胜负（改善/持平/恶化）；
  3. 与 A2（官方 pose 31.7823）和 pose 控制（本脚本同实现 31.7795）做配对比较（EVAL ONLY）；
  4. 报告必须把 legacy（零 RGB）与 corrected（真实 RGB）分开展示，**不得混用**；corrected 数字才是“真实 RGB+pose”的本地数字。
- 当前状态：**BLOCKED（等 p0a 配置）**，不是实现失败；legacy 回归已验证 adapter 集成层无问题（50/50）。

## 4. 1086 全量（待 Lead 调度）

- 在 Pilot-50 legacy PASS、corrected 配置就绪后，按 Lead 排期执行：
  1. **legacy 1086**：必须与 `runs/unisign-rgb-pose-1086-20261027/predictions_rgb_pose.jsonl` 逐字一致（预期 1086/1086）；指标容差 ≤0.01；
  2. **corrected 1086**（如 Lead 要求）：真实手部 crop 的全量数字，与 legacy/A2/pose 控制配对比较；同时给出 1086 上的 legacy vs corrected 覆盖率（edit≥0.5/0.75/0.9）。
- 命令模板（同一 runner）：
  ```powershell
  .\.conda-env\python.exe -m experiments.p1.e01.run_adapter_regression \
      --bundle models/external/uni_sign_rgb_pose/model_bundle --mode full \
      --run-dir experiments/p1/e01/runs/adapter-regression-1086-20261027
  ```
- 注意：1086 的 corrected 跑批需在 GPU 排期窗口内；legacy 1086 可复用本 runner（约 25 分钟）。

## 5. 产物与工具

| 产物 | 路径 |
|---|---|
| 回归比对工具 | `experiments/p1/e01/regression_rgb_pose_adapter.py`（逐字差异分类 + canonical 指标重算 + PASS/FAIL） |
| adapter run 工具 | `experiments/p1/e01/run_adapter_regression.py`（`load_model_adapter` 正常路径 + 冻结配置校验 + GT-free 输出） |
| 冻结基线清单 | `experiments/p1/e01/frozen_baseline_manifest.json`（pilot/1086 predictions/metrics/protocol sha256 与指标） |
| legacy pilot run | `experiments/p1/e01/runs/adapter-regression-pilot50-20261027/` |
| p0a adapter | `slbridge/uni_sign_rgb_pose.py`（未改；已注册 `unisign_rgb_pose_slt`） |
| bundle | `models/external/uni_sign_rgb_pose/model_bundle/manifest.json`（回归 runner 会生成 run-local 绝对路径副本） |

## 6. GT / 安全 / 边界

- adapter 预测输出均过 `gt_guard`（0 GT-key leaks）；GT 仅在本报告/比对工具中离线重算指标（EVAL ONLY）。
- 未改 `slbridge/` 生产代码；未改参考答案/评测协议；GPU 单进程串行；用前记录 nvidia-smi 基线；不 commit。
- 输入偏差（历史遗留，仍需在 corrected 报告保留）：RGB 帧来自本地近似句段 trimmed 视频（官方 sentence-crop 缺失）；帧映射偏移（start−best_start）median −1、min/max 见 1086 protocol（−170/+14）。
- 本报告会在 corrected 段完成后更新为完整版（含 legacy vs corrected 指标/胜负与手部帧统计）。
