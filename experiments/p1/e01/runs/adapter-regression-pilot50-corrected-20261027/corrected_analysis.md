# Corrected RGB+pose adapter analysis (pilot)

- 目的：用真实手部 crop（`hand_conf_layout="squeeze"`）替换 legacy 的零 RGB 路径，给出**真实 RGB+pose** 数字。
- legacy run：`experiments\p1\e01\runs\adapter-regression-pilot50-20261027`
- corrected run：`experiments\p1\e01\runs\adapter-regression-pilot50-corrected-20261027`
- 样本数：50；GT 只用于评测侧（EVAL ONLY）。

## 0. 根因（关键）：released checkpoint 的 fusion gate 全零

- checkpoint `models\external\uni_sign\csl_daily_rgb_pose_slt.pth`（sha256 `c2ba5d7ae32f3fa4…`）的 `fusion_gate` 共 4 个张量：**全零**。
  - `fusion_gate.0.bias` shape=[256] nonzero=0 absmax=0.0
  - `fusion_gate.0.weight` shape=[256, 512, 1] nonzero=0 absmax=0.0
  - `fusion_gate.2.bias` shape=[1] nonzero=0 absmax=0.0
  - `fusion_gate.2.weight` shape=[1, 256, 1] nonzero=0 absmax=0.0
- 上游 `models.py` 对 `fusion_gate` 的 Conv1d 权重/偏置做 zero-init；gate_score=0 时模型计算为 `fused_post = 0*fused + 1*pose`，**RGB 分支对输出没有任何影响**，与手部 crop 是否真实无关。
- 逐条文本一致：legacy 与 corrected **完全一致 = True**（50/50 无字符差异）。
- 结论：**corrected 并不是"真实 RGB+pose"数字**；released checkpoint 实际等价于 pose-only（stage2 权重）。26.3026 不是论文 26.36 协议；即使 corrected 手部帧统计正常（见 §3），模型也不会使用 RGB。需要作者提供 fusion gate 非零的权重或重新训练/校准 gate 后才能评测真实 RGB+pose。

## 1. 指标

| 系统 | BLEU-4 | chrF | char edit |
|---|---:|---:|---:|
| A（Uni-Sign video） | 31.2308 | 27.1849 | 0.4831 |
| A2（官方 pose） | 31.7823 | 27.9108 | 0.4885 |
| pose 控制（本脚本） | 31.7795 | 27.9274 | 0.4885 |
| legacy RGB+pose（零 RGB） | 33.1731 | 28.1884 | 0.5020 |
| corrected RGB+pose（真实手部） | 33.1731 | 28.1884 | 0.5020 |

## 2. 配对 signer bootstrap（1000 次）

| 比较 | n | ΔBLEU [95%CI] | Δedit [95%CI] | 改善/持平/恶化 |
|---|---:|---|---|---|
| corrected − legacy | 50 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0/50/0 |
| corrected − A2 | 50 | 1.3908 [-3.7980, 7.1415] | 0.0136 [-0.0406, 0.0791] | 21/12/17 |
| corrected − pose 控制 | 50 | 1.3936 [-3.7945, 7.1409] | 0.0136 [-0.0406, 0.0791] | 21/12/17 |
| corrected − A | 50 | 1.9423 [-4.4029, 8.0787] | 0.0189 [-0.0269, 0.0761] | 20/15/15 |
| legacy − A2 | 50 | 1.3908 [-3.7980, 7.1415] | 0.0136 [-0.0406, 0.0791] | 21/12/17 |
| legacy − pose 控制 | 50 | 1.3936 [-3.7945, 7.1409] | 0.0136 [-0.0406, 0.0791] | 21/12/17 |

## 3. 手部帧覆盖统计（由 run-local bundle 配置 + adapter 预处理函数计算）

| 模式 | left frames mean/min/max (zero) | right frames mean/min/max (zero) | 双手 0 帧样本 |
|---|---|---|---:|
| legacy | 0.0/0/0 (50) | 0.0/0/0 (50) | 50 |
| corrected | 10.32/1/18 (0) | 11.66/5/17 (0) | 0 |

## 4. 边界

- 50 条 pilot 探索性；corrected 才是真实 RGB+pose 数字；legacy 只用于兼容性回归。
- RGB 帧仍来自本地近似 trimmed 视频（官方 sentence-crop 缺失）；帧映射偏移 min/median/max 见 1086 protocol（如本 run 的 protocol/manifest 记录）。
- GT 只在评测侧 join；预测 GT-free；不 commit；GPU 单进程串行、用前记基线。
