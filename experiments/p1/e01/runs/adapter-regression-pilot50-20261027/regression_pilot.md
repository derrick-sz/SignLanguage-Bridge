# RGB+pose adapter 回归结果：pilot50

- 状态：**PASS**
- 冻结基线：`experiments\p1\e01\runs\unisign-rgb-pose-pilot50-20261027\predictions_rgb_pose.jsonl`（sha256 `d64bc2aac46e39943a25470b2f81ac6423fb09e75b8febdd1612f0b683121ab4`）
- adapter 预测：`E:\githubproject\SignLanguage-Bridge\experiments\p1\e01\runs\adapter-regression-pilot50-20261027\predictions_rgb_pose_adapter.jsonl`（sha256 `7154a81dd9030f766b229d7531aa8f17d802e6fb8caabf65834f2a9fc98819c2`）
- 样本：冻结 50 / adapter 50 / 交集 50；缺失 0，多余 0
- 逐条逐字一致：**50/50**（100.00%）

## 指标对照（EVAL ONLY）

| 指标 | 冻结基线 | adapter run | Δ | ≤0.01? |
|---|---:|---:|---:|---|
| bleu4_13a | 33.1731 | 33.1731 | 0.0000 | PASS |
| chrf | 28.1884 | 28.1884 | 0.0000 | PASS |
| mean_char_edit_similarity | 0.5020 | 0.5020 | 0.0000 | PASS |

## 差异清单

无差异：逐条逐字一致。

## GT 纪律

- adapter 预测为 GT-free；GT 仅在本工具中用于离线指标重算（EVAL ONLY）。
