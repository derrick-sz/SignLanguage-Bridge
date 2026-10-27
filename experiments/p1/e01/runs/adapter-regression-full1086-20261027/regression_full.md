# RGB+pose adapter 回归结果：full1086

- 状态：**PASS**
- 冻结基线：`experiments\p1\e01\runs\unisign-rgb-pose-1086-20261027\predictions_rgb_pose.jsonl`（sha256 `f335548822511bbc321b14b9a7830f1b59c9bced148e47d3fe0c72cd76b825d5`）
- adapter 预测：`E:\githubproject\SignLanguage-Bridge\experiments\p1\e01\runs\adapter-regression-full1086-20261027\predictions_rgb_pose_adapter.jsonl`（sha256 `345d459ff103c610a58b1c657b4d153d86c929a7f7a79b51c943116e7df52c42`）
- 样本：冻结 1086 / adapter 1086 / 交集 1086；缺失 0，多余 0
- 逐条逐字一致：**1086/1086**（100.00%）

## 指标对照（EVAL ONLY）

| 指标 | 冻结基线 | adapter run | Δ | ≤0.01? |
|---|---:|---:|---:|---|
| bleu4_13a | 26.3026 | 26.3026 | 0.0000 | PASS |
| chrf | 23.0005 | 23.0005 | 0.0000 | PASS |
| mean_char_edit_similarity | 0.4358 | 0.4358 | 0.0000 | PASS |

## 差异清单

无差异：逐条逐字一致。

## GT 纪律

- adapter 预测为 GT-free；GT 仅在本工具中用于离线指标重算（EVAL ONLY）。
