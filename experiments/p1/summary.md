# P-1 实验执行汇总（自动生成）

- 生成时间：2026-10-26T14:28:46+00:00
- Git：`cf291c2a81558faf6de0dc5a7243e730531e4367`（branch codex/model-contract-auto-captions，dirty=True，133 项）
- 环境：Python 3.10.22 / torch 2.9.1+cu128 / NVIDIA GeForce RTX 5060 Ti

## 实验状态

| 实验 | 状态 | 主产物 |
|---|---|---|
| E00 | PASS (audit-side); G0 smoke evidence from E01 | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e00\model_registry.yaml` |
| E01 | PASS | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e01\reproduction_report.md` |
| E02 | PASS (offline replay); B1/C/D/E/F/G BLOCKED | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e02\report.md` |
| E03 | PASS (SLRT score calibration); Uni-Sign confidence BLOCKED | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e03\report.md` |
| E04 | PASS | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e04\hardware_report.md` |
| E05 | PASS | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e05\streaming_ability_matrix.md` |
| E06 | PASS (Rule-only); LLM/Jev/D-G BLOCKED | `E:\githubproject\SignLanguage-Bridge\experiments\p1\e06\report.md` |

## 运行目录

| run_id | 实验 | 状态 | 预测行数 | GT-free |
|---|---|---|---:|---|
| `20261026_e03_slrt_confidence` | E03 SLRT confidence calibration + error-risk prediction (CPU only, no model inference) | PASS | 0 | yes |
| `20261026_e06_rule_judge_cascade` | E06 Rule-only arbitration + cascade shadow analysis (CPU only; measured imported candidates, simulated routing) | PASS | 0 | yes |
| `e01-pilot50-s1-slrt-gloss-20261026` | E01 S1 SLRT Gloss reproduction on Pilot-50 | PASS | 50 | yes |
| `e01-pilot50-u1-unisign-video-20261026` | E01 video reproduction on Pilot-50 | PASS | 50 | yes |
| `e01-pilot50-u2-unisign-pose-20261026` | E01 pose reproduction on Pilot-50 | PASS | 50 | yes |
| `e02-offline-1086-20261026T134508Z` | E02 | PASS | 3258 | yes |
| `e04-gpu-20261026T2202Z` | Uni-Sign(pose/video) and SLRT(hrnet-npz) cold/warm/continuous-load segmented latency, VRAM, serial vs parallel on one GPU (Pilot-50 signer-stratified subset) | PASS | 0 | yes |
| `e05-prefix-20261026T2226Z` | frozen observed-prefix stress test (12.5/25/50/75/100%) on Uni-Sign pose and video paths; first-stable prefix, rewrite rate, future-frame audit | PASS | 0 | yes |
| `e05-prefix-video-20261026T2232Z` | frozen observed-prefix stress test (12.5/25/50/75/100%) on Uni-Sign pose and video paths; first-stable prefix, rewrite rate, future-frame audit | PASS | 0 | yes |

## GT 隔离扫描（推理/融合载荷）

- `E:\githubproject\SignLanguage-Bridge\experiments\p1\e02\candidate_outputs.jsonl`：3258 行，GT-free=True
- `E:\githubproject\SignLanguage-Bridge\experiments\p1\e06\candidate_inputs.jsonl`：1086 行，GT-free=True

> 完整机器可读结果：`experiments/p1/summary.json`；每个 run 的 manifest/指标在其目录内。
