# G0 证据表（E00 资产侧判定）

日期：**2026-10-26**；审计员：auditor（P-1 E00）。
**重要边界：本表不包含 GPU/模型推理 smoke。E00 审计员按团队约定不跑 GPU，smoke 证据必须来自 E01（reproduction_report）或已记录的历史运行报告；本表没有、也不允许伪造 smoke PASS。**
因此 Gate 结论：**G0 资产清点完成（本表）**；E00 计划书要求的“至少一条 Uni-Sign 视频路径 + 一条 SLRT Gloss 路径完成 smoke”尚未由 E01 提供 → **G0 整体 = PENDING（等待 E01 smoke），不是 PASS**。

## 1. 判定表

| 模型 ID | 状态 | 关键证据（路径） | 缺口 / 说明 |
|---|---|---|---|
| `unisign_rgb`（视频→RTMPose 133→69→pinned pose-only SLT） | **READY**（资产+历史运行；GPU smoke 待 E01） | checkpoint `models/external/uni_sign/csl_daily_pose_only_slt.pth`，SHA256 `73bea87a…c1a6c`，1,186,925,112 B（`_hash_raw.txt` L2，与 model_bundle/manifest.json 一致）；mT5 `180573b5…d614`；RTMPose/YOLOX ONNX 哈希见 model_registry；1086 视频运行 BLEU-4 24.8157（pitfalls L117，参考=上游 test_tmp_refs.txt） | 1086/1176 覆盖；官方 sentence_frames zip 缺；smoke（E01）未跑 |
| `unisign_pose_released`（released pose→pinned SLT） | **READY**（资产+历史运行；GPU smoke 待 E01） | 同权重；官方 pose 归档 `artifacts/cslnews-eval/csl-daily-pose/pose-format`（1,176 文件 / 556,506,793 B，zip 4,594,281,295 B）；test_tmp_refs `042cd94c…aab`（1,176 行）；官方 pose 复现 BLEU-4 25.55（pitfalls L12）；1086 released-pose BLEU-4 25.5846（pitfalls L117） | pose 归档只解出 test；smoke（E01）未跑 |
| `unisign_pinned`（生产固定权重） | **READY** | `slbridge/uni_sign_assets.py:20-22` 固定 SHA256 校验；manifest `model_bundle`（`462ea712…`）/`model_bundle_stride5`（`c4789c6f…`）；实测哈希一致 | 仅此一份权重可通过生产校验 |
| `unisign_rgb_pose_checkpoint`（RGB+pose 权重） | **BLOCKED** | 权重存在且 SHA256 `c2ba5d7a…70f6`/1,197,268,371 B 实测=extra-downloads.json 记载（`_hash_raw.txt` L1） | 仓库无接入代码/协议/运行记录；不得用 pose-only 开 RGB 开关冒充 |
| `slrt_online`（EMNLP24 Online CSLR，CSL-Daily） | **READY**（资产+已跑通历史运行；GPU smoke 待 E01） | `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt`，SHA256 `e9fdd3a6…cbd4d`/499,554,423 B（实测=文档）；词表 `0c4984f0…4c76`/28,821 B；1,086 条 test 运行 **1,086/1,086 成功、0 空输出、corpus WER 0.3018**（pitfalls L119；`docs/reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md`）；代码 `slbridge/slrt/` 实测 import OK 且 `python -m slbridge.slrt.cli --help` 可用 | 运行是 RTMPose 全帧滑窗近似对齐官方 keypoints 裁句段，**不是官方 sentence_frames-512x512 协议**；缺 90 条 test 视频；`slbridge/slrt/` 未提交；smoke（E01）未跑 |
| `slrt_twostream_slr`（TwoStream-SLR CSLR） | **BLOCKED** | 主权重 S2G `F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_s2g\ckpts\best.ckpt` SHA256 `54d2cb65…09739`/504,360,673 B（实测=docs 记载；另有 Papers 镜像同哈希）；joint/keypoint/video 变体各 1,748,231,929 B 哈希见 registry；官方源码镜像 `Paper-algorithm/SLRT-main/TwoStreamNetwork` 存在 | 无项目内 adapter、无严格加载/推理记录；官方旧依赖栈（timm/ctcdecode/wandb/旧 torch）缺；在线特征 pkl 缺；**许可 UNKNOWN** |
| `slrt_twostream_slt`（CSL-Daily 句子翻译） | **BLOCKED** | 缺官方 CSL-Daily G2T best.ckpt 与作者 mBART_zh（`docs/reports/slrt-online-reproduction-gap-2026-10-25.md` L28-29、`slrt-assets-bootstrap-2026-10-25.md` §4）；本地仅 base mBART `6e55bd7b…cf84`（2,444,517,405 B）+ gloss 侧 + 全词表 fallback；另有离线 SingleStream `csl-daily_s2t` 权重 `b6bdf7c2…d66c`/1,503,041,809 B（未接入，非官方 G2T） | 缺 G2T 权重、作者 mBART_zh、csl_pred.dev/test、旧依赖栈；未做任何 SLT 推理 |
| `rf_mediapipe_legacy`（RF + MediaPipe225 遗留） | **READY**（项目本地遗留） | `data/gestures/gesture_model.pkl` SHA256 `06205b68…61b4`/332,473 B；Libras RF 试点 `artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009/weights/model.joblib` SHA256 `3450d09c…db35d`/440,668 B；`.venv-libras` 实测 sklearn 1.4.0 | MediaPipe225 不能替代 HRNet 133 点；RF 产物未入库；仓库无 LICENSE |
| 支撑：`slrt_hrnet_frontend` | **READY** | HRNet `f5726563…3c1`/254,976,311 B、Faster-RCNN `d022e227…bf9`/165,671,301 B 实测；legacy 环境 mmpose 0.29.0/mmdet 2.25.3/mmcv 1.7.0 import OK；202 帧提取跑通记录（bootstrap 报告 §2） | 仅 CPU legacy 环境；官方 prediction_slide 旧栈仍不可跑 |
| 支撑：base mBART / gloss 资产 / K400 S3D | **READY（仅资产）** | base mBART `6e55bd7b…cf84`；gloss_embeddings `b293fdad…4ff5`；map_ids/old2new `df5ac826…6e63`；S3D `6b7f0229…b2d2` | 不是作者 mBART_zh；K400 仅训练/融合需要 |

## 2. 数据与运行前提（G0 相关）

- 数据：`manifest_1086` 1,086 行（18 列）SHA256 `22cd0e49…851f`；pilot_50 50 行 `f0f46125…99c4b`；gt_1086（EVAL ONLY）1,086 行 `feb1b42d…7bb7`；本地视频 19,528 / 26,959,704,328 B；官方 test 覆盖 1,086/1,176（缺 90）。详见 `data_snapshot.json`。
- 环境：`.conda-env` = Python 3.10.22 + torch 2.9.1+cu128（CUDA 12.8, cuDNN 91002, RTX 5060 Ti 16,311 MiB, driver 616.92, CUDA UMD 13.4）；legacy = Python 3.8.20 + torch 1.13.1+cpu + mmpose/mmdet/mmcv；`.venv-libras` = Python 3.12.7 无 torch。详见 `environment_manifest.json`。
- Git：HEAD `cf291c2a…` + 132 项脏工作区（SLRT/Uni-Sign 接入代码未提交）。详见 `inventory_report.md`。

## 3. G0 判定（审计员视角）

| Gate 条目 | 判定 | 依据 |
|---|---|---|
| 模型/权重清单可追溯（路径+SHA256+许可） | **PASS（资产侧）** | 本表 + `model_registry.yaml` + `_hash_raw*.txt`；许可缺口已显式标 UNKNOWN（SLRT） |
| 环境可共存且记录（Python/torch/CUDA/ffmpeg/GPU/驱动） | **PASS（记录侧）** | `environment_manifest.json`；三环境分离已验证 |
| 数据快照（manifest/pilot/GT/视频/pose/参考文件） | **PASS（记录侧）** | `data_snapshot.json` |
| 入口审计（真实 CLI/模块/README 差异/git 治理） | **PASS（记录侧）** | `inventory_report.md` |
| 至少 1 条 Uni-Sign 视频路径 smoke | **PENDING（E01）** | 审计员不跑 GPU；历史运行 1086 视频 BLEU 24.8157 属已记录证据，不能替代本轮 E01 受控 smoke |
| 至少 1 条 SLRT Gloss 路径 smoke | **PENDING（E01）** | 历史 1,086 条 WER 0.3018 是已记录证据；本轮 smoke 待 E01 |
| 伪造/夸大风险 | 无 | 本表所有 PASS 均指向实测文件哈希或明确引用；smoke 一律不写 PASS |

**G0 当前结论：资产/环境/数据/入口四类审计已完成并可复核；G0 最终通过需等待 E01 提供受控 smoke 证据（模型加载+1 条真实输入）。在那之前，任何“G0 PASS”声明都不成立。**
