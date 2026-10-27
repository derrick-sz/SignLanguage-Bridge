# P-1 实验执行报告（E00–E06 + G0–G4 + ADR）

> 计划来源：`SignLanguageBridge_P-1_实验计划包/`（副本 `docs/experiments/P-1/`）。
> 执行日期：**2026-10-26**（UTC 13:1x–14:3x / +08:00 21:1x–22:3x）。
> 机器可读汇总：`experiments/p1/summary.json`；每次运行的 manifest/日志在各 `experiments/p1/runs/<run_id>/`。
> 状态口径：**PASS** 仅用于真实执行并保存证据的分支；缺权重/端点/人评的分支一律 **BLOCKED**，未运行不写 PASS。

## 0. 执行摘要

| 实验 | 状态 | 一句话结论 |
|---|---|---|
| E00 资产/环境/数据审计 | **PASS（审计侧）** | 6+ 模型逐项实测 SHA256/大小/许可并判定 READY/BLOCKED；环境、数据快照、真实入口、Git 脏树风险全部记录 |
| E01 Pilot-50 复现 | **PASS** | 三条路径 50/50 实跑成功，历史输出 **100% 逐字复现**（视频 50/50、官方 pose 50/50、SLRT gloss 50/50），无静默失败 |
| E02 同任务翻译基线 | **PASS（离线回放）** | A(视频) BLEU-4 **24.8157**、A2(官方 pose) **25.5846**、B0(Gloss 确定性 join) **12.2183**；A/A2 oracle 上限 26.598；B1/C/D/E/F/G **BLOCKED** |
| E03 置信度/风险校准 | **PASS（SLRT）；Uni-Sign BLOCKED** | SLRT score 排序可用（AUC 0.745–0.753）但**不是正确概率**（raw ECE 0.30）；OOF Platt 后 ECE 0.026；Uni-Sign 无导出分数 |
| E04 GPU 性能/串并行 | **PASS** | 见 §7：冷/热启动、分段耗时、VRAM、串行 vs 并行实测 |
| E05 前缀/在线可行性 | **PASS（探针+源码审计）** | 见 §8：已观测前缀下的改写/首稳行为、未来帧需求审计 |
| E06 Rule-only 仲裁/级联 | **PASS（Rule-only）；LLM/Jev BLOCKED** | 规则级联 A2 调用率 38.1%、+0.17 BLEU、错停优胜率 5.2%；LLM 分支无端点不可测 |
| G0–G4 | **G0/G1 PASS；G2 部分（BLOCKED 分支）；G3/G4** | 详见 §10、§11 |

**最重要的工程结论（给 V3.0 / P0 的输入）**：

1. 权重与预处理在目标机的**可复现性成立**：同 checkpoint、同解码、同输入下，Pilot-50 三路输出与历史快照 100% 一致（E01）。这解除了"模型不可复现"的最大风险。
2. **Uni-Sign 官方 pose 路径仅比视频端到端高约 0.77 BLEU**（1086 全量）/ 0.55（Pilot-50）；视频路径可以作为正式链路，不必等待官方 `sentence_frames-512x512`。
3. 双模型融合的**潜在空间存在但收益很小且难以兑现**：A/A2 逐样本强弱交叉（A2 更好 241、A 更好 203），离线 oracle 上限 +1.78 BLEU；但规则级联只拿到 +0.17 BLEU（E02/E06）。
4. **不要启用无证据的自动融合/自动接受**：SLRT 分数只能排序、Uni-Sign 无分数、LLM/Jev 无端点；任何"自动接受"目前都无法证明误接受率可接受（E03/E06）。
5. 在线/实时：Uni-Sign 是非因果整句模型（对称时间卷积 ±6 帧 + 双向 mT5 encoder），**不支持原生 Partial**；SLRT 的 partial 依赖右上下文 8 帧 + 投票等待 3 窗。不得把离线整句延迟写成实时首稳时延（E05）。

## 1. 执行环境与数据边界

- 机器：Windows 10，32 GB RAM；GPU **RTX 5060 Ti 16 GB**（驱动 616.92，CUDA UMD 13.4，torch 2.9.1+cu128 / CUDA 12.8）。
- Python：`.conda-env`（3.10.22，模型推理）；`.venv-libras`（3.12.7，无 torch）；`F:\DSH-envs\slrt-legacy`（3.8，legacy OpenMMLab，仅 HRNet 提取）。
- 仓库：Git `cf291c2a81558faf6de0dc5a7243e730531e4367`，branch `codex/model-contract-auto-captions`，**dirty tree 132 项**（`slbridge/slrt`、35 个实验脚本、P-1 全部未提交）。所有 run manifest 都记录了该脏树状态——这是当前最大的复现风险（见 §13）。
- 数据：CSL-Daily test 本地 **1,086 / 1,176**（缺 90 条偏长句视频）；全部为 `retrospective_seen_1086`，**不是新盲测**。SLRT 的句段裁剪是"RTMPose 全帧滑窗对官方 keypoints 的近似对齐"，不是官方 `sentence_frames-512x512` 协议。
- 参考口径：canonical = 上游 `test_tmp_refs.txt`（半角标点、逐字空格）；官方标注（全角标点）仅作诊断。E01 实测同一份视频预测在两套参考下 BLEU 差 **3.13**（28.10 vs 31.23），跨运行比较只能固定一套。

## 2. 方法与基础设施（`experiments/p1/`）

- **GT 隔离**：`experiments/p1/gt_guard.py` + `common.write_jsonl` 默认守卫。推理/融合载荷出现 `reference/gt_*/wer/edit_similarity/...` 会抛 `GroundTruthLeakError`。全部预测/候选 JSONL 已扫描：E02 candidate_outputs 3,258 行、E06 candidate_inputs 1,086 行、各 run 的 model_outputs 共 3,408 行，**GT 泄漏 0**。
- **canonical 指标**：`eval/metrics.py`（sacreBLEU 13a + 上游 mT5 逐字加空格后处理；chrF；token 级 WER；char ROUGE-L/编辑相似度；按 signer 聚类 bootstrap）。回归测试断言：对既冻结快照重算必须得到 **24.8157 / 21.7974**（已通过，见 `tests/test_p1_infra.py`：12 passed）。
- **运行记录**：每实验一个 `runs/<run_id>/`，含 `manifest.json`（git/硬件/权重/数据哈希/命令/样本数/失败数/状态）、`model_outputs.jsonl`（预测-only）、`metrics.json`、`eval_records.jsonl`（GT 仅存在于 EvaluationWorker 输出）。
- **measured / imported / simulated**：本轮 E01 是 measured；E02/E03/E06 的模型文本是 imported 历史快照，路由/选择/级联是 simulated 离线分析；报告逐项标注。

## 3. E00 · 资产、环境与数据治理审计

产物：`experiments/p1/e00/`（`model_registry.yaml`、`environment_manifest.json`、`data_snapshot.json`、`inventory_report.md`、`g0_evidence.md`、`provenance.md`）。

| 模型 | 判定 | 关键证据 |
|---|---|---|
| `unisign_rgb` / `unisign_pose_released` / `unisign_pinned` | **READY** | `csl_daily_pose_only_slt.pth` 实测 SHA256 `73bea87a…c1a6c`、1,186,925,112 B；mT5 `180573b5…d614`；官方 pose 归档 1,176 pkl |
| `slrt_online`（CSLR CSL-Daily） | **READY** | `cslr_best.ckpt` 实测 SHA256 `e9fdd3a6…cbd4d`、499,554,423 B；词表 `0c4984f0…`；1,086/1,086 历史运行 |
| `slrt_twostream_slr` | **BLOCKED** | S2G 权重在（`54d2cb65…`），但无项目 adapter/严格加载/旧依赖栈；许可 UNKNOWN |
| `slrt_twostream_slt`（CSL-Daily 句子翻译） | **BLOCKED** | 缺官方 CSL-Daily G2T `best.ckpt`、作者 `mBART_zh`、`csl_pred.dev/test`；仅 Phoenix SLT 权重 |
| `rf_mediapipe_legacy` | **READY（遗留）** | 本地 RF 模型哈希已记录；MediaPipe225 不能替代 HRNet 133 |
| `unisign_rgb_pose_checkpoint` | **BLOCKED** | 权重在（`c2ba5d7a…`），但仓库无 RGB+pose 接入代码，不得用 pose-only 开 RGB 开关冒充 |

环境：三套 Python 分离、CUDA/ffmpeg 可用；`.conda-env` 未安装 `slbridge-slrt` 入口（用 `python -m slbridge.slrt.cli`）。数据快照哈希：`manifest_1086` `22cd0e49…`、`gt_1086` `feb1b42d…`、上游 refs `042cd94c…`。G0 的 smoke 门槛由 E01 提供。

## 4. E01 · Pilot-50 预处理一致性与推理复现（measured）

产物：`experiments/p1/e01/`（`video_probe.csv`、`preprocess_audit.csv`、`compare_*.csv`、`summary.json`、`reproduction_report.md`）+ 3 个 run 目录。

| 路径 | run_id | 成功 | canonical BLEU-4 / chrF / WER | 与历史快照逐条 |
|---|---|---:|---|---|
| U1 Uni-Sign 视频端到端 | `e01-pilot50-u1-unisign-video-20261026` | 50/50 | BLEU 31.2308 / chrF 27.1849 / edit 0.4831 | 原样 **50/50**、NFKC **50/50** |
| U2 Uni-Sign 官方 pose | `e01-pilot50-u2-unisign-pose-20261026` | 50/50 | BLEU 31.7823 / chrF 27.9108 / edit 0.4885 | 原样 **50/50**、NFKC **50/50** |
| S1 SLRT Gloss | `e01-pilot50-s1-slrt-gloss-20261026` | 50/50 | corpus WER **0.3170**（S72/D21/I17，347 参考 token） | gloss_sequence **50/50** |

- 失败 0、空输出 0、OOM 0；单进程串行；历史 `artifacts/` 只读未改（时间戳核验）。
- 输入一致性：trimmed 视频 SHA256 == 对应 NPZ `metadata.video_sha256` **50/50**；帧数（ffprobe/trimmed/NPZ/alignment/标注）**50/50** 一致；对齐 mean 6.62 px、low-confidence 0。
- 路径间：视频 − pose = **−0.5515 BLEU / −0.0054 edit**（方向与 1086 全量的 −0.77/−0.0055 一致）。
- 延迟（measured）：U1 P50 **3.625 s** / P95 5.336 s（RTMPose+模型）；U2 P50 0.351/P95 0.476 s；S1 每样本 CLI mean 14.46 s / P50 14.19 / P95 19.17（**含每子进程约 3.9 s 权重加载**）。VRAM peak reserved：U1 2,708 MB / U2 2,746 MB。
- 边界：S1 只重跑 infer（align 复用已审计的 byte-identical 历史产物，不能写成"重新提取 pose"）；50 条片段均 ≤172 帧，未覆盖 >256 帧随机采样分支；重复性/跨机器 NOT_RUN。
- 复现报告：`experiments/p1/e01/reproduction_report.md`。

## 5. E02 · 同任务翻译基线与互补收益（imported 快照 + 离线回放）

产物：`experiments/p1/e02/`（`candidate_outputs.jsonl`、`translation_metrics.json`、`fusion_ablation.csv`、`comparison_table.md`、`error_analysis.md`、`report.md`）+ run `e02-offline-1086-20261026T134508Z`。

| 分支 | n | BLEU-4 | chrF | 平均字符编辑相似度 | 完全匹配 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| A Uni-Sign 视频 | 1,086 | **24.8157** | 21.7974 | 0.4157 | 15 | IMPORTED_MEASURED |
| A2 Uni-Sign 官方 pose | 1,086 | **25.5846** | 22.4284 | 0.4212 | 18 | IMPORTED_MEASURED |
| SLRT Gloss（CSLR） | 1,086 | — | — | — | — | WER **0.3018**（canonical 回溯拆分 S1,330/D717/I434；历史 R18 拆分 S1,154/D805/I522，总编辑数同为 2,481，差异仅 Levenshtein 并列回溯顺序） |
| B0 Gloss 确定性 join | 1,086 | 12.2183 | 13.8087 | 0.3484 | 0 | DERIVED_BASELINE（不是训练型 Gloss2Text） |
| B1 / C / D / E / F / G | — | — | — | — | — | **BLOCKED**（缺 G2T 权重 / 无 LLM 端点） |
| GT oracle A/A2 | 1,086 | **26.598** | 23.2554 | 0.4430 | — | OFFLINE_ORACLE（不可部署） |

- A2 − A：平均编辑相似度 **+0.0055**（95% CI −0.0013 … +0.0116，**含 0**）；BLEU-4 **+0.7689**（95% CI +0.2299 … +1.4372，不含 0）。
- 逐样本胜负：A2 更好 **241**、A 更好 **203**、持平 **642**；纠正（A 差→A2 明显更好）93，误改（A 好→A2 明显更差）50。
- 误差重叠：两者都差 661、仅 A 差 43、仅 A2 差 29、两者都好 353 → 存在互补空间，但需要**无 GT 的可观测信号**才能利用。
- Legacy vs canonical：视频 legacy BLEU 24.8156828061 → canonical 24.8157；SLRT legacy WER 0.3018248175 → 0.3018（口径一致）。
- **人评未做**（`NOT_RUN`）：严重语义错误率/关键事实正确率必须由人工评审协议给出，本文不冒充。

### E02-G2 · 真实 LLM 融合消融（探索性 50 条，2026-10-26 追加）

端点：wb-proxy `http://127.0.0.1:8788/v1/chat/completions`，`deepseek-v4.1-flash`，temperature=0，串行；样本 = E01 Pilot-50（10 signer × 5）；分组按用户批复字母（A 直通 / C=Uni-Sign→LLM / D=+SLRT Gloss / E=+证据约束）。

| 组 | v1 prompt BLEU-4 | v2 prompt BLEU-4 | v2 chrF | v2 edit | v2 n |
|---|---:|---:|---:|---:|---:|
| A Uni-Sign 原句（基准） | 31.2308 | 31.2308 | 27.1849 | 0.4831 | 50 |
| C 润色对照 | 26.6838 | **28.9518** | 24.8547 | 0.4858 | 50 |
| D + SLRT Gloss | 26.9450 | **28.5872** | 24.2824 | 0.4831 | 50 |
| E + 证据约束 | 28.6302 | **29.1880** | 24.8223 | 0.4930 | 45（5 次长推理截断） |

- v1 配对：C−A **−4.55**（CI −8.02…−1.38，不含 0）；D−C −0.04（n.s.）；E−D +1.62（n.s.）；**E−C edit +0.040（CI +0.0019…+0.0821，显著）**。
- v2 配对：C−A **−2.28**（CI −4.04…−0.70，仍显著为负）；D−C −0.365（n.s.）；E−D −0.416（n.s.）；edit 差均 n.s.。
- 成本（v2）：**113k tokens**（C 25.7k / D 38.9k / E 48.7k）；P50 C 2.04 s、D 3.01 s、E 4.48 s；P95 5.0–9.5 s。
- 事故（v1 首轮）：`max_tokens=256` 导致 **73/150 空 completion**（推理 token 吃满预算），该轮不进入对比表；v2 用 2048，剩 3/150（v1 复跑 3 次失败）+ v2 5/150（E）。
- **结论：两组 prompt 下 LLM 融合都没有超过 A；D−C（Gloss 新增价值）与 E−D（约束价值）均不显著。按 Gate G2 的"无质量提升则保留单模型路线"规则处理。** 边界：50 条探索批、retrospective_seen、单端点单模型、prompt 未系统调优；最终结论需新独立集 + 人评（Round-1 评审包已就绪，待用户回填）。

## 6. E03 · 原始分数、校准与错误风险

产物：`experiments/p1/e03/`（`score_definition.md`、`per_sample_scores.csv`、`calibration_results.json`、`risk_coverage.csv`、`report.md`）+ run `20261026_e03_slrt_confidence`。

- SLRT token score 语义：`uncalibrated_mean_ensemble_probability`（对象=gloss token；7 窗投票平均）。**不是正确概率**：直接当 P(correct) 时 ECE **0.3041** / Brier 0.2323（AUC 0.7528）。
- OOF Platt（10 签者 leave-one-signer-out，阈值内层选择）：bad（WER≥0.5）AUC **0.7449** / ECE **0.0256** / Brier 0.1353。
- FDR≤15% 策略：coverage **0.8481**、error-ACCEPT 率 **0.1531**、missed-upgrade 率 **0.6744**（should_upgrade 代理 43 条，漏 29）。
- 标签分布：bad 216（19.9%）、error_any 928（85.5%）、bad_03 462（42.5%）。
- **Uni-Sign 置信度 = BLOCKED**：runtime `TextPrediction(score=None)`，历史 JSONL 无 score/logprob；不得用编辑相似度/文本稳定性冒充。
- 结论：SLRT 分数只能当**风险排序信号**；不能当概率、不能跨模型共用阈值；高置信区间仍有不可忽略的 bad 风险。

## 7. E04 · GPU 性能、显存与双模型调度成本（measured）

产物：`experiments/p1/e04/`（`benchmark_p1.py`、`perf_trace.jsonl`、`hardware_report.md`、`parallel_vs_serial.csv`、`metrics.json`）+ run `e04-gpu-20261026T2202Z`。
环境：RTX 5060 Ti 16 GB、驱动 616.92/CUDA 12.8、测量前整卡 ~2.87 GB（桌面基线，无其它 CUDA compute app）；单进程串行；失败尝试保留在 `failed_attempt_{1,2}/` 且不计入任何数字。

| 模型 | 冷启动加载 P50 / P95 | 热启动（30 次）e2e P50 / P95 | RTF | 吞吐 | 显存（torch peak alloc / reserved；整卡峰值） |
|---|---|---:|---:|---:|---|
| Uni-Sign **pose**（模型-only） | 17.66 / 18.74 s | **277.6 / 344.3 ms** | **0.078** | 3.27 条/s | 1,206 / 2,738 MB；5.64 GB |
| Uni-Sign **video**（端到端） | 26.62 / 65.06 s（2 次 CUDA OOM 重试成功） | **2,690.4 / 3,283.9 ms**（pose 2,156 ms = 80%；decode 290、forward 36、文本 200） | **0.748** | 0.339 条/s | 1,206 / 2,738 MB；5.66 GB |
| SLRT（hrnet-npz） | 3.14 / 3.15 s | **9,865.1 / 11,189.7 ms**（≈90 ms/窗 ×109 窗；S3D forward 3,701 ms，其余 ~5.6 s 是逐窗 CPU 热力图预处理） | **2.534** | 0.099 条/s | 374 / 600 MB；3.73 GB |

- 连续负载 2 分钟：pose 419 次（3.49/s）、video 42 次（0.348/s）、SLRT 13 次（0.105/s），**0 失败 / 0 OOM、无显存增长**。
- 串行 vs 两进程并行（同 5 条样本）：串行 batch 和 **40.28 s**；并行 compute wall **34.10 s**（serial_sum/parallel_wall = **1.181**，总墙钟 −15%），但单模型延迟变差：video 2.36→4.27 s/条（1.81×）、SLRT 5.69→6.69 s/条（1.18×）；整卡峰值 **6.45 GB**、GPU util 均值仅 27.8% → **并行不提高 GPU 效率，只适合离线批处理**。
- 结论：**H1 单卡串行可行**（重试后 0 OOM）；**H3 Uni-Sign 延迟 80% 在姿态前端**（pose 2,152 ms vs forward 36 ms）；**H2 修正**：并行缩短总墙钟 15% 但抬高单条 P95 与显存，实时/交互语义默认串行。
- 与 E01 交叉核对：E01 的 50 条视频端到端 P50 3.625 s / P95 5.336 s，与本批 5 条（更短片段）P50 2.690 s 量级一致。
- 如实记录的失败：首轮编排 argparse bug；第二轮 cold video 在宿主 commit 压力下 2 次 `CUDA error: out of memory`（GPU 空闲 >13 GB，committed 55.7/65.2 GB），按坑点 §7 重试后 3/3 成功——**不计模型失败**。

## 8. E05 · 视频前缀压力测试与真正在线能力核验

产物：`experiments/p1/e05/`（`prefix_probe.py`、`prefix_probe.jsonl`、`prefix_probe_summary.md`、`prefix_summary.json`、`future_frame_audit.md`、`streaming_ability_matrix.md`）；run：`e05-prefix-20261026T2226Z`（pose）、`e05-prefix-video-20261026T2232Z`（video）。

**已观测前缀探针（不是实时流；无未来帧填充）**

- Uni-Sign **pose** 路径（10 条 pilot，1 条/signer；12.5/25/37.5/50/62.5/75/87.5/100% 共 80 次推理，0 失败）：`first_stable_prefix` **mean 90% / median 100%**，分布 62.5%:1、75%:2、87.5%:1、100%:**6**；相邻前缀改写率 **67%**；空输出 0；与作者快照 10/10 逐字一致；100% 输入重复两次文本相同（决定论）。
- Uni-Sign **video** 路径（10 条 × 25/50/75/100%，40/40 成功、0 空输出；最短可用前缀 **20 帧**）：各前缀 P50 e2e 820–937 ms、pose 558–599 ms、forward 52–54 ms；与 pose 路径同前缀对比 40 对逐字相同 16、平均字符相似度 0.69（前端不同，仅参考）。
- 含义：**6/10 句要到 100% 前缀才出现最终文本**，平均 90% 前缀才稳定 → Uni-Sign 输出是整句重估而非增量提交，**不能当流式模型**。

**源码审计（future_frame_audit.md，带行号）**

- Uni-Sign：对称时间卷积（kernel=5、3 层，接收野 ≈ ±6 帧有界）+ **双向 mT5 encoder + 单次 generate** → 追加未来帧会改写整句 embedding；`needs_future_frames=YES`、`supported_partial=NO`、`supported_final=NO`。上游 `demo/online_inference.py` 实为"读视频路径整段推理"，不是实时采集。
- 本地 `continuous.py` 滑窗：每个窗口完整重算、无跨窗状态，`confirmed` 只表示连续窗口文本相同，仍可被改写 → **伪 partial**。
- SLRT Online：窗口左 7 / 右 8 + 投票等待 3 窗 → 中心帧 partial 需帧到 `c+11`（**11 帧已观测 lookahead**，30 fps ≈ 0.37 s，不含推理/排队/采集）；`partial` 可改写/撤回，`final` 为 run 结束/EOF（非人工确认）。
- 结论：**P2 实时字幕需要独立在线分支或显式披露窗口等待与首稳时延**；不得把离线整句延迟写成实时指标。

## 9. E06 · Rule/Jev/LLM 仲裁与调度可行性（Rule-only 实测；LLM BLOCKED）

产物：`experiments/p1/e06/`（`candidate_inputs.jsonl`、`router_ablation.jsonl`、`judge_error_analysis.md`、`policy_candidate.yaml`、`report.md`）+ run `20261026_e06_rule_judge_cascade`。

| 策略 | BLEU-4 | chrF | A2 调用率 | error-ACCEPT | missed-upgrade | 纠错率 | 误改率 | missed-winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 always A | 24.8157 | 21.7974 | 0.0 | 0.6483 | 1.0 | 0 | 0 | 0.046 |
| P0 always A2 | 25.5846 | 22.4284 | 1.0 | 0.6354 | 0.0 | 0.0611 | 0.0759 | — |
| P2 rule cascade A | **24.9861** | 21.9575 | **0.3812** | 0.5952 | 0.5568 | 0.0071 | 0.0052 | **0.0521** |
| P2c cascade + score | 25.1061 | 22.0563 | 0.4153 | 0.5874 | 0.5227 | 0.0085 | 0.0052 | 0.0504 |
| P3 full-run + rule select | 24.9559 | 21.9326 | 1.0 | 0.6455 | 0.0 | 0.0071 | 0.0052 | — |
| GT oracle A/A2 | 26.598 | 23.2554 | — | — | — | 0.0611 | 0 | — |

- 规则质量：numbers 触发 122（误报 48%、检出 51%）、negation 125（32%/59%）、person 164（24%/32%）、question 79（46%/43%）。
- 结论：级联"低风险、低收益"——错停优胜仅 5.2%，但相对 A 仅 +0.17 BLEU、纠错率 0.7%；A/A2 高度相关（507/1086 文本相同）是主因。FUSE 状态机合成算例可执行，本批真实触发 0。
- **LLM/Jev/D/E/F/G 全部 BLOCKED**：本机无任何 LLM 端点/密钥（环境变量名核对、repo grep 无 provider 客户端）；Jev 无实现。解锁需要固定端点+模型版本+prompt 版本+预算。

## 10. Gate 判定（G0–G4）

| Gate | 结论 | 依据 |
|---|---|---|
| **G0 资产可追溯** | **PASS** | E00 model_registry（实测 SHA256/许可）+ E01 受控 smoke（Uni-Sign 视频 50/50、SLRT 50/50）；两条既有路径均真实可运行；缺权重的 SLT/TwoStream-SLR 分支显式 BLOCKED |
| **G1 输入与推理复现** | **PASS** | E01：三路 50/50 成功、历史文本 100% 逐字复现、零差异样本、输入 SHA/帧数/裁剪逐条审计、无 silent preprocess mismatch |
| **G2 同任务价值** | **FAIL（融合）/ PARTIAL（基线）** | 可比的 A（24.8157）与 B0（12.2183）已取得，A2（25.5846）为同任务第二路径；**真实 LLM 融合 v1/v2 两组 prompt 均未超过 A**（v1 C−A −4.55 显著；v2 C−A −2.28 显著；D−C、E−D 均不显著；成本 113k tokens、P50 2–4.5 s）→ 按 Gate 规则**保留单模型路线，融合不进主链路**；C/B1（SLRT 句子翻译）仍 BLOCKED，人评 NOT_RUN |
| **G3 性能及流式能力** | **PASS（以实测为准）** | E04：三模型冷/热/负载/串并行全部实测、0 失败（重试后）、峰值显存与 P50/P95 有 trace；E05：前缀探针 80/80 + 40/40、future-frame 源码审计、流式能力矩阵；如实分类"Uni-Sign 无原生 Partial，SLRT partial 需 11 帧 lookahead" |
| **G4 决策归档** | **PASS（本轮）** | 本报告 §11 ADR + `experiments/p1/summary.json` 证据索引；每个决策可回溯到 run_id/产物 |

## 11. ADR 决策记录（P-1 → P0/P1）

### ADR-P1-001 输入预处理：维持每模型独立输入规范
- 状态：ACCEPTED；对应 E00/E01；G0/G1。
- 背景：Uni-Sign 需要 RTMPose 133 取 69 点（xy+confidence、isotropic/官方 pose），SLRT 需要 HRNet 133 的 63 点热力图 + RGB。
- 证据：E00 资产卡；E01 三路复现 100%。
- 结论：**不强制统一 MediaPipe 225**；每模型保留原生输入规范与协议版本字段（方向/归一化/帧采样已进入 manifest）。
- 回退：若未来统一前端，需先做配对精度/成本对照。

### ADR-P1-002 翻译路线：默认单模型 Uni-Sign；SLRT Gloss 不进入句子链路
- 状态：ACCEPTED；对应 E02；G2。
- 背景：B0 确定性 Gloss→Text 只有 12.22 BLEU（远低于 A 24.82），真正 Gloss2Text/SLT 权重缺失（BLOCKED）。
- 结论：**生产默认 A（视频端到端）或 A2（官方 pose，离线/批处理场景）**；不做 Gloss→LLM 伪翻译；双模型 late fusion 暂不启用。
- 解锁条件：取得 CSL-Daily G2T/SLT 权重或固定 LLM 端点后重做 B/C/D–G。

### ADR-P1-003 推理方式：单卡串行/低成本级联优先，不默认双模型并行
- 状态：ACCEPTED；对应 E04/E06；G3/G4。
- 背景：E06 规则级联把 A2 调用率降到 38.1%，质量增益仅 +0.17 BLEU；E04 实测同 5 条样本两进程并行总墙钟 40.28→34.10 s（−15%），但 Uni-Sign video 单条 2.36→4.27 s、SLRT 5.69→6.69 s，整卡峰值显存 6.45 GB、GPU util 均值 27.8%。
- 结论：**默认串行单模型**；级联作为可选路由（需无 GT 信号与 shadow 审计）；**并行只用于离线批处理**（需 ≥6.5 GB 显存预算），不用于实时/交互路径。
- 回退：若未来多卡或更强 GPU，可重测并行使单条延迟不劣化的条件。

### ADR-P1-004 在线/流式：Uni-Sign 只做整句识别；不得宣称无等待实时翻译
- 状态：ACCEPTED；对应 E05；G3。
- 证据（源码审计）：Uni-Sign 对称时间卷积 kernel=5×3 层（±6 帧）+ 双向 mT5 encoder + 单次 generate → `needs_future_frames=YES`、`supported_partial=NO`；上游 `online_inference.py` 实为"读视频路径整段推理"；SLRT 原生 partial 依赖右上下文 8 帧 + 投票等待 3 窗（约 11 帧 lookahead）。
- 证据（E05 实测前缀）：Uni-Sign pose 10 条 × 8 前缀 80/80 成功；`first_stable_prefix` mean **90% / median 100%**（6/10 句只有 100% 前缀才出现最终文本）；相邻前缀改写率 **67%**；video 路径 40/40 成功、最短可用前缀 20 帧。
- 结论：**Uni-Sign 定位为"动作结束后识别"**；P2 实时字幕需要独立在线模型（如 SLRT Online 类）或显式披露窗口等待与首稳时延。

### ADR-P1-005 Jev/LLM：不进入主链路（BLOCKED，保留接口）
- 状态：BLOCKED → 不进 P0；对应 E06。
- 结论：无端点/密钥/版本，无法固定 prompt 与预算；保留 `FusionController` 接口与证据合同，不做自动融合。

### ADR-P1-006 可靠性：SLRT 分数只做排序信号，不启用自动接受
- 状态：ACCEPTED（保守）；对应 E03；G2/G4。
- 证据：raw score 非概率（ECE 0.30）；OOF Platt 后 AUC 0.745、FDR≤15% 策略 error-ACCEPT 0.153、missed-upgrade 0.674；Uni-Sign 无分数。
- 结论：**不启用"高置信自动接受"**；阈值选择必须按 signer 分组 OOF，且需要新数据复验。

### ADR-P1-007 Libras：不使用 CSL 权重推断
- 状态：ACCEPTED；对应 PLAN §2；G4。
- 结论：Libras 需要独立权重、数据许可与 signer 泛化实验；当前 `SignSystemRouter` 的 Libras 能力标 `UNTRAINED`。

### ADR-P1-008 双轨推进：补齐 G2 实验 + P0-A 后端基础并行（2026-10-26 用户批复）
- 状态：ACCEPTED；对应 G2；影响 P0/P1 排期。
- 背景：P-1 已完成复现/性能/流式初验；真正阻塞 V3.0 核心方案的是 G2（双模型融合有效性未验证）。
- 结论：**不等全部实验完成才启动后端**。路线 A（补齐 G2：SLRT G2T → LLM 融合 → 人工评审 → G2 验收）与路线 B（P0-A 基础：Uni-Sign Adapter/API/预测记录/视频管理/错误处理）并行；G2 结果决定融合是否进主链路（有稳定收益才接入，否则保持可插拔）。
- 暂缓项：Jev 集成、学习式动态调度、Early Exit、CSL/Libras 自动识别（理由见 `NEXT_PHASE_PLAN.md` §6）。
- 详见：`docs/experiments/P-1/NEXT_PHASE_PLAN.md`。

### ADR-P1-009 G2 实验程序与验收标准（2026-10-26 用户批复）
- 状态：ACCEPTED；对应 RQ3/RQ7、E02/E06；G2。
- 实验组（用户字母；括号内为计划包原字母）：A 基准 / B=SLRT→G2T（计划 C）/ C=Uni-Sign→LLM（计划 E）/ D=Uni-Sign+SLRT Gloss→LLM（计划 F）/ E=双模型+证据约束→LLM（计划 G）。
- 核心判据：**D vs C**（Gloss 新增价值）、**E vs D**（约束是否降低严重误译）；B 权重 BLOCKED 时以 B0 退化基线代替，且 12.22 不得解读为"SLRT 不值得用"。
- 验收三问：①质量（canonical BLEU/chrF + signer 聚类 bootstrap，禁个案）②关键事实错误（六槽位盲评、纠对/改错、双评审）③成本（调用数/tokens/P50/P95 vs 单模型基线）。
- LLM 前置条件：固定端点+模型版本+prompt 版本（sha256 记录）+预算；GT 严禁入 LLM 输入；探索用 1086，结论需新独立集+人评。
- 详见：`NEXT_PHASE_PLAN.md` §2。

### ADR-P1-010 G2 结论：融合不进主链路（2026-10-27）
- 状态：ACCEPTED；对应 RQ3/RQ7、E02-G2、Gate G2。
- 证据：两组 prompt（v1/v2）的 50 条探索批均显示 LLM 融合不优于 A——v1 C−A −4.55（CI −8.02…−1.38）、v2 C−A −2.28（CI −4.04…−0.70，仍显著为负）；D−C（Gloss 价值）与 E−D（约束价值）在两版均不显著；选择性级联（默认 A、规则触发 28% 才调 E_v2）BLEU **30.7487**（Δ −0.482，CI −2.532…+1.061，n.s.），成本 22,981 tokens；全调 E(v2) 27.9650（Δ −3.266，CI −6.115…−0.461）、61,393 tokens；v2 批合计 125,976 tokens、per-call P50 3.16 s / P95 10.82 s。
- 结论：**G2 = FAIL（无质量增益证据）**；生产默认单模型（A/A2），融合只保留可插拔接口与离线管线；不扩到 1086；B（SLRT G2T/SLT）仍 BLOCKED，B0 不得用于否定 SLRT。
- 重启条件与边界：见 `G2_DECISION_MEMO.md` §5–§6（需新独立集、开发集 prompt 调优、人评"纠对>改错"、成本上限、CI 不含 0）。
- 待补：Round-1 人工评审回填（"关键事实错误"维度目前**未验证**，不是通过）。

## 12. 交付物索引

| 内容 | 路径 |
|---|---|
| 计划文档副本 | `docs/experiments/P-1/` |
| 执行基础设施/说明 | `experiments/p1/README.md`、`experiments/p1/common.py`、`gt_guard.py`、`eval/` |
| 数据台账（GT-free） | `experiments/p1/datasets/manifest_1086.csv`、`pilot_50.csv`、`source_meta.json` |
| GT（EVAL ONLY） | `experiments/p1/references/gt_1086.csv`、`source_meta.json` |
| E00 | `experiments/p1/e00/` |
| E01 | `experiments/p1/e01/` + 3 个 run 目录 |
| E02 | `experiments/p1/e02/` + run `e02-offline-1086-20261026T134508Z` |
| E03 | `experiments/p1/e03/` + run `20261026_e03_slrt_confidence` |
| E04 | `experiments/p1/e04/` + run `e04-gpu-…` |
| E05 | `experiments/p1/e05/` |
| E06 | `experiments/p1/e06/` + run `20261026_e06_rule_judge_cascade` |
| 汇总 | `experiments/p1/summary.json` / `summary.md` |
| 项目报告 | `docs/reports/p1-experiments-2026-10-26.md` |

## 13. 风险、边界与下一步

**边界（不可越过的声明）**
- 1,086 是 `retrospective_seen`，不是新盲测；无独立 signer 泛化结论。人评 NOT_RUN。
- E01 只覆盖 50 条诊断样本；S1 未重跑 align；未覆盖 >256 帧采样分支与跨机器复现。
- E02/E03/E06 使用 imported 历史预测；路由/级联/校准均为离线/模拟，不是线上实测时延或收益。
- E04/E05 的单卡测量结果不能外推到多卡/异构 GPU；实时结论受源码事实约束。

**风险**
1. Git 脏树（132 项）是最大复现风险：`slbridge/slrt`、`scripts/*` 等实验入口未提交，任何清理都可能丢失。建议下一步提交独立 commit/分支并打 tag。
2. 官方 `sentence_frames-512x512.zip` 缺失、test 视频缺 90 条、SLRT 近似对齐协议未复现官方输入。
3. SLRT 上游许可 UNKNOWN；Uni-Sign 权重非商用（CC BY-NC 4.0）。

**下一步最小行动（P0/P1 backlog）**
1. P0：把 P-1 基础设施与 E01/E02 评测器合并为一个只读子包并提交（先保 git 状态）。
2. P0：用 E01 的 canonical 评测器替换零散评测脚本的 BLEU/WER 口径，防止全角/半角参考再次混用。
3. P1：Uni-Sign 增加 `output_scores`（生成 logprob/beam）→ 解锁 E03 Uni-Sign 置信度与 E06 路由。
4. P1：补 90 条 test 视频与官方 sentence frames；取得 SLRT G2T/SLT 权重后重做 C/B。
5. P1：固定 LLM 端点/版本/prompt 后做 D–G 融合消融；在此之前不启用自动融合。
6. P2：在线模型单独验证（窗口/右上下文/首稳），不改造 Uni-Sign 为假流式。

### ADR-P1-011 Uni-Sign RGB+pose：视频可用时的首选单模型（2026-10-27）
- 状态：**ACCEPTED（分层默认：视频→RGB+pose；仅 pose→A2）**；对应 E01、G2 改进、RQ2/RQ3。
- 背景：7 种文本级融合全部负结果后，转向模型内部融合——Uni-Sign 自带的 RGB+pose 权重（1012 键，含 `rgb_support_backbone`/`fusion_gate`）此前未接入。
- 证据（1086 test，canonical p1-v1，`experiments/p1/e01/rgb_pose_report_1086.md`）：
  | 系统 | BLEU-4 | chrF | char edit |
  |---|---:|---:|---:|
  | A（视频） | 24.8157 | 21.7974 | 0.4157 |
  | A2（官方 pose） | 25.5846 | 22.4284 | 0.4212 |
  | pose 控制（同实现） | 25.6395 | 22.4766 | 0.4223 |
  | **RGB+pose** | **26.3026** | **23.0005** | **0.4358** |
  - **论文复现**：pose 25.64 vs 25.61、RGB+pose 26.30 vs 26.36、Δ +0.66 vs +0.75 → 输入协议与实现可信
  - 配对 signer bootstrap：ΔBLEU +0.66 [−0.35,+1.32]（**含 0**）、Δedit +0.0135 [+0.0045,+0.0192]（**显著**）
  - 补充检验（Lead）：逐样本 edit 配对 t=3.03（p≈0.0025）、符号检验 468 胜/363 负（p≈3e-4）、样本级 BLEU bootstrap [−0.115,+1.429]
  - 覆盖率：A2 36.5% @0.5 → 三模型并集 **46.6%（+10.1pp）**；@0.75 12.2% → 17.3%
  - 成本：+26 ms P50（pilot 同序）、+8 MB 显存；1086/1086 成功、0 空输出
- 决策：
  1. **有视频输入时首选 RGB+pose**（BLEU 26.30 / edit 0.436，均优于 A/A2；延迟与 A 同量级，约 2.7 s）
  2. **只有官方 pose 归档（无视频）时用 A2**（25.58，0.28 s）
  3. **批处理/离线**可用三模型并集提升覆盖率（46.6% @0.5），但需人工复核兜底
- 保留条件：BLEU 的 signer 聚类 CI 含 0（10 个 signer 的保守 CI）；RGB 输入用本地近似 trimmed 视频（官方 `sentence-crop` 帧缺失）。**在 Round-2 人评 + 官方帧协议复核完成前，不把 RGB+pose 写死为唯一默认**。
- 回退：A2（pose）/ A（视频，若手部 crop 不可用）。
- 影响：主链路模型选择从"单模型 A/A2"变为"按输入可用性分层"；`slbridge` 现有 adapter 支持 pose-only，RGB+pose 需新增 adapter（P1 任务）。
