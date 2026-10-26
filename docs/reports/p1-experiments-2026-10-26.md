# P-1 实验计划执行报告（E00–E06，2026-10-26）

日期：**2026-10-26**。执行对象：`SignLanguageBridge_P-1_实验计划包/`（PLAN.md, docs/00–07）。
本报告是项目级索引；逐实验细节在 `docs/experiments/P-1/P1_EXECUTION_REPORT.md`，原始证据在 `experiments/p1/`。
**110 分钟实跑窗口**（21:18–23:0x +08:00），1 名 Lead + 4 名 Agent 并行；GPU 串行调度，CPU 分析并行。

## 结论先行

| 项 | 结果 |
|---|---|
| E00 资产/环境/数据审计 | **PASS（审计侧）**：6+ 模型实测 SHA256/许可；环境/数据/入口/Git 风险记录；G0 smoke 由 E01 提供 |
| E01 Pilot-50 复现 | **PASS**：三路 50/50 成功，历史文本 **100% 逐字复现**（视频、官方 pose、SLRT gloss 均 50/50） |
| E02 同任务翻译 | **PASS（离线回放）**：A 24.8157 / A2 25.5846 / B0 12.2183 BLEU-4；A-A2 配对 Δ +0.7689（CI +0.23…+1.44）；oracle 上限 26.598；C/D–G BLOCKED |
| E03 置信/风险 | **PASS（SLRT）；Uni-Sign BLOCKED**：raw score 非概率（ECE 0.30）；OOF Platt AUC 0.745 / ECE 0.026；FDR≤15% 策略 error-ACCEPT 0.153、missed-upgrade 0.674 |
| E04 GPU 性能 | **PASS**：热启动 e2e P50——Uni-Sign pose 277.6 ms / video **2,690.4 ms**（pose 占 80%）/ SLRT **9,865.1 ms**；串行 40.28 s vs 并行 34.10 s（−15% 总墙钟但单条变慢 1.18–1.81×、峰值显存 6.45 GB） |
| E05 前缀/在线 | **PASS（探针+源码审计）**：Uni-Sign pose 80/80、`first_stable_prefix` mean 90%/median 100%、改写率 67%，6/10 句要到 100% 前缀才稳定 → 无原生 Partial；video 40/40、最短 20 帧；SLRT partial 需 11 帧 lookahead |
| E06 Rule 仲裁 | **PASS（Rule-only）；LLM/Jev BLOCKED**：级联 A2 调用率 38.1%、+0.17 BLEU、错停优胜 5.2% |
| G0/G1 | **PASS** |
| G2 | **PARTIAL**（A/B0 可比；SLRT 句子侧与人评、LLM 融合 BLOCKED） |
| G3/G4 | 见执行报告 §10–11 |

## 关键实测数字（全部可回溯 run_id）

1. **复现（E01，measured，Pilot-50）**
   - Uni-Sign 视频端到端：BLEU-4 **31.2308** / chrF 27.1849 / edit 0.4831；与历史 1086 快照逐条 **50/50 原样一致**；P50 3.625 s / P95 5.336 s。
   - Uni-Sign 官方 pose：BLEU-4 **31.7823** / chrF 27.9108 / edit 0.4885；**50/50 一致**；P50 0.351 s。
   - SLRT Gloss：corpus WER **0.3170**（S72/D21/I17，347 参考 token）；gloss 序列 **50/50 一致**；单条 CLI 14.46 s（含 ~3.9 s 权重加载）。
   - 视频 − pose = **−0.5515 BLEU / −0.0054 edit**（1086 全量：−0.77/−0.0055，方向一致）。
   - 参考口径实测：同一份预测，官方全角参考 BLEU **28.10** vs 上游半角参考 **31.23**（差 **3.13**）。
2. **同任务基线（E02，imported 1086）**
   - A 视频 24.8157 / 21.7974 / 0.4157；A2 官方 pose 25.5846 / 22.4284 / 0.4212；B0 Gloss join 12.2183 / 13.8087 / 0.3484。
   - A2−A：Δedit +0.0055（95% CI −0.0013…+0.0116，**含 0**）；ΔBLEU +0.7689（CI +0.2299…+1.4372）。
   - 逐样本：A2 更好 241 / A 更好 203 / 持平 642；纠正 93 / 误改 50；误差重叠：都差 661、仅 A 差 43、仅 A2 差 29。
   - 离线 GT oracle A/A2：BLEU 26.598（选择 A 830 / A2 256）。
3. **置信/风险（E03，SLRT 历史 1,086）**
   - raw token score 当 P(correct)：ECE **0.3041** / Brier 0.2323 / AUC 0.7528；不是正确概率。
   - OOF Platt（10 签者 LOSO）：AUC **0.7449** / ECE **0.0256** / Brier 0.1353。
   - FDR≤15% 策略：coverage 0.8481、error-ACCEPT 0.1531、missed-upgrade 0.6744。
   - Uni-Sign 无导出分数（`TextPrediction(score=None)`）→ 置信度 BLOCKED。
4. **规则级联（E06）**：P2 BLEU 24.9861、A2 调用率 38.1%、missed-winner 5.2%、纠错 0.7%、误改 0.5%；P3 全跑 24.9559；oracle A/A2 26.598。
5. **性能/流式（E04/E05，measured；RTX 5060 Ti 16 GB，桌面基线上测量）**
   - 冷启动加载 P50：Uni-Sign pose 17.7 s / video 26.6 s（P95 65.1 s，2 次宿主内存压力导致的 CUDA OOM 已重试成功）/ SLRT 3.1 s。
   - 热启动 30 次 e2e P50/P95：pose **277.6/344.3 ms**（RTF 0.078，3.27 条/s）、video **2,690.4/3,283.9 ms**（RTF 0.748，0.339 条/s，pose 前端占 80%）、SLRT **9,865.1/11,189.7 ms**（RTF 2.53，0.099 条/s，~90 ms/窗，其中 S3D forward 3.7 s、逐窗 CPU 预处理 ~5.6 s）。
   - 连续负载 2 分钟 0 失败/0 OOM/无显存增长；torch peak alloc：Uni-Sign 1,206 MB（reserved 2,738）、SLRT 374 MB（reserved 600）；整卡峰值 单模型 5.66 GB、并行 6.45 GB。
   - 串行 40.28 s vs 两进程并行 34.10 s（−15% 总墙钟），但单条延迟变差（video 1.81×、SLRT 1.18×）、GPU util 仅 27.8% → 实时默认串行，并行只适合离线批处理。
   - 前缀探针：Uni-Sign pose 10 条 × 8 前缀 **80/80**；`first_stable_prefix` mean **90% / median 100%**，分布 62.5%:1、75%:2、87.5%:1、100%:6；相邻前缀改写率 **67%**、空输出 0、与作者快照 10/10 一致；video 路径 40/40、最短可用前缀 20 帧。结论：Uni-Sign 输出是整句重估，**不能当流式模型**。
   - 源码审计：Uni-Sign `needs_future_frames=YES`、`supported_partial=NO`（对称时间卷积 ±6 帧 + 双向 mT5 encoder + 单次 generate；上游 `online_inference.py` 是读视频路径整段推理）；SLRT partial 需 11 帧已观测 lookahead（30 fps ≈ 0.37 s，不含推理/排队）。

## 新增工程判断（供 V3.0 接收）

- **模型池**：Uni-Sign（视频/官方 pose）与 SLRT（Gloss）均可在目标机复现；SLRT 的句子翻译（SLT/G2T）不在模型池（权重缺失，BLOCKED）。
- **融合**：A/A2 互补空间存在（oracle +1.78 BLEU）但规则级联只能兑现 +0.17；无可靠无 GT 信号前**不启用自动融合**。
- **在线**：不得把整句离线延迟写成实时首稳；实时能力需要独立在线模型或明确窗口等待披露。
- **可靠性**：不启用自动接受；SLRT 分数只做排序，且必须 signer 分组校准。
- **LLM/Jev**：本轮无端点、无证据 → 只保留接口，不进主链路。

## G2 追加：真实 LLM 融合首轮结果（2026-10-27，探索性 50 条）

> 详见 `docs/experiments/P-1/G2_DECISION_MEMO.md`、ADR-P1-010。端点 = wb-proxy / `deepseek-v4.1-flash`（temperature 0，串行）；样本 = Pilot-50；`retrospective_seen`，非最终盲测。

| 组 | v1 BLEU-4 | v2 BLEU-4 | 关键配对（v2） |
|---|---:|---:|---|
| A Uni-Sign 原句 | **31.2308** | **31.2308** | 基准 |
| C 润色对照 | 26.6838 | 28.9518 | C−A **−2.28 [−4.04,−0.70]**（显著为负） |
| D + SLRT Gloss | 26.9450 | 28.5872 | D−C −0.365（n.s.） |
| E + 证据约束 | 28.6302 | 29.1880（n=45） | E−D −0.416（n.s.） |
| 选择性级联（触发 28%） | — | 29.089 | 相对 A −2.14 [−5.27,+0.44]；15k tokens |

- **G2 结论：FAIL（无质量增益证据）** → 融合不进主链路，保留可插拔接口；不扩 1086；SLRT 的 Gloss 价值未被否定（B=G2T/SLT 仍 BLOCKED）。
- 成本：v2 共 113k tokens，P50 2.04–4.48 s / P95 5.0–9.5 s；全调 E 为 −6.56 BLEU。
- 事故隔离：v1 首轮 `max_tokens=256` 导致 73/150 空 completion，该轮仅作证据、未进入对比表。
- 未验证：六槽位人工评审（Round-1 包已完成、待用户回填）——"关键事实错误"维度不是通过。
- 可复算证据：`experiments/p1/e02/llm_fusion/runs/g2a-wbproxy-50-v2*-20261026/metrics.json`、`cascade_analysis_v2.json`、`report_prompt_v2.md`。

## 产物

- 执行报告与 Gate/ADR：`docs/experiments/P-1/P1_EXECUTION_REPORT.md`
- 计划副本：`docs/experiments/P-1/`（原包保留在 `SignLanguageBridge_P-1_实验计划包/`）
- 基础设施/数据/评测：`experiments/p1/{common.py,gt_guard.py,datasets/,references/,eval/,tests/}`
- 逐实验产物：`experiments/p1/e00/` … `experiments/p1/e06/`
- 运行记录：`experiments/p1/runs/<run_id>/`（7+ 个 run，全部 GT-free 预测 + manifest + metrics）
- 汇总：`experiments/p1/summary.json` / `summary.md`

## 边界（不得越过的声明）

- 1,086 条为 `retrospective_seen_1086`（本地 test 覆盖 1,086/1,176，缺 90 条偏长句），**不是新盲测**；无独立 signer 泛化结论。
- E01 只覆盖 50 条诊断样本；S1 未重跑 align（复用已审计历史产物）；未覆盖 >256 帧采样分支、重复性与跨机器。
- E02/E03/E06 的模型文本来自 imported 历史快照；路由/级联/校准是离线/模拟，不是线上时延或线上收益。
- 人评 **NOT_RUN**（严重语义错误率/关键事实正确率必须由人评协议给出）。
- SLRT 为近似句段裁剪，不是官方 `sentence_frames-512x512` 协议；SLRT 上游许可 UNKNOWN；Uni-Sign 权重 CC BY-NC 4.0。
- Git 脏树 132 项（`slbridge/slrt`、脚本、P-1 全部未提交）是最大复现风险。
