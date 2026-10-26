# E03 · SLRT 分数定义、校准与错误风险（P-1）

- run_id: `20261026_e03_slrt_confidence`
- 生成时间（UTC）: 2026-10-26T13:36:44+00:00
- 数据集: 1086 条 CSL-Daily 本地视频子集，split=test，`analysis_scope=retrospective_seen_1086`（历史已见数据，**不是新盲测**）
- GT（仅评测侧）: `experiments\p1\references\gt_1086.csv` sha256 `feb1b42daa74ed3e…`
- SLRT 历史输出: `artifacts\csldaily-video-test-627-20261026\slrt`（1086 条样本，缺失 0 条；本批全部 status=final、0 空 gloss）
- 状态: **PASS**（SLRT raw score 校准与风险分析已完成）；**Uni-Sign 置信度 = BLOCKED**（见 §6）

## 1. 结论摘要

- SLRT corpus WER（gloss 对 GT gloss）: **0.3018** (S=1330/D=717/I=434, 参考 8220 tokens)。总编辑数与历史 [R18] 一致（S1,154/D805/I522 = 2,481 次编辑、WER 0.3018）；本表 S/D/I 拆分不同仅因 Levenshtein 回溯并列时的选择顺序，**不影响 WER 与总编辑数**。
- 标签: error_any(WER>0) **928** 条 (0.8545), bad(WER≥0.5) **216** 条 (0.1989), bad_03(WER≥0.3) **462** 条 (0.4254); 签者 10 人
- **raw token score 不是正确概率**：把 `token_score_mean` 直接当 P(correct) 时 ECE=**0.3041**、Brier=**0.2323**。分数范围被压缩（mean ∈ [0.0627, 0.6640]），且系统性低估正确率：score≈0.44 的样本实际 70% 可接受（1-bad）、score≈0.50 时 83%、score≈0.56 时 93% —— 直接当概率会严重失校准（见 §3）。排序信号仍可用：AUC=0.7528。
- raw window score 同样未校准：ECE=**0.2348**、Brier=**0.1969**、AUC=**0.7401**。
- 单变量 Platt/logistic（token score）OOF 后：主标签 bad 上 AUC=**0.7449**、ECE=**0.0256**、Brier=**0.1353**；在 FDR≤0.15 策略下覆盖 **0.8481**、error-ACCEPT 率 **0.1531**、missed-upgrade 率 **0.6744**。
- 同一策略下 raw token score 阈值：覆盖 0.8573、error-ACCEPT 率 0.1568、missed-upgrade 率 0.6744；raw window score：覆盖 0.8214、error-ACCEPT 率 0.1570、missed-upgrade 率 0.7209。
- 结论：SLRT 分数**可用作风险排序信号**（AUC 明显 > 0.5），但**不能当作正确概率**；阈值必须在 signer 分组 OOF 上选择，且高置信样本仍有不可忽略的 bad 风险（见 §4 risk-coverage）。

## 2. 分数定义与模型卡

| 分数 | score_type | 对象 | 来源/语义 | 可用性 |
|---|---|---|---|---|
| SLRT token score | `uncalibrated_mean_ensemble_probability` | **输出 gloss token**（非帧、非整句） | 7 窗多数投票中，该 token 的每次 partial 事件取「参与投票窗口的该类别平均概率」，再对该 token 生命周期内事件取均值；见 `slbridge/slrt/decoder.py`。**未校准**，不是 P(correct) | READY（1086/1086） |
| SLRT window top_probability | 无单独 score_type 字段（softmax 最大类概率） | **单个 16 帧滑窗** | `*.windows.jsonl` 每窗 `top_probability`；每个 anchor frame 一窗，窗数≈帧数（如 52 帧→52 窗），长度依赖 | READY（1086/1086） |
| Uni-Sign 句级置信度 | — | — | runtime `TextPrediction(score=None)`（`slbridge/uni_sign.py:1007`）；历史 `predictions.jsonl` **没有** score/logprob 字段 | **BLOCKED** |

详见 `score_definition.md`（含字段样本与代码引用）。

## 3. 校准结果（OOF，bad = WER≥0.5）

| 方法 | 预测对象 | ECE(以bad为正类) | Brier | AUC |
|---|---|---|---|---|
| raw_token | token | 0.3041 | 0.2323 | 0.7528 |
| raw_window | window | 0.2348 | 0.1969 | 0.7401 |
| logistic_token | token | 0.0256 | 0.1353 | 0.7449 |
| logistic_window | window | 0.0169 | 0.1388 | 0.7308 |
| logistic_combo | token+window | 0.0275 | 0.1355 | 0.7460 |

raw score 直接当 P(correct) 的可靠性表（token score，全部 15 箱；后 5 箱无样本）：

| bin | n | mean_pred | observed_positive_rate |
|---|---|---|---|
| [0.0000 , 0.0667) | 1 | 0.0627 | 0.0000 |
| [0.0667 , 0.1333) | 1 | 0.1125 | 0.0000 |
| [0.1333 , 0.2000) | 1 | 0.1680 | 0.0000 |
| [0.2000 , 0.2667) | 1 | 0.2645 | 0.0000 |
| [0.2667 , 0.3333) | 16 | 0.3154 | 0.1875 |
| [0.3333 , 0.4000) | 64 | 0.3771 | 0.4531 |
| [0.4000 , 0.4667) | 234 | 0.4398 | 0.7009 |
| [0.4667 , 0.5333) | 384 | 0.4989 | 0.8255 |

## 4. 风险-覆盖与阈值策略（signer 分组 OOF）

阈值策略：在 9 个训练签者的**内层 OOF 置信度**上选最大覆盖阈值，使训练集 error-ACCEPT 率 ≤ target；再应用到留出签者。主 target = 0.15。

| 方法 | coverage | error-ACCEPT 率（接受中 bad 占比） | 占全部 bad 比例 | missed-upgrade 率 |
|---|---|---|---|---|
| raw_token | 0.8573 | 0.1568 | 0.6759 | 0.6744 |
| raw_window | 0.8214 | 0.1570 | 0.6481 | 0.7209 |
| logistic_token | 0.8481 | 0.1531 | 0.6528 | 0.6744 |
| logistic_window | 0.8168 | 0.1556 | 0.6389 | 0.7209 |
| logistic_combo | 0.8370 | 0.1518 | 0.6389 | 0.6977 |

missed-upgrade 的评测侧定义：bad=1 且 Uni-Sign 视频历史文本 char edit similarity ≥ 0.5（即另一条路径本可给出可用译文）。这是**离线代理真值**，不是人工升级真值。

升级机会（评测侧代理）：should_upgrade = **43/1086** 条，其中 **29** 条仍被策略 ACCEPT（missed-upgrade 率 **0.6744**）。即 SLRT 分数只能识别「bad」的一部分，「bad 且另一路本可给可用译文」的升级机会与低分区间只有部分重叠；仅凭 SLRT 分数决定升级会漏掉多数升级机会，需 E06 的级联/冲突规则补充。

risk-coverage 关键点（OOF 排序，FDR = 接受样本中 bad 占比）：

| 方法 | coverage=25% | 50% | 75% | 100% |
|---|---|---|---|---|
| raw_token | 0.0478 | 0.0921 | 0.1241 | 0.1989 |
| raw_window | 0.0588 | 0.0902 | 0.1314 | 0.1989 |
| logistic_token | 0.0551 | 0.0921 | 0.1314 | 0.1989 |
| logistic_window | 0.0625 | 0.0884 | 0.1339 | 0.1989 |
| logistic_combo | 0.0551 | 0.0866 | 0.1302 | 0.1989 |

完整曲线见 `risk_coverage.csv`（coverage 5%–100%，含 threshold / accepted_n / fdr）。

## 5. 敏感性（error_any / bad_03）

| 标签 | 方法 | AUC | ECE | 主 target 下 coverage | error-ACCEPT 率 |
|---|---|---|---|---|---|
| error_any | raw_token | 0.7199 | 0.3564 | 0.0009 | 1.0000 |
| error_any | raw_window | 0.7410 | 0.4329 | 0.0083 | 0.2222 |
| error_any | logistic_token | 0.7072 | 0.0228 | 0.0000 | null |
| error_any | logistic_window | 0.7335 | 0.0253 | 0.0092 | 0.2000 |
| error_any | logistic_combo | 0.7283 | 0.0258 | 0.0083 | 0.1111 |
| bad_03 | raw_token | 0.7305 | 0.1347 | 0.0552 | 0.2000 |
| bad_03 | raw_window | 0.7314 | 0.1207 | 0.1611 | 0.1771 |
| bad_03 | logistic_token | 0.7259 | 0.0346 | 0.0387 | 0.2619 |
| bad_03 | logistic_window | 0.7271 | 0.0277 | 0.1464 | 0.1635 |
| bad_03 | logistic_combo | 0.7347 | 0.0364 | 0.1906 | 0.1498 |

注意：`error_any` 包含仅插入错误（WER>0 但 GT 内容全对），因此 bad=0 时也可能 error_any=1；两个标签的风险含义不同，不能混用。`error_any` 基线错误率 85.5%，FDR≤15% 的阈值策略在它上面几乎不接受任何样本（coverage≈0），说明 token score 无法区分「局部/插入错误」与「基本正确」；实际路由应以 bad（WER≥0.5）为主标签而非「任何错误」。

## 6. 边界与阻塞

1. **Uni-Sign 置信度 = BLOCKED**：当前 runtime 与历史产物都没有 logprob/score。证据：
   - `slbridge/uni_sign.py:1007` 构造 `TextPrediction(..., score=None)`；`slbridge/contracts.py:364-369` / `slbridge/api_schemas.py:137-140` 的 `score` 默认 `None`。
   - 历史 `unisign-video-1086-paperrefs/predictions.jsonl` 每行字段：`sample, frames, reference, prediction, status, error, gloss, char_edit_similarity, rouge_l_f1_char, exact_match, e2e_seconds, pose_ms, model_total_ms` — **无 score/logprob**。
   - `docs/agents/model-evaluation-pitfalls.md` §6：「当前 TextPrediction 返回整句文本、score=None；beam=4 不代表已经提供 N-best、词级概率或动作时间对齐」。
   - 不得用编辑相似度/文本稳定性冒充模型置信度；文本稳定性只能在 E05 前缀数据上作为弱信号，且必须标注为「非模型置信度」。
2. 1086 是 retrospective_seen 子集（本地已有 test 视频的 1086/1176），不是新盲测；SLRT 为近似句段裁剪的 GLOSS 输出，不是 `sentence_frames-512x512` 官方协议。
3. SLRT 交付物是 gloss 序列，不是自然语言句子；本报告的 WER 是 gloss 级指标，不能与 Uni-Sign 的 BLEU/chrF 直接比较。
4. raw score 阈值/校准结果不能跨模型复用（E03 协议 §2：不同模型不可共用分数阈值）。
5. 标签阈值（bad=0.5）是分析选择，不是官方验收阈值；`bad_03`/`error_any` 已作敏感性。

## 7. 产物

| 文件 | 说明 |
|---|---|
| `experiments/p1/e03/per_sample_scores.csv` | 1086 行逐样本分数/WER/标签（含 OOF 概率列，缺失保留 null） |
| `experiments/p1/e03/calibration_results.json` | ECE/Brier/AUC、可靠性分箱、每折阈值与权重、风险策略 |
| `experiments/p1/e03/risk_coverage.csv` | 5 方法 × 3 标签 × 20 coverage 点（5%–100%） |
| `experiments/p1/e03/score_definition.md` | 分数对象/语义/字段证据 |
| `experiments/p1/runs/20261026_e03_slrt_confidence/manifest.json` | 命令、脚本/数据 sha256、样本数、状态 |

