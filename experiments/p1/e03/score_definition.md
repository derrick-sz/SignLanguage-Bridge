# E03 · Score definition（SLRT / Uni-Sign）

运行：`20261026_e03_slrt_confidence`，生成于 2026-10-26T13:36:44+00:00。本文件只描述**分数的对象与语义**，数值结果见 `report.md` / `calibration_results.json`。

## 1. SLRT token score

- 字段：`slrt/<sample>.json` → `tokens[].score`，伴生 `tokens[].score_type`。
- `score_type`：**`uncalibrated_mean_ensemble_probability`**（1086 条样本全部为该值；无其他类型）。
- 对象：**单个输出 gloss token**（事件级），不是帧、不是全句、不是 CTC 帧级后验。
- 语义（`slbridge/slrt/decoder.py`）：
  1. 每个 16 帧滑窗输出 2001 类 softmax；7 窗中心多数投票（vote_size=7）决定该位置 token id。
  2. 对参与投票的 7 个窗，取该 token id 类别概率的均值 `p`（`decode._drain`：`p = mean([r['prob'][k] for r in window])`）。
  3. token 生命周期内每次 partial 事件累计 `p`，final 时取 `score = score_sum / count`（`decode._event`）。
- 含义：**投票一致性与类别概率的混合量**，值域 [0,1]，但**未做任何校准**，不是 P(输出正确)、不是 P(token 正确)。
- 缺失：本批 1086/1086 都有 score；空输出样本不在本批（0 空 gloss），若未来出现，token 级分数为空，应填 `null` 而不是 1.0。

字段样本（`S000020_P0004_T00`）：

```json
{"event_id": "0:0", "token_id": 2, "gloss": "他", "status": "final",
 "score": 0.6312378015783098, "score_type": "uncalibrated_mean_ensemble_probability",
 "reason": "run_ended", "boundary_source": "sliding_window_anchor_estimate"}
```

## 2. SLRT window top_probability

- 字段：`slrt/<sample>.windows.jsonl` → 每行一个滑窗的 `top_id` / `top_probability` / `inference_ms` / `preprocess_ms`。
- 对象：**单个 16 帧滑窗**（帧级窗口分类结果），每个 anchor frame 一个窗口；窗数≈视频帧数（52 帧样本有 52 窗）。
- 语义：该窗口 2001 类输出中最大类别的概率。**未校准**；且窗口有左右重复 padding（`left_edge_repeats`/`right_edge_repeats`），边界窗口的概率不可与中间窗口等价解读。
- 与 token score 的区别：token score 是 7 窗投票后的 token 生命周期均值；window top_probability 是单窗瞬时值。二者**不可混用阈值**。

字段样本：

```json
{"segment_id": 0, "anchor_frame": 0, "anchor_time": 0.0, "top_id": 2,
 "top_probability": 0.7509263753890991, "left_edge_repeats": 7, "eos_padding": false}
```

## 3. Uni-Sign 句级置信度 = BLOCKED

- 当前生产 runtime：`slbridge/uni_sign.py:1007` → `TextPrediction(text=..., language="zh", score=None)`。
- 数据合同默认值：`slbridge/contracts.py:364-369` 与 `slbridge/api_schemas.py:137-140` 的 `score: float | None = None`。
- 历史 1086 产物：`artifacts/csldaily-video-test-627-20261026/unisign-video-1086-paperrefs/predictions.jsonl` 与 `artifacts/cslnews-eval/csldaily-test-1086-pose-20261026/predictions.jsonl` 均**没有** score/logprob/beam 字段（字段见 report §6）。
- 项目坑点记忆 §6：整句文本、`score=None`；beam=4 不代表 N-best/词级概率/时间对齐。
- 结论：**BLOCKED**。要做 Uni-Sign 校准，需要先让 runtime 导出 `generate(output_scores=True, return_dict_in_generate=True)` 的 token logprob 或 beam score，并用同一批输入重跑；不得用编辑相似度、文本稳定性、多次解码一致性冒充模型置信度（后者只能作为弱信号且须标注）。

## 4. 本报告使用的分数特征

| 特征 | 来源 | 对象 | 方向 |
|---|---|---|---|
| `token_score_mean/min/median/p10` | tokens[].score | token | 越高越自信（风险越低） |
| `window_top_prob_mean/min/p10` | windows[].top_probability | window | 越高越自信（风险越低） |
| `low_prob_window_frac_05/03` | 同上 | window | 越高风险越高 |
| `gloss_length_ratio` | pred_tokens/ref_tokens | 序列长度 | 诊断用（过/欠生成） |

## 5. 模型卡（分数对象）

| model_id | 任务 | score_source | score_type | 对象 | 校准状态 |
|---|---|---|---|---|---|
| `slrt-emnlp24-csl-daily` | CSLR（gloss 序列） | `tokens[].score` | `uncalibrated_mean_ensemble_probability` | token | 本报告训练了 OOF Platt 映射（仅风险排序，不是绝对概率） |
| `slrt-emnlp24-csl-daily` | CSLR | `windows[].top_probability` | 单窗 softmax max | window | 同上（raw 未校准） |
| `unisign`（视频/官方 pose） | CSLT（中文句子） | 无 | — | — | **BLOCKED：无导出分数** |

