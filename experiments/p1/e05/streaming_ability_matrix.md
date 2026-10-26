# E05 · 流式能力矩阵（streaming_ability_matrix）

> **状态**：`PASS`（实测）。源码事实列来自只读审计
> `future_frame_audit.md`；实测列由 `experiments/p1/e05/prefix_probe.py` 在
> `e05-prefix-20261026T2226Z`（pose）与 `e05-prefix-video-20261026T2232Z`（video）
> 两个 run 中产生。仍标 `NOT_RUN` 的单元格表示本轮未跑，不得用历史数字填充。
>
> **核心口径**：本表把「已观测前缀」与「实时流」严格分开。前缀探针是冻结前缀的离线
> 回放，只能回答“译文在哪个已有前缀比例上稳定”，**不是** `first_stable_latency`
> （后者必须由真实 PTS + 墙钟事件定义）。

## A. 术语与判定标准

| 术语 | 本表定义 |
|---|---|
| `needs_future_frames` | 模型发布某个时间点/某个 token 的结果，是否必须看到该时间点之后的输入帧。`YES` = 非因果；`BOUNDED:n` = 需要 n 帧已到达的右侧上下文（不是未来泄漏）；`NO` = 逐帧因果 |
| `supported_partial` | 模型/运行时是否提供**原生**、可改写、按时间推进的部分输出事件；反复对完整窗口重算得到的文本不算 |
| `supported_final` | 是否有明确的“该 run 结束”事件；`final` 仍只是模型假设，不是人工确认/语义真值 |
| 前缀指标（E05 探针） | 见 §B；只对 **已观测前缀** 定义 |
| 未来帧审计 | `experiments/p1/e05/future_frame_audit.md`（含行号证据） |

## B. 冻结前缀探针的指标定义（与 `prefix_probe.py` 实现一致）

- 取样：Pilot-50 中按 signer 分层选 10 条（1 条/signer，确定性排序）。
- **已观测前缀**：官方发布 pose pkl 先按 `start`/`end` 裁剪得到句子片段（T 帧），
  前缀 = 前 `N = max(2, round(T·f))` 帧；视频路径 = trimmed mp4 前 N 个解码帧。
  **不复制、不外推、不填充任何未来帧**；`N > T` 不可能发生。
- `first_match_prefix`：第一次出现与 100% 前缀相同文本的前缀比例。
- `first_stable_prefix`：从该前缀起，后续所有前缀的文本都等于 100% 文本的**最早**
  前缀比例（= 最终文本第一次出现且不再改变）。这是“冻结前缀稳定性”，不是在线
  首稳时延。
- `rewrite_rate_adjacent`：相邻前缀对文本发生变化的占比（每样本），再取样本均值。
- `empty_rate`：各前缀空输出占比。
- `similarity_to_100`：每个前缀输出与 100% 输出之间的字符编辑相似度（无 GT）。
- 历史一致性：100% 前缀输出 vs 作者发布 `test_tmp_pres.txt` 快照（同 checkpoint
  家族；帧采样 linspace≠random.sample，不要求逐字相同）。
- 决定论：固定 seed、beam=4；对同一样本 100% 输入重复两次比对。

## C. 能力矩阵

| # | 路径 / 入口 | 输入单元 | needs_future_frames | 未来帧 / 上下文要求 | supported_partial | supported_final | 修订语义 | 证据 |
|---|---|---|---|---|---|---|---|---|
| 1 | Uni-Sign pose（官方 released pkl，离线） | 整句姿态片段 | **YES** | 时间卷积 ±2帧/层×3层≈**±6 帧**有界感受野；但 mT5 encoder 双向全注意力 + 单次 `generate` 整句输出 → 追加未来帧会改写整句 embedding，模型没有任何增量解码接口 | **NO** | **NO**（只有一次整句输出） | 无协议层修订；不同前缀=独立重算，文本可任意改写 | `models.py:93,139,206,260-323`；`stgcn_block.py:66-88,121-127`；本地 `uni_sign.py:446-482,988` |
| 2 | Uni-Sign video（trimmed mp4 → RTMPose133 → mT5） | 整句视频片段 | **YES**（同 1，且先整段解码/提姿态） | 同 1；`_video_pose_window` 必须先解码到 EOF/600 帧上限，才 `prepare_pose_window` + 一次推理 | **NO** | **NO** | 同 1；前端（RTMPose vs 官方 pose）另有差异，不构成流式能力 | `uni_sign.py:853-946,1029-1045,988` |
| 3 | Uni-Sign 连续滑窗（本地 `continuous.py` + `TextStabilizer`） | 完整窗口（≤8 s / ≤600 帧） | **YES**（每个窗口都要完整，且右边界是“窗口截止”） | 每 `update_interval`（默认 0.5 s）对**完整候选窗口**重新推理；窗口内帧必须全部到齐；无跨窗状态 | **NO（伪 partial）** | **NO**（`stability_state=confirmed` 仅表示连续 2 次窗口文本相同，仍可被后续窗口改写） | 完整重算 → 文本修订；`confirmed` 不是人工确认 | `continuous.py:10-24,63-126`；`uni_sign.py:151-204` |
| 4 | **SLRT Online（原生滑窗，hrnet-npz / rtmpose-experimental）** | 逐帧 `FramePacket`（时间戳+帧号） | **BOUNDED: 11 帧**（stride=1, window=16, vote=7） | 窗口左 7 / 右 8；中心帧 `c` 的 partial 需帧 `c+1..c+8` 已到达（8 帧，右上下文），且投票需窗口 `c..c+3` 全部推送 → 需帧到 `c+11`（再 +3 帧）→ 合计 **11 帧已观测未来帧**（30 fps ≈ 0.37 s，不含推理/排队/采集）；EOS 时才允许复制右边界 | **YES**（`partial`，可改写/撤回） | **YES**（`final` = 投票 run 结束或自然 EOF；非人工确认） | partial 可被后窗改写；`interrupt()` 发 `retracted`；final 可能晚于 partial | `stream.py:24-29,66,82-91,129-159,161-169`；`decoder.py:9-22,48-95`；`cli.py:128-131` |
| 5 | SLRT 文件视频（`cli video`，当前批量路径） | 完整 mp4 | BOUNDED（同 4 的窗口语义） | 逐个窗口顺序跑完整个视频；无实时 pacing，输出在整段处理完后汇总 | 名义上有（窗口事件 JSONL），但**不是实时流** | 同上 | 批处理；墙钟执行时间短于视频时长也不等于实时交付 | `slbridge/slrt/backend.py:64-108`；`cli.py:128-135` |
| 6 | SLRT replay（`cli replay`，按原时间轴回放） | 逐帧、真实 pacing | BOUNDED（同 4） | 同 4；用于检验队列/交付/首稳墙钟 | YES | YES | 同 4 | `slbridge/slrt/capture.py`（本 E05 不重复跑，引用既有报告口径） |

## D. 实测列（measured；run_id `e05-prefix-20261026T2226Z` pose / `e05-prefix-video-20261026T2232Z` video）

### D.1 Uni-Sign pose 前缀曲线（`prefix_probe.py` pose phase，10 条 × 12.5/25/37.5/50/62.5/75/87.5/100%）

| 指标 | 值 |
|---|---|
| 样本数 / 推理次数 / 失败 | 10 条（1 条/signer）/ **80/80 成功** / 0 失败；模型加载 6.8 s |
| `first_stable_prefix` P50(median) / mean / 分布 | **100% / 90.0%**；{62.5%:1, 75%:2, 87.5%:1, 100%:6} |
| `first_match_prefix` mean | **90.0%**（与 first_stable 完全一致：最终文本一旦出现就不再改写） |
| 相邻前缀改写率（均值） | **0.6714**（67% 的相邻前缀对文本改变） |
| 空输出率 / 100% 空输出数 | **0.0 / 0** |
| 各前缀 P50 e2e / forward ms | 12.5% 204/52；25% 235/48；37.5% 239/50；50% 256/39；62.5% 261/49；75% 278/54；87.5% 299/54；100% 269/51（P95 e2e 276→405 ms） |
| 与历史快照一致性（逐字/相似度） | **10/10 逐字相同**，平均字符相似度 1.0（这些样本 ≤256 帧，确定性 linspace 未触发采样，故与作者 `test_tmp_pres.txt` 完全一致） |
| 决定论复跑（same seed=0, beam=4, 100%×2） | **identical=true** |
| 归一化漂移证据 | 各前缀 `body_scale` 均值 0.630→0.787（前缀越短 bbox 越不稳定）；每行已存输入 SHA256/body_scale/valid_body_fraction |

**解读**：6/10 句只有看到 100% 前缀才出现最终文本，平均要到 90% 前缀才稳定；
前缀每增加 12.5% 有约 2/3 概率整句改写。这是 Uni-Sign 无增量解码、整句重估的直接
证据，**不是**在线 Partial 修订率。

### D.2 Uni-Sign video 前缀可行性（`prefix_probe.py` video phase，10 条 × 25/50/75/100%，clip 模式）

| 指标 | 值 |
|---|---|
| 模式（clip / file）/ 成功 / 失败 / 空输出 | clip（内存帧，无重编码）/ **40/40 成功** / 0 失败 / 0 空输出 |
| 最短可用前缀帧数（可行性下界） | **20 帧**（更短的未尝试；≥2 帧是 runtime 硬约束，本批最小 12.5%≈20 帧） |
| 各前缀 P50 e2e / pose / forward ms | 25% 820/585/52；50% 870/599/53；75% 938/584/53；100% 901/558/54 |
| 视频前缀 vs pose 前缀 一致性（前端不同，仅参考） | 40 对：**16 对逐字相同**，平均字符相似度 **0.6906**（官方 released pose vs 本地 RTMPose，前端不同，不构成正确性判断） |

### D.3 SLRT 原生流首稳（不在本 E05 探针内重复跑）

| 指标 | 值 |
|---|---|
| partial 首现 lookahead（算法） | **11 帧 ≈ 0.37 s @30fps**（源码推导，见 §C4） |
| `first_stable_wall_latency_ms` | `NOT_RUN`（需真实回放流；引用既有报告时须注明运行条件） |
| 窗口/投票实际时延 P50/P95 | 见 E04 `slrt_hrnet_npz` 行的 `forward_ms`/`e2e_ms`（批处理口径，非实时交付） |

## E. 判断与边界（写作约束）

1. **Uni-Sign 不是流式模型**：可继续作为“整句高质量后备翻译模型”；不得把冻结前缀
   稳定性写成 Online Partial/Final，也不得把 `first_stable_prefix × 视频时长` 当成
   在线首稳时延（它缺少捕获等待、队列、推理服务与显示）。
2. **SLRT 是唯一原生在线路径**，但当前证据边界：CE-CSL 跨域质量低、摄像头稳定
   Gloss 未验收、CPU 回退不代表实时可用（`model-evaluation-pitfalls.md` §10）。
3. 前缀实验的 100% 文本与作者快照不同属正常（帧采样/归一化/前端差异），不能当
   模型退化或提升；不得用 references/GT 计算前缀“逐词准确率”。
4. 前缀归一化会随前缀长度变化（`prepare_pose_window` 对当前前缀重算 body
   bbox/scale，`uni_sign_preprocess.py:142-167`），报告需同时给出 `body_scale`、
   `valid_body_fraction` 与输入哈希，便于复核。
5. 所有结论必须注明样本数（10 条 pilot，诊断用，非独立测试集）与 GPU 是否独占。
