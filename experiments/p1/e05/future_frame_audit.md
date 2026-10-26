# E05 · future-frame 源码审计（Uni-Sign vs SLRT）

> **范围**：只读源码，未修改任何源文件。审计对象为上游 Uni-Sign 源码
> `models/external/uni_sign_source/`（pinned commit `eed438b`）、本地运行时
> `slbridge/uni_sign.py`、`slbridge/continuous.py`、`slbridge/slrt/stream.py`、
> `slbridge/slrt/decoder.py`、`slbridge/slrt/cli.py`。
> 结论中的行号对应本次审计时的磁盘内容；`needs_future_frames`/`supported_partial`
> 是源码事实判断，不是性能数字；实测首稳/时延见 `streaming_ability_matrix.md` 与
> `prefix_probe_summary.md`（GPU 释放后填写）。

## 0. 结论速览

| 路径 | needs_future_frames | supported_partial | 结论一句话 | 主要证据 |
|---|---|---|---|---|
| Uni-Sign 模型本体（pose-only mT5 SLT） | **YES（非因果）** | **NO** | 时间卷积只有约 ±6 帧有界感受野，但 mT5 encoder 是双向全注意力、且每次只对“完整给定序列”生成一句文本；追加未来帧会改变整句 embedding，输出可整句重写 | `models.py:93,139,206,260-323`；`stgcn_layers/stgcn_block.py:68-88,121-127` |
| 上游 `demo/online_inference.py` | 需要完整视频文件 | **NO** | 名称带 online，实际入口是 `--online_video {video_path}`：先把整段视频全部解码、全部提姿态，再一次性 forward+generate；不是实时采集，也不是增量推理 | `online_inference.py:22,26-29,62,64-104,106-142`；`datasets.py:606-665` |
| 本地 Uni-Sign 视频/姿态入口 | 需要完整片段 | **NO（只有反复重算伪 partial）** | `recognize_file` 先解码整个文件再单次推理；`continuous.py` 的窗口是“完整候选窗口再次整段推理”，`confirmed` 只表示连续两次文本相同，不是流式 Partial/Final | `slbridge/uni_sign.py:853-946,988,1029-1045`；`slbridge/continuous.py:63-126` |
| SLRT Online（本地 `slbridge/slrt/stream.py`） | **需要已到达的右侧上下文**：每窗左 7 右 8 帧；投票再等 3 个窗 → 相对中心帧约 **11 帧** 已观测未来帧（30 fps ≈ 0.37 s） | **YES**（原生 `partial` 可改写/撤回，run 结束才 `final`） | 官方在线分支：16 帧窗、stride 1、7 窗多数投票、只在自然 EOF 复制右边界；没有跨窗口 S3D 特征缓存，也不得访问未来帧 | `stream.py:24-29,129-159,161-169`；`decoder.py:9-22,48-95`；`cli.py:128-131` |

一句话总结：**只有 SLRT 的 online 分支具备“无未来帧泄漏的原生 Partial”**；Uni-Sign
两条路径都是“整段输入、整句输出”，可以作为整句高质量后备翻译模型，但不能宣称原生
流式。E05 的冻结前缀探针只能回答“如果把当前已观测前缀当成一个完整短句，译文最早在
哪个前缀比例稳定”，**不等于实时流首稳时延**。

## 1. Uni-Sign 是否需要未来帧

### 1.1 时间卷积：有界、但对称（需要未来帧）

- `models.py:88-96` 为四个部位（body/left/right/face_all）建立空间 GCN 与融合 GCN；
  `models.py:93` 的 temporal 链使用 `kernel_size=(5, spatial_kernel_size)`。
- `stgcn_layers/stgcn_block.py:66-70`：`assert kernel_size[0] % 2 == 1`，
  `padding = ((kernel_size[0] - 1) // 2, 0)` → 时间核 5 时左右各 pad 2；
  `stgcn_block.py:77-88` 的 TCN `Conv2d(..., padding)` 同样是对称 padding。
- `stgcn_block.py:121-127` 的 `get_stgcn_chain(..., 'temporal', ...)` 使用
  `block_args = [[256, 3]]`，即 3 层 temporal block。
- 因此：**单帧特征要看到约 ±2 帧/层 × 3 层 ≈ ±6 帧**（最后 6 帧之外的未来帧不再
  影响该帧的视觉特征）。这是有界窗口，但仍然需要未来帧才能计算“当前最后一帧”的
  最终特征；模型没有“只用过去帧”的因果卷积开关。
- `models.py:206` 的 `forward(src_input, tgt_input)` 一次消费整个 `src_input`
  （全部采样帧），`models.py:260-263` 对每个部位执行 temporal GCN 后取时间均值池化
  `gcn_feat.mean(-1)` —— 池化本身也是对整个窗口求平均，窗口一旦变化，整段特征都变。

### 1.2 mT5 encoder：双向全注意力，追加未来帧会改写全部 token

- `models.py:139-140`：`self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)`;
- `models.py:266-277` 把多部位特征投影成 `inputs_embeds`，并前置语言提示 token；
- `models.py:279-295` 以**完整** `attention_mask` 调用
  `self.mt5_model(inputs_embeds=..., attention_mask=..., labels=...)` —— 这是标准
  T5 **encoder**（双向注意力），不是因果 encoder。
- `models.py:312-323` 的 `generate()` 只接收 `pre_compute_item['inputs_embeds']`
  与 `attention_mask`，一次 beam search 生成整句；没有“把已生成历史 + 新增帧”增量
  送入的接口。
- 本地运行时的 `TorchUniSignRuntime.infer`（`slbridge/uni_sign.py:446-482`）每次
  推理都重新构造 `source` 张量与 `attention_mask`，调用一次
  `self._model(source, target)` 再做一次 `self._model.generate(...)`；没有跨调用的
  encoder/decoder 状态复用。`slbridge/uni_sign.py:988` 也明确写下
  `"streaming": False`。

**判定**：Uni-Sign SLT 的输出是“给定整段序列后的一个句子假设”。即使只传入已观测
前缀（E05 的做法），它也会把该前缀当作完整句子重新编码；追加新帧后整句 embedding
改变，译文可能整句改写。因此 `needs_future_frames = YES`（非因果模型），且
`supported_partial = NO`（没有可发布的部分解码文本，只有“重算整窗”）。

### 1.3 上游 `demo/online_inference.py` 是读视频路径，不是实时采集

| 行 | 事实 |
|---|---|
| `online_inference.py:22` | `pose_data = pose_extraction(args.online_video)` —— 输入是一个视频路径参数 |
| `online_inference.py:64-104` | `pose_extraction()` 用 `cv2.VideoCapture(video_path)`，`while True: ret, frame = cap.read()` 把**整段视频所有帧读进内存** `vid_data`（86-92），线程池对每一帧跑 Wholebody，最后返回全部帧姿态 |
| `online_inference.py:26-29` | 把整段 `pose_data` 一次性赋给 `S2T_Dataset_online` 的 `rgb_data`/`pose_data` |
| `online_inference.py:62,106-142` | `inference()` 对整个 dataloader（`__len__ == 1`）跑一次 `model(src_input, tgt_input)` + 一次 `model.generate(..., num_beams=4)`，打印一条完整句子 |
| `datasets.py:606-623` | `S2T_Dataset_online.__len__` 返回 1（整段视频=一个样本） |
| `datasets.py:634-665` | `load_pose()` 在 `duration > max_length` 时用 `random.sample` 抽整段视频的帧子集（641），不做任何前缀截断/滑窗 |

结论：该 demo 的 “online” 只是文件名/参数名，实际是**离线整段推理**；不能作为
“直播功能已就绪”的证据（与 `01_实验协议.md` E05 节的提醒一致）。

## 2. 本地 Uni-Sign 实现：没有真实 Partial

| 位置 | 事实 |
|---|---|
| `slbridge/uni_sign.py:853-946` `_video_pose_window` | 打开视频后循环 `capture.read()` 直到文件结束（或 600 帧上限），每帧过 RTMPose，全部提完才 `prepare_pose_window` 并交给模型；推理发生在整段解码之后 |
| `slbridge/uni_sign.py:1029-1045` `recognize_file` | 一次调用 = 完整文件 → 完整姿态窗口 → 一次 `runtime.infer`；无窗口循环、无部分结果 |
| `slbridge/uni_sign.py:1005-1008` | 输出是单个 `TextPrediction`，`score=None`；没有词级 token/时间对齐 |
| `slbridge/uni_sign.py:151-204` `TextStabilizer` | `confirm_after=2`：只有**连续两次窗口完整推理文本相同**才把状态标 `confirmed`（175-193）；`reset()`/空文本会清零。它是“重复窗口一致性”启发式，不是流式 Partial/Final |
| `slbridge/continuous.py:10-24` `ContinuousConfig` | 窗口时长 ≤ 8 s、`max_frames` ≤ 600、`max_result_age_seconds` 等；窗口是完整候选片段 |
| `slbridge/continuous.py:63-126` `RollingWindow` | `snapshot()` 只在积累到 `minimum_frames`/`minimum_seconds` 后返回**完整 clip**（115-126）；`append()` 在时间跳变 > `max_observation_gap_seconds` 时 `clear()`（95-96）；滑窗推理 = 每次拿完整窗口重新调模型，无跨窗特征/状态复用（与坑点 §6 一致） |

**local partial/final 语义**（必须写进报告）：

- Uni-Sign：没有协议层 `partial`/`final` 事件；只有 `stability_state ∈
  {empty, temporary, confirmed}` 的文本一致性标注，`confirmed` 仍可能被后续窗口
  改写（`uni_sign.py:192-204`）。E05 冻结前缀实验里的“prefix 间改写”正是这种
  完整重算的改写，而不是在线 Partial 的修订。

## 3. SLRT Online：原生滑窗 / 投票 / 首稳

### 3.1 窗口几何与未来帧约束

| 行 | 事实 |
|---|---|
| `stream.py:24-29` | 文档字符串：训练协议 16 帧、左 7 右 8；只在自然 EOF 允许复制右边界；无跨窗口特征缓存 |
| `stream.py:129-131` | `right = window_size // 2`（=8）；`while self._next_anchor + right < self._received: self._predict()` —— 锚点窗口的右侧 8 帧必须已到达 |
| `stream.py:134-140` | `left = window_size // 2 - 1`（=7）；`start, stop = anchor-left, anchor-left+window`；非 EOS 时 `stop > received` 抛 `RuntimeError('Attempted to access future video frames')` |
| `stream.py:150-157` | `available = self._frames[-1].packet.timestamp`；`left_edge_repeats`/`right_edge_repeats` 只在 EOS 时非 0；`evidence = (frames[0].timestamp, frames[-1].timestamp)` |
| `stream.py:161-169` | `flush()` 只在 EOF 调用：补算剩余锚点（`eos=True` 复制右边界）并 `decoder.flush()`；手动 stop/断线不伪造 final |
| `stream.py:82-91` | `interrupt()` 会把当前 partial 标 `retracted` 并重置窗口；pause/reset/输入不连续都走这里 |
| `stream.py:66` | 事件 `available_time_source = 'latest_required_input_timestamp_not_wall_delivery'` —— `available_time` 是源时间戳，不是实际交付墙钟 |

**每个窗口的未来帧数**：锚点帧 `a` 需要帧 `a+1 … a+8` 已到达（8 帧右侧上下文）。

### 3.2 投票等待 3 个窗 = 再 +3 帧 lookahead

| 行 | 事实 |
|---|---|
| `decoder.py:15-22` | `vote_size=7`（奇数）；`_left = deque(maxlen=vote_size//2)` 保存已投票的左侧窗口 |
| `decoder.py:48-57` | `_drain`：`while self._pending and (eos or len(self._pending) > half)` —— 非 EOS 时需要 **4 个**已到达窗口（中心 1 + 右侧 3）才投票；`right = pending[:half+1]`，`left` 不足时用 `self._first` 复制 |
| `decoder.py:58-64` | 多数票：`counts.max() > vote_size//2` 才输出非 blank；token 变化时把上一 run 标 `final`（`run_ended`） |
| `decoder.py:65-77` | 同一 token run 期间持续发 `partial`（`vote`），分数是 `uncalibrated_mean_ensemble_probability` |
| `decoder.py:88-95` | `flush()` 只在 EOS：eos=True 排空 pending，未结束的 run 标 `final`（`eos`） |

**首稳 lookahead 推导**（stride=1, window=16, vote=7）：

- 窗口锚点 `c` 需要帧 `c+8` 已观测（8 帧未来）；
- 投票中心 `c` 需要窗口 `c, c+1, c+2, c+3` 都已推送（4 个 pending → 3 个右侧窗）；
- 窗口 `c+3` 需要帧 `(c+3)+8 = c+11` 已观测；
- 因此：**中心帧 `c` 的 partial 最早在帧 `c+11` 到达后才可发布 → lookahead
  = `window_size//2 + (vote_size//2)*stride = 8 + 3 = 11` 帧**；30 fps 下 ≈ **0.37 s**
  （不含推理服务时间，也不含采集/排队等待）。
- `final` 需要“run 结束后某个窗口的投票变为新 token”，因此比 partial 更晚；EOF 时
  `flush()` 用右边界复制补 final（`stream.py:161-169`、`decoder.py:88-95`）。
- `evidence_time_range` 覆盖参与投票的左右窗（`decoder.py:69,72-73`），报告不得把它
  当人工标注边界。

### 3.3 本地 partial/final 语义（写报告时必须原样保留）

- `partial`：当前投票 run 的中间结果，可被后续窗口改写；`score` 是未校准的
  ensemble 平均概率（`decoder.py:83-85`），不是已校准正确率。
- `final`：该 token run 在投票意义上结束（`run_ended`）或自然 EOF（`eos`）；
  **不是人工确认、不是语义真值**（`decoder.py:11-14` 注释）。
- `retracted`：`stream.interrupt()` 主动撤回（pause/reset/断流/无人），
  `stream.py:82-91`。
- 时间边界来自滑窗锚点估计（`stream.py:153-157`、`decoder.py:85`
  `boundary_source='sliding_window_anchor_estimate'`），不是精确词边界；`cli.py:130-131`
  另外单独记录 `first_stable_wall_latency_ms`，且注明
  `wall_latency_scope='command_start_including_model_and_frontend_initialization'`。

## 4. 对 E05 探针设计的直接约束（已落实到 `prefix_probe.py`）

1. **不填充未来帧**：前缀只取 `keypoints[:N]`，`N = round(T·fraction)` 且 ≥2，
   `N ≤ T`；视频路径只解码前 N 帧（`in_memory clip` 或物理截断 mp4），不复制/不外推。
2. **明确标注“已观测前缀”**：报告不得把 prefix 改写率当在线 Partial 修订率，也不得
   把 `first_stable_prefix` 当 `first_stable_latency`（后者需真实流 + PTS + 墙钟）。
3. **归一化会随前缀变化**：`prepare_pose_window` 在每次调用时对**当前前缀**重算身体
   bbox/scale（`slbridge/uni_sign_preprocess.py:142-167,181-236`），所以同一前缀比例
   在不同长度下坐标不同；`prefix_probe.py` 每行记录 `body_scale`、
   `valid_body_fraction` 与输入 SHA256，报告必须说明这一点（与坑点 §6 一致）。
4. **100% 对照口径**：100% 前缀=官方 pkl 经 `start/end` 裁剪后的整段；与作者发布
   `test_tmp_pres.txt` 快照比对只作“历史一致性”参考，帧采样是确定性 `linspace`，
   不是上游 `random.sample`，不要求逐字相同。
5. **SLRT 不需要前缀探针**：它本身是滑窗状态机，未来帧需求已由窗口几何/投票规则
   明确（11 帧 lookahead + 推理时间）；其 Partial/Final 验收应在真实回放流中做
   （E05 矩阵中如实标 `NOT_RUN` 或引用既有证据，不伪造）。

## 5. 证据文件清单

| 证据 | 路径 |
|---|---|
| 上游在线 demo（整段读取+单次生成） | `models/external/uni_sign_source/demo/online_inference.py`（22,64-104,106-142） |
| 上游在线数据集（整段、random.sample） | `models/external/uni_sign_source/datasets.py`（606-623,634-665） |
| Uni-Sign 模型结构（对称时间卷积/双向 mT5/单次 generate） | `models/external/uni_sign_source/models.py`（72-140,206,260-295,312-323） |
| 对称时间卷积实现 | `models/external/uni_sign_source/stgcn_layers/stgcn_block.py`（66-88,121-127） |
| 本地视频入口（整段解码后推理、`streaming: false`） | `slbridge/uni_sign.py`（853-946,988,1005-1008,1029-1045） |
| 本地连续窗口（完整窗口重算、confirmed 启发式） | `slbridge/continuous.py`（10-24,63-126）；`slbridge/uni_sign.py`（151-204） |
| SLRT 窗口状态机（左7右8、future-frame 断言、EOS 才复制右边界） | `slbridge/slrt/stream.py`（24-29,66,82-91,129-159,161-169） |
| SLRT 投票/Partial/Final | `slbridge/slrt/decoder.py`（9-22,48-95） |
| SLRT CLI 墙钟口径 | `slbridge/slrt/cli.py`（128-131） |
| 坑点记忆（滑窗 vs 原生流式、bbox 重算、SLRT Online 边界） | `docs/agents/model-evaluation-pitfalls.md` §6、§10 |
