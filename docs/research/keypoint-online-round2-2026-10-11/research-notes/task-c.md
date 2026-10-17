---
task_id: c
role: Repository and streaming-input protocol auditor
objective: Identify input, quality, timing, and event-contract gaps before a native keypoint route
status: complete
confidence: high for read-only code facts; medium for proposed protocol
sources_found: local code and existing audit artifacts
acceptance_met: yes
---

# 线程 C：在线关键点协议与当前工程差距

日期：2026-10-11。只读核查主工作区 `HEAD=de316f52bc618387e0536daeee80899b947fd5bd`，另只读训练工作区。未修改代码、未提取新数据、未训练、未启动摄像头、未运行验收测试。以下“拟议”接口与实验均未实现。既有审计数字来自读取其产物，未在本轮重新扫描全部原数据。

## 1. 新的、足以影响方案选择的事实

### C1：同一原始 graph 输出有缓存机会，当前输出却不足以复用

`MediaPipeActivityObserver` 持有 `_graph`，首次按 `static_image_mode=False/smooth_landmarks=True` 打开，此后每个收到的 packet 调用一次 `_graph.process(rgb)`。它在函数内部已取得 `result`，包括肩部与双手原始坐标，但返回值只有 `ActivityObservation(timestamp,hands_visible,quality_ok,motion,pose_active,left/right_visible)`；内部仅留上一帧手部二维坐标和帧身份。没有坐标序列出口，也没有实际识别输出。

证据：[segmentation.py:73](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:73)、[segmentation.py:397](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:397)、[segmentation.py:412](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:412)、[segmentation.py:436](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:436)、[segmentation.py:480](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:480)。短支持片段：`result = self._graph.process(rgb)`，随后 `return ActivityObservation(...)`。

**分析**：最佳复用位置是同一次 process 后生成不可变 landmark packet，并由它派生活动观察和时序输入，不是把 observer 现有布尔输出还原为关节。识别缓存新增成本应分别测量；不能把省去第二次提取自动写成 FPS 或端到端收益。

### C2：活动质量与分类质量不相同，缓存前必须明确取哪一种

活动观察：某只手必须 21 点全部有限且 xy 均在图内才算 visible；如果某只已出现的手有一点无效，即 `invalid_hand=True`，整帧 `quality=False`。同时需双肩 visibility≥0.5、肩距有效。运动只在同手前后均有效且时间差≤0.5 秒时计算，取每点位移均值除以真实时间差；无共同手或超间隔时运动保留 0，此 0 不证明静止。

分类归一化：双肩无效时整帧为 -999；否则逐点写入有效位置，无效点继续 -999；一只手只要至少一个点有效就 `observed[side]=True`。它的 hand-frame fraction 因而并非活动 observer 的完整手可见比例。

证据：[segmentation.py:439](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:439)、[segmentation.py:459](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:459)、[segmentation.py:468](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:468)、[media.py:228](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:228)、[media.py:263](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:263)。支持片段：observer 为 `len(points) == 21 and all(...)`；分类为 `bool((values[offset:offset + 21, 0] != -999).any())`。

**分析**：新 packet 应保存逐点观察 mask、肩锚质量和“差分是否有效”，允许分类/边界各自判定；不要将缺失、时间断档或 0 motion 合并成 background。当前代码不读取手部逐点 probability，不能替每个手点编造 0～1 confidence。pose visibility 如需导出应保留其实际来源及适用点集。

### C3：225 是二维身体与双手，z=0 是已观察坐标，不是缺失

点序为 pose33 + anatomical left hand21 + anatomical right hand21。归一化先将 x 乘图像宽高比，再减双肩中点、除肩距；写入 `(xy[0],xy[1],0.0)`。脸部不在导出特征里。缺失点三坐标全 -999。`prepare_array` 由 `values != -999` 构造逐坐标 mask，再将缺失值置 0；因此有效点的 z=0 mask=True，缺失点的数值0 mask=False。合法的 x=0/y=0 同样不是缺失。

证据：[media.py:224](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:224)、[media.py:243](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:243)、[media.py:251](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:251)、[features.py:45](/E:/githubproject/SignLanguage-Bridge/slbridge/features.py:45)、[features.py:131](/E:/githubproject/SignLanguage-Bridge/slbridge/features.py:131)。支持片段：`values[index] = (xy[0], xy[1], 0.0)`。

送入提取器与模型的图像不镜像；桌面镜像 checkbox 仅 `_render_preview` 翻转显示。不得为了匹配示范模板随意交换左右手或翻转 x，把镜像预览当成特征增强。是否按主手规范化是另一个需手语专家审核及消融的协议。

证据：[media.py:45](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:45)、[desktop.py:597](/E:/githubproject/SignLanguage-Bridge/slbridge/desktop.py:597)。

### C4：丢帧、时间和预检已改变实时输入；“每帧”不等于摄像头每一帧

`FramePacket` 仅 image、timestamp、frame_index、source_id。CameraSource 在 `VideoCapture.read()` 返回图像后取 `perf_counter()`，记录的是应用获得帧的时间，不是曝光时间。frame_index 是应用读取序号；不能据此审计摄像头驱动内部未交付的帧。

VideoSource 在 open 时完整预解码全部帧检查时间和图形，再 rewind/reopen；有效 POS_MSEC 优先，无可靠媒体时间才用帧索引/FPS。这个完整预检发生在发出第一个 packet 之前。文件回放随后按媒体时间节奏 sleep，pause 时调整墙钟起点。

证据：[sources.py:107](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:107)、[sources.py:157](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:157)、[sources.py:183](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:183)、[sources.py:226](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:226)、[session.py:226](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:226)。

观察 queue 最大为 2，满时弹出最旧任务再插入最新任务。实际 observer 接收的是依负载而稀疏的时间序列。AutoSegmenter 只在 observer 完成后 feed 这个 packet，因此其动作 RGB 缓冲亦是被观察的帧，不是所有 camera captures。现有 dropped_frames 是此队列弃帧计数；输出不含逐帧 queue-age、观察器耗时、遗漏间隔序列。

证据：[session.py:58](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:58)、[session.py:280](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:280)、[session.py:336](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:336)、[session.py:343](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:343)。

**分析**：秒速度必须用有效点且实际 dt；模型若需要固定采样节奏，应明确选择 drop、过去保持加未观察标志、或等待新帧，任何“填帧”不能冒充新观察。固定 30 帧完整动作是相对动作相位，不是固定 Hz 摄像头输入。回放质量实验可以复用原时间，但首次 open 耗时应单列；回放帧龄要建立媒体→单调墙钟映射，不能直接拿两种 clock 相减。

### C5：当前训练输入显式声明 reset + 全段抽样，持续结果不能直接注入 RF

`ExtractionConfig.frames=30`。离线 decode_video 检查完整视频之后按整段索引均匀选至多 30 帧，Holistic 在每个 clip 开始 reset，只对选中图像 process。captured clip 也是已完成段均匀选至多 30 图、开新 graph、reset，然后提取。PreparedInput 再用首末时间线性重采样到 target_frames。未来右端点参与插值，整段首末位置定义来自已完成 clip，因此这不是过去数据可立即复用的因果特征管线。RF 最后将坐标与 mask 整段 flatten，并不是状态模型。

证据：[media.py:29](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:29)、[media.py:134](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:134)、[media.py:161](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:161)、[media.py:299](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:299)、[live.py:59](/E:/githubproject/SignLanguage-Bridge/slbridge/live.py:59)、[features.py:147](/E:/githubproject/SignLanguage-Bridge/slbridge/features.py:147)、[features.py:192](/E:/githubproject/SignLanguage-Bridge/slbridge/features.py:192)。

旧 extractor identity 记录 `temporal_policy=uniform_selected_frames_after_full_decode` 与 `tracking_reset=before_each_clip`，还记录 MediaPipe 0.10.14、资源 hash、media.py hash、坐标点序。`.npz` loader 严格匹配 metadata 字段和 extractor hash。新持续跟踪全帧缓存即使 width=225 也不能标为同一 `mediapipe225_rgb_v1` 提取协议，以躲过这些校验。

证据：[media.py:91](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:91)、[media.py:120](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:120)、[features.py:86](/E:/githubproject/SignLanguage-Bridge/slbridge/features.py:86)、[features.py:104](/E:/githubproject/SignLanguage-Bridge/slbridge/features.py:104)。

### C6：已有字幕保证顺序与旧结果隔离，可复用；没有逐帧候选修订语义

AutoSegment 在完成时保留全局源时间 start/end/action_end，但 CapturedClip 的每帧 timestamp 减去片段首帧 origin。字幕 reserve 保存源时间 start/end，因此在线 token/window 边界若来自 clip 局部时间，必须加显式 origin 映射，不能把局部 0 作为全局 0。

`CaptionLedger` 用 session/source_epoch/model_generation/request/segment 绑定，pending 完成后按 reserve 的动作顺序 committed。相同文本的不同动作有不同 segment ID，不应按文本去重。它只接纳一次 pending completion，没有 revision/upsert partial 文本或边界更新；当前 SessionEvent candidate 是 AutoSegmenter 状态提示，不是 recognition partial。

证据：[segmentation.py:258](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:258)、[caption_events.py:48](/E:/githubproject/SignLanguage-Bridge/slbridge/caption_events.py:48)、[caption_events.py:80](/E:/githubproject/SignLanguage-Bridge/slbridge/caption_events.py:80)、[caption_events.py:116](/E:/githubproject/SignLanguage-Bridge/slbridge/caption_events.py:116)、[session.py:347](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:347)。

Capabilities/InferenceRequest 可声明字符串 mode=`stream`，但 ModelAdapter 只有 warmup/infer/close，输入种类只有 file/captured_clip/prepared；ModelManager 实际只有这些 infer 方法，没有会话 begin/feed/end/reset 状态生命周期。字符串允许值是扩展插槽，不是 native stream 已实现证明。

证据：[model_adapters.py:148](/E:/githubproject/SignLanguage-Bridge/slbridge/model_adapters.py:148)、[model_adapters.py:177](/E:/githubproject/SignLanguage-Bridge/slbridge/model_adapters.py:177)、[runtime.py:356](/E:/githubproject/SignLanguage-Bridge/slbridge/runtime.py:356)。

### C7：现有 total_ms 无法支持“动作到字幕”延迟结论

ModelManager total_ms 在拿到推理 lease 后开始计时，涵盖 adapter 调用；queue_ms 在自动已完成段排队后另计算。活动提取、起止判定、默认末端停顿 0.65 秒、字幕显示均不在这里。observer 另设 gap/质量门限，自动结束等待使用 max(end_candidate_seconds,post_buffer_seconds)。现有自动模型 pending 满 1 段会丢新段；stream 模式亦需各层背压行为，不能只测平均 per-frame model_ms。

证据：[runtime.py:359](/E:/githubproject/SignLanguage-Bridge/slbridge/runtime.py:359)、[session.py:369](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:369)、[session.py:401](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:401)、[segmentation.py:22](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:22)、[segmentation.py:369](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:369)。

## 2. 新数据适配范围与缺口

既有全量审计产物 `artifacts/libras-updated-data-audit-20261010/landmark-audit.json` 的 `full_numeric_stream_scan` 确认：hand 5,347 文件 587,575 帧，137,355 帧整个126块零；full 5,335 文件与 hand 尾部192:318逐值相同。进一步的手部质量统计：741 文件至少一半帧整个126块零，P95 文件零行比例约0.605。全零不是已审核的“空闲”标签。

审计文档明确 raw NPY 无额外 timestamps/提取器信息，全局签者身份未核验、dev/test 有12组已确认相同PNG序列。当前 318 与225维的点序、归一化、z、时间协议不同。不能用 shape 或近似人脸/手点数代替实际映射。

证据：[新增特征审计:18](/E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:18)、[新增特征审计:22](/E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:22)、[新增特征审计:26](/E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:26)、[新增特征审计:36](/E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:36)。

训练工作区的 `libras_full318_blockmask321_v1` 为318训练折标准化特征+3块 observed-if-any-nonzero 标志，分块0:192/192:255/255:318；非逐点mask。该代码明确 `global_signer_id=None`、heldout signer evaluation=False。`temporal.py` 左侧 padding 是因果卷积，但分类结果对整个有效 T 求平均，只有 full forward，无逐层 cache/feed。因此现有新数据分类器是离线对照，不是此次225流模板的直接模型。

证据：[训练数据:31](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:31)、[训练数据:112](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:112)、[训练数据:158](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:158)、[训练数据:586](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:586)、[TCN:141](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/temporal.py:141)。

**限制**：本轮未读取上游提供的实际逐列 schema，也未运行 MediaPipe 结果 A/B；不能判断持续跟踪变化大小或哪种质量阈值最好。CNN/MediaPipe内部处理相邻帧时间的实际契约另需官方核查。没有真实长流 background/OOV/边界标签，没有新签者 ground truth。

## 3. 最小新协议建议（设计，未实现）

复用 normalize_landmarks 的平面坐标方法，新增明确持续提取版本，建议名如 `pose_hands225_xy_stream_v1`；名称只是草案，注册和模型数据生成需实际实现。schema 至少记录：

| 字段组 | 建议字段与语义 |
|---|---|
| 状态身份 | session_id、source_id、source_epoch、tracking_epoch、model_generation、schema_id/extractor_hash、chunk_sequence |
| 时间 | source_timestamp及单位/时钟域、capture_received_monotonic、extract_start/end、frame_index、dt、frames_skipped_since_last、timestamp_origin与媒体→墙钟映射 |
| 几何 | values[75,3]或明确xy子集、point_observed[75]、shoulder_anchor_valid、点序/anatomical handedness、source/extraction尺寸、mirror_applied=false；z_zero不能宣称3D |
| 质量 | hand_present、每手有效点比例、实际pose visibility及其适用点集、motion_valid；没有数值confidence则留空 |
| 来源 | media_hash、raw_frame_range、extractor/package/model/options/codehash、sampling_policy、tracking/reset_policy、label_manifest_hash |

所有资源 hash 只冻结最终实际使用者，不以冗余多环境版本替代当前图。除了现 media.py，持续 wrapper/归一化/采样/质量代码及 tracker 生命周期也应纳入 identity。

`begin(config,binding)` 建独占流状态；`feed(packet/chunk)` 只接受本binding且严格递增时间；`end(reason)` 明确结束类型，不把 EOF 自动变成完整动作；`reset(reason,new_epoch)` 同时清提取跟踪历史、dt前帧、DTW行或卷积/GRU历史、端点、候选修订。source/shape/schema/模型切换、长时间gap均须明确reset/拒绝规则。短缺失可以保留过去模型状态，但需要训练/验证相同策略；不能把0数值当新观察。

输出另设 `candidate_id,revision,status=partial|committed|retracted,source_time_range,evidence_end_time,label_candidates,score_type,reason`。partial 仅当前可撤回候选；动作身份保留到提交/撤回。跨窗同动作合并依据候选身份和时间证据；同词连做应产生两个动作。重用 CaptionLedger 的 binding、序号、最终commit与历史，但增加真实partial协议之后才向UI显示它。标注 `model_score` 未校准时不写词义概率。

## 4. 可直接执行的验证顺序（尚未执行）

### E0：拆开跟踪、采样、处理速度三个变化

冻结相同原视频、分段边界、图像尺寸、模型版本、同一normalize函数，并分别生成：A旧“整段抽30+reset”；B处理同样抽30图但不reset且带同一真实前缀，用来隔离跟踪初始条件；C同动作全帧+reset、之后仅从已生成坐标取A的索引，用来隔离处理采样密度；D完整连续流全帧persistent；E按实际 queue/drop 轨迹重放persistent。A～E 均保留实际观察mask和帧索引，不能只比较最终经插值的数值。

比较对应索引共有有效点的坐标误差、观察mask/anchor分歧、前几帧检测质量、旧RF结果改变，以及预处理/排队成本。旧RF用于监测域变化；不能以E旧RF可运行就宣称免重训。B/D 用相同前缀和重启种子，避免把不同场景初始化混为跟踪差异。每种提取产物标自己的协议。

### E1：从原RGB定义新模板集，不转换318

选现有已审核小词表的一部分原RGB加新采示范，每段原始hash+原标签+语言+人工起止+全局参与者ID/确认状态。按D新协议全帧提取，保留前后背景、缺失段和原时间；为每类保存多个训练源模板。分词条/固定短语，不能把自然文本列表切词冒充gloss。raw318可另作离线对照；本路线模板与训练不依赖其未冻结逐列定义。

训练/验证/测试按源内容组拆分，模板只取train；同视频裁剪、重复、增强、同场景拼接派生必须同组。未知签者身份就标 unknown 并仅报告内容组独立，新签者目标另采有真实global ID的人。需未知手语动作、相近但不同词、日常抓头/挥手/拿物、遮挡、双手交叉、同词重复、动作中长停顿、连续不回静止、速度变化测试。把孤立片段拼成流只能验证软件与受控输入，不能替代自然共构/协同发音流验收。

### E2：真实端到端计时及错误恢复

记录每个源/到达/提取/推理/候选/提交/UI显示时点；分别报告提取与模型P50/P95、处理帧龄、丢帧比例/最大gap、动作结束到首次候选和最终字幕、未知误接收/重复漏检/字幕抖动、长流内存。camera只拿到读返回时点，帧龄标为 application-receipt age；要总曝光→显示延迟须额外可核验的视频/外部计时，不假称已有硬件时点。

TCN/GRU 增量一致性应在冻结新协议后比较相同前缀批计算与多种chunk分割；追加未来帧不改变旧逐帧logits，并检验reset隔离；这些软件检查通过仍不证明词义与低延迟实机成功。DTW/模型/端点阈值仅validation定，test固定后测。当前模型推理total_ms和0.65s pause默认都可作为明确的旧系统对照项。

## 5. 决策建议

近期最小研发对象应是“版本化持续关键点录制/回放 + 多模板匹配 + 可撤回候选/稳定提交”的可对照闭环。先验证同一实际坐标协议能在新签者及背景/OOV中工作；之后再从同一冻结流数据比较DTW和增量TCN/GRU。持续跟踪缓存确实可复用一次提取结果，但属于输入域与状态生命周期变化，直接改旧RF提取路径会混入三个变化并破坏原manifest承诺。
