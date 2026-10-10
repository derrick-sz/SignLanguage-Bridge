# 自动动作分段

`slbridge.segmentation` 根据连续帧的真实时间戳观察手部活动，形成一个完整动作片段，再交给现有 `clip` 模型。每词或短句结束后持续暂停可触发一次片段提交。它没有为模型添加 `stream` 能力，也不能把活动边界、无手或低质量判断当作未知手语拒识。

## 接口

```python
from slbridge.segmentation import AutoSegmenter, MediaPipeActivityObserver, SegmentConfig

config = SegmentConfig()
segmenter = AutoSegmenter(config)
observer = MediaPipeActivityObserver(config)
try:
    # source.read() 返回未经镜像的 BGR FramePacket。
    while (packet := source.read()) is not None:
        observation = observer.observe(packet)
        for notice in segmenter.feed(packet, observation):
            if notice.kind == "ready":
                segment = notice.segment
                # 在所属 session 中排队，并保留来源、序号和会话代次。
                # recognizer.recognize_clip(segment.clip, sample_id=segment.segment_id)
            elif notice.kind == "rejected":
                # 展示明确的分段失败原因，不追加字幕。
                pass
    notices = segmenter.flush()
finally:
    observer.close()
```

`feed(packet, observation)` 返回 `tuple[SegmentNotice, ...]`；没有变化时返回空元组。通知的 `kind` 为 `state`、`ready` 或 `rejected`，并携带 `state` 和 `reason`。只有 `ready.segment` 含有 `AutoSegment`，字段为 `segment_id`、`clip`、`start_time`、`end_time`、`action_end_time`、`source_id` 和递增 `sequence`。

`start_time`、`end_time` 是含缓冲区的实际首末帧源时间；`action_end_time` 是最终有效结束候选的第一帧时间。输出 `CapturedClip` 的帧时间减去实际首帧时间，从零开始；源帧序号保持不变。结束确认等待期间超出后缓冲区的帧不会送入模型。

`reset()` 清除状态、缓冲和源时间轴；同一个分段器的完成序号仍然递增。片段 ID 使用实例 UUID 加序号，跨关闭再启用的新实例也唯一，且仅含适合 API 的 ASCII 字符；含空格、斜杠或 Unicode 的源路径单独保留在 `source_id`。更换来源后须调用 `reset()`，并重新创建活动观察器。`flush()` 在 EOF 拒绝尚未完成的开始候选、活动或结束候选；EOF 不构成暂停确认。观察器独立拥有一个 MediaPipe Holistic 图，`close()` 可重复调用；它不访问或重置 `live.prepare_clip` 和文件提取器的图。

## 状态与边界

| 状态 | 条件与结果 |
| --- | --- |
| `WAITING` | 无手、低质量或低位静止的手不启动；运动达阈值或有效静态姿态进入开始候选。 |
| `START_CANDIDATE` | 活动持续达到开始候选时间后进入捕获；中断则取消候选。 |
| `CAPTURING` | 连续保存完整动作；低运动进入结束候选。纯静态姿态会先保持至少最小动作时间。 |
| `END_CANDIDATE` | 手仍可见且质量合格的暂停持续达到结束候选时间和后缓冲时间后提交；恢复运动取消结束候选。 |
| `REARM` | 提交后保持同一静态动作不重复提交；持续释放姿态或重新运动可开始新动作。被拒绝的动作须先释放，防止把仍持续的超长动作切成多段。 |

质量不合格、手消失或动作中丢失曾观察到的一侧手会取消结束候选。短于质量间隙上限的中断可恢复，但须重新观察完整暂停；达到上限则 `quality_lost` 拒绝，即使达到上限的这一帧已经恢复。中断帧可留在恢复后的片段中，模型预处理仍需执行原有质量校验。断续时间戳超过观察间隙上限会 `timestamp_gap` 拒绝活动；不能把卡顿或丢帧期间当作可观察的连续暂停。

静态保持一次提交后，新运动或释放再做可生成新的 ID 和序号。分段器不接收识别文本，因而不会按文字去重：连续两次同词可以分别提交。会话层仍须按会话代次与片段 ID 丢弃重复、迟到或切换来源前的识别回调。

## 配置与存储上限

`SegmentConfig` 是冻结 dataclass，`config_version=1`，`to_dict()` 返回可序列化配置。

| 参数 | 默认值 | 单位 / 意义 |
| --- | --- | --- |
| `start_candidate_seconds` | 0.12 | 连续开始活动的秒数 |
| `end_candidate_seconds` | 0.65 | 连续有效暂停的秒数 |
| `pre_buffer_seconds` / `post_buffer_seconds` | 0.25 / 0.10 | 片段前后缓冲秒数，以实际可用帧为准 |
| `min_segment_seconds` / `max_segment_seconds` | 0.25 / 8.0 | 候选动作开始至结束候选首帧的秒数；不含缓冲及确认等待 |
| `quality_gap_seconds` | 0.20 | 可恢复质量中断上限 |
| `max_observation_gap_seconds` | 0.50 | 相邻真实观察的最大间隔 |
| `rearm_seconds` | 0.20 | 连续释放或新运动的秒数 |
| `start_motion_threshold` / `end_motion_threshold` / `rearm_motion_threshold` | 0.35 / 0.15 / 0.35 | 每秒肩宽归一化的手部平均位移 |
| `max_buffer_seconds` / `max_frames` / `max_bytes` | 10.0 / 600 / 134217728 | 环形缓冲的秒、帧、像素字节上限 |
| `max_width` | 640 | 内部保留图像与活动观察最大宽度；比例缩放，无镜像 |
| `pose_visibility` | 0.5 | 左右肩部锚点可见性阈值 |
| `static_active_y` / `static_active_x` | 0.85 / 1.8 | 以肩部中点、肩宽归一化后的腕部最大纵坐标和横向绝对值 |

等待与重新准备时旧帧可从环形缓冲淘汰；活动和候选期间必须保存完整动作，达到上限会返回 `buffer_time_limit`、`buffer_frame_limit` 或 `buffer_byte_limit`，清除本段并拒绝，不输出被截断的片段。超长动作返回 `segment_too_long`，太短动作返回 `segment_too_short`。高分辨率或高帧率可先达到字节/帧上限，8 秒默认最大动作时间不是容量承诺；例如 640×480 BGR 每帧约 0.88 MiB，应按部署的实际帧率配置容量或降低 `max_width`。

首尾时间、帧序号、同源和图像尺寸严格校验，不用帧计数推算动作时间、不合成时间戳。每帧图像复制并设为只读，源复用图像内存不会改写已缓冲片段。只保存前一帧的时间与尺寸元数据，不额外持有原始大图。

## 真实活动观察与验证范围

`MediaPipeActivityObserver.observe(packet)` 将 BGR 转为不镜像的 RGB，用真实 Holistic 左手、右手和左右肩部结果计算 `ActivityObservation(timestamp, hands_visible, quality_ok, motion, pose_active, left_hand_visible, right_hand_visible)`。手部每个点必须在图像内且坐标有限；肩部须通过可见性检查且肩宽有效。运动为两侧可对应手部的 21 个点的平均归一化位移除以真实观察间隔；坐标先校正图像宽高比，再相对肩部中点与肩宽归一化。第一帧及质量中断后的首个有效帧没有运动基线，运动为零。

静态 `pose_active` 使用有效腕部相对肩部的位置；所以低运动不自动等于空闲。一个可见的有效手也可启动单手手语；动作中一旦看到另一侧，则之后丢失该侧会被保守视为质量中断。合成观察可省略左右手字段，真实观察器总是提供两侧结果。

这些开始、结束、静态区域和质量阈值均是**尚未在标注手语视频上验证的启发式**。行为测试只验证状态机契约与时间、存储约束，不能证明真实手语边界准确率、字幕内容正确率或未知类拒识能力。默认结束确认等待约 0.65 秒的源时间，后面仍有提取、识别和队列延迟；实际延迟必须实测。上线前应以包含静态词、内部停顿、连续同词、双手遮挡、离开画面和长句的标注视频检查漏段、误切、重复和起止误差，再调参或替换为经过训练的活动观察模型。

## 本次验证证据

使用 `.venv-libras/Scripts/python.exe -m pytest tests/test_segmentation.py tests/test_sources.py tests/test_media.py tests/test_live.py -q`，结果为 **119 passed、1 skipped**。其中分段新增 42 项参数化行为测试；现有 opt-in 真实模型 parity 测试跳过，不计作通过。`sources.py`、`media.py`、`live.py` 未改变。

真实 MediaPipe 图处理 320×240 空画面的结果为：无左手、无右手、`hands_visible=False`、`quality_ok=False`、`motion=0`、`pose_active=False`。

实际冻结样本 `Libras-EQT-UECE (RGB)/105_Árvore/INFORMANTE 3/IMG_2971.mp4` 使用原始 VideoSource 时间轴、真实活动观察器和默认 `SegmentConfig`：93 帧，源首末跨度 3.0667 秒，处理壁钟 3.486 秒；手可见 50 帧、质量合格 48 帧、静态活动 48 帧。状态经过开始确认、捕获、暂停候选、活动恢复，再以 `quality_lost` 拒绝；**ready=0，不能记为自动字幕成功**。内部缓冲峰值 60 帧、41,472,000 字节、1.9667 秒。此单次壁钟测量仅供复现，不是吞吐量基准。

为寻找默认配置可完成的真实回放，又从当前 RF 12 类中的 8 类各选最短的、不超过 8 秒的视频，依次回放；下表路径均相对 `Libras-EQT-UECE (RGB)`，时间均为视频源秒数。没有修改阈值、模型或数据，也没有延长、复制或补帧。

| 真实视频 | 帧数 | 捕获确认时间 | 结束候选与中断 | 分段结果 |
| --- | --- | --- | --- | --- |
| `2_Sim/INFORMANTE 4/IMG_0506.mp4` | 53 | 0.1667 | 1.3333 暂停，1.3667 质量中断 | 1.5667 `quality_lost`；ready=0 |
| `7_Quero/INFORMANTE 2/IMG_0696.mp4` | 55 | 0.3000 | 1.5000 暂停，1.5667 恢复，1.6333 再次暂停 | EOF `incomplete_at_eof`；ready=0 |
| `92_Energia/INFORMANTE 4/IMG_3049.mp4` | 59 | 0.5667 | 0.6667、0.7667 暂停随后恢复；1.0667 暂停，1.1000 质量中断 | 1.3000 `quality_lost`；ready=0 |
| `93_Solicitar/INFORMANTE 4/IMG_3058.mp4` | 53 | 0.9333 | 未形成结束候选 | 1.4333 `quality_lost`；ready=0 |
| `94_Serviço/INFORMANTE 4/IMG_3061.mp4` | 52 | 0.9000 | 1.2333 暂停，1.2667 质量中断 | 1.4667 `quality_lost`；ready=0 |
| `100_Cair/INFORMANTE 5/IMG_4217.mp4` | 61 | 0.7333 | 未形成结束候选 | EOF `incomplete_at_eof`；ready=0 |
| `106_Poste/INFORMANTE 4/IMG_3131.mp4` | 55 | 0.7667 | 未形成结束候选 | 1.4000 `quality_lost`；ready=0 |
| `110_Menos/INFORMANTE 4/IMG_3155.mp4` | 49 | 0.7333 | 未形成结束候选 | 1.0333 `quality_lost`；ready=0 |

上述 9 个真实回放都没有产生可提交的完整片段，因此没有执行片段 RF 推理，**尚无默认配置下真实视频自动分段→RF接受→追加字幕的成功证据**。这些短视频的质量中断或 EOF 不能强行当作动作结束；补帧、弱化质量判断或为某个样本调整阈值也不能算同一默认配置的验收。后续需要符合结束确认协议的真实录制，或通过有边界标注的视频完成参数验证。
