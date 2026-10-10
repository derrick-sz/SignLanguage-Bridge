# 统一任务接口与字幕事件

HTTP v3 使用同一个推理入口承载 `classification`、`gloss_sequence` 和 `text`。具体输入表示、预处理版本、词表、运行设备及 `clip`/`stream` 能力由注册适配器声明。接口不会把旧的 225 维输入自动转成 321 维，也不会把 CTC gloss 序列当成自然语言翻译。

## 服务启动与语言路由

模型目录保持原来的严格 v1 JSON 格式；路径只由服务器配置。单模型运行兼容原命令：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.api_cli --catalog configs/runtime-models.example.json --model libras-onnx --device cpu
```

同时提供两种语言时，为每种语言配置一个独立 `ModelManager`。示例中的别名须存在于实际 catalog，语言名称须与模型能力声明完全一致：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.api_cli --catalog path/to/catalog.json --route libras=libras-onnx --route csl=csl-tfnet --device cpu --max-caption-sessions 64 --inference-timeout-seconds 120
```

`--route` 可以重复；同一种语言不能重复，且不能与 `--model` 同用。每个语言管理器拥有自己的推理容量、generation、切换与回退状态；一个语言正在推理时，另一种语言可独立接收请求。底层 GPU 是否实际并行由运行库决定。Python 嵌入方式可向 `create_app(..., router=router)` 传入已经注册管理器的 `ModelRouter`；应用 lifespan 负责关闭它们。服务仍使用单 worker；非 loopback 绑定要求配置 bearer token。

## 能力协商与推理

| 接口 | 说明 |
| --- | --- |
| `GET /v3/health` | 能力、各语言状态、字幕会话容量 |
| `GET /v3/models` | 服务 catalog 别名、已加载模型能力和 generation |
| `POST /v3/models/negotiate` | JSON 声明 language、representation、task、model、preprocessing、generation、mode 并校验匹配 |
| `POST /v3/models/switch` | JSON `sign_language`、catalog `model_id` 和可选运行参数；预热且检查语言后激活 |
| `POST /v3/models/rollback?sign_language=libras` | 只回退指定语言；generation 递增 |
| `POST /v3/recognitions` | 单个 multipart 文件和可选声明，调用实际注册适配器 |

能力包括 `model_id`、`model_version`、`sign_language`、`task_type`、`inputs`、`vocabulary`、`runtime`、`devices`、`modes` 和 `score_type`。`inputs` 的每项声明 `representation_id`、shape、dtype、`preprocessing_version` 和 `input_kinds`。`model_id` 是运行模型的实际 ID；切换接口的 `model_id` 是 catalog 别名，两者可能不同。

推理支持 `.mp4/.avi/.mov/.mkv/.npy/.npz`；文件的真实可用性继续由模型输入契约及其解析器校验。multipart 字段包括：

| 字段 | 约束 |
| --- | --- |
| `file` | 必填，唯一文件；受上传大小限制 |
| `request_id` | 可省略生成 UUID；1–128 字符标识，允许字母数字与 `_.:-` |
| `sign_language` | 路由服务必填，且与所选模型一致 |
| `representation_id` | 任意已注册表示；须与模型能力一致 |
| `task_type` | `classification`、`gloss_sequence` 或 `text` |
| `model_id`、`preprocessing_version` | 可选精确绑定 |
| `expected_generation` | 可选正整数；字幕请求必填 |
| `mode` | 默认 `clip`；模型须声明该能力 |
| `timeout_ms` | 可选 1–3600000 毫秒，实际期限不超过服务配置 |

未知字段、重复 multipart 字段和上传模型权重均被拒绝。预检查与实际推理租赁都会验证绑定；模型 generation 或 ID 已改变返回 HTTP 409 `model_generation_conflict`，不执行识别。能力、任务、语言或预处理不匹配返回 422。

响应 `schema_version=3`；保留 `request_id`、`model_generation`、`result`、`runtime`、`timing_ms`，新增实际 `model_id`、`model_version`、`preprocessing_version` 和会话/片段上下文。`result.task_type` 判别严格类型的 `prediction`：

- 分类：`label_id`、`score`、完整 `scores`、`display_text`。
- Gloss 序列：有序 `tokens`（包括重复 token）及 `decoder`；未知 token 时间保留 null。
- 文本：`text`、`language`、可空 `score`。

质量/阈值拒识属于 HTTP 200 `result.status=rejected`，不是服务异常。HTTP 错误保留原有 `schema_version=1`、`request_id`、`error.code/message` 的错误 envelope。

`mode=stream` 只表示适配器接受该次调用模式；本接口仍上传一个完整文件，不提供帧级实时连接。现有 RF/ONNX/TFNet 默认仅声明 `clip`。单模型服务器继续支持 `/v1/recognitions` 分类和 `/v2/recognitions` 序列，其响应外层保持原版本字段。路由服务器也支持这两个推理端点，但须声明语言；模型管理请使用 v3。

## 稳定字幕会话

显式创建会话后才可提交字幕片段：

```json
{"session_id":"conversation-1","sign_language":"libras","source_epoch":0}
```

发送到 `POST /v3/caption-sessions`，成功返回 201 和 `snapshot`；同一当前绑定的创建重试返回 200。`session_id` 可省略生成 UUID。`snapshot` 包含会话 ID、source epoch、model generation、revision、`last_sequence`、已提交 `events` 和未提交 `candidates`。

字幕推理 multipart 必须额外提供 `session_id`、`segment_id`、递增 `sequence`（从 1 开始）、`source_epoch`、`expected_generation`、`start_time`、`end_time`、`source_id` 和 `sign_language`。开始/结束时间必须是有限、非负、顺序正确的源时间。摄像头重开、视频 seek 或换源时，客户端提升 source epoch 后再提交新片段。

```powershell
curl.exe -X POST http://127.0.0.1:8765/v3/recognitions -F "file=@clip.mp4" -F "request_id=recognition-1" -F "session_id=conversation-1" -F "segment_id=segment-1" -F "sequence=1" -F "source_epoch=0" -F "expected_generation=1" -F "start_time=0" -F "end_time=2.4" -F "source_id=camera-front" -F "sign_language=libras"
```

响应新增 `caption_event_id` 和 `caption_committed`；会话 `GET /v3/caption-sessions/{id}` 是当前历史的权威读回。事件身份由会话与片段确定，包含源时间、模型/预处理版本、原始识别结果、字幕、运行信息和耗时。相同文本的两个独立片段形成两个事件。提交顺序由 sequence 确定；拒识、超时或取消的片段不会进入成功历史。

每个会话缓存请求 ID 与文件 SHA256、全部声明和片段元数据。相同 ID、相同数据重试返回同一个响应和事件且不再次推理；正在执行的相同请求返回 409 `request_inflight`。ID 相同而数据不同返回 409 `request_conflict`。字幕撤回或清空后重试旧请求不会重新插入事件。普通无会话推理不会缓存请求 ID。

会话操作发送到 `POST /v3/caption-sessions/{id}/actions`：

```json
{"action_id":"undo-1","action":"undo"}
```

| action | 效果 |
| --- | --- |
| `undo` | 撤回最后一个已提交事件 |
| `clear` | 取消候选并清空成功历史 |
| `cancel` | 取消当前全部候选，保留历史 |
| `invalidate_source` | 要求额外的、严格递增 `source_epoch`；取消旧候选并绑定新源 |
| `close` | 关闭会话，取消候选；保留快照供读回 |

同一 action ID 与相同操作的重试返回原响应；重用 ID 执行另一操作返回 409。`DELETE /v3/caption-sessions/{id}` 关闭并删除会话、释放容量；重复删除仍成功。删除后重新创建同一个 ID 是新的会话生命周期。

会话数量默认上限 64，请求去重记录和 action 记录默认分别上限 1024，成功历史默认上限 200，候选默认上限 2。到达容量返回 429，不默默淘汰去重 ID；客户端可以关闭、删除并创建新会话。会话只保存在当前服务进程内，重启后需重新创建。独立字幕场景应分别使用不同会话。

## 截止时间与取消

超时立即返回 504 `inference_timeout`；会话取消立即使等待请求返回 409 `request_cancelled`。原生推理不能安全强制中断：实际工作、模型租赁与临时文件保留至原生调用返回，然后释放并丢弃迟到结果。此期间该语言仍可能返回 409 `runtime_busy`。应用关闭会取消字幕候选并等待已启动工作清理。

服务临时路径和 CUDA DLL 目录不作为公开响应字段返回。保留既有 bearer 验证、ASGI 流式 body 限额、multipart 数量/字段限额和临时文件清理。

边界回归可运行 `.venv-libras/Scripts/python.exe -m pytest tests/test_unified_api.py tests/test_api.py tests/test_sequence_api.py -q`。新测试使用确定性的类型适配器验证 HTTP、路由、去重、取消和时限；这些测试不证明真实视频识别质量。真实模型视频回放必须单独记录模型、输入和响应读回证据。
