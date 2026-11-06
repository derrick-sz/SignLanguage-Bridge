# 三维手语动画集成：当前工程入口审计

日期：2026-10-28（Asia/Taipei）。范围：`E:\githubproject\SignLanguage-Bridge` 当前代码与两个现有 Python 环境的模块可见性。只读审计；没有新增运行时代码、安装依赖、启动模型或训练。本文支持选型，不代表集成完成。

## 1. 工程结论

**首期优先采用“独立 Three.js 动画面板 + 当前 FastAPI”的接口集成。桌面应用增加打开面板与发送已确认输入的入口；QtWebEngine 嵌入可作为后续壳层，不建议先重写 Qt/OpenGL 动画引擎。**

原因是现有识别结果、任务状态与证据字段可以复用，而本项目尚没有三维播放器、静态前端或动画服务。将 SignBridge 的渲染与资产处理作为独立模块接入，比把浏览器扩展完整搬入 Qt 更容易保持原识别链稳定。

## 2. 已有接口与拟新增边界

| 当前事实 | 可复用部分 | 仍须新增 |
|---|---|---|
| `create_app` 可按模型目录或语言路由创建 FastAPI；路径属于服务端配置，客户端不提交任意文件系统路径。[1] | 原模型管理、上传限制、鉴权、错误映射 | 动作注册表、规划/确认接口、动画资源白名单服务与前端静态页面。 |
| `/v3/recognitions` 返回统一结果，支持 session、segment、source epoch、model generation。[2] | 文本与 Gloss 输入、来源与版本绑定 | 把用户明确选中的输入转换为独立动作计划；识别成功不能直接自动批准动作。 |
| `/v1/videos/infer` 和 `/v1/jobs/{job_id}` 已有异步任务入口，结果轮询可获得终态。[3] | 文件识别进度、失败处理、证据 ID | 前端上传/结果确认 UI，以及与播放计划的关联。当前没有 WebSocket/SSE 传输实现。 |
| `/v3/caption-sessions`、`actions` 有取消、撤销、清空、来源失效和关闭。[4] | 生命周期与幂等 action ID 的设计 | **人工确认是新增操作**，当前没有 confirm/play 动作。 |
| Qt 桌面已有文件/摄像头输入、结果区、候选列表与导出。[5] | 当前输入、模型选择与候选 UI | 明确的“确认并播放”操作与动画窗口入口。 |

本次搜索 `slbridge` 未找到 `StaticFiles`、`HTMLResponse`、`FileResponse`、`CORSMiddleware`、`QWebEngineView` 或已集成三维 UI。此项是当前应用源码搜索结果，不代表安装这些库或复制 HTML 后就已经接通产品。

### 三条集成路线

| 方案 | 工程工作 | 适用性与限制 |
|---|---|---|
| 浏览器面板 + FastAPI（建议首期） | 新增动画资源/计划 API 和同源静态面板；保留 Three.js 渲染 | 可以直接复用原播放器、GLB、动作资源及浏览器调试。桌面第一版可以只加打开入口。需要补选定模型环境中的 HTTP 依赖。 |
| QtWebEngine 嵌入 Three.js | 在上一方案基础上增加 WebEngine 宿主、启动/关闭生命周期、消息或 HTTP 连接 | 用户体验可统一在现有桌面窗口内；当前两个环境都缺 WebEngine 模块，须另行解决依赖与打包、GPU/浏览器进程生命周期。 |
| Qt/OpenGL 原生重写 | 重建资产加载、蒙皮、动画混合、材质、相机、骨架映射与同步 | QtOpenGLWidgets 模块存在，但不能直接复用 Three.js 播放器实现；代码和验收负担最大，不适合先证明业务闭环。 |

三者都可以最终提供同样的动作选择/播放功能，画面效果主要取决于角色资产、手部/面部驱动、重定向与渲染调校，不能仅凭“原生 Qt”预期质量提升。

## 3. 环境实际探测

通过 `importlib.util.find_spec` 检查模块可见性；**没有实例化 Qt 窗口或运行 API，因此这不是 GPU、页面或模型加载验收**。

| 环境 | QtWidgets | QtOpenGLWidgets | QtWebEngineWidgets | FastAPI / Uvicorn |
|---|---|---|---|---|
| `.conda-env/python.exe` | 有 | 有 | **无** | **无 / 无** |
| `.venv-libras/Scripts/python.exe` | 有 | 有 | **无** | 有 / 有 |

依赖清单只明确 `PySide6-Essentials==6.8.3`，API 是独立 extra。[6] 因此当前不能宣称“将 Three.js 嵌进去即可运行”，也不能因为 `.venv-libras` 能启动 HTTP 就推断它具备 Uni-Sign/SLRT 的全部模型依赖。

若首期选浏览器方案，推荐在现有已验证的目标模型环境中增补明确版本的 API extra，并做依赖/模型回归。若为了隔离新增单独动画静态服务，应显式处理同源代理或窄范围 CORS；不要用 `file://` 页面直接读取 G 盘私有资源作为产品入口。具体依赖安装不在本轮执行。

## 4. 人工确认与原结果保护

现有 `CandidateLedger` 的每个条目明确为 `status='unconfirmed'`，快照声明重叠窗口、句界未知、分数未校准及 `automatic_translation=False`。[7]

另一个容易误读的点是 `CaptionLedger`：`completed → committed` 是按请求顺序保存识别结果的程序状态，**不是人类批准其内容**。[8] `/v1/jobs` 的 `subtitle.final` 同样表示推理任务终态，不能拿这个 FINAL 当作“可自动播放的准确 CSL”。[3]

首期可采用如下产品流程（均为拟新增）：

1. 识别结果继续显示原文本或原 Gloss，并保留来源。
2. 用户选择“确认并播放”；若允许编辑，另存编辑后的输入，保留原识别证据。
3. 规划器优先匹配完整短语和明确审核范围的动作资产，返回匹配依据、缺项与语言状态。
4. 未匹配/歧义或必要否定、数字、指代无法表达时，显示需复核，不静默删除成分后拼播。
5. 播放器根据确定的资产版本播放，不修改原识别结果、字幕证据或冻结实验。

直接键入文本、从候选选择输入、回放历史结果都需要明确输入来源；按原始显示文本去重会错误合并不同动作，沿用现有 event/request identity 的思路更合适。

## 5. 会话、迟到结果与取消

可复用现有约束：

- `InferenceRequest/Response` 已携带 `session_id`、`segment_id`、`source_epoch`、`expected_generation/model_generation` 与 request ID。[9]
- `RecognitionSession` 根据 request/epoch 忽略迟到结果，连续模式还检查窗口年龄、观察间隔等。[10]
- `CaptionLedger.bind` 在来源/模型变化时取消 pending；完成时再次比对 session/request/epoch/generation。[8]
- API 的 `invalidate_source` 要求 source epoch 增加；clear/close/cancel 会取消 pending。[4]

新增动作计划应保存上述**来源关联**，并另外拥有 `plan_id`、输入修订号、`playback_generation`、目标手语/变体、具体 motion/rig 版本与哈希。源 epoch 不是动画播放修订号；明确回放历史结果时应保留历史来源，在当前播放请求中重新绑定操作身份。

必须验证以下情况：

1. 用户修改文本或换动作后，前一规划/资源加载的迟到回调不应重新启动旧动作。
2. 切来源、切语言、关闭会话时停止旧播放/排队；新页面或新动作不能继承旧姿态队列。
3. 同一次确认重试不会重复排队；不同操作即使文字相同仍可分别播放。
4. 暂停/停止动画与取消模型推理是两个操作。当前 `JobRunner.cancel` 是内部方法，[11] 不能据此宣称已有公开的 job-cancel API。
5. 当前 Qt `cancel()` 调用 session invalidation；动画模块需明确订阅对应停止事件，不能假定识别取消会自动关闭 Three.js mixer。[5][10]

这些是开发验收建议，不是已存在的动画状态机。

## 6. 文件路径、鉴权和资产许可边界

原 API 设计将模型路径限制为服务端目录配置。[1] 动画资产应保持同样边界：客户端提交逻辑资产 ID，由服务端注册表映射到已验证的资产内容；不要让浏览器提交 G/E 盘任意路径，再直接用文件响应返回。

已有归档可以在受控只读导入/缓存目录中解包，验证原清单和哈希，保留原始出处；导入不等于语言批准。对 G 盘视频，只在确实需要对照或重建时按注册来源访问，不暴露整个归档目录。

`api_cli.py` 默认绑定 `127.0.0.1`；非回环地址必须提供 bearer token。[12] 鉴权中间件在 HTTP 请求层处理 Authorization。[13] 如果同一 app 新增受保护静态页面，需要设计页面启动与静态资源访问方式，不能把 token 放 URL 或假定普通 `<script src>` 自动附带 bearer header。首期本地同源页面可沿用回环服务的明确访问策略，不扩大监听范围。

代码、角色模型、纹理、动作来源、SMPL-X 派生资源分别保留许可信息；SignBridge 代码的 MIT 标识不能覆盖所有资产。直接复用已存在的演示包仍应保留其批准范围、研究用途和出处说明，详见本项目此前的[资产复用报告](csl-animation-existing-assets-reuse-2026-10-28.md)。

## 7. 可作为开发验收的最小范围

推荐先实现独立动画面板、静态动作清单、明确确认后的完整短语匹配与播放控制，再对接一个现有识别入口。以少量可追溯片段验证：资源加载/错误、精确选片、暂停/停止/调速、重复播放、迟到结果丢弃、资产缺失与审核状态展示。视觉保真和目标手语者理解另设验收，程序通过不能替代。

本轮 **NOT_RUN**：FastAPI 启动/实际请求、模型推理、浏览器/WebEngine 渲染、桌面交互、GPU 帧率、资源打包、语义审核与产品集成。模块探测与源码审计不证明这些项目通过。

## 源码证据

[1]: ../../slbridge/api.py#L288
[2]: ../../slbridge/api.py#L731
[3]: ../../slbridge/api.py#L993
[4]: ../../slbridge/api.py#L656
[5]: ../../slbridge/desktop.py#L121
[6]: ../../pyproject.toml#L14
[7]: ../../slbridge/candidate_context.py#L46
[8]: ../../slbridge/caption_events.py#L33
[9]: ../../slbridge/runtime_contracts.py#L43
[10]: ../../slbridge/session.py#L723
[11]: ../../slbridge/job_runner.py#L341
[12]: ../../slbridge/api_cli.py#L55
[13]: ../../slbridge/api.py#L80

- [API 创建与服务端路径边界][1]：`slbridge/api.py:288–320`。
- [统一识别请求][2]：`slbridge/api.py:731` 起；字段定义 `slbridge/api_schemas.py:171–187`。
- [文件推理与 job 结果][3]：`slbridge/api.py:993`、`:1104–1133`。
- [Caption 会话操作][4]：`slbridge/api.py:656–729`；动作枚举 `slbridge/api_schemas.py:210–213`，没有 confirm。
- [Qt 界面][5]：结果/候选区 `slbridge/desktop.py:258–339`，取消 `:651`，候选显示 `:741`，结果更新 `:877`，关闭 `:1132`。
- [依赖清单][6]：API extra `pyproject.toml:14`，PySide6 Essentials `:39`。
- [未确认候选][7]：`slbridge/candidate_context.py:46–99`。
- [字幕程序提交与生命周期][8]：`slbridge/caption_events.py:33–40`、`:80–100`、`:116–133`。
- [请求/响应元信息][9]：`slbridge/runtime_contracts.py:43–117`。
- [桌面 session worker 有效性][10]：`slbridge/session.py:723–828`。
- [内部 job 取消][11]：`slbridge/job_runner.py:341`。
- [回环默认与鉴权约束][12]：`slbridge/api_cli.py:63–83`；[HTTP 中间件][13] `slbridge/api.py:80–99`。
