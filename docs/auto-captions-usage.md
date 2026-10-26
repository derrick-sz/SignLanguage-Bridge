# 自动字幕与模型接入

桌面提供连续视频窗口识别、停顿确认追加字幕和手动录制三种路径。现有模型仍按完整窗口执行 clip 推理，未声明原生 stream。默认端点、连续窗口准确率和未知动作拒识尚未经真人长流验证。

## 无需停顿的连续识别

识别主界面预览下方新增“显示关键点”。勾选后显示身体（绿色）、左手（蓝色）、右手（橙色）的有效关键点与连线；摄像头镜像时骨架同步翻转。开关默认关闭，手动录制期间也可切换；关闭后保留正常视频预览和识别状态。关键点与其原始帧配对显示，检测过慢或暂停时会清除骨架；不使用旧骨架套在新画面上。与自动／连续识别共用活动观察图，不将显示坐标替代模型保存的特征提取协议。该开关不保存新的关键点文件，样本录制仍使用独立工作台。

打开视频或摄像头、加载模型后，勾选“连续识别视频流（无需停顿）”。动作中持续刷新“当前窗口结果”；它会随后续画面修正，不会重复写入正式字幕历史。积累窗口最少 1.2 秒、最多 2.4 秒，目标提交间隔 0.5 秒；只有一个后台模型任务，忙时跳过旧窗口，空闲时取最新画面。实际刷新受模型与设备吞吐量限制。

手不可见或观察质量丢失会清空当前窗口显示并重新积累；暂停、取消、换源、停止模式也使当前结果失效，但已归档的候选序列仍保留。超过 3 秒的结果不显示或归档。恢复连续识别不依赖停顿。此机制没有训练新的流式模型，不能据此声称任意连续句子均能准确翻译。

### 保留候选和导出 LLM 上下文

右侧默认打开“候选序列”。后台每次有效连续识别完成就按顺序归档，界面只显示最新大字结果不会覆盖这些记录；不会保留尚未执行的窗口，拒识、取消、过期返回也不作为有效候选。默认保留 200 条，可在页内改成 1–1000 条，超出后淘汰最早记录并显示累计数量。降低条数会立即淘汰，调高条数不恢复旧项。

列表显示候选编号、手语种类、源时间范围和释义；悬停可看完整文本、来源轮次、分类前三项分数、模型和词典审核状态。“清空候选”独立于“清空记录”，不会删除正式字幕，也不会关闭视频、摄像头或连续识别。清空前正在运行的旧窗口不会补回；下一次新窗口仍可产生候选。取消、停止、换源、切换模型和暂时移出双手都保留既有记录，后续结果通过来源轮次、模型代次、窗口版本区分。

“导出上下文”将当前保留记录原子写为 UTF-8 JSON。记录只驻留当前会话，不自动写盘；退出前需要导出。导出包括 `schema_version/kind/session_id/revision/max_history/dropped_count/evidence_policy/candidates`。每条候选含稳定 ID、全局顺序、请求 ID、来源与模型版本、`start_time/end_time`、原始 `result`、显示时的 `caption` 和词典审核状态。分类保留全部原始 `scores`；中文保留原始 `tokens`、解码器和可用的词级分数，模型未提供的分数、词级边界或序列备选保持缺失。词级时间如有提供，相对于窗口起点；源窗口时间仍单独保留。导出不含视频或关键点帧。

所有项均为 `unconfirmed`。`overlaps_candidate_id` 指示同一连续段中与前一候选重叠；最早的前项可能已因容量淘汰。相同结果的不同窗口也保留为不同修订，不能据此计为多个动作。句界未知，窗口中可能有重复或遗漏；后续 LLM 应按来源和窗口版本取证，不把这份数据当已确认聊天历史。当前没有接入 LLM、自动联想或生成翻译。

程序可调用 `RecognitionSession.candidate_context()` 读取深拷贝快照，以 `after_revision` 跳过未变化内容；`set_candidate_limit()`、`clear_candidates()`、`export_candidate_context(path)` 提供相同操作。它们是本地会话方法，尚未新增 HTTP 或 LLM API。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Language csl -Camera 0
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Language libras -Video "Libras-EQT-UECE (RGB)/2_Sim/INFORMANTE 5/IMG_1316.mp4"
```

脚本优先使用依赖齐全的 `.venv-gpu`，其次 `.venv-libras`；两者均不可用则明确报错。GPU 环境需额外安装 `requirements-desktop.txt`；实际 provider 与 CPU 回退原因见模型详细信息。Libras RF 只支持 CPU，其主要耗时是 MediaPipe 提取，GPU 启动不会将 RF 自动转成 GPU 模型。`-NoLaunch` 可检查所选解释器与参数，`-Manual` 关闭连续启动。

也可直接运行 `.venv-gpu/Scripts/python.exe -m slbridge.desktop_cli --continuous --model <模型目录> --camera 0 --device auto`。带 `--continuous` 的启动会先等模型预热完成再打开输入，避免短视频在模型加载时已播放完毕。停止连续识别、等待当前原生任务返回后可切换手语，再重新开启。

## 启动桌面

在仓库根目录使用现有媒体环境：

```powershell
.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt -r requirements-api.txt -r requirements-onnx.txt
.venv-libras/Scripts/python.exe -m pip install -e .
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli
```

加载支持 `captured_clip` 的模型，打开摄像头或视频回放，再勾选“停顿确认后追加字幕记录”。完成每个词或固定表达后保持可见姿态并稍作停顿。候选状态与正式字幕分开；右侧“字幕记录”按动作追加，可撤销上一条或独立清空记录。“支持的手语词”在相邻标签页。

### 主界面切换手语

在视频／摄像头按钮旁的“识别手语”选择“中国手语”或“巴西手语”，无需先打开设置。系统沿用现有后台模型加载与预热流程，摄像头不会因模型切换而重开。加载期间原手语仍显示为当前选择，状态栏提示正在准备的目标；成功后更新选择、词表和模型标题，失败则保留原模型与预览。

录制、识别、加载和取消后仍未结束的模型任务期间，切换控件不可用。切换清除当前结果和未完成动作，保留已提交的字幕历史；自动分段停止，切换完成后可重新开启。加载期间暂停识别，热切换不保证零等待。

默认候选来自启动工作目录下的两个现有模型包：`artifacts/models/cecsl-tfnet-onnx-20261009` 和 `artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009`。对应包不存在时显示“未配置”。可以在“设置 → 选择识别模型”加载自定义包；加载成功后，已知手语的候选使用该模型目录。“恢复上个模型”也同步主界面选择，其他手语模型显示为“其他手语模型”。

这里选择的是要识别的手语种类。系统不自动判断中／巴手语，中文 CTC 仍输出完整片段的词序列，Libras RF 仍覆盖既有的 12 类。

关闭自动模式后，可继续手动录制。取消、切换来源或模型会使旧任务失效，但不删除正式字幕历史；历史只由“清空记录”删除。历史清空同时取消尚未提交的候选，避免旧结果补回。原生模型调用可能继续运行至返回；再次识别或切换需等待它释放容量。

参数使用 `SegmentConfig`，可保存部分字段为 JSON，再用 `--segment-config path/to/config.json` 加载，例如：

```json
{"config_version":1,"end_candidate_seconds":0.65,"max_segment_seconds":8.0}
```

这是配置示例，不是已选定的质量最优参数。全部字段及状态机见[分段契约](automatic-segmentation.md)。低运动不是空闲；手离开、遮挡、时间戳缺口和过长片段会拒绝或中止，不会自动确认成功。静态保持只提交一次，释放或新的动作运动后可以再次提交相同词。

## 统一 API 与两语言路由

```powershell
.venv-libras/Scripts/python.exe -m slbridge.api_cli --catalog configs/unified-runtime-models.example.json --model libras-rf --device cpu
```

示例 catalog 引用本机既有产物；新克隆需提供自己的可信模型包并更新路径。HTTP 输入是上传片段，摄像头完整采集仍由桌面或客户端负责。

两个语言实例：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.api_cli --catalog configs/unified-runtime-models.example.json --route libras=libras-rf --route csl=csl-tfnet --device cpu
```

每种语言拥有独立 manager、generation、容量和回退状态。语言路由隔离不等于硬件同时执行，也不是双语语义映射。桌面当前是单路输入；双路由能力通过 API 提供。

v3 能力、typed 输出、超时、字幕 session 和事件操作见[API 契约](unified-api.md)。旧 v1 分类与 v2 CTC 序列端点仍可用。中文 CTC 输出 gloss，未进行整句葡语翻译。现有 RF 中文释义为草案；未审核内容不会自动变成已审核映射。

## 新模型包与适配器

现有 RF/ONNX 分类和 CTC 包格式及校验保持兼容。模型包声明 backend、输入 spec、词表、依赖、权重和处理产物校验和；加载器校验包后才初始化权重。新 adapter 的具体 manifest 由其 loader 校验，统一能力由 `ModelCapabilities`、`ModelInputContract` 发布。示例不定义新的全局万能权重格式。

注册一个新的输入表示和 `AdapterDefinition(backend_id, load, wrap)`；实现真实校验、预处理、warmup、infer 和 close，再返回 typed `RecognitionResult`。调用者只使用 `ModelManager` 或 `ModelRouter`。桌面按 `captured_clip` 能力加载，不按算法或固定表示名称判断。

请求可绑定 model ID、generation、完整预处理版本、session/segment/request ID 和 source epoch。实际响应返回模型与预处理版本哈希、provider/device 和阶段时延。Prepared 对象自身的处理版本必须匹配；请求声明不能覆盖错误对象身份。分类、gloss 和 text 分别保留自己的 payload，raw score 不自动被称作概率。

当前 225 维 MediaPipe 与实验 321 维 Libras174 TCN 表示互不兼容；本轮没有接入训练工作树的 TCN，也没有修改或重训权重。新模型需要冻结并验证自己的在线提取协议，不能补零或截断。可运行接入示例见[adapter 示例](model-adapter-example.md)，示例 NumPy 求和不代表手语识别效果。

## 可复现验证

```powershell
.venv-libras/Scripts/python.exe -m pytest -q -ra
```

默认参数的原始视频回放可能拒绝。真实分段状态、RF HTTP 调用、事件去重读回、Qt 已挂载控件及截图验证：

```powershell
.venv-libras/Scripts/python.exe -m experiments.validate_auto_captions --model artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009 --video "Libras-EQT-UECE (RGB)/7_Quero/INFORMANTE 2/IMG_0696.mp4" --pause-at 1.5 --output artifacts/auto-captions-smoke
```

`--pause-at` 明确生成“真实 RGB 前缀＋末帧静态保持”的合成停顿；它是工程回放，不能当作自然停顿或真人准确率证据。省略该参数则直接回放原视频，无完成片段会明确失败。HTTP 使用真实 RF 包、lossless AVI 上传和 localhost socket；桌面复用真实 MediaPipe 观察缓存回放，再按保存的提取器重新处理 RGB。缓存观察用于可重复的工程验证，不能冒称实时观测速度。

输出包括 `fixture.json`、`replay.json`、分段 AVI、服务日志和 `mounted-history.png`。只保存在忽略目录，不上传原始视频。模型、数据和图形桌面是此 smoke 的前提。

provider 和两个实际模型探测：

```powershell
.venv-libras/Scripts/python.exe -m experiments.probe_unified_runtime --libras-model artifacts/runtime-acceptance-20261009/models/libras-rf-onnx-20261009 --libras-video "Libras-EQT-UECE (RGB)/7_Quero/INFORMANTE 2/IMG_0696.mp4" --csl-model artifacts/models/cecsl-tfnet-onnx-20261009 --csl-video D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4
```

记录实际 ORT provider、fallback、显式 CUDA 不兼容及两个任务结果；不开展训练或大规模数据处理。初次 CTC CPU 探测曾耗时约 11.5 分钟；性能修复后，同一完整视频连续三次为 6.31–6.62 秒，仍未达到动作结束后 2 秒内显示字幕的目标。只探测 RF/provider 时省略两个 `--csl-*` 参数。报告分阶段保存，但原生同步调用仍需等待返回。

API、桌面和 `RuntimeOptions` 默认使用 4 个 ORT 推理线程；CPU CTC 默认仅在内存副本中将卷积权重的 float32 subnormal 值归零。原始模型包不改写，运行响应的 `cpu_numerics` 记录源文件与执行模型哈希、策略及处理数量。需要复核原始数值行为时使用 `--cpu-conv-subnormals preserve`；复现旧单线程配置还需加 `--intra-op-threads 1`。其他模型/视频应重新验证数值一致性，不应把此样本的完全一致推广为全数据集准确率证明。

独立完整视频基准会保存逐次阶段耗时和完整 logits；可指定参考 logits 做数值及逐帧 argmax 对比：

```powershell
.venv-libras/Scripts/python.exe -m experiments.benchmark_ctc_cpu --model artifacts/models/cecsl-tfnet-onnx-20261009 --video D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4 --output artifacts/ctc-cpu-benchmark --runs 3
```

根因、单变量对照、原始 693 秒记录的复现边界及 HTTP 证据见[CPU 性能修复报告](reports/ctc-cpu-performance-fix-2026-10-11.md)。

## GPU 推理

本机 RTX 5060 Ti 已用独立 `.venv-gpu` 验证完整 CTC 视频。原 `.venv-libras` 安装的是 CPU ORT，启动 GPU 推理应使用 GPU 环境的 Python。GPU 依赖现在包含 CUDA/cuDNN 运行库；Windows 加载器会补充预加载同版本 cuDNN 引擎。统一运行器、API 和桌面的默认显存池上限为 8192 MB，可通过 `--gpu-mem-limit-mb` 覆盖；上限不会预先占用全部显存。

使用 Python 3.12 创建独立环境：

```powershell
python -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r requirements-api.txt -r requirements-onnx-gpu.txt "opencv-contrib-python==4.10.0.84" "scipy==1.16.3"
.venv-gpu/Scripts/python.exe -m pip install --no-deps -e .
.venv-gpu/Scripts/python.exe -m pip check
.venv-gpu/Scripts/python.exe -m slbridge.api_cli --catalog configs/unified-runtime-models.example.json --model csl-tfnet --device cuda
```

`--device cuda` 要求实际 CUDA provider 激活，依赖不兼容会明确失败。`auto` 可在初始化失败时回退 CPU；真实设备和回退原因见 `/v3/health`、识别响应的 `runtime`。API 完整视频使用 `/v3/recognitions`，请求 `sign_language=csl`、`task_type=gloss_sequence`。GPU 依赖与 CPU `onnxruntime` 分别安装在各自环境。

同一完整视频的 GPU 模型前向为 0.54–0.56 秒，文件推理总耗时 1.92–1.99 秒，真实 HTTP 上传为 2.07–2.25 秒，输出与 CPU 参考一致；详细数值及限制见[GPU 实测报告](reports/ctc-gpu-inference-2026-10-11.md)。本次是完整片段推理，动作结束到字幕显示的 P95 目标仍需单独验收。

GPU 基准可保存完整 logits，并核对数值、逐帧 argmax 和最终 CTC token：

```powershell
.venv-gpu/Scripts/python.exe -m experiments.benchmark_ctc_runtime --device cuda --model artifacts/models/cecsl-tfnet-onnx-20261009 --video D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4 --output artifacts/ctc-gpu-benchmark --runs 3 --reference-logits artifacts/ctc-perf-diagnosis/production-original-one/logits-1.npy
```

参考 logits 是本机 CPU 对照产物；其他环境可先用同一入口的 `--device cpu` 保存自己的参考。旧 `experiments.benchmark_ctc_cpu` 命令继续兼容。当前 `.venv-gpu` 实测范围是模型及 API；桌面启动还需安装 `requirements-desktop.txt`，再使用 `.venv-gpu/Scripts/python.exe -m slbridge.desktop_cli --device cuda --model artifacts/models/cecsl-tfnet-onnx-20261009`。

实际摄像头活动观察与释放探测（不保存像素，不评估动作准确率）：

```powershell
.venv-libras/Scripts/python.exe -m experiments.validate_auto_camera --camera 0 --model artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009 --output artifacts/auto-camera-probe.json
```

运行报告见[完成与限制](reports/auto-captions-2026-10-11.md)。
