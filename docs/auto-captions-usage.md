# 自动字幕与模型接入

本轮核心实现保留现有手动路径。自动模式是持续采集、启发式停顿分段和完整片段推理；现有模型未声明原生 stream。默认端点和未知动作拒识尚未经真人长流验证。

## 启动桌面

在仓库根目录使用现有媒体环境：

```powershell
.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt -r requirements-api.txt -r requirements-onnx.txt
.venv-libras/Scripts/python.exe -m pip install -e .
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli
```

加载支持 `captured_clip` 的模型，打开摄像头或视频回放，再勾选“自动追加字幕”。完成每个词或固定表达后保持可见姿态并稍作停顿。候选状态与正式字幕分开；右侧“字幕记录”按动作追加，可撤销上一条或独立清空记录。“支持的手语词”在相邻标签页。

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

实际摄像头活动观察与释放探测（不保存像素，不评估动作准确率）：

```powershell
.venv-libras/Scripts/python.exe -m experiments.validate_auto_camera --camera 0 --model artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009 --output artifacts/auto-camera-probe.json
```

运行报告见[完成与限制](reports/auto-captions-2026-10-11.md)。
