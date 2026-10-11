# CTC CPU 性能问题：根因、修复与实测

同一段 8.2 秒完整视频在修复后连续三次总耗时为 **6.31–6.62 秒**。本轮可重复的原始单线程配置为 54.93 秒，整体改善约 8.6 倍；完整 `62 × 3516` logits 与原模型逐元素相等。初次报告的 692.94 秒历史记录仍保留，但本轮未复现那次额外变慢，不能把全部 693 秒都归因于已确认的两个瓶颈。

后续已建立独立 GPU 环境并修复 cuDNN 引擎预加载与显存池上限：相同完整输入的 CUDA 模型前向约 0.54–0.56 秒，文件推理总耗时约 1.92–1.99 秒，真实 HTTP 上传约 2.07–2.25 秒。GPU 数值、实际算子及 HTTP 证据见[GPU 实测报告](ctc-gpu-inference-2026-10-11.md)；下文保留本次 CPU 对照的原始配置与结果。

## 已确认的原因

1. **视觉卷积权重含大量 float32 subnormal 值。** 原 ONNX 的 9 个卷积初始化张量包含 1,438,942 个非零但绝对值小于 float32 最小正规数（约 `1.175e-38`）的权重。这些值会增加本机 CPU 浮点计算成本。最重的 `/visual/layer4/layer4.1/conv2/Conv` 包含 1,188,895 个此类权重；原始单线程 profile 中该层耗时约 17.07 秒。
2. **API、桌面和统一 RuntimeOptions 默认仅使用 1 个 ORT 推理线程。** 该视觉网络需要处理 260 帧 padding 后的 RGB 张量，本机单线程成本很高。相同计算在 4 线程下明显改善。
3. **CTC 解码成本很小。** 原始完整输入的 profile 显示 `Conv + FusedConv` 合计约 53.04 秒，模型前向总共约 53.62 秒；prefix beam 解码仅约 28–32 毫秒。后续不启用 profiler 的生产入口复测也得到约 53.5 秒模型耗时，支持视觉卷积为主要瓶颈的判断。

上述原因通过单变量对照确认。没有证据证明 693 秒记录中其余差异来自某个具体进程、内存压力或 CPU 调度，因此这些可能性不作为已确认原因。

## 实现

- CPU CTC 会话在原模型校验完成后，对**内存执行副本**的 float32 Conv 权重进行位级检测，将 subnormal 转为带原符号的零。正常权重与源模型文件不改写；输入帧、预处理和解码参数保持原协议。
- 共享初始化张量若还被其他算子、图输入/输出或嵌套子图使用，为卷积单独创建清理后的副本，保留其他消费者的原值。
- API、桌面和 RuntimeOptions 默认线程数统一为 4，可通过原线程参数覆盖。
- 实际 CPU 降级也应用同一策略；实际 CUDA 会话保留原始权重。分类模型不应用此数值策略。
- CLI 新增 `--cpu-conv-subnormals flush_to_zero|preserve`，HTTP 模型切换字段为 `cpu_conv_subnormals`。默认 `flush_to_zero`；`preserve` 用于严格源数值对照。
- CPU 响应增加 `runtime.cpu_numerics`，记录策略、处理数量、源和执行副本 SHA-256。`model_version` 仍追溯经过验证的源包，执行副本哈希单独披露。
- CTC 响应增加 `timing_ms.model_ms/decode_ms`，既有 `inference_ms` 保留模型与解码的合计意义。

未采用 ORT 的全局 `session.set_denormal_as_zero` 作为生产修复：官方说明该配置与首次会话、主线程和线程池有关；本项目还会同时加载 RF、CTC 并跨工作线程调用，单独改某个 SessionOptions 不足以保证一致行为。内存权重清理避免依赖这个全局状态。[ORT 1.23.2 配置定义](https://github.com/microsoft/onnxruntime/blob/v1.23.2/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)、[ORT 线程配置](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)。

## 同视频、同协议对照

环境为 Windows、Intel Core i5-14600KF（14 核 / 20 逻辑处理器）、32 GB 内存、ORT 1.23.2、ONNX 1.19.1、CPUExecutionProvider。此环境未提供 CUDAExecutionProvider，所有下述推理均为真实 CPU 执行。

输入为本地 `D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4`：247 源帧、260 padded frames；完整解码、无抽帧、无截断。源时间范围为 `[0, 8.2]`；媒体容器时长 8.2333 秒。

以下生产入口均执行 `ModelManager.switch → warmup → infer_file`，且不启用 ORT profiler。模型加载与 warmup 单独计时，不包含在每段服务耗时内。

| 配置 | 预处理 ms | 模型 ms | 解码 ms | 总耗时 ms |
| --- | ---: | ---: | ---: | ---: |
| 原权重，1 线程 | 1379.58 | 53507.81 | 29.67 | 54927.23 |
| 清理极小权重，1 线程 | 1403.09 | 18089.32 | 27.37 | 19529.74 |
| 原权重，4 线程 | 1520.12 | 15003.69 | 26.58 | 16560.63 |
| 清理极小权重，4 线程，第 1 次 | 1364.18 | 5215.32 | 27.82 | 6617.71 |
| 清理极小权重，4 线程，第 2 次 | 1314.44 | 5027.58 | 28.44 | 6380.57 |
| 清理极小权重，4 线程，第 3 次 | 1387.27 | 4886.88 | 28.92 | 6311.90 |

只清理权重使单线程模型前向从 53.51 秒降至 18.09 秒；保持清理策略再从 1 线程改至 4 线程，前向降至约 4.89–5.22 秒。原始配置到新默认配置的服务耗时按三次中位数为约 **8.61 倍**改善。三次修复测量的模型加载与 warmup 合计为 1705.44 ms，另计于首段耗时之外。

额外按初次 `probe_unified_runtime` 的完整顺序复测：加载 ONNX RF → 加载并预热 CTC → RF / MediaPipe 视频推理 → CTC 完整视频，显式保留原权重和单线程，得到 54.86 秒总耗时。它仍未复现历史 692.94 秒。这一记录与当前受控对照分开，避免把一次历史异常直接当成稳定基线。

## 数值和 HTTP 验证

保存完整 logits，比较形状、最大绝对误差、逐帧 argmax 与最终 token。原始 1 线程、原始 4 线程、修复 1 线程及修复 4 线程三次之间，本视频全部满足：

- logits 形状 `62 × 3516`，最大绝对误差 **0**，逐帧 argmax 一致。
- prefix beam 输出均为“平方 / 我 / 自己 / 。”。
- 保持 247 源帧、260 padded frames、完整视频解码和原词表。

真实 localhost socket smoke 使用生产 API CLI 默认值上传原始完整 MP4。最新一次 HTTP 200，socket 壁钟 **7168.07 ms**，服务总耗时 **7148.48 ms**（预处理 1507.96、模型 5601.18、解码 29.02 ms）。响应确认 4 线程、1,438,942 个处理值和两个模型哈希；随后真实 `/v3/models/switch` 依次切换 `preserve`、`flush_to_zero`，读回实际策略成功，测试所属服务进程已释放。该项与直接三次基准分别记录，不用于 P95 推断。

## 回归、构建与身份

全量回归：**546 passed、9 skipped、18 warnings，41.50 秒**。启用既有真实 RF 视频跨路径一致性测试；9 个跳过项因媒体环境缺少 Torch，均为原生模型/导出测试。已保存的 ONNX CTC 在本次真实运行，不依赖这些跳过项。

新增 CPU 数值回归先确认原实现在生产会话边界保留 subnormal 导致测试失败，再实现修复。最终覆盖正负 subnormal、带符号的零、最小正规数保留、源文件不改写、共享与嵌套图消费者、preserve、分类模型隔离、实际 CUDA 数值隔离、显式/隐式 CPU fallback，以及 API 策略传递。

wheel 隔离构建成功；`python -I` 从 wheel 路径加载模块，再实际初始化同一 CPU CTC 模型，确认默认线程数、处理数量、策略与哈希均正确。wheel SHA-256 为 `7ce887b8c59a3f423b9adb3b527da85e7219108916d2ff4530b252c79413fbee`。

| 身份 | SHA-256 |
| --- | --- |
| 原始 ONNX | `e52bf9ff768d7d9d4d4bb1c73ddae5f1693060cdd239d718d1fe65e36ba3f6c6` |
| CPU 内存执行副本 | `c7082c37c5697db088ef989bc2f712f70d720427e96ba39ae7832c9fc69b9097` |
| 输入 MP4 | `d0279bde09688fe1edae603d8e7c3e41cd93342cee5982fdef357c76e0bdb45b` |

本地忽略目录 `artifacts/ctc-perf-diagnosis/` 保存：

- `production-original-one/`、`production-original-four/`、`production-fixed-one/`、`production-fixed-four/`：生产入口阶段耗时、完整响应和 `.npy` logits。
- `original-mixed-probe.json`：保持历史双模型顺序的原配置复测。
- `reference-full-one-thread.json` 及对应 ORT trace：卷积节点耗时；profiling 仅用于诊断。
- `http-fixed/report.json`、`server.log`：真实上传、策略切换和服务释放结果。
- `tests.xml`、`wheels/`、`wheel-validation.json`：回归、构建和从 wheel 实际加载证明。

原始历史证据仍为 `artifacts/unified-runtime-probe-20261011.json`，不覆盖。模型和视频留在本地，不纳入源码提交。

## 复现与边界

```powershell
# 旧单线程配置；先保存参考 logits
.venv-libras/Scripts/python.exe -m experiments.benchmark_ctc_cpu --model artifacts/models/cecsl-tfnet-onnx-20261009 --video D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4 --output artifacts/ctc-reference --threads 1 --cpu-conv-subnormals preserve --runs 1

# 新默认配置；完整输出对照及连续三次测量
.venv-libras/Scripts/python.exe -m experiments.benchmark_ctc_cpu --model artifacts/models/cecsl-tfnet-onnx-20261009 --video D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4 --output artifacts/ctc-fixed --runs 3 --reference-logits artifacts/ctc-reference/logits-1.npy
```

修复解决了可重复的 CPU 性能瓶颈，尚未证明整个系统达到“动作结束到字幕显示 ≤ 2 秒”或 P95 目标。没有对该视频做标注评分；token 和 logits 一致只证明本次修复没有改变样本输出，不证明识别内容正确或整个数据集准确率不变。新模型/其他视频需重新验证数值行为，可选择 `preserve`。CPU 上此视觉网络仍有约 5 秒的完整片段前向成本，GPU、量化或原生流式模型属于后续独立工作。
