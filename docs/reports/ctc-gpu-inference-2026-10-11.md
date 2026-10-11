# CTC GPU 推理：环境修复与完整视频验证

RTX 5060 Ti 已通过正式模型加载器和真实 HTTP 上传执行同一段 8.2 秒完整视频。最终锁定依赖的三次模型前向为 **0.541–0.559 秒**，本地文件推理总耗时为 **1.917–1.990 秒**；三次 HTTP 上传的客户端耗时为 **2.069–2.250 秒**。完整 logits 与 CPU 参考在既定容差内一致，最终词序列完全相同。HTTP 与字幕显示不是相同计时边界，尚未证明字幕端到端 P95 ≤ 2 秒。

## 为何之前没有使用 GPU

1. **运行环境安装了 CPU ORT。** 原 `.venv-libras` 的 ORT 1.23.2 只有 Azure、CPU provider。显卡和系统 CUDA 的存在不能替代 `onnxruntime-gpu`。本次保留该环境，新建 `.venv-gpu` 安装 GPU ORT 及 CUDA/cuDNN Python 运行库。
2. **ORT 1.23.2 的 DLL 预加载列表遗漏新版 cuDNN 引擎。** 当前 cuDNN 9.27 的 `cudnn_engines_tensor_ir64_9.dll` 已安装，但首次真实卷积仍报 `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`。仅添加 DLL 搜索目录仍失败；显式预加载该引擎后，同一模型预热通过。生产加载器现定位实际已加载的 `cudnn64_9.dll`，从同目录按绝对路径预加载 `cudnn_engines*_9.dll`，避免混用不同 cuDNN 安装。此补加载仅用于 CTC；CPU 和分类模型不加载额外引擎。
3. **统一运行器的默认显存池上限过小。** 2 GiB 配置可以通过短预热，但完整 260 帧输入在 `/visual/motorAttention1/reLU/LeakyRelu` 报错：剩余 627,029,504 字节，小于本次需要的 834,928,640 字节。相同完整输入改为 8 GiB 上限后通过。`RuntimeOptions`、API、桌面的默认值现统一为 8192 MB，与既有 CE-CSL 工具一致；它是分配上限，不是预留全部显存。

依赖回归还发现：自动解析得到的 Protobuf 7.36.2 拒绝 `skl2onnx 1.19.1` 发出的布尔值整数属性，导致 5 个已有 RF 导出测试失败。GPU requirements 和 package extra 固定 `protobuf==4.25.8` 后，包含这些测试的 96 项回归全部通过。

此前 CPU 性能问题已通过卷积 subnormal 权重清理和 4 线程修复，稳定对照为 6.31–6.62 秒，见[CPU 报告](ctc-cpu-performance-fix-2026-10-11.md)。历史 692.94 秒记录未在本轮复现，不能把全部时间差归因于本次已确认的问题，也不能据此宣称实测加速 300 倍。

## 冻结输入与环境

| 项目 | 实测配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti，16311 MiB，driver 616.92 |
| OS / Python | Windows 10 19045 / Python 3.12.7，独立 `.venv-gpu` |
| ORT / ONNX | `onnxruntime-gpu[cuda,cudnn]==1.23.2` / `onnx==1.19.1` |
| NVIDIA 库 | CUDA runtime 12.9.79、cuBLAS 12.9.2.10、cuDNN 9.27.0.42 |
| 其他主要库 | NumPy 1.26.3、OpenCV contrib 4.10.0.84、Protobuf 4.25.8 |
| 运行器 | strict CUDA、device 0、4 CPU threads、8192 MB arena、TF32 关闭 |
| 内存池 | `kSameAsRequested`，每次调用后 `gpu:0` arena shrinkage |
| 模型 | `artifacts/models/cecsl-tfnet-onnx-20261009`，完整原始 ONNX |
| 视频 | `D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4` |
| 输入 | 247 源帧、260 padded frames，完整解码，无截断、无降帧 |
| 输出 | `62 × 3516` float32 logits，prefix beam 解码 |

源 ONNX SHA256：`e52bf9ff768d7d9d4d4bb1c73ddae5f1693060cdd239d718d1fe65e36ba3f6c6`。

manifest SHA256：`d69d700b7d120720292cc3d98bcb1bdd90791790ed2028acb12c945dff50ae40`。

视频 SHA256：`d0279bde09688fe1edae603d8e7c3e41cd93342cee5982fdef357c76e0bdb45b`。

CPU 原始参考 logits SHA256：`0216c3347259061c1810d3193e1cbe6694a78e09c4d0378fad1b694fd82091b6`。

GPU 使用源模型字节，未应用 CPU 的权重清理。真实 HTTP 前后源文件哈希相同。原 `.venv-libras` 仍为 CPU ORT，其依赖未替换。

## 完整文件推理

最终环境的加载加短预热为 1.738 秒，以下计时从已加载模型开始；第 1 次是短预热后的首次完整视频，不包含服务启动。每次仍执行完整视频预处理和原生前向。

| 次数 | 预处理 ms | 模型 ms | CTC 解码 ms | 运行器总计 ms |
|---|---:|---:|---:|---:|
| 1 | 1409.052 | 540.895 | 29.213 | 1990.051 |
| 2 | 1329.334 | 550.338 | 27.808 | 1917.067 |
| 3 | 1385.935 | 558.790 | 28.119 | 1982.244 |

初次 GPU 探测的三个完整调用为 1.987–2.052 秒，模型为 0.543–0.592 秒，说明短样本耗时有波动。Protobuf 固定解决导出兼容性，未建立它与耗时差异的因果关系。

三次最终完整 logits 的最大绝对误差均为 `7.62939453125e-06`。`rtol=1e-4, atol=1e-4` 全张量对照通过，所有逐帧 argmax 与最终 CTC token ID 均相同，输出为“平方 / 我 / 自己 / 。”。没有在本轮重新评分标注；输出一致性不代表识别内容正确或全数据集准确率不变。

## 实际 CUDA 算子与显存

对相同原模型执行短预热和完整视频，并捕获原生 ORT profile。以下是两次调用合计的 kernel 事件，不是图中唯一节点数：

| 算子 | kernel 事件 | 实际 provider |
|---|---:|---|
| Conv | 112 | CUDAExecutionProvider |
| LSTM | 8 | CUDAExecutionProvider |
| MatMul | 6 | CUDAExecutionProvider |

所有 Conv 事件均在 CUDA 上执行；少量 Cast、Concat、Gather、Reshape、Unsqueeze 仍由 CPU 处理。因而健康接口的 session provider 加上 profile 的实际算子共同证明 GPU 计算路径。profile 会增加开销，其 host event duration 未用作无 profiler 的性能结果。

最终基准期间 `nvidia-smi` 采样到设备总显存从 1634 MiB 达到 4272 MiB，进程退出后为 1629 MiB，GPU 利用率最高 100%。这是设备级采样，包含其他应用，不是该进程的精确峰值；8 GiB arena cap 也不是总显存占用承诺。

## 真实 HTTP 上传

使用 `.venv-gpu` 启动正式 `slbridge.api_cli`，指定 `--device cuda`，**省略显存参数**验证共享默认值。健康接口、三次结果及最终健康读回均确认 CUDA 激活、无回退、TF32 关闭、默认 8192 MB、模型 generation 1；三次独立 request ID 完整上传原 MP4。

| 次数 | 客户端上传至响应 ms | 服务运行器总计 ms | 预处理 ms | 模型 ms | CTC 解码 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 2249.503 | 2230.026 | 1639.356 | 547.765 | 29.610 |
| 2 | 2145.024 | 2127.177 | 1520.774 | 565.496 | 30.305 |
| 3 | 2068.558 | 2046.396 | 1455.002 | 556.302 | 26.108 |

HTTP status 均为 200，源帧数、padding、无截断、模型身份及四个 token 均验证通过。测试服务器退出、监听端口明确连接拒绝，源模型未改变。Windows 空端口返回连接拒绝约需 2 秒；验证脚本保留有限等待，不把短超时视为已释放。

当前主要剩余耗时是视频解码、RGB/JPEG 处理和张量准备，而非约 0.55 秒的模型前向。三次 HTTP 均超过 2 秒；自然停顿分段、字幕提交与显示、真人长流以及端到端 P95 仍需独立验证。

## 回归与产物

- 原完整 CPU 环境：**552 passed / 9 skipped**，包含真实模型/视频测试；9 项缺少 Torch 的导出测试跳过。
- 新 GPU 环境：**96 passed**，覆盖 ONNX provider、CTC、CPU 数值策略、RF 导出和 DLL 回归；不是整个 GUI/media/Torch 测试环境。
- `pip check`：GPU 环境通过；依赖完整 freeze 已保存。
- wheel 构建、独立 target 安装和真实导入：通过；安装包包含新 helper、8192 默认值及 GPU extras / Protobuf 固定配置。
- wheel SHA256：`9beaaf771d9ecaeaa46f028530b89cbfa1e85510644706368e73415d9e5b7e89`。
- GPU 环境本轮验证模型与 API；未安装或验收该环境的图形桌面。

证据保存在忽略目录，不上传视频或模型权重：

| 产物 | 内容 |
|---|---|
| `artifacts/ctc-gpu-validation/cuda-2gb/benchmark.json` | 完整输入 2 GiB cap 的原始失败 |
| `artifacts/ctc-gpu-validation/cuda-8gb/benchmark.json` | 初次 GPU 三次完整张量、对照与计时 |
| `artifacts/ctc-gpu-validation/cuda-final/benchmark.json` | 最终锁定依赖的三次原生调用 |
| `artifacts/ctc-gpu-validation/cuda-final/logits-*.npy` | 每次完整 logits |
| `artifacts/ctc-gpu-validation/cuda-final/gpu-monitor.json` | 有明确 scope 的设备级显存采样 |
| `artifacts/ctc-gpu-validation/profile/report.json` | 最终环境实际算子 provider 与原始 trace 路径 |
| `artifacts/ctc-gpu-validation/http-pinned/report.json` | 三次 HTTP、health/readback、源哈希、资源释放 |
| `artifacts/ctc-gpu-validation/cpu-tests.xml` | CPU 完整回归 |
| `artifacts/ctc-gpu-validation/gpu-tests-pinned.xml` | GPU 96 项回归 |
| `artifacts/ctc-gpu-validation/gpu-requirements-freeze.txt` | 最终 GPU 环境 freeze |

环境创建、正式启动和带参考 logits 的完整基准命令见[使用说明](../auto-captions-usage.md#gpu-推理)。安装与预加载机制参考 [ORT CUDA provider 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 和 [NVIDIA Windows cuDNN 安装文档](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)。
