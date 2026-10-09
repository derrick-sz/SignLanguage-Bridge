# 模型运行 API 与 ONNX 验收

日期：2026-10-09；代码版本 0.2.0；Windows / Python 3.12.7。本次落实用户已同意的模型可替换架构及新增 API、GPU/ONNX 要求。

## 实现结果

- 统一 `InferenceRequest` / `InferenceResponse`：请求 ID、模型 generation、识别结果、实际运行 provider 和分阶段耗时。
- ModelManager 单请求容量、候选校验与预热、提交/丢弃、旧模型回退和关闭后延迟释放；准备期间暂停新推理，失败保留当前模型。
- 桌面接入管理器，录制期间禁止切换，设置增加“恢复上个模型”，支持启动时配置执行设备/DLL 搜索目录。
- FastAPI v1：健康状态、模型别名、切换/回退、单片段上传识别及 OpenAPI；native 调用在线程执行；请求/文件/展开数组限制、损坏文件结构化错误和临时文件清理。
- RF/ONNX 注册与各自完整模型包契约；ONNX NC/NTC + mask、概率/logits、固定单片段 batch、复用 Session、图优化、预热、CPU/auto/strict CUDA。
- RF 导出使用 hash 固定的真实留出样本，验证概率/标签/阈值决策后发布新包；原 RF 包、标签与 split 不改写。ONNX external weights 被拒绝，避免未校验的外部权重引用。

## 测试与独立复审

全量命令在已有桌面/API/ONNX 环境执行，并设置实际模型与视频：

```powershell
$env:SLBRIDGE_LIVE_VIDEO='E:/githubproject/SignLanguage-Bridge/Libras-EQT-UECE (RGB)/105_Árvore/INFORMANTE 3/IMG_2971.mp4'
$env:SLBRIDGE_LIVE_MODEL='E:/githubproject/SignLanguage-Bridge/artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009'
.venv-libras/Scripts/python.exe -m pytest -q -rs
```

结果：**285 passed，0 failed，0 skipped，31.69 秒**。18 条第三方弃用提示来自 Starlette/anyio、Matplotlib/pyparsing 与 MediaPipe 使用的 protobuf。损坏视频测试预期产生 `moov atom not found` 解码日志。

独立复审最终 PASS，无剩余 P0/P1/P2。复审发现并修复：激活失败不再杀死桌面工作线程；WEBM 声明与实际输入格式对齐；API 可选依赖缺失时测试正常跳过；NPZ 头部及后部压缩数据损坏均转为输入错误。并发测试覆盖候选预热期间取消/关闭、旧请求失效、容量超限和 native 推理退出后释放。

`git diff --check` 与主环境 `pip check` 通过。

## 真实 Libras RF/ONNX 对比

来源：已有 12 条目/360 视频试点的冻结 test split，**72 个 NPZ 输入逐个验证 SHA256**。旧 RF 与 ONNX 包的输入、标签顺序、数据/split hash、拒识阈值一致。最大类别概率差 **7.6452891×10⁻⁷**，低于固定 absolute tolerance 1×10⁻⁶；72/72 预测/拒识决策一致；额外全缺失坐标输入的质量拒识一致。

单线程、batch=1、并发=1，预热后每后端 216 次（72×3），输入为已有 NPZ 和温热文件缓存：

| 阶段，毫秒 | 原生 RF P50 / P95 | ONNX CPU P50 / P95 |
|---|---:|---:|
| 预处理 | 2.131 / 2.785 | 2.341 / 3.035 |
| 模型推理 | 4.805 / 5.299 | 0.235 / 0.317 |
| 总运行时间 | 6.986 / 7.841 | 2.617 / 3.300 |

这些数值**不包含原始 RGB 视频解码与关键点提取**，不证明识别准确率改善，也不能直接作为摄像头端到端时延。

实际 TCP HTTP 上传另测原生/ONNX 各 72 个真实缓存输入，预测一致；HTTP 往返 P50/P95 为 **13.823/16.137 ms → 8.807/10.357 ms**（单遍，每后端 72 次）。新进程管理器及 HTTP 服务完成切换/回退，generation 为 1→2→3，退出后服务均已停止。

实际 RGB `IMG_2971.mp4` 上传、RF/ONNX识别、回退、严格 CUDA 加载 RF 失败且保留旧模型也通过。两次顺序视频请求的预处理约 2648 ms / 1415 ms，模型推理约 5.05 ms / 0.28 ms。前者含首次库初始化，后者在温热进程执行；只证明视频链路及预处理占主要耗时，不用于宣称整个视频流水线加速比。

## GPU 实际执行

RTX 5060 Ti 16 GiB、驱动 616.92；隔离 GPU 环境使用 ORT GPU 1.23.2 与本机 CUDA 12.9/cuDNN 9.10 DLL。默认加载最初因 cuDNN 目录未被搜索而回退 CPU；采用进程内 preload 和 DLL 目录配置后通过，未修改系统 PATH。

- 独立 MatMul/Conv 图 profile 确认 CUDA 计算节点。
- 生产 API → ModelManager → Recognizer → ONNX adapter → CUDA 通过实际 HTTP NPY 上传与 CPU 数值对比；夹具是未训练图，不能视为手语识别准确率证据。
- 主代理运行已提交的 `experiments.validate_onnx_gpu`：20 次推理 + 1 次预热记录 **21 个 CUDA MatMul 事件**，CPU/GPU 输出一致。
- 真实 RF 导出图虽可创建 CUDA Session，**73 个 TreeEnsembleClassifier 节点均在 CPU，CUDA 节点为 0**。本次 RF 性能收益来自 ONNX CPU 路径。

后续真实 TCN/ST-GCN 训练与导出仍待开展。需要分别检查图兼容性、GPU节点分配、精度与 P95 时延；CUDA Session 有效不代表所有算子执行于 GPU。GPU arena 参数是单 Session 选项，不是整个进程显存硬限制。

## 包与使用验证

正常隔离构建 wheel 成功：`signlanguage_bridge_core-0.2.0-py3-none-any.whl`，SHA256 `d9667ca254fb37702c543414ad43175f84f2ba1b8c26c633c19068829d872e9d`。无隔离构建曾因主环境未安装 setuptools 失败，切回项目声明的正常 build isolation 后成功，无需改变源码。

新建 wheel 环境，从仓库之外目录确认导入来自 site-packages：core-only 导入没有加载 FastAPI/ORT；API/export CLI help 可用；旧 hand63 模型独立推理返回 A。补装可选 API/ONNX 依赖后，安装包的真实管理器上传 NPY、request_id/generation、识别结果与 OpenAPI 验证通过；`slbridge-api --help` 和该环境 `pip check` 通过。

本机产物（不纳入 Git）：

- `artifacts/runtime-acceptance-20261009/cpu-parity/runtime_validation.json`：逐输入 hash、数值一致性和216次耗时。
- `cpu_api_validation.json`、`model_manager_switch.json`：真实上传及版本切换/回退。
- `tcp-video/api_video_validation.json`：原始视频 HTTP 链路。
- `gpu-api/gpu_api_validation.json`、`gpu-reproducible/gpu_validation.json`：API和生产适配器CUDAprofile。
- `rf-node-placement/rf_node_placement.json`：RF实际CPU节点分配。
- `export_cli_readback.json`：主环境正常导出CLI与72个留出输入验证。
- `models/libras-rf-onnx-20261009`：示例配置使用的导出模型。

使用入口：[API与ONNX使用说明](../模型运行API与ONNX_使用说明_2026-10-09.md)。设计入口：[设计与执行](../模型运行API与ONNX_设计与执行_2026-10-09.md)。
