# SLRT-CSL 在线 Gloss 模块

此模块接入 EMNLP 2024 *Towards Online Continuous Sign Language Recognition and Translation* 的 CSL-Daily 权重，输出原始 Gloss。当前实测采用显式选择的 **RTMPose 实验前端**，尚未完成官方 HRNet 端到端复现。详细结果与未通过项见[交付报告](reports/slrt-integration-2026-10-24.md)，接入依据见[架构与兼容性分析](reports/slrt-architecture-compatibility-2026-10-24.md)。**在本地 CE-CSL 数据集上的批量识别率与延迟实测（test 500 条 WER、方向探针、实时回放结论）见[数据集实测报告](reports/slrt-dataset-eval-2026-10-24.md)**；批量评测命令为 `scripts/evaluate_slrt_datasets.py`。

当前范围仅覆盖 `Online/CSLR` 的 CSL-Daily 在线 Gloss；`Online/SLT`（wait-2 G2T）与 `Online/CTC_fusion` 尚未接入。权重、数据、配置与官方复现要求的逐项差距见[SLRT Online 复现差距核查](reports/slrt-online-reproduction-gap-2026-10-25.md)，非数据资产的实际补齐记录见[资产补齐记录](reports/slrt-assets-bootstrap-2026-10-25.md)。

## 资源与环境

默认资源目录：`models/external/EMNLP24_Online/CSL-Daily`。

| 资产 | 默认相对路径 | 身份 |
|---|---|---|
| 模型 | `ckpts/cslr_best.ckpt` | SHA256 `e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d` |
| 词表 | `meta/csl_iso_with_blank.vocab` | SHA256 `0c4984f0dd3e7897b8c8d9e6e99a848b60814a5d9cf7a49baf7cc9b9cd124c76` |
| 配置示例 | `configs/slrt-csl.example.json` | JSON 中相对路径以配置文件所在目录为基准 |

checkpoint 的 938 个参数键严格加载，三路分类头均为 2001 类，blank=0。指定另一 checkpoint／词表时须在 JSON 中同时声明其 SHA256；仅输出维度相同不能证明词表顺序相同。类索引和 Gloss 严格按原 JSON 列表，不重新排序。

2026-10-25 数据补充：上游 `Paper-algorithm/SLRT-main/data/csl-daily` 已通过 Junction 指向 `F:\model_weigth\data\csl-daily`，含作者包的 `csl-daily-keypoints.pkl`（20,654 个视频、`[T,133,3]` float16）、`csl.train/dev/test`（18,401/1,077/1,176 句）和 S2G 词表；`Online/CTC_fusion/results/csl-daily_s2g/ckpts/best.ckpt` 已复制（504,360,673 B，SHA256 `54d2cb65…9739`）。目前唯一仍缺的数据文件是 `sentence_frames-512x512.zip`（CSL-Daily 视频帧）。

当前可复用 `.conda-env/python.exe`，已装 Torch 2.9.1+cu128、torchvision 0.24.1、OpenCV、Pillow、RTMLib 和 PySide6。此次没有升级或替换原环境依赖。`.venv-gpu` 名称虽含 GPU，目前没有 Torch，不能用于 SLRT 真实权重推理。

如需新环境，先建立独立 Python 3.10–3.12 环境，安装与设备匹配的 Torch/torchvision，再安装 [requirements-slrt.txt](../requirements-slrt.txt)；Qt 面板另需 `PySide6-Essentials`。不要将旧版 OpenMMLab 的全量依赖强装进现有环境。官方 HRNet/Faster-RCNN 提取另需 **Python 3.8 legacy 环境**，见 [requirements-slrt-legacy.txt](../requirements-slrt-legacy.txt) 与[资产补齐记录](reports/slrt-assets-bootstrap-2026-10-25.md)。

### 官方 HRNet 模式（默认）

需要配置这四个本地文件：

- `detector_config`：官方 `faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py`。
- `detector_checkpoint`：`faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth`。
- `hrnet_config`：`hrnet_w48_coco_wholebody_384x288_dark_plus.py`。
- `hrnet_checkpoint`：`hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth`。

提取器复用上游 `gen_pose.py` 的 legacy MMPose/MMDetection API，依赖 MMPose 0.x、MMDetection 2.x 和与 Torch/CUDA 相容的 mmcv-full。配置中的 `_base_` 也需保留在相应源码树中。本机已按 [requirements-slrt-legacy.txt](../requirements-slrt-legacy.txt) 建立独立 Python 3.8 CPU 环境（torch 1.13.1+cpu、mmcv-full 1.7.0、mmdet 2.25.3、mmpose 0.29.0），并跑通官方 HRNet 提取；`EMNLP24_Online` 包本身不含这些权重/配置，不能把旧版依赖装进 `.conda-env`。

官方资源入口：[SLRT Online](https://github.com/FangyunWei/SLRT/tree/main/Online)、[官方 gen_pose.py](https://github.com/FangyunWei/SLRT/blob/main/Online/CSLR/gen_pose.py)、[MMPose WholeBody 模型库](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html)。该旧版栈需要单独配置并验证，不能将“安装命令能执行”当成模型已通过。

2026-10-25 验证：用 `scripts/extract_slrt_hrnet_pose.py` 在 CE-CSL `dev-00015` 全 202 帧上跑通官方 HRNet-W48 DARK + Faster-RCNN（0 丢帧、733.7 s CPU），生成 `hrnet-npz` 后由本地 `--pose-backend hrnet-npz` 消费，输出 `拒绝 不`；同一视频的 `rtmpose-experimental` 输出 `拒绝 去 不`。这是**跨域链路验证**，不是 CSL-Daily 同域准确率，也不能据此判断哪个前端更准。项目 `.conda-env` 仍不能直接 import mmpose，因此进程内 `hrnet` 模式未验证，文件推理请走 NPZ；详见[资产补齐记录](reports/slrt-assets-bootstrap-2026-10-25.md)与[复现差距核查](reports/slrt-online-reproduction-gap-2026-10-25.md)。

2026-10-26 同域批量验证：`D:\Zenodo\csl-daliy` 与官方 `csl-daily.test` 相交的原始视频，经 RTMPose 全帧滑窗对齐官方 keypoints 后裁剪，以 `--pose-backend hrnet-npz` 跑 pinned `cslr_best.ckpt`：视频补充后 **1,086 条 test** 全部成功（**1,086/1,086、0 空输出，corpus WER 0.3018**，S1,154/D805/I522，8,220 参考 token；627 基线 WER 0.3032），对齐 mean 7.23 px / median 6.20 / P95 12.60，12 条 >25 px（其 WER 0.33–1.0），剔除后 WER 0.2974。首轮 627 阶段在内存不足时有 47 条失败，分片重跑全部成功。这仍是**近似句段裁剪**，不是官方 `sentence_frames-512x512` 协议，也不能当完整 1,176 test（缺 90 条）。逐条 GT Gloss + SLRT 预测与 Uni-Sign 对照见[1,086 条补充报告](reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md)与[627 条报告](reports/csldaily-video-slrt-unisign-2026-10-26.md)。

### RTMPose 实验模式

命令显式加 `--pose-backend rtmpose-experimental`。复用已有 `RtmposeExtractor` 的 CUDA DLL 处理与本地 lightweight ONNX 缓存；缓存缺失时拒绝，避免隐藏下载。默认 `rtmpose_cache_dir` 是 `models/external/uni_sign/rtmlib-cache`，可配置其他位置。

RTMPose 与 HRNet 均使用 COCO-WholeBody133 的解剖定义和 x/y/confidence，第 3 通道不是 z。这里按官方排序选择 hands42、mouth10、upper-body11，映射至 512 方形像素参考系；保留 RGB 和点位的同帧身份。检测器、姿态估计器和训练分布仍不同，必须保留结果里的 `pose.experimental=true`。MediaPipe225 输入会被拒绝。

### 已提取 HRNet NPZ

独立视频 CLI 支持 `--pose-backend hrnet-npz --pose-npz path/to/pose.npz`，NPZ 使用 `allow_pickle=False`，需要：

- `keypoints`：float 数组 `[T,133,3]`，原始完整点序，x/y 为 512 方形参考坐标，第 3 通道为 [0,1] confidence。
- `frame_indices`：整数 `[0..T-1]`，覆盖每帧。
- `timestamps`：严格递增的原视频时间戳（秒）。
- `metadata_json`：标量 JSON 字符串，包含 `video_sha256`、`frontend="hrnet_w48_dark"`、`coordinate_system="coco_wholebody_133_xyc_512_v1"`，建议另存提取源码、模型、配置的哈希。

视频哈希、帧号和时间戳须匹配。该模式只供已具备可追溯 HRNet 数据的文件推理，不能伪造 `frontend` 元数据，将 RTMPose 缓存当成 HRNet。摄像头及共享 Adapter 不接受这种文件绑定模式。可用 `scripts/extract_slrt_hrnet_pose.py` 生成；示例配置为 `configs/slrt-csl-hrnet-npz.example.json`。

独立视频命令示例：

```powershell
.conda-env\python.exe -m slbridge.slrt.cli video `
  --video 'D:/path/sign.mp4' `
  --config configs/slrt-csl-hrnet-npz.example.json `
  --pose-npz artifacts/slrt-hrnet/sign.npz `
  --output artifacts/slrt-hrnet/result.json `
  --events-jsonl artifacts/slrt-hrnet/events.jsonl `
  --windows-jsonl artifacts/slrt-hrnet/windows.jsonl
```

## 运行命令

以下均在项目根目录 PowerShell 执行。

严格检查真实权重（仅模型加载，不证明姿态前端和视频已就绪）：

```powershell
.conda-env/python.exe -m slbridge.slrt.cli inspect --config configs/slrt-csl.example.json --output artifacts/slrt-inspect.json
```

普通 MP4，当前可运行的实验前端：

```powershell
.conda-env/python.exe -m slbridge.slrt.cli video --video 'D:/path/sign.mp4' --pose-backend rtmpose-experimental --output artifacts/slrt-video/result.json --events-jsonl artifacts/slrt-video/events.jsonl --windows-jsonl artifacts/slrt-video/windows.jsonl --require-gloss
```

官方 HRNet 资源配置完成后，将 `--pose-backend` 改为 `hrnet`，并传入包含四项资源路径的 `--config`。`--device cpu` 可强制 CPU；CPU 已完成真实窗口功能验证，速度不足以支撑当前实时目标。

按原时间轴实时回放 MP4，以检验排队与交付：

```powershell
.conda-env/python.exe -m slbridge.slrt.cli replay --video 'D:/path/sign.mp4' --pose-backend rtmpose-experimental --output artifacts/slrt-replay.json
```

摄像头持续采集，Ctrl+C 结束；`--seconds` 可限定时长：

```powershell
.conda-env/python.exe -m slbridge.slrt.cli camera --camera 0 --pose-backend rtmpose-experimental --seconds 30 --output artifacts/slrt-camera.json --events-jsonl artifacts/slrt-camera-events.jsonl
```

独立 Qt 面板提供开始、暂停、继续、重置、停止及 JSON 导出，并可打开原 Legacy 工作区：

```powershell
.conda-env/python.exe -m slbridge.desktop_cli --slrt --slrt-pose-backend rtmpose-experimental
# 同一个面板的独立入口：
.conda-env/python.exe -m slbridge.slrt.cli desktop --pose-backend rtmpose-experimental --video 'D:/path/sign.mp4'
```

原有 `slbridge-desktop --model ...`、`--unisign`、RF 训练、数据采集、CE-CSL 和 Libras 路由保留。`--slrt` 与其他模型模式互斥，CSL 权重不会路由成 Libras。

安装项目后的入口为 `slbridge-slrt`；直接从工作区执行上述 `python -m` 无需重新安装项目。

## 统一模型接口

共享 `ModelManager` 后端注册名是 `slrt_online_csl`，输入表示名为 `slrt_rgb_hrnet133_v1`。生成本地模型包清单：

```powershell
.conda-env/python.exe -m slbridge.slrt.cli bundle --config configs/slrt-csl.example.json --pose-backend rtmpose-experimental --output models/external/slrt_csl_bundle/manifest.json
```

`load_model_adapter()` 读取该清单；`warmup()` 同时检查真实网络和姿态前端，缺前端时模型切换准备阶段失败，原模型继续保留。共享接口支持 file/captured_clip，返回 `RecognitionResult(task_type="gloss_sequence")`、`GlossSequencePrediction(decoder="slrt_window_majority")`，不把非 CTC 网络冒充 CTC。共享合同当前声明 `clip`；在线生命周期由下面的独立接口承载。

```python
from slbridge.slrt.assets import SLRTConfig
from slbridge.slrt.backend import SLRTBackend

backend = SLRTBackend(SLRTConfig(pose_backend='rtmpose-experimental'))
stream = backend.new_stream(event_sink=print)
try:
    # 由调用方持续送入 slbridge.sources.FramePacket(BGR, timestamp, frame_index, source_id)
    # stream.feed(packet)
    # stream.pause(); stream.resume(); stream.reset()
    report = stream.result()
finally:
    stream.close()
    backend.close()
```

`stream.flush()` 仅用于已知自然 EOF；手动停止、断线和源变更撤回尚未结束的 partial，不用尾部填充制造 final。后台采集使用 `SLRTCaptureSession`：`start/pause/resume/reset/request_stop/stop/poll/result`。同一 backend 应在所有流结束后再 close。

## 输入、解码和实时边界

- 默认窗口 16、stride=1、投票 7。输入率保持原样，JSON 保存原始 FPS 和时间戳；可靠时使用 POS_MSEC，否则显式记录 FPS 回退。不会静默重采样不同帧率。
- RGB：320 方形 → 中心裁 270 → 224，BGR，[-1,1]。热力图：512 参考坐标 → 170，sigma=8 → 中心裁 143 → 112，[-1,1]。官方 threshold=0 时 Gaussian 不乘 confidence，前端的无人／低可靠单签者门控另行处理。
- 默认左 7、右 8 帧，当前窗只有实际右上下文到达才推理；投票再等待 3 个窗口。没有跨窗口 S3D 特征缓存。`--window` 可配置 16–64 的 8 倍数，但只有 16 属于已训练协议；更改长度会标记实验配置。
- `--stride` 可配置；大于 1 会改变投票对应的时间跨度，且固定在线边缘填充与官方依赖整段长度的填充未必一致，JSON 有明确标志。
- 多数投票后相邻 run 压缩，再删除 blank。`A blank A` 会保留两个 A；同一连续 run 的真实重复且没有 blank 分隔，模型仍可能漏分，不能靠全局去重修复。
- `partial` 可修改或撤回；`final` 表示该预测 run 已结束，仍是模型假设。词级时间是滑窗 anchor 估算，`evidence_time_range` 包含投票用到的左右窗口，不是人工标注边界。
- 分数来自真实三头 ensemble 概率，标注 `uncalibrated_mean_ensemble_probability`；不是已校准正确率。`available_time` 是所需输入的源时间戳，实际交付另看 wall latency，二者不可混用。
- 帧队列默认 32，保留帧最大 512 方形；模型窗口、事件队列、Gloss 历史和统计样本均有上限。输入过快丢旧帧并记录丢帧，遇帧号／时间中断重置局部上下文。可能导致漏词，必须通过真实回放评估。
- 关闭期间若 native 摄像头 read 尚未返回，仍保持运行／释放中状态并阻止重开或释放 backend；不会把线程超时当成摄像头已释放。

默认配置在当前硬件与实验前端下的实际速度见交付报告。能持续采集和使用有界队列，不代表能以原视频帧率准确实时输出。

## 测试、性能和后续扩展

定向测试：

```powershell
.conda-env/python.exe -m pytest tests/test_slrt_protocol.py tests/test_slrt_stream.py tests/test_slrt_capture.py tests/test_slrt_resources_video.py -q
.conda-env/python.exe -m pytest tests/test_slrt_slt_export.py tests/test_slrt_official_layout.py -q
```

资产与官方目录：

```powershell
.conda-env\python.exe scripts\setup_slrt_official_layout.py
.conda-env\python.exe scripts\check_slrt_reproduction_assets.py
```

官方 HRNet 提取（独立 Python 3.8 legacy env，CPU 约 3.6 s/帧）：

```powershell
F:\DSH-envs\slrt-legacy\python.exe scripts\extract_slrt_hrnet_pose.py `
  --video 'D:/path/sign.mp4' --output artifacts/slrt-hrnet/sign.npz --device cpu
.conda-env\python.exe -m slbridge.slrt.cli video --video 'D:/path/sign.mp4' `
  --pose-backend hrnet-npz --pose-npz artifacts/slrt-hrnet/sign.npz --output artifacts/slrt-hrnet/result.json
```

真实源码数值对照（需要本地上游源码、真实视频和实验姿态缓存）：

```powershell
.conda-env/python.exe scripts/verify_slrt_parity.py --video 'D:/path/sign.mp4' --output artifacts/slrt-parity/result.json
```

进程 CPU/RSS 和整张 GPU 利用率采样：

```powershell
.conda-env/python.exe scripts/benchmark_slrt.py --mode video --video 'D:/path/sign.mp4' --pose-backend rtmpose-experimental --output-dir artifacts/slrt-benchmark
```

`--reference '词1 词2'` 只接受实际 Gloss 真值，计算逐词 WER/S/D/I；没有真值不输出准确率。`--require-gloss` 对正样本零 final 返回 3，资源／运行错误返回 2；正常空输出也必须检查窗口和门控计数。测试替身只验证调度机制；黑帧负样本只验证无人门控，均不能冒充真实识别质量。

后续 Libras 需要独立训练权重、词表、语言身份和验证集。Gloss-to-Text 应消费带版本／状态的 Gloss 事件，保留原始视觉结果和生成文本；当前 `translation=null`。已补齐 base mBART、CSL gloss 侧资产与全词表 fallback，以及 `prediction_slide` 结果到 SLT 注释的转换脚本；但作者 `mBART_zh` 的文本剪枝与 CSL-Daily G2T 权重仍缺，`mBART_zh_t2g` 不是已验证的 Gloss-to-Text，full-vocab fallback 也不能当作作者等价复现。LLM 不应向视觉结果补写未识别内容。
