# SLRT 接入前架构与兼容性分析

日期：2026-10-24。目标与范围来自本次用户给出的 Phase 1–3 开发任务。

## 当前实际架构

- 桌面入口 `slbridge.desktop_cli`，界面是 PySide6，采集和推理通过 `session.py` 的后台线程执行；并非浏览器前端。独立 HTTP 服务为 FastAPI，入口 `slbridge.api_cli`。
- `sources.py` 已提供带时间戳的 BGR `FramePacket`、`VideoSource`、`CameraSource`。`model_adapters.py`、`runtime.py` 提供模型合同、路由和生命周期；可复用文件与采集片段的统一结果合同。
- 现有 RF 接收 MediaPipe225/hand63；CE-CSL TFNet 输出 Gloss；Uni-Sign 输出中文整句。它们的训练、词表、预处理、字幕词典和已有桌面功能均保留。
- 当前连续桌面调度面向完整片段重推和可修订候选，并不负责逐窗口在线 Gloss 提交。SLRT 使用独立流会话和独立可选面板，避免把已有片段候选错误拼成词序列。
- Git 工作区存在大量已有修改与未跟踪文件；本次新增独立 `slbridge/slrt/`，仅在模型注册、入口和依赖说明处做有限接线。修改前保存待改文件副本及哈希，不提交权重、视频、缓存或本次成果到远程。

## 已检查的本地资产与官方协议

- `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt` 为 499,554,423 字节，`torch.load(weights_only=True)` 可解析；`model_state` 有 938 个键。RGB 首层 3 通道，姿态首层 63 通道，RGB/pose/fuse 三个线性头均 2001 类，维度分别为 1024/1024/2048。
- `meta/csl_iso_with_blank.vocab` 是有序 JSON 列表，2001 类，`<blank>` 索引 0。元数据不是视频、HRNet 权重或关键点缓存。
- 官方 `Online/CSLR` 的 `S3D_base.py`、`S3D.py`、`two_stream.py`、`fusion.py`、`Visualhead.py` 和 `recognition.py` 表明这是双向 lateral 的双流 S3D，5 个 block、三个平均池化线性头，ensemble 为三个头 softmax 概率的平均。
- HRNet COCO-WholeBody133 的选点顺序为按名称排序的 hand 91:133、mouth_half 71:91:2、pose 0:11，共 63 点；每点为原图像素 x/y/score。原始 CSL-Daily 方形参考坐标为 512×512。
- 视频按官方 CSL 分支缩放到 320×320，中心裁 270，再至 224；送入 S3D 前 RGB 反转成 BGR，范围映射到 [-1,1]。先由 512 坐标生成 170×170、sigma=8 的 Gaussian，再中心裁 int(270*170/320)=143 到 112，范围映射到 [-1,1]。官方阈值为 0 时热力图不乘置信度，不能擅自“修正”为乘置信度。
- 官方默认 win=16、stride=1，整段长度决定对称边缘复制填充；stride=1 时左 7、右 8 帧。真正在线要等待窗口右侧上下文，EOS 才能复制末帧。多数投票窗口也要额外等待，输出时间必须记录可用时间和估算来源。
- 官方 naive 解码先压缩相邻重复，再删 blank，保留 A blank A。投票配置单独记录，不能全局去重。

## 资源缺口与实施选择

`.conda-env` 已有 Torch 2.9.1+cu128、torchvision、RTMLib、OpenCV、PySide6，CUDA 实测可用；没有 mmpose/mmdet 及官方 HRNet-W48 DARK 和 Faster-RCNN 人体检测权重。压缩包并未提供这些资源。

采用以下接入：

1. 将权重实际使用的 S3D 推理部分放入独立包，保留官方参数路径；严格加载全部 938 个键，不依赖 K400 初始化、训练标签、wandb、TensorBoard、DDP 或数据集文件。
2. 默认姿态模式为官方 `hrnet`，复用旧版 MMPose/MMDetection API，缺配置、权重或依赖时明确失败；另支持带视频 SHA256 和逐帧索引的原始 HRNet133 NPZ 输入。
3. 另提供需显式选择的 `rtmpose-experimental`：RTMPose 与 HRNet 都输出 COCO-WholeBody133，几何和 confidence 可验证，但检测器、估计器和分布不同。真实推理报告必须标注该差异；不称官方复现，不把 MediaPipe225 转成 HRNet。
4. MP4 和在线摄像头共用同一个有限窗口推理器及增量解码器；保留原帧率/时间戳，不把高帧率静默当作官方帧率。流式实现使用窗口内完整计算，明确右上下文等待，后台有界队列、间断重置和 EOS flush。
5. 复用现有 `ModelAdapter` 文件合同；流会话另提供 feed/pause/resume/reset/flush/close。新独立 CLI 和 Qt 面板可选 SLRT-CSL，原入口和 Legacy 功能保留。

## 验证边界

先验证真实权重严格加载，再对真实 MP4 运行；随机输入只验证数值/结构，不作为功能或质量验收。单元测试覆盖错误协议、词表、blank/重复、未来帧不可见、重置、EOS、积压及释放。记录真实权重/视频/前端的哈希、时间与设备。没有对应标注不报告 WER；现场摄像头和无手语误触发需要各自实测。
