# 关键点视频 CLI 实际验收（2026-10-11）

本次 **通过真实 MediaPipe 提取 → 保存 → 独立进程检查 → 逐帧严格读回**。验证对象是离线视频关键点采集基础功能，未加载手语识别模型，未验证真实摄像头或手语准确率。

## 复现入口与证据

在项目根目录运行：

```powershell
.\.venv-libras\Scripts\python.exe experiments/validate_keypoint_video.py
```

脚本每次创建新的输出目录，不覆盖已有录制。可通过 `--video` 和 `--output-dir` 指定其他样本与产物位置。

- 脚本：`experiments/validate_keypoint_video.py`
- 成功产物：`artifacts/keypoint-video-cli-20261011/20261011T151227Z-f1a8e9e7/`
- 完整机器可读证据：该目录的 `acceptance.json`
- 已保存文件：`test-00001.slkp.jsonl`
- 子进程输出：`extract.stdout.json`、`inspect.stdout.json`
- 实际 MediaPipe 日志：`extract.stderr.log`；包含 `Created TensorFlow Lite XNNPACK delegate for CPU`。

验收时工作区基于 `de316f52bc618387e0536daeee80899b947fd5bd`，包含本次尚未提交的关键点基础功能。录制元数据保存协议哈希和关键实现文件 SHA256，不能仅凭基线提交判断当前实现。

## 样本与环境

| 项目 | 实测值 |
| --- | --- |
| 输入 | `D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4` |
| 输入 SHA256 | `d0279bde09688fe1edae603d8e7c3e41cd93342cee5982fdef357c76e0bdb45b` |
| 输入文件大小 | 2,846,233 字节 |
| 完整解码帧数 | 247 |
| 原始图像尺寸 | 1920 × 1080 |
| 原始帧索引 | 0 至 246 |
| 首／末帧媒体时间 | 0.0／8.2 秒 |
| 时间源 | `pos_msec`，无 FPS 回退 |
| MediaPipe | 0.10.14，持久 Holistic 图，CPU XNNPACK |
| OpenCV | 4.10.0，distribution 为 `opencv-contrib-python 4.10.0.84` |
| NumPy / Python | 1.26.3 / 3.12.7 |
| 操作系统 / 逻辑 CPU | Windows 10 19045 / 20 |

关键点提取配置使用现有默认 `max_width=640`。录制中的图像坐标仍对应共同的原始图像坐标系；不把缩放后的检测图尺寸冒充原始帧尺寸。8.2 秒是首末帧时间跨度，不是容器时长测量。

## 保存与逐帧读回

独立运行的 `extract` 与 `inspect` CLI 均返回 0，两个摘要完全相等。再次用 `KeypointRecording` 读完整文件，与重新解码的原视频逐帧比对：

- 247 帧全部保存并读回；源索引、时间戳、源身份逐帧完全一致。
- 严格读取器接受完整头部、帧数据及校验尾部，拒绝规则由单元测试另行覆盖。
- 缺失点均为零坐标加 `false` 掩码；没有把缺失点当作有效观测。
- 数据 schema 为 `pose_hands_xy75_stream_v1`，身体 33 点加双手各 21 点；每点为二维坐标。
- 示例标签的 `review_status` 保持 `draft`；标签只是采集注释，没有产生识别结果。
- 文件为 1,739,468 字节；内容为坐标及元数据，没有 RGB 视频帧。

录制帧数据 SHA256：

```text
20f544ac6c4cc3cf2e3c59827ceece6a4204f6efebd800e1ba2f64819dc7b675
```

## 质量与重置信息

| 记录项 | 实测值 |
| --- | --- |
| 有效身体归一化锚点 | 247 / 247 帧 |
| 返回完整左手组 | 234 / 247 帧 |
| 返回完整右手组 | 218 / 247 帧 |
| 有效图像观测点 | 13,551 / 18,525 点 |
| 身体／左手／右手有效点 | 5,659 / 4,012 / 3,880 |
| 有效归一化点 | 13,551 点，与图像掩码一致，因为本样本所有锚点有效 |
| 有效运动观测 | 232 / 247 帧 |
| 源帧跳过数 | 0 |
| tracking epochs | `[0]` |
| 重置原因 | `initial` 1 次，无额外自动重置 |

“返回完整手部组”只表示检测器返回了结构完整的 21 点组；有效逐点掩码另行统计。这些比例不能作为手部检测准确率，因为本次没有人工关键点真值。

## 用时与解释

| 测量范围 | 实测值 |
| --- | --- |
| `extract` 子进程总墙钟时间 | 13.290 秒 |
| `inspect` 子进程总墙钟时间 | 0.242 秒 |
| 247 帧 packet 处理时间总和 | 9.913 秒 |
| packet 平均 / 中位数 | 40.133 / 33.366 毫秒 |
| packet P95 | 50.431 毫秒 |
| packet 最小 / 最大 | 27.869 / 1,311.243 毫秒 |

首帧索引 0 的处理时间为 1,311.243 毫秒，与最大值相同，包含首次原生图初始化等开销。packet 处理时间从进入 `KeypointExtractor.process` 前的接收时刻到提取结果时刻，包含 MediaPipe 和数据转换；CLI 总用时还包括进程启动、视频预解码、JSONL 写入和严格检查。

这是单个样本的一次本机测量，不能据此宣称实时 30 FPS，也不能与旧 CTC 整段识别耗时作等价算法比较。实时工作台采用有界队列及逐帧质量信息；本验收没有实际连接摄像头，也未测量摄像头到屏幕的端到端延迟。

## 验收脚本修正记录

首次运行的真实 CLI 提取和检查已经完成，但脚本收集环境版本时假定存在 `opencv-python` distribution，实际安装的是 `opencv-contrib-python`，因此未生成成功验收 JSON。脚本已改为按 `cv2` 的实际 distribution 记录版本；以上数据均来自修正后完整重跑且退出码为 0 的一次验收。
