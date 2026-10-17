# 关键点采集与回放

工作台用于观察身体和双手关键点，录制动作轨迹并检查、回放保存的数据。它不加载手语词义识别模型，尚未接入 DTW、TCN、GRU、动作分类或字幕生成。采集使用现有 MediaPipe Holistic 关键点检测器；回放直接读取已保存的坐标，不再运行检测器。

## 启动

在项目根目录使用 Python 3.12 和桌面依赖：

```powershell
.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt
.venv-libras/Scripts/python.exe -m pip install -e .
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --keypoints
```

`--keypoints` 直接打开独立工作台，不需要准备识别模型目录或字幕词典。也可启动时打开自己的视频或摄像头：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --keypoints --video "D:/videos/sample.mp4"
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --keypoints --camera 0
```

`--video` 与 `--camera` 二选一。摄像头设备编号允许 0–32，0 通常是默认设备；编号可选不代表设备已经连接。关键点模式不能与 `--model`、`--catalog` 或 `--segment-config` 同时使用。原识别模型的 `--device` 等参数不配置此采集器。

已有桌面主界面也提供“关键点工作台”入口。打开时会停止主界面的输入来源，工作台关闭后可重新启动主界面来源。

## 采集、保存和回放

1. 点击“导入视频”或选择设备后点击“使用摄像头”。让双手和上半身进入画面，等待第一帧处理完成。
2. 查看叠加骨架及左手、右手有效点数和肩锚状态。勾选“仅显示骨架”可隐藏背景画面。
3. 可填写样本标签、手语种类、示范者编号，再点击“开始录制关键点”，选择一个尚不存在的 `.slkp.jsonl` 文件路径。未填写的字段可保持为空。
4. 动作完成后点击“完成并保存”。窗口显示保存路径后，本次录制才形成可读回的完整文件。视频播放结束后，已录制的数据仍需点击该按钮保存。
5. 点击“打开关键点录制”，选择保存的文件，按记录的源时间间隔回放骨架。支持“暂停”“继续”；回放不需要原始视频，也不能再次作为采集源录制。

录制开始之前的帧不会补入文件；只保存录制期间成功处理的关键点帧。实时工作台的输入队列最多容纳 2 帧，预览只保留最新结果。处理赶不上输入时会丢弃旧帧，并保留帧编号、时间差和跳帧信息。因此它适合交互采集；需要提取视频每一帧时，使用下方 `extract` 命令。

“暂停”冻结结果交付，继续后重新建立跟踪；视频和录制回放的播放时钟也随暂停后移。“重置跟踪”可以在采集、录制过程中使用，新的跟踪段通过 `tracking_epoch` 和重置原因区分。重置后的首帧不提供有效运动差分。切换输入来源会更换来源代次，并丢弃旧来源迟到的处理结果。

点击“取消录制”会删除本次临时数据。录制期间需要先完成或取消，才能切换或停止来源；关闭工作台会取消尚未完成的录制。空录制、写入失败或超限不会发布为完整文件。

保存内容为关键点、质量、时间和采集配置，**不包含原始 RGB 视频或音频**。样本标签只是人工采集备注，统一标记为 `review_status: "draft"`；选择“中国手语”或“Libras”不会核验动作语义，也不会建立识别能力。

## 命令行提取与检查

安装上述依赖后，可在没有图形界面的环境逐帧提取视频。输出目录须已存在，输出文件须尚不存在：

```powershell
New-Item -ItemType Directory -Path artifacts/keypoints -Force | Out-Null
.venv-libras/Scripts/python.exe -m slbridge.keypoint_cli extract --video "D:/videos/sample.mp4" --output artifacts/keypoints/sample.slkp.jsonl --label sample-001 --language csl --performer signer-01
.venv-libras/Scripts/python.exe -m slbridge.keypoint_cli inspect artifacts/keypoints/sample.slkp.jsonl
```

`--label`、`--language` 和 `--performer` 均可省略。它们同样是未经审核的采集备注。`extract` 同步处理视频各帧，完成后重新读取并校验文件，返回 JSON 摘要；`inspect` 只校验现有录制并返回帧数、时间范围、质量统计、配置和哈希，无需运行 MediaPipe。失败时返回非零退出码和 JSON 错误。

两条命令均支持 `--max-frames` 和 `--max-bytes`。例如限制为 1,000 帧、64 MiB：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.keypoint_cli extract --video "D:/videos/sample.mp4" --output artifacts/keypoints/bounded.slkp.jsonl --max-frames 1000 --max-bytes 67108864
```

超过任一上限时拒绝整份录制，不会静默保存截断文件。安装当前版本后，也可使用 `slbridge-keypoints inspect ...` 或 `slbridge-keypoints extract ...`。

## 数据和质量含义

录制格式为 `slbridge.keypoints.jsonl` 第 1 版，关键点 schema 为 **`pose_hands_xy75_stream_v1`**。按身体 33 点、解剖左手 21 点、解剖右手 21 点排列，每点仅有二维 `xy`。它是独立的流数据协议，不能通过补零、截断或改形状直接充当旧模型的 225/321 维输入。

| 字段 | 含义 |
| --- | --- |
| `image_xy` | 输入图像的共同坐标系，按图像宽、高归一化到 `[0, 1]`，不镜像；用于骨架叠加。 |
| `point_observed` | 75 个逐点有效标记。缺失点写为 `(0, 0)`，必须结合此掩码使用，不能将零值当作有效位置。 |
| `normalized_xy`、`normalized_observed` | 按图像宽高比校正后，以双肩中心、肩宽归一化的坐标及有效标记；不含深度。 |
| `anchor_valid` | 双肩均有效且肩宽足够时为真。肩锚缺失时，身体归一化坐标全部无效，图像坐标仍可独立使用。 |
| `pose_visibility` | 检测器提供的 33 个身体点可见性。手部没有编造逐点置信度。 |
| `left_present`、`right_present` | 检测器是否返回对应手部点组；仍需结合逐点掩码检查坐标有效性。 |
| `frame_index`、`dt`、`frames_skipped` | 源帧编号、相邻已处理观测的源时间差及中间跳过的帧数。 |
| `session_id`、`source_id`、`source_epoch`、`tracking_epoch` | 会话、来源与跟踪段身份，用于隔离切源和重置前后的轨迹。 |
| `motion`、`motion_valid`、`reset_reason` | 已观察到的手部归一化点运动差分及有效性、重置原因；不是手势分类结果。 |

视频时间域为 `media_seconds`，优先使用解码器时间戳；如果视频时间戳不严格递增，来源会记录 `fps_fallback`，改用帧率推导时间。摄像头时间域为 `application_monotonic`，使用应用读帧时的单调时钟，不能当作硬件曝光时间。具体模式保存在录制头部的 `source_timing_mode`。

界面“提取”统计提取调用耗时；“结果年龄”从应用读帧完成算到关键点处理完成，可能包含排队和视频播放等待。二者都不是摄像头到屏幕、手语到字幕的端到端延迟。界面时间差反映观测间隔，不能单凭它判定丢帧原因；应结合源帧编号、跳帧和跟踪重置记录。回放不把历史单调时钟当作当前帧龄，相关处理耗时显示为“—”。

## 文件完整性和资源边界

默认录制、读取上限为 **100,000 帧、256 MiB**，单行上限为 1 MiB。文件包含头部、逐帧数据及尾部；尾部记录帧数和逐帧数据行的 SHA-256。读取时检查版本、身份、时间及帧号递增、坐标和掩码形状、完整尾部、哈希及资源上限。

保存先写同目录下的唯一 `.partial` 文件，完成后通过硬链接提交，已有目标文件不会被覆盖。目标目录须支持硬链接；本地 NTFS 可使用，不支持硬链接的文件系统或挂载位置会导致保存失败，应改选兼容的本地目录。取消、异常或超限会清理本次临时文件；若文件被锁定或目录权限阻止清理，界面会报告错误，采集可恢复。如果完整文件已成功发布、仅临时文件清理失败，界面会显示“已保存”及清理提示，CLI 返回 `cleanup_warning`；完整文件仍可回放。清理失败或进程被强制结束时可能残留 `.partial`，它不是用于回放的正式录制。

该功能提供采集、质量观测、数据保存和回放基础。软件测试与真实视频验证的结果见 [开发验收报告](reports/keypoint-workbench-2026-10-11.md)；摄像头硬件、真人动作语义及识别准确率需另行验证。
