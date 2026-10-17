# 关键点采集与回放开发验收（2026-10-11）

本次完成独立关键点工作台、共享采集协议、录制文件、回放和 CLI。范围遵循“现不考虑模型算法”：复用现有 MediaPipe 检测关节，不增加 DTW、TCN、GRU、词义识别、训练或字幕生成。工作区基线为 `de316f52bc618387e0536daeee80899b947fd5bd`，本次代码尚未提交。

## 可使用的功能

- 独立启动：`.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --keypoints`；支持 `--video` 或 `--camera`，主界面另有入口。
- 视频／摄像头持续采集，原画面骨架叠加或纯骨架预览；身体 33 点、双手各 21 点，显示有效点、肩锚、处理用时、队列和丢帧。
- 开始录制、完成保存、取消；标签／手语／示范者仅保存为 draft 备注。支持采集中的暂停、继续、跟踪重置。
- `.slkp.jsonl` 严格检查与按源时间回放；回放不重新创建 MediaPipe 提取器，不把历史处理时间当作当前帧龄。
- `python -m slbridge.keypoint_cli extract --video ... --output ...` 全帧离线提取，`inspect <file>` 独立检查。

操作与协议见 [使用说明](../keypoint-workbench-usage.md)。关键实现为 `slbridge/keypoints.py`、`keypoint_session.py`、`keypoint_recording.py`、`keypoint_cli.py` 和 `keypoint_desktop.py`；旧分段观察器增加一个共享原生结果出口，既有返回接口保留。

## 数据与资源约束

新 schema 为 `pose_hands_xy75_stream_v1`，75 个二维点，分别保存公共图像坐标与肩部归一化坐标及逐点掩码。没有深度或虚构的手部逐点置信度，不能直接作为旧 225/321 维特征输入。

保留会话／来源／跟踪身份、源帧号、时间域、时间来源、跳帧和重置原因。视频记录 `pos_msec` 或 `fps_fallback`，摄像头记录应用单调时钟。协议头部包含配置、后端版本、实现文件 SHA256 和协议哈希。

采集与检测各一个资源所属线程，输入队列最多 2 帧、事件队列最多 48 条，预览与结果保留最新一项。切源、暂停、重置和关闭均使旧的在途结果失效。已经进入原生调用的操作要等调用返回，才能在其所属线程释放资源；本次没有增加强制终止原生调用的能力。

录制上限 100000 帧／256 MiB，行上限 1 MiB；先写同目录唯一 partial，完成时以硬链接发布，既有文件不覆盖。读取校验 identity、版本、严格时间／帧号、点／掩码、完整尾部与数据 SHA256。哈希用于完整性校验，不证明采集真实性。仅保存坐标和元数据，不保存 RGB 视频或音频。

## 自动化验证

完整本地回归命令：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.venv-libras/Scripts/python.exe -m pytest -q -rs -p no:cacheprovider --basetemp artifacts/keypoint-delivery-verified-pytest-20261011-a --junitxml artifacts/keypoint-workbench-pytest-verified-20261011.xml
```

最终结果为 **667 passed、10 skipped、15 warnings，50.69 秒，exit 0**；其中新增关键点六模块 116 项。跳过的 9 项 TFNet 导出测试因本环境缺少 Torch；另一项旧模型跨路径验收未设置 `SLBRIDGE_LIVE_VIDEO`／`SLBRIDGE_LIVE_MODEL`。这些项目不能计为本次通过，既有依赖弃用警告未修改。

新增测试覆盖一次原生提取、逐点缺失、时间／身份校验、重置、背压与迟到结果隔离、源时钟不能通过 reset 绕过、保存读回／损坏／上限／竞争不覆盖、Qt 控件流程和独立无识别模型启动。最终默认 Windows Qt 桌面测试另行 **7 passed，2.47 秒，exit 0**；截图像素断言已正确按 DPR 换算，产品绘制未为测试改色。审查与修正记录见 [独立评审](keypoint-workbench-review-2026-10-11.md)。

磁盘写入与 partial 清理同时失败的会话恢复另有测试：采集线程继续，重新打开来源后可以恢复。录制器保留主异常类型／文本并报告清理因果；CLI 错误 JSON 保留两项信息。完整目标已经通过硬链接发布后，清理失败不将它改判为保存失败、不删除目标文件，返回保存成功和 `cleanup_error`；会话、Qt 及 CLI 对应显示已保存并报告临时残留。真实录制器的故障注入、会话读回和 Qt 提示测试均包含在最终回归。

最终回归前曾出现 **1 failed、666 passed、10 skipped**，失败项是旧 `test_model_adapters.py` 的 deadline-during 测试。独立实测 40 次 `sleep(.09)` 中 7 次 `time.monotonic()` 仅增加 0.078 秒，实际 `perf_counter()` 已经过约 0.09 秒；本环境 `GetTickCount64` 粒度为 0.015625 秒，因此该测试没有保证 0.08 秒截止时间已过。只将此测试的请求时钟替换为受控时钟，在真实 adapter 线程被阻塞时推进至 deadline 后再释放，生产超时和模型代码未改。四项参数化取消／超时检查通过，之后完整复跑得到上述最终结果。失败快照保留为 `artifacts/keypoint-workbench-pytest-final-20261011.xml`，最终成功快照为 `artifacts/keypoint-workbench-pytest-verified-20261011.xml`。

## 真实视频 CLI 验收

对 `D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4` 使用真实 MediaPipe CPU XNNPACK，保存并在独立进程检查，重新解码原视频逐帧核对。

- 全部 **247 帧**保存读回，源索引、时间戳和身份完全一致；时间范围 0–8.2 秒、跳帧 0。
- `extract` 总墙钟 13.290 秒；`inspect` 0.242 秒。
- 单帧 packet 平均 40.133 ms、P95 50.431 ms；首帧含初始化为 1311.243 ms。

这是一次本机离线全帧提取，不证明实时 30 FPS，也不能与旧 CTC 的整段识别速度作等价算法比较。完整质量、输入哈希、证据和复现脚本见 [CLI 验收报告](keypoint-video-cli-2026-10-11.md)。

## 真实视频 Qt 控件流程

使用生产 VideoSource、KeypointSession、MediaPipe、录制器及读取器，仅由脚本提供文件选择结果，用 QTest 点击已挂载控件：

```powershell
$env:QT_QPA_PLATFORM='windows'
.venv-libras/Scripts/python.exe experiments/validate_keypoint_workbench.py --video D:/Zenodo/CE-CSL/CE-CSL/video/test/A/test-00001.mp4 --output-dir artifacts/keypoint-qt-windows-final-20261011-a
```

默认 Windows Qt 平台在最终代码上再次执行，**exit 0**。导入视频 → 录制 → 重置 → 暂停 0.3 秒（计数冻结）→ 继续 → 完成保存 → 停止 → 打开回放 → 关闭线程走通；保存 23 帧，回放 23 帧，最终回放 packet 与保存 packet 字段完全一致。记录的 `source_timing_mode` 为 `pos_msec`，备注保持 draft，保存文件 157636 字节。

本轮同时运行完整回归，采集处理不及输入；有界队列丢弃 56 帧并记录对应缺口，文件保留三个跟踪段。这验证背压及记录行为，不作为速度基准。回放无丢弃、处理用时和当前帧龄为 null。退出后两条所属线程均已关闭。

证据：`artifacts/keypoint-qt-windows-final-20261011-a/acceptance.json`、同目录 `qt-real-video.slkp.jsonl` 与截图；另有 offscreen Qt 流程也以 exit 0 完成。offscreen 截图缺少中文字体渲染，以下为默认 Windows Qt 截图，已检查中文显示与骨架叠加／纯骨架回放。

![真实视频关键点叠加](E:/githubproject/SignLanguage-Bridge/artifacts/keypoint-qt-windows-final-20261011-a/capture.png)

![保存后的骨架回放](E:/githubproject/SignLanguage-Bridge/artifacts/keypoint-qt-windows-final-20261011-a/replay.png)

这里是 Qt widget 控件自动化验收，文件对话框结果由脚本提供；没有通过操作系统自动化验证真实文件对话框，也没有连接实体摄像头。

## 尚未验证或未纳入范围

实体摄像头、设备断开后的硬件重连、真实使用者动作、语义／准确率、摄像头到屏幕的端到端延迟没有验收。跨平台 GUI、非 NTFS 文件系统也未验收；不支持硬链接时保存明确失败。

没有增加新识别算法、GPU 关键点后端或训练接口。后续算法可以消费版本化的逐帧轨迹和质量信息，但需要独立数据、契约与验证，不能将本次采集标签或质量比例当作识别结果。
