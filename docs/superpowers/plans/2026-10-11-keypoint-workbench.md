# Keypoint Workbench Implementation Plan

> **For agentic workers:** Implement the authorized plan in this session with parallel ownership and an independent review. Steps use checkbox tracking; do not commit unrelated research or write to Git metadata.

**Goal:** 不加载识别模型即可持续采集、观察、录制并回放版本化关键点。

**Architecture:** keypoints模块封装持续提取与不可变帧，recording模块封装安全有界文件。KeypointSession以固定两线程和最新slot管理资源/来源隔离，独立Qt工作台和CLI复用这些模块。

**Tech Stack:** Python≥3.10；已有MediaPipe0.10.14/OpenCV/PySide6；JSONL；pytest。

**Spec:** ../specs/2026-10-11-keypoint-workbench-design.md

## Global Constraints

- 新schema `pose_hands_xy75_stream_v1`；75点二维，逐点mask；不转换225/321输入。
- 输入队列最多2帧；预览与帧各最新slot；不全量缓存流。
- 录制上限100000帧/256MiB；不覆盖既有文件，不发布中止partial。
- labels只能draft；不增加识别算法、词义字幕或精度/延迟宣传。
- 保留现有未提交docs和审计脚本，本次不stage/commit/push。

## Review Focus

- Native调用尚未返回时切源/reset：旧结果不得进入预览/录制。
- 录制完成/磁盘错误/达到上限：只能完整保存或明确取消。
- EOF后仍有待处理帧：必须排完再报告回放/采集结束。
- 手部局部缺失/尺寸变化：质量来源明确，差分与跟踪重置。
- 带摄像头历史monotonic时间的回放：不重新计为当前帧龄。

### Task 1: 持续关键点协议

Files: keypoints.py、segmentation.py、test_keypoints.py。
Interface: `LandmarkFrame.to_dict/from_dict`；`KeypointExtractor.process(packet,received_monotonic=None), reset,close,metadata`。

- [x] 测试一次process、部分缺失、真正逐点mask、时间/源严格性、手动/gap/尺寸reset、JSON读回及不可变值。
- [x] 实现新packet与共享原图结果出口，旧observe返回不变。
- [x] 执行核心与旧segmentation tests。

### Task 2: 安全录制与CLI

Files: keypoint_recording.py、keypoint_cli.py及对应tests。
Interface: `KeypointRecorder(path,metadata=...).append/finish/abort`；`KeypointRecording(path).summary/iter_frames`；`python -m slbridge.keypoint_cli inspect|extract`。

- [x] 测试同源单调、footer/hash/损坏/超限、空文件/不覆盖、取消清partial。
- [x] 实现流式JSONL及inspect/extract，不加载识别模型。
- [x] 运行录制/CLI tests，读回实际保存文件。

### Task 3: 会话资源与背压

Files: keypoint_session.py、test_keypoint_session.py。
Interface: open_source/open_recording、pause/resume、start/finish/cancel_recording、reset_tracking、latest preview/keypoints、poll/status、close。

- [x] 测试无模型录制读回、慢提取有界丢帧、切源迟到隔离、reset延迟隔离、回放不创建提取器、线程资源归属。
- [x] 实现固定采集/提取worker；epoch/version检查在native返回后再次执行。
- [x] 运行会话及相关旧来源/会话回归。

### Task 4: 桌面入口与工作台

Files: keypoint_desktop.py、desktop.py、test_keypoint_desktop.py。

- [x] 实现独立工作台、骨架预览、质量/耗时、录制保存/取消、回放暂停/继续、主窗口互斥入口。
- [x] 注入fake session操作Qt控件，验证保存/打开/释放路径；旧桌面回归。
- [x] 实际视频提取并回放检查，独立代码评审修复实质问题。

### Task 5: 交付

- [x] 加slbridge-keypoints可安装CLI入口与用户文档。
- [x] 汇总真实通过/失败/跳过/未测，保留验收产物。
- [x] 再核查工作区diff只含授权代码和新增文档。
