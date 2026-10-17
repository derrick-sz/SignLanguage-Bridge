# 关键点工作台独立代码审查（2026-10-11）

范围：`keypoints.py`、`keypoint_session.py`、`keypoint_recording.py`、`keypoint_cli.py`、`keypoint_desktop.py`、桌面入口与最小 `segmentation.py` 修改，以及对应测试。仅审查算法无关的采集、时序、录制、回放和界面；未训练、未使用摄像头、未评价识别准确率。

最终结论：初审的三项 P2 已修正并复核，未发现残留的代码阻断问题。修正了独立 Qt 像素测试在 Windows 125% 缩放下的取样错误；没有为测试修改产品绘制。

## 初审发现与处理

1. **P2：暂停漏计播放时钟，且可能丢掉待显示帧。** 初审位置 `keypoint_session.py` 的 `_capture()` 定时等待（当时第 312–318 行）。对媒体时间为 0/1 秒的受控视频源，在实际 0.145 秒暂停、0.495 秒继续，第二帧仍在 1.008 秒处理，遗漏了 0.350 秒暂停时长。若暂停跨越目标展示时刻，`self.paused` 分支会直接 `continue`，丢掉已读取的帧。**已修复并复核：** pause/resume 在调用时记录累计暂停时间；`_wait_capture()` 以最多 20 ms 等待检查暂停和停止，保留待展示帧，并在交付锁内重新核对平移后的 deadline。新增 0.25 秒与 1.1 秒暂停回归通过，第二帧未丢失，媒体时间保留为 1 秒。此等待逻辑同时用于视频与录制回放；新增计时测试使用受控视频源。
2. **P2：桌面录制缺少视频时间来源。** CLI 已保存 `source_timing_mode`，桌面 `start_recording()` 初审只有来源类型和路径；`VideoSource` 的 `pos_msec` 与 `fps_fallback` 因而不可区分。**已修复并复核：** 来源打开后保存其 `timestamp_mode`，摄像头记录 `application_monotonic`，并在桌面录制元数据中保留；`fps_fallback` 保存读回回归通过。
3. **P2：直接桌面入口可能打开错误摄像头。** `--keypoints --camera N` 初审直接写入范围为 0–32 的 `QSpinBox`，Qt 会把 -1/100 静默改成 0/32。**已修复并复核：** 在 Qt 创建前解析错误拒绝范围外编号；-1、33、100 参数回归通过。
4. **测试问题：默认 Windows Qt 像素断言忽略 DPR。** 初次默认图形环境测试为 5 passed、1 failed。受控实测：400×220 widget 的 grab 是 500×275，DPR=1.25；逻辑点 (100,70) 直接读取图像 (100,70) 为黑色，正确物理点 (125,88) 为骨架色 `#008bba`。按 root 授权，仅将 `tests/test_keypoint_desktop.py` 三处截图采样改为乘 `pixmap.devicePixelRatio()`；默认 Windows Qt 六项测试全部通过，offscreen 回归亦通过。

## 已检查的边界

- 来源 epoch 与跟踪 version 均在旧 native 调用返回后检查；切源、暂停、重置或关闭期间旧结果不会发布或写入新录制。native 资源由各自工作线程关闭。
- 采集队列最多 2 帧、事件队列最多 48 条、预览和关键点各保留最新一项；没有长期积累视频或关键点历史。录制逐帧写入，有帧数、文件大小和行大小上限。
- 录制固定 source/session/epoch/time domain，检查源时间与帧序严格递增、跟踪 epoch 不回退；重置不等于可绕过源时间次序。
- 完成文件先写唯一 partial，帧哈希和 footer 验证后通过同目录硬链接提交；既有文件和提交竞争不覆盖，写入失败、取消和空录制正常清理 partial；文件锁或权限阻止清理时保留错误提示，不能宣称 partial 已删除。回放前完整扫描验证；回放使用保存坐标，没有再次创建提取器。
- 新协议只含 pose + 双手的二维坐标、显式缺失掩码、肩锚、时间和质量信息；无深度估计、无 225/321 兼容伪装、无识别词义。备注保持 draft，不能把采集标签伪装成已审核结果。
- 桌面使用原始公共图像坐标绘制骨架，画面与关键点 frame identity 不同时不叠加；采集与录制错误在界面显示，可恢复输入，EOF 保留完成保存按钮。
- 独立 `--keypoints` 入口在模型和字幕目录选择前进入工作台，工作台不会构造识别模型。

## 验证记录

初次沙箱 pytest：51 passed、40 个临时目录 setup errors，原因是 pytest 临时目录 `.lock` PermissionError，不能记为功能通过。第二次正常权限执行使用不存在的临时父目录，仍为 setup errors，未产生产品失败结论。

正常权限、项目专用临时目录复跑：

```text
.venv-libras/Scripts/python.exe -m pytest tests/test_keypoints.py tests/test_keypoint_session.py tests/test_keypoint_recording.py tests/test_keypoint_cli.py tests/test_keypoint_launcher.py -q -p no:cacheprovider --basetemp .pytest-keypoint-review-20261011-2
95 passed in 2.00s
```

三项修正与 DPR 测试修改后的独立复核（下述清理异常链补强前）：

```text
# 默认 Windows Qt 图形环境，DPR=1.25
.venv-libras/Scripts/python.exe -m pytest tests/test_keypoint_desktop.py -q -p no:cacheprovider --basetemp .pytest-keypoint-review-20261011-qt-dpr
6 passed in 1.95s

# QT_QPA_PLATFORM=offscreen；六个关键点模块
.venv-libras/Scripts/python.exe -m pytest tests/test_keypoints.py tests/test_keypoint_session.py tests/test_keypoint_recording.py tests/test_keypoint_cli.py tests/test_keypoint_launcher.py tests/test_keypoint_desktop.py -q -p no:cacheprovider --basetemp .pytest-keypoint-review-20261011-final
107 passed in 5.90s
```

受控图源、Qt 控件交互和注入提取器验证不等于真实摄像头、真实 MediaPipe 性能或真人手语识别验收。真实视频 Qt 流程的验收由 root 另行完成，不属于本独立审查证据。

## 追加：录制写入与清理同时失败的恢复

对 root 随后的 `keypoint_session.py` 小改动进行了只读复核：`_cancel_recording()` 在先解除录制引用后捕获 `abort()` 清理异常，返回清理错误描述并发出错误与取消事件；写入失败和采集失败路径将此描述并入实际错误；`finish_recording()` 也捕获清理异常而继续抛出其收到的 finish 异常。因而清理失败不会再沿 `_cancel_recording()` 杀死 owner worker，后续输入和切源仍可执行。

新增 `BrokenRecorder` 回归同时注入 append 的 `disk full` 与 abort 的 `partial locked`，检查错误均可见、录制解除、处理帧数继续增加和切源恢复。root 报告相关 51 项测试通过；本次追加独立工作为代码及测试阅读，没有自行再执行该组测试，不将其并入上面的 107 项独立执行计数。

历史发现：当时真实 `KeypointRecorder.append()/finish()` 内部仍有 `except Exception: self.abort(); raise`，其内部 abort 失败可能在进入 session 前覆盖原始写入/提交异常。此具体链条已告知 root，随后修正，现已撤销为未解决限制。

## 最后复核：内部清理异常链与成功保存警告

按 root 指示，仅复核最后相关代码和测试，未扩大范围或修改产品、测试：

- `KeypointRecorder._abort_after_failure()` 保留主异常类型和文字，把清理失败保存为 `keypoint_cleanup_error`，通过异常 cause 和 note 附带清理错误。`abort()` 对 close 与 unlink 同时失败也保留两者的诊断，之后可再次尝试清理。
- `finish()` 将成功发布作为边界：硬链接成功后，partial 清理失败仍返回有效的目标路径，设置只读 `cleanup_error`；既不删除目标，也不把保存成功误报成未保存。保存目标可以完整读回，partial 的残留得到明确说明。
- CLI 仅在必要时增加成功结果的 `cleanup_warning` 或失败 JSON 的 `cleanup_error`，主错误类型和文本不被清理错误替换。
- session 先发 `recording_saved` 再发 `recording_cleanup_warning`；Qt 保留已保存路径与录制标签，状态显示已保存及清理提示。真实 Recorder→Session 集成测试验证读回有效目标、录制解除和两种事件；Qt 控件测试验证警告不覆盖已保存状态。

已阅读新增的 append/finish 内部主异常保留、close/unlink 双故障、成功发布后清理失败、CLI 结果与真实 Recorder→Session / Qt 提示回归。recording agent 报告相关 42 项通过，root 报告默认 Windows Qt 7 项通过；这两组不是本审查者独立执行，不与此前 107 项相加。root 的最终全量回归正在另外执行。

最后结论：此前内部链诊断限制已解决，最后这条链未发现残留代码阻断。异常测试使用注入文件操作失败，验证控制流与诊断，不等同于真实磁盘耗尽、系统文件锁或权限故障现场验收。
