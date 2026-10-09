# 视频、摄像头与字幕开发验收

日期：2026-10-09。执行依据：[专项方案](../视频摄像头输入与字幕_执行方案_2026-10-09.md)，操作入口：[使用说明](../视频摄像头与字幕_使用说明_2026-10-09.md)。

## 1. 交付结果

已实现统一FrameSource、完整录制缓冲、捕获片段到现有模型的推理接口、独立双语词典、后台会话和新Tk界面。文件与摄像头均复用现有模型契约和预测入口。app/desktop_app.py启动新界面，旧原型保留为app/legacy_desktop_app.py。新增可安装的 slbridge-desktop 和 python -m slbridge.desktop_cli 入口。

一个捕获worker独占设备open/read/close；一个模型worker串行执行，队列和预览均有界。epoch/request_id隔离晚到结果。超限、取消、切源、设备断开及拒识不显示上一动作的字幕。

## 2. 实际验证

| 验证 | 结果 |
|---|---|
| 最终完整pytest，启用真实模型一致性测试 | **189 passed**，23.50秒，0失败/0跳过；19条已有依赖弃用警告 |
| 真实视频跨输入路径 | IMG_4868.mp4的119帧经VideoSource/ClipBuffer后，prepared values、mask、时间、概率及最终结果与旧路径完全一致 |
| 真实Tk控件与已发布模型 | 测试集105_Árvore/INFORMANTE 3/IMG_2971.mp4显示“树 / Árvore”，item:105，分数0.89 |
| 词典编辑保存到显示 | 编辑JSON后通过实际重载按钮更新已挂载字幕控件，恢复原词典成功 |
| 实际文件处理期间UI响应 | 界面更新持续，取消清空字幕，关闭后两个worker均退出 |
| 摄像头0连通性 | 读取两帧640×480 BGR，时间递增并释放；不保存摄像头图像 |
| 摄像头0真实桌面链路 | 实时预览→开始录制2秒→结束并推理，得到insufficient_hand_frames质量拒识；中文提示“手部未完整入镜，请重新录制”，葡语为空；取消清空及worker退出通过 |
| 普通wheel分发 | wheel构建成功，实际非editable安装至隔离target，从项目外临时目录加载包资源，12条默认词典及“树”中文存在 |
| 独立审查 | 输入层两项P2、默认词典打包P2、模型显示持久状态和硬件时钟问题均已修复并复核；无剩余P0/P1/P2 |

本地证据在 artifacts/desktop-acceptance-20261009：tests.xml、desktop-validation.json、desktop-file-caption.png、camera-probe.json、camera-desktop-validation.json、wheel-validation.json。摄像头图像未落盘，文件截图仅来自已有Libras测试视频。数据、模型和本地产物不纳入Git。

## 3. 验证中修复的实际问题

- 视频POS_MSEC中途无效时，先预检全段并统一选择回退时间线；不混合媒体时间与fps推导时间。
- 完整解码帧数必须与声明帧数一致；缺帧和多帧都拒绝，预检时间数组最多8MB。
- 预览请求高度曾挤掉字幕，改为仅预览区域伸缩；默认及最小窗口下字幕/按钮可见。
- 模型完成事件可能被切源清空，界面改从会话持久模型同步版本与支持词汇。
- Windows实机time.monotonic使用GetTickCount64，分辨率15.625ms；摄像头快速读帧产生相同时间。默认改为单调的perf_counter（实机QueryPerformanceCounter，分辨率1e-7秒），不伪造epsilon时间、不丢帧。
- 默认词典随包分发，保留cwd可编辑词典优先和--catalog覆盖，普通安装可显示中文草案。

## 4. 模型与工作范围

media.py SHA-256仍为 `a08ed4a430c08b473405b5ebc9d081c2f3a14d5ea707cd3026cd7b44b0c4a3f2`，匹配现有模型的extractor identity；.gitattributes保护该文件字节不被Git换行处理改变。未修改RF权重或原始数据，试点测试准确率仍为63.89%。

硬件录制检查验证了输入、拒识与呈现链路，没有手语动作真值标注，不构成摄像头识别准确率评估。当前仍是完整片段分类，未实现自动分段、连续手语翻译或原生流式模型。中文仍是待核验释义草案。F5完成桌面部分，旧采集／训练脚本迁移另行推进。

## 5. 仓库同步

本地原目录无Git，接续指定远程已有main初始提交 `3519d23e5884bce35dc658462404c0869bc29247` 的历史。源码、配置、测试、文档同步；原数据、模型、虚拟环境和运行产物排除。origin保留指定SSH仓库标识；本机22端口SSH不可用，使用GitHub CLI已登录凭据经HTTPS传输，无强制推送或历史覆盖。
