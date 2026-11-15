# 实时网页：当前字幕隔离与采集停滞恢复

日期：2026-11-11，Asia/Taipei。场景依据仍为中巴竞赛的有边界表达、确认与纠错需求；本轮修复上一轮已见的“零帧结束显示旧译文”，不把页面提示或模型非空输出当作语言正确。实现位于动画产品工作树，模型工作树与识别算法未改。

## 已修复的工程行为

- 主字幕只选择当前本地epoch的草稿或已提交句；新尝试、换源、中止不会回退显示上一轮字幕。历史、原文、来源、人工确认与辅助版本保护仍保留。正常完成仍显示本轮译文。
- 收到真实native summary的处理帧数0后再finished，明确显示“本次未采集到视频画面，请确认预览正在播放后重试。”，不提示译文已交付；处理过画面但没有译文/词时另给未识别提示。
- 采集开始读取当前首帧，随后仍按requestAnimationFrame和实际墙钟索引采样，不补造或重编号。独立250ms观察器发现2秒没有JPEG交付进展则停止，提示保持前台、确认播放后重试。它检测采集交付进度，不能判断静态手势与解码画面冻结；系统事件循环阻塞时也不保证严格2秒触发。
- EOF等待编码同样受停滞检测保护；中止会解除pending等待。旧编码回调不能清除新轮次的pending，也不能晚发帧。画布读取异常转为可操作错误，首帧同步错误不能被start的后续成功提示覆盖。
- 折叠的“实时状态”新增已采集/已发送/已处理计数。换源清零，ACK只更新轻量指标，断线/中止发布最终计数。终态同时停止active和draining，后续watchdog不能覆盖已有服务错误。

这改变了浏览器首帧采样时机，模型权重、FPS上限、512方形JPEG、姿态/窗口/分句规则与LLM策略保持。因此仍需真实浏览器回归，不能直接沿用旧0.99秒样例或旧模型成绩作为本轮验收。

## 验证与部署

| 项目 | 结果 | 证据范围 |
|---|---|---|
| 当前client的零帧/旧字幕回退 | 2项RED→GREEN | 真client事件处理，历史确认不丢失 |
| 采集回调与取消/挂起/画布错误 | 5项RED→GREEN | 真Capture与可控时钟、RAF、编码回调 |
| 真页面handler的同步错误覆盖 | 有效RED→GREEN | app+真client；另覆盖EOF/历史 |
| 审查发现的两个Minor | 2项RED→GREEN，关闭 | 排空期间服务错误被覆盖、未ACK断线漏最终计数 |
| 最终定向Node | **43 passed / 0 failed** | 原29项及本轮14项，独立审查也复跑43通过 |
| 产品端真实静态资源 | **4/4字节SHA一致** | HTML/app/client/capture，Cache-Control=no-cache |
| 真实空输入WS→当前client重放 | **通过** | ready→summary(frames=0)→finished；当前client为error/currentCaption=null |
| 本轮真实网页采样/截图 | **未通过** | 浏览器控制连接持续超时，不能用Node或HTTP代替 |

产品静态路由是直接读取磁盘的FileResponse，已经配置no-cache。本轮HTTP回读确认新字节已生效，无需重启。监听PID始终为识别67636、产品45240，未重载模型或更改profile；`final-http/`绑定最后两项审查修复后的源码，而早先`http-readback.json`保留第一次回读。

真实空输入请求经产品WS进入既有模型服务，但未发送任何画面，也未产出word/sentence。原始`final-http/empty-events.json`直接重放到当前RealtimeClient，结果`status=error`、`currentCaption=null`及上述重试提示。这验证协议与client组合，不是浏览器采集或语言质量验收。

初版page-red测试缺少Node环境的显式socket URL，3项ReferenceError/无socket失败属于仪器错误，记录为无效；补注入URL后的`page-red-valid.log`才是同步错误被成功提示覆盖的产品RED。没有把首轮仪器失败算产品缺陷或隐藏失败。

## 浏览器边界仍保留

CUA新建隐藏验证页attach超时，既有tab3可读取旧DOM。该页只含本任务前几轮的3条测试记录，已先保存`browser-before-reload.txt`；尝试刷新加载新模块后，Runtime.evaluate及Emulation.setFocusEmulationEnabled继续超时，未能验证刷新版本、操作修复后的页面或取得新截图。没有复用旧截图，也没有把原页面的旧字幕当新输出。

因此本轮修复有代码、真实模块回读、协议回放和43项定向测试证据，**实际网页采样恢复尚未证明**。不能据“识别服务ready”宣布浏览器可用，不能将上述错误唯一归因为模型、内存或用户操作；控制连接和页面采样需恢复后另验。

## 产物与后续

证据目录：realtime-capture-recovery-20261111（本地产物：`artifacts/realtime-capture-recovery-20261111`）。含4份source-before、各阶段RED/GREEN、独立review、真实HTTP/WS/client重放、监听PID及最终源码绑定`verification.json`。产品改动为`realtime-client.mjs`、`realtime-app.mjs`、`realtime-capture.mjs`、`realtime.html`及测试；两个工作树既有修改保留，未提交/推送。

下一步恢复实际网页操作和采样验收，并继续面向完整交流闭环推进。物理摄像头、新签者、用户原“我想睡觉”模仿、真人语义/手形、双侧中巴任务、24小时稳定性与全仓库/远端CI仍NOT_RUN。本轮也未重新测量首词速度，前轮长时退化根因仍待定位。
