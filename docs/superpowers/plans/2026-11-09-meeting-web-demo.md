# 会议字幕 Web Demo Implementation Plan

> **For agentic workers:** Use inline execution and one fresh code review after implementation. Steps use checkbox syntax for tracking.

**Goal:** 在现有产品交付可运行的屏幕共享、视频框选、分段识别和字幕记录Demo，复用正在优化的识别服务。

**Architecture:** 新会议页面使用独立选区、采集、字幕会话模块，复用既有RecognitionController。保留原辅助页和API；前端按返回任务类型显示词序列或句子文本。

**Tech Stack:** 原生ES modules、Canvas、MediaRecorder、现有FastAPI和Node test runner。

**Spec:** [design](../specs/2026-11-09-meeting-web-demo-design.md)

## Global Constraints

- 活动工作树复用；保留所有既有未提交手形修订和另一会话模型修改。
- 不启用摄像头或音频，不更换模型权重，不发布GitHub。
- 独立完整视频片段；单请求、有界队列、来源世代隔离；原文与人工编辑分离。

## Review Focus

- 拖框起点在留白、反向拖动及视频实际尺寸变化：裁剪必须合法或明确重选。
- 权限迟到、用户停止共享及Recorder失败：释放所属轨道、阻止旧回调恢复采集。
- 暂停/重选/换语种期间旧HTTP完成：不能写入新字幕。
- 推理慢于采集、错误/空识别：有界且可见，重复词不因文字相同去重。
- 新整句识别结果：保持模型返回语义，不继续标成TFNet词序列。

## Tasks

### 1. 选区与字幕会话

- [ ] 为`selection.mjs`公开换算接口写已知16:9/4:3留白、反向拖框及非法值用例，观察RED，再实现并GREEN。
- [ ] 为`caption-session.mjs`写真实识别HTTP契约、顺序、重复文字、暂停/重选隔离和有界队列用例；RED→GREEN。复用`recognition-controller.mjs`，不修改原调用方。

### 2. 视频来源与独立片段编码

- [ ] 为`screen-capture.mjs`写共享迟到取消、轨道结束、录像切段、空片段及采集失败用例；RED→GREEN。
- [ ] 实现屏幕来源与上传回放共用Canvas选区管线；每段独立Recorder、时间标记、按停顿/定时/手动提交，暂停保留来源预览，结束释放资源。

### 3. 产品页面与资源路由

- [ ] 原`index.html`保留为`assist.html`；新默认页面`index.html`、`meeting-app.js`和`meeting.css`接入以上模块。
- [ ] 添加字幕可编辑原文保留、导出、清空撤销、设置能力、服务重连和辅助页导航。
- [ ] 路由允许新增资源，测试会议默认页/辅助页/资源/路径限制；RED→GREEN。确认wheel包括新静态文件。

### 4. 实机与整合

- [ ] 原API、辅助控制器和动画回归；真实浏览器上传回放框选、录制分段、HTTP结果与字幕核对，保存截图/响应。
- [ ] 读取另一会话最新模型进度，核对模型运行身份和整句结果兼容性；只重启必要所属产品服务。
- [ ] 经过一次独立代码审查，处理重要问题；更新用户文档、项目坑点和验收报告，区分回放/屏幕共享/语义质量未测范围。

## Execution notes

实现与证据留在当前工作树及`artifacts/meeting-web-demo-20261109`；最终提交/推送不在本轮必需范围。
