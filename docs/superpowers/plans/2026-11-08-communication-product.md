# Communication Product Implementation Plan

**Goal:** 把已有识别和text-to-gesture接成听障沟通产品，替代研究演示主入口。

**Architecture:** 独立v3识别服务 + 有界轻量产品网关；同一前端复用现有动作控制器/人物舞台。来源、用户编辑、对话消息分别管理，取消与换源有世代隔离。

**Spec:** `docs/superpowers/specs/2026-11-08-communication-product-design.md`

- [x] 真实后端契约盘点与独立识别服务启动；记录可用语种和模型范围。
- [x] 先写有意义的网关负例/绑定/文件清理与对话防串结果回归，再实现communication服务和路由。
- [x] 完成产品双区界面、摄像头/上传、识别核对、文字表达、消息复制/导出和设置；复用原始动作与取消保护。摄像头生命周期仅单测，未访问真实设备。
- [x] 接入scene_cli并把根入口切向产品；独立本机一键启动两类服务，保护已有进程。
- [x] 运行定向回归、真实上传视频/识别与组合链路、WebM转换和浏览器检查；用截图复核界面与窄屏。依用户选择，物理摄像头NOT_RUN。
- [x] 更新产品运行说明、坑点与验收记录，保留未运行的设备与质量项目。详见`docs/reports/communication-product-2026-11-08.md`。
