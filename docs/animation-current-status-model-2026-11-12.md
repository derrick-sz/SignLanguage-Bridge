# 当前沟通产品与动画工作入口

更新：2026-11-12。面对面沟通页已连接独立识别服务，另有 SLRT 增量词＋Uni-Sign 句末中文的实时入口。数字人由动作数据在浏览器实时渲染；快速模式检索/组合已有动作，整段准备后播放。画面实时渲染不等于动作数据流式生成或手语表达质量已验收。

- [沟通产品](http://127.0.0.1:8879/communicate/)：上传手语视频、识别核对、文字回复与数字人表达，在同一页面完成。
- [实时识别](http://127.0.0.1:8879/communicate/realtime.html)；协议、计时和限制见[实时说明](realtime-csl.md)。
- **接续开发优先阅读：[当前缺陷、证据与改进计划](superpowers/plans/2026-11-12-avatar-motion-quality.md)。** 含 AV-01～AV-13 缺陷、M0～M6 依赖顺序、交付物和验收条件。
- [数字人 25 条单词回认](reports/avatar-train-recognition-2026-11-12.md)、[30 条完整句与改进试验](reports/avatar-train-sentence-expansion-2026-11-12.md)、[最近的姿态驱动迭代](reports/avatar-retarget-iteration-2026-11-12.md)。当前候选未达到接入条件，未替换产品动作。
- [产品使用与范围](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/communication-product.md)；[本轮产品验收](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/reports/communication-product-2026-11-08.md)。
- [文字动作调试](http://127.0.0.1:8879/generate/)与手形对照／原句库保留为开发工具。
- [2026-11-08 历史整合总表](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/reports/animation-consolidated-status-2026-11-08.md)，后续质量状态以上述 2026-11-12 报告和计划为准。
- 本机启动脚本（本地产物：`artifacts/product-bridge-20261108/start-product.ps1`）。

实现工作区为 `C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge` 的 `codex/local-godette-v14a-demo-20261107` 分支；本主检出保留原识别分支。此页是本机跳转入口，能力/测试/待办以总表和实际源码、运行记录为准。

最近数字人评测采用“画布录像 → 转码 → 按视频节奏回放实时接口”，不是物理摄像头或数字人直接实时输入验收。姿态关键点、有限碰撞采样、请求正常结束及真人训练视频成绩均不能替代数字人语言审核；用户原“我想睡觉→绿色”录制尚未独立复现。
