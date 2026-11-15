# 实时中文手语工程实施计划

用户已确认并授权实施的依据：[策略评估](../../research/real-time-recognition-strategy-2026-11-09.md)。本会话直接执行，按吞吐→网页增量事件→Uni-Sign句末→LLM辅助的顺序交付。首词1–2秒、句末P95≤2秒是验收目标，不是完成声明。

## 工作边界

- 模型/API：E:/githubproject/SignLanguage-Bridge，当前codex分支；产品：C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge，复用现有功能工作树。保留两个检出的已有改动，不提交无关文件。
- 8765保留Uni-Sign clip与Libras；新流式能力单独声明。共享现有Uni-Sign实例，避免重新加载整套句子模型。一个中文实时会话持有模型租约，阻止中途切换和竞争上传；不影响独立Libras。
- 不改权重、不把GT送入推理/LLM，不改变P1冻结产物。新姿态/精度协议另记；同视频回放不是物理摄像头/新签者验收。
- 运行记录：artifacts/realtime-csl-engineering-20261109/progress.md。测试与实测输出保存在该目录。每阶段先测试缺失行为再实现；阶段评测依据P1 canonical评测器。

## Task 1：SLRT吞吐与可追溯协议

接口：SLRTConfig增加显式pose_detect_stride与precision，默认保持detect=1/FP32；RTMPose可复用检测框，质量失效时重置；CUDA FP16可选。现有帧缓存保持。

- RED：配置边界/默认兼容、前端检测间隔传递及失效重检、CPU拒绝FP16。
- GREEN：修改assets.py、pose.py、runtime.py；定向测试test_slrt_stream/capture/frame_cache/protocol及新配置测试。
- 新回放器固定上轮6正例与两控制，逐项比较FP32→FP16→检测复用，必要时再评估stride=2；保存原FPS事件、队列、计时、词列与输入SHA；GT只进独立评测。
- 通过行为/吞吐/质量检查后确定显式实时配置；未达成指标如实记录，继续提供可观察的过载处理。

## Task 2：实时会话与网页事件

接口：realtime会话协议独立版本；开始消息绑定CSL、source_epoch；连续帧绑定index/timestamp；事件包含会话、句段、修订、源帧索引。单会话、输入/事件/任务/缓冲均有上限。

- RED：乱序/断帧、过载、换源/停止迟到、断线native生命周期、独立会话能力与认证。
- GREEN：模型端realtime模块和独立API路由；产品轻量HTTP/WS桥、浏览器取帧和双层字幕。复用选区/摄像头/屏幕预览；不把MediaRecorder片段当可独立解码增量。
- 新增页面模式，保留原上传/分片识别；只在CSL且服务支持时启用。暂停/换源立即失效旧事件。
- 验证Python协议/网关、Node控制器与真实服务视频推送。浏览器权限由用户控制，不伪报物理摄像头验收。

## Task 3：Uni-Sign句末翻译

接口：共用RTMPose133原始点，分别适配SLRT512参考像素与Uni-Sign归一化；新共享前端协议明确命名。Uni-SignBackend增加受锁保护的pose输入入口，不重复提取视频姿态。

- RED：姿态几何/质量/方向、静止拒答、句界/长句上限、不同句任务不互相覆盖、取消不制造EOF、原始clip结果不被破坏。
- GREEN：多信号活动/停顿句段器、前后上下文缓存、句末模型调度、中文draft/committed事件；工程提交不等于语义确认。
- 原FPS回放记录真实句界/中文交付时刻及文本对照；保存新前端结果，不借用原clip指标。现有HTTP clip定向回归。

## Task 4：可选LLM异步辅助

接口：禁用为默认；启用后只发送实际词证据与当前原译文，结构化返回建议/风险；有超时预算，不阻塞词/句事件。

- RED：超时/异常不影响交付、旧修订拒绝、否定/数量/人称/时间变化拦截、拒答清除旧建议、日志不含凭据/GT。
- GREEN：实现可注入辅助器，复用已有OpenAI兼容端点的配置约定；建议和原文分别展示，不自动代替事实。
- 用现有端点做小规模实际请求验证并记录成功/拒答/失败与耗时；不将合规JSON当语义正确。

## Task 5：审查、持久化启动与交付

- 定向Python/Node回归，真实模型串联和产品桥验证；检查新增静态资源随wheel打包。
- 启动profile/launcher增加显式实时配置、身份检查和足够预热时间，识别能力就绪后才可开启。
- 按requesting-code-review技能派一个独立审查代理，范围限本任务改动；修复重要问题并复测。
- 更新两份坑点记忆与实施报告，区分工程通过、性能目标、语言/摄像头NOT_RUN；不推送/合并无关改动。

## Review Focus

1. WebSocket认证/Origin、帧尺寸和长度/时间/速率上限，不因绕过HTTP中间件漏保护。
2. 客户端断线时不得释放仍在native使用的模型；重新开始不得接收上个会话结果。
3. 只丢弃同一句旧修订，前一句句末任务不可被下一句覆盖；有界缓存不能静默截断句义。
4. 输入丢帧/姿态无效重置必须与句段同步，不将中断flush成完整译文。
5. 输出计时包含采集/队列/浏览器交付，模型分数和工程final不作为语义正确保证。
