# Free Text Sign Demo Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent worker/frontend tasks and a final fresh review.

**Goal:** 自由中文网页输入经真实预训练生成得到可播放人物演示。

**Architecture:** 进程隔离的串行GPU任务管理器调用现有认证CLI；新HTTP与UI复用MotionStage，独立于整句匹配。

**Tech Stack:** 现有Python/FastAPI/NumPy/JavaScript/Three.js和本机SignSparK CUDA环境，无新增模型训练。

**Spec:** `docs/superpowers/specs/2026-11-08-free-text-sign-demo-design.md`

**Execution status (2026-11-08):** 主链路已实现，两条不同库外新句已实际生成并通过发布校验、在浏览器人物播放。资源修复采用只读checkpoint映射、CPU FP32文本缓存与CUDA逐层offload，执行配置在worker manifest中单独记录。root已确认第二条播放结束`0:04 / 0:04`、console error／warning均为空；新鲜回归分组为Python82通过、独立runtime48通过、Node18通过。最终浏览器证据与执行报告由root归档。语言、手形和自然度均未获得人工验收。

## Global Constraints

- 自由输入1–200字符；4/6/8/10秒，50步seed123；不依赖库内完整句命中。
- 一个GPU worker；非终态最多8、状态最多32；原始模型/数据只读。
- 只有真实complete和校验通过的产物可播放；取消/失败不能冒充成功；语言NOT_RUN。
- 不覆盖并发stage/rig/retarget/api修改，不提交或推送。

## Review Focus

- 输入在库外时仍进入真实生成，不能改字幕配旧动作。
- subprocess不拼shell，输入不能注入参数/路径；取消终止自有进程树。
- queued/running/failed/cancelled互不冒充complete；晚到结果不能重启播放。
- 发布前校验worker请求文本/帧数、源空观测、archive摘要及生成asset身份。
- 输出数据不与旧请求或不同模型/处理身份混用，客户端不获得本地路径。

### Task 1: GPU任务管理

Files: 新增`slbridge/animation/text_generation_jobs.py`、`tests/test_text_generation_jobs.py`。

- [x] 阅读spec；先写状态机、库外文字、队列/取消、失败和发布校验测试并观察RED。
- [x] 实现GenerationService及真实CLI runner；subprocess argv、隐藏窗口；读取实际manifest进度，严格发布与有界状态。
- [x] 用fake runner/真实小产物fixture运行定向tests至GREEN，不在单元测试启动GPU。

### Task 2: 自由输入页面

Files: 新增`web/generate.html/js/css`、`web/generate-controller.mjs`、`tests/animation_generate.test.mjs`。

- [x] 先测自由输入、轮询/取消/输入变化、旧结果不播放、fetch绑定与失败显示，观察RED。
- [x] 实现主输入和生成/进度/人物演示流程；不要求选择已有表达；复用MotionStage。
- [x] Node定向行为验证至GREEN；真实浏览器由Task3检查。

### Task 3: HTTP接线、真实生成与报告

Files: 新增`text_generation_routes.py`和测试；修改本轮自有`scene_cli.py`/`scene_routes.py`接线；新增报告。

- [x] HTTP测试先行：库外请求入队、未完成motion409、未知404、队列429、invalid422、运行时缺失503。
- [x] 固定server runtime配置，挂载/generate/；已有/scenes/进入新主流程，/scenes/library保留旧库。
- [x] 从浏览器提交两个真实库外新句并等待GPU生成complete；保留任务、worker manifest和motion产物。见[任务1](../../../artifacts/free-text-20261108/accepted/job-1.json)、[生成1](../../../artifacts/free-text-20261108/accepted/generation-1.json)、[任务2](../../../artifacts/free-text-20261108/accepted/job-2.json)、[生成2](../../../artifacts/free-text-20261108/accepted/generation-2.json)。
- [x] 两句人物实际播放；第二条结束`0:04 / 0:04`、console error／warning均为空（root观察）。两份archive和raw body摘要不同，产物身份分开保留。
- [x] root归档最终浏览器证据与数值差异审计，保留早期失败尝试。`accepted/browser-cua-audit.json`记录两句播放结束及截图接口限制；`accepted/artifact-audit.json`核对实际参数、SHA、来源及数值差异。
- [x] 独立review及资源修复静态补审；未发现剩余可操作P1/P2，见[review](../../../.superpowers/sdd/2026-11-08-free-text-sign-demo/final-review.md)。
- [x] root收尾回归：Python82通过（服务／HTTP／场景／两CLI）；独立runtime48通过（readonly／offload／generation／empty-loading）；Node18通过（generate＋scenes）。按组记录，不叠加早期重复测试数。
- [x] 完善[最终执行报告](../../reports/free-text-sign-demo-2026-11-08.md)与坑点。语言质量仍NOT_RUN。
