# Basic Scene Expression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 形成可实际预览的12类基本场景原句库、受控多句组合与可复现覆盖证据。

**Architecture:** 新增只读SceneLibrary索引与组合器；独立路由及CLI复用现有动画资产和MotionStage，不修改并发工作的既有文件。

**Tech Stack:** 现有Python、SQLite、NumPy、FastAPI、原生JavaScript/Three.js；不增加依赖。

**Spec:** `docs/superpowers/specs/2026-11-07-basic-scene-expression-design.md`

## Global Constraints

- 12场景、每场景8目标，共96条；train完整素材优先且默认唯一来源划分。
- 多句最多6段、总帧数1800、插入9帧/30FPS过渡；保留全部源帧。
- 整体缺项阻断，保留问号/感叹号/数值/否定/专名；不声明真人语言验收通过。
- 独立8879端口，不改已有并发修改文件，不写语料、不提交/推送。

## Review Focus

- 模糊检索结果不能自动冒充输入精确匹配；候选分组不能当场景覆盖。
- 小数、否定、疑问与专名变化必须阻断，不能被规范化吞掉。
- 多句后段缺失不能部分播放；输入改变/停止/乱序不得恢复旧任务。
- 计划创建后参数或模板变化时拒绝，不能执行客户端伪造计划。
- 组合源帧必须原样保留，额外过渡帧有来源区分且预算有界。

### Task 1: SceneLibrary与动作组合器

**Files:** Create `slbridge/animation/scenes.py`; Test `tests/test_animation_scenes.py`。

**Interfaces:** 读取CorpusStore和scene definitions；产出spec定义的catalog/coverage/plan/motion。

- [x] 测试先行并观察失败：仅train完整项进入索引；文本JSON正确解码与去重；场景分页检索；固定目标exact覆盖与keyword候选分开；问题/小数/否定/姓名不能被吞；任意缺段整体blocked；确认条件、6段和1800帧上限；source变化拒绝；过渡旋转最短弧且源片段逐帧相等。
- [x] 实现库与组合器；可用既有gesture四元数帮助函数，绑定源模板/时基及SHA，返回MotionStage可读结构。
- [x] Run `.venv-libras/Scripts/python.exe -m pytest tests/test_animation_scenes.py -q`；Expected: 全部通过。记录RED/GREEN。

### Task 2: 场景页面

**Files:** Create `slbridge/animation/web/scenes.html`, `scenes.js`, `scenes.css`; Test `tests/animation_scenes.test.mjs`。

**Interfaces:** 消费spec约定HTTP JSON，加载`/animation/static/stage.js`的MotionStage；不改既有网页文件。

- [x] 先写最小真实行为测试，验证输入变化与停止失效旧请求，源文本显示及缺项阻断，观察失败。
- [x] 实现场景切换、分页、检索、加入表达、输入、多句预览、停止及计划下载；全部动态文本用textContent，禁止自动播放，包含确认复选框。
- [x] Run `node --test tests/animation_scenes.test.mjs`；Expected: 全部通过。真实浏览器由Task3执行。

### Task 3: 目标清单、集成与证据

**Files:** Create `slbridge/animation/basic_scenes_v1.json`, `slbridge/animation/scene_routes.py`, `scene_cli.py`, `scripts/animation/audit_basic_scenes.py`; Test `tests/test_animation_scene_routes.py`; report `docs/reports/basic-scene-expression-2026-11-07.md`。

**Interfaces:** 导入SceneLibrary，固定本地配置，挂载routes与静态资源；复用CorpusStore/MotionRegistry/PresentationBundle。

- [x] 固定96条独立目标（不先挑现有原句来提高分母），保存定义；审阅各场景真实候选是否自包含，记录缺口。
- [x] 路由测试先失败：未知plan404、缺词禁止motion、确认条件、来源漂移409、分页边界、固定路径保护。
- [x] 实现路由、CLI和审计脚本，启动8879；审计12场景候选各若干条的真实NPZ与MotionStage结构；报告固定目标exact/组合/blocked，不输出虚假语言通过率。
- [x] 相关Python/Node回归通过后，真实浏览器逐场景播放至少一条、一次新多句组合、一次缺项阻断与输入变更取消；保存轻量截图和机器证据。
- [x] 独立代码审查，修正重要问题，更新报告、计划复选框和项目坑点索引。

## 执行记录

本轮沿用现有功能工作树，文件互不覆盖。使用独立报告记录进展；没有为本轮自动提交任何文件，保留用户审阅和同步选择。
