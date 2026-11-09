# Instant Sign Composition Implementation Plan

> **For agentic workers:** implement independent data, planner and controller modules against the interfaces below; root integrates and dispatches independent review.

**Goal:** 自由中文默认进入快速词级动作组合并实际播放。

**Architecture:** 离线认证词片段库，持久mBART文字规划进程，CPU词候选选择与轻量SLERP，复用MotionStage。

**Tech Stack:** Python/FastAPI/NumPy/SciPy、现有SignSparK隔离runtime的Transformers、原生JavaScript。

**Spec:** `docs/superpowers/specs/2026-11-08-instant-sign-composition-design.md`

## Global Constraints

- 1–200字符；最多32词、1800帧、词内源帧保留、段间6帧过渡。
- ISO必须绑定SignSparK LMDB时基；原权重/数据只读；语言NOT_RUN。
- 不以测试自身动作或最近词替换冒充自由新句；缺词不播放部分句。
- 不覆盖并发人物文件，不关闭Docker/WSL，不提交/推送。

## Review Focus

- 同ID不等于SOKE帧时基；解码左手不二次翻转/加均值。
- 模型截断、空输出、超时与错请求ID不能产生看似完整动作。
- 改动输入/模式、取消、旧HTTP/加载结果不能继续播放。
- 文件身份变化必须拒绝；不能以缓存继续使用旧身份。
- 页面时间与后台组合计时分别报告，不承诺未达到的1–2秒。

### Task 1: 资产及离线词库

Files: `word_library_build.py`、`scripts/animation/build_word_library.py`、`tests/test_word_library_build.py`。

- [x] 续传固定revision两个资源并验证SHA，保留下载身份/日志。
- [x] RED测试受限ISO、数值NPZ、范围/帧时基、左右手协议、词帧源不变及错误隔离。
- [x] 实现`build_word_library(lmdb_path,annotations,base_archive,output,*,expected_lmdb_sha256,expected_annotations_sha256,max_candidates=3)`；返回manifest；新输出目录，单位schema见spec。
- [x] GREEN后构建真实词库并保存有效词/失败片段统计。

### Task 2: 持久文字规划器

Files: `text_gloss_planner.py`、`scripts/animation/text_gloss_worker.py`、`tests/test_text_gloss_planner.py`。

- [x] RED测试真实JSONL子进程fixture：ready、请求ID、时限、退出、错误、busy、输入拒绝、资产身份。
- [x] 实现`TextGlossPlanner(config).plan/status/close`及真实mBARTworker；stdout纯协议、日志stderr；未知词不替换，输入/输出不截断。
- [x] GREEN；真实权重测试由root协调，以免Windows提交额度竞争。

### Task 3: 快速组合核心与HTTP

Files: `word_composition.py`、`word_composition_routes.py`、`tests/test_word_composition.py`、`tests/test_word_composition_routes.py`；root修改`scene_cli.py`。

- [x] RED测试多候选/缺词/来源区间/实际SLERP/原词帧/摘要变化/预算/HTTP状态。
- [x] 实现`WordLibrary(path).compose(text,glosses)`、`CompositionService(library,planner).compose/status/close`。返回schema见spec；SHA绑定实际clip，缓存有界。
- [x] 挂载GET capabilities与POST composition，启动配置以server-owned JSON提供，独立子进程warmup并关闭自有worker。
- [x] GREEN；实际库外新句、缺词和分阶段延迟验证。

### Task 4: 默认快模式与实测

Files: `web/fast-composition-controller.mjs`、`web/generate.html/js/css`、`tests/animation_fast_composition.test.mjs`；新增报告。

- [x] RED测试自由输入POST、ready加载自动播放、缺词、输入/模式变化、取消/晚到及停止重播。
- [x] 实现默认快速组合/显式慢生成，展示词序/缺词/实际首播耗时。
- [x] GREEN；root真实浏览器验证、独立review、复核报告与坑点。性能未达到目标时保留真实数据，语言仍NOT_RUN。
