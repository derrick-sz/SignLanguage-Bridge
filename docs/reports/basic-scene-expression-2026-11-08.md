# 基本场景表达：实施与验证（2026-11-08）

**入口更新（2026-11-08）：** 自由文字生成主链路现已实现，主入口为`http://127.0.0.1:8879/generate/`；`/scenes/`现重定向到该入口，原句库保留在`/scenes/library`作为辅助素材入口。下文记录此前完整表达复用与多句接续的实施及验收，库内未匹配时阻断的规则只适用于该旧库流程，不能用来限制新的自由文字生成。新流程已实际生成两条不同库外新句；浏览器播放收尾、最终测试汇总及语言验收范围见[自由文字设计与状态](../superpowers/specs/2026-11-08-free-text-sign-demo-design.md)及[执行计划](../superpowers/plans/2026-11-08-free-text-sign-demo.md)。语言审核仍为`NOT_RUN`。

计划于2026-11-07开始，实际验证于2026-11-08（Asia/Taipei）完成。用户授权由本次工作制定计划并验证，优先利用现有连贯语句覆盖基本场景。

## 已形成的能力

原句库当前入口：**http://127.0.0.1:8879/scenes/library**（本报告此前验证时使用`/scenes/`）。

- 12类基本场景，各精选8条真实完整原句，共96条不同文本／96个样本；全部train，95条选签者0，1条选签者8。文本相关性由本轮检查，手语语言／自然度未人工认证。
- 可检索train完整动作中的6,572种文本，索引17,708个样本、2,000个Gloss标签。场景关键词候选与固定需求精确覆盖分别显示。
- 可把2–6个完整表达接续为新的多句表达；每段全部源帧保留，在段间新增9帧最短弧四元数SLERP。原句内部不切词、不重采样。
- 完整输入先查整句；未匹配时只在显式句末标点／换行处分段。任何一段缺失则整体阻断，不能替词、漏词或播放已有部分。
- 每次播放绑定服务器保存的计划、源样本和参数摘要、模板／骨架身份；计划篡改或来源变化拒绝。可下载来源与缺项计划。

这是**完整表达复用与多句接续**，不是句内任意词替换或任意中文词级翻译。现有源拟合抖动、手形、接触和完整非手部语法未因本轮接续获得修复或语言批准。30FPS沿用现有参数库的播放约定，不是新增的原视频逐帧映射证明。

## 计划与实现

设计：[2026-11-07-basic-scene-expression-design.md](../superpowers/specs/2026-11-07-basic-scene-expression-design.md)。
执行计划：[2026-11-07-basic-scene-expression.md](../superpowers/plans/2026-11-07-basic-scene-expression.md)。
场景及固定目标：[basic_scenes_v1.json](../../slbridge/animation/basic_scenes_v1.json)。

复用工作树：`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，原HEAD `e8477dcd5c53495c0f555ae55fa8a1d97213f9a2`，分支 `codex/local-godette-v14a-demo-20261107`。本次只新增scene相关模块／页面／测试及文档；已有api、stage、rig、retarget、人物和pyproject并发修改未由本次覆盖或提交。既有8874/8878服务保留。

## 场景盘点与需求差距

以下固定目标在执行前写定，每场景8条；精选原句不能反向替换分母。exact保留问号、感叹号、数字、否定、姓名，句末句号可忽略，因此“你好。”不会自动当作“你好！”。这是一种保守的字面资源覆盖口径，不是实际沟通成功率。语义等价的其他说法需明确选用其原文。

| 场景 | 精选真实原句 | 关键词／精选候选文本 | 固定目标严格匹配 |
|---|---:|---:|---:|
| 问候礼貌 | 8 | 40 | 2/8 |
| 身份家庭 | 8 | 396 | 1/8 |
| 时间安排 | 8 | 411 | 1/8 |
| 饮食 | 8 | 183 | 1/8 |
| 购物支付 | 8 | 303 | 0/8 |
| 出行问路 | 8 | 174 | 0/8 |
| 学校学习 | 8 | 556 | 0/8 |
| 工作沟通 | 8 | 346 | 0/8 |
| 健康就医 | 8 | 197 | 0/8 |
| 居家日常 | 8 | 196 | 1/8 |
| 情绪社交 | 8 | 540 | 0/8 |
| 求助安全 | 8 | 158 | 0/8 |

候选组可重叠，不能相加为不同表达数，关键词命中也可能不符合实际意图。固定96目标：6条exact、0条可由当前严格完整表达拼出、90条blocked。这个结果说明“拥有对应场景的句子”与“覆盖指定基本说法”差别很大，不能称12场景全面覆盖。

完整短句“再见、稍等、我知道、我明白、救命、请报警、我迷路了、我饿了、我头疼、我不舒服”等未在完整候选文本中直接命中；部分概念存在于较长句。健康语料的“我身体不舒服，想去医院。”不能只改字幕充当“我头疼”。居家卫生、通用等待回应与紧急呼救是优先补齐方向。没有从长句按字幕裁字冒充词级边界。

## 实际验证

### 参数与来源

最终证据：[run02/numeric-audit.json](../../artifacts/basic-scenes-20261108/run02/numeric-audit.json)、[coverage.json](../../artifacts/basic-scenes-20261108/run02/coverage.json)。

- 96条精选原句全部加载实际NPZ，校验来源、帧数、`T×55×3`、有限值与骨架结构，0失败。
- 6条新的多句表达实际构建，源动作切片逐帧 `np.array_equal` 相等；新增过渡区单独记载，0失败。
- 4条库外／标点负控全部blocked，未发生部分播放。
- 参数身份、模板／骨架漂移、计划篡改、确认条件、6段／1800帧预算等由定向回归验证。

| 新多句表达 | 源段数 | 输出帧数 |
|---|---:|---:|
| 你好！请坐。你想喝什么？ | 3 | 108 |
| 你好！我想喝茶。谢谢！ | 3 | 128 |
| 对不起！我不懂你在讲什么。请帮助我！ | 3 | 122 |
| 你好！我是手语翻译。很高兴认识你。 | 3 | 135 |
| 请帮助我！我丢了钥匙。谢谢！ | 3 | 131 |
| 我身体不舒服，想去医院。我不得不请假。 | 2 | 203 |

### 真实浏览器

Chrome可视浏览器真实WebGL／人物加载，页面1360×1050；[browser-audit.json](../../artifacts/basic-scenes-20261108/browser/browser-audit.json)为逐项证据。

- 12场景各播放一条真实原句，全部运行至结束；不是只调用后端或只看截图。
- “你好！请坐。你想喝什么？”三段接续实际播放至结束，页面显示3段原文来源。
- 库外后段“我想去火星看流星雨。”阻断整段；人为延迟motion响应后修改输入，旧响应未恢复播放。
- 390px手机宽度无横向溢出、人物canvas存在；最终一轮控制台error=0、warning=0。
- 单张画面仅作界面证据，不承担动作自然度或语言认证。

最终服务已按新默认配置路径重启；[final-live-smoke.json](../../artifacts/basic-scenes-20261108/final-live-smoke.json)再次确认2000标签、108帧三段组合、缺项阻断和500字符页面限制，并保存本轮模块与复用人物播放器源码摘要。

![基本场景与多句研究预览](../../artifacts/basic-scenes-20261108/browser/scene-sequence-desktop.png)

### 本地回归

- Python动画相关：**112 passed**，1条既有Starlette/anyio弃用警告；[日志](../../artifacts/basic-scenes-20261108/python-animation-tests.log)。不是全项目／CI／真实模型重跑。
- Node动画相关，使用现有Three.js loader与人物依赖：**79 passed / 12 skipped / 0 failed**；[日志](../../artifacts/basic-scenes-20261108/node-animation-tests-final.log)。12条未配置其他版本人物或真实手／词动作fixture，仍为未运行；本轮scene控制器8条均通过。
- 独立审查发现前端2000／核心3000／HTTP500长度不一致，已统一500并增加程序化append/preview守卫；超长输入测试RED→GREEN。
- 实库发现Gloss发布格式为JSON字符串而非测试fixture中的列表；已兼容两种，修正2,000标签统计和多字Gloss搜索，RED→GREEN。
- 浏览器首轮揭示fetch脱离global receiver导致Illegal invocation，已绑定globalThis并增加行为回归，RED→GREEN；随后完整浏览器通过。

初次服务启动未就绪的连接失败、fetch修复前报告和Node未配置loader／错误Windows loader路径的失败日志均保留；修正环境后才报告上述最终结果。没有把失败或skip改写成通过。

## 使用与重现

工作目录为上述功能工作树。启动独立服务：

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -X utf8 -m slbridge.animation.scene_cli --archive E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip --expected-sha256 4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5 --presentation E:/githubproject/SignLanguage-Bridge/artifacts/godette-v14a-demo-20261107/presentation --corpus G:/Archive/Zenodo/SOKE/full-library-20261107 --port 8879
```

打开 `/scenes/`；选择场景，选用原句或逐句“加入表达”，核对后勾选研究预览并播放。输入改变会停止旧播放；缺项出现在页面下方，不能自动替换为近似句。

新增审计（output必须是新目录）：

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -X utf8 scripts/animation/audit_basic_scenes.py --corpus G:/Archive/Zenodo/SOKE/full-library-20261107 --output artifacts/basic-scenes-next-run
```

浏览器验证脚本随本地证据保存为 `artifacts/basic-scenes-20261108/browser/playwright-check.js`；依赖已有Playwright和Chrome。大语料、人物和完整运行目录仍是本机资源，本次未打包分发或核准商用。

## 下一阶段计划与未验收范围

本轮3项任务已实施。进一步覆盖基本需求，应按以下顺序推进；下列内容未计作本轮完成：

1. 对90条固定缺口建立文本意图与可选原句对照，区分可明确选用的等价表达、含额外事实的表达、确实缺动作；不自动改变否定、数字、专名或问题类型。
2. 按需求优先制作等待／回应、告别、居家卫生、常见身体不适与紧急呼救模块。先检查句级素材是否已有同义完整表达，再决定审核过的词级组合或补录。
3. 对配套`csl_iso.train`与SignSparK动作库做边界试点，先20–30词／短语、多候选；SOKE帧映射核实后再扩大现有库切词，不凭同ID直接切。
4. 由目标地区CSL使用者核对词义与句法，盲看实际人物复述关键事实，并分别评自然度／手形／空间指代／非手部表达。当前语言理解、自然度、接触与完整非手部语法均 **NOT_RUN**。

本轮决定：优先完整表达、只在句界接续，代价是句内改词仍受限；复用既有脏工作树且不提交／推送，代价是后续同步需单独选取本轮文件；配置最终放在包旁`basic_scenes_v1.json`以避开全局`data/`忽略规则。这些决定及逐步验证记录在本轮ledger中，不影响旧实验成绩。
