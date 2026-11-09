# 动画工作汇总与本地整合

日期：2026-11-08，Asia/Taipei。按用户要求，将[另一条会话](codex://threads/01a1b69f-5d64-7920-a377-075f0e66ba28)的自定义表达、词库扩容工作与当前 v14A 人物/手形/接触轨工作汇总为一个本地演示。

**后续产品整合：** 面向使用者的主入口已升级为[面对面沟通](http://127.0.0.1:8879/communicate/)，接通上传视频识别、人工核对、文字回复与数字人表达。本文保留动画整合时的历史记录；新入口和最新验收见[产品报告](communication-product-2026-11-08.md)。下列`/generate/`现为开发调试入口。

## 当前统一入口

**[http://127.0.0.1:8879/generate/](http://127.0.0.1:8879/generate/)** 默认快速词级组合；同页可选择较慢的模型生成。页面顶部现在可以相互切换“文字演示”“动作与手形对照”“原句场景库”。

| 入口/功能 | 当前实际接入 | 状态与范围 |
|---|---|---|
| [文字演示](http://127.0.0.1:8879/generate/)：快速组合 | 中文 → 固定 beam4 mBART 规划 → 原生词动作检索 → 6 帧过渡 → v14A 播放 | 1–200 字；1,990 个有效词/短语标签、14,317 个动作候选；每词最多 8 例。缺词或已检测到的锚点遗漏会阻止播放 |
| 同页“模型生成（较慢）” | 现有 SignSparK 身体/双手/面部模型及 readonly/offload 执行配置 | 可选 4/6/8/10 秒。此前两条真实新句完成生成与播放，约 5 分钟/条；本轮合并只复核入口/服务状态，没有重跑完整 GPU 生成 |
| [动作与手形对照](http://127.0.0.1:8879/animation/) | 原 100 条动作、SOKE 全量检索、原始/制作修正版同帧对照、慢放与来源诊断 | 全量索引 20,654；完整已转换 19,877 段，去重句子文本 7,369 种；775 段缺帧、2 段缺序列仍隔离 |
| [原句场景库](http://127.0.0.1:8879/scenes/library) | 既有精选场景原句与最多 6 句完整表达接续 | 需要已有整句匹配；不以这个入口冒充词级新句生成 |
| 共用人物与显示链 | 冻结 Godette v14A、30 根指骨保留原始局部旋转、保守身体避让 | 所有入口使用同一 `MotionStage`/retarget。指定 S000931 源动作保留整段接触轨；不会把该轨套给任何含“西瓜”的新组合句 |

原会话标题中的“还不能把任意新句子翻译成手语动作”是较早阶段的记录：现已增加自由文字词级组合和可选模型生成。**这仍不表示任意中文都能准确、完整、自然地翻译或播放。**

## 两边成果如何合到一起

两边实现原本就位于同一个工作树、同一未提交工作区：

`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`

分支 `codex/local-godette-v14a-demo-20261107`，当前提交基点 `e8477dcd5c53495c0f555ae55fa8a1d97213f9a2`。没有两个独立提交分支需要 Git merge；本轮统一运行配置、页面导航、质量提示、验证与文档。实现仍为本地未提交改动，没有推送或合并 PR。`E:/githubproject/SignLanguage-Bridge` 主检出保留其原识别分支；本机启动脚本明确指向上述动画工作树。

当前只复用 **8879** 服务。旧 8874/8878 已在另一会话按用户选择关闭以释放资源；本轮没有重启它们，也没有停止现有规划器、Docker 或 WSL。

本机启动入口：[start-demo.ps1](E:/githubproject/SignLanguage-Bridge/artifacts/animation-consolidated-20261108/start-demo.ps1)。已有相同配置服务时核对人物/词库身份后直接复用；冷启动时后台运行统一 CLI。对应 [stop-demo.ps1](E:/githubproject/SignLanguage-Bridge/artifacts/animation-consolidated-20261108/stop-demo.ps1) 只定位已记录且命令匹配的本服务及其子进程。启动复用路径已实跑，冷启动分支与停止脚本本轮仅做语法/静态核查，没有为了验证而打断当前服务。

```powershell
& 'E:/githubproject/SignLanguage-Bridge/artifacts/animation-consolidated-20261108/start-demo.ps1'
```

当前词库配置为动画工作树 `artifacts/corpus-expansion-20261108/config-after.json`；库目录 `G:/Archive/Zenodo/SignSparK_Retrieval/word-library-v2-expanded-20261108`，`bundle_id=csl-product-95379e49911c11908e420a94`。它从 5,916 扩到 14,317 片段，但标签仍为 1,990；train/dev/test 混合候选明确标为 `usage_role=product-demo`、`evaluation_eligible=false`，不能用于声称独立测试成绩。

v14A 人物 SHA256：`535fc9ab96ec3d837e2568817abc41d593fac1c4fcad8565b45b29c5f4472f5e`。原动作、源参数、人物 GLB、原词片段、原权重及既有报告保持各自身份；没有通过改字幕替换旧动作。

## 当前手形与穿模进度

上轮已修复显示层独立平滑改变指骨手形的问题，并针对“女儿写作业”“药物过敏吗？”减少避让额外跳动；S000931“今天水果很新鲜，买一个西瓜回家。”使用精确源身份绑定的整段接触轨。上轮 104 段/28,668 帧检查中，860,040 个指骨四元数键不变，源输入修改为 0，剩余核心重叠线索为 0；S000931 另有全段 2,577 个插值时刻检查。详见[手形与接触轨报告](godette-word-refinement-2026-11-08.md)。这些是指定片段和保守核心的工程结果。

**本轮合并发现的新限制：** 从在线扩展词库实际组合的“我想喝水。”为 50 帧，1,500 个指骨键保持不变，输入未修改；但 v14A 显示避让仍有 **1 个核心重叠线索**，最大额外腕部偏移步长约 **0.05826 模型单位（约 5.83 cm/帧）**。它在工程上可播放，手形/接触质量没有通过验收。不能把旧 104 段的通过结论推广给新组合句。

已把现有 renderer 的两种诊断接入文字主页面：有重叠线索时提示重点核对手部与身体接触；额外偏移步长超过既有 0.015 阈值时提示核对腕部衔接。本轮没有用新的通用平滑覆盖指骨，也没有为通过指标放宽单次 0.06 位移上限。后续应对新词片段/衔接做接触轨调校，并保留原始/修订对照。

当前核心体积不是完整皮肤、袖口、头发、服装或双手碰撞检测；源拟合与规划本身仍可能错误。慢模型产物中的旧人物下颌候选也不等于 v14A 完整口型或非手控语法已验收。

## 本轮新鲜合并验证

证据集中在 `E:/githubproject/SignLanguage-Bridge/artifacts/animation-consolidated-20261108/`，不随 Git 发布。

| 检查 | 本轮结果 |
|---|---|
| 同一在线服务身份 | 8879：规划器 ready、beam4、1,990 标签/14,317 候选、冻结 v14A、19,877 转换段/7,369 文本、慢生成入口 enabled |
| 三条真实 HTTP | “我想喝水。” ready 且有 `retrieved_word_composition` 动作；“我想去成都。” blocked/缺成都且无 motion；“明天早上我们一起去医院看朋友。” needs_review/漏医院且无 motion。单次热 HTTP 约 147/104/128 ms，只是本轮三次功能复核，不是 P95 或首像素指标 |
| 路由与导航 | `/` 和 `/scenes/`进入文字演示；`/generate/`、`/animation/`、`/scenes/library` 同服务可达，并互相提供入口 |
| 真 GLB/真实动作重定向 | 从同一服务取得组合句与 S000931；共 6,360 个指骨键逐分量不变、输入未改。S000931 命中指定轨、核心重叠 0；新组合句保留上述未通过的接触问题 |
| Node 动画完整集合 | 114 passed / 4 skipped；跳过为未配置旧版本角色实物的可选 GLB 检查，含本轮真实 v14A 指骨/接触检查 |
| Python 定向整合 | 196 passed / 2 skipped，1 个 Starlette 弃用警告；跳过为 Windows 符号链接权限。覆盖动画/场景、组合/HTTP、生成任务/HTTP、CLI 和真实 v14A 身份 |
| 浏览器 | 已通过顶部导航实际切换三个入口；同服务 S000931 0.40 s 显示指定接触轨；快速组合播放结束，缺成都/漏医院均未播放。新增穿模/腕部衔接提示可见；一次预热提交至 playing 状态 0.24 s，控制台 warn/error 为 0。`browser-audit.json`、`unified-demo.jpg`、`hand-comparison.jpg` |
| 全仓库/CI、完整慢模型 GPU 生成 | 本轮 NOT_RUN。上一轮全仓库一次 ONNX/SciPy 子进程导入失败及其隔离成功结果仍按原报告保留，不重写为全绿 |

关键产物：`http-audit.json`、`composition-status.json`、`presentation-status.json`、`compose-{ready,missing,omitted}.json`、`retarget-integration.json`、`python-integration.log`、`node-integration.log`。页面复核与截图另存本目录。

## 后续工作与证据索引

当前优先项是新组合句的手部/身体接触与衔接，再做目标 CSL 使用者的词义、语序、否定/数量/人名/指代、非手控语法和自然度审核。缺源的 `45、利益、小姐、成都、瓜、谣言、阅兵` 不能靠增加同词候选数解决；需取得相应三维动作或完成视频重建与审核。地点/时间锚点 guard 只拦截已知遗漏，不是完整语义验证。

| 工作 | 详细记录 |
|---|---|
| 本地 v14A 接入、PR 基点与六段对照 | [本地演示报告](godette-v14a-local-demo-2026-11-07.md) |
| 手形回退原因与第一轮避让 | [手形回退与穿模](godette-handshape-clearance-2026-11-07.md) |
| 其他词句与 S000931 整段接触轨 | [词句修订](godette-word-refinement-2026-11-08.md) |
| 场景完整表达检索与接续 | [基本场景](basic-scene-expression-2026-11-08.md) |
| 网页自由文字慢模型生成、Windows 内存修复 | [真实生成](free-text-sign-demo-2026-11-08.md) |
| 快速词级组合、mBART 规划及遗漏保护 | [快速组合](instant-sign-composition-2026-11-08.md) |
| 词动作候选扩至 14,317，标签不变 | [词库扩容](word-corpus-expansion-2026-11-08.md) |
| 外部资产、建库、运行与复现 | [工作台使用说明](../csl-animation-workspace.md) |


![统一文字演示与当前质量提示](E:/githubproject/SignLanguage-Bridge/artifacts/animation-consolidated-20261108/unified-demo.jpg)
