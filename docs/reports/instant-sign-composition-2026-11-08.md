# 自由输入快速词级组合：实施与实测

日期：2026-11-08，Asia/Taipei。实现位于动画功能工作树`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，本轮未提交、推送，未覆盖并发人物修订。

## 当前行为

`/generate/`默认进入**快速组合**：自由中文 → 持久mBART文字规划器 → 真实词级动作检索 → CPU过渡 → 人物自动播放。`/`、`/scenes/`进入同一主流程，完整原句库保留在`/scenes/library`。页面可显式切换到“模型生成（较慢）”；缺词或规划可疑时不自动调用慢模型，也不播放部分句子。

快速模式支持1–200字符、最多32个Gloss、组合总长最多1,800帧。时长由动作片段决定；不显示慢模式的4/6/8/10秒选择。输入、模式变化及取消／停止均使旧请求或晚到加载失效。仍是标有`NOT_RUN`的实验演示，**不能据播放成功推断中国手语含义正确、手形准确或自然连贯**。

## 原始资源与词库

本轮完整续传并核验作者固定版本的T2G权重与配套train LMDB，不读取partial文件，也不将ISO边界套入SOKE参数。来源为[作者Retrieval模型与标注](https://huggingface.co/LionelLow/SignSparK_Retrieval/tree/a883bc2d13a5d7f85e64591b32bae4cb48950b97)及[配套动作数据](https://huggingface.co/datasets/LionelLow/SignSparK_data/tree/9fe7e04ef59204c787cd6094c747f889efff6ba0)。

| 文件 | 完整SHA256 |
|---|---|
| `t2g/CSL-Daily/model.safetensors`，1,444,104,872字节 | `93ef6db7b706a6d0f6906fbd1bcfad8b46878918b8cb21fc600193f3630aa2be` |
| `train/CSL-Daily_reopt_train.lmdb/data.mdb`，3,327,815,680字节 | `7dbc5dc932c6e34ea1d7d47976e8db8fb90ba4c2aff3eb05a6991c7008366222` |
| `annotations/csl_iso.train` | `92cabbe0fcc97770abd491802be00cb35883a2237cabb7f0c5de847845355f2c` |

原标注133,687条，排除2条重复后有133,685条候选。每词优先选接近中位长度的最多3例，得到**1,990个有效标签、5,916个动作单元**。95个候选在配套LMDB缺源，已隔离；`45、利益、小姐、成都、瓜、谣言、阅兵`七个标签没有可用动作，不以近义词补齐。

词库位于本机`G:/Archive/Zenodo/SignSparK_Retrieval/word-library-v1-20261108`，`bundle_id=csl-words-f845628cb6ae7843c8c636d6`，manifest SHA256为`41e7bddc7013da503cde36c6c42f562905e19b2def1deea44ea6d0ff4648ce80`。文件与权重不随Git分发。

建库先验证完整源文件SHA，ISO禁用class／persistent反序列化；LMDB只读，NPZ数值字段使用`allow_pickle=False`并限制展开量。body严格126列、取`[:,66:]`，双手各90列，face56列；使用现有codec和`smplx-left`，不再次翻转左手、不加SOKE手均值。输出每词float32 `poses[T,55,3]`与`expression[T,50]`，记录原clip、`[start,end)`和train来源。

5个分布抽样单元与原ISO／LMDB直接解码的poses及expression逐位一致，源文件完整SHA复核未变。在线在片段间新增6帧shortest-arc SLERP，选候选时参考相邻关节角距离。**源帧保留仅指参数层；现有renderer仍可施加显示修订，不能声称画面与原始录像逐帧相同。**

所有动作以30 FPS预览；这是明确的产品播放时钟。LMDB没有独立时间戳，作者展示脚本另有默认FPS，原始采集FPS尚未核验，不能把30 FPS称为原片精确时间基。

## 文字规划与已发现的失败

规划器使用固定作者mBART CSL-Daily模型、官方提示模板、`zh_CN`与**beam4**；不按汉字机械切词，不用随机embedding，不采用作者参考代码中“最近Gloss”替换缺词的回退。权重、tokenizer／配置的7个文件均绑定SHA；进程驻留，stdout为带请求ID的JSONL，模型加载及一次固定文本warmup在服务准备阶段完成。

当前beam4 worker的[服务状态记录](../../artifacts/instant-composition-20261108/current-service-status.json)显示启动load约11.43秒、warmup约15.01秒。这是该次服务初始化，包含身份检查／依赖与模型准备，不是每句输入都重新等待。旧`probe-v2`仍记录早期beam1，不能用它冒充当前beam4启动证据。

真实测试发现“明天早上我们一起去医院看朋友。”的Gloss仍漏掉“医院”。已加入**有限的已知地点／时间词保留检查**：原文中的已知锚点若在Gloss中遗漏，返回`needs_review`及`omitted_terms`，不生成动作、不自动补译。这个检查并不覆盖所有概念、否定、数量、空间指代或错误词序；换成beam4也没有修复这条漏译。语言准确性仍未验收。

## 真实HTTP与浏览器证据

本轮`http-audit-v2.json`使用实际beam4服务。五条输入在全20,654条索引中，原句及去标点匹配均为0。

| 输入 | 实际状态 | 后台规划＋组合 | 热HTTP往返 |
|---|---|---:|---:|
| 明天早上我想去图书馆学习。 | ready，8个Gloss／114帧 | 183.94 ms | 485.00 ms |
| 今天晚上我们一起做饭。 | ready，5个Gloss／77帧 | 112.64 ms | 187.83 ms |
| 今天我不去学校。 | ready，5个Gloss／73帧 | 109.02 ms | 126.99 ms |
| 明天早上我们一起去医院看朋友。 | needs_review，遗漏“医院”，无motion | 160.68 ms | 177.59 ms |
| 我想去成都。 | blocked，缺“成都”动作，无motion | 125.51 ms | 148.61 ms |

前三条返回参数已核对源词帧、SHA及`source_kind=retrieved_word_composition`。ready只表示工程上得到可播放组合，不表示文字规划或手语语言已批准。例如第一条Gloss为“明天／早／我／想／去／书／房子／学”，其表达是否可被目标CSL使用者正确理解仍须人工验收。

本轮在真实浏览器完成前三条库外输入的beam4组合及自动播放至结束。页面记录的**提交至首个playing状态**约0.33秒、0.20秒、0.20秒；人物及规划器均已准备。计时来自`performance.now()`和播放器状态回调，**不是首个实际像素、冷启动或摄像头端到端延迟**。这三例达到此前1–2秒的预热交互目标，但不足以报告P95、全场景达标或任意中文保证。完整页面截图已由CUA成功观察，尚未导出本地图片文件；[最终浏览器记录](../../artifacts/instant-composition-20261108/browser-cua-audit.json)保留操作与状态证据。

早期HTTP／beam1失败记录保留；不能用新结果覆盖旧失败状态。此前完整句组合54.8–67.5 ms的微计时也只测后台片段组合，详见[前期调研](../research/instant-sign-composition-2026-11-08.md)，不与本轮网页计时混写。

## 定向回归与范围

本轮回归分组：Python **109通过**（服务、组合核心及旧场景／CLI），隔离runtime **49通过**（builder16、planner33），Node **33通过**。具体命令与范围见[验证记录](../../artifacts/instant-composition-20261108/verification.json)。这是定向检查，不是全仓库／CI。

语言、手形、非手控语法、自然度、目标地区CSL使用者理解、原始视频时间基、全仓库／CI及相机端到端均`NOT_RUN`。慢模型选项保留，本轮没有重新跑完整动作生成GPU验收；其两条4秒演示约5分钟的历史结果仍见[前轮报告](free-text-sign-demo-2026-11-08.md)，原run02仍见[CLI生成报告](custom-expression-generation-2026-11-07.md)。

本机证据位于`artifacts/instant-composition-20261108/`：`assets-status.json`、`builder-fixture-audit.json`、`word-library-build-audit.json`、`http-audit-v2.json`及逐条`http-v2-*.json`，另含`current-service-status.json`、`browser-cua-audit.json`与`verification.json`。旧probe、HTTP及资源不足记录保留；本机产物不随Git发布。

复现资产、建库和启动步骤见[动画工作台说明](../csl-animation-workspace.md)及[配置示例](../../configs/animation/word-composition.example.json)；设计／任务见[spec](../superpowers/specs/2026-11-08-instant-sign-composition-design.md)和[plan](../superpowers/plans/2026-11-08-instant-sign-composition.md)。
