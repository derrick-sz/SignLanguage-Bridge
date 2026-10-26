# Uni-Sign 后续核查：剩余低相似度、实时窗口与词级 LLM 接入

日期：2026-10-24。基于当前工作区实际源码和新运行证据。

**仍有前轮遗漏。** 本轮确认并修复了一处桌面窗口无法就绪的调度错误，实测确认每次完整推理重复执行两次 mT5 encoder，并补做了动作裁剪与视频前缀诊断。调度和重复计算影响实时链路，不能解释离线 500 条回归的全部低相似度。当前权重仍是 CSL-Daily 句子 SLT，没有经过 CE-CSL 适配。

**当前 Uni-Sign 可以被完整窗口反复调用；尚不具备稳定逐词、原生增量视频识别能力。** 用户希望的“手语词识别 → LLM 意思联想”需要词级事件和提交规则，不能直接把当前整句生成结果拆成汉字当作已识别动作。

## 1. 新发现与对低分的解释范围

| 项目 | 本轮证据 | 结论与状态 |
|---|---|---|
| CLI 窗口调度边界 | 旧配置最多 48 帧，要求首末跨度至少 48/30=1.6 秒；理想 30 FPS 的 48 帧跨度只有 47/30≈1.5667 秒。旧机制 30/29.97 FPS 各 360 帧均不能就绪 | 确定代码错误，**已修复**。影响连续候选调度，不改变完整文件的 BLEU |
| 重复 mT5 编码 | 在 24 条 dev 完整推理上挂真实 encoder hook，全部观察到每次调用 2 次 encoder | 确定效率遗漏，**尚未优化**。上游先算未使用的 sentinel loss，随后 generate 再编码；未声称可获得特定加速倍数 |
| 姿态前端不等价 | 当前 lightweight 为 RTMW-L/M 256×192 配置；论文 A.1 写 RTMPose-x | 确定存在配置差异，准确率贡献 **尚未量化**。更强前端的对照仍待执行 |
| 完整视频包含等待帧 | 上游数据 loader 使用已有 start/end；本地默认整片。新 24 条 dev 使用固定运动分位裁剪，逐片真实重新提取姿态 | 有影响，但本次收益混合，未证明是剩余低分的主因 |
| 输入帧率与短窗训练 | 本次 dev 24 条中 23 条 30 FPS、1 条 60 FPS；识别前端按收到的帧处理，没有训练时序对齐实验 | 不应将“48 帧”无条件等同于固定动作时长；帧率归一化和短窗训练仍待控制实验 |
| 目标任务／训练域 | 当前代码固定 `CSL_Daily, task=SLT, rgb_support=False`，输出整句中文 | CE-CSL 目标域适配尚未进行；“统一框架”不等于当前权重已掌握所有数据集的词级动作 |

论文明确按下游数据集分别微调，且区分 ISLR、CSLR、SLT 的任务与指标；当前检查的权重属于 SLT。[Uni-Sign 原论文 §4.1–4.2、A.1](https://arxiv.org/html/2501.15187v2)

前轮权重加载已检查 missing/unexpected key 均为零，且本轮新提取的 24/24 完整视频预测复现开发集旧结果；目前没有证据支持“权重没有正确加载”作为剩余低分的原因。

## 2. 24 条 dev 新实验

沿用已锁定的 dev 方向表，选前轮与校准样本不重叠的 24 条 dev 验证视频。它们现在用于诊断，不能再当作未接触的盲测集。本轮不读取 test 参考句，不重新选择方向，不改权重。

从视频重新提取原始姿态，再分别构造完整片段和固定 `05-95_pad5` 运动分位裁剪：累计运动量取 5%–95%，两侧各保留 5 帧；该规则在观察结果前固定。每个裁剪片段重新做身体 bbox/scale 与手／脸归一化，未沿用完整片段的归一化坐标。

| 同一批 24 条 dev | 完整片段 | 固定裁剪 |
|---|---:|---:|
| BLEU-ZH | 10.3210 | 10.7150 |
| chrF | 13.2748 | 13.2585 |
| 字符编辑相似度 | 31.1895% | 30.5079% |

24 条均实际裁剪，完整输入的预测 24/24 复现前轮。裁剪使 BLEU 小幅增加，但其他两项没有改善；这是启发式动作边界实验，未使用人工逐词／逐句边界。不能据此认定边界问题已解决，也不支持将低分主要归因于等待帧。

前缀实验使用视频最初 32、64、128 帧（仅取短于原片的长度）及完整片段，每次按同一方向重新推理。共 70 次相邻输出比较，**58 次新文本未保留上一条完整文本作为前缀**；24 条视频均至少发生一次此类改写。24 条均存在旧帧身体坐标随窗口 bbox/scale 改变的现象。

这些是文本稳定性和输入表示变化的诊断，不是将部分视频对照整句参考得到的“逐词准确率”。句子模型看到更多视频后修订文本并不意外，但应用必须处理撤回和改写，不能把每次输出都追加为新词。

证据：[summary.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/summary.json)、[实验脚本](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/probe_boundaries_and_prefixes.py)。原始重新提取的姿态位于同目录 `raw-dev-poses/`，与历史缓存分开。

## 3. 窗口错误的修复和验收

新增可选 `ContinuousConfig.minimum_frames`：缺省时保留现有按时间判断的行为；指定时按实际收到的帧数判断就绪。Uni-Sign CLI 和会话 smoke 脚本设置 `minimum_frames=N, max_frames=N`，并保留 8 秒硬时间界限。标称 FPS 继续用于步长换算和参数预检，不再把 N 个时间戳要求成 N/F 秒的跨度。

| 检查 | 结果 |
|---|---|
| 新回归测试 RED → GREEN | 新增 11 项先失败；修复后与 continuous/session/desktop 组合 **42 passed** |
| 实际 CLI 配置接线 | 25、29.97、30、31.26、60 FPS 均在第 48 帧就绪，后续 360 帧内窗口保持有界 |
| 普通时间窗口、来源中断、过期和 EOF 处理 | 独立 CPU 复审通过，默认时间模式语义保留 |
| 全项目回归 | **797 passed、10 skipped、15 warnings**；9 项未安装 Torch 的 TFNet 测试和 1 项未配置真实 live 输入跳过 |
| 修改范围 | `slbridge/continuous.py`、`slbridge/desktop_cli.py`、`scripts/smoke_unisign_session.py` 和新窗口测试；没有改权重或姿态识别代码 |

请求帧数仍需在 8 秒和字节预算内积满。帧率过低或预算不足时不能承诺该配置必定出窗；帧数就绪也不表示已经识别正确。

代码：[continuous.py](E:/githubproject/SignLanguage-Bridge/slbridge/continuous.py)、[desktop_cli.py](E:/githubproject/SignLanguage-Bridge/slbridge/desktop_cli.py)、[窗口测试](E:/githubproject/SignLanguage-Bridge/tests/test_unisign_window_readiness.py)。修改前快照与哈希保存在 `artifacts/unisign-cecsl-eval/followup-20261024/before/` 和 `before-sha256.json`。

**真实会话候选交付尚未通过。** 另用 dev-00015 原视频、CUDA、48/16 帧配置运行实际 `RecognitionSession`：无抛出的运行错误，但 `continuous_result_count=0`，日志记录质量门控清空和 EOF 停止。此结果不能算“实时识别成功”，也不能仅据这些日志就断言全部归因于某个门控阈值。需要进一步记录活动观察、实际积累长度、任务提交与 EOF 前交付，区分门控、窗口与推理耗时。

该 smoke 脚本的退出码 0 只由 `error is None` 决定，验收必须读取候选计数和稳定性字段；本轮明确将该真实回放列为未通过候选交付。未为获得结果而关闭质量门控。

证据：[session-frame-readiness.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/session-frame-readiness.json)、[全量测试日志](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/full-pytest.log)。现场摄像头、真人语义和实时延迟 P95 均 **NOT_RUN**。

## 4. 当前模型能提供的粒度

| 能力 | 当前 CSL-Daily Uni-Sign SLT |
|---|---|
| 完整片段 → 中文句子 | 已实现，跨域语义质量仍低 |
| 持续采集 → 完整窗口反复推理 | 已有框架；本轮修复帧数就绪，真实候选交付仍有未验收项 |
| 新帧到来后复用过去视觉 encoder 状态 | 当前未实现；时间卷积含未来邻域，mT5 encoder 双向，全窗口归一化也会改变旧输入 |
| 稳定逐词 Gloss 及动作时间边界 | 当前权重／adapter 不提供 |
| 字符或 token 流式输出 | mT5 的文本自回归属于单次文本生成；不等于视频增量识别，文本 token 也不等于手语动作 |
| 跨窗口增量解码 | 当前无跨窗口 encoder／decoder 状态复用；单次 generate 内部可能使用自身 KV cache，不应说“完全没有 KV cache” |
| 词候选置信度／N-best | 当前 `TextPrediction` 为整句 `text`、`score=None`，beam=4 最终仍只返回一条文本 |
| LLM 意思联想 | 现有候选快照与 JSON 导出可供后续消费，实际 LLM 尚未接入 |

完整源定位见 [源码审计](E:/githubproject/SignLanguage-Bridge/docs/reports/unisign-streaming-source-audit-2026-10-24.md)。上游名为 `online_inference.py` 的示例先读取完整文件再处理，并不构成原生摄像头增量推理。

## 5. 如何适配“单字／词识别后交给 LLM”

建议将识别单元定义为**手语词／Gloss 或审核过的动作词项**，而不是中文字符。生成文本的子词 token 与视觉动作之间没有当前可用的一一时间对齐。

```mermaid
flowchart LR
    A[连续摄像头帧] --> B[按目标词表训练的在线词级识别]
    B --> C[背景与未知处理]
    C --> D[词候选去重及稳定提交]
    D --> E[词义与本句上下文]
    E --> F[LLM 表达草案或澄清]
```

这条链路与当前 Uni-Sign 的整句翻译用途不同。现有 CE-CSL TFNet 的输出是 Gloss 序列，更接近中间接口，但其 manifest 也明确是完整片段、非因果模型；不能因为用了 CTC 就宣称支持原生流式。

可以沿本地已研究的 Zuo 等在线 CSLR 路线验证：离线教师对齐构造词样本，训练带背景的短窗词级识别器，再通过投票、重复处理与提交规则形成连续 Gloss。作者工作还使用了受监督 gloss-to-text 网络；将该模块换成通用 LLM 属于本项目的新实验，性能需要重新验证。[EMNLP 2024 原论文](https://aclanthology.org/2024.emnlp-main.619/)、[本地既有研究](E:/githubproject/SignLanguage-Bridge/docs/reports/online-cslr-study-2026-10-23/README.md)

LLM 可以组织语法和给出带来源的意思草案；输入需保留原始词候选、时间范围、是否稳定以及分数的实际含义。缺词或歧义时提供澄清，原始识别与辅助表达分开保留。当前的候选账本是重叠窗口修订记录，不能直接拼接成一句已经识别的完整话语；也没有可假造的 Uni-Sign 词级概率。

## 6. 下一步验证顺序

1. **准确率链路**：在固定 dev 条件下对照更强姿态前端，并补同权重的官方 CSL-Daily 姿态参照，分开定位前端、模型与目标域差异。再决定 CE-CSL 的 SLT 适配或词级 CSLR 训练；不按 test 分数继续选参数。
2. **实时链路**：记录真实会话的门控合格率、积累帧数、模型提交和结果撤回，先达到可重复的候选交付，再优化重复 encoder 计算。度量应覆盖摄像头到稳定词提交，不能只用网络 forward 耗时。
3. **LLM 链路**：先固定词级事件合同和审核词义，再接表达辅助；分别测词识别 WER／未知拒识、提交延迟／改写率、LLM 语义保真与端到端交流效果。

本轮保留前轮 500 条缓存重解码的 BLEU 12.3019 及其证据；新调度修复不改变该成绩。没有重新训练模型、接入 LLM 服务、现场摄像头验收、提交或推送代码。
