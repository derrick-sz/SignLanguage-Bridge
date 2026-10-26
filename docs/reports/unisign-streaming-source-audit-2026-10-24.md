# Uni-Sign 流式逐词识别与 LLM 表达辅助：源码审计

日期：2026-10-24（Asia/Taipei）。工作区：`E:\githubproject\SignLanguage-Bridge`；Git HEAD `cf291c2a81558faf6de0dc5a7243e730531e4367`。本报告审计的是当前工作区文件，其中 Uni-Sign、continuous、candidate_context 等包含未提交或未跟踪内容，不能把 HEAD 当成这些文件的完整版本身份。上游源码固定为 `eed438bcb49e30405cd6ccdfcccca330c134e830`，权重固定为 Hugging Face revision `182913ca7a8b7dd9837d7d6f10717353f4408f80` 的 `csl_daily_pose_only_slt.pth`。

范围：本审计只读代码、配置、官方论文及文档，另用 CPU 合成时间戳核查窗口边界；只新增／更新本报告，未自行启动 GPU、训练、视频评测或 LLM 服务，未修改生产代码、配置或历史证据。后续更新读取并引用主任务已经完成的修复、测试和24条dev诊断证据；没有重跑GPU，也没有新增产品验收通过结论。

## 结论

**当前系统能持续接收视频并重复生成窗口级中文候选，尚不能提供经过验证的“每完成一个手语词就稳定提交一个词／字”的原生流式识别。** 输入可以持续采集，模型仍按有限窗口整体重算，输出可以撤回或重写。

**“先识别手语词义，再由 LLM 组织意思”可以作为后续分层目标，但当前 Uni-Sign 权重承担的是 SLT 句子翻译。** 它已经把视觉动作组织成中文句子；把这些句子再交给 LLM，属于对已有译文的表达辅助，不能称为已经实现了独立的逐词识别层。

**候选上下文保存与导出已实现，LLM 接入尚未实现。** 当前可以提供带来源和重叠关系的候选快照，没有实际 provider 调用、LLM 草案返回、澄清或采纳闭环。依据：[当前运行配置](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:385)、[候选合同](E:/githubproject/SignLanguage-Bridge/slbridge/candidate_context.py:87)、[界面导出提示](E:/githubproject/SignLanguage-Bridge/slbridge/desktop.py:729)。

**审计发现的帧数／时间跨度冲突已由主任务修复。** 原配置在30及29.97 FPS合成输入中360帧均未产生快照；当前Uni-Sign入口改用实际收到的帧数就绪，保留8秒硬上限及原有时间模式。回归测试通过，但真实RecognitionSession回放仍得到0条连续识别结果，不能据此宣称实际流式链路验收通过。详见第4节。

| 能力 | 当前源码行为 | 不能据此声称 |
|---|---|---|
| 持续读取摄像头／视频 | 按源时间戳缓存有限窗口，后台调度 | 每个手语词都已被识别 |
| 重叠窗口复用 | 复用每帧 RTMPose 点与分数 | 复用视觉 encoder 或语言 decoder 的跨窗口状态 |
| 中文文本输出 | 每窗一次完整 `generate`，返回整串文本 | 字／词的视觉时间对齐或稳定前缀 |
| 连续两窗文字相同 | 标记 `confirmed`，下次不同仍可修订 | 人工确认、语义真值或不可撤销提交 |
| 候选上下文 | 保存原始结果、来源、重叠和修订 | 无重叠的词序列、已确认句子或已接 LLM |

## 1. 当前模型输出的是 SLT 中文句子

[UniSign runtime](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:384) 明确初始化 `dataset="CSL_Daily", task="SLT", rgb_support=False`。上游训练代码只有在 `task=="CSLR"` 时，才把目标由 `gt_sentence` 改为 `gt_gloss`；这一改变发生在训练监督里，不能只改推理参数就让句子权重变成 Gloss 识别器。见[上游本地 fine_tuning.py:213](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/fine_tuning.py:213)及[固定官方源码](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/fine_tuning.py#L207-L219)。论文也分别把 ISLR、CSLR、SLT 的目标定义为动作描述、Gloss 序列、翻译文本。[论文 §3.2](https://arxiv.org/html/2501.15187v2#S3.SS2)。

当前返回 [TextPrediction](E:/githubproject/SignLanguage-Bridge/slbridge/contracts.py:364)，在 [uni_sign.py:920](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:920) 填写整句 `text`、`language="zh"`、`score=None`。没有逐词／字的 `time_range`、Gloss ID、N-best 列表或 token 分数。Beam=4 目前用于搜索一条最终输出，代码没有请求多条候选及其生成分数。`T5Tokenizer` 的 token 是 SentencePiece 编码单位，不能直接认作一个手语词或一个汉字；界面逐字显示一条译文也不等于逐字视觉识别。[官方 v4.40.0 tokenizer 文档](https://huggingface.co/docs/transformers/v4.40.0/en/model_doc/mt5#transformers.T5Tokenizer)。

当前 [GlossSequencePrediction](E:/githubproject/SignLanguage-Bridge/slbridge/contracts.py:341) 还限定 decoder 为 CTC 的 `greedy/prefix_beam`。若未来接入 Uni-Sign 的生成式 CSLR，需要对应的目标域权重及明确的生成式 Gloss 输出合同；不能悄悄借用 CTC 的分数、边界和 decoder 名称。

## 2. 三处结构使它不具备当前输入下的增量不变性

### 2.1 时间卷积使用未来邻帧

[models.py:91](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/models.py:91) 用 temporal kernel=5 创建每分支的时间网络。[stgcn_block.py:67](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/stgcn_layers/stgcn_block.py:67) 使用对称 padding `(kernel_size−1)//2`，不是只向左填充；[同文件:121](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/stgcn_layers/stgcn_block.py:121) 的 temporal 链有3层。对单独的这段卷积而言，输出会用到左右各最多6个采样位置，新增未来帧可能改变右边界附近的旧输出。这不是完整模型的固定6帧延迟保证，因为后面还有全窗口处理。[固定官方源码](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/stgcn_layers/stgcn_block.py#L67-L128)。

### 2.2 mT5 encoder 处理全窗口

本地依赖锁定 Transformers **4.40.0**，并与已安装 `transformers/__init__.py:21` 一致。该版 [modeling_mt5.py:1582](E:/githubproject/SignLanguage-Bridge/.conda-env/Lib/site-packages/transformers/models/mt5/modeling_mt5.py:1582) 将 encoder 设为 `is_decoder=False/use_cache=False`；[modeling_utils.py:1058](E:/githubproject/SignLanguage-Bridge/.conda-env/Lib/site-packages/transformers/modeling_utils.py:1058) 对 encoder 的二维输入掩码只做 padding 广播，不添加未来位置的因果掩码。本地 runtime 又把所有有效帧的 mask 设为1。因此 encoder 可以查看窗口内全部视觉位置，新增帧可改变早期表示。[官方同版本 mT5 源码](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/mt5/modeling_mt5.py#L1582-L1592)。

### 2.3 每个窗口重新计算 body 坐标基准

[uni_sign_preprocess.py:142](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign_preprocess.py:142) 从当前窗口全部有效 body 点求 xmin/xmax/ymin/ymax，并据此生成 scale、平移；[prepare_pose_window:195](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign_preprocess.py:195) 对每窗重复执行。手和脸也除以这个窗口的 body scale。窗口扩展或左侧旧帧被丢弃时，即便某一帧 RTMPose 点没变，它的归一化坐标仍可能改变。

这是一处容易遗漏的流式障碍：**缓存原始点不代表已缓存可直接追加的神经特征。** 不能仅把 Conv 改成因果卷积，就宣称旧 encoder 表示可以永久复用；更改归一化合同本身也需训练／回归验证。

主任务后续在24条dev视频上检查增长前缀，24/24均观测到已包含帧的body归一化坐标相对完整视频发生变化。这是当前样本的机制证据，不是所有帧或所有输入均会改变的断言。[followup summary](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/summary.json)。

## 3. 文本自回归、文本流式显示和视频流式识别是三件事

上游 [models.py:313](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/models.py:313) 的 `generate` 在**一个已固定的视频窗口**上，依次生成目标文本 token。模型运行时没有在生成途中接收新视频帧；现有 adapter 也在 `generate` 结束后才 `batch_decode` 并返回整串文本，见 [uni_sign.py:421](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:421)。

单次 `generate` 可以内部使用 decoder KV cache；本地 mT5 config 的 `use_cache=true`，官方文档说明这种缓存用于加速序列解码。**当前缺失的是跨视频窗口的 encoder 状态、decoder KV 和已提交文本前缀复用，不是“库完全没有 KV cache”。** 上游调用不传 `past_key_values`、`encoder_outputs`、`decoder_input_ids`；runtime 也没有保存这些状态。变化了全窗口视觉上下文后，直接沿用旧 cross-attention KV 不能保证与当前完整解码等价。[官方 v4.40.0 cache 合同](https://huggingface.co/docs/transformers/v4.40.0/en/model_doc/mt5#transformers.MT5ForConditionalGeneration.forward)、[固定上游 generate](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/models.py#L313-L321)。

即使增加 token streamer，它也只是把当前固定窗口的译文更早显示出来。当前 Transformers 4.40.0 的 [generation/utils.py:1504](E:/githubproject/SignLanguage-Bridge/.conda-env/Lib/site-packages/transformers/generation/utils.py:1504) 还明确拒绝 streamer 与 `num_beams>1` 同用，而本地默认 beam=4。更换 beam 或解码方式需要单独测速度／质量，不能把接口变化计作现成的逐词流式能力。[官方同版本实现](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/generation/utils.py#L1504-L1507)。

### 一处确定的额外计算

目前上游 [models.py:291](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/models.py:291) 先调用 mT5 完成带标签的 forward/loss，但返回值只有视觉 `inputs_embeds`、mask、loss，没有 `encoder_outputs`。后续 `generate` 再以 embedding 启动，[Transformers generation/utils.py:1449](E:/githubproject/SignLanguage-Bridge/.conda-env/Lib/site-packages/transformers/generation/utils.py:1449) 会重新运行 encoder。主任务后续的24条dev完整调用均计数为 **2次encoder调用**，与静态定位相符。这是性能优化候选；没有量化重复部分的净毫秒占比，也没有证明消除它就能达到实时目标。句点 sentinel 不进入生成条件，不构成参考文本泄漏。[实测计数来源](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/summary.json)。

本地 mT5 与 generation 文件的 Git blob SHA 分别为 `84a9f78ca91ec53f95b5302b61baa78e7875748d`、`002cea9d73ca9d5c6af1eef2d454188595ef137b`，与 GitHub 插件读取 v4.40.0 官方源码相等。

## 4. 当前“连续识别”实际如何工作

[RollingWindow](E:/githubproject/SignLanguage-Bridge/slbridge/continuous.py:57) 保存有限 RGB 帧；满时间、帧数或内存上限会丢弃最早帧。当前指定minimum_frames时以实际收到的帧数判断就绪；未指定时继续使用minimum_seconds，随后返回完整 `CapturedClip`。[session.py:212](E:/githubproject/SignLanguage-Bridge/slbridge/session.py:212) 仅在前一次任务结束、手部观察仍有效且达到更新间隔时提交完整窗口，过期结果被丢弃，再处理最新画面。因此配置更新间隔不是每个词的处理时延，也不保证每个间隔都产生候选。

Uni-Sign CLI 默认48帧窗口、16帧名义步长；按默认标称30 FPS换算，更新间隔约0.533秒。[desktop_cli.py:90](E:/githubproject/SignLanguage-Bridge/slbridge/desktop_cli.py:90) 仍用 `capture_fps` 派生更新间隔、名义时长预检及结果时效预算，**没有在这里把实际帧流重采样到30 FPS**。修复后48帧的就绪不再取决于标称1.6秒，但仍受8秒窗口、内存、观察门控等约束。不同实际 FPS 或丢帧率下的响应需实测，不能把48帧固定写成摄像头真实1.6秒。

### 4.1 标称帧率附近的窗口跨度冲突：已修复

修复前CLI同时设置 `minimum_seconds=window_frames/capture_fps`、`window_seconds=相同值`、`max_frames=window_frames`，先将帧数裁到上限，再检查末帧减首帧的跨度。

对理想等间隔F FPS，N帧的跨度为 `(N−1)/F`，小于原先要求的 `N/F`。这与解码速度无关。在修复前默认配置下，只用32×32零图像、合法递增FramePacket时间戳，连续调用append/snapshot 360次，CPU结果如下：

| 标称FPS | 实际合成FPS | minimum_seconds | 最后缓存首末跨度 | 有效snapshot次数 |
|---:|---:|---:|---:|---:|
| 30 | 30 | 1.6 | 1.566667 | 0 |
| 30 | 29.97 | 1.6 | 1.568235 | 0 |
| 30 | 60 | 1.6 | 0.783333 | 0 |
| 30 | 25 | 1.6 | 1.600000 | 320 |

25 FPS是能达原先时间门槛的机制对照。320是逐次检查得到的快照数，不是实际模型调度数或吞吐。此表保留为历史RED证据，**不描述当前修复后行为**；独立滑窗benchmark绕过RollingWindow时不能覆盖该缺陷。

主任务现已在 [continuous.py:20](E:/githubproject/SignLanguage-Bridge/slbridge/continuous.py:20) 增加可选 `minimum_frames`，并在 [snapshot:113](E:/githubproject/SignLanguage-Bridge/slbridge/continuous.py:113) 分开处理帧数模式与原有时间模式。Uni-Sign [desktop_cli.py:102](E:/githubproject/SignLanguage-Bridge/slbridge/desktop_cli.py:102) 和 [smoke_unisign_session.py:37](E:/githubproject/SignLanguage-Bridge/scripts/smoke_unisign_session.py:37) 均指定minimum_frames=N、max_frames=N、window_seconds=8。新增[就绪回归测试](E:/githubproject/SignLanguage-Bridge/tests/test_unisign_window_readiness.py:16) 覆盖25/29.97/30/31.26/60 FPS等机制场景；[RED记录](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/window-red.log:65) 为11 failed，修复后[组合回归](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/window-green.log:2) 为42 passed，[全量记录](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/full-pytest.log:53) 为797 passed、10 skipped。这些验证帧就绪及软件回归，不证明词级准确率或实时性。

主任务随后用真实RecognitionSession回放 `dev-00015.mp4`，[session-frame-readiness.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/session-frame-readiness.json) 记录error=null，但 **continuous_result_count=0、confirmed_count=0**，事件含“等待双手清晰进入画面”、质量门控清空及EOF。因此不能把“无异常”写成真实连续识别成功，也不能把剩余无输出全部归于已修复的帧就绪缺陷；仍需独立定位观察质量、源生命周期和调度等环节。

### 4.2 原始点缓存与文字重复提示

缓存发生在 [uni_sign.py:767](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:767)：键是 source/frame_index/height/width，值是每帧点和分数。每次窗口仍重跑预处理、GCN、mT5 和解码。[运行信息](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:614) 写 `sliding_window_over_complete_windows`，[结果元数据](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:903) 写 `streaming=False`，与实际代码一致。

[TextStabilizer](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:150) 只在完整文本连续重复达到阈值后标记 `confirmed`。新窗口文字不同会重置 streak、增加 revision；没有最长公共前缀提交、词级等待策略、视觉对齐、句尾检测或词级撤销事件。`confirmed` 只表示窗口一致；[captions.py:103](E:/githubproject/SignLanguage-Bridge/slbridge/captions.py:103) 明确显示“尚未人工确认”。

主任务在24条dev视频上逐步增加前缀，共70次相邻文本转换，其中58次没有保留上一条完整文本作为前缀，24/24视频都至少出现一次改写。这支持“当前窗口候选不是稳定词前缀”的判断；它是截断视频的稳定性诊断，不是完整词级在线准确率／延迟，也没有人工证明每次改写的语义对错。[followup summary](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/summary.json)。

## 5. 候选上下文能作为 LLM 输入材料，但没有接入 LLM

[CandidateLedger](E:/githubproject/SignLanguage-Bridge/slbridge/candidate_context.py:46) 在有效推理完成后保存原结果和来源；[snapshot](E:/githubproject/SignLanguage-Bridge/slbridge/candidate_context.py:87) 明确声明 `status=unconfirmed`、`overlapping_windows=True`、`sentence_boundaries=unknown`、分数未校准和 `automatic_translation=False`。相邻窗口用 `overlaps_candidate_id` 标注关联；重复文本仍可能是同一动作的多次修订。

[session.py:768](E:/githubproject/SignLanguage-Bridge/slbridge/session.py:768) 在 UI 合并事件前归档，另提供[快照／JSON 导出](E:/githubproject/SignLanguage-Bridge/slbridge/session.py:308)。它没有在此调用 LLM。当前 slbridge 源码搜索未发现实际 `LLMProvider`、`semantic_assistance` 或 chat/provider 调用；[desktop.py:736](E:/githubproject/SignLanguage-Bridge/slbridge/desktop.py:736) 也写“LLM尚未接入”。

用户设想中的“词义→LLM联想”还缺三类实质能力：有来源的词义事件与边界；跨窗去重／修订和提交规则；实际异步 LLM 调用与草案／澄清／确认处理。现有[设计草案:3](E:/githubproject/SignLanguage-Bridge/docs/LLM中葡表达辅助智能体_设计草案_2026-10-11.md:3) 及[候选交付报告](E:/githubproject/SignLanguage-Bridge/docs/reports/recognition-candidate-context-2026-10-22.md:3) 都没有把这些写成已完成实现。

如果先用当前窗口候选做表达辅助原型，应把多条候选当成**同一动作的可能修订**，让模型保留歧义与来源，而不是将其直接拼成“已经说过的一串词”。尤其 Uni-Sign 当前没有 N-best 和词级分数，不能假装下游拿到了视觉模型的可靠词级概率。

## 6. 残余低相似度中仍值得隔离的输入协议

以下是已定位差异及待实验范围，不是新的原因贡献比例。已有几何／方向修复与回归以[方向修复记录](E:/githubproject/SignLanguage-Bridge/docs/reports/unisign-cecsl-direction-repair-2026-10-24.md:1)为准；本报告不把更早的7.99当成最新修复成绩。

| 项目 | 确定的源码事实 | 仍未证实的部分 |
|---|---|---|
| 姿态提取模型 | 论文 A.1 为RTMPose-x；当前demo/manifest为lightweight RTMW-L/M 256×192 | 作者发布pose包的精确checkpoint、换强前端的净收益 |
| 真实时间尺度 | 按实际帧序进入网络；max_frames采样无FPS重采样；模型不接收实际时间戳 | 各设备帧率/丢帧对动作速度分布的损失 |
| 训练与在线动作边界 | CSL-Daily loader使用已有start/end；在线窗口受有限帧数／8秒硬上限约束；超长序列抽样方式也不同 | 已有24条dev固定启发式trim对照，不能替代正确动作边界的消融 |
| 进入姿态器前的分辨率 | 连续窗口超过640宽先等比缩小；文件路径默认读取原帧 | 缩小对检测、手指定位和文本质量的影响 |
| 低分数点与人物身份 | confidence≤0.3清零，实例默认取第0人；没有该路径下的跨帧签者跟踪 | 遮挡／多人场景误选人的发生率和影响 |
| RGB细节 | 当前权重rgb_support=False，只有选定69点的几何/分数 | 对应RGB-pose权重在同域/目标域的收益，不能给pose-only直接开开关 |
| 训练目标域 | 当前为CSL-Daily SLT；公开配置未包含CE-CSL | CE-CSL适配训练的净改善量及陌生签者泛化 |

本地位置：[姿态点和阈值](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign_preprocess.py:118)、[文件读帧/FPS仅用于时长](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:825)、[在线缩小](E:/githubproject/SignLanguage-Bridge/slbridge/continuous.py:93)、[确定性采样](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign_preprocess.py:100)、[上游训练边界/随机抽样](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/datasets.py:461)。官方来源：[论文A.1](https://arxiv.org/html/2501.15187v2#A1)、[前端mode映射](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/demo/rtmlib-main/rtmlib/tools/solution/wholebody.py#L53-L76)。

主任务后续在同一24条dev视频重新提取姿态，完整输入预测24/24复现先前记录；固定启发式动作trim确实裁剪24/24条。其BLEU-zh由 **10.32099→10.71497**，平均字符编辑相似度由 **31.1895%→30.5079%**。两个指标方向不一致，不支持“裁剪已经显著修复残余差距”的结论；此处不是人工标注边界，也不是新的盲测／产品成绩。[summary.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/followup-20261024/summary.json)。

官方名为 `online_inference.py` 的 demo 也先把整个视频读入 `vid_data`，完成提点后才推理；文件名本身不是原生流式证据。[本地demo:81](E:/githubproject/SignLanguage-Bridge/models/external/uni_sign_source/demo/online_inference.py:81)、[官方固定版本](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/demo/online_inference.py#L81-L104)。

## 7. 对用户目标的可行性判断与验收缺口

当前代码可以作为“持续窗口候选＋可追溯上下文”的实验底座，不能把现有句子SLT直接标为逐词SLR。若目标必须是稳定逐词／字事件，应明确目标单位（Gloss词、孤立词、指拼字符或译文字符），采用对应监督及输出合同，再建立在线窗口／状态、词级提交与撤销。单独压缩窗口、加文本streamer或接大语言模型，都没有解决这些视觉与时间合同。

后续验证应至少区分：完整片段质量、逐前缀新增证据后的输出修订、已提交词的撤回／替换、重复与遗漏，以及动作可辨识时刻→原始词可见→稳定提交→LLM草案可见的各自延迟。窗口结束后的model_ms不是这一整条延迟；生成中文更流畅也不证明原始手语词识别更准。

**本审计已完成：** 当前源码与官方固定版本的静态研究、实际依赖版本核对、输出／缓存／非因果结构定位、上下文与LLM接入状态核查、360帧CPU合成时间戳的历史窗口边界复现；后续只读核对主任务的帧数就绪修复及测试日志、24条dev真实提点／trim／增长前缀／encoder计数，以及0条输出的真实Session回放记录。

**仍为NOT_RUN：** 原生增量模型与稳定词提交验收、真实摄像头端到端P95、系统性FPS／分辨率／强弱pose对照、人工标注动作边界消融、词级准确率与时间对齐、LLM调用和语义验收、目标域训练。24条增长前缀诊断和软件回归均不能替代这些验收；本审计没有重跑GPU。
