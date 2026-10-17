---
task_id: c
role: Project and data fit auditor
objective: Compare practical applications against current code, datasets, and verification boundaries; support both a usable product and a truthful China-Brazil competition demonstration.
status: complete
confidence: high_for_scoped_repository_facts_medium_for_product_priority
sources_found: 14
acceptance_met: yes
---

## Sources

All sources below are local first-party project records/code inspected on 2026-10-11. They establish this project's state, not user demand or commercial viability. Source audit documents were read; raw videos/NPYs were not re-audited in this thread.

- [C1] 场景语义覆盖审查与候选 CSV | `E:/githubproject/SignLanguage-Bridge/docs/中巴手语竞赛_场景语义覆盖审查_2026-10-09.md:7-76`; `docs/reports/libras-scene-candidates-2026-10-09.csv`; `docs/reports/csl-scene-candidates-2026-10-09.csv` | LOCAL_AUDIT / CANDIDATE_ANNOTATION | 2026-10-09. CSV group sums were recomputed in this thread.
- [C2] Libras 新增特征数据审计 | `E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:7-43,77-96` | LOCAL_AUDIT / PROPOSED_EXPERIMENT | 2026-10-10.
- [C3] 共同语义审核清单及当前 JSON | `E:/githubproject/SignLanguage-Bridge/docs/CE-CSL_Libras共同语义审核清单_2026-10-09.md:3-23,70-91`; `docs/reports/cecsl-libras-semantic-review-2026-10-09.json` | LOCAL_AUDIT / UNREVIEWED_CATALOG | 2026-10-09. Current JSON summary and all 178 records were counted in this thread.
- [C4] 当前字幕词典及显示逻辑 | `E:/githubproject/SignLanguage-Bridge/slbridge/resources/libras_caption_catalog.json:1-19`; `slbridge/captions.py:20-59,96-124` | LOCAL_CODE / CURRENT_RESOURCE | live. Actual caption JSON review statuses were counted in this thread.
- [C5] 数据/模型契约及当前表示注册 | `E:/githubproject/SignLanguage-Bridge/slbridge/contracts.py:33-79,341-375`; `slbridge/model_adapters.py:64-66,121-168,240-262`; `slbridge/sequence.py:94-112,510-511`; `slbridge/media.py:219-264` | LOCAL_CODE | live.
- [C6] 自动分段及完整片段推理 | `E:/githubproject/SignLanguage-Bridge/slbridge/session.py:327-405`; `slbridge/live.py:15-89` | LOCAL_CODE | live.
- [C7] 字幕事件与 HTTP 会话 | `E:/githubproject/SignLanguage-Bridge/slbridge/caption_events.py:14-31,51-108,114-126,149-169`; `slbridge/api.py:605-678,680-815` | LOCAL_CODE | live.
- [C8] 桌面结果及字幕列表、视频源 | `E:/githubproject/SignLanguage-Bridge/slbridge/desktop.py:171-227,438-450,497-521`; `slbridge/sources.py:140-218`; `slbridge/session.py:219-258` | LOCAL_CODE | live.
- [C9] GPU 验证报告 | `E:/githubproject/SignLanguage-Bridge/docs/reports/ctc-gpu-inference-2026-10-11.md:3,43-53,71-90` | LOCAL_BENCHMARK | 2026-10-11.
- [C10] 自动字幕已测与未测 | `E:/githubproject/SignLanguage-Bridge/docs/reports/auto-captions-2026-10-11.md:7-23,31-57,80-85` | LOCAL_ACCEPTANCE_REPORT | 2026-10-11; opening GPU update supersedes historical CPU-environment CUDA-unavailable entries.
- [C11] LLM 中葡表达辅助草案 | `E:/githubproject/SignLanguage-Bridge/docs/LLM中葡表达辅助智能体_设计草案_2026-10-11.md:3-25,43-67,90-121,139-150` | PROPOSED_DESIGN | 2026-10-11.
- [C12] 数据与训练基础设施 | `E:/githubproject/SignLanguage-Bridge/slbridge/extraction.py:81-113,116-223`; `slbridge/training.py:80-101,178-186`; `slbridge/contracts.py:133-165` | LOCAL_CODE | live.
- [C13] 另一训练工作树的 TCN 输入/池化 | `C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:31-32,112-172`; `slbridge/temporal.py:81-99,136-142` | LOCAL_CODE_IN_SEPARATE_CHECKOUT | live; not registered in current app checkout.
- [C14] 中文序列 API 与逐词时间限制 | `E:/githubproject/SignLanguage-Bridge/docs/CE-CSL_TFNet_ONNX接入与验证_2026-10-09.md:72-75`; `docs/CE-CSL_Libras共同语义审核清单_2026-10-09.md:82` | LOCAL_CONTRACT_REPORT | 2026-10-09.

## Findings (facts only)

1. **现有数据的垂直方向不对称。** Libras 场景候选文件含 178 个来源条目、5,347 个视频文件。重新汇总 CSV 得到：停电故障 42 / 1,259，复电/服务申请 16 / 479，缴费支付 23 / 690，账单日期 25 / 749，账单查询 18 / 540，联系方式 22 / 665，导航 8 / 246，确认否定等辅助 23 / 689，不明确 1 / 30。该分组是依源葡语文字制作的待审候选；含重复，不能等同独立语义或独立签者数量。 [C1]
2. **模型已支持范围比数据小。** 当前冻结试点 12 类仅涉及上述故障 5 类、服务 3 类、辅助 4 类；缴费/日期/账单各候选组当前模型支持数均为 0。原数据有完整表达，不能推成当前模型已能识别。 [C1][C3][C4]
3. **校园选题有材料但缺配对语料。** CE-CSL 候选 CSV 实际 50 条记录、49 个不同中文句；多数具体句在候选表仅一条录制。Libras 来源文本没有直接标注“我来自巴西”“欢迎来学校”“请说慢一点”“请再说一遍”的完整对应表达。此处是文字清单核查，不能证明全部未标注视频没有相关概念。 [C1]
4. **可运行的审核语义为零。** 当前实际字幕 JSON 12 条全部 `draft`。实际共同语义 JSON 的 178 个记录中，非空 `shared_semantic_id` 为 0，启用 production mapping 为 0；summary 的 `reviewed_label_meaning_mappings`、`approved_production_mappings`、`reviewed_caption_count`、`verified_sign_form_alignments` 均为 0。31 个条目有比较方向，未获人工语言批准。 [C3][C4]
5. **Libras T 包不是现成 gloss 监督。** 既有全量审计记录 5,335 行 `label_gloss=[]`、176 种葡语自然文本序列；原始 178 条目与文本类别不一一对应。模板中的省略号不提供实际月份、账户等槽值。原 split 有 12 对跨 dev/test 完全重复和 6 对 train 冗余；局部 `INFORMANTE 1–5` 不能作为全球唯一签者身份。 [C2]
6. **严格输入契约已实现，新的骨架路由未接入。** 当前注册表示为 hand63、两种 pose/hands225 和 TFNet RGB。225 提取器为 33 pose + 双手 42 点、有效 z=0、肩宽归一化，没有脸部特征。另一训练工作树的表示名为 `libras_full318_blockmask321_v1`，318 数值加 3 个候选块观察标记；其因果卷积分类头对完整已给定长度做平均池化，不能据此声明已有原生增量流推理。225 补零/截断不能形成 321 协议。 [C5][C13]
7. **字幕工程可复用。** CaptionLedger 保存片段起止、来源、动作 event ID、会话/source epoch、模型 generation；提交结果还保存原 prediction、字幕、模型/预处理版本、runtime 与计时。HTTP v3 会话能创建、读回、撤销、清空、关闭、换源失效，具有冲突/容量保护；桌面可载入字幕词典、顺序预览视频、暂停、呈现字幕列表并撤销。 [C7][C8]
8. **它还不是字幕编辑器。** 所查当前 `slbridge` UI/ledger/API 未找到逐条文字编辑、人工审核修订、持久工程、SRT/VTT 导入导出、任意时间轴定位/裁切、音频/ASR 转录入口。结果文字是可选择的 QLabel，历史是 QListWidget；ledger 数据保存在内存，默认历史最多 200 条，提交时裁去更早事件。视频源现有 seek 是首轮时钟预检查后回到零帧，不能算编辑器的随机定位。 [C7][C8]
9. **识别输出与发布字幕仍有一层语义工作。** CSL CTC 返回 gloss；显示明确说明尚未变成自然中文/葡语句子。`prefix_beam` 不估计逐词对齐，time_range 为 null；CE-CSL 没有人工词级起止。Libras 固定分类与当前中文草案也未完成审校。 [C4][C5][C14]
10. **GPU 修复仅缩短完整样例计算。** 同一 8.2 秒视频三次 GPU 完整文件耗时 1.917–1.990 秒、HTTP 2.069–2.250 秒，模型前向 0.541–0.559 秒；真实算子 CUDA 和 CPU 输出等价得到验证。未重新对照动作真值打分、未证明动作结束到字幕显示 P95≤2 秒。自动字幕报告仍披露 9 个原始视频默认端点均未产生 ready、摄像头探测没有动作真值、未知动作拒识及自然长流质量未验收。 [C9][C10]
11. **流采集不等于流识别。** 当前活动 observer→AutoSegmenter 等到 ready→`infer_clip(segment.clip)`。RF captured clip 再建立/重置 Holistic，并采样完整片段提取；当前 CTC `begin_stream` 抛 capability unavailable。 [C5][C6]
12. **语义辅助仍是设计。** LLM 草案明确本轮只交付设计。当前 `slbridge` 搜索没有 SemanticCatalog、LLMProvider、semantic_assistance、UtteranceBuilder、ExpressionValidator 或共享场景/义项字段的实现命中。其设计要求独立草案、人工确认、审核义项/模板、事件来源和不覆盖原始预测。 [C5][C11]
13. **数据工具有现成基础但缺专家工作界面。** 已有源/派生 hash、质量隔离、可恢复提取缓存、分组划分契约、分类 confusion matrix/逐类召回/coverage 等；现有训练代码明确没有 unknown/OOV 验证集，低分阈值仅是启发式。未发现能让专家逐条观看、记录语义决定、审改边界并冻结发布的产品界面。 [C12][C3][C8]

## Analysis (our synthesis)

优先级是基于代码/数据的条件性判断，实际需求强弱由 A 线程证据及中巴本地试点决定。

| 方向 | 可以复用的核心 | 首版可完成的真实任务 | 尚需补齐 | 条件性次序 |
|---|---|---|---|---|
| **视频字幕/语料审校工作台** | 时间/来源、typed model 草案、字幕 ledger、视频预览、数据 hash/质量审计 | 专家或创作者校正一段视频的草案、调整片段时间、保存审核版本、导出可发布字幕/训练清单 | 原文与人工修订分离；定位/重复播放；人工边界；自然语言审校；持久工程；导入/导出。口语字幕需求还需要独立 ASR 来源或外部字幕导入 | 产品入口优先候选；不等自动识别完美即可验证编辑任务，但需要验证真实工作节省 |
| **已审核真人短句视频 + 个人沟通板** | 稳定来源键、标签/文本分离、审核状态与双语显示；可接同一事件/语义目录 | 用户选定常用表达→确认→展示文字/合法授权的完整真人短句视频；识别仅作为可关闭快捷入口 | 真实用户短语选择、审核/授权视频目录、可点击播放、文本/数值输入和双方确认；当前不是已经完成的功能 | 与审校共享资产，端到端任务验证较易；不能把逐词视频串联称为自然手语句子 |
| **电力/服务有限意图助理** | 巴西故障/服务/日期候选较多，模型/API可独立按语言路由，结果可撤销 | 例如选择/识别“停电报修”意图，确认对象/地址/回拨信息，再由工作人员核对；文本/点击承担未覆盖槽位 | 目标少量完整意图语义审核与重训、多签者/OOV/自然流试验；中国侧电力完整表达与任务数据可能需补采；准确的否定/日期/数量确认 | 数据适配最强的识别驱动产品候选；产品流程先限制于提交前草案，不能由分类分数直接执行缴费/改账期 |
| **校园/中巴活动接待交流** | 中文已有校园/澄清选题、同一 API/事件/审核核心 | 问路、报名、约见、请求重复等一组双方自然表达；用户确认译文，文字/视频回退 | Libras 目标表达补采、双侧完整意图审核，真实交流频次与参与伙伴；候选单条录像不够训练 | 竞赛主线适合；现有数据复用率低于电力，不能靠旧词表改名字解决 |
| **学习与练习助手** | 视频预览、模型词表、关节点表示，可使用同一审核视频目录 | 查阅完整参考、录制练习、并排/慢放回看，由教师给反馈；几何偏差仅作线索 | 教师审核参考、变体范围、练习任务与用户试验；模型标签正确不等于手形/语法全部正确，225 无脸部/z信息限制反馈 | 可做内容/回看工具；自动“标准手语评分”后置，需要专用错误标注 |
| **口译员/字幕员辅助** | 事件来源和时间、候选草案、审校版本设计 | 离线术语检索或短片草案审查，保留人工最终责任及原视频 | 长片持久存储/定位、专业术语与目标语言审核；实时旁路、干扰/纠错成本试验 | 同审校台自然延伸；尚不能称长会话自动翻译器或专业口译替代 |

### 双路径：同一核心服务产品和竞赛

- **共享核心建议**：录制/导入 → 来源/时间契约 → 原始识别草案 → 语义/文字审核修订 → 双语确认 → 授权真人视频或文本呈现。审校数据和版本化语义目录可同时为产品与竞赛服务，识别模型可更换而原证据保持可追溯。
- **产品路径**：先用一段真实待处理视频或一个真实沟通任务比较当前人工方法与工具；计总完成时间、修订量、错误遗漏、确认/回退和接受范围，而不只展示模型 accuracy。
- **竞赛路径**：准备明确声明词/固定表达范围的双侧自然表达集；随机留出的真人签者现场识别、包括否定/未知动作/重做，再让另一方核对“实际收到的意图”。摄像头实际识别、手动点击表达与预录参考播放在界面及结果中分别标明。可先用手动结束完整动作；不要用合成停顿/训练视频证明自然流质量。
- **竞争演示可验证的两项价值**：一是识别→确认→另一方正确理解的闭环；二是发现误识别后如何回看、纠正、保持原记录。已审核预录短句可用于回复/回退，但不能代替被宣称的现场识别。
- **近期顺序判断**：字幕/语料审校和短语目录是可以与模型优化并行的实用入口；电力有限任务优先复用巴西数据；校园中巴演示按两侧伙伴/补采条件建设。不要把“再扩大词表”当成独立的产品价值。

## Support snippets / paraphrases for top claims

- Claim: 产品缺口不只有模型速度，审核资产尚为空。
  Source: [C3][C4]
  Support: 实际资源文件 12 个显示字幕全为 draft；实际 178 条语义记录全部共享义项空、生产映射关闭。此线程以 JSON 重新计数确认，没有只引用旧路线图。
- Claim: 电力服务最符合现有巴西来源，但无法现成覆盖完整客服。
  Source: [C1]
  Support: 候选 CSV 重新汇总故障 42 / 1,259、服务 16 / 479；当前 12 类没有任何 payment/date/billing_query 类，文件数量并非当前分类能力。
- Claim: 视频字幕审校台复用程度高，但必须补编辑与持久保存。
  Source: [C7][C8]
  Support: ledger 已记录片段时间和模型证据、API 有读回/撤销；现桌面展示控件只读，ledger 是内存列表并按 max_history 截尾。没有发现 SRT/VTT、编辑/持久工程或 ASR 管线。
- Claim: 完整 GPU 样例快了不证明实际实时字幕可用。
  Source: [C9][C10][C6]
  Support: GPU 报告明确计时边界和未打真值分；自然原视频默认分段均没 ready；当前代码仍等待片段 ready 再 infer_clip。
- Claim: 新骨架/葡语任务不能直接复用旧输入/输出假设。
  Source: [C2][C5][C13]
  Support: 321 是 318+3 mask 的另一工作树输入；当前表示注册没有该协议；Libras label_gloss 为空而 label_char 是自然葡语，不能 padding 成225或按空格伪造gloss。

## Conflicts / unresolved issues

- 2026-10-09 场景审查记载 T.zip 不可用；2026-10-10 新特征审计已经读取 T 解包资产。本线程以新审计判断 T 标签状态，同时保留旧候选场景统计的身份边界。[C1][C2]
- 自动字幕报告正文保存旧 CPU 环境 CUDA 不可用/693秒历史；同报告开头及新 GPU 报告确认环境修复。不可重复旧不可用结论，也不可把历史 693秒与最终样例时间当成稳定加速比。[C9][C10]
- 174 ready 条目、另训练工作树的321骨架模型及178来源均不同于当前12类部署范围；它们没有建立可运行双语语义目录、自由句翻译或新签者/长流接受证据。[C2][C3][C5][C13]
- CaptionCatalog 的 reviewed 字段为展示目录状态，不能自动证明 CSL/Libras 成对动作/完整意图审核。应用需单独保存语言、表达范围、专家、来源视频和版本。[C3][C4]

## Leads discovered

- 可直接以 `docs/reports/cecsl-libras-semantic-review-2026-10-09.json` 的 source clip examples/csl examples 设计专家审核任务；禁止自动提高现有状态。
- 字幕/标注工作的共同记录应是源视频+时间段+原模型输出+人工修订+审核版本；与当前实时 ledger 的会话容量分开设计持久项目。
- 电力候选可从完整报修/否定/日期表达选择少量目标；需要业务工作人员与 Libras 使用者共同核对流程。竞赛校园选题则针对真实缺失表达补采。

## Gaps

- 本线程未访谈/联系任何用户、教师、口译员或机构，不能建立中国/巴西需求频次、愿付费、部署预算或试点可获取性。
- 没有逐段观看候选语料，所以只能说“来源文字候选”，不能认定其动作与中文完全等义。社区成员的自然用法和变体范围需审核。
- 商业/公开展示所需数据许可、录制人物授权和二次传播权未在本线程完成核查；不能因数据已下载就认作可公开用于产品素材。
- 既有模型的跨签者、未知动作、自由长流及实际任务成功率仍缺接受证据；本线程没有训练、摄像头测试或重新评分。
- 中国聋人创作者研究若主要针对口语字幕校正，本项目现有 CSL 视频识别不是 ASR 替代。应把用户任务先定义清楚，再决定导入外部字幕还是新增独立语音来源；不能把两种需求混为同一识别任务。
- 能改变优先级的关键新证据：已有两侧参与伙伴愿意持续做真实任务；审核表达与合法参考资产可取得；与当前人工方法相比审校/沟通工具确实减少耗时和重要误解；现场留出签者的可接受范围得到独立验证。
