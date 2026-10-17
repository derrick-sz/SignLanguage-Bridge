---
task_id: a
role: Online DTW algorithm researcher
objective: Decide a falsifiable multitemplate trajectory-matching baseline for a continuous keypoint stream
status: complete
confidence: medium
sources_found: 4
acceptance_met: yes
---

# A：连续关键点流的 DTW 起止、距离与提交

核查时间：2026-10-11（Asia/Shanghai）。本线程只研究算法，不实现代码，不测量本项目性能。原论文年代较早，但动态规划定义仍可核验；它们的其他语言、设备与数据结果均不迁移为本项目准确率或延迟。

## Search log

六个起始检索串：

1. `SPRING subsequence matching time series stream dynamic time warping Sakurai Faloutsos Yamamuro 2007 pdf`
2. `Shokoohi Yekta multivariate dynamic time warping dependent independent 2017 author pdf`
3. `Jangyodsuk Athitsos sign language recognition dynamic time warping 2014 PETRA pdf`
4. `"Generalizing DTW" "multi-dimensional" pdf ucr`
5. `"UCR Suite" "Searching and mining trillions" pdf author`
6. `"online dynamic time warping" gesture spotting initialization threshold original`

随后直接获取四份原文，按算法初始化、提交规则、相对位置/速度、多通道定义与剪枝假设定位段落。未采用搜索摘要、ResearchGate、二手实现博客作为核心证据。Jangyodsuk 的新版作者站与 UCR 的作者 PDF 在工具打开时失败，分别改用作者旧大学站 PDF 与 PMC 作者稿。检索返回的抓取日期不当成论文发表日期。

## Sources

- [A1] Yasushi Sakurai, Christos Faloutsos, Masashi Yamamuro. **Stream Monitoring under the Time Warping Distance**. ICDE 2007, pp. 1046–1055, April 2007. [作者/大学原文 PDF](https://www.cs.cmu.edu/~christos/PUBLICATIONS/spring-icde07.pdf). PRIMARY; authority high, mechanism relevance high, depth high; freshness low for contemporary deployment, adequate for the original DP theorem. Fetched full paper; §§3.1–3.3, Fig. 4, §4. Sources checked on 2026-10-11.
- [A2] Pat Jangyodsuk, Christopher Conly, Vassilis Athitsos. **Sign Language Recognition using Dynamic Time Warping and Hand Shape Distance Based on Histogram of Oriented Gradient Features**. PETRA 2014, May 2014; DOI 10.1145/2674396.2674421. [作者/大学原文 PDF](https://crystal.uta.edu/~athitsos/publications/jangyodsuk_petra2014.pdf). PRIMARY; authority high, sign feature relevance high, online relevance limited, depth medium. Fetched full paper; §3 and §4. This is isolated ASL retrieval, not a MediaPipe online validation.
- [A3] Mohammad Shokoohi-Yekta, Bing Hu, Hongxia Jin, Jun Wang, Eamonn Keogh. **Generalizing DTW to the multi-dimensional case requires an adaptive approach**. Online publication 2016-02-15; DMKD 31(1), January 2017, pp. 1–31; DOI 10.1007/s10618-016-0455-0. [PMC 作者稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC5668684/). PRIMARY; authority high, multichannel alignment relevance high, depth high; neither missing-mask SL nor online event handling tested. Fetched full article; §2.1, §3, §5, §7.
- [A4] Thanawin Rakthanmanon et al. **Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping**. SIGKDD 2012, pp. 262–270; DOI 10.1145/2339530.2339576. [PMC 作者稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC6816304/). PRIMARY; authority high, exact-search optimization relevance medium, depth high. Fetched full article; §1.2 and §4. Focused on normalized subsequence search; its benchmarks are not this project's runtime budget.

## Findings (facts only)

- 完整两段 DTW 以 `D(0,0)=0`、非零边界为无穷大，取末端匹配；SPRING 改为允许流中任意起点的 `D(t,0)=0`，其他初始模板前缀为无穷大，传播最优起点，并每帧更新整份模板的 DP 列。[A1]
- SPRING 完整模板末端提供“到当前帧结束的候选”。原文的 disjoint query 只报告重叠候选中的局部最优；首次距离过阈值立即输出可能错过更优重叠片段，严格规则会继续等待不可被后续重叠候选替换的条件。[A1]
- 固定模板长度为 `m` 时，SPRING 的状态与每帧更新均为 `O(m)`，不随已流入的总帧数增长；保留完整对齐路径另需历史信息。原文使用非负可加局部距离并说明可以替换该距离。[A1]
- 手语研究使用相对面部的手位置、运动方向和双手关系，再与手形外观结合。其运动差分使用 `t+1`；查询已是一个完整词，单/双手类型事先给定，评价包含 top-k 检索。[A2]
- DTW_D 用各维合成的帧距离共享一条时间路径；DTW_I 对各维分别对齐后累加。作者实验没有发现其中一种普遍占优；采用逐维 z-normalization，且分类与跨通道对齐方法的关系依赖数据。[A3]
- UCR Suite 使用逐候选归一化、下界级联与提前放弃来加速其精确搜索；可剪枝的前提是所用量确实是目标 DTW 的下界。[A4]

## Analysis (proposed design, not a source-proven project result)

### 1. 分开三个问题：对齐、判别、提交

Open-start/closed-template-end 的 DP 解决“过去哪一段最像完整示范”；它并不回答“是否在打手语”“是否属于库内词”“能否现在提交字幕”。前缀阶段可显示暂定候选，但两个词若共享相同前缀，观察尚未出现的区分动作前，坐标匹配无法消除歧义。即使整份模板能被当前帧勉强匹配，横向/纵向重复也可能造成时间过度压缩或拉长。

建议维护 `candidate → tentative → committed` 与单独的质量状态。提交要求至少检查完整动作覆盖、持续时间/允许速度、有效观测比例、类别最佳距离、第二名间隔和事件结束证据。距离/间隔阈值必须用同协议的 held-out 库内词、库外动作、过渡与日常动作标定；其输出是校准后的拒识决策，原始 DTW 距离不等于语义概率。

### 2. 长度归一化与约束不能随意加在标准 SPRING 上

原累计距离容易受模板长度与路径长度影响，跨词比较需要诊断。可报告 `sum_cost`、`path_length`、`mean_cost`、真实持续时间、压缩/拉伸统计；但**先求最小累计和路径，再除以该路径长度，并不等于求最小平均路径**。修改优化目标时要明确是“累计和最优路径的事后平均”启发式，还是带路径长度状态的另一种算法；不能继续声称继承原最优性定理。

同理，只传播一个最优起点，最后检查其持续时间上下界，可能拒掉非法最优路径而漏掉合法次优路径。若要精确遵守持续时间、斜率或状态约束，应把所需历史纳入状态，或维护一组有界起点候选。候选银行的每帧更新近似 `O(S × Σ m_k × d)`，`S` 是保留起点数、`d` 是比较维数；此处为设计成本推导，不是论文或实测时长。超时起点可按事先声明的任务持续时间范围淘汰，不把这一范围宣称为所有自然手语的语言事实。

普通 Sakoe–Chiba 对角带以已知两段起止为基础；不应直接把 `|template_i-global_stream_t|<r` 施于未知流起点。在线有界候选可以对相对候选时间/进度设带或使用限制连续横/竖步的状态，但支持的快慢比与路径集合都改变了，需重新评估。完整 DTW 是当前输入上的精确匹配；带约束 DTW 只在允许路径集合内精确；降采样改变信号，FastDTW 一类近似路径搜索则可能遗漏最优路径，四者必须在报告中区别。

### 3. 默认共享路径，独立路径作为消融

建议小词表首先用 DTW_D，因为身体、左右手和各关节同时发生的关系本身有意义。DTW_I 可放宽跨通道延迟，但也可能把不同时刻的左手与右手拼成一个貌似匹配的动作。它应在相同 split、同拒识要求下作消融，而不是按“多变量总是更准”决定。可选分组路径（单手内部共享，双手/身体不同组）同样是新设计，需专门验证其是否损失双手同步信息。

### 4. 形状与身体相对轨迹分组，避免过度归一化

拟议特征分为两路：手部局部形状可扣除腕点并用手掌尺度归一化；手的整体位置仍以可靠身体锚点和身体尺度表达，同时保留两手相对位置。只把所有手坐标扣除腕点会抹掉手在哪个身体部位与移动方向。逐段逐坐标 z-normalization 在通用时序分类中有用途，却可能抹掉手在额头/胸口、移动幅度与保持姿态等区分信息，还依赖当前候选的完整统计；不将 [A3][A4] 的归一化要求原封不动当成手语特征规则。尺度估计也需版本化、过去信息限定与失败状态。

速度可用过去帧差分除以真实 `Δt`，并声明过滤/插值策略；不能照搬 [A2] 的中心差分后声称零前视。零运动时单位方向向量未定义，应有专门状态，不能用任意方向。

### 5. 缺失不能变成“完美相似”

拟议局部距离按特征组及共同有效观测加权，同时记录/惩罚覆盖不足；需要明确定义某手为该词必需、某帧身体锚点是否有效。仅对重合有效维取均值会让只剩少数易匹配点的坏帧取得低距离。双方都缺失或没有公共有效点的帧不得默认为距离零并推动模板完成。长缺失应暂停/失效当前候选，记录恢复/reset；短缺失插值是实验策略，不能隐藏为真实观测。检测失败与背景空闲为不同状态。阈值来自校准，而非任意设置“超过 x% 就可信”。

当 mask/confidence 是每对帧的固定非负局部成本时，普通可加 DP 仍可以求该**新定义累计和**的最小值；这不证明该成本在缺失时符合语义，也不证明 LB_Keogh 对它仍成立。加入路径平均、动态权重或历史约束，更需重新审视状态是否充分。

### 6. 多模板、未知、重复词与规模

每词保留多签者/多次示范可涵盖变体，类别最小距离作可解释基线；但模板更多的类别有更多碰巧匹配机会。需记录每类模板预算，测试模板增长后库外误报，代表模板选择只在训练集合内进行。shared-path 两列 DP 的在线算术成本约 `O(d × Σ m_k)`，DP 状态约 `O(Σ m_k)`，模板特征存储另占 `O(d × Σ m_k)`；没有语言/设备无关的每帧毫秒保证。

同一真实事件被多模板/多个末端帧匹配时，应按时间重叠合并，并保留最好证据，而非每帧输出一词。但不同类别重叠的仲裁规则是本项目的新设计，SPRING 单查询的 disjoint 证明不替系统解决它。连续两次同词 `A,A` 要允许新的独立事件；仅按文本去重或固定 cooldown 会吞掉第二个词。语内重复次数属于一个词的内部结构时，又不能误拆成两个词。必须同时有 `A,A`、内部重复词、`A→B` 无停顿、相近共享前缀词以及日常动作测试。

先测精确 DP，再按实测成本选择少量代表模板、物理意义明确的维数/组、稳定重采样与受约束路径。UCR 的剪枝技巧值得参考，但本线程没有建立自定义 mask 加权多变量距离的 LB_Keogh 下界；不能宣称直接复制仍零漏检。降采样候选筛选如果只保留排名前 k，也可能丢词，除非给出适用下界/验证；可标成近似检索并测其漏检。

## Small-vocabulary baseline decision

建议先做**同特征、同示范库的精确完整片段 DTW_D 对照 + 有界在线候选 DTW_D**。前者测“有正确边界时是否能分词”，后者测“自然连续流能否找事件并拒识”，拆开对齐距离与边界故障。完整片段对照不得用人工切片成绩替代在线成绩。

在线实现可先用上述有界起点候选和明确累计和目标，保证约束状态可检查；SPRING open-start 作为低状态成本对照，明确其后过滤/提交是否启发式。两者都产生带证据时间段与模板 ID 的事件，正式字幕通过共同提交器输出。具体词数/模板数/阈值与时间范围需由审核词表、录制和 held-out 校准决定，本轮不凭其他论文给固定参数。

**会推翻这条选择的证据**：正确边界下跨签者的近形近轨迹词仍不可分；满足同一误报预算时 DTW 漏识/延迟不可接受；模板增长导致每帧处理积压；自然过渡与 OOV 持续被命中；或必要词义区分只存在未纳入的面部/外观。前四种促使在同数据同协议上比较窗口训练的因果 TCN/单向 GRU；最后一种先补特征/RGB，而不是仅换时序算法。若只在模板签者自己表现好，可将基线定位个人少量命令而非跨人手语识别。

## Support snippets / paraphrases for top claims

- Claim: 连续流可以允许任意起点，并只保留滚动状态。[A1] Support paraphrase: Eq. (4) 的 `d(t,0)=0` 用零代价星号前缀跳过流开头，Eq. (5) 传播起点；§3.3 的滚动数组每次更新模板长度个代价与起点。
- Claim: 立即阈值输出与最优重叠片段输出不同。[A1] Support paraphrase: §3.3.1 直接指出首次过阈值输出会遗漏更优重叠片段，Fig. 4 保留候选并等候无法被替换条件；示例在匹配结束后才报告。
- Claim: 单纯腕点局部形状不保留手语轨迹位置。[A2] Support paraphrase: §3 同时比较相对面部的位置、运动方向与双手位置关系；算法还使用完整词首末帧的 HoG。
- Claim: 多维共享/独立路径必须选择并验证。[A3] Support paraphrase: Definitions 3–4 分别给每维独立时间变形和合成距离共享路径，实验表明选择依赖数据结构，非一方普遍胜出。
- Claim: 不能无证移植下界剪枝。[A4] Support paraphrase: §4 的提前放弃依赖累计部分代价和剩余下界超过当前最好值，所用包络距离必须确为目标距离下界。

## Conflicts / unresolved issues

- [A3][A4] 强调各自时序任务的逐段 z-normalization；[A2] 使用手相对面部的位置以保留空间线索。两者不是直接矛盾，但说明归一化针对不同不变量，不能跨任务套用。
- [A1] “在线”指流式更新与尽早报告满足数学条件的匹配，并不保证动作结束前完成词义判断或固定提交延迟。
- [A2] 在较少样本时采用 DTW，但它不证明小词表在新签者、跟踪丢失和连续过渡下有效。

## Leads discovered

- [A3] 作者支持页 `https://sites.google.com/site/dtwAdaptive` 可用于核对实验代码；本轮没有据其宣称执行复现。
- [A4] 作者 UCR Suite 支持页可作为后续 exact-search 优化代码参考，先保持可重复的无剪枝基准。

## Gaps

- 未找到并验证适用于本项目 mask/confidence 权重、部分三维/身体相对坐标、动态候选长度的安全下界；当前建议不承诺 LB_Keogh 剪枝零漏检。
- 没有该项目的连续 keypoint 流、审核事件起止、真实 OOV/日常动作、新签者 split；拒识与提交阈值尚不可定。
- 没有模板规模/维数/提取开销/丢帧的实测结果；论文复杂度不能转换为本项目毫秒数。
- 未判定局部形状、身体轨迹、面部与手部 RGB 的必要组合；需要以审核词表中的最小混淆对消融。
