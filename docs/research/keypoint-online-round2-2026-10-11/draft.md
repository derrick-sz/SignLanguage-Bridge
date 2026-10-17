# 关键点轨迹在线识别：第二轮研究与可执行验证路线

2026-10-11｜增量研究｜主工作区 de316f52bc618387e0536daeee80899b947fd5bd

**建议把下一条路线收敛为：小词表、一次持续提取、共享时间路径的多模板 DTW、独立的拒识与字幕确认；随后在同一输入上比较因果 TCN 和单向 GRU。** 这条路线有明确的算法依据和工程入口，尚没有本项目的摄像头准确率、未知动作拒识率或新延迟结果。本轮补充的是上一轮未解决的输入协议、在线匹配和增量状态问题。[上一轮记录](/E:/githubproject/SignLanguage-Bridge/docs/reports/keypoint-streaming-feasibility-2026-10-11.md)

## 1. 关键点应该分成“手形”和“手在哪里、怎样移动”

MediaPipe 手部 world 坐标以**每只手自己的几何中心**为原点，不能直接把左右手的这些坐标相减，当作两手的全局物理距离。拟议特征应同时保留局部手形、身体相对的腕部轨迹和同一坐标框架中的双手关系；若只保留每只手扣腕后的坐标，会丢掉手在额头还是胸口、整只手向哪里移动等信息。Tasks Hand Landmarker 默认只检测一只手，双手任务需显式配置；它与项目当前 legacy Holistic 不是同一接口。[1]

工程核查发现，活动观察器已在一次 `process()` 中获得身体和双手坐标，却只输出可见性和运动量；分类随后对完整动作重新提取。**复用原始结果的入口存在，但不能直接复用它的质量布尔值**：观察器要求一只手的21点全部有效，分类归一化则允许逐点缺失，任一点有效就把该手记为 observed。当前225维表示的有效 z 固定为0，脸不在输入中；它不是完整3D手语表示。[代码1]

因此，先建立新版本的持续关键点包：点序、逐点观察 mask、身体锚点质量、源时间及其时钟域、左右手身份、采样/丢帧、提取器版本、tracking/reset epoch。活动判断和识别共用坐标，各自保留判定条件。持续全帧跟踪与旧“每段 reset、完整动作最多抽30帧”的提取分布不同；即使同为225维，也不能冒用旧协议。318/321输入另有定义，不能补齐或截断接入。[代码1][模型1]

## 2. DTW 能边看边算，提交字幕还需要另一层判断

SPRING 原论文给出了任意流起点、每来一帧更新模板匹配的动态规划；但它也明确指出，**第一次距离过阈值就输出，可能错过更好的重叠匹配**。在线计算不代表词义此刻已确定；共享前缀词仍要等区分动作出现。[2]

建议先采用 DTW_D：把身体、双手各维的帧距离合成，沿同一时间路径对齐，保留双手同时运动的关系。逐维各自对齐的 DTW_I 作为消融。多变量研究没有证明其中一种普遍更好，因此这只是针对同步关系的基线选择。[3]

拟议实现必须防止四个陷阱：

- **缺失越多反而越像。** 仅平均共同有效点的距离，可能奖励坏帧；无共同有效点不能记零距离并推进模板完成。必须另记覆盖不足、必要手部缺失和断档。
- **归一化悄悄改变目标。** 先求最小累计和路径，再除路径长度，不等于求最小平均路径；前者只能按声明的启发式评分使用。
- **时长过滤漏掉合法匹配。** 只保留一个最佳起点，再把时长不合格的路径过滤，可能丢掉合法次优路径。严格时长/斜率约束需增加状态或维护有界起点候选。
- **文本去重吞掉重复词。** 同一次动作的重叠命中要合并；连续两次同词应保留两个事件。词内部的重复运动又可能属于一个词，需要分别标注测试。

这些是本轮提出的设计检查项，不是原论文已经验证的手语系统。先做精确、可检查的对照；UCR 的下界剪枝依赖目标距离与下界相容，本轮没有证明其对自定义缺失mask多维距离仍安全。[9]

## 3. 小模型真正增量运行，训练上下文也必须一致

作者 TCN 源码每个残差块有两次卷积，原 `forward` 没有状态缓存接口。每层应缓存自己的历史输入激活；仅保存原始帧窗口仍会重算中间层。对两卷积、核宽k、L块、dilation为2^l、stride=1的结构，感受野推导为 `R=1+2(k−1)(2^L−1)`。例如k=3、L=4得到61帧范围，含当前帧和60帧历史；这不是61帧未来等待。单向 GRU 可把上一块返回的 `h_n` 传给下一块的 `h_0`，省略则默认从零开始。[4][5]

更关键的是：**“持续编码后平均最近W帧”与“每个W帧窗口 reset 后重新编码”通常不等价**，因为窗头看到的历史不同。训练须采用持续上下文，或带左侧历史的窗口并只监督目标区域。当前实验TCN对完整有效片段取平均；即使可用 sum/count 精确更新前缀均值，也没有证明完整片段分类头的前缀分数可以确认词义。[模型1] 归一化、插值、mask和reset同样必须只使用允许的过去信息；当前依赖完整片段右端点的时间插值不能直接搬入零前视管线。[代码1]

EMNLP2024 的在线手语研究支持“窗口训练＋过渡背景＋投票”的思路，但实际使用 **RGB＋63点热图双流S3D**，词典构建依赖预训练CTC与真实gloss序列强制对齐；不能直接套到只有自然语言整段标签的数据。[6] 作者发布的评估入口读取整段张量、生成全部窗口再批量推理，属于数据集评估；论文在线机制可以改造成到达帧尾窗，仍需独立验证摄像头、端点填充和字幕等待。[7]

## 4. 按这个顺序验证，能定位失败在哪一层

下表均为**待实施实验**。词表先取经审核、能找到真实使用者的小集合，模板只来自训练集；不预设论文里的帧数、阈值或准确率。

| 步骤 | 实验 | 能回答的问题 | 后续选择 |
|---|---|---|---|
| E0 输入 | 相同原视频比较旧reset抽帧、持续跟踪、全帧提取及真实丢帧回放，分别隔离变量 | 缓存后坐标、mask、初始化和成本变化多大？ | 冻结新协议，重新生成模板/训练输入 |
| E1 可分性 | 正确动作边界下的精确完整片段DTW_D，多示范，按源内容切分；身份可核验时另按签者切分 | 已观察完整动作时，小词表本身是否可分？ | 不可分先找缺失线索；可分再研究在线端点 |
| E2 连续流 | 同模板的有界起点在线DTW，独立候选与提交器 | 自然过渡、未知动作、内部停顿和重复词会不会误发字幕？ | 在固定误报预算下评估漏识、覆盖与提交延迟 |
| E3 学习模型 | 同一输入/切分/确认策略，一个小因果TCN＋一个单向GRU | 跨人变体或模板规模问题能否改善？ | 用实测错误与耗时决定保留哪个 |

DTW_D完整片段对照是定位工具，人工切片成绩不代替连续流成绩。孤立片段拼接只能做受控软件检查，不能代表自然连续动作。采样保留真实Δt；现代Hand Landmarker异步模式忙时会忽略新输入，回调完成时间不能代替观测时间。可把1€因果滤波列为单项消融，比较抖动改善与附加滞后，保留原坐标；通用滤波文献不证明它提高本项目识别。[1][8]

训练/验证/测试按源内容组冻结，重复、裁剪和增强同组；签者身份不清时只能报告内容组独立。已有Libras审计有缺失和dev/test重复记录，不能直接宣称新签者测试；从原RGB按新协议生成示范优先于转换未冻结schema的NPY。[数据1]

## 5. 拒识与延迟比孤立分类准确率更决定实用性

输出至少区分：**无目标活动、正在观察但证据不足、词表外/无法可靠识别、观测质量不足、可确认候选**。低运动不自动代表动作结束；静态保持和词内部停顿都可能仍有意义。原始DTW距离、softmax和连续几帧相同argmax，均不能单独保证可确认。

验证集同时放入已知词、近似库外词、日常动作、半动作、遮挡、双手交叉、共享前缀、同词两次、内部重复和不回静止的相邻词。比较事件precision/recall、每小时误提交、OOV误接收、拒识后的覆盖率、重复漏检，以及动作结束到最终字幕P50/P95；阈值在validation定，test冻结后一次评价。等待/拒识阈值形成质量与延迟的曲线，而不是只选最快设置。

当前账本可复用来源绑定、动作身份和按序最终提交，但没有逐帧候选的 revision/retraction 协议；需要可撤回的 partial，才能让早期提示和稳定字幕分别有清晰意义。相同文本的不同动作身份必须保留。[代码2]

计时也需修正口径：当前model `total_ms` 从取得推理lease后开始，不含活动提取、动作末端等待和UI显示；摄像头timestamp是应用读帧返回时刻，文件源启动前则做整文件时间戳预检。媒体时间与单调墙钟不能直接相减。报告排队、处理帧龄、丢帧/最大gap、提取、模型、确认及渲染各阶段；没有曝光时点就标应用接收后的帧龄。[代码2]

已验证的CTC CUDA运行不证明关键点提取图或小TCN也用GPU。新路线应分别测提取器和batch=1模型的CPU/GPU耗时及传输成本，再选择设备；本轮没有新的速度结果。

## 6. 更实用的落点与升级条件

结合真实产品和中巴交流演示的双重目标，优先做两种受限闭环：**示范跟练的轨迹反馈**，以及**经审核的少量词条/固定表达识别后供使用者确认**。前者可先验证形状与轨迹比较是否能帮助完成练习；后者需要自然流的拒识与稳定字幕。两种手语语言分别管理词表、示范和标签，展示相似度不代表语法评价，识别表达ID也不自动获得逐词gloss或双语句子翻译。

升级依据应来自错误：完整动作在新签者上仍不可分，先检查关键点/手形/身体位置是否丢信息；模板可分而连续流误发，先修边界与拒识；DTW模板规模或变体受限，再引入同协议TCN/GRU。只有混淆分析证明精细接触、面部或外观是缺失线索，才增加对应RGB/面部观测；若仅对示范者本人有效，可定位个人少量命令，不能推广为跨人手语识别。

**下一项最值得实施的是持续关键点录制/回放与DTW对照闭环。** 它把现有提取成本、输入变化和词表可分性同时变成可测证据，再决定是否需要学习模型。

## 研究边界

正文采纳9项外部一手/官方来源，并核对本地代码；未实现DTW/缓存接口、重提特征、训练或测试摄像头。VISAPP2022骨架monotonic论文与ICCV2025骨架分析PDF访问失败，未据其摘要采用机制/性能数字。旧数据质量数字只来自已有审计产物，不是本轮重跑。人工与独立评审用于检查论据和设计表述，不能当作产品验收。

## 本地证据

- [代码1]：[活动提取与质量](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:436)、[归一化质量与二维z](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:228)、[提取协议](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:134)、[完整动作分类](/E:/githubproject/SignLanguage-Bridge/slbridge/live.py:59)。更多采样/插值支持见[工程笔记](/E:/githubproject/SignLanguage-Bridge/docs/research/keypoint-online-round2-2026-10-11/research-notes/task-c.md)。
- [代码2]：[文件时间预检](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:157)、[相机时钟](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:227)、[字幕提交](/E:/githubproject/SignLanguage-Bridge/slbridge/caption_events.py:80)、[模型计时](/E:/githubproject/SignLanguage-Bridge/slbridge/runtime.py:359)、[自动队列](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:369)。
- [数据1]：[已有Libras特征审计](/E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:18)。
- [模型1]：[实验TCN](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/temporal.py:141)、[321数据协议](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:31)。均为只读核查。

## 参考文献

[1] [MediaPipe Hand Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).

[2] Sakurai et al. [Stream Monitoring under the Time Warping Distance](https://www.cs.cmu.edu/~christos/PUBLICATIONS/spring-icde07.pdf). ICDE 2007.

[3] Shokoohi-Yekta et al. [Generalizing DTW to the multi-dimensional case requires an adaptive approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC5668684/). DMKD 2017.

[4] Bai/Kolter/Koltun. [TCN author implementation, fixed file commit](https://github.com/locuslab/TCN/blob/8845f88f31def1e7ffccb8811ea966e9f58d9695/TCN/tcn.py).

[5] PyTorch. [torch.nn.GRU, current main documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html).

[6] [Towards Online Continuous Sign Language Recognition and Translation](https://aclanthology.org/2024.emnlp-main.619.pdf). EMNLP 2024.

[7] FangyunWei/SLRT. [Online CSLR prediction_slide.py](https://raw.githubusercontent.com/FangyunWei/SLRT/main/Online/CSLR/prediction_slide.py), main snapshot retrieved 2026-10-11.

[8] Casiez/Roussel/Vogel. [1€ Filter author page and CHI 2012 paper/implementation links](https://gery.casiez.net/1euro/).

[9] Rakthanmanon et al. [Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping](https://pmc.ncbi.nlm.nih.gov/articles/PMC6816304/). KDD 2012.
