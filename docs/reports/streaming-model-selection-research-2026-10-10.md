# 实时手语模型选型研究笔记

日期：2026-10-10。范围：模型算法推荐与验证方案；本轮未训练、未运行 GPU／摄像头，也未修改产品代码。下述 Pose-TCN 是本项目拟设计的轻量变体，不是已验证产品型号。

> 新数据更新：`D:/Zenodo` 的 Libras 126／318 维完整序列已做实际审计。后续以[新增特征算法策略与执行路线](../Libras新增特征数据_算法策略与执行路线_2026-10-10.md)落实输入分组、缺失处理和数据协议。其葡语标注的 gloss 字段全空，近期先做词条／固定句式分类；318 特征不兼容旧 225 输入，发布 dev/test 存在已核验重复。

## 建议收敛到一个首选

**首选：双手／上身分组编码 + 共享轻量因果 TCN（本文简称 Pose-TCN）。** 用已有 MediaPipe 坐标与缺失 mask 建立第一版；保持中国手语与 Libras 的独立权重／词表和任务头，复用模型代码与服务契约。空间 GCN、手部 RGB 作为后续按消融结果增加的模块。RF 保留为基线，现有 TFNet 保留为中文离线基线及视觉预训练／教师候选。

这是现有 [Backbone 实验路线](../Backbone模型选型与实验路线_2026-10-09.md) B1／B4 的收敛方案，并非另外铺开一组必须全部实现的模型。选择理由是现有关键点契约可复用、模型可只依赖已到达帧、历史缓存可以减少滑窗重复计算。其在本项目的准确率和时延均尚未测量；不能预先承诺超过 RF 或达到任意毫秒值。

## 一手证据与边界

| 来源 | 可以支持的结论 | 不能据此推断的内容 |
|---|---|---|
| [TCN 原论文](https://arxiv.org/abs/1803.01271)、[作者实现](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py) | 因果、膨胀、残差时间卷积是成熟的序列建模方案；作者代码通过 padding 后裁掉右端实现因果 | 原论文的通用序列实验不能证明本项目手语识别优于 GRU／RF；官方代码也没有直接提供本项目的增量缓存 API |
| [SAM-SLR 论文](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html)、[SL-GCN 官方说明](https://github.com/jackyjsy/CVPR21Chal-SLR/blob/main/SL-GCN/readme.md) | 关节、骨向量、关节运动、骨运动四流和 RGB 互补是手语建模的有价值参考 | 论文 RGB 轨 98.42% 来自 AUTSL 挑战的 SAM-SLR 多模态系统，不能当作本地 Libras 结果、轻量骨架单流结果或摄像头准确率 |
| [SL-GCN 官方源码](https://github.com/jackyjsy/CVPR21Chal-SLR/blob/main/SL-GCN/model/decouple_gcn_attn.py) | 原版时间卷积 kernel=9、双侧 padding=4；注意力和末端池化也使用整个时间轴 | 原版并非零前视流式模型。改为左侧 padding、取消整段时间注意力／池化后，需要重新训练和验证，不能沿用原成绩 |
| [MobileNetV3 原论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | Small／Large 是面向不同资源预算的轻量视觉候选，可研究局部 RGB 分支 | ImageNet／手机 CPU 结果不能换算成 RTX、ONNX 或本项目手语时延／正确率 |

本项目的 [CE→Libras 冻结视觉探针](../CE-CSL_Libras迁移验证_2026-10-09.md) 在相同 12 类、72 条测试集上取得 100%，而历史 RF 为 63.89%。这提示视觉表示值得保留，但输入、表示和分类头均不同，尚不能把提升完全归因于模型结构。该划分未建立全局签者／背景独立性，探针使用完整视频，不是连续识别或实时能力证据。

## 与几类候选的取舍

| 候选 | 实时条件 | 对本项目的用途 |
|---|---|---|
| Pose-TCN | 所有时间操作只看当前及过去；导出状态／缓存；提交规则另计时延 | 第一优先，检验时序学习能否改善现有骨架分类；之后接 CTC |
| 单向 GRU／LSTM | 明确输入输出隐藏状态，禁止双向层 | 一个低成本对照；状态容易表达，但是否更快、更准仍需同设备实测 |
| SL-GCN／多流骨架 | 原版全片段，流式需要因果改造和去除全局时间依赖 | 若 Pose-TCN 的空间关系错误明显，再添加小型空间图模块；先不用四个完整网络集成 |
| 骨架 + MobileNetV3 局部 RGB | 裁剪只依赖已到达帧，视觉特征按时间缓存 | 若手形、遮挡错误较多再验证；手部裁剪不能取代非手部语义，按需要保留面部／上身视图 |
| 现有 TFNet | 当前双向时序网络使用完整片段 | 中文离线基线、预训练表示／蒸馏来源；导出 ONNX 不会自动变成流式模型 |

## 用户提供的 Omdena 项目能贡献什么

[项目概述](https://omdenaai.github.io/SaoPauloBrazilChapter_BrazilianSignLanguage/)与[详细报告](https://omdenaai.github.io/SaoPauloBrazilChapter_BrazilianSignLanguage/report#results)提供了一个可复查的孤立词基线：25 类、每类 6 条视频，最佳 LSTM 为 2 层、256 hidden，使用 20 帧、189 维工程特征，报告 Top-1 为 66.37%，且承认过拟合。测试也从每个原视频随机抽取多个 20 帧子样本；它与本项目 12 类、72 条完整视频测试的 RF 63.89% 不是同一评估单位，不能计算直接提升幅度。未见摄像头端到端 P50／P95 数据。

最适合借鉴的是指间距离、手指角度、手到身体的相对位置及帧间运动特征，而非直接复制其模型配置。按同一划分分别验证这些输入带来的增益；增强仅作用于训练源视频，多个派生样本始终保持源视频分组。

源码中的 [LSTM](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage/blob/main/code/model/models/rnn_classifiers.py)是单向结构，但[Demo API](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage/blob/main/webapp/backend/api/routes.py)仍接收完整视频，并没有暴露跨帧隐藏状态。[元数据配置](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage/blob/main/code/model/configs/features/metadata.yaml)及[特征实现](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage/blob/main/code/model/features/feature_processor.py)使用最终动作帧数、时长与相对位置；后向填补／线性插值同样可能需要未来帧。在线版本应采用时间戳、已观测时长、缺失 mask 及仅过去帧的处理，并重新训练和验证。

## 拟设计的最小结构

```text
带时间戳的 RGB 帧
  → 一次关键点提取与质量判断
  → 双手／上身坐标 + 缺失 mask
  → 分组的小型线性投影 → 共享因果 TCN → 语言专属任务头
                                      ├─ 完整动作分类头
                                      ├─ CTC 词序列头（有 gloss 标签时）
                                      └─ 活动／起止辅助头（有相应标注时）
  → 拒识与字幕提交策略 → partial / committed 结果
```

- 初版先使用现有 75 点 x/y 和 mask；`mediapipe225_rgb_v1` 的 z 固定为 0，不能当作真实三维信息。模型不再展平时间轴。骨向量、过去帧差分速度分别消融，避免把新增输入收益误认作网络收益。
- 可从 4 个小残差块、kernel=3、dilation=1/2/4/8 的配置起步，用验证集确定通道数与历史范围。这是训练配置起点，不是最优结论。采用每帧归一化或训练折冻结统计，避免依赖整段／未来帧。
- 保留真实时间间隔；动态丢帧时速度按有效时间差计算。禁止使用未来坐标插值、整片段长度归一化或事后均匀抽样来冒充在线输入。新增采样／提取策略须升级输入版本并重训评估。
- 模型编码器可以零前视，但字幕稳定化／结束确认可能需要等待。二者分别记录，不能用“因果”宣称字幕无需等待。
- 因果网络仍需实现每层卷积历史缓存。先验证同一已观测前缀的一次性推理与不同 chunk 划分的增量输出一致；不以不断重跑整个前缀代替缓存。导出的状态、时间戳、reset 和语言／模型版本绑定是新的流式契约，现有 clip API 不自动具备它。
- 起止检测与空闲识别不等于未知动作拒识。需要空闲、过渡、遮挡和词表外动作数据；阈值在验证集选择，报告拒识覆盖率与误接收率。
- 现有 Libras 孤立词数据可以训练分类头，不能凭目录名生成连续句子监督。中文 CE-CSL gloss 可支持 CTC 实验；葡语连续 CTC 需要相应序列标注。两侧不是直接共享最后一层，也不是简单翻译标签即可识别另一种手语。

## 验证后再决定部署模型

1. 冻结同一份训练／验证协议，训练 Pose-TCN；单向 GRU 作为一个小对照，RF 保留现有参照。先保持输入信息一致，再分别增加运动特征、空间图模块和 RGB 分支。
2. 当前历史测试集已被查看和反复分析，新实验登记可见性；用训练／验证折调参，扩大词表后建立新冻结测试集。核查跨条目全局签者与重复视频；新签者、新背景、不同设备另设评价。
3. 分类报告 accuracy、macro-F1、逐类召回、拒识覆盖率；连续词序列报告 corpus WER、整句完全匹配与关键词错误。RGB 教师／蒸馏只能使用训练部分，最终按学生自身表现评价。
4. 同设备 batch=1 实测采集、关键点、编码器、解码、排队和显示；报告 P50／P95、处理吞吐、帧龄、动作结束到稳定字幕。两路同时运行时重测。冷启动单列。
5. 用未人工分段的长流验证空闲误触发、重复输出、未知动作和字幕修订；比较流式与离线准确率。若准确率下降，显式评估有限前视的质量／延迟曲线。

ONNX CPU／CUDA／FP16 是部署实验维度，不是算法质量证明。必须验证实际节点执行位置与 PyTorch 输出一致性，不能因为模型小或参数少就跳过端到端测量。

## 接入当前核心框架的开发顺序

1. 保持当前 RF／TFNet 模型包作为可回退基线；先登记有时间戳、可变长度、仅依赖已到达帧的关键点输入版本，重新生成训练／验证数据，保留旧 30 帧样本用于旧协议对照。
2. 训练最小 Pose-TCN 分类器及一个单向 GRU／LSTM 对照，统一源视频和输入信息。验证达到可部署条件后再增加运动特征、空间图或 RGB 分支，不默认全部启用。
3. 增加流式状态管理、缓存及 partial／committed 结果；采集、特征提取和字幕提交分别计时，避免队列积压掩盖帧龄。长流验证需要空闲、过渡、未知动作与起止标注。
4. 在 CE-CSL 上单独训练并评估因果 CTC，比较离线、流式及有限前视的 WER／时延；Libras 先完成孤立词闭环，获取连续序列标注后再扩展。
5. 最后导出 ONNX 并验证 CPU／CUDA／FP16，测本机摄像头、两路并发和新签者。中葡语义与字幕映射继续使用审核词典，不把识别词序列自动视作自然语言翻译。
