# SOKE / CSL-Daily 句级资源数量核查

核查日期：2026-11-06。范围：官方页面、论文与发布索引的静态核查；没有运行模型或进行整库动作／语言验收。

## 先区分计数单位

| 单位 | 官方统计 | 含义与边界 |
|---|---:|---|
| 句级视频样本 | 20,654 | 原始 CSL-Daily 的样本总数；不是不同中文句子数量 |
| train / dev / test 样本 | 18,401 / 1,077 / 1,176 | 原始论文 Table 2 的 segments；SOKE 具体发布索引另行核对 |
| 各划分不同句子 | 6,598 / 797 / 798 | 原始论文 Table 2 的 unique sentences；不能未查跨划分文本交集就相加称全局去重数 |
| Gloss 词汇 | 2,000 | 原始论文的词汇量；不是 2,000 个完整句子，也不能据此推断 SOKE 公开独立词条资产数量 |
| 逐句三维动作与语言审核通过数 | 未公布／本轮未验收 | 下载链接、样本 ID、视频存在或单帧 SMPL-X 字段都不能充当质量通过证据 |

这些精确原始统计来自 [CSL-Daily 原始论文，Table 1–2](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf)。该论文 Table 3 的训练对数另写作 18,402，与 Table 2 不一致；本笔记以 Table 2 的划分统计为参照，具体资产数量以发布索引实查为准。

## SOKE 官方资料能证明什么

[SOKE 项目页](https://2000zrl.github.io/soke/)提供 CSL-Daily 三维数据入口；[官方仓库 README](https://github.com/2000ZRL/SOKE#data)分别指向原始视频、split files 和 SMPL-X poses，未列逐句审核通过清单或精确保留／剔除数量。

[SOKE 论文 v3](https://arxiv.org/pdf/2411.17799v3) §4 将 CSL-Daily 概括为约 20K samples；§3.1 和补充材料 A 描述 OSX 身体、WiLoR 手部及 MediaPipe 重投影优化。补充材料说明 CSL-Daily 没有三维真值，其拟合定量对照使用另外的 SGNify mocap 数据；§E 承认重建误差不可避免。因此，论文对拟合方法的评价不能换算为 CSL-Daily 全部句子质量验收通过。

## 本机与发布索引实查

本轮实际取得官方项目页链接的 [Dropbox CSL-Daily-Fittings 目录](https://www.dropbox.com/scl/fo/3lwcdq60q8yg3foj6ulh0/AOw8pBv6LEk0JR0KvgVR-m4?rlkey=9ewwux5vnslljv88hbb5ibhds&dl=0)中的 `csl_clean.train`、`csl_clean.val`、`csl_clean.test`；train 沿用同日已下载文件，val/test 本轮下载，并非从 README 另列的 Google Drive split files 取得。方法为 gzip 解压后的 pickle opcode 静态解释；`GLOBAL`／`REDUCE` 仅保存为惰性数据，没有导入或调用，嵌套 torch storage bytes 也未执行。机器可读结果及每个索引的大小、SHA256 见 counts.json（本地产物：`artifacts/animation-upgrade/soke-sentence-count/counts.json`）。

| 划分 | 唯一样本 ID | 精确 `text` 去重 | 精确 Gloss 序列去重 | 本地同名视频 |
|---|---:|---:|---:|---:|
| train | 18,401 | 6,578 | 6,587 | 17,438 |
| val | 1,077 | 797 | 797 | 1,004 |
| test | 1,176 | 798 | 798 | 1,086 |
| 全局去重／匹配合计 | **20,654** | **7,376** | **7,386** | **19,528** |

三个划分之间样本 ID 交集均为 0；全局 `text` 去重直接对全部记录的原始字符串取集合，未做标点、空格等规范化，不能将各划分去重数相加。`val` 是 SOKE 发布文件名，对应验证划分。

**本问题若问发布资源有多少语句，可回答：20,654 条句级样本，覆盖 7,376 种精确中文文本；本机原视频按文件名可匹配 19,528 条。** 同一句话可以有不同签者／不同动作实例；7,376 是发布 clean 索引的文本统计，不是经语言学归并的句型数。训练文本去重为 6,578，与原始论文的 6,598 不同，保留差异，不将原始论文数字替代本次发布索引结果。7,386 是完整 Gloss 序列种数，也不是独立 Gloss 词条数。

索引声明帧数合计 2,467,508，只是 metadata。按名称匹配视频也没有核对视频内容哈希或三维文件逐项对应。项目页链接的 [CSL-Daily-Fittings 发布目录](https://www.dropbox.com/scl/fo/3lwcdq60q8yg3foj6ulh0/AOw8pBv6LEk0JR0KvgVR-m4?dl=0&e=1&rlkey=9ewwux5vnslljv88hbb5ibhds&st=bwkaqujt)未在本轮完成整包存在性／完整性核验；单个三维帧样本存在 SMPL-X 字段不能外推整库。本轮逐句可播放、动作质量及词义验收均为 **NOT_RUN**，没有可报告的“质量审核通过总数”。

## 使用范围

面向用户分别报告发布索引、本地视频匹配与动作审核数量。不能写成“两万种句子均可直接播放”或“两万条已审核标准 CSL 动画”。原始 [USTC 官方 CSL-Daily 页面](https://ustc-slr.github.io/datasets/2021_csl_daily/)明确数据面向大学与研究机构的研究用途，并提供访问申请流程；发布数量不改变其使用条件。
