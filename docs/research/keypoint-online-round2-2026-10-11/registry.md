# 引文登记表：关键点在线路线第二轮

核查日期：2026-10-11，Asia/Shanghai。正文为 delta；9 项外部一手/官方来源，4 组直接读取的本地证据。排序与 draft.md 一致。来源评分只是研究者判断，不是可信度概率。脚本检查数字引文、URL 与重复/遗漏；本地证据及引文语义另由人工与独立评审核对。

| 编号 | 来源与 URL | 类型/日期 | 采纳范围与读取位置 | 评分 Aut/Rec/Rel/Dep |
|---|---|---|---|---|
| 1 | [MediaPipe Hand Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) | 官方；更新2026-08-17 | Python running modes/config/output：busy丢输入、双手配置、逐手world原点；不证明当前legacy图GPU能力 | 10/10/10/9 |
| 2 | [Stream Monitoring under the Time Warping Distance](https://www.cs.cmu.edu/~christos/PUBLICATIONS/spring-icde07.pdf) | 作者大学PDF；ICDE2007 | 全文§§3.1–3.3/Fig4：任意起点滚动DP与重叠最优提交；不迁移性能 | 10/5/10/10 |
| 3 | [Generalizing DTW to the multi-dimensional case requires an adaptive approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC5668684/) | 作者稿；2017卷期/2016在线 | 全文§§2.1/3/5/7：共享与独立路径无普遍优胜；未验证手语缺失mask | 10/6/10/10 |
| 4 | [TCN author source](https://github.com/locuslab/TCN/blob/8845f88f31def1e7ffccb8811ea966e9f58d9695/TCN/tcn.py) | 作者源码；2018-04-16文件提交 | TemporalBlock两卷积、dilation、forward无state；RF/缓存公式是结构推导 | 10/5/10/10 |
| 5 | [PyTorch GRU](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html) | 当前官方main文档 | h_0/h_n与bidirectional；不证明安装版或本项目支持 | 10/10/9/8 |
| 6 | [Towards Online Continuous Sign Language Recognition and Translation](https://aclanthology.org/2024.emnlp-main.619.pdf) | EMNLP2024论文全文 | §§3–4/A.1–A.3/Algorithm2：RGB+热图、CTC/gloss前提、投票机制；V100/理论延迟不能迁移 | 10/7/10/10 |
| 7 | [Author prediction_slide.py](https://raw.githubusercontent.com/FangyunWei/SLRT/main/Online/CSLR/prediction_slide.py) | 作者源码；抓取2026-10-11，main可变 | sliding_windows/evaluation_slide：整段张量生成窗、批量评估、端点padding；不否定论文可在线机制 | 10/8/10/10 |
| 8 | [1€ Filter author page](https://gery.casiez.net/1euro/) | 作者说明；CHI2012论文入口 | tuning/实现入口：抖动和滞后权衡；未获取全文或做手语实验 | 10/5/8/8 |
| 9 | [Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping](https://pmc.ncbi.nlm.nih.gov/articles/PMC6816304/) | KDD2012作者稿全文 | §§1.2/4：下界级联剪枝前提；自定义mask多维距离的适用证明未建立 | 10/6/8/10 |

## 本地证据（不混入 web source pool）

- [代码1]：主工作区 de316f52bc618387e0536daeee80899b947fd5bd。[segmentation.py:436](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:436)、[media.py:134](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:134)、[media.py:228](/E:/githubproject/SignLanguage-Bridge/slbridge/media.py:228)、[live.py:59](/E:/githubproject/SignLanguage-Bridge/slbridge/live.py:59)。同一次process的缓存机会、质量差异、完整clip采样/reset。
- [代码2]：[sources.py:157](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:157)、[sources.py:227](/E:/githubproject/SignLanguage-Bridge/slbridge/sources.py:227)、[caption_events.py:80](/E:/githubproject/SignLanguage-Bridge/slbridge/caption_events.py:80)、[runtime.py:359](/E:/githubproject/SignLanguage-Bridge/slbridge/runtime.py:359)、[session.py:369](/E:/githubproject/SignLanguage-Bridge/slbridge/session.py:369)、[segmentation.py:22](/E:/githubproject/SignLanguage-Bridge/slbridge/segmentation.py:22)。时间域、整文件预检、事件和计时边界。
- [数据1]：[新增特征审计:18](/E:/githubproject/SignLanguage-Bridge/docs/Libras新增特征数据_算法策略与执行路线_2026-10-10.md:18)。既有审计及JSON产物只读，未重新扫描原数据；丢点/重复/签者身份缺口，不作为本轮训练结果。
- [模型1]：[temporal.py:141](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/temporal.py:141)、[libras_temporal_data.py:31](/C:/Users/ADMIN/.codex/worktrees/bilingual-tcn-training/SignLanguage-Bridge/slbridge/libras_temporal_data.py:31)。训练工作区只读，完整有效帧平均、321不同输入协议、未实现状态feed。

## 机器读取编号索引

以下9项与上表逐项对应，供 run-summary helper 计数，不增加来源。

[1] MediaPipe Hand Landmarker Python
[2] SPRING ICDE2007
[3] Multivariate DTW DMKD2017
[4] TCN author source
[5] PyTorch GRU official documentation
[6] Towards Online CSLR/SLT EMNLP2024
[7] SLRT prediction_slide author source
[8] One Euro Filter author page
[9] UCR Suite KDD2012 author manuscript

## 未采纳与不确定项

- VISAPP2022 skeleton monotonic attention 原文多次403，未用其细节/数字。
- ICCV2025 closer-look PDF403，只发现元数据及作者仓库，未用其机制/成绩。
- 姿态适用性论文只读摘要，不支撑细分失效数字。
- 没有把二手博客、搜索摘要、离线数据集精度、V100速度当作本项目结果。
- 实验次序、缺失惩罚、时长候选、确认规则、特征分组均标为拟议设计/数学推导；并非以上论文证明的手语工程方案。
