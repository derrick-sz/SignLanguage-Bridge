---
task_id: d
role: Lead technical-doc investigator and synthesis editor
objective: Verify coordinate and timing assumptions, then integrate algorithm and repository evidence
status: complete
confidence: high for inspected interfaces; medium for proposed adaptation
sources_found: 2 accepted external sources
acceptance_met: yes
---

# D：坐标语义、异步帧与证据整合

检索/核查：2026-10-11，Asia/Shanghai。没有新摄像头、特征提取、训练或性能测试。

## Sources

- D1：MediaPipe Hand Landmarker Python，官方文档。https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python 。页面 last updated 2026-08-17 UTC；读取配置、运行模式及结果字段，Aut:10 Rec:10 Rel:10 Dep:9 = 9.75。
- D2：Casiez/Roussel/Vogel，1€ Filter 作者说明、原论文入口及实现。https://gery.casiez.net/1euro/ 。原论文 CHI 2012；本轮获取作者页，不将示例或交互追踪结果当作手语效果。Aut:10 Rec:5 Rel:8 Dep:8 = 7.75。
- 本地支持：task-c.md 精确链接对应主工作区 de316f52bc618387e0536daeee80899b947fd5bd。主代理另复核 segmentation.py:439–483、media.py:228–264、caption_events.py:80–126 及 sources.py 时间戳相关定义。

## Search log

主代理执行 5 个 web search query strings：

1. MediaPipe hand landmarker live stream busy frame dropped timestamp normalized world landmarks
2. Evaluating the Immediate Applicability of Pose Estimation for Sign Language Recognition
3. A Closer Look at Skeleton-based Continuous Sign Language Recognition ICCV 2025
4. "A Closer Look at Skeleton-based Continuous Sign Language Recognition" arxiv github
5. "1€ Filter" Casiez Roussel Vogel 2012 pdf

另外用 Context7 resolve_library_id + query_docs 核对官方 MediaPipe 文档与 Python detect_async/callback 时间戳接口。检索串为本轮操作语义记录，非逐字符日志复刻；计数按实际搜索 query 项计。没有把 Context7 调用计入 web search count。

## Findings (facts only)

- Hand Landmarker `num_hands` 默认 1；live stream 的 detect_async 立即返回，忙碌时忽略新输入，结果经 callback 返回输入时间戳。video/live 模式跟踪可避免每帧重复 palm detection。[D1]
- normalized x/y 是图像尺度；world landmarks 的米制原点是每只手的几何中心。handedness category 的 score 是类别分数，文档未给每个手点的置信度。[D1]
- 1€ Filter 根据速度调整低通，较低最小截止频率可减小慢动抖动，同时增加滞后；作者提供 Python 等实现。[D2]
- 本地主代理复核：观察器对出现的手要求 21 点全部有效，归一化则允许逐点缺失并以任一点有效判断该手 observed；两者没有同义质量定义。

## Analysis (proposals)

- 手局部形状与身体相对轨迹分开；不能拿两只手各自为原点的 world 坐标直接相减声称测得双手全局物理距离。用共同图像/身体框架表达双手关系，并对不同坐标系统另设版本。
- 回调完成时刻不能替代观测帧时间来计算速度；忙碌丢帧后实际 Δt、缺口与质量必须保留。API 是否异步与实际处理帧率、语义延迟是不同指标。
- 1€ Filter 只作为有/无因果滤波的消融；不能据通用交互结果认定能提高手语识别。保留原坐标及滤波版本，测附加滞后与细动作损失。
- DTW、TCN、GRU 均须用同提取协议、数据切分、拒识预算与提交政策比较。已经验证的 CTC CUDA provider 不能为新关键点图或小模型的 GPU 吞吐作证。

## Support paraphrases

- D1 Run the task：busy 时输入被忽略，callback 返回 input timestamp；Output：world 原点逐手，handedness score 属于分类。
- D2 Tuning：慢动降 mincutoff 减抖但增 lag，快速运动调 beta 减 lag。

## Conflicts / unresolved issues

- 现代 Hand Landmarker Tasks 文档与当前项目 legacy Holistic 是不同接口；这里只用来核对拟议新分支，不声称当前图已经具备这些 API/设备特性。
- 新 ICCV 2025 骨架论文主 PDF 返回 403，虽然找到官方作者仓库和出版社元数据，未读全文，不采纳其机制/准确率；姿态适用性论文仅读摘要，未采纳具体失效数字。

## Leads / gaps

- https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Min_A_Closer_Look_at_Skeleton-based_Continuous_Sign_Language_Recognition_ICCVW_2025_paper.pdf ：访问失败；可在后续空间模型升级时再读。
- 新协议的双手追踪一致性、脸/手 RGB 必要性、最优过滤参数、CPU/GPU batch=1 性能仍须实测。本轮不提供新性能数字。
