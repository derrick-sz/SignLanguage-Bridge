# 第一轮交接

上一轮完整记录：[keypoint-streaming-feasibility](/E:/githubproject/SignLanguage-Bridge/docs/reports/keypoint-streaming-feasibility-2026-10-11.md)。另参照[模型选型笔记](/E:/githubproject/SignLanguage-Bridge/docs/reports/streaming-model-selection-research-2026-10-10.md)。本轮负责人已读取两者。

已知：每手21点，目标手语仍需自己的标签/示范；动态词需时间轨迹；现RF处理完成动作，活动观察与识别重复提取；225二维身体+双手与Libras318/321不同；小词表DTW基线、因果TCN/单向GRU候选；自然长流/新签者/OOV及字幕P95未验收。

前轮没有实现DTW、状态缓存或新的在线模型，没有摄像头新性能结果。本轮不重复宣称可行性，着重填补如何比较、如何提交以及数据能否训练相同输入的细节。
