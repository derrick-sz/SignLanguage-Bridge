# SignLanguage-Bridge 后端开发实施清单 V3.0

## 基础要求

- [ ] 冻结 `contracts/v1`，所有模块只通过类型化契约交换预测与证据
- [ ] `sample_id / session_id / utterance_id / window_id / candidate_id` 贯穿全链路
- [ ] GT 只由 evaluation 模块读取，不进入 Judge/Fusion/LLM 的实际输入
- [ ] 记录 checkpoint、预处理版本、prompt/策略版本，预测可复现
- [ ] 能区分 model WER 和最终翻译指标，不能拿不同任务的数字直接比较
- [ ] 保留原有项目 Demo，先加 `backend/`，不强制替换当前前端

## 里程碑 P0：离线回放与融合（先完成）

- [ ] `Pydantic`：EvidenceCandidate / TimeSpan / SemanticFrame / DecisionIntent / OutputEvent
- [ ] 加载 `csldaily-video-comparison-1086.xlsx` 共 1086 条，含 GT 但隔离
- [ ] 构建 `ReplayAdapter`，逐条模拟 Uni-Sign Text 与 SLRT Gloss
- [ ] `RuleJudge`：关键事实风险分类、候选完整性与时间窗校验
- [ ] `FusionController`：槽位与来源、严重冲突明细
- [ ] `LLMFusionEngine`：只用候选证据，严格 JSON 输出且失败可回退
- [ ] `EvidenceValidator`：来源检查、禁止新增未经支持事实
- [ ] A-F 实验统一运行；C 与 E、E 与 F 成对比较
- [ ] 输出 per-sample JSONL + 汇总 CSV；可手工回查 75 条潜在补证样本
- [ ] 增加案例回归测试：儿子/女儿、赚钱不多/没赚钱、公共场所吸烟/烟花

**验收**：无 GPU 视觉推理情况下可复验离线融合流程、复现统计口径；缺失或错误的 Jev/LLM 输出不造成假成功。

## 里程碑 P1：真实模型 Worker + Jev + API

- [ ] 分别记录 Uni-Sign 与 SLRT 使用的权重与依赖锁文件
- [ ] `PoseSchema` 支持 distinct HRNet / Uni-Sign / MediaPipe225，断言维度与坐标系
- [ ] 真实视频 Worker 与 ReplayAdapter 输出同一 EvidenceCandidate Schema
- [ ] 推理调度器：队列、每 Worker 并发、GPU 预算、超时、错误码和 fallback
- [ ] `JevJudge` 仅接结构化文本/JSON；可开关/可 Mock
- [ ] HTTP 上传 Job、查询 Job、模型健康 API
- [ ] 测 G/H 组正确升级率、错误接受率、LLM 调用率、额外时延
- [ ] 不依赖 Jev/LLM 可用性即可给出单模型候选或不确定状态

**验收**：上传一个完整视频时能通过 API 返回带时间与来源的预测，模型失败时不会假造结果。

## 里程碑 P2：实时输入 + 字幕

- [ ] WebSocket 协议支持单调帧序号与输入时间戳
- [ ] 有界 ring buffer，积压背压，主动清理失效窗口
- [ ] 按 utterance 关联候选，拒绝跨句错误融合
- [ ] PARTIAL revision 递增，FINAL 不被迟到推理覆盖
- [ ] 断流 / 重连 / 超时 / GPU OOM 端到端演练
- [ ] 采集实际 first-partial、first-stable、P50/P95、RTF、peak VRAM

**验收**：连续两句输入时字幕不串句、不回退最终结果、无死循环和无限缓存增长。

## 里程碑 P3：动态早退 + Libras

- [ ] 基于开发集校准 ACCEPT 与升级阈值；不使用线上不可见的 GT 指标
- [ ] Adaptive Cascade + 有比例的 full-run shadow audit
- [ ] 统计错停较优模型比例、GPU 时间与整体准确率权衡
- [ ] Libras Worker 必须是相应语料验证过的权重，不能复用未适配 CSL checkpoint
- [ ] 手动语言路由先上线，再以 CSL/Libras/Unknown 独立识别器替代
- [ ] 跨签手/跨设备/跨语言盲测 + 缺失非手部信息的风险报告

**验收**：既报告效率收益，也报告是否损失关键语义；对 Libras 的实际质量有独立实验支撑。

## 原有功能兼容检查

- [ ] 原随机森林手势 Demo 可通过 `LegacyGestureAdapter` 继续运行
- [ ] 文本检索与播放预置骨架动画保留为独立 `LegacyAnimationRenderer`
- [ ] 旧 `.npy` 与新版 `semantic_id` 迁移通过明示映射完成，不依赖目录名猜测语义
- [ ] 不将预置骨架动画宣传为自由文本自动生成完整 CSL/Libras 手语

## 合并前检查（Definition of Done）

- [ ] 自动化测试绿灯，类型检查通过，所有接口变更同步版本化
- [ ] 用户输入、视频时间段、模型版本和翻译输出可按 trace 查询
- [ ] GT 不进入任何运行时推理组件
- [ ] 未校准置信度输出 `null`，拒绝展示未经证明的“准确率 95%”
- [ ] Jev、LLM、备用 Worker 全部故障时按显式错误或 UNCERTAIN 回退
- [ ] 原始视频默认不持久化，测试录像经授权且规定保留周期
- [ ] 能从模型/预处理/prompt 配置中复现实验设置
