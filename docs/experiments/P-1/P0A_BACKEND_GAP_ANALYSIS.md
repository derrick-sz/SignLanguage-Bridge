# P0-A 后端基础缺口分析与实施计划

> 任务：task-9（阶段 3 · P0-A 后端基础，双轨路线 B）。执行人：p0a-backend。
> 日期：2026-10-26。状态：**分析完成（本文为分析与计划，不含产品代码改动）**。
> 基线：分支 `codex/model-contract-auto-captions`（HEAD b33b3b4，工作树干净）。
> 规范依据：`docs/架构开发文件/SignLanguage_Bridge_后端架构设计_V3.0.md`（下称 **V3.0**）、`docs/架构开发文件/SignLanguage_Bridge_API_Contract_V1.yaml`（下称 **API 契约**）、`docs/架构开发文件/SignLanguage_Bridge_开发实施清单_V3.0.md`（下称 **实施清单**）。
> 实测依据：`docs/experiments/P-1/P1_EXECUTION_REPORT.md`（E00–E06/Gates/ADR）、`docs/experiments/P-1/NEXT_PHASE_PLAN.md` §3、`docs/agents/model-evaluation-pitfalls.md`。
> 范围纪律：P0-A 只做**不依赖 G2 结论**的部分——统一模型调用协议、API、预测记录（JSONL/候选上下文）、视频管理、错误处理；FusionController 只留接口；Jev/LLM 明确标"等 G2"。不迁移 UI、不微服务化、不动已验证推理路径。

---

## 0. 执行摘要

| 层次 | 现成度 | 一句话判断 |
|---|---:|---|
| ModelAdapter / 统一模型调用协议 | **~90% 可复用** | `slbridge/model_adapters.py` 的类型化契约 + Uni-Sign/SLRT 两个真实 adapter 已落地并被测试覆盖；差的只是 `InferenceResponse → EvidenceCandidate` 的证据映射层 |
| 运行时/调度（ModelManager 单容量） | **~85% 可复用** | E04 证实单卡串行即默认形态，现有单容量租约 + 世代号 + 事务切换正好匹配 ADR-P1-003，无需新建 InferenceScheduler |
| HTTP API | **~70% 可复用** | v3 端点（negotiate/统一识别/字幕会话/幂等/鉴权/错误码）齐全；缺 V1 契约的**异步 job 模式**（`/v1/videos/infer` → `/v1/jobs/{job_id}`）与路径命名对齐 |
| 错误处理 | **~80% 可复用** | 错误码体系完整且分层；需对齐 API 契约 `ErrorResponse` 枚举（`MODEL_UNAVAILABLE`/`INFERENCE_TIMEOUT` 等）并补 `LOW_EVIDENCE` 语义位 |
| 视频管理 | **~65% 可复用** | `sources.py`/`continuous.py`/`session.py` 的有界采集、滑窗、会话、候选上下文均现成；服务端"上传视频→job→媒体元数据记录"需新建；WS 流属 P2 不在 P0-A |
| 预测记录（JSONL/评测口径） | **~50%（实验侧现成）** | `experiments/p1/` 的 JSONL/manifest/gt_guard/canonical 评测器已冻结并回归，但**不在产品运行路径上**；需以它为统一口径接入 API 路径 |
| SignSystemRouter | **~60% 可复用** | `ModelRouter` 已有"按手语语言路由"语义（csl 一条真实路由）；缺 LIBRAS stub 显式 `MODEL_UNAVAILABLE` 与 V3.0 §8.3 的路由枚举对齐 |
| EvidenceNormalizer | **需新建（小）** | 现有 `RecognitionResult`/`InferenceResponse` 字段语义接近 V3.0 §4.2，但标识/时间/质量结构不同，需一个薄转换层 |
| FusionController / RuleJudge / LLM / Jev / SemanticFrame | **需新建（P0-A 只留接口）** | 产品代码无对应物；E06 有 Rule-only 离线原型可作 Rule 分支参考，但按 ADR-P1-005/008 不进主链路，等 G2 |
| 字幕状态（OutputStateManager） | **~55% 可复用** | `caption_events.py:CaptionLedger` 已实现有序、防迟到覆盖的事件账本；但无 PARTIAL/FINAL/UNCERTAIN 三态——且 E05 证明当前模型不支持真 PARTIAL，P0-A 不应伪造 |

**P0-A 真正的新建工作量**集中在四件事：① EvidenceCandidate 证据层 + JSONL 预测记录；② 异步 job API（对齐 V1 契约）；③ 上传视频的作业化视频管理（元数据 + 生命周期）；④ SignSystemRouter 显式化（LIBRAS stub）。其余均为对现有 `slbridge` 组件的**薄改造或直接复用**。

---

## 1. 分析方法与证据基线

- 所有"现有实现"论断均来自本轮对仓库源码的逐文件核读（文件与符号见 §2 各层，行号为当前 HEAD 实测）；未凭 README 或历史报告推测。
- 规范要求逐条对应 V3.0 章节号（形如 V3.0 §3/§4.2）；API 面对应 API 契约路径。
- 实测约束来自 P-1 执行报告 E04/E05/E06 与 ADR-P1-001~009；坑点约束来自 `docs/agents/model-evaluation-pitfalls.md`（下称坑点，按 §n 引用）。
- 本轮只读代码 + 写本文档；未修改任何 `slbridge/`、`scripts/` 代码，未跑 GPU。

## 2. 逐层差距分析

### 2.1 视频管理（VideoInput → VideoStreamManager）

**V3.0 要求**：
- §3 模块表：`VideoInput` 验证媒体、解码、帧率与时间戳、保留原始 PTS（阶段 P1）；`VideoStreamManager` ring buffer、滑窗、utterance ID、积压丢帧策略（阶段 P2）。
- §7.1：文件处理记录 FPS/PTS/尺寸/编码；摄像头进入有界 ring buffer；滑窗参数 `window_frames/hop_frames/max_wait_ms` 可配置并记录各段耗时。
- §4.1：`start_ms/end_ms` 使用同一视频流单调媒体时间轴，与 monotonic 处理计时分开。

**现有实现**（均已核读）：
- `slbridge/sources.py`：`FramePacket`（L22，含时间戳与单调性校验 `_check_next` L44）、`CapturedClip`（L54，有界）、`VideoSource`（L140，文件解码 + `_timestamp` L199）、`CameraSource`（L221）、`ClipBuffer`（L257，max_seconds/max_frames 有界，`finish()` 产出 clip）。
- `slbridge/continuous.py`：`ContinuousConfig`（L11，窗口/字节预算/过期丢弃）、`RollingWindow`（L63，`append/snapshot/buffered_bytes`）。坑点 §5 记录的帧数窗口边界 bug 已修复并有定向测试。
- `slbridge/session.py`：`RecognitionSession`（L29）统一管理来源开闭（`open_source/stop_source` L350-378）、采集 worker（L391 `_capture`）、epoch 失效（L109-121）、候选上下文（见 §2.7）。
- 测试：`tests/test_sources.py`、`test_continuous.py`、`test_continuous_session.py`、`test_continuous_gate.py`、`test_session.py`。

**差距**：
1. **缺（P0-A 范围内需新建）**：服务端"上传完整视频 → 异步 job → 媒体元数据（FPS/尺寸/编码/时长）记录"。现有 `api.py` 的 `_save_upload`（L171-188）只把上传落成临时文件并在请求结束后删除（L484-487、L831），**没有 job 化的视频生命周期与元数据记录**——这正是 V3.0 §10 `POST /v1/videos/infer` + `GET /v1/jobs/{job_id}` 的形态。
2. **可复用**：FramePacket/CapturedClip/RollingWindow 的时间轴纪律与 V3.0 §4.1 一致（媒体时间戳与处理计时分离，`InferenceRequest.deadline_monotonic` 是独立 monotonic 时钟，见 `runtime_contracts.py` L56/L87-91）。
3. **超出/不迁移**：`session.py` 的连续采集 + 自动分段（`segmentation.py`）属桌面实时栈，对应 V3.0 P2 范围；P0-A 不动它（纪律：不迁移 UI、不动已验证推理路径）。
4. **P-1 约束（E05/ADR-P1-004）**：Uni-Sign 是非因果整句模型（对称时卷积 ±6 帧 + 双向 mT5 encoder），`RollingWindow` 的滑窗重算是**伪 partial**（坑点 §6；E05 §8 实测 first_stable_prefix mean 90%/median 100%）。P0-A 的视频管理层**不得承诺流式字幕**，字幕/流式预期按"动作结束后整句识别"设计（见 §2.7）。

**最小实施建议**：新建薄 job 层（建议 `slbridge/api_jobs.py` 或并入 `api.py`）：复用 `_save_upload` 的落盘与限额、复用 `ModelManager.infer_file`，新增 job 状态机（枚举对齐 API 契约 `JobStatus`：QUEUED/PREPROCESSING/RUNNING/SUCCEEDED/FAILED/TIMED_OUT/CANCELLED）与媒体元数据采集（ffprobe 只读）。滑窗/实时栈不动。

### 2.2 SignSystemRouter（CSL/LIBRAS/UNKNOWN）

**V3.0 要求**：§3 模块表（阶段 P3，初期默认显式指定）；§8.3：P0/P1 由用户明确选择 CSL；Libras 未接训练权重时返回 `MODEL_UNAVAILABLE`，不得偷偷使用 CSL 权重；§8.4：自动识别属 P3。ADR-P1-007：Libras 需独立权重，当前标 `UNTRAINED`。

**现有实现**：
- `slbridge/runtime.py:ModelRouter`（L407-469）：按 `sign_language` 注册/隔离独立 `ModelManager`；`register` 校验路由语言与模型能力一致（L429-431）；`manager_for` 未注册语言抛 `ModelUnavailable`（L434-441）。
- `slbridge/api.py`：`create_app(routes=...)` 支持多语言路由（L304-311、L323-340）；`selected_manager`（L387-396）把 `ModelUnavailable` 映射为 404 `unknown_language`（当前 v1/v2）或经 `operation` 映射为 503 `model_unavailable`。
- 两个真实 adapter 均声明 `sign_language='csl'`（`uni_sign.py` `UniSignBackend` capabilities；`slrt/adapter.py` L29-31）。测试：`test_runtime.py`、`test_unified_api.py`、`test_desktop_language_switch.py`。

**差距**：
1. **部分具备（可复用）**：路由隔离、语言-模型一致性校验、未注册语言报错——语义上就是 §8.3 的前半句。
2. **需改造（小）**：对齐 V3.0/API 契约枚举：`sign_system` 用 `CSL|LIBRAS|UNKNOWN`（API 契约 `SignSystem` schema）；LIBRAS 应显式返回契约错误码 `MODEL_UNAVAILABLE`（当前是 404 unknown_language / 503 model_unavailable 二者之一，需统一到一个契约错误码）；`UNKNOWN` 在 P0-A 只是"未指定时拒绝"（V3.0 §3：初期默认显式指定），自动识别不做（NEXT_PHASE_PLAN §6 暂缓清单）。
3. **禁止项（纪律）**：绝不注册 LIBRAS→CSL 权重路由（ADR-P1-007）。

**最小实施建议**：在 API 层加 `sign_system` 枚举校验（Pydantic Literal），LIBRAS 未有权重时固定返回 503 `MODEL_UNAVAILABLE`；ModelRouter 本体不动。

### 2.3 ModelAdapter / Worker（含统一模型调用协议）

**V3.0 要求**：§3：`ModelAdapter` 输入 `ModelRequest` 输出 `EvidenceCandidate`，含预处理、权重、输出、健康检查、耗时（阶段 P0）；§14 P0：ModelAdapter 可先实现 ReplayAdapter（从 Excel/JSON 读已有预测），真实视觉 Worker 是之后步骤；§4.2：`task=CSLR 配 GLOSS、task=CSLT 配 TEXT` 强类型校验；`preprocess_version` 必须记录；未校准置信度必须 `null`。

**现有实现**（这是现成度最高的一层）：
- `slbridge/model_adapters.py`：
  - `ModelAdapter` Protocol（L176-186）：`capabilities` + `runtime_info` + `warmup` + `infer(source, request, input_kind) -> (RecognitionResult, timing)` + `close`。
  - `ModelCapabilities`（L120-173）：model_id/model_version/sign_language/task_type/inputs/vocabulary/runtime/devices/modes/score_type；`negotiate()`（L157-173）做请求-能力协商（model_id/sign_language/task_type/mode/representation/preprocessing_version 六重匹配）。
  - `ModelInputContract`（L69-117）：表示、形状、dtype、`preprocessing_version`（对完整冻结处理流程做 `canonical_hash`，L248-250）、`validate_prepared`（L92-117）。
  - `validate_adapter_result`（L189-236）：**task↔payload 强类型校验**（L196-198：classification/gloss_sequence/text 各配对应 prediction 类型）、rejected 不得携带已接受预测（L201-208）、Gloss payload 必须命中词表（L233-236）。
  - 输入表示注册表（L43-66）：hand63/pose225/mediapipe225/tfnet_rgb，Uni-Sign 与 SLRT 各自注册专属表示（`uni_sign.py` L61、`slrt/adapter.py` L14/L25）。
  - 后端注册表（L390-393、L401）：`random_forest_v1`/`onnx_classifier_v1`/`onnx_ctc`/`uni_sign_pose_slt`/`slrt_online_csl`。
- **两个真实视觉 adapter**：
  - Uni-Sign：`slbridge/uni_sign.py:UniSignAdapter`（L1048-1089，注册名 `uni_sign_pose_slt`）→ `UniSignBackend`（L509+）实现 file/captured_clip 两路；`_make_result`（L963-1008）产出 `TextPrediction(text, language='zh', score=None)`（L1007，**未校准分数为 null**，符合 V3.0 §4.2 规则）+ 拒绝路径（空文本/身体帧不足，L994-1004）+ timing（`infer` L1076-1081 返回 pose_ms/inference_ms/total_ms）。
  - SLRT：`slbridge/slrt/adapter.py:SLRTAdapter`（L17-109，注册名 `slrt_online_csl`）：capabilities 声明 task `gloss_sequence`、score_type `uncalibrated_mean_ensemble_probability`（L29-31）；权重与词表 SHA256 在构造期校验（L34-37）；`infer`（L69-103）产出 `GlossSequencePrediction`（含 token score 与 time_range）+ timing（preprocess/inference/total，L101-102）；hrnet-npz 只允许独立视频 CLI（L19-20、L77-78，共享 adapter 需实时 pose 前端）。
- `slbridge/runtime.py:ModelManager`（L106-405）：单容量租约 `_lease`（L330-349）、世代号 + `ModelGenerationConflict`（L180-183）、事务性切换 prepare/activate/discard（L202-306）、rollback、`infer`（L356-380）完成协商→执行→校验→timing→`InferenceResponse`。
- 测试：`tests/test_model_adapters.py`、`test_uni_sign_adapter.py`、`test_slrt_protocol.py`、`test_slrt_stream.py`、`test_slrt_resources_video.py`、`test_runtime.py`、`test_unified_api.py`、`test_sequence_api.py`。

**差距**：
1. **可复用（直接）**：类型化协议、六重协商、task 强类型、preprocess_version 哈希、拒绝语义、timing——即 V3.0 §4.2 大部分规则已在协议层强制。E01 已证明两条真实路径 50/50 逐字复现（P1 报告 §4），"不动已验证推理路径"意味着这层只加不改。
2. **需新建（小）**：`ReplayAdapter`（V3.0 §3 L79：从 JSONL/Excel 读已有预测的回放 adapter，注册为普通 backend）——P-1 已有全部素材（`experiments/p1/e02/candidate_outputs.jsonl` 等），实现约等于"读 JSONL → 逐条构造 `RecognitionResult`/证据对象"。这为 G2 融合实验与后端联调提供无 GPU 通道。
3. **需新建（证据形状）**：V3.0 §4.2 的 `EvidenceCandidate`（candidate_id/utterance_id/window_id/time_span_ms/model{...}/output{kind}/quality/timing）与现有 `InferenceResponse`（`runtime_contracts.py` L94-115：request_id/model_generation/result/runtime/timing）**字段不同构**：现有无 utterance/window 分层（用 session_id/segment_id/source_epoch 近似）、无 `output.kind` 枚举（用 task_type+typed payload 表达）、quality 是自由 dict（Uni-Sign 塞入大量稳定性指标）。→ 见 §2.4 EvidenceNormalizer。
4. **Worker 进程隔离**（V3.0 §1 ADR-1：Uni-Sign/SLRT 由可替换 Worker 运行）：当前是**进程内 adapter**，无独立 Worker 进程。按 E04/ADR-P1-003（单卡串行、并行只适合离线批处理），P0-A **不需要**引入进程隔离；若 P1 需要，以 `slbridge/cli.py`/`slrt/cli.py` 子进程模式演进（坑点 §7：并发重任务会产生假失败，并发上限按可用内存定）。

**最小实施建议**：协议层零改动；新增 ReplayAdapter 注册项 + 证据映射层（§2.4）。

### 2.4 EvidenceNormalizer

**V3.0 要求**：§3 模块表（阶段 P0）：原始结果 → 标准证据（字幕/Gloss、时间轴、模型版本、optional calibrated score）；§4.2 标准证据契约及规则（calibrated_confidence 可 null；preprocess_version 记录；GT 只在 evaluation.* 隔离区）。

**现有实现**：
- **无独立 EvidenceNormalizer**。最接近的物是：
  - `InferenceResponse`（`runtime_contracts.py` L94-115）：已携带 model_id/model_version/preprocessing_version/timing_ms/session_id/segment_id/source_epoch。
  - `validate_adapter_result`（`model_adapters.py` L189-236）：结果规范化前的强校验。
  - `caption_events.py:caption_from_result`（经 `captions.py` L87）：把 RecognitionResult 转成字幕条目（面向桌面字幕目录）。
- **实验侧**：`experiments/p1/e02/candidate_outputs.jsonl` + `common.py` 的写盘守卫（GT 隔离）事实上是离线版证据规范化（imported 快照 → 候选行），并已通过 3,258 行扫描 0 GT 泄漏（P1 报告 §2）。

**差距**：
1. **需新建（薄层）**：`EvidenceCandidate` Pydantic/dataclass schema（对齐 V3.0 §4.2 与 API 契约 `EvidenceCandidate` schema：required = schema_version/candidate_id/session_id/utterance_id/window_id/time_span_ms/sign_system/model/output；quality.calibrated_confidence 可 null；timing_ms.preprocess/inference 可 null）+ 一个 `InferenceResponse → EvidenceCandidate` 转换函数。**注意字段映射决策**：现有 `segment_id`↔`utterance_id`、`request_id`↔`candidate_id` 的对应关系需在 schema 文档中一次性写死（标识贯穿，实施清单"基础要求"第 2 条）。
2. **校准分数语义**：SLRT 的 `uncalibrated_mean_ensemble_probability`（E03：raw ECE 0.304，只能排序不能当概率）与 Uni-Sign `score=None`——P0-A 一律映射为 `calibrated_confidence: null`，E03 的 OOF Platt 校准器**不进 P0-A**（它依赖 GT 训练，属 evaluation 侧资产）。
3. **GT 隔离守卫**：实施清单/05 号文要求"后端 schema 校验应阻止包含 GT 派生字段的推理请求"（05 文 L16）。P-1 的 `experiments/p1/gt_guard.py` 已有现成守卫逻辑（出现 reference/gt_*/wer/edit_similarity 抛 `GroundTruthLeakError`）；P0-A 把**同规则**复刻进证据层校验（不直接 import 实验子包，避免产品依赖实验目录；或经薄 re-export）。

**最小实施建议**：新建 `slbridge/evidence.py`（schema + 转换 + GT 守卫），单测含"携带 GT 字段被拒绝"负例（05 文 L16 要求）。

### 2.5 FusionController

**V3.0 要求**：§3 模块表（阶段 P0）：paired evidence → conflicts（实体/动作/数量/否定/疑问/时间的冲突与来源映射）；§4.3 SemanticFrame；§19 伪代码（L433-460）。

**现状与差距**：
- **产品代码无对应物**（grep 全仓无 fusion/controller 类）。
- **按纪律只留接口**：NEXT_PHASE_PLAN §3（"FusionController 只留接口（是否启用等 G2 结论，见阶段 5）"）与 ADR-P1-005（Jev/LLM 不进主链路）。
- **可复用的参考实现**：E06 的规则级联与 FUSE 状态机算例（`experiments/p1/e06/`：`candidate_inputs.jsonl`、`router_ablation.jsonl`、`policy_candidate.yaml`；P1 报告 §9：级联 A2 调用率 38.1%、+0.17 BLEU、FUSE 真实触发 0）可作为 Rule 分支的**原型与测试素材**，但结论是"低风险、低收益"，不构成进主链路的理由。
- **P0-A 动作**：定义接口签名（输入：同一 utterance 的 `EvidenceCandidate` 集合；输出：`DecisionIntent` 枚举 ACCEPT/FUSE/RETRY/UNCERTAIN + 冲突明细），不写实现、不接线。放 `slbridge/fusion.py` 或暂缓到 G2 后——建议**只写接口文件 + 单测锁定签名**，避免 G2 结论出来后返工。

### 2.6 RuleJudge / JevJudge / LLMFusionEngine

**V3.0 要求**：§3（三者的阶段标记：RuleJudge P0、JevJudge P1、LLMFusionEngine P0）；§5 门控优先顺序（硬规则优先于 Jev；Jev 仅结构化文本；LLM 受证据约束）；§5.2/L164：GT 不进线上路由。

**现状与差距**：
- **RuleJudge**：产品代码无；E06 的离线规则（数字/否定/人物/疑问触发统计，P1 报告 §9 L153）+ `experiments/p1/e06/policy_candidate.yaml` 是可迁移原型。P0-A 范围内**不实现**（它服务融合路由，等 G2）；但注意 E06 的规则素材已在实验侧冻结，迁移成本可控。
- **JevJudge / LLMFusionEngine**：**明确标"等 G2"**（ADR-P1-005：无端点/密钥/版本，无法固定 prompt 与预算；NEXT_PHASE_PLAN §6：Jev 集成暂缓）。P0-A 不建任何 LLM/Jev 客户端。G2 的 LLM 端点前置条件见 NEXT_PHASE_PLAN §2.3（Codex CLI 或 OpenAI 兼容端点）。
- **候选上下文（LLM 输入的半成品）已现成**：`slbridge/candidate_context.py:CandidateLedger`（L14-87，有界历史、revision 快照）+ `session.py:export_candidate_context`（L332-336，JSON 导出）。坑点 §9：候选 `confirmed` 只是连续窗口文字一致，不代表确认语义；重叠候选不能串接成词序列。P0-A 把"候选上下文导出"作为 LLM 输入的**唯一既有通道**保留，不改语义。
- **DecisionIntent schema**：API 契约已有定义（`DecisionIntent` schema：action/origin/policy_version/reason_codes）；P0-A 可随 FusionController 接口一并锁定 schema，实现留空。

### 2.7 字幕与日志（OutputStateManager / SubtitleEvent / EvaluationLogger）

**V3.0 要求**：§3（OutputStateManager P2：部分结果修订、最终确认、拒判；EvaluationLogger P0：记录模型/版本/样本/决策/延迟/错误与审计）；§7.2 字幕状态机（COLLECTING/PARTIAL/FUSING/FINAL/UNCERTAIN，`session_id+utterance_id+revision` 排序去重，迟到输出不覆盖 FINAL）；§10 SubtitleEvent 事件 JSON；§11 观测（request/session/utterance/window/candidate ID、Worker revision、耗时、决策规则版本）。

**现有实现**：
- **字幕事件账本**：`slbridge/caption_events.py:CaptionLedger`（L14-168）：`reserve`（L48-78，segment 唯一性 + sequence 单调 + 容量上限）、`complete`（L80-100，四重绑定核验 session/request/source_epoch/model_generation，防串话）、`reject`（L102）、`_drain`（L116-126，**按序提交**：pending 队首完成才 commit，迟到/过期不覆盖已 committed 事件）、`undo/clear/cancel/invalidate_source`（L128-161）。API 侧由 `api.py` `_CaptionSession`（L242-260）+ `/v3/caption-sessions` 端点（L605-678）暴露，action 幂等（L644-678）。测试：`tests/test_caption_events.py`、`test_unified_api.py`。
- **会话内字幕与候选**：`session.py`（桌面栈，caption_catalog）+ `captions.py:CaptionCatalog/caption_from_result`（L20-30/L87）。
- **日志/观测**：`InferenceResponse.timing_ms`（runtime.py L372 计 total_ms）+ Uni-Sign quality 里的 per-stage 计时与 GPU peak（`uni_sign.py` L970-993）+ SLRT report timings（`slrt/adapter.py` L101-102）+ api.py request_id 贯穿 + 路径脱敏（`_redact` L202-229）。**但没有落盘的 EvaluationLogger**（JSONL per-run 记录）——预测记录见 §2.8。

**差距**：
1. **PARTIAL/FINAL/UNCERTAIN 三态**：现有账本状态是 `pending/completed/committed/rejected`（clip 会话语义），**不是** V3.0 §7.2 的流式三态。而 E05/ADR-P1-004 证明 Uni-Sign 无原生 Partial、SLRT partial 需 11 帧 lookahead——**P0-A 不应伪造三态**。正确姿势：P0-A 的 job 结果只产 `subtitle.final`（整句识别完成）与不确定时的 reason（`UNCERTAIN` 可作为 API 契约 `SubtitleEvent.status` 枚举预留），流式三态等 P2 在线模型（ADR-P1-004 结论）。`CaptionLedger` 的顺序提交与防迟到覆盖逻辑在 P2 仍可复用为事件存储内核。
2. **SubtitleEvent JSON 对齐**：现有 v3 响应体（`api_schemas.py` 的 UnifiedRecognitionResponse）与契约 `SubtitleEvent` 字段不同（schema_version/session_id/utterance_id/revision/time_span_ms/sign_system/status/text/evidence_ids/reason_codes）。P0-A 在 job 结果侧新增契约形状的 final 事件即可，不必改 v3 现有响应（向后兼容）。
3. **EvaluationLogger（P0-A 的"预测记录"核心缺口）**：见 §2.8。

### 2.8 评测与预测记录（EvaluationLogger / ExperimentReplay / canonical 口径）

**V3.0 要求**：§3（ExperimentReplay P0：1086 预测表 → evidence 文本证据回放）；§12 实验矩阵与指标分层；§11 日志字段；实施清单 P0"输出 per-sample JSONL + 汇总 CSV"。

**现有实现（实验侧，已冻结）**：
- `experiments/p1/common.py`：`write_jsonl` 默认 GT 守卫；run 目录 + manifest（git/硬件/权重/数据哈希/命令/状态）。
- `experiments/p1/eval/metrics.py`：**canonical 评测器**（`METRIC_DEFINITION_VERSION = "p1-v1"`，L27；sacreBLEU 13a + 上游 mT5 逐字加空格；chrF 原始串；token WER），对 1086 快照回归断言 24.8157/21.7974（`experiments/p1/tests/test_p1_infra.py` 12 passed）。
- `experiments/p1/gt_guard.py`：GT 泄漏守卫。
- `experiments/p1/e02/`、`e06/`：候选 JSONL 与规则回放素材。

**差距**：
1. **产品路径无预测记录**：API/会话推理的结果**不落 JSONL**，无 run manifest。P0-A 需新建一个最小 `PredictionRecorder`：每次 job/识别完成追加一行（request_id/job_id/model_id/model_version/preprocessing_version/sign_system/task_type/text 或 gloss_tokens/timing_ms/time_range/错误码），带 per-run manifest（模型权重 SHA、源码 commit、协议字段——坑点 §3 要求预测绑定权重/协议，防混成绩）。
2. **统一口径**：P1 报告 §13 下一步 2 与 NEXT_PHASE_PLAN 均要求 canonical 评测器成为统一口径。P0-A 的记录格式应保证：同一 JSONL 可直接喂 `experiments/p1/eval`（或其后继只读子包）重算 BLEU/WER——即字段命名/文本后处理与 p1-v1 兼容（`character_separated` 等由评测器做，记录层存原始文本）。
3. **ExperimentReplay（1086 Excel importer）**：属 P0 离线回放核心，但**依赖 G2 实验设计**（A–F 分组）；P0-A 只需保证 JSONL 记录形状与其兼容（上条），importer 本身归 G2 轨道。

### 2.9 API 层对照（API 契约 V1）

**契约要求 → 现有对应**：

| 契约（API_Contract_V1.yaml） | 现有（slbridge/api.py） | 判定 |
|---|---|---|
| `GET /health/live`（仅进程存活） | `/v1/health`（L420-425，返回模型状态——比 live 重） | 需新增轻量 live 端点（或保留 /v1/health 作 ready） |
| `GET /health/ready`（依赖可用性） | `/v3/health`（L525-535，capabilities+各路由 status） | 可复用，改路径/形状对齐 |
| `GET /v1/models`（ModelInfo：id/revision/task/sign_systems/status/preprocess_version） | `/v3/models`（L537-546，models+capabilities+status） | 可复用；capabilities 已含 task_type/sign_language/preprocess 版本（经 capabilities.inputs），需映射成契约 ModelInfo 形状 |
| `POST /v1/videos/infer`（202 + job_id；sign_system/mode） | `/v3/recognitions`（L680-864，**同步** multipart 识别） | **需新建异步 job 模式**（P0-A 核心） |
| `GET /v1/jobs/{job_id}`（JobStatus + final_event + metrics） | 无 | **需新建** |
| `WS /v1/stream/{session_id}` | 无（x-websocket-protocol 也注明需单独传输契约与性能测试） | **P2，不在 P0-A** |
| bearer 鉴权 | `_BoundedRequestMiddleware`（L64-108）+ `api_cli.py` token-env（L73-83，非回环强制 token） | 可复用 |
| 上传限额/413 | middleware content-length + `_save_upload`（L92-95、L171-188） | 可复用 |
| `ErrorResponse` 枚举（INVALID_INPUT/MODEL_UNAVAILABLE/INVALID_POSE_SCHEMA/OUT_OF_VRAM/INFERENCE_TIMEOUT/JUDGE_UNAVAILABLE/LOW_EVIDENCE/INTERNAL_ERROR） | `operation` 映射（L398-418）：inference_timeout/runtime_busy/model_unavailable/model_generation_conflict/request_cancelled/invalid_input/invalid_model/internal_error | **~70% 对齐**；缺 INVALID_POSE_SCHEMA/OUT_OF_VRAM/LOW_EVIDENCE/JUDGE_UNAVAILABLE（后两者等 G2/融合期才有语义） |
| EvidenceCandidate/DecisionIntent/SubtitleEvent schema | api_schemas.py 的响应模型 | 需按契约新增（随 §2.4/§2.5/§2.7） |

**现有 API 的额外强项（超出契约但保留）**：请求幂等（fingerprint + request_id 重放，L744-754）、字幕会话 action 幂等、路径脱敏、有界 multipart 解析、协作式取消 + deadline（L840-851）。这些是 V3.0 §13 测试闸门（stream/scheduler 项）的现成基础。

## 3. P-1 实测约束 → P0-A 设计决策标注

| 实测结论 | 出处 | 对 P0-A 的硬约束 |
|---|---|---|
| Uni-Sign 无原生 Partial（对称时卷积 ±6 帧 + 双向 mT5；first_stable_prefix mean 90%） | E05 §8、ADR-P1-004 | 字幕层只做整句 FINAL；`RollingWindow`/`TextStabilizer` 的 confirmed 是伪 partial（坑点 §6），不得对外当流式能力；WS/三态字幕推迟 P2 |
| 单卡串行默认；并行只适合离线批处理（总墙钟 −15% 但单条延迟 1.18–1.81×、峰值 6.45 GB） | E04 §7、ADR-P1-003 | P0-A 调度 = 现有 `ModelManager` 单容量租约，**不建**并发 InferenceScheduler；job 队列天然串行 |
| SLRT 分数只能排序（raw ECE 0.304）；Uni-Sign score=None | E03 §6、ADR-P1-006 | 证据层 `calibrated_confidence` 一律 null；不启用自动接受 |
| Rule-only 级联低风险低收益（+0.17 BLEU、错停优胜 5.2%） | E06 §9 | RuleJudge/FusionController 只留接口不进主链路；E06 素材作 G2 后参考实现 |
| 前缀/回放口径冻结：canonical p1-v1、上游半角参考、retrospective_seen 1086 | P1 报告 §1/§2、坑点 §11 | 预测记录 JSONL 与 canonical 评测器同口径；任何新数字不得与历史混比 |
| 32 GB Windows 并发假失败（OOM 类错误需重试后判定）；SLRT CLI 每条 3.9 s 权重加载 | 坑点 §7、P1 报告 §13 | job 执行器单进程串行 + 失败重试一次再判 FAILED；批量场景复用长驻进程而非每 job 起子进程 |
| GT 隔离（gt_guard 0 泄漏） | P1 报告 §2 | 证据层/记录层复刻同守卫；测试含负例 |
| hrnet-npz 是视频绑定后端，不进共享 adapter | `slrt/adapter.py` L19-20、坑点 §10 | P0-A 的 SLRT 接入走独立视频 CLI 路径（`slbridge/slrt/cli.py`），共享 adapter 保持实时 pose 前端约束 |

## 4. P0-A 实施计划（阶段拆分）

> 总原则（NEXT_PHASE_PLAN §3 + 05 号文）：TDD/测试先行；不迁移 UI；不动已验证推理路径（`uni_sign.py`/`slrt/` 推理内核零改动）；未运行字段保持 null/BLOCKED；每阶段可独立合并、独立验收。目录遵守 V3.0 §9 建议**但落地在现有 `slbridge/` 包内**（05 文 L7：不为统一目录贸然移动旧功能；V3.0 §9 L276 也注明路径仅是建议命名）。

### 阶段 A0 · 证据契约与预测记录（第一优先，无 GPU 依赖）

- **内容**：新建 `slbridge/evidence.py`：`EvidenceCandidate` schema（对齐 V3.0 §4.2 + API 契约 schema；`calibrated_confidence` 可 null；`output.kind=TEXT|GLOSS` 与 task 强绑定）+ `InferenceResponse → EvidenceCandidate` 转换 + GT 字段拒绝守卫（复刻 `gt_guard` 规则）。新建 `slbridge/prediction_log.py`（或并入 evidence）：JSONL 追加写 + per-run manifest（model_id/version/preprocess_version/权重 SHA/源码 commit/协议字段）。
- **测试先行**：`tests/test_evidence.py`——① 正常转换字段齐全；② null 置信度/空 gloss/空文本/坏 JSON/迟到输出的契约反例（V3.0 §13 contracts 项）；③ 携带 GT 派生字段（reference/gt_*/wer/...）被拒绝（05 文 L16）；④ JSONL 可被 canonical 评测器（`experiments/p1/eval` 或其只读拷贝）解析重算。
- **验收**：跑一次 CPU-only 的 ReplayAdapter（A1 产物）或注入的假 `InferenceResponse`，产出 JSONL；manifest 完整；GT 负例全红→全绿。
- **涉及文件**：新增 2 个模块 + 1 个测试文件；不改现有文件。

### 阶段 A1 · ReplayAdapter（打通无 GPU 全链路）

- **内容**：`slbridge/replay_adapter.py`：读 `experiments/p1/e02/candidate_outputs.jsonl`（或指定 JSONL）注册为 backend `replay_jsonl_v1` 的 ModelAdapter（实现同一 Protocol，逐条回放 uni/slrt 预测）。
- **测试**：`tests/test_replay_adapter.py`——回放 50 条与源 JSONL 逐字一致；经 `ModelManager.infer_file` 走完整协商/校验路径。
- **验收**：`python -m slbridge.api_cli --catalog <含 replay 项> ...` 起 API 后，上传/回放请求产出 EvidenceCandidate JSONL，全程 0 GPU。
- **涉及文件**：新增 adapter + 注册行（`model_adapters.py` 加一行注册，属"加不改"）+ 测试。

### 阶段 A2 · 异步 job API（对齐 V1 契约）

- **内容**：`api.py` 新增（不改现有端点）：`POST /v1/videos/infer`（multipart video + sign_system + mode=parallel_fusion|adaptive_cascade|single_model；P0-A 实际只实现 single_model 语义，其余 202 但返回明确"未启用"错误——诚实原则）→ 202 + job_id；`GET /v1/jobs/{job_id}` → JobStatus（QUEUED/PREPROCESSING/RUNNING/SUCCEEDED/FAILED/TIMED_OUT/CANCELLED + final SubtitleEvent + metrics 时延摘要）。执行器复用 `ModelManager.infer_file` + 现有 deadline/cancellation；job 状态机单线程推进（E04 串行约束）；失败重试一次（坑点 §7）。
- **测试先行**：`tests/test_jobs_api.py`——① 提交→轮询→SUCCEEDED 带时间/来源/时延；② 超时→TIMED_OUT 不假造结果；③ 模型缺失→FAILED+MODEL_UNAVAILABLE；④ 上传超限→413；⑤ job 文件生命周期（SUCCEEDED/FAILED 后临时视频删除）；⑥ 幂等（同 job_id 重复查询一致）。
- **验收**：对一段真实 mp4（CPU 环境可用 replay backend；GPU 环境用 uni_sign）端到端跑通；错误路径全部显式错误码。
- **涉及文件**：`api.py`（新增端点 + job 注册表）、`api_schemas.py`（JobStatus/VideoInferRequest）；现有 v1/v2/v3 端点零改动。

### 阶段 A3 · 视频管理与媒体元数据

- **内容**：job 化上传的媒体信息采集（ffprobe 只读：FPS/宽高/编码/时长/帧数），写入 job 元数据与 EvidenceCandidate 的 time_span 校验（V3.0 §7.1"记录 FPS、PTS、尺寸、编码类型"）；输入校验（允许格式 mp4/webm/avi/mov/mkv，沿用 `_SUFFIXES`）。
- **测试**：元数据字段齐全且与 ffprobe 一致；坏文件→INVALID_INPUT 而非 500；time_span_ms 与媒体时长一致。
- **涉及文件**：`api.py`/新 `api_jobs.py`；不改 `sources.py`。

### 阶段 A4 · SignSystemRouter 显式化 + 错误码对齐

- **内容**：`sign_system` 枚举校验（CSL/LIBRAS/UNKNOWN，API 契约 SignSystem）；LIBRAS→503 `MODEL_UNAVAILABLE`（固定文案"no Libras-trained weights"）；错误响应增加契约枚举映射（INVALID_INPUT/INVALID_POSE_SCHEMA/OUT_OF_VRAM——pose schema 错误在 negotiate 失败路径挂接；OUT_OF_VRAM 在 CUDA OOM 重试仍失败时挂接）；`/health/live` 轻量端点 + `/health/ready` 形状对齐。
- **测试**：LIBRAS 拒绝；UNKNOWN 显式拒绝（P0/P1 必须显式指定，V3.0 §8.3）；错误码枚举与契约一致（契约测试逐码断言）。
- **涉及文件**：`api.py`、`api_schemas.py`。

### 阶段 A5 · FusionController 接口占位（等 G2，可选末位）

- **内容**：`slbridge/fusion.py` 仅含 `DecisionIntent` 枚举 + `FusionController` Protocol 签名 + 单测锁定接口形状；**零实现零接线**。G2 结论（NEXT_PHASE_PLAN §5 决策规则）出来后决定实现与否。
- **测试**：接口存在性/类型签名锁定（防 G2 后返工签名）。
- **明确不做（标注"等 G2"）**：JevJudge、LLMFusionEngine、EvidenceValidator、SemanticFrame、A–F 实验编排、1086 Excel importer、自动路由识别、Early Exit、学习式调度（NEXT_PHASE_PLAN §6 暂缓清单）。

### 顺序与依赖

```text
A0 (证据+记录) ──► A1 (ReplayAdapter) ──► A2 (job API) ──► A3 (视频元数据) ──► A4 (路由/错误码) ──► A5 (Fusion 接口,可并行)
```
A0/A1 无 GPU；A2–A4 用 replay backend 做 CI，GPU 仅在验收 smoke（不动推理内核）。各阶段独立 PR，均遵守"先审计、后修改、再实测"（05 文 L11）。

## 5. 风险与边界

1. **字段映射决策风险**：`segment_id↔utterance_id`、`request_id↔candidate_id` 的映射一旦写入 JSONL 即成为长期口径；A0 需在模块 docstring 与本文档同步固化，并经 Lead 复核。
2. **双轨互斥**：G2 轨道可能引入 LLM 端点与 prompt 版本字段；P0-A 的记录层预留 `policy_version`/`prompt_version` 可空字段即可，不实现消费。
3. **不要为了"对齐 V3.0 §9 目录"迁移现有包**（05 文 L7 明令）；`slbridge/` 内新增模块即可，V3.0 目录是建议命名（V3.0 L276）。
4. **UI/桌面栈零接触**：`session.py`/`desktop*.py`/`live.py` 不在 P0-A diff 中出现。
5. **所有时延声明**：P0-A 产出的 metrics 只含实测墙钟分段（已有 timing_ms 通道）；不得引用 E04 数字冒充 API 实测（P1 报告 §13：E04 数字是基线不是 API 后的）。
6. **回放数据边界**：ReplayAdapter 输入必须用 GT-free 的 `candidate_outputs.jsonl`/`model_outputs.jsonl`（manifest 已扫描 0 泄漏），绝不读 `references/gt_1086.csv`（EVAL ONLY，NEXT_PHASE_PLAN §1 入库边界）。

---

### 附：本文引用的代码锚点索引

| 组件 | 文件:行 |
|---|---|
| ModelAdapter Protocol | slbridge/model_adapters.py:176-186 |
| ModelCapabilities.negotiate | slbridge/model_adapters.py:120-173 |
| validate_adapter_result（task 强类型） | slbridge/model_adapters.py:189-236 |
| adapter 注册（uni_sign/slrt） | slbridge/model_adapters.py:390-393,401 |
| ModelManager 租约/infer | slbridge/runtime.py:106,330-380 |
| ModelRouter | slbridge/runtime.py:407-469 |
| InferenceRequest/Response | slbridge/runtime_contracts.py:43-115 |
| create_app/v3 端点/字幕会话 | slbridge/api.py:272-287,680-864,605-678 |
| 错误映射 operation | slbridge/api.py:398-418 |
| CaptionLedger | slbridge/caption_events.py:14-168 |
| CandidateLedger/导出 | slbridge/candidate_context.py:14-87; slbridge/session.py:317-336 |
| FramePacket/ClipBuffer | slbridge/sources.py:22,54,140,221,257 |
| RollingWindow | slbridge/continuous.py:11,63 |
| UniSignAdapter/拒绝与分数 | slbridge/uni_sign.py:963-1008,1048-1089 |
| SLRTAdapter | slbridge/slrt/adapter.py:17-109 |
| api_cli serve 入口 | slbridge/api_cli.py:55-109 |
| canonical 评测器 p1-v1 | experiments/p1/eval/metrics.py:27 |
| caption_from_result | slbridge/captions.py:20-30,87 |
