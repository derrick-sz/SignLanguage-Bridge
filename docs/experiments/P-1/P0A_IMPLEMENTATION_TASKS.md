# P0-A 后端实施任务卡（可直接领取）

> 上游依据：`docs/experiments/P-1/P0A_BACKEND_GAP_ANALYSIS.md`（A0–A5 阶段与代码锚点，基线 HEAD b33b3b4）。
> 规范依据：`docs/架构开发文件/SignLanguage_Bridge_后端架构设计_V3.0.md`（下称 **V3.0**）、`docs/架构开发文件/SignLanguage_Bridge_API_Contract_V1.yaml`（下称 **API 契约**）、`docs/架构开发文件/SignLanguage_Bridge_开发实施清单_V3.0.md`（下称 **实施清单**）。
> 实测依据：`docs/experiments/P-1/P1_EXECUTION_REPORT.md` §7–§11（E04/E05/E06/ADR-P1-001~009）。
> 本文为**计划文档**，不含产品代码实现；不 git commit、不跑 GPU/模型。
> 符号与行号均经源码逐文件核实（见文末「引用索引」），禁止臆测。
> 领取纪律（每张卡通用）：TDD/测试先行；不迁移 UI；不动已验证推理路径（`slbridge/uni_sign.py`、`slbridge/slrt/` 推理内核零改动）；未运行字段保持 `null`/BLOCKED；每张卡独立 PR、独立验收。

---

## 0. 总览矩阵（按 A0→A5 排序）

| ID | 标题 | 阶段 | 前置依赖 | 工作量(人日) | 依赖 G2 结论 | 领取状态 |
|---|---|---|---|---|---|---|
| **T-A0-1** | 证据契约与预测记录 | A0 | 无 | 2.5–3 | 否 | ✅ 可直接开始 |
| **T-A1-1** | ReplayAdapter（`replay_jsonl_v1`） | A1 | T-A0-1 | 1–1.5 | 否 | ⚠️ 需先决条件 |
| **T-A2-1** | V1 契约 Pydantic schema 对齐 | A2 | T-A0-1 | 1–2 | 否 | ⚠️ 需先决条件 |
| **T-A2-2** | 异步 job 状态机与执行器 | A2 | T-A0-1、T-A2-1 | 2–3 | 否 | ⚠️ 需先决条件 |
| **T-A2-3** | `POST /v1/videos/infer` + `GET /v1/jobs/{job_id}` | A2 | T-A2-1、T-A2-2 | 1.5–2 | 否 | ⚠️ 需先决条件 |
| **T-A3-1** | 视频媒体元数据与 job 生命周期清理 | A3 | T-A2-2 | 1–1.5 | 否 | ⚠️ 需先决条件 |
| **T-A4-1** | `sign_system` 路由显式化（CSL/LIBRAS/UNKNOWN） | A4 | T-A2-1 | 1–1.5 | 否 | ⚠️ 需先决条件 |
| **T-A4-2** | 健康端点 + `ErrorResponse` 错误码对齐 | A4 | T-A2-1 | 1–1.5 | 否 | ⚠️ 需先决条件 |
| **T-A5-1** | FusionController 接口占位（等 G2） | A5 | 无 | 0.5–1 | 否（实现等 G2） | ✅ 可直接开始（可并行） |

依赖链：`T-A0-1 → T-A1-1 → T-A2-2 → T-A2-3`；`T-A2-1` 与 T-A0-1 并行推进后为 A2/A4 提供契约 shape；`T-A3-1` 挂在 T-A2-2 之后；`T-A4-1/T-A4-2` 依赖 T-A2-1；`T-A5-1` 全程可并行。

---

## 1. 任务卡

### T-A0-1 · 证据契约与预测记录（Phase A0 第一张卡）

- **目标**：落地 V3.0 §4.2 的 `EvidenceCandidate` 标准证据契约，打通 `InferenceResponse → EvidenceCandidate` 转换、GT 字段拒绝守卫、以及产品运行路径上的 JSONL 预测记录（含 per-run manifest），作为 P0-A 全链路的统一口径。
- **涉及文件**（均为**新建**，零改动现有文件）：
  - `slbridge/evidence.py`：`EvidenceCandidate` schema（dataclass 或 Pydantic，二选一并在 docstring 写明）、`from_slbridge_response(...)` 转换函数、GT 守卫（复刻 `experiments/p1/gt_guard.py` 规则，**不 import 实验子包**）。
  - `slbridge/prediction_log.py`（或并入 `slbridge/evidence.py`）：`PredictionRecorder`，JSONL 追加写 + `manifest.json` 生成。
  - `tests/test_evidence.py`、`tests/test_prediction_log.py`。
- **依赖**：无。
- **验收标准（可测试断言）**：
  1. 转换后 `candidate_id == InferenceResponse.request_id`、`utterance_id == InferenceResponse.segment_id`（字段映射见 §2 决策表，须在 docstring 写死）。
  2. `result.task_type == "text"` ⇒ `output.kind == "TEXT"` 且 `output.text` 非空；`result.task_type == "gloss_sequence"` ⇒ `output.kind == "GLOSS"` 且 `output.gloss_tokens` 命中词表（对齐 `slbridge/model_adapters.py:validate_adapter_result` L196–198、L233–236）。
  3. `calibrated_confidence` 恒为 `null`：Uni-Sign `TextPrediction(score=None)`（`slbridge/uni_sign.py:1007`）与 SLRT `uncalibrated_mean_ensemble_probability`（`slbridge/slrt/adapter.py:31`）均不得映射成概率（ADR-P1-006）。
  4. `timing_ms.preprocess/inference` 可空；`preprocess_version` 必填且来自 `InferenceResponse.preprocessing_version`。
  5. 携带 `reference/gt_*/wer/edit_similarity/...` 键的 payload 被 GT 守卫抛 `GroundTruthLeakError`（复刻 `experiments/p1/gt_guard.py:FORBIDDEN_GT_KEYS` L21–48 全集，负例测试）。
  6. `PredictionRecorder` 每行 JSON 可被 `experiments/p1/common.py:iter_jsonl` 解析；行含 `request_id/job_id/model_id/model_version/preprocessing_version/sign_system/task_type/text|gloss_tokens/timing_ms/error`；`manifest.json` 含 `git_sha/branch/dirty_tree`（对齐 `common.py:git_state` L149–173）与模型/预处理字段。
  7. 单测覆盖 V3.0 §13 第 1 条 contracts 反例：`None` 置信度、空字符串、空 Gloss、坏 JSON、迟到输出。
- **建议测试文件**：`tests/test_evidence.py`、`tests/test_prediction_log.py`（注入假 `InferenceResponse`，CPU-only，0 GPU）。
- **风险**：
  - 字段映射一旦写入 JSONL 即成为长期口径（风险 §5.1 上游已点名）：需 Lead 复核 §2 决策表，尤其 `window_id` 与 `time_span_ms` 单位换算。
  - `RecognitionResult.time_range` 为**秒级 float**（`slbridge/uni_sign.py:968` 用其算 `window_duration_seconds`），而 API 契约 `TimeSpan` 为 **integer 毫秒** ⇒ 转换需 `×1000` 并取整，单测锁定；禁止静默丢单位。
  - 若 `evidence.py` 直接 `import experiments.p1`，会把实验子包耦进产品运行路径（反模式）⇒ 守卫规则在 `slbridge/` 内复刻一份独立常量，禁止 import。
- **预估工作量**：2.5–3 人日（含字段映射决策复核）。
- **G2 依赖**：否。
- **状态**：✅ 可直接开始。

---

### T-A1-1 · ReplayAdapter（`replay_jsonl_v1`）

- **目标**：注册一个实现 `ModelAdapter` Protocol 的 `ReplayAdapter`，从 GT-free JSONL（`experiments/p1/e02/candidate_outputs.jsonl`）逐条回放 Uni-Sign/SLRT 历史预测，打通无 GPU 的"上传→识别→证据→记录"全链路（V3.0 §3 表 + §14 P0 的 ReplayAdapter 要求）。
- **涉及文件**：
  - 新建 `slbridge/replay_adapter.py`（实现 Protocol：`capabilities/runtime_info/recognizer/warmup/infer/close`，逐条回放）。
  - 改 1 行：`slbridge/model_adapters.py` 末尾追加 `register_model_adapter(AdapterDefinition("replay_jsonl_v1", _load_replay))`（属「加不改」，参照 L390–393/401 的 `uni_sign_pose_slt`/`slrt_online_csl` 注册行）。
  - 新建 `tests/test_replay_adapter.py`。
- **依赖**：T-A0-1（回放产物需能产出/校验 `EvidenceCandidate`）。
- **验收标准（可测试断言）**：
  1. 回放 N 条与源 JSONL 逐字一致（文本/gloss 与 `candidate_outputs.jsonl` 一致）。
  2. 经 `ModelManager.infer_file`（`slbridge/runtime.py:382`）走完整 negotiate→执行→`validate_adapter_result` 校验路径，不绕过协议。
  3. 回放 `capabilities` 声明 `sign_language='csl'`，与 `ModelRouter` 语言路由一致；`score_type` 诚实标注 uncalibrated。
  4. 全程 0 GPU；空 JSONL / 坏行 / 缺字段行均显式报错而非静默跳过。
- **建议测试文件**：`tests/test_replay_adapter.py`（用 `candidate_outputs.jsonl` 小切片 fixture）。
- **风险**：回放输入必须是 GT-free 的 `candidate_outputs.jsonl`（gap §5.6 边界：绝不读 `experiments/p1/references/gt_1086.csv`）；`hrnet-npz` 视频绑定约束（`slbridge/slrt/adapter.py:19-20`）不影响回放，但回放 SLRT 的 `time_range` 需按 T-A0-1 的单位换算口径处理。
- **预估工作量**：1–1.5 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A0-1）。

---

### T-A2-1 · V1 契约 Pydantic schema 对齐

- **目标**：在 `slbridge/api_schemas.py` 补齐 API 契约 V1 的 HTTP 响应模型，作为 A2/A4 端点的形状单一真源；与 T-A0-1 的 `EvidenceCandidate` 形状保持一致（防漂移）。
- **涉及文件**：
  - 改 `slbridge/api_schemas.py`：新增 `SignSystem`/`InferenceMode`/`ModelTask`/`TimeSpan`/`ModelInfo`/`HealthResponse`/`JobStatus`/`SubtitleEvent`/`ErrorResponse`（契约 V1 字段与 enum 逐一对齐，现有 `ErrorResponse` L18 与 `UnifiedRecognitionResponse` L171 保持兼容，新增而不替换）。
  - 新建 `tests/test_api_contract_v1.py`（契约测试，逐码断言）。
- **依赖**：T-A0-1（`EvidenceCandidate` 形状单一真源：`slbridge/evidence.py` 为唯一定义处，HTTP 若需暴露则 re-export/reuse，不重复定义）。
- **验收标准（可测试断言）**：
  1. `JobStatus.status` enum == 契约 `{QUEUED,PREPROCESSING,RUNNING,SUCCEEDED,FAILED,TIMED_OUT,CANCELLED}`（V3.0 §6 + 契约 `JobStatus`）。
  2. `SubtitleEvent` 必填 `event/schema_version/session_id/utterance_id/revision/time_span_ms/sign_system/status`，`status ∈ {PARTIAL,FINAL,UNCERTAIN}`，`event ∈ {subtitle.partial,subtitle.final,subtitle.uncertain,subtitle.correction}`（契约 `SubtitleEvent`）。
  3. `ErrorResponse.code` enum == 契约 8 值 `{INVALID_INPUT,MODEL_UNAVAILABLE,INVALID_POSE_SCHEMA,OUT_OF_VRAM,INFERENCE_TIMEOUT,JUDGE_UNAVAILABLE,LOW_EVIDENCE,INTERNAL_ERROR}`（契约 `ErrorResponse` + V3.0 §11）。
  4. `SignSystem` == `{CSL,LIBRAS,UNKNOWN}`、`InferenceMode` == `{parallel_fusion,adaptive_cascade,single_model}`、`ModelTask` == `{CSLR,CSLT}`（契约）。
  5. 契约测试：对每个 enum 逐码断言，缺失/多出即失败。
- **建议测试文件**：`tests/test_api_contract_v1.py`。
- **风险**：与现有 `api_schemas.py` 的 `ErrorResponse`/`ErrorDetail`（L13/L18）命名冲突需显式区分（如 `ContractErrorResponse` 或复用改造）；不得破坏现有 v1/v2/v3 响应模型（向后兼容）。
- **预估工作量**：1–2 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A0-1 的 `EvidenceCandidate` 单一真源决策）。

---

### T-A2-2 · 异步 job 状态机与执行器

- **目标**：新建 job 层，实现 V3.0 §6 的调度任务状态机（`QUEUED → PREPROCESSING → RUNNING → SUCCEEDED|FAILED|TIMED_OUT|CANCELLED`），复用现有 `ModelManager` 单容量租约与 deadline/cancellation，单线程推进 + 失败重试一次（ADR-P1-003 单卡串行；坑点 §7 OOM 类错误重试后判定）。
- **涉及文件**：
  - 新建 `slbridge/api_jobs.py`：`JobStatus` 状态机、`JobStore`（内存注册表，job_id 幂等）、`JobRunner`（复用 `ModelManager.infer_file` + `InferenceRequest` + `CancellationToken`/`deadline_monotonic`）。
  - 新建 `tests/test_jobs_api.py`。
- **依赖**：T-A0-1（预测落盘）、T-A2-1（`JobStatus`/`SubtitleEvent` schema）。
- **验收标准（可测试断言）**：
  1. 合法转移序列 `QUEUED→PREPROCESSING→RUNNING→SUCCEEDED`（或 `FAILED/TIMED_OUT/CANCELLED`）可观测；非法跳转被拒。
  2. 单容量串行：同一时刻至多一个 `RUNNING`（对齐 `ModelManager._lease` `slbridge/runtime.py:330-349` 的 busy 语义）。
  3. 失败重试一次后仍失败 ⇒ `FAILED` + 显式错误码；不静默重试无限次。
  4. 超时 ⇒ `TIMED_OUT`，`final_event` 为空、`error.code=INFERENCE_TIMEOUT`（对齐 `operation` 映射 `slbridge/api.py:403-404` 的 `inference_timeout` 语义）。
  5. 取消 ⇒ `CANCELLED`，不产出伪造结果（E05/ADR-P1-004：不伪造 PARTIAL）。
  6. `SUCCEEDED` 后写一行 `PredictionRecorder` JSONL（T-A0-1 产物）。
- **建议测试文件**：`tests/test_jobs_api.py`（用 `replay_jsonl_v1` backend 做 CPU-only 驱动）。
- **风险**：job 队列天然串行，不要在 P0-A 引入并发调度器（ADR-P1-003）；`Parallel/adaptive` mode 语义留到 G2 后，P0-A 只实现 `single_model` 语义。
- **预估工作量**：2–3 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A0-1、T-A2-1）。

---

### T-A2-3 · `POST /v1/videos/infer` + `GET /v1/jobs/{job_id}`

- **目标**：在 `api.py` 新增（不改现有端点）V1 契约的视频异步 job 端点：提交视频 → `202 + job_id`；查询 → `JobStatus + final_event(SubtitleEvent) + metrics`（V3.0 §10 + 契约 `POST /v1/videos/infer`、`GET /v1/jobs/{job_id}`）。
- **涉及文件**：
  - 改 `slbridge/api.py`：新增 `POST /v1/videos/infer`、`GET /v1/jobs/{job_id}`（复用 `_save_upload` L171–188 落盘与限额、`_SUFFIXES` L41 格式校验、`selected_manager` L387–396、`operation` L398–418 错误映射）。
  - 新建 `tests/test_v1_jobs_endpoints.py`。
- **依赖**：T-A2-1、T-A2-2。
- **验收标准（可测试断言）**：
  1. `POST /v1/videos/infer`（multipart `video` + `sign_system` + `mode`）返回 `202` + `job_id` + `status=QUEUED`。
  2. `GET /v1/jobs/{job_id}` 轮询到 `SUCCEEDED`，含 `final_event`（`SubtitleEvent` 契约形状，仅 `subtitle.final`/`subtitle.uncertain`，不伪造 `partial`）与 `metrics` 时延摘要。
  3. 未知 `job_id` → `404`；上传超限 → `413`；空文件/坏格式 → `400/422`。
  4. `mode=parallel_fusion|adaptive_cascade` 在 P0-A 返回明确「未启用」错误（诚实原则，V3.0 §6 效率模式等 G2 验证）。
  5. 幂等：同 `job_id` 重复查询结果一致（V3.0 §18 幂等键思想）。
  6. 现有 `/v3/recognitions`（`api.py:680-864`）行为零改动（回归测试）。
- **建议测试文件**：`tests/test_v1_jobs_endpoints.py`（`TestClient` + `replay_jsonl_v1`）。
- **风险**：异步 job 需要事件循环内安全的执行线程（复用现有 `run_in_threadpool` 模式）；`metrics` 只含实测墙钟分段（timing_ms），不得引用 E04 数字冒充 API 实测（gap §5.5）。
- **预估工作量**：1.5–2 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A2-1、T-A2-2）。

---

### T-A3-1 · 视频媒体元数据与 job 生命周期清理

- **目标**：为 job 化上传补充媒体信息采集（ffprobe 只读：FPS/宽高/编码/时长/帧数）写入 job 元数据与 `EvidenceCandidate.time_span_ms` 校验；并固化临时视频的生命周期（SUCCEEDED/FAILED 后删除），对齐 V3.0 §7.1「记录 FPS、PTS、尺寸、编码类型」。
- **涉及文件**：
  - 新建 `slbridge/media_probe.py`（ffprobe 只读封装）或并入 `slbridge/api_jobs.py`。
  - 改 `slbridge/api_jobs.py`（job 元数据 + 清理钩子）；`slbridge/api.py`（清理接线）。
  - 新建 `tests/test_media_probe.py`。
- **依赖**：T-A2-2（job 生命周期钩子）。
- **验收标准（可测试断言）**：
  1. ffprobe 元数据字段（fps/width/height/codec/duration/nb_frames）与实测一致；缺 ffprobe 时显式降级而非 500。
  2. 坏文件/非视频 → `INVALID_INPUT`（而非 500）。
  3. `EvidenceCandidate.time_span_ms` 与媒体时长一致（单位 ms）。
  4. job `SUCCEEDED/FAILED/TIMED_OUT/CANCELLED` 后临时视频文件删除（`path.unlink(missing_ok=True)`，参照 `api.py:831` 现有清理语义）。
  5. 允许格式对齐 `_SUFFIXES`（`.mp4/.avi/.mov/.mkv`，`.npy/.npz` 仍属旧端点专属）。
- **建议测试文件**：`tests/test_media_probe.py`。
- **风险**：`sources.py`/`continuous.py` 的实时栈不动（gap §2.1 纪律）；ffprobe 只读，不重编码、不修改 PTS。
- **预估工作量**：1–1.5 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A2-2）。

---

### T-A4-1 · `sign_system` 路由显式化（CSL/LIBRAS/UNKNOWN）

- **目标**：在 API 层加 `sign_system` 枚举校验（`CSL|LIBRAS|UNKNOWN`），LIBRAS 无权重时固定返回 `503 MODEL_UNAVAILABLE`（固定文案「no Libras-trained weights」），UNKNOWN 在 P0/P1 显式拒绝（V3.0 §8.3；ADR-P1-007 绝不注册 LIBRAS→CSL 权重路由）。
- **涉及文件**：
  - 改 `slbridge/api.py`（sign_system 校验挂接 `selected_manager`/`operation`）。
  - 改 `slbridge/api_schemas.py`（`SignSystem` 校验，复用 T-A2-1）。
  - 新建 `tests/test_sign_system_routing.py`。
- **依赖**：T-A2-1（`SignSystem`/`ErrorResponse`）。
- **验收标准（可测试断言）**：
  1. `sign_system=LIBRAS` → `503 MODEL_UNAVAILABLE`（不落到 CSL 权重）。
  2. `sign_system=UNKNOWN` → 显式拒绝（P0/P1 必须显式指定 CSL）。
  3. `sign_system=CSL` 正常路由到 `csl` 模型（对齐 `ModelRouter.manager_for` `slbridge/runtime.py:434-441`）。
  4. 负例测试：断言 `ModelRouter` 未注册 `libras→csl` 权重路由（ADR-P1-007）。
- **建议测试文件**：`tests/test_sign_system_routing.py`。
- **风险**：`ModelRouter` 本体不动（只加 API 层校验）；LIBRAS 能力在 P0-A 标 `UNTRAINED`。
- **预估工作量**：1–1.5 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A2-1）。

---

### T-A4-2 · 健康端点 + `ErrorResponse` 错误码对齐

- **目标**：新增轻量 `/health/live`（仅进程存活）与 `/health/ready`（依赖可用性）形状对齐；补齐 `ErrorResponse` 契约枚举映射（`INVALID_POSE_SCHEMA`/`OUT_OF_VRAM` 挂接，`LOW_EVIDENCE`/`JUDGE_UNAVAILABLE` 仅预留枚举位）（V3.0 §10 + §11 + 契约 `HealthResponse`/`ErrorResponse`）。
- **涉及文件**：
  - 改 `slbridge/api.py`：新增 `/health/live`、`/health/ready`（复用 `/v3/health` L525–535 的 capabilities/status 数据，改形状）；`operation` 错误映射（L398–418）增加契约 code 映射分支。
  - 改 `slbridge/api_schemas.py`（复用 T-A2-1 的 `HealthResponse`/`ErrorResponse`）。
  - 新建 `tests/test_health_and_errors.py`。
- **依赖**：T-A2-1。
- **验收标准（可测试断言）**：
  1. `GET /health/live` → `200 {"status":"alive"}`（无需加载模型，轻于现有 `/v1/health` L420–425）。
  2. `GET /health/ready` → `ready|not_ready` + `workers` 形状对齐契约 `HealthResponse`。
  3. `ErrorResponse.code` 枚举与契约 8 值一致；`negotiate` 失败路径挂接 `INVALID_POSE_SCHEMA`（对齐 `validate_adapter_result`/`ModelInputContract.validate_prepared` 的 pose/schema 校验），CUDA OOM 重试仍失败挂接 `OUT_OF_VRAM`。
  4. `JUDGE_UNAVAILABLE`/`LOW_EVIDENCE` 在 P0-A 仅定义枚举位、不产生（诚实原则，等 G2/融合期才有语义）。
  5. 现有错误码（`inference_timeout/runtime_busy/model_unavailable/...`）映射不变，回归测试绿。
- **建议测试文件**：`tests/test_health_and_errors.py`。
- **风险**：`OUT_OF_VRAM` 需可模拟（或注入 fake OOM）；不得在健康端点泄露堆栈/密钥（V3.0 §11，复用 `_redact`）。
- **预估工作量**：1–1.5 人日。
- **G2 依赖**：否。
- **状态**：⚠️ 需先决条件（T-A2-1）。

---

### T-A5-1 · FusionController 接口占位（等 G2，可选末位）

- **目标**：定义 `DecisionIntent` 枚举 + `FusionController` Protocol 签名，用单测锁定接口形状，**零实现、零接线**（V3.0 §3 + §5.1；ADR-P1-005：Jev/LLM 不进主链路，FusionController 只留接口）。
- **涉及文件**：
  - 新建 `slbridge/fusion.py`：`DecisionIntent`（`action ∈ {ACCEPT,FUSE,RETRY,UNCERTAIN}`、`origin`、`policy_version`、`reason_codes`）+ `FusionController` Protocol（输入：同一 utterance 的 `EvidenceCandidate` 集合；输出：`DecisionIntent` + 冲突明细）。
  - 新建 `tests/test_fusion_interface.py`（类型签名锁定）。
- **依赖**：无（可并行）。
- **验收标准（可测试断言）**：
  1. `DecisionIntent.action` enum == 契约 `{ACCEPT,FUSE,RETRY,UNCERTAIN}`；`origin` enum == `{rules,jev,timeout_fallback}`（契约 `DecisionIntent`）。
  2. `FusionController` Protocol 签名存在且类型化；单测对签名锁定（防 G2 后返工）。
  3. grep 断言：`slbridge/fusion.py` 未被 `slbridge/api.py`/`slbridge/runtime.py` import（零接线）。
- **建议测试文件**：`tests/test_fusion_interface.py`。
- **风险**：接口形状最终由 G2 结论修正（NEXT_PHASE_PLAN §5）；P0-A 只锁最小签名，不为未来猜太多字段。
- **预估工作量**：0.5–1 人日。
- **G2 依赖**：否（仅接口；实现与接线等 G2 结论）。
- **状态**：✅ 可直接开始（可并行）。

---

## 2. Phase A0 第一张卡（T-A0-1）的最小可合并变更范围

> 目标：以最小 diff 交付「证据契约 + 预测记录」，可独立合并、独立验收、可回滚。

### 2.1 新增文件（精确清单）

| 文件 | 内容（计划层面） | 是否改动现有文件 |
|---|---|---|
| `slbridge/evidence.py` | `EvidenceCandidate` schema（字段对齐 API 契约 `EvidenceCandidate` required 12 字段）；`from_slbridge_response(InferenceResponse) -> EvidenceCandidate`；`assert_gt_clean`（复刻 `gt_guard` 规则常量） | 否（新建） |
| `slbridge/prediction_log.py` | `PredictionRecorder`：JSONL 追加写（`sort_keys` 稳定）+ `manifest.json`（git/模型/预处理字段）；可并回 `evidence.py` | 否（新建） |
| `tests/test_evidence.py` | 转换正例 + 契约反例（null 置信度/空 gloss/空文本/坏 JSON/迟到输出）+ GT 负例（全红→全绿） | 否（新建） |
| `tests/test_prediction_log.py` | JSONL 可被 `common.iter_jsonl` 解析 + manifest 字段齐全 | 否（新建） |

**明确不改动**：`slbridge/` 与 `experiments/p1/` 现有任何文件（含 `model_adapters.py`/`api.py`/`api_schemas.py`/`gt_guard.py`）；不 import `experiments.p1`（守卫规则在 `slbridge/` 内复刻一份常量）。

### 2.2 字段映射决策（一次性写死，写入 `evidence.py` docstring + 本小节）

| 契约字段（V3.0 §4.1/§4.2 + 契约 EvidenceCandidate） | 来源（`slbridge` 真实符号） | 决策 |
|---|---|---|
| `candidate_id` | `InferenceResponse.request_id`（`runtime_contracts.py:96`，= `RecognitionResult.sample_id`，经 `validate_adapter_result` L192 校验一致） | `candidate_id = request_id` |
| `session_id` | `InferenceResponse.session_id`（`runtime_contracts.py:102`） | 直通；缺省时用 `job_id` 或 `"offline"` 占位（需 Lead 定） |
| `utterance_id` | `InferenceResponse.segment_id`（`runtime_contracts.py:103`） | `utterance_id = segment_id`（P0-A 整句/分段语义最接近） |
| `window_id` | 无对应字段（V3.0 §4.1：切片 ID 绑定 start/end） | P0-A file 路径取整段窗口，`window_id = utterance_id + "-w1"`（派生规则写死，需 Lead 复核） |
| `time_span_ms` | `RecognitionResult.time_range`（`contracts.py:389`，**秒级 float**） | `×1000` 取整为 ms（`TimeSpan` integer）；缺省/无 time_range 时置 `{start:0,end:0}` 或 null 并记录（需 Lead 定） |
| `sign_system` | `RecognitionResult.sign_language`（`"csl"`） | 规范化 `"csl"→"CSL"`（契约 enum 大写） |
| `model.id/revision/task/preprocess_version` | `InferenceResponse.model_id/model_version/preprocessing_version` + `result.task_type` | 直通；`task_type text→CSLT`、`gloss_sequence→CSLR`（契约 `ModelTask` 大写） |
| `output.kind/text/gloss_tokens` | `TextPrediction` / `GlossSequencePrediction`（`contracts.py:363/341`） | `text→TEXT`、`gloss_sequence→GLOSS`；强类型不混填（V3.0 §4.2） |
| `quality.calibrated_confidence` | Uni-Sign `score=None`、SLRT `uncalibrated_mean_ensemble_probability` | 恒 `null`（ADR-P1-006） |
| `timing_ms` | `InferenceResponse.timing_ms`（`runtime_contracts.py:100`） | 直通，可空 |

### 2.3 验收口径（T-A0-1 单独可跑）

- `pytest tests/test_evidence.py tests/test_prediction_log.py` 全绿；GT 负例先红后绿（TDD）。
- 用注入的假 `InferenceResponse`（text 与 gloss 各一）产出两行 JSONL；`manifest.json` 完整。
- 不触碰任何现有测试（回归不破坏）。

### 2.4 边界

- 不实现 `ExperimentReplay`（1086 Excel importer，属 G2 轨道）；只保证 JSONL 记录形状与其兼容（记录层存原始文本，`character_separated` 等后处理由评测器做，对齐 `experiments/p1/eval/metrics.py:METRIC_DEFINITION_VERSION="p1-v1"`）。
- 记录层预留 `policy_version`/`prompt_version` 可空字段（gap §5.2），不实现消费。
- UI/桌面栈（`session.py`/`desktop*.py`/`live.py`）零接触。

---

## 3. 引用索引（经源码核实）

### 3.1 V3.0 章节

| 主题 | 章节 |
|---|---|
| 模块表（ModelAdapter/EvidenceNormalizer/FusionController/RuleJudge/JevJudge/LLMFusionEngine/InferenceScheduler/OutputStateManager/EvaluationLogger/ExperimentReplay） | §3 |
| 标识与时间（session/utterance/window/candidate/request/trace；start/end 媒体时间轴 vs monotonic 计时分离） | §4.1 |
| 标准模型证据（EvidenceCandidate 示例与规则；calibrated_confidence 可 null；preprocess_version 记录；GT 隔离） | §4.2 |
| 语义证据 SemanticFrame | §4.3 |
| 门控动作枚举与优先顺序 | §5.1、§5.2 |
| 两种推理模式（parallel_fusion/adaptive_cascade）与调度任务状态机 `QUEUED→…→SUCCEEDED|FAILED|TIMED_OUT|CANCELLED` | §6 |
| 输入（FPS/PTS/尺寸/编码）与输出（字幕状态机 PARTIAL/FINAL/UNCERTAIN） | §7.1、§7.2 |
| 时延定义 T_visible 与测量要求 | §7.3 |
| 跨手语路由（LIBRAS 无权重返回 MODEL_UNAVAILABLE；target_language 与 sign_system 独立） | §8.3、§8.5 |
| 推荐目录（仅建议命名，落地在 `slbridge/`） | §9 |
| 对外 API（`/v1/videos/infer`、`/v1/jobs/{job_id}`、`/health/live`、`/health/ready`、`/v1/models`、`/v1/experiments/replay`） | §10 |
| 观测/错误码（MODEL_UNAVAILABLE/OUT_OF_VRAM/INVALID_POSE_SCHEMA/INFERENCE_TIMEOUT/JUDGE_UNAVAILABLE/LOW_EVIDENCE） | §11 |
| 自动化测试闸门（contracts/fusion/replay/judge/scheduler/stream） | §13 |
| 实施顺序（P0 离线回放与融合核心） | §14 |
| 参考调度伪代码（职责分离，非可运行实现） | §19 |

### 3.2 `slbridge` 真实符号（文件:行）

| 符号 | 位置 |
|---|---|
| `InferenceRequest`（request_id/segment_id/session_id/source_epoch/deadline_monotonic/cancellation/expected_mode） | `slbridge/runtime_contracts.py:43-91` |
| `InferenceResponse`（request_id/model_generation/result/runtime/timing_ms/session_id/segment_id/model_id/model_version/preprocessing_version） | `slbridge/runtime_contracts.py:94-115` |
| `CancellationToken` / `InferenceCancelled` / `InferenceDeadlineExceeded` | `slbridge/runtime_contracts.py:31/23/27` |
| `RecognitionResult`（status/sample_id/model_id/sign_language/prediction/quality/reason/task_type/score_type/time_range） | `slbridge/contracts.py:378-392` |
| `TextPrediction`（text/language/score） / `GlossSequencePrediction`（tokens/decoder） / `GlossToken`（token_id/gloss/score/time_range） | `slbridge/contracts.py:363-375 / 341-360 / 320-338` |
| `ModelAdapter` Protocol / `ModelCapabilities`（negotiate）/ `ModelInputContract` / `validate_adapter_result` | `slbridge/model_adapters.py:176-186 / 120-173 / 69-117 / 189-236` |
| `AdapterDefinition` / `register_model_adapter` / `load_model_adapter` | `slbridge/model_adapters.py:300-304 / 310-318 / 343-361` |
| 注册行 `uni_sign_pose_slt` / `slrt_online_csl` | `slbridge/model_adapters.py:393 / 401` |
| `ModelManager`（`_lease`/`infer`/`infer_file`/`prepare`/`activate`/`discard`/`rollback`） | `slbridge/runtime.py:106-405`（租约 `330-349`、infer `356-380`、infer_file `382-383`） |
| `ModelRouter`（`register`/`manager_for`/`capabilities`） | `slbridge/runtime.py:407-469`（`419-432 / 434-441 / 457-459`） |
| `create_app` / `_save_upload` / `_SUFFIXES` / `_redact` / `_fingerprint` | `slbridge/api.py:272-320 / 171-188 / 41 / 202 / 263` |
| `operation`（错误映射） / `selected_manager` | `slbridge/api.py:398-418 / 387-396` |
| `/v3/recognitions`（同步识别） / `/v3/health` / `/v3/models` / `/v3/caption-sessions` | `slbridge/api.py:680-864 / 525 / 537 / 605-678` |
| `_CaptionSession` / `CaptionLedger` | `slbridge/api.py:243-260`；`slbridge/caption_events.py:14-168` |
| `ApiModel` / `ErrorResponse` / `UnifiedRecognitionResponse` | `slbridge/api_schemas.py:9 / 18 / 171` |
| `main`（serve 入口）/ `load_catalog` | `slbridge/api_cli.py:55-109 / 26` |
| `UniSignAdapter` / `UniSignBackend` / `_make_result`（`TextPrediction(score=None)`）/ `load_uni_sign_adapter` | `slbridge/uni_sign.py:1048 / 509 / 963 / 1007 / 1087` |
| `SLRTAdapter`（`REPRESENTATION='slrt_rgb_hrnet133_v1'`、score_type、hrnet-npz 约束）/ `load_slrt_adapter` | `slbridge/slrt/adapter.py:17-109 / 14 / 31 / 19-20 / 112` |
| `FramePacket` / `CapturedClip` / `ClipBuffer` / `RollingWindow` | `slbridge/sources.py:22/54/257`；`slbridge/continuous.py:63` |

### 3.3 实验侧（P-1）真实符号

| 符号 | 位置 |
|---|---|
| `FORBIDDEN_GT_KEYS` / `GroundTruthLeakError` / `assert_gt_clean` / `find_gt_keys` / `sanitize` | `experiments/p1/gt_guard.py:21-48 / 51 / 76 / 66 / 86` |
| `write_jsonl` / `append_jsonl` / `iter_jsonl` / `RunDirectory` / `git_state` / `environment_fingerprint` | `experiments/p1/common.py:110 / 94 / 82 / 203 / 149 / 176` |
| `METRIC_DEFINITION_VERSION = "p1-v1"` / `character_separated` | `experiments/p1/eval/metrics.py:27 / 30` |
| 回放输入（GT-free） | `experiments/p1/e02/candidate_outputs.jsonl`（另有 `slrt_gloss_predictions.jsonl`/`b_gloss_join_predictions.jsonl`） |

### 3.4 P-1 实测约束 → 硬约束（§7–§11）

| 结论 | 出处 | 约束 |
|---|---|---|
| 单卡串行默认；并行只适合离线批处理（−15% 墙钟、单条 1.18–1.81×、峰值 6.45GB） | E04 §7 / ADR-P1-003 | job 队列天然串行，不建并发 InferenceScheduler |
| Uni-Sign 无原生 Partial（first_stable_prefix mean 90%/median 100%，相邻改写率 67%）；SLRT partial 需 11 帧 lookahead | E05 §8 / ADR-P1-004 | 字幕层只做整句 FINAL/UNCERTAIN，不伪造 PARTIAL |
| SLRT 分数只排序（raw ECE 0.304）；Uni-Sign score=None | E03 / ADR-P1-006 | `calibrated_confidence` 一律 null |
| Rule-only 级联低风险低收益（+0.17 BLEU、错停优胜 5.2%、FUSE 触发 0） | E06 §9 / ADR-P1-005 | FusionController/RuleJudge 只留接口不进主链路 |
| GT 隔离 0 泄漏（gt_guard） | P1 报告 §2 | 证据层/记录层复刻同守卫，测试含负例 |
| 32GB Windows 并发假失败；SLRT CLI 每条约 3.9s 权重加载 | 坑点 §7 | job 单进程串行 + 失败重试一次再判 FAILED |

---

## 4. 依赖图与领取顺序

```text
T-A0-1 (证据+记录) ──► T-A1-1 (ReplayAdapter) ──► T-A2-2 (job 执行器) ──► T-A2-3 (V1 端点)
      │
      └──► T-A2-1 (契约 schema) ──► T-A2-2 / T-A4-1 / T-A4-2
                                        │
                                        └──► T-A3-1 (视频元数据, 挂 T-A2-2 之后)

T-A5-1 (Fusion 接口) ── 全程可并行（零实现零接线）
```

- **第一优先（无 GPU，可立即领取）**：T-A0-1、T-A5-1、T-A2-1（其中 T-A2-1 需等 T-A0-1 的 `EvidenceCandidate` 单一真源决策）。
- **第二优先（CPU-only 全链路）**：T-A1-1 → T-A2-2 → T-A2-3 → T-A3-1。
- **第三优先（契约收口）**：T-A4-1、T-A4-2。
- 所有卡均不依赖 G2 结论；G2 结论只影响 T-A5-1 的后续实现与接线，以及 A2 的 `parallel_fusion/adaptive_cascade` 是否在后续启用（P0-A 内明确「未启用」）。

---

## 实施进度（Lead 维护，2026-10-27）

| 任务卡 | 状态 | 提交 | 验收 |
|---|---|---|---|
| T-A0-1 证据契约与预测记录 | **DONE** | `d8df2c6` | 48 tests；GT 键集 24/24 与实验侧一致；全量 873 |
| T-A1-1 ReplayAdapter（`replay_jsonl_v1`） | **DONE** | `98bc6e4` | 16 tests；Pilot-50 A 候选 50/50 逐字一致；全量 889 |
| T-A2-1 V1 契约 schema 对齐 | **DONE** | `90f1239` | 26 tests（直读契约 YAML）；枚举逐值一致；全量 915 |
| T-A2-2 异步 job 状态机与执行器 | **DONE** | `1855dc5` | 14 tests（7 态矩阵/幂等/持久化/超时/取消/重试/串行/E2E）；`.conda-env` 930/10、`.venv-libras` 950/13 |
| T-A2-3 `POST /v1/videos/infer` + `GET /v1/jobs/{job_id}` | **DONE** | `01cca8f` | 11 tests；api 子集 63（52+11）全绿；api.py 纯新增 159 行；双环境 930/11、961/13 |
| T-A3-1 视频媒体元数据与 job 生命周期 | IN PROGRESS（task-18） | — | ffprobe 元数据/探针失败如实 null/TTL 清理与证据分离 |
| T-A4-1 `sign_system` 路由显式化 | PENDING | — | LIBRAS→503 MODEL_UNAVAILABLE |
| T-A4-2 健康端点 + 错误码对齐 | PENDING | — | 依赖 A2-1（已具备） |
| T-A5-1 FusionController 接口占位 | PENDING（等 G2 重启条件） | — | G2=FAIL，仅接口 |

回归基线：`.conda-env` repo root `pytest -q` = **930 passed / 11 skipped**；`.venv-libras` = **961 passed / 13 skipped**（两环境互补：前者无 pydantic/FastAPI，后者无 torch；v3 端点 API 测试在 .venv-libras 完整覆盖 52/52）。


