# P-1 → 下一阶段计划（双轨推进：补齐 G2 实验 + P0-A 后端基础并行）

> 决策来源：用户 2026-10-26 对 `P1_EXECUTION_REPORT.md` 的批复（原话要点保留在本文，执行以本文为准）。
> 状态：**ACTIVE**（阶段 1 已执行；阶段 2/3 已启动；阶段 4/5 排队）。
> 原则：G2（双模型融合有效性）是唯一仍阻塞 V3.0 核心方案的 Gate；**不等所有实验完成才启动后端开发**。

## 0. 总路线

```text
阶段 1  冻结 P-1 成果（git 提交、口径与权重哈希固化）          [本次已执行]
阶段 2  补齐 G2：SLRT G2T/SLT → LLM 融合消融 → 人工语义评审     [已启动，见 §2]
阶段 3  P0-A 后端基础开发（不依赖 G2 结论的部分）              [已启动，见 §3]
阶段 4  性能优化对照（RTMPose 前端 / SLRT 逐窗预处理）          [排队，可与 3 并行]
阶段 5  G2 决策：融合进主链路 或 保持可插拔不进主链路            [等阶段 2 证据]
```

## 1. 阶段 1：实验资产保护（已执行）

- Git 提交拆分：①既有集成工作（SLRT/Uni-Sign 接入、评测脚本、报告 R1–R18）；②P-1 执行包（基础设施 + E00–E06 证据 + 报告 + 本计划）。
- 入库边界（详见 `experiments/p1/FREEZE.json`）：
  - **不入 git**：`Paper-algorithm/`（vendored 上游源码 1.48 GB、许可 UNKNOWN、含数据 Junction）；run 目录内 `artifact/`（媒体）、`predictions-raw/`、`inputs/`、`eval_records.jsonl`（含原始 GT，均可由 `model_outputs + references` 脚本重建）；`references/gt_1086.csv`（原始 GT，`build_references.py` 可确定性重建）。
  - 派生指标（WER/BLEU/编辑相似度等数值与逐样本标签）按既有报告惯例入库。
  - 仓库当前为 **private**；若未来转 public，须先清除报告/表格中的 GT 例句与模板 xlsx（sheet 01 含完整 GT 列）。
- 不 push（由用户决定推送时机）；密钥/权重/视频/关键点一律不入库（`.gitignore` 已覆盖 `models/`、`artifacts/`、`.env*` 等）。

## 2. 阶段 2：补齐 G2（最高优先）

### 2.1 优先级（用户原表）

| # | 任务 | 现状 | 交付 |
|---|---|---|---|
| 0 | Git 冻结 | ✅ 本次完成 | 可复现稳定版本 |
| 1 | SLRT CSL-Daily G2T/SLT | **BLOCKED（外部，2026-10-27 实网复核）**：`g2t_asset_tracker.md` 逐项实测——CSL-Daily G2T 从未发布（issue #17 无回复、#54 0 回复）；官方 9 个 SharePoint 链接全部 404；作者 2026-08 Drive 替换包只有 `ckpts.zip/data.zip/EMNLP24_Online.zip/mBart_de_t2g.zip/mBart_zh_t2g.zip/TA.zip`，**无 G2T/SLT**；HF 无可追溯转载；仓库 `license=null`。**FOUND 但非 B 所需**：CSL-Daily SingleStream-S2T（video→sentence，本地 `F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\csl-daily_s2t\ckpts\best.ckpt`，SHA256 b6bdf7c2…）——它是独立句子翻译路径，值得做一次可行性评估（不等同 G2T） | SLRT 中文翻译基线（B）或 S2T 替代基线 |
| 2 | Uni-Sign + SLRT + LLM 融合消融 | harness 已建（`experiments/p1/e02/llm_fusion/`，stub 验证通过）；**等 LLM 端点选择**（§2.3） | 统一任务消融结果 |
| 3 | 人工关键语义评审 | 协议 + Round-1 评审表（A/A2/B0）生成中；**需要用户本人评审** | 纠错率/误改率/严重错误率 |
| 4 | 预处理优化对照 | 排队（阶段 4） | 优化前后质量与延迟 |
| 5 | SLRT Online 独立验证 | replay 首稳探针已派发 | RTF、首稳时延、字幕改写率 |

其中 1–3 直接决定双模型融合能否进后端主链路；1 拿不到权重时的替代路线 = 固定版本 LLM 后期融合（必须带 C 对照组，见 2.2）。

### 2.2 G2 核心对照实验（字母按用户 2026-10-26 批复；括号为计划包原字母）

| 组 | 方案 | 目的 | 判据 |
|---|---|---|---|
| A | Uni-Sign 原始译文 | 基准 | — |
| B | SLRT → G2T/SLT（计划 C） | 验证独立翻译路径 | 权重 BLOCKED 时以 B0（确定性 join）为退化基线，**不得据此否定 SLRT**（12.22 BLEU 是拼接的分数，不是翻译能力） |
| C | Uni-Sign → LLM（计划 E） | 控制 LLM 本身的润色收益 | C vs A |
| D | Uni-Sign + SLRT Gloss → LLM（计划 F） | 验证 Gloss 的新增价值 | **D vs C**（核心判据） |
| E | 双模型 + 证据约束 + LLM（计划 G） | 验证约束是否减少误译 | E vs D（严重语义错误率） |

- 输入纪律：参考答案/GT/GT 派生分数**不得进入 LLM 输入**（gt_guard 强制）；prompt 版本、温度、模型版本全部固定并记录 sha256。
- 探索可用既有 1086（retrospective_seen）；**最终结论需新的独立测试集 + 人工评审**。

### 2.3 LLM 端点（已选定，2026-10-26 用户拍板）

- **选定：wb-proxy 的 DeepSeek v4.1 flash**（本地 OpenAI 兼容代理，即 DSH harness 自用通道）。
- 实测参数（2026-10-26 smoke 验证，1.22 s / usage 正常）：
  - `base_url = http://127.0.0.1:8788/v1`（`POST /chat/completions`，`api: openai-completions`）
  - `model = deepseek-v4.1-flash`
  - 认证：Bearer，密钥读自 `C:\Users\ADMIN\.dsh\.credentials.yaml` 的 `refs.WB_PROXY_API_KEY`（**密钥只允许运行时内存读取：不打印、不落盘、不入库、不进任何 JSONL/日志**）；备选同名环境变量。
- 记录纪律：每次 run 的 manifest 必须写 `provider=wb-proxy`、`model=deepseek-v4.1-flash`、`base_url`、`prompt_version + sha256`、`temperature/max_tokens`、逐调用 latency 与 usage tokens——满足 G2 验收三问的"成本"维度。
- 本机其余候选（Codex CLI 已登录）保留为备选；grok/kimi/hermes 未安装。
- 风险与边界：该代理同时服务 DSH 自身；实验调用需串行或 ≤2 并发、瞬时错误退避重试并记录；代理属用户自有基础设施，**不得**把密钥或代理日志写入仓库。

### 2.4 G2 验收三问（用户定义，逐条落实）

1. **质量是否提高**：统一 BLEU-4/chrF（canonical `p1-v1` 口径）+ 按 signer 聚类 Bootstrap CI；禁止凭个案判定。
2. **是否引入新的关键事实错误**：人物/否定/数字/时间/动作/对象六槽位盲评（0–4 分 + 是/否/不确定），统计纠对/改错；双评审 + 分歧仲裁。
3. **成本是否值得**：LLM 调用数、tokens、P50/P95 时延，与单模型基线比较；质量小幅提升但延迟大增的方案不进实时链路。

### 2.5 G2 首轮真实结果（2026-10-26，探索性 50 条）

- 端点/参数：wb-proxy `http://127.0.0.1:8788/v1/chat/completions`，`deepseek-v4.1-flash`，temperature=0，`max_tokens=2048`，严格串行；50 条 pilot（10 signer × 5）× C/D/E = 150 次调用，成功 147（3 次长推理截断）。
- 结果（canonical p1-v1，GT 上游参考）：A **31.2308** / C **26.6838** / D **26.9450** / E **28.6302** BLEU-4；Δedit A→C −0.033（CI 不含 0）、E−D +0.011（n.s.）、**E 相对 C +0.040（CI +0.0019…+0.0821，显著）**；D−C（Gloss 价值）ΔBLEU −0.04、Δedit +0.026（边缘 n.s.）。
- 成本：150 次调用 **113,653 tokens**；P50 **3.16 s** / P95 **8.89 s**（串行 HTTP）。
- 判定（暂定）：**当前 prompt 下"每句都调 LLM"不可取**；无条件重写显著变差。D−C/E−D 未达显著，E 相对 C 的改善说明"证据约束"方向可能有效但需 prompt 改进。
- 下一动作：① prompt v2（C 最小必要修正；D/E 冲突才改）50 条对照；② **选择性级联**（默认 A，E06 RuleJudge 触发才调 LLM）成本-质量表；③ 仍不扩到 1086；④ 最终 G2 结论需新独立集 + Round-1 人评（评审包已就绪，待用户回填）。
- 产物：`experiments/p1/e02/llm_fusion/runs/g2a-wbproxy-50-v2-20261026/`、`report_real_50.md`、`fusion_ablation_g2.csv`；事故证据：`runs/g2a-wbproxy-50-20261026/incident_note.md`（max_tokens=256 截断，未进入对比表）。
- **v2 prompt 结果（2026-10-27，同 50 条）**：A **31.2308** / C **28.9518** / D **28.5872** / E **29.1880**（n=45）；C−A **−2.28 [−4.04,−0.70]**（仍显著为负）；D−C −0.365、E−D −0.416（均 n.s.）；edit 差全部 n.s.。成本 113,291 tokens，P50 C 2.04 s / D 3.01 s / E 4.48 s，P95 5.0–9.5 s。
- **选择性级联**（默认 A，E06 槽位冲突触发 28% 才调 E_v2）：BLEU **30.7487**（Δ −0.482 [−2.532,+1.061]，n.s.）、22,981 tokens、P50 8.62 s/P95 14.77 s（整批口径）；全调 D(v2) −2.644、全调 E(v2) −3.266（CI 不含 0）。结论：省钱但不增值。
- **G2 结论（ADR-P1-010，2026-10-27）：FAIL（无质量增益证据）** → 融合不进主链路、保留可插拔接口、不扩 1086；人评维度仍未验证。详见 `G2_DECISION_MEMO.md`。

## 3. 阶段 3：P0-A 后端基础（已收口，2026-10-27）

- **A0–A5 共 9 张任务卡全部 PASS**（详见 `P0A_IMPLEMENTATION_TASKS.md`「P0-A 收口总结」）：新增 `evidence.py`/`prediction_log.py`/`replay_adapter.py`/`api_contract_v1.py`/`job_runner.py`/`media_probe.py`/`fusion.py`，144 项新测试；回归 `.conda-env` 939/13、`.venv-libras` 990/13；v3 端点与已验证推理路径零改动。
- 规范基线：V3.0 架构文档在 `docs/架构开发文件/`（已入库）；缺口分析 `P0A_BACKEND_GAP_ANALYSIS.md`（ModelAdapter ~90% 可复用，E04 单卡串行与现有单容量租约一致）。
- 当前 API 面：`POST /v1/videos/infer`、`GET /v1/jobs/{id}`、`/v1/health/live|ready`、鉴权/413/幂等/错误码统一；推理由 `replay_jsonl_v1` 驱动（无 GPU 全链路验证）。
- 纪律：不迁移 UI、不做微服务化、不动已验证推理路径；FusionController 只留接口（等 G2 重启条件）。

- **Uni-Sign 前端（CSL-Daily test，同 checkpoint/解码/帧采样，配对 signer bootstrap）**
  - `pose_sample_only`：pilot-50 全部 ≤172 帧 → 0 跳帧、50/50 预测逐字相同（no-op）；全部 12 条 >256 帧片段上跳 13.2% 帧、**12/12 预测逐字相同**、e2e 7013→6271 ms（−10.6%）。→ **可安全开启**（零质量变化），但该域仅 1.1% 片段 >256 帧，不能宣称数据集级提速。
  - `pose_detect_stride=5`：e2e −36.4%、pose −40%，但 BLEU −1.69、**Δedit −0.0205（CI [−0.0563,+0.0153]）**、仅 32/50 相同 → **质量门槛 FAIL，不建议默认开启**；CSL-News/CE-CSL 的"无显著差异"不能外推。
  - 生产默认值未改；开启 sample_only 需同步更新姿态 protocol 身份。
- **SLRT 逐窗预处理（数值等价缓存）**：定位 stride=1 下同帧 RGB+热力图重复计算 ~16×；内容 hash 帧缓存（原始函数 + cat）→ RGB/heatmap **max abs diff 0.0**、`torch.equal` True、Gloss **20/20 一致**、WER **0.27007 不变**；e2e P50 **5923→4836 ms（−18.4%）**、P95 −16.4%、prepare −70.3%。→ **建议合入生产**（纯工程收益），下一瓶颈 canonical_frame（28%）与 S3D（39.5%）。
- 证据：`experiments/p1/e04/opt_unisign_frontend.md`、`opt_slrt_preprocess.md`、`opt-runs/`（3 run）、`hardware_report.md` §H。

## 5. 阶段 5：G2 决策规则（预先约定）

- 若 D−C 有稳定增益（新独立数据 + CI 不含 0）**且** E 相对 D 的严重语义错误率下降、人评纠错>误改、成本可接受 → 融合模块进主链路。
- 否则 → 保留可插拔接口，交付可靠单模型系统；Jev/学习式调度继续暂缓。

## 6. 暂缓与不重复清单（用户批复）

| 工作 | 决定 | 原因 |
|---|---|---|
| E00 审计 / E01 复现 / E04 基础性能 | **不重复**（保留基准） | 已通过/已实测 |
| Jev 决策集成 | 暂缓 | 融合收益未定 |
| 学习式动态调度 | 暂缓 | 缺可靠质量预测标签（Uni-Sign 无分数） |
| 模型内部 Early Exit | 暂缓 | 成本高、当前价值不足 |
| CSL/Libras 自动识别 | 暂缓 | 无双语训练/评估数据 |
| Libras 权重 | 禁止用 CSL 权重冒充 | ADR-P1-007 |

## 7. 任务板映射

| 任务 | 内容 | 状态 |
|---|---|---|
| task-6 | LLM 融合消融 harness（C/D/E）+ stub dry-run | analyst 执行 |
| task-7 | 人工评审协议 + Round-1 评审表（A/A2/B0） | analyst 执行 |
| task-8 | SLRT replay 首稳探针（RTF/首稳/撤销率） | perf-streamer 执行 |
| task-9 | P0-A 后端缺口分析与实施计划 | p0a-backend 执行 |
| task-10 | 预处理优化对照（Uni-Sign 前端 + SLRT 窗口） | pending（阶段 4） |
| task-11 | SLRT CSL-Daily G2T/SLT 权重获取 | BLOCKED（外部作者发布；跟踪上游 repo/issue） |
