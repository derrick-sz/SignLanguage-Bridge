# E01 · 预处理一致性与推理复现报告（Pilot-50：U1 / U2 / S1）

> 模板：`SignLanguageBridge_P-1_实验计划包/docs/07_实验报告模板.md`（A–E）。
> 本报告所有数字来自本次实际运行产物；历史 1086 数字一律标注 `imported`，未重新推理。
> 状态口径：只有真实跑通并保存证据的分支写 PASS。

## A. 实验基本信息

- 实验编号：`E01`（P-1 · 预处理一致性及推理可复现性，P0）
- run_id：
  - U1 视频：`e01-pilot50-u1-unisign-video-20261026`
  - U2 官方 pose：`e01-pilot50-u2-unisign-pose-20261026`
  - S1 SLRT Gloss：`e01-pilot50-s1-slrt-gloss-20261026`
- 日期 / 负责人：2026-10-26 21:32–22:00（+08:00）／reproducer（P-1 实验团队；独占 GPU、单进程串行）
- 状态：`PASS`（三条路径全部实跑并保存证据；50/50 逐条可审计，历史输出 100% 逐字复现）
- 实验假设：在**同一 checkpoint、同一权重哈希、同一预处理/解码协议**下，Pilot-50 三路重跑应逐条复现历史 1086 快照的文本输出；若不一致，差异必须能归因到输入、采样、权重或环境；输入差异（trimmed 视频 vs 官方 pose pkl）必须可解释。
- 样本范围：**Pilot-50**（`experiments/p1/datasets/pilot_50.csv`，50 条；10 名签者 P0000–P0009 各 5 条；cohort：existing_627=20、added_459=30）。它取自 `selection-1086.json` = 本地可用的 1,086 条 CSL-Daily test 片段（官方 test 共 1,176 条，本地缺 90 条偏长句视频）。
  - **子集边界**：指标是 50 条子集数字，不可与全量 1,086 或论文 1,176 test 数字直接比较；仅用于「同输入同协议复现」与路径间横向对照。
  - usage 列 = `retrospective_diagnostic_only`（回溯诊断，不是盲测）。
- 运行配置与 Git SHA：
  - `git_sha = cf291c2a81558faf6de0dc5a7243e730531e4367`，branch `codex/model-contract-auto-captions`，`dirty_tree = true`（132 个路径的会话前既有改动；E01 未修改 `scripts/`、`slbridge/`、`artifacts/`，未 git commit/push）。
  - 解释器 `.conda-env\python.exe`（Python 3.10.22）；torch `2.9.1+cu128`；设备 NVIDIA GeForce RTX 5060 Ti 16,311 MiB，驱动 616.92，CUDA 12.8；精度 bf16（Uni-Sign）/ bf16 CUDA（SLRT）；batch=1；单进程独占 GPU；未使用 Docker。
- 模型 checkpoint SHA256（本次实读文件计算，与 bundle manifest 一致）：
  - Uni-Sign：`models/external/uni_sign/csl_daily_pose_only_slt.pth`，1,186,925,112 B，SHA256 `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c`；bundle manifest SHA256 `462ea712c73ba0b0c7a91a4c6d5e596e8ff5bd58854dca53f8c73e03b8cdba7e`，mt5-base 权重文件 SHA256 `180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614`（两者均取自 bundle manifest 并记录在 run manifest/model 段）。
  - SLRT：`models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt`，499,554,423 B，SHA256 `e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d`，vocabulary SHA256 `0c4984f0dd3e7897b8c8d9e6e99a848b60814a5d9cf7a49baf7cc9b9cd124c76`。
  - 解码：`num_beams=4`、`max_new_tokens=100`、`max_frames=256`（三条路径与历史一致）。
- 预处理版本、视频路径哈希、FPS/PTS 取得方式：
  - 原始视频逐条实测 SHA256、解码帧数、尺寸、PTS/FPS：`video_probe.csv`（ffprobe `-count_frames`，ffprobe 来自 `E:\ffmpeg\...\bin\ffprobe.exe`；fps 取自 ffprobe `avg_frame_rate`，50 条为 29.97003 或 30.0，`nb_read_frames == nb_frames` 50/50）。
  - 句子片段输入（三条路径共用同一裁剪片段）：`artifacts/csldaily-video-test-627-20261026/trimmed/<sample>.mp4`；逐条 SHA256 与对应 pose NPZ `metadata_json.video_sha256` **50/50 完全一致**；帧数与 alignment `trim.frames`、`ffprobe` 解码帧数、标注 `num_frames` **50/50 三方一致**（`preprocess_audit.csv`）。
  - Uni-Sign 视频路：trimmed mp4 → 本地 RTMPose wholebody 133（rtmlib `mode=lightweight`，实际 `rtmw-dw-l-m`）→ `isotropic_short_side` 归一化 → `pose_orientation=preserve` → `max_frames=256` 确定性 `linspace` 采样（`slbridge/uni_sign_preprocess.py:100-115`）。
  - Uni-Sign 官方 pose 路：`artifacts/cslnews-eval/csl-daily-pose/pose-format/<sample>.pkl`（官方发布的 RTMPose pose 归档）→ 按上游 `load_pose` 语义取 `keypoints[start:end]`、instance 0 → 同一 `prepare_pose_window`（`rescale_x=1.0`，官方坐标本身为 x/W,y/H；50 条帧均为正方形，故与 isotropic 等价）。
  - SLRT 路：trimmed mp4 + 官方 HRNet 对齐 NPZ（`coco_wholebody_133_xyc_512_v1`，`hrnet_w48_dark`）→ 官方 16 帧窗口、stride=1、7 窗多数投票（`window_size=16, stride=1, vote_size=7`）。
- 指标工具版本、tokenizer/sacreBLEU signature：
  - 统一评测器 `experiments/p1/eval/evaluate.py`（`metric_definition_version = p1-v1`），BLEU 采用 sacreBLEU `tokenize=13a`，并复刻上游 `fine_tuning.py:282-284` 的 `' '.join(list(text.replace(" ","").replace("\n","")))` 重写；参考文本使用**上游半角、逐字空格**参考（`gt_1086.csv :: ref_sentence_upstream`）。
  - WER：`(ΣS+ΣD+ΣI)/ΣN_gt_gloss`（`experiments/p1/eval/metrics.py :: corpus_wer`），参考 = `gt_1086.csv :: gt_gloss`，**EVAL ONLY**。

## B. 复现实验步骤

```powershell
# 环境
$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'
cd E:\githubproject\SignLanguage-Bridge

# 0) 输入清单（50 条 sample_id，保持 pilot_50.csv 顺序）
#    experiments/p1/e01/inputs/pilot_50_ids.txt

# 1) 原始视频探针（ffprobe -count_frames；逐条 SHA256/帧数/尺寸/FPS/码率/PTS）
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\build_video_probe.py

# 2) 预处理一致性审计（trimmed SHA vs NPZ metadata；帧数三方对齐；两条 Uni-Sign 输入协议）
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\build_preprocess_audit.py

# 3) U1 Uni-Sign 视频端到端重跑（50 条 -> 新 run 目录）
.\.conda-env\python.exe -X utf8 scripts\evaluate_unisign_csldaily_video.py `
  --selection artifacts\csldaily-video-test-627-20261026\selection-1086.json `
  --artifact-dir artifacts\csldaily-video-test-627-20261026 `
  --output-dir experiments\p1\runs\e01-pilot50-u1-unisign-video-20261026 `
  --only-file experiments\p1\e01\inputs\pilot_50_ids.txt

# 3b) 归档为 P-1 run 产物：GT-free model_outputs + manifest + canonical metrics + 历史比对
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\finalize_unisign_run.py --kind video

# 4) U2 Uni-Sign 官方 pose 重跑（脚本无 --only：先把 refs/pres 过滤成 pilot 50）
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\prepare_u2_inputs.py
.\.conda-env\python.exe -X utf8 scripts\evaluate_unisign_csldaily_pose.py `
  --pose-dir artifacts\cslnews-eval\csl-daily-pose\pose-format `
  --refs experiments\p1\runs\e01-pilot50-u2-unisign-pose-20261026\inputs\refs-pilot50.txt `
  --upstream-pres experiments\p1\runs\e01-pilot50-u2-unisign-pose-20261026\inputs\pres-pilot50.txt `
  --output-dir experiments\p1\runs\e01-pilot50-u2-unisign-pose-20261026 --device cuda
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\finalize_unisign_run.py --kind pose

# 5) S1 SLRT Gloss 重跑（新 artifact-dir bootstrap：只复制 trimmed/pose/alignment，不重跑 align）
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\prepare_s1_inputs.py
$ids = Get-Content 'experiments\p1\e01\inputs\pilot_50_ids.txt' | Where-Object { $_.Trim() -ne '' }
.\.conda-env\python.exe -X utf8 scripts\align_slrt_csl_daily_subset.py --stage infer `
  --selection experiments\p1\runs\e01-pilot50-s1-slrt-gloss-20261026\inputs\selection-pilot50.json `
  --annotation 'F:\model_weigth\data\csl-daily\csl-daily.test' `
  --artifact-dir experiments\p1\runs\e01-pilot50-s1-slrt-gloss-20261026\artifact `
  --only $ids --force
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\finalize_slrt_run.py

# 6) 汇总
.\.conda-env\python.exe -X utf8 experiments\p1\e01\tools\build_e01_summary.py

# 7) 回归测试（既有 P-1 基础设施）
.\.conda-env\python.exe -m pytest experiments\p1\tests\test_p1_infra.py -q   # 12 passed in 1.02s
```

- 运行日志：各 run 目录 `predictions-raw/run.log`（U1/U2 的冻结脚本 stdout；U2 的 run.log 含逐条 progress）。
- 输入表 / manifest：`experiments/p1/datasets/pilot_50.csv`（SHA256 `f0f46125…f4c99c4b`）、`experiments/p1/e01/inputs/pilot_50_ids.txt`（`30c6b086…`）、U2 `inputs/{refs-pilot50.txt, pres-pilot50.txt}`、S1 `inputs/selection-pilot50.json`；每条 run 的 `manifest.json` 记录 git/硬件/权重哈希/数据快照/命令/样本数/失败数/状态。
- 原始 JSONL 预测：`predictions-raw/predictions.jsonl`（冻结脚本自身输出，按脚本设计内嵌 reference 文本，仅作 eval 侧证据，**不**并入 model_outputs 的推理记录）；对外交付的 GT-free 推理记录是 `model_outputs.jsonl`。
- 失败、超时/OOM 样本：**无**。U1/U2 各 50/50 成功、0 失败、0 空输出；S1 50/50 `exit_code=0`、`status=final`、0 空 Gloss。无 OOM 重试（单进程独占，避免坑点 §7 的并发 OOM 假失败）。
- 只读证据：`artifacts/` 下没有任何文件在本次会话窗口（21:30 之后）被修改（文件时间戳核验）。

## C. 结果（measured / imported 严格区分）

### C.1 本次实测（measured，全部为 50 条 Pilot-50）

| 分支 | 样本数 | 成功数 | BLEU-4 | chrF | Gloss WER ↓ | 严重语义错误率 | P50 | P95 | VRAM | 证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| U1 Uni-Sign **视频**端到端 | 50 | 50 | **31.2308** | 27.1849 | — | NOT_RUN | 3.625 s | 5.336 s | peak reserved 2,708 MB（peak alloc 1,183 MB） | `experiments/p1/runs/e01-pilot50-u1-unisign-video-20261026/` |
| U2 Uni-Sign **官方 pose** | 50 | 50 | **31.7823** | 27.9108 | — | NOT_RUN | 0.351 s | 0.476 s | peak reserved 2,746 MB（peak alloc 1,249 MB） | `experiments/p1/runs/e01-pilot50-u2-unisign-pose-20261026/` |
| S1 SLRT **Gloss** | 50 | 50 | — | — | **0.3170**（S72/D21/I17，参考 347 gloss token） | NOT_RUN | 14.193 s | 19.172 s | 未记录（`null`，不估计）；进程 RSS≈1,805 MB | `experiments/p1/runs/e01-pilot50-s1-slrt-gloss-20261026/` |

- 指标口径：C.1 的 BLEU/chrF 为 canonical（`evaluate.py`，参考 `gt_1086.ref_sentence_upstream`，上游半角逐字空格参考），WER 为 canonical CSLR；三者都在**全部 50 条成功样本**上计算（`metric_scope = successful_subset_all_succeeded_corpus_metrics`），无失败样本被剔除。
- 延迟口径：U1/U2 的 P50/P95 是逐样本 `e2e_seconds`（`time.perf_counter()`，包含 RTMPose 前端 + mT5 生成；模型加载/warmup 不计入逐样本）。S1 是每样本 SLRT CLI 子进程 `elapsed_seconds`，**含每条子进程约 3.9 s 的固定权重加载开销**（坑点 §7），因此不能当作纯推理时延。
- 与 UI/服务链路的关系：本次是离线文件入口；SLRT 的 `status=final` 指单段离线结果，不代表实时流式交付已验收。

### C.2 与历史快照的逐条复现（measured vs imported）

| 路径 | 历史快照（imported，1086 条） | 本次 50 条 | 原样匹配 | NFKC 规范化匹配 | 差异样本 |
|---|---|---:|---:|---:|---|
| U1 视频 | `artifacts/csldaily-video-test-627-20261026/unisign-video-1086-paperrefs/predictions.jsonl` | 50/50 成功 | **50/50（100%）** | **50/50（100%）** | 0 |
| U2 官方 pose | `artifacts/cslnews-eval/csldaily-test-1086-pose-20261026/predictions.jsonl` | 50/50 成功 | **50/50（100%）** | **50/50（100%）** | 0 |
| S1 SLRT Gloss | `artifacts/csldaily-video-test-627-20261026/slrt/<sample>.json` | 50/50 成功 | **50/50 gloss_sequence 完全一致** | 50/50 | 0 |

- 同一 50 条子集上的 canonical 指标也逐位相同（历史预测 vs 本次重跑）：U1 均 31.2308 / 27.1849 / edit 0.4831；U2 均 31.7823 / 27.9108 / edit 0.4885。即**文本层 100% 复现**，不存在"平均值相近但样本不同"的情况。
- U2 另与**上游作者自报预测**（`test_tmp_pres.txt` 的 50 条）对照：`this_bleu4=31.7823` vs `upstream_bleu4=32.0631`（−0.28），逐字相同 `48/50`；差异归因于上游 `random.sample(range(duration), k=max_length)`（随机帧采样）与本仓库确定性 `linspace`（坑点 §1），属预期而非模型差异。
- 历史 1086 全量基线（imported，仅作口径参照，**不是本次测量**）：Uni-Sign 视频 BLEU-4 24.8157 / chrF 21.7974 / edit 0.4157 / exact 15；官方 pose 25.5846 / 22.4284 / 0.4212 / exact 18；SLRT corpus WER 0.3018（1,086 条，S1,154/D805/I522）。Pilot-50 子集分数高于全量（31.2/31.8 vs 24.8/25.6），是子集差异，**不得**写成模型提升。

### C.3 三条路径的横向对照（同 50 条、同权重、同解码、同参考）

| 对照 | ΔBLEU-4 | ΔchrF | Δedit_sim | 解释 |
|---|---:|---:|---:|---|
| 视频 − 官方 pose | **−0.5515** | −0.7259 | **−0.0054** | 仅有姿态来源（本地 RTMPose vs 官方 HRNet 归档）与帧集合不同；方向与 1086 全量的 −0.77 / −0.0055 一致 |

### C.4 输入一致性实测（`video_probe.csv` / `preprocess_audit.csv`）

| 检查项 | 结果 |
|---|---|
| 原始视频探针 | 50/50 成功；raw 解码帧数 min 70 / median 157 / max 332（合计 8,133）；尺寸仅 1280×1280 与 1080×1080（正方形）；fps 29.97003 或 30.0；codec h264 / yuv420p；总 67.56 MB；50 条 SHA256 全部记录 |
| 标注帧数 ≠ 原始帧数 | 50/50 不同（标注 `frames` 是句子片段长度：min 50 / median 116 / max 172；`equals_raw_frame_count=0`），与 E01 预期一致 |
| trimmed SHA256 == NPZ metadata `video_sha256` | **50/50** |
| trimmed 帧数 == NPZ 帧数 == alignment `trim.frames` == ffprobe 解码帧数 | **50/50**（标注帧数也一致，50/50） |
| 对齐质量 | alignment `best_mean_px` min 3.695 / mean 6.622 / max 13.265；`low_confidence` 0/50；`confidence_clipped` 合计 56,267（官方 pkl confidence 曾被 clip 到 [0,1]，见坑点 §R12） |
| NPZ 协议 | 50/50 `coordinate_system=coco_wholebody_133_xyc_512_v1`、`frontend=hrnet_w48_dark`、shape `T×133×3` |
| Uni-Sign 视频路帧采样 | 50 条句子片段 ≤172 帧，均 ≤256 → **全部未下采样**（`pilot_clips_over_256=0`） |
| Uni-Sign pose 路裁剪字段 | 官方 pkl 的 `T_raw` == 原始视频解码帧数（50/50）；模型实际消费 `end−start` 帧。与标注帧数相比：median **+3**、min −2、max **+140**，45/50 非零、15 条 \|Δ\|>5；其中 1 条（`S002128_P0005_T00`，crop 303 帧）触发 256 帧下采样 |
| 归一化等价性 | 视频与官方 pose 两条路都在**正方形**帧上工作（1280×1280 / 1080×1080），故 `isotropic_short_side` 与官方 x/W,y/H 数值等价；差异只剩姿态前端与帧集合 |
| 输入差异可解释 | 50/50（`input_diff_explained=True`），不构成"未解释的静默失败" |

## D. 错误分析

- **模型输出与历史参考的差异**：0 条。U1/U2 文本 50/50 原样一致；S1 gloss_sequence 50/50 一致。`compare_unisign_video.csv`、`compare_unisign_pose.csv`、`compare_slrt.csv` 逐条列出两侧预测文本、匹配位与差异分类（本次全部 `identical`），可供审计。
- **与前处理/对齐相关的可疑失败**：无失败样本。对齐质量良好（mean 6.62 px、0 条 low_confidence），但需注意官方 pose pkl 的 `[start,end)` 与标注句长不等（median +3、max +140 帧）——这是**上游发布归档的属性**，历史 run 与我们使用同一 `load_pose` 语义，因此不影响本次复现结论；但它解释了为什么官方 pose 路径精度与视频路径不完全同级，也是后续若要严格对齐需要单独审计的点。
- **参考文本口径（本次实测的新坑证据）**：同一份 U1 视频预测、同一切分，
  - 官方标注参考（全角标点、无空格，`selection-1086.json::reference`）：BLEU-4 = **28.1011**、chrF 24.2261；
  - 上游半角逐字空格参考（`gt_1086.ref_sentence_upstream`）：BLEU-4 = **31.2308**、chrF 27.1849；
  - **差 3.1297 BLEU**，与坑点 §R18 记录的 ~2.8 BLEU 同向。因此所有跨运行比较只使用 canonical（上游参考）口径；冻结脚本 summary 里的 28.10 是"官方全角参考"口径，**不得**与历史"paperrefs"数字并列比较。
- **数据泄漏检查**：三份 `model_outputs.jsonl`（各 50 条，共 150 条）逐个通过 `experiments.p1.gt_guard.assert_gt_clean`，**0 泄漏**；GT 只出现在 `metrics.json` / `eval_records.jsonl`（评测侧）与 `references/gt_1086.csv`；`compare_*.csv` 不含参考文本。冻结脚本的 `predictions-raw/predictions.jsonl` 按脚本设计内嵌 reference，已隔离在 `predictions-raw/` 并标注为 eval 侧，绝不作为推理输入或交付推理记录。
- **静默失败**：0。判定不只看退出码：U1/U2 检查 `status=ok` 且 prediction 非空；S1 检查 `exit_code==0` **且** `status==final` **且** 结果文件可解析且 gloss 非空——50/50 全部满足，空输出 0。
- **反例（新方法把原本正确样本改错）**：不适用（本次是复现，不是替换实验），未出现。
- **人工语义评分**：`NOT_RUN`（E01 不含人工评审；模板中的"严重语义错误率"留空，不填估计值）。
- **正确性边界说明**：本次 100% 一致只证明「已记录输入 + 已固定协议」下的推理可复现；本地 RTMPose 前端与官方 HRNet 归档仍不是同一估计器，且未重跑 SLRT align 阶段（复制的是已审计的历史对齐产物，并用 NPZ metadata `video_sha256` 逐条校验 byte-identical）。

## E. 结论与工程选择

- **是否支持原假设**：`YES`。三条路径在 Pilot-50 上文本层 100% 复现历史 1086 快照（U1/U2/S1），所有输入差异（trimmed vs 官方 pose 前端、官方 pkl 裁剪帧数、正方形帧归一化等价性）均有实测证据与逐条记录；无未解释的静默失败。
- **Gate 相关判断**：E01 操作性通过规则满足——Pilot-50 无未解释静默失败；与参考结果不一致样本数 = 0；规范化完全匹配率 100% ≥ 95% 调查起点。**E01 = PASS**（三条路径各自 PASS）。
- **对 V3.0 / 后续组件的影响**：
  1. 该 checkpoint 权重哈希、预处理字段（`pose_normalization=isotropic_short_side`、`pose_orientation=preserve`、`max_frames=256`、确定性 linspace）可固化为与历史可比的协议；每条 run 的 `protocol_id`（U1 `1a8fe60053698eddb443f7613bec137a0b56aabe29348b52763e7cd6b7f28169`、U2 `ef3354909d49649db33dbe5a4822179ebee2c59094d2eb0b843d6555b613bdaf`，`rescale_x=1.0`）可直接作为 E02/E05 的输入边界。
  2. **参考口径必须固化**：canonical 指标只允许用 `gt_1086.ref_sentence_upstream`；`gt_sentence_official`（全角）只做审计说明，禁止与 paperrefs 数字混用。
  3. SLRT 的 S1 只验证了"推理阶段复现"；`align` 阶段复用历史产物必须在任何引用处标注，不能写成"重新提取的 pose"。
- **计划推进**：P0（E01 完成）；E02 可直接复用本次 U1/U2/S1 的 run_id 与 GT-free `model_outputs.jsonl` 作为 A/A2/B 分支的"本机同协议"记录（或继续沿用历史 1086，但必须保留 imported 标注）。
- **不确定性与泛化限制**：
  - Pilot-50 仅 10 名签者、每签者 5 条；per-signer BLEU 波动极大（U1: P0009 12.11 … P0004 46.44），不可据此做签者级结论。
  - 本次 50 条句子片段全部 ≤256 帧，**未覆盖**"输入 >256 帧时确定性 linspace vs 上游 random.sample"的随机性路径；U2 vs 上游预测的 48/50 一致也只在短片段上验证。
  - 未做同输入重复推理 repeatability（≥2 次）实验；未跑 S2（官方 crop/keypoint 工具复核，可选项）；未跨机器/跨驱动复现；未做许可证法务核对（Uni-Sign 权重 CC-BY-NC-4.0、CSL-Daily 数据条款适用）。
  - 延迟只代表本机独占 GPU 的单进程条件，不可与历史含 GPU 争用的延迟数字（如视频 e2e mean 6.11 s、SLRT mean 16.83 s）直接比较；本次测得 U1 e2e mean 3.776 s、S1 mean 14.46 s 是**本机独占条件**下的实测。
- **下一步最小行动**：
  1. E02 复用本次 run 产物时，在 manifest 里显式带 `run_id + protocol_id + canonical 参考口径`，并保留 historical 1086 数字的 `imported` 标注。
  2. 追加一个"长片段（>256 帧）采样随机性"探针（固定种子对照 `random.sample` 与 `linspace`），量化 E01 未覆盖的采样分支；这属于 E01 的补充证据，不改变本次 PASS 结论。
  3. 若要声称"预处理链路全量可复现"，需另立任务重跑 SLRT `align` 阶段（本机 GPU 逐帧 RTMPose，50 条约需数分钟）并与本次复制的对齐记录逐条比对；当前结论边界是"推理阶段 + 已审计对齐输入"。
  4. 在 E02 的评测代码里加一条断言：同一对比中参考文件 SHA256 必须一致，防止全角/半角参考再次混用。

## 附录 · 产物与校验

- `experiments/p1/e01/`
  - `video_probe.csv`（50 行；原始视频实测：sha256/frames/width/height/fps/duration/codec/bytes）
  - `preprocess_audit.csv`（50 行；trimmed SHA vs NPZ metadata、帧数三方一致、对齐指标、两条 Uni-Sign 输入协议、pose pkl 裁剪字段）
  - `compare_unisign_video.csv` / `compare_unisign_pose.csv`（50 行；历史 vs 重跑逐条预测、raw/NFKC 匹配、差异分类；本次全 identical）
  - `compare_slrt.csv`（50 行；历史 vs 重跑 gloss_sequence、exit code、耗时、空输出标记）
  - `summary.json`（机器可读汇总：inputs / 三路 path 状态与指标 / 参考口径 / measured-vs-imported / limitations）
  - `reproduction_report.md`（本文件）
  - `inputs/pilot_50_ids.txt`、`tools/*.py`（可复现的探针、审计、比对与汇总脚本）
- `experiments/p1/runs/<run_id>/`（每个 run 均有）
  - `manifest.json`（git SHA/dirty tree、硬件与 torch/CUDA、模型与权重 SHA256、数据集快照哈希、真实命令、样本数/失败数、状态、GT 政策）
  - `model_outputs.jsonl`（GT-free，guarded writer）、`runtime_metrics.jsonl`
  - `metrics.json` + `eval_records.jsonl`（canonical 指标与逐条评测记录，含 GT，仅评测侧）
  - `comparison.json`（与历史快照的匹配率、差异清单、同子集指标、延迟）
  - `predictions-raw/`（冻结脚本原始输出 + `protocol.json` + `run.log`；内嵌 reference，eval 侧证据）
- 回归校验：`pytest experiments/p1/tests/test_p1_infra.py -q` → **12 passed in 1.02s**；三份 `model_outputs.jsonl` GT-guard 全绿；`artifacts/` 历史文件时间戳未变（只读）。
