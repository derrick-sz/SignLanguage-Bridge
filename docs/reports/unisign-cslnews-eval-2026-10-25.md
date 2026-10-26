# Uni-Sign 在 CSL-News 上的准确率与延迟验证（2026-10-25）

本报告回答两个问题：项目当前集成的 Uni-Sign 权重在 `data/CSL-NEWS` 上**准确率**如何、**延迟**如何。
所有数字来自本机实跑，脚本、协议文件与逐样本预测都在 `artifacts/cslnews-eval/` 下可复核。
**这不是产品验收，也不是 CSL-News 泛化准确率**，理由见第 2 节。

## 0. 结论先行

1. **项目当前集成的 `csl_daily_pose_only_slt.pth`（CSL-Daily 微调权重）在这批 CSL-News 片段上不可用**：
   379 条配对样本上 BLEU-zh **1.27**、字符编辑相似度 **7.90%**、exact match 0；输出是通顺但与内容无关的中文。
2. **换成作者同时发布的 `csl_stage1_weight.pth`（CSL-News pose-only 预训练权重）后，同一批数据、同一姿态，
   BLEU-zh 从 1.27 升到 16.75（+15.5）**；再配上作者发布的 pose 文件（同一 rtmlib 模型、但用 `x/W, y/H` 归一化），升到 **24.67**。
   在**全部 1694 条**上：stage1 + 上游 pose 为 BLEU-zh **23.76** / 编辑相似度 **30.07%**；stage1 + 本地提取 + `wh` 为 **21.69** / **27.72%**（差 −2.07 BLEU）。
3. **延迟瓶颈完全在姿态前端**：单条 8.7 秒片段端到端 **5.54 s**，其中解码+RTMPose **5.13 s（92.6%）**，
   mT5 只占 **0.40 s**。逐帧成本回归 `pose_extract_ms ≈ -67 + 20.12 × 帧数`（n=182，R²=0.994）。
4. **姿态前端确实影响精度，但主因是归一化而不是模型**：本地 rtmlib 提取与作者脚本是**同一个模型**，
   作者脚本用 `x/W, y/H`，本地默认 `isotropic_short_side`。把归一化改成 `wh` 后，同一份本地提取从
   BLEU 15.67 升到 **22.02**、编辑相似度 19.95%→**27.12%**；而"本地估计器 vs 上游估计器"在相同归一化下
   只差 **1.51 BLEU / 1.33 pp**。详见 §10。跨域的 CSL-Daily 权重对以上变化不敏感（BLEU 1.03–1.27）。

## 1. 数据与任务

| 项 | 实测 |
|---|---|
| 视频 | `data/CSL-NEWS/archive_001` 共 **1694** 个 mp4，全部 204×170 / h264 / 30 fps |
| 帧数 / 时长 | 51–500 帧（中位 225），合计 **424,898 帧 = 3:56:03** |
| 标签 | `CSL_News_Labels.csv` 共 **722,715** 行（`video,pose,text`）；本地 1694 条恰为第 **1..1694** 行 |
| 参考文本 | 2–95 字（中位 34），**零重复**；字符集 2146，与 CSL-Daily train 字符集交集覆盖本地 80.38% |
| pose | 本地 **0** 个 `.pkl`（上游要 `pose_format/*.pkl`）→ 本报告的姿态由本地 RTMPose 现跑，或从上游 pose 分片读取 |

数据审计明细与复现命令见 [dataset-audit.md](../../artifacts/cslnews-eval/audit/dataset-audit.md)。

## 2. 三条硬限制（决定这些分数怎么读）

1. **发布里没有 CSL-News 下游微调权重。** HF `ZechengLi19/Uni-Sign` 的 8 个 `.pth` 全在本地：
   `csl_daily_pose_only_slt`、`csl_daily_rgb_pose_slt`、`csl_stage1_weight`、`csl_stage2_weight`、
   `how2sign_pose_only_slt`、`openasl_pose_only_slt`、`wlasl_pose_only_islr`、`wlasl_rgb_pose_islr`。
   其中**没有** CSL-News 微调权重 → 论文 CSL-News 那一行无法本地复现。
2. **本地这批片段不是留出集。** 上游 `datasets.py:527-532`（`S2T_Dataset_news`）把 annotation 顺序的
   `[0, 99%)` 当 train、`[99%, 100%)` 当 test；本地 1694 条落在 `[0, 1694)` → **train 段**。
   因此 stage1 在这批数据上的 23.76 BLEU 是**训练集拟合**，不是泛化。
3. **权重血缘**：`csl_daily_pose_only_slt` 与 `csl_stage1_weight` 的 627 个张量名完全一致（stage2 多 385 个 RGB 张量），
   可判别参数上 stage3 更接近 stage1（`lm_head` cos 0.946 vs 0.915）；上游 `train_stage3.sh` 的 pose-only 分支
   正是从 stage1 初始化。→ **CSL-News 是该权重的预训练域**。但权重不含数据清单，**不能证明这 1694 条进过预训练语料**。

## 3. 准确率

### 3.1 四格消融（同一批 379 条、同一解码、同一帧采样）

| 权重 | 姿态来源 | BLEU-zh | chrF | ROUGE-L | 平均字符编辑相似度 | P95 | ≥0.5 条数 |
|---|---|---:|---:|---:|---:|---:|---:|
| `csl_daily_pose_only_slt`（项目集成） | 本地 rtmlib RTMW-L | **1.27** | 2.95 | 0.133 | **7.90%** | 0.176 | 0 |
| `csl_daily_pose_only_slt` | 上游 pose 文件（rtmlib + x/W,y/H） | 1.27 | 3.02 | 0.136 | 8.09% | 0.176 | 0 |
| `csl_stage1_weight`（CSL-News 预训练） | 本地 rtmlib RTMW-L | 16.75 | 14.82 | 0.302 | 20.40% | 0.535 | 29 |
| `csl_stage1_weight` | 上游 pose 文件（rtmlib + x/W,y/H） | **24.67** | 21.68 | 0.397 | **28.97%** | 0.715 | 71 |

- 姿态来源的影响：**在能用的权重上很大**（+7.9 BLEU、+8.6 pp、P95 0.535→0.715），
  在跨域权重上几乎为零（+0.19 pp，379 条里 21 条预测逐字相同）。
- 权重选择的影响：**决定性**。同一份姿态，换权重差 15.5–23.4 BLEU。

### 3.2 全量 1694 条

| 指标 | stage1 + 上游 pose | stage1 + 本地提取 + wh |
|---|---:|---:|
| BLEU-zh / BLEU-char | **23.76** / 23.76 | **21.69** / 21.69 |
| chrF | 20.92 | 19.12 |
| ROUGE-L（字符级 F1） | 0.4116 | 0.3849 |
| 平均字符编辑相似度（P50 / P95） | **30.07%**（0.25 / 0.723） | **27.72%**（0.229 / 0.696） |
| exact match | **8 / 1694（0.47%）** | 6 / 1694（0.35%） |
| 编辑相似度 ≥ 0.5 | 319（18.8%） | 272（16.1%） |
| 预测/参考长度比 | 1.057 | 1.054 |
| 空预测 / 失败 | 0 / 0 | 0 / 0 |

两列是**同一权重、同一批片段**，只换姿态来源；差值即「用本地提取替代作者发布 pose 文件」的代价（−2.07 BLEU / −2.36 pp）。
项目集成权重（CSL-Daily）在同样全量上的结果是 BLEU-zh 0.74 / 编辑相似度 7.52%（见 §3.1 的跨域说明）。

### 3.3 失败模式（人工速读，非盲评）

- 大多数输出是**通顺且主题正确**的新闻中文；典型成功例：
  `好，欢迎收看新闻频道…共同关注。` → `各位好,欢迎收看新闻频道,每天十八点为您直播的共同关注。`（编辑相似度 0.929）
- 主要错误类型是**信息压缩、实体替换、句子合并**，不是乱码。
- 存在少量 **mT5 重复塌缩**（如 `对对对对…`、`以美谋毒,以美谋毒…`），集中在信息量低或长窗口样本。
- 对项目集成权重（CSL-Daily）的 30 条分层速读：20 条无关 / 8 条部分相关 / 2 条基本相关
  （AI 单标注、分层抽样、非盲，仅作参考；正式人工语义评审 **NOT_RUN**）。

## 4. 延迟

### 4.1 分解（379 条，配对；RTX 5060 Ti 16 GB，bf16，beam=4）

| 环节 | 均值 | P50 | P95 | 说明 |
|---|---:|---:|---:|---|
| 纯视频解码（独立基准 n=40） | 37 ms | 34 ms | 55 ms | cv2 读帧，约 8,900 fps |
| 解码 + RTMPose（逐帧） | 5,152 ms | 4,861 ms | 9,026 ms | 204×170 → 每帧 20.12 ms |
| 姿态预处理 | 1.05 ms | 1.06 ms | 1.44 ms | 可忽略 |
| 模型 forward | 36 ms | 33 ms | 52 ms | |
| mT5 generation | 357 ms | 352 ms | 496 ms | beam=4，max_new_tokens=100 |
| 模型合计 | 394 ms | 386 ms | 537 ms | |
| **端到端（含姿态）** | **5.54 s** | **5.27 s** | **9.42 s** | |
| 端到端（姿态来自磁盘） | **0.409 s** | 0.406 s | 0.558 s | 纯模型路径 |
| 冷启动（加载+两次 warmup） | 20.63 s | — | — | 已排除在逐条统计外 |

- **RTF**：全链路 0.643（P95 0.683）；纯模型 0.047。
- **吞吐**：10.78 条/分钟、47.0 帧/秒；1694 条约 2.6 小时。
- 姿态前端占端到端 **92.6%**。
- **口径提醒 1**：全量 1,694 条那次运行（`pinned-wh-full-20261025`）期间同时跑了 CSL-Daily 复现与两个基准脚本，e2e 观测为 6.12 s / P95 11.37 s / RTF 0.731，**明显受并发争用影响，不得作为基准引用**；本表数字来自无争用的 300 条运行。
- **口径提醒 2（模型侧）**：`scripts/evaluate_unisign_cslnews_checkpoint.py` 在 2026-10-25 之前没有 warmup，**首条记录把一次性初始化算进了 e2e（16.2–16.6 s）**，使「姿态来自磁盘」组均值虚高约 1%。剔除首条后：stage1+上游 pose 0.529 s（max 1.79 s）、stage1+本地 wh 0.536 s（max 1.79 s）。脚本已补 warmup 并记录 `warmup_seconds_before_first_sample`；引用极值时务必剔除首条。

### 4.2 为什么姿态前端这么慢（实测归因）

- 逐帧回归：`pose_extract_ms = -67 + 20.12 × frames`，R²=0.994，截距≈0 → **纯逐帧成本，无固定开销**。
- 每帧跑两个 ONNX 模型（`rtmlib.Wholebody.__call__`：`det_model` → `pose_model`）：
  YOLOX-tiny 检测器输入 **416×416**（`Wholebody` 从 MODE 表取 `det_input_size`；204×170 画面被 letterbox 放大 2.04×），RTMW-DW-L 姿态 192×256（已用实例核对 `det_model.model_input_size=[416,416]`、`pose_model.model_input_size=[192,256]`）。
  并发测得 det 19.8 ms / pose 12.4 ms / 全链路 33.5 ms（稳态约 20 ms），**检测器占大头**。
- `mode="lightweight"` 名不副实：实际加载 `rtmw-dw-l-m`（L 体型），`performance` 才是 `rtmw-dw-x-l@384×288`。
- **每帧都提但模型只用 256 帧**：>256 帧的片段占 39%，其中 30–50% 的姿态计算被丢弃。
- batch=1、每帧同步一次；ORT 报部分节点回退 CPU；GPU 利用率约 30%。

## 5. 复现

```powershell
$env:PYTHONPATH = 'E:\githubproject\SignLanguage-Bridge'

# 1) 项目集成权重（跨域基线），可断点续跑
.\.conda-env\python.exe scripts\evaluate_unisign_cslnews.py `
  --output-dir artifacts\cslnews-eval\run-cslnews-20261025

# 2) 上游 pose 对照（同一权重）
.\.conda-env\python.exe scripts\evaluate_unisign_cslnews_upstream_pose.py --limit 379 `
  --output-dir artifacts\cslnews-eval\pinned-upstreampose-20261025

# 3) 其他发布权重（实验路径，不改变生产校验）
.\.conda-env\python.exe scripts\evaluate_unisign_cslnews_checkpoint.py `
  --checkpoint models\external\uni_sign\csl_stage1_weight.pth `
  --pose-source upstream-pkl --output-dir artifacts\cslnews-eval\stage1-upstreampose-full-20261025
```

生产路径 `UniSignBackend.load_model()` 仍然只校验 pinned CSL-Daily 权重的字节与 SHA256；
其他权重只能走上面的实验脚本，避免把实验权重混进产品契约。

## 6. 限制与不得宣称

- **不得**把 23.76 BLEU 称为“Uni-Sign 在 CSL-News 上的准确率”：它是 CSL-News 预训练权重在其 train 段片段上的拟合。
- **不得**把 1.27 BLEU 称为“模型坏了”：同一管线在 stage1 上正常，说明跨域是主因。
- 没有 CSL-News 官方留出集（1% 尾部）的本地视频，**未做**真正的泛化评测。
- 没有 CSL-Daily 本地数据，**未做**同域复现（论文 25.70 BLEU 仍只能引用上游快照）。
- 人工语义评审、摄像头实拍、流式首字延迟：**NOT_RUN**。
- 延迟数字是单进程、独占 GPU；并发或多实例未测。
- `csl_daily_rgb_pose_slt` / `csl_stage2_weight` 含 385 个 RGB 分支张量，本仓库的 pose-only 运行时无法加载，**未评测**。

## 7. 建议

**准确率侧**

1. 若目标域是新闻手语，**必须用 CSL-News 域权重**（当前唯一可用的是 stage1 预训练权重）；项目集成的 CSL-Daily 权重不要用于新闻场景。
2. 把归一化改成与训练一致（`wh`）可再拿约 **+7 个百分点**编辑相似度、+6 BLEU，见 §10。
3. 想拿到可对外引用的 CSL-News 数字，需要官方 1% 尾部视频或自行留出划分 + 重新训练/微调。

**延迟侧**

| 措施 | 预期效果（未验证） |
|---|---|
| 只对采样出的 256 帧做提取 | **已实测（6 条）**：475 帧片段 **1.87×**（16.93→9.06 s）、≤256 帧片段 1.0×；按全量帧分布推算跳过 **14.7%** 的姿态提取，姿态阶段 **1.17×**，e2e 5.54→**4.73 s**。张量与 `sampled_indices` 逐位一致（`scripts/benchmark_pose_sampling.py`，`pose-sampling-benchmark.json`） |
| 单人固定机位复用检测 bbox（或每 N 帧检测一次） | 砍掉约 60% 每帧成本 |
| 跨帧复用检测 bbox（`pose_detect_stride`） | **已实现并有精度验收**（300 条，stage1）：stride 5 → 姿态阶段 5,206→**2,932 ms（1.78×）**、e2e 5.62→**3.33 s**、RTF 0.649→**0.390**、吞吐 10.7→**18.0 条/分**；BLEU-zh 22.02→22.36，配对 Δ 编辑相似度 **+0.0036（95% CI [−0.0051, +0.0125]）**，胜/负/平 91/87/122 → **在本地静态新闻片段上无显著精度损失**。默认仍为关闭，换场景需重新验证 |
| 姿态模型 batch=4~8 | 提高 GPU 利用率 |
| 直接用上游 pose 文件（若有） | 姿态提取归零，端到端 ≈ 0.41 s/条 |

**两项优化叠加实测（300 条，wh，4 组对照）**

| 配置 | 姿态提取 mean | e2e mean | e2e P95 | RTF mean | 吞吐 |
|---|---:|---:|---:|---:|---:|
| 基线（stride 1，全帧） | 5,206 ms | 5.62 s | 9.79 s | 0.649 | 10.7 条/分 |
| `pose_detect_stride=5` | 2,932 ms | 3.33 s | 5.60 s | 0.390 | 18.0 条/分 |
| `pose_sample_only`（全量推算） | −14.7% 帧 | 4.73 s（推算） | — | ~0.55 | ~12.7 条/分 |
| **两项同时开** | **2,483 ms** | **2.89 s** | **3.54 s** | **0.357** | **20.8 条/分** |

→ 端到端 **1.95×**（5.62→2.89 s），RTF 0.649→**0.357**，帧率 46→**90 帧/秒**。两项都只影响姿态提取阶段，可叠加。
（若以 isotropic 稳态基线 5.54 s 计则为 1.92×；两种基线都来自无争用运行，差别只在同一配置的取样。）

**`pose_sample_only` 确实生效——按片段长度分桶的证据**（组合 vs 仅 stride 5，姿态提取 mean）：

| 片段帧数 | n | 组合 | 仅 stride 5 | Δ mean |
|---|---:|---:|---:|---:|
| <200 | 73 | 1,596 ms | 1,589 ms | **+8 ms** |
| 200–299 | 125 | 2,635 ms | 2,659 ms | **−24 ms** |
| 300–399 | 63 | 2,928 ms | 3,759 ms | **−831 ms** |
| ≥400 | 39 | 2,939 ms | 4,988 ms | **−2,049 ms** |

只有「会触发 256 帧降采样」的长片段收益显著、短片段几乎不变，与「只对进入窗口的帧做姿态提取」的语义一致。
这同时解释了组合运行 **e2e mean 2.888 s 小于 P50 3.184 s**：73 条短片约 1.9 s，把左偏分布的均值拉低——引用时 mean 与 P50/P95 要一起给。

## 8. 证据索引

| 内容 | 路径 |
|---|---|
| 数据审计 | `artifacts/cslnews-eval/audit/dataset-audit.md`、`dataset-audit.json` |
| 项目集成权重逐样本预测（379 条，已暂停） | `artifacts/cslnews-eval/run-cslnews-20261025/predictions.jsonl`、`run-protocol.json` |
| stage1 + 上游 pose 全量 1694 | `artifacts/cslnews-eval/stage1-upstreampose-full-20261025/` |
| 四格消融其余两格 | `artifacts/cslnews-eval/stage1-cslnews-20261025/`、`pinned-upstreampose-20261025/` |
| 延迟归因微基准 | `artifacts/cslnews-eval/pose-bench.py` |
| 独立汇总与协议复核 | `artifacts/cslnews-eval/summary/summary-interim.md`（309 条中间版） |
| 评测脚本 | `scripts/evaluate_unisign_cslnews.py`、`scripts/evaluate_unisign_cslnews_upstream_pose.py`、`scripts/evaluate_unisign_cslnews_checkpoint.py`、`scripts/summarize_unisign_eval.py` |

## 9. 附录：论文结果复现状态与缺口清单

### 9.1 论文报告的到底是什么

依据 arXiv:2501.15187v2/v3（ICLR 2025）正文表格：

- **CSL-News 在论文里只是预训练语料**（Table 1：1,985 小时、5K 词表、TV 来源），
  **论文没有任何 CSL-News 翻译指标**。因此"用 CSL-News 复现论文结果"这个命题不成立。
- 论文真正可对照的下游结果：

| 论文表格 | 任务/数据 | Uni-Sign pose-only | Uni-Sign RGB+pose |
|---|---|---|---|
| Table 8 | CSL-Daily SLT（test） | BLEU-4 **25.61** / ROUGE **54.92** | 26.36 / 56.51 |
| Table 8 | CSL-Daily SLT（dev） | 25.27 / 54.34 | 26.25 / 56.03 |
| Table 7 | CSL-Daily CSLR（test WER） | 27.4 | 26.0 |
| Table 9 | How2Sign SLT（test） | 14.5 / 34.3 | 14.9 / 36.0 |
| Table 9 | OpenASL SLT（test） | 22.67 / 42.77 | 23.14 / 43.22 |
| Table 10 | MSASL1000 ISLR Top-1 | 77.88（P-I）/ 76.55（P-C） | — |

### 9.2 本机能核到哪一步

| 项 | 状态 | 证据 |
|---|---|---|
| 论文**指标口径** | **已对齐** | 用上游 1,176 条 test 预测+参考、按论文口径（`sacrebleu` tokenize=13a）重算：BLEU-4 **25.70**（论文 25.61，差 0.09）；ROUGE-L F1 53.59（论文 54.92，差 1.33，ROUGE 实现不同） |
| 本机**重跑 CSL-Daily test** | **已完成（2026-10-25）** | 用作者发布的 pose 归档跑 1,176 条：BLEU-4 **25.55** vs 论文 25.61；87.5% 预测与上游快照逐字相同。详见 §11 |
| CSL-News 对照论文 | **不适用** | 论文无 CSL-News 指标 |
| 预训练复现（stage1/2） | **未做** | `pre_training.py` 不在固定提交源码副本（README 指向 Issue #15）；本机 CSL-News 仅 1,694 条 = 3:56:03，占论文 1,985 小时的 **0.2%** |
| 下游微调复现（stage3） | **未做** | 无 CSL-Daily 训练数据；`fine_tuning.py` 可读但未运行 |

**结论：论文结果目前只有"指标口径"被复现，模型推理没有本机重跑，因此不构成复现。**

### 9.3 缺什么（按优先级）

1. ~~**CSL-Daily 数据集**~~ → **已部分补齐（2026-10-25）**：已下载 `csl_daily_pose_format.zip`（4.59 GB）并完成 test 集复现（§11）。仍缺**视频**（用于本地重提取与端到端复现）与官方 `labels.*`（参考句已从上游 `test_tmp_refs.txt` 获得）。
2. **与发布脚本一致的姿态前端**：`demo/pose_extraction.py` 用 rtmlib `Wholebody`（默认 lightweight）+ `x/W, y/H` 归一化；本地是同一模型但用 isotropic，已实测差 +6 BLEU（§10）。
3. **评测协议固定**：上游 eval 用 `random.sample` 抽 256 帧（**随机**）、`sacrebleu` 13a、ROUGE 用 `rouge` 包、beam=4；本仓库是确定性 linspace 采样 + 自有指标实现。→ 逐位复现不现实，合理目标是 **±0.5 BLEU** 内。
4. **训练代码**：`pre_training.py` 缺失 → 无法从零复现预训练，只能复现下游评测。
5. **数据规模**：CSL-News 本地覆盖 0.2%，无法复现预训练阶段。

### 9.4 最短复现路径（**已于 2026-10-25 执行，见 §11**）

1. 下载 `csl_daily_pose_format.zip`（4.38 GB）；
2. 从上游 `test_tmp_refs.txt` 提取 1,176 个 `S000020_P0004_T00` 形式的 ID，直接匹配 pkl 文件名；
3. 用本仓库管线跑这 1,176 条（姿态来自磁盘，模型推理约 0.4 s/条 → 约 8 分钟，无需 RTMPose）；
4. 按论文口径算 BLEU-4 / ROUGE，并与上游 `test_tmp_pres.txt` 逐条比对；
5. 判定：BLEU-4 落在 25.1–26.1 且预测一致率高 → 视为复现；否则按「帧采样 / 分词 / 前端 / dtype」逐项定位。

**已知风险**：CSL-Daily 的 pkl 含 `start/end` 裁剪字段（`datasets.py:464-470`），必须按上游 `S2T_Dataset.load_pose` 处理，否则帧范围会整体错位。

## 10. 姿态前端补齐（2026-10-25 追加）

### 10.1 官方前端其实就是 rtmlib，差异在归一化

- 上游 `demo/pose_extraction.py` 导入的是 `from rtmlib import Wholebody`，默认 `--mode lightweight`——
  **与本地运行时完全同一个模型**（`rtmw-dw-l-m_simcc-cocktail14_270e-256x192` + `yolox_tiny`）。
- 但它保存时做的是 `keypoints / np.array([W, H])`，即 **x/W、y/H（各轴分别归一化）**；
  本地运行时默认 `isotropic_short_side`（两轴同除短边）。**这才是 +7.9 BLEU 差距的来源，不是模型。**
- 身份核对（5 条片段 × 40 帧，`frontend-identity-check.json`）：本地提取 vs 上游 pkl×(W,H)，
  78–82% 关键点差 <1 px、87–91% <2 px，均值 2–3 px → 同一估计器，仅少量检测框抖动。

### 10.2 四格验证（同一批 300 条、同一解码、同一帧采样）

| 权重 | 姿态来源 / 归一化 | BLEU-zh | chrF | ROUGE-L | 编辑相似度 | P95 | ≥0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| pinned（CSL-Daily） | 本地 rtmlib / isotropic | 1.27 | 2.95 | 0.133 | 7.84% | 0.146 | 0 |
| pinned | 本地 rtmlib / **wh** | 1.03 | 2.78 | 0.131 | 7.85% | 0.150 | 0 |
| pinned | 上游 pkl / isotropic | 1.15 | 2.90 | 0.134 | 8.00% | 0.162 | 0 |
| stage1（CSL-News） | 本地 rtmlib / isotropic | 15.67 | 14.15 | 0.295 | 19.95% | 0.527 | 21 |
| stage1 | 本地 rtmlib / **wh** | **22.02** | 19.37 | 0.369 | **27.12%** | 0.714 | 52 |
| stage1 | 上游 pkl / isotropic | 17.75 | 15.92 | 0.326 | 22.28% | 0.627 | 31 |
| stage1 | 上游 pkl / **wh** | **23.53** | 20.63 | 0.386 | **28.45%** | 0.715 | 53 |

**全量 1,694 条复核（同权重、同片段，只换姿态来源）**：

| 权重 | 姿态来源 | BLEU-zh | chrF | ROUGE-L | 编辑相似度 | ≥0.5 | exact |
|---|---|---:|---:|---:|---:|---:|---:|
| stage1 | 上游 pose pkl | **23.76** | 20.92 | 0.4116 | **30.07%** | 319 | 8 |
| stage1 | 本地 rtmlib + wh | **21.69** | 19.12 | 0.3849 | **27.72%** | 272 | 6 |
| 差值 | — | **−2.07** | −1.80 | −0.027 | **−2.36 pp** | −47 | −2 |

→ 用本地提取替代作者发布的 pose 文件，全量代价只有 **约 2 BLEU / 2.4 个编辑相似度点**；配对 bootstrap（1,694 对，10,000 次）给出 Δ 编辑相似度 **−0.0236，95% CI [−0.0287, −0.0184]**，539 胜 / 822 负 / 333 平，CI 不含 0 —— 这个差距是真实的，不是采样噪声。

### 10.3 归因

| 因素 | 效果（stage1，300 条） |
|---|---|
| **归一化** isotropic → wh（本地估计器） | BLEU **+6.35**、编辑相似度 **+7.17 pp** |
| 归一化 isotropic → wh（上游估计器） | BLEU +5.78、编辑相似度 +6.17 pp |
| **估计器** 本地 → 上游（isotropic） | BLEU +2.08、编辑相似度 +2.33 pp |
| 估计器 本地 → 上游（wh） | BLEU +1.51、编辑相似度 +1.33 pp |
| pinned 权重下以上全部因素 | BLEU 1.03–1.27，几乎无变化 |

**结论：前端要补的是归一化，不是换模型。** 把 `pose_normalization` 设为 `wh` 后，本地管线
（自己的 rtmlib 提取 + 自己的缓存）在这批数据上达到 **BLEU-zh 22.02 / 编辑相似度 27.12%**，
与"直接用作者发布 pose"的 23.53 只差 1.51 BLEU——**这个残差才是估计器版本差异的真实代价**。

### 10.4 落地建议

1. 对 CSL-News 域（4:3 左右的新闻片段 + stage1 权重）：`pose_normalization = "wh"`，这是作者脚本的口径。
2. 对 CE-CSL 等宽屏数据：先不要改。历史结论（isotropic 4.72→7.99）是在 CSL-Daily 权重 + 自建提取上得到的，
   与本节数据域不同；要判定必须检查 CSL-Daily 官方 pose 的归一化（`csl_daily_pose_format.zip`，4.38 GB）。
3. 归一化应随"训练数据生产者"记录，而不是全局固定：同一权重换归一化可差 6 BLEU，属于协议字段，必须进 manifest 与评测协议。






## 11. CSL-Daily in-domain 复现（2026-10-25 完成）

### 11.1 做法

- 下载作者发布的 `csl_daily_pose_format.zip`（**4.59 GB，20,654 个 pkl**），按上游 `test_tmp_refs.txt`
  的 1,176 个 sample ID（`S000020_P0004_T00` 形式）精确抽取 test pose（556.5 MB）。
- 用本仓库运行时（pinned `csl_daily_pose_only_slt`、bf16、beam=4、max_new_tokens=100）在 1,176 条上推理，
  姿态来自磁盘，**不需要 RTMPose**，模型侧约 8 分钟。
- 两个关键细节：① pkl 含 `start/end` 裁剪字段（例如 170 帧文件只取 `[92,150)`），已按上游
  `S2T_Dataset.load_pose` 处理；② pkl 的 `w_h` 字段显示 CSL-Daily 视频是**正方形**
  （1280×1280 / 1080×1080），因此在 CSL-Daily 上 `wh` 与 `isotropic` 两种归一化等价。

### 11.2 结果

| 项 | BLEU-4 | ROUGE |
|---|---:|---:|
| 论文 Table 8（pose-only, test） | 25.61 | 54.92 |
| 上游快照重算（1,176 条） | 25.70 | 53.59* |
| **本机重跑（1,176 条）** | **25.55** | 53.41* |
| 差值（本机 − 论文） | **−0.06** | — |

\* ROUGE 用字符级 LCS 近似，与上游 `rouge` 包实现不同，系统性低约 1.3；**BLEU 才是可比值**。

- 与上游预测逐条比对：**1,029 / 1,176（87.5%）逐字相同**；其余差异来自帧采样
  （上游 `random.sample` vs 本仓库确定性 `linspace`）。
- 未做格式修正时，本机平均字符编辑相似度 42.04%。

### 11.3 一个必须知道的协议陷阱

上游 `fine_tuning.py:282-284` 在算指标前对预测做了
`' '.join(list(r.replace(" ", "").replace("\n", "")))`（注释写 "fix mt5 tokenizer bug"），
即**去空格后逐字插入空格**；参考句本身也是逐字空格分隔。

不做这一步、直接用 `sacrebleu` 13a 计算连续中文预测时，BLEU-4 会从 25.55 掉到 **0.0005**
（每句退化为 1 个 token，BP 0.0026）——**看起来像"模型完全失败"，实际只是格式不同**。
chrF 22.39 与**字符级** ROUGE 0.534 保持正常，正是识别这类假失败的关键旁证；注意**token 级 ROUGE 会一起崩到 0**，只有字符级 ROUGE 不受影响（独立复核员 2026-10-25 指出）。

### 11.4 结论

- **CSL-Daily SLT（pose-only）本地复现成立**：BLEU-4 25.55 vs 论文 25.61（差 0.06），
  87.5% 预测与作者快照逐字一致。
- 复现边界：用的是**作者发布的 pose 文件**（不是本地 RTMPose 重提取），帧采样为确定性而非随机；
  因此不能声称"从视频端到端复现"，但足以证明模型构造、checkpoint 加载、bf16 推理、beam 设置与
  `prepare_pose_window` 预处理与上游一致。
- 对 CSL-News 的意义：既然同一套运行时能在 CSL-Daily 上复现论文，那么 CSL-News 上的低分
  （pinned 权重 1.27）只能归因于**权重域不匹配**，而不是管线缺陷。













## 12. 「CSL-News 低分是不是特征处理不一致造成的？」（2026-10-25 追加）

**结论：不是。** 四条独立证据。

### 12.1 两个发布 pose 归档的实际格式（逐字段核对）

| 项 | CSL-News pose（`CSL-News_pose/archive_001`） | CSL-Daily pose（`csl_daily_pose_format.zip`） |
|---|---|---|
| keys | `keypoints`, `scores`, `w_h` | `keypoints`, `scores`, `w_h`, **`start`**, **`end`** |
| `keypoints` | (T,1,133,2) float64 | (T,1,133,2) float64 |
| `w_h` | **[204, 170]**（非方形） | **[1280,1280] / [1080,1080]**（方形） |
| 归一化 | **x/W, y/H** | **x/W, y/H**（方形 ⇒ 与 isotropic 等价） |
| 裁剪 | 无（片段本身就是裁好的） | 有，如 170 帧中只取 `[92,150)` |
| 关键点约定 | 133 点；手/脸簇结构正常（簇内最大偏移 0.08–0.11） | 同左（0.078–0.087） |

→ 两者**同一个 133 点约定、同一个 `x/W, y/H` 归一化**，真正的差异只有「视频是不是方形」，
因此 CSL-News 必须用 `wh`、CSL-Daily 上二者等价。这已在本仓库按域处理（§10）。

### 12.2 特征固定、只换权重 → 差 22 BLEU

用**作者自己的 CSL-News pose 文件**（即完全保持其原始特征处理），仅更换 checkpoint：

| 权重 | BLEU-zh | 编辑相似度 |
|---|---:|---:|
| `csl_daily_pose_only_slt`（项目集成） | **1.27** | 7.90% |
| `csl_stage1_weight` | **23.76** | 30.07% |

特征处理在两边完全相同，差距只能来自权重域。这是最直接的排除。

### 12.3 本机特征管线在 in-domain 上被验证正确

用作者发布的 CSL-Daily pose 复现论文：**BLEU-4 25.55 vs 论文 25.61**，87.5% 预测与作者快照逐字相同（§11）。
说明 `load_part_kp` 移植、`prepare_pose_window`、置信度通道与推理链都是对的。

### 12.4 特征处理误差的量级上界：2–6 BLEU

在 CSL-Daily in-domain 300 条上故意错配 x 缩放（模拟归一化做错）：

| x 缩放 | BLEU-4 | 编辑相似度 | 相对正确配置 |
|---|---:|---:|---:|
| ×1.0（正确，as-is） | **36.51** | **54.05%** | — |
| ×0.75 | 34.57 | 52.68% | −1.94 BLEU |
| ×0.5625（按 16:9 处理） | 30.45 | 48.69% | **−6.06 BLEU** |

对照：CSL-News 上 pinned 与 stage1 的差距是 **22 BLEU**。**任何合理量级的特征处理错误都解释不了 22 BLEU。**

### 12.5 仍然存在的、真实但很小的特征差异

- 本机提取 vs 作者 pose（同权重、同片段、全量 1,694）：23.76 → 21.69，**−2.07 BLEU**（§10），来自估计器实现差异（78–82% 关键点差 <1 px）与帧采样方式。
- 这条残差与「权重域」是两件事，不应混为一谈。

## 13. 目标域（日常手语）CE-CSL 与 stride=5 第二数据集验收（2026-10-25 追加）

产品口径已收敛为「动作结束后识别」，目标域为日常手语 → 权重用 pinned `csl_daily_pose_only_slt`，
归一化 `isotropic_short_side`，方向用 dev 校准映射（B/E/G/I/L = mirror）。

### 13.1 CE-CSL test 500（stride=1，方向映射）

| 指标 | 数值 |
|---|---:|
| BLEU-zh | **12.3019** |
| chrF / ROUGE-L | 13.3098 / 0.4038 |
| 平均字符编辑相似度 | **30.63%** |
| exact match | 1 / 500 |
| 空预测 / 失败 | 0 / 0 |
| 端到端（mean / P50 / P95 / max） | **5.55 / 5.33 / 8.41 / 10.47 s** |
| 姿态提取 mean | 5,213 ms |
| 模型 mean | 307 ms |

该结果与历史 `repair-20261024/test-locked-directions` 的 12.3019 **逐位一致**，确认代码路径未漂移。

### 13.2 stride=5（重新校准的方向映射 + 新姿态协议）

方向映射绑定姿态协议：`pose_detect_stride` 一旦进入协议，旧映射会被拒绝；因此用 `--pose-detect-stride 5`
重跑了 dev 校准（36 校准 + 24 验证），新映射方向与旧映射**完全一致**（B/E/G/I/L=mirror），随后跑 test 500。

| 指标 | stride=1 | stride=5 | 差值 |
|---|---:|---:|---:|
| BLEU-zh | 12.3019 | 12.1470 | −0.155 |
| chrF | 13.3098 | 13.2773 | −0.033 |
| ROUGE-L | 0.4038 | 0.4095 | +0.006 |
| 平均字符编辑相似度 | 30.63% | **31.32%** | **+0.69 pp** |
| exact | 1 | 2 | +1 |
| 端到端 mean | 5.55 s | **3.42 s** | **1.62×** |
| 端到端 P50 / P95 | 5.33 / 8.41 s | 3.31 / 5.15 s | — |
| 姿态提取 mean | 5,213 ms | **3,077 ms** | **1.69×** |
| 模型 mean | 307 ms | 310 ms | 不变 |

配对检验（500 条逐条编辑相似度差）：**+0.0068，95% CI [−0.0011, +0.0152]**（含 0，不显著），
胜/负/平 **91/88/321**，285/500 预测逐字相同。

→ **第二个数据集独立复现了 CSL-News 上的结论**：`pose_detect_stride=5` 在静态单人片段上无显著精度损失，
端到端约 1.6×。至此该优化在 300 条（CSL-News）+ 500 条（CE-CSL）两个数据集上均通过验收。

### 13.3 三数据集汇总（同一权重、同一 post-hoc 协议）

| 数据集 | 角色 | BLEU-zh | 编辑相似度 | e2e P50 | e2e P95 |
|---|---|---|---:|---:|---:|---:|
| CE-CSL test 500（日常） | **目标域** | **12.30 / 12.15*** | 30.63% / 31.32%* | 5.33 / 3.31 s* | 8.41 / 5.15 s* |
| CSL-News 1,694（新闻） | 跨域对照 | 0.74 | 7.52% | — | — |
| CSL-Daily test 1,176（域内证明） | 管线正确性 | 25.55 | — | — | — |

\* 斜杠后为 `pose_detect_stride=5` 的数值。
