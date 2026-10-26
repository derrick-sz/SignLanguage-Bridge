# E04 · GPU 性能 / 显存 / 串并行硬件报告（measured）

> **状态**：`PASS`（所有计划 phase 均有实测数据；失败尝试保留在 run 目录
> `failed_attempt_*/`，未用于任何数字）
> **run_id**：`e04-gpu-20261026T2202Z`
> **日期/负责人**：2026-10-26 / perf-streamer
> **数据**：`experiments/p1/runs/e04-gpu-20261026T2202Z/`
> （`perf_trace.jsonl` 706 KB、`metrics.json`、`metrics_summary.md`、`phase-*.json`、
> `device-*.jsonl`、`parallel_vs_serial.csv`、`manifest.json`）

## A. 实验基本信息

| 字段 | 值 |
|---|---|
| 实验编号 | E04（GPU 性能、显存、双模型串行/并行） |
| 状态 | `PASS`；冷启动视频路径有 2 次分配类失败已按坑点 §7 重试成功（详见 §E） |
| 实验假设 | H1 单 GPU 串行可行；H2 两进程并行不增加总吞吐/会推高延迟与显存；H3 Uni-Sign 延迟以姿态前端为主 |
| 样本范围 | Pilot-50 中按 signer 分层确定性抽取：warm/load 5 条（5 signer），serial/parallel 5 条；样本 ID 记录在 `phase-*.json` 与 trace 行内，未使用 GT |
| 执行方式 | **单进程串行**（一次只有一个模型进程），每 phase 独立新进程；并行 phase 是唯一的两进程场景；见 §C 与 §E 的 GPU 独占说明 |

## B. 硬件 / 环境 / 权重身份

| 字段 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti，16311 MiB（`torch` 报 16283.4 MB），算力 12.0 |
| 驱动 / CUDA | 616.92 / CUDA 12.8（Torch 2.9.1+cu128） |
| 测量前整卡占用 | **2872–2897 MiB / 10% util**（`gpu-baseline-before-e04.json`；`nvidia-smi --query-compute-apps` 只有 explorer/chrome/WeChat 等桌面进程，**无其它 CUDA 计算负载**） |
| 门限 | phase 启动时整卡 `memory_used > 4500 MB` 则拒绝启动；本次所有 phase 均未触发。门限高于桌面基线（~2.9 GB）以避免桌面波动中止，但低于任何真实模型进程（≥1 GB 新增） |
| CPU / RAM | i5-14600KF（14C/20T）/ 31.84 GB；测量期间 host 内存偏紧（见 §E），并行 gate 记录 12.3 GB available（通过 7 GB 门限） |
| Python / 精度 | `.conda-env` Python 3.10.22；Uni-Sign bf16，SLRT float32；batch=1 |
| Docker | 无（本机 conda 环境） |
| 输入 | `trimmed/<sample>.mp4`（Uni video、SLRT）、官方 `pose-format/<sample>.pkl`（Uni pose）；h264/mp4v，30/29.97 FPS |
| Uni-Sign 权重 | `csl_daily_pose_only_slt.pth`，1,186,925,112 B，SHA256 `73bea87a…1a6c`；source commit `eed438b`，模型 revision `182913ca…`，CC-BY-NC-4.0 |
| SLRT 权重 | `cslr_best.ckpt`，SHA256 `e9fdd3a6…cbd4d`；词表 `0c4984f0…4c76`；938 键严格加载，2001 类 blank=0 |
| 环境/SHA 记录 | run `manifest.json`：git SHA/dirty tree、`environment`、`model_assets`、命令列表 |

## C. 测量口径

- CPU/总耗时 `time.perf_counter()`；GPU 段在 runtime 内部以 `torch.cuda.synchronize()`
  界定（Uni-Sign `TorchUniSignRuntime.infer`；SLRT `SLRTRuntime.predict`），
  逐行 `timing_method` 记录。
- 分段：`video_decode_ms` / `pose_ms` / `forward_ms` / `text_decode_ms` / `queue_ms` / `e2e_ms`；
  缺失写 `null`（Uni pose 无 video decode；SLRT 无文本头）。
  - Uni video 的 `video_decode_ms` 由“包装 `_extract_cached` 后从 `pose_extract_ms` 扣除”得到（解码与逐帧 RTMPose 交错，运行时无原生拆分；报告按插桩值引用）。
  - SLRT `pose_ms` 是 hrnet-npz 数组索引 + 前端开销（≈4–6 ms，如实记录）；S3D 前向在 `forward_ms`。
- RTF = `e2e_ms/1000 ÷ 容器 FPS×帧数`；视频时长由容器 FPS 计算，不用帧数硬推。
- P50/P95 仅对成功样本；同时报失败/OOM（本次计时阶段 0 失败 / 0 OOM）。
- 显存：`torch.cuda.max_memory_allocated/reserved`（本进程 allocator）+ `nvidia-smi`
  整卡采样（含桌面进程；scope 在 JSON 中注明）。
- 冷启动 = 3 次全新进程（导入+资产校验+模型加载+1 次推理）；热启动 = 预热 2 次后
  30 次计时；连续负载 = 2 分钟不间断。**注意**：负载阶段与 warm 阶段的绝对耗时受
  其它 CPU 进程/系统内存压力影响（见 §E），同一指标跨 phase 有可见波动。

## D. 结果（全部 measured）

### D.1 冷启动（每次全新进程；`phase-cold-*.json`）

| 模型 | 成功/尝试 | 模型加载 P50 / P95 s | 进程墙钟 P50 s | 首次推理 e2e P50 ms | 加载后驻留 alloc MB | 进程峰值 alloc MB | reserved MB |
|---|---|---:|---:|---:|---:|---:|---:|
| unisign_pose | 3/3 | 17.66 / 18.74 | 24.21 | 303 | 1149 | 1206 | 2738 |
| unisign_video | 3/3（另 2 次分配失败重试） | 26.62 / 65.06 | 30.52 | 2950 | 1149 | 1206 | 2738 |
| slrt_hrnet_npz | 3/3 | 3.14 / 3.15 | 13.55 | 9559 | 170 | 520 | 600 |

- Uni-Sign 冷加载包含运行时加载与一次 warmup 前向（pose runtime 的
  `cold_first_forward_ms` = **2079 ms**，模型加载本身 17.0 s）；video 路径另含
  RTMPose ONNX 会话装载（`model_warmup_ms` warm 阶段实测 14.8 s）。
- video 冷启动 2 次 `CUDA error: out of memory`（mT5 warmup 内）发生在宿主 commit
  压力峰值，重试后成功；`spawn_attempts` 已写入 trace，**不计模型失败**（§E）。
- SLRT 冷进程墙钟 13.5 s 里只有 3.1 s 是权重加载，其余是 torch/导入/首个视频前端。

### D.2 热启动（预热 2 次后 30 次计时）与连续负载（2 分钟）

| 模型/阶段 | n | 失败/OOM | e2e P50 / P95 ms | forward P50 ms | pose P50 ms | video decode P50 ms | text decode P50 ms | 平均 RTF | 吞吐 样本/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| unisign_pose / warm | 30 | 0 / 0 | **277.6 / 344.3** | 38.986 | 0.675 | — | 233.7 | **0.078** | 3.27 |
| unisign_video / warm | 30 | 0 / 0 | **2690.4 / 3283.9** | 35.896 | **2156.4** | 290 | 200.5 | **0.748** | 0.339 |
| slrt_hrnet_npz / warm | 30 | 0 / 0 | **9865.1 / 11189.7** | 3701.4 | 6.144 | 432 | — | **2.534** | 0.099 |
| unisign_pose / load 120.0 s | 419 | 0 / 0 | 276.3 / 345.2 | 37.1 | 0.6 | — | 237.9 | 0.077 | 3.49 |
| unisign_video / load 120.6 s | 42 | 0 / 0 | 2697 / 3225 | 32.3 | 2140 | 289 | 200.1 | 0.744 | 0.348 |
| slrt_hrnet_npz / load 124.3 s | 13 | 0 / 0 | 8038 / 10322 | 3329 | 5.9 | 361 | — | 2.190 | 0.105 |

**分段解读（measured，不是估计）**

- **Uni-Sign pose（模型-only）**：e2e 276 ms = 姿态窗口准备 ~0.7 ms + encoder 前向
  38.9 ms + beam 生成/分词 233.7 ms；RTF 0.078（按 3.5 s 平均视频计），吞吐 3.3 条/s。
- **Uni-Sign video（端到端）**：e2e 2679 ms 中 **pose_ms ~2152 ms（80%）**、
  video decode ~290 ms、forward 35.8 ms、文本生成 ~200 ms；RTF 0.748。
  与坑点 §7「延迟 93% 在姿态前端」同向（本批平均视频更短，比例 80%）。
- **SLRT hrnet-npz**：e2e 9775 ms / 109 窗口 ≈ 90 ms/窗；其中 S3D 前向 3690 ms、
  视频解码 432 ms，**其余 ~5.6 s 是每窗口 CPU 热力图预处理（170→143→112 的
  Gaussian 采样）+ Python 循环**（`timings` 中 `preprocess_ms`/`window_total_ms`），
  不是模型 forward；RTF 2.53（比视频实时慢 2.5×）。这是单进程文件路径的真实成本，
  优化点在窗口预处理而不是 S3D。
- 连续负载 2 分钟内吞吐与 warm 一致，无 OOM/失败、无显存增长（见 D.3 峰值），
  说明 2 分钟观察窗内无泄漏证据。

### D.3 显存（峰值）

| 场景 | 进程 torch peak alloc MB | reserved MB | 整卡 used 峰值 MB（采样） | 备注 |
|---|---:|---:|---:|---|
| unisign_pose warm/load | 1206 / 1202 | 2738 | 5641 / 5644 | 整卡含桌面基线 ~2.4 GB |
| unisign_video warm/load | 1206 / 1202 | 2738 | 5659 / 5656 | ORT RTMPose 会话另占显存（整卡口径） |
| slrt warm/load | 374 / 374 | 600 | 3729 / 3667 | 权重 499 MB fp32 |
| serial（两模型先后） | 各模型峰值同上 | — | 未采样（serial 无 device sampler；以 parallel 为上限参考） | |
| **parallel（两模型同时）** | video 1206 + slrt 374 | — | **6451** | 均值 5629 MB；GPU util 均值 27.8%、峰值 85% |
| 冷启动 slrt | 520 | 600 | — | 首次分配含 CUDA context |
| 桌面基线（无模型） | — | — | 2872–2897 | 无其它 CUDA compute app |

### D.4 串行 vs 两进程并行（同一 5 条样本；`parallel_vs_serial.csv`）

| 模式 | Uni video batch s（5 条） | SLRT batch s（5 条） | 总 batch s | 整卡峰值 used MB | GPU util 均值/峰值 | 失败/OOM |
|---|---:|---:|---:|---:|---|---|
| serial（先后） | 11.813 | 28.463 | **40.276** | 未采样 | 未采样 | 0/0 |
| parallel（两进程同时） | 21.330（**1.81× 变慢**） | 33.452（**1.18× 变慢**） | **34.100（compute wall）** | **6451** | 27.8% / 85% | 0/0 |

- 串行 batch 和 40.28 s；并行 compute wall 34.10 s → **serial_sum / parallel_wall =
  1.181**，即两进程重叠使“最后一条完成”的时间比两批串行之和短约 **15%**。
- 但**每个模型的单样本延迟都变差**：Uni video 2.36 s→4.27 s/条，SLRT 5.69 s→6.69 s/条；
  整卡峰值显存 6451 MB（两模型常驻 + 桌面基线），GPU util 均值仅 27.8%——说明
  并行没有提高 GPU 计算效率，只是把两段各自低利用率的负载叠在一起。
- 结论（H2 部分成立/部分不成立）：**并行降低“总墙钟”但不降低单条延迟，也不提高
  GPU 利用率，反而抬高显存和两条路径的 P95**。对“先出 Uni-Sign、再出 SLRT”的
  级联产品语义，串行是默认；并行只适合离线批量。5 条样本的批次很小，只作方向性
  证据，不作 SLA。

## E. 失败 / OOM / 环境事件（如实记录，未计入模型失败）

| 事件 | 证据 | 处理 |
|---|---|---|
| 首次编排尝试：`all` 把 phase 专属 argv 传给所有子命令 → 9 个 cold/warm/load 全部 argparse exit 2；随后串行 phase 在宿主内存耗尽时子进程 `CUDA error: unknown error`，pwsh 宿主退出 OOM | `failed_attempt_1/orchestrator.log`、`failed_attempt_1/serial-unisign_video-summary.json`、`failed_attempt_1/README.md` | 修复 argv 构造；杀掉 3 个孤儿 python 进程；改为逐 phase 独立进程重跑；所有数字来自第二轮 |
| 第二轮 cold video：attempt 0/1 `CUDA error: out of memory`（mT5 warmup 内，GPU 空闲 >13 GB；当时 Windows committed 55.7/65.2 GB，free RAM ~4.5 GB，ORT 报 `fatal: Memory allocation failure`） | `failed_attempt_2/README.md`、`dropped_trace_rows.jsonl`、`cold-unisign_video-*.log` | 判定为宿主内存/commit 压力导致的分配失败（非模型缺陷）；加入“分配类错误重试 2 次、每次间隔 25 s”并在 trace 记 `spawn_attempts`；重跑 3/3 成功 |
| 计时阶段失败/OOM | `metrics.json` | **0 失败 / 0 OOM**（cold 3 模型、warm 3×30、load 3×2 min、serial/parallel 各 5 条全部成功） |

> 32 GB Windows 上并发的其它桌面/agent 进程会造成 commit 压力并产生 `CUDA error:
> out of memory` 假失败；必须重试后再判定模型失败（坑点 §7）。本报告所有延迟数字
> 都是在“无其它 CUDA compute app、但有桌面/agent CPU 负载”的条件下测得，**不是**
> 干净机房独占数字。

## F. 结论与工程选择

1. **H1 成立**：单 GPU、单进程串行可稳定跑完三模型，冷启动 3.1–26.6 s、热启动
   0.34–3.3 条/s，全部 0 OOM（重试后）。
2. **H3 成立**：Uni-Sign 端到端 80% 时间是姿态前端（pose 2152 ms vs forward 36 ms）；
   mT5 生成再慢一个量级也赶不上前端。SLRT 文件路径 90 ms/窗中 S3D 前向仅 ~34 ms/窗。
3. **H2 修正**：两进程并行使总墙钟 −15%（40.28→34.10 s），但单模型延迟 +18%~+81%、
   整卡峰值显存 6.45 GB、GPU util 仍只有 27.8%。对实时/交互语义，**默认串行**；
   并行仅用于离线批处理，且需 6.5 GB 以上显存预算。
4. 选型建议：常驻 Uni-Sign（1.2 GB alloc / 2.7 GB reserved）+ SLRT（0.37 GB）
   的显存总和在 16 GB 卡上安全；CPU/RAM 才是本机瓶颈（commit 压力导致假 OOM），
   部署时应串行调度并限制并发进程数。
5. 与 E01 交叉核对：E01（`e01-pilot50-u1-unisign-video-20261026`）50 条视频端到端
   e2e P50 3.625 s / P95 5.336 s、pose mean 3,311 ms、model 444.8 ms；本 E04 的 5 条
   子集 e2e P50 2.679 s（样本更短 3.6 s vs 4.9 s），量级一致，差异来自样本/时长。

## G. 复现命令（真实执行）

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$run = 'e04-gpu-20261026T2202Z'; $dir = "experiments\p1\runs\$run"
# 每个 phase 独立新进程、单进程串行；GPU 门限 4500 MB（桌面基线 ~2.9 GB）
.\.conda-env\python.exe experiments\p1\e04\benchmark_p1.py cold --model <unisign_pose|unisign_video|slrt_hrnet_npz> --run-id $run --run-dir $dir --cold-repeats 3 --warmup 2 --max-foreign-vram-mb 4500
.\.conda-env\python.exe experiments\p1\e04\benchmark_p1.py warm --model <M> --run-id $run --run-dir $dir --warm-repeats 30 --samples 5 --warmup 2 --max-foreign-vram-mb 4500
.\.conda-env\python.exe experiments\p1\e04\benchmark_p1.py load --model <M> --run-id $run --run-dir $dir --load-minutes 2 --samples 5 --warmup 2 --max-foreign-vram-mb 4500
.\.conda-env\python.exe experiments\p1\e04\benchmark_p1.py serial   --run-id $run --run-dir $dir --samples 5 --warmup 1 --max-foreign-vram-mb 4500
.\.conda-env\python.exe experiments\p1\e04\benchmark_p1.py parallel --run-id $run --run-dir $dir --samples 5 --warmup 1 --max-foreign-vram-mb 4500 --min-available-ram-gb 7
.\.conda-env\python.exe experiments\p1\e04\benchmark_p1.py summarize --run-id $run --run-dir $dir
```

产物副本：本目录 `hardware_report.md`、`parallel_vs_serial.csv`、`perf_trace.jsonl`、
`metrics.json`、`metrics_summary.md`、`device_samples.jsonl`（复制自 run 目录）。
