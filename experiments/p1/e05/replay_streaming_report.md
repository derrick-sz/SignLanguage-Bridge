# task-8 · SLRT replay 首稳探针报告（真实墙钟流式指标）

> **状态**：`PASS`（两轮独立 replay 重复，均无失败；实测数字见下）
> **证据等级**：**按原时间轴回放的半真实证据**。`SLRTCaptureSession` 用
> `video_packets` 的源时间戳对帧做真实 pacing，但**不是摄像头采集**：没有相机曝光/
> 传输抖动、没有真人现场操作、没有桌面端渲染链路。**不得宣称“现场摄像头验收通过”。**
> **实验编号**：task-8（G2 阶段 2-C）；上游 E05 结论对齐对象：`future_frame_audit.md` §3、
> `streaming_ability_matrix.md` D.3
> **负责人/日期**：perf-streamer / 2026-10-26
> **run 目录**（`experiments/p1/e05/` 下，未写 `slbridge/`，未改源码）：
> - `replay-probe-20261027T0002Z`（文档内记为 **run A**；`run_id=replay-probe-20261026T232802Z`）
> - `replay-probe-runB-20261026T2333Z`（**run B**）
>
> 每轮：10 条 pilot trimmed mp4（1 条/signer，hrnet-npz）+ 1 条 warm-up（不计入统计）
> + 3 条压力片段（黑帧 / 镜像 / 两段拼接）。单进程串行；GPU 基线 2196–2595 MiB
> 桌面占用、无其它 CUDA compute 进程。

## 1. 方法（区分 available_time 与 wall delivery）

- 入口：`SLRTCaptureSession(video=<trimmed mp4>, pose_npz=<HRNet npz>)`，即 CLI
  `replay` 的同一实现（`slbridge/slrt/cli.py` replay 分支）；**未修改库**。
- 插桩（本脚本内只读包装，见 `replay_probe.py`）：
  - `event_sink`：捕获每个 partial/final/retracted 事件及其**墙钟**接收时刻
    （capture session 自带 `wall_elapsed_ms`，相对 session start）；
  - `runtime.predict` 包装：记录每个**窗口推理**开始/结束墙钟、anchor 帧、模型内部分段
    （`preprocess_ms`/`inference_ms`/`window_total_ms`）；
  - `stream.feed` / 捕获队列 `put` 包装：记录每帧**到达队列墙钟**与 worker 取走墙钟 →
    队首等待；`enabling_frame = anchor + 8` 的到达时刻用于“窗口可用 → 交付”延迟。
- `available_time` 是事件所需的**源时间戳**（`latest_required_input_timestamp`），
  与墙钟交付**分开保存**；本报告所有延迟均为墙钟，除特别标注。
- **首稳定义**（两种，全部报告）：
  1. `first_final_wall`：首个 `final` 事件的墙钟（decoder 的 final 是 append-only，
     后续 partial 不会撤回 final；这是“首个不再被改写的词”的可部署口径）；
  2. `first_stable_after_last_retraction`：最后一次 `retracted` 之后的首个 final
     （更保守；retraction 只发生在输入不连续/中断时的 active partial，不是 final 撤回）。
- RTF = replay 墙钟 / 视频源时长；吞吐 = `windows_processed / replay 墙钟`。
- 环境敏感：两轮重复的绝对耗时差异明显（见 §3），报告给出两轮全部数字，不取“最好轮”。

## 2. 与 E05 的 11 帧 lookahead 推导对齐（验证/修正）

E05 源码推导：中心帧 `c` 的 partial 需要（i）窗口右上下文 8 帧已到达，
（ii）投票 7 窗需要再等 3 个窗（stride=1）→ **11 帧已观测 lookahead ≈ 0.37 s @30fps**。

**实测验证**：
- `window_events.jsonl` 中最早出现 partial 的窗口为 `window_index=3, anchor=3`
  （4 个 pending 窗口才投第一票），其 `enabling_frame = anchor+8 = 11` —— 与推导一致；
- 但**首个非空 partial 常在 window_index 3–8**（早期窗口投票为 blank，不产生事件），
  因此“首个可见 partial”的墙钟 = 11 帧 lookahead（~0.37 s）+ 播放启动开销 + **队列积压**。
- 实测首 partial 在回放开始后 **P50 2.0–2.2 s**（run A/B），远大于 0.37 s —— 差额来自
  吞吐不足（见 §3），不是投票规则更慢。**结论：11 帧推导是算法下限，正确；但真实
  首字延迟由吞吐/队列决定，不能只报 0.37 s。**

## 3. 正样本结果（10 条 pilot，hrnet-npz，两轮重复）

| 指标 | run A P50 / P95 | run B P50 / P95 |
|---|---|---|
| replay 墙钟 (s) | 10.19 / 12.53 | 8.86 / 12.42 |
| 视频源时长 (s) | 3.87 / 4.95 | 3.87 / 4.95 |
| **RTF（墙钟/源时长）** | **2.58 / 3.07** | **2.23 / 2.96** |
| **首 partial 墙钟（自 session 起点）** | 2249 / 2547 ms | 2109 / 2446 ms |
| 首 partial − 首帧到达 | 2161 / 2418 ms | 2020 / 2331 ms |
| **首 final 墙钟** | 3918 / 8039 ms | 3727 / 7547 ms |
| **首 final − 首帧到达（≈首稳）** | **3873 / 7144 ms** | **3666 / 5110 ms** |
| 保守口径：最后 retraction 后首个 final | 9231 / 11614 ms | 6007 / 10478 ms |
| **吞吐（窗口/秒）** | 7.49 / 10.33 | 7.91 / 14.30 |
| 单窗口推理 P50（每视频 P50 的分布） | 87.5 ms | 57.1 ms |
| 队首→交付（窗口推理开始→事件发布） | 52.4 / 101.9 ms | 49.7 / 71.8 ms |
| 帧到达→worker 取走（队首前等待） | 1733 / 2095 ms | 1661 / 2037 ms |
| partial 改写率（死胡同 partial 占比） | 0.225 / 0.427 | 0.189 / 0.547 |
| retraction 总数（10 条合计） | 24 | 19 |
| 丢帧总数（capture 队列丢帧，= stream 不连续跳帧） | 61 | 52 |
| 空输出 | 0/10 | 0/10 |

**核心结论（可部署性）**

1. **不能在原时间轴实时运行**：两轮 RTF P50 2.2–2.6（P95 3.0）；吞吐仅需 ~30 窗口/s
   才能跟上 30 fps，实测 7.5–7.9 窗口/s（窗口推理 57–88 ms > 33 ms 帧间隔）。
2. **丢帧是结构性后果**：队列 32 帧仅约 1 s 缓冲，worker 追不上时 capture 队列丢帧
   （每轮 52–61 帧），`stream.feed` 检测到帧号跳变即 `interrupt('input_discontinuity')`
   → 两轮共 19–24 次 retraction。**retraction 主要不是模型投票撤销，而是输入不连续**。
3. **首稳不早于视频结束**：首 final token 出现在回放开始后 P50 3.7–3.9 s，而视频 P50
   3.87 s —— 即**第一个稳定的词只能在句末附近交付**；P95 5.1–7.1 s。实时字幕要求
   “句内早出词”，本路径不满足。
4. **队首→交付本身不慢**（P50 50 ms），慢在**队首前等待**（帧到达→worker 取走 P50
   1.7 s）与逐窗吞吐；优化应针对窗口预处理/批处理/降 FPS，而不是 decoder。
5. partial 改写率中位 0.19–0.23、P95 0.43–0.55：即使 partial 能出，仍有一半左右在
   P95 样本上会被后续改写；与 E05 “SLRT 原生 partial 可改写”一致。
6. **warm-up 会话明显更差**（首 session：queue_dropped 58–59，首 partial ~5 s）：
   首次 session 的前端加载 + 首窗 CUDA kernel 使队列瞬间爆满；产品需预热常驻。

**逐条 gloss（两轮）**：run B 多数恢复出可读词序列（如 `['爸爸','我','惭愧']`、
`['驾驶','注意','驾驶']`），但伴随丢词/重复；这是跨域 CE-CSL/CSL-Daily 权重的
识别结果，本探针**不评估准确率**（无 GT 参与），仅用于流式交付指标。

## 4. 压力/负样本（rtmpose-experimental 前端；黑帧/镜像片段无法与官方 HRNet NPZ 配对）

| 片段 | 轮次 | 帧数 | 窗口数 | 质量拒绝帧 | retraction | 输出 | 墙钟/RTF | 判读 |
|---|---|---:|---:|---:|---:|---|---|---|
| 黑帧（无手语） | A | 108 | **0** | **108/108** | 0 | 空 | 5.35 s / 1.50 | ✅ 无人门控有效：全帧拒绝、0 窗、无输出（与坑点 §10 的黑帧负样本结论一致） |
| 黑帧（无手语） | B | 108 | **0** | **108/108** | 0 | 空 | 9.82 s / 2.75 | 同上 |
| 水平镜像 | A | 108 | 50 | 0 | 1 | 空（无 final） | 7.13 s / 2.00 | ❌ 镜像**未被门控**（0 拒绝）；本次无 final、输出不稳定 |
| 水平镜像 | B | 108 | 53 | 0 | 1 | `['累']` | 12.24 s / 3.43 | ❌ 镜像未被识别为无效输入，仍会输出词 |
| 两段拼接（串句） | A | 202 | 63 | 0 | 1 | `['我','惭愧']` | 11.83 s / 1.77 | 无句界输出：未虚构句号/句界，但前句内容在丢帧下丢失，只剩第二句尾部 |
| 两段拼接（串句） | B | 202 | 66 | 0 | 1 | `['我','惭愧']` | 20.01 s / 2.99 | 同上 |

**负样本结论**：
- **无人门控有效**（黑帧 2 轮均 0 窗/空输出/108 帧全拒绝）；这是 replay 路径的可用
  安全边界证据。
- **镜像检测缺失**：左右镜像片段不被质量门控拒绝，rtmpose 前端仍给出姿态并输出 gloss
  （run B 有 1 词）；镜像/反向输入必须由上层方向策略处理（与 E05/uni-sign 方向坑一致）。
- **串句无句界**：两段无停顿拼接不会产生句界或凭空补全；但当前实现也没有句子切分能力，
  长视频只会输出一条（受丢帧影响的）gloss 序列——产品层需要外部断句。

## 5. 证据范围与限制（不得越界声明）

1. **replay ≠ 摄像头**：帧 pacing 由源时间戳驱动；相机抖动、曝光、丢帧模式、真人等待/
   遮挡、桌面端渲染与队列均未覆盖。不能宣称“现场摄像头验收通过”。
2. **无 GT / 不评准确率**：本节所有输出与指标只用于流式交付与时延；未与参考句/Gloss
   对齐（replay 路径的文本质量应按 E02/E03 的离线口径另行评估）。
3. **环境敏感**：两轮 RTF P50 差 16%（2.23 vs 2.58）、单条差异更大（1.39–3.06）；
   32 GB Windows 主机上其它 agent/桌面进程会抢占 CPU（SLRT 的瓶颈是窗口 CPU 预处理）。
   引用数字时必须同时给出 run_id 和轮次，不要把单轮当 SLA。
4. **前端差异**：正样本用官方 HRNet NPZ（与离线评测同协议）；负样本用本地 RTMPose
   （因为黑帧/镜像无法绑定官方 NPZ）。负样本结论只针对门控/方向，不是 HRNet 精度。
5. **未覆盖**：真实摄像头长时间稳定性、pause/resume/reset、多段连续对话、不同采集
   FPS（25/60）——这些仍是 NOT_RUN。

## 6. 复现命令

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.conda-env\python.exe experiments\p1\e05\replay_probe.py run `
  --run-dir experiments\p1\e05\replay-probe-<stamp> `
  --samples 10 --negatives 2 --concat --max-foreign-vram-mb 4500
# 产物：manifest.json / window_events.jsonl / stream_events.jsonl / summary.json / negatives/*.mp4
```

- 本轮未修改 `slbridge/` 源码；插桩在 `experiments/p1/e05/replay_probe.py` 内以实例级
  包装实现，`git status` 可核验。
- GPU 用前记录基线（2196–2595 MiB 桌面占用、无其它 CUDA compute 进程），结束后无
  本实验遗留进程。

## 7. 对 G2 决策的建议

1. **不要把 SLRT replay 当作可实时上线的字幕路径**：当前硬件+现有实现 RTF 2.2–2.6，
   首稳在句末附近；需要至少 2–3× 的窗口处理加速（批处理/降 FPS/优化热力图预处理）
   才有讨论空间。
2. **保留 SLRT 作为整句 Gloss 后备**；在线首字/首稳需要在 **离线批处理** 或
   **降采样到 ≤10–12 fps** 的采集条件下重新验收。
3. 无人门控可作为安全前置（黑帧负样本通过）；镜像/方向必须由应用层显式策略处理。
4. 与 E05 一致：Uni-Sign 无原生 partial（首稳前缀 90%），SLRT 有原生 partial 但当前
   实时吞吐不足 → P2 实时链路应优先解决“窗口前处理吞吐”，而不是继续加模型。
