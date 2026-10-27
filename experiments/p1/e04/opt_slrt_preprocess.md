# 阶段 4 · SLRT 逐窗预处理优化对照（opt_slrt_preprocess）

> **状态**：`PASS`（20 条样本 × 2 臂，0 失败；优化数值等价性逐张量验证）
> **run 目录**：`experiments/p1/e04/opt-runs/slrt-preprocess-20261027T1135Z`
> （`manifest.json`、`stage-profile.json`、`equivalence.json`、
> `predictions-baseline.jsonl`、`predictions-optimized.jsonl`、`metrics.json`）
> **纪律**：生产默认值未修改；优化只在 benchmark 进程内以实例/模块级包装实现
> （`slbridge/` 无改动，`git status` 可核验）；GT Gloss 仅在推理后离线 join 计算 WER。

## 1. 基线瓶颈定位（run A 实测，20 条 pilot，2406 窗口）

E04 报告中的“~5.6 s/条 非 forward 开销”在本轮被拆开（`stage-profile.json` +
插桩总计）。**注意时间口径**：`prepare_inputs` 的 torch 操作是异步的，CPU 墙钟只测到
kernel 入队；真实 GPU 准备时间会在下一次 `torch.cuda.synchronize`（模型推理入口）时
才结算。因此下表同时给出“同步可见”和“有效/含隐藏”两种口径：

| 阶段（20 条合计） | 同步可见 ms | 有效/含隐藏 ms | 占 e2e | 说明 |
|---|---:|---:|---:|---|
| 视频解码（`video_packets`） | 5,782 | 5,782 | 4.7% | OpenCV 逐帧 |
| `canonical_frame`（PIL 512→320，每帧一次） | **34,445** | **34,445** | **28.0%** | **单帧 ~14.3 ms，是当前最大预处理开销** |
| `prepare_inputs`（逐窗 RGB+热力图） | 14,495 | **29,073**（+14,578 GPU 异步） | **23.6%** | 2406 窗口 × ~12 ms 有效 |
| S3D 前向（模型） | 48,758 | 48,758 | 39.5% | 2406 窗口 |
| 其余 Python/decoder/收尾 | — | ~4,900 | 4.0% | |
| **e2e 合计** | | **123,227** | 100% | 每条约 6,161 ms |

**冗余发现**：滑窗 stride=1 时每个窗口 16 帧、相邻窗口步进 1 帧 → **同一帧的热力图/
RGB 会被重复计算约 16 次**（2406 窗口 = 20 条共 ~2,300 帧，却做了 2406×16 ≈ 38,500 次
帧级准备）。这是可以在**不改变任何数值**的前提下消除的冗余。

## 2. 优化实现（数值语义不变）

`opt_slrt_preprocess.py::Instrumentation.install(use_cache=True)`：

1. 在 benchmark 进程内把 `slbridge.slrt.preprocess.prepare_inputs` 替换为带缓存的版本
   （`SLRTRuntime.predict` 内部 `from .preprocess import prepare_inputs`，因此运行时
   自然使用替换后的函数，**不改库代码**）；
2. 缓存版对每个窗口逐帧处理：**用原始未修改的 `prepare_inputs` 在单帧 (T=1) 上计算**
   `(rgb, heatmap)`，按 `(帧字节 hash, 原始关键点字节 hash)` 缓存后 `torch.cat` 组装；
3. 缓存上限 128 帧（FIFO 整体清理，VRAM 峰值增加 <0.5 GB），跨样本不会无限增长；
4. 因为数值全部来自同一原始函数、同一输入，**只消除重复计算，不改变算术**。

**等价性证据（`equivalence.json`，cuda:0，8 个真实窗口）**：

| 张量 | shape | max abs diff | torch.equal |
|---|---|---:|---|
| RGB（优化 vs 原始） | (1,3,T,224,224) | **0.0** | **True** |
| 热力图（优化 vs 原始） | (1,63,T,112,112) | **0.0** | **True** |

→ 逐位相同（不是“≤1e-6”，而是精确相等）；进一步地，20 条样本的最终 Gloss 序列
**20/20 逐字相同**、corpus WER 完全一致。

## 3. 优化前后对照（同一 20 条 pilot、同协议 hrnet-npz、window16/stride1/vote7）

| 指标 | baseline | optimized | 变化 |
|---|---:|---:|---:|
| 成功 / 失败 | 20 / 0 | 20 / 0 | — |
| **e2e wall P50 / P95 ms** | **5,923.4 / 8,557.1** | **4,835.7 / 7,152.4** | **−18.4% / −16.4%（speedup 1.225×）** |
| 每样本 e2e 配对 Δ（20 条） | — | 全部改善 | 均值 **−1,120.8 ms**（min −749, max −1,542） |
| `prepare_inputs` 同步可见 P50 | 709.7 ms | 385.9 ms | −45.6% |
| `prepare_inputs` 有效（含 GPU 异步）总计 | 29,073 ms | 8,629 ms | **−70.3%** |
| `canonical_frame` 总计 | 34,445 ms | 34,046 ms | −1.2%（未优化，噪声） |
| S3D 前向总计 | 48,758 ms | 47,511 ms | −2.6%（噪声/缓存效应） |
| **corpus WER**（GT Gloss，137 ref tokens） | **0.27007**（S24/I5/D8） | **0.27007**（S24/I5/D8） | **0.0（20/20 Gloss 逐字相同）** |
| Gloss 序列逐字相同 | — | **20/20** | 决定性等价 |

> 说明：e2e 改善（−18.4%）大于“同步可见 prepare”改善（−45.6% of 710ms），因为
> 基线 prepare 的 GPU kernel 是异步的，其排队时间隐藏到模型推理同步点；缓存版把
> 逐帧小 kernel 重新排序后显著缩短了这条“隐藏队列”，同时几乎消除了 16× 重复计算。

## 4. 结论与建议

1. **该优化可以合入生产路径（建议）**：数值逐位相同、WER 完全不变、20 条配对
   e2e P50 −18.4%（5.92→4.84 s），是纯工程收益。实现建议：把“帧级 `(rgb, heatmap)`
   缓存”落到 `SLRTRuntime`/`SLRTStream`（按 `frame_index` 或内容 hash），缓存上限
   按窗口×2 帧控制；需要同步更新姿态/缓存 protocol 身份，避免跨运行误用。
2. **下一个最大瓶颈是 `canonical_frame`**：PIL 512→320 每帧 ~14.3 ms，占 e2e 28%；
   若换成 cv2/自定义双线性会改变像素 → 必须做逐像素等价研究后再替换（本轮未做，
   不声称收益）。其次仍是 S3D 前向（39.5%）。
3. **E04 报告旧措辞需修正**：此前“~5.6 s/条 花在逐窗 CPU 热力图预处理”不准确。
   实测拆分：canonical_frame ~1.72 s/条（28%）+ prepare_inputs 有效 ~1.45 s/条
   （23.6%，其中一半是 GPU 异步且 CPU 计时不可见）+ S3D ~2.44 s/条（39.5%）+
   解码 ~0.29 s/条。**热力图在 GPU 上算，不是 CPU**。`hardware_report.md` 已加
   优化前后对照表（§H）。
4. 局限：20 条 pilot（10 signer）子集；e2e 在单进程、有桌面/agent 后台负载的机器上
   测得；缓存实现是 benchmark 包装，不是已合入代码；`canonical_frame`/模型前向未变，
   故不能宣称“整体 2×”。

## 5. 复现命令

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.conda-env\python.exe experiments\p1\e04\opt_slrt_preprocess.py run `
  --run-dir experiments\p1\e04\opt-runs\slrt-preprocess-<stamp> --samples 20
```

产物：`manifest.json`、`stage-profile.json`、`equivalence.json`、
`predictions-{baseline,optimized}.jsonl`、`metrics.json`。GPU 基线 2054 MiB 桌面占用、
无其它 CUDA compute 进程；单进程串行；0 次 OOM/失败。
