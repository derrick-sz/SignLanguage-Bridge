# CSL-Daily 视频端到端验证：SLRT 与 Uni-Sign 预测 vs 真值（627 条本地 test）

日期：**2026-10-26**。对象：`D:\Zenodo\csl-daliy` 中与官方 `csl-daily.test` 相交的 **627 条 test** 原始句子视频（本地共 10,456 条 mp4，与训练标注相交 test 627 / dev 552 / train 9,277）。同一批裁剪片段上同时跑两个已发布模型，输出**真值（句子 + Gloss）与两套预测的逐条对比**：

> **更正（2026-10-26 下午）**：视频随后补充到 19,528 条（test 覆盖 1,086），且发现本报告原先把"视频（标注文本参考）"与"pose（上游参考）"的 BLEU 直接比较，参考文本的全角/半角标点差异放大了差距。本报告数字已按**统一上游参考**（`test_tmp_refs.txt`）重算：视频 BLEU 由 21.9944 更正为 **24.9485**，相对官方 pose 的差距由 **−3.82 更正为 −0.87 BLEU**。全量 1,086 条结果见 [补充报告](csldaily-video-slrt-unisign-full-1086-2026-10-26.md)。

- **Uni-Sign**（`csl_daily_pose_only_slt.pth`，pose-only SLT）：视频端到端 → 中文句子；指标 BLEU-4 / chrF / 字符 ROUGE-L / 字符编辑相似度。
- **Uni-Sign（官方 pose 对照）**：同一 checkpoint 吃作者发布的 `csl_daily_pose_format` 官方 512 像素姿态；这是 627 条上的论文口径参照。
- **SLRT**（`cslr_best.ckpt`，在线 Gloss）：裁剪 RGB + 官方 HRNet133 关键点 → Gloss 序列；指标词级 WER。

两者任务不同（翻译 vs Gloss 识别），**不是排行榜**；对比表是把同一批样本的真值和两套输出并排放，指标分别读。

## 数据与协议

| 项 | 内容 |
|---|---|
| 视频 | `D:\Zenodo\csl-daliy`，10,456 个 mp4（约 16.0 GB），未裁剪原始录制（1280/1080 方形，30 / 29.97 fps） |
| 官方标注 | `F:\model_weigth\data\csl-daily\csl-daily.test`（gzip-pickled，1,176 条：`name / signer / gloss / text / num_frames`） |
| 官方姿态 | `F:\model_weigth\data\csl-daily\csl-daily-keypoints.pkl`（20,654 条，`[T,133,3]`，512 参考系） |
| 选取 | `video ∩ csl-daily.test` = **627 / 1,176**；selection：`artifacts/csldaily-video-test-627-20261026/selection.json`（sorted，含 GT 句子/Gloss/num_frames） |
| 帧数 | min 49 / mean 145.07 / median 140 / max 340；Gloss 平均 8.42 token，参考句平均 17.72 字 |
| 签者 | 本地 627 条覆盖 10 个签者（P0000…P0009），P0000 占 211 条 |

**语句段裁剪（SLRT 与 Uni-Sign 视频路径共用的同一份片段）**：官方 `sentence_frames-512x512.zip` 不在本机，被裁剪的只是"从哪里开始"这件事——

1. 在原始视频上全帧跑 RTMPose（`rtmpose-experimental`，CUDA），取 SLRT 的 63 点；
2. 与官方 HRNet keypoints 的同 63 点做长度 T 的**穷举 stride-1 滑窗**，均值欧氏距离最小处即 `best_start`；打分只使用两侧都可靠的点（官方 conf≥0.2 且坐标在 ±64..576、RTMPose conf≥0.1 且检出人）；
3. 以 `[best_start, best_start+T)` 裁出 mp4（保留原 fps，mp4v 重编码），并用官方 keypoints 的裁剪段生成 SLRT 的 `hrnet-npz`（保留全部官方点，confidence 只 clip 到 [0,1]）。

对齐质量（627 条）：**mean 7.06 px / median 6.19 / P90 9.85 / P95 12.14 / P99 19.17 / max 88.58**；仅 3 条 >25 px（`S001266_P0002_T00` 88.6 px、`S003537_P0001_T00` 76.5 px、`S003206_P0000_T00` 45.4 px）。

| 模型 | 输入与协议 |
|---|---|
| Uni-Sign 视频端到端 | 裁剪 mp4 → 本地 RTMPose wholebody 133 点（pinned `rtmlib-cache`，isotropic 短边归一化）→ pose window（max 256 帧）→ pinned mT5/`csl_daily_pose_only_slt.pth`，bf16、beam=4；checkpoint SHA `73bea87a…c1a6c` |
| Uni-Sign 官方 pose | 作者 `pose-format` pkl（`[T,133,2]+scores`，pinned 权重同款 256 帧确定性 linspace）→ 同 checkpoint；这就是此前完整 1,176 test 得 25.5504 的口径 |
| SLRT | 裁剪 mp4 + 官方 keypoints NPZ（`--pose-backend hrnet-npz`），checkpoint SHA `e9fdd3a6…cbd4d`（938 键 / 2001 类），window 16 / stride 1 / vote 7，未用 `--require-gloss`；WER 为 token Levenshtein（S/D/I，无文本归一化），参考为 `gloss` 字段 |

**Uni-Sign 的 BLEU 口径**：参考与预测都先逐字加空格，再按上游 `fine_tuning.py:282-284` 的 `' '.join(list(text.replace(' ','')))` 重写后算 SacreBLEU 13a（不做这步 BLEU 会假崩到 ~0，见坑点记忆）。

## 结果

| 项目 | Uni-Sign 视频端到端 | Uni-Sign 官方 pose | SLRT |
|---|---:|---:|---:|
| 成功 / 尝试 | **627 / 627** | **627 / 627** | **627 / 627** |
| 主指标 | BLEU-4 **24.9485** | BLEU-4 **25.8142** | corpus WER **0.3032** |
| chrF | 22.0051 | 22.6634 | — |
| 字符 ROUGE-L F1 | 0.5252 | 0.5344 | — |
| 平均字符编辑相似度 | 0.4123（P50 0.3571） | 0.4179（P50 0.3750） | — |
| 完全匹配 | 8 / 627 | 9 / 627 | 78 条 WER=0 |
| 误差结构 | BP 0.9125；1–4 gram 精度 57.56 / 32.98 / 20.79 / 14.16 | BP 0.9228；58.19 / 33.75 / 21.41 / 14.57 | S 748 / D 493 / I 359（共 1,600 edits，参考 5,277 token） |
| 耗时 | e2e 均值 7.42 s（P50 5.93 / P95 16.18）；姿态 6.71 s、模型 0.70 s | 模型侧 ~0.4–0.7 s/条 | CLI wall 均值 16.89 s（P50 16.43 / P95 26.70，含 3.9 s 权重加载） |

SLRT 逐条 WER 分布（627 条）：=0 **78**、≤0.1 **18**、≤0.3 **273**、≤0.5 **166**、>0.5 **92**；逐条均值 0.2969、中位 0.25。剔除 3 条低置信对齐后 corpus WER **0.2999**（624 条）。

### 读法

1. **视频端到端相对官方 pose 的代价约 −0.87 BLEU**（24.9485 vs 25.8142，同 627 条，统一上游参考），配对编辑相似度差 +0.0055（官方 pose 更好），完全匹配 9 vs 8。差距来自本地 RTMPose 前端、mp4v 重编码与"裁剪只是近似语句段"的叠加，量级很小；**不能用它否定任一份权重**。此前写的 −3.82 BLEU 是"标注文本参考 vs 上游参考"的口径错误，已在[补充报告](csldaily-video-slrt-unisign-full-1086-2026-10-26.md)更正。完整 1,176 test 的官方 pose 口径仍是 **25.5504**，627 子集官方 pose 为 25.8142，属子集波动。
2. **SLRT 同域 WER 30.3%**，与 52 条试点 33.0% 同向（52 条参考句平均 23.6 字，627 条 17.7 字，长句更难），远好于 CE-CSL 跨域 89.18%。这仍不是论文 `sentence_frames-512x512` 协议，**不能直接对标论文在线 29.3% 或 CSLR test 27.4%**。
3. **误差主要不是对齐造成的**：对齐距离与 WER 的 Pearson 全量 0.116、剔除 3 条低置信后 −0.083；3 条低置信样本恰好 WER 0.86–0.93，说明只有极端失配时对齐才主导。SLRT 的删除（493）多于替换（748）但仍以替换为主，插入 359 说明模型在部分句子上过生成。
4. **句长与签者影响明显**：≤10 字参考句 SLRT WER 0.212 / Uni-Sign 编辑相似度 0.544；>20 字 WER 0.327 / 编辑相似度 0.324。签者层面 SLRT WER P0008 0.188、P0000 0.244、P0005 0.364；Uni-Sign 视频编辑相似度 P0008 0.575、P0005 0.317。
5. 两个任务的逐条分数只有弱相关（Pearson(Uni-Sign 视频编辑相似度, SLRT WER) = **−0.231**）。这只是观察，**不能**据此把任一指标当作另一任务的质量代理。

## 逐条对比产物（真值 + 两套预测）

- **合并表（推荐，统一上游参考）**：`artifacts/csldaily-video-test-627-20261026/combined-627-paperrefs/comparison-both.md` / `.csv` / `summary.json`
- **Excel（627 行 + Summary 页）**：`artifacts/csldaily-video-test-627-20261026/combined-627-paperrefs/csldaily-video-comparison-627.xlsx`
- **子群体补充统计**：`artifacts/csldaily-video-test-627-20261026/combined-627-paperrefs/analysis-extra.json`
- Uni-Sign 视频端到端（统一参考）：`artifacts/csldaily-video-test-627-20261026/unisign-video-627-paperrefs/{predictions.jsonl,summary.json}`；标注文本参考的原始版本保留在 `unisign-video/`（仅审计，不可与上游参考数字混用）
- Uni-Sign 官方 pose（627）：`artifacts/cslnews-eval/csldaily-test-627-pose-20261026/{predictions.jsonl,summary.json}`
- SLRT：627 基线快照 `artifacts/csldaily-video-test-627-20261026/summary-627.json`、`comparison-aligned-627.md`、`comparison-aligned-627.csv`（目录内的 `summary.json` / `comparison-aligned.*` 在 2026-10-26 下午补充 459 条新视频后被 1,086 条全量结果覆盖，见 [补充报告](csldaily-video-slrt-unisign-full-1086-2026-10-26.md)），逐条 `slrt/<sample>.json`、`infer/<sample>.json`、`alignment/<sample>.json`
- 裁剪片段：`trimmed/`（627 基线时 627 个 mp4、共 1,648 MB；2026-10-26 下午补充后 1,086 个、约 2.26 GB）；SLRT 姿态 NPZ：`pose/`；RTMPose 全帧缓存：`rtmpose/`
- 输入清单：`selection.json`、`refs-627.txt`；52 条试点产物复用记录：`reused-52-provenance.json`（对齐/推理记录仅重写了绝对路径，sha256 仍绑定复制后的字节）

## 产物校验与读表注意

- `unisign-video/predictions.jsonl` 是逐条追加式记录，含重试产生的重复行（共 650 行 / 627 个唯一样本）；`summary.json` 与合并表按样本名取**最后一条**，逐条以 `combined-627-paperrefs/comparison-both.csv` 为准。
- SLRT 的 `slrt/` 下每个样本有 3 个文件（`<name>.json`、`.events.jsonl`、`.windows.jsonl`），627 基线阶段 1,881 个，补充后共 3,258 个；`infer/<name>.json` 是逐条退出码/耗时记录。
- 结构自检：`artifacts/csldaily-video-test-627-20261026/selfcheck.json`——全量 627 条 NPZ/裁剪结构不变量 **627/627 通过、0 mismatch、627 个 SLRT 结果文件**；随机 2 条（`S004692_P0003_T00`、`S004242_P0007_T00`）逐帧复核：声明帧数=解码帧数=标注帧数（89 / 112），首末帧与源视频 `best_start` 对应帧平均绝对差 2.26–2.51（mp4v 有损重编码），NPZ 全有限、confidence∈[0.0198, 1.0]，SLRT 结果 `video_sha256` 与实际裁剪文件一致、`status=final`。若自检未覆盖到某类异常，以逐样本记录为准。

## 复现命令

```powershell
$env:PYTHONIOENCODING = 'utf-8'
Set-Location 'E:\githubproject\SignLanguage-Bridge'

# 0) 选取 627 条（test 与本地视频相交；同时校验官方 keypoints 帧数）
.\.conda-env\python.exe scripts\build_csldaily_video_selection.py --split test --check-keypoints `
  --expected-total 1176 --output artifacts\csldaily-video-test-627-20261026\selection.json

# 1) 语句段对齐 + 裁剪 + SLRT 姿态 NPZ（4 分片；可重复执行续跑）
foreach ($i in 0..3) {
  .\.conda-env\python.exe scripts\align_slrt_csl_daily_subset.py --stage align `
    --selection artifacts\csldaily-video-test-627-20261026\selection.json `
    --artifact-dir artifacts\csldaily-video-test-627-20261026 `
    --annotation F:\model_weigth\data\csl-daily\csl-daily.test `
    --shard-index $i --shard-count 4
}

# 2) SLRT 推理（3 分片；失败样本重跑同命令会自动重试）+ 汇总
foreach ($i in 0..2) {
  .\.conda-env\python.exe scripts\align_slrt_csl_daily_subset.py --stage infer `
    --selection artifacts\csldaily-video-test-627-20261026\selection.json `
    --artifact-dir artifacts\csldaily-video-test-627-20261026 `
    --annotation F:\model_weigth\data\csl-daily\csl-daily.test `
    --shard-index $i --shard-count 3
}
.\.conda-env\python.exe scripts\align_slrt_csl_daily_subset.py --stage report `
  --selection artifacts\csldaily-video-test-627-20261026\selection.json `
  --artifact-dir artifacts\csldaily-video-test-627-20261026 `
  --annotation F:\model_weigth\data\csl-daily\csl-daily.test

# 3) Uni-Sign 视频端到端（自动续跑）
.\.conda-env\python.exe scripts\evaluate_unisign_csldaily_video.py `
  --selection artifacts\csldaily-video-test-627-20261026\selection.json `
  --artifact-dir artifacts\csldaily-video-test-627-20261026 `
  --output-dir artifacts\csldaily-video-test-627-20261026\unisign-video --device cuda

# 4) Uni-Sign 官方 pose 对照（627 条）
.\.conda-env\python.exe scripts\evaluate_unisign_csldaily_pose.py `
  --pose-dir artifacts\cslnews-eval\csl-daily-pose\pose-format `
  --refs artifacts\csldaily-video-test-627-20261026\refs-627.txt `
  --output-dir artifacts\cslnews-eval\csldaily-test-627-pose-20261026

# 5) 合并真值/两模型预测 + Excel
.\.conda-env\python.exe scripts\build_csldaily_video_comparison.py `
  --selection artifacts\csldaily-video-test-627-20261026\selection.json `
  --unisign-video-summary artifacts\csldaily-video-test-627-20261026\unisign-video\summary.json `
  --slrt-summary artifacts\csldaily-video-test-627-20261026\summary.json `
  --unisign-pose-predictions artifacts\cslnews-eval\csldaily-test-627-pose-20261026\predictions.jsonl `
  --unisign-pose-summary artifacts\cslnews-eval\csldaily-test-627-pose-20261026\summary.json `
  --output-dir artifacts\csldaily-video-test-627-20261026\combined
<bundled-python> scripts\export_csldaily_comparison_xlsx.py `
  --summary artifacts\csldaily-video-test-627-20261026\combined\summary.json `
  --output artifacts\csldaily-video-test-627-20261026\combined\csldaily-video-comparison.xlsx
```

## 运行异常与重试（不得算作模型失败）

- **SLRT 第一轮 627 条中有 47 条失败**，全部是内存分配失败（`numpy.core._exceptions._ArrayMemoryError`、OpenCV/FFmpeg `Insufficient memory`、`CUDA error: out of memory`、`Memory allocation failure`），发生在与 Uni-Sign 进程并发、整机可用内存仅 2–3 GB 的时候。用同一命令分 3 片重跑后 **627/627 成功、0 空输出**。逐条首轮错误记录仍在 `infer/*.json` 的历史里，最终成功记录覆盖同名文件。
- **Uni-Sign 视频端到端第一轮在 240/627 条时进程崩溃**（日志尾部为 FFmpeg/OpenCV `get_buffer() failed` + `Insufficient memory`），随后在内存宽裕时用同一命令续跑到 **627/627 成功**。耗时统计因此包含并发争用，**不能当生产延迟**。
- 开始前有一个上一轮的 100 条 train 拟合任务（`align_slrt_csl_daily_train100.py --stage all`）仍在占用 GPU/内存，已停止以释放资源；其产物目录保留、可续跑，与本报告无关。

## 适用边界（不得越界引用）

- **627 / 1,176**：本地只有 1,176 条测试视频中的 627 条（另 549 条缺视频）；这不是完整官方 test，也不是新盲测（样本集合由"本地有哪些文件"决定）。
- **SLRT 仍是近似帧协议**：官方 `sentence_frames-512x512.zip` 不在本机，句段由 RTMPose↔官方 keypoints 滑窗对齐得到；对齐均值 7.06 px 说明多数样本可信，但 3 条 >25 px 的样本其 WER 显著偏高。
- **Uni-Sign 视频路径不是论文协议**：mp4v 重编码 + 本地 RTMPose + 确定性 linspace；论文口径请引用官方 pose 的 25.5504（完整 1,176）或本报告 627 条官方的 25.8142。
- **两个模型权重都是 CSL-Daily 训练域**，627 条属于同域 test；不能外推到陌生签者、移动机位或其它手语。
- 延迟数字来自单进程、与其他 GPU 任务共享机器的测量，不构成产品 SLA。
