# SLRT Online 非数据资产补齐记录

日期：**2026-10-25**。前置报告：[SLRT Online 复现差距核查](slrt-online-reproduction-gap-2026-10-25.md)。本轮先在没有 CSL-Daily 视频帧的前提下补齐非视频资产；2026-10-25 晚作者 Google Drive 补充包（`F:\model_weigth`）又补齐了 CSL-Daily 关键点、句级标注和 TwoStream-SLR S2G，记录见 1.1 节。数据与依赖数据的权重仍明确标记为缺口，不用占位文件冒充。

结论：

1. 官方在线 CSLR 权重、词表、词级/句级 meta、HRNet 关键点、S2G 词表与 TwoStream-SLR S2G 权重、HRNet/Faster-RCNN 权重、K400 S3D 已放入官方目录结构，并可用 `scripts/setup_slrt_official_layout.py` 幂等重建。
2. 官方 HRNet 前端已在独立 Python 3.8 legacy CPU 环境跑通：CE-CSL `dev-00015` 全 202 帧零丢帧，输出 `.npz` 可被本地 `--pose-backend hrnet-npz` 消费；官方 HRNet 与实验 RTMPose 在同一视频上分别输出 `拒绝 不` 与 `拒绝 去 不`。这只是跨域链路验证，不是 CSL-Daily 同域复现。
3. SLT 的 **gloss 侧**资产（`gloss2ids.pkl`、`gloss_embeddings.bin`）已由 base mBART + CSL 词表生成；文本侧剪枝需要 `csl.train/dev/test`，现在数据已到手但尚未重跑剪枝，**明确标注的全词表 fallback** 仍不是作者 `mBART_zh`。CSL-Daily G2T 权重仍未获得。
4. 作者 Google Drive 补充包提供 TwoStream-SLR S2G（`ckpts/TwoStreamSLT/TwoStream/csl-daily_s2g/ckpts/best.ckpt`，504,360,673 B，SHA256 `54d2cb65…9739`）；原 SharePoint 链接仍 404。`prediction_slide` 输出的 SLT 预测转换脚本已补齐并有单测。
5. 仍然缺的只有：`sentence_frames-512x512.zip`（CSL-Daily 视频帧）、CSL-Daily G2T 权重/作者 `mBART_zh`、官方 `prediction_slide.py` 旧依赖栈。其余资产清单可用 `scripts/check_slrt_reproduction_assets.py` 一条命令核对。

## 1. 官方目录结构（可重建）

`scripts/setup_slrt_official_layout.py` 以 `Paper-algorithm/SLRT-main` 为上游根，建立官方 `prediction_slide.py` 期望的路径，默认幂等、硬链接优先、目标内容不同则失败：

| 目标 | 来源 | 结果 |
|---|---|---|
| `Online/CSLR/results/csl-daily_ISLR/ckpts/best.ckpt` | `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt`（499,554,423 B，SHA256 `e9fdd3a6…cbd4d`） | 硬链接 |
| `data/csl-daily/csl_iso_with_blank.vocab` | `…/meta/csl_iso_with_blank.vocab`（28,821 B，SHA256 `0c4984f0…4c76`） | 硬链接 |
| `data/csl-daily/csl_iso_center_label_bag2items_halfblk.train` | 词级 ISLR meta（177,213,738 B，200,530 个片段） | 硬链接 |
| `data/csl-daily/csl_iso_with_blank.dev` | 词级 ISLR meta（1,076,884 B，14,059 个片段） | 硬链接 |
| `data/csl-daily/csl_iso_with_blank.test` | 词级 ISLR meta（1,215,757 B，15,826 个片段） | 硬链接 |
| `data/csl-daily/csl_s2g_gloss2ids.pkl` | 上游镜像 `TwoStreamNetwork/data/csl-daily/gloss2ids.pkl`（2,004 项，`<si>`=0） | 硬链接 |
| `data/csl-daily/csl.{train,dev,test}` | 作者包 `csl-daily.{train,dev,test}`（18,401 / 1,077 / 1,176 句；`name/gloss/text/num_frames`） | 硬链接别名 |
| `data/csl-daily/keypoints_hrnet_dark_coco_wholebody{.pkl,_iso.pkl}` | 作者包 `csl-daily-keypoints.pkl`（1,971,425,633 B，20,654 个视频，`[T,133,3]` float16） | 同一文件的两个硬链接 |
| `Online/CTC_fusion/results/csl-daily_s2g/ckpts/best.ckpt` | 作者包 `ckpts/TwoStreamSLT/TwoStream/csl-daily_s2g/ckpts/best.ckpt`（504,360,673 B，SHA256 `54d2cb65…9739`，912 键、2,004 类） | 复制 |
| `pretrained_models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth` | `models/external/uni_sign/…`（254,976,311 B） | 硬链接 |
| `pretrained_models/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth` | `models/external/uni_sign/…`（165,671,301 B） | 硬链接 |
| `pretrained_models/s3ds_actioncls_ckpt` | 下载自 kylemin/S3D（33,453,443 B，SHA256 `6b7f0229…b2d2`） | 硬链接 |
| `pretrained_models/mBart` | 基础 mBART（见第 3 节） | 目录联接（Junction） |

`data/csl-daily/EXPECTED_FILES.md` 列出仍缺的数据文件及字段要求；`SLRT_ASSET_LAYOUT.json` 记录链接、数据存在性与缺失清单，不会生成或填充占位数据。

```powershell
.conda-env\python.exe scripts\setup_slrt_official_layout.py `
  --data-source 'F:\model_weigth\data\csl-daily' `
  --two-stream-checkpoint 'F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_s2g\ckpts\best.ckpt'
.conda-env\python.exe scripts\check_slrt_reproduction_assets.py
```

### 1.1 作者 Google Drive 补充包（`F:\model_weigth`）

作者在 issue #107 的评论里给出 Google Drive 文件夹，用来替代已失效的 SharePoint。文件夹里六个压缩包的实际取舍：

| 文件 | 内容 | 是否使用 |
|---|---|---|
| `Copy of data.zip` | `data/csl-daily/csl-daily-keypoints.pkl`（1.97 GB）、`csl-daily.train/dev/test`（gzip pickle）、`gloss2ids.pkl`（2,004 项）；另有 Phoenix/MSASL/NMFs/WLASL 数据 | **使用**：CSL-Daily 关键点 + 句级标注 + S2G 词表 |
| `Copy of ckpts.zip` | `ckpts/TwoStreamSLT/TwoStream/csl-daily_s2g/ckpts/best.ckpt`（481 MB）及 SingleStream S2G/S2T、feature pkl；NLA-SLR/LCSA/Phoenix 权重 | **按需使用**：只取 TwoStream CSL-Daily S2G；其余与本任务无关 |
| `Copy of EMNLP24_Online.zip` | 与本地 `models/external/EMNLP24_Online` 重复 | 不用 |
| `Copy of mBart_zh_t2g.zip` | Spoken2Sign **Text2Gloss** 资产（config vocab 6,308；`old2new_vocab.pkl` 6,308；`gloss2ids.pkl` 2,009） | **不用**：不是 Online SLT 的 `mBART_zh` G2T 权重 |
| `Copy of mBart_de_t2g.zip` | Phoenix-2014T 的德文 T2G 资产 | 只做 Phoenix T2G 时才用 |
| `Copy of TA.zip` | COMP2011/2012 课程文件 | 不用，与 SLRT 无关 |

作者包**没有** `sentence_frames-512x512.zip`（CSL-Daily 视频帧），也没有 online `csl-daily_g2t` 权重；issue #17 也记录 CSL-Daily G2T 缺失。

## 2. 官方 HRNet 前端：legacy 环境 + NPZ

### 2.1 环境

`requirements-slrt-legacy.txt` 固定一套**独立**的 Python 3.8 CPU 环境（不能与项目 `.conda-env` 的 Torch 2.9 混用）：

- Python 3.8.20
- torch 1.13.1+cpu / torchvision 0.14.1+cpu
- mmcv-full 1.7.0（OpenMMLab 预编译 CP38 Windows wheel）
- mmdet 2.25.3
- mmpose 0.29.0
- pycocotools-windows 2.0.0.2、xtcocotools 1.14.3、chumpy 0.70 等

本轮实际环境：`F:\DSH-envs\slrt-legacy`（不属仓库，不入库）。旧 CUDA 10.2 栈不支持 RTX 5060 Ti，因此 CPU 运行；202 帧提取耗时 **733.674 s**（约 3.63 s/帧），无丢帧。配置树来自 OpenMMLab 官方 tag：`mmpose v0.29.0`、`mmdetection v2.25.3`，位于 `models/external/legacy_openmmlab/`（gitignore 的 `models/` 下）。

### 2.2 提取脚本

`scripts/extract_slrt_hrnet_pose.py`：

- Faster-RCNN person 检测 + HRNet-W48 wholebody DARK；`--det-score-thr 0.5`、`--det-area-thr 1600`、512×512 参考系。
- 默认 `--color-order rgb`，与上游 `gen_pose.py` 的 PIL 读取路径一致；`--color-order bgr` 仅作对照。
- 写出 `keypoints [T,133,3]`、`frame_indices`、`timestamps`、`metadata_json`，元数据含视频 SHA256、`frontend="hrnet_w48_dark"`、`coordinate_system="coco_wholebody_133_xyc_512_v1"`、检测/姿态模型及配置的 SHA256、缺失帧列表。
- 默认遇到无人/无姿态帧即失败；`--allow-missing` 只用于诊断，并记录缺失帧。

```powershell
F:\DSH-envs\slrt-legacy\python.exe scripts\extract_slrt_hrnet_pose.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\dev\A\dev-00015.mp4' `
  --output artifacts\slrt-hrnet-20261025\dev-00015-full-hrnet.npz --device cpu

.conda-env\python.exe -m slbridge.slrt.cli video `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\dev\A\dev-00015.mp4' `
  --config configs/slrt-csl-hrnet-npz.example.json `
  --pose-backend hrnet-npz --pose-npz artifacts\slrt-hrnet-20261025\dev-00015-full-hrnet.npz `
  --output artifacts\slrt-hrnet-20261025\hrnet-full-video.json `
  --events-jsonl artifacts\slrt-hrnet-20261025\hrnet-full-events.jsonl `
  --windows-jsonl artifacts\slrt-hrnet-20261025\hrnet-full-windows.jsonl --require-gloss
```

### 2.3 验证结果（CE-CSL dev-00015，202 帧，跨域）

| 前端 | 输出 Gloss | 事件分数 | 本地推理墙钟 | 说明 |
|---|---|---|---|---|
| 官方 HRNet-W48 DARK（本脚本，RGB） | `拒绝 不` | 0.28 / 0.46 | 22.75 s（8.88 FPS，含加载） | 202 窗、0 丢帧；NPZ 元数据通过本地校验 |
| RTMPose experimental（BGR，320→512） | `拒绝 去 不` | 0.25 / 0.24 / 0.53 | 41.16 s | 同一视频、同一权重、同一投票协议 |

这不是“HRNet 更准确”的证据：两者都是 CSL-Daily 权重在 CE-CSL 上的跨域输出，且没有同域真值；它证明的是**官方 HRNet 通道可运行、输出稳定、与 RTMPose 有可观测差异**。

几何对照（`dev-00015` 前 20 帧，63 个 SLRT 选点，512 参考系）：

- HRNet vs RTMPose：平均 **3.59 px**、中位 3.03 px、P95 7.85 px、最大 13.59 px；分组：躯干 3.65、手 3.34、嘴 4.60 px。
- 置信度均值：HRNet 0.848、RTMPose 0.811。
- 同一 HRNet 的 RGB vs BGR 输入：平均 **1.36 px**、中位 1.17 px、P95 3.14 px；颜色顺序有可测但不大的影响。上游 `gen_pose.py` 走 RGB；本地进程内 `HRNetFrontend` 当前走 BGR，仍属未验证路径，支持的是本脚本的 NPZ 流程。

## 3. SLT 资产

### 3.1 已补齐（数据无关部分）

- **base mBART**：`facebook/mbart-large-cc25`（作者未公布其 `pretrained_models/mBart` 的具体来源；这里按 notebook 中 `mBart` 的角色使用公开 cc25 模型），下载到 `models/external/SLRT_Online/pretrained_models/mBart`；`pytorch_model.bin` 2,444,517,405 B，SHA256 `6e55bd7b…dcf84`。
- **CSL gloss 侧**：`scripts/prepare_slt_gloss_assets.py` 生成 `mBART_zh_gloss_assets/`：
  - `gloss_embeddings.bin`：2,009 行（9 个特殊/语言别名 + 2,000 个 CSL gloss），每行 1,024 维；
  - `gloss2ids.pkl`：连续 ID，`<s>`=0、`<pad>`=1、`</s>`=2、`<unk>`=3、`<mask>`=4、`zh_CN`=5、`de_DE`=6、`zh_CSL`=7、`de_DGS`=8；
  - 2026-10-25 校验：ID 连续、所有 gloss 均有 embedding、特殊 token 顺序符合 `GlossTokenizer` 断言。
- **全词表 fallback**：`scripts/assemble_slt_mbart_fullvocab.py` 生成 `mBART_zh_fullvocab/`（配置/模型/tokenizer 用硬链接，`old2new_vocab.pkl`、`map_ids.pkl` 为 250,027 项 identity map，gloss 资产硬链接）。它可让 G2T 配置在数据到位后直接训练，但**文本词表没有按 CSL 语料剪枝，不是作者发布的 `mBART_zh`**；`README_FALLBACK.md` 已写明。

### 3.2 仍依赖 CSL-Daily 数据

作者的 `prune_embedding.ipynb` 需要 `csl.train/dev/test` 中的 `text`/`gloss` 统计来构造文本子词剪枝和 `old2new_vocab.pkl`。当前没有这些文件，所以：

- 不能生成作者等价的 `mBART_zh`；
- 不能训练 CSL-Daily G2T；
- 不能复现论文 20.63 BLEU。

一旦数据到位，可按 notebook 重跑文本侧剪枝；gloss 侧与 base 模型已经就绪。

### 3.3 SLT 预测转换

`scripts/convert_slrt_predictions_to_slt.py` + `slbridge/slrt/slt_export.py`：

- 输入：`prediction_slide.py` 的 `{split}_results.pkl` + 原句级 `csl.dev/test`；
- 输出：SLT 需要的 `csl_pred.dev/.test` pickle（保留 `text`/`num_frames`，`gloss` 换成预测值，`gls_ref` 保留原参考）；
- 默认使用 `window_greedy_7_gls_hyp`，缺省回退 `naive_greedy_gls_hyp`；缺样本默认报错，`--allow-missing` 仅用于诊断并统计空预测。

```powershell
.conda-env\python.exe scripts\convert_slrt_predictions_to_slt.py `
  --results artifacts\...\test\test_results.pkl `
  --sentence-meta Paper-algorithm\SLRT-main\data\csl-daily\csl.test `
  --output Paper-algorithm\SLRT-main\Online\CSLR\results\online_slr_csl\prediction_slide\csl_pred.test `
  --summary artifacts\...\test\csl-pred-test-summary.json
```

## 4. 仍未补齐

| 缺口 | 状态 | 说明 |
|---|---|---|
| `sentence_frames-512x512.zip` | **缺** | CSL-Daily 官方视频帧；没有它就无法做 RGB 流同域评测。作者 Google Drive 包不含此文件。HF `csl-daily-sentence-crop` 的 `split_*.zip` 是**未裁剪原始视频**（抽查帧数 898/314/256/526 vs 标注 146/146/200/490，最佳句段起点 25/152/16/23 帧），不能直接当 sentence frames；需官方 frame zip 或句子起止元数据 |
| CSL-Daily G2T `best.ckpt` / 作者 `mBART_zh` | **缺** | 本地发布包没有；`mBart_zh_t2g` 是 Spoken2Sign T2G，不是 Online G2T；只有全词表 fallback，不能称等价复现 |
| 官方 `prediction_slide.py` 旧依赖栈 | **未装（阻塞）** | 官方 requirements 是 torch 1.9.0+cu102、transformers 4.11.3、timm 0.4.12、`Lintel`、ctcdecode、wandb；CUDA 10.2 与 RTX 5060 Ti 不兼容，`Lintel` PyPI 0.1.0 是文档检查器而非视频解码包。需容器/另一台机器或现代 Torch 移植；本轮 legacy 环境只覆盖 HRNet/Faster-RCNN 提取 |

已补齐并可核对：CSL-Daily 句级标注（18,401 / 1,077 / 1,176）与关键点（20,654 个视频，`[T,133,3]` float16）逐条匹配——0 缺失、0 帧数/形状不匹配；`gloss2ids.pkl` 与 `csl_s2g_gloss2ids.pkl` 均为 2,004 项且内容相同；TwoStream-SLR S2G 为 912 键、2,004 类。官方 pkl 的 confidence 最大约 1.71（超出本地 `select_keypoints` 的 [0,1] 限制）；官方 `slide_csl-daily.yaml` 的 `heatmap_cfg` 未启用 threshold，因此不影响官方 Gaussian 生成，但本地 NPZ 路径不能直接喂这份 pkl。

`scripts/check_slrt_reproduction_assets.py` 当前输出：**唯一缺失的数据文件是 `sentence_frames-512x512.zip`**；权重/配置/工具全部 `present`。

## 5. 复现命令（数据到位后）

```powershell
# 1) 资产清单
.conda-env\python.exe scripts\check_slrt_reproduction_assets.py

# 2) 官方目录（CSL-Daily keypoints/meta 已从作者包挂到 data/csl-daily；S2G 已复制）
.conda-env\python.exe scripts\setup_slrt_official_layout.py `
  --data-source 'F:\model_weigth\data\csl-daily' `
  --two-stream-checkpoint 'F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_s2g\ckpts\best.ckpt'

# 3) 官方 HRNet 关键点（若有视频；否则使用作者 pkl）
F:\DSH-envs\slrt-legacy\python.exe scripts\extract_slrt_hrnet_pose.py `
  --video 'D:\path\csl-daily-test.mp4' `
  --output artifacts\slrt-hrnet\csl-daily-test.npz --device cpu

# 4) 本地 SLRT（NPZ 路径）
.conda-env\python.exe -m slbridge.slrt.cli video --video 'D:\path\csl-daily-test.mp4' `
  --config configs/slrt-csl-hrnet-npz.example.json `
  --pose-backend hrnet-npz --pose-npz artifacts\slrt-hrnet\csl-daily-test.npz `
  --output artifacts\slrt-hrnet\csl-daily-test.json --require-gloss

# 5) 官方入口（需 legacy 环境/容器）
#    cd Paper-algorithm\SLRT-main\Online\CSLR
#    python -m torch.distributed.launch --nproc_per_node 1 --use_port 29999 --use_env prediction_slide.py --config=configs/slide_csl-daily.yaml
```

注意：官方入口仍需要旧依赖（torch 1.9.0+cu102、transformers 4.11.3、timm 0.4.12、ctcdecode、wandb 等）和 `prediction_slide.py` 的完整数据加载；本轮的 legacy 环境只覆盖 HRNet/Faster-RCNN 提取，不代表官方入口可在该环境直接运行。

## 6. 验证与证据

- 新增测试：`tests/test_slrt_slt_export.py`、`tests/test_slrt_official_layout.py`，**9 passed**（`.conda-env`）；SLRT 定向合计 44 passed。
- 作者包校验：`csl-daily.train/dev/test` = 18,401 / 1,077 / 1,176；keypoints 20,654 个视频，`[T,133,3]` float16；测试集 1,176 条全部有 keypoints，0 帧数/形状不匹配；confidence 最大约 1.71。`gloss2ids.pkl` 与 `csl_s2g_gloss2ids.pkl` 均 2,004 项且相同。
- 数据文件哈希：`keypoints_..._{iso,noniso}.pkl`（同一文件）SHA256 `23f8ab10…7461`，1,971,425,633 B；`csl.train` `ef77f56e…481f`；`csl.dev` `f14660c4…bd4a`；`csl.test` `ebbbb48a…2458`。
- 数据一致性/哈希汇总：`artifacts/slrt-gdrive-20261025/data-consistency.json`（1,303 B；含 18,401/1,077/1,176、20,654 视频、0 缺失/0 形状不匹配、confidence 0.005–1.713）。
- TwoStream S2G：504,360,673 B，SHA256 `54d2cb65c7d47085e1868326abbbb3a6e21774c383f57cd29f999d2009609739`，912 键、2,004 类。
- HRNet 提取：`artifacts/slrt-hrnet-20261025/dev-00015-full-hrnet.json`（202 帧、0 缺失、733.674 s、Torch 1.13.1+cpu、mmdet 2.25.3、mmpose 0.29.0）。
- 本地推理：`hrnet-full-video.json`（`拒绝 不`、202 窗、0 丢帧、`pose.experimental=false`）；`rtmpose-full-video.json`（`拒绝 去 不`）。
- 关键点对照：`artifacts/slrt-hrnet-20261025/frontend-compare-first20.json`；RGB/BGR 对照 `dev-00015-first20-hrnet-bgr.npz`。
- 资产清单：`scripts/check_slrt_reproduction_assets.py` 输出（见第 4 节）；本次保存为 `artifacts/slrt-hrnet-20261025/asset-inventory.json`（9,477 B，2026-10-25）。
- 边界：HRNet/RTMPose 结果均为 **CSL-Daily 权重在 CE-CSL 上的跨域输出**；没有同域真值，不报告 WER/准确率，不把 `拒绝 不` 与 `拒绝 去 不` 的差异归因为“HRNet 更准”。
