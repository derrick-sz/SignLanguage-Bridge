# E00 审计来源、命令与哈希溯源（provenance.md）

- 审计员：auditor（P-1 实验团队 / 共享任务板 `task-1`）
- 审计日期：**2026-10-26**（本机 Asia/Taipei；GPU/系统快照时间 21:37:59 +08:00）
- 工作目录：`E:\githubproject\SignLanguage-Bridge`
- 纪律：只读审计，未修改 `experiments/p1/` 公共设施、`artifacts/`、`scripts/`、`slbridge/`；未 git commit/push；未运行 GPU 推理或模型加载。

## 1. 检索/核对来源

### 1.1 项目内文档（读取并引用）

| 文档 | 用途 |
|---|---|
| `docs/experiments/P-1/PLAN.md` | P-1 目标、模型候选、E00 范围 |
| `docs/experiments/P-1/docs/01_实验协议.md`（E00 节） | E00 操作与退出门槛 |
| `docs/experiments/P-1/docs/05_Agent执行指令.md` | 回报格式、入口“拟实现”约束 |
| `docs/experiments/P-1/config/model_registry.template.yaml` | 注册表结构 |
| `docs/agents/model-evaluation-pitfalls.md`（§1、§7、§10 及证据锚点） | 任务/指标口径、Windows/性能坑、SLRT 协议边界、历史成绩索引 |
| `docs/uni-sign-integration.md` | Uni-Sign 权重/SHA256/许可/环境/输入协议 |
| `docs/slrt-integration.md` | SLRT 在线 CSLR 资源、命令、环境、输入/解码协议、`slbridge-slrt` 入口陈述 |
| `docs/reports/slrt-online-reproduction-gap-2026-10-25.md` | 权重盘点（官方要求 vs 本地）、缺失项 |
| `docs/reports/slrt-assets-bootstrap-2026-10-25.md` | 资产补齐记录、S2G/mBART/HRNet 证据 |
| `docs/reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md` | 1,086 条 SLRT/Uni-Sign 运行记录（WER 0.3018 / BLEU 24.8157 的上游报告） |
| `README.md`（工作区版本）、`pyproject.toml` | README 与实际入口差异、console script 声明 |
| `experiments/p1/datasets/source_meta.json`、`experiments/p1/references/source_meta.json` | manifest/GT 生成来源与哈希 |
| `models/external/uni_sign/model_bundle{,_stride5}/manifest.json`、`download_manifest.json`、`extra-downloads.json` | Uni-Sign 固定 revision、权重/源码/mT5 哈希、许可声明 |
| `models/external/slrt_csl_bundle/manifest.json`、`slrt_hrnet_bundle/manifest.json` | SLRT bundle 的 checkpoint/vocab SHA256 记载 |
| `Paper-algorithm/SLRT-main/SLRT_ASSET_LAYOUT.json` | 官方目录硬链接/存在性记录 |
| `models/external/uni_sign_source/UPSTREAM_COMMIT.txt` | 源码 pinned commit + zip sha |

### 1.2 外部 URL（引用自本地文档；本次未联网复核）

| URL | 内容 | 引用处 |
|---|---|---|
| https://github.com/ZechengLi19/Uni-Sign | Uni-Sign 源码仓库（pinned `eed438bcb49e30405cd6ccdfcccca330c134e830`） | uni-sign-integration.md、docs/reports |
| https://huggingface.co/ZechengLi19/Uni-Sign | Uni-Sign 发布权重（revision `182913ca7a8b7dd9837d7d6f10717353f4408f80`）；CC BY-NC 4.0 | model_bundle/manifest.json、uni-sign-integration.md |
| https://huggingface.co/google/mt5-base （revision `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f`，Apache-2.0） | mT5 基座 | model_bundle/manifest.json |
| https://arxiv.org/abs/2501.15187 | Uni-Sign 论文（CSL-Daily pose-only test BLEU-4 25.61 等） | model_bundle/manifest.json、pitfalls |
| https://github.com/FangyunWei/SLRT （`Online/`、`TwoStreamNetwork/`） | SLRT 上游；本地镜像 `Paper-algorithm/SLRT-main`（无 .git；`latest_directory_commit=cb393becf08e47ecdf4196de4cd41449a4b02cbc` 引用自差距报告 §6） | slrt-integration.md、slrt-online-reproduction-gap |
| https://github.com/FangyunWei/SLRT/blob/main/Online/CSLR/gen_pose.py | 官方 HRNet/Faster-RCNN 前端脚本 | slrt-integration.md |
| https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html | HRNet-W48 DARK / Faster-RCNN 权重来源 | slrt-integration.md |
| https://github.com/kylemin/S3D | `s3ds_actioncls_ckpt` 来源（33,453,443 B，SHA256 `6b7f0229…b2d2`） | slrt-assets-bootstrap §1 |
| https://github.com/FangyunWei/SLRT/issues/107 （作者 Google Drive 补充包，引用自报告） | 作者包来源（S2G ckpt、data.zip） | slrt-assets-bootstrap §1.1 |
| https://huggingface.co/datasets/… `csl-daily-sentence-crop`（报告提及，HF） | 被排除的“未裁剪原始视频” | pitfalls §10 / bootstrap §4 |

**许可来源与缺口**
- Uni-Sign 权重：**CC BY-NC 4.0**（`docs/uni-sign-integration.md` §许可与源码限制；`model_bundle/manifest.json` license 字段；HF model card）。上游 pinned 源码树**无 LICENSE 文件**（同两处来源）。
- mT5-base：**Apache-2.0**（model_bundle/manifest.json）。
- SLRT（EMNLP24 Online / TwoStreamNetwork）：**UNKNOWN** —— 本地 `Paper-algorithm/SLRT-main` 与 `models/external/EMNLP24_Online`、`SLRT_Online` 均无 LICENSE 文件；上游仓库许可本次未联网核验。**缺口需向作者/仓库确认后才能用于再分发或商用判定。**
- 项目仓库根目录无 LICENSE 文件；`rf_mediapipe_legacy` 属项目本地产物，再分发许可 **UNKNOWN**。

## 2. 实际执行命令（可复跑）

所有产物在 `experiments/p1/e00/`（辅助脚本也在该目录，均为本次新建，只读性质）。

### 2.1 资产哈希（SHA256）
```powershell
# 24 个关键资产/数据文件：权重、mT5、HRNet/Faster-RCNN、RTMPose/YOLOX ONNX、源码 zip、
# cslr_best、mBART、gloss2ids/embeddings、三个 CSV、F: 的 keypoints/test/gloss2ids、上游 refs
powershell -File experiments\p1\e00\_hash_assets.ps1      # -> _hash_raw.txt（24 行）
# 7 个 SLRT TwoStream 权重（含 S2G 与 joint/keypoint/video、SingleStream s2t/s2g、Paper 镜像）
powershell -File experiments\p1\e00\_hash_slrt_weights.ps1  # -> _hash_raw_slrt.txt（7 行）
```

### 2.2 环境探测（无 GPU 推理/无模型加载）
```powershell
\.conda-env\python.exe experiments\p1\e00\_env_probe.py > experiments\p1\e00\_env_probe_conda.json
\.venv-libras\Scripts\python.exe experiments\p1\e00\_env_probe.py > experiments\p1\e00\_env_probe_libras.json
F:\DSH-envs\slrt-legacy\python.exe experiments\p1\e00\_env_probe.py > experiments\p1\e00\_env_probe_legacy.json
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu --format=csv
Get-CimInstance Win32_OperatingSystem / Win32_Processor / Win32_LogicalDisk
Get-ChildItem Env: | Where-Object { $_.Name -match 'CUDA|CUDNN|TORCH|NVIDIA' }
```

### 2.3 数据快照（不做全量视频哈希）
```powershell
\.conda-env\python.exe experiments\p1\e00\_data_snapshot.py > experiments\p1\e00\_data_snapshot_raw.json
# 内部：csv 行数/列/哈希；D:\Zenodo\csl-daliy mp4 计数与字节；pose-format 计数；
# F:\model_weigth\data\csl-daily 小文件哈希（大 keypoints 哈希取自 _hash_assets.ps1）；
# gzip pickle 句级标注条数；test_tmp_refs 与本地视频覆盖交集
```

### 2.4 Git / 入口 / 资产清单
```powershell
git rev-parse --abbrev-ref HEAD; git rev-parse HEAD; git status --porcelain=v1; git ls-files -v
git -c core.quotepath=false ls-files -z  # + 逐项 Test-Path（0 缺失）
\.conda-env\python.exe -m slbridge.slrt.cli --help
\.conda-env\python.exe -m slbridge.cli --help
\.conda-env\python.exe -m slbridge.desktop_cli --help
\.conda-env\python.exe -m experiments.p1.eval.evaluate --help
\.conda-env\python.exe -m pytest experiments\p1\tests\test_p1_infra.py -q   # 12 passed in 1.37s
\.conda-env\python.exe scripts\check_slrt_reproduction_assets.py            # 唯一 missing: sentence_frames-512x512.zip
Get-ChildItem models\external -Recurse -File -Include *.pth,*.ckpt,*.pt,*.bin,*.onnx
```

## 3. 哈希清单（本机实测，2026-10-26）

完整机器可读记录：`_hash_raw.txt`（24 行，格式 `sha256|bytes|relative_or_absolute_path`）、`_hash_raw_slrt.txt`（7 行）。核心摘录：

| 资产 | bytes | SHA256 |
|---|---:|---|
| `models/external/uni_sign/csl_daily_pose_only_slt.pth`（生产 pinned） | 1,186,925,112 | `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c` |
| `models/external/uni_sign/csl_daily_rgb_pose_slt.pth` | 1,197,268,371 | `c2ba5d7ae32f3fa40f7c6807b5004042be81fca192abe7d8ed7182c0d13470f6` |
| `models/external/uni_sign/mt5-base/pytorch_model.bin` | 2,329,735,129 | `180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614` |
| `models/external/uni_sign/faster_rcnn…d022e227.pth` | 165,671,301 | `d022e22795c17ecd53915ea8231bac43dd7b62ae2b322f1d59bf5b66af217bf9` |
| `models/external/uni_sign/hrnet_w48…f5726563.pth` | 254,976,311 | `f572656370050193134ef50e27745cdcdda54986d396ff78e6169608efcdc3c1` |
| `models/external/uni_sign/rtmlib-cache/…/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx` | 128,976,616 | `68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76` |
| `models/external/uni_sign/rtmlib-cache/…/yolox_tiny…onnx` | 20,283,006 | `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250` |
| `models/external/.downloads/Uni-Sign-eed438….zip` | 8,600,007 | `8022b073bb1de962a8e3f424e9858ee50465e46228bc8854b966d54107d8cc97` |
| `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt` | 499,554,423 | `e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d` |
| `models/external/EMNLP24_Online/CSL-Daily/meta/csl_iso_with_blank.vocab` | 28,821 | `0c4984f0dd3e7897b8c8d9e6e99a848b60814a5d9cf7a49baf7cc9b9cd124c76` |
| `models/external/SLRT_Online/pretrained_models/mBart/pytorch_model.bin` | 2,444,517,405 | `6e55bd7b105d99e6e25fafd4125cd5acb956ad2daef209b74fc4fb72113dcf84` |
| `…/mBART_zh_gloss_assets/gloss_embeddings.bin` | 8,810,325 | `b293fdade93d2d0406e606300852979ab4a77b79a7ce37ed9bf5f005a09e4ff5` |
| `…/mBART_zh_fullvocab/gloss2ids.pkl` | 22,293 | `393d48da67de647b34a1b156573e3b67d4cb31c92af42fbe079481dc64392732` |
| `…/mBART_zh_fullvocab/map_ids.pkl` 与 `old2new_vocab.pkl`（同哈希） | 2,238,436 | `df5ac826e15fb24d2aabfa193238707f8d777de3e46f2c6bcb56b1262adc6e63` |
| `…/SLRT_Online/pretrained_models/s3ds_actioncls_ckpt` | 33,453,443 | `6b7f0229a0797525d3b14074d8063f4a7dfc7ff2d65bf8ee80c5ecb5f9e0b2d2` |
| `F:/model_weigth/.../TwoStream/csl-daily_s2g/ckpts/best.ckpt`（及 Paper 镜像同哈希） | 504,360,673 | `54d2cb65c7d47085e1868326abbbb3a6e21774c383f57cd29f999d2009609739` |
| `F:/model_weigth/.../TwoStream/csl-daily_joint/ckpts/best.ckpt` | 1,748,231,929 | `4e83f36695bfe63db5413d00e04b9fbfd7a84dced12b88f11c7c9c90b0f2f7b3` |
| `F:/model_weigth/.../TwoStream/csl-daily_keypoint/ckpts/best.ckpt` | 1,748,231,929 | `fe884f416dfd1e11bc8fd7f51680f762cbd9b5c34c8e4cc9dfe1fd6c288ffe0c` |
| `F:/model_weigth/.../TwoStream/csl-daily_video/ckpts/best.ckpt` | 1,748,231,929 | `02812b445b114f9b696fc720ccaeb9f9394a7abb7ebf231e01b779a831daedec` |
| `F:/model_weigth/.../SingleStream/csl-daily_s2t/ckpts/best.ckpt` | 1,503,041,809 | `b6bdf7c2d8babe4d8ced724783f966f101d27468fd0494be864e73e03e8bd66c` |
| `F:/model_weigth/.../SingleStream/csl-daily_s2g/ckpts/best.ckpt` | 60,404,865 | `55bb1cd5710779d4647905b1415729fd8e4f77be55baacbddf1145f3c711e477` |
| `experiments/p1/datasets/manifest_1086.csv` | 454,569 | `22cd0e4969557ce7e40f96e83f3ff0f466c1e3081a883c267fbabe8f1119851f` |
| `experiments/p1/datasets/pilot_50.csv` | 16,871 | `f0f46125a6acf9768e696d1053450652ddca1146bb6ac2962be2b4efe4c99c4b` |
| `experiments/p1/references/gt_1086.csv`（EVAL ONLY） | 196,651 | `feb1b42daa74ed3efa992b50746c29f145550d381a80460669060ca00c087bb7` |
| `experiments/p1/references/pilot_50_diagnostic_EVAL_ONLY.csv` | 3,339 | `924bd4b7a7cf0be21e195f3a022f1e69e99bb1f9fe1ca5ddbcb4145c56b40046` |
| `F:/model_weigth/data/csl-daily/csl-daily-keypoints.pkl` | 1,971,425,633 | `23f8ab1063cfcfb882cc912f5ca918e77ee1dcc8d8ad2fac28c195b627107461` |
| `F:/model_weigth/data/csl-daily/csl-daily.test` | 105,205 | `ebbbb48a3ff4430106eac39977759398a82afe2f448ca6378a0012ade4972458` |
| `F:/model_weigth/data/csl-daily/csl-daily.train` | 1,282,351 | `ef77f56e517cf5ecf16a7bf8035de1965975164b92a4ca3348c920912365481f` |
| `F:/model_weigth/data/csl-daily/csl-daily.dev` | 99,259 | `f14660c46f9714425db2de33d0f037a8ed19cf945fd70ae9e0cde2efbd62bd4a` |
| `F:/model_weigth/data/csl-daily/gloss2ids.pkl` | 35,491 | `3b347a8a3c81ad2a2e79eaa28b89dcc0502cd09070f29175bb3e86f338afdf4d` |
| `models/external/uni_sign_source/out/eval_csl_daily_pose_only/test_tmp_refs.txt`（1,176 行） | 123,714 | `042cd94c44f9652f9fa127134565877a7a1cd1c58d2a12776c6698bb65150aab` |
| `models/external/uni_sign/model_bundle/manifest.json` | 2,191 | `462ea712c73ba0b0c7a91a4c6d5e596e8ff5bd58854dca53f8c73e03b8cdba7e` |
| `models/external/uni_sign/model_bundle_stride5/manifest.json` | 2,298 | `c4789c6f39712d9909692c6b873ae3687d5ee01021e2db98f54afad4e1ce6228` |
| `data/gestures/gesture_model.pkl`（RF 遗留） | 332,473 | `06205b684a7120d71b9e45a5000c42ca5ad48c46110ed6e613decf05dc8861b4` |
| `artifacts/experiments/libras-rf-pilot-20261009/.../weights/model.joblib` | 440,668 | `3450d09ccd44ae60cdf19d5dd90dc3d34f1810ad0edf65218ae1c2bb555db35d` |

未计算哈希（按任务约束或时间/规模原因）：19,528 个原始 MP4；`csl_daily_pose_format.zip`（4,594,281,295 B）；`Phoenix-2014T/ckpts/slt_best.ckpt`（1,434,238,071 B）与 `Phoenix cslr_best.ckpt`（456,022,647 B）——仅记录大小，未经本次实测哈希，注册表中相应字段为 null 并注明。

## 4. 交叉核对结果（实测 vs 文档）

- Uni-Sign pose-only：实测 `73bea87a…` = `model_bundle/manifest.json`/`download_manifest.json`/`docs/uni-sign-integration.md` 记载 ✅
- Uni-Sign rgb_pose：实测 `c2ba5d7a…` = `extra-downloads.json` 的 `sha256_expected` ✅
- mT5：实测 `180573b5…` = manifest + download_manifest ✅
- 源码 zip：实测 `8022b073…` = UPSTREAM_COMMIT.txt + bundle manifest ✅
- SLRT cslr_best：实测 `e9fdd3a6…` = `slrt_csl_bundle/manifest.json` + `docs/slrt-integration.md` ✅（两份副本同哈希）
- SLRT 词表：实测 `0c4984f0…` = 差距报告记载 ✅
- SLRT S2G：实测 `54d2cb65…` = bootstrap 报告 §6 记载 ✅（F: 与 Paper 镜像同哈希）
- HRNet/Faster-RCNN：文件名自带的短哈希（f5726563 / d022e227）与实测 SHA256 前缀一致 ✅
- 数据文件：`csl-daily.test` 实测 `ebbbb48a…` = bootstrap §6 记载 ✅；keypoints `23f8ab10…` = 记载 ✅；`csl.train/dev/gloss2ids` 哈希与 bootstrap §6 一致 ✅
- 官方句级标注条数：test/train/dev = 1,176/18,401/1,077（实测 gzip pickle 载入），与报告一致 ✅

## 5. 未验证 / 不能声明

- **未运行任何 GPU 推理或模型加载**（团队约定；E01 负责）。model_registry 中 READY 均为“资产+历史运行记录/本地入口可导入”级证据，不是本轮 smoke PASS。
- 未联网复核上游 GitHub/HF 的当前 commit/许可/模型卡（引用的 URL 与 pinned SHA 均来自本地文档/manifest；检索日期=文档日期）。SLRT 上游许可未查到 → UNKNOWN。
- 未验证 Docker daemon/GPU passthrough、官方 `rzuo/pose:sing_ISLR` 镜像（沿用既有报告，标未验证）。
- 未启动 API/桌面服务；未验证 `http://127.0.0.1:8765/docs`。
- 未对 19,528 个视频/4.59 GB pose zip 做哈希（明确排除）。
- 未修改、未提交任何代码；本目录外的仓库状态与审计前一致。
