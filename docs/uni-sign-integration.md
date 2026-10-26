# Uni-Sign Pose-only 接入与运行说明（2026-10-23）

## 结论先行

当前项目已以**可选实验后端**接入 Uni-Sign `csl_daily_pose_only_slt.pth`：

- 官方权重、`google/mt5-base` 与固定源码提交均已下载并通过哈希/大小核验。
- 独立 Conda 环境在 RTX 5060 Ti 16 GB 上完成 PyTorch CUDA 与 RTMPose CUDA 推理验证。
- 真实中文手语视频可输出中文句子；同一视频重复推理文本一致。
- 已实现滑动窗口实验、历史姿态复用、单推理任务、过期结果丢弃、临时候选/多窗口一致候选标记。
- 现有 RF / ONNX / CTC / 桌面测试保持通过。

**重要边界：** 本地可用于验收的中文视频来自 CE-CSL，而该权重是 CSL-Daily Pose-only SLT 模型。两者不是同一数据域，因此 CE-CSL 上的分数不能当作 CSL-Daily 准确率或产品验收。本轮同时修正了 16:9 视频将 `x/W, y/H` 分别归一化造成的横向几何压缩；旧 4.72 BLEU 基线已被修正后的 7.99 BLEU 结果取代，但纠正预处理不等于解决跨域问题。

滑动窗口不是 Uni-Sign 原生流式模型，也没有实现 mT5 KV Cache 或因果编码器改造。

## 资源与磁盘分配

本轮坚持把大文件放在 E 盘项目目录：

| 路径 | 用途 | 占用（约） |
|---|---|---:|
| `E:\githubproject\SignLanguage-Bridge\.conda-env` | 独立 Python 3.10 + PyTorch + 桌面/RTMPose 依赖 | 8.3 GB |
| `E:\githubproject\SignLanguage-Bridge\.cache` | pip 缓存 | 4.9 GB |
| `E:\githubproject\SignLanguage-Bridge\models\external\uni_sign` | 权重、mT5、模型包、RTMPose 缓存 | 3.4 GB |
| `E:\githubproject\SignLanguage-Bridge\models\external\uni_sign_source` | 固定提交的上游源码副本（本地） | < 0.1 GB |

实测 E 盘在开始前约剩余 **71.45 GB**，上述资产和环境完成后约剩余 **54.61 GB**。下载脚本会在写入前检查剩余空间，并保留可配置的安全余量（默认 2 GB）。

Hugging Face 缓存与 pip 缓存均显式指向项目目录；RTMPose 缓存通过 `TORCH_HOME` 指向 `models/external/uni_sign/rtmlib-cache`。不要把 `models/`、`.cache/`、`.conda-env/` 提交到 Git。

诊断时曾直接调用 rtmlib，产生了 `C:\Users\ADMIN\.cache\rtmlib` 约 150 MB 的一次性重复缓存；正式代码已固定使用 E 盘项目缓存，但该诊断副本未在任务中自动递归删除。

## 许可与源码限制

- Uni-Sign 权重：**CC BY-NC 4.0**，要求署名，禁止商业使用。
- `google/mt5-base`：**Apache-2.0**。
- 固定提交 `eed438bcb49e30405cd6ccdfcccca330c134e830` 的上游源码树**没有 LICENSE 文件**。本仓库不复制、不再分发该源码；商业或再分发用途须先向作者取得书面澄清。
- 本项目只把上游源码作为本地外部依赖使用，并记录固定提交与 ZIP SHA256。

## 固定版本

| 项 | 固定值 |
|---|---|
| 模型仓库 | `ZechengLi19/Uni-Sign` |
| 模型 revision | `182913ca7a8b7dd9837d7d6f10717353f4408f80` |
| 权重 | `csl_daily_pose_only_slt.pth` |
| 权重 SHA256 | `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c` |
| 权重大小 | `1,186,925,112` bytes |
| mT5 revision | `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f` |
| 源码提交 | `eed438bcb49e30405cd6ccdfcccca330c134e830` |
| 源码 ZIP SHA256 | `8022b073bb1de962a8e3f424e9858ee50465e46228bc8854b966d54107d8cc97` |

## 环境安装（Windows / PowerShell）

官方旧依赖是 Python 3.9 + PyTorch 2.1.1 + CUDA 12.1；本机 RTX 5060 Ti 属于 Blackwell，旧内核不适用。本机验证组合为 Python 3.10.22、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128。

```powershell
# 如网络需要代理，只对当前命令会话设置
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'

# 下载固定源码 / 权重 / mT5
python scripts\download_uni_sign.py
python scripts\download_uni_sign.py --check-only

# 创建并安装独立环境；脚本会把 pip 缓存和临时目录放到 E 盘项目
powershell -ExecutionPolicy Bypass -File scripts\setup_unisign_env.ps1
```

如果环境已创建，只需复跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_unisign_env.ps1 -SkipCreate
```

RTMPose 固定使用上游 `demo/rtmlib-main`，以 editable、`--no-deps` 安装；PyTorch 必须先于 ONNX Runtime/RTMPose 导入。Windows 上反向顺序会让 `torch` 的 cuDNN DLL 与 ONNX Runtime 的 cuDNN 会话冲突并报 `WinError 127`。

## 权重与模型包

下载脚本生成：

```text
models/external/uni_sign/
  csl_daily_pose_only_slt.pth
  mt5-base/
  model_bundle/manifest.json
  download_manifest.json
  rtmlib-cache/
```

模型包 manifest 使用相对路径，可整体移动。下载脚本在最终文件完成前只写 `.downloads/` 暂存目录；只有大小和哈希都通过的完整文件才会被原子替换到正式路径。残缺文件不会被标记成功。

## 独立真实推理

```powershell
$env:PYTHONPATH = 'E:\githubproject\SignLanguage-Bridge'

.\.conda-env\python.exe scripts\verify_uni_sign.py `
  --bundle models/external/uni_sign/model_bundle `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --device cuda --runs 2 `
  --output artifacts/unisign-verification/cecsl-test-00001-runs2-final.json
```

模型加载时执行一次姿态和模型 warmup，因此首次加载约 27 秒；之后同一窗口推理不再承担冷启动。实际窗口缓存复用后，完整 247 帧文件的第二次推理约 1.14 秒。

本地 CE-CSL 参考句只用于说明**数据域不匹配**，不用于得出 Uni-Sign 准确率：

- 参考句（CE-CSL）：`40平米的房子比较适合一个人住。`
- Uni-Sign 输出：`他用白色的牙齿使人感到害羞。`
- 两次推理文本完全一致。
- 单样本字符级 BLEU / 编辑相似度不代表 CSL-Daily 质量。

## CE-CSL 完整测试集的跨域指标

在本地 CE-CSL test split 的 **500/500** 条视频上直接运行 CSL-Daily 权重，不做微调。旧 W/H 归一化存在已确认的 16:9 横向压缩，下表同时保留被修正基线与本轮正式结果：

| 指标 | 旧 W/H 基线 | 修正短边（正式） |
|---|---:|---:|
| 严格字符匹配 | 0 / 500（0.00%） | **0 / 500（0.00%）** |
| BLEU-zh | 4.72 | **7.99** |
| chrF | 6.33 | **9.38** |
| 字符级 ROUGE-L F1 | 0.2449 | **0.3004** |
| 平均字符编辑相似度 | 17.88% | **22.51%** |
| 空预测 | 0 | **0** |

配对 Bootstrap 的 BLEU-zh 增量为 **+3.27（95% CI +2.32 至 +4.26）**，字符编辑相似度增量为 **+4.64 个百分点（95% CI +3.31 至 +6.01）**。这说明归一化修复有效，但结果仍是明显跨域表现，不能称为 CSL-Daily 或通用中国手语准确率。CE-CSL 有中文参考句和 Gloss；部分签者 BLEU-zh 仍低于 1.11，剩余误差主要应按签者、场景和模型域差异继续分析。详细口径、逐译者结果和逐句对比见 [CE-CSL 跨域指标报告](reports/unisign-cecsl-cross-domain-accuracy-2026-10-23.md)。
## 滑动窗口实验

离线视频、32 帧窗口、12 帧步长、640 像素最大宽度：

```powershell
.\.conda-env\python.exe scripts\benchmark_uni_sign.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --device cuda --window-frames 32 --stride-frames 12 `
  --max-width 640 --max-seconds 8.25 `
  --output artifacts/unisign-benchmark/cecsl-test-00001-32-12-final.json
```

已实现的调度边界：

- 窗口只保留有限帧数，旧帧按上限淘汰。
- 重叠窗口按 `(source_id, frame_index, shape)` 复用 RTMPose 关键点。
- 模型 worker 同时最多一个任务；新窗口在旧任务运行时不排队堆积。
- 每个结果带窗口起止时间、姿态缓存命中数、模型耗时、RTF 和 GPU 显存。
- 结果带 `temporary` / `confirmed` 状态；`confirmed` 只表示多个连续窗口文本一致，**不是人工确认**。
- 会话层继续使用 generation/epoch/request 规则，过期结果不能覆盖新结果。

## 桌面端

完整环境已包含 PySide6 与 MediaPipe。设置页新增“加载 Uni-Sign（实验）”按钮；主界面仍保留现有 RF / ONNX / CTC 模型。

启动 Uni-Sign 连续视频实验：

```powershell
.\.conda-env\python.exe -m slbridge.desktop_cli `
  --unisign `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --continuous --device cuda `
  --continuous-window-frames 32 --continuous-stride-frames 12
```

启动摄像头实验：

```powershell
.\.conda-env\python.exe -m slbridge.desktop_cli `
  --unisign --camera 0 --continuous --device cuda `
  --continuous-window-frames 32 --continuous-stride-frames 12
```

也可以在设置中选择普通模型目录或点击 Uni-Sign 实验按钮。Uni-Sign 不会被自动设为默认生产模型。

offscreen Qt 真实桌面链路可用同一脚本验证：`scripts\smoke_unisign_desktop.py`。

同一 `RecognitionSession` 异步链路可用无 GUI 冒烟脚本验证：

```powershell
.\.conda-env\python.exe scripts\smoke_unisign_session.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --device cuda --window-frames 32 --stride-frames 12 `
  --output artifacts/unisign-session/cecsl-test-00001-32-12.json
```

## 输入协议

Uni-Sign Pose-only 使用 RTMPose Wholebody 的 **133 个关键点**，不是 MediaPipe 225 维协议。默认 `pose_normalization=isotropic_short_side`：横纵坐标共同除以 `min(width, height)`，用于保持非方形视频的动作几何；`wh` 仅用于复现旧基线。

| 图分支 | 点数 | 归一化 |
|---|---:|---|
| body | 9 | 全窗口 body 外接框缩放并平移 |
| left hand | 21 | 减去手腕点后按 body scale 缩放 |
| right hand | 21 | 减去手腕点后按 body scale 缩放 |
| face | 18 | 减去 nose 参考点后按 body scale 缩放 |
| 第 3 通道 | 1 | **原始关键点置信度**，不是 z 坐标 |

超过 256 帧时使用确定性等距采样并保留首尾；短窗口不采样。正式推理使用 `model.eval()`、`torch.inference_mode()`，不读取测试参考句。

### 输入方向校正（2026-10-24）

`runtime.pose_orientation` 默认为 `preserve`。对于已确认需要校正的输入，可显式配置为 `mirror`；校正同步反射x坐标、交换左右身体／手／面部点及其置信度。文件推理和完整采集片段均使用同一入口，结果质量字段记录实际方向。此设置与桌面预览镜像独立，后端不会根据预测文本、测试参考句或签者姓名自动猜测方向。

单个视频可直接校验：

```powershell
.\.conda-env\python.exe scripts\verify_uni_sign.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\dev\B\dev-00069.mp4' `
  --pose-orientation mirror --device cuda `
  --output artifacts\unisign-direction-check.json
```

为摄像头或桌面实验固定输入方向时，可在所用模型包manifest的`runtime`对象中设置`"pose_orientation": "mirror"`；模型包默认保持原方向。CE-CSL的开发集方向映射只适用于对应录制数据，不自动用于陌生摄像头。

### 输入归一化必须跟随「训练数据的生产者」（2026-10-25 补充）

`pose_normalization` 不是全局最优项，而是**协议字段**：同一权重换归一化可差 6 BLEU。判断依据是训练时 pose 是怎么产出的：

| 权重/数据域 | 训练时输入 | 应使用的归一化 |
|---|---|---|
| CSL-News 预训练（`csl_stage1_weight` / `csl_stage2_weight`） | 作者 `demo/pose_extraction.py` 用 rtmlib `Wholebody`（默认 `lightweight`）+ **`x/W, y/H`**，片段 204×170 等非方形 | **`wh`**（实测比 isotropic 高 +6.35 BLEU / +7.17 pp） |
| CSL-Daily 微调（`csl_daily_pose_only_slt`） | 发布 pkl 的 `w_h` 为 **1280×1280 / 1080×1080（正方形）**，`wh` 与 `isotropic` 等价 | 非方形输入用 **`isotropic_short_side`**（保持长宽比，最接近方形训练分布；与 CE-CSL 历史结论一致） |

注意：上游 `demo/pose_extraction.py` 与本地运行时是**同一个姿态模型**（rtmlib `rtmw-dw-l-m` + `yolox_tiny`），逐点差异 78–82% 在 1 px 内；前端差异主要是归一化，不是模型。

### 文件推理的采样帧优化（`pose_sample_only`，2026-10-25）

`_video_pose_window` 默认对解码出的**每一帧**跑 RTMPose，随后 `prepare_pose_window(max_frames=256)` 只保留 256 帧——超过 256 帧的片段（本数据集 39%）有 30–50% 的姿态计算被丢弃。把 `runtime.pose_sample_only` 设为 `true` 后：

- 先用容器的帧数算出确定性采样下标，跳过帧只 `grab()`（不解码成 BGR），只对采样帧跑 RTMPose；
- 张量与 `sampled_indices` 与全帧路径**逐位一致**（`tests/test_unisign_pose_sampling.py`，7 项）；
- 实测（真实视频，经后端入口）：425/450/475 帧片段 **1.65× / 1.74× / 1.94×**；≤256 帧片段 1.0×（本来就全用）；
- 按全量帧分布推算：跳过 **14.7%** 的姿态提取，姿态阶段 **1.17×**，e2e **5.54 s → 4.73 s**；
- 默认**关闭**，需在模型包 manifest 的 `runtime` 里显式加 `"pose_sample_only": true`；只影响文件推理，连续窗口/摄像头路径（`_prepare_frames`）不受影响。

结果里新增 `pose_sample_only` / `pose_skipped_frames` 两个质量字段，`quality` 中同时记录 `source_frames` 与 `skipped_frames`，便于审计实际跳过了多少帧。

### 跨帧复用检测 bbox（`pose_detect_stride`，2026-10-25，**默认关闭；已在 CSL-News 上通过精度验收**）

检测器（YOLOX-tiny @416×416，约占每帧姿态成本的 60%）可以每 K 帧跑一次、中间帧复用上一次的人体框：

- `runtime.pose_detect_stride`（默认 `1` = 每帧检测，范围 1–30）；`>1` 时 `RtmposeExtractor.extract` 改走 `det_model` / `pose_model` 两步，并在输入源变化时由 `_clear_for_source` 调用 `reset_detection()` 丢弃旧框。
- 实测（6 条 × 120 帧，仅姿态阶段）：stride 2/3/5 → **1.45× / 1.70× / 1.95×**；关键点在 2 px 内的比例 98.9% / 98.2% / 97.4%；但个别帧的规范姿态最大差可达 0.36。
- **精度验收（2026-10-25 完成，CSL-News 300 条，stage1 权重、wh 归一化）**：stride 5 vs stride 1 的配对 Δ 编辑相似度 **+0.0036，95% CI [−0.0051, +0.0125]**（含 0，无显著差异），胜/负/平 91/87/122；BLEU-zh 22.02→22.36。同一批上姿态阶段 5,206→2,932 ms、e2e 5.62→3.33 s、RTF 0.649→0.390。
- **第二个数据集独立复现（2026-10-25，CE-CSL test 500，目标域=日常手语）**：`pose_detect_stride=5` vs `1` 的配对 Δ编辑相似度 **+0.0068，95% CI [−0.0011, +0.0152]**（含 0），胜/负/平 91/88/321，285/500 预测逐字相同；BLEU-zh 12.3019→12.1470（−0.16）、编辑相似度 30.63%→31.32%；端到端 **5.55→3.42 s（1.62×）**、姿态阶段 5,213→3,077 ms（1.69×）。
- **方向映射绑定姿态协议**：`pose_detect_stride` 会进入 pose protocol，因此旧的 dev 校准映射不能直接用于 stride=5 —— 必须用 `calibrate_unisign_cecsl_direction.py --pose-detect-stride 5` 重新校准（本次重跑后方向与旧映射完全一致，说明 stride 不影响方向判定）。协议格式版本（schema_version 2→3）的漂移已允许，但任何语义字段（归一化/mode/设备/帧数/方向/stride）变化都会硬失败。
- 验收边界：**只覆盖静态单人片段**（204×170 新闻、1080p CE-CSL，固定机位）。移动镜头、多人、快速大位移场景必须重新验证；因此默认仍为关闭。测量脚本 `scripts/benchmark_detector_reuse.py`，验收运行 `artifacts/cslnews-eval/pinned-wh-stride5-20261025` 与 `stage1-wh-stride5-20261025`。
- 与采样帧优化可叠加：两项同时开启时实测（300 条、wh）姿态提取 5,206→**2,483 ms**、e2e 5.62→**2.89 s**、RTF 0.649→**0.357**、吞吐 10.7→**20.8 条/分**（端到端 **1.95×**）。
- **两个开关都会进入姿态缓存身份**：`scripts/evaluate_unisign_cslnews.py` 在非默认值时把 `pose_sample_only` / `pose_detect_stride` 写入 pose protocol（默认值不写，因此历史运行目录的协议保持字节一致、仍可续跑）。不加这条会让 stride-5 的姿态被 stride-1 的运行静默复用。


### 适用范围：只做「动作结束后识别」（2026-10-25 产品决定）

Uni-Sign 相关模型**只用于动作结束后的整段识别**（`recognize_file` / `recognize_clip`），不做同步字幕。
因此连续窗口的「双手可见观察门控」与「稳定候选（confirmed）」**不是该模型的功能依赖**；
连续窗口路径上实现的门控放宽仅作为文件回放的默认语义保留。

post-hoc 路径实测（CSL-News 片段、冷姿态缓存、RTX 5060 Ti）：

| 片段帧数 | 时长 | `recognize_clip`（内存帧） | 开 `pose_detect_stride=5` 后 | 加速 |
|---:|---:|---:|---:|---:|
| 76 | 2.5 s | 1.79 s | **1.16 s** | 1.55× |
| 87 | 2.9 s | 2.10 s | **1.39 s** | 1.52× |
| 175 | 5.8 s | 4.00 s | **2.42 s** | 1.65× |
| 250 | 8.3 s | 5.59 s | **3.43 s** | 1.63× |
| 均值（4 条） | — | 3.37 s | **2.10 s** | — |

- `recognize_file` 与 `recognize_clip` 耗时基本一致（视频解码仅约 40 ms），瓶颈仍是逐帧 RTMPose。
- 模型侧（姿态来自磁盘）下限约 **0.39 s**；`pose_detect_stride` 对两条入口都生效，
  `pose_sample_only` 只对 `recognize_file` 生效（采集时无法预知最终帧数）。
- **延迟随动作长度增长**：2–3 s 的动作 ≈ 1.2–1.4 s；5.8 s 的动作 ≈ 2.4 s；8.3 s 的动作 ≈ 3.4 s。

### CSL-News 评测（2026-10-25）

```powershell
# 项目集成权重（CSL-Daily）在本地 CSL-News 子集上：跨域，分数不可当 CSL-News 准确率
.\.conda-env\python.exe scripts\evaluate_unisign_cslnews.py `
  --output-dir artifacts\cslnews-eval\run-cslnews-20261025

# 其他发布权重（实验路径，不改生产校验）
.\.conda-env\python.exe scripts\evaluate_unisign_cslnews_checkpoint.py `
  --checkpoint models\external\uni_sign\csl_stage1_weight.pth `
  --pose-source upstream-pkl --output-dir artifacts\cslnews-eval\stage1-upstreampose-full-20261025

# 上游 pose 对照
.\.conda-env\python.exe scripts\evaluate_unisign_cslnews_upstream_pose.py --limit 379 `
  --output-dir artifacts\cslnews-eval\pinned-upstreampose-20261025
```

关键事实：`data/CSL-NEWS/archive_001` 的 1,694 条恰为 `CSL_News_Labels.csv` 第 1..1694 行，
落在上游 `datasets.py:527-532` 的 **train 段**；发布里**没有** CSL-News 下游微调权重。
因此 CSL-News 上的分数只能是「跨域回归」或「训练集拟合」，不得写成 CSL-News 准确率。

### CSL-Daily in-domain 复现（2026-10-25 完成）

```powershell
# 1) 下载作者发布的 pose 归档（4.59 GB），按 test_tmp_refs.txt 的 1,176 个 sample ID 抽 pkl
# 2) 跑复现
.\.conda-env\python.exe scripts\evaluate_unisign_csldaily_pose.py `
  --pose-dir artifacts\cslnews-eval\csl-daily-pose\pose-format `
  --refs models\external\uni_sign_source\out\eval_csl_daily_pose_only\test_tmp_refs.txt `
  --upstream-pres models\external\uni_sign_source\out\eval_csl_daily_pose_only\test_tmp_pres.txt `
  --output-dir artifacts\cslnews-eval\csldaily-indomain-20261025
```

- 结果：BLEU-4 **25.55**（论文 25.61，差 0.06），1,029/1,176（87.5%）预测与作者快照逐字相同。
- **必须复刻上游的格式后处理**：`fine_tuning.py:282-284` 在算指标前执行
  `' '.join(list(text.replace(" ", "")))`（注释 "fix mt5 tokenizer bug"）。不做这一步，
  同一份预测的 BLEU-4 会从 25.55 掉到 **0.0005**（每句退化成 1 个 token），而 chrF/ROUGE 正常。
- pkl 含 `start/end` 裁剪字段，必须按上游 `S2T_Dataset.load_pose` 处理，否则帧范围错位。

### CSL-Daily 本地视频端到端（2026-10-26，1,086 条 test 子集）

`D:\Zenodo\csl-daliy` 现有 **19,528 条**未裁剪原始 mp4（约 25.1 GB），与官方标注相交 test **1,086/1,176**（缺 90 条、偏长句）、dev 1,004、train 17,438。同一批"按官方 keypoints 对齐裁剪出的句子片段"上，视频端到端与官方 pose 口径对照：

| 口径 | 样本 | BLEU-4 | chrF | char ROUGE-L | 编辑相似度 | exact |
|---|---:|---:|---:|---:|---:|---:|
| 视频端到端（裁剪 mp4 → 本地 RTMPose → pinned ckpt） | 1,086 | **24.8157** | 21.7974 | 0.5273 | 0.4157 | 15 |
| 官方 pose（同 1,086 条） | 1,086 | **25.5846** | 22.4284 | 0.5351 | 0.4212 | 18 |
| 官方 pose（完整 1,176） | 1,176 | 25.5504 | 22.3893 | 0.5342 | 0.4204 | 19 |

- 命令链：`scripts/build_csldaily_video_selection.py` → `scripts/align_slrt_csl_daily_subset.py --stage align`（句段裁剪，SLRT 与 Uni-Sign 共用）→ `scripts/evaluate_unisign_csldaily_video.py`（bf16、beam=4、isotropic、max 256 帧）；官方 pose 对照用 `scripts/evaluate_unisign_csldaily_pose.py`，子集汇总用 `scripts/summarize_unisign_predictions.py`。
- **视频路径比官方 pose 低 0.77 BLEU / 0.0055 编辑相似度**（627 基线为 −0.87），差距来自本地 RTMPose 前端 + mp4v 重编码 + 近似句段裁剪，均属小量；不是权重差异。
- **参考文本必须统一**：上游 `test_tmp_refs.txt` 用半角标点，官方标注 `text` 用全角；1,086 条中 589 条仅标点不同，同一份预测两套参考 BLEU 差 **2.78**（24.82 vs 22.03）。跨运行比较 BLEU/chrF 必须先固定参考文件，本表用上游参考。
- 延迟：单进程且与其它 GPU 任务共享机器时 e2e 均值 6.11 s（P50 5.04 / P95 14.92），姿态 5.50 s、模型 0.59 s；**不可当独占延迟或产品 SLA**。
- 完整逐条对比（GT 句子 + GT Gloss + 两套预测、Excel）见 [1,086 条补充报告](reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md)；627 基线见 [627 条报告](reports/csldaily-video-slrt-unisign-2026-10-26.md)（其中 −3.82 BLEU 的口径已更正为 −0.87）。


### CE-CSL校准、评测与续跑

`scripts/calibrate_unisign_cecsl_direction.py`只接受官方515条dev标签，固定每签者首／中／末3条用于校准、两个四分位样本用于验证。方向在验证前锁定；验证分数不改变选择。它不会读取test标签选择方向。

```powershell
.\.conda-env\python.exe scripts\calibrate_unisign_cecsl_direction.py `
  --output-dir artifacts\unisign-cecsl-eval\direction-calibration

.\.conda-env\python.exe scripts\evaluate_unisign_cecsl.py `
  --split test --device cuda `
  --direction-map artifacts\unisign-cecsl-eval\direction-calibration\direction-map.json `
  --output-dir artifacts\unisign-cecsl-eval\test-with-locked-directions
```

评测保存完整运行协议，绑定权重、分词器版本、姿态参数、实际设备／dtype、解码设置、方向映射及输入视频SHA256。姿态缓存保持原方向，按协议分目录，解码前再应用方向；从缓存加载的张量保持只读。更换协议必须使用新输出目录，旧版无协议的预测不会被自动续用。历史缓存导入须另有明确的来源审计，不能将其计作重新提取视频。

当前修复及验收边界见`docs/reports/unisign-cecsl-direction-repair-2026-10-24.md`。锁定开发集方向后的测试成绩仍是已检查测试集上的回归结果。

### 连续窗口补充核查（2026-10-24）

Uni-Sign CLI和会话smoke脚本使用显式帧数就绪：收到请求的N帧后才构造候选窗口，保留8秒和字节预算硬上限。标称FPS用于步长换算和参数预检；普通模型默认的时间窗口行为保持不变。这修复了“N帧只覆盖(N-1)/FPS秒，却要求N/FPS秒才能就绪”的阻塞问题。

当前仍是完整窗口重复推理，输出整句中文；没有稳定词级Gloss、动作时间对齐或跨窗口模型状态复用。LLM表达辅助尚未接入。新的真实会话回放尚未交付候选，不能因窗口单元测试通过而称实时识别已验收。实验和后续路线见`docs/reports/unisign-followup-accuracy-streaming-2026-10-24.md`。

## 测试

```powershell
$env:PYTHONPATH = 'E:\githubproject\SignLanguage-Bridge'

# 现有桌面/回归环境（Python 3.12）
.\.venv-libras\Scripts\pytest.exe -q

# Uni-Sign 独立环境（Python 3.10）
.\.conda-env\python.exe -m pytest -q
```

方向修复回归为786 passed、10 skipped；Uni-Sign定向测试为34 passed、无跳过。后续窗口修复全量回归为797 passed、10 skipped，窗口／会话组合为42 passed，详见两份修复与核查报告。单元测试不依赖网络或 GB 级权重。真实 CUDA 推理单独作为实验证据，而不是普通单元测试。

## 仍未完成 / 不得宣称

- **CSL-Daily test 的 pose 口径已复现**（2026-10-25）：BLEU-4 **25.55** vs 论文 25.61，87.5% 预测与作者快照逐字相同（见上文「CSL-Daily in-domain 复现」）。帧采样为确定性 `linspace`，与上游 `random.sample` 不同，复现精度约 ±0.5 BLEU。
- **视频端到端只覆盖 1,086/1,176（2026-10-26）**：本地有 1,086 条 test 视频（缺 90 条偏长句），用近似句段裁剪 + 本地 RTMPose 得 BLEU-4 **24.82**（统一上游参考）；同批官方 pose 为 **25.58**。这不是完整官方 test，也不是论文协议，见[1,086 条报告](reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md)与[627 条报告](reports/csldaily-video-slrt-unisign-2026-10-26.md)。
- **CSL-News 上没有任何可对照论文的数字**：论文里 CSL-News 只是预训练语料；本地 1,694 条又落在上游 train 段。项目集成权重在本地 CSL-News 子集上 BLEU-zh 1.27（跨域），换 `csl_stage1_weight` 后 22.02–23.76（训练集拟合）——两者都**不是** CSL-News 准确率。
- 没有达到“稳定字幕从视频流开始计算 P95 <= 2 秒”的候选目标；当前单条实验视频的流级稳定候选 P95 约 4.59 秒。
- 没有实现原生流式 mT5、KV Cache 或因果编码器。
- 摄像头阻塞式实验与桌面异步链路的耗时不同；桌面端仍需真人长时、多场景验收。
- 没有训练或微调 CSL-Daily；没有下载完整 CSL-Daily 数据集（现有：pose 归档 4.59 GB、19,528 条未裁剪原始 mp4、官方句级标注与 keypoints；仍缺 90 条 test 视频和官方 `sentence_frames-512x512.zip`）。
- 没有进行商业使用授权评估；权重许可明确为非商业。




