# Uni-Sign 预训练权重与 SignLanguage-Bridge 接入验收报告

- 日期：2026-10-23（Asia/Taipei）
- 仓库：`E:\githubproject\SignLanguage-Bridge`
- 分支：`codex/model-contract-auto-captions`
- HEAD：`cf291c2 feat: add keypoint capture workbench`
- 状态：工作树原有未提交改动与本轮改动共存；本轮未提交、未合并、未发布 PR。

## 结论

本轮完成了固定版本资源下载、独立环境、模型严格加载、真实视频推理、可选 Backend、滑动窗口、桌面入口、真实摄像头路径和完整回归测试。

已跑通的最小真实链路：

```text
真实中文手语 MP4
  -> OpenCV 解码
  -> RTMPose Wholebody 133 关键点（CUDA）
  -> Uni-Sign 四分支预处理（body/left/right/face，置信度第 3 通道）
  -> PyTorch Uni-Sign + mT5（CUDA bfloat16）
  -> 中文句子
```

但**没有同域 CSL-Daily 测试集**。本地可用视频来自 CE-CSL，与本模型训练数据域不同，因此：

- 模型加载与工程链路：通过。
- 重复推理一致性：通过。
- CSL-Daily 同域准确率：NOT_RUN。
- CE-CSL 500 条跨域测试：修正短边归一化后严格匹配 0/500，BLEU-zh 7.99，chrF 9.38，字符编辑相似度 22.51%；旧 W/H 基线为 4.72 / 6.33 / 17.88%，已标记为被修正结果。
- 输入协议修复：旧脚本对 16:9 视频按 `x/W, y/H` 分别归一化，已改为默认短边等比例归一化；配对 Bootstrap BLEU 增量 +3.27（95% CI +2.32 至 +4.26）。
- 稳定字幕 P95 <= 2 秒候选目标：未达到。
- 原生流式 / KV Cache / 因果编码器：未实现。

## A. 修改文件清单

### 新增：核心与下载

| 文件 | 作用 |
|---|---|
| `slbridge/uni_sign_assets.py` | 固定 revision、大小/SHA256、磁盘余量、模型包 manifest |
| `slbridge/uni_sign_preprocess.py` | RTMPose-133 四分支预处理；置信度通道；确定性采样 |
| `slbridge/uni_sign.py` | PyTorch runtime、Backend、Adapter、姿态缓存、文本稳定器 |
| `scripts/download_uni_sign.py` | 官方 HF 断点续传/暂存/校验/原子落盘；源码和 mT5 |
| `scripts/setup_unisign_env.ps1` | 独立 Conda 环境、CUDA 12.8 PyTorch、依赖和 rtmlib |
| `requirements-unisign.txt` | Python 3.10 验证依赖组合 |
| `scripts/verify_uni_sign.py` | 单视频真实推理、重复性、哈希与计时证据 |
| `scripts/benchmark_uni_sign.py` | 滑动窗口、缓存复用、P50/P95、RTF、显存与文本变化 |
| `scripts/smoke_unisign_session.py` | 真实 `RecognitionSession` 异步视频窗口冒烟 |
| `scripts/smoke_unisign_desktop.py` | offscreen Qt 真实加载、打开视频、手动识别 |

### 新增：测试与文档

| 文件 | 作用 |
|---|---|
| `tests/test_uni_sign_assets.py` | 大小、哈希、缺失、原子 JSON、磁盘保护 |
| `tests/test_uni_sign_preprocess.py` | 133 点、置信度通道、归一化、采样、错误输入 |
| `tests/test_uni_sign_adapter.py` | Adapter 注册、缓存复用、稳定状态、失败清理 |
| `tests/test_unisign_desktop.py` | 桌面按钮、安装状态、许可提示 |
| `docs/uni-sign-integration.md` | 完整安装、运行、协议、许可与限制 |
| `docs/reports/unisign-integration-2026-10-23.md` | 本验收报告 |

### 修改

| 文件 | 作用 |
|---|---|
| `.gitignore` | 忽略 `.conda-env/`、`.cache/`、`models/`、临时 Uni-Sign 目录 |
| `README.md` | 增加 Uni-Sign 可选实验入口与边界 |
| `pyproject.toml` | 新增 `unisign` optional dependency |
| `slbridge/model_adapters.py` | 注册 `uni_sign_pose_slt` 后端 |
| `slbridge/desktop_models.py` | 默认 Uni-Sign 模型包路径 |
| `slbridge/desktop.py` | 设置页实验模型按钮、延迟启动输入、文本候选状态、许可/耗时展示 |
| `slbridge/desktop_cli.py` | `--unisign`、窗口帧/步长、摄像头/视频实验入口 |
| `slbridge/desktop_presenter.py` | Uni-Sign 算法名、中文句子任务显示 |
| `slbridge/captions.py` | 连续窗口文本候选/多窗口一致提示 |
| `tests/test_desktop.py` | Python 3.10 `tomllib` 兼容测试入口 |

> 注意：`README.md`、`slbridge/desktop.py`、`slbridge/desktop_cli.py`、`slbridge/desktop_presenter.py`、`slbridge/captions.py`、`tests/test_desktop.py` 在原工作树中已有未提交改动；本轮是在其基础上追加，未使用 reset/checkout 覆盖。

## B. 模型下载和完整性报告

| 项 | 值 |
|---|---|
| 模型仓库 | `ZechengLi19/Uni-Sign` |
| revision | `182913ca7a8b7dd9837d7d6f10717353f4408f80` |
| 文件 | `models/external/uni_sign/csl_daily_pose_only_slt.pth` |
| 大小 | `1,186,925,112` bytes（1.105 GiB） |
| SHA256 | `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c` |
| 校验 | PASS |
| mT5 revision | `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f` |
| mT5 文件 | `models/external/uni_sign/mt5-base/`（6 个模型/tokenizer 文件） |
| mT5 `pytorch_model.bin` | `2,329,735,129` bytes；SHA256 `180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614`；PASS |
| 源码 | `models/external/uni_sign_source` |
| 源码提交 | `eed438bcb49e30405cd6ccdfcccca330c134e830` |
| 源码 ZIP SHA256 | `8022b073bb1de962a8e3f424e9858ee50465e46228bc8854b966d54107d8cc97` |
| 本地模型包 | `models/external/uni_sign/model_bundle/manifest.json` |
| 本地下载报告 | `models/external/uni_sign/download_manifest.json` |

`python scripts\download_uni_sign.py --check-only` 返回 0，全部资产 `ok: true`。

### 许可

- Uni-Sign 权重：**CC BY-NC 4.0**。
- mT5：**Apache-2.0**。
- 固定提交的上游源码：**没有 LICENSE 文件**。本地只作外部依赖使用，不复制进仓库；再分发/商业用途需作者书面澄清。

## C. 接入实现说明

### C.1 Backend 与 Adapter

- 新后端 ID：`uni_sign_pose_slt`。
- 输入表示：`uni_sign_rgb_v1`，布局 `T,H,W,C`，uint8 BGR 视频帧。
- 输出任务：`text`，语言 `csl`，`score_type=uncalibrated_text`。
- 与现有 `tfnet_rgb_v1`、`mediapipe225_rgb_v1`、ONNX CTC 完全隔离。
- `model_manager` 仍负责 generation、模型切换、回滚、单请求 lease 和资源释放。
- Uni-Sign 异常不会把现有 RF/ONNX/CTC 模型包转换为新格式。

### C.2 关键点与输入

使用官方 `demo/online_inference.py` 的 RTMPose Wholebody 133 点协议：

| 分支 | 点索引 | 处理 |
|---|---|---|
| body | `[0] + 3..10` | 全窗口 body bbox 缩放/平移 |
| left | `91..111` | 减手腕，按 body scale 缩放 |
| right | `112..132` | 减手腕，按 body scale 缩放 |
| face_all | `[23,25,...,39] + [83..90] + [53]` | 减 nose，按 body scale 缩放 |
| 第 3 通道 | 每点 | 原始 confidence；不是 z 坐标 |

预处理已与官方 `load_part_kp` 在合成输入上比较：left/right/face 完全一致，body 仅存在 `1.19e-7` 的 float32 舍入差异。

### C.3 加载与推理

- `torch.load(..., weights_only=True)`。
- 固定 `CSL_Daily`、`SLT`、`rgb_support=False`、`hidden_dim=256`。
- 严格检查 missing/unexpected；实际 627 个 state-dict key，缺失/多余均为 0。
- `model.eval()`，CUDA bfloat16，`torch.inference_mode()`。
- 不读取参考句；上游 forward 中未使用的 label 路径用固定 sentinel `"."` 保持可执行。
- Windows 上先导入 PyTorch，再创建 ONNX Runtime/RTMPose 会话；反向顺序会触发 `WinError 127`。
- Python 3.10 下在导入 torchvision 前将 `typing.Self` 兼容为 TypeVar，避免 PySide6 已导入时 `torch._dynamo` import 失败。

### C.4 滑动窗口与缓存

- 关键点缓存键：`(source_id, frame_index, height, width)`。
- 窗口重叠时只对新帧运行 RTMPose；实测 32/12 第二窗口缓存命中 > 0。
- `RollingWindow` 保留有限帧数，桌面 CLI 的 Uni-Sign 实验使用精确 `max_frames=window_frames`。
- Session 现有单 worker / generation / epoch 机制限制同时推理数并丢弃过期结果。
- 文本稳定器：
  - 第一次文本：`temporary`
  - 连续相同：`confirmed`（表示窗口一致，不是人工确认）
  - 文本变化：重新回到 `temporary`，统计“已确认后被撤回”
- 滑动窗口仍是完整窗口重复调用，不是原生流式模型。

### C.5 桌面端

- 设置页增加“加载 Uni-Sign（实验）”按钮和许可提示。
- `--unisign` 入口可启动视频或摄像头。
- Uni-Sign 启动时可延迟打开输入，避免 27 秒模型加载期间短视频播放完毕。
- 显示模型加载状态、中文句子、窗口耗时、RTF、姿态缓存命中、候选稳定性。
- 现有模型仍作为默认；Uni-Sign 不自动设为生产模型。

## D. 运行方法

```powershell
$env:PYTHONPATH = 'E:\githubproject\SignLanguage-Bridge'
$env:HTTP_PROXY = 'http://127.0.0.1:7897'   # 仅当前会话，按需
$env:HTTPS_PROXY = $env:HTTP_PROXY

# 1. 下载并校验
python scripts\download_uni_sign.py
python scripts\download_uni_sign.py --check-only

# 2. 安装独立环境
powershell -ExecutionPolicy Bypass -File scripts\setup_unisign_env.ps1

# 3. 单视频真实推理（2 次一致性）
.\.conda-env\python.exe scripts\verify_uni_sign.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --device cuda --runs 2

# 4. 滑动窗口
.\.conda-env\python.exe scripts\benchmark_uni_sign.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --device cuda --window-frames 32 --stride-frames 12 `
  --max-width 640 --max-seconds 8.25

# 5. 桌面视频/摄像头
.\.conda-env\python.exe -m slbridge.desktop_cli --unisign `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4' `
  --continuous --device cuda --continuous-window-frames 32 --continuous-stride-frames 12

.\.conda-env\python.exe -m slbridge.desktop_cli --unisign `
  --camera 0 --continuous --device cuda `
  --continuous-window-frames 32 --continuous-stride-frames 12

# 6. 测试
.\.venv-libras\Scripts\pytest.exe -q
.\.conda-env\python.exe -m pytest -q
```

## E. 验收报告

### 已完成并通过

| 项 | 证据 |
|---|---|
| 权重/版本/哈希/大小 | `download_manifest.json`；`--check-only` 返回 0 |
| mT5 六个文件 | 大小与模型 SHA256 通过 |
| 固定源码提交 | `UPSTREAM_COMMIT.txt` + ZIP SHA256 |
| 模型严格加载 | 627 keys，missing/unexpected = 0 |
| RTMPose CUDA | detection/estimation 均为 `CUDAExecutionProvider` |
| 真实视频中文输出 | CE-CSL `test-00001.mp4` 返回中文句子 |
| 重复推理一致性 | `runs2-final.json` 两次文本完全相同 |
| 单推理任务/资源清理 | Session/Adapter 单测 + 真实 Session 冒烟 |
| 滑动窗口与缓存 | 18 个窗口 0 skipped；重叠窗口断言命中 |
| 桌面真实链路 | offscreen Qt 真实加载模型、打开视频、手动识别得到中文结果 |
| 摄像头路径 | `camera0-32-12.json`，3 个窗口，0 skipped |
| 回归 | `.venv-libras`: 772 passed, 10 skipped；`.conda-env`: 723 passed, 10 skipped；均 0 failed；两个环境 `pip check` 均无破损依赖 |
| CE-CSL 完整跨域评估 | 500/500 successful，0 empty；修正短边后严格匹配 0/500；BLEU-zh 7.99；旧 W/H 基线 4.72 |

### 已实现但尚未验证

| 项 | 状态 |
|---|---|
| 长时真人摄像头稳定性 | 只跑了约 5 秒、3 个窗口；不能代表长时验收 |
| 真实 CSL-Daily 数据域准确率 | 没有同域数据；NOT_RUN |
| BLEU/ROUGE 正式语料级结果 | 没有同域参考集；NOT_RUN |
| P95 产品目标 | 单片段描述性结果，不是大规模产品 P95 |
| 桌面真人交互视觉验收 | 仅 offscreen Qt 自动化验收，未完成人工视觉走查 |

### 实验结果未达到要求

- 稳定候选从视频流开始计算的 P95 约 **4,587 ms**，超过候选目标 **2,000 ms**。
- 640 宽、32/12 配置的实测 RTF 约 **1.22**，不能称为严格实时。
- 24/8 和 48/16 也未达到流级 2 秒目标；详见 F 节。
- 文本变化/撤回偏多；这是模型在跨域和不完整窗口上的真实行为，没有用硬编码文本掩盖。

### 因环境/依赖限制无法完成

- 没有 CSL-Daily 同域测试视频/标注；没有下载完整数据集。
- 没有进行 mT5 KV Cache 或因果编码器改造。
- 权重 CC BY-NC 4.0，未进行商业使用授权。

## F. 性能报告

硬件：NVIDIA GeForce RTX 5060 Ti 16 GB，驱动 616.92；PyTorch 2.9.1+cu128；Windows 10.0.19045；Python 3.10.22。

测试视频：`D:\Zenodo\CE-CSL\CE-CSL\video\test\A\test-00001.mp4`，1920×1080，30 FPS，247 帧，8.23 秒。该视频与 Uni-Sign 的 CSL-Daily 训练域不同，仅用于工程与延迟实验。

| 窗口/步长 | 窗口数 | 处理 P50 | 处理 P95 | 首输出 | 首稳定候选 | 实际 FPS | RTF | 文本变化/重复/撤回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 / 8 | 28 | 389.4 ms | 501.6 ms | 959.2 ms | 3,152.2 ms | 19.44 | 1.54 | 21 / 6 / 4 |
| 32 / 12 | 18 | 483.7 ms | 560.9 ms | 925.8 ms | 1,946.7 ms | 24.56 | 1.22 | 13 / 4 / 2 |
| 48 / 16 | 13 | 584.8 ms | 1,044.2 ms | 1,877.2 ms | 3,109.1 ms | 25.57 | 1.17 | 10 / 2 / 2 |

### 32/12 的稳定候选细分

- confirmed 窗口：4
- confirmed 窗口结束后到结果的 P50/P95：**498.2 / 519.4 ms**
- 从视频流开始到 confirmed 结果的 P50/P95：**3,343.7 / 4,587.3 ms**
- 因此：
  - “窗口结束后推理延迟”约 0.52 秒 P95；
  - 但从流开始出现稳定候选的 P95 约 4.59 秒；
  - **不能**用前者宣称满足后者的 2 秒目标。

### GPU

- 模型常驻：约 1,148.9 MB allocated；2,688 MB reserved。
- 峰值：约 1,248.8 MB allocated；2,708 MB reserved。
- 640 宽、32/12 的最终 benchmark 峰值：1,183.5 MB allocated；2,708 MB reserved。

### 摄像头

`camera:0`，32/12，5 秒上限，阻塞式 benchmark：

- 65 帧，3 个窗口，0 skipped
- 处理 P50/P95：532.3 / 1,190.2 ms
- 首输出：3,161.9 ms
- 无 confirmed 稳定输出
- 实际 FPS：11.67
- RTF：2.57

这是阻塞式采集 + 推理实验，不等同于桌面异步链路的性能。

### 异步 Session 冒烟

`smoke_unisign_session.py` 在真实 `RecognitionSession` 上运行同一视频：

- 3 个 continuous_result
- 0 个 confirmed
- 无异常
- 采集线程与模型 worker 分离；窗口结果未写入人工确认字幕历史

## G. Git 变更与回滚

- 当前分支：`codex/model-contract-auto-captions`
- HEAD：`cf291c2`
- 未提交、未推送、未创建 PR。
- 大模型、mT5、源码、Conda 环境和缓存均在 ignored 路径，不会进入 Git。
- 工作树在本轮前已有多项未提交改动；本轮没有执行 `git reset`、`git checkout`、强制清理或覆盖式删除。
- 受控运行使用 E 盘项目缓存；诊断时产生的 `C:\Users\ADMIN\.cache\rtmlib` 约 150 MB 重复缓存未自动递归删除。

回滚原则：

1. 新增文件可按清单删除；不需要触碰原模型或数据。
2. 已修改的既有文件包含本轮前用户改动，禁止直接对该文件做无差别 `git checkout`。
3. 回滚应按文件中的 Uni-Sign hunk 定向移除。
4. `artifacts/unisign-integration-20261023/` 保留了本轮源码/补丁快照，便于人工比对恢复。

## 证据文件

- `artifacts/unisign-verification/cecsl-test-00001-runs2-final.json`
- `artifacts/unisign-benchmark/cecsl-test-00001-24-8.json`
- `artifacts/unisign-benchmark/cecsl-test-00001-32-12-final.json`
- `artifacts/unisign-benchmark/cecsl-test-00001-48-16-fullwarm.json`
- `artifacts/unisign-benchmark/camera0-32-12.json`
- `artifacts/unisign-session/cecsl-test-00001-32-12.json`
- `artifacts/unisign-desktop/cecsl-test-00001-real.json`（由 `scripts/smoke_unisign_desktop.py` 生成）
- `artifacts/unisign-cecsl-eval/full-isotropic-20261023/summary.json`
- `artifacts/unisign-cecsl-eval/full-isotropic-20261023/predictions.csv`
- `artifacts/unisign-cecsl-eval/full-20261023/`（旧 W/H 被修正基线）
- `artifacts/unisign-cecsl-eval/comparison-20261023/bootstrap_compare.json`
- `artifacts/unisign-cecsl-eval/comparison-20261023/cecsl-unisign-comparison.csv`
- `outputs/01a16889-c4d3-72a1-8eb7-5b7ebb887710/CE-CSL_UniSign_结果对比.xlsx`
- `docs/reports/unisign-cecsl-cross-domain-accuracy-2026-10-23.md`

## 最终判定

**工程接入：完成。**
**CE-CSL 500 条跨域严格匹配：0/500；修正短边 BLEU-zh 7.99（旧 W/H 4.72）。**
**真实 CSL-Daily 准确率：未验证。**
**近实时实验：可运行但不是原生流式；未达到 2 秒稳定字幕候选目标。**
**产品验收：未完成，需同域数据、真人长时摄像头、多场景和人工视觉验收。**











- 源码快照：`artifacts/unisign-integration-20261023/unisign-source-snapshot.zip`；SHA256 见同目录 `unisign-source-snapshot.sha256`。
