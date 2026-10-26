# E00 / T00 入口与环境审计报告（inventory_report.md）

日期：**2026-10-26**；审计员：auditor（P-1 实验团队）；范围：T00 真实项目入口、Git 治理状态、README 与实际差异。
纪律：本轮**只读**，未修改任何代码/配置；所有命令见文末与 `provenance.md`。

## 1. Git 状态（实测）

| 项 | 值 |
|---|---|
| 分支 | `codex/model-contract-auto-captions` |
| HEAD | `cf291c2a81558faf6de0dc5a7243e730531e4367`（2026-10-18 00:08:44 +0800，`feat: add keypoint capture workbench`） |
| 跟踪文件数 | 207（`git ls-files`） |
| 跟踪文件在磁盘缺失 | **0**（`git -c core.quotepath=false ls-files -z` 逐项 Test-Path） |
| skip-worktree（`git ls-files -v` 的 `S`） | **0** —— pitfalls §10 记录的“35 个模块被 skip-worktree 隐藏、实际不在磁盘”在当前工作区已不成立 |
| 脏工作区 | **132 项**：16 个已跟踪文件被修改（` M`，如 README/pyproject/slbridge/{captions,contracts,desktop*,model_adapters,session,sources}.py、tests/）；116 项未跟踪（`??`） |

### 未提交改动对实验的影响（关键）

HEAD **不包含** P-1 需要的大多数实验入口与模型接入代码：

- `slbridge/slrt/`（19 个模块，SLRT 在线 Gloss 后端/CLI/流式）→ 全部未跟踪，`git ls-files slbridge/slrt*` 为 0；HEAD 无法独立复现 SLRT 路径。
- `slbridge/uni_sign.py`、`uni_sign_assets.py`、`uni_sign_preprocess.py`、`continuous.py`、`candidate_context.py`、`desktop_models.py`、`preview_landmarks.py` → 未跟踪。
- `scripts/` 下 35 个实验脚本（`evaluate_unisign_csldaily_video.py`、`align_slrt_csl_daily_subset.py`、`extract_slrt_hrnet_pose.py`、`check_slrt_reproduction_assets.py` 等）→ 未跟踪；git 里只剩 3 个 2026-10 的旧脚本。
- `docs/experiments/P-1/`、`experiments/p1/`、`docs/reports/*`（2026-10-22 之后的报告）、`configs/slrt-csl*.json`、`requirements-{slrt,slrt-legacy,unisign}.txt`、`AGENTS.md`、`Paper-algorithm/`、根目录计划包副本 → 未跟踪。
- 已跟踪文件也有 16 个修改（含 README、pyproject 增加 slrt/unisign extras 与 `slbridge-slrt` 入口）。当前运行行为 = 工作区状态，**不是 HEAD 状态**。

**影响**：所有 E01–E06 运行记录必须同时记录 `git HEAD + 工作区脏状态`（建议 `git status --porcelain` 摘要或 `experiments.p1.common.git_state()` 输出）；任何人用 HEAD 复现都缺 SLRT/Uni-Sign 实验代码。本报告不修改、不提交这些文件。

## 2. 真实入口清单（存在/不存在，实测）

### 2.1 控制台脚本（pyproject `[project.scripts]`）

| 入口 | pyproject 声明 | .conda-env 已安装 `.exe` | 实际可用性判定 |
|---|---|---|---|
| `slbridge` | `slbridge.cli:main` | ✅ `slbridge.exe` | ✅ `python -m slbridge --help` 实测输出 `{index,validate,select,extract,split,train,predict}` |
| `slbridge-desktop` | `slbridge.desktop_cli:main` | ✅ `slbridge-desktop.exe` | ✅ `python -m slbridge.desktop_cli --help` 实测（含 `--unisign/--slrt/--keypoints/--continuous`） |
| `slbridge-api` | `slbridge.api_cli:main` | ✅ | 声明+已安装；本轮未启动服务 |
| `slbridge-cecsl` | `slbridge.cecsl_cli:main` | ✅ | 声明+已安装；本轮未跑 |
| `slbridge-keypoints` | `slbridge.keypoint_cli:main` | ✅ | 声明+已安装；本轮未跑 |
| `slbridge-slrt` | `slbridge.slrt.cli:main`（工作区 pyproject） | ❌ **缺失** | ⚠️ `docs/slrt-integration.md` 第111行称“安装项目后的入口为 slbridge-slrt”，**当前 .conda-env 未安装该入口**；只能用 `python -m slbridge.slrt.cli`（已实测可用） |

### 2.2 Python 模块入口（实测帮助）

| 命令 | 结果 |
|---|---|
| `.conda-env\python.exe -m slbridge.slrt.cli --help` | ✅ usage: `slbridge-slrt [-h] [--config] [--resource-dir] [--checkpoint] [--vocabulary] [--device] [--pose-device] [--pose-backend] ...` |
| `.conda-env\python.exe -m slbridge.cli --help` | ✅ |
| `.conda-env\python.exe -m slbridge.desktop_cli --help` | ✅ |
| `.conda-env\python.exe -m experiments.p1.eval.evaluate --help` | ✅ `--predictions --references --task {CSLT,CSLR} --output --run-dir --bootstrap` |
| `python -m experiments.p1.evaluate`（05 指令建议写法） | ❌ 不存在；实际路径是 `experiments.p1.eval.evaluate` |
| `python -m experiments.p1.inventory / replay / profile / stream_probe` | ❌ 均不存在（05 指令明示为“拟实现”）；`experiments/p1` 现有可执行模块只有 `datasets/build_manifest.py`、`references/build_references.py`、`eval/evaluate.py`（+ `eval/metrics.py`） |

### 2.3 脚本入口（`scripts/`，均为**未跟踪**工作区文件）

| 脚本 | 存在 | 备注 |
|---|---|---|
| `evaluate_unisign_csldaily_video.py` | ✅ | 1086 视频端到端评测（历史 BLEU 24.8157 来源） |
| `evaluate_unisign_csldaily_pose.py` | ✅ | 官方 pose 复现（BLEU 25.55） |
| `align_slrt_csl_daily_subset.py` / `align_slrt_csl_daily_train100.py` | ✅ | SLRT 视频-关键点滑窗近似对齐 |
| `extract_slrt_hrnet_pose.py` | ✅ | 官方 HRNet NPZ（legacy Python 3.8） |
| `setup_slrt_official_layout.py` | ✅ | 幂等重建官方目录 |
| `check_slrt_reproduction_assets.py` | ✅ | 本轮实际运行：唯一 missing = `sentence_frames-512x512.zip` |
| `convert_slrt_predictions_to_slt.py` / `assemble_slt_mbart_fullvocab.py` / `prepare_slt_gloss_assets.py` | ✅ | SLT 资产/转换 |
| `verify_uni_sign.py` / `download_uni_sign.py` / `setup_unisign_env.ps1` | ✅ | Uni-Sign 资产/环境 |
| `evaluate_slrt_datasets.py` / `benchmark_slrt.py` / `verify_slrt_parity.py` / `probe_slrt_replay_latency.py` | ✅ | SLRT 评测/性能/协议对照 |
| `summarize_unisign_eval.py`（236 KB） / `summarize_unisign_predictions.py` | ✅ | 汇总工具 |
| `launch_desktop.py` / `start_demo.ps1` / `smoke_unisign_*.py` | ✅ | 桌面/冒烟 |
| 其余（build_csldaily_video_*, export_csldaily_comparison_xlsx 等） | ✅ | 均为工作区未跟踪脚本 |

## 3. README 与实际差异（只列可核实的）

| README/文档陈述 | 实际核对 | 判定 |
|---|---|---|
| README 第2行称“当前已完成首轮识别核心、真实 Libras RGB 试点…”，第163行新增 SLRT-CSL 模块 | README 本身被修改（` M`），SLRT 段落仅存在于工作区；HEAD README 无该段 | 文档超前于 HEAD，属未提交改动 |
| README 桌面安装：`.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt` 再 `-m slbridge.desktop_cli` | `.venv-libras` 实测 **没有安装本项目**（`import slbridge` 失败；`Scripts` 无 `slbridge*.exe`）；只在仓库根用 `python -m` 可运行（cwd 进 sys.path） | 命令隐含“先 pip install -e .”；新克隆需补跑安装 |
| README 称 `.venv-libras/Scripts/python.exe` 为“已验证的 Python 3.12 环境” | 实测 Python **3.12.7**，无 torch；有 numpy/cv2/onnxruntime/sklearn/PySide6/mediapipe | 一致（但不能跑 torch 模型） |
| README 第54/75行默认 RF 模型 `artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009` | ✅ 目录存在：`weights/model.joblib`（440,668 B，SHA256 `3450d09c…db35d`）+ labels/manifest/metrics/resolved_config + splits | 一致 |
| README/文档称权重不入库（models/、artifacts/、data/ 在 .gitignore） | ✅ `.gitignore` 含 `.conda-env/ .venv-libras/ .venv-gpu/ artifacts/ data/ models/ .cache/` | 一致；`data/gestures/gesture_model.pkl` 不在 git 跟踪内 |
| README 第5行称支持“随机森林训练、模型包及独立推理” | `slbridge.cli` 帮助实测含 `train/predict` 子命令 | 一致 |
| README 称 API 文档在 `http://127.0.0.1:8765/docs` | 本轮未启动服务，未验证 | 未验证（勿当证据） |
| 05 指令写的 `python -m experiments.p1.evaluate` 等命令 | 不存在（见 2.2） | 05 已标“拟实现”，实现时需改为 `experiments.p1.eval.evaluate` |
| P-1 计划包位置 | `docs/experiments/P-1/` 存在全部 8 个文档+config/schemas/templates；根目录另有未跟踪副本 `SignLanguageBridge_P-1_实验计划包/` | 内容一致（副本冗余，未核对逐字节） |

## 4. 基础设施与环境健康

- `experiments/p1/tests/test_p1_infra.py`：**12 passed in 1.37s**（`.conda-env` pytest，本轮实测）。
- `experiments/p1/` 公共设施（RunDirectory/sha256_file/git_state/environment_fingerprint、gt_guard、manifest/pilot/references、evaluate）齐备，未被本轮修改。
- 三套 Python 环境分离见 `environment_manifest.json`；`.conda-env` 可 import `slbridge` + `slbridge.slrt` + torch CUDA；`F:\DSH-envs\slrt-legacy` 可 import mmpose/mmdet/mmcv（torch CPU）；`.venv-libras` 无 torch。
- `scripts/check_slrt_reproduction_assets.py` 本轮实测：权重/配置/工具全部 present，**唯一 missing 数据文件 = `sentence_frames-512x512.zip`**。

## 5. 结论与交接

1. 项目入口真实存在且主要 CLI 可调用；但 **`slbridge-slrt` 控制台入口未安装**、05 指令的部分模块名是“拟实现”。
2. **Git 治理是 E00 最大风险**：132 项脏改动、SLRT/Uni-Sign 实验代码全部未提交。任何结果必须绑定“HEAD + 工作区状态”，不能只写 tag/SHA。
3. 本轮无 skip-worktree 隐藏文件问题（`S`=0、跟踪文件 0 缺失），但该风险已写入 pitfalls §10，后续拉取/切分支后应重新跑本项检查。
4. 未修改任何代码；证据文件均在本目录（`_hash_raw*.txt`、`_env_probe_*.json`、`_data_snapshot_raw.json`、辅助脚本）。

## 附：本轮实际执行命令（摘要）

```powershell
git rev-parse --abbrev-ref HEAD; git rev-parse HEAD; git status --porcelain=v1; git ls-files -v
git -c core.quotepath=false ls-files -z | python 逐项存在性检查
Get-ChildItem models\external -Recurse -File -Include *.pth,*.ckpt,*.pt,*.bin,*.onnx
Get-FileHash <24 个关键文件> -Algorithm SHA256          # experiments/p1/e00/_hash_assets.ps1
Get-FileHash <7 个 SLRT TwoStream 权重> -Algorithm SHA256  # _hash_slrt_weights.ps1
\.conda-env\python.exe -m pytest experiments\p1\tests\test_p1_infra.py -q
\.conda-env\python.exe scripts\check_slrt_reproduction_assets.py
\.conda-env\python.exe -m slbridge.slrt.cli --help（及 slbridge.cli / desktop_cli / experiments.p1.eval.evaluate）
\.conda-env\python.exe experiments\p1\e00\_env_probe.py；\.venv-libras\Scripts\python.exe 同脚本；F:\DSH-envs\slrt-legacy\python.exe 同脚本
nvidia-smi；Get-CimInstance Win32_Processor/Win32_OperatingSystem/Win32_LogicalDisk；Get-ChildItem Env:
```
