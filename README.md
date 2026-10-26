# 中巴手语桥 · SignLanguage Bridge

面向中国与巴西听障人群交流的项目。当前已完成首轮识别核心、真实 Libras RGB 试点，以及视频／摄像头片段输入与中葡字幕桌面链路。

核心支持 NPY／带时间戳的关键点序列索引、严格校验、分组切分、随机森林训练、模型包及独立推理。Libras RGB 已完成全量索引、标签冲突隔离和 12 条目／360 视频试点提取、训练及测试。0.3.0 接入 CE-CSL TFNet：完整 RGB 视频返回中文手语词序列，使用 ONNX CPU/CUDA 与独立 v2 API；该双向模型需要结束片段后识别。跨语言句子翻译仍待建设。现有 A–D 标签的手语语种及释义未经核验。

## CE-CSL 中文词序列识别

已冻结官方词表、预处理及划分，运行全部 500 段中文测试视频。修正 prefix beam 解码下，PyTorch/CUDA 逐句平均 WER 为 32.4794%、语料 WER 为 32.8289%；最终 ONNX CUDA 分别为32.4394%和32.7961%，两份报告499/500条词序列一致，差异项单独保留；WER 不是分类准确率。约 266 MB 的 ONNX 模型已完成动态帧数、数值、CTC 解码和实际 CUDA 算子对照，并接入共享模型切换、回滚和字幕。变长重复推理采用每次请求后回收空闲显存池区的策略。

```powershell
python -m pip install -e '.[api,rgb-sequence,onnx-gpu]'
python -m slbridge.api_cli --catalog configs/cecsl_runtime_catalog.example.json --model csl-tfnet --device cuda --intra-op-threads 4 --gpu-mem-limit-mb 8192
```

模型与上游源码保存在本地，仓库不分发权重；CUDA/cuDNN DLL 的安装或显式目录配置见 [CE-CSL 接入与执行说明](docs/CE-CSL_TFNet_ONNX接入与验证_2026-10-09.md)。上传完整视频使用 `/v2/recognitions`；桌面可选择同一个模型目录，中文字幕为识别词序列，葡语表达待审核。共同词义候选见 [审核清单](docs/CE-CSL_Libras共同语义审核清单_2026-10-09.md)，视觉表示迁移结果见 [迁移验证](docs/CE-CSL_Libras迁移验证_2026-10-09.md)。

2026-10-11 已修复当前 GPU 环境的 cuDNN 引擎加载及统一运行器 2 GB 显存池上限问题。同一 8.2 秒完整视频在 RTX 5060 Ti 上模型前向约 0.54–0.56 秒，文件推理总耗时约 1.92–1.99 秒，真实 HTTP 上传约 2.07–2.25 秒；完整 logits 与 CPU 参考在既定容差内一致，最终词序列相同。独立 GPU 环境和 v3 API 启动见[GPU 使用说明](docs/auto-captions-usage.md#gpu-推理)，数值、算子、HTTP 证据及验收边界见[实测报告](docs/reports/ctc-gpu-inference-2026-10-11.md)。

## Uni-Sign 中文句子实验（可选）

Uni-Sign `csl_daily_pose_only_slt.pth` 已作为独立 PyTorch 实验后端接入，不替换现有 ONNX CTC 或 MediaPipe 225 维协议。权重为 CC BY-NC 4.0，仅用于非商业研究；上游源码在固定提交下没有 LICENSE 文件，本仓库不分发该源码。模型、mT5、源码和独立 Conda 环境均放在 E 盘项目目录；Hugging Face/pip/RTMPose 运行缓存也配置到 E 盘。诊断期间产生的约 150 MB C 盘 RTMLib 副本未自动删除。

```powershell
python scripts\download_uni_sign.py --check-only
.\.conda-env\python.exe -m slbridge.desktop_cli --unisign --continuous --device cuda --camera 0
.\.conda-env\python.exe -m pytest -q
```

本机 RTX 5060 Ti 的真实视频滑动窗口实验已完成：32 帧窗口/12 帧步长下，窗口处理 P50/P95 约 0.48/0.56 秒，首条稳定候选约 1.95 秒；但流级稳定候选 P95 约 4.59 秒，未达到 2 秒候选目标。CE-CSL 500 条跨域评估发现并修正了 16:9 视频的横纵归一化缺陷：严格整句仍为 **0/500**，BLEU-zh 由旧 W/H 的 **4.72** 提升到修正短边的 **7.99**，chrF **9.38**，字符编辑相似度 **22.51%**，仍只用于跨域分析。CE-CSL 有中文参考句和 Gloss；没有本地同域 CSL-Daily 视频测试，因此不能给出 CSL-Daily 或产品准确率结论。完整安装、许可、输入协议和限制见 [Uni-Sign 接入说明](docs/uni-sign-integration.md)，逐译者、逐句与 Bootstrap 证据见 [CE-CSL 评估报告](docs/reports/unisign-cecsl-cross-domain-accuracy-2026-10-23.md)。

## 模型 API 与 ONNX

0.2.0 新增统一模型运行管理器、HTTP v1 API、模型校验/预热/切换/回退，以及 ONNX CPU/CUDA 适配器。桌面使用同一运行管理逻辑，录制期间禁止切换，设置中可恢复上个模型。现有 RF 导出模型已用全部 72 个冻结测试输入验证数值与预测一致性；CUDA 计算路径已验证，RF 的 TreeEnsemble 算子仍在 CPU 上执行。

```powershell
.venv-libras/Scripts/python.exe -m pip install -r requirements-api.txt -r requirements-onnx.txt
.venv-libras/Scripts/python.exe -m slbridge.api_cli --catalog configs/runtime-models.example.json --model libras-onnx --device cpu
```

本地接口文档为 http://127.0.0.1:8765/docs。模型包仍保留本地，新克隆先训练或导出并修改目录配置。GPU 环境选择 `requirements-onnx-gpu.txt`，不要与 CPU ORT 包共同安装。完整输入输出、设备配置、导出和实测边界见[使用说明](docs/模型运行API与ONNX_使用说明_2026-10-09.md)、[设计与执行](docs/模型运行API与ONNX_设计与执行_2026-10-09.md)及[验收报告](docs/reports/model-runtime-api-2026-10-09.md)。

## 视频与摄像头字幕桌面

在项目根目录使用已验证的 Python 3.12 环境。当前桌面使用 PySide6，左侧预览视频或摄像头，右侧显示中文与葡语释义及支持的手语词。支持完整文件、手动录制，以及无需停顿的连续视频窗口识别。连续模式更新当前窗口结果，内容可修正，不自动写入正式字幕记录；现有模型仍按片段推理，尚未实现连续整句翻译。Libras 中文释义为待核验草案。

```powershell
python -m venv .venv-libras
.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt
.venv-libras/Scripts/python.exe -m pip install -e .
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli
```

桌面依赖锁定 `PySide6-Essentials==6.8.3`，由 `requirements-desktop.txt` 安装。已在本机完成试点实验时，默认加载 `artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009`；新克隆的仓库不包含训练模型或视频，应先准备数据并运行下方实验，或在“设置 → 选择识别模型”选择兼容模型目录。

导入视频后点击“识别视频”；使用摄像头后点击“开始录制”，动作完成后点击“完成并识别”。按钮会随当前任务显示“取消识别”“取消录制”或“清空结果”。模型路径、设备编号和释义文件在“设置”中调整，标签编号和未经校准的模型分数在“详细信息”中查看。

需要动作中持续刷新时，打开输入后勾选“连续识别视频流（无需停顿）”。默认积累 1.2 秒后开始，最多使用最近 2.4 秒，目标提交间隔 0.5 秒；模型忙时持续保留最新画面，不排队处理旧窗口。移出双手、暂停或关闭时清除当前候选。停止连续识别并等待当前任务返回后，可以切换手语；“停顿确认后追加字幕记录”保留原来的片段确认流程，与连续模式互斥。

右侧“候选序列”保留连续识别已返回的最近 200 次有效候选，可调整为 1–1000 条、清空或“导出上下文”为 JSON。停止识别、换视频或切换手语后记录仍保留，并标明来源、语言、模型、时间范围及窗口重叠关系。分类模型保留完整备选分数，中文模型保留原始词序列及模型实际提供的词级分数；不会补造序列备选。记录仅在当前会话内保存，退出前需导出。该上下文供后续 LLM 接入使用，当前未调用 LLM；候选未确认、重叠窗口可能重复，不能直接拼接为正式句子。

预览下方的“显示关键点”可随时显示或隐藏身体与双手骨架，视频、摄像头和手动录制均可使用。默认关闭；开启时复用后台活动检测，并将关键点与对应帧一起显示。摄像头镜像预览会同步翻转骨架；关闭、暂停或换源后清除旧骨架。此开关只影响显示，不改写录制画面、模型输入或字幕记录；独立关键点工作台仍用于样本保存与回放。

本机可用以下脚本优先选择依赖齐全的 GPU 环境并开启连续模式；`-Language libras` 选择巴西手语，`-Video "视频路径"` 使用文件实时回放，`-Manual` 使用手动模式。首次配置 GPU 桌面需在独立 GPU 环境安装 `requirements-desktop.txt`，实际设备见“详细信息”。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Language csl -Camera 0
```

主界面视频／摄像头按钮旁的“识别手语”可直接切换“中国手语”与“巴西手语”。模型在后台加载，摄像头预览保持开启；加载成功后才更新当前选择与词表，失败保留原模型。录制、识别和加载期间禁止再次切换；连续模式需先停止。切换会停止自动分段、清除当前结果并保留字幕记录，需要时重新勾选对应模式。候选显示“未配置”时，可在设置里选择该手语的本地模型文件夹。此功能需要手动选择手语种类，当前不自动判断中／巴手语。

可指定自己的模型、词典和来源：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --model artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009 --catalog configs/libras_caption_catalog.json --camera 0
```

安装后也可使用 `slbridge-desktop`；原启动方式 `python app/desktop_app.py` 继续有效。详细操作、输入边界与验收命令见[使用说明](docs/视频摄像头与字幕_使用说明_2026-10-09.md)，当前界面设计见[桌面改版方案](docs/桌面界面改版_设计与执行_2026-10-09.md)，验证结果见[新版验收报告](docs/reports/desktop-redesign-2026-10-09.md)。

## 关键点采集与回放

新增独立工作台，可使用视频或摄像头查看身体与双手关键点、质量和时间信息，录制为 `.slkp.jsonl`，保存后回放骨架。它不加载手语词义识别模型，采集使用 MediaPipe，尚未接入动作分类或字幕生成；录制不含原始视频，人工标签保留为待审核草案。

```powershell
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --keypoints
```

主界面也可点击“关键点工作台”。需要提取视频每一帧时使用 `python -m slbridge.keypoint_cli extract`，已保存文件使用 `inspect` 校验。启动参数、完整操作、独立关键点协议和资源限制见[使用说明](docs/keypoint-workbench-usage.md)，验证范围见[开发验收报告](docs/reports/keypoint-workbench-2026-10-11.md)。

## 开始使用核心

推荐使用 Python 3.12。已验证版本为 Python 3.12.7、NumPy 1.26.3、scikit-learn 1.4.0、joblib 1.4.2。核心不需要摄像头、MediaPipe、OpenCV 或 GUI。

```powershell
python -m pip install -r requirements-core.txt
python -m slbridge --help
```

在项目根目录运行。完整说明见 [核心开发与使用](docs/核心开发与使用_2026-10-09.md)。

### 索引、切分、训练、推理

```powershell
python -m slbridge index --source legacy --root data/sequences --dataset-id legacy-hand63 --representation hand63_v1 --output artifacts/datasets/legacy-hand63/manifest.json
python -m slbridge split --manifest artifacts/datasets/legacy-hand63/manifest.json --output artifacts/datasets/legacy-hand63/split.json
python -m slbridge train --manifest artifacts/datasets/legacy-hand63/manifest.json --split artifacts/datasets/legacy-hand63/split.json --config configs/rf_hand63.json --output-dir artifacts/models
```

训练输出 JSON 中的 `model` 是新模型包路径。指定这个目录进行推理，例如本次已经验证的模型：

```powershell
python -m slbridge predict --model artifacts/models/core-hand63-20261009 --input data/sequences/A_000.npy
```

结果为 JSON，包括稳定标签、所有类别概率、质量信息、模型版本和拒识原因；显示词典为空仍会返回识别标签。概率标为未经校准的类别概率。

### 测试

全量测试覆盖视频输入和桌面控件，需要在已安装桌面依赖的 Python 3.12 环境运行：

```powershell
.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt "pytest>=8"
.venv-libras/Scripts/python.exe -m pip install -r requirements-api.txt -r requirements-onnx.txt
.venv-libras/Scripts/python.exe -m pytest -q
```

真实模型与视频一致性测试的配置见[输入与字幕使用说明](docs/视频摄像头与字幕_使用说明_2026-10-09.md)。

上面是完整 CPU 测试环境；GPU 环境将 `requirements-onnx.txt` 替换为 `requirements-onnx-gpu.txt`。未安装可选 API/ONNX 依赖时，相应模块测试正常跳过；完整验收应安装依赖并设置真实模型/视频环境变量。

### Libras RGB 实验

```powershell
python -m venv .venv-libras
.venv-libras/Scripts/python.exe -m pip install -r requirements-libras.txt
.venv-libras/Scripts/python.exe -m experiments.libras_rf
.venv-libras/Scripts/python.exe -m experiments.validate_libras
```

完整流程见[巴西手语训练与验证](docs/巴西手语训练与验证_2026-10-09.md)。试点测试 accuracy 为 63.9%、macro-F1 为 0.603；范围是 12 个有限条目的完整片段分类，local_informant 分组不代表全局签者独立。评价见[真实数据实验报告](docs/reports/libras-rf-pilot-2026-10-09.md)。

## 目录

```text
slbridge/          核心契约、数据、特征、模型、训练、模型包、推理、CLI
configs/           特征与 RF 配置、中葡字幕词典
tests/             公共接口与跨进程闭环验收
artifacts/         本地索引、冻结 split、版本化模型和验证结果（不纳入版本控制）
app/               新桌面启动包装、保留的旧原型与旧采集／训练脚本
data/              原始 NPY 与旧模型资产
Libras-EQT-UECE (RGB)/  原始 RGB 数据
docs/              设计、数据审计、开发与验收文档
```

`app/desktop_app.py` 已接入新核心；旧桌面原型保留为 `app/legacy_desktop_app.py`。旧采集和训练脚本尚未完成全面迁移，当前训练请使用核心 CLI。新模型包与旧的 `gesture_model.pkl` 使用不同路径。项目中没有 `app/main.py`。原始数据、训练模型和运行产物保持本地，不纳入 Git。

[核心设计](docs/核心框架设计_2026-10-09.md) · [核心验收](docs/reports/core-implementation-2026-10-09.md) · [当前桌面验收](docs/reports/desktop-redesign-2026-10-09.md) · [首版输入与字幕验收](docs/reports/video-camera-captions-2026-10-09.md) · [文档入口](docs/README.md)

统一模型 adapter、独立语言路由、停顿自动分段和字幕事件历史已提供实现。启动、接入和回放命令见[自动字幕与模型接入](docs/auto-captions-usage.md)，API 见[v3 契约](docs/unified-api.md)，实测状态见[开发报告](docs/reports/auto-captions-2026-10-11.md)。自动端点与未知动作拒识仍需真实长流验证；当前能力是持续采集加完整片段推理。

CTC CPU 的卷积极小浮点权重处理与默认线程配置已修复；8.2 秒完整视频当前实测总耗时 6.31–6.62 秒，完整 logits 与原模型一致。对照基准、复现命令和限制见[性能修复报告](docs/reports/ctc-cpu-performance-fix-2026-10-11.md)。

## SLRT-CSL 连续 Gloss 实验模块

新增 EMNLP 2024 SLRT 双流 S3D 后端，支持独立 MP4 JSON 推理、有界连续帧会话及可选 Qt 面板，保留现有 RF、TFNet、Uni-Sign 和 Libras 路由。权重严格加载及实验 RTMPose 前端的真实视频已验证；官方 HRNet 资源仍需单独配置，识别质量与实时交付尚未通过验收。

```powershell
.conda-env/python.exe -m slbridge.slrt.cli inspect
.conda-env/python.exe -m slbridge.slrt.cli video --video 'D:/path/sign.mp4' --pose-backend rtmpose-experimental --output artifacts/slrt-result.json
.conda-env/python.exe -m slbridge.desktop_cli --slrt --slrt-pose-backend rtmpose-experimental
```

资源、摄像头命令、输入协议和统一接口见 [SLRT 接入说明](docs/slrt-integration.md)，实际结果见[交付报告](docs/reports/slrt-integration-2026-10-24.md)。输出是 CSL Gloss，翻译模块及 Libras 的 SLRT 权重尚未接入。
