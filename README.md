# 中巴手语桥 · SignLanguage Bridge

面向中国与巴西听障人群交流的项目。当前已完成首轮识别核心、真实 Libras RGB 试点，以及视频／摄像头片段输入与中葡字幕桌面链路。

核心支持 NPY／带时间戳的关键点序列索引、严格校验、分组切分、随机森林训练、模型包及独立推理。Libras RGB 已完成全量索引、标签冲突隔离和 12 条目／360 视频试点提取、训练及测试；当前可直接识别兼容模型对应的原始视频。连续识别与跨语言翻译仍待实现。现有 A–D 标签的手语语种及释义未经核验。

## 视频与摄像头字幕桌面

在项目根目录使用已验证的 Python 3.12 环境。当前桌面使用 PySide6，左侧预览视频或摄像头，右侧显示中文与葡语释义及支持的手语词。支持 MP4/AVI/MOV/MKV 完整文件识别；摄像头先录制一个动作，再识别片段。中文显示为待核验释义草案，当前模型不支持边输入边翻译。

```powershell
python -m venv .venv-libras
.venv-libras/Scripts/python.exe -m pip install -r requirements-desktop.txt
.venv-libras/Scripts/python.exe -m pip install -e .
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli
```

桌面依赖锁定 `PySide6-Essentials==6.8.3`，由 `requirements-desktop.txt` 安装。已在本机完成试点实验时，默认加载 `artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009`；新克隆的仓库不包含训练模型或视频，应先准备数据并运行下方实验，或在“设置 → 选择识别模型”选择兼容模型目录。

导入视频后点击“识别视频”；使用摄像头后点击“开始录制”，动作完成后点击“完成并识别”。按钮会随当前任务显示“取消识别”“取消录制”或“清空结果”。模型路径、设备编号和释义文件在“设置”中调整，标签编号和未经校准的模型分数在“详细信息”中查看。

可指定自己的模型、词典和来源：

```powershell
.venv-libras/Scripts/python.exe -m slbridge.desktop_cli --model artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009 --catalog configs/libras_caption_catalog.json --camera 0
```

安装后也可使用 `slbridge-desktop`；原启动方式 `python app/desktop_app.py` 继续有效。详细操作、输入边界与验收命令见[使用说明](docs/视频摄像头与字幕_使用说明_2026-10-09.md)，当前界面设计见[桌面改版方案](docs/桌面界面改版_设计与执行_2026-10-09.md)，验证结果见[新版验收报告](docs/reports/desktop-redesign-2026-10-09.md)。

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
.venv-libras/Scripts/python.exe -m pytest -q
```

真实模型与视频一致性测试的配置见[输入与字幕使用说明](docs/视频摄像头与字幕_使用说明_2026-10-09.md)。

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
