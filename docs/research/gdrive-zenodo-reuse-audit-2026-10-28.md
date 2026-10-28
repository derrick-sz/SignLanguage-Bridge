# G 盘 Zenodo 归档：手语资源复用盘点

日期：2026-10-28（Asia/Taipei）。检查范围：`G:\Archive\Zenodo`。采用只读文件清单、CSV 解析、NPY 文件头、SHA256 与现有项目来源记录比对；**没有加载 pickle/PyTorch 权重、执行归档代码、解压大包、搬动文件或做模型推理**。

## 1. 结论

该目录已包含可继续复用的识别模型来源权重、CSL 视频与标签，以及 Libras 的关键点/帧数据。它可以明显减少重新下载和重新制作原始资料的工作，但本次展开目录中没有发现可直接作为三维动画库的 SMPL-X/MANO/SOKE 动作参数或 glTF/GLB 资产。

优先复用顺序：

1. 已迁移、且哈希与部署来源一致的 TFNet checkpoint，用于既有识别/特征管线；不能反向调用生成动画。
2. CE-CSL 与 CSL-Daily 的真人视频，作为后续三维重建、人工裁剪/审核与短语候选的源素材。
3. Libras 的 126/318 维序列用于其对应的识别/重建研究输入；它们不是 SMPL-X 旋转，也不应强行接到既有 225 维输入。

另一个目录 `G:\ML\temp_feture` 由主报告另查，本报告不对其内容作结论。

## 2. 当前实际文件结构与数量

按 `rg --files --hidden --no-ignore G:/Archive/Zenodo` 的展开文件名盘点，共 **622,684 个文件**。

| 实际根目录 | 文件数量与类型 | 对三维路线的可复用部分 | 当前不能据此声称 |
|---|---|---|---|
| `G:\Archive\Zenodo\CE-CSL` | 5,992：5,988 MP4、3 CSV、1 PTH | 真人视频、中文/Gloss 标注、TFNet 识别来源权重 | 已有 5,988 段可播放三维动作或已审核孤立词库 |
| `G:\Archive\Zenodo\csl-daliy` | 19,528，均为平铺 MP4 | CSL-Daily 真人源视频，可复用已有选择清单查找候选 | 已具完整数据集/全部 test；此目录本身没有标注或三维参数 |
| `G:\Archive\Zenodo\Libras-EQT-UECE` | 597,164：586,477 PNG、10,685 NPY、1 README、1 ZIP | 关键点序列、T 版预处理帧、划分文件 | 已有 Libras 三维动画或所有原始 RGB 视频 |

合计：25,516 MP4、586,477 PNG、10,685 NPY、3 CSV、1 PTH、1 ZIP、1 Markdown。该合计是文件系统清单，不包含 ZIP 内部条目的计数。

### CE-CSL

- 数据根：`G:\Archive\Zenodo\CE-CSL\CE-CSL`。
- 标注：`label/train.csv`、`label/dev.csv`、`label/test.csv`，字段为 `Number,Translator,Chinese Sentences,Gloss,Note`。
- 当前解析行数为 **4,973 / 515 / 500**。逐条按 `video/<split>/<Translator>/<Number>.mp4` 检查，**5,988 条预期路径全部存在**，各 split 视频数量也对应。
- 这是文件存在性核验，未逐条做视频解码、内容哈希或语言审核。句级 Gloss 也不提供已确认的词级动作边界，做孤立词资产前仍需对齐/裁剪审核。

### CSL-Daily

- 实际目录拼写是 **`csl-daliy`**，不是 `csl-daily`，调用时保留当前真实路径。
- 文件例：`S000000_P0000_T00.mp4`。本次只见 19,528 个 MP4，没有伴随 CSV、PKL、NPZ 或模型文件。
- 当前项目的 `experiments/p1/datasets/manifest_1086.csv` 的 1,086 个 `sample_id` 均能在新根找到同名 MP4，**1,086/1,086 路径存在**。
- 该检查不重新产生评测成绩；清单属于既有 `retrospective_seen_1086`。同名文件存在不能替代原视频哈希，也不能证明余下 90 条 test 已补齐。

### Libras

主要结构：

```text
G:\Archive\Zenodo\Libras-EQT-UECE\
  README.md
  Libras-EQT-UECE-T.zip
  Landmarks\
    Libras-EQT-UECE (Hand Landmarks)\
    Libras-EQT-UECE (Landmarks)\
  Libras-EQT-UECE\
    annotations\
    features\
```

- Hand Landmarks 有 **5,347 个 NPY**；全 Landmarks 有 **5,335 个 NPY**。两边数量不同，复用前需要按身份键连接并报告缺项，不能按目录顺序 zip。
- 在同一 `100_Cair/INFORMANTE 1/IMG_4868.npy` 样例上，仅解析 NPY 文件头，分别为 **`(119,126)`** 与 **`(119,318)`**，float32、小端、非 Fortran 顺序。这只证明该样例的形状/类型；没有认证每个维度的坐标语义，也没有把它当作真实 3D 旋转或校准深度。
- `annotations/train.npy`、`dev.npy`、`test.npy` 文件头均为 **object dtype、scalar shape**。本次未反序列化其 pickle 内容，不据此推定 Gloss 监督或划分质量。
- 本目录展开文件清单中 **没有 MP4**。本地 README 描述 RGB 包与 T 包是分开的；目前此根实际具有 T 版帧/annotations 和 Landmarks，不能写成“原 RGB 视频已在这里全部恢复”。
- `Libras-EQT-UECE-T.zip` 大小为 **35,604,881,215 字节**；本次未解压或检查 ZIP 内部目录，不能以展开清单证明压缩包内部绝无其他资源。
- README 仅标示 “LIBRAS-EQT-UECE Free Use License”，没有在本目录见到独立完整许可文本。来源授权应继续引用既有登记或取得原始条款，不从该名称推断全部使用范围。

## 3. 已核验的迁移身份

### 识别来源权重：内容哈希一致

实际文件：`G:\Archive\Zenodo\CE-CSL\TFNet-CE-CSL-CSLDaily-32.46.pth`，大小 **879,605,267 字节**。

实算 SHA256：

```text
160d35a42d9f8f52a090d0131586df1631d2fcf8c9f41ee94729cae1fb6b0386
```

与本仓库 `artifacts/models/cecsl-tfnet-onnx-20261009/manifest.json` 中的来源 `checkpoint_sha256` 完全一致；`provenance.json` 也记录同一摘要。这支持**新路径该文件与既有部署来源权重字节一致**，不是仅凭文件名识别。未执行 PyTorch 加载或新推理。

### 代表视频：单文件迁移身份一致

实际文件：`G:\Archive\Zenodo\CE-CSL\CE-CSL\video\dev\A\dev-00015.mp4`。

实算 SHA256：

```text
caba5a2e1a78990b5d7319aa1608486b73ce64ad320a28ad5c944cbdccb4475e
```

与 `docs/reports/model-input-compatibility-audit-2026-10-24.md` 的原 `D:\Zenodo\...\dev-00015.mp4` 一致。**该证据只覆盖该样本，不能写成整个数据集已逐文件验证迁移完整。**

## 4. 旧路径现已失效

本次 `Test-Path` 检查：`D:\Zenodo`、其 CE-CSL 数据根、`csl-daliy`、Libras 数据根均不存在，因此这些被检查路径不存在可承接访问的 junction。

仍有旧 D 盘路径的例子：

- `scripts/build_csldaily_video_selection.py`：`DEFAULT_VIDEO_DIR`。
- `scripts/align_slrt_csl_daily_subset.py` 与 `align_slrt_csl_daily_train100.py`：`VIDEO_DIR`。
- `scripts/evaluate_slrt_datasets.py`：CE-CSL 标签/视频根。
- `scripts/evaluate_unisign_cecsl.py`：CE-CSL 根。
- `scripts/audit_libras_landmarks_20261010.py`、`audit_libras_join_20261010.py`：默认输入根。
- `experiments/p1/datasets/manifest_1086.csv`：`raw_video_path` 仍是原 D 盘路径；其历史 `raw_video_present=True` 不是当前存在性判断。

建议后续运行时显式使用 G 盘输入根，或由统一的运行时数据根解析器映射。**不要全仓替换历史证据中的 D 路径、改写冻结清单或重新计算旧协议身份来掩盖迁移。** 新运行另存新路径解析记录，并继续核对内容哈希。本文仅记录风险，没有修改脚本、配置、manifest 或 junction。

## 5. 复用到三维路线的具体边界

| 资源 | 可直接复用 | 需要新增/转换 | 状态 |
|---|---|---|---|
| TFNet 来源权重 | 既有识别网络/特征来源身份 | 接当前环境的加载与推理；输出映射到已审核词条 | 哈希核验 PASS；本轮加载/推理 NOT_RUN |
| CE-CSL 视频+句级中文/Gloss | 原始视频选择、标注索引、重建源素材 | 手部/身体/脸估计、SMPL-X 拟合、词界或完整短语审核、重定向 | 文件路径核验 PASS；三维制作/语言审核 NOT_RUN |
| CSL-Daily 19,528 视频 | 现有样本 ID 选择与真人对照素材 | 引用项目已有标注/划分，补来源和内容哈希，再做重建 | 清单 1,086 路径存在；完整迁移哈希/完整 test NOT_VERIFIED |
| Libras 126/318 序列 | 对应特征协议研究、片段时序参考 | 坐标协议核验、缺项连接、3D 拟合与独立 Libras 语言审核 | 文件头样例核验；不是现成 SMPL-X |
| Libras PNG/T版标注 | 既有预处理帧、重新审计的候选输入 | 来源视频身份、裁剪/帧率/监督核验 | 未解码帧、未加载 object 标注；NOT_RUN |

**本次没有发现足以取消三维重建与语言审核环节的现成资产。** 应优先利用已有真人视频制作少量可核验短语候选，复用识别模型提供候选文本/Gloss；不要把识别权重、二维关键点或句级标签直接标成三维生成成果。

## 6. 可复核的一手证据入口

- 文件系统：上述 G 盘路径、文件名/计数、NPY 原始头、CSV 内容和两个实算 SHA256。
- 本地 README：`G:\Archive\Zenodo\Libras-EQT-UECE\README.md`。
- 本项目来源 manifest：`artifacts/models/cecsl-tfnet-onnx-20261009/manifest.json` 的嵌套 `provenance` 中 `checkpoint_path` / `checkpoint_sha256` 字段（本次在第 431–432 行），以及同目录 `provenance.json`。
- 视频历史身份：`docs/reports/model-input-compatibility-audit-2026-10-24.md`。
- 既有样本清单：`experiments/p1/datasets/manifest_1086.csv`；本次仅用 ID 与新根连接，不更改其冻结内容。

复核命令种类：`Get-ChildItem`、`rg --files`、UTF-8 CSV 解析、按 NPY 魔数/版本读取文件头、`Get-FileHash -Algorithm SHA256`、`Test-Path`。没有以本次资源盘点刷新或重写任何模型成绩。
