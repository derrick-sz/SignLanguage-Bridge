# Libras RGB 首轮真实训练与验证

日期：2026-10-09（Asia/Shanghai）。状态：训练、测试及独立进程原始视频验证已完成。

## 1. 数据和实验定义

采用此前规划的 12 个条目：2、7、92、93、94、100、105、106、107、110、111、113，共 360 条真实 Libras 视频。所有视频完整解码成功，并通过当前肩部／手部质量门槛；无提取失败或拒识样本。

在特征提取前按条目内 `local_informant` 冻结来源 split。每个类别 18 条训练、6 条验证、6 条测试，合计 **216／72／72**。提取后的 NPZ 保持原始集合归属，禁止同源／重复字节跨集合。没有数据增强，没有根据测试集调整配置。

分类器为 300 棵树、max_depth=20、min_samples_leaf=1、n_jobs=1 的 RF，seed=42。输入为 30 帧身体＋双手的肩部归一化 x/y、恒定 z=0 及缺失 mask。特征契约、提取代码／模型资源 hash 和依赖版本均保存在模型包。

## 2. 实际结果

| 指标 | Validation（72） | Test（72） |
|---|---:|---:|
| Accuracy | 50.00%（36/72） | **63.89%（46/72）** |
| Macro-F1 | 0.4835 | **0.6034** |
| Coverage | 100% | 100% |
| 选定概率阈值 | 0 | 0 |

阈值候选 0、0.4、0.6，只在 validation 上按 accuracy／coverage 选择，最低覆盖要求 80%。实际选定 0，因此上述结果没有通过拒识减少测试分母。质量拒识入口仍保留。

| 条目 | 目录葡语文本 | Test recall | Test F1 | Test support |
|---|---|---:|---:|---:|
| 2 | Sim | 0.833 | 0.909 | 6 |
| 7 | Quero | 0.667 | 0.500 | 6 |
| 92 | Energia | 1.000 | 0.857 | 6 |
| 93 | Solicitar | 0.000 | 0.000 | 6 |
| 94 | Serviço | 0.500 | 0.667 | 6 |
| 100 | Cair | 1.000 | 0.923 | 6 |
| 105 | Árvore | 1.000 | 0.923 | 6 |
| 106 | Poste | 0.833 | 0.909 | 6 |
| 107 | Raio | 0.667 | 0.571 | 6 |
| 110 | Menos | 0.833 | 0.714 | 6 |
| 111 | Mais | 0.333 | 0.267 | 6 |
| 113 | Religar | 0.000 | 0.000 | 6 |

![冻结测试集混淆矩阵](../../artifacts/experiments/libras-rf-pilot-20261009/test_confusion_matrix.png)

## 3. 独立视频与工程验收

另一个 Python 进程从保存的模型包加载，每个类别选择一条已冻结的测试视频，直接解码和重新提取。12 条均完成推理，其中 8 条分类正确；这组示例检查不是完整测试集指标。

12 条原始视频与对应 NPZ 的预处理特征最大差值为 **0**，缺失 mask 全部一致，类别概率最大差值为 **0**。因此本次验证没有依赖训练进程内存中的分类器或缓存替代原始视频路径。

核心 CLI 对 Árvore 的测试视频也成功返回 `item:105` 与葡语显示文本，概率为 0.7433。强制 Windows GBK 环境下仍输出 UTF-8 JSON，中文／葡语不因终端编码而失败。

**104 项测试通过**，覆盖源保护、时间信息、两种旧格式兼容、新输入契约、质量拒识、恢复缓存、冻结 split、原始字节重复约束、产物校验和及独立 CLI。

原始试点视频 hash 在训练后与冻结清单一致。首轮提取约 494 秒，提取与训练总计约 502 秒（8 分 22 秒）；这是本机本次离线批处理耗时。

实际重跑缓存验证：360／360 命中，原特征文件 mtime 与派生清单 fingerprint 均保持不变；无需重新运行提取模型。记录保存在 `cache_resume_validation.json`。

## 4. 产物位置

实验目录：`artifacts/experiments/libras-rf-pilot-20261009/`。

- `definition.json`、`source_manifest.json`、`source_split.json`：提取前冻结的实验和来源划分。
- `features/clips/*.npz`、`feature_manifest.json`、`feature_split.json`：实际特征、来源和投影后的集合。
- `training_config.json`、`environment-lock.txt`：解析配置和实际独立环境版本。
- `models/libras-rf-pilot-20261009/`：可加载模型包、完整 metrics 和 labels。
- `result.json`：训练／验证／测试汇总。
- `independent_video_validation.json`、`cli_video_prediction.json`：独立原始视频与 CLI 验证结果。
- `cache_resume_validation.json`：真实 360 条缓存恢复验证。
- `test_confusion_matrix.png`、`tests.xml`：可查看图和测试记录。

运行与实现说明见[巴西手语训练与验证](../巴西手语训练与验证_2026-10-09.md)。

## 5. 结果边界和下一步

这证明本地巴西手语数据已经可以进入真实训练与推理闭环。当前能力是 12 个来源条目的完整片段分类，不是 174／178 类全量效果、跨全局签者泛化或连续自由翻译。葡语文本来自数据目录，中文释义尚未审核。

Solicitar 和 Religar 的测试召回为 0，优先检查这些条目的动作相似性和手部提取情况；随后在相同冻结划分下对比轻量 TCN／ST-GCN。扩大词表、增加手部／面部 RGB 分支分别设实验，不用本次测试集反复筛选模型。
