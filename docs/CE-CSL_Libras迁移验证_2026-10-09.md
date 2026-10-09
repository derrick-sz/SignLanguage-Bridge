# CE-CSL → Libras 冻结视觉表示迁移探针

迁移实验已完成，以下数值来自本地实际运行。

固定协议标识：`cfa1bcedde1b1ca3c1ade9332e1737af2a8555223d8e710c7c81f584c8c955ac`。协议及逐条视频身份记录见 `artifacts/cecsl-integration-20261009/transfer/protocol.json`；可提交实现见 `scripts/experiments/cecsl_libras_transfer.py`。当前样本数360、类别数12，沿用历史 feature_split 的216训练/72验证/72测试成员及顺序，核对原始视频与历史特征文件哈希，不重新划分或补入曾被 MediaPipe 过滤的样本。

冻结命令：`python scripts/experiments/cecsl_libras_transfer.py freeze --source-dir artifacts/cecsl-tfnet-audit-20261009/TFNet`。实际运行命令：`python scripts/experiments/cecsl_libras_transfer.py run --checkpoint D:/Zenodo/CE-CSL/TFNet-CE-CSL-CSLDaily-32.46.pth --device cuda:0`。PyTorch/CUDA 与独立审计源码、下载 checkpoint 均属于本地可选实验依赖；未将上游源码或原 checkpoint 复制到新脚本。

CE-CSL 对照使用固定 checkpoint 严格加载后的 MAM ResNet34 视觉骨干；随机对照使用同一审计源码架构与 torch seed42。两者均 eval、冻结参数和归一化缓冲，不使用原 CE-CSL CTC 分类头。每段完整原始 RGB 视频采用相同 TFNet JPEG/裁剪/归一化/边界填充，仅将原始帧位置6:6+n的512维视觉特征按时间均值与总体标准差拼接成1024维向量。

两组都只从训练集拟合特征标准化与 one-hot 岭分类头，固定 λ 候选0.01/1/100；仅用验证准确率选 λ，平局按候选顺序选较小 λ，先保存配置再读取测试预测，不重训 train+validation。所有缓存键绑定视频、模型状态、源码、预处理、执行环境和协议哈希。

CE 组验证选中 λ=100（其余 λ=0.01、1 的验证准确率均为94.4444%）；随机组验证选中 λ=0.01（λ=1、100 均为12.5000%）。两个配置及训练头都先于任一组测试结果保存；测试结果未参与调参。

| 方法 | 验证准确率 | 测试准确率 | 测试 macro-F1 | 测试 N |
|---|---:|---:|---:|---:|
| MediaPipe + RF（历史上下文） | 50.0000% | 63.8889% | 0.603377 | 72 |
| cecsl_pretrained + ridge | 95.8333% | 100.0000% | 1.000000 | 72 |
| random_seed42 + ridge | 13.8889% | 23.6111% | 0.220878 | 72 |

单个seed的有限子集结果不能证明统计显著性或完整 Libras 数据集能力。历史分组只使用 local_informant，跨条目全局签者身份尚未核实；保留旧 MediaPipe 就绪子集的选择偏差。RF 与本探针的输入表示和分类头都不同，因此 RF 表格只提供上下文；CE 与随机 MAM 之间的同头对照才对应当前协议下的预训练表示差异。这里没有 CE-CSL/Libras 词表对齐、中葡句子翻译或实时因果推理证据。

实际结果及逐类混淆/逐样本预测见 `transfer/result.json`。CE 相对随机的测试准确率差为 76.3889 个百分点，macro-F1 差为 0.779122；该差值仅描述当前固定划分。

另用独立 CPU 读回脚本 `artifacts/cecsl-integration-20261009/transfer/verify_saved_results.py` 核对完成结果，证据为同目录 `verification.json`。重新核对360个原始视频与360个历史就绪特征文件哈希；三个划分之间原始视频 SHA256 重合数均为0。两组720份缓存的哈希、样本顺序和向量逐项匹配保存的360×1024 float32 特征矩阵；每组360行特征均不同、特征非恒定，模型状态哈希及两组特征不同。只取216个训练成员独立复算训练均值/标准差和三个 λ 的岭解，与保存的训练头及验证选型一致；从保存头的分数重新计算两组验证和测试的全部预测，与结果文件一致。测试每类各6条，CE正确72/72、随机正确17/72。该核对排除了完全相同视频的跨划分重复，不建立全局签者或背景独立性。

实际运行使用 RTX 5060 Ti、PyTorch 2.9.1+cu130、NumPy 2.2.6、OpenCV 4.12.0、float32视觉前向；CUDA矩阵运算与cuDNN的TF32均关闭，启用确定性运算，随机种子42。完成总耗时608.42秒，CE特征提取362.36秒，随机特征提取244.01秒；这些是本次运行记录。实验开始时与完整CE-CSL ONNX评估共享GPU，后者随后因其ORT内存池上限失败而结束；因此这些耗时不属于受控速度基准。运行上下文、日志和退出状态保存在 `transfer/execution_context.json`、`run.stdout.log`、`run.stderr.log`、`state.json`，实际退出码0且标准错误为空。
