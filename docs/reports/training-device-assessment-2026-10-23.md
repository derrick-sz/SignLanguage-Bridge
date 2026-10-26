# 当前设备训练能力评估

评估日期：2026-10-23（Asia/Taipei）  
基线：分支 `codex/model-contract-auto-captions`，HEAD `cf291c2a81558faf6de0dc5a7243e730531e4367`  
范围：本机硬件、PyTorch/CUDA、论文 Online/CSLR ISLR 结构的合成前向/反向与 Adam 单步探测，以及本地数据/权重存在性。未启动真实训练，未报告准确率。

## 结论

**当前设备可以支持本地手语模型训练，但当前项目环境不能直接原样运行 Zuo 2024 的官方训练命令。**

- 适合直接开展：已有 ONNX 模型评估、Libras 现有特征上的轻量分类器/TCN、冻结视觉骨干后的小规模训练、官方模型短批数据探查。
- 有条件可以开展：Zuo 的双流 S3D ISLR 训练。需要现代 PyTorch + BF16，或把 FP32 微批次降到 12 个片段并处理损失/批次语义变化。
- 当前不能宣称可复现：官方训练脚本在当前环境中缺少训练依赖、数据/权重，且 AMP 与梯度累积路径不能按配置可靠工作。

## 硬件与系统

| 项目 | 实测 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti，16,283 MiB（nvidia-smi 显示 16,311 MiB） |
| Compute Capability | 12.0（Blackwell） |
| 驱动 | 616.92；驱动 UMD 显示 CUDA 13.4 |
| CPU | Intel Core i5-14600KF，14 核 / 20 线程 |
| 内存 | 31.8 GiB |
| 磁盘空闲 | D: 80.9 GiB；E: 71.5 GiB；F: 262.7 GiB |
| CUDA Toolkit | 12.9（nvcc 12.9.41）；这只是工具链版本，PyTorch 使用自带 CUDA runtime |
| Docker | CLI 存在，但 daemon 未启动 |
| WSL | 仅有停止的 `docker-desktop`，没有可用通用 Linux 发行版 |

## PyTorch 兼容性

已在 `D:\anaconda3\python.exe` 中实际执行 GPU 张量和矩阵乘法：

```text
torch = 2.9.1+cu130
compiled CUDA = 13.0
cuda_available = True
device = NVIDIA GeForce RTX 5060 Ti
capability = (12, 0)
```

结论：当前驱动与这套 PyTorch 对 Blackwell `sm_120` 可用，不只是“装上了但无法运行”。

但各虚拟环境状态不同：

| Python 环境 | torch/torchvision | transformers/timm | 结论 |
|---|---|---|---|
| `.venv-gpu` | 缺失 | 缺失 | 当前桌面/ONNX 环境，不能训练 PyTorch 论文模型 |
| `.venv-libras` | 缺失 | 缺失 | 不能直接训练 PyTorch 网络 |
| `D:\anaconda3\python.exe` | 有，torch 2.9.1+cu130 | 缺失 | GPU 可用；需补训练依赖后才能运行原代码 |
| 系统 `python` | 与 Anaconda 相同 | 缺失 | 同上 |

## 模型级显存探测

探测使用 CSL-Daily ISLR 配置的实际双流模型结构：两个 S3D 流、RGB 224×224、关键点热图 112×112、16 帧、2001 类分类头。总参数约 **41,554,867**。探测使用 scratch 结构和合成输入，只测结构与算子内存，不是训练结果。

官方 ISLR 数据配置的 `batch_size=4` 会按 `bag_size=6` 展开为 **24 个片段**；这不是普通的 4 片段批次。

| 微批次片段数 | 精度 | 操作 | 峰值分配 | 峰值保留 | 判断 |
|---:|---|---|---:|---:|---|
| 4 | FP32 | 双流前向 + 反向 | 3.1 GiB | 4.1 GiB | 宽松 |
| 12 | FP32 | 双流前向 + 反向 | 8.5 GiB | 11.5 GiB | 可运行，留约 4.5 GiB |
| 24 | FP32 | 双流前向 + 反向 | 16.8 GiB | 20.7 GiB | 超过物理显存，发生严重超额使用，不适合 |
| 24 | BF16 autocast | 双流前向 + 反向 + Adam step | 10.8 GiB | 11.8 GiB | 通过，合成 loss 为有限值 |

补充：24 片段 FP32 单步耗时约 35.6 秒；24 片段 BF16 + Adam 合成单步约 5.3 秒。两者只证明不会立即 OOM，**不能推导完整 epoch 时间、训练准确率或真实吞吐**。实际训练还会受到 MediaPipe/HRNet、视频解码、增强、I/O 和显存碎片影响。

## 当前训练阻塞项

1. **依赖未就绪。** 官方 requirements 固定旧版 `torch==1.9.0+cu102`，不适合 RTX 5060 Ti 的 `sm_120`。应建立独立现代环境，不能覆盖当前桌面环境。
2. **AMP 路径有代码缺陷。** AMP 开启后模型没有按原路径包装，后续仍访问 `model.module.set_train()`；需要修复单卡/非 AMP 或单进程 DDP 分支。
3. **梯度累积未完成。** 源码中 `grad_acc_step` 的除法和 step 条件被注释，不能靠改 YAML 直接降低显存。
4. **官方资产缺失。** `Paper-algorithm/SLRT-main/data`、`pretrained_models`、`Online/CSLR/results` 当前均不存在；缺 K400 S3D、HRNet 关键点和 ISLR checkpoint。
5. **本地数据存在但任务不同。** CE-CSL 有 5,988 个 MP4、约 9.17 GiB，且标注含句级 Gloss；Libras 本地有约 34.17 GiB PNG、约 0.97 GiB NPY。Libras 历史 Gloss 审计为空，不能直接把 `label_char` 当作 Gloss，也不能直接复现 Online CSLR 的监督训练。
6. **Docker/WSL 备用环境未启动。** Docker CLI 存在但 daemon 停止；当前没有可直接使用的 Linux 训练发行版。

## 建议的训练门槛

### 可以立即执行

- 用现有 ONNX/CUDA 运行时做推理、特征提取和基线评估。
- 在现有 Libras 预提取特征上训练 NumPy/轻量 TCN/分类头，固定训练/验证/测试划分和学习率预算。
- 为 CE-CSL 建立数据审计、词表、划分和数据加载器；先做 CPU/小批量 smoke test。

### 在正式训练前必须完成

1. 新建隔离环境，固定 `torch 2.9.1+cu130` 及对应 `torchvision`、`timm`、`transformers`、`tensorboard` 版本。
2. 修正 BF16 autocast 与单 GPU 训练循环；先做单步前向、反向、Adam step、checkpoint save/load。
3. 原有效批次 24 采用 BF16；若坚持 FP32，则将有效片段数降到 12 并明确这改变了原批次的 gloss-level loss 语义。
4. 用真实 CE-CSL 小样本验证 RGB/关键点同步、张量形状、语音/字幕字段隔离和显存泄漏。
5. 只在 dev 集选模型和超参数，固定版本后测试 test；记录 data hash、模型 hash、随机种子和逐样本输出。

## 状态边界

| 项目 | 状态 |
|---|---|
| GPU/CUDA/PyTorch 兼容 | 已实测通过 |
| 合成双流前向/反向/Adam 显存 | 已实测 |
| 官方未修改训练入口 | 未运行；依赖与资产未就绪 |
| CE-CSL/Libras 真实训练 | NOT_RUN |
| 准确率、收敛、训练 epoch 时间 | NOT_RUN |
| 摄像头长时训练/推理热稳定性 | NOT_RUN |
