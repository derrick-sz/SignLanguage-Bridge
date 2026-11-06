# 本机中国手语三维词典制作条件核查（2026-11-06）

结论：现有硬件与已运行工具支持先做动作整理、三维姿态导入、修订、GLB 导出及小规模词典试产；原视频到高质量三维动作的完整重建链路尚未配置和实跑验证。本轮只读检查目录、依赖和系统状态，并做小规模 CUDA 运算；未安装软件、下载模型、启动训练、修改驱动或现有识别配置。

## 1. 实测硬件与工具

| 项目 | 本次结果 | 解释 |
|---|---|---|
| CPU | Intel Core i5-14600KF，14 核/20 线程 | 可承担视频预处理、资源整理与离线制作；未测全库吞吐 |
| 内存 | 31.84 GiB（约 32 GB） | 建议逐段处理，避免把全部视频和关键点一次载入 |
| GPU | RTX 5060 Ti，16311 MiB（16 GB 级），驱动 617.42 | CUDA 设备能力 12.0；具体拟合模型显存和速度待单段实测 |
| 系统 | Windows 10 专业版，10.0.19045 | 当前动画制作/播放已使用此系统 |
| 磁盘可用 | C 52.9、D 137.3、E 78.0、F 193.7、G 513.6 GiB | 大体量拟合中间文件可优先规划到 G 盘；本轮未移动文件 |
| Blender | 4.5.9 LTS 便携版，实际可启动 | 同会话已有六段 GLB 制作和 1138 帧导出核验；并非视频到三维拟合证明 |
| FFmpeg / Node | 命令可运行，项目已用于视频/动画工具链 | 不需为现有资源整理重新搭建播放器 |
| CUDA 编译工具 | PATH 中 nvcc 为 12.9.41 | 自定义 CUDA 扩展的完整编译链未验证；不等同于 Torch wheel 的 CUDA 运行时版本 |
| WSL2 | docker-desktop，Running，Version 2 | 本次查询可用；未见独立 Ubuntu 发行版，也未测试容器内 GPU/拟合工具 |

WSL 状态是当前查询结果；不能由它推定 Ubuntu、Docker 拟合镜像或 Linux GPU 应用已经准备好。

## 2. Python 环境实际检查

| 环境 | 关键结果 | 可用范围 |
|---|---|---|
| `E:/githubproject/SignLanguage-Bridge/.conda-env` | Python 3.10.22、torch 2.9.1+cu128、torchvision 0.24.1+cu128，CUDA available；设备 12.0、wheel 含 sm_120 | 已通过实际 CUDA 小矩阵运算；可作为当前 GPU 运行基础，未测试完整三维重建模型 |
| `.venv-gpu` | Python 3.12.7、onnxruntime-gpu 1.23.2，无 torch/smplx | ORT 列出 CUDA/TensorRT provider；只列出 provider 不代表任意模型的 GPU 推理通过 |
| `.venv-libras` | Python 3.12.7、CPU onnxruntime 1.23.2，无 torch/smplx | 当前动画服务、资源制作驱动和测试环境；Blender 在独立进程执行 |
| `F:/DSH-envs/slrt-legacy` | Python 3.8.20、torch 1.13.1+cpu、mmcv-full 1.7.0、mmpose 0.29.0、mmdet 2.25.3 | 有旧 OpenMMLab 包，但 CUDA unavailable，不能当成现成 GPU 三维拟合环境 |

CUDA 实测为两个 16×16 的全一 GPU 张量相乘并同步，均值应为 16.0，实际为 16.0。它证明本次 torch/CUDA/显卡可执行这一算子，不证明 Uni-Sign、SMPLer-X、SMPL-X 模型加载、显存占用、质量或端到端重建均通过。

四个已检查环境中都未发现 smplx；当前 GPU 环境还缺少所检查的 mmcv/mmpose/mmdet/timm/pyrender/trimesh。不同拟合路线需要的依赖不同，不建议将这些包无选择地装进现有识别环境。

## 3. 语料与三维资产

本次再次确认 CSL-Daily/CE-CSL 原视频目录、CSL-Daily 训练标注与 keypoints 文件存在。数量仍引用同日的[全量目录清点](animation-jitter-and-corpus-expansion-2026-11-06.md)：19,528 个 CSL-Daily MP4、5,988 个 CE-CSL MP4；本次没有重新验证每个视频内容或原始 SHA。

对以下限定根目录重新搜索相关模型/数据名称，包含 `.npz/.pkl/.pth/.pt/.ckpt/.bin/.tar/.zip`，覆盖 `.pth.tar` 检查点：

- `E:/githubproject/SignLanguage-Bridge/models`
- `E:/githubproject/signbridge`
- `F:/model_weigth`
- `G:/Archive/Zenodo/Spoken2Sign`

仅找到 Spoken2Sign 源码所带 VPoser 快照 `TR00_E096.pt`（2,702,962 B）这一相关候选，未确认完整 SMPL-X/MANO 模板、SMPLer-X/手部重建权重或 SOKE/DETO 生成权重组。未反序列化这些模型文件。限定名称/目录搜索不能证明其他磁盘位置绝对没有资源。

已有 Godette GLB 和 100 段动作归档可用于播放、整理和修订；GLB 播放角色不等于 SMPL-X/MANO 人体/手部模型参数。同日后续[资源核查](../research/csl-reusable-resources-2026-11-06.md)已实际取得 SOKE 训练索引和一个逐帧参数样本，确认完整三维 ZIP 约 2.69 GiB；未下载整包，也未验证参数与当前角色的兼容性。该部分拟合数据可以优先验证复用，不能据此称本机视频重建链已配置。

## 4. 环境兼容性判断

[SMPLer-X 官方 README](https://github.com/caizhongang/SMPLer-X#install)列出的原环境为 Python 3.8、PyTorch 1.12.0、CUDA 11.3、mmcv-full 1.7.1；其 Docker 示例说明测试设备为 RTX3090 和 WSL2 Ubuntu 20.04。[PyTorch 官方 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)介绍了新架构和 CUDA 12.8 wheel 的支持。

据此及本机实际设备能力 12.0 判断，旧教程不能直接作为 RTX5060 Ti 上可用的证明；改用 Linux/容器本身也不会自动解决旧 Torch/CUDA 扩展的架构兼容。当前可运行的 cu128 环境应保留，拟合路线应单独验证兼容版本、模型加载和一段原片的实际运行。

## 5. 能开展的工作与下一步

| 工作 | 当前状态 |
|---|---|
| 现有动作整理、元数据/词条索引、审核、修订、导出、播放 | 有可用工程基础与实际产物；语言审核仍需人员参与 |
| 导入已有 CSL 三维姿态并制作首批词典 | 可开展；具体来源序列、格式、含义与角色重定向需样本验证 |
| 从本机原视频重新估计三维动作 | 需补齐独立依赖、人体/手部模型及所选方法权重；NOT_CONFIGURED / NOT_RUN |
| 全库自动重建或大型文生动作模型训练 | 本次未验证，不作显存、工期或质量承诺 |

建议先以现有硬件开展首批词典试产：先验证一段公开三维序列导入到当前播放器，再补齐重建环境并用一段本地原视频跑通，记录显存、耗时和手形质量后逐步扩大到建议的 50 个概念词条/20 个短语。三维重建、动作审核和语言审核分别验收。

核查证据位于忽略目录 `artifacts/animation-upgrade/environment-audit/`：hardware.json、conda.json、venv-gpu.json、venv-libras.json、slrt-legacy.json、resources.json、wsl.json 和 probe.py。未改现有生产代码，不重复以项目单元测试代替环境/模型实测。
