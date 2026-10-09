# CE-CSL TFNet 接入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 将本地 CE-CSL TFNet 权重转换为可验证、可切换的 ONNX 完整片段词序列模型。

**Architecture:** 保留分类 v1，新增 RGB/CTC 序列模型包 v2。共享模型管理、上传边界与设备配置；原生基线和导出工具使用显式指定的上游源码，部署只依赖 ONNX Runtime。

**Tech Stack:** Python 3.12、PyTorch 2.9.1/cu130、ONNX 1.19.1（opset17）、ONNX Runtime 1.23.2、FastAPI、PySide6。

**Spec:** `docs/CE-CSL与Libras_桥接评估及接入建议_2026-10-09.md`。

## Global Constraints

- 上游固定提交 `ee7cb6a253a711e8f3699a1e8e0943179268060c`；权重 SHA256 `160d35a42d9f8f52a090d0131586df1631d2fcf8c9f41ee94729cae1fb6b0386`。
- 词表保留原顺序和 blank=0；RGB 全帧、JPEG 参考路径、中心裁剪与边界填充固定。
- TFNet FFT 沿 512 特征轴、forward 归一化；融合输出为 TBV logits。
- 完整片段能力不声明流式；gloss 不声明自然语言翻译；分数不声明正确率。
- 模型、数据、环境留在忽略的 artifacts 中，不上传权重和上游源码。
- 官方 split 与说话人跨 split 重叠如实记录；测试集不用于调参。

## Review Focus

- 合法重复 gloss 必须保留，空白输出不得借用其他样本。
- 输入超限或无法解码须拒绝，不能无声截断。
- v1/v2 请求在模型租约内检查任务，避免切换竞态和错误响应结构。
- CUDA provider 与实际算子设备分别验证，CPU 回退须可见。
- 不同原始帧数模 4 的原生与 ONNX 输出长度、数值及解码须一致。

## Task 1: 冻结与预处理

Files: `slbridge/cecsl.py`, `tests/test_cecsl.py`。

- [x] 实现 `RGBInputSpec`、`PreparedRGB`、`prepare_rgb_file`、`prepare_rgb_clip`。
- [x] 冻结三个官方 CSV、词表、逐视频索引及预处理 JSON。
- [x] 核验上游像素路径、规范化、填充、超限拒绝和冻结重跑一致性。

## Task 2: PyTorch/CUDA 基线

Files: `slbridge/tfnet_native.py`, `slbridge/tfnet_export.py`, `tests/test_tfnet_export.py`。

- [x] 显式加载可信上游；禁用无关预训练下载；weights_only 安全加载并严格匹配全部键。
- [x] 在真实 CE-CSL 测试视频上运行融合头，保存来源、词序列、耗时、设备、指标口径。
- [x] 将历史 README/权重指标与本地实测分开。

## Task 3: 序列接口与显示

Files: `slbridge/sequence.py`, `slbridge/contracts.py`, `slbridge/runtime.py`, `slbridge/runtime_contracts.py`, `slbridge/onnx_backend.py`, `slbridge/api.py`, `slbridge/api_schemas.py`, `slbridge/session.py`, `slbridge/captions.py`, `slbridge/desktop_presenter.py`。

- [x] v2 序列模型包验证全部哈希、词表和输入契约；分类 v1 保持兼容。
- [x] 实现正确 CTC greedy / 有界 prefix beam；逐 token 时间未知时为 null。
- [x] 新增 `/v2/recognitions`，复用切换/回滚/上传限制，阻止任务错配。
- [x] 桌面视频与结束录制可识别中文词序列，文案按模型任务切换；实视频/Qt离屏验证完成，未新增物理摄像头验收。
- [x] 核验分类回归、序列 API、热切换和错误输入。

## Task 4: 共同语义审核

Files: 新增语义候选清单及审核说明。

- [x] 固定双方词条身份，标出候选、歧义、人工审核状态与证据。
- [x] 未经双侧语言审核不得标成 reviewed，不加入自动翻译或训练标签；人工批准0项。

## Task 5: ONNX 与迁移验证

Files: 导出/打包 CLI、执行说明与真实验证报告。

- [x] 替换 FFT 为同轴实数 DFT、B1 无 pack LSTM；先通过原生数值对照。
- [x] 导出动态时间 ONNX 并验证实际 CPU/CUDA 推理、输出长度和 CTC 一致性；5个数值探针通过，修复CUDA内存池策略后500段执行通过，与原生全量报告499/500词序列相同；差异项在显式FP32下通过原生/CPU/CUDA数值和双解码对照，原报告保留。
- [x] 创建完整模型包，真实上传视频验证 API，并核验桌面适配。
- [x] 建立 Libras 迁移对照实验协议；实际720次全视频表示提取、同划分ridge对照及独立读回核验完成。
- [x] 运行适当回归测试，文档区分 passed/failed/blocked/not_run；选择性提交用于远程同步，不纳入模型、数据和环境。
