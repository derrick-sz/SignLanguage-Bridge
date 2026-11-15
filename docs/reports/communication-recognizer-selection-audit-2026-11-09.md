# 沟通页面识别模型选择核查

日期：2026-11-09（Asia/Taipei）。范围：本机 `http://127.0.0.1:8879/communicate/` 的模型路由、已有替代模型和录制入口；未修改服务或产品实现。

## 结论与证据

- **已确认配置绕过默认策略。** 8879 网关连接 8765；实际进程参数为 `--route csl=csl-tfnet --route libras=libras-rf --device cpu --intra-op-threads 4`。`GET /v3/health` 的中文模型为 `cecsl-tfnet-onnx-20261009`，任务为 `gloss_sequence`，实际 provider 为 `CPUExecutionProvider`。
- 动画工作树 `C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge/scripts/communication_product.ps1:73,93` 固定 TFNet 别名及 CPU。本机 `artifacts/product-bridge-20261108/recognition-catalog.json` 只有 TFNet 与 Libras RF，没有 Uni-Sign 或 SLRT；`slbridge/api_cli.py` 依据这个 catalog 和显式 route 加载模型，不读取 `configs/model_defaults.json`。
- `configs/model_defaults.json` 的既有策略仍是 dev/test 用 `unisign_stage2_pose`、回退 `unisign_pose_stage1`；prod 用基础版 `unisign_pose_stage1`。当前源码中 `desktop_models.py` 调用该策略，不能推定网页/API 自动沿用。这里是配置接线差异，不能解释为 TFNet 在模型比较中胜出。
- **替代权重实际存在。** 本轮重新计算基础版 `csl_daily_pose_only_slt.pth` 的 SHA256 为 `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c`（1,186,925,112 B），S2 子集 `csl_daily_stage2_pose_subset.pth` 为 `951601a39fbbe19f633e0df9690df432056b4286609f75a3ee723c8f001fc2e3`（1,947,137,670 B），均与各自 manifest 一致。mT5 的 `pytorch_model.bin`、tokenizer 和 Uni-Sign adapter 也在；本轮未重新加载模型推理。
- Uni-Sign 的目标是 CSL-Daily 整句中文翻译，比直接把 TFNet 的 Gloss 当整句沟通输出更符合页面目标。S2 是开发/测试候选，不能称为已证明全面更好：P-1 执行报告 ADR-P1-013 的 Round-3 更新维持基础版生产默认，S2 无显著语义优势，放行条件未满足。SLRT Online 是另一条已有的 CSL-Daily Gloss 路径；SignSparK/SOKE 动作生成资源不属于视频识别模型。

## “我想睡觉”被识别为“绿色”

- 这是**用户报告的真实使用现象**，本轮没有取得该录制文件，未复现该单例，不能认定用户动作错误，也不能确定镜像、裁剪、录制或动作选材各自的贡献。
- 当前 TFNet 的词表同时有“我”“想”“睡觉”“绿色”，因此不是简单的缺失“睡觉”输出标签。
- `slbridge/sequence.py` 的 `_predict_profiled` 在预处理通过后解码 CTC；无 token 时以 `ctc_all_blank` 拒绝，有 token 时返回 `recognized`。这里没有针对陌生拍摄/签者的语义可靠性判定。`recognized` 不代表识别正确，token posterior 也不等于校准后的正确概率。
- 当前动作端配置使用 CSL-Daily T2G 规划器及词动作库，识别端却是 CE-CSL TFNet；数据来源与任务没有构成相互验证的逆向系统。即使换为 CSL-Daily 模型，也仍需验证数字人动作、真人模仿及摄像头输入，不能承诺自动识别回原文。
- `capture.mjs` 直接录制摄像头流；镜像设置只改变预览 CSS，未进入录制流。WebM 在网关转码为 MP4。以上是源码核查，**物理摄像头与本次用户录制未复测**。CPU 模式本身不是此次语义错误的证据。

## 后续选择原则

整句中文沟通应接入已有 Uni-Sign 路径并遵循既有环境策略，TFNet 保留为显式 Gloss 对照。用同一份原始录制对照基础版与 S2，保留各自原始结果；先定位镜像、画面范围和时序，再决定默认模型。不能用右侧输入文字或预期答案替代视觉识别输出。

验证范围：当前 HTTP 状态、启动脚本/catalog、adapter/defaults 调用关系、两份权重的字节数与 SHA256。模型重跑、同输入模型优劣比较、用户录制复现和真人语义验收均为 **NOT_RUN**；本轮没有新增准确率、BLEU 或 WER 结果，没有切换/重启服务。

## 后续执行更新

用户随后明确授权模型切换；同日已完成Uni-Sign基础版实际路由与MP4/WebM网关验证。上述“未切换”描述保留为核查阶段历史；当前状态见[切换报告](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/reports/communication-unisign-switch-2026-11-09.md)。
