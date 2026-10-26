# 视频输入、骨架兼容性与低分原因核验

日期：2026-10-24。针对“MediaPipe 225 维是否被直接送入论文权重”的疑问，核对当前文件评测链路、真实视频输入张量、历史逐条错误及官方训练来源。

**结论：输入必须与权重一致，这个排查方向正确；但当前文件评测没有直接误喂 MediaPipe225。已确认的差异是 SLRT 使用实验性 RTMPose 替代官方 HRNet、任意宽屏视频的构图尚未对齐训练输入，以及镜像评测入口和失败统计存在错误。不能据此把两个模型的全部低分归为同一个原因。**

## 1. 当前实际输入

| 路径 | 真实视觉输入 | 权重与任务 | 核验结果 |
|---|---|---|---|
| Legacy RF | MediaPipe 75 点的 x/y/z，共 225 维 | 孤立手势分类 | 原有接口保留 |
| Uni-Sign 文件评测 | 原视频 BGR → RTMPose Wholebody133 → 身体9、左手21、右手21、面部18；每点 x/y/confidence | `csl_daily_pose_only_slt.pth`；CSL-Daily 整句 SLT | 未使用 MediaPipe225；当前 lightweight RTMW 与论文强姿态前端仍有差别 |
| SLRT 文件评测 | 原视频 → RGB + `rtmpose-experimental` 的133点 → 63通道热力图 | `cslr_best.ckpt`；CSL-Daily 双流 S3D 短窗分类与滑窗投票 | 未使用 MediaPipe225；不是官方 HRNet 端到端复现 |

入口和实现：[`evaluate_unisign_cecsl.py`](../../scripts/evaluate_unisign_cecsl.py)、[`uni_sign.py`](../../slbridge/uni_sign.py)、[`uni_sign_preprocess.py`](../../slbridge/uni_sign_preprocess.py)、[`evaluate_slrt_datasets.py`](../../scripts/evaluate_slrt_datasets.py)、[`pose.py`](../../slbridge/slrt/pose.py)、[`preprocess.py`](../../slbridge/slrt/preprocess.py)。MediaPipe 在桌面活动观察、门控和 Legacy 分类中的使用，不能作为论文模型实际输入的证据。

真实 CE-CSL 视频抽样得到 Uni-Sign 的 `[17,9,3] / [17,21,3] / [17,21,3] / [17,18,3]` 四分支；SLRT 的 RGB 为 `[1,3,16,224,224]`，热力图为 `[1,63,16,112,112]`。第3通道是置信度，不是 MediaPipe 的深度 z。单测会拒绝 MediaPipe225 和错误关键点结构；不能用补零、截断或重排来假装兼容。

## 2. 形状一致不等于前端等价

SLRT 官方配置需要 HRNet-W48 DARK 及人体检测器提取的姿态。当前 ZIP 包不含这些模型，`slrt_hrnet_bundle/manifest.json` 也只是配置，四项 detector/HRNet 资源仍为 null。真实视频运行显式采用 `rtmpose-experimental`，使用已缓存的 RTMW 模型。

COCO-WholeBody133 点位名称可以对应，不代表检测误差、手指与嘴部精度、抖动、缺失模式及置信度分布与训练时一致。当前已证明的是严格加载938键、2001类和“相同姿态输入时网络/热力图实现的数值对照”；**这不能证明 RTMPose 与官方 HRNet 等价，也不能证明视频输入分布已对齐。** 官方方法参见 [SLRT 代码](https://github.com/FangyunWei/SLRT/tree/main/Online/CSLR)和[EMNLP 论文](https://aclanthology.org/2024.emnlp-main.619/)。

Uni-Sign 已经使用所需的 RTMPose 表示，但当前 lightweight 前端也不能视为论文前端质量的完整复现。历史已修复的宽屏归一化和方向问题，确实改变过结果，见[方向修复报告](unisign-cecsl-direction-repair-2026-10-24.md)。

## 3. 发现实际的构图与裁剪风险

样本：`D:\Zenodo\CE-CSL\CE-CSL\video\dev\A\dev-00015.mp4`，1920×1080、30 FPS、202帧。SHA256：`caba5a2e1a78990b5d7319aa1608486b73ce64ad320a28ad5c944cbdccb4475e`。

当前 SLRT 实验前端先将原图压成320×320，再估计姿态并换算到512参考坐标；RGB 还会中心裁270×270，再变224×224。这保留了官方已准备方形帧上的后续网络变换，但**不能自动保证任意原始宽屏视频具备相同人物比例和动作范围**。

本次抽取17帧（0、12、…、192），对比320方形估计与保留宽高比的640×360估计，并映射回共同坐标：

| 观察 | 数值 | 解释边界 |
|---|---|---|
| 当前方形路径中落在最终 RGB 裁剪外的手部点 | 423个点次，置信度 >0.3 | 17帧共714个可能手点；含等待、垂手与结束帧，不能换算为识别损失率 |
| 两种估计的手点位移 | 各帧手点中位位移的中位数约4.54像素，512参考坐标 | 说明先变形再检测会改变输入；没有人工关键点真值 |
| 手部原始平均置信度 | 方形0.8135、保留比例0.8589 | 未校准置信度，不是关键点准确率；也不是识别收益 |

保留比例估计仍有419个手点次在同一裁剪外，说明仅换估计分辨率不能自动解决构图问题。部分活跃动作帧也存在手点落在裁剪外；尚无人工动作边界，不能量化对词汇丢失的贡献。**本次没有擅自改变默认裁剪，也没有把稀疏17帧当成识别评测。**

证据：[逐帧几何数据](../../artifacts/slrt-input-audit-20261024/pose-geometry-audit.json)、[诊断脚本](../../artifacts/slrt-input-audit-20261024/audit_pose_geometry.py)。下图红框为最终 RGB 保留范围；绿色点为手点，两列分别为方形估计与保留比例估计。

![姿态与裁剪检查](../../artifacts/slrt-input-audit-20261024/pose-geometry-contact.png)

## 4. 已修复的镜像与评测错误

历史 `artifacts/slrt-dataset-eval-20261024/dev-mirror/detail-dev.jsonl` 中36条全部为 `mirror_keypoints_133 requires T x 133 x 3`，每条1帧、0窗。入口向批次接口传了单帧数组，并且没有同步变换外层 RGB；汇总又将异常当成空预测。**原“镜像比正常更差、方向结论与 Uni-Sign 相反”的说法已撤回。** 原始记录保留，正常方向 test500条无该错误，其历史 WER 0.8918 未改写。

本次修复：

1. 外层 RGB 镜像与姿态反射/左右点交换配对，正确补 T 维再恢复单帧。
2. 失败行 WER 为 null，失败不计为空识别；只有预期清单完整、无失败且成功记录为 final 时，发布整批 corpus WER；未完成或无协议记录只保留明确的成功子集指标。
3. resume 验证协议自身哈希、逐行协议、视频内容、参考词序列、方向、split和唯一ID；重算WER，拒绝孤立、重复、混合或被修改记录。
4. 协议绑定实际权重/词表、标签、输入视频、源码（含姿态提取器与WER实现）、实际设备/dtype和前端资产。协议变化使用新目录，不接续旧无协议评测。
5. OOV相关 `min(OOV,D)` 仅标记为删除上界，不能据此扣除“词表原因”或计算词表内WER。

修复后两条真实 dev 视频完成236/183帧及相应窗口，0输入错误；其中一条仍为空，另一条输出真实 Gloss。此项是入口跑通验证，**不是36条方向校准，也不能证明修复提高总体识别质量**。证据在 `artifacts/slrt-input-audit-20261024/mirror-corrected-probe/`，含当时协议及准确源码快照；其旧版摘要中的方向描述字符串以 `mirror_all=true`、逐条方向和 manifest 为准，该摘要标签也已在后续源码中修正。

## 5. 训练语料包含 CE-CSL 的前提不成立

当前两份权重对应 CSL-Daily。Uni-Sign 官方脚本给出 CSL-News 预训练和 CSL-Daily 微调；未发现该权重使用 CE-CSL 训练的证据。SLRT 论文报告的是 Phoenix14、Phoenix14T 和 CSL-Daily。CE-CSL 是另行采集的复杂环境数据集；中文词语重叠不等于相同视频、手语动作、签者或标注规范已经被训练过。[Uni-Sign 论文](https://arxiv.org/html/2501.15187v2)、[SLRT 论文](https://aclanthology.org/2024.emnlp-main.619/)、[CE-CSL 数据集论文](https://arxiv.org/html/2409.11960v1)。

另外，当前 Uni-Sign pose-only 权重直接生成中文句子，SLRT 输出 Gloss。文本 BLEU/编辑相似度与 Gloss WER 是不同指标，不能把两者低分简单视为同一种分类失败。

## 6. 验证状态与下一步

| 范围 | 状态 |
|---|---|
| 实际输入链路与张量检查 | 已用真实视频核验；未见 MediaPipe225 误喂 |
| 镜像入口与评测完整性 | 单元/负向测试通过；独立CPU审查14 passed |
| 相关回归 | `.conda-env` 64 passed，含实际Torch预处理；日志 `targeted-ml-regression.log`。较轻环境62 passed/2 skipped（缺Torch），单列不代替ML验证 |
| 镜像入口修复后的真实视频 | 2条通过链路验证；完整36条方向实验 NOT_RUN |
| Qt 控件与资源释放 | Python3.10 `typing.Self` 导入兼容修复后，真实权重 offscreen 检查通过启动/暂停/恢复/重置/停止/释放、Legacy窗口可见；不代表语义与实时交付通过 |
| 官方 HRNet 与实验 RTMPose 的识别对照 | **未验证：缺模型和兼容环境** |
| CSL-Daily 同域视频基线 | **未验证：本地有元数据和部分预提取姿态，未找到对应原始视频** |
| 改变裁剪/前端后的质量提升 | **未验证：没有完成受控识别对照** |

本次修改以 `scripts/evaluate_slrt_datasets.py`、`tests/test_slrt_evaluation.py`、`slbridge/slrt/runtime.py` 的窄范围修复为主；另更正两份 SLRT 报告并维护项目坑点记忆。原始500条结果、模型、数据与 Legacy 功能保留。

下一步先在相同的少量有标注 dev 视频上固定权重、方向、采样、窗口与解码，比较官方HRNet/当前RTMPose及人物构图方案；同时补同域CSL-Daily视频基线。确认输入等价之后再讨论跨域训练收益。需要新增大型模型或数据资源时单独配置，不自动下载数GB文件；不能把本次形状匹配、链路通过或裁剪诊断写成准确率提升。
