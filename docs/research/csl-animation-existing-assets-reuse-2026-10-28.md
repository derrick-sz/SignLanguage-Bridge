# CSL 三维路线：现有成果复用与 G 盘路径核验

日期：2026-10-28（Asia/Taipei）。本报告接续[技术路线评审](csl-3d-animation-route-review-2026-10-28.md)，按用户提供的 `G:\Archive\Zenodo`、`G:\ML\temp_feture` 盘点已有成果，并检查相关本地模型与 SignBridge 仓库。**没有重新训练、下载模型、搬动文件、更新其他仓库或改写冻结实验。**

## 1. 结论：首轮可以直接从已有动作开始

无需为了验证播放链路先重训 SOKE 或批量重建全部视频。本机已有可复用的动作演示归档、识别权重、真人视频及标注；应先把这些成果接到正确层级。

| 已有成果 | 当前真实位置 | 复用方式与证据边界 |
|---|---|---|
| 100 段三维演示动作及配套播放器 | `E:\githubproject\signbridge\docs\assets\signbridge-chrome-extension-motion-demo-20261022.zip` | 直接作为技术播放候选；100 段结构/有限值检查通过，语言理解本轮未测。 |
| Uni-Sign 句子模型、stage2 pose 候选 | 当前项目 `models/external/uni_sign/`，及 `configs/bundles/unisign_stage2_pose/` | 继续复用原识别链提供文本；两份目标权重大小与 SHA256 均匹配对应 manifest。不是文本到动作模型。 |
| SLRT CSL-Daily 识别模型与词表 | 当前项目 `models/external/EMNLP24_Online/CSL-Daily/ckpts/`、`meta/` | 继续提供原始 Gloss 候选；checkpoint 与词表 SHA256 均匹配 bundle。 |
| TFNet 来源权重 | `G:\Archive\Zenodo\CE-CSL\TFNet-CE-CSL-CSLDaily-32.46.pth` | 全文件 SHA256 与现有部署来源记录一致，可复用原识别/特征成果。 |
| CE-CSL 真人视频与中文/Gloss 标签 | `G:\Archive\Zenodo\CE-CSL\CE-CSL` | 后续重建/完整短语候选来源；5,988 条标注对应视频路径全存在。 |
| CSL-Daily 真人视频 | `G:\Archive\Zenodo\csl-daliy` | 当前 19,528 MP4，原 P-1 的 1,086 样本均有同名文件；未逐文件证明迁移内容一致。 |
| Libras 关键点、T 版帧/标注 | `G:\Archive\Zenodo\Libras-EQT-UECE` | 对应协议的识别与动作研究输入；展开目录没有原 RGB MP4，不是现成 SMPL-X 动作。 |
| PS2 时序实验产物 | `G:\ML\temp_feture\ps2_matrix` | 清单属于 FEED、REACTOR、SEPARATION 等工业过程实验；不纳入手语资源库。 |

静态盘点的可复用，不等于本轮已完成模型加载、Three.js 播放或语言验收。

## 2. 最有价值的现成动作归档

本地 SignBridge HEAD 为 `7ab6de3828aacf99aa758c68b5699089cb33ef75`，工作区检查为空；**不同于上一轮核验的远端 SHA `3503ff7802bc991ccf5e0753c2fe41b79689b860`**。本次没有 fetch、checkout 或更新该仓库。

本机源码没有 `local-data/`、SOKE/SMPLer-X 依赖目录，也没有远端版本的 `signbridge/assets/public-motion-v1/`。但已保存发布归档：

- 文件：`E:\githubproject\signbridge\docs\assets\signbridge-chrome-extension-motion-demo-20261022.zip`。
- 大小：**40,752,631 字节**。
- 实算 SHA256：`4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`，匹配同目录 `.sha256`。
- ZIP 161 个成员 CRC 检查无损坏；`BUILD.json` 的 **143 个文件哈希全部匹配**。
- 动作结构检查：**14 个 T08 参考 + 3 个 T08 候选 + 83 个 SMPL-X 拟合片段 = 100 段**。逐段检查姿态数组为 `frames × 55 × 3`、数值有限、帧数一致，100/100 通过。
- 归档中有 100 个不同文本标签，但包含完整句子，**不能改称 100 个标准孤立词**。

直接复用所需入口：

| ZIP 内成员 | 内容 |
|---|---|
| `demo/t08-motion.json` | 14 段预计算 token 解码动作；`text_to_sign_trained=false`。 |
| `demo/t08-candidates.json` | 3 段候选动作。 |
| `demo/experimental-index.json` | 83 段拟合动作的标签、帧率、来源摘要、资源路径及质量状态。 |
| `demo/experimental/*.json` | 83 段独立姿态资源；逐条匹配索引声明的资源哈希。 |
| `demo/experimental/motion-approval.json` | 83 段的 `motion_quality_only` 批准声明。 |
| `avatar/t08-index.json` | 参考/候选运行索引，17 条。 |

当前本地 `signbridge/dist/demo/t08-motion.json` 与 `experimental-index.json` 都是 `{"unavailable":true}`。因此，**当前 dist 不能代表这份完整动作包**。后续播放验证应使用独立目录中的完整归档，保留其原始索引、审批和运行时代码；本轮没有覆盖 dist 或解压部署。

归档还明确 `training_ready=false`、`csl_translation_trained=false`；T08 为预计算解码，而不是对任意文本现场生成。当前姿态包含 root/躯干等静止、面部系数未重定向的限制。100/100 是结构通过，**不是 100/100 语义正确**。

已生成[100 段候选目录](local-motion-candidate-catalog-2026-10-28.json)：保留原 clip ID、原文标签、归档成员位置、资源摘要、帧数/帧率与质量字段；所有条目的本轮语言审核状态为未验证。该目录是调查产物，不是生产 CSLPlanner 词典，不生成虚构的 `sign_id`，也没有开启自动播放。

## 3. 已有模型可复用到哪里

本轮只做字节与清单核对，没有使用 `torch.load` 或反序列化任意权重。

| 模型文件 | 本轮核验 | 复用目标 |
|---|---|---|
| `csl_daily_pose_only_slt.pth` | 1,186,925,112 字节；SHA256 `73bea87a…c1a6c` 匹配 bundle | 原 Uni-Sign CSL-Daily 句子识别链。 |
| `csl_daily_stage2_pose_subset.pth` | 1,947,137,670 字节；SHA256 `951601a3…fc2e3` 匹配 manifest | 原 dev/test 的 stage2 pose 候选，保持既有默认决策。 |
| `CSL-Daily/ckpts/cslr_best.ckpt` | 499,554,423 字节；SHA256 `e9fdd3a6…7cbd4d` 匹配 bundle；词表摘要也匹配 | SLRT Gloss 候选输入。默认文件位置按 `slbridge/slrt/assets.py` 的 `ckpts/`、`meta/` 约定解析。 |
| G 盘 `TFNet-CE-CSL-CSLDaily-32.46.pth` | 879,605,267 字节；SHA256 `160d35a4…6b0386` 匹配部署来源 | 既有 CE-CSL 识别/特征管线。 |

完整路径、大小与 SHA256 见[机器可读资源清单](local-model-resource-inventory-2026-10-28.json)。这些模型与已发表/已完成的实验成果可以复用；本轮没有刷新其 BLEU、WER、分类指标或运行能力。

**衔接方式**：保留 `EvidenceCandidate` 和模型原输出 → 明确目标手语/变体 → 人工确认或匹配已审核完整短语 → 原动作资产播放。SLRT 的 token ID 必须连同词表命名空间解释；不能用字符串相近就自动认定新动作的语义。

## 4. G 盘迁移的影响

详细数据检查见[G 盘 Zenodo 审计](gdrive-zenodo-reuse-audit-2026-10-28.md)。

| 历史根 | 当前根 | 当前检查 |
|---|---|---|
| `D:\Zenodo\CE-CSL\CE-CSL` | `G:\Archive\Zenodo\CE-CSL\CE-CSL` | 旧根不存在；5,988 个预期视频路径存在；代表样本 SHA256 与历史相同。 |
| `D:\Zenodo\csl-daliy` | `G:\Archive\Zenodo\csl-daliy` | 旧根不存在；P-1 的 1,086 个样本在新根有同名文件；原视频全量迁移哈希未验证。 |
| `D:\Zenodo\Libras-EQT-UECE` | `G:\Archive\Zenodo\Libras-EQT-UECE` | 旧根不存在；当前展开内容以 landmarks 和 PNG/T 版为主，不能认作原 RGB 包完整迁移。 |
| `D:\temp_feture` | `G:\ML\temp_feture` | 旧路径仍是指向新根的 junction；其中资源不属于手语实验。 |

`manifest_1086.csv` 绑定的是裁剪视频哈希，不能用它直接认证 G 盘未裁剪原视频。旧根失效后，后续运行应显式指定新输入根，并在新运行记录中保留路径解析和内容摘要。历史报告、冻结清单、旧指标及协议身份保持原样。

本次资源 JSON 中加入路径定位记录，**没有给现有应用自动接线，也没有修复全部脚本的 D 盘默认值**。例如 `build_csldaily_video_selection.py` 有 `--video-dir`，可在未来新运行时显式指向 `G:\Archive\Zenodo\csl-daliy`；对硬编码脚本则应按所选任务局部改造，避免全仓替换历史路径。

## 5. 按复用优先的下一步

1. **先验证已有动作播放**：用完整归档的 5–10 段动作做独立播放验证，检查中文标签、骨架、双手、暂停/调速、重复播放和缺项行为；不需要先启动训练或 GPU 重建。
2. **选可审核完整表达**：由目标 CSL 使用者判断现成片段的实际含义、方向和非手控缺口，决定哪些能成为首轮短语资产；若现有片段不足，再从 G 盘视频中补充。
3. **接确认后的识别结果**：复用现有 Uni-Sign/SLRT 的接口和证据记录，保持原输出可见，缺资源返回待审核；不把识别分数当语言批准。
4. **只重建缺失资源**：复用 SignBridge 现有 `sign-motion.py`/SMPLer-X 导入逻辑和固定 SOKE 特征约定；本机当前没有找到其完整依赖/私有重建产物，需要补齐后再运行。暂不重训全量 SOKE。

## 6. 本轮验证边界

- **PASS（静态）**：指定根目录盘点；CE-CSL 标签→视频路径；两份 Uni-Sign 权重、SLRT 权重/词表、G 盘 TFNet 来源权重的哈希；动作归档 checksum/CRC/143 文件清单；100 段动作形状、帧数、有限值及 83 段资源摘要。
- **未验证**：整个归档迁移的逐文件内容身份、全部数据许可细则、重定向后的动作正确性、独立语言审核。
- **NOT_RUN**：本轮模型加载/识别推理、三维播放器现场运行、动作视觉验收、SMPL-X 重建、文本到新动作生成、训练和产品端到端验收。

此次更新把“需要从零准备的路线”收敛为“先复用已有动作演示成果，再按缺口补资源”的执行顺序；没有把资源准备写成产品集成完成。
