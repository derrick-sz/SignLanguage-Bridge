# Uni-Sign 跨数据集方向校正：修复与校验记录

日期：2026-10-24。代码工作区：`E:\githubproject\SignLanguage-Bridge`。

本轮已修复输入方向校正缺失、评测缓存与续跑身份校验不足、裁剪探针键名不匹配三类问题。使用开发集锁定的方向配置，CE-CSL 500 条测试集回归的 BLEU-ZH 从 **7.9934 提升到 12.3019**，字符编辑相似度从 **22.5123% 提升到 30.6325%**。改善说明方向对齐确实贡献了部分跨域损失；目标域泛化问题仍未解决。

本轮没有训练权重。500 条回归使用历史姿态缓存重新解码；另外用真实视频重新提取验证公开推理入口。测试集在此前诊断时已经被检查过，因此这份结果属于修复回归，不能作为新的盲测成绩或产品识别准确率。

## 1. 结果与证据边界

| 校验对象 | 修复前 | 修复后 | 证据范围 |
|---|---:|---:|---|
| dev 独立验证 24 条：BLEU-ZH | 6.2325 | 10.3210 | 与 36 条方向校准样本无交集；同一批签者 |
| dev 独立验证 24 条：字符编辑相似度 | 22.1977% | 31.1895% | 验证分数不参与方向选择 |
| test 500 条：BLEU-ZH | 7.9934 | 12.3019 | 历史姿态缓存重新解码，500/500 成功 |
| test 500 条：chrF | 9.3751 | 13.3098 | 相同参考句与计分实现 |
| test 500 条：字符编辑相似度 | 22.5123% | 30.6325% | 平均增加 8.1202 个百分点 |
| test 500 条：字符 ROUGE-L F1 | 30.0397% | 40.3772% | 字符级 F1 |
| test 500 条：整句完全匹配 | 0/500 | 1/500 | 不能将 BLEU 或编辑相似度当成整句准确率 |

BLEU-ZH 使用 SacreBLEU 的 `tokenize=zh`；字符编辑相似度为 `1 - Levenshtein距离 / max(参考长度, 预测长度)`，先执行相同文本归一化。BLEU/chrF 按其 0–100 量表列出。

逐条配对后，169 条编辑相似度上升、306 条不变、25 条下降。保持原方向的 295 条预测全部复现原结果，剩余 205 条使用锁定的镜像方向。10,000 次按样本配对 bootstrap 的编辑相似度均值差 95% 区间为 **+6.62 至 +9.65 个百分点**；这是当前已检查样本上的描述性区间，不涵盖新签者、新设备或新数据集的不确定性。

前一轮几何归一化修复的 4.72 → 7.99 BLEU 已在历史报告记录；本轮方向修复的直接比较基线是 7.99。上游 CSL-Daily 的约 25.70 BLEU 来自其预测快照，并非本机同条件重新推理，不能据此精确分摊剩余跨域差距。

## 2. 已落实的代码修复

| 问题 | 最终行为 | 代码位置 |
|---|---|---|
| 没有显式姿态方向校正 | `preserve` 为默认；`mirror` 同步反射 x、交换身体／面部左右点与左右手，保留置信度；不修改原缓存 | [uni_sign_preprocess.py](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign_preprocess.py:43) |
| 文件与采集片段方向行为可能不一致 | 两个公开推理入口均在规范姿态之后应用同一校正；结果记录实际方向 | [uni_sign.py](E:/githubproject/SignLanguage-Bridge/slbridge/uni_sign.py:932) |
| 缓存只靠大小和时间判断视频身份 | 加入视频内容 SHA256；协议绑定姿态模式、实际设备、帧数与归一化，缓存加载后保持只读 | [evaluate_unisign_cecsl.py](E:/githubproject/SignLanguage-Bridge/scripts/evaluate_unisign_cecsl.py:106) |
| 变更解码或方向后可能混入旧预测 | 运行协议绑定权重、源版本、分词器配置、实际设备／dtype、beam、输出长度、方向表、标签及所选视频；不匹配时要求新目录 | [evaluate_unisign_cecsl.py](E:/githubproject/SignLanguage-Bridge/scripts/evaluate_unisign_cecsl.py:129) |
| 校准续跑可混入无来源或错误记录 | 拒绝无 manifest 的孤立预测；逐行检查样本、阶段、参考句、视频内容及重算分数 | [calibrate_unisign_cecsl_direction.py](E:/githubproject/SignLanguage-Bridge/scripts/calibrate_unisign_cecsl_direction.py:52) |
| 裁剪探针生成 `5-95`，调用却请求 `05-95` | 两位数格式统一；索引改为直接查表，键名错误立即失败；结果写入新文件 | [trim_eval_stratified.py](E:/githubproject/SignLanguage-Bridge/.tmp-unisign/trim_eval_stratified.py:35) |

方向操作以请求的最终方向为准，重复调用不会二次翻转。单元测试覆盖身体、手、面部的独立解剖位置，以及置信度、质量字段、只读性和默认行为。

## 3. 方向配置如何产生

脚本从官方 515 条 dev 标签中，对每位签者按编号取五个等距位置：首／中／末共 3 条用于校准，两个四分位位置共 2 条用于验证。因此共 **36 条校准、24 条验证**。这属于同签者不同视频验证。

每位签者只有当校准样本的平均字符编辑相似度满足 `mirror - preserve > 0.02` 时才选择镜像，否则保持原方向。方向表在验证开始前落盘锁定；验证得分不会改写选择。没有使用 test 参考句逐条挑选两种方向中的更优答案。

| 方向 | CE-CSL 签者 |
|---|---|
| mirror | B、E、G、I、L |
| preserve | A、C、D、F、H、J、K |

H 的 dev 校准增益约 0.0133，未达到预设阈值，因此保持原方向。L 在 2 条 dev 验证上的平均编辑相似度略降，说明固定方向并非每条视频都受益；本轮保留锁定配置，没有依据验证／测试表现回调方向。

该表仅适用于这批 CE-CSL 录制数据；它不是自动摄像头镜像检测器，也不能证明所有被校正的视频都在像素层面被镜像过。惯用手、姿态前端和训练分布都可能造成方向敏感性。

## 4. 真实视频公开入口校验

实际使用 CUDA、RTMPose lightweight 和 bfloat16 Uni-Sign，通过 `UniSignAdapter` 完成以下 11 次推理：

| 检查 | 次数 | 结果 |
|---|---:|---|
| B/E/G/D 各 1 条 dev 验证视频，两种方向各一次，重新从视频提取 | 8 | 8/8 识别调用完成；预测文本与校准时相同方向的缓存预测完全一致 |
| 将 B/D 原视频逐帧水平翻转、重新编码，再重新提取 | 2 | 2/2 调用完成；未强求像素翻转后的神经姿态提取等同于原姿态坐标反射 |
| B 的完整视频缩放为 640×360，构造采集片段走 `recognize_clip` | 1 | 调用完成，记录 mirror；这是视频回放，不是现场摄像头 |

11/11 的“通过”指运行状态、方向字段和预期入口契约通过。它不代表 11 句语义正确。真人语义评审与现场摄像头验收均 **NOT_RUN**。这些任务曾与其他 GPU 工作并行，耗时不用于性能对比。

完整逐次预测、质量字段与设备信息见 [real-video-validation.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/real-video-validation.json)。

## 5. 测试与独立复审

| 校验 | 结果 |
|---|---|
| Uni-Sign 定向测试：预处理、公开入口、资产配置、评测协议 | **34 passed**，无跳过；独立复审另在禁用 GPU 时复跑通过 |
| 全项目回归 `.venv-gpu` | **786 passed, 10 skipped, 15 warnings**，退出 0；跳过项不计为通过 |
| 公开入口负控：临时将方向操作替换为无操作 | 2 个 mirror 用例失败，2 个 preserve 用例通过，证明测试能捕获入口未接入校正 |
| 缓存内容负控 | 文件大小、mtime 不变但内容改变时，SHA256 校验拒绝使用缓存 |
| 续跑负控 | 方向、beam、设备或校准身份不匹配时拒绝混用；无 manifest 的校准预测被拒绝 |
| 校准 CLI 完整续跑 | 退出 0；独立复审确认模型加载 0 次，重新核验 60 个视频哈希并复算分数一致 |
| 裁剪键修复 | 200 帧合成输入上，3 个请求变体均实际裁剪；全语料裁剪指标重跑 **NOT_RUN** |
| 独立代码／dev 证据复审 | 无剩余必须修正项；该复审未运行 GPU，500 条回归与真实视频由主任务另行完成 |

全项目日志：[final-pytest.log](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/final-pytest.log)。复审中发现的缓存设备遗漏、校准孤立记录、实际 dtype、dev 域标记、视频内容身份与只读属性丢失均已修复后再验。

## 6. 固定版本、来源与历史保护

- 权重：`csl_daily_pose_only_slt.pth`，SHA256 `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c`。
- 上游源码：[Uni-Sign 固定提交 eed438b](https://github.com/ZechengLi19/Uni-Sign/tree/eed438bcb49e30405cd6ccdfcccca330c134e830)。
- 参数：RTMPose lightweight、isotropic_short_side、max_frames=256、max_video_frames=600、beam=4、max_new_tokens=100；实际姿态设备 cuda、模型 bfloat16。
- 最终方向表 SHA256：`4643536e4205634733bdf6cc0a088f76baeda0918d5211d1cf9e4e9ba7fe2730`。
- test 运行协议 ID：`52217b578a1159502386c3540075901e63c0ea5d0441536aa7d617495d3373b6`。
- 60 条 dev 姿态来自本轮真实提取；最终协议目录迁移保持其数值不变。500 条 test 姿态来自已审计历史缓存，导入时保留原文件，逐条校验数组、视频元数据并记录视频内容哈希。历史缓存生成时没有内容哈希，本轮补录不能反向证明当时像素完全相同；295 条未改方向的预测全部复现、另外 8 次真实视频输出一致，为来源提供补充证据。
- 用户提供的历史工作簿、原始比较 CSV、CE-CSL test 标签与 CSL-Daily train 标签当前 SHA256 均与诊断时一致，见 [historical-input-integrity.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/historical-input-integrity.json)。原历史报告保留；本轮代码修改前快照与哈希在 `repair-20261024/before/` 及 `before-sha256.json`。

默认模型包仍保持原方向，按 CE-CSL 签者应用校准表的行为仅在显式传入 `--direction-map` 时启用。没有下载目标域训练权重，没有做 CE-CSL 微调或同域 CSL-Daily 视频测试，没有提交或推送代码。

## 7. 使用与复现

在项目根目录执行，输出目录需为空或与现有协议完全一致。以下 test 命令使用新的输出目录，复用本轮已审计、绑定协议的缓存重新解码。若输出目录已经完成且协议相同，脚本只核验并续用已有预测；省略 `--pose-cache-dir` 并使用全新的输出目录与缓存位置，才会重新从视频提取。

```powershell
.\.conda-env\python.exe scripts\evaluate_unisign_cecsl.py `
  --split test --device cuda `
  --direction-map artifacts\unisign-cecsl-eval\repair-20261024\dev-calibration-final\direction-map.json `
  --pose-cache-dir artifacts\unisign-cecsl-eval\repair-20261024\test-canonical-cache `
  --output-dir artifacts\unisign-cecsl-eval\repair-20261024\test-locked-directions-replay
```

重新执行 dev 校准时使用新目录：

```powershell
.\.conda-env\python.exe scripts\calibrate_unisign_cecsl_direction.py `
  --device cuda --output-dir artifacts\unisign-cecsl-eval\direction-calibration-new
```

单个已确认需要镜像校正的视频可以通过公开脚本直接运行：

```powershell
.\.conda-env\python.exe scripts\verify_uni_sign.py `
  --video 'D:\Zenodo\CE-CSL\CE-CSL\video\dev\B\dev-00069.mp4' `
  --pose-orientation mirror --device cuda `
  --output artifacts\unisign-direction-check.json
```

测试命令：

```powershell
.\.conda-env\python.exe -m pytest -q tests/test_uni_sign_preprocess.py tests/test_uni_sign_adapter.py tests/test_uni_sign_assets.py tests/test_unisign_evaluation.py
.\.venv-gpu\Scripts\python.exe -m pytest -q --tb=short
```

## 8. 交付文件

| 文件 | 用途 |
|---|---|
| [original-vs-repaired.csv](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/original-vs-repaired.csv) | 500 条参考、原预测、修复预测、方向与编辑相似度差；人工语义判定留空 |
| [repair-comparison-summary.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/repair-comparison-summary.json) | 配对比较、指标、区间、复现计数 |
| [direction-map.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/dev-calibration-final/direction-map.json) | dev 锁定方向与模型／姿态协议 |
| [dev summary.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/dev-calibration-final/summary.json) | 24 条独立 dev 验证成绩 |
| [test summary.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/test-locked-directions/summary.json) | 500 条完整回归状态与分签者成绩 |
| [run-protocol.json](E:/githubproject/SignLanguage-Bridge/artifacts/unisign-cecsl-eval/repair-20261024/test-locked-directions/run-protocol.json) | 运行参数、输入身份与方向表哈希 |
| [集成说明](E:/githubproject/SignLanguage-Bridge/docs/uni-sign-integration.md) | 单视频与评测入口使用说明 |

