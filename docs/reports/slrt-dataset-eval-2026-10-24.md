# SLRT 在本地数据集上的识别率与延迟实测

日期：2026-10-24。对象：[SLRT-CSL 在线 Gloss 模块](../slrt-integration.md)（EMNLP 24 CSL-Daily 双流 S3D，`rtmpose-experimental` 前端）。本报告只回答两个问题：**识别率**（Gloss 词级 WER）和**延迟**（吞吐、窗口耗时、首词延迟、实时回放交付）。结论先行：

> **输入审计更正（2026-10-24）：** 原镜像探针的 36 条逐行记录均为 `ValueError: mirror_keypoints_133 requires T x 133 x 3 COCO-WholeBody points`，每条仅收到 1 帧、完成 0 窗。此前把它们当成空识别并归因于模型方向，结论无效，现已撤回。正常方向 test 500 条无该异常，其历史 WER 保留。修复和新的输入审计见[输入兼容性审计](model-input-compatibility-audit-2026-10-24.md)。

1. **本机没有 CSL-Daily 连续视频**，只有上游捆绑的文本标注和 3 个官方姿态快照 pkl，因此 SLRT 的同域 WER 本轮无法产出。实测全部是 **CE-CSL 跨域**（test 500 条全量 + dev 36 条方向探针）。
2. **识别率：CE-CSL test 500 条 corpus WER 0.8918**（替换 706 / 删除 1980 / 插入 25，参考共 3,040 词），113/500 条输出为空。这是"权重训练域之外零样本 + 实验姿态前端"下的结果，不代表该模型在 CSL-Daily 域内的能力。
3. **方向结论撤回**：镜像 36 条均为输入错误，不能与正常方向对比。默认 preserve 暂时保留；没有有效证据支持“SLRT 与 Uni-Sign 的方向结论相反”，也不能直接套用其他模型的方向表。
4. **延迟：批处理吞吐中位 10.5 FPS**，窗口 p50 30.6 ms + 姿态提取中位 38.9 ms/帧；首 final 的协议固有源延迟中位 2.47 s。**实时回放未通过**：按原帧率回放 12 条，11 条交付 0 个 final Gloss，且每条都有队列丢帧——与此前摄像头 12 秒试跑结论一致。

## 1. 数据可用性盘点

| 数据 | 本机状态 | 本轮用途 |
|---|---|---|
| CE-CSL（`D:\Zenodo\CE-CSL`） | 完整：train/dev/test 视频 + UTF-8 Gloss 标注（dev 515、test 500，按译者 A–L 分目录） | 全部实测 |
| CSL-Daily 连续视频 | **不存在**。`models/external/uni_sign_source/data/CSL_Daily` 仅有 `labels.*` 文本；3 个 `S00xxxx.pkl` 是官方姿态快照（曾用于方形归一化验证），不是视频 | 无法评测；同域 WER 待视频到位 |
| 孤立词 CSL（`C:\Users\ADMIN\Downloads\CSL_basic_dataset.zip` 235 词、`CSL_common_dataset.zip` 863 词，文件名为 GBK 编码需修复解码） | 与 SLRT 2001 词词表重叠仅 33.2% / 12.6%（命中样本：不、为什么、书、什么、今天…；未命中：三明治、丹麦、7点40分…） | 未评测；孤立词任务与连续 CSLR 协议不同，仅可作低覆盖补充 |

CE-CSL test 参考 Gloss 的 **38.0% 词不在 SLRT 词表内**（`。`、数字词、`一些1` 这类变体词）。WER 不做参考侧清洗，OOV 单独报告；模型结构上无法产出这些词。

## 2. 识别率

协议：`rtmpose-experimental` 前端、窗口 16、stride=1、投票 7、规范方向；WER 与 SLRT CLI 同一 token Levenshtein 实现，无文本归一化；逐条记录见 `artifacts/slrt-dataset-eval-20261024/`。

| 范围 | 条数 | corpus WER | 空 输出 | 参考 OOV 率 | 说明 |
|---|---:|---:|---:|---:|---|
| CE-CSL test（全量） | 500 | **0.8918** | 113 | 38.0% | S 706 / D 1980 / I 25 |
| CE-CSL dev（方向探针基准） | 36 | 0.8986 | 9 | 37.2% | 排除 OOV 删除后 0.9154，**无改善** |

test 逐译者 WER（n 为该译者条数）：D 0.719（47）、F 0.769（58）、A 0.859（55）、C 0.878（42）、K 0.873（46）、J 0.871（38）、I 0.881（9）、H 0.975（9）、G 0.985（56）、E 0.985（54）、L 0.988（29）、B 0.992（57）。译者间差异大但没有一位接近可用；H/I 样本过少（各 9 条）不单独解读。

三个关键读法：

- **错误构成以删除为主**（1,980 删除 vs 706 替换，插入仅 25）：模型倾向输出空/弃答，而不是编造 Gloss。113 条空输出是同一行为的极端情况；帧质量门控 0 次拒绝（有人），空输出来自模型自身投票。
- **OOV 不能用简单扣除作因果排除**：旧汇总将每条 `min(OOV数, 删除数)` 视为 OOV 删除，没有逐词编辑路径归属。该量只是上界，扣除后的 WER 不能证明词表因素已经排除，更不能单独证明错误发生在哪一层。参考词 OOV 和可生成词的识别表现需要分别按明确协议统计。
- **与历史基线不可横比**：TFNet 在同一 test 500 条上的记录是 12.60% Gloss 完全匹配（`TFNet-CE-CSL-CSLDaily-32.46.pth`，**CE-CSL 域内训练**）；SLRT 是 CSL-Daily 权重零样本跨域。两者差距主要来自训练域，不是模型优劣的证明。真正的同域参照需要 CSL-Daily 视频。

## 3. 方向适配（历史经验应用）

按坑点记忆规则（先 dev 校准、锁定后验证、不用测试集选方向），在 dev 36 条（每译者 3 条）做 preserve/mirror 配对探针。镜像通过 [`mirror_keypoints_133`](../../slbridge/slrt/preprocess.py)（COCO-WholeBody133 全表交换 + 关于中心线反射，与几何管线翻转等价性有单测）实现，同步翻转 RGB。

结果审计：**镜像侧 36/36 均为输入维度异常，0 个窗口完成，不能形成有效方向实验。** 旧代码向只接受 `[T,133,3]` 的函数传了 `[133,3]`，同时未将外层 RGB 同步镜像；汇总又把异常记作空输出。当前已补批次维、配对 RGB/pose 方向变换，并将失败计入独立失败字段；有失败时不再给出整批 corpus WER。修复后的完整 36 条方向实验尚未重跑，默认 preserve 不是经该探针证明的最优方向。

修复后两条真实视频入口验证为0输入错误（236/183窗），仅证明链路可运行。新增评测绑定逐条协议/视频/参考/方向/源码，resume拒绝混合记录；缺样本或非final成功记录不能发布完整成绩。详细测试与前端/裁剪边界见[输入兼容性审计](model-input-compatibility-audit-2026-10-24.md)。

## 4. 延迟

| 指标 | 数值 | 口径 |
|---|---|---|
| 批处理吞吐 | 中位 **10.5 FPS**（p95 14.2、最慢 8.9） | 逐视频 `frames/wall`，含解码+姿态+推理 |
| 窗口推理 | p50 30.6 ms、p95 36.2 ms | 每 16 帧窗一次前向 |
| 姿态提取 | 中位 38.9 ms/帧 | RTMLib lightweight ONNX |
| 首 final 源延迟 | 中位 **2.47 s**、最大 7.97 s | 右上下文 8 帧 + 7 窗投票的协议固有等待，非墙钟 |
| 实时回放（12 条，按原 FPS 喂入） | **11 条交付 0 个 final**；唯一交付的一条在 9.37 s 墙钟给出 1 个词 | 每条队列丢帧 14–86（有界队列 32） |

读法：批处理约 70 ms/帧的端到端成本（姿态 39 + 摊销窗口 31）决定了吞吐上限约 14 FPS，低于 CE-CSL 的 30 FPS 源帧率；实时回放因此持续积压、丢帧、窗口无法凑齐右上下文——**在当前硬件与实验前端下不具备实时识别能力**。这不因"退出码 0、会话正常结束"而改变。改进方向仍是降低单帧成本（姿态前端提速、窗口批处理）或降采样协议（有正确性代价，须标注实验配置）。

## 5. 本轮代码改动与修复

- `slbridge/slrt/preprocess.py`：新增 `mirror_keypoints_133`（显式方向诊断用），含交换表自逆、坐标反射、与几何管线翻转等价的单测（`tests/test_slrt_protocol.py`，11 passed）。
- `slbridge/uni_sign.py`：修复 `_configure_windows_cuda` 重复前插 PATH 的缺陷——长会话多次构造提取器会使 PATH 超过 Windows 32767 字符上限（本轮 500 条评测在 51 条处实际触发）。回归测试 `test_windows_cuda_path_entries_are_not_duplicated`。
- 新增 `scripts/evaluate_slrt_datasets.py`（批量 WER+延迟评测，支持 `--resume`、方向表、逐条 JSONL 增量落盘）与 `scripts/probe_slrt_replay_latency.py`（实时回放延迟探针）。
- 全量回归：**784 passed / 10 skipped**（`.conda-env`，61 s）。历史日志中的 827 与当前收集数（791）的调用范围差异无法从旧日志复原，不作为对比基线；本轮验证以当前树为准。

## 6. 复现命令

```powershell
# 批量 WER + 延迟（新目录；旧历史结果无绑定协议，不允许直接 --resume）
.conda-env\python.exe scripts\evaluate_slrt_datasets.py --split test --output-dir artifacts\slrt-dataset-eval-new\test-preserve
# 方向探针（dev 校准专用）
.conda-env\python.exe scripts\evaluate_slrt_datasets.py --split dev --limit 36 --mirror-all --output-dir artifacts\slrt-dataset-eval-new\dev-mirror
# 实时回放延迟
.conda-env\python.exe scripts\probe_slrt_replay_latency.py --per-translator 1 --output artifacts\slrt-dataset-eval-20261024\replay-latency.json
```

## 7. 证据索引与边界

- 逐条记录：`artifacts/slrt-dataset-eval-20261024/test-preserve/detail-test.jsonl`（500 条）、`dev-preserve/`、`dev-mirror/`、`replay-latency.json`；汇总 `summary-*.json`。
- 未验证 / NOT_RUN：CSL-Daily 同域 WER（无视频）、官方 HRNet 前端对照（无资源）、孤立词集评测（词表覆盖不足，另立任务）、真人语义评审。
- 本报告数字不构成 SLRT 的"识别准确率"声明：跨域、实验前端、WER 是词级指标；与 TFNet/Uni-Sign 的历史数字分属不同指标与训练域。
