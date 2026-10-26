# Uni-Sign on CE-CSL：跨域句子级指标与预处理修正报告

- 日期：2026-10-23
- 模型：Uni-Sign `csl_daily_pose_only_slt.pth`
- 模型 revision：`182913ca7a8b7dd9837d7d6f10717353f4408f80`
- 测试集：本地 CE-CSL test split，500 条
- 运行状态：**500 successful，0 failed/empty**
- 硬件：RTX 5060 Ti 16 GB；PyTorch 2.9.1+cu128；CUDA bfloat16

## 结论先行

CE-CSL **有完整的中文参考句和 Gloss 标注**，不存在“没有类似 CSL-Daily 的中文”这一问题。500 条测试标签都来自 `Chinese Sentences` 和 `Gloss` 两列；缺的是与 CSL-Daily 相同的**签者、录像场景、动作实例和句子分布**，不是中文标签。

本轮同时确认并修复了一个真实的输入协议错误：旧实现按 `x/W, y/H` 分别归一化 RTMPose 像素坐标，而官方 CSL-Daily pose 快照使用方形 `w_h`。CE-CSL 大量视频为 16:9，旧实现会把横向几何压缩为原来的 9/16。修正为短边等比例归一化后，500 条全量结果为：

| 评估对象 | 样本 | 严格整句匹配 | BLEU-zh | chrF | 字符编辑相似度 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| CSL-Daily × Uni-Sign（上游预测快照） | 1176 | 1.53% | 25.70 | 22.48 | 42.27% | 0.5360 |
| CE-CSL × Uni-Sign（旧 W/H，被修正基线） | 500 | 0.00% | 4.72 | 6.33 | 17.88% | 0.2449 |
| CE-CSL × Uni-Sign（修正短边，本轮正式） | 500 | 0.00% | **7.99** | **9.38** | **22.51%** | **0.3004** |
| CE-CSL × TFNet（原生中文词序列） | 500 | 12.60% Gloss 完全匹配 | — | — | — | — |

修正不是调参后“看起来好一点”，而是配对比较通过了区间检验：

- BLEU-zh：**+3.27**，95% CI **[+2.32, +4.26]**
- 字符编辑相似度：**+4.64 个百分点**，95% CI **[+3.31, +6.01]**
- ROUGE-L：**+0.0555**，95% CI **[+0.0410, +0.0703]**
- 逐句字符编辑相似度：237 条改善、105 条不变、158 条变差

但修正后仍是明显跨域结果。**严格整句 0/500 不等于“识别准确率 0”**；即使 CSL-Daily 上游同域预测快照，严格整句完全相同也只有 1.53%。BLEU/chrF/ROUGE 是句子相似度，不是分类 accuracy；TFNet 的 WER 又是词级 Gloss 指标，不能直接与 Uni-Sign 的句子 BLEU 横比。

## 为什么“都是中文”仍会这么低

CE-CSL 与 CSL-Daily 的文本层并不完全断裂：

- CE-CSL 参考句平均 **10.77** 字，CSL-Daily 测试参考句平均 **16.40** 字。
- 500 条 CE-CSL 参考句没有任何一条与 CSL-Daily 测试参考句完全相同。
- CE-CSL 字符类型有 **83.73%** 出现在 CSL-Daily 文本中，但 2-gram / 3-gram / 4-gram 类型覆盖率分别只有 **34.67% / 10.84% / 2.43%**。
- CE-CSL Gloss 的**出现次数**有 **80.55%** 能在 CSL-Daily 参考文本中做字面匹配，但字面相同不等于手语动作、词序、方言或语法相同。

所以用户的直觉有一半是对的：两边共享大量普通中文词汇；但模型看到的不是“中文文本”，而是具体签者的 133 点动作序列。CE-CSL 的签者、相机距离、室内/室外场景和动作实现与 CSL-Daily 不同。修正几何后，部分签者明显恢复，另一些仍失败，支持这种解释。

## 归一化缺陷与修复

官方 CSL-Daily pose 抽样中：

- `S004805_P0003_T00.pkl`：`w_h = 1080 × 1080`
- `S005000_P0000_T00.pkl`：`w_h = 1280 × 1280`
- `S005021_P0003_T00.pkl`：`w_h = 1080 × 1080`

官方 `datasets.py` 的 `crop_scale` 以像素尺度计算 body 外接框；方形画幅下 `x/W, y/H` 不产生几何畸变。CE-CSL 的 `test-00001` 是 1920×1080，`test-00056` 是 1280×720，旧实现的横向坐标被额外压缩。

当前默认协议：

- `pose_normalization = isotropic_short_side`
- 横纵坐标共同除以 `min(width, height)`，保留动作几何比例
- `wh` 模式仅作为旧基线复现，不再作为默认
- 评估结果和姿态缓存均记录 `pose_normalization`，不同协议的缓存不会被误复用

回归测试先失败后通过，覆盖 wide-screen 几何保持、旧 `wh` 兼容、非法尺寸/模式拒绝，以及运行时配置记录。

## 逐译者结果

| 译者 | n | 旧 BLEU | 新 BLEU | 新 chrF | 新编辑相似度 | 新整句匹配 | ΔBLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 55 | 6.35 | 14.46 | 14.36 | 33.80% | 0 | +8.11 |
| B | 57 | 0.44 | 0.53 | 2.32 | 5.77% | 0 | +0.09 |
| C | 42 | 7.38 | 14.91 | 16.30 | 31.95% | 0 | +7.53 |
| D | 47 | 11.49 | 22.74 | 21.65 | 43.83% | 0 | +11.25 |
| E | 54 | 0.50 | 0.44 | 2.81 | 10.23% | 0 | -0.06 |
| F | 58 | 7.45 | 11.21 | 11.93 | 28.93% | 0 | +3.76 |
| G | 56 | 0.65 | 0.74 | 3.43 | 10.35% | 0 | +0.09 |
| H | 9 | 2.55 | 2.26 | 3.71 | 10.15% | 0 | -0.28 |
| I | 9 | 0.64 | 1.02 | 1.82 | 5.08% | 0 | +0.38 |
| J | 38 | 7.42 | 7.04 | 9.47 | 25.48% | 0 | -0.38 |
| K | 46 | 6.18 | 11.20 | 12.37 | 31.34% | 0 | +5.01 |
| L | 29 | 0.63 | 1.10 | 3.26 | 10.68% | 0 | +0.47 |

结果呈明显双峰：D、C、A、K、F 的 BLEU-zh 为 **11.20–22.74**，而 B、E、G、L 低于 **1.11**，H/I 只有 9 条，不能外推为稳定签者结论。这个分布比单一总体均值更能说明部分场景并非“完全不会输出中文”，而是模型在部分签者/拍摄域上无法稳定对齐动作语义。该现象需要在 CE-CSL 同域微调或同域 CSL-Daily 视频上继续验证，不能直接归因于“译者水平”。

## 代表性逐句对比

| 样本 | 参考中文 | 旧 W/H 预测 | 新短边预测 | 旧字符相似度 | 新字符相似度 |
|---|---|---|---:|---:|
| test-00003 | 下雨后空气会变好。 | 放假结束后,天赋格外引人注目。 | 下雨结束了,天气很好。 | 13.3% | 36.4% |
| test-00004 | 不可以不做吗？ | 不要做坏事。 | 你不要做兼职吗? | 14.3% | 37.5% |
| test-00008 | 人生最重要的是学习。 | 人生不应该灰心,而应该值得你忘记。 | 人生中最重要的就是学习。 | 17.6% | 83.3% |
| test-00010 | 今天你吃了什么蔬菜？ | 宝宝,你吃什么菜? | 今天你吃什么菜? | 50.0% | 80.0% |
| test-00011 | 今天我吃了香蕉。 | 宝宝喜欢吃香蕉。 | 今天要吃香蕉。 | 37.5% | 75.0% |
| test-00013 | 今天晚上我可以喝酒，明天不用读书。 | 妈妈,晚饭不要喝牛奶。 | 今天晚上喝酒了,明天不看书。 | 11.8% | 64.7% |
| test-00016 | 今晚不适合去海边。 | 我去爬山,不要和你打乒乓球。 | 今天海边不适合捕鱼。 | 14.3% | 40.0% |
| test-00019 | 他们的防守太厉害了。 | 怎么没关系? | 他把手表弄丢了。 | 0.0% | 30.0% |
| test-00022 | 他喜欢学英语。 | 她笑的时候,露出一口洁白的牙齿。 | 她笑的时候,露出一口洁白的牙齿。 | 6.2% | 6.2% |
| test-00024 | 他的弟弟把车开走了。 | 他有点害怕跑步。 | 他弟弟要开车去。 | 20.0% | 50.0% |

完整 500 条逐句表见 `artifacts/unisign-cecsl-eval/comparison-20261023/cecsl-unisign-comparison.csv`；可筛选工作簿见 `outputs/01a16889-c4d3-72a1-8eb7-5b7ebb887710/CE-CSL_UniSign_结果对比.xlsx`。

## 同域参照与原生 CE-CSL 参照

### CSL-Daily 上游预测快照

固定源码提交中包含 `out/eval_csl_daily_pose_only/test_tmp_pres.txt` 与 `test_tmp_refs.txt`，各 1176 条。对其按同一套文本规范化复算：

- BLEU-zh：**25.70**
- chrF：**22.48**
- 字符编辑相似度：**42.27%**
- ROUGE-L：**0.5360**
- 严格整句匹配：**18/1176（1.53%）**

边界：这是上游源码包随附的预测/参考快照，不是本轮重新解码 CSL-Daily 视频；不能用它替代同域复现实验，但它清楚表明该 checkpoints 在熟悉域能生成连贯中文，而不是模型不会中文。

### CE-CSL 原生 TFNet

项目已有 CE-CSL TFNet ONNX Gloss 识别基线：

- 500 条语料 WER：**32.7961%**
- 编辑距离：997 / 3040 个参考词
- 完全相同的 Gloss 序列：**63/500（12.60%）**

这是词序列识别，不是 Uni-Sign 句子翻译，不能横比 BLEU；它证明 CE-CSL 标签和视频本身是可学习的，低分不是数据集缺少中文或完全不可识别。

## 运行成本

- 500 条总处理墙钟：约 **2460 秒（41.0 分钟）**
- 单条处理 P50 / P95：**4.79 / 7.10 秒**
- 模型部分 P50 / P95：**295 / 414 ms**
- 平均源帧数 / 平均采样帧数：**198.8 / 195.2**
- 姿态缓存：**500 个 NPZ，约 53.6 MB**

## 解释边界

- 这是 **CE-CSL × CSL-Daily 权重**的跨域评估，不是 CSL-Daily 同域准确率，也不是通用中国手语准确率。
- 没有人工逐句语义复核；`Manual_Semantic_Judgment` 仍为空，人工语义验收为 **NOT_RUN**。
- BLEU/ROUGE 不能替代聋人手语者的可懂度、自然度或业务场景验收。
- 没有针对 CE-CSL 微调，也没有融合 CE-CSL 词表或语言模型。
- 逐译者结果不能外推为“模型支持/不支持某位译者”。
- 上游 CSL-Daily 快照是证据参照，不是本轮重新推理。

## 复现

```powershell
$env:PYTHONPATH = 'E:\githubproject\SignLanguage-Bridge'

# 修正短边默认协议
.\.conda-env\python.exe scripts\evaluate_unisign_cecsl.py `
  --output-dir artifacts\unisign-cecsl-eval\full-isotropic-20261023 `
  --device cuda --save-every 25

# 显式复现旧 W/H 基线（需要独立输出目录）
.\.conda-env\python.exe scripts\evaluate_unisign_cecsl.py `
  --output-dir artifacts\unisign-cecsl-eval\full-wh-replay `
  --pose-normalization wh --device cuda --save-every 25
```

## 证据文件

- `artifacts/unisign-cecsl-eval/full-isotropic-20261023/summary.json`
- `artifacts/unisign-cecsl-eval/full-isotropic-20261023/predictions.jsonl`
- `artifacts/unisign-cecsl-eval/full-isotropic-20261023/predictions.csv`
- `artifacts/unisign-cecsl-eval/full-isotropic-20261023/pose-cache/`
- `artifacts/unisign-cecsl-eval/full-20261023/`（旧 W/H 被修正基线）
- `artifacts/unisign-cecsl-eval/comparison-20261023/bootstrap_compare.json`
- `artifacts/unisign-cecsl-eval/comparison-20261023/cecsl-unisign-comparison.csv`
- `artifacts/unisign-cecsl-eval/comparison-20261023/summary.json`
- `artifacts/unisign-cecsl-eval/csldaily-vs-cecsl-protocol-audit-20261023.json`

## 最终判定

**CE-CSL 中文标签存在：是，500/500 行都有参考句和 Gloss。**
**500/500 修正短边全量运行：通过。**
**归一化修复有效：通过（配对 Bootstrap 不跨 0）。**
**严格整句正确率：0/500（0.00%）。**
**跨域句子相似度：BLEU-zh 7.99、chrF 9.38、字符编辑相似度 22.51%。**
**CSL-Daily 本轮本地重跑：NOT_RUN。**
**人工语义验收 / 产品级准确率：NOT_RUN / 未完成。**
