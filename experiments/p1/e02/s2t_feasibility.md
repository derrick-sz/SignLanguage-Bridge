# CSL-Daily SingleStream-S2T（SLT）本地可行性评估（只读）

- 评估日期：**2026-10-27**
- 对象：`F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\csl-daily_s2t\ckpts\best.ckpt`
  （1,503,041,809 B；SHA256 `b6bdf7c2d8babe4d8ced724783f966f101d27468fd0494be864e73e03e8bd66c`；
  来自作者 Google Drive `ckpts.zip`，见 `g2t_asset_tracker.md`）
- 代码：`Paper-algorithm/SLRT-main/TwoStreamNetwork`（vendored 官方实现，未改动）
- 方法约束：**只读**评估 —— `torch.load(weights_only=True, mmap=True, map_location='cpu')` 读 state_dict 键集/形状、
  gzip-pickle 只读检查特征/标注、阅读代码与配置；**未下载、未训练、未改代码、未跑任何模型前向**。
- 结论：**BLOCKED（但全部为本地可解项；没有外部新权重缺口）**。详见 §6。**注意：这是 S2T（video→sentence）模型，不是 G2T，不能当 B 分支的 Gloss→Text 使用。**

## 1. 这是什么模型（由 ckpt 键集/形状判定）

ckpt 顶层只有 `{"model_state": ...}`（539 个键，**无 epoch/hyper_parameters**），三个模块：

| 模块 | 键数 | 参数量 | 关键形状 | 判定 |
|---|---:|---:|---|---|
| `recognition_network` | 18 | 10.3 M | `visual_head.fc1.weight (512, 832)`；`visual_head.gloss_output_layer.weight (2004, 512)`；`pe.pe (1, 5000, 512)` | 视觉 head：吃 **832 维预提取 S3D 特征**，输出 **2004 类 gloss logits**；**无 S3D backbone 权重**（feature 模式下不需要 S3D） |
| `translation_network` | 520 | 382.7 M | mBART encoder/decoder 各 **12 层、d_model=1024**；`shared.weight (6308, 1024)`；`lm_head.weight (6308, 1024)`；`gloss_embedding.weight (2004, 1024)`；`final_logits_bias (1, 6308)` | 经过剪枝的 mBART_zh：**输出文本词表 6308**、gloss 词表 2004（与本机 `data/csl-daily/gloss2ids.pkl` 的 2004 项一致） |
| `vl_mapper` | 1 | 2.05 M | `mapping.weight (1024, 2004)` | 把 recognition 的 gloss_probabilities（2004 维）线性映射到 1024 维文本 embedding（config `VLMapper.type=embedding` 与此一致） |

合计 395.1 M 参数（538 个 fp32 张量 + 1 个 int64）。对应官方配置
`experiments/configs/SingleStream/csl-daily_s2t.yaml`（task=`S2T`；论文/文档 BLEU-4 **24.09**，R/B1/B2/B3/B4 = 53.35/53.53/40.68/31.04/24.09）。

**输入规格**：`data.test_head_rgb_input` 指向 gzip-pickle 特征文件；`dataset/Dataset.py` 将其载入为
`head_rgb_input`，`recognition.py` 的 `input_type=='feature'` 分支直接把 `head_rgb_input` 喂 `visual_head`
（`load_batch_feature` 会 pad 到 batch 内最长）。本机特征实测：每条视频 `(T, 832)`，T = 7–85。
**输出**：一句中文文本（lm_head 6308 → TextTokenizer 解码），不是 gloss。

## 2. 本地资产核对（全部只读实测）

| 资产 | 本地路径/状态 | 结论 |
|---|---|---|
| S2T ckpt（本对象） | `F:\model_weigth\...\SingleStream\csl-daily_s2t\ckpts\best.ckpt` | **有** |
| 预提取 S3D 特征（inference 输入） | `F:\model_weigth\...\SingleStream\csl-daily_s2g\extract_features\head_rgb_input\{train,dev,test}.pkl`（519.7 MB / 31.3 MB / 35.6 MB） | **有**；`test` 1176 条（`(T,832)`）、`dev` 1077；**本机 1086 条评测样本全部覆盖（1086/1086，缺 0）** |
| CSL-Daily 句级标注 | `...\data\csl-daily\csl-daily.{train,dev,test}`：18,401 / 1,077 / 1,176 条；字段 name/signer/gloss/text/sign/num_frames | **有**（test 1176 与特征一一对应） |
| gloss 词表 | `data/csl-daily/gloss2ids.pkl`：**2,004 项**（与 ckpt `gloss_output_layer` 2004、`gloss_embedding` 2004 一致） | **有** |
| base mBART（初始化/基座） | `models/external/SLRT_Online/pretrained_models/mBart`（config/tokenizer/pytorch_model.bin 2.44 GB） | **有** |
| 作者剪枝 mBART_zh 文本资产（vocab 6308 的 config/tokenizer/old2new_vocab/map_ids） | 本地只有 `mBART_zh_fullvocab`（**全词表 fallback，vocab 250,027，identity map**；README_FALLBACK 明确写不是作者剪枝版） | **缺**（见 §4-①） |
| `s3ds_glosscls_ckpt`（S3D 预训练） | 本地只有 `s3ds_actioncls_ckpt`（34 MB，且是 action-classification，不是 glosscls） | **feature 模式下不需要**（ckpt 内无 S3D 权重，`input_type='feature'` 分支不构建 S3D） |
| 官方 `sentence_frames-512x512.zip` | 缺 | **feature 模式不需要**（已用作者预提取特征） |
| 旧依赖环境 | `.conda-env` = Python 3.10 / torch 2.9.1 / transformers 4.40 / 无 tensorflow、wandb；`F:\DSH-envs\slrt-legacy` = Python 3.8 / torch 1.13.1+cpu / 无 transformers、tensorflow、timm、wandb | **缺可用环境**（见 §4-②） |

## 3. 本地推理入口与所需文件（代码阅读结论）

- 入口：`TwoStreamNetwork/prediction.py --config <cfg> --ckpt_name best.ckpt`
  → `build_model`（task=S2T：RecognitionNetwork(input_type='feature') + TranslationNetwork + VLMapper）
  → 把 ckpt 中 `model_state`（含三个模块）通过 `neq_load_customized` 加载 → `build_dataloader` 在 dev/test 上跑
  `model.generate_txt` → 内部用 `utils.metrics.bleu/rouge/wer_list` 打印 BLEU/ROUGE/WER。
- 需要的配置项（`experiments/configs/SingleStream/csl-daily_s2t.yaml`）：
  `data.dev_head_rgb_input` / `data.test_head_rgb_input`；`data.{dev,test}` 标注；`GlossTokenizer.gloss2id_file`；
  `TranslationNetwork.TextTokenizer.pretrained_model_name_or_path` + `pruneids_file`；
  `TranslationNetwork.load_ckpt`（**该配置写的是 `experiments/outputs/SingleStream/csl-daily_g2t/ckpts/step_1000.ckpt`**，用于 S2T 训练时初始化翻译网络）；
  `training.model_dir`（`prediction.py` 从 `model_dir/ckpts/best.ckpt` 读权重）。
- 依赖（env yml + 源码 import）：python 3.6/3.8、torch 1.9、torchvision 0.10、numpy 1.18、**transformers 4.11.3**
  （`MBartForConditionalGeneration/MBartTokenizer/MBartConfig/PreTrainedModel/ModelOutput`）、
  **tensorflow 2.2**（`modelling/recognition.py` 顶部 `import tensorflow as tf`，用于 ctc beam decoder；S2T 评测默认 `do_recognition=True` 会走到它）、
  **wandb**（`prediction.py`/`utils/misc.py` import）、**protobuf 3.19**（`google.protobuf.reflection.ParseMessage`，新版 protobuf 已无此符号）、
  tqdm/Pillow/PyYAML 等。`av`/`lintel`/`timm`/`torchtext` **源码未 import**（VideoLoader 只依赖 PIL+ZipReader），不需要。
- ⚠️ `prediction.py` 的 `__main__` 里 **硬编码 `cfg['device']=torch.device('cuda')` 与 `map_location='cuda'`**：
  纯 CPU 运行需要 1 行设备覆盖（或使用 GPU 机器）。本轮不改代码，故记为适配项。

## 4. 缺什么（逐项）

| # | 缺口 | 现状 | 是否外部新权重 | 解除方式（本轮未执行） |
|---|---|---|---|---|
| ① | **作者剪枝 mBART_zh 文本侧资产**（vocab=6308 的 `config.json`/`tokenizer.json`/`old2new_vocab.pkl`/`map_ids.pkl`） | 本地 `mBART_zh_fullvocab` 是 250,027 全词表 identity fallback，与 ckpt 的 `lm_head/shared (6308,1024)` **形状不匹配**，无法直接构建/加载 | **否**（作者 Drive 已有可用副本） | 两个本地路径：**(a)** 作者 Drive 的 `mBart_zh_t2g.zip` 在 R12 文件级检查中记录为 *config vocab 6,308 / old2new_vocab.pkl 6,308*（T2G 的文本侧与 S2T/G2T 文本侧同一剪枝词表；gloss 侧 2,009 与本地 2,004 不同，但文本侧正是所需）——需下载该 zip 并只取文本侧文件验证；**(b)** 完全离线重建：ckpt 的 `shared.weight (6308,1024)` 是 base mBART 词表行的子集拷贝，可与 base `model.shared.weight (250054,1024)` 做**精确行匹配**反推 old→new 映射（需要一个新脚本 + 校验，本轮未写）。任一方式都**不需要作者再发布新权重**。 |
| ② | **可用运行环境**（python3.8 + torch1.9/1.13 + transformers4.11.3 + tensorflow2.x + wandb + protobuf3.19） | 无：`.conda-env` py3.10/torch2.9/transformers4.40/无 tf；`slrt-legacy` py3.8/torch1.13/无 transformers/tf/wandb | 否（依赖可用 pip/conda 安装） | 新建独立 CPU 环境（建议 py3.8 + CPU wheels）；为遵守"不安装/不下载"约束，本轮不建。 |
| ③ | **配置/路径适配**（不改上游代码） | 特征目录名 `extract_features` vs 配置 `extract_feature`；`pretrained_model_name_or_path` 需指向剪枝 mBART_zh；`load_ckpt` 指向不存在的 G2T `step_1000.ckpt`；`model_dir/ckpts/best.ckpt` 需放置本 ckpt；`prediction.py` 硬编码 cuda | 否 | 复制 config 到独立实验目录并改 5 个路径字段 + 用环境变量/一行 device override（或 GPU 机器）。属于"配置适配"，不改变模型/预处理逻辑。 |

**不需要的资产**（排除以缩小工作范围）：S3D `glosscls` ckpt、官方 sentence frames zip、CSL-Daily 视频帧、`av`/`lintel`。
（本机 1086 条评测样本均有作者预提取特征，因此**无需视频解码/关键点**即可做 S2T 端到端。）

## 5. 若解除后能得到什么（价值与边界）

- 可复现 **官方 CSL-Daily SingleStream-SLT** 在本机 1086 子集上的端到端 BLEU（论文 24.09，索引集 1176；本机 1086 是子集，
  预期与 24.09 同量级，**不能宣称"复现论文数字"除非用 1176 全测试集**）。
- 可额外导出 `recognition_network` 的 **gloss logits/概率**，与 SLRT 的 gloss 输出做上游对比（WER/序列差异），
  为 G2 的"B 路径上游质量"提供证据。
- **边界**：它是 **S2T（video→sentence）**，输入是视频特征；**不是 G2T（gloss→text）**，不能直接用于 B 分支
  （SLRT Gloss→G2T→sentence）。若要 G2T，仍需 §4-① 的文本资产 + 单独的 G2T ckpt（仍外部缺失，见 task-13）。
- 估算：395 M 参数、CPU 上 beam=5 生成 1086 句预计**数十分钟量级**（未实测；本轮不跑）；GPU 现在被 E01/E04 占用，需排期。

## 6. 结论（二选一）

> **BLOCKED（逐项缺 3 项：① 剪枝 mBART_zh 文本资产；② 旧依赖环境；③ 配置/路径适配），
> 但全部为本地可解，无外部新权重缺口。** 解除后按 §7 最小验证。

- 不是 "RUNNABLE_WITH_LOCAL_ASSETS"：虽然模型/特征/标注/gloss 词表/基座都在本地，但缺 **vocab=6308 的文本侧 tokenizer 资产**，
  且当前没有任何一个本地 Python 环境能 import 该代码（缺 transformers 4.11/tensorflow/wandb）。
- 也不是外部 BLOCKED：① 可来自作者 Drive 已有的 `mBart_zh_t2g.zip`（或离线行匹配重建），② 是依赖安装，③ 是路径/设备适配。

## 7. 最小验证步骤（解除后执行，约 1–2 小时，不含长跑）

1. 建独立 CPU 环境（py3.8）：`torch==1.13.1+cpu`（或 1.9）+ `transformers==4.11.3` + `tensorflow-cpu`（2.x，确认 `tf.nn.ctc_beam_search_decoder` 可用或用 `tf.compat.v1`）+ `wandb` + `protobuf==3.19` + `torchvision` + `numpy==1.23`；不污染 `.conda-env`。
2. 准备 `pretrained_models/mBART_zh/`（vocab 6308）：从作者 Drive `mBart_zh_t2g.zip` 取文本侧 config/tokenizer/old2new_vocab 并**核对** `len(old2new)==6308`、`config.vocab_size==6308`；或离线用 base mBART 行匹配重建并校验。
3. 复制 `SingleStream/csl-daily_s2t.yaml` 到独立实验目录：把 `*_head_rgb_input` 指向 `F:\model_weigth\...\extract_features\head_rgb_input\*.pkl`；`pretrained_model_name_or_path` 指向步骤 2 的 mBART_zh；去掉/改写 `load_ckpt`；`model_dir` 指向新目录并放入 `ckpts/best.ckpt`（copy 或 symlink 本 ckpt）；device 改 CPU（或上 GPU）。
4. **先 3–5 条**（从 test 标注里按名字抽 3–5 条，全部输入均 **GT-free**）跑 `prediction.py --ckpt_name best.ckpt`，检查：
   输出是通顺中文、非重复/空、`recognition` 的 gloss logits 可导出；同时记录耗时。
5. 评测侧（EVAL ONLY）再用本机 1086 子集跑一次离线评测（GT 只 join 评测），与 E02 的 A/A2 指标并列；
   若 BLEU 与论文 24.09 相差 >2（在 1176→1086 子集漂移预期内），先查分词/词表/beam 配置，不得直接宣称"模型坏了/复现成功"。
6. 明确写：这是 **S2T 基线**，**不是 G2T**；G2 的 B 分支仍按 task-13 维持外部 BLOCKED。

## 8. 边界与免责

1. 本文件是**只读可行性评估**，不是运行结果；未执行任何 forward，未加载到 GPU/CPU 推理。
2. 许可：SLRT 仓库 `license=null`；作者 Drive 包无显式许可 —— 任何使用/再分发前需作者确认（同 `g2t_asset_tracker.md` §9）。
3. 特征/标注来自作者包（`F:\model_weigth`），SHA/大小见 `data-consistency.json` 与本文 §2；本机 1086 视频无需参与 S2T 推理（feature 模式）。
4. 论文 BLEU 24.09 的来源是官方 1176 test 集；本机 1086 是其子集，数字不可直接等同。
5. 结论只在 2026-10-27 的本地资产/代码快照下成立；若后续作者发布 G2T/mBART_zh 或环境就绪，需重新评估。
