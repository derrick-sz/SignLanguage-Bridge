# SLRT Online 本地权重与配置复现差距核查

日期：**2026-10-25**。对象：`https://github.com/FangyunWei/SLRT` 的 `Online/`（EMNLP 2024 *Towards Online Continuous Sign Language Recognition and Translation*，Zuo et al.）。本地对象：`slbridge/slrt/`、`models/external/EMNLP24_Online/`、`configs/slrt-csl.example.json`、`models/external/slrt_*_bundle/manifest.json`，以及上游源码镜像 `Paper-algorithm/SLRT-main/`。

本轮是**只读核查**：未训练、未下载大文件、未改动模型或数据。结论区分“本地已有证据”“官方要求”“仍缺什么”，不把权重存在当成已复现，也不把历史跨域成绩当成同域结果。

> **2026-10-25 后续状态：** 本报告列出的非数据缺口已按顺序补齐（官方目录、HRNet legacy 栈与 NPZ 提取、S3D K400、base mBART/gloss 资产、SLT pred 转换、检查脚本）；作者 Google Drive 补充包又补齐了 CSL-Daily 关键点、句级标注和 TwoStream-SLR S2G。仍缺 `sentence_frames-512x512.zip`、CSL-Daily G2T 权重/作者 `mBART_zh` 和官方旧依赖栈；详见[资产补齐记录](slrt-assets-bootstrap-2026-10-25.md)。

## 1. 结论先行

1. **CSL-Daily 在线 CSLR 的主权重、词表与本地推理协议基本齐备。** `cslr_best.ckpt` 严格加载 938 个参数键、三头 2001 类，输入分别 3 通道 RGB / 63 通道热力图，与官方 `slide_csl-daily.yaml` 的 `use_block: 5`、`triplehead_cat_bilateral`、`raw_size 512 / map 170 / sigma 8 / input 112` 一致；本地预处理、16 帧窗、stride=1、左 7 右 8 边缘复制、7 窗多数投票与官方 `prediction_slide.py` 的默认 CSL 路径等价（见第 4 节）。此前已记录同窗口数值对照 `max_abs=0`。
2. **真正卡住同域复现的是数据与评测入口，不是 CSLR 权重。** 本机没有 CSL-Daily 连续视频、官方 HRNet 关键点 `.pkl`、句级 `csl.train/dev/test`；现有 `EMNLP24_Online/CSL-Daily/meta` 只是**词级 ISLR** 标注（dev 14,059 / test 15,826 个词片段，train 200,530 个），不能直接喂给 `slide_csl-daily.yaml`。因此无法产出与论文同协议的 CSL-Daily 在线 WER。
3. **官方姿态前端只差“配置 + 运行栈”。** 官方 `gen_pose.py` 需要的 HRNet-W48 DARK 与 Faster-RCNN person 权重**本机实际已有**（`models/external/uni_sign/` 下，文件名与官方发布一致），但 mmdet/mmpose 的 Python 配置、`_base_` 继承树、legacy mmpose 0.x + mmdet 2.x + mmcv-full 运行栈都缺失；`gen_pose.py` 自身默认 `configs/det.yaml` 在上游仓库也不存在。此前文档“压缩包不含这些资源”仍成立，但不能再说“本机没有这两个权重”。
4. **Online/SLT 与 CTC_fusion 在本机没有可复现权重。** 本地这份发布包只有 Phoenix-2014T 的 `slt_best.ckpt`（520 个 `translation_network.*` 键）；CSL-Daily 只有 `cslr_best.ckpt`，没有 G2T 权重，官方 SLT README 也没有给出 CSL-Daily ckpt 链接（是否另有发布需向作者确认）。CSL-Daily 的 wait-2 mBART 还需要 `mBART_zh/` 的 `gloss_embeddings.bin`、`gloss2ids.pkl`、`old2new_vocab.pkl`，本机与上游仓库都没有。SLT 的 dev/test 输入 `csl_pred.dev/.test` 也不由 `prediction_slide.py` 生成，需要自己转换。CTC_fusion 还缺 TwoStream-SLR S2G 权重、K400 S3D、在线特征 pkl 和 `csl_s2g_gloss2ids.pkl`。
5. **官方 README 没有在线目标数字，论文目标不能写成仓库 README 的承诺。** 本地论文卡记录：在线 CSLR 测试 WER 为 Phoenix-2014 22.1%、Phoenix-2014T 22.1%、CSL-Daily **29.3%**；在线 SLT BLEU-4 为 Phoenix-2014T 23.69、CSL-Daily **20.63**（PDF p.7 表 1–2，未在本机复现）。仓库文档只在 `TwoStreamNetwork/docs/TwoStream-SLR.md` 给出离线 TwoStream-SLR 的 18.8/19.3/25.3 WER，不能当在线基线。
6. **本地 SLRT 是 CSL 专用推理路径，不是通用 Online 复现器。** `SLRTConfig` 固定 `sign_language='csl'`，`load_vocabulary` 固定 2001 类；Phoenix 的 1116 类权重存在也无法通过当前配置加载。CTC_fusion 所需的 block5 特征、`save_fea`、`pred_src='fuse'`、`model_ex`、`split_setting` 等模型变体也没有实现。

## 2. 权重盘点：官方要求 vs 本地

官方在线 CSLR 的 checkpoint 期望放在 `{training.model_dir}/ckpts/best.ckpt`，CSL-Daily 的 `model_dir` 是 `results/csl-daily_ISLR`（`Online/CSLR/configs/slide_csl-daily.yaml:49`），加载逻辑见 `prediction_slide.py:590,621-624`。本地把同一角色的权重命名为 `cslr_best.ckpt`，放在资源目录而不是 `results/` 下。

| 角色 | 官方要求（路径/文件名） | 本地状态 | 差距与影响 |
|---|---|---|---|
| CSL-Daily 在线 CSLR 主权重 | `Online/CSLR/results/csl-daily_ISLR/ckpts/best.ckpt` | `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt`，499,554,423 B，SHA256 `e9fdd3a6…cbd4d`；938 键，头 `(2001,1024)/(2001,1024)/(2001,2048)`，epoch 89 | **权重可用**；名称/目录不符合官方 loader。官方 loader 找不到文件只记日志并继续（`prediction_slide.py:627-629`），本地严格加载不会静默用随机权重。官方未发布哈希，未做跨源字节对照 |
| CSL-Daily 词表 | `data/csl-daily/csl_iso_with_blank.vocab` | `models/external/EMNLP24_Online/CSL-Daily/meta/csl_iso_with_blank.vocab`，28,821 B，2001 类，`<blank>`=0，SHA256 `0c4984f0…4c76` | **可用**；官方路径需复制/软链 |
| Phoenix-2014T 在线 CSLR | `results/phoenix-2014t_ISLR/ckpts/best.ckpt` | `Phoenix-2014T/ckpts/cslr_best.ckpt`，456,022,647 B | 文件存在，但当前 `slbridge/slrt` 只支持 CSL/2001 类，**不可直接运行** |
| Phoenix-2014T 在线 SLT（mBART-de） | `results/g2t_wait2/ckpts/best.ckpt` | `Phoenix-2014T/ckpts/slt_best.ckpt`，1,434,238,071 B，520 个 `translation_network.*` 键，无 visual 键 | 文件存在，未接入；不是 CSL-Daily 的 G2T |
| CSL-Daily 在线 SLT（mBART-zh） | `results/g2t_wait2_csl/ckpts/best.ckpt` | **缺失**；本地发布包 CSL-Daily 目录只有 `cslr_best.ckpt`；官方 SLT README 无 ckpt 链接 | **阻塞**：论文 20.63 BLEU 无法用本地这份发布包中的权重复现；是否另有发布需向作者确认 |
| `mBART_zh` 预训练资源 | `pretrained_models/mBart_zh/{gloss_embeddings.bin,gloss2ids.pkl,old2new_vocab.pkl,HF 权重}` | **缺失**；仓库无此目录、无下载链接 | 阻塞 SLT 训练/推理 |
| `mBART_de_2/gloss2ids.pkl`（ISLR 用） | `pretrained_models/mBart_de_2/gloss2ids.pkl` | **缺失**；本地另有 `Paper-algorithm/SLRT-main/TwoStreamNetwork/data/csl-daily/gloss2ids.pkl`，但它是 2004 项、含 `<si>/<unk>/<pad></s>` 的 TwoStream 词表，**不是** ISLR/G2T 的同角色文件 | 官方 `prediction_slide.py` 只把 `gloss2id_file` 用于构建 tokenizer；本地推理用 JSON 词表，不受影响。但跑官方训练/评测仍缺 |
| K400 S3D 预训练 | `pretrained_models/s3ds_actioncls_ckpt` | **缺失** | 仅训练/融合初始化需要；发布权重推理不需要 |
| TwoStream-SLR S2G 权重 | `Online/CTC_fusion/results/{ds}_s2g/ckpts/best.ckpt` | **缺失** | 切词、CTC_fusion 训练/在线 TwoStream 推理需要 |
| HRNet-W48 DARK 姿态 | `gen_pose.py` 默认从 OpenMMLab 下载 `hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth` | `models/external/uni_sign/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth`，254,976,311 B（2026-10-25 12:25） | **权重在，配置/运行栈不在**；不属于 `EMNLP24_Online` 发布包；来源未记录在 `extra-downloads.json`，未与 OpenMMLab 公布哈希对照 |
| Faster-RCNN person 检测 | `faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth` | `models/external/uni_sign/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth`，165,671,301 B | 同上；还缺 mmdet 配置与 `person` 类别校验所需的 legacy 栈 |
| SLT 预测输入 | `../CSLR/results/online_slr_csl/prediction_slide/csl_pred.dev/.test` | **缺失**；`prediction_slide.py` 只写 `*_results.pkl`/`*_evaluation_results.pkl`/`*_logits.pkl` | **阻塞**；上游没有生成脚本，需自己实现 pkl→pred 转换 |
| CTC_fusion 在线特征 | `CSLR/results/csl-daily_ISLR/prediction_slide/{split}/{split}_features.pkl` | **缺失**；本地网络不导出 block5 特征 | 仅跑 `prediction_slide.py --save_fea 1` 才能生成 |

角色佐证：`cslr_best.ckpt` 的 `epoch=89`，与在线 ISLR 配置 `total_epoch: 100`（`slide_csl-daily.yaml:54`）相容，而离线 S2G 配置是 `total_epoch: 40`（`csl-daily_s2g.yaml:49`）；结合 938 键、三头 2001 类，支持它是**在线 ISLR 权重**而不是离线 S2G 权重。仍未与作者发布包的哈希做跨源字节对照，官方也没有公布哈希。

## 3. 数据资产盘点：现有 meta 不是句级评测集

| 资产 | 官方要求 | 本地状态 | 说明 |
|---|---|---|---|
| CSL-Daily 连续视频/帧 zip | `data/csl-daily/sentence_frames-512x512.zip`（512×512 帧） | **缺失**；`data/` 只有 CSL-NEWS 标签和项目自有序列 | 没有视频就无法生成 RGB 流；仅有姿态也不够（双流需要 RGB） |
| HRNet 关键点 pkl | `data/csl-daily/keypoints_hrnet_dark_coco_wholebody_iso.pkl` | **缺失**；只有 3 个 `artifacts/unisign-csldaily-probe/S00*.pkl` 快照 | 在线评测需要逐帧 `[T,133,3]`，512 参考像素、x/y/confidence |
| 句级标注 | `data/csl-daily/csl.train/dev/test` | **缺失**；已有 `csl_iso_center_label_bag2items_halfblk.train`（dict，200,530 个词片段）、`csl_iso_with_blank.dev`（14,059 个词片段）、`.test`（15,826 个词片段），字段 `video_file/name/label/seq_len/start/end` | 这些是 **ISLR 词级** 标注，`dataset_name: csl` 的句级评测不能直接复用 |
| ISLR 词表 | `csl_iso_with_blank.vocab` | **已有**，2001 类 | 词级训练/评估可用；不等于句级评测数据 |
| ISLR 训练 meta | `csl_iso_center_label_bag2items_halfblk.train` | **已有**（169 MB，dict 200,530） | 可支持词级 ISLR 训练，不足以支持 `prediction_slide.py` |
| Phoenix-2014T 数据 | 视频 zip、关键点 pkl、句级 meta、词表 | 本地仅有词表与 meta（`phoenix_iso_*`），无视频/关键点 | 与 CSL-Daily 同类的缺口 |

数据来源是官方 CSLR README 给出的作者 SharePoint（`Online/CSLR/README.md:22`）。本轮没有下载这些数据；本地 `EMNLP24_Online` 是模型/词表/词级 meta 包，不是完整数据包。

## 4. 配置与协议对照

官方 CSL-Daily 在线配置 `slide_csl-daily.yaml` 的关键字段：`num_output_frames: -1`、`win_size: 16`、`stride: 1`、`split_size: 24`、`prob_thr: [-1]`、`model_dir: results/csl-daily_ISLR`、`testing.beam_size: 10`（第 20–25、45–49 行）；模型配置含 `use_block: 5`、`freeze_block: 0`、`heatmap_cfg raw_size 512 / map_size 170 / input_size 112 / sigma 8`（第 76–90 行）。

| 配置/行为 | 官方 | 本地 | 差距判断 |
|---|---|---|---|
| 窗口/步长 | 16 / 1 | `window_size=16` / `stride=1`（`configs/slrt-csl.example.json`；`assets.py:63-64`） | **一致**；本地允许 stride>1，但文档已标为实验配置，且官方 stride=1 的居中边缘复制与本地首/末帧 clamp 等价 |
| 投票 | `decode_win_size ∈ {3,5,7,9,11,13}` 全部计算；窗内多数票 `count > B/2`，否则 blank；再压缩相邻重复、删 blank（`prediction_slide.py:336-430`） | `vote_size` 单一值（默认 7）；`counts.max() > half`，否则 blank；事件式压缩/删 blank（`decoder.py:48-78`） | 默认 B=7 语义**等价**；本地不输出 3/5/9/11/13 多组结果，也不保存原始 `win_index`/投票序列 |
| 概率阈值 | `prob_thr: [-1]` → 全部窗口保留（`prediction_slide.py:259-264`） | 无 `prob_thr`；所有窗口参与投票 | 对 CSL 配置**等价**（-1 等于禁用）；若换用其他配置的 0.2，本地无法表达“先过滤窗口再投票” |
| `blank_thr` | 默认 0.5，只记录日志，不参与决策（`prediction_slide.py:170,219,592,598`） | 无 | 不影响结果 |
| 解码 | 每窗 `argsort` top-k=10，取第 0 位（`recognition.py:385-389`） | 每窗 `argmax`（`decoder.py:35-42`） | **等价**（top-1） |
| 三头 ensemble | `log(mean(softmax(rgb), softmax(keypoint), softmax(fuse)))`（`recognition.py:874-875`） | `mean` 三个 `softmax`（`network.py:68-71`） | 数学等价；已记录同窗口数值对照 |
| `split_size` | 每 24 个窗拼一个 batch 前向（`prediction_slide.py:227`） | 逐窗前向 | 只影响显存/速度，不改变切窗协议；本地没有整段预切窗，天然是在线逐窗 |
| checkpoint 定位 | `{model_dir}/ckpts/best.ckpt`（`prediction_slide.py:621`） | `{resource_dir}/ckpts/cslr_best.ckpt`（`assets.py:102-104`） | **路径/命名不一致**，需要复制或改配置 |
| 模型结构配置 | YAML 驱动：`use_block/freeze_block/lateral/pyramid/aux_head/visual_head/split_setting` | `network.py` 硬编码 CSL-Daily 的 5-block 双流 + 三头；不读 YAML | 只覆盖一种已训练变体；Phoenix、融合 adapter、`model_ex`、`split_setting` 不可表达 |
| 数据集字段 | `keypoint_file/zip_file/train/dev/test/vocab_file` | 只有 `resource_dir`/`vocabulary`，无视频/关键点/句级标注加载器 | 官方 `prediction_slide.py` 无法用本地 JSON；本地 CLI 也无法读官方 YAML 的数据项 |
| 语言/类别数 | CSL 2001；Phoenix 1116 | 固定 `sign_language='csl'`、2001 类（`assets.py:37-48,78-79`） | Phoenix 权重存在但**配置层拒绝**；不是“缺权重”而是缺扩展 |
| 特征导出/融合 | `--save_fea 1` 导出 block4/block5 特征；`pred_src ∈ {ensemble,fuse}`；`model_ex` 可选 | 无 `save_fea`、无 `pred_src`、无 `model_ex` | CTC_fusion/离线增强**未接入** |
| 设备 | 脚本硬编码 `cfg['device']=cuda`（`prediction_slide.py:600`） | `device=auto/cpu/cuda` | 本地更宽松；官方入口在无 CUDA 时会直接失败 |
| 输出与指标 | `{split}_results.pkl`、`{split}_evaluation_results.pkl`（WER/DEL/INS/SUB，百分数） | 事件 JSONL + `result.json`，WER 为**比例**（0–1） | 输出协议不同；比较论文数字时本地值需 ×100，且需确认同一 token 化/清洗 |

### 4.1 协议等价性的边界

- **stride=1 时边缘填充等价**：官方 `sliding_windows` 取 `num_clips=ceil(T/stride)`，`final_frames=(num_clips-1)*stride+win_size`，`pad_left=(final_frames-T)//2`（`prediction_slide.py:96-121`）；stride=1 时恒为左 7 右 8，首/末帧重复。本地 `stream.py:134-159` 以 anchor 为中心、越界索引 clamp 到首/末帧，结果相同。**stride>1 时不保证等价**，本地文档已把非 1 标为实验。
- **投票边缘填充等价**：官方对投票序列和 logits 都做 `B//2` 的首/末重复（`prediction_slide.py:343-352`）；本地 `decoder.py:51-57` 用 `_first/_last` 重复，等价。
- **`A blank A` 保留**：官方先压缩相邻重复再删 blank；本地在 blank 处结束 run、后续 A 另起 run，最终两个 A。语义一致，但本地事件流会给出两个 `final`，与官方离线拼串不同。
- **本地不做官方 `index2token` 的 Phoenix 大小写映射**（`prediction_slide.py:59-71`）；CSL 中文不受影响。
- **官方评测是整段视频预切窗 + 后处理**，不是摄像头流；本地流式会话的“在线”是另一层工程实现。两者在同协议下应能对齐，但不能把本地流式示例当作官方 `prediction_slide.py` 的逐位复现。

## 5. 环境与工具链差距

官方 `Online/requirements.txt` 固定：`torch==1.9.0+cu102`、`torchvision==0.10.0`、`numpy==1.21.6`、`transformers==4.11.3`、`timm==0.4.12`、`opencv-python==4.2.0.32`、`Pillow==9.2.0`、`ctcdecode`、`wandb==0.12.17`、`Lintel==1.0`、`gensim`、`azureml*`；推荐 Docker 镜像 `rzuo/pose:sing_ISLR`（`Online/README.md` 环境段）。`gen_pose.py` 另需从源码安装 **mmdet + mmpose**，版本未给，且其默认 `--config configs/det.yaml` 在仓库中不存在。

本机 `.conda-env` 实测：Python **3.10.22**、Torch **2.9.1+cu128**、torchvision **0.24.1+cu128**、numpy 1.26.3、transformers **4.40.0**、Pillow 12.3.0、opencv 4.10、RTMLib 0.0.13；**没有** `mmpose/mmdet/mmcv/timm/ctcdecode/wandb/tensorboard/azureml/gensim/Lintel`。项目 `requirements-slrt.txt` 只声明 Torch/torchvision/numpy/Pillow/opencv/psutil，并明确不把旧版 OpenMMLab 强装进当前环境。Docker CLI 为 29.7.2，但官方 `rzuo/pose:sing_ISLR` 镜像未拉取、daemon 与 GPU 透传本轮未验证。

| 影响 | 说明 |
|---|---|
| 官方入口不能直接跑 | `prediction_slide.py` 顶部 import `wandb`、`tensorboard`；`modelling/two_stream.py` 依赖 `timm`，`modelling/resnet3d.py` 依赖 `mmcv`；当前 `.conda-env` 缺包 |
| 旧栈无法原样复刻 | torch 1.9.0+cu102 没有 Python 3.10 轮子，也不支持 RTX 5060 Ti（Blackwell/sm_120）；官方镜像的 CUDA 10.2 与该 GPU 不匹配。要跑官方代码必须在另一台机器/容器或用 CPU，或做现代 Torch 移植 |
| Docker 镜像 | 官方推荐 `rzuo/pose:sing_ISLR`；本机只有 Docker CLI 29.7.2，镜像未拉取，daemon/GPU 透传未验证（参见 `docs/reports/online-cslr-study-2026-10-23/README.md` 的本机运行条件表） |
| HRNet 前端需要独立环境 | mmpose 0.x + mmdet 2.x + mmcv-full 与 torch 2.9/CUDA 12.8 不兼容；即便权重已在本地，也需要单独 legacy 环境（或改写成现代实现并重新验证分布） |
| `ctcdecode` 不是在线 CSLR 必需 | `prediction_slide.py` 不 import ctcdecode，`testing.beam_size` 对在线滑窗评测无效；它只影响离线 `prediction.py` 和融合/训练路径 |
| 指标库版本 | 论文用 SacreBLEU 1.4.2；本地 `sacrebleu 2.2.0`、`rouge_score 0.1.2`。做 SLT BLEU 对照时必须固定版本与 tokenizer，否则不能与 20.63 直接比 |

## 6. 已核验事实与未验证边界

**已核验（有本地证据）**
- 真实 CSL-Daily 权重严格加载：938 键、2001 类、blank=0；三头形状 `(2001,1024)/(2001,1024)/(2001,2048)`；pose 首层 63 通道，RGB 3 通道；epoch 89。
- 词表 2001 类、`<blank>` 在 0；本地网络/预处理与官方 CSL 配置的几何、窗口、投票、ensemble 语义一致；此前记录与官方源码窗口级数值对照 `max_abs=0`（`docs/reports/slrt-integration-2026-10-24.md` 第 3、17 行）。
- 真实 MP4 可产出 Gloss；CE-CSL 跨域历史 WER 0.8918、空输出 113/500（`docs/reports/slrt-dataset-eval-2026-10-24.md` 第 8、28 行）——**跨域、实验姿态前端、不是 CSL-Daily 同域复现**。

**未验证 / 不能声明**
- CSL-Daily 同域在线 WER（无论 29.3 还是其他）：缺视频、官方关键点、句级标注。
- 官方 HRNet 端到端：缺配置与 legacy 栈；本地真实视频走的是 `rtmpose-experimental`，估计器分布不同。
- 在线 SLT（CSL-Daily 20.63 / Phoenix-2014T 23.69）：缺 CSL-Daily G2T 权重、mBART_zh、`*_pred.dev/test` 转换。
- CTC_fusion（离线增强）：缺 TwoStream-SLR S2G、K400、在线特征、S2G 词表。
- 摄像头实时交付：历史试跑未通过，与本报告的数据/权重差距是不同问题。

**上游源码核对**：本地上游镜像 `Paper-algorithm/SLRT-main/Online` 与 GitHub `main` 的 580 个 blob 比对，**577 个 blob SHA1 相同**，3 个 `utils/video_transformation.py` 不同（训练增强工具，不影响推理路径）。本地记录的 `latest_directory_commit` 为 `cb393becf08e47ecdf4196de4cd41449a4b02cbc`，GitHub 当前 `main` 树为 `38a4f7b…`；引用具体行号前以本地镜像为准，并注意这 3 个文件的版本差异。

## 7. 最小补齐路径（按可执行顺序）

**A. 先做 CSL-Daily 在线 CSLR（目标：论文 29.3% WER 的同协议对照）**
1. 取得作者发布的预处理数据（`Online/CSLR/README.md:22` 的 SharePoint 链接）或 CSL-Daily 原始视频，准备：
   - `sentence_frames-512x512.zip`（或确认 zip 内帧布局与官方 `read_jpg` 一致）；
   - `keypoints_hrnet_dark_coco_wholebody_iso.pkl`（512 参考像素、133 点、x/y/confidence）；
   - `csl.train/dev/test` 句级标注。
2. 把本地 `cslr_best.ckpt` 复制/软链为官方期望的 `Online/CSLR/results/csl-daily_ISLR/ckpts/best.ckpt`，逐字节记录 SHA256；不要依赖官方 loader 的“缺文件继续”分支。
3. 选择评测路径：
   - **路径 1（官方入口）**：在独立 legacy 环境/容器中安装旧依赖，跑 `prediction_slide.py --config=configs/slide_csl-daily.yaml`；需要接受 torch 1.9/CUDA 10.2 与当前 GPU 不兼容，或在其他机器上跑。
   - **路径 2（本地入口移植）**：给 `slbridge/slrt` 增加“官方数据集目录”输入（读 `csl.dev/test` + 关键点 pkl + 帧 zip），复用现有 `preprocess.py`/`network.py`/`decoder.py`，再按官方 `wer_list` 口径输出 WER/DEL/INS/SUB（百分比）。这条路径不修改官方数值，但必须把数据加载、切窗、投票和指标实现逐项与 `prediction_slide.py` 做配对测试。
4. 比较时固定：词表、blank=0、B=7、W=16/S=1、同一切窗/边缘复制、同一参考文本、百分比口径。先复现 `naive_greedy`，再复现 `window_greedy_7`。

**B. 官方 HRNet 前端**
- 用本地已有的两个权重，补 mmdet/mmpose 配置（`faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py`、`hrnet_w48_coco_wholebody_384x288_dark_plus.py` 及其 `_base_`），在隔离 legacy 环境跑 `gen_pose.py`（显式传 `--config`，不要依赖不存在的 `configs/det.yaml`）。
- 验证：133 点排序/坐标/置信度、512 参考系、逐帧索引与视频时间戳一致；再用同一段视频对比 `rtmpose-experimental` 与 HRNet 的 Gloss/分数，量化前端差异。未完成前，任何结果继续标 `pose.experimental=true`。

**C. 在线 SLT**
- 先确认作者是否提供 CSL-Daily G2T checkpoint；当前发布包没有。若没有：
  1. 准备 `mBART_zh`（含 `gloss_embeddings.bin`、`gloss2ids.pkl`、`old2new_vocab.pkl`）；
  2. 用 `Online/SLT` 的 `configs/g2t_wait2_csl.yaml` 训练（`wait_k=2`，训练/验证配置见该文件）；
  3. 把 CSLR 输出的 `*_results.pkl` 转成 SLT 需要的 `csl_pred.dev/.test`（上游没有转换脚本，需要自己写并记录协议）；
  4. 用固定 SacreBLEU 1.4.2 与同一 tokenizer 报告 BLEU-4，并区分 GT-Gloss 与 predicted-Gloss 两种输入。
- Phoenix-2014T 的 `slt_best.ckpt` 只能用于 Phoenix 复现；当前 CSL-only 模块需先扩展词表/语言与模型加载，不能直接套用。

**D. CTC_fusion（可选、离线增强）**
- 需要 TwoStream-SLR S2G `best.ckpt`、K400 S3D、`csl_s2g_gloss2ids.pkl`、由官方 `prediction_slide.py --save_fea 1` 生成的在线特征；本地网络不导出特征，不能替代。此项不是在线 CSLR 主链，建议单独立项。

## 8. 证据索引

- 官方配置：`Paper-algorithm/SLRT-main/Online/CSLR/configs/slide_csl-daily.yaml:7,12-25,45-49,75-90`；`slide_phoenix-2014t.yaml:7-25,45`；`Online/SLT/configs/g2t_wait2_csl.yaml:4-6,57-75`。
- 官方推理/加载：`Online/CSLR/prediction_slide.py:96-121`（切窗/填充）、`166-171`（B/阈值/方法列表）、`205-276`（阈值过滤/切 batch）、`336-430`（多 B 投票/去重/删 blank）、`481-491`（输出）、`590-629`（ckpt 路径与静默继续）。
- 官方数据/前端：`Online/CSLR/dataset/Dataset.py:9-25,56-74`（63 点选择）、`dataset/VideoLoader.py:48-88`（512 帧读取/320 缩放）、`Online/CSLR/gen_pose.py:52-63`（HRNet/Faster-RCNN 默认权重）。
- 官方 SLT/融合：`Online/SLT/configs/g2t_wait2_csl.yaml`；`Online/CTC_fusion/configs/csl-daily_s2g.yaml:8,44,70-79`；`csl-daily_fuse_online.yaml:21-23,47,75-82`。
- 本地权重/配置：`slbridge/slrt/assets.py:37-48,78-79,102-108`；`slbridge/slrt/network.py:60-80`；`slbridge/slrt/preprocess.py:69-103`；`slbridge/slrt/decoder.py:48-78`；`slbridge/slrt/stream.py:134-159`；`slbridge/slrt/pose.py:18-30,66-71`；`configs/slrt-csl.example.json`；`models/external/slrt_csl_bundle/manifest.json`、`models/external/slrt_hrnet_bundle/manifest.json`。
- 本地历史证据：`docs/reports/slrt-integration-2026-10-24.md`（938 键、2001 类、数值对照、未验证 HRNet）；`docs/reports/slrt-dataset-eval-2026-10-24.md`（数据可用性、CE-CSL 跨域 WER、实时回放未通过）。
- 论文目标：`docs/reports/online-cslr-study-2026-10-23/zuo-2024/paper-card.md:147-152`（W=16/B=7、wait-2、SacreBLEU 1.4.2、WER 22.1/22.1/29.3、BLEU-4 23.69/20.63）。
