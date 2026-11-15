# 项目记忆：模型、评测与实时链路的坑

最近核验：**2026-10-28**（本次补充三维生成路线核验；既有实验仍以各节日期为准）。适用于当前 SignLanguage-Bridge 工作区。这里保存易重复踩到的原因与处理规则；运行命令见[集成说明](../uni-sign-integration.md)，逐条证据见文末索引。复用结论前核对当前源码、manifest、运行协议和产物；缺失的本地证据标为缺失。

动画专项补充核查：**2026-11-06**，仅核查动画功能分支的显示处理与开源实现器；无新增动作、性能或语言验收通过结论，见 [R23]。

## 1. 任务和指标先对齐

- 当前 `csl_daily_pose_only_slt.pth` 是 **CSL-Daily、pose-only、整句 SLT** 权重。框架支持多个任务，不代表这份权重已经适配 CE-CSL 或能直接输出逐词 Gloss。修改 task 名称不能替代对应训练。
- Uni-Sign 的中文译文、TFNet 的 Gloss 序列、孤立动作分类是不同目标。分别报告文本指标、WER、分类准确率；字符编辑相似度或 `1-WER` 均不能改称产品识别准确率。
- CSL-Daily 约 25.70 BLEU 的既有参照是上游 **1,176 条预测快照**，不是本机同权重同协议视频重跑。比较时保留该来源边界。
- **论文（ICLR 2025 / arXiv:2501.15187）没有 CSL-News 的翻译指标**：CSL-News 只是 1,985 小时预训练语料（Table 1），下游表格是 CSL-Daily（pose-only test BLEU-4 **25.61** / ROUGE **54.92**；RGB+pose 26.36 / 56.51；CSLR test WER 27.4 / 26.0）、How2Sign（14.5 / 34.3）、OpenASL（22.67 / 42.77）、MSASL1000（77.88 / 76.55）。**不要拿 CSL-News 上的任何分数去"对照论文"**。
- 2026-10-25 核对：用上游 1,176 条 test 预测+参考、按论文口径（`sacrebleu` tokenize=13a）重算得 BLEU-4 **25.70**、ROUGE-L F1 53.59 → 指标口径可对上（BLEU 差 0.09；ROUGE 差 1.33 来自实现不同），但**这是重算上游预测，不是本机重跑**。上游 `SLRT_metrics.translation_performance` 用 `rouge` 包 + `sacrebleu` 13a；CSL-Daily 参考句是**空格分字**（`他 今 年 四 岁 。`）。[R10]
- **CSL-Daily 已在本地复现（2026-10-25）**：下载作者发布的 `csl_daily_pose_format.zip`（4.59 GB / 20,654 pkl），按 `test_tmp_refs.txt` 的 1,176 个 sample ID 抽 test pose，用本仓库运行时（pinned 权重、bf16、beam=4）跑出 BLEU-4 **25.55**（论文 25.61，差 0.06），**1,029/1,176（87.5%）预测与作者快照逐字相同**；残差来自帧采样（上游 `random.sample` vs 本仓库确定性 linspace）。脚本 `scripts/evaluate_unisign_csldaily_pose.py`。[R10]
- **上游有个必须复刻的格式后处理，否则 BLEU 会假崩**：`fine_tuning.py:282-284` 在算指标前执行 `' '.join(list(r.replace(" ","").replace("\n","")))`（注释 "fix mt5 tokenizer bug"），即把连续中文逐字加空格；参考句同样是逐字空格分隔。**不做这一步，同一份预测的 BLEU-4 从 25.55 掉到 0.0005**（每句退化成 1 个 token，BP 0.0026），而 chrF 22.39 与**字符级** ROUGE 0.534 保持正常（token 级 ROUGE 会一起崩到 0）——**chrF 与字符级 ROUGE 正常而 BLEU 接近 0，首先怀疑分词/空格格式，不要当成模型失败**。[R10]
- **CSL-Daily 视频是正方形**：发布 pkl 的 `w_h` 字段为 1280×1280 或 1080×1080，因此在 CSL-Daily 上 `wh` 与 `isotropic_short_side` 两种归一化**等价**；这也解释了为什么 CE-CSL（16:9）上 isotropic 更接近方形训练分布。pkl 还含 `start/end` 裁剪字段（如 170 帧只取 [92,150)），必须按上游 `load_pose` 处理。[R10]
- 上游 eval 的帧采样是 `random.sample(range(duration), k=max_length)`（**随机**，`datasets.py:472-475`），本仓库用确定性 `linspace`。因此逐位复现论文数字不现实，合理目标是 ±0.5 BLEU；严格对照必须固定同一采样器。
- 分析词汇覆盖应读取训练标注；比较两套测试文本不能代表训练覆盖。汉字覆盖也不等于对应手语动作覆盖。[R1]、[R2]
- **CSL-News 是 Uni-Sign 作者的发布数据集，但发布里没有它的下游微调权重**（HF `ZechengLi19/Uni-Sign` 只有 CSL-Daily/WLASL/How2Sign/OpenASL 下游权重 + stage1/2 预训练权重）。2026-10-25 实测：项目集成的 `csl_daily_pose_only_slt.pth` 在 CSL-News 片段上 BLEU-zh **1.27**、编辑相似度 **7.90%**；换成 `csl_stage1_weight.pth`（CSL-News pose-only 预训练，627 张量、与 pose-only 模型键集完全一致）后同一批数据升到 **16.75**，配上上游 MMPose pose 为 **24.67**。**选错权重比选错前端严重一个量级。** [R10]
- **本地 `data/CSL-NEWS/archive_001` 的 1694 条不是留出集**：它们恰是 `CSL_News_Labels.csv` 第 1..1694 行，落在上游 `datasets.py:527-532` 的 train 段 `[0, 99%)`。stage1 在其上的 23.76 BLEU 是训练集拟合，不得写成 CSL-News 准确率。要可对外引用必须拿到 `[99%, 100%)` 尾部视频或自建留出划分。[R10]

## 2. 姿态协议与方向会显著改变结果

- Uni-Sign 输入为 RTMPose Wholebody **133 点的 x、y、confidence**，再取身体／左右手／面部共 69 点。第 3 通道是置信度，不能换成 z，也不能直接套用 MediaPipe 225 维协议。
- 宽屏视频分别除以 W/H 会改变动作几何。当前使用 `isotropic_short_side`，两个轴同除以短边；`wh` 仅用于明确标注的历史复现。单测要覆盖非方形输入。
- **归一化必须跟随「训练数据的生产者」，而不是全局固定（2026-10-25 更正）**：上游 `demo/pose_extraction.py` 用的就是 rtmlib `Wholebody`（默认 `mode=lightweight`，与本地运行时同一个模型），但它保存时做 `keypoints / [W, H]`（**x/W、y/H**）。CSL-News 300 条配对实测：stage1 权重下 isotropic→wh 让 BLEU-zh **15.67→22.02**、编辑相似度 **19.95%→27.12%**；换成上游 pkl 也是 17.75→23.53。**同一权重换归一化可差 6 BLEU，属于协议字段**，必须写进 manifest 与评测协议。对 CSL-News 域应设 `pose_normalization="wh"`；CE-CSL 的 isotropic 结论是另一数据域 + 另一权重的经验结果，不要直接外推，要判定需先查 CSL-Daily 官方 pose 的归一化。[R10]
- **前端模型本身差异很小**：本地提取与上游 pkl×(W,H) 逐点比对，78–82% 关键点差 <1 px、87–91% <2 px（均值 2–3 px）。在 wh 归一化下，本地估计器与上游只差 **1.51 BLEU / 1.33 pp**——把前端问题归因于"模型不同"会错，主因是归一化。[R10]
- 镜像需要同步反射 x、交换左右身体／面部点和左右手，并保留置信度、只读性和方向记录；重复应用同一目标方向应保持不变。单纯把画面显示为镜像不能代替后端校正。
- 同时验证文件和采集片段的公开入口，不能只测预处理 helper。独立进程将方向操作替换为无操作时，mirror 用例应失败；前轮该负控实际得到 2 个 mirror 失败、2 个 preserve 通过，证明入口测试能发现漏接线。
- **方向映射绑定姿态协议，改任何姿态相关开关都要重校准（2026-10-25 实测）**：`pose_detect_stride` 进入 pose protocol 后，旧的 dev 校准映射会被 `load_direction_map` 拒绝；重新校准（`--pose-detect-stride 5`）得到的方向与旧映射完全一致（B/E/G/I/L=mirror），说明 stride 不影响方向判定，但**必须走重校准流程**，不能手工改映射文件。另：协议 `schema_version` 从 2 升到 3 时旧映射也会被拒——只允许"格式版本漂移、语义字段全等"这一种兼容，其余一律硬失败。[R10]
- 方向依赖录制条件。先在 train/dev 校准，锁定后验证；生产默认 `preserve`。不能统一翻转全部数据，也不能按测试参考句逐条选择更优预测。
- 2026-10-24 的 CE-CSL 锁定表仅适用于该批数据：B/E/G/I/L 为 mirror，其余 preserve。H 未达到 dev 预设增益阈值，保持原方向。此表不适用于任意摄像头／陌生签者。[R1]、[R2]

## 3. 缓存与续跑必须绑定输入和完整协议

- 大小、mtime 相同不代表视频内容相同。持久缓存与评测输入绑定视频 SHA256，并核对姿态模式、实际设备、归一化、帧数和预处理版本；加载数组后恢复只读属性。
- 预测续跑绑定权重、源码／分词器版本、实际设备与 dtype、解码参数、方向表、标签和所选输入。协议不同就使用新输出目录，避免混合成绩。
- SLRT 续跑逐条核验协议、视频SHA、参考、方向、split与唯一ID，并重算WER；成功行必须final。缺样本、失败或无清单时不得标记完整，也不得发布整批corpus WER。失败不是模型空识别。[R9]
- 校准续跑拒绝无 manifest 的孤立记录；逐行验证样本身份、校准／验证阶段、参考句、视频和重算分数。验证阶段不能反向修改方向选择。
- 姿态缓存保持规范原方向，解码前应用方向。旧缓存只通过有来源审计的显式导入使用；补录内容哈希不能反向证明旧提取时的视频内容。数值不变的缓存迁移也不能写成视频重新提取。[R2]

## 4. 数据划分与裁剪探针的陷阱

- CE-CSL 官方 train/dev/test 为 4,973/515/500；A–L 签者跨划分重叠，成绩不是陌生签者泛化。校准、验证、test 的用途和样本交集必须核验。
- 已检查的 test 上重新解码属于修复回归，不能改称新盲测。失败、空预测和未完成样本须计数；成功子集指标应明确标注。
- 历史裁剪探针生成 `5-95`／`2-98`，调用却请求 `05-95`／`02-98`，再静默回退整片，导致两组实际没有裁剪。当前已统一两位键名并改为直接索引。先验证实际区间／帧数改变，再解释指标。
- 不能把整体低分都归因于 256 帧上限：历史 test 500 条中只有 62 条被下采样。比较裁剪或前端时固定方向、权重、解码和样本，避免多因素一起变化。[R1]、[R2]

## 5. 帧数窗口与时间窗口分开判断

- **已修复的边界错误：** N 帧在 F FPS 下首末跨度为 `(N-1)/F`。旧 CLI 同时限制 N 帧、要求跨度至少 `N/F`，默认 48 帧／30 FPS 可永远无法就绪；独立模型 benchmark 绕过调度器时发现不了它。
- 当前 `ContinuousConfig.minimum_frames` 缺省保留原时间模式；指定时按收到的帧数就绪。Uni-Sign CLI 和会话 smoke 同步采用 `minimum_frames=N, max_frames=N`，保留 8 秒及字节预算上限。
- 帧数模式要覆盖实际 25、29.97、30、31.26、60 FPS，而非只测试标称 FPS。还要检查过期丢弃、来源中断、EOF 取消，以及预算不足时有界。预算不足可能积不满窗口，不能承诺任意配置必定出窗。
- 标称 FPS 仍参与步长、参数预检与结果年龄界限；帧数就绪不等于动作时长归一化，更不等于语义识别正确。[R3]、[R4]

## 6. 滑窗、文本生成与原生流式的区别

- 当前是持续采集加完整窗口重复推理；`online_inference.py` 的名称不是视频增量能力证据。当前模型使用对称时间卷积、双向 mT5 encoder，窗口变动还会重算身体 bbox/scale，使旧帧归一化坐标变化。
- 缓存 RTMPose 点不等于缓存可直接追加的神经特征。单次 generate 内部可用 decoder KV cache，但当前没有跨视频窗口的 encoder／decoder 状态复用。
- 中文字符或 SentencePiece token 不等于手语词。当前 `TextPrediction` 返回整句文本、`score=None`；beam=4 不代表已经提供 N-best、词级概率或动作时间对齐。增加文字 streamer 也不能替代视频增量识别，且需检查锁定库版本与 beam 的兼容约束。
- 前缀诊断要衡量改写和提交稳定性，不能拿截断视频对照整句参考当作逐词准确率。词级功能需先确定 Gloss／孤立词／指拼等目标单位及对应监督。[R3]、[R4]

## 7. 性能与 Windows 环境的坑

- **重复训练计算已修复（2026-11-11）：** 历史上游forward先执行未使用的sentinel loss，再generate，24条完整推理均调用mT5 encoder两次。当前runtime临时捕获上游视觉embedding/mask，在训练段前退出并移除hook，再使用原generate；固定30条官方train姿态加4/256帧合成控制，32输入embedding/mask摘要一致，重复3次的96对token/text一致，encoder两次变一次。独占模型阶段平均324.53→300.50ms。这是效率改善，不是低BLEU解释或摄像头延迟保证；详见[本轮报告](../reports/unisign-generation-optimization-2026-11-11.md)。
- **延迟 93% 在姿态前端，且是纯逐帧成本（2026-10-25 实测）**：单条 8.7 s 片段端到端 5.54 s，其中解码+RTMPose 5.13 s，mT5 仅 0.394 s。回归 `pose_extract_ms ≈ -67 + 20.12×frames`（n=182，R²=0.994）说明没有可摊销的固定开销，约 50 fps。每帧跑两个 ONNX 模型：YOLOX-tiny 检测器输入 **416×416**（`Wholebody` 从 MODE 表取，不是 YOLOX 类默认的 640；204×170 画面 letterbox 放大 2.04×，实测约占 60%）+ RTMW-DW-L 姿态 192×256。三个易踩点：① rtmlib `mode="lightweight"` 实际加载 `rtmw-dw-l-m`（L 体型），`performance` 才是 `rtmw-dw-x-l@384×288`；② 管线把每帧都过一遍姿态提取（上限 600 帧），但模型只用 256 帧，>256 帧的片段（本数据集 39%）有 30–50% 计算被丢弃；③ batch=1、每帧同步，GPU 利用率约 30%。优化方向：只提采样帧（**已实现** `pose_sample_only`，实测长片段 1.65–1.94×，全量推算跳过 14.7% 提取）、跨帧复用检测 bbox（有精度风险，需配对验证）、姿态模型批处理。注意检测输入已是官方 416，不要为了提速再降尺寸。[R10]
- **其他权重不能走生产加载路径**：`UniSignBackend.load_model()` 固定校验 pinned CSL-Daily 权重的字节与 SHA256，换 checkpoint 会报「incomplete or corrupt」。实验必须用 `TorchUniSignRuntime` 直连并在自己的协议文件里记录实际 SHA256（见 `scripts/evaluate_unisign_cslnews_checkpoint.py`），不要为了跑实验放宽生产校验。[R10]
- 论文 RTMPose-x 与当前 lightweight RTMW-L/M 256×192 存在前端差异；强前端收益尚未量化。不能给 pose-only 权重直接打开 RGB 开关并称为已切换模型。
- **前端差异已量化（2026-10-25，CSL-News 379 条配对）**：同一权重、同一解码、同一帧采样，只把姿态来源从本地 rtmlib RTMW-L 换成本文上游发布的 MMPose `.pkl`，`csl_stage1_weight` 上 BLEU-zh 16.75→**24.67**、编辑相似度 20.40%→**28.97%**、P95 0.535→0.715；但在跨域的 `csl_daily_pose_only_slt` 上几乎无变化（+0.19 pp，21/379 预测逐字相同）。**结论：前端差异只在模型能用上细节时体现；跨域低分时不要归因于前端，也不要反过来用跨域结果否定前端。** [R10]
- **两个发布 pose 归档都自带 `w_h`，且都是 `x/W, y/H`（各轴分别归一）**：CSL-News 的 `w_h=[204,170]`（非方形）、CSL-Daily 的 `w_h=[1280,1280]`/`[1080,1080]`（方形，故此时 anisotropic 与 isotropic 等价）。所以 CSL-News 域必须用 `wh`，不能套 isotropic；`_crop_scale` 只抵消**统一**预缩放，不会把各向异性变回各向同性。CSL-Daily 的 pkl 另带 `start/end` 裁剪字段（如 170 帧只取 `[92,150)`），必须按 `load_pose` 处理。[R10]
- **特征处理误差的量级有上界**：在 CSL-Daily in-domain 300 条上故意错配 x 缩放，BLEU-4 从 **36.5**（正确）掉到 34.6（×0.75）/ 30.5（×0.5625）。即归一化错误通常只吃掉 **2–6 BLEU**，不会把 20+ BLEU 的差距解释成"特征处理不一致"；遇到数量级差距先查权重域与样本划分。[R10]
- **32 GB Windows 上并发重任务会产生假失败，必须重试后再判模型失败（2026-10-26 实测）：** 4 个 RTMPose 对齐进程 + 3 个 SLRT CLI（每条子进程重载 499 MB 权重）+ 1 个 Uni-Sign(torch) 同时跑时，可用内存跌到 2–3 GB；SLRT 首轮 **47/627** 失败、Uni-Sign 进程在 240/627 处崩溃，错误全部是内存分配类（`numpy ..._ArrayMemoryError`、OpenCV/FFmpeg `Insufficient memory`、`CUDA error: out of memory`、`fatal: Memory allocation failure`、`get_buffer() failed`）。分片、独占机器重跑后两者都 627/627 成功。**规则：并发上限按可用内存定（本机实测同时最多 3 个 SLRT CLI + 1 个 Uni-Sign 已偏紧）；失败样本必须单独重试；首轮 OOM 不得写成模型失败、空识别或准确率下降。** 另：SLRT CLI 每条样本固定约 3.9 s 加载权重（16.9 s wall 中约 1/4），批量评测的主要固定开销。[R17]
- Windows 已遇到先创建 ORT 会话、后导入 Torch 导致 `WinError 127`；保持当前先 Torch、后 ORT／RTMPose 的加载顺序。Python 3.10 的 `typing.Self` 兼容与 GUI 导入顺序也有既有处理，改依赖时需回归。[R5]
- 环境名含 GPU 不证明具备 Torch 或实际使用 CUDA。读取 runtime 的 device、dtype 和 providers；`.conda-env` 用于 Uni-Sign ML 推理，其他测试环境的 Torch 缺失跳过不能计作该模型通过。[R3]

## 8. 无异常、非空文本、稳定标记都不等于验收

- `smoke_unisign_session.py` 当前退出码 0 仅表示 `error is None`。验收同时读取候选计数、稳定性、取消／丢弃原因和实际显示，不能只看返回码。
- 2026-10-24 的 dev-00015 真实会话回放为 **error=null、候选 0 条**，事件含质量门控清空与 EOF，候选交付未通过。剩余原因需沿观察质量、窗口积累、任务提交和结果交付定位；没有为制造输出而关闭门控。
- 文件推理、缓存重解码、视频实时回放、现场摄像头和真人语义评审分开报告。RTMPose 输入、MediaPipe 活动观察门控和桌面缩放是不同环节；某一环节通过不能替代整链路证据。
- 候选 `confirmed` 仅指连续窗口文字一致，仍可被后窗改写；不代表人工确认、语义真值或不可撤销词提交。原始识别、辅助草案与用户确认分别保存。[R3]、[R4]

## 9. LLM 接入前先建立可靠的词级事件

- Uni-Sign 内部已有 mT5 句子生成；独立的“词级识别 → LLM 表达辅助”尚未接入。当前仅有候选上下文快照与 JSON 导出。
- 重叠候选是同一动作的可能修订，不能直接串接成已识别词序列。保留来源、重叠关系、时间范围、未确认状态及分数的实际含义。
- 面向词级功能先补目标词表／监督、背景与未知处理、去重、稳定提交和修订。TFNet 虽输出 Gloss，但自身也是完整片段非因果模型；有 CTC 不等于可原生流式。
- LLM 可组织表达或提供澄清，不应覆盖原始识别、虚构视觉证据或把模型分数改称已校准置信度。真实词义／句界审核与 LLM 接口另行验收。[R3]、[R4]

## 10. SLRT Online 接入也有独立的协议和验收边界

- 新增 `slbridge/slrt/` 使用 EMNLP24 CSL-Daily 双流 S3D：938 个参数键、2001 类、blank=0，RGB 与 COCO-WholeBody133 选出的 63 点热力图同时输入。不能把 Uni-Sign 的 69 点分支或 MediaPipe225 输入交给它。路径和实际命令见 [SLRT 集成说明](../slrt-integration.md)。
- 官方 HRNet-W48 DARK、Faster-RCNN 及 legacy MMPose/MMDetection 不在 `EMNLP24_Online` 包中。当前真实 MP4 采用显式 `rtmpose-experimental`，几何/点位协议可核验，估计器分布仍不同；不得将这种结果写成官方 HRNet 复现。共享 Adapter 的 warmup 必须验证姿态前端，失败时保留原模型。
- 官方默认 16 帧左 7 右 8，7 窗多数投票再等待 3 个窗；只在自然 EOF 填充右边。先压缩相邻重复、再删 blank，保留 `A blank A`。投票的证据范围要覆盖参与投票的左右窗。用 `avg_pool3d` 保留官方 CUDA 累加方式，不能仅凭数学上等价的 mean 宣称数值相同。
- 快速 pause/resume 不得覆盖 reset。输入 epoch 在阻塞 read 前采样，旧 native 调用即使返回也不能发布到新 epoch；reset 后立刻 stop 不能回写旧 final。`is_running` 同时涵盖采集和推理线程，native read 未退出时不能声称摄像头已释放。声明帧数与实解码不一致须报错，禁止用 EOS 生成伪 final。
- 本次实测有真实 Gloss 输出，但 CE-CSL 单段跨域质量较低、摄像头试跑没有稳定 Gloss；有界持续采集不等于实时词汇交付通过。黑帧负样本只证明无人门控，CPU 回退单窗口也不代表 CPU 实时可用。详见[SLRT 实测报告](../reports/slrt-integration-2026-10-24.md)。
- CLI 的 JSON/JSONL 输出路径与源视频、pose、配置及权重必须不同，错误处理也不能覆盖源文件。原始 Gloss 与后续生成文本分别保存；当前 `translation=null`，`mBart_zh_t2g` 不视为已验证的 Gloss-to-Text。
- **先核查异常再解释方向（2026-10-24 更正）：** SLRT 镜像 36 条均因向 `[T,133,3]` 接口传入 `[133,3]` 而失败，实际 0 窗；此前“方向结论与 Uni-Sign 相反”已撤回。还需配对变换 RGB 和 pose。修复后失败单列，整批有失败时 corpus WER 为 null；完整方向探针仍待重做。SLRT 是双流 S3D 短窗分类与投票，不是该权重已训练 CTC 的证明。
- **多次构造 RtmposeExtractor 会使 PATH 无界增长：** `_configure_windows_cuda` 每次前插 DLL 目录，Windows 32767 字符上限在 51 次构造后触发 `putenv` 失败。已修复为按条目去重；批量评测复用单个前端实例，同时省去逐视频重建 ONNX 会话。
- WER 接近 1 时先分解 S/D/I、异常与空输出：SLRT 正常方向跨域 CE-CSL 以删除为主（1,980 D vs 706 S）。`min(OOV,D)` 只是 OOV 删除上界，不能把简单扣除后的数当成词表内 WER，也不能据此排除词表因素或确定视觉层原因。[R8]
- **输入表示与实际入口分开核验（2026-10-24）：** 当前文件评测没有把MediaPipe225直接喂入Uni-Sign/SLRT；前者RTMPose133选69点，后者实验RTMPose133选63点热力图+RGB。点位能对应不等于HRNet估计器分布等价，严格加载与网络数值对照不能证明前端正确。[R9]
- **任意宽屏不能只照搬方形训练帧的后半段处理：** 当前SLRT压320方形再中心裁270，dev-00015的17帧中423个高置信手点次落在RGB裁剪外，包含等待/垂手帧。构图有风险，尚无识别收益对照；不要把点次当丢词率，也不要直接修改官方裁剪并宣称修复。权重训练集CSL-Daily与CE-CSL为不同数据来源，没有证据因中文词汇重叠就认定包含该测试集。[R9]
- **官方在线权重与本地路径不同名（2026-10-25）：** 官方 `prediction_slide.py` 只从 `{model_dir}/ckpts/best.ckpt` 加载，CSL-Daily 的 `model_dir` 是 `results/csl-daily_ISLR`；本地实际文件是 `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt`（499,554,423 B、938 键、2001 类、SHA `e9fdd3a6…cbd4d`）。官方 loader 找不到文件只记日志并以随机权重继续（`prediction_slide.py:627-629`）；本地严格加载不会。跑官方入口前必须建立官方目录并逐字节核对，不能以退出码 0 证明权重正确。[R11]
- **CSL-Daily 的 `meta` 是词级 ISLR 标注，不是句级评测集（2026-10-25；当晚已补句级数据，见下条）：** `csl_iso_center_label_bag2items_halfblk.train` 是 200,530 个词片段的 dict，`csl_iso_with_blank.dev/test` 是 14,059/15,826 个词片段（字段 `video_file/name/label/seq_len/start/end`）；`slide_csl-daily.yaml` 需要的 `csl.train/dev/test`、`sentence_frames-512x512.zip`、`keypoints_hrnet_dark_coco_wholebody_iso.pkl` 当时均缺。不要把这些词级 meta 当句级评测输入。[R11]
- **HRNet/Faster-RCNN 权重本机已有，缺的是配置与 legacy 栈（2026-10-25 更正；后续已补齐，见下条）：** `models/external/uni_sign/` 下已有 `hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth`（254,976,311 B）和 `faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth`（165,671,301 B），文件名与官方默认一致；但 `EMNLP24_Online` 包不含 mmdet/mmpose 配置，`.conda-env` 没有 mmpose 0.x + mmdet 2.x + mmcv-full 兼容栈，`gen_pose.py` 默认的 `configs/det.yaml` 在上游仓库也不存在。来源与哈希未记录在 `uni_sign/extra-downloads.json`，未与 OpenMMLab 公布值对照。正确说法是“权重在、配置和运行栈缺”，不是“本机没有 HRNet 权重”。[R11]
- **CSL-Daily 在线 SLT 在本地发布包中没有权重（2026-10-25；后续已补数据无关资产，见下条）：** 本地 `EMNLP24_Online/CSL-Daily` 只有 `cslr_best.ckpt`；Phoenix-2014T 才有 `slt_best.ckpt`（1,434,238,071 B，520 个 `translation_network.*` 键）。`mBART_zh` 的 `gloss_embeddings.bin/gloss2ids.pkl/old2new_vocab.pkl` 本机与仓库都没有；作者 Google Drive 的 `mBart_zh_t2g` 是 Spoken2Sign Text2Gloss 资产，不是 Online G2T 的 `mBART_zh`。SLT 需要的 `csl_pred.dev/.test` 也不是 `prediction_slide.py` 的输出。官方 SLT README 未给 ckpt 链接，是否另有发布需向作者确认；论文 20.63 BLEU 不能声称“有发布权重可复现”。[R11]
- **本地 SLRT 当前只覆盖 CSL/2001 类（2026-10-25）：** `SLRTConfig` 拒绝非 `csl`，`load_vocabulary` 固定 2001 类；Phoenix-2014T 的 CSLR/SLT 权重虽在本地，也无法通过当前配置加载。引用 Phoenix 在线结果前先扩展语言/类别并重新验证，不要把“文件存在”当“可运行”。[R11]
- **官方 HRNet 前端已于 2026-10-25 跑通（CE-CSL 跨域）：** 独立 Python 3.8 legacy 环境（torch 1.13.1+cpu、mmcv-full 1.7.0、mmdet 2.25.3、mmpose 0.29.0）配合 `scripts/extract_slrt_hrnet_pose.py` 生成 NPZ，再用 `--pose-backend hrnet-npz`；dev-00015 全 202 帧 0 缺失、733.7 s CPU，HRNet 输出 `拒绝 不`，同视频 RTMPose 输出 `拒绝 去 不`。20 帧 63 选点平均差 3.59 px（RGB/BGR 变体 1.36 px）。这仍是 CSL-Daily 权重在 CE-CSL 上的跨域链路验证，**不得**写成 CSL-Daily 同域结果或“HRNet 更准”。[R12]
- **SLT 只补齐了数据无关资产（2026-10-25）：** base mBART 与 CSL gloss 侧（`gloss2ids.pkl`、`gloss_embeddings.bin`，2,009 行）已生成；全词表 fallback（250,027 词表、identity prune map）可让 G2T 配置运行，但**不是作者剪枝后的 `mBART_zh`**。作者文本剪枝、CSL-Daily G2T 权重和 `csl_pred.dev/.test` 仍需数据/权重；`prediction_slide` 结果的转换脚本已补。[R12]
- **TwoStream-SLR S2G 已从作者 Google Drive 包补齐（2026-10-25 晚）：** 原 SharePoint 三个链接 404；作者在 issue #107 评论给出 Google Drive 文件夹，`ckpts/TwoStreamSLT/TwoStream/csl-daily_s2g/ckpts/best.ckpt` 为 504,360,673 B、SHA256 `54d2cb65…9739`、912 键、2,004 类，已复制到 `Online/CTC_fusion/results/csl-daily_s2g/ckpts/best.ckpt`。`csl_s2g_gloss2ids.pkl` 2,004 项且与包内 `gloss2ids.pkl` 相同。CTC_fusion 还缺在线特征（需 `prediction_slide --save_fea`）。[R12]
- **作者 Google Drive 包的六类文件不要混用（2026-10-25 晚）：** `data.zip` 补 CSL-Daily keypoints（20,654 视频、`[T,133,3]` float16）与句级 `csl-daily.train/dev/test`（18,401/1,077/1,176），但**没有视频帧**；`ckpts.zip` 只取 TwoStream CSL-Daily S2G，里面的 `SingleStream/csl-daily_s2t` 是离线 S2T；`mBart_zh_t2g`/`mBart_de_t2g` 是 Spoken2Sign **Text2Gloss** 资产（vocab 6,308），不是 Online SLT 的 `mBART_zh` G2T；`EMNLP24_Online.zip` 与本地重复；`TA.zip` 是 COMP2011/2012 课程文件。[R12]
- **作者 keypoints pkl 不能直接喂本地 NPZ 入口（2026-10-25）：** `csl-daily-keypoints.pkl` 的 confidence 最大值约 1.71，超出本地 `select_keypoints` 的 [0,1] 校验；官方 `slide_csl-daily.yaml` 的 `heatmap_cfg` 未启用 threshold（Gaussian 不乘 confidence），所以官方代码不受影响。若要转成本地 `hrnet-npz`，必须显式处理 confidence（裁剪或记录超界），不能静默当 [0,1]。[R12]
- **HF `csl-daily-sentence-crop` 的 `split_*.zip` 不是 `sentence_frames-512x512.zip`（2026-10-25）：** 抽查 4 条原始 MP4（1280×1280、30fps）的帧数为 898/314/256/526，而对应标注 `num_frames` 为 146/146/200/490；用官方 keypoints 做滑窗匹配得到的最佳起点为 25/152/16/23 帧，偏移不固定。它们是**未裁剪原始视频**，不能按“取前 N 帧/居中 N 帧”直接生成官方 sentence frames；标注/ground truth 已在作者 `data.zip`，无需从 split 里找。要复现应找官方 frame zip 或带句子起止的 segment 元数据；逐视频姿态对齐只能算近似。[R12]
- **Uni-Sign 必须用作者 pose archive，不能拿 SLRT keypoints 直接转（2026-10-25）：** 用 `F:\model_weigth\data\csl-daily\csl-daily-keypoints.pkl`（HRNet 133 点、512 像素）转 `[T,133,2]/512` 后，52 条 test 子集 50/52 成功、BLEU-4 **15.16**，与上游同 52 条只有 **2/50** 逐字相同。改用作者 `csl_daily_pose_format.zip`（内部 `pose-rtmpose-192-clean`）后 52/52 成功、BLEU-4 **24.16**、chrF 21.22、平均编辑相似度 0.379，与上游 **35/52** 逐字相同。不要把 15.16 或关键点转换结果当 Uni-Sign 准确率；该 52 条是视频子集，不是完整 1,176 test。[R13]
- **SLRT 52 条近似对齐试点可跑，但只能当子集结果（2026-10-25）：** 原始视频未裁剪，用 RTMPose 全帧 63 点与官方 keypoints 做过滤后的滑窗匹配（均值 6.88 px、中位 6.31），裁剪到 `num_frames` 后 `hrnet-npz` 推理；52/52 成功、0 空输出、corpus WER **33.0%**（S94/D37/I65）。这不是官方 `sentence_frames-512x512` 协议，也不是完整 1,176 test；最差样本高 WER 时先查对齐（该样本对齐 4.94 px，属模型过生成而非裁错）。[R14]
- **Uni-Sign 657 train 拟合检查：BLEU 85.82、exact 654/657（2026-10-25）：** 用官方 pose archive 在本地视频子集的 657 条 train 上跑，BLEU-4 **85.8240**、chrF 79.62、char ROUGE 0.99979、平均编辑相似度 0.99968；同一模型完整 test 1,176 只有 BLEU **25.55**、52 子集 24.16。训练集近乎记忆，**不能把 85.82 当泛化准确率**；657 也不是完整 train（18,401）或留出集。[R15]
- **SLRT 100 条 train 拟合检查：WER 20.91%（2026-10-26）：** 从 657 train 按 signer 分层抽 100 条，RTMPose 近似对齐裁剪后 SLRT：100/100、0 空输出/错误、corpus WER **0.20914**（S88/D84/I98，参考 1,291 token），对齐均值 6.87 px、0 低置信度；test 52 条 WER 33.0%。train 低于 test 说明有拟合，但不是 Uni-Sign 式记忆；这是 100 条子集/近似对齐，不能当完整 train 或泛化准确率。[R16]
- **本地 CSL-Daily 视频覆盖已两次扩充（2026-10-26）：** 上午 10,456 条（test 627 / dev 552 / train 9,277），下午补到 **19,528 条（约 25.1 GB；test 1,086 / dev 1,004 / train 17,438）**，仍缺 **90 条 test**（偏长句：缺失平均 159.6 帧 vs 已有 125.1）与官方 `sentence_frames-512x512.zip`。引用覆盖时必须写清是哪一版；"1,086 子集"仍不是完整官方 test。[R17][R18]
- **Uni-Sign 视频端到端 vs 官方 pose 的真实差距约 −0.8 BLEU（2026-10-26 更正）：** 统一用上游参考 `test_tmp_refs.txt` 打分时，1,086 条 test 上视频端到端 BLEU-4 **24.8157**、chrF 21.80、编辑相似度 0.4157、exact 15；官方 pose **25.5846** / 22.43 / 0.4212 / 18 → **视频路径合计 −0.77 BLEU / −0.0055 编辑相似度**（627 基线 −0.87、新增 459 −0.61）。此前 R17 写的 −3.82 BLEU 是"标注文本参考 vs 上游参考"的口径错误，已更正。[R17][R18]
- **BLEU/chrF 的参考文本必须是同一套（2026-10-26 实测，新增坑）：** 官方标注 `csl-daily.test` 的 `text` 用全角 `？，。`，上游 `test_tmp_refs.txt` 用半角 `? , .`；1,086 条中 **589 条**仅这类标点不同。同一份 Uni-Sign 视频预测：上游参考下 BLEU **24.82**，标注文本参考下 BLEU **22.03（−2.78）**、chrF 21.80→19.44。**跨运行比较 BLEU/chrF 必须先固定参考文件**，也不要把某一套的绝对值与另一套的历史数字相比。[R18]
- **同域 SLRT 本地 test 结果（2026-10-26）：** 视频补充后 1,086 条 **1,086/1,086 成功、0 空输出、corpus WER 0.3018**（S1,154/D805/I522，8,220 参考 token）；627 基线 0.3032（S748/D493/I359）。对齐 mean 7.23 px / median 6.20 / P95 12.60，**12 条 >25 px**（WER 0.33–1.0）；剔除后 WER 0.2974，对齐–WER Pearson 全量 0.154、剔除后 −0.037 → **对齐不是主导误差**。仍不是官方 `sentence_frames-512x512` 协议，不能对标论文在线 29.3% / CSLR test 27.4%。[R17][R18]
- **官方 `prediction_slide.py` 旧依赖栈仍未装（2026-10-25）：** 官方 requirements 是 torch 1.9.0+cu102、transformers 4.11.3、timm 0.4.12、`Lintel`、ctcdecode、wandb；CUDA 10.2 与 RTX 5060 Ti 不兼容，`Lintel` PyPI 0.1.0 是文档检查器而不是视频解码包。legacy HRNet 环境只覆盖 mmpose/mmdet 提取，不能直接跑官方入口；需要容器/另一台机器或现代 Torch 移植。[R12]
- **工作区可能有 skip-worktree 隐藏的已跟踪文件（2026-10-25）：** `git status` 不显示，但 `slbridge/_io.py`、`slbridge/runtime_contracts.py` 等 35 个模块实际不在磁盘，运行 CLI 会报 `ModuleNotFoundError`。用 `git ls-files -v` 查 `S` 标记，对需要缺失的路径执行 `git update-index --no-skip-worktree -- <path>` 再 `git checkout HEAD -- <path>`；不要整目录 checkout 覆盖未提交修改。[R12]

## 11. P-1 实验执行（2026-10-26）新增事实与约束

- **P-1 基础设施位置（2026-10-26）：** `experiments/p1/` 是计划包的可执行实现与证据库（GT-free manifest、EVAL-ONLY 参考、canonical 评测器、run 目录、GT guard、汇总 `summary.json`）。P-1 执行结论见 `docs/experiments/P-1/P1_EXECUTION_REPORT.md` 与 `docs/reports/p1-experiments-2026-10-26.md`；E00–E06 均已落地，BLOCKED 项在报告中如实记录。[R19]
- **canonical 指标器已冻结并回归：** `experiments/p1/eval`（sacreBLEU 13a + 上游 mT5 逐字加空格；chrF 原始串；token WER）对历史 1086 快照重算必须得到 BLEU-4 **24.8157**、chrF **21.7974**；`pytest experiments/p1/tests/test_p1_infra.py` 用该断言做回归（2026-10-26 实跑 12 passed）。新的评测不要手写 BLEU，先复用该模块。
- **E01 文本层可复现性成立（2026-10-26 实测）：** 同 checkpoint/同解码/同冻结输入下，Pilot-50 三路输出与历史快照 **100% 逐字一致**（视频 50/50、官方 pose 50/50、SLRT gloss 50/50）；条件是使用 `artifacts/csldaily-video-test-627-20261026/trimmed/` 与 `pose/` 的同一批冻结片段。新运行必须写到 `experiments/p1/runs/`，**禁止覆盖历史 artifacts**。S1 只重跑 infer（align 复用已审计产物），不能写成"重新提取 pose"。[R19]
- **Pilot-50 分数不能与 1086 全量比较：** 同协议下 Pilot-50 BLEU-4 视频 **31.2308** / 官方 pose **31.7823**（历史 1086 为 24.8157/25.5846）；这是 50 条诊断子集的样本差，不是模型提升。视频 − pose = −0.5515 BLEU，与全量 −0.77 方向一致。
- **参考口径在同一子集上的实测差：** 同一份 Pilot-50 视频预测，官方全角参考 BLEU **28.10** vs 上游半角参考 **31.23**（差 **3.13**）；比 1,086 全量的 ~2.8 更大。跨运行只允许固定 `gt_1086.ref_sentence_upstream`。
- **Uni-Sign 没有可用分数（仍 BLOCKED）：** runtime `TextPrediction(score=None)`，历史 predictions JSONL 无 score/logprob；E03 的 Uni-Sign 置信度必须标 BLOCKED。**不得用编辑相似度/文本稳定性冒充模型置信度**；要解锁需增加 `output_scores` 重跑。
- **SLRT 分数语义：** `uncalibrated_mean_ensemble_probability`（对象=gloss token）不是 P(correct)：直当概率 ECE **0.304**；OOF Platt 后 AUC 0.745 / ECE 0.026；阈值必须按 signer 分组 OOF。FDR≤15% 策略 error-ACCEPT 0.153、missed-upgrade 0.674。不同模型不能共用阈值。[R19]
- **B0（SLRT Gloss 确定性 join）不是翻译器：** 1086 条 BLEU-4 仅 **12.22**；char 编辑相似度会奖励 gloss token 覆盖，含 B 的 oracle（27.295）偏乐观，A/A2 句子级 oracle 才是 26.598。不要用 B 的 edit sim 当语义质量，也不要把 B0 当"Gloss2Text 已接入"。
- **A/A2 高度相关：** 1086 条中 **507 条文本完全相同**；离线 oracle 上限 +1.78 BLEU，但规则级联只能兑现 +0.17（A2 调用率 38.1%、错停优胜 5.2%、纠错 0.7%）。在没有可靠无 GT 信号（Uni-Sign 分数 BLOCKED、LLM/Jev 无端点）前，不启用自动融合/自动接受。[R19]
- **LLM/Jev 分支不可测：** 本机无任何 LLM 端点/密钥（环境变量名核对且未打印值），repo 无 provider 客户端；D/E/F/G、LLM-only、Rule→LLM 全部 BLOCKED。解锁需要固定端点+模型版本+prompt 版本+预算。[R19]
- **SLRT CLI 单条固定开销：** 每条样本子进程重载 ~3.9 s 权重（499 MB）；批量评测/延迟报告必须扣除或单列，不能当纯推理时延；批量要分片并防并发 OOM。[R19]
- **P-1 新增的实时约束（源码审计）：** Uni-Sign 对称时间卷积（kernel=5×3 层，±6 帧）+ 双向 mT5 encoder → 需要未来帧、无原生 Partial；上游 `online_inference.py` 是"读视频路径整段推理"。SLRT 原生 partial 需右上下文 8 帧 + 投票等待 3 窗（约 11 帧 lookahead）。不得把整句离线延迟写成实时首稳时延。[R19]
- **P-1 性能实测（2026-10-26，RTX 5060 Ti 16 GB，桌面基线上、单进程串行）：** 热启动 30 次 e2e P50/P95：Uni-Sign pose **277.6/344.3 ms**（RTF 0.078）、Uni-Sign video **2,690.4/3,283.9 ms**（RTF 0.748，pose 前端占 80%）、SLRT hrnet-npz **9,865.1/11,189.7 ms**（RTF 2.53，≈90 ms/窗，其中 S3D forward 3.7 s、逐窗 CPU 热力图预处理 ~5.6 s）。冷启动加载 P50：pose 17.7 s / video 26.6 s / SLRT 3.1 s。**两进程并行**总墙钟 −15%（40.28→34.10 s）但单条延迟变差 1.18–1.81×、整卡峰值 6.45 GB、GPU util 仅 27.8% → 实时默认串行，并行只适合离线批处理。**引用时以 `experiments/p1/e04/metrics.json` 为准**（报告曾四舍五入到 276/2679/9775，已按实测对齐）。[R19]
- **P-1 前缀探针（2026-10-26）：** Uni-Sign pose 路径 10 条 × 8 个已观测前缀 80/80 成功，`first_stable_prefix` mean **90% / median 100%**（6/10 句只有 100% 前缀才出现最终文本）、相邻前缀改写率 **67%**、空输出 0、与作者快照 10/10 一致；video 路径 40/40、最短可用前缀 20 帧。**输出是整句重估，不是增量提交**；这些是冻结前缀稳定性，不是 `first_stable_latency`。[R19]
- **P-1 GPU 性能测量的环境坑（2026-10-26）：** 32 GB Windows 上桌面/agent 进程造成 host commit 压力（committed 55.7/65.2 GB、free ~4.5 GB）时，GPU 空闲 >13 GB 也会在 mT5 warmup 报 `CUDA error: out of memory`；必须按分配类错误重试（本次 2 次后 3/3 成功）并在 trace 记 `spawn_attempts`，不得写成模型失败。性能数字要记录整卡基线（本次 2.87 GB）与"非干净机房"条件；并行前必须设 RAM 门（≥7 GB available）。[R19]
- **推理型 LLM 的 max_tokens 截断坑（2026-10-26 实测，G2）：** `deepseek-v4.1-flash`（wb-proxy，OpenAI chat-completions 兼容）在输出最终句子前会消耗内部推理 token；`max_tokens=256` 时 **73/150 次返回空 completion**（HTTP 200、`finish_reason=length`、completion 顶满），且**按组分布不均**（C 8/D 34/E 31）——若只统计异常、不统计 `status=error`，会得到"选择性偏差"的假指标。修正：`max_tokens≥2048`（本轮 2048 后仅 3/150 截断），并把 `finish_reason`/空 completion 计入失败数。**不要用 256 之类的小预算跑中文长句生成。**
- **G2 首轮真实结论（2026-10-26，50 条 pilot 探索批，wb-proxy/deepseek-v4.1-flash，temp 0）：** 无条件"每句调 LLM 重写"**显著劣于原始 Uni-Sign**：A BLEU 31.2308 vs C 26.6838（ΔBLEU −4.55，95% CI −8.02…−1.38）、Δedit −0.033（CI 不含 0）。D−C（Gloss 新增价值）ΔBLEU −0.04（CI 含 0）、Δedit +0.026（边缘）；E−D（证据约束）ΔBLEU +1.62（CI 含 0）、Δedit +0.011（不显著）；但 E 相对 C 的 edit **+0.040（CI +0.0019…+0.0821，显著）**。成本：150 次调用 113,653 tokens、P50 3.16 s/P95 8.89 s。**结论：当前 prompt 下 LLM 融合不应进主链路；需 prompt v2（最小修正/冲突才改）+ 选择性级联（默认 A）再评估**，最终仍需新独立集+人评。[R19]
- **SLRT replay 真流探针（2026-10-26 两轮独立重复）：** 按原时间轴回放 10 条 pilot（hrnet-npz）的实测 **RTF 2.23–2.58**、吞吐 **7.5–7.9 窗/秒**（30 fps 需 ~30 窗/秒）→ 当前实现**不能实时**；首个 final token 墙钟 P50 **3.7–3.9 s**（P95 5.1–7.1 s），队首→交付仅 ~50 ms，瓶颈是帧到达→worker 取走 P50 1.7 s + 单窗推理 57–88 ms > 33 ms 帧间隔；capture 丢帧 52–61 引起 19–24 次 partial retraction。黑帧无人门控有效（108/108 拒绝），**水平镜像未被门控**（需应用层处理）。11 帧 lookahead 推导实测验证（最早投票窗口 anchor=3，enabling=11 帧）——0.37 s 是算法下限，实测首 partial 2.0–2.2 s 还含启动与队列积压。[R19]
- **P-1 执行的最大风险仍是 Git 脏树：** 2026-10-26 执行时 HEAD `cf291c2a…` + 132 项脏工作区（`slbridge/slrt`、实验脚本、P-1 全部未提交）。所有 run manifest 记录 dirty tree；清理/提交前先复核。

## 12. 阶段 4 预处理优化实测（2026-10-27）——两个域依赖结论
- **`pose_detect_stride=5` 在 CSL-Daily 上是质量退化，不是免费加速（2026-10-27 实测）：** CSL-Daily test pilot-50 上 e2e P50 3086→1964 ms（−36.4%）、pose −40%，但 BLEU 31.23→29.54（−1.69）、Δedit **−0.0205**（95% CI [−0.0563, +0.0153]）、仅 32/50 预测相同 → **质量门槛 FAIL，不建议在该域默认开启**。此前 CSL-News 300 条与 CE-CSL 500 条上 stride=5 与 stride=1 无显著差异（CI 含 0），**但两个数据域的结论不能外推**：每换一个域必须做配对质量门槛（建议 Δedit ≥ −0.005 且 CI 下界 > −0.01），并同步更新 pose protocol 身份与方向重校准。生产默认值未改。
- **CSL-Daily test 只有 1.1% 片段 >256 帧（12/1086）：** 因此 `pose_sample_only` 在该域几乎无收益——pilot-50 全部 ≤172 帧、0 跳帧、50/50 预测完全相同（no-op）；在全部 12 条长片段上跳 13.2% 帧、12/12 预测逐字相同、e2e −10.6%。结论：可安全开启（零质量变化），但**不能宣称"CSL-Daily 提速 10%"**（域级平均约 0.1%）。
- **SLRT 逐窗预处理瓶颈与优化（2026-10-27 修正 E04 旧措辞）：** 旧说"~5.6 s/条是 CPU 热力图预处理"不准确——热力图在 **GPU** 上算；实测拆分（20 条/2406 窗口）：canonical_frame（PIL 512→320，14.3 ms/帧）28.0%、prepare_inputs 有效 23.6%（约一半 GPU 异步、CPU 计时不可见）、S3D 前向 39.5%、解码 4.7%。**stride=1 时同一帧的 RGB+热力图被重复计算约 16 次**；用"逐帧调用原始未改函数 + 内容 hash 缓存 + cat 组装"（仅 benchmark 进程内替换）得到 RGB/heatmap **max abs diff 0.0 / 0.0、torch.equal True/True**（8 真实窗口 cuda:0）、Gloss **20/20 逐字相同**、corpus WER **0.27007 不变**，e2e P50 **5923→4836 ms（−18.4%）**、P95 −16.4%、prepare 有效总计 −70.3%。合入生产需落到 `SLRTRuntime`/`SLRTStream` 帧级缓存并更新 protocol；下一瓶颈是 canonical_frame 与 S3D。
- **优化对照必须同协议逐位核验：** 任何"只改性能不改语义"的优化都要给出等价性证据（张量 max abs diff / `torch.equal` / 输出逐字一致 / WER 不变），否则只能算另一条算法分支，不能写进默认路径。

## 13. Uni-Sign 1012 键权重 / RGB+pose / rtmpose 接入（2026-10-27）

- **`csl_daily_rgb_pose_slt.pth` 的 RGB 分支数学上恒等消失，不要把它当 RGB+pose 用：** 该 checkpoint 的 `fusion_gate` 两个 Conv 是 zero-init 且权重全零（`fusion_gate.0.weight` 0/131072、`fusion_gate.2.weight` 0/256、bias 全零），上游公式 `fused_post = gate*fused + (1-gate)*gcn`（`models.py` L192-194）⇒ `gate≡0`，输出恒等于 pose 分支。实测：给真实手部 crop（squeeze，left 10.3/right 11.7 帧/样本）与全零支持**逐条完全一致（1086/1086, Δ=0）**。它的 +0.66 BLEU（1086，26.3026 vs A2 25.5846）来自 **stage2 的 pose/mT5 权重**（与 pose-only checkpoint 相比 41.6% 参数超 1e-6 容差），与 RGB 无关；"复现论文 26.36"是巧合。**结论：按"更优 pose-only 模型"使用（627 键子集 + pose-only 架构），不做手部裁剪；真实 RGB+pose 需作者非零 gate 权重。**
- **官方 pose pkl 的 `scores` 是 `(T,1,133)`，手部采样索引要带中间维：** `confs[:, 91:112]` 在 3 维输入上取到空 → 手部帧 `None` → RGB 支持全零。正确写法 `confs[:, 0, 91:112]`（或先 squeeze）。任何"手部可用性"断言都应检查 `hand_frames_left/right > 0`，而不是只看"没报错"。
- **pose-only adapter 的 checkpoint 校验曾是硬编码常量，导致替代 checkpoint 无法加载：** `verify_checkpoint()` 用模块级 `UNISIGN_CHECKPOINT_BYTES/SHA256`；必须改为按 bundle manifest 声明的 `checkpoint_bytes`/`checkpoint_sha256` 走 `verify_file()`（2026-10-27 已修，`slbridge/uni_sign.py`）。否则 stage2 子集（1.95 GB）必然 `size_mismatch`。
- **rtmpose 生产路径接入三个硬要求（2026-10-27 实测）：** ①`RtmposeExtractor(backend=...)` 必须是 `"onnxruntime"`（值原样传给 `rtmlib.Wholebody`）；②`cache_dir` 必须复用 `models/external/uni_sign/rtmlib-cache`——rtmlib 走 `TORCH_HOME/hub/checkpoints`，换空目录会触发**联网下载**；③cv2 读到 EOF 返回 `(False, None)`，帧尺寸要在解码循环内取，否则 `frame.shape` 抛 AttributeError（rtmpose 路径曾因此任何真实视频都跑不通）。
- **归一化口径要跟 A 路径对齐：** rtmpose 分支用逐轴 `x/W, y/H` 与 A 路径 `isotropic_short_side` 数值不同（CSL-Daily 是 1280×1280 正方形所以看不出来，**非方形/摄像头输入才会暴露**）；必须复用 `normalize_rtmpose_pixels(mode="isotropic_short_side")` 并在 metadata 记 `pose_normalization` + 版本。
- **"同一权重不同输入协议"的等价性要靠逐字比对证明：** stage2 子集在 pose-only 架构上与 legacy RGB+pose 1086/1086 逐字一致，才允许说"RGB 分支可去掉"；单样本一致不算（采样/窗口可能不同）。

## 14. Round-3 双评审仲裁清单的协议差异（2026-10-27）

- **状态：已修复（2026-10-27 修正 `arbitration_rows` 为协议口径，并同时输出 `material_disagreement_rows`；Round-3 双评已实跑：协议口径 36/90=40%、实质分歧 8/90=8.9%）。** 触发条件：Round-3 的 A/B 两位评审评分仅差 1 分，且严重语义错误判断相同。Round-3 README（本地产物：`experiments/p1/e06/human_review/round3/README.md`） 和模板填写说明要求对评分或严重错误的任意分歧仲裁，但 `make_round3_package.py`（本地产物：`experiments/p1/e06/human_review/round3/make_round3_package.py`） 的 `agreement.arbitration_rows` 仅筛选 `abs(score_A-score_B) >= 2` 或严重错误不同，因而会漏掉该类 1 分分歧。本次只完成 AI 评审人 A，未读取 B 评审意见，不能据此声称一致性或仲裁已通过。
- 避免办法：B 到齐后，按协议额外以“评分不同或严重错误不同”逐行扫描，并由第三人或用户裁定；不能把现有 `arbitration_rows` 当成全部协议分歧。独立记录 AI/人工身份、量表版本与已读历史文本，不能将同一 AI 的两次输出计为两名独立人工评审。当前单 A 结果与边界见 Round-3 AI A 评审摘要（本地产物：`outputs/01a17b59-e3b8-7590-b21b-250b3f41ca6f/round3/round3_reviewer_A_ai_summary.md`）。

## 15. 三维动画生成路线的能力边界（2026-10-28）

- **待验证：动作质量批准不等于语言认证。** 触发：将上游 SignBridge 的资源启用/审批当成“已审核标准 CSL 词库”。证据：固定 SHA `3503ff7802bc991ccf5e0753c2fe41b79689b860` 的 100 段公开快照中，83 段拟合资产的批准为 `motion_quality_only`，manifest 明示 `semantic_accuracy_certified=false`；片段数也不是标准词条数。避免：分别记录语言、动作与来源审核，绑定具体资产和重定向版本；语言验收 **NOT_RUN**。[R20]
- **待验证：SMPL-X 重定向已有原型，但非手控表达未完整实现。** 触发：把上游播放器可播放理解成完整 CSL 连续生成。证据：上游已有 40 骨骼映射，但 jaw/eyes 静止、表情系数未重定向；片段顺播和四元数插值没有证明词间接触、空间指代与语法正确。避免：完整短语优先，单词与新拼接句分别验收；不把源码审计写成运行通过。[R20]
- **待验证：同为 133 的协议不可互换。** 触发：复用 Uni-Sign 的 RTMPose 数据来直接驱动 SOKE/SMPL-X。证据：前者为 `T×133×3` 的二维点与 confidence；SOKE 论文的 133 维是人体动作参数表示。避免：显式绑定 feature schema、旋转/坐标/时基与重定向配置，禁止凭维度 reshape 或把置信度当深度。[R20]
- **待验证：生成候选与跨语言资源须另行审核。** 触发：把 SignAvatars、VLibras 或 SOKE 的公开资料视为通用 CSL/Libras 动作授权与语言保证。证据：各自语种、数据和许可不同；SOKE 的任务 checkpoint 完整性、MVSign 的受限访问以及本地运行均未验证。避免：保留独立语法/词典和原始识别证据，未知词/否定缺项不能静默跳过后自动播放。以上为调研约束，没有新增实验成绩或验收通过结论。[R20]
- **已核验结构，驱动效果待验证（2026-10-28）：脸部静止不能直接归因于模型没有表情结构。** 本机 Godette GLB 没有 morph targets，但包含带非零顶点权重的嘴/眼睑/眉部骨骼；当前重定向只取约 40 根头颈/上肢/双手骨骼，脸部保持静止。先验证现有骨骼驱动和局部变形，再决定用 Blender 修权重、补 Shape Keys 或换角色。骨骼/权重存在不等于表情控制或 CSL 非手控语法通过；本轮只抽查旧演示两个片段的浏览器播放和一个暂停帧。[R22]
- **待实现（2026-10-28）：动画确认不能复用程序终态冒充人工批准。** `CaptionLedger.committed` 与 job `subtitle.final` 是程序提交/任务终态；当前没有人工 confirm/play 操作。动画计划需另存明确输入确认、来源会话与播放代次，识别取消/来源失效不等于已有 Three.js 播放器会自动停止。集成方案已另写草案，本轮未新增这些产品能力。[R22]

- **已核验源码，质量改善待验证（2026-11-06）：显示稳定不能当作手形/接触修复。** 触发：把动画工作台的“显示优化”或提高渲染帧率称为动作质量已解决。证据：动画功能分支 `codex/csl-animation-workspace` 的 `86e0d45…` 中，`stage.js:118-119` 共用同一原始重定向轨，再调用孤立尖峰修正，阈值为残差 2°、邻帧差 12°、最多修正 3°；`retarget.js:176` 仍保留面部静止。避免：分别对照源视频、源姿态和目标角色，校准手形/掌向与接触，再检查原速、慢速和人物导出；单帧截图不能证明全片自然度。当前聊天主检出仍是 `782dee3…`，不能将功能分支能力写成主检出已集成；本轮没有新播放、性能或语言通过结论。[R23]
- **待验证（2026-11-06）：开源实现器的两种模式不能自动补齐现有姿态的非手控表达。** 触发：将 Performs 的关键帧模式与 BML/SiGML 脚本模式理解为任意 SMPL-X 输入都能自动获得正确表情/接触。证据：Performs 固定提交 `dfe48af8bc0213df0dae73929b0d69538e54b785` 的 README、API 和 KeyframeApp/ScriptApp 区分两条路径；SignON-realizer README 已指向该后继项目。避免：先在少量完整 CSL 短句验证角色配置、格式转换、通道混合及播放控制；未知表情不得由微笑/眨眼冒充语法，Libras/NGT 词典不得作为 CSL 词典。新角色、新重定向和语言效果均需各自复核。[R23]

- **已核实数量，动作质量待验证（2026-11-06）：句级样本、文本去重和可发布动作数量分开。** 触发：将 SOKE CSL-Daily 的样本数当成不同中文句子数，或将全部发布索引记为通过质量审核。证据：实查 `csl_clean.train/val/test` 为 18,401/1,077/1,176 个样本，共 20,654；精确 text 全局去重为 7,376，本地同名视频匹配 19,528。训练 text 去重 6,578 与原始论文 6,598 不同，保留统计口径差异。避免：以具体发布索引和 SHA 记录样本规模，跨划分取全集去重，不直接相加各划分文本数；视频同名不当作三维整包完整、角色兼容或语言审核通过。[R31]

## 词级组合调研补充（2026-11-07）

- **已核验源码，接入待验证：** 触发：把 `SignSparK_Retrieval` 的非空输出／`status=ok` 当作完整保义生成。证据：固定提交 `b1ed92705cafd2c9ff0c72a89df14b0a9ee62258` 的 `oov.py`／`pipeline.py` 可按模糊匹配或语义近邻替词，未解决的词被丢弃；只要还有动作片段仍可能标记 ok。避免：显式核对规划与 `used_glosses`、mapped、dropped；缺词阻断或人工核准改写，不能以更高输出覆盖冒充语义正确。本轮没有修改或运行该外部实现。
- **已核实发布合同，映射待验证：** 触发：将公开 `csl_iso.train` 的 start/end 直接用于现有 SOKE 参数。证据：作者模型卡明确它们是对应 SignSparK LMDB clip 的 `[start,end)` 帧索引；本机 SOKE 到原片／派生 LMDB 的精确映射尚未验证。避免：先在配套 LMDB 验证，跨包切片绑定 ID、裁剪、采样和帧映射；CTC 峰不能当完整词边界，FAST 段数不能强配 Gloss 数。
- **已核验源码，独立泛化验收未运行：** 触发：把 `best_case` 检索或 dense 关键帧保留当作新句语言通过。证据：`dictionary.py` 的 best_case 可取测试句自身动作；`signspark.py` 的 dense merge 贴回词内保留帧，但不证明接缝、空间指代、接触或表情正确。避免：正式词库限 train，规范化整句去重的组合留出集另建；语言理解／自然度保留 NOT_RUN。

来源、外部一手链接和本次只读 train／全库统计见词级动作组合调研（本地产物：`.survey/csl-word-composition-2026-11-07/solutions.md`）及其 context（本地产物：`.survey/csl-word-composition-2026-11-07/context.md`）。这里记录研究约束，不追加模型或语言验收成绩。

## 截至核验日期的证据锚点

三维与路径补充（2026-10-28，原实验成绩不变）：

- **已核验定位，运行接线待验证：** `D:\Zenodo` 已不存在，实际数据根是 `G:\Archive\Zenodo`；CE-CSL 5,988 条预期视频路径全存在，CSL-Daily 原 1,086 样本在新根有同名文件。TFNet 来源权重与一个代表视频的 SHA256 匹配旧证据，但没有整库迁移内容认证。新运行显式解析 G 盘路径，保留冻结 manifest 中的历史路径与裁剪视频摘要，不能全仓替换 D 路径或把同名原片当作已匹配裁剪哈希。`G:\ML\temp_feture\ps2_matrix` 属工业过程实验，不能按其中模型名推断为手语权重。[R21]
- **已核验资源，播放/语言验收待验证：** 本机 SignBridge HEAD `7ab6de38…` 的普通 dist 研究资源是 `unavailable` 占位；但 `docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip` 保存完整 100 段动作。归档 SHA/CRC、143 文件清单、100 段姿态结构/有限值与 83 段资源摘要通过本轮静态核验；这不代表现场播放或语义正确。优先从完整包验证现有动作，避免仅因 dist 为空重复训练/重建；本机未找到上游文档引用的私有 `local-data/`、SOKE/SMPLer-X 依赖或完整训练产物。不要把远端源码能力、旧训练报告和本机可用文件混写。[R21]

以下是历史运行记录，不是后续任务可直接复用的“当前通过”声明。

| 范围 | 已记录结果 | 边界 |
|---|---|---|
| 几何与方向修复后的 test 500 条 | BLEU-ZH 7.9934 → 12.3019；字符编辑相似度 22.5123% → 30.6325% | 历史姿态缓存重新解码，已检查 test；不是 500 条重新提取 |
| dev 方向校准 | 36 条校准、24 条不重叠验证；验证 BLEU 6.2325 → 10.3210 | 同签者不同视频；方向锁定后不回调 |
| 新 dev 裁剪诊断 | BLEU 10.3210 → 10.7150，编辑相似度 31.1895% → 30.5079% | 24 条重新提取；收益混合，非人工真值边界 |
| 新 dev 前缀诊断 | 70 次转换中 58 次改写；24/24 旧帧归一化变化 | 稳定性诊断，不是逐词准确率 |
| 窗口修复测试 | 定向 42 passed；全量 797 passed、10 skipped | 9 项缺 Torch 的 TFNet 测试、1 项未配置 live 输入跳过 |
| 实时／语义验收 | 指定真实会话 0 候选；现场摄像头、真人语义、实时 P95 为 NOT_RUN | 软件回归通过不能补写这些结果 |
| CSL-News 准确率（2026-10-25） | 项目集成 CSL-Daily 权重：BLEU-zh 1.27、编辑相似度 7.90%（379 条配对）；stage1 预训练权重：BLEU-zh 16.75（本地 pose）/ 24.67（上游 pose）；全量 1694 条 stage1+上游 pose：BLEU-zh 23.76、编辑相似度 30.07%、exact 8/1694 | 本地 1694 条在上游 train 段，23.76 是训练集拟合；四格消融只覆盖 379 条配对样本；无 CSL-News 官方留出集视频 |
| CSL-News 延迟（2026-10-25） | 端到端 5.54 s/条（P50 5.27、P95 9.42），RTF 0.643，10.78 条/分、47 帧/秒；姿态来自磁盘时 0.409 s/条 | 单进程独占 GPU、bf16、beam=4；并发、多实例与摄像头路径未测 |
| CSL-Daily 本地 test 视频 1,086 条（2026-10-26） | Uni-Sign 视频端到端 BLEU-4 24.8157 / chrF 21.80 / 编辑相似度 0.4157 / exact 15（627 基线 24.9485 / 8）；同一批官方 pose BLEU-4 25.5846 / exact 18；SLRT corpus WER 0.3018（S1,154/D805/I522），对齐 mean 7.23 px | 1,086/1,176 子集（缺 90 条偏长句视频）；SLRT 为近似句段裁剪；Uni-Sign 视频路径非论文协议；统一上游参考打分，标注参考版本仅审计 |
| P-1 执行（2026-10-26） | E00–E06 全部落地：E01 三路 50/50 逐字复现；E02 A/A2/B0 BLEU 24.8157/25.5846/12.2183、A/A2 oracle 26.598；E03 SLRT 风险 AUC 0.745 / raw ECE 0.304；E06 规则级联 +0.17 BLEU、A2 调用率 38.1%、错停优胜 5.2% | 1,086 为 retrospective_seen；E02/E03/E06 模型文本 imported；路由/校准 offline；人评 NOT_RUN；LLM/Jev BLOCKED；SLT/G2T 权重缺失 |

## 证据索引与维护方式

- [R1：初始差距诊断](../reports/unisign-cecsl-gap-diagnosis-2026-10-24.md)：几何、方向敏感性、训练文本覆盖和历史裁剪错误；以其日期理解后续已修复项。
- [R2：方向修复与评测完整性](../reports/unisign-cecsl-direction-repair-2026-10-24.md)：方向校准、缓存身份、500 条回归及真实文件入口验证。
- [R3：后续准确率与实时链路核查](../reports/unisign-followup-accuracy-streaming-2026-10-24.md)：本轮窗口修复、24 条新诊断、797 项通过与真实会话未交付。
- [R4：流式源码审计](../reports/unisign-streaming-source-audit-2026-10-24.md)：时序卷积、mT5、缓存、输出合同和 LLM 状态的源码行号。
- [R5：初始接入与环境记录](../reports/unisign-integration-2026-10-23.md)：Windows 加载顺序与环境兼容；旧分数、benchmark 和未验收项按当时条件解释。
- R6：最近交付核验 JSON（本地产物：`artifacts/unisign-cecsl-eval/followup-20261024/delivery-verification.json`）：输入和产物哈希、测试摘要及明确的未通过／NOT_RUN 状态。
- [R7：SLRT 集成交付报告](../reports/slrt-integration-2026-10-24.md)：权重严格加载、官方数值对照零差异、真实 MP4 Gloss 输出与吞吐实测、摄像头实时交付未通过及未验证边界。
- [R8：SLRT 数据集实测](../reports/slrt-dataset-eval-2026-10-24.md)：CE-CSL test 500 条历史 WER 0.8918、镜像 36 条输入错误及结论撤回、吞吐与实时回放、CSL-Daily 视频缺失边界。
- [R9：模型输入兼容性核验](../reports/model-input-compatibility-audit-2026-10-24.md)：实际输入张量、HRNet替代前端、宽屏裁剪证据、镜像与评测修复、训练数据来源及未验证对照。
- [R10：CSL-News 准确率与延迟验证](../reports/unisign-cslnews-eval-2026-10-25.md)：本地子集=上游 train 段、无 CSL-News 下游权重、权重×姿态四格消融、前端差异量化收益、延迟归因到逐帧 RTMPose。
- [R11：SLRT Online 复现差距核查](../reports/slrt-online-reproduction-gap-2026-10-25.md)：权重/数据/配置/环境逐项差距、官方 `best.ckpt` 路径与本地命名差异、HRNet 权重已存在的更正、CSL-Daily SLT 缺失及最小补齐路径。
- [R12：SLRT Online 非数据资产补齐](../reports/slrt-assets-bootstrap-2026-10-25.md)：官方目录硬链接、legacy OpenMMLab 环境、HRNet NPZ 提取与跨域验证、K400/base mBART/gloss 资产、SLT 转换、作者 Google Drive 包（CSL-Daily keypoints/句级 meta/TwoStream S2G）及剩余视频帧/G2T 缺口。
- [R13：Uni-Sign 52 条 CSL-Daily test 子集](../reports/unisign-csldaily-52test-2026-10-25.md)：官方 pose archive 输入、52/52、BLEU-4 24.16、与上游 35/52 逐字相同；及 HRNet keypoints 直转被否决的 15.16 对照。
- [R14：SLRT 52 条 CSL-Daily test 近似对齐试点](../reports/slrt-csldaily-52test-2026-10-25.md)：RTMPose 对齐裁剪、52/52、corpus WER 33.0%、对齐均值 6.88 px；非官方 frame 协议与完整 test。
- [R15：Uni-Sign 657 条 train 拟合检查](../reports/unisign-csldaily-train657-2026-10-25.md)：BLEU 85.82、exact 654/657、编辑相似度 0.9997；与完整 test 25.55 对照，证明训练拟合而非泛化。
- [R16：SLRT 100 条 train 拟合检查](../reports/slrt-csldaily-train100-2026-10-25.md)：分层抽样、100/100、WER 20.91%、对齐均值 6.87 px；test 52 条 WER 33.0%，属拟合检查而非泛化。
- [R17：CSL-Daily 视频端到端 SLRT vs Uni-Sign（627 条 test）](../reports/csldaily-video-slrt-unisign-2026-10-26.md)：本地 10,456 条视频与 test 相交 627 条；同一裁剪片段上 Uni-Sign 视频 / 官方 pose、SLRT WER 30.32%；逐条 GT+两模型预测合并表、Excel、子群体统计与 OOM 重试记录。**数字已按统一上游参考更正（视频 24.95 vs pose 25.81），原 −3.82 口径已废。**
- [R18：CSL-Daily 视频补充测试（1,086 条）与参考口径更正](../reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md)：视频扩到 19,528 条、test 覆盖 1,086；新增 459 条补跑；统一上游参考下视频 24.82 vs 官方 pose 25.58（−0.77）、SLRT WER 30.18%；记录全角/半角参考文本造成的 ~2.8 BLEU 陷阱。
- [R19：P-1 实验计划执行（E00–E06，2026-10-26）](../reports/p1-experiments-2026-10-26.md)：Pilot-50 三路 100% 复现、E02 同任务基线/oracle、E03 SLRT 校准、E04 GPU 性能、E05 前缀与未来帧审计、E06 Rule-only 级联；G0–G4 与 ADR。详细版：`docs/experiments/P-1/P1_EXECUTION_REPORT.md`，基础设施与证据：`experiments/p1/`。

- [R20：CSL 三维动画路线评审与一手来源调研（2026-10-28）](../research/csl-3d-animation-route-review-2026-10-28.md)：SignBridge 固定源码/100 段资源边界、VLibras 分层、SOKE/SignAvatars/SGNify/MVSign、动作协议、许可标识、阶段与语言验收设计；运行/真人理解/端到端集成均 NOT_RUN。附 [SignBridge/VLibras 源码审计](../research/csl-animation-source-audit-2026-10-28.md)。
- [R21：现有成果与 G 盘复用核验（2026-10-28）](../research/csl-animation-existing-assets-reuse-2026-10-28.md)：本机完整动作归档、候选资源目录、识别权重与数据迁移身份、D/G 路径边界；附 G 盘详细盘点和机器可读资源清单。本轮仅静态核验，模型推理/动作播放/语言理解均 NOT_RUN。
- [R22：动画集成与画面/动作升级设计草案（2026-10-28）](../research/csl-animation-integration-and-visual-upgrade-design-2026-10-28.md)：本机 GLB/蒙皮、旧版少量实际浏览器播放、Blender 离线职责与三条工程路线；附集成代码审计。旧版能显示和播放不代表新集成、所有片段、性能或语义验收通过。
- R23：动画动作与人物画面优化调研（2026-11-06）（本地产物：`.survey/csl-animation-quality-2026-11-06/solutions.md`）：动画功能分支的显示/重定向边界、两张历史截图、VLibras/WikiLibras、Performs、MMS 与 Spoken2Sign 一手资料比较；给出小批完整 CSL 短句与目标角色共同修复的建议，新播放/性能/语言验收均 NOT_RUN。
- [R31：SOKE CSL-Daily 句级资源数量核查（2026-11-06）](../research/soke-csl-daily-sentence-count-2026-11-06.md)：发布三划分索引的样本/文本去重、本地视频文件名匹配和原始论文统计的区别；整包三维、播放与语言审核均未完成。

后续维护时在对应主题更新“触发条件 → 证据 → 避免办法”，给已解决项保留回归入口。新运行结果写新报告并在这里更新日期和索引；保留旧产物。完成条件是相关源码事实与证据边界都已核验，文档链接可用，未知事项有明确状态。

[R1]: ../reports/unisign-cecsl-gap-diagnosis-2026-10-24.md
[R2]: ../reports/unisign-cecsl-direction-repair-2026-10-24.md
[R3]: ../reports/unisign-followup-accuracy-streaming-2026-10-24.md
[R4]: ../reports/unisign-streaming-source-audit-2026-10-24.md
[R5]: ../reports/unisign-integration-2026-10-23.md
[R7]: ../reports/slrt-integration-2026-10-24.md
[R8]: ../reports/slrt-dataset-eval-2026-10-24.md
[R9]: ../reports/model-input-compatibility-audit-2026-10-24.md
[R10]: ../reports/unisign-cslnews-eval-2026-10-25.md
[R11]: ../reports/slrt-online-reproduction-gap-2026-10-25.md
[R12]: ../reports/slrt-assets-bootstrap-2026-10-25.md
[R13]: ../reports/unisign-csldaily-52test-2026-10-25.md
[R14]: ../reports/slrt-csldaily-52test-2026-10-25.md
[R15]: ../reports/unisign-csldaily-train657-2026-10-25.md
[R16]: ../reports/slrt-csldaily-train100-2026-10-25.md
[R17]: ../reports/csldaily-video-slrt-unisign-2026-10-26.md
[R18]: ../reports/csldaily-video-slrt-unisign-full-1086-2026-10-26.md
[R19]: ../reports/p1-experiments-2026-10-26.md
[R20]: ../research/csl-3d-animation-route-review-2026-10-28.md
[R21]: ../research/csl-animation-existing-assets-reuse-2026-10-28.md
[R22]: ../research/csl-animation-integration-and-visual-upgrade-design-2026-10-28.md
[R23]: ../../.survey/csl-animation-quality-2026-11-06/solutions.md
[R31]: ../research/soke-csl-daily-sentence-count-2026-11-06.md






## 基本场景接续验证补充（2026-11-08）

- **已实现工程预览，语言验收未运行：** 触发：将12类精选96原句或关键词候选数当成基本需求全面覆盖。证据：独立场景入口8879已验证96条源参数、6条保留源帧的多句组合、12场景各一条真实人物播放；固定96需求严格匹配仅6条，90条缺资源。避免：分别记录精选素材数、关键词候选、固定需求覆盖与真人理解；句内切词和新句翻译不由句界接续冒充。
- **已修复并回归：** 触发：新场景索引把发布Gloss JSON字符串只按列表处理，导致标签数0及多字检索失败；或浏览器fetch被作为控制器方法调用导致Illegal invocation。证据：均有RED→GREEN，最终实库2000标签、浏览器error/warn为0；前端/核心/HTTP长度统一500。避免：测试使用真实字段格式并运行真实浏览器，不能只用mock fetch断言。
- **范围：** 功能仅在动画功能工作树新增scene模块，本主检出是否包含这些模块需另行核对；既有服务与并发改动保留。详细[实施报告](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/reports/basic-scene-expression-2026-11-08.md)。原视频到参数精确时基、手形/接触/非手部语法和自然度仍未认证。

## 自由文字生成主流程补充（2026-11-08）

- **主链路已实现、语言验收未运行：** 触发：将场景目录、相似句检索或整句接续当成“自由输入文字实现演示”。证据：两条在全部20,654条索引中均无原句/去标点命中的输入，从网页真实生成120帧并播放至结束；相同seed/步数下身体、左右手、面部参数均不同。避免：输入不得经过整句命中门槛，也不能改字幕配旧动作；参数不同和播放成功不能推出手语句义正确。现为预训练文本条件生成，词级组合仍未验收。
- **本机执行配置已实跑通过，环境外待验证：** 触发：Windows主机提交额度接近68.4GB上限，CUDA仍有大量空闲却无法分配16MiB。证据：旧全模搬运失败目录保留；只读checkpoint映射、真实CPU FP32文本缓存和逐层CUDA offload后，两次真实任务完成，manifest分别记录mapping/profile。避免：核对系统commit和实际配置；不以显存空闲排除主机额度问题，不声称CPU/CUDA编码逐位等价。只读加载固定PyTorch2.9.1内部接口，未知版本拒绝；原权重不写入。
- **验收范围：** Python服务/HTTP/场景/CLI定向82通过、生成runtime小fixture48通过、Node控制器18通过；两句浏览器播放结束、数值产物审计通过。截图接口失败，成功页截图缺失；语言、手形、自然度、完整表情语法、相机、全仓库/CI仍NOT_RUN。只按用户授权关闭旧8874/8878服务，Docker/WSL未动。1–200字符、显式4/6/8/10秒，两次4秒演示约5分钟。具体代码在动画功能工作树；证据报告路径为 C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge/docs/reports/free-text-sign-demo-2026-11-08.md，不能假定本主检出已包含实现。

## 快速词级组合补充（2026-11-08）

- **已实现工程链路、语言验收未运行：** 触发：自由输入每次调用完整神经生成，约5分钟等待不符合快速演示。证据：已从配套ISO/SignSparK train LMDB建1,990标签、5,916词片段，真实mBART beam4规划后检索组合，三条库外短句浏览器提交至playing回调显示0.20–0.33秒并播放结束。避免：分别报告预热输入延迟、首次进程加载/预热、HTTP和首像素；不能把三例推广为P95或任意文字保证。冷启动本次load约11.43秒、warmup约15.01秒，页面按ready门控。
- **已修复prompt误拒绝：** 触发：裁词表中的固定英文prompt产生9个UNK，对整段prompt检查会拒绝普通中文。证据：原6句全拒绝、原始token审计与修后6句规划均保留。避免：保留作者exact prompt，仅检查用户正文UNK，并拒绝输入/输出截断；不能删prompt或用伪Gloss替代。NUM_BEAMS固定4并纳入身份，旧beam1结果不能混用。
- **有限遗漏已阻断，整体语义仍未通过：** 触发：‘明天早上我们一起去医院看朋友。’在beam1/4都漏医院。证据：当前真实HTTP与浏览器返回needs_review/omitted_terms，禁止播放；‘我想去成都。’为资源缺失blocked，不能混为同一原因。避免：缺词不最近词替换，不播放部分句，不自动补译；固定地点/时间存在性检查不是完整语义验证，否定、数量、人名、空间指代和其他遗漏仍须真人审查。
- **词边界/显示时钟有边界：** ISO start/end只能用于配套LMDB，不直接套SOKE；左手smplx-left不再翻转/加均值，body严格126列取[:,66:]。30FPS为本产品显式预览时钟；LMDB无独立时间戳、作者展示默认20FPS，不能叫原始采集FPS已核验。原词帧保留只指组合参数，既有renderer仍可显示修订。7个无有效片段标签与95个缺源候选已保留；不得把1997原标签全部称为可播词库。
- **验证范围：** 定向Python109通过、生成runtime中builder/planner49通过、Node33通过；真实源片段抽样逐位/HTTP摘要与来源/三句播放/两种阻断通过，独立静态审查无新增P1/P2。语言、手形、自然度、完整非手控语法、全仓库/CI及本轮慢生成GPU复测仍NOT_RUN。仅重启本任务服务及其自有worker，Docker/WSL未动。报告在动画功能工作树 C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge/docs/reports/instant-sign-composition-2026-11-08.md；不能假定本主检出已包含实现。

## 词动作候选库扩容补充（2026-11-08）

- **已接入本机演示、未新增词义：** 固定配套train/dev/test资源合并后由5,916增至14,317片段（+8,401），标签仍1,990；33条ready中27条实际选择新增片段52次。固定36条输入及同一T2G身份前后仍33ready/1needs_review/1blocked/1HTTP422，无退化、无新增ready。避免：片段数、词义数与可播输入状态分别报告，不能把ready叫语言准确率。
- **已核验缺源范围：** 全量train ISO有200个源ID／1,698词例缺配套LMDB，旧95只是每词限3选例时遇到的失败。原7缺词的21源ID在三套LMDB均无记录，不能拿SOKE同ID直接套帧补齐，也不能用4+5冒充已审核45表达。新发布包仅提供更多候选，不会自动补成都或修复T2G漏医院。
- **已修复源身份与上限约束：** 原5,916单位ID/原源字段/NPZ字节全保留；每词≤8、单位≤20,000、来源≤16、索引≤16MiB及缓存有界。产品池须usage_role=product-demo/evaluation_eligible=false，dev/test源split与源库/LMDB/ISO摘要逐项绑定，已知官方SHA对不能伪改split。合并器曾漏16源上限、可发布运行时拒绝的库；现17源fixture验证创建输出前拒绝。
- **验收范围：** 定向预览Python117过、builder/merge36过；两条含新增候选短句浏览器播放结束，页面预热提交至playing为0.15/0.23秒；成都仍阻断，console error/warn为空。人工CSL语义/手形/自然度、相机及全仓库/CI仍NOT_RUN。新外部6707词视频来源并非现成指骨旋转，未重建的词不能计入可播库。报告在动画功能工作树 C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge/docs/reports/word-corpus-expansion-2026-11-08.md；本主检出未必包含实现。


## 跨会话本地整合与新组合句接触补充（2026-11-08）

- **已整合本地入口，语言验收未运行：** 自定义表达/扩展词库与v14A/手形修订原本位于同一动画工作树；统一使用8879服务，顶部互通文字演示、动作与手形对照、原句场景库。旧8874/8878按另一会话的用户选择已关闭，本轮不重启。现有19,877完整动作/7,369文本与1,990词标签/14,317候选是不同维度；扩候选不代表新增词义。总表：[当前动画工作入口](../animation-current-status.md)。
- **新组合句接触验收未通过：** 触发：将旧104段的核心检查通过推广到词级组合。证据：在线扩展库组合“我想喝水。”50帧，1,500指骨键不变/源输入未改，但仍有1个核心重叠线索、最大额外避让步长0.05826模型单位；同服务S000931专用轨仍命中且核心重叠0。避免：分别验证原句/组合句与来源，不能按文字相同套专用接触轨；主页面已显示穿模线索和腕部避让过渡提示，后续需单独调校。
- **本轮功能验证范围：** 合并定向Python196过/2跳过、Node114过/4跳过；真实HTTP与浏览器分别验证ready、缺成都blocked、漏医院needs_review及共用v14A。慢模型GPU、全仓库/CI与语言自然度未重测；保留之前ONNX/SciPy全仓库失败历史。工程可播放不等于接触/语义通过。

## 双向沟通产品整合补充（2026-11-08）

- **已接通工程链路，语言与动作质量未验收：** 动画工作树新增`/communicate/`统一上传视频识别、原文保留、人工核对、文字回复与数字人表达，独立8765识别服务与8879产品服务通过按次v3协商绑定。真实CE-CSL示例只返回“平方 我 自己 。”；完整房子句是测试者依据标注手动修正，不能算自动翻译正确。Libras仍是12词试点与中葡草案，未接入Libras生成或完整中葡句子互译。入口、报告和源码位置见[本机当前入口](../animation-current-status.md)。
- **已修复并定向回归：** 取消/换源拒绝迟到结果；断开网关任务先取消等待再清理临时文件；旧录制回调不影响新流和新计时器。模型`inferencing`须视为已加载但忙，不能因另一语种ready而停止重连并永久禁用本语种。wheel必须包含communication子目录与scene默认定义。
- **验收边界：** 用户本轮明确只验证上传视频，物理摄像头与真实浏览器录制NOT_RUN；WebM是数据集视频转码，重编码可能改变模型输出。设备单测、工程可播与界面流程通过均不等于识别语义、手形、穿模或自然度通过。原有新组合句接触失败继续保留，历史成绩与全仓库失败不覆盖。


## 完整原生词条与短切片补充（2026-11-09）

- **已修复“谢谢”选材和动作完整性，语言质量待验收：** 截图对应S000122“谢谢医生”[4,13)的9帧词片段，按预览30FPS仅0.3秒，右拇指活动不足约5度。时长中位数候选筛选漏掉了只有一个Gloss的S000003“谢谢！”原句。重新核对官方配对LMDB和标注后，选择完整30帧源动作；不改源指骨、蒙皮或FPS。三份独立“谢谢！”完整序列加入候选并标记，句内也优先完整词条，纯词输入另有精确匹配；原1,989个其他词候选及5,916原单位保持。
- **已加失败关闭约束：** 必须先统计同一源视频全部标注，再按请求标签筛选；不能过滤掉“医生”后误报独立“谢谢”。完整标记绑定单标签数量、原标注范围、全片范围和原blob摘要，合并器与运行时均验证；换骨架、篡改词条、伪完整标记均拒绝。源object元数据仅作无执行的pickle指令检查，构建和运行不反序列化它。
- **边界：** 官方ISO裁剪实现本身也是直接取配对LMDB区间；本轮没有凭感觉乘时间系数、拉慢FPS或替换同ID的SOKE参数。完整词条补充不解决T2G全部语义规划或其他词条质量；通用手间/袖口/网格碰撞与真人语言自然度仍待验收。详细[本轮报告](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/reports/thanks-native-gesture-2026-11-09.md)（实现位于动画工作树）。

## 沟通页面模型路由补充（2026-11-09）

- **已确认、待修复接线：** 用户反馈模仿“我想睡觉”却返回“绿色”。核查8879→8765实际中文路由为CE-CSL TFNet；启动器写死`csl=csl-tfnet`和CPU，catalog没有Uni-Sign。已有`model_defaults.json`分环境策略由桌面入口使用，网页API没有读取，不能因默认配置或本地权重存在就声称页面使用Uni-Sign。两份Uni-Sign权重本轮SHA256均与manifest一致；S2仍遵守ADR-P1-013的开发/测试范围，不能宣称全面更优。
- **避免错误归因：** TFNet词表有“我/想/睡觉/绿色”；CTC有非空token即标记recognized，不等于语义通过。CSL-Daily动作库与CE-CSL识别没有模仿往返验收，不能凭数字人能播放就要求真人必能识别回来。未取得用户原录制，本例原因未复现；不能归因为用户动作、CPU或镜像。源码仅确认镜像为预览CSS、录制走原始流。
- **范围：** 本轮只核查实际HTTP/配置/源码/资产哈希，没有切换服务、模型重跑或摄像头验收；详见[路由核查](../reports/communication-recognizer-selection-audit-2026-11-09.md)。


## Uni-Sign页面切换与启动保护补充（2026-11-09）

- **已修复并实机核验模型路由：** 用户授权后，8765中文从TFNet切为CSL-Daily Uni-Sign基础版，实际task=text，翻译与RTMPose均CUDA；8879的真实MP4/WebM请求返回整句，MP4与直连原文一致。Libras独立RF仍通过。本机profile显式配置csl_model/libras_model/device/startup_timeout_seconds；旧profile省略字段仍用TFNet/CPU兼容值，不可只看默认示例推定已迁移。S2与LLM融合未自动启用。
- **已修复启动保护：** 模型/设备使用完整参数及相邻值匹配，防止unisign_pose误匹配unisign_pose_stage1或从别处匹配auto。Stop/RestartProduct在停止任何进程前预核验所有受影响服务；6个隔离场景从2通过/4失败变为6通过，实际错误别名Start被拒且监听PID不变。
- **资源与验收边界：** TFNet仍在线时影子加载因Windows提交内存不足失败；仅释放身份核验的旧识别进程后，顺序加载成功，不能将首次OOM记为模型质量失败。使用含Torch/API依赖的.conda-env；D盘通用Python的旧Starlette/HTTPX不能用于本次TestClient回归。Python定向50过，上传格式适配不是物理摄像头验收；本次用户录制、识别语义、实时流、全仓库/CI均NOT_RUN。SLRT有独立new_stream，但共享v3仍为clip；不能把Uni-Sign切窗重跑或模型切换写成原生流式已接入。
- 详细记录：[切换与实时模型范围](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-product-20261112/docs/reports/communication-unisign-switch-2026-11-09.md)。


## SLRT + LLM小闭环实测补充（2026-11-09）

- **已运行、语言/实时中文验收未通过：** 固定30条CSL-Daily已见test短句（排除旧Pilot-50），当前帧缓存v2+RTMPose CUDA的完整视频输出全非空，WER34.52%。受约束deepseek-v4.1-flash整理后19草稿/9信息不足/2次4096-token空正文，调用P50 5.58s、P95 28.84s；同19成功子集没有字符编辑相似度净收益证据。不能把覆盖63.3%叫准确率，也不能把拒答/失败从分母删掉。
- **增量词可交付不等于完整实时字幕：** 同6条原FPS回放均在采集EOF前交付final词（工程状态），但5条队列丢帧，608采集帧只接收592帧，5条词列与完整文件不同，同6条WER37.5%→50%。EOF前中文草稿0/6，最终4草稿/2拒答；有效最终草稿在EOF后额外等待P50 6.29s（仅4成功）。这是数据集回放/共享机器实测，不能当物理摄像头P95或全局SLRT性能结论。
- **合法JSON/词下标不保证语义保持：** AI文字初审19草稿发现4条明确语义风险、6条待复核；不是/他/去/见面/同学在两签者视频均变成他不是去见同学，改变否定作用域。参考原文只进评测，不能用练习句回填生成端。AI初审不等于真人盲评。最新词版本拒答时必须清除/标记旧草稿，不能把历史的去工作重新算最终输出。
- **测量与选材约束：** 回放前修正异步异常静默、混用计时原点、取消伪EOF、native仍运行时释放backend、draft空证据等仪器问题；7项定向测试通过，阶段源码SHA有记录。关键词水会误命中水平，本轮保留该条主分母并披露主题误分；后续用概念/完整词校验，不把子串主题筛选称封闭场景覆盖。新签者、用户我想睡觉原录制、真人摄像头、网页E2E、长时会议与CI仍NOT_RUN。详见[本轮报告](../reports/slrt-llm-scene-validation-2026-11-09.md)。

## Uni-Sign同视频对照补充（2026-11-09）

- **离线API对照已运行、泛化/实时语义待验证：** 同上轮30条、同派生视频、同p1-v1参考，Uni-Sign基础版BLEU26.1916、编辑相似度0.4607；SLRT直接拼词16.7320/0.3792，SLRT+LLM全请求7.5870/0.3023（含9拒答2失败）。同19条LLM已输出子集，Uni-Sign31.6415/0.5258，对照LLM20.1049/0.4774，但编辑增益95%签者聚类区间[−0.0870,+0.1805]仍含0。避免：全请求与相同成功子集均报告，不将BLEU/非空率叫准确率，不将本批观察推广到新签者。
- **静止输入拒答验收未通过：** 黑帧输出空，静止人物重复帧被翻成“我肚子鼓鼓的。”；相同控制SLRT无词。Uni-Sign非空输出不保证有效动作或语义正确，也存在“小猫睡觉”→“你有早睡觉吗?”等明显误译。避免：评估静止/无手语控制，后续验证活动检测与拒答，不能靠句子流畅度放行。完整文件HTTP P50 2.28s/P95 3.23s不含录制时长，不能当原生流式延迟。详见[同视频对照](../reports/unisign-scene-comparison-2026-11-09.md)。

## 实时策略与当前吞吐审计补充（2026-11-09）

- **瓶颈已确认、优化待验证：** 当前帧缓存v2已启用；同6条回放profiling中姿态约25–32ms/帧、canonical约3ms/帧、窗口约25ms/窗，stride=1稳态串行超过30FPS预算。不能将旧报告的加入帧缓存再次当成未做优化，也不能以加大队列掩盖积压。候选人体框复用/流水线/锚点stride=2需分别验证，不跳过采集帧后伪造连续时序。共享133点仍需按两模型分别适配，不能直接复用63点热图或已经归一化的输入。
- **方案建议、未实施：** 用户选择动作中约1–2秒出词、停顿后补完整中文；建议SLRT词与Uni-Sign句末翻译分工，LLM不阻塞交付。当前网页仍以完整片段POST识别，HTTP请求字节流不是模型增量流。上游wait2 G2T示例为德语，本地未发现中文在线G2T，不能拿Text2Gloss倒用。ReSCUE官方仓库本次仍Coming Soon，论文实时分数不作为本地能力。详见[策略评估](../research/real-time-recognition-strategy-2026-11-09.md)。

## 实时四阶段工程补充（2026-11-10）

- **已接入工程、语言验收待完成：** 新入口`/communicate/realtime.html`使用独立`csl-realtime-v1` WS；SLRT增量词、共享RTMPose133点的Uni-Sign句末中文、可选LLM风险提示分开交付。触发：旧完整片段POST无法提供动作中的词。证据：真实产品桥及H264浏览器样例已出词/中文，网页显示原译文与“酒/饭店”冲突提示，未覆盖原文。避免：`native_streaming_translation=false`，不能把整套架构称Uni-Sign原生流式或与数字人动作互为逆映射；模型/API与产品仍在两个既有检出。
- **已固定显式调度协议、不是泛化提升：** CUDA FP16/detect5/stride4/vote3/window16，采集与姿态仍30FPS、窗口推理7.5Hz。v4产品桥30正例+2控制全部完成；首partial P50/P95=0.678/1.142s，末句在原视频EOF后附加P95=1.422s（含1.2秒重复尾帧）；词WER29.76%、文本BLEU25.454/edit0.4205。避免：同批已见test、同调度诊断不是新签者实验；这些数值不是准确率或逐词动作完成延迟。默认离线SLRT协议没有被改成该配置，旧512px Uni-Sign file成绩不能混用于共享320px新前端。
- **短片通过不能替代持续负载；句末调度缺口已修复并通过限定合成负载：** 触发：逐条32请求均完成，但同一WS连续送帧27.965秒后overload；词队列已有积压时立即运行句末翻译会耗尽16帧缓冲。证据：`continuous-10m-v4-r2`为有效失败；v5同一WS604秒18,112帧全部ack/处理、0丢帧、128句、0过载/翻译拒答，输入队列等待P95=218ms、峰值12帧，句末等待峰值797ms。避免：先处理当前帧/积压，最多2句FIFO等待，空闲执行Uni-Sign，等1.5秒明确拒答；保持输入队列16，EOF与取消区别处理。调度RED/GREEN及增量独立审查通过；两固定视频循环不是自然会议/语言验收，模型RSS增加16MiB不宣称长期无泄漏。
- **测试仪器阻塞已修复：** 触发：在发送协程枚举全部进程资源阻塞6秒，过期帧突发补发造成伪过载。证据：首次`continuous-10m-v4`最大生产者落后11.2秒，此轮无效。避免：只按启动器已核验PID采样；发送落后250ms立即中止测量，不突发追赶也不改帧号掩盖不连续。
- **LLM时间预算与UI传参已修复：** 触发：只设置HTTPX逐操作timeout仍可被慢字节流长期占用；UI复选框未传start则真实页面从未启用辅助。证据：独立审查2项Important及RED/GREEN；修复为async整体8秒截止/64KiB、主动关闭、单运行/单等待，实际2条产品WS后HTTP查询分别2.72/3.07秒返回risk。避免：示例默认禁用、本机能力启用但每次页面默认不勾选；仅发送实际词与原译文，不发送GT/目标练习句/视频；建议不覆盖人工编辑。保守字符/人称否定等拦截不是完整语义证明。首次三次HTTPStatusError也保留，不能只报告成功请求。
- **启动内容身份已实机核验：** 修改同一实时JSON路径的根/嵌套配置后，Start均拒绝复用，两个服务PID不变，恢复原字节后可用。避免：同时核对完整相邻参数和配置SHA，不以路径相同推定已加载新配置；更换profile内路径前先按旧profile停止。Stop后监听端口可能短暂残留，立即Start曾保守拒绝；确认端口自然释放后同一启动器重试，未终止不明PID。
- 证据与最终验收范围见[实施报告](../reports/realtime-csl-engineering-2026-11-10.md)及[使用说明](../realtime-csl.md)。物理摄像头、共享画面权限、新签者、用户原“我想睡觉”录制、真人手语语义/手形审核与远端CI仍NOT_RUN。

## 训练样本应用验证补充（批次创建于2026-11-10 UTC）

- **官方训练划分已核验、应用短句回放已测：** 30条/30种表达/8签者均属于官方train，与dev/test无ID交集。按作者姿态区间取视频、固定≤200帧（1.23–6.67秒）后签者轮询，不按识别结果挑样。完整视频Uni-Sign规范化整句匹配30/30，实时含1.2秒补尾帧29/30，SLRT词WER18.90%（S17/D27/I4，参考254词），LLM关闭。同权重官方姿态历史预测参照30/30，仅为imported，不冒充本轮重跑。避免：训练回放可支持有边界的应用路线，但不能把句匹配率、词WER、新人模仿或真实需求覆盖率合并为一个准确率。
- **自动分句误追加验收未通过：** S000068的“小猫在桌子下面。”先正确输出，视频末段另被切句并误加“没钱，我出去玩。”。按原文件自然EOF、去掉补尾帧的单例复测只出原句，不能据此把主29/30重算为30/30。避免：明确单句结束/确认；不要把句后运动都当新表达，也不要硬删与参考不符的句子。该例LLM对正确首句和错误次句均提示风险，不能作为语义裁判。
- **运行性能漂移已缓解、根因待定位：** 同源码/配置服务已运行22.8小时，训练单例、旧已通过test单例、静止控制均overload；黑帧通过。冻结现场后用身份核验启动器重建同权重服务，旧探针恢复、完整v2实验通过。避免：ready不等于吞吐稳定；保留v1不完整失败，不把过载记为识别空结果，不把重建恢复写成根因修复/24小时稳定通过。未改生产代码或权重。
- **应用覆盖应核对完整表达：** 18,401条训练标注中“我想睡觉”无逐字相同句，但含睡觉113样本/41文本，也有想睡觉的相关表达。不能从字面无同句推断没有语义覆盖，也不能从训练过某个词推定任意组合可靠。先对齐场景、常用句、同源示范动作与逐句验收；用户照着数字人模仿仍待验证。证据见[训练应用报告](../reports/training-application-validation.md)，原始时间/协议/视频SHA及逐条表在train-application-validation-v2产物内，历史失败在v1保留。

## 实时延迟维护补充（2026-11-11）

- **重复预处理与ACK重绘已修复、保持输入协议：** RTMPose完成的320px canonical像素供SLRT直接复用，避免第二次PIL转换；同8输入事件完全一致，重复处理均值3.311→0.005ms/有效帧，ACK排队P95 82.43→48.56ms。ACK不再触发整字幕/历史snapshot。避免：不能减帧后重新编号、扩大队列藏延迟，或把先前已有缓存再次当新增优化。
- **准备时间必须计入体验：** 同native worker有界预热现有detector/pose/S3D，清检测和帧缓存，不把模拟帧送入识别；首词从开始按钮计时，并保留采集起点口径。同6正样例含准备首词P95 1.112s，采集起点P95 0.922→0.943s并未下降；不将吞吐余量改善写成所有首词延迟变快。状态中保留active/last有界数字计时，取消/过载也可查；重复快照必须幂等，终止时补最终跳帧指标。
- **10分钟持续回放已测、老化根因仍待验证：** 单WS 603.92s、18112帧全部处理、零丢帧/过载、128句，排队P95 171.96ms。这是固定视频合成负载；该阶段之后只调整了初始化元数据取锁位置，最终部署另做短测，不冒充该最后改动后的再次10分钟测试。空闲任务进程工作集回收未复现35ms持续慢速，不能归因驻留内存，也不能把重建恢复当24小时稳定/根因已修复。
- **明确完成与中止不同：** “完成本句并暂停”先排空编码/已有帧、再finish并等待句末；“中止识别”、换源/后台仍丢弃未完成句，不能伪造EOF。这是应用收尾改进，不能宣称自动分句误追加或用户模仿已验收。native租约内避免反向请求manager元数据锁，pipeline可在取得租约前冻结；仍保留ready和expected_model校验。
- 证据与分阶段验收见[延迟优化报告](../reports/realtime-latency-optimization-2026-11-11.md)。物理摄像头、新签者、原“我想睡觉”视频和真人语言验收仍NOT_RUN。
- **浏览器采样需另测：** 同一训练示范的H264自然EOF首词1.29s/零采样跳帧；加尾帧后手动于3.050/5.833s完成，首词0.92s但采样跳帧1，最终句相同。保留该跳帧和native间断诊断，不能用Python持续回放零丢帧推定浏览器/摄像头零丢帧，也不能把工具等待超时直接当应用失败。网页按钮、媒体暂停状态和实际译文共同验收。

## 实时句末核对迭代（2026-11-11）

- **误追加诊断边界已澄清：** S000068末尾确有额外手部动作，第二段是新的“没有”词证据，并非上一句残词或只有重复静止帧。其交际意图仍未人工审核；不能自动删除所有再次活动。Uni-Sign仅凭该段生成“没钱，我出去玩”，自动误追加本身尚未修复。
- **新增提示不是正确性门槛：** 仅一词且译文内容字符>max(4,2*词字符)附加重点核对提示，无GT/LLM，不拒答、不覆盖原文、不剔除指标行。64固定回放全部完成、原始语义事件64/64一致，train整句仍29/30、已见test仍2/30；train提示1/31段，test0/30。未提示不等于正确，不能把零提示写成验收通过，也不能把人工修订后文本混入模型成绩。
- **本人核对必须逐句绑定：** word_evidence、原译文、review_hint及advice revision随各自caption保留。人工确认/修改后主显示同步；被拒绝的native修订不改变人工文本或原译文辅助任务。真正接受新机器修订时清旧advice/取消旧轮询；已提交同revision不能换文字。避免：旧句建议留在新译文、确认后pending任务被取消却一直显示等待、编辑只更新历史而主显示仍旧。相关RED/GREEN与独立复查均已留证。
- **布局瞬态应安全跳过：** ResizeObserver可能在video已有尺寸而preview为0时调用paint，旧路径抛视频尺寸RangeError。四尺寸positive finite才算videoRect，瞬态隐藏选区、恢复后重算；真实回调测试及最终浏览器重载/识别/核对流程无新console warn/error。不能把一次顺畅的截图推定尺寸始终有效。
- 最终真实浏览器训练H264样例首词0.98s、编码3ms、零采样跳帧；实际完成确认、明确词级修订及原文回读。自动化模拟操作只验证机制，不证明选择的“没有”是手势真实语义。报告：[实时句末证据与核对](../reports/realtime-utterance-review-2026-11-11.md)。物理摄像头、新签者、真人语言/中巴任务、24h稳定及CI仍NOT_RUN。

## 可复现示范与来源隔离（2026-11-11）

- **训练示范已接入，跟练与语义验收未运行：** 4条官方train来源/标注/哈希独立核验后转H264，3条成功回放、1条已知误追加明确保留。4/4完成、首词含准备783–971ms；这是按已有结果选的演示输入，不能据3条成功推出整体准确率。示范文字只显示在视频外，不进入WS/LLM；验收必须读取生成区，不能把同屏示范标注当模型输出。
- **动作组合不是识别的逆映射：** 当前“我想睡觉”实际组合计划39帧/30FPS，3词来自不同原句且含dev素材，语言审核NOT_RUN；动作池为product-demo/evaluation_eligible=false。跟练应另验，不能用动画可播或模型训练包含某词证明模仿必成功，也不能据该计划单独归因用户原录制错误。
- **启动身份缺席检查已修复：** profile禁用可选开关时，只检查正向token会误复用旧启用服务；完整选项名须同时识别空白/等号/引号边界，再核对开启时相邻值和清单SHA。17项隔离回归通过；实际禁用/同路径变更均拒绝，恢复后复用，两端PID未变。FileResponse哈希检查不宣称能抵抗本地并发恶意换文件的原子快照。
- **换源竞态已修复、延迟仍分阶段报告：** 旧video.play Promise迟到不能停止新来源，加载回调绑定sourceVersion；历史逐句保留来源。浏览器校园回放首词0.99s/零采样跳帧，问题回放1.18s/零采样跳帧，但排队P95 225.51ms，不能把单帧零排队推广全程。原误追加、长时退化根因、摄像头/新签者/真人语言验收仍待验证。证据见[示范与网页延迟复核](../reports/realtime-demo-catalog-2026-11-11.md)。

## 句末计算与资源复核（2026-11-11）

- **模型去重已验证，不等于所有体验变快：** 32冻结姿态输入/96对生成token及文本相同，encoder 2→1。8个实际WS输入语义事件相同，ACK排队P95 146.45→130.30ms，首词含准备P95却1143→1202ms。分别保留模型、准备、词和句末耗时，不能拿某一项加速宣传全程提升。权重、beam和SLRT规则未改。
- **Windows物理可用不是提交余量：** 服务驻留时另载一份模型，物理仍约4.7GiB可用而commit余量仅约0.7GB，其他轻量进程MemoryError/WinError1455/PowerShell启动失败。已有配对任务正常完成，不将资源失败写成模型空识别。基准现在在模型加载前检查至少10GiB free commit，实际拒绝通过；按身份释放空闲服务后顺序测。预检不预留资源，不把它当老化根因修复。
- **本轮网页复验未通过：** IAB新标签attach、鼠标和截图均有控制超时；既有页面键盘操作后视频到EOF但native收帧0，仍显示旧主字幕。不能读同屏旧记录/示范标注冒充当前输出，也不能据此单独归因模型。保留browser-empty/native-zero证据，后续需修空帧提示和当前会话显示，再恢复真实网页验收。详见[本轮报告](../reports/unisign-generation-optimization-2026-11-11.md)。

## 当前字幕与采集恢复（2026-11-11）

- **旧字幕回退/零帧成功提示已工程修复，真实网页待验：** 主显示按本地epoch取当前草稿/本轮句子，换源/取消/新尝试不回退旧历史；真实summary0→finished在当前client显示未采集画面及currentCaption=null。历史原文/确认不丢弃，不用“清空历史”解决显示错误。43 Node和真实HTTP/WS重放通过不是实际网页采样通过。
- **采集挂起与取消已修复：** 首帧立即读，后续仍按RAF/墙钟索引；250ms观察器发现2秒无JPEG交付则停止，不用重复旧帧凑连续流。EOF编码不回调也可停止并解除等待；旧回调不能清新pending。终态要同时停止active/draining，避免watchdog覆盖原服务错误；断线前尚未ACK也须发布最终sent计数。该watchdog不是动作/冻结画面判别，也不保证事件循环阻塞时严格2秒。
- **静态部署与浏览器证据分开：** FileResponse直接读磁盘且no-cache，4份HTTP字节一致可免重启，识别67636/产品45240均保持；已打开页仍需重新加载模块。本轮新tab attach和旧tab刷新/控制继续超时，未获得新版截图或正常视频交付，不把旧截图/Node页面替身当浏览器验收。详见[恢复报告](../reports/realtime-capture-recovery-2026-11-11.md)。

## 实时字幕接续对话（2026-11-11）

- **已实现工程接线，真实跨窗口待验：** 触发：实时字幕与面对面回复原为两个独立流程。现须暂停采集并人工确认后，才同源交接到assist；保留实时历史、对话草稿、机器原文/词证据/训练来源，导入不自动生成。68 Node/独立68及32接口通过，8模块HTTP字节核验；它们不代表实际浏览器或完整比赛闭环通过，也不新增首词性能结论。
- **超时分流已修复：** 触发：10秒无ACK即清target，重试新开页使原草稿与后续对话分离。实际Sender RED→GREEN后只取消nonce，保留尚未关闭窗口；错误origin/source/nonce测试须提供正确expiresAt，避免由其他无效字段遮蔽守卫。Node会忽略命令中不存在的部分测试文件，最终须先核对显式清单，旧对话回归为communication_product.test.mjs。
- **静态白名单和桌面控制边界：** 新模块需产品重启后才从404变200；按身份分步重启仅产品45240→59624，识别67636保持。宿主截图/窗口激活也失败，不能臆定锁屏或网站故障；待桌面控制恢复再做实际跨页验收。证据见[本轮报告](../reports/realtime-conversation-2026-11-11.md)，历史零帧/语义/长期退化问题不因本轮接线消失。

## 译文分数独立诊断（2026-11-11）

- **已取得离线beam分数，线上仍未接入：** 原61历史句段重建133点后文字全部相同，加2静止直接探针，共63输入/126对token一致；取分数模型均值274.37→274.89ms、P95415.87→434.91ms，完整调用CUDA allocated相对峰值均值差24.37MiB。分数是搜索log分数，非正确概率；只读标量仍需计入生成全beam逐步score的真实成本，选中token长度与总score步数不可混用。
- **排序代理不等于语义判定：** 已见train30中29逐字匹配、test30中2匹配，代理风险AUC1.0/0.9464分别只有1个负类/2个正类。已知扩写−1.548比首句−0.136低，但人称误译−0.315仍高于部分匹配句；语序同义也会被逐字代理惩罚。不能据此自动放行/删句或宣称94.6%准确率，静止控制也不能跳过live词/活动门控。
- **权重和阶段源码身份须实证：** UniSignConfig只核声明，直接Torch runtime不会代为核验checkpoint SHA；独立脚本须load前核实际bytes/SHA，并覆盖load/metadata异常close。prepare、运行中的旧extract、新infer/后续evaluate应分别绑定，不能用最终磁盘SHA替代旧执行源码。7机制测试及审查修复保留。
- **恢复与浏览器范围：** 原profileStop/Start后32160/49052，8HTTP模块字节保持，真实3WS/325帧全处理、两无动作控制0句；不是摄像头/长期稳定验收。原IAB实际innerWidth/Height0、visible、videoWidth512；0宽截图和尺寸set/reset均失败，可见性false已恢复，override状态未知。见[完整诊断](../reports/unisign-score-diagnostic-2026-11-11.md)，原误追加、真人语义与长期退化未消失。

## 网页控制恢复审计（2026-11-11）

- **真实网页验收仍受阻：** 应用原生open_in_codex面板请求多次等待无返回，停止的是观察脚本，不推定应用动作已取消/失败或审批拒绝。随后实际IAB仍原2/3页，原页innerWidth/Height0、visible、videoWidth512；“实时服务已就绪”和旧字幕不能证明新采样/新模块验收。不要反复开页或用其他技术绕过指定控制通路。
- **服务与展示状态分开：** 本轮8HTTP模块、3模型/协议源码SHA均保持，8765/8879仍32160/49052，ready/idle及回复规划器ready；无重启/产品代码变更。正常浏览器是否可用与自动控制失败是不同事项。已请求手动展开Codex面板并确认桌面可操作，真实闭环未通过且不缩小为工程通过；见[审计报告](../reports/realtime-ui-recovery-2026-11-11.md)。

## 数字人与真人同源动作识别（2026-11-12）

- **同源回认验收未通过，原因待验证：** 触发：对同一训练样本的真人视频与数字人重定向画布做实时接口回放。真人“谢谢”被识别为“谢谢”；数字人识别为“骄傲”，句末为“这个杯子十块钱。”，两次无输入丢帧。避免：同源 ID、动画可播放、姿态存在及真人训练样本识别正确，都不能证明数字人动作会被正确识别；词分数未经校准，不能当正确率。30 帧动作与123帧源视频并非逐帧等长，录制静止段不同，根因和人工手语语义审核均未验证。完整协议、时长和证据见[同源对照报告](../reports/avatar-human-recognition-comparison-2026-11-12.md)。
- **组合示范回认未通过：** “我想睡觉”由三个不同样本的词素材拼成，含 dev 素材，语言/动作/组合均未人工审核；数字人录屏99帧全部处理但无词/句事件。不能把该附测作为同一句真人视频配对或复现用户摄像头“绿色”，也不能把正常结束理解为语义识别成功。

## 数字人动作几何优化与回认（2026-11-12）

- **待验证，隔离候选未接入：** 同一“我想睡觉”动作触发避碰最大相邻修正0.05826、305个1/8帧采样累计61次核心点命中。冻结输入上的离线候选降低至0.01402和0次，但最大偏移0.07952超过当前0.06，手语语义未人工审核；实际数字人视频经SLRT/Uni-Sign依旧无词/句。避免：不得仅凭固定样本、稀疏核心几何采样或回认输出宣称语义正确、全部碰撞消除或直接提高全局偏移上限。限制0.06的平滑网格候选增加关键帧命中，关键帧零命中也曾有13次插值采样命中；详见[交叉核对报告](../reports/avatar-crosscheck-optimization-2026-11-12.md)。
- **慢放不是已证实的识别修复：** 同一“谢谢”和“我想睡觉”各一次0.5倍速画布回放均无词/句。它们是固定采集条件下的单例，不代表其他速度、模型或真人摄像头。

## 训练视频实时复核（2026-11-12）

- **固定已见样本复测已完成，真实摄像头仍待验：** 触发：用户要求重新核对训练集识别。证据：原30条短train视频、2控制在当前产品WS全部完成，30例6113帧全部处理、零输入丢帧；句末规范化29/30、Gloss整序列11/30、词WER18.90%（替17/删27/插4），预测与2026-11-10同批完全一致。避免：三种指标不可混称正确率；同批重跑不是新增独立样本，不代表全训练集、数字人或新签者；唯一句末失配仍为小猫句后误追加。详见[本轮报告](../reports/train-realtime-recheck-2026-11-12.md)。

## 训练来源数字人实时回认（2026-11-12）

- **数字人回认验收未通过：** 触发：用户要求识别训练集来源的数字人手语。证据：26个完整单Gloss train源中25个在已验证LMDB可用、1个缺失；25段数字人视频全部处理、0输入丢帧，最终词匹配6/25、句末文本匹配8/25，14条无词/句。避免：含空输出、模型链路不同指标和素材缺失要分别报告；这是8词3签者的product-demo诊断，不代表官方测试、全部训练数据或专家手语验收；不把上一批真人训练视频29/30当数字人识别率。
- **录制变体可能改变回认结果：** 同一“谢谢”动作和骨架两份各90帧视频不是逐像素相同，前者固定重放3次均“骄傲”及错误句，后者3次均无输出；具体原因未知，不得归因模型随机性或某一步。完整协议和原始产物见[批量报告](../reports/avatar-train-recognition-2026-11-12.md)。

## 完整训练句的数字人回认与迭代（2026-11-12）

- **数字人完整句回认未通过：** 同源30条train完整原生动作的数字人画布，词整序列0/30、词WER90.16%（替30/删199/插0）、句末匹配4/30；先前对应真人视频为11/30、18.90%、29/30。原始视频裁剪、数字人时钟与视觉域均不同，不能单独归因渲染或模型；不代表全部训练集/官方基准。
- **放大不是全面修复、直接补句有误接受：** 对同画布固定放大，25单词词匹配6→6、WER76→80%；不重合30完整句WER90.16→83.46%、句匹配4→3，均不可直接上线。事后选14个实时空输出改用文件接口5条句匹配，但静止数字人被误译“腰酸背痛。”；不能取消门控或用LLM润色掩盖无证据。方法、样本及后续验收见[扩展报告](../reports/avatar-train-sentence-expansion-2026-11-12.md)。

## 基于训练姿态的动作迭代（2026-11-12）

- **受限候选验收未通过、未接入产品：** 30条同源训练样本按签者分为17拟合/13保留，肩宽归一化高置信手腕点的拟合偏移在坐标层有改善，但真人有效区间与native帧仍仅线性暂配准。8个不同签者实际候选中6段通过相对原动作的有界位移与1/4帧核心命中不增约束，2段回退；保留签者4条真实重渲染手腕平均偏差0.462→0.437肩宽，但词WER81.58%不变；拟合组句匹配1/4→0/4。避免：估计关键点、有限核心采样和理想二维拟合均不是专家语义或完整碰撞验收；原动作回退的重录画面仍可能改变预测，不能把全部差异归因于动作修正。新增可复核评估工具与完整证据见[本轮报告](../reports/avatar-retarget-iteration-2026-11-12.md)。

## 数字人缺陷与后续计划入口（2026-11-12）

- [当前缺陷、证据与改进计划](../superpowers/plans/2026-11-12-avatar-motion-quality.md)集中维护 AV-01～AV-13、M0～M6 和验收条件。先完成可靠时序与可重复 A/A 输入，再解释动作候选的因果收益；本轮只有文档归档，不新增模型、摄像头或动作语义通过结论。
- 动作数据准备、浏览器实时渲染和测试画布录像是三个不同环节；快速组合与可选完整生成均先准备整段再播放。不得把录制后实时节奏回放写成数字人直接实时采集、流式动作生成或摄像头验收；历史实验数值继续留在对应报告，新结果另写。
