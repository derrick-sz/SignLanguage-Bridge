# 项目记忆：模型、评测与实时链路的坑

最近核验：**2026-10-28**（本次补充三维生成路线核验；既有实验仍以各节日期为准）。适用于当前 SignLanguage-Bridge 工作区。这里保存易重复踩到的原因与处理规则；运行命令见[集成说明](../uni-sign-integration.md)，逐条证据见文末索引。复用结论前核对当前源码、manifest、运行协议和产物；缺失的本地证据标为缺失。

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

- **仍待优化：** 当前上游 forward 先执行未使用的 sentinel loss 路径，再 generate；24 条完整推理均实测调用 mT5 encoder 两次。这是效率问题，不是离线低 BLEU 的直接解释。优化时验证生成结果、编码次数和端到端耗时。
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

- **状态：已修复（2026-10-27 修正 `arbitration_rows` 为协议口径，并同时输出 `material_disagreement_rows`；Round-3 双评已实跑：协议口径 36/90=40%、实质分歧 8/90=8.9%）。** 触发条件：Round-3 的 A/B 两位评审评分仅差 1 分，且严重语义错误判断相同。[Round-3 README](../../experiments/p1/e06/human_review/round3/README.md) 和模板填写说明要求对评分或严重错误的任意分歧仲裁，但 [`make_round3_package.py`](../../experiments/p1/e06/human_review/round3/make_round3_package.py) 的 `agreement.arbitration_rows` 仅筛选 `abs(score_A-score_B) >= 2` 或严重错误不同，因而会漏掉该类 1 分分歧。本次只完成 AI 评审人 A，未读取 B 评审意见，不能据此声称一致性或仲裁已通过。
- 避免办法：B 到齐后，按协议额外以“评分不同或严重错误不同”逐行扫描，并由第三人或用户裁定；不能把现有 `arbitration_rows` 当成全部协议分歧。独立记录 AI/人工身份、量表版本与已读历史文本，不能将同一 AI 的两次输出计为两名独立人工评审。当前单 A 结果与边界见 [Round-3 AI A 评审摘要](../../outputs/01a17b59-e3b8-7590-b21b-250b3f41ca6f/round3/round3_reviewer_A_ai_summary.md)。

## 15. 三维动画生成路线的能力边界（2026-10-28）

- **待验证：动作质量批准不等于语言认证。** 触发：将上游 SignBridge 的资源启用/审批当成“已审核标准 CSL 词库”。证据：固定 SHA `3503ff7802bc991ccf5e0753c2fe41b79689b860` 的 100 段公开快照中，83 段拟合资产的批准为 `motion_quality_only`，manifest 明示 `semantic_accuracy_certified=false`；片段数也不是标准词条数。避免：分别记录语言、动作与来源审核，绑定具体资产和重定向版本；语言验收 **NOT_RUN**。[R20]
- **待验证：SMPL-X 重定向已有原型，但非手控表达未完整实现。** 触发：把上游播放器可播放理解成完整 CSL 连续生成。证据：上游已有 40 骨骼映射，但 jaw/eyes 静止、表情系数未重定向；片段顺播和四元数插值没有证明词间接触、空间指代与语法正确。避免：完整短语优先，单词与新拼接句分别验收；不把源码审计写成运行通过。[R20]
- **待验证：同为 133 的协议不可互换。** 触发：复用 Uni-Sign 的 RTMPose 数据来直接驱动 SOKE/SMPL-X。证据：前者为 `T×133×3` 的二维点与 confidence；SOKE 论文的 133 维是人体动作参数表示。避免：显式绑定 feature schema、旋转/坐标/时基与重定向配置，禁止凭维度 reshape 或把置信度当深度。[R20]
- **待验证：生成候选与跨语言资源须另行审核。** 触发：把 SignAvatars、VLibras 或 SOKE 的公开资料视为通用 CSL/Libras 动作授权与语言保证。证据：各自语种、数据和许可不同；SOKE 的任务 checkpoint 完整性、MVSign 的受限访问以及本地运行均未验证。避免：保留独立语法/词典和原始识别证据，未知词/否定缺项不能静默跳过后自动播放。以上为调研约束，没有新增实验成绩或验收通过结论。[R20]
- **已核验结构，驱动效果待验证（2026-10-28）：脸部静止不能直接归因于模型没有表情结构。** 本机 Godette GLB 没有 morph targets，但包含带非零顶点权重的嘴/眼睑/眉部骨骼；当前重定向只取约 40 根头颈/上肢/双手骨骼，脸部保持静止。先验证现有骨骼驱动和局部变形，再决定用 Blender 修权重、补 Shape Keys 或换角色。骨骼/权重存在不等于表情控制或 CSL 非手控语法通过；本轮只抽查旧演示两个片段的浏览器播放和一个暂停帧。[R22]
- **待实现（2026-10-28）：动画确认不能复用程序终态冒充人工批准。** `CaptionLedger.committed` 与 job `subtitle.final` 是程序提交/任务终态；当前没有人工 confirm/play 操作。动画计划需另存明确输入确认、来源会话与播放代次，识别取消/来源失效不等于已有 Three.js 播放器会自动停止。集成方案已另写草案，本轮未新增这些产品能力。[R22]

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
- [R6：最近交付核验 JSON](../../artifacts/unisign-cecsl-eval/followup-20261024/delivery-verification.json)：输入和产物哈希、测试摘要及明确的未通过／NOT_RUN 状态。
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





