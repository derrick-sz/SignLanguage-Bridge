# SOKE CSL-Daily 下载与首批三维语料构建（2026-11-06）

同日后续按用户“1+2”完成手势修订并扩展100条的工作见[后续报告](gesture-refinement-expansion-2026-11-06.md)。本文保留首批20条的条件和当时上限，当前CLI/展示资源上限以后续报告为准。

已下载官方三维拟合包，完成整包目录与发布索引的覆盖核对，并构建 20 段句级候选、数值参数与独立人物动画。当前交付是可查看和继续修订的研究候选语料，不是语言/动作审核通过的中文手语词典。

## 产物入口

- 本地预览：<http://127.0.0.1:8876/animation/>，20 段；原有 8875 工作台未替换。
- 原始数据：`G:/Archive/Zenodo/SOKE/CSL-Daily-Fittings/csl-daily_pose.zip`。
- 构建目录：`G:/Archive/Zenodo/SOKE/corpus-candidate-20261106-r1/`。
- `corpus.jsonl`：全部 20,654 个索引样本的句子、Gloss、划分、帧覆盖状态和本地视频文件名存在状态。
- `sequences/*.npz` + 同名 JSON：20 段原始身体/双手/下颌/形体/表情/相机参数、原始 133D 特征、有效 55 关节姿态、帧号与来源哈希。
- `candidate-motion.zip` 及 `.sha256`：独立候选动作包，可由现有 MotionRegistry 加载。
- `presentation/`：复用现有人物的 20 个 GLB、同钟运行时轨、几何检查、导出一致性结果和 manifest。

本轮 G 盘 SOKE 目录合计约 **3.665 GiB**，包括原始拟合 ZIP、模板 ZIP、单独 neutral 模板及候选产物；当前候选目录约 **94.1 MiB**。这些是文件体积统计，没有测量整机功耗或峰值内存/显存。

## 真实覆盖范围

| 检查 | 数量 | 含义 |
|---|---:|---|
| 发布索引样本 | 20,654 | train/val/test 标注总数 |
| ZIP 中实际序列 | 20,652 | 实际存在逐帧参数成员的 sample ID |
| 帧号与索引帧数完整 | 19,877 | 从 0 开始、无缺号、帧数等于索引；未等同于全序列数值或语言通过 |
| 帧数/帧号不完整 | 775 | 保留为待处理；首批不补帧、不纳入 |
| 索引存在但 ZIP 无序列 | 2 | `S003751_P0000_T00`、`S005362_P0000_T00` |
| 实际逐帧参数文件 | 2,461,316 | ZIP64 总目录项 2,481,971 包含目录，不能算作语句数 |
| 已转换/导出候选 | 20 段、2,562 帧、10 位签者 | 不是全库转换或审核通过数量 |

目录采用流式扫描，没有在 Windows 全量展开约 248 万个小文件。对选中完整序列才读取、解压并核对每个成员 CRC 和参数形状/有限值；全包每个成员的 CRC 与数值检查 **NOT_RUN**。三个试产候选因不完整被排除，记录于 `excluded-pilot-candidates.json`。

## 下载与模板

来源为 [SOKE 官方项目页](https://2000zrl.github.io/soke/)链接的 [CSL-Daily-Fittings](https://www.dropbox.com/scl/fo/3lwcdq60q8yg3foj6ulh0/AOw8pBv6LEk0JR0KvgVR-m4?rlkey=9ewwux5vnslljv88hbb5ibhds&dl=0)。ZIP 为 2,886,912,220 B（约 2.69 GiB），本次下载/本地哈希流程记录 161.4 s，本地 SHA256：

`cf026121ef61c0da0ffab8f9d861405cd0a621f004ba77cddb4d54cfec786229`

未取得作者发布的独立校验和，不能将本地哈希等同于作者签名认证。

为正确恢复手形，另从 [SOKE 官方 Human Models 入口](https://github.com/2000ZRL/SOKE#human-models)取得公开模板包 `smpl_models.zip`，840,847,898 B，本地 SHA `ca511fd61d4c01034e782f1b11a49cd2456f708615aefe1c74f7238eb42d2833`，记录 271.4 s；只取其中 `SMPLX_NEUTRAL.npz`，108,752,058 B。没有登录、提交访问申请或接受额外条款表单；Google Drive 的公开大文件下载确认仅用于取得数据。

neutral 模板 SHA `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` 与已有骨架来源完全一致；重算 rest joints 与原播放器骨架最大差 `2.98e-7`。`hands_meanl/r` 为有限 `(2,45)` 数组，按 [SOKE 参数解释](https://github.com/2000ZRL/SOKE/blob/main/mGPT/utils/human_models.py)及现有播放器的 non-flat hand convention 加入一次，原始手参数保持不变。

静态 pickle 解释器不导入或调用 `GLOBAL/REDUCE`，numpy 数组仅由验证后的浮点 dtype、尺寸与字节载荷重建；标注内嵌 torch 内容保持惰性数据。NPZ 以 `allow_pickle=False` 可读。模板与公开拟合数据没有变成训练好的中文文生动作权重。

## 时间与显示边界

本机 MP4 为未裁剪原片，首批 20 段都与发布参数帧数不同。复现例：`S000000_P0000_T00` 参数为 52 帧，本机 MP4 为 168 帧、30 FPS。此前集成文档也明确这些视频为未裁剪原始 MP4。第一轮按“帧数必须相同”严格拒绝所有候选，保留其输出目录；随后将同名原片绑定与时间对齐分开。

本次没有拉伸、截断或猜测裁剪偏移。参数按 [CSL-Daily 原始论文 §4.1](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf)的 native **30 FPS** 播放；发布包没有逐帧时间戳供本轮独立验证，`pose_fps_basis` 明示这一依据。原片帧数已实查，时间对齐仍为 **unverified / NOT_RUN**。

播放采用 [SOKE 的 upper-body 表示](https://github.com/2000ZRL/SOKE/blob/main/mGPT/data/H2S.py)：root 和关节 1..11 归零，12..21 保留，手加均值，眼部静止；原始 root/body/camera 全部另存。Godette 面部系数、下颌与形体参数没有完整重定向，当前人物表情保持原工作台的中性处理，不代表非手控语言表达已实现。

20 段都有相邻关节旋转超过 30° 的数值线索，最大约 **171.52°**。这不是语言错误率；也不能据“参数有限/导出通过”称动作自然。此次沿用保守显示处理，手指校正不超过 3°、骨长变化 0，保留问题供对照与后续修订，没有用强平滑冒充手形修复。

## 验证结果

- 相关 Python 回归 **58 passed**，含新增静态解析、缺帧拒绝、CRC、均值单次应用、候选审核隔离、时间绑定及烘焙批量入口限制；不是全项目模型测试。
- Node 动画回归 **26 passed**。
- 20 个 GLB 全部 2,562 帧与运行时目标关节位置一致性通过，最大差 `3.0446458e-6`；这验证烘焙/导出，不验证原片动作或语义。
- fresh-context review 独立核对 NPZ 哈希、raw→有效姿态→ZIP 全等、模板来源、PresentationBundle 与 GLB 时间轨，最大时间差 `2.22524e-7 s`。
- 浏览器实际完整播放“老师们每周五下午都会开会。”，检查 1.6 s 原始/制作同钟对照、导出下载，控制台无 error/warn。下载 GLB SHA 与 manifest 一致：`21ebd37db2c075e13266473d51ab849530ac713667a04ec191d8e3efdf34cb1c`。
- 专家手语/词义、动作可理解性、原片精确对齐、完整非手控表达及全库数值验收均 **NOT_RUN**。这批是完整短句候选，尚未切成审核词条。

![开会短句在 1.6 秒的原始与制作人物对照](../../artifacts/animation-upgrade/soke-build/preview-meeting.png)

独立审查指出 `--count 21..40` 与烘焙上限 20 不一致。已按会导致合法输入在昂贵烘焙后失败处理，增加入口提前拒绝，并做 RED→GREEN 回归；原始参数批次仍支持至 40，带人物烘焙每批最多 20。没有遗留 Critical/Important 或延期 Minor。

## 复用方式与下一步

角色、播放器和已有 100 段素材保持原样，新增包使用独立服务。后续先补原片裁剪对齐，再挑选跳变明显和复杂接触短句修订、由手语使用者审核，通过后才分词或扩大转换数量。

可重新启动候选工作台：

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -m slbridge.animation.cli --archive G:/Archive/Zenodo/SOKE/corpus-candidate-20261106-r1/candidate-motion.zip --expected-sha256 2bdd06f3c34c59dbc21e6abae03f15b5903c9a6e3a23d8413b067c3e934035b1 --presentation G:/Archive/Zenodo/SOKE/corpus-candidate-20261106-r1/presentation --port 8876
```

命令在当前动画工作树运行。构建命令接口见 `scripts/animation/build_soke_corpus.py --help`，新输出目录不会覆盖已有产物。源码和报告保留在 `codex/csl-animation-sync-20261106` 工作树，未提交或发布。

使用条件分别跟随 SOKE 仓库、CSL-Daily 数据、人体模板及 Godette 角色；[SOKE 仓库许可](https://github.com/2000ZRL/SOKE/blob/main/license.txt)为 CC BY-NC-ND 4.0，不能把本地候选的生成视为公开分发或商用授权。

证据位于 `artifacts/animation-upgrade/soke-build/` 的 download-status.json、model-download.json、hand-mean-validation.json、browser-verification.json、final-review.md、progress.md；构建目录保有完整 summary、inventory、selected、geometry-report、export-verification 和原始参数。
