# 中国手语动画工作台使用说明

更新：2026-11-07。本页面向新克隆使用者；下列盘符、环境和数据目录均需替换为自己的路径，命令在仓库根目录运行。

| 入口 | 当前功能 | 边界 |
|---|---|---|
| 原有动作 | 100 段动作、完整表达匹配、确认后播放 | 语言含义未审核 |
| 全量语料 | SOKE CSL-Daily 分页搜索、筛选、选中后加载单条参数 | 检索已有整句，不拼接词语生成新句 |
| SignSparK CLI | 从自定义中文、显式帧数和随机种子生成三维动作候选 | 需外部预训练权重；浏览器只能预览产出，尚无任意文本 UI 生成 |

## 先准备外部资产

Git 仓库包含转换器、播放器、生成脚本和小型身份清单；**不包含数据集、模型权重、人体模板、动作 ZIP、人物 GLB、生成结果或本机 `artifacts/` 证据**。仅克隆代码不能立即播放或推理。普通预览使用安装了项目 `api` 可选依赖及 NumPy／SciPy 的 Python；SignSparK 另用隔离环境，见后文。

| 资产 | 来源与身份 |
|---|---|
| 原 100 条动作基包 | `signbridge-chrome-extension-motion-demo-20261022.zip`，SHA256 `4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`；已有资产的来源说明见[资源审计](research/csl-reusable-resources-2026-11-06.md)。它是外部项目资产，不是本仓库自动下载项 |
| 可选 SOKE 100 条手势修订包 | `gesture-corpus-100-20261106/candidate-motion.zip`，SHA256 `36d319957b66562f8a74c6c44c1249bd24daa82987948389ae3e011c3103ddb5`；与下列校准人物制作包配套，是外部派生产物，不等于原 100 条基包 |
| SOKE 参数包与标注 | SOKE 官方 Data 入口的 CSL-Daily-Fittings ZIP，以及同一发布的 `csl_clean.train`、`csl_clean.val`、`csl_clean.test`；来源链接与下载身份见[构建报告](reports/soke-corpus-build-2026-11-06.md)。本次 ZIP SHA256 `cf026121ef61c0da0ffab8f9d861405cd0a621f004ba77cddb4d54cfec786229` |
| SMPL-X neutral 模板 | 官方 Human Models 入口及提取方式见上述构建报告；`SMPLX_NEUTRAL.npz` SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` |
| 已校准人物制作包 | 含 `manifest.json` 与 `avatar-clean.glb`。新句生成当前固定使用已验证身份：manifest SHA256 `a9fa82f145861c731d6aadc63957df549e7008fb701cb282a5f8dcd7143d80c8`，GLB SHA256 `b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c`；制作背景见[人物升级报告](reports/animation-quality-upgrade-2026-11-06.md) |

请按每个来源分别核对许可。SOKE 仓库标注 CC BY-NC-ND 4.0；CSL-Daily 数据、SMPL-X、角色及 SignSparK 权重各有自己的条件。公开下载、技术转换成功及本仓库代码许可，均不代表这些资产的商用或再分发权限。

## 启动原有动作与全量语料

先准备变量（示例路径不要求采用相同盘符）：

```powershell
$PreviewPython = 'D:/envs/signbridge-preview/Scripts/python.exe'
$BaseArchive = 'D:/sign-assets/signbridge-chrome-extension-motion-demo-20261022.zip'
$BaseSha = '4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5'
$Presentation = 'D:/sign-assets/calibrated-presentation'
$Corpus = 'D:/sign-assets/soke-full-library'

# 仅原有 100 条；不需要生成模型或 GPU
& $PreviewPython -m slbridge.animation.cli --archive $BaseArchive --expected-sha256 $BaseSha --port 8874

# 完成全库构建后，在另一终端启动带分页检索的入口（重新设置上述变量）
& $PreviewPython -m slbridge.animation.cli --archive $BaseArchive --expected-sha256 $BaseSha --corpus $Corpus --port 8876
```

浏览器访问 `http://127.0.0.1:8874/animation/` 或所选端口。服务只监听本机 loopback；同一端口只运行一个服务。`--presentation` 可省略，但校准人物和对应嘴部候选需要匹配的制作包。不要把不同动作归档的制作包混用：动作、人物与处理协议均有摘要绑定。

若已取得配套的 SOKE 100 条手势修订包和上述校准人物包，使用下面的组合替代第二条启动命令；原基包仍可在独立端口使用。这也是本次全库嘴部预览使用的组合：

```powershell
& $PreviewPython -m slbridge.animation.cli `
  --archive 'D:/sign-assets/gesture-corpus-100-20261106/candidate-motion.zip' `
  --expected-sha256 36d319957b66562f8a74c6c44c1249bd24daa82987948389ae3e011c3103ddb5 `
  --presentation $Presentation --corpus $Corpus --port 8876
```

首次构建全库（已有有效库可跳过）：

```powershell
& $PreviewPython scripts/animation/build_soke_library.py `
  --source-archive 'D:/sign-assets/CSL-Daily-Fittings.zip' `
  --indices 'D:/sign-assets/soke-indices' `
  --base-archive $BaseArchive `
  --smplx-model 'D:/sign-assets/SMPLX_NEUTRAL.npz' `
  --output $Corpus
```

`--indices` 目录须含上述三个 `csl_clean.*` 文件。输出为 `corpus.sqlite3`、`sequences/<分片>/<ID>.npz` 与进度记录；相同输入可续跑，已有文件仍核验身份和摘要。可先加 `--limit 20` 做小规模构建，再去掉限制续跑；不要同时启动两个写入者。

实测 20,654 条索引中，**19,877 段、2,359,884 帧**完整转换；775 段帧号或数量不符、2 段缺失，保留诊断状态，不补帧。原始参数与有效姿态分开保存，SOKE 手均值只加一次。2,000 个 Gloss 标签是整句标注覆盖，未建立对应的词级时间边界或审核后的词片段。30 FPS 的依据及原片裁剪尚未对齐的边界见[全量库报告](reports/soke-full-library-2026-11-07.md)和[首批构建报告](reports/soke-corpus-build-2026-11-06.md)。

## 检索、修订与嘴部预览

1. 在“原有动作”或“全量语料”选择整句。全库支持中文、Gloss、样本 ID 搜索，以及划分／跳变线索筛选和分页；缺帧项可看状态，不能播放。
2. 明确确认后预览，使用暂停、重播、时间轴、速度、正面／侧面／近景检查；“同时对照”共享时间轴。
3. 查看“动作来源与显示处理”。原始有效姿态保留，手势修订派生新候选并绑定来源摘要；常规修订限制手指 3°、身体 5°，较大的孤立跳变修复单独留痕，不代表手形或句义已被纠正。
4. 匹配人物与动作协议时，可显式打开“源数据嘴部候选”。它从 SMPL-X jaw 提取受限开口量；默认关闭，手动下颌检查可覆盖，恢复按钮释放覆盖。旧动作无该协议时不会凭空补出嘴部轨。

全库只在选中时读取一条 NPZ，不将近两万段一次性装入 `MotionRegistry`；每条最多 3,000 帧，文件及展开参数各限 32 MiB。数值“未见大跳变”不等于语言、手形、接触或碰撞通过。

完整面部表达尚未重定向：生成的原生 **50 维 expression 保存在参数中**，`expression` 与 `jaw_swing` 仍标为 unmapped。下颌开合候选不等于口语音素、完整嘴形或 CSL 非手控语法。**包含新下颌轨的 GLB 导出不支持**；原制作包已有的烘焙 GLB 下载不包含这条实时轨。详见[嘴部候选报告](reports/nonmanual-jaw-candidate-2026-11-07.md)。

没有完整表达资源时，网页规划器保留原文并阻止播放，不截取前半句、不换成相近句。例如“认识你很高兴不要”不能冒充“认识你很高兴”。需要真正生成新句时使用下一节 CLI。

## SignSparK：CLI 生成自定义中文候选

这条路径使用公开预训练生成器，不训练新模型，不向模型提供目标动作或已观测关键帧；帧数由 `--frames` 明确指定，并非从文本自动预测。检索增强、词典锚点与任意文本网页生成尚未接入。

### 固定资产与隔离环境

源码来自作者 `JianHe0628/SignSparK`，固定 commit `916402d0ecd6322fb4a9e5f513229a94a57bf4b6`。克隆到任意外部目录时保留原始 LF 文件内容，避免 Windows 自动 CRLF 改写使 blob 校验失败：

```powershell
git -c core.autocrlf=false clone https://github.com/JianHe0628/SignSparK.git D:/sign-assets/SignSparK/source
git -C D:/sign-assets/SignSparK/source -c core.autocrlf=false checkout 916402d0ecd6322fb4a9e5f513229a94a57bf4b6
```

认证小清单位于本仓库 `configs/signspark/source-manifest.json` 与 `configs/signspark/text-assets-manifest.json`，记录固定源码文件／文本资产身份，不能通过手改摘要适配未知版本。文本资产按清单的相对目录放在 `text-assets/mclip/` 与 `text-assets/xlm-roberta-large/`：

| 上游 Hugging Face 仓库 | 固定 revision | 所需文件 |
|---|---|---|
| `LionelLow/SignSparK` | `b907f25f9bb9f5a0adcaf131a23a155dc471d9a8` | `body/`、`hand/`、`face/` 各自的 `ema_0.9999_200000.pt`，合计约 16.74 GB；左右手共用 hand 权重 |
| `M-CLIP/XLM-Roberta-Large-Vit-B-16Plus` | `15c3aee9b9e96fd7de1c1d488810a604bab98e78` | 小型配置与 tokenizer 文件，按认证清单获取 |
| `FacebookAI/xlm-roberta-large` | `c23d21b0620b635a76227c604d44e43a9f0ee389` | `modelBase=xlm-roberta-large` 对应配置与 tokenizer 文件，按认证清单获取 |

文本配置／tokenizer 共 9 个文件、约 28.3 MB，可由 Hugging Face 的 `https://huggingface.co/<仓库>/resolve/<固定revision>/<文件名>` 地址取得；全部通过身份校验后仅本地读取。完整 EMA 已包含冻结文本编码器和 1024→640 投影，strict 加载全部参数后不需要另下重复的 M-CLIP／XLM-R 大权重，更不能用随机编码器替代。

使用**独立 Python 3.10 环境**，不要升级原 Uni-Sign 环境来兼容生成器。实际成功环境为 Python 3.10.22；关键版本如下（来自本机 constraints、安装锁定记录及运行环境查询，非自动安装全量清单）：

```text
torch==2.9.1+cu128            torchvision==0.24.1+cu128
numpy==1.26.3                scipy==1.15.3
transformers==4.56.1         tokenizers==0.22.1
accelerate==0.29.2           multilingual-clip==1.0.10
sentence-transformers==5.1.2 sentencepiece==0.1.97
torchdiffeq==0.2.5           einops==0.8.1
hydra-core==1.3.2            omegaconf==2.3.0
lmdb==2.2.0                 wandb==0.21.3
blobfile==3.0.0
```

本仓库的 `unisign` 可选依赖仍固定另一组 Transformers 版本，不适合作为上述环境的安装命令。需另满足官方源码导入依赖及项目依赖，并先核对环境；单条 `pip install -e .` 不会配全此生成环境。真实推理在 RTX 5060 Ti 16 GB 上按身体／手／脸顺序运行 FP32；首次尝试曾因内存分配失败退出，后续采用空参数构造、CPU 非持久缓冲区及完整 strict 权重加载成功。16 GB 显存不构成任意主机可用 RAM／提交额度的保证。

### 下载、生成与预览命令

先完成上面的资产准备，并沿用前面的 `$PreviewPython`、`$BaseArchive`、`$Presentation` 变量；以下命令不会自动补齐源码、文本资产、基包、模板或校准人物。模型下载器需要 `curl`，采用固定 revision、分块续传及完整 SHA256 验证，预留权重和分块／组装临时空间。

```powershell
$GenerationPython = 'D:/envs/signspark/Scripts/python.exe'
$SignSparK = 'D:/sign-assets/SignSparK'
$Generated = 'D:/sign-assets/generated-csl-run01'  # 必须是新的输出目录

& $GenerationPython scripts/animation/download_signspark_models.py --output "$SignSparK/models"

& $GenerationPython scripts/animation/generate_custom_signspark.py `
  --text '明天下午我去医院看朋友。' --text '今天我在家休息。' `
  --frames 128 --steps 50 --seed 123 `
  --source "$SignSparK/source" `
  --source-manifest 'configs/signspark/source-manifest.json' `
  --text-assets "$SignSparK/text-assets" --models "$SignSparK/models" `
  --base-archive $BaseArchive --source-presentation $Presentation `
  --smplx-model 'D:/sign-assets/SMPLX_NEUTRAL.npz' --output $Generated

# 仅在 generation-manifest.json 的 state 为 complete 后启动候选预览
& $PreviewPython -m slbridge.animation.cli `
  --archive "$Generated/candidate-motion.zip" `
  --presentation "$Generated/presentation" --port 8878
```

可以替换 `--text`，可重复传入多句；`--frames` 范围 16–304，`--steps` 范围 1–100。每次使用新输出目录。输出包括 `generation-manifest.json`、逐请求的 `raw-stream-*-{body,left,right,face}.npz`、55 关节姿态与 expression 参数、候选动作 JSON／ZIP，以及仅人物与新动作绑定信息的 `presentation/`。每个原始流完成后立即保存；失败目录保留诊断，不当成完整结果展示。SignSparK 手流已是其 flat-hand 约定，转换时不再套用 SOKE 手均值。

### 已完成验证与限制

本机 run02 实际完成上述两句，每句 128 帧、50 步，共 **256 帧**。身体／手／脸三个 checkpoint 各 **859 个张量 strict 加载**，missing／unexpected 均为空；目标动作为空、已观测关键帧数为 0。完全屏蔽的观察量负控差异为 0；同种子不同文本生成不同且非静态的身体轨。与失败 run01 已完成的 6 条身体／左右手原始流逐项比较，均 bitwise 相同；run01 缺脸部流，仍保持 failed。

原 run02 完成后发现展示目录遗漏跳变警示，已另派生 `custom-csl-20261107-run02-preview-v2`，保留原 run02。**通过产物一致性审计的是该 preview-v2**：原始流 SHA、最终 NPZ／JSON／registry、重解码一致性、人物与新动作摘要绑定均通过。两句修订后最大相邻帧变化仍约 **37.60°／40.90°**，分别有 2／1 个超过 30° 的事件，预览明确标为有跳变线索。

上述是功能与来源一致性验证，语言、动作及表情审核仍为 **NOT_RUN**，不能推出句义正确、自然流畅或可商用。完整本机证据位于 `artifacts/animation-upgrade/custom-expression/`（不随 Git）：`runtime-constraints.txt`、`runtime-local-lock.txt`、`run02-preview-final-artifact-audit.json` 及原始失败／修订审计记录；实际运行目录位于本机外部数据盘。新克隆需自行运行并保留自己的 manifest，不能把这些本机路径当成已下载资产。

可随Git查看的运行摘要、截图、资源实测与回归范围见[本轮生成报告](reports/custom-expression-generation-2026-11-07.md)。[requirements-signspark.txt](../requirements-signspark.txt)记录本机验证的主要依赖版本，不是完整传递依赖锁；建议在独立Python3.10环境使用。两句同seed试点可运行`python scripts/animation/audit_signspark_pilot.py --run-dir <输出目录> --output <目录外的新JSON>`，执行只读一致性审计。

## 从现有识别结果进入

网页“接入识别结果”接受已登记模型输出；结果进入待核对输入框，人工修订不当成模型正确输出。启动服务时可加 `--catalog <本地catalog.json> --model csl`，catalog 格式如下；识别仍使用已配置 CUDA 和识别依赖的环境：

```json
{"schema_version":1,"models":{"csl":"D:/sign-assets/uni_sign/model_bundle"}}
```

Qt 桌面结果区“在动画工作台核对”将 CSL 文本／Gloss 候选送入浏览器草稿，不自动播放。默认地址为 `http://127.0.0.1:8874/animation/`，可用 `SLBRIDGE_ANIMATION_URL` 指向同机其他端口；独立 SLRT 专用面板可手动复制 Gloss。识别模型详情见[Uni-Sign 集成说明](uni-sign-integration.md)，历史工作台范围见[首期实施记录](reports/csl-animation-workspace-acceptance-2026-10-28.md)。

普通回放和新句候选预览不要求 Blender。需要重新烘焙旧制作包、改网格／蒙皮或角色时，使用离线制作流程并重新验证导出；见[人物升级报告](reports/animation-quality-upgrade-2026-11-06.md)和[制作交接](animation-dot-blender-handoff-2026-11-06.md)。
