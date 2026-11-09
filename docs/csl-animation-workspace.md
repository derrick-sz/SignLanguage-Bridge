# 中国手语动画工作台使用说明

更新：2026-11-08。本页面向新克隆使用者；下列盘符、环境和数据目录均需替换为自己的路径，命令在仓库根目录运行。

面向使用者的入口已改为`/communicate/`，把视频识别、核对、文字回复和数字人表达接成同一对话，见[产品使用说明](communication-product.md)。本页继续说明外部资产与开发工具；`/generate/`保留为文字动作调试入口，不再是产品首页。

本机两条会话的当前成果、统一 8879 入口、v14A 手形/接触修订、扩展词库、最新验证与剩余问题，集中见[动画工作汇总与本地整合](reports/animation-consolidated-status-2026-11-08.md)。快速组合的新句已发现接触残余，不能沿用指定原句的接触通过结论。

| 入口 | 当前功能 | 边界 |
|---|---|---|
| 快速词级组合 `/generate/` 默认模式 | 自由中文经持久文字规划器检索词动作，组合后自动播放 | 1–200字符；缺词或已知锚点遗漏会阻止播放；语言、手形和自然度未验收 |
| 模型生成（较慢），同页显式切换 | 从头运行身体／双手／面部模型，生成后点击播放 | 时长显式选4/6/8/10秒；历史两例约5分钟，本轮未重测完整GPU生成 |
| 原有动作 | 100 段动作、完整表达匹配、确认后播放 | 语言含义未审核 |
| 全量语料 | SOKE CSL-Daily 分页搜索、筛选、选中后加载单条参数 | 检索已有整句，不拼接词语生成新句 |
| SignSparK CLI | 从自定义中文、显式帧数和随机种子生成三维动作候选 | 需外部预训练权重；与网页慢模式共用已认证生成器，语言审核NOT_RUN |

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

## 启动自由文字快速组合主入口

`scene_cli`提供`/generate/`，默认“快速组合”；`/scenes/`重定向到它，原句库保留在`/scenes/library`。快模式为中文→Gloss→词级动作检索→轻量衔接→人物播放，不要求原句命中，也不每次从头生成动作。时长来自词动作；每词参数保留，片段间新增6帧过渡。显示层仍可应用已有修订，不能把参数保留说成与原片画面一致。

先取得作者配套资源，保持下表的固定revision；不要把这些ISO帧边界套到SOKE。资源不随Git发布，下载后仍需完整SHA验证。

| 仓库与固定版本 | 所需相对路径 |
|---|---|
| [LionelLow/SignSparK_Retrieval](https://huggingface.co/LionelLow/SignSparK_Retrieval/tree/a883bc2d13a5d7f85e64591b32bae4cb48950b97)，`a883bc2d13a5d7f85e64591b32bae4cb48950b97` | `annotations/csl_iso.train`；`t2g/CSL-Daily/`中的`model.safetensors`、`config.json`、`generation_config.json`、`sentencepiece.bpe.model`、`signspark_t2g.json`、`special_tokens_map.json`、`tokenizer_config.json` |
| [LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data/tree/9fe7e04ef59204c787cd6094c747f889efff6ba0)，`9fe7e04ef59204c787cd6094c747f889efff6ba0` | `train/CSL-Daily_reopt_train.lmdb/data.mdb` |

在具有NumPy、SciPy及`lmdb`的独立Python环境建库；本机使用下文SignSparK隔离环境，未导入Torch或使用GPU。输出目录必须全新：

```powershell
$GenerationPython = 'D:/envs/signspark/Scripts/python.exe'
$WordAssets = 'D:/sign-assets/SignSparK_Retrieval'
$BaseArchive = 'D:/sign-assets/signbridge-chrome-extension-motion-demo-20261022.zip'

& $GenerationPython scripts/animation/build_word_library.py `
  --lmdb "$WordAssets/train/CSL-Daily_reopt_train.lmdb" `
  --annotations "$WordAssets/annotations/csl_iso.train" `
  --base-archive $BaseArchive --output "$WordAssets/word-library-v1" `
  --expected-lmdb-sha256 7dbc5dc932c6e34ea1d7d47976e8db8fb90ba4c2aff3eb05a6991c7008366222 `
  --expected-annotations-sha256 92cabbe0fcc97770abd491802be00cb35883a2237cabb7f0c5de847845355f2c
```

实际构建得到1,990个标签、5,916个词动作；95个缺源候选被隔离，2条重复标注被排除。七个标签无有效词动作：`45、利益、小姐、成都、瓜、谣言、阅兵`。每词最多3个候选，NPZ保存真实`poses`与`expression`及原视频／帧区间来源。30 FPS仅为产品预览时钟，原始LMDB时间戳／采集FPS未独立核验。

将[配置示例](../configs/animation/word-composition.example.json)复制到例如`D:/sign-assets/word-composition.json`，替换全部路径。规划器的`python`应为装有已验证Torch／Transformers的隔离环境；`script`指当前仓库`text_gloss_worker.py`，`model_dir`指上表完整T2G目录，`word_library`指新建词库。示例使用CUDA；CPU也可配置，但本轮速度仅在实际CUDA环境验证。固定模型的beam4及提示模板由代码绑定，不能从网页改写。

```powershell
$PreviewPython = 'D:/envs/signbridge-preview/Scripts/python.exe'
& $PreviewPython -m slbridge.animation.scene_cli `
  --archive 'D:/sign-assets/signbridge-chrome-extension-motion-demo-20261022.zip' `
  --expected-sha256 4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5 `
  --presentation 'D:/sign-assets/current-verified-presentation' `
  --corpus 'D:/sign-assets/soke-full-library' `
  --composition-config 'D:/sign-assets/word-composition.json' `
  --generation-config 'D:/sign-assets/generation-config.json' --port 8879
```

人物制作包须与基包匹配；`generation-config.json`用于可选慢模式，配置见下一节。页面等待人物与文字规划器准备后可输入1–200字并点击“组合并演示”，完成后自动播放。`GET /v1/animation/composition`可看准备状态，`POST /v1/animation/composition`只接受`{"text":"..."}`；客户端不能指定模型或本地路径。

规划器为持久进程，启动时加载模型并warmup。本机当前beam4服务的一次启动状态记录load约11.43秒、warmup约15.01秒；这不属于每句等待。前三个实际库外短句热HTTP约0.485／0.188／0.127秒，浏览器三句提交至首个playing状态约0.33／0.20／0.20秒，均以人物和模型已准备为前提。它不是首个真实像素／冷启动计时，也不是P95或任意文本保证。

缺词返回`blocked`且无动作，例如“我想去成都。”；已知地点／时间锚点遗漏返回`needs_review`且无动作，例如“明天早上我们一起去医院看朋友。”仍漏“医院”。不自动替换词、不补译或播放半句。该guard只覆盖有限已知锚点，不能证明不存在其他漏译／错译。语言、手形和自然度仍`NOT_RUN`。证据、失败样例与回归范围见[快速组合实施报告](reports/instant-sign-composition-2026-11-08.md)。

## 词动作候选扩容与回退

2026-11-08的后续候选扩容已将本机在线库切到14,317片段，每词最多8例；原有5,916片段的ID和NPZ字节全部保留。词标签仍为1,990，没有新增词义。固定36输入前后均33条ready，未把片段增长当成新句覆盖增长。结果与限制见[扩容报告](reports/word-corpus-expansion-2026-11-08.md)。

沿用上述固定revision，增补`annotations/csl_iso.dev`、`annotations/csl_iso.test`及数据仓库的`dev/CSL-Daily_reopt_dev.lmdb/data.mdb`、`test/CSL-Daily_reopt_test.lmdb/data.mdb`。它们必须与各自split配套。下面示例保留旧库，并在新目录构建／合并：

```powershell
$Retrieval = 'D:/sign-assets/SignSparK_Retrieval'
$WordSourceHashes = @{
  train = @('7dbc5dc932c6e34ea1d7d47976e8db8fb90ba4c2aff3eb05a6991c7008366222', '92cabbe0fcc97770abd491802be00cb35883a2237cabb7f0c5de847845355f2c')
  dev = @('c01bded1a737ba0405d7deedd13de3c6d8173584daadc47682b83449b93bd6d5', '93ba7e1757ef83a65878c3371ec340099854bd87ea2c4046e327538db1378ec5')
  test = @('c9a851b879925880eb6f87cbbb6495361c86d158dbede2ccf0f31a2eceb4319f', '901ac0ac89755c0500f029e0235dccc2149ebfaae2ce068ffd52838abc9975a0')
}
foreach ($splitName in @('train', 'dev', 'test')) {
  & $GenerationPython scripts/animation/build_word_library.py `
    --lmdb "$Retrieval/$splitName/CSL-Daily_reopt_$splitName.lmdb" `
    --annotations "$Retrieval/annotations/csl_iso.$splitName" `
    --base-archive $BaseArchive --split $splitName --max-candidates 8 `
    --expected-lmdb-sha256 $WordSourceHashes[$splitName][0] `
    --expected-annotations-sha256 $WordSourceHashes[$splitName][1] `
    --output "$Retrieval/word-library-$splitName-8"
  if ($LASTEXITCODE -ne 0) { throw 'Word library build failed; stop before merging' }
}
& $GenerationPython scripts/animation/expand_word_library.py `
  --base "$Retrieval/word-library-v1" `
  --addition "$Retrieval/word-library-train-8" `
  --addition "$Retrieval/word-library-dev-8" `
  --addition "$Retrieval/word-library-test-8" `
  --output "$Retrieval/word-library-v2-expanded" --max-candidates 8
```

确认构建成功后，在服务端配置中将`word_library`改为新库目录并重启本任务服务；回退时指回旧目录。不要直接改动正在使用的manifest或NPZ。新池保留train/dev/test及各源摘要，标记`usage_role=product-demo`、`evaluation_eligible=false`；不能用它宣称同源样本的独立留出成绩。

## 可选：模型生成（较慢）

在`/generate/`显式切换到“模型生成（较慢）”后，自由文字进入完整文本条件生成，不因库内没有整句而阻断。页面显示实际身体／双手／面部阶段，完成校验后加载人物并点击播放；可暂停、停止和重播。修改输入或取消会使旧请求失效。以下为此前已验证配置，本轮没有重跑完整慢模型GPU验收。

先准备前述资产及下文固定的SignSparK源码、权重和文本资产。将服务器专用配置保存为例如`D:/sign-assets/generation-config.json`，替换其中全部路径；`worker_script`必须指向当前仓库中的脚本。网页请求只传文字与秒数，不能选择这些路径。

```json
{
  "python": "D:/envs/signspark/Scripts/python.exe",
  "worker_script": "D:/SignLanguage-Bridge/scripts/animation/generate_custom_signspark.py",
  "source": "D:/sign-assets/SignSparK/source",
  "source_manifest": "D:/SignLanguage-Bridge/configs/signspark/source-manifest.json",
  "text_assets": "D:/sign-assets/SignSparK/text-assets",
  "models": "D:/sign-assets/SignSparK/models",
  "base_archive": "D:/sign-assets/signbridge-chrome-extension-motion-demo-20261022.zip",
  "source_presentation": "D:/sign-assets/calibrated-presentation",
  "smplx_model": "D:/sign-assets/SMPLX_NEUTRAL.npz",
  "output_root": "D:/sign-assets/web-generation",
  "readonly_checkpoint_mmap": true,
  "cpu_offload": true
}
```

```powershell
$PreviewPython = 'D:/envs/signbridge-preview/Scripts/python.exe'
& $PreviewPython -m slbridge.animation.scene_cli `
  --archive 'D:/sign-assets/gesture-corpus-100-20261106/candidate-motion.zip' `
  --expected-sha256 36d319957b66562f8a74c6c44c1249bd24daa82987948389ae3e011c3103ddb5 `
  --presentation 'D:/sign-assets/calibrated-presentation' `
  --corpus 'D:/sign-assets/soke-full-library' `
  --generation-config 'D:/sign-assets/generation-config.json' --port 8879
```

访问`http://127.0.0.1:8879/generate/`。正文1–200字符，换行／制表符规范为空格；时长可选4／6／8／10秒，对应120／180／240／300帧，固定50步、seed123。时长来自显式选择。一次只运行一个GPU worker，非终态队列最多8个。两条实际库外新句的4秒演示在本机各需约5分钟生成；其他时长、硬件与负载的耗时需实测。

上述配置使用`fp32-layer-offload-cached-text-v1`低提交额度执行路径：经SHA及strict校验的权重使用OS只读映射，原始文本编码器在CPU FP32生成缓存，再将各动作层按需搬到CUDA；没有改成低精度或随机文本编码器。只读映射限定已检查的PyTorch 2.9.1接口。它降低本轮Windows提交额度压力，不构成CPU/CUDA文本编码逐位等价或任何主机内存必足的保证；manifest记录实际执行配置和加载结果。薄服务只装载动画／生成路由，不初始化识别运行时。

2026-11-08已从网页实际生成两条库外新句，均完成真实GPU任务及人物播放；字幕绑定生成clip文字，语言、手形和自然度仍为`NOT_RUN`，两条仍有跳变警示。当前实现与证据边界见[实测报告](reports/free-text-sign-demo-2026-11-08.md)、[自由文字设计](superpowers/specs/2026-11-08-free-text-sign-demo-design.md)和[执行计划](superpowers/plans/2026-11-08-free-text-sign-demo.md)。以下原有`slbridge.animation.cli`整句检索入口继续保留。

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

原有整句检索／规划器没有完整表达资源时保留原文并阻止播放，不截取前半句、不换成相近句。例如“认识你很高兴不要”不能冒充“认识你很高兴”。自由文字新句可使用前述`/generate/`网页或下一节CLI；该旧库阻断规则不适用于自由生成。

## SignSparK：CLI 生成自定义中文候选

这条路径使用公开预训练生成器，不训练新模型，不向模型提供目标动作或已观测关键帧；帧数由 `--frames` 明确指定，并非从文本自动预测。网页慢模式接入同一生成器；默认快速模式已独立接入词级检索组合，但尚未把词动作锚点送入此生成器做过渡精修。以下run02数值是此前CLI实验的独立记录。

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
