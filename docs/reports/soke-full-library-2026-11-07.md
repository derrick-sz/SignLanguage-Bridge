# SOKE 全量参数库与按需展示（2026-11-07）

用户要求持续推进已下载数据的利用。本轮扩展原有100句试产：全部标注身份进入SQLite，帧范围完整的序列逐段校验、转换并建立数值诊断；展示通过分页检索、选中后加载一份NPZ实时重定向，保留原100条制作版GLB。持续目标仍包含原片对齐、非手控表达、残余跳变和词级独立审核，不能由本轮工程转换替代。

## 当前产物与运行

- 数据库/参数根目录：`G:/Archive/Zenodo/SOKE/full-library-20261107/`。
- `corpus.sqlite3`：标注身份、原始逐帧成员位置、来源/模板/协议绑定、逐段状态、参数文件SHA与旋转诊断。
- `sequences/<ID前三位数字>/*.npz`：全部原始8个字段、有效55关节姿态、原始133D特征及连续帧号。没有展开约246万原始PKL小文件。
- `progress.json`与本工作树`artifacts/animation-upgrade/full-library/build.log`记录真实运行。全量构建正常退出0；随后相同命令续跑核验已有文件，`attempted_this_run=0`，无重建、无输入漂移。
- 当前入口：<http://127.0.0.1:8876/animation/>，选择“全量语料”。原100条GLB可在“原有动作”切换，旧8875入口保留。临时8877验证服务已关闭。

| 终态项目 | 数量 |
|---|---:|
| 已索引标注身份 | 20,654 |
| 完整序列已转换并审计 | **19,877** |
| 已验证帧 | **2,359,884** |
| 不完整帧范围 / 缺少序列 | 775 / 2 |
| 转换失败 / 待转换完整序列 | 0 / 0 |
| 已转换不同文本 / 全索引不同文本 | 7,369 / 7,376 |
| 签者 | 10 |
| 已转换序列涉及Gloss标签 / 全索引标签 | 2,000 / 2,000 |

完整帧范围集合转换覆盖100%，全部标注样本覆盖约96.24%；7种文本没有完整实例，清单见`coverage-and-disk.json`。这不是词汇量或语言准确率。库目录实测约3.64GiB（含数据库、成员索引、参数及诊断队列）；首轮索引/转换内部计时761.625秒，未测量功耗或全过程峰值内存/显存。构建中一次观察到的Python工作集约400MB，仅是该时刻快照。

Gloss覆盖按作者标注中空格分隔的原始标签统计，2000个标签全部出现在已转换句段中；**不等于已经制作了2000个有边界、审核通过的词级动作**。当前planner与全量CorpusStore仍要求完整句匹配。实际新句“明天下午三点，我去医院看望刚做完手术的朋友。”在现有/全库计划入口均被阻止，没有改字幕或换相近句冒充自定义生成；证据为`custom-expression-current-contract.json`。

![全库检索与同帧原始/修订人物对照](../../artifacts/animation-upgrade/full-library/preview-full-library.png)

## 转换与完整性

ZIP目录流式读取，成员位置批量写入SQLite，单序列读取共用已打开的源文件；保留逐成员CRC、尺寸、有限值和静态pickle解析检查。19,877段帧范围完整是待转换集合，775段不完整与2段缺失保留状态，不猜测补帧。

构建采用单写者操作系统锁。NPZ先写临时文件、flush/fsync后原子替换，再提交SQLite。续跑重算源包/标注/模板/手均值/处理协议身份，核对已有产物SHA，缺失或损坏重新转换；非有限参数和损坏压缩数据不能发布成可播放项。数据库WAL支持独立只读检索和逐段提交同时运行。

真实CLI对接发现`MotionRegistry.motion()`只返回父子关系与rest坐标，模板摘要被省略。生产者现从已核实模板显式绑定SHA；首轮运行的同骨架元数据只允许在构建锁内补全这一字段，前提是输入协议、骨架/rest坐标完全不变。没有放宽只读端校验，也未从第二个进程修改运行中的数据库。

## 按需展示

CLI新增`--corpus ROOT`。原100条API和PresentationBundle保持独立；新接口位于`/v1/animation/corpus`：状态、`/assets`分页搜索、`/assets/{sample_id}`、`/plans`及`/motions/{sample_id}`。

检索支持句子、Gloss标签和sample ID，并按train/val/test、数值跳变及转换状态筛选。每页30条，上限100；页面搜索请求有防抖、取消和代次校验，翻页不丢弃已选动作。选择完整句后明确确认研究预览；未转换项可查看状态但不能播放。

只读端核验单条NPZ摘要、路径、NPY数组头/展开尺寸、有限值和帧号；单文件及展开内容各限32MiB、每句最多3000帧。原始有效姿态不重复加手均值，按需生成有明确profile的修订候选；动作JSON摘要绑定计划与请求，派生ID包含来源/处理版本，避免缓存复用错误。语言、动作、表情审核始终unverified；没有将NPZ摘要冒充动作JSON摘要。

## 验证证据

- 修复后相关Python：**101 passed**，1条既有Starlette依赖弃用警告；Node动画**31 passed**。
- 修复后全项目测试：**1143 passed、25 skipped、15 warnings**，日志为`final-project-tests.log`。此前1140次通过保留为历史日志；不是实际摄像头、模型准确率或语言验收。
- 最终只读审计核对**19,877段/2,359,884帧**：文件摘要、全部原始参数形状/有限值、帧号、手均值单次恢复、有效关节与133D重拼全部通过，`full_numeric_conversion_complete=true`。另外101,432个原始帧文件属于未转换的不完整序列，未声称这部分已全量数值/CRC审核。
- 原100条在全量库中的所有原始参数、有效姿态、133D特征及帧号，与上一批NPZ逐项严格相等；没有通过更换身份或原始数据取得新覆盖结果。
- 新归档夹具覆盖续跑缺失/损坏、标注或来源漂移、单写者互斥、非有限参数、损坏DEFLATE、原始字段保留、审计区分部分/全量及模板绑定恢复；均有RED→GREEN证据。
- 独立代码审查发现2项Important：畸形pickle的AttributeError可中断健康序列，审计器缺NPY声明形状分配预检。两项均用真实文件夹具复现RED后修复GREEN，再运行上述相关/全项目回归。保留pre-fix报告和修复账本，不把发现记录改写为从未存在。
- 真实浏览器：正式8876入口显示19,877条，30条/页；翻页后仍可播放前页选中的`S000000_P0000_T00`“你们好！”，完整结束并在0.51秒原始/修订同帧近景对照。文本检索“开会”返回6个实例；按ID定位最长完整句`S005786_P0006_T00`，490帧/16.33秒完整播放。检查JSON下载与实际派生动作SHA一致，无控制台warn/error；未烘焙的语料不显示GLB下载入口。

原始数值诊断中19,877段均至少有一个驱动关节相邻变化>30°，>60°事件总计305,434，最大约179.99°；全部进入`quality-review-queue.jsonl`，不是305,434个已确认语言错误。全量离线保留原始数值，修订在单条预览时生成；没有声称全库修订后的改善幅度已经测量，也未把可加载等同于自然、可理解。

已知Minor待后续：超出SQLite整数范围的极大分页offset会返回500，尚未规范为422；正常UI分页不触发。审查未判定的语言理解、原片同步和非手控表达保持未验收；ZIP全局目录歧义仍整体拒绝，逐段可归属错误才隔离。对于可信已验证产物，续跑用已提交SHA核验；匹配SHA但结构异常的文件由独立审计拒绝。

## 继续运行与审核

在当前动画工作树运行以下命令。首次建立索引，后续相同命令核对输入及已有产物后续跑；`--retry-failed`用于明确重试失败项，默认不掩盖错误。

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -u -X utf8 scripts/animation/build_soke_library.py --source-archive G:/Archive/Zenodo/SOKE/CSL-Daily-Fittings/csl-daily_pose.zip --indices E:/githubproject/SignLanguage-Bridge/artifacts/animation-upgrade/soke-sentence-count --base-archive E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip --smplx-model G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz --output G:/Archive/Zenodo/SOKE/full-library-20261107
```

只读产物核查：

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -u -X utf8 -m slbridge.animation.full_corpus_audit --library G:/Archive/Zenodo/SOKE/full-library-20261107 --report artifacts/animation-upgrade/full-library/final-artifact-audit.json
```

预览重启（8876空闲时）：

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -u -X utf8 -m slbridge.animation.cli --archive G:/Archive/Zenodo/SOKE/gesture-corpus-100-20261106/candidate-motion.zip --expected-sha256 36d319957b66562f8a74c6c44c1249bd24daa82987948389ae3e011c3103ddb5 --presentation G:/Archive/Zenodo/SOKE/gesture-corpus-100-20261106/presentation --corpus G:/Archive/Zenodo/SOKE/full-library-20261107 --port 8876
```

原片精确时间映射仍未取得；官方演示时钟不作为源帧对齐证据。调查记录与一手源码入口见[对齐核查](../../artifacts/animation-upgrade/full-library/alignment-investigation.md)。后续应绑定实际拟合输入帧/来源表及原MP4的PTS，估计匹配保持未验证。

本轮只把发布拟合参数变为可检索、可继续制作的句级候选；没有新增文生动作模型训练。全库CRC/数值校验范围以最终审计的已验证帧数为准，不包含缺帧序列中未转成员。面部与下颌、接触/穿插、真实手形含义、词级边界和手语使用者理解仍需继续推进。来源和使用条件沿用[首批报告](soke-corpus-build-2026-11-06.md)，本地转换不等于公开分发或商用授权。

本轮工程计划完成，持续目标保持active，下一阶段围绕可追溯词级候选制作与残余动作/非手控表达推进；专家语言审核和原片精确映射不由程序自动批准。原始数值与既有产物保留，代码未提交/推送。

## 分页整数边界修复附记（2026-11-07）

上文记录的极大offset返回500问题已在源码修复：`CorpusStore.search`在进入SQLite前限制 `0 <= offset <= 2**63-1`，超范围抛出明确参数ValueError，由既有API转换为422；没有改动正常分页或接口结构。RED阶段真实SQLite/TestClient复现 `2**63`、1000位数字以及直接reader调用的3项OverflowError失败；修复后0、正常第二页、`2**63-1`保持200（超出数据末尾为空页），`2**63`、1000位和10000位数字均为422。10000位输入由既有请求解析器直接拒绝。

运行 `pytest tests/test_animation_corpus.py tests/test_animation_api.py tests/test_animation_nonmanual.py -q`：**29 passed**，1条既有Starlette弃用警告，包含7项新增边界用例。未重启运行服务、未改UI或生成代码；当前运行进程是否已加载该修复需在下一次正常服务更新后单独确认。
