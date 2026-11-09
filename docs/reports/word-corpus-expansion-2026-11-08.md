# 词级动作候选库扩容与复测

日期：2026-11-08，Asia/Taipei。代码与证据在动画功能工作树`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，未提交／推送，未改并发人物修订或识别评测产物。

本轮已把在线词级动作库从**5,916片段扩至14,317片段，新增8,401段真实动作候选**，每词上限3例扩大为8例。**词标签仍为1,990种；没有新增词义标签。** 新库已接入8879服务的快速组合入口，旧库与旧配置保留可回退。

## 扩容内容与来源

| 项目 | 旧库 | 当前库 |
|---|---:|---:|
| 词标签 | 1,990 | 1,990 |
| 动作片段 | 5,916 | 14,317 |
| 每词候选上限 | 3 | 8 |
| 固定36句中ready状态 | 33 | 33 |

新增来自作者固定版本的配套CSL-Daily train/dev/test原生LMDB及各自ISO词边界。dev/test四个增量文件约430MB，逐文件核验完整大小与SHA；下载已正常结束。具体身份见本机`artifacts/corpus-expansion-20261108/asset-inventory.json`及`annotation-inventory.json`。未读取未完成文件，也未把边界套入同名SOKE片段。

先导出train8=14,299、dev8=4,628、test8=4,973候选，再合并：保留原库全部5,916单位；余下名额从不同源轮流补入，重复源区间去重。最终来源为train **10,410**、dev **2,341**、test **1,566**。每条保留原split、源库摘要、原动作文件摘要及`[start,end)`区间；逐文件核对5,916旧NPZ字节完全一致。

新库本机目录：`G:/Archive/Zenodo/SignSparK_Retrieval/word-library-v2-expanded-20261108`。

- bundle：`csl-product-95379e49911c11908e420a94`
- manifest SHA256：`98f17a0e3fa55547dcb26aec8d08f765d45ca2e1ddbe7e46fd1930d9c1e2db28`
- `usage_role=product-demo`，`evaluation_eligible=false`

它是产品演示素材池，包含dev/test来源，不能再以这些源样本宣称独立留出泛化。T2G模型、beam4、原文输入和有限遗漏检查未改；没有训练新模型。30FPS仍为显式预览时钟，原始采集时基未独立认证；源帧保留仅指参数层，既有播放器仍可执行显示修订。

## 可播放范围没有扩大

三份ISO的标签并集仍是1,997，原先缺少的`45、利益、小姐、成都、瓜、谣言、阅兵`七个标签，在三套配套LMDB中都找不到对应的21个原train源ID。全量键检查发现train有200个缺源ID／1,698词例；此前建库遇到的95次失败只是按每词3候选遍历时的子集，不能当全量缺失总数。

固定36条输入保持原文、输入文件摘要及模型身份相同。扩容前后都为：**33 ready、1 needs_review、1 blocked、1 HTTP422**，没有ready退化，也没有新增ready。ready是工程可播放状态，不是正确翻译或语言准确率。

- “明天早上我们一起去医院看朋友。”仍因规划漏“医院”待核对。
- “我想去成都。”仍因缺“成都”动作阻断。
- 数字45探针仍未通过文字规划，不能靠把4和5拼接冒充已审核的45表达。

33条ready输入中，**27条实际选用了新增片段，共52次**。这说明新增候选已进入在线选择，不能据此声称手形、语义、衔接或自然度改善。相关人工验收仍为NOT_RUN。

本次热HTTP往返样本中位数由124.71ms变为144.67ms，最大值由181.63ms变为216.71ms；它们包含规划、组合与HTTP，不是浏览器首像素或跨环境性能保证。浏览器“请等一下。”“我想喝水。”均实际播放至结束，页面预热提交至playing回调为0.15秒／0.23秒，后者重复提交为0.16秒；两句在HTTP审计中均实际选用新增单位。亚秒级第一句的页面整数时钟显示0:00，实际帧数在HTTP记录中。成都缺词仍禁播，console error/warn为空；原生CUA截图已观察，未导出本地文件。浏览器记录另存`browser-cua-audit.json`。

## 实现与定向验证

`build_word_library`新增显式split和最多8候选，已知官方SHA对不能被改标为其他split。新增`merge_word_libraries`及`expand_word_library.py`，校验骨架、FPS、安全路径、NPZ数值及SHA；同ID冲突拒绝，原库不能被截断；最多16来源／20,000单位／8候选／16MiB索引。新manifest只在复制与复核完成后发布，旧库不改。

运行库接受显式产品来源绑定的dev/test单位；逐条核对源库ID、manifest SHA、LMDB／ISO SHA及split，保留旧train-only库兼容和有界缓存。缺词、遗漏、输入失效与旧结果阻断逻辑保持。

- 预览环境定向Python：**117 passed**，1条Starlette第三方弃用警告。
- 原生builder／merge真实小LMDB fixture：**36 passed**（23 builder＋13 merge，无Torch／GPU推理）。
- 独立静态审查的来源数上限P2已修复，并以17来源fixture验证创建输出前拒绝；无剩余可行动P1/P2。
- 本轮没有运行全仓库／CI、相机或人工语言/手形/自然度验收。

固定输入前后结果、全部旧单位字节复核与实际选用情况位于`artifacts/corpus-expansion-20261108/{before,after,comparison,build-audit}.json`；脚本为同目录`check_inputs.py`与`compare_inputs.py`。源统计、下载失败与重试记录也保留。数据、权重及本机artifacts不随Git发布。

## 真正扩展词义的后续来源

本地一Gloss完整片仅26段／8标签，无法补上述七词或额外APP/P/PPT；不能对整句猜边界后计成可播新词。外部一手来源中，[NationalCSL-DP作者论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12009393/)提供6,707个中国手语词的视频资源，但本轮未核验到可直接导入的SMPL-X／完整指骨旋转发布物。视频词数不能计入当前可播词数。

SLR500的25个Kinect关节、NMFs-CSL的RGB视频也需要额外手形重建；SignAvatars的WLASL词级子集属于美语，不能改中文标签补中国手语。来源与访问边界见本机`source-options.md`。真正新增词义需要选择缺词视频进行三维重建和手语审核，或取得相应录制动作；本轮未把未核验视频／生成候选当成已覆盖。

复现与回退见[工作台指南](../csl-animation-workspace.md)，任务与界限见[实施计划](../superpowers/plans/2026-11-08-word-corpus-expansion.md)。
