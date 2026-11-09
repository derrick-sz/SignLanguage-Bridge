# 词级演示语料扩展

用户授权继续扩展当前快速词库。沿用已批准的原生LMDB/ISO解码与快速组合架构，只扩展可回溯素材；不训练T2G、不改已有语言保留检查、不覆盖旧库、不提交推送。

## 目标与边界

- 当前基线1990标签/5916词片段。使用作者固定revision的CSL train/dev/test各自配套ISO及LMDB，完整SHA后才读取；split不能混写为train。
- 每词候选上限3扩为8；原有unit ID、源帧和NPZ字节必须保留。新增候选数与新增标签数分别报告。
- 新组合库明确`usage_role='product-demo'`、`evaluation_eligible=false`，保留每源摘要和unit原始split。不能拿库内test素材宣称独立留出泛化。
- 原生codec/smplx-left/30FPS预览时钟/词内帧不改/缺词与needs_review行为保持；不从SOKE套词边界、不替换近义词。
- T2G、语言、手形与自然度未因扩库自动通过；固定36条输入前后计数只称工程可播放状态，不称准确率。

## Task 1 资产审计

- [x] 固定model revision a883bc2d13a5d7f85e64591b32bae4cb48950b97、dataset revision 9fe7e04ef59204c787cd6094c747f889efff6ba0，补齐dev/test ISO和LMDB；下载有界监督，SHA与字节数验证。
- [x] 相对旧库统计新标签、七个旧缺词出现位置及匹配clip；保留无来源诊断。

## Task 2 构建与可回退合并

Files: 修改word_library_build.py/build_word_library.py及tests/test_word_library_build.py；新增word_library_merge.py、scripts/animation/expand_word_library.py、tests/test_word_library_merge.py。

- [x] RED→GREEN：`build_word_library(..., split='train', max_candidates=3)`扩展允许train/dev/test和最多8，不破旧默认。官方source身份须按配套hash绑定；unit.source.split真实保留。
- [x] 新`merge_word_libraries(base, additions, output, max_candidates=8)`：读取安全manifest/NPZ（不可解压到任意路径）、核对source字节/形状/骨架/FPS，所有base条目及字节保留；重复同unit ID必须身份一致。余下候选按来源轮询，最多8，同词新增来源优先于某单源垄断。复制实际文件到全新输出，写sources来源及每unit绑定、product-only标记、固定bundle摘要、增量统计。
- [x] 新CLI接受--base/重复--addition/--output/--max-candidates；真实先建train8/dev8/test8，再合并旧base为新v2目录，不覆盖旧库。

## Task 3 运行接入与前后验证

Root owns word_composition.py/tests/test_word_composition.py。允许合法split=train/dev/test，但dev/test必须来自product-only且可关联明确源库及SHA；单位≤20000，manifest仍≤16MiB，候选≤8，缓存有界。旧train-only库兼容。

- [x] RED→GREEN：8候选、非train来源约束、旧unit不变、超过上限/伪train标记/身份冲突/骨架不一致拒绝。
- [x] 固定36条basic-inputs.json，当前服务跑baseline，再切换新库跑同文字同模型身份；分别保留ready/blocked/needs_review/错误、源片段与延迟，不挑成功样例代替整批。
- [x] 若新库有效，保留旧配置可回退，重启本任务服务与自有worker；真实浏览器验证新增候选实际参与和至少两条旧句播放（本轮新增词标签为0），快慢模式及缺词保护不变。
- [x] 独立静态review、定向回归、报告与项目坑点更新。语言/完整语义验收仍NOT_RUN。
