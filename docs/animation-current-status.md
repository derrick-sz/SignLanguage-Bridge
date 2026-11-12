# 当前沟通产品与动画入口

更新：2026-11-12。当前功能分支把视频识别、人工核对、文字回复与数字人表达接成同一页面，并已接入独立的实时增量词/句末中文入口。数字人画面根据动作数据实时渲染，快速模式先检索/组合整段动作再播放。

- [本机沟通产品](http://127.0.0.1:8879/communicate/)；首次启动和外部资产配置见[产品说明](communication-product.md)。
- [实时识别](http://127.0.0.1:8879/communicate/realtime.html)；[缺陷、证据与改进计划](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/superpowers/plans/2026-11-12-avatar-motion-quality.md)是后续动作迭代入口，含优先级、任务依赖和验收条件。
- [数字人回认扩展测试](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/reports/avatar-train-sentence-expansion-2026-11-12.md)与[最近姿态修正迭代](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/reports/avatar-retarget-iteration-2026-11-12.md)。候选尚未通过，产品默认动作未因此替换。
- [上传视频与产品流程实测](reports/communication-product-2026-11-08.md)，[GitHub同步与复查记录](reports/github-sync-2026-11-09.md)。
- [动画、词库与v14A整合](reports/animation-consolidated-status-2026-11-08.md)。`/generate/`、`/animation/`与`/scenes/library`保留为开发工具。

中国手语完整视频路径提供 Uni-Sign 中文候选，实时路径提供 SLRT 词与 Uni-Sign 句末中文；Libras 为12词试点及中葡草案，文字动作输出目前支持中国手语。完整中葡句子互译、Libras动作输出尚未接入。最近数字人测试是画布录像后按视频节奏回放，未覆盖数字人直接实时采集或物理摄像头；组合句接触、动作语言质量与用户跟练仍待验收，不能将工程可播放或请求正常结束视为语义、手形或自然度通过。
