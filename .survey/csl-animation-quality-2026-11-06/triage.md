# Triage

- Problem: 当前 CSL 动画的动作与人物画面均需要改善；核查现有实现，并筛选 VLibras 等开源项目中能落地到本项目的做法。
- Audience: 手语动画展示使用者、CSL 使用者与审核人员、播放器及动作资产开发者。
- Why now: 首期工作台已经形成工程切片，但现有显示优化没有覆盖角色网格、手部蒙皮、接触约束、表情和连续表达；用户于 2026-11-06 明确选择“动作和人物画面都需要改善”。

## Research Contract

```yaml
survey_run:
  primary_mode: workflow-landscape
  scope: medium
  evidence_floor: primary-pages-first
  output_language: user-language
  needs_platform_map: false
  reuse_existing: false
```

范围问题：在保留现有 CSL 识别及完整表达检索入口的前提下，怎样同时改善角色可读性、动作质量与展示效果，哪些开源能力值得复用？

“LIBRAS”按巴西手语及 VLibras/WikiLibras 工程生态检索；不把 Libras 词典当作中国手语词典。四条证据线分别为本地上下文、开源方案、文档/源码体现的实际工作流，以及保留现有实现或替换播放器的替代路线。

取证日期：2026-11-06（Asia/Taipei）。本地源码读取、两张历史对照图检查、一手官网/仓库读取；网络证据采用 `direct page retrieval`。没有安装上游、训练模型或进行新的动作播放/语言测试。

研究产物：[上下文与现状](context.md)、[开源比较与优化建议](solutions.md)。这是调研建议，不是获批的实现设计或已完成的质量升级。
