# 实时识别、数字人评测及文档同步（2026-11-12）

本次将 2026-11-09 以来的模型服务改动、实验工具和最新数字人质量记录发布到独立增量分支。网页产品与模型服务仍使用两个检出，通过 HTTP / WebSocket 协作；本次没有合并主分支或重启运行服务。历史 PR #1、#2 保留原范围。

| 范围 | 同步分支 | 增量基线 |
| --- | --- | --- |
| 模型服务与评测文档（本分支） | `codex/realtime-model-20261112` | `782dee3f128f727373dcdd3e4c2590d7bdc987da`，`codex/model-contract-auto-captions` |
| 网页产品与数字人评估工具 | [`codex/realtime-product-20261112`](https://github.com/betterway614/SignLanguage-Bridge/tree/codex/realtime-product-20261112) | `ca20cbb94acf6cc26c02e07db78ced165d41393c`，`codex/local-godette-v14a-demo-20261107` |

## 本次内容

- SLRT 增量词、共享姿态前端、Uni-Sign 句末中文、词优先有界队列和可选 LLM 辅助核对；保持原始词、译文与人工修改的区别。
- 首词及排队计时、预处理复用、初始化预热、句末核对提示、训练应用与场景验证工具。
- 识别模型选择、实时工程、延迟、训练样本和数字人回认报告；补齐此前本地研究文档。
- [数字人缺陷与改进计划](../superpowers/plans/2026-11-12-avatar-motion-quality.md)中的 AV-01～AV-13、M0～M6，以及[小型证据快照](evidence/avatar-motion-quality-2026-11-12/summary.json)。最新动作候选未达到接入条件，未替换产品默认动作。
- 跨检出的文档和源码链接改为对应 GitHub 分支链接；只在本机保存的原始实验资料明确标为本地产物。

## 本轮同步检查

模型环境执行 30 个相关 Python 测试文件：**258 passed、1 skipped、0 failed**。覆盖所有本轮新增/修改测试，以及既有 SLRT、Uni-Sign、runtime、API 相关测试文件。完整命令和输出见[验证摘要](evidence/realtime-sync-2026-11-12/verification.json)及[测试日志](evidence/realtime-sync-2026-11-12/model-tests.log)。这是定向工程检查，不是全仓库、GPU 数据集或真实摄像头验收；跳过项不记为通过。

产品分支另行记录 Python、Node 和启动器检查，不与本分支计数相加。提交前检查候选文件范围、JSON 格式、源码/文档链接和 `git diff --cached --check`，提交后核对远端分支 SHA 与 PR head。

## 验收与资料范围

历史真人训练回放、数字人录像回放和姿态候选实验保留各自日期、输入协议及指标；本次同步没有重新运行这些模型质量实验。数字人回认、原“我想睡觉→绿色”用户视频、物理摄像头、新签者及专家手语审核均不能由工程测试推定通过。远端 CI 在此文档创建时尚未运行。

外部权重、完整数据集、LMDB、词动作库、原视频、完整 `artifacts/`、`.survey/` 草稿以及本地 Windows 测试工具包不在本次增量中。使用时仍须按[实时说明](../realtime-csl.md)准备外部资产与本机配置。
