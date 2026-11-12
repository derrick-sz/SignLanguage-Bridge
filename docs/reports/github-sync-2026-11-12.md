# 实时网页、数字人工具及文档同步（2026-11-12）

本次产品增量发布到 `codex/realtime-product-20261112`，基线为 `codex/local-godette-v14a-demo-20261107` 的 `ca20cbb94acf6cc26c02e07db78ced165d41393c`。模型服务另见 [`codex/realtime-model-20261112`](https://github.com/betterway614/SignLanguage-Bridge/tree/codex/realtime-model-20261112)，模型增量基线为 `782dee3f128f727373dcdd3e4c2590d7bdc987da`。两个检出继续通过 HTTP / WebSocket 协作；本次不合并主分支、不重启服务，保留旧 PR #1、#2 的范围。

## 本次内容

- 中国手语完整片段切换 Uni-Sign，独立实时增量词/句末中文网页、共享画面选区、逐句核对、训练示范与问题回放。
- 实时字幕转入对话、窗口接续、源身份和修订绑定、取消/EOF 区分，以及匹配实际配置内容的启动器复用保护。
- 原生完整动作的词库来源绑定、构建/组合保护及相关测试。
- 置信度过滤、签者/源句隔离、缓存 SHA 核验和时序反转对照的[姿态评估模块](../../slbridge/animation/retarget_pose_evaluation.py)与[CLI](../../scripts/animation/evaluate_retarget_pose.py)。此工具输出诊断，不自动修改产品动作，也不提供手语语义认证。
- 产品使用说明、模型选择/会议画面/原生“谢谢”报告、小型证据及最新计划链接。

后续动作迭代以[数字人缺陷与改进计划](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/superpowers/plans/2026-11-12-avatar-motion-quality.md)为入口。报告位于模型分支，评估源码位于本产品分支；跨分支链接可直接在 GitHub 阅读。

## 本轮同步检查

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 产品 Python | **93 passed、1 skipped、0 failed** | 8 个相关文件，含词库/组合、通信桥、示范与姿态评估、动画导出核验 |
| Node，使用项目已有 Three.js 测试 loader | **189 passed、15 skipped、0 failed** | `tests/*.test.mjs`，仅提供部署 Three.js 模块；未提供的真实人物/动作 fixture 保留跳过 |
| PowerShell 启动器 | **17 passed、0 failed** | 配置与进程归属的隔离模拟，无真实服务启动/终止 |

首次直接运行 Node 得到 162 passed、32 skipped、2 failed：两个旧动画测试在导入阶段无法解析浏览器绝对路径 `/animation/vendor/three.module.js`。随后按仓库现有 `tests/animation-test-loader.mjs` 配置真实部署 Three.js 模块，得到表中结果；没有修改生产导入或通过删除测试规避。首次日志与配置后日志一并保留。

完整命令、环境和输出见[验证摘要](evidence/realtime-sync-2026-11-12/verification.json)、[Python 日志](evidence/realtime-sync-2026-11-12/product-python-tests.log)、[Node 最终日志](evidence/realtime-sync-2026-11-12/product-node-tests-with-loader.log)、[Node 首次日志](evidence/realtime-sync-2026-11-12/product-node-tests.log)与[启动器结果](evidence/realtime-sync-2026-11-12/launcher-results.json)。不与模型分支的测试计数相加，也不将跳过项计为通过。

## 未通过与未运行边界

本次没有重做 GPU 质量实验、浏览器/摄像头交互、持续负载或专家手语审核。数字人回认与新动作候选仍有未通过项，默认动作未因研究候选而替换。工程、来源与有限几何检查不能证明动作语义正确。远端 CI 在此文档创建时尚未运行。

只同步源码、配置示例、测试、报告及小型必要证据；外部权重、完整数据/词库、原视频与完整本机实验产物留在本地。提交后以远端分支 SHA 和草稿 PR head 读回核对上传结果。
