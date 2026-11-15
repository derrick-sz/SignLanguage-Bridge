# PR 整合记录（2026-11-15）

本分支从 PR #4 的头提交 `4e4860380edf4b1a59b6d4e70a358dac64b04fb6` 出发，合入 PR #3 的头提交 `8f0c86121da356667bf35ea975c5aea086d3823a`。#1 的提交已经是 #2 基线历史的一部分，#2 的头提交是 #4 的基线；#3 的模型/API 与 #4 的产品桥通过 HTTP / WebSocket 配套。

直接以本分支合入 `main` 不只是四个 PR 的增量：本次核对时 `main` 为 `b668453456f4ddffef1cbdfda679ebea2b8aacbd`，到 #4 头已有 80 个提交；整合后的暂存树相对 main 涉及 1191 个文件。应另行梳理历史范围与主线部署，不能把本次定向检查写成全仓或主线验收。

冲突处理：`slbridge/api.py` 同时保留 `animation_registry` 和 `realtime_config`；`pyproject.toml` 的 API 依赖保留 `httpx` 和 `websockets`。同名历史文档及验证摘要分别保留产品版和以 `-model` 标注的模型版；坑点记忆链接模型侧补充。历史摘要仍绑定其原始分支、源码及当时的测试，不能作为整合后的结果。

本次针对整合工作树运行 18 个相关 Python 文件，189 passed、1 skipped；实时网页、对话和会议相关 Node 文件 76 passed、0 skipped。两者均为定向工程检查；未重新运行完整仓库、GPU 质量实验、物理摄像头或专家语义审核。首次 Python 运行因默认临时目录不可写而未启动测试，改用项目 artifacts 临时目录后运行；前述计数来自后一次完整运行。数字人动作候选仍未通过语言质量验收。
