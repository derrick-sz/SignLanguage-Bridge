# 项目协作入口

- 涉及手语模型接入、识别质量、数据集评测、摄像头／连续窗口或 LLM 表达辅助时，先读取[项目坑点记忆](docs/agents/model-evaluation-pitfalls.md)，再依据当前源码和运行产物开展工作。
- 运行或修改 Uni-Sign 时，再读取[集成说明](docs/uni-sign-integration.md)；具体实验条件与证据从坑点记忆中的报告索引进入。
- 发现新坑时维护坑点记忆：记录触发条件、证据、避免办法及日期，区分“已修复”“待验证”“验收未通过”。历史实验成绩保留在对应报告中，新结果另写并注明输入、协议和验收范围。
- P-1 实验计划包与执行证据：[PLAN 副本](docs/experiments/P-1/PLAN.md)、[执行报告（E00–E06/Gate/ADR）](docs/experiments/P-1/P1_EXECUTION_REPORT.md)、[实验基础设施与运行记录](experiments/p1/README.md)；项目级结果见[R19](docs/reports/p1-experiments-2026-10-26.md)。做模型评测、评测指标或复现实验前先读该执行报告与 `experiments/p1/eval/`。
