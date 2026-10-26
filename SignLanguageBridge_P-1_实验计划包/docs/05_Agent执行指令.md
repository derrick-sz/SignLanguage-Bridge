# 05 · 开发 Agent 执行约束与交付规则

## 放入现有项目的方式

- 不要直接覆盖现有 `AGENTS.md`。可以在 AGENTS.md 中加一行相对路径链接：`docs/experiments/P-1/README.md`，让项目智能体按需加载。
- 新建 `docs/experiments/P-1/`，复制本包所有 Markdown、config、schemas 和采集模板。
- 实现代码建议逐步放在 `experiments/p1/`、`backend/adapters/`、`backend/evaluation/`，实际相对路径先由 Agent 根据当前仓库审查确认；不要为了“统一目录”贸然移动旧功能。

## 总体执行要求

1. 实行 **先审计、后修改、再实测**。任何涉及模型的正确性陈述都必须指向真实权重和运行日志。
2. 先完成 T00–T04，不同时重构 UI、RF 训练或 V3.0 整体路由。
3. 将任何新增 CLI 名称标注为“拟实现”，不要暗示项目当前已有该命令。
4. 在目标 GPU 机器上验证 CUDA/PyTorch，单独为 Uni-Sign、SLRT 环境创建锁定依赖或镜像；默认不混装到项目原环境。
5. 使用稳定 `sample_id`、`run_id`，不让预测器或 Jev/LLM 读取任何 GT 文件。
6. 后端 schema 校验应该阻止包含 GT 派生字段的推理请求。测试必须包含“错误地携带 GT 仍被拒绝”的负例。
7. 实验失败时保存可复现的输入配置、异常栈、最后成功阶段和明确的修复候选。
8. 运行结束提交经过验证的真实数据；未运行的字段保持 `null`/`BLOCKED`，不写估计延迟。
9. 源视频/数据/权重只在合法的本地或授权环境使用，不提交到 Git；使用 `.gitignore` 并审查 Git diff。
10. 任何融合实验都记录 prompt 版本和 LLM 参数。不要基于已见测试答案反复修改 prompt 后再声称是独立提升。

## 建议的开发 PR 划分

| PR | 范围 | 主要测试 |
|---|---|---|
| PR1 | 数据索引/manifest/hash/GT 隔离 | sample ID 唯一、627∈1086、日志无 GT |
| PR2 | 模型注册和 Adapter 接口 | 模型能力约束、错误输出 schema |
| PR3 | 参考模型复现工具 | 50 条 smoke、有错误分析、版本追踪 |
| PR4 | 独立 EvalWorker | BLEU/chrF/WER 口径一致、历史与新定义分离 |
| PR5 | 翻译与融合 replay | A/B/C/E/F/G 按实际可用分支，隐私/GT 测试 |
| PR6 | GPU profiler + Stream probe | 冷热启动/P95、future-frame 约束、Partial/Final |
| PR7 | Gate 审核与 ADR | 所有关键决策有 run_id 和证据路径 |

## Agent 每个任务的最终回报格式

```text
任务 ID：
状态：PASS | FAIL | BLOCKED | IMPLEMENTED_UNTESTED
修改文件：
源 Git SHA / 权重 SHA256 / Docker image digest：
实际执行命令：
输入样本数与失败样本数：
输出路径（相对仓库）：
运行结果和指标：
测试命令及结果：
遇到问题/剩余阻塞：
下一任务：
```

## 只在需要时实施的命令接口（拟开发）

以下仅为建议的接口规范；**并非当前仓库已经存在的命令**。

```bash
python -m experiments.p1.inventory --config experiments/p1/configs/run.yaml
python -m experiments.p1.replay --manifest experiments/p1/datasets/pilot_50.csv --model-id unisign_rgb
python -m experiments.p1.evaluate --predictions experiments/p1/runs/EXAMPLE/model_outputs.jsonl --references experiments/p1/references/gt_1086.csv
python -m experiments.p1.profile --config experiments/p1/configs/gpu_profile.yaml
python -m experiments.p1.stream_probe --config experiments/p1/configs/stream_probe.yaml
```

实现前应先比对仓库中是否已有等价工具，可复用而不重复开发。
