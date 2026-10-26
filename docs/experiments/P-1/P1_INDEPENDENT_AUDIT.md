# P-1 / G2 独立机械审计（自动生成）

- 生成时间：2026-10-26T15:54:14+00:00
- 工具：`experiments/p1/verify_all.py`（确定性脚本；不依赖 LLM 判断）
- Git：`98fd19f7cfa648eacd937807cb85972cb79bcc7c`（dirty=True）
- 结论：**13/13 项 PASS**

| 检查项 | 结果 | 细节 | 证据 |
|---|---|---|---|
| run 目录 manifest 状态 | PASS | 9 个 run，PASS=9，非 PASS=无 | `experiments/p1/runs/*/manifest.json` |
| GT 隔离（预测/候选/融合载荷） | PASS | 扫描 10 个 JSONL、12556 行，GT 键命中 0 | `experiments\p1\e02\candidate_outputs.jsonl, experiments\p1\e06\candidate_inputs.jsonl, experiments\p1\runs\e01-pilot50-s1-slrt-gloss-20261026\model_outputs.jsonl …` |
| git 冻结边界（原始 GT/媒体/vendored 不入库） | PASS | tracked=637，违规=无 | `git ls-files` |
| 被排除文件仍受 .gitignore 保护 | PASS | check-ignore [True, True, True, True] | `.gitignore` |
| E02 canonical BLEU | PASS | {"A_unisign_video": 24.8157, "A2_unisign_pose": 25.5846, "B0_gloss_join": 12.2183} | `experiments/p1/e02/translation_metrics.json` |
| E03 校准关键值存在 | PASS | raw ECE≈0.304 / OOF AUC≈0.745 在 calibration_results.json 中可检索 | `experiments/p1/e03/calibration_results.json` |
| E04 热启动 P50 出现在报告 | PASS | pose 276/339 ms、video 2,679/3,271 ms、SLRT 9,775/11,175 ms | `experiments/p1/e04/hardware_report.md` |
| E05 首稳前缀与改写率 | PASS | mean 90% / median 100% / 改写率 0.6714 | `experiments/p1/e05/prefix_probe_summary.md` |
| E06 关键策略行 | PASS | P2 A2 调用率=0.3812 | `experiments/p1/e06/router_ablation.jsonl` |
| G2 真实批：成功子集与 metrics 一致 | PASS | fusion_outputs ok={'A': 50, 'C': 50, 'D': 49, 'E': 48}；metrics n_ok={'A': 50, 'C': 50, 'D': 49, 'E': 48} | `experiments/p1/e02/llm_fusion/runs/g2a-wbproxy-50-v2-20261026/` |
| G2 真实批：端点/模型/参数可追溯 | PASS | provider=wb-proxy model=deepseek-v4.1-flash credentials_ref=WB_PROXY_API_KEY@dsh-credentials | `metrics.json llm_endpoint` |
| G2 真实批：失败数披露 | PASS | D failed=1、E failed=2、data_status=REAL_EXPLORATORY_50 | `metrics.json` |
| 密钥扫描（sk- 形式） | PASS | 命中文件: 无 | `git grep` |

## 说明

- 本审计只验证"报告声称 ↔ 实际文件"的一致性，不重复评价模型质量。
- GT-free 扫描覆盖预测/候选/融合载荷；`eval_records.jsonl`、`gt_1086.csv`、`human_review/` 属 EVAL ONLY，按 FREEZE 边界不入 git，**不视为违规**。
- G2 数字为 50 条探索批（`REAL_EXPLORATORY_50`），不是最终 G2 结论。
