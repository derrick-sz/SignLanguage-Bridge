# 事故记录：首轮真实批 max_tokens=256 截断（g2a-wbproxy-50-20261026）

- 日期：2026-10-26
- 端点：wb-proxy `http://127.0.0.1:8788/v1/chat/completions`，model `deepseek-v4.1-flash`
- 参数：temperature=0，**max_tokens=256**，prompt `v1_zh`
- 现象：C/D/E 共 150 次调用中 **73 次返回空 completion**（HTTP 200，`finish_reason=length`，
  completion tokens 顶到 256；按组 C=8 / D=34 / E=31）。
- 诊断：`deepseek-v4.1-flash` 在输出最终句前消耗内部推理 token；256 的预算在较长 prompt（D/E 证据更多）上
  不足以同时完成推理与作答，因此 `message.content` 为空且被截断。**不是代理故障，也不是评测口径问题。**
- 处置：
  1. 本目录结果因**选择性偏差**（仅短推理样本成功）**不可用于质量结论**，不进入 `fusion_ablation_g2.csv`；
  2. 正式数据改为 `runs/g2a-wbproxy-50-v2-20261026/`，固定 `max_tokens=2048`（prompt 文本未变，仍为 `v1_zh`；
     run 级 override 记录在 manifest.decoding.override）；
  3. 说明：本目录 `manifest.json` 的 `outputs.n_errors` 记为 0 是当时 harness 的计数缺陷
     （只统计抛异常，未统计后端返回的 `status=error`）；已修复：后续 run 会把 `status != ok` 计入错误数。
     本目录 `fusion_outputs.jsonl` 中 73 条 `status="error"` 行是真实记录。
- 安全：密钥仅运行时内存读取，本文件与任何运行目录均不含密钥。
