# P-1 实验执行包（SignLanguage-Bridge）

本目录是 `SignLanguageBridge_P-1_实验计划包/`（计划原包，已复制到 `docs/experiments/P-1/`）的**实际执行实现与运行证据**。
所有 PASS 都指向真实 run 目录、真实日志与哈希；缺少权重/数据的实验如实标 `BLOCKED`。

## 1. 目录结构

```text
experiments/p1/
  common.py              # run 目录、哈希、git/环境指纹、JSONL guarded writer
  gt_guard.py            # GT 隔离守卫（推理/融合载荷禁止 GT 键）
  collect_summary.py     # 汇总所有实验 → summary.json / summary.md
  schemas/               # 计划包 schema 副本（EvidenceCandidate / run_record）
  datasets/
    build_manifest.py    # 从冻结 1086 快照生成 GT-free manifest
    manifest_1086.csv    # 1086 样本索引（无 GT）
    pilot_50.csv         # 计划包的 50 条诊断样本 + 解析后的路径
    source_meta.json     # 源文件哈希与计数
  references/            # ⚠ EVALUATION ONLY（见其 README）
    build_references.py  # 生成 gt_1086.csv（两套参考口径 + GT Gloss）
    gt_1086.csv
    pilot_50_diagnostic_EVAL_ONLY.csv
  eval/
    metrics.py           # canonical BLEU-4/chrF/WER/编辑相似度 + signer 聚类 bootstrap
    evaluate.py          # 离线 EvaluationWorker（唯一允许 join GT 的入口）
  e00/                   # E00 资产/环境/数据审计（model_registry.yaml、environment_manifest.json…）
  e01/                   # E01 Pilot-50 复现（video_probe、preprocess_audit、compare_*、summary.json、report）
  e02/                   # E02 同任务翻译基线（candidate_outputs、translation_metrics、fusion_ablation…）
  e03/                   # E03 SLRT 分数校准（score_definition、calibration_results、risk_coverage…）
  e04/                   # E04 GPU 性能（perf_trace、hardware_report、parallel_vs_serial）
  e05/                   # E05 前缀/在线可行性（prefix_probe、future_frame_audit、streaming_ability_matrix）
  e06/                   # E06 Rule-only 仲裁（candidate_inputs、router_ablation、policy_candidate）
  runs/<run_id>/         # 每次运行的不可变记录（manifest.json / model_outputs.jsonl / metrics.json / eval_records.jsonl）
  summary.json / summary.md   # 一键汇总（collect_summary.py 生成）
  tests/test_p1_infra.py # GT 隔离、manifest、canonical 指标回归测试
```

## 2. 常用命令

```powershell
$env:PYTHONPATH = 'E:\githubproject\SignLanguage-Bridge'

# 基础设施（幂等，读冻结快照生成 manifest/references）
.\.conda-env\python.exe -m experiments.p1.datasets.build_manifest
.\.conda-env\python.exe -m experiments.p1.references.build_references

# 基础设施测试（含“canonical 指标必须复现历史 24.8157/21.7974”的回归断言）
.\.conda-env\python.exe -m pytest experiments/p1/tests/test_p1_infra.py -q

# E02 离线回放（imported 快照，不跑模型）
.\.conda-env\python.exe -m experiments.p1.e02.run_e02

# 每个实验的入口脚本
.\.conda-env\python.exe experiments/p1/e03/build_e03.py
.\.conda-env\python.exe experiments/p1/e06/build_e06.py
.\.conda-env\python.exe experiments/p1/e04/benchmark_p1.py all --cold-repeats 3 --warm-repeats 30 --load-minutes 2 --samples 5
.\.conda-env\python.exe experiments/p1/e05/prefix_probe.py run --phases pose,video

# 汇总
.\.conda-env\python.exe -m experiments.p1.collect_summary
```

## 3. 核心纪律（已由代码强制）

1. **GT 隔离**：`common.write_jsonl` 默认调用 `gt_guard.assert_gt_clean`；推理/融合载荷出现
   `reference/gt_*/wer/edit_similarity/...` 会抛 `GroundTruthLeakError`（测试覆盖嵌套键）。
   GT 只存在于 `references/` 与各 run 的 `eval_records.jsonl`（EvaluationWorker 输出）。
2. **不可变运行记录**：每次运行一个 `runs/<run_id>/`，`manifest.json` 记录 git SHA + dirty tree、
   硬件、权重/数据哈希、命令、样本数、失败数、status；`model_outputs.jsonl` 为预测-only。
3. **measured / imported / simulated 三分**：报告与 summary 中每个数字都标明来源；
   历史 1086 快照重算写 `imported`，本轮重跑写 `measured`，离线选择/级联写 `simulated`。
4. **缺失即 BLOCKED**：没有权重/端点/人评就写 BLOCKED 与缺口，不填估计值。

## 4. 当前状态

见 `summary.md`（机器可读：`summary.json`）：

| 实验 | 状态 |
|---|---|
| E00 资产/环境/数据审计 | PASS（审计侧）；G0 smoke 由 E01 提供 |
| E01 Pilot-50 复现 | PASS（U1/U2/S1 各 50/50） |
| E02 同任务翻译基线 | PASS（离线回放）；B1/C/D/E/F/G BLOCKED |
| E03 置信/风险校准 | PASS（SLRT）；Uni-Sign 置信度 BLOCKED |
| E04 GPU 性能 | 见 e04/hardware_report.md |
| E05 前缀/在线可行性 | 见 e05/streaming_ability_matrix.md |
| E06 Rule 仲裁/级联 | PASS（Rule-only）；LLM/Jev BLOCKED |

## 5. 重要边界

- 1086 条是 `retrospective_seen_1086`（本地 test 视频 1086/1176，缺 90 条偏长句），**不是新盲测**；
  统计显著性/泛化结论需要新的、未见 signer/环境的数据。
- 参考句必须统一口径：上游 `test_tmp_refs.txt`（半角标点、逐字分隔）vs 官方标注（全角标点），
  同一份预测两套参考的 BLEU 可差约 2.8–3.1；本包统一以上游参考为 canonical。
- SLRT 为 RTMPose 全帧滑窗对齐 + 近似句段裁剪，不是官方 `sentence_frames-512x512` 协议。
- Uni-Sign 当前没有导出生成分数（`score=None`），置信度相关工作必须标 BLOCKED 或另行重跑导出。
- 本轮**没有**可用的 LLM/Jev 端点，D/E/F/G、LLM-only、Rule→LLM 全部 BLOCKED。
