# EVALUATION ONLY — do not feed to inference / fusion / LLM

| 文件 | 用途 | 读取方 |
|---|---|---|
| `gt_1086.csv` | 1086 条 test 参考句（上游口径 + 官方标注口径）与参考 Gloss | 离线 EvaluationWorker |
| `pilot_50_diagnostic_EVAL_ONLY.csv` | Pilot-50 的 GT 派生分层（cohort class、WER、字符相似度） | 仅人工分析/报告，禁止进入推理载荷 |

规则：推理 Worker、Jev/LLM、融合控制器都不得读取本目录；一旦 GT 出现在
预测/融合载荷中，`experiments/p1/gt_guard.py` 会抛 `GroundTruthLeakError`。

两个参考口径（全角/半角标点）不能混用：同一份预测在两套参考上的 BLEU 可差约 2.8。

## Git 冻结边界（见 `../FREEZE.json`）

`gt_1086.csv` 与 `pilot_50_diagnostic_EVAL_ONLY.csv` **不入 git**（含原始 GT），
但可从本地冻结快照确定性重建：

```powershell
$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'
.\.conda-env\python.exe -m experiments.p1.datasets.build_manifest     # 先建 datasets manifest
.\.conda-env\python.exe -m experiments.p1.references.build_references # 重建 gt_1086.csv 等
```

重建输入（只读，均在 .gitignore 内）：`artifacts/csldaily-video-test-627-20261026/combined-1086-paperrefs/comparison-both.csv`
与 `models/external/uni_sign_source/out/eval_csl_daily_pose_only/test_tmp_refs.txt`；
源哈希记录在 `source_meta.json`。重建后 sha256 应与 `source_meta.json` 一致。
