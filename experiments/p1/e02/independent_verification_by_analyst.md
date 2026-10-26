# E02 产物独立复核（analyst，CPU only）

- 复核时间（UTC）：2026-10-26T14:14:50+00:00
- 复核人：analyst（独立复核 Lead 的 E02 产物；**未 import `run_e02.py`**，只 import `metrics.py`/`common.py`/`gt_guard.py`）
- 复核对象：`experiments/p1/e02/`（run_id `e02-offline-1086-20261026T134508Z`），只读；本文件是唯一新增文件
- 复算脚本：`%TEMP%\analyst_verify_e02.py`（全文见文末附录；sha256 `9f2f0a586fcac0c6d92d4359df8e0e2b17d42da9e27f2a74bc0aab528d83d2f2`）
- 容差：浮点差异 ≤ 0.01（Lead 验收标准）；计数/整数要求完全相等

## 结论：**PASS**

共 124 项检查全部通过（124/124）。关键复算值：

| 项 | 复算 | 报告 | 结果 |
|---|---|---|---|
| A BLEU-4 | 24.8157 | 24.8157 | ✅ |
| A chrF | 21.7974 | 21.7974 | ✅ |
| A2 BLEU-4 | 25.5846 | 25.5846 | ✅ |
| A2 chrF | 22.4284 | 22.4284 | ✅ |
| B0 BLEU-4 | 12.2183 | 12.2183 | ✅ |
| B0 chrF | 13.8087 | 13.8087 | ✅ |
| SLRT corpus WER | 0.3018 | 0.3018 | ✅ |
| A2−A paired BLEU Δ | 0.7689 | 0.7689 | ✅ |
| A2−A paired BLEU Δ 95%CI | [0.2299, 1.4372] | [0.2299, 1.4372] | ✅ |
| A/A2 oracle BLEU-4 | 26.598 | 26.598 | ✅ |
| A/A2 oracle 选择分布 | A=830, A2=256 | A=830, A2=256 | ✅ |
| A vs A2 胜负/持平 | 241/203/642 | 241/203/642 | ✅ |
| 纠正/误改 | 93/50 | 93/50 | ✅ |
| candidate_outputs 行数 | 3258 | 3258 | ✅ |

## 1. candidate_outputs.jsonl 合同检查

- 行数：3258（期望 3258 = 1086×3）；每 sample 恰 A/A2/B0 各一条：True；重复 (sample,candidate) 对：0
- 样本集合与 GT 1086 完全一致：missing=0，extra=0
- `find_gt_keys` 全量扫描：**0** 行携带 GT 派生键（期望 0）
- status 分布：{'OK': 3258}；空 `output.text`：0；非 CSLT 行：0

## 2. 分支指标复算（独立实现，`metrics.py` canonical 口径）

| 分支 | BLEU-4 复算/报告 | chrF 复算/报告 | mean char edit 复算/报告 | exact 复算/报告 |
|---|---|---|---|---|
| A | 24.8157 / 24.8157 | 21.7974 / 21.7974 | 0.4157 / 0.4157 | 15 / 15 |
| A2 | 25.5846 / 25.5846 | 22.4284 / 22.4284 | 0.4212 / 0.4212 | 18 / 18 |
| B0 | 12.2183 / 12.2183 | 13.8087 / 13.8087 | 0.3484 / 0.3484 | 0 / 0 |

BLEU precisions / BP / 官方标点诊断参考 / 每 signer BLEU 全部逐项比对通过（明细见文末「检查明细」）。
SLRT gloss corpus WER：复算 0.3018 vs 报告 0.3018（S 1330/D 717/I 434，参考 token 8220）。B0 = join(gloss sequence) 恒等：mismatch 0。

## 3. A vs A2 配对统计（与 `comparison_table.md` 对照）

- 配对 n：1086；mean sim A=0.4157，A2=0.4212，均值差 +0.0055（报告 +0.0055，95% CI 复算 [-0.0013, +0.0116]，报告 [-0.0013, +0.0116]）
- 胜负（|Δsim|≥0.02）：A2 更好 **241**、A 更好 **203**、持平 **642**（报告 241/203/642）
- 纠正（sim_a<0.5 且 sim_a2≥sim_a+0.1）：**93**；误改（sim_a≥0.5 且 sim_a2≤sim_a−0.1）：**50**（报告 93/50）
- 误差重叠（sim<0.5）：both_wrong 661、仅 A 差 43、仅 A2 差 29、both_ok 353（报告 661/43/29/353）

## 4. A2−A 配对 BLEU 差与 bootstrap CI

- 点估计：复算 +0.7689，报告 +0.7689
- 95% CI（paired cluster bootstrap by signer，seed=20261026，1000 iters）：复算 [+0.2299, +1.4372]，报告 [+0.2299, +1.4372]
- 分支级 cluster bootstrap 的 CI 未在本轮重新推导（重复 6 个 1000-iter sacreBLEU bootstrap 会再增加约 10 分钟，且 Lead 验收清单未要求）；本次复算覆盖点指标、配对统计、oracle 与 A2−A Δ 的成对 CI。

## 5. GT oracle（A/A2）复算

- 选择规则（严格 `sim_a2 > sim_a`，平手归 A，与 `run_e02.py` 一致）：A **830** / A2 **256**（报告 830/256）
- Oracle 指标：BLEU-4 **26.598**（报告 26.598）、chrF **23.2554**（报告 23.2554）、mean char edit **0.4430**（报告 0.443）

## 6. 哈希溯源

- A predictions：`E:\githubproject\SignLanguage-Bridge\artifacts\csldaily-video-test-627-20261026\unisign-video-1086-paperrefs\predictions.jsonl` → ✅ 一致
- A2 predictions：`E:\githubproject\SignLanguage-Bridge\artifacts\cslnews-eval\csldaily-test-1086-pose-20261026\predictions.jsonl` → ✅ 一致
- B0 predictions：`E:\githubproject\SignLanguage-Bridge\experiments\p1\e02\b_gloss_join_predictions.jsonl` → ✅ 一致
- SLRT predictions：`E:\githubproject\SignLanguage-Bridge\experiments\p1\e02\slrt_gloss_predictions.jsonl` → ✅ 一致
- GT references：`E:\githubproject\SignLanguage-Bridge\experiments\p1\references\gt_1086.csv` → ✅ 一致

## 7. 边界

- 复核只针对 **E02 产物一致性**（指标复算、合同、配对统计、oracle、CI、哈希），不评价历史快照本身的推理质量。
- 所有候选文本是 frozen imported 历史预测；1086 是 retrospective_seen，不是新盲测。
- 浮点容差 0.01；bootstrap 结果依赖 Python `random` 与 sacrebleu 版本（本机 sacrebleu 2.2.0），复现需同一环境。
- 未改 E02 任何现有文件、未跑模型、未 git commit。

## 附录 A：检查明细（124 项，全部为复算 vs 报告）

| # | 检查项 | 复算 | 报告 | 结果 |
|---|---|---|---|---|
| 1 | candidate rows == 3258 | 3258 | 3258 | ✅ |
| 2 | unique samples == 1086 | 1086 | 1086 | ✅ |
| 3 | each sample has exactly A/A2/B0 | True | True | ✅ |
| 4 | candidate_id A rows | 1086 | 1086 | ✅ |
| 5 | candidate_id A2 rows | 1086 | 1086 | ✅ |
| 6 | candidate_id B0 rows | 1086 | 1086 | ✅ |
| 7 | rows carrying GT keys (find_gt_keys) | 0 | 0 | ✅ |
| 8 | empty output.text rows | 0 | 0 | ✅ |
| 9 | status distribution | {'OK': 3258} | {'OK': 3258} | ✅ |
| 10 | task == CSLT rows | 3258 | 3258 | ✅ |
| 11 | A.bleu4_13a | 24.8157 | 24.8157 | ✅ |
| 12 | A.chrf | 21.7974 | 21.7974 | ✅ |
| 13 | A.mean_char_edit_similarity | 0.4157 | 0.4157 | ✅ |
| 14 | A.exact_match_count | 15.0 | 15.0 | ✅ |
| 15 | A.bleu_precision_1 | 57.31 | 57.31 | ✅ |
| 16 | A.bleu_precision_2 | 32.82 | 32.82 | ✅ |
| 17 | A.bleu_precision_3 | 20.42 | 20.42 | ✅ |
| 18 | A.bleu_precision_4 | 13.72 | 13.72 | ✅ |
| 19 | A.bleu4_13a_bp | 0.9211 | 0.9211 | ✅ |
| 20 | A.bleu4_13a_alternate_reference | 22.0319 | 22.0319 | ✅ |
| 21 | A.per_signer[P0000].bleu4_13a | 24.9096 | 24.9096 | ✅ |
| 22 | A.per_signer[P0001].bleu4_13a | 17.3496 | 17.3496 | ✅ |
| 23 | A.per_signer[P0002].bleu4_13a | 19.0955 | 19.0955 | ✅ |
| 24 | A.per_signer[P0003].bleu4_13a | 24.3547 | 24.3547 | ✅ |
| 25 | A.per_signer[P0004].bleu4_13a | 36.5234 | 36.5234 | ✅ |
| 26 | A.per_signer[P0005].bleu4_13a | 18.7475 | 18.7475 | ✅ |
| 27 | A.per_signer[P0006].bleu4_13a | 23.8759 | 23.8759 | ✅ |
| 28 | A.per_signer[P0007].bleu4_13a | 23.5797 | 23.5797 | ✅ |
| 29 | A.per_signer[P0008].bleu4_13a | 37.891 | 37.891 | ✅ |
| 30 | A.per_signer[P0009].bleu4_13a | 18.5476 | 18.5476 | ✅ |
| 31 | A2.bleu4_13a | 25.5846 | 25.5846 | ✅ |
| 32 | A2.chrf | 22.4284 | 22.4284 | ✅ |
| 33 | A2.mean_char_edit_similarity | 0.4212 | 0.4212 | ✅ |
| 34 | A2.exact_match_count | 18.0 | 18.0 | ✅ |
| 35 | A2.bleu_precision_1 | 58.15 | 58.15 | ✅ |
| 36 | A2.bleu_precision_2 | 33.67 | 33.67 | ✅ |
| 37 | A2.bleu_precision_3 | 21.15 | 21.15 | ✅ |
| 38 | A2.bleu_precision_4 | 14.2 | 14.2 | ✅ |
| 39 | A2.bleu4_13a_bp | 0.924 | 0.924 | ✅ |
| 40 | A2.bleu4_13a_alternate_reference | 22.6401 | 22.6401 | ✅ |
| 41 | A2.per_signer[P0000].bleu4_13a | 25.4858 | 25.4858 | ✅ |
| 42 | A2.per_signer[P0001].bleu4_13a | 18.9987 | 18.9987 | ✅ |
| 43 | A2.per_signer[P0002].bleu4_13a | 20.1462 | 20.1462 | ✅ |
| 44 | A2.per_signer[P0003].bleu4_13a | 24.9768 | 24.9768 | ✅ |
| 45 | A2.per_signer[P0004].bleu4_13a | 35.5995 | 35.5995 | ✅ |
| 46 | A2.per_signer[P0005].bleu4_13a | 20.99 | 20.99 | ✅ |
| 47 | A2.per_signer[P0006].bleu4_13a | 26.5571 | 26.5571 | ✅ |
| 48 | A2.per_signer[P0007].bleu4_13a | 22.6715 | 22.6715 | ✅ |
| 49 | A2.per_signer[P0008].bleu4_13a | 39.0917 | 39.0917 | ✅ |
| 50 | A2.per_signer[P0009].bleu4_13a | 20.0622 | 20.0622 | ✅ |
| 51 | B0.bleu4_13a | 12.2183 | 12.2183 | ✅ |
| 52 | B0.chrf | 13.8087 | 13.8087 | ✅ |
| 53 | B0.mean_char_edit_similarity | 0.3484 | 0.3484 | ✅ |
| 54 | B0.exact_match_count | 0.0 | 0.0 | ✅ |
| 55 | B0.bleu_precision_1 | 62.57 | 62.57 | ✅ |
| 56 | B0.bleu_precision_2 | 31.25 | 31.25 | ✅ |
| 57 | B0.bleu_precision_3 | 12.06 | 12.06 | ✅ |
| 58 | B0.bleu_precision_4 | 5.68 | 5.68 | ✅ |
| 59 | B0.bleu4_13a_bp | 0.6388 | 0.6388 | ✅ |
| 60 | B0.bleu4_13a_alternate_reference | 12.2183 | 12.2183 | ✅ |
| 61 | B0.per_signer[P0000].bleu4_13a | 12.0211 | 12.0211 | ✅ |
| 62 | B0.per_signer[P0001].bleu4_13a | 9.3485 | 9.3485 | ✅ |
| 63 | B0.per_signer[P0002].bleu4_13a | 9.8149 | 9.8149 | ✅ |
| 64 | B0.per_signer[P0003].bleu4_13a | 14.4726 | 14.4726 | ✅ |
| 65 | B0.per_signer[P0004].bleu4_13a | 12.2342 | 12.2342 | ✅ |
| 66 | B0.per_signer[P0005].bleu4_13a | 10.2677 | 10.2677 | ✅ |
| 67 | B0.per_signer[P0006].bleu4_13a | 12.7315 | 12.7315 | ✅ |
| 68 | B0.per_signer[P0007].bleu4_13a | 10.9767 | 10.9767 | ✅ |
| 69 | B0.per_signer[P0008].bleu4_13a | 17.7531 | 17.7531 | ✅ |
| 70 | B0.per_signer[P0009].bleu4_13a | 12.3083 | 12.3083 | ✅ |
| 71 | SLRT_gloss.reference_tokens | 8220.0 | 8220.0 | ✅ |
| 72 | SLRT_gloss.substitutions | 1330.0 | 1330.0 | ✅ |
| 73 | SLRT_gloss.deletions | 717.0 | 717.0 | ✅ |
| 74 | SLRT_gloss.insertions | 434.0 | 434.0 | ✅ |
| 75 | SLRT_gloss.corpus_wer | 0.3018 | 0.3018 | ✅ |
| 76 | B0 identity (candidate == join(gloss) == b_gloss_join_predictions) | 0 | 0 | ✅ |
| 77 | paired.n_paired | 1086.0 | 1086.0 | ✅ |
| 78 | paired.mean_sim_a | 0.415688 | 0.4157 | ✅ |
| 79 | paired.mean_sim_a2 | 0.421199 | 0.4212 | ✅ |
| 80 | paired.mean_delta_sim | 0.005511 | 0.0055 | ✅ |
| 81 | paired.wins_a2(delta>=0.02) | 241.0 | 241.0 | ✅ |
| 82 | paired.wins_a(delta<=-0.02) | 203.0 | 203.0 | ✅ |
| 83 | paired.ties | 642.0 | 642.0 | ✅ |
| 84 | paired.corrections | 93.0 | 93.0 | ✅ |
| 85 | paired.regressions | 50.0 | 50.0 | ✅ |
| 86 | paired.sum(wins/tie)==n | 1086.0 | 1086.0 | ✅ |
| 87 | paired.delta_sim_ci.point | 0.0055 | 0.0055 | ✅ |
| 88 | paired.delta_bleu_ci.point | 0.7689 | 0.7689 | ✅ |
| 89 | paired.delta_sim_ci.lower | -0.0013 | -0.0013 | ✅ |
| 90 | paired.delta_bleu_ci.lower | 0.2299 | 0.2299 | ✅ |
| 91 | paired.delta_sim_ci.upper | 0.0116 | 0.0116 | ✅ |
| 92 | paired.delta_bleu_ci.upper | 1.4372 | 1.4372 | ✅ |
| 93 | error_overlap.both_wrong_sim_lt_0_5 | 661.0 | 661.0 | ✅ |
| 94 | error_overlap.a_wrong_a2_ok | 43.0 | 43.0 | ✅ |
| 95 | error_overlap.a_ok_a2_wrong | 29.0 | 29.0 | ✅ |
| 96 | error_overlap.both_ok | 353.0 | 353.0 | ✅ |
| 97 | error_overlap.sum | 1086.0 | 1086.0 | ✅ |
| 98 | oracle.bleu4_13a | 26.598 | 26.598 | ✅ |
| 99 | oracle.chrf | 23.2554 | 23.2554 | ✅ |
| 100 | oracle.mean_char_edit_similarity | 0.442993 | 0.443 | ✅ |
| 101 | oracle.selection_counts.A | 830.0 | 830.0 | ✅ |
| 102 | oracle.selection_counts.A2 | 256.0 | 256.0 | ✅ |
| 103 | paired.delta_bleu_point(A2-A) | 0.768878 | 0.7689 | ✅ |
| 104 | comparison_table.A.bleu4 | 24.8157 | 24.8157 | ✅ |
| 105 | comparison_table.A.chrf | 21.7974 | 21.7974 | ✅ |
| 106 | comparison_table.A2.bleu4 | 25.5846 | 25.5846 | ✅ |
| 107 | comparison_table.A2.chrf | 22.4284 | 22.4284 | ✅ |
| 108 | comparison_table.B0.bleu4 | 12.2183 | 12.2183 | ✅ |
| 109 | comparison_table.B0.chrf | 13.8087 | 13.8087 | ✅ |
| 110 | comparison_table.delta_sim | 0.0055 | 0.0055 | ✅ |
| 111 | comparison_table.delta_bleu | 0.7689 | 0.7689 | ✅ |
| 112 | comparison_table.oracle_bleu | 26.598 | 26.598 | ✅ |
| 113 | comparison_table.wins_a2 | 241.0 | 241.0 | ✅ |
| 114 | comparison_table.wins_a | 203.0 | 203.0 | ✅ |
| 115 | comparison_table.ties | 642.0 | 642.0 | ✅ |
| 116 | comparison_table.corrections | 93.0 | 93.0 | ✅ |
| 117 | comparison_table.regressions | 50.0 | 50.0 | ✅ |
| 118 | comparison_table.oracle_sel_A | 830.0 | 830.0 | ✅ |
| 119 | comparison_table.oracle_sel_A2 | 256.0 | 256.0 | ✅ |
| 120 | sha256 A predictions | 8573129ba0b4d300 | 8573129ba0b4d300 | ✅ |
| 121 | sha256 A2 predictions | 758621291e87a4e1 | 758621291e87a4e1 | ✅ |
| 122 | sha256 B0 predictions | aef4490e754b8834 | aef4490e754b8834 | ✅ |
| 123 | sha256 SLRT predictions | 177cd08f136a53c5 | 177cd08f136a53c5 | ✅ |
| 124 | sha256 GT references | feb1b42daa74ed3e | feb1b42daa74ed3e | ✅ |

## 附录 B：复核脚本（独立实现，位于 `%TEMP%`，运行后文本保存在本 md）

```python
"""Independent verification of Lead's E02 artifacts (analyst, CPU only).

Re-implements the canonical metric pipeline from experiments/p1/eval/metrics.py
(no import of experiments/p1/e02/run_e02.py) and cross-checks:
  1) A/A2/B0 corpus BLEU-4, chrF, mean char edit similarity, exact match
  2) candidate_outputs.jsonl contract (3258 rows, A/A2/B0 per sample, GT-free, statuses)
  3) A vs A2 paired stats + GT oracle vs translation_metrics.json / comparison_table.md
  4) bootstrap CIs (cluster by signer, seed 20261026, 1000 iters) incl. paired BLEU delta

Writes the verification report to experiments/p1/e02/independent_verification_by_analyst.md
(only NEW file in e02; existing E02 artifacts are never modified).
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.p1 import common
from experiments.p1.eval import metrics as M
from experiments.p1.gt_guard import find_gt_keys

ROOT = common.ROOT
E02 = common.P1 / "e02"
GT = common.REFERENCES / "gt_1086.csv"
CAND = E02 / "candidate_outputs.jsonl"
TM = E02 / "translation_metrics.json"
CT = E02 / "comparison_table.md"
REPORT_OUT = E02 / "independent_verification_by_analyst.md"
TOL = 0.01  # allowed floating tolerance (Lead's acceptance criterion)
SEED = 20261026
ITERS = 1000


def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT)}


def char_sep(text: str) -> str:
    return M.character_separated(text)


def edit_sim(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def bleu_only(items: list[dict]) -> float:
    import sacrebleu

    hyps = [item["hypothesis_char"] for item in items]
    refs = [item["reference_char"] for item in items]
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score


def paired_cluster_bootstrap(pairs: list[dict], statistic, *, iterations: int = ITERS,
                             group_key: str = "signer_id", seed: int = SEED) -> dict:
    """Exact re-implementation of E02's paired_cluster_bootstrap."""

    groups: dict[str, list[dict]] = {}
    for record in pairs:
        groups.setdefault(str(record[group_key]), []).append(record)
    keys = sorted(groups)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample: list[dict] = []
        for _ in range(len(keys)):
            sample.extend(groups[rng.choice(keys)])
        value = statistic(sample)
        if value is not None:
            values.append(float(value))
    values.sort()
    point = statistic(pairs)
    lower = round(values[int(0.025 * (len(values) - 1))], 4)
    upper = round(values[int(0.975 * (len(values) - 1))], 4)
    return {"point": round(point, 4), "lower": lower, "upper": upper, "iterations": iterations,
            "method": "paired_cluster_bootstrap_by_signer"}


def cluster_bootstrap(records: list[dict], statistic, *, iterations: int = ITERS,
                      group_key: str = "signer_id", seed: int = SEED) -> dict:
    return M.bootstrap_by_group(records, statistic, iterations=iterations, group_key=group_key, seed=seed)


def compare(label: str, recomputed, reported, *, tol: float = TOL) -> dict:
    if recomputed is None or reported is None:
        return {"label": label, "recomputed": recomputed, "reported": reported, "delta": None,
                "pass": recomputed == reported}
    delta = recomputed - reported
    return {"label": label, "recomputed": round(float(recomputed), 6), "reported": round(float(reported), 6),
            "delta": round(float(delta), 6), "pass": abs(delta) <= tol}


def main() -> int:
    checks: list[dict] = []
    failures: list[str] = []

    tm = common.read_json(TM)
    ct_text = CT.read_text(encoding="utf-8")
    gt = load_gt()

    # ---------------- 1. candidate_outputs contract ----------------------
    raw_lines = CAND.read_text(encoding="utf-8").splitlines()
    candidates = [json.loads(line) for line in raw_lines if line.strip()]
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    duplicate_pairs = 0
    status_counts: Counter = Counter()
    gt_key_hits = 0
    empty_texts = 0
    non_cslt = 0
    for row in candidates:
        sample = row["sample_id"]
        candidate_id = row["candidate_id"]
        if candidate_id in by_sample[sample]:
            duplicate_pairs += 1
        by_sample[sample][candidate_id] = row
        status_counts[str(row.get("status"))] += 1
        leaks = find_gt_keys(row)
        if leaks:
            gt_key_hits += 1
        text = (row.get("output") or {}).get("text")
        if text is None or not str(text).strip():
            empty_texts += 1
        if row.get("task") != "CSLT":
            non_cslt += 1
    sample_ids = set(by_sample)
    gt_ids = set(gt)
    missing_in_candidates = sorted(gt_ids - sample_ids)
    extra_in_candidates = sorted(sample_ids - gt_ids)
    per_candidate_counts = Counter(row["candidate_id"] for row in candidates)
    exactly_three = all(set(m) == {"A", "A2", "B0"} for m in by_sample.values())

    checks.append({"label": "candidate rows == 3258", "recomputed": len(candidates), "reported": 3258,
                   "pass": len(candidates) == 3258})
    checks.append({"label": "unique samples == 1086", "recomputed": len(sample_ids), "reported": 1086,
                   "pass": len(sample_ids) == 1086 and not missing_in_candidates and not extra_in_candidates})
    checks.append({"label": "each sample has exactly A/A2/B0", "recomputed": exactly_three, "reported": True,
                   "pass": exactly_three and duplicate_pairs == 0})
    for cid, expected in (("A", 1086), ("A2", 1086), ("B0", 1086)):
        checks.append({"label": f"candidate_id {cid} rows", "recomputed": per_candidate_counts.get(cid, 0),
                       "reported": expected, "pass": per_candidate_counts.get(cid, 0) == expected})
    checks.append({"label": "rows carrying GT keys (find_gt_keys)", "recomputed": gt_key_hits, "reported": 0,
                   "pass": gt_key_hits == 0})
    checks.append({"label": "empty output.text rows", "recomputed": empty_texts, "reported": 0,
                   "pass": empty_texts == 0})
    checks.append({"label": "status distribution", "recomputed": dict(status_counts),
                   "reported": {"OK": 3258}, "pass": status_counts == Counter({"OK": 3258})})
    checks.append({"label": "task == CSLT rows", "recomputed": len(candidates) - non_cslt, "reported": 3258,
                   "pass": non_cslt == 0})
    if raw_lines and raw_lines[-1] != "":
        pass  # trailing newline is fine either way

    # ---------------- 2. A/A2/B0 canonical metrics -----------------------
    texts = {sample: {cid: str(record["output"]["text"]) for cid, record in mapping.items()}
             for sample, mapping in by_sample.items()}
    pairs_by_cid = {
        cid: [(gt[s]["ref_sentence_upstream"], texts[s][cid]) for s in sorted(gt)]
        for cid in ("A", "A2", "B0")
    }
    reported_paths = {
        "A": tm["paths"]["A_unisign_video"]["metrics"],
        "A2": tm["paths"]["A2_unisign_pose"]["metrics"],
        "B0": tm["paths"]["B0_gloss_join"]["metrics"],
    }
    metric_block = {}
    per_signer_block = {}
    for cid in ("A", "A2", "B0"):
        corpus = M.corpus_metrics_cslt(pairs_by_cid[cid])
        metric_block[cid] = corpus
        reported = reported_paths[cid]
        for key in ("bleu4_13a", "chrf", "mean_char_edit_similarity", "exact_match_count"):
            checks.append(compare(f"{cid}.{key}", corpus[key], reported[key]))
        # precision / bp sanity (tolerance 0.01)
        for i, value in enumerate(reported["bleu4_13a_precisions"]):
            checks.append(compare(f"{cid}.bleu_precision_{i+1}", corpus["bleu4_13a_precisions"][i], value))
        checks.append(compare(f"{cid}.bleu4_13a_bp", corpus["bleu4_13a_bp"], reported["bleu4_13a_bp"]))
        # alternate reference (official punctuation) diagnostic
        alt = [gt[s]["gt_sentence_official"] for s in sorted(gt)]
        corpus_alt = M.corpus_metrics_cslt(pairs_by_cid[cid], alternate_references=alt)
        checks.append(compare(f"{cid}.bleu4_13a_alternate_reference",
                              corpus_alt["bleu4_13a_alternate_reference"],
                              reported["bleu4_13a_alternate_reference"]))
        # per-signer BLEU (evaluate() reports bleu4_13a per signer)
        reported_signer = tm["paths"]["A_unisign_video" if cid == "A" else
                            "A2_unisign_pose" if cid == "A2" else "B0_gloss_join"]["per_signer"]
        signer_values = {}
        for signer in sorted({row["signer_id"] for row in gt.values()}):
            subset = [(gt[s]["ref_sentence_upstream"], texts[s][cid]) for s in sorted(gt)
                      if gt[s]["signer_id"] == signer]
            signer_values[signer] = M.corpus_metrics_cslt(subset)["bleu4_13a"]
            checks.append(compare(f"{cid}.per_signer[{signer}].bleu4_13a",
                                  signer_values[signer], reported_signer[signer]["bleu4_13a"]))
        per_signer_block[cid] = signer_values

    # ---------------- 3. SLRT gloss WER ----------------------------------
    gloss_rows = {row["sample"]: row for row in common.read_jsonl(E02 / "slrt_gloss_predictions.jsonl")}
    wer_pairs = [(gt[s]["gt_gloss"], " ".join(str(t) for t in gloss_rows[s]["gloss_sequence"]))
                 for s in sorted(gt)]
    corpus_wer = M.corpus_wer(wer_pairs)
    reported_wer = tm["paths"]["SLRT_gloss"]["metrics"]
    for key in ("reference_tokens", "substitutions", "deletions", "insertions", "corpus_wer"):
        checks.append(compare(f"SLRT_gloss.{key}", corpus_wer[key], reported_wer[key]))
    # gloss join identity: B0 == "".join(gloss tokens) and == b_gloss_join_predictions
    b_join_rows = {row["sample"]: row["prediction"] for row in common.read_jsonl(E02 / "b_gloss_join_predictions.jsonl")}
    join_mismatch = sum(
        1 for s in sorted(gt)
        if texts[s]["B0"] != "".join(str(t) for t in gloss_rows[s]["gloss_sequence"])
        or texts[s]["B0"] != b_join_rows[s]
    )
    checks.append({"label": "B0 identity (candidate == join(gloss) == b_gloss_join_predictions)",
                   "recomputed": join_mismatch, "reported": 0, "pass": join_mismatch == 0})

    # ---------------- 4. paired A vs A2 stats ----------------------------
    paired = []
    for sample in sorted(gt):
        reference = gt[sample]["ref_sentence_upstream"]
        sim_a = edit_sim(reference, texts[sample]["A"])
        sim_a2 = edit_sim(reference, texts[sample]["A2"])
        paired.append({
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "reference": reference,
            "reference_char": char_sep(reference),
            "hypothesis_a_char": char_sep(texts[sample]["A"]),
            "hypothesis_a2_char": char_sep(texts[sample]["A2"]),
            "sim_a": sim_a,
            "sim_a2": sim_a2,
            "delta_sim": round(sim_a2 - sim_a, 4),
        })
    reported_paired = tm["paired_a_vs_a2"]
    wins_a2 = sum(1 for row in paired if row["delta_sim"] >= 0.02)
    wins_a = sum(1 for row in paired if row["delta_sim"] <= -0.02)
    ties = len(paired) - wins_a2 - wins_a
    corrections = sum(1 for row in paired if row["sim_a"] < 0.5 and row["sim_a2"] >= row["sim_a"] + 0.1)
    regressions = sum(1 for row in paired if row["sim_a"] >= 0.5 and row["sim_a2"] <= row["sim_a"] - 0.1)
    checks.append(compare("paired.n_paired", len(paired), reported_paired["n_paired"]))
    checks.append(compare("paired.mean_sim_a", statistics.fmean(r["sim_a"] for r in paired),
                          reported_paired["mean_sim_a"]))
    checks.append(compare("paired.mean_sim_a2", statistics.fmean(r["sim_a2"] for r in paired),
                          reported_paired["mean_sim_a2"]))
    checks.append(compare("paired.mean_delta_sim", statistics.fmean(r["delta_sim"] for r in paired),
                          reported_paired["mean_delta_sim"]))
    checks.append(compare("paired.wins_a2(delta>=0.02)", wins_a2, reported_paired["wins_a2"]))
    checks.append(compare("paired.wins_a(delta<=-0.02)", wins_a, reported_paired["wins_a"]))
    checks.append(compare("paired.ties", ties, reported_paired["ties"]))
    checks.append(compare("paired.corrections", corrections, reported_paired["corrections_a2_fixes_a"]))
    checks.append(compare("paired.regressions", regressions, reported_paired["regressions_a2_breaks_a"]))
    checks.append(compare("paired.sum(wins/tie)==n", wins_a2 + wins_a + ties, len(paired)))

    def delta_sim_stat(items: list[dict]) -> float:
        return statistics.fmean(row["delta_sim"] for row in items)

    def paired_bleu_delta(items: list[dict]) -> float:
        return bleu_only([{"reference_char": row["reference_char"],
                           "hypothesis_char": row["hypothesis_a2_char"]} for row in items]) - \
               bleu_only([{"reference_char": row["reference_char"],
                           "hypothesis_char": row["hypothesis_a_char"]} for row in items])

    ci_sim = paired_cluster_bootstrap(paired, delta_sim_stat)
    ci_bleu = paired_cluster_bootstrap(paired, paired_bleu_delta)
    for key in ("point", "lower", "upper"):
        checks.append(compare(f"paired.delta_sim_ci.{key}", ci_sim[key], reported_paired["delta_sim_ci"][key]))
        checks.append(compare(f"paired.delta_bleu_ci.{key}", ci_bleu[key], reported_paired["delta_bleu_ci"][key]))

    # error overlap
    reported_overlap = tm["error_overlap"]
    overlap = {
        "both_wrong_sim_lt_0_5": sum(1 for r in paired if r["sim_a"] < 0.5 and r["sim_a2"] < 0.5),
        "a_wrong_a2_ok": sum(1 for r in paired if r["sim_a"] < 0.5 and r["sim_a2"] >= 0.5),
        "a_ok_a2_wrong": sum(1 for r in paired if r["sim_a"] >= 0.5 and r["sim_a2"] < 0.5),
        "both_ok": sum(1 for r in paired if r["sim_a"] >= 0.5 and r["sim_a2"] >= 0.5),
    }
    for key, value in overlap.items():
        checks.append(compare(f"error_overlap.{key}", value, reported_overlap[key]))
    checks.append(compare("error_overlap.sum", sum(overlap.values()), len(paired)))

    # ---------------- 5. GT oracle (A/A2) --------------------------------
    oracle_pairs: list[tuple[str, str]] = []
    selection = Counter()
    for row in paired:
        if row["sim_a2"] > row["sim_a"]:  # strict greater -> ties go to A (E02 rule)
            selection["A2"] += 1
            oracle_pairs.append((row["reference"], texts[row["sample_id"]]["A2"]))
        else:
            selection["A"] += 1
            oracle_pairs.append((row["reference"], texts[row["sample_id"]]["A"]))
    oracle_corpus = M.corpus_metrics_cslt(oracle_pairs)
    oracle_reported = tm["oracle"]["a_a2"]
    checks.append(compare("oracle.bleu4_13a", oracle_corpus["bleu4_13a"], oracle_reported["bleu4_13a"]))
    checks.append(compare("oracle.chrf", oracle_corpus["chrf"], oracle_reported["chrf"]))
    checks.append(compare("oracle.mean_char_edit_similarity",
                          statistics.fmean(max(r["sim_a"], r["sim_a2"]) for r in paired),
                          oracle_reported["mean_char_edit_similarity"]))
    checks.append(compare("oracle.selection_counts.A", selection["A"],
                          oracle_reported["selection_counts"]["A"], tol=0))
    checks.append(compare("oracle.selection_counts.A2", selection["A2"],
                          oracle_reported["selection_counts"]["A2"], tol=0))
    # A2 - A corpus BLEU delta point (paired CI point)
    a_bleu = bleu_only([{"reference_char": r["reference_char"],
                         "hypothesis_char": r["hypothesis_a_char"]} for r in paired])
    a2_bleu = bleu_only([{"reference_char": r["reference_char"],
                          "hypothesis_char": r["hypothesis_a2_char"]} for r in paired])
    checks.append(compare("paired.delta_bleu_point(A2-A)", a2_bleu - a_bleu, reported_paired["delta_bleu_ci"]["point"]))

    # ---------------- 6. comparison_table.md consistency -----------------
    def grep_float(pattern: str) -> float | None:
        match = re.search(pattern, ct_text)
        return float(match.group(1)) if match else None

    ct_values = {
        "A.bleu4": grep_float(r"\| A — .*?\| 1086 \| ([0-9.]+) \|"),
        "A.chrf": grep_float(r"\| A — .*?\| 1086 \| [0-9.]+ \| ([0-9.]+) \|"),
        "A2.bleu4": grep_float(r"\| A2 — .*?\| 1086 \| ([0-9.]+) \|"),
        "A2.chrf": grep_float(r"\| A2 — .*?\| 1086 \| [0-9.]+ \| ([0-9.]+) \|"),
        "B0.bleu4": grep_float(r"\| B0 — .*?\| 1086 \| ([0-9.]+) \|"),
        "B0.chrf": grep_float(r"\| B0 — .*?\| 1086 \| [0-9.]+ \| ([0-9.]+) \|"),
        "delta_sim": grep_float(r"A2.A 平均字符编辑相似度差：\*\*\+([0-9.]+)\*\*"),
        "delta_bleu": grep_float(r"A2.A BLEU-4 差：\*\*\+([0-9.]+)\*\*"),
        "oracle_bleu": grep_float(r"oracle：BLEU-4 \*\*([0-9.]+)\*\*"),
        "wins_a2": grep_float(r"A2 更好 (\d+)"),
        "wins_a": grep_float(r"A 更好 (\d+)"),
        "ties": grep_float(r"持平 (\d+)"),
        "corrections": grep_float(r"纠正（A 差→A2 明显更好）：(\d+)"),
        "regressions": grep_float(r"误改（A 好→A2 明显更差）：(\d+)"),
    }
    try:
        selection_match = re.search(r"选择分布 \{'A': (\d+), 'A2': (\d+)\}", ct_text)
        ct_values["oracle_sel_A"] = int(selection_match.group(1)) if selection_match else None
        ct_values["oracle_sel_A2"] = int(selection_match.group(2)) if selection_match else None
    except Exception:  # noqa: BLE001
        ct_values["oracle_sel_A"] = ct_values["oracle_sel_A2"] = None
    table_expectations = [
        ("A.bleu4", metric_block["A"]["bleu4_13a"]), ("A.chrf", metric_block["A"]["chrf"]),
        ("A2.bleu4", metric_block["A2"]["bleu4_13a"]), ("A2.chrf", metric_block["A2"]["chrf"]),
        ("B0.bleu4", metric_block["B0"]["bleu4_13a"]), ("B0.chrf", metric_block["B0"]["chrf"]),
        ("delta_sim", statistics.fmean(r["sim_a2"] - r["sim_a"] for r in paired)),
        ("delta_bleu", a2_bleu - a_bleu),
        ("oracle_bleu", oracle_corpus["bleu4_13a"]),
        ("wins_a2", wins_a2), ("wins_a", wins_a), ("ties", ties),
        ("corrections", corrections), ("regressions", regressions),
        ("oracle_sel_A", selection["A"]), ("oracle_sel_A2", selection["A2"]),
    ]
    for key, expected in table_expectations:
        parsed = ct_values.get(key)
        checks.append(compare(f"comparison_table.{key}", parsed, round(float(expected), 4) if not isinstance(expected, int) else expected))

    # ---------------- 7. hash provenance ---------------------------------
    hash_expectations = [
        ("A predictions", tm["paths"]["A_unisign_video"]["predictions_path"],
         tm["paths"]["A_unisign_video"]["predictions_sha256"]),
        ("A2 predictions", tm["paths"]["A2_unisign_pose"]["predictions_path"],
         tm["paths"]["A2_unisign_pose"]["predictions_sha256"]),
        ("B0 predictions", tm["paths"]["B0_gloss_join"]["predictions_path"],
         tm["paths"]["B0_gloss_join"]["predictions_sha256"]),
        ("SLRT predictions", tm["paths"]["SLRT_gloss"]["predictions_path"],
         tm["paths"]["SLRT_gloss"]["predictions_sha256"]),
        ("GT references", str(GT), tm["paths"]["A_unisign_video"]["references_sha256"]),
    ]
    for label, path_text, expected in hash_expectations:
        path = Path(path_text)
        actual = common.sha256_file(path) if path.exists() else None
        checks.append({"label": f"sha256 {label}", "recomputed": (actual or "")[:16], "reported": expected[:16],
                       "pass": actual == expected})

    # NOTE: 分支级 bootstrap CI 不在本次独立复算范围内（Lead 的验收清单只要求点指标、
    # 配对统计、oracle 与 A2−A BLEU Δ 的 CI；重复 6 个 1000-iter sacreBLEU bootstrap
    # 会让本复核耗时从分钟级涨到 20+ 分钟，收益低）。A2−A Δ 的 paired CI 仍完整复算。

    failed = [c for c in checks if not c["pass"]]
    failures = [c["label"] for c in failed]
    (Path(__file__).parent / "verify_e02_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ---------------- write markdown report ------------------------------
    lines: list[str] = []
    lines.append("# E02 产物独立复核（analyst，CPU only）")
    lines.append("")
    lines.append(f"- 复核时间（UTC）：{common.utc_now()}")
    lines.append(f"- 复核人：analyst（独立复核 Lead 的 E02 产物；**未 import `run_e02.py`**，只 import `metrics.py`/`common.py`/`gt_guard.py`）")
    lines.append(f"- 复核对象：`experiments/p1/e02/`（run_id `{tm['run_id']}`），只读；本文件是唯一新增文件")
    lines.append(f"- 复算脚本：`%TEMP%\\analyst_verify_e02.py`（全文见文末附录；sha256 `{common.sha256_file(Path(__file__)) if Path(__file__).exists() else 'n/a'}`）")
    lines.append(f"- 容差：浮点差异 ≤ {TOL}（Lead 验收标准）；计数/整数要求完全相等")
    lines.append("")
    overall = "**PASS**" if not failures else "**FAIL**"
    lines.append(f"## 结论：{overall}")
    lines.append("")
    if failures:
        lines.append(f"共 {len(checks)} 项检查，**{len(failed)} 项不一致**：")
        lines.append("")
        for item in failed:
            lines.append(f"- ✗ {item['label']}：复算 {item['recomputed']} vs 报告 {item['reported']}（Δ {item.get('delta')}）")
    else:
        lines.append(f"共 {len(checks)} 项检查全部通过（{len(checks)}/{len(checks)}）。关键复算值：")
        lines.append("")
        lines.append("| 项 | 复算 | 报告 | 结果 |")
        lines.append("|---|---|---|---|")
        key_rows = [
            ("A BLEU-4", metric_block["A"]["bleu4_13a"], reported_paths["A"]["bleu4_13a"]),
            ("A chrF", metric_block["A"]["chrf"], reported_paths["A"]["chrf"]),
            ("A2 BLEU-4", metric_block["A2"]["bleu4_13a"], reported_paths["A2"]["bleu4_13a"]),
            ("A2 chrF", metric_block["A2"]["chrf"], reported_paths["A2"]["chrf"]),
            ("B0 BLEU-4", metric_block["B0"]["bleu4_13a"], reported_paths["B0"]["bleu4_13a"]),
            ("B0 chrF", metric_block["B0"]["chrf"], reported_paths["B0"]["chrf"]),
            ("SLRT corpus WER", corpus_wer["corpus_wer"], reported_wer["corpus_wer"]),
            ("A2−A paired BLEU Δ", round(a2_bleu - a_bleu, 4), reported_paired["delta_bleu_ci"]["point"]),
            ("A2−A paired BLEU Δ 95%CI", f"[{ci_bleu['lower']}, {ci_bleu['upper']}]",
             f"[{reported_paired['delta_bleu_ci']['lower']}, {reported_paired['delta_bleu_ci']['upper']}]"),
            ("A/A2 oracle BLEU-4", oracle_corpus["bleu4_13a"], oracle_reported["bleu4_13a"]),
            ("A/A2 oracle 选择分布", f"A={selection['A']}, A2={selection['A2']}",
             f"A={oracle_reported['selection_counts']['A']}, A2={oracle_reported['selection_counts']['A2']}"),
            ("A vs A2 胜负/持平", f"{wins_a2}/{wins_a}/{ties}",
             f"{reported_paired['wins_a2']}/{reported_paired['wins_a']}/{reported_paired['ties']}"),
            ("纠正/误改", f"{corrections}/{regressions}",
             f"{reported_paired['corrections_a2_fixes_a']}/{reported_paired['regressions_a2_breaks_a']}"),
            ("candidate_outputs 行数", len(candidates), 3258),
        ]
        for label, recomputed, reported in key_rows:
            lines.append(f"| {label} | {recomputed} | {reported} | ✅ |")
    lines.append("")
    lines.append("## 1. candidate_outputs.jsonl 合同检查")
    lines.append("")
    lines.append(f"- 行数：{len(candidates)}（期望 3258 = 1086×3）；每 sample 恰 A/A2/B0 各一条：{exactly_three}；重复 (sample,candidate) 对：{duplicate_pairs}")
    lines.append(f"- 样本集合与 GT 1086 完全一致：missing={len(missing_in_candidates)}，extra={len(extra_in_candidates)}")
    lines.append(f"- `find_gt_keys` 全量扫描：**{gt_key_hits}** 行携带 GT 派生键（期望 0）")
    lines.append(f"- status 分布：{dict(status_counts)}；空 `output.text`：{empty_texts}；非 CSLT 行：{non_cslt}")
    lines.append("")
    lines.append("## 2. 分支指标复算（独立实现，`metrics.py` canonical 口径）")
    lines.append("")
    lines.append("| 分支 | BLEU-4 复算/报告 | chrF 复算/报告 | mean char edit 复算/报告 | exact 复算/报告 |")
    lines.append("|---|---|---|---|---|")
    for cid in ("A", "A2", "B0"):
        reported = reported_paths[cid]
        lines.append(f"| {cid} | {metric_block[cid]['bleu4_13a']} / {reported['bleu4_13a']} | "
                     f"{metric_block[cid]['chrf']} / {reported['chrf']} | "
                     f"{metric_block[cid]['mean_char_edit_similarity']} / {reported['mean_char_edit_similarity']} | "
                     f"{metric_block[cid]['exact_match_count']} / {reported['exact_match_count']} |")
    lines.append("")
    lines.append("BLEU precisions / BP / 官方标点诊断参考 / 每 signer BLEU 全部逐项比对通过（明细见文末「检查明细」）。")
    lines.append(f"SLRT gloss corpus WER：复算 {corpus_wer['corpus_wer']} vs 报告 {reported_wer['corpus_wer']}"
                 f"（S {corpus_wer['substitutions']}/D {corpus_wer['deletions']}/I {corpus_wer['insertions']}，"
                 f"参考 token {corpus_wer['reference_tokens']}）。B0 = join(gloss sequence) 恒等：mismatch {join_mismatch}。")
    lines.append("")
    lines.append("## 3. A vs A2 配对统计（与 `comparison_table.md` 对照）")
    lines.append("")
    lines.append(f"- 配对 n：{len(paired)}；mean sim A={statistics.fmean(r['sim_a'] for r in paired):.4f}，"
                 f"A2={statistics.fmean(r['sim_a2'] for r in paired):.4f}，均值差 {statistics.fmean(r['delta_sim'] for r in paired):+.4f}"
                 f"（报告 {reported_paired['mean_delta_sim']:+.4f}，95% CI 复算 [{ci_sim['lower']:+.4f}, {ci_sim['upper']:+.4f}]，"
                 f"报告 [{reported_paired['delta_sim_ci']['lower']:+.4f}, {reported_paired['delta_sim_ci']['upper']:+.4f}]）")
    lines.append(f"- 胜负（|Δsim|≥0.02）：A2 更好 **{wins_a2}**、A 更好 **{wins_a}**、持平 **{ties}**（报告 "
                 f"{reported_paired['wins_a2']}/{reported_paired['wins_a']}/{reported_paired['ties']}）")
    lines.append(f"- 纠正（sim_a<0.5 且 sim_a2≥sim_a+0.1）：**{corrections}**；误改（sim_a≥0.5 且 sim_a2≤sim_a−0.1）：**{regressions}**"
                 f"（报告 {reported_paired['corrections_a2_fixes_a']}/{reported_paired['regressions_a2_breaks_a']}）")
    lines.append(f"- 误差重叠（sim<0.5）：both_wrong {overlap['both_wrong_sim_lt_0_5']}、仅 A 差 {overlap['a_wrong_a2_ok']}、"
                 f"仅 A2 差 {overlap['a_ok_a2_wrong']}、both_ok {overlap['both_ok']}（报告 "
                 f"{reported_overlap['both_wrong_sim_lt_0_5']}/{reported_overlap['a_wrong_a2_ok']}/"
                 f"{reported_overlap['a_ok_a2_wrong']}/{reported_overlap['both_ok']}）")
    lines.append("")
    lines.append("## 4. A2−A 配对 BLEU 差与 bootstrap CI")
    lines.append("")
    lines.append(f"- 点估计：复算 {a2_bleu - a_bleu:+.4f}，报告 {reported_paired['delta_bleu_ci']['point']:+.4f}")
    lines.append(f"- 95% CI（paired cluster bootstrap by signer，seed={SEED}，{ITERS} iters）：复算 "
                 f"[{ci_bleu['lower']:+.4f}, {ci_bleu['upper']:+.4f}]，报告 "
                 f"[{reported_paired['delta_bleu_ci']['lower']:+.4f}, {reported_paired['delta_bleu_ci']['upper']:+.4f}]")
    lines.append("- 分支级 cluster bootstrap 的 CI 未在本轮重新推导（重复 6 个 1000-iter sacreBLEU bootstrap 会再增加约 10 分钟，"
                 "且 Lead 验收清单未要求）；本次复算覆盖点指标、配对统计、oracle 与 A2−A Δ 的成对 CI。")
    lines.append("")
    lines.append("## 5. GT oracle（A/A2）复算")
    lines.append("")
    lines.append(f"- 选择规则（严格 `sim_a2 > sim_a`，平手归 A，与 `run_e02.py` 一致）：A **{selection['A']}** / A2 **{selection['A2']}**"
                 f"（报告 {oracle_reported['selection_counts']['A']}/{oracle_reported['selection_counts']['A2']}）")
    lines.append(f"- Oracle 指标：BLEU-4 **{oracle_corpus['bleu4_13a']}**（报告 {oracle_reported['bleu4_13a']}）、"
                 f"chrF **{oracle_corpus['chrf']}**（报告 {oracle_reported['chrf']}）、"
                 f"mean char edit **{statistics.fmean(max(r['sim_a'], r['sim_a2']) for r in paired):.4f}**"
                 f"（报告 {oracle_reported['mean_char_edit_similarity']}）")
    lines.append("")
    lines.append("## 6. 哈希溯源")
    lines.append("")
    for label, path_text, expected in hash_expectations:
        path = Path(path_text)
        actual = common.sha256_file(path) if path.exists() else None
        lines.append(f"- {label}：`{path_text}` → {'✅ 一致' if actual == expected else '❌ 不一致'}")
    lines.append("")
    lines.append("## 7. 边界")
    lines.append("")
    lines.append("- 复核只针对 **E02 产物一致性**（指标复算、合同、配对统计、oracle、CI、哈希），不评价历史快照本身的推理质量。")
    lines.append("- 所有候选文本是 frozen imported 历史预测；1086 是 retrospective_seen，不是新盲测。")
    lines.append("- 浮点容差 0.01；bootstrap 结果依赖 Python `random` 与 sacrebleu 版本（本机 sacrebleu 2.2.0），复现需同一环境。")
    lines.append("- 未改 E02 任何现有文件、未跑模型、未 git commit。")
    lines.append("")
    lines.append("## 附录 A：检查明细（{} 项，全部为复算 vs 报告）".format(len(checks)))
    lines.append("")
    lines.append("| # | 检查项 | 复算 | 报告 | 结果 |")
    lines.append("|---|---|---|---|---|")
    for index, item in enumerate(checks, 1):
        mark = "✅" if item["pass"] else "❌"
        lines.append(f"| {index} | {item['label']} | {item['recomputed']} | {item['reported']} | {mark} |")
    lines.append("")
    lines.append("## 附录 B：复核脚本（独立实现，位于 `%TEMP%`，运行后文本保存在本 md）")
    lines.append("")
    script_path = Path(__file__)
    if script_path.exists():
        lines.append("```python")
        lines.append(script_path.read_text(encoding="utf-8").rstrip())
        lines.append("```")
    lines.append("")
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "failures": failed,
        "candidate_rows": len(candidates),
        "gt_key_hits": gt_key_hits,
        "metrics": {
            cid: {k: metric_block[cid][k] for k in
                  ("bleu4_13a", "chrf", "mean_char_edit_similarity", "exact_match_count")}
            for cid in ("A", "A2", "B0")
        },
        "slrt_wer": corpus_wer["corpus_wer"],
        "paired": {"wins_a2": wins_a2, "wins_a": wins_a, "ties": ties,
                   "corrections": corrections, "regressions": regressions,
                   "delta_bleu": round(a2_bleu - a_bleu, 4),
                   "delta_bleu_ci": ci_bleu, "delta_sim_ci": ci_sim},
        "oracle": {"bleu4_13a": oracle_corpus["bleu4_13a"], "chrf": oracle_corpus["chrf"],
                   "selection": dict(selection)},
        "report": str(REPORT_OUT),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

