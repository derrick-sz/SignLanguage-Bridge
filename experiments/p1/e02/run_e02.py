"""E02 — same-task translation baselines and complementary-gain analysis.

This script replays the frozen 1086 historical predictions (A: Uni-Sign video,
A2: Uni-Sign released pose, SLRT: gloss tokens) through the canonical P-1
evaluator and adds a deterministic Gloss-to-Text baseline (B0).  It never
re-runs a model, so every number is labelled ``IMPORTED``/``DERIVED``; the
matched re-run evidence lives in E01.

Outputs (``experiments/p1/e02/`` and one run directory):

* ``candidate_outputs.jsonl``   GT-free candidates for A/A2/B0 (consumed by E06)
* ``slrt_gloss_predictions.jsonl`` / ``b_gloss_join_predictions.jsonl``
* ``translation_metrics.json``  canonical metrics, bootstrap CIs, paired deltas,
                                error overlap, GT-oracle upper bounds
* ``fusion_ablation.csv``       A/A2/B0 measured-vs-imported + blocked branches
* ``comparison_table.md``, ``error_analysis.md``, ``report.md``
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from experiments.p1 import common
from experiments.p1.eval import metrics as M
from experiments.p1.eval.evaluate import evaluate

HIST_VIDEO = common.SHARED_ARTIFACTS / "unisign-video-1086-paperrefs" / "predictions.jsonl"
HIST_POSE = common.ROOT / "artifacts" / "cslnews-eval" / "csldaily-test-1086-pose-20261026" / "predictions.jsonl"
SLRT_DIR = common.SHARED_ARTIFACTS / "slrt"

BOOTSTRAP_ITERATIONS = 1000
RNG_SEED = 20261026


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"e02-offline-1086-{stamp}"


def load_predictions(path: Path) -> dict[str, dict]:
    rows = {}
    for row in common.iter_jsonl(path):
        sample = row.get("sample") or row.get("sample_id")
        if sample:
            rows[str(sample)] = row
    return rows


def load_slrt_gloss() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(SLRT_DIR.glob("*.json")):
        if path.name.endswith((".events.jsonl", ".windows.jsonl")):
            continue
        record = common.read_json(path)
        name = record.get("name") or path.stem
        rows[name] = {
            "sample": name,
            "gloss_sequence": record.get("gloss_sequence") or [],
            "status": record.get("status"),
            "error": record.get("error"),
        }
    return rows


def build_b0(gloss_rows: dict[str, dict]) -> dict[str, dict]:
    """Deterministic Gloss-to-Text baseline: join gloss tokens in order."""

    rows: dict[str, dict] = {}
    for sample, record in gloss_rows.items():
        tokens = [str(token) for token in record.get("gloss_sequence") or []]
        text = "".join(tokens)
        rows[sample] = {
            "sample": sample,
            "prediction": text,
            "status": "ok" if tokens else "empty_output",
            "error": None if tokens else "empty gloss sequence",
        }
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    common.write_jsonl(path, rows)
    return path


def cluster_bootstrap(
    records: list[dict],
    statistic,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    group_key: str = "signer_id",
    seed: int = RNG_SEED,
) -> dict:
    return M.bootstrap_by_group(records, statistic, iterations=iterations, group_key=group_key, seed=seed)


def paired_cluster_bootstrap(
    pairs: list[dict],
    statistic,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    group_key: str = "signer_id",
    seed: int = RNG_SEED,
) -> dict:
    """Paired cluster bootstrap where each record carries both A and A2 values."""

    groups: dict[str, list[dict]] = {}
    for record in pairs:
        groups.setdefault(str(record[group_key]), []).append(record)
    keys = sorted(groups)
    import random

    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = []
        for _ in range(len(keys)):
            sample.extend(groups[rng.choice(keys)])
        value = statistic(sample)
        if value is not None:
            values.append(float(value))
    values.sort()
    point = statistic(pairs)
    if not values:
        return {"point": point, "lower": None, "upper": None}
    alpha = 0.025
    return {
        "point": round(point, 4) if point is not None else None,
        "lower": round(values[int(alpha * (len(values) - 1))], 4),
        "upper": round(values[int((1 - alpha) * (len(values) - 1))], 4),
        "iterations": iterations,
        "method": "paired_cluster_bootstrap_by_signer",
    }


def _bleu_only(items: list[dict]) -> float:
    """Fast BLEU-only statistic (bootstrap inner loop; pre-tokenized char strings)."""

    import sacrebleu

    hyps = [item.get("hypothesis_char") or M.character_separated(item["hypothesis"]) for item in items]
    refs = [item.get("reference_char") or M.character_separated(item["reference"]) for item in items]
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score


def bleu_of(items: list[dict]) -> float:
    return _bleu_only(items)


def mean_edit_of(items: list[dict]) -> float:
    return statistics.fmean(item["char_edit_similarity"] for item in items)


def build_candidate_outputs(
    video: dict[str, dict], pose: dict[str, dict], b0: dict[str, dict], run_id: str
) -> list[dict]:
    samples = sorted(set(video) | set(pose) | set(b0))
    candidates: list[dict] = []
    specs = [
        ("A", "unisign_rgb", "RGB_video", video, "unisign-video-1086-paperrefs"),
        ("A2", "unisign_pose_released", "released_pose", pose, "csldaily-test-1086-pose-20261026"),
        ("B0", "slrt_twostream_slr__gloss_join_deterministic", "gloss_tokens", b0, "slrt-1086"),
    ]
    for sample in samples:
        for candidate_id, model_id, modality, source_rows, source_label in specs:
            row = source_rows.get(sample)
            text = None
            status = "SKIPPED"
            if row is not None:
                if "prediction" in row and row.get("prediction") is not None:
                    text = row.get("prediction")
                elif row.get("gloss_sequence"):
                    text = "".join(str(token) for token in row["gloss_sequence"])
                status = "OK" if row.get("status") in ("ok", "final") else str(row.get("status") or "FAILED").upper()
            candidates.append(
                {
                    "run_id": run_id,
                    "sample_id": sample,
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "task": "CSLT",
                    "sign_system": "CSL",
                    "modality": modality,
                    "model_revision": source_label,
                    "preprocess_version": "imported_historical_snapshot",
                    "window": {
                        "start_frame": 0,
                        "end_frame_exclusive": None,
                        "start_pts_ms": None,
                        "end_pts_ms": None,
                        "needs_future_frames": None,
                    },
                    "output": {
                        "kind": "TEXT",
                        "gloss_tokens": None,
                        "text": text,
                        "spoken_language": "zh",
                    },
                    "status": status if status in ("OK", "FAILED", "TIMEOUT", "SKIPPED", "BLOCKED") else "FAILED",
                }
            )
    return candidates


def main() -> int:
    run_id = _run_id()
    run = common.RunDirectory(run_id, experiment_id="E02")
    e02 = common.P1 / "e02"
    e02.mkdir(parents=True, exist_ok=True)
    references_path = common.REFERENCES / "gt_1086.csv"

    video = load_predictions(HIST_VIDEO)
    pose = load_predictions(HIST_POSE)
    gloss = load_slrt_gloss()
    b0 = build_b0(gloss)

    run.write_manifest(
        run_status="RUNNING",
        purpose="E02",
        scope="retrospective_seen_1086 (imported historical predictions; no model re-run)",
        code={"repository": "SignLanguage-Bridge", **common.git_state()},
        host={"host_alias": common.host_alias()},
        hardware={"device": "n/a (offline CPU evaluation)"},
        dataset={
            "dataset_name": "CSL-Daily",
            "split": "test",
            "scope": "retrospective_seen_1086",
            "manifest": "experiments/p1/datasets/manifest_1086.csv",
            "manifest_sha256": common.sha256_file(common.DATASETS / "manifest_1086.csv"),
            "gt_reference_path": str(references_path),
            "gt_reference_sha256": common.sha256_file(references_path),
            "fps_source": "not_applicable_for_text_metrics",
        },
        sources={
            "A_unisign_video": {"path": str(HIST_VIDEO), "sha256": common.sha256_file(HIST_VIDEO), "rows": len(video)},
            "A2_unisign_pose": {"path": str(HIST_POSE), "sha256": common.sha256_file(HIST_POSE), "rows": len(pose)},
            "slrt_gloss_dir": {"path": str(SLRT_DIR), "result_jsons": len(gloss)},
        },
        inference={"note": "imported snapshots; no inference in this run"},
        metrics={"metric_definition_version": M.METRIC_DEFINITION_VERSION},
    )

    # -- predictions files for the canonical evaluator ----------------------
    slrt_pred_path = write_jsonl(
        e02 / "slrt_gloss_predictions.jsonl",
        [
            {
                "sample": sample,
                "gloss_sequence": record["gloss_sequence"],
                "status": "final" if record.get("status") == "final" else (record.get("status") or "final"),
                "error": record.get("error"),
            }
            for sample, record in sorted(gloss.items())
        ],
    )
    b0_pred_path = write_jsonl(
        e02 / "b_gloss_join_predictions.jsonl",
        [
            {
                "sample": sample,
                "prediction": record["prediction"],
                "status": "ok" if record.get("status") == "ok" else "error",
                "error": record.get("error"),
            }
            for sample, record in sorted(b0.items())
        ],
    )
    candidates = build_candidate_outputs(video, pose, b0, run_id)
    candidate_path = e02 / "candidate_outputs.jsonl"
    write_jsonl(candidate_path, candidates)

    # -- canonical metrics ---------------------------------------------------
    evaluations: dict[str, dict] = {}
    paths = {
        "A_unisign_video": (HIST_VIDEO, "CSLT"),
        "A2_unisign_pose": (HIST_POSE, "CSLT"),
        "B0_gloss_join": (b0_pred_path, "CSLT"),
        "SLRT_gloss": (slrt_pred_path, "CSLR"),
    }
    for name, (path, task) in paths.items():
        payload, records = evaluate(path, references_path, task=task)
        evaluations[name] = {"payload": payload, "records": {row["sample_id"]: row for row in records}}

    # -- bootstrap CIs (cluster by signer) -----------------------------------
    bootstrap: dict[str, dict] = {}
    for name in ("A_unisign_video", "A2_unisign_pose", "B0_gloss_join"):
        records = []
        for sample, row in evaluations[name]["records"].items():
            records.append(
                {
                    "sample_id": sample,
                    "signer_id": row["signer_id"],
                    "reference": row["reference"],
                    "hypothesis": row["hypothesis"],
                    "reference_char": M.character_separated(row["reference"]),
                    "hypothesis_char": M.character_separated(row["hypothesis"]),
                    "char_edit_similarity": row["char_edit_similarity"],
                }
            )
        bootstrap[name] = {
            "bleu4_13a": cluster_bootstrap(records, bleu_of),
            "mean_char_edit_similarity": cluster_bootstrap(records, mean_edit_of),
        }

    # -- paired A vs A2 comparison ------------------------------------------
    common_samples = sorted(
        set(evaluations["A_unisign_video"]["records"]) & set(evaluations["A2_unisign_pose"]["records"])
    )
    paired = []
    for sample in common_samples:
        record_a = evaluations["A_unisign_video"]["records"][sample]
        record_a2 = evaluations["A2_unisign_pose"]["records"][sample]
        paired.append(
            {
                "sample_id": sample,
                "signer_id": record_a["signer_id"],
                "reference": record_a["reference"],
                "reference_char": M.character_separated(record_a["reference"]),
                "hypothesis_a": record_a["hypothesis"],
                "hypothesis_a2": record_a2["hypothesis"],
                "hypothesis_a_char": M.character_separated(record_a["hypothesis"]),
                "hypothesis_a2_char": M.character_separated(record_a2["hypothesis"]),
                "sim_a": record_a["char_edit_similarity"],
                "sim_a2": record_a2["char_edit_similarity"],
                "delta_sim": record_a2["char_edit_similarity"] - record_a["char_edit_similarity"],
            }
        )
    corrections = [row for row in paired if row["sim_a"] < 0.5 and row["sim_a2"] >= row["sim_a"] + 0.1]
    regressions = [row for row in paired if row["sim_a"] >= 0.5 and row["sim_a2"] <= row["sim_a"] - 0.1]
    wins_a2 = [row for row in paired if row["delta_sim"] >= 0.02]
    wins_a = [row for row in paired if row["delta_sim"] <= -0.02]
    ties = len(paired) - len(wins_a2) - len(wins_a)

    def paired_bleu_delta(items: list[dict]) -> float:
        return _bleu_only(
            [{"reference_char": row["reference_char"], "hypothesis_char": row["hypothesis_a2_char"]} for row in items]
        ) - _bleu_only(
            [{"reference_char": row["reference_char"], "hypothesis_char": row["hypothesis_a_char"]} for row in items]
        )

    paired_summary = {
        "n_paired": len(paired),
        "mean_sim_a": round(statistics.fmean(row["sim_a"] for row in paired), 4),
        "mean_sim_a2": round(statistics.fmean(row["sim_a2"] for row in paired), 4),
        "mean_delta_sim": round(statistics.fmean(row["delta_sim"] for row in paired), 4),
        "delta_sim_ci": paired_cluster_bootstrap(paired, lambda items: statistics.fmean(row["delta_sim"] for row in items)),
        "delta_bleu_ci": paired_cluster_bootstrap(paired, paired_bleu_delta),
        "wins_a2": len(wins_a2),
        "wins_a": len(wins_a),
        "ties": ties,
        "corrections_a2_fixes_a": len(corrections),
        "regressions_a2_breaks_a": len(regressions),
        "correction_definition": "A sim<0.5 and A2 sim>=A sim+0.1",
        "regression_definition": "A sim>=0.5 and A2 sim<=A sim-0.1",
        "example_corrections": [
            {"sample_id": row["sample_id"], "sim_a": row["sim_a"], "sim_a2": row["sim_a2"]} for row in corrections[:10]
        ],
        "example_regressions": [
            {"sample_id": row["sample_id"], "sim_a": row["sim_a"], "sim_a2": row["sim_a2"]} for row in regressions[:10]
        ],
    }

    # -- error overlap -------------------------------------------------------
    error_overlap = {
        "both_wrong_sim_lt_0_5": sum(
            1 for row in paired if row["sim_a"] < 0.5 and row["sim_a2"] < 0.5
        ),
        "a_wrong_a2_ok": sum(1 for row in paired if row["sim_a"] < 0.5 and row["sim_a2"] >= 0.5),
        "a_ok_a2_wrong": sum(1 for row in paired if row["sim_a"] >= 0.5 and row["sim_a2"] < 0.5),
        "both_ok": sum(1 for row in paired if row["sim_a"] >= 0.5 and row["sim_a2"] >= 0.5),
        "threshold": "char edit similarity 0.5 (analysis threshold, not a calibrated probability)",
    }

    # -- GT oracle upper bounds (offline only, not deployable) --------------
    def oracle_metrics(sources: list[str]) -> dict:
        hypotheses: list[str] = []
        refs: list[str] = []
        chosen: dict[str, int] = {source: 0 for source in sources}
        for row in paired:
            best_source = None
            best_sim = -1.0
            for source in sources:
                sim = row["sim_a"] if source == "A" else row["sim_a2"]
                if sim > best_sim:
                    best_sim = sim
                    best_source = source
            chosen[best_source] += 1
            hypotheses.append(row["hypothesis_a"] if best_source == "A" else row["hypothesis_a2"])
            refs.append(row["reference"])
        corpus = M.corpus_metrics_cslt(list(zip(refs, hypotheses)))
        return {
            "sources": sources,
            "n": len(hypotheses),
            "bleu4_13a": corpus["bleu4_13a"],
            "chrf": corpus["chrf"],
            "mean_char_edit_similarity": round(
                statistics.fmean(
                    max(row["sim_a"], row["sim_a2"]) for row in paired
                ),
                4,
            ),
            "selection_counts": chosen,
            "note": "offline GT oracle: picks the better candidate per sample using the reference; theoretical upper bound, not a deployable method",
        }

    oracle_a_a2 = oracle_metrics(["A", "A2"])

    # -- per-signer table ----------------------------------------------------
    per_signer: dict[str, dict] = {}
    for signer in sorted({row["signer_id"] for row in paired}):
        subset = [row for row in paired if row["signer_id"] == signer]
        per_signer[signer] = {
            "n": len(subset),
            "bleu4_a": M.corpus_metrics_cslt([(row["reference"], row["hypothesis_a"]) for row in subset])["bleu4_13a"],
            "bleu4_a2": M.corpus_metrics_cslt([(row["reference"], row["hypothesis_a2"]) for row in subset])["bleu4_13a"],
            "mean_sim_a": round(statistics.fmean(row["sim_a"] for row in subset), 4),
            "mean_sim_a2": round(statistics.fmean(row["sim_a2"] for row in subset), 4),
        }

    # -- legacy vs canonical -------------------------------------------------
    legacy = {
        "unisign_video_legacy_bleu4": 24.8156828061,
        "unisign_video_canonical_bleu4": evaluations["A_unisign_video"]["payload"]["metrics"]["bleu4_13a"],
        "legacy_source": "csldaily-video-comparison-1086.xlsx Summary (as documented in R18)",
        "slrt_legacy_corpus_wer": 0.3018248175,
        "slrt_recomputed_corpus_wer": evaluations["SLRT_gloss"]["payload"]["metrics"]["corpus_wer"],
        "note": "canonical recomputation over the same frozen snapshots; differences would indicate a metric-definition drift",
    }

    # -- fusion ablation table ----------------------------------------------
    ablation_rows: list[dict] = []

    def add_row(branch, description, status, payload=None, evidence=""):
        metrics = payload["metrics"] if payload else {}
        ablation_rows.append(
            {
                "branch": branch,
                "description": description,
                "status": status,
                "n": payload["successful"] if payload else "",
                "bleu4_13a": metrics.get("bleu4_13a", ""),
                "chrf": metrics.get("chrf", ""),
                "mean_char_edit_similarity": metrics.get("mean_char_edit_similarity", ""),
                "exact_match_count": metrics.get("exact_match_count", ""),
                "evidence": evidence,
            }
        )

    add_row("A", "Uni-Sign RGB/video -> sentence (frozen 1086 snapshot)", "IMPORTED_MEASURED", evaluations["A_unisign_video"]["payload"], "artifacts/.../unisign-video-1086-paperrefs")
    add_row("A2", "Uni-Sign released pose -> sentence (frozen 1086 snapshot)", "IMPORTED_MEASURED", evaluations["A2_unisign_pose"]["payload"], "artifacts/cslnews-eval/csldaily-test-1086-pose-20261026")
    add_row("B0", "SLRT Gloss -> deterministic token join (fallback baseline, not a translator)", "DERIVED_BASELINE", evaluations["B0_gloss_join"]["payload"], "experiments/p1/e02/b_gloss_join_predictions.jsonl")
    add_row("SLRT_gloss", "SLRT Gloss (CSLR) on frozen 1086 snapshot", "IMPORTED_MEASURED", None, "see translation_metrics.json SLRT_gloss")
    add_row("ORACLE_A_A2", "offline GT oracle over A/A2 (upper bound, not deployable)", "OFFLINE_ORACLE", None, "translation_metrics.json")
    blocked = [
        ("B1", "SLRT Gloss -> mBART full-vocab fallback G2T", "author-pruned mBART_zh and CSL-Daily G2T (Gloss2Text) weights are absent; full-vocab fallback assets exist but are unvalidated and need 1086 GPU generations -> not in this round"),
        ("C", "SLRT TwoStream-SLT -> sentence", "no CSL-Daily SLT/G2T checkpoint; only Phoenix-2014T slt_best.ckpt (different language) exists"),
        ("D", "SLRT Gloss -> LLM -> sentence", "no LLM endpoint/API key configured for experiments; fixed prompt/model version cannot be recorded"),
        ("E", "Uni-Sign RGB -> LLM (polish control)", "same LLM blocker"),
        ("F", "Uni-Sign + SLRT gloss -> LLM", "same LLM blocker"),
        ("G", "dual model + evidence-constrained LLM", "same LLM blocker"),
    ]
    for branch, description, reason in blocked:
        ablation_rows.append(
            {
                "branch": branch,
                "description": description,
                "status": "BLOCKED",
                "n": "",
                "bleu4_13a": "",
                "chrf": "",
                "mean_char_edit_similarity": "",
                "exact_match_count": "",
                "evidence": reason,
            }
        )

    # -- persist -------------------------------------------------------------
    common.write_csv_rows(
        e02 / "fusion_ablation.csv",
        list(ablation_rows[0].keys()),
        ablation_rows,
    )

    translation_metrics = {
        "metric_definition_version": M.METRIC_DEFINITION_VERSION,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "scope": "CSL-Daily test, retrospective_seen_1086, imported historical predictions (no model re-run)",
        "reference_caliber": "upstream test_tmp_refs.txt (half-width punctuation, character-separated)",
        "paths": {name: result["payload"] for name, result in evaluations.items()},
        "bootstrap_by_signer": bootstrap,
        "paired_a_vs_a2": paired_summary,
        "error_overlap": error_overlap,
        "oracle": {"a_a2": oracle_a_a2},
        "per_signer": per_signer,
        "legacy_vs_canonical": legacy,
        "human_review": {
            "status": "NOT_RUN",
            "reason": "requires human reviewers; automated metrics cannot be a substitute for the PLAN/02 §6 semantic review",
        },
        "boundaries": [
            "1086/1176 test videos; 90 long-sentence test videos are still missing locally",
            "A is the local video end-to-end path (RTMPose + approximate sentence-clip trimming), not the paper's pose protocol",
            "A2 uses the released pose archive and is the closest available in-domain reference path",
            "B0 is a deterministic join of SLRT gloss tokens, not a trained Gloss-to-Text model",
            "BLOCKED branches (B1/C/D/E/F/G) must not be counted as negative results",
        ],
    }
    common.write_json(e02 / "translation_metrics.json", translation_metrics)

    # comparison table markdown
    def fmt(value, digits=4):
        if value in ("", None):
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# E02 同任务翻译基线与互补性（imported 1086 快照）",
        "",
        f"- run_id：`{run_id}`",
        "- 样本：CSL-Daily test 1086 条（retrospective_seen，非新盲测）",
        "- 参考口径：上游 `test_tmp_refs.txt`（半角标点、逐字分隔）",
        "- 状态：A/A2 为历史快照重算（IMPORTED_MEASURED）；B0 为确定性 Gloss join（DERIVED_BASELINE）；C–G/B1 BLOCKED",
        "",
        "| 分支 | n | BLEU-4 | chrF | 平均字符编辑相似度 | 完全匹配 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['branch']} {('— ' + row['description']) if row['description'] else ''} | {fmt(row['n'])} | "
            f"{fmt(row['bleu4_13a'])} | {fmt(row['chrf'])} | {fmt(row['mean_char_edit_similarity'])} | "
            f"{fmt(row['exact_match_count'])} | {row['status']} |"
        )
    lines += [
        "",
        "## A vs A2 配对（同一批 1086 样本）",
        "",
        f"- A2−A 平均字符编辑相似度差：**{paired_summary['mean_delta_sim']:+.4f}**"
        f"（95% CI {paired_summary['delta_sim_ci']['lower']:+.4f} … {paired_summary['delta_sim_ci']['upper']:+.4f}）",
        f"- A2−A BLEU-4 差：**{paired_summary['delta_bleu_ci']['point']:+.4f}**"
        f"（95% CI {paired_summary['delta_bleu_ci']['lower']:+.4f} … {paired_summary['delta_bleu_ci']['upper']:+.4f}）",
        f"- 逐样本胜负：A2 更好 {paired_summary['wins_a2']}、A 更好 {paired_summary['wins_a']}、持平 {paired_summary['ties']}",
        f"- 纠正（A 差→A2 明显更好）：{paired_summary['corrections_a2_fixes_a']}；"
        f"误改（A 好→A2 明显更差）：{paired_summary['regressions_a2_breaks_a']}",
        "",
        "## 误差重叠（sim<0.5 视为差）",
        "",
        f"- 两者都差：{error_overlap['both_wrong_sim_lt_0_5']}；仅 A 差：{error_overlap['a_wrong_a2_ok']}；"
        f"仅 A2 差：{error_overlap['a_ok_a2_wrong']}；两者都好：{error_overlap['both_ok']}",
        "",
        "## GT Oracle 上限（仅离线分析，不可部署）",
        "",
        f"- A/A2 逐样本 oracle：BLEU-4 **{oracle_a_a2['bleu4_13a']}**，chrF **{oracle_a_a2['chrf']}**，"
        f"平均编辑相似度 **{oracle_a_a2['mean_char_edit_similarity']}**（选择分布 {oracle_a_a2['selection_counts']}）",
        "",
        "## Legacy vs Canonical",
        "",
        f"- Uni-Sign 视频 legacy BLEU-4 {legacy['unisign_video_legacy_bleu4']} → canonical {legacy['unisign_video_canonical_bleu4']}",
        f"- SLRT legacy corpus WER {legacy['slrt_legacy_corpus_wer']} → canonical {legacy['slrt_recomputed_corpus_wer']}",
    ]
    (e02 / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # error analysis markdown (examples are evaluation-only)
    examples = []
    for row in sorted(paired, key=lambda item: item["delta_sim"], reverse=True)[:5]:
        examples.append(
            f"| {row['sample_id']} | {row['sim_a']:.3f} | {row['sim_a2']:.3f} | {row['hypothesis_a']} | {row['hypothesis_a2']} |"
        )
    worst = []
    for row in sorted(paired, key=lambda item: item["delta_sim"])[:5]:
        worst.append(
            f"| {row['sample_id']} | {row['sim_a']:.3f} | {row['sim_a2']:.3f} | {row['hypothesis_a']} | {row['hypothesis_a2']} |"
        )
    error_lines = [
        "# E02 错误分析与互补候选（EVAL ONLY，含预测文本）",
        "",
        "> 本文件含 GT 相关分析，仅供离线报告；不得作为推理/融合输入。",
        "",
        "## A2 相对 A 的最大改进样本",
        "",
        "| sample | sim A | sim A2 | A 预测 | A2 预测 |",
        "|---|---:|---:|---|---|",
        *examples,
        "",
        "## A2 相对 A 的最大退步样本",
        "",
        "| sample | sim A | sim A2 | A 预测 | A2 预测 |",
        "|---|---:|---:|---|---|",
        *worst,
        "",
        "## 观察",
        "",
        f"- 两者都差 {error_overlap['both_wrong_sim_lt_0_5']} 条，是融合的主要目标区域；",
        f"- 仅 A2 好 {error_overlap['a_wrong_a2_ok']} 条、仅 A 好 {error_overlap['a_ok_a2_wrong']} 条，说明两路并非简单包含关系；",
        "- 逐样本胜负受句长、签者和词表影响，需在 E03/E06 中做无 GT 的路由可行性分析；",
        "- 未做人评；严重语义错误率需要人评协议（本文件不冒充人评）。",
    ]
    (e02 / "error_analysis.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    # report.md (docs/07 template condensed, measured vs imported explicit)
    report = f"""# E02 实验报告 — 同任务翻译基线与互补收益（离线回放）

## A. 实验基本信息
- 实验编号：E02
- run_id：`{run_id}`
- 状态：**PASS（离线回放部分）**；B1/C/D/E/F/G **BLOCKED**
- 样本范围：CSL-Daily test **1086 / 1176**（retrospective_seen_1086，10 名 signer；缺 90 条偏长句视频）
- 数据来源：冻结历史快照（A=Uni-Sign 视频端到端、A2=官方 pose、SLRT=Gloss），**本轮没有重新推理**
- 指标定义：`p1-v1`（sacreBLEU 13a + 上游 mt5 逐字加空格后处理；chrF 原始字符串；WER token 级）
- 参考口径：上游 `test_tmp_refs.txt`（半角标点、空格分字）；官方标注（全角标点）仅诊断列
- Git SHA：{common.git_state()['git_sha']}（dirty tree={common.git_state()['dirty_tree']}）

## B. 复现实验步骤
```bash
$env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
.\\.conda-env\\python.exe -m experiments.p1.e02.run_e02
```
- 输入快照与哈希见 `experiments/p1/runs/{run_id}/manifest.json`
- 数据台账：`experiments/p1/datasets/manifest_1086.csv`
- GT（eval-only）：`experiments/p1/references/gt_1086.csv`

## C. 结果（imported / derived，非本轮重新推理）
| 分支 | 样本数 | BLEU-4 | chrF | 平均字符编辑相似度 | 完全匹配 |
|---|---:|---:|---:|---:|---:|
| A Uni-Sign 视频 | {evaluations['A_unisign_video']['payload']['successful']} | {evaluations['A_unisign_video']['payload']['metrics']['bleu4_13a']} | {evaluations['A_unisign_video']['payload']['metrics']['chrf']} | {evaluations['A_unisign_video']['payload']['metrics']['mean_char_edit_similarity']} | {evaluations['A_unisign_video']['payload']['metrics']['exact_match_count']} |
| A2 Uni-Sign 官方 pose | {evaluations['A2_unisign_pose']['payload']['successful']} | {evaluations['A2_unisign_pose']['payload']['metrics']['bleu4_13a']} | {evaluations['A2_unisign_pose']['payload']['metrics']['chrf']} | {evaluations['A2_unisign_pose']['payload']['metrics']['mean_char_edit_similarity']} | {evaluations['A2_unisign_pose']['payload']['metrics']['exact_match_count']} |
| B0 Gloss 确定性 join（fallback baseline） | {evaluations['B0_gloss_join']['payload']['successful']} | {evaluations['B0_gloss_join']['payload']['metrics']['bleu4_13a']} | {evaluations['B0_gloss_join']['payload']['metrics']['chrf']} | {evaluations['B0_gloss_join']['payload']['metrics']['mean_char_edit_similarity']} | {evaluations['B0_gloss_join']['payload']['metrics']['exact_match_count']} |
| SLRT Gloss（CSLR, corpus WER） | {evaluations['SLRT_gloss']['payload']['successful']} | — | — | — | — |

- SLRT corpus WER：**{evaluations['SLRT_gloss']['payload']['metrics']['corpus_wer']}**（S {evaluations['SLRT_gloss']['payload']['metrics']['substitutions']} / D {evaluations['SLRT_gloss']['payload']['metrics']['deletions']} / I {evaluations['SLRT_gloss']['payload']['metrics']['insertions']}）
- Legacy vs canonical：视频 legacy BLEU {legacy['unisign_video_legacy_bleu4']} vs canonical {legacy['unisign_video_canonical_bleu4']}；SLRT legacy WER {legacy['slrt_legacy_corpus_wer']} vs {legacy['slrt_recomputed_corpus_wer']}
- Bootstrap（按 signer 聚类）与 A/A2 配对结果见 `translation_metrics.json` / `comparison_table.md`

## D. 错误分析
- A vs A2 误差并不完全重叠（两者都差 {error_overlap['both_wrong_sim_lt_0_5']} 条，仅 A 差 {error_overlap['a_wrong_a2_ok']}、仅 A2 差 {error_overlap['a_ok_a2_wrong']}）→ 存在可融合空间，但需要无 GT 的可观测信号（E03/E06）。
- 未做人评；`human_review.status = NOT_RUN`。严重语义错误率必须由人评协议给出，本文不替代。

## E. 结论与工程选择
- 是否支持原假设：**INCONCLUSIVE（质量提升潜力存在，但融合未实测）**
- 可引用结论：A/A2 两路在同一 1086 条上分别为 BLEU {evaluations['A_unisign_video']['payload']['metrics']['bleu4_13a']} / {evaluations['A2_unisign_pose']['payload']['metrics']['bleu4_13a']}；差 {paired_summary['delta_bleu_ci']['point']:+.4f}；逐样本强弱交叉存在。
- C 分支需先取得 CSL-Daily SLT/G2T 权重；B1 需作者剪枝 mBART 或验证 full-vocab fallback；D–G 需固定 LLM 端点与预算。
- 对 V3.0：保留双模型接口；生产默认单模型，融合需等 E06 无 GT 路由证据。
- 下一步最小行动：E03（SLRT score 校准）+ E06（Rule-only 路由）。

## 边界
- 1086 是 retrospective_seen，不是新盲测；所有结论不能外推到陌生 signer/新采集条件。
- 历史快照的推理协议见对应报告（R17/R18）；本报告只重算指标。
"""
    (e02 / "report.md").write_text(report, encoding="utf-8")
    (e02 / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "created_at": common.utc_now(),
                "frozen_historical_1086": {
                    "A": {"path": str(HIST_VIDEO), "sha256": common.sha256_file(HIST_VIDEO), "rows": len(video)},
                    "A2": {"path": str(HIST_POSE), "sha256": common.sha256_file(HIST_POSE), "rows": len(pose)},
                    "SLRT_gloss": {"path": str(SLRT_DIR), "result_jsons": len(gloss)},
                },
                "note": "read-only evidence; do not overwrite these artifacts",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # copy GT-free candidate outputs and metrics into the run directory
    common.write_jsonl(run.record("model_outputs.jsonl"), candidates)
    common.write_json(run.record("metrics.json"), translation_metrics)
    (run.record("report.md")).write_text(report, encoding="utf-8")
    run.update_manifest(
        run_status="PASS",
        outputs={
            "candidate_outputs_jsonl": str(candidate_path),
            "translation_metrics_json": str(e02 / "translation_metrics.json"),
            "fusion_ablation_csv": str(e02 / "fusion_ablation.csv"),
            "comparison_table_md": str(e02 / "comparison_table.md"),
            "error_analysis_md": str(e02 / "error_analysis.md"),
            "report_md": str(e02 / "report.md"),
        },
        counts={"samples": 1086, "candidates": len(candidates), "blocked_branches": len(blocked)},
    )

    print(json.dumps({"run_id": run_id, "A_bleu": evaluations["A_unisign_video"]["payload"]["metrics"]["bleu4_13a"],
                      "A2_bleu": evaluations["A2_unisign_pose"]["payload"]["metrics"]["bleu4_13a"],
                      "B0_bleu": evaluations["B0_gloss_join"]["payload"]["metrics"]["bleu4_13a"],
                      "slrt_wer": evaluations["SLRT_gloss"]["payload"]["metrics"]["corpus_wer"],
                      "oracle_bleu": oracle_a_a2["bleu4_13a"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
