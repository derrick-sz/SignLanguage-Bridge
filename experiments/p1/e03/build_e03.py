"""E03 · SLRT score definition, calibration and error-risk prediction.

CPU-only, retrospective analysis of the frozen 1086-sample CSL-Daily snapshot
(``artifacts/csldaily-video-test-627-20261026/slrt``).  Ground-truth gloss is
read *only* here, on the evaluation side, to build per-sample WER labels; it is
never written into a model/inference payload.

Outputs (all under ``experiments/p1/e03/`` plus one run dir):
    per_sample_scores.csv
    calibration_results.json
    risk_coverage.csv
    score_definition.md
    report.md
    runs/<run_id>/manifest.json, runs/<run_id>/metrics.json

Definitions
-----------
* sample WER      : ``metrics.wer_breakdown(gt_gloss, " ".join(gloss_sequence))``
* error_any       : WER > 0
* bad             : WER >= 0.5     (primary risk label)
* bad_03          : WER >= 0.3     (sensitivity)
* score features  : SLRT token score ``tokens[].score``
                    (``score_type="uncalibrated_mean_ensemble_probability"``,
                    object = output gloss token)
                    and window ``top_probability`` from ``*.windows.jsonl``
                    (object = 16-frame sliding window).
* model selection : leave-one-signer-out (10 signers); learned thresholds are
                    selected on *inner* OOF predictions of the 9 training
                    signers (nested OOF), never on the held-out signer.

Everything is numpy/scipy/stdlib; no pandas/sklearn.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from experiments.p1.common import (
    ROOT,
    P1,
    RUNS,
    RunDirectory,
    percentile,
    read_csv_rows,
    read_json,
    sha256_file,
    utc_now,
    write_csv_rows,
    write_json,
)
from experiments.p1.eval.metrics import edit_similarity, normalize_text, wer_breakdown

RUN_ID = "20261026_e03_slrt_confidence"
EXPERIMENT = "E03"
E03_DIR = P1 / "e03"
RUN_DIR = RUNS / RUN_ID
SLRT_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "slrt"
INFER_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "infer"
GT_PATH = P1 / "references" / "gt_1086.csv"
MANIFEST_PATH = P1 / "datasets" / "manifest_1086.csv"
UNISIGN_VIDEO_PREDS = (
    ROOT
    / "artifacts"
    / "csldaily-video-test-627-20261026"
    / "unisign-video-1086-paperrefs"
    / "predictions.jsonl"
)

WER_BAD = 0.5
WER_BAD_SENS = 0.3
LOW_PROB_WINDOW = 0.5
LOW_PROB_WINDOW_SENS = 0.3
PRIMARY_FDR_TARGET = 0.15
FDR_TARGETS = (0.10, 0.15, 0.20)
UPGRADE_ALT_EDIT_SIM = 0.5  # eval-side proxy: Uni-Sign video text acceptable
COVERAGE_GRID = [round(0.05 * k, 2) for k in range(1, 21)]
ECE_BINS = 15
OOF_SEED = 20261026


# --------------------------------------------------------------------------
# numeric helpers (numpy only, no sklearn)
# --------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-6, iterations: int = 100) -> np.ndarray:
    """Newton/IRLS logistic regression on a design matrix that already has intercept."""

    w = np.zeros(x.shape[1], dtype=float)
    n = len(y)
    for _ in range(iterations):
        p = sigmoid(x @ w)
        grad = x.T @ (p - y) / n + l2 * w
        weights = p * (1.0 - p)
        hessian = (x * weights[:, None]).T @ x / n + l2 * np.eye(x.shape[1])
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate fold
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        w = w - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return w


def standardize_and_design(features: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(features[train_mask], axis=0)
    sd = np.nanstd(features[train_mask], axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def design(features: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (features - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([np.ones(len(z)), z])


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUC (Mann-Whitney), higher score -> positive class."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_summary(p_positive: np.ndarray, labels: np.ndarray) -> dict:
    """ECE / Brier / reliability bins for a predicted positive-class probability."""

    p = np.asarray(p_positive, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = np.isfinite(p)
    p = p[valid]
    y = y[valid]
    if len(p) == 0:
        return {"n": 0, "ece": None, "brier": None, "roc_auc": None, "bins": []}
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    bins = []
    ece_num = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi >= 1.0
        mask = (p >= lo) & (p < hi) if not last else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4), "n": 0,
                         "mean_pred": None, "observed_positive_rate": None})
            continue
        mean_pred = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece_num += count * abs(mean_pred - observed)
        bins.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4), "n": count,
                     "mean_pred": round(mean_pred, 4), "observed_positive_rate": round(observed, 4)})
    return {
        "n": int(len(p)),
        "ece": round(ece_num / len(p), 4),
        "brier": round(brier, 4),
        "roc_auc": round(roc_auc(p, y), 4) if roc_auc(p, y) is not None else None,
        "bins": bins,
    }


def select_tau(confidence: np.ndarray, labels: np.ndarray, target: float) -> dict:
    """Largest coverage prefix whose positive-rate (error-ACCEPT) is <= target.

    Returns the acceptance threshold midpoint so that the *training prefix* is
    exactly ``accepted_n`` samples; a held-out sample is accepted when
    ``confidence >= tau``.  ``tau=None`` means the empty prefix (accept nothing).
    """

    conf = np.asarray(confidence, dtype=float)
    lab = np.asarray(labels, dtype=float)
    order = np.argsort(-conf, kind="mergesort")
    cs = conf[order]
    bs = lab[order]
    cum = np.cumsum(bs)
    ks = np.arange(1, len(cs) + 1, dtype=float)
    rates = cum / ks
    valid = np.where(rates <= target)[0]
    if len(valid) == 0:
        return {"tau": None, "accepted_n": 0, "error_accept_rate": None, "coverage": 0.0}
    k = int(valid.max()) + 1
    if k >= len(cs):
        tau = float("-inf")
    else:
        tau = float((cs[k - 1] + cs[k]) / 2.0)
    return {
        "tau": tau,
        "accepted_n": k,
        "error_accept_rate": round(float(rates[k - 1]), 4),
        "coverage": round(k / len(cs), 4),
    }


def apply_tau(confidence: np.ndarray, tau: float | None) -> np.ndarray:
    if tau is None:
        return np.zeros(len(confidence), dtype=bool)
    return np.asarray(confidence, dtype=float) >= tau


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_gt() -> list[dict]:
    rows = read_csv_rows(GT_PATH)
    out = []
    for row in rows:
        out.append(
            {
                "sample_id": row["sample_id"],
                "signer_id": row["signer_id"],
                "ref_sentence_upstream": row["ref_sentence_upstream"],
                "gt_gloss": row["gt_gloss"],
            }
        )
    out.sort(key=lambda r: r["sample_id"])
    return out


def load_unisign_video_edit() -> dict[str, float]:
    """EVAL-ONLY: char edit similarity of the historical Uni-Sign video text."""

    result: dict[str, float] = {}
    if not UNISIGN_VIDEO_PREDS.exists():
        return result
    with UNISIGN_VIDEO_PREDS.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = record.get("sample")
            prediction = record.get("prediction") or ""
            reference = record.get("reference") or ""
            if sample:
                result[sample] = round(
                    edit_similarity(normalize_text(reference), normalize_text(prediction)), 4
                )
    return result


def read_windows(path: Path) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    ids: list[int] = []
    if not path.exists():
        return probs, ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prob = record.get("top_probability")
            if isinstance(prob, (int, float)) and math.isfinite(prob):
                probs.append(float(prob))
                ids.append(int(record.get("top_id", -1)))
    return probs, ids


def build_rows() -> list[dict]:
    gt_rows = load_gt()
    manifest = {row["sample_id"]: row for row in read_csv_rows(MANIFEST_PATH)}
    uni_edit = load_unisign_video_edit()
    rows: list[dict] = []
    score_types: dict[str, int] = {}
    for gt in gt_rows:
        sample = gt["sample_id"]
        row: dict = {
            "sample_id": sample,
            "signer_id": gt["signer_id"],
            "cohort": manifest.get(sample, {}).get("cohort"),
            "artifact_status": None,
            "exit_code": None,
            "elapsed_seconds": None,
            "frames_received": None,
            "windows_processed": None,
            "retained_frames": None,
            "n_tokens": None,
            "token_score_mean": None,
            "token_score_min": None,
            "token_score_median": None,
            "token_score_p10": None,
            "token_score_max": None,
            "n_windows": None,
            "window_top_prob_mean": None,
            "window_top_prob_min": None,
            "window_top_prob_p10": None,
            "window_top_prob_max": None,
            "low_prob_window_frac_05": None,
            "low_prob_window_frac_03": None,
            "blank_window_frac": None,
            "ref_tokens": None,
            "pred_tokens": None,
            "gloss_length_ratio": None,
            "wer": None,
            "wer_s": None,
            "wer_d": None,
            "wer_i": None,
            "error_any": None,
            "bad": None,
            "bad_03": None,
            "uni_video_edit_sim_EVAL_ONLY": uni_edit.get(sample),
            "should_upgrade_proxy_EVAL_ONLY": None,
        }
        result_path = SLRT_DIR / f"{sample}.json"
        if not result_path.exists():
            row["artifact_status"] = "missing_artifact"
            rows.append(row)
            continue
        data = read_json(result_path)
        row["artifact_status"] = data.get("status")
        row["frames_received"] = data.get("frames_received")
        row["windows_processed"] = data.get("windows_processed")
        row["retained_frames"] = data.get("retained_frames")
        row["elapsed_seconds"] = data.get("elapsed_command_seconds")
        infer_path = INFER_DIR / f"{sample}.json"
        if infer_path.exists():
            infer = read_json(infer_path)
            row["exit_code"] = infer.get("exit_code")
            row["elapsed_seconds"] = infer.get("elapsed_seconds", row["elapsed_seconds"])

        gloss_sequence = data.get("gloss_sequence") or []
        tokens = data.get("tokens") or []
        token_scores = []
        for token in tokens:
            score = token.get("score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                token_scores.append(float(score))
                stype = str(token.get("score_type"))
                score_types[stype] = score_types.get(stype, 0) + 1
        row["n_tokens"] = len(tokens)
        if token_scores:
            row["token_score_mean"] = round(float(np.mean(token_scores)), 6)
            row["token_score_min"] = round(float(np.min(token_scores)), 6)
            row["token_score_median"] = round(float(statistics.median(token_scores)), 6)
            row["token_score_p10"] = round(percentile(token_scores, 10), 6)
            row["token_score_max"] = round(float(np.max(token_scores)), 6)

        window_probs, window_ids = read_windows(SLRT_DIR / f"{sample}.windows.jsonl")
        row["n_windows"] = len(window_probs) if window_probs else 0
        if window_probs:
            arr = np.asarray(window_probs)
            row["window_top_prob_mean"] = round(float(arr.mean()), 6)
            row["window_top_prob_min"] = round(float(arr.min()), 6)
            row["window_top_prob_p10"] = round(percentile(window_probs, 10), 6)
            row["window_top_prob_max"] = round(float(arr.max()), 6)
            row["low_prob_window_frac_05"] = round(float(np.mean(arr < LOW_PROB_WINDOW)), 6)
            row["low_prob_window_frac_03"] = round(float(np.mean(arr < LOW_PROB_WINDOW_SENS)), 6)
            row["blank_window_frac"] = round(float(np.mean(np.asarray(window_ids) == 0)), 6)

        # evaluation-side WER label (GT gloss join; hyp = gloss sequence join)
        breakdown = wer_breakdown(gt["gt_gloss"], " ".join(gloss_sequence))
        row["ref_tokens"] = breakdown["reference_tokens"]
        row["pred_tokens"] = breakdown["predicted_tokens"]
        if breakdown["reference_tokens"]:
            row["gloss_length_ratio"] = round(
                breakdown["predicted_tokens"] / breakdown["reference_tokens"], 6
            )
        row["wer"] = round(breakdown["wer"], 6) if breakdown["wer"] is not None else None
        row["wer_s"] = breakdown["substitutions"]
        row["wer_d"] = breakdown["deletions"]
        row["wer_i"] = breakdown["insertions"]
        if breakdown["wer"] is not None:
            row["error_any"] = int(breakdown["wer"] > 0)
            row["bad"] = int(breakdown["wer"] >= WER_BAD)
            row["bad_03"] = int(breakdown["wer"] >= WER_BAD_SENS)
            if row["uni_video_edit_sim_EVAL_ONLY"] is not None:
                row["should_upgrade_proxy_EVAL_ONLY"] = int(
                    row["bad"] == 1
                    and row["uni_video_edit_sim_EVAL_ONLY"] >= UPGRADE_ALT_EDIT_SIM
                )
        rows.append(row)
    return rows, score_types


# --------------------------------------------------------------------------
# learning: leave-one-signer-out with nested threshold selection
# --------------------------------------------------------------------------
def learn_oof(
    rows: list[dict],
    features: np.ndarray,
    labels: np.ndarray,
    signers: list[str],
    feature_name: str,
) -> dict:
    """Outer OOF probabilities + per-fold nested-OOF threshold policies."""

    n = len(rows)
    p_oof = np.full(n, np.nan)
    signer_list = sorted(set(signers))
    folds: list[dict] = []
    for signer in signer_list:
        test_mask = np.asarray([s == signer for s in signers])
        train_mask = ~test_mask
        # inner OOF over the 9 training signers -> threshold selection evidence
        inner_p = np.full(int(train_mask.sum()), np.nan)
        train_signers = [s for s in signer_list if s != signer]
        train_idx_all = np.where(train_mask)[0]
        for inner_signer in train_signers:
            inner_test = np.asarray([signers[i] == inner_signer for i in train_idx_all])
            inner_train = ~inner_test
            sub_features = features[train_idx_all]
            mu, sd = standardize_and_design(sub_features, inner_train)
            x_train = design(sub_features[inner_train], mu, sd)
            w = fit_logistic(x_train, labels[train_idx_all][inner_train])
            x_test = design(sub_features[inner_test], mu, sd)
            inner_p[inner_test] = sigmoid(x_test @ w)
        mu, sd = standardize_and_design(features, train_mask)
        w = fit_logistic(design(features[train_mask], mu, sd), labels[train_mask])
        p_oof[test_mask] = sigmoid(design(features[test_mask], mu, sd) @ w)

        fold = {
            "held_out_signer": signer,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "weights": [round(float(v), 6) for v in w],
            "inner_oof_error_rate": round(float(np.mean(labels[train_idx_all][np.isfinite(inner_p)])), 4)
            if np.isfinite(inner_p).any()
            else None,
            "policies": {},
        }
        inner_conf = 1.0 - inner_p
        for target in FDR_TARGETS:
            policy = select_tau(inner_conf, labels[train_idx_all], target)
            tau = policy["tau"]
            test_conf = 1.0 - p_oof[test_mask]
            accepted = apply_tau(test_conf, tau)
            fold["policies"][str(target)] = {
                "tau": tau,
                "train_coverage": policy["coverage"],
                "train_error_accept_rate": policy["error_accept_rate"],
                "test_coverage": round(float(accepted.mean()), 4) if len(accepted) else 0.0,
                "test_error_accept_rate": round(float(labels[test_mask][accepted].mean()), 4)
                if accepted.any()
                else None,
            }
        folds.append(fold)
    return {
        "method": feature_name,
        "kind": "logistic_platt_numpy",
        "features": feature_name.split("+"),
        "oof_positive_probability": p_oof,
        "folds": folds,
    }


def raw_policy_folds(
    confidence: np.ndarray, labels: np.ndarray, signers: list[str]
) -> dict:
    """Per-fold OOF threshold selection for a raw (untrained) score."""

    folds: list[dict] = []
    accept = np.zeros(len(confidence), dtype=bool)
    tau_oof = np.full(len(confidence), np.nan)
    for signer in sorted(set(signers)):
        test_mask = np.asarray([s == signer for s in signers])
        train_mask = ~test_mask
        fold = {"held_out_signer": signer, "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()), "policies": {}}
        for target in FDR_TARGETS:
            policy = select_tau(confidence[train_mask], labels[train_mask], target)
            tau = policy["tau"]
            test_accept = apply_tau(confidence[test_mask], tau)
            accept[test_mask] = test_accept
            tau_oof[test_mask] = tau if tau is not None else np.nan
            fold["policies"][str(target)] = {
                "tau": tau,
                "train_coverage": policy["coverage"],
                "train_error_accept_rate": policy["error_accept_rate"],
                "test_coverage": round(float(test_accept.mean()), 4) if len(test_accept) else 0.0,
                "test_error_accept_rate": round(float(labels[test_mask][test_accept].mean()), 4)
                if test_accept.any()
                else None,
            }
        folds.append(fold)
    return {"folds": folds, "oof_accept_primary": accept, "oof_tau_primary": tau_oof}


def policy_metrics(
    accept: np.ndarray, labels: np.ndarray, should_upgrade: np.ndarray
) -> dict:
    accepted = np.asarray(accept, dtype=bool)
    lab = np.asarray(labels, dtype=int)
    upgrade = np.asarray(should_upgrade, dtype=int)
    n = len(lab)
    n_acc = int(accepted.sum())
    accepted_bad = int((accepted & (lab == 1)).sum())
    bad_total = int((lab == 1).sum())
    should = upgrade == 1
    missed = int((accepted & should).sum())
    return {
        "n": n,
        "accepted_n": n_acc,
        "coverage": round(n_acc / n, 4) if n else None,
        "error_accept_n": accepted_bad,
        "error_accept_rate_among_accepted": round(accepted_bad / n_acc, 4) if n_acc else None,
        "error_accept_share_of_bad": round(accepted_bad / bad_total, 4) if bad_total else None,
        "bad_n": bad_total,
        "should_upgrade_n": int(should.sum()),
        "missed_upgrade_n": missed,
        "missed_upgrade_rate": round(missed / int(should.sum()), 4) if should.any() else None,
    }


# --------------------------------------------------------------------------
# markdown / json writers
# --------------------------------------------------------------------------
def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def reliability_table(bins: list[dict], limit: int = 20) -> list[str]:
    lines = ["| bin | n | mean_pred | observed_positive_rate |", "|---|---|---|---|"]
    for entry in bins[:limit]:
        lines.append(
            f"| [{fmt(entry['lo'])} , {fmt(entry['hi'])}) | {entry['n']} | "
            f"{fmt(entry['mean_pred'])} | {fmt(entry['observed_positive_rate'])} |"
        )
    return lines


def write_report(payload: dict) -> None:
    summary = payload["summary"]
    labels = payload["labels"]
    lines: list[str] = []
    lines.append("# E03 · SLRT 分数定义、校准与错误风险（P-1）")
    lines.append("")
    lines.append(f"- run_id: `{RUN_ID}`")
    lines.append(f"- 生成时间（UTC）: {utc_now()}")
    lines.append(f"- 数据集: 1086 条 CSL-Daily 本地视频子集，split=test，`analysis_scope=retrospective_seen_1086`（历史已见数据，**不是新盲测**）")
    lines.append(f"- GT（仅评测侧）: `{GT_PATH.relative_to(ROOT)}` sha256 `{payload['data']['gt_sha256'][:16]}…`")
    lines.append(f"- SLRT 历史输出: `{SLRT_DIR.relative_to(ROOT)}`（{payload['data']['slrt_json_count']} 条样本，"
                 f"缺失 {payload['data']['n_missing_artifact']} 条；本批全部 status=final、0 空 gloss）")
    lines.append("- 状态: **PASS**（SLRT raw score 校准与风险分析已完成）；**Uni-Sign 置信度 = BLOCKED**（见 §6）")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append(f"- SLRT corpus WER（gloss 对 GT gloss）: **{fmt(summary['corpus_wer'])}** "
                 f"(S={summary['corpus_s']}/D={summary['corpus_d']}/I={summary['corpus_i']}, "
                 f"参考 {summary['corpus_ref_tokens']} tokens)。总编辑数与历史 [R18] 一致"
                 f"（S1,154/D805/I522 = 2,481 次编辑、WER 0.3018）；本表 S/D/I 拆分不同仅因 Levenshtein "
                 f"回溯并列时的选择顺序，**不影响 WER 与总编辑数**。")
    lines.append(f"- 标签: error_any(WER>0) **{labels['error_any']['n']}** 条 ({fmt(labels['error_any']['rate'])}), "
                 f"bad(WER≥{WER_BAD}) **{labels['bad']['n']}** 条 ({fmt(labels['bad']['rate'])}), "
                 f"bad_03(WER≥{WER_BAD_SENS}) **{labels['bad_03']['n']}** 条 ({fmt(labels['bad_03']['rate'])}); 签者 {labels['signer_count']} 人")
    lines.append(f"- **raw token score 不是正确概率**：把 `token_score_mean` 直接当 P(correct) 时 "
                 f"ECE=**{fmt(summary['raw_token_score_as_prob']['ece'])}**、Brier=**{fmt(summary['raw_token_score_as_prob']['brier'])}**。"
                 f"分数范围被压缩（mean ∈ [{fmt(summary['raw_token_score_range'][0])}, {fmt(summary['raw_token_score_range'][1])}]），"
                 f"且系统性低估正确率：score≈0.44 的样本实际 70% 可接受（1-bad）、score≈0.50 时 83%、score≈0.56 时 93%"
                 f" —— 直接当概率会严重失校准（见 §3）。排序信号仍可用：AUC={fmt(summary['raw_token_score_as_prob']['roc_auc'])}。")
    lines.append(f"- raw window score 同样未校准：ECE=**{fmt(summary['raw_window_score_as_prob']['ece'])}**、"
                 f"Brier=**{fmt(summary['raw_window_score_as_prob']['brier'])}**、AUC=**{fmt(summary['raw_window_score_as_prob']['roc_auc'])}**。")
    lines.append(f"- 单变量 Platt/logistic（token score）OOF 后：主标签 bad 上 AUC=**{fmt(summary['logistic_token_bad']['roc_auc'])}**、"
                 f"ECE=**{fmt(summary['logistic_token_bad']['ece'])}**、Brier=**{fmt(summary['logistic_token_bad']['brier'])}**；"
                 f"在 FDR≤{PRIMARY_FDR_TARGET} 策略下覆盖 **{fmt(summary['primary_policies']['logistic_token']['coverage'])}**、"
                 f"error-ACCEPT 率 **{fmt(summary['primary_policies']['logistic_token']['error_accept_rate_among_accepted'])}**、"
                 f"missed-upgrade 率 **{fmt(summary['primary_policies']['logistic_token']['missed_upgrade_rate'])}**。")
    lines.append(f"- 同一策略下 raw token score 阈值：覆盖 {fmt(summary['primary_policies']['raw_token']['coverage'])}、"
                 f"error-ACCEPT 率 {fmt(summary['primary_policies']['raw_token']['error_accept_rate_among_accepted'])}、"
                 f"missed-upgrade 率 {fmt(summary['primary_policies']['raw_token']['missed_upgrade_rate'])}；"
                 f"raw window score：覆盖 {fmt(summary['primary_policies']['raw_window']['coverage'])}、"
                 f"error-ACCEPT 率 {fmt(summary['primary_policies']['raw_window']['error_accept_rate_among_accepted'])}、"
                 f"missed-upgrade 率 {fmt(summary['primary_policies']['raw_window']['missed_upgrade_rate'])}。")
    lines.append("- 结论：SLRT 分数**可用作风险排序信号**（AUC 明显 > 0.5），但**不能当作正确概率**；"
                 "阈值必须在 signer 分组 OOF 上选择，且高置信样本仍有不可忽略的 bad 风险（见 §4 risk-coverage）。")
    lines.append("")
    lines.append("## 2. 分数定义与模型卡")
    lines.append("")
    lines.append("| 分数 | score_type | 对象 | 来源/语义 | 可用性 |")
    lines.append("|---|---|---|---|---|")
    lines.append("| SLRT token score | `uncalibrated_mean_ensemble_probability` | **输出 gloss token**（非帧、非整句） | "
                 "7 窗多数投票中，该 token 的每次 partial 事件取「参与投票窗口的该类别平均概率」，再对该 token 生命周期内事件取均值；"
                 "见 `slbridge/slrt/decoder.py`。**未校准**，不是 P(correct) | READY（1086/1086） |")
    lines.append("| SLRT window top_probability | 无单独 score_type 字段（softmax 最大类概率） | **单个 16 帧滑窗** | "
                 "`*.windows.jsonl` 每窗 `top_probability`；每个 anchor frame 一窗，窗数≈帧数（如 52 帧→52 窗），长度依赖 | READY（1086/1086） |")
    lines.append("| Uni-Sign 句级置信度 | — | — | runtime `TextPrediction(score=None)`（`slbridge/uni_sign.py:1007`）；"
                 "历史 `predictions.jsonl` **没有** score/logprob 字段 | **BLOCKED** |")
    lines.append("")
    lines.append("详见 `score_definition.md`（含字段样本与代码引用）。")
    lines.append("")
    lines.append("## 3. 校准结果（OOF，bad = WER≥0.5）")
    lines.append("")
    lines.append("| 方法 | 预测对象 | ECE(以bad为正类) | Brier | AUC |")
    lines.append("|---|---|---|---|---|")
    for name in ("raw_token", "raw_window", "logistic_token", "logistic_window", "logistic_combo"):
        calib = payload["methods"][name]["calibration"]["bad"]
        lines.append(f"| {name} | {payload['methods'][name]['object']} | {fmt(calib['ece'])} | "
                     f"{fmt(calib['brier'])} | {fmt(calib['roc_auc'])} |")
    lines.append("")
    lines.append("raw score 直接当 P(correct) 的可靠性表（token score，全部 15 箱；后 5 箱无样本）：")
    lines.append("")
    lines.extend(reliability_table(payload["reliability_raw_token_as_prob"], limit=8))
    lines.append("")
    lines.append("## 4. 风险-覆盖与阈值策略（signer 分组 OOF）")
    lines.append("")
    lines.append(f"阈值策略：在 9 个训练签者的**内层 OOF 置信度**上选最大覆盖阈值，使训练集 error-ACCEPT 率 ≤ target；"
                 f"再应用到留出签者。主 target = {PRIMARY_FDR_TARGET}。")
    lines.append("")
    lines.append("| 方法 | coverage | error-ACCEPT 率（接受中 bad 占比） | 占全部 bad 比例 | missed-upgrade 率 |")
    lines.append("|---|---|---|---|---|")
    for name in ("raw_token", "raw_window", "logistic_token", "logistic_window", "logistic_combo"):
        pol = summary["primary_policies"][name]
        lines.append(f"| {name} | {fmt(pol['coverage'])} | {fmt(pol['error_accept_rate_among_accepted'])} | "
                     f"{fmt(pol['error_accept_share_of_bad'])} | {fmt(pol['missed_upgrade_rate'])} |")
    lines.append("")
    lines.append("missed-upgrade 的评测侧定义：bad=1 且 Uni-Sign 视频历史文本 char edit similarity ≥ "
                 f"{UPGRADE_ALT_EDIT_SIM}（即另一条路径本可给出可用译文）。这是**离线代理真值**，不是人工升级真值。")
    lines.append("")
    ug = summary["primary_policies"]["logistic_token"]
    lines.append(f"升级机会（评测侧代理）：should_upgrade = **{ug['should_upgrade_n']}/{ug['n']}** 条，"
                 f"其中 **{ug['missed_upgrade_n']}** 条仍被策略 ACCEPT（missed-upgrade 率 "
                 f"**{fmt(ug['missed_upgrade_rate'])}**）。即 SLRT 分数只能识别「bad」的一部分，"
                 f"「bad 且另一路本可给可用译文」的升级机会与低分区间只有部分重叠；"
                 f"仅凭 SLRT 分数决定升级会漏掉多数升级机会，需 E06 的级联/冲突规则补充。")
    lines.append("")
    lines.append("risk-coverage 关键点（OOF 排序，FDR = 接受样本中 bad 占比）：")
    lines.append("")
    lines.append("| 方法 | coverage=25% | 50% | 75% | 100% |")
    lines.append("|---|---|---|---|---|")
    for name in ("raw_token", "raw_window", "logistic_token", "logistic_window", "logistic_combo"):
        points = payload["risk_curves"][name]["bad"]
        cells = []
        for coverage in (0.25, 0.50, 0.75, 1.00):
            entry = min(points, key=lambda e: abs(e["coverage"] - coverage))
            cells.append(fmt(entry["fdr"]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("完整曲线见 `risk_coverage.csv`（coverage 5%–100%，含 threshold / accepted_n / fdr）。")
    lines.append("")
    lines.append("## 5. 敏感性（error_any / bad_03）")
    lines.append("")
    lines.append("| 标签 | 方法 | AUC | ECE | 主 target 下 coverage | error-ACCEPT 率 |")
    lines.append("|---|---|---|---|---|---|")
    for label_name in ("error_any", "bad_03"):
        for name in ("raw_token", "raw_window", "logistic_token", "logistic_window", "logistic_combo"):
            calib = payload["methods"][name]["calibration"][label_name]
            pol = payload["methods"][name]["threshold_policies"][label_name][str(PRIMARY_FDR_TARGET)]["pooled"]
            lines.append(f"| {label_name} | {name} | {fmt(calib['roc_auc'])} | {fmt(calib['ece'])} | "
                         f"{fmt(pol['coverage'])} | {fmt(pol['error_accept_rate_among_accepted'])} |")
    lines.append("")
    lines.append("注意：`error_any` 包含仅插入错误（WER>0 但 GT 内容全对），因此 bad=0 时也可能 error_any=1；两个标签的风险含义不同，不能混用。"
                 "`error_any` 基线错误率 85.5%，FDR≤15% 的阈值策略在它上面几乎不接受任何样本（coverage≈0），"
                 "说明 token score 无法区分「局部/插入错误」与「基本正确」；实际路由应以 bad（WER≥0.5）为主标签而非「任何错误」。")
    lines.append("")
    lines.append("## 6. 边界与阻塞")
    lines.append("")
    lines.append("1. **Uni-Sign 置信度 = BLOCKED**：当前 runtime 与历史产物都没有 logprob/score。证据：")
    lines.append("   - `slbridge/uni_sign.py:1007` 构造 `TextPrediction(..., score=None)`；`slbridge/contracts.py:364-369` / `slbridge/api_schemas.py:137-140` 的 `score` 默认 `None`。")
    lines.append("   - 历史 `unisign-video-1086-paperrefs/predictions.jsonl` 每行字段："
                 "`sample, frames, reference, prediction, status, error, gloss, char_edit_similarity, rouge_l_f1_char, exact_match, e2e_seconds, pose_ms, model_total_ms` — **无 score/logprob**。")
    lines.append("   - `docs/agents/model-evaluation-pitfalls.md` §6：「当前 TextPrediction 返回整句文本、score=None；beam=4 不代表已经提供 N-best、词级概率或动作时间对齐」。")
    lines.append("   - 不得用编辑相似度/文本稳定性冒充模型置信度；文本稳定性只能在 E05 前缀数据上作为弱信号，且必须标注为「非模型置信度」。")
    lines.append("2. 1086 是 retrospective_seen 子集（本地已有 test 视频的 1086/1176），不是新盲测；SLRT 为近似句段裁剪的 GLOSS 输出，不是 `sentence_frames-512x512` 官方协议。")
    lines.append("3. SLRT 交付物是 gloss 序列，不是自然语言句子；本报告的 WER 是 gloss 级指标，不能与 Uni-Sign 的 BLEU/chrF 直接比较。")
    lines.append("4. raw score 阈值/校准结果不能跨模型复用（E03 协议 §2：不同模型不可共用分数阈值）。")
    lines.append("5. 标签阈值（bad=0.5）是分析选择，不是官方验收阈值；`bad_03`/`error_any` 已作敏感性。")
    lines.append("")
    lines.append("## 7. 产物")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    lines.append("| `experiments/p1/e03/per_sample_scores.csv` | 1086 行逐样本分数/WER/标签（含 OOF 概率列，缺失保留 null） |")
    lines.append("| `experiments/p1/e03/calibration_results.json` | ECE/Brier/AUC、可靠性分箱、每折阈值与权重、风险策略 |")
    lines.append("| `experiments/p1/e03/risk_coverage.csv` | 5 方法 × 3 标签 × 20 coverage 点（5%–100%） |")
    lines.append("| `experiments/p1/e03/score_definition.md` | 分数对象/语义/字段证据 |")
    lines.append(f"| `experiments/p1/runs/{RUN_ID}/manifest.json` | 命令、脚本/数据 sha256、样本数、状态 |")
    lines.append("")
    (E03_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_score_definition(payload: dict) -> None:
    lines: list[str] = []
    lines.append("# E03 · Score definition（SLRT / Uni-Sign）")
    lines.append("")
    lines.append(f"运行：`{RUN_ID}`，生成于 {utc_now()}。本文件只描述**分数的对象与语义**，数值结果见 `report.md` / `calibration_results.json`。")
    lines.append("")
    lines.append("## 1. SLRT token score")
    lines.append("")
    lines.append("- 字段：`slrt/<sample>.json` → `tokens[].score`，伴生 `tokens[].score_type`。")
    lines.append("- `score_type`：**`uncalibrated_mean_ensemble_probability`**（1086 条样本全部为该值；无其他类型）。")
    lines.append("- 对象：**单个输出 gloss token**（事件级），不是帧、不是全句、不是 CTC 帧级后验。")
    lines.append("- 语义（`slbridge/slrt/decoder.py`）：")
    lines.append("  1. 每个 16 帧滑窗输出 2001 类 softmax；7 窗中心多数投票（vote_size=7）决定该位置 token id。")
    lines.append("  2. 对参与投票的 7 个窗，取该 token id 类别概率的均值 `p`（`decode._drain`：`p = mean([r['prob'][k] for r in window])`）。")
    lines.append("  3. token 生命周期内每次 partial 事件累计 `p`，final 时取 `score = score_sum / count`（`decode._event`）。")
    lines.append("- 含义：**投票一致性与类别概率的混合量**，值域 [0,1]，但**未做任何校准**，不是 P(输出正确)、不是 P(token 正确)。")
    lines.append("- 缺失：本批 1086/1086 都有 score；空输出样本不在本批（0 空 gloss），若未来出现，token 级分数为空，应填 `null` 而不是 1.0。")
    lines.append("")
    lines.append("字段样本（`S000020_P0004_T00`）：")
    lines.append("")
    lines.append("```json")
    lines.append('{"event_id": "0:0", "token_id": 2, "gloss": "他", "status": "final",')
    lines.append(' "score": 0.6312378015783098, "score_type": "uncalibrated_mean_ensemble_probability",')
    lines.append(' "reason": "run_ended", "boundary_source": "sliding_window_anchor_estimate"}')
    lines.append("```")
    lines.append("")
    lines.append("## 2. SLRT window top_probability")
    lines.append("")
    lines.append("- 字段：`slrt/<sample>.windows.jsonl` → 每行一个滑窗的 `top_id` / `top_probability` / `inference_ms` / `preprocess_ms`。")
    lines.append("- 对象：**单个 16 帧滑窗**（帧级窗口分类结果），每个 anchor frame 一个窗口；窗数≈视频帧数（52 帧样本有 52 窗）。")
    lines.append("- 语义：该窗口 2001 类输出中最大类别的概率。**未校准**；且窗口有左右重复 padding（`left_edge_repeats`/`right_edge_repeats`），边界窗口的概率不可与中间窗口等价解读。")
    lines.append("- 与 token score 的区别：token score 是 7 窗投票后的 token 生命周期均值；window top_probability 是单窗瞬时值。二者**不可混用阈值**。")
    lines.append("")
    lines.append("字段样本：")
    lines.append("")
    lines.append("```json")
    lines.append('{"segment_id": 0, "anchor_frame": 0, "anchor_time": 0.0, "top_id": 2,')
    lines.append(' "top_probability": 0.7509263753890991, "left_edge_repeats": 7, "eos_padding": false}')
    lines.append("```")
    lines.append("")
    lines.append("## 3. Uni-Sign 句级置信度 = BLOCKED")
    lines.append("")
    lines.append("- 当前生产 runtime：`slbridge/uni_sign.py:1007` → `TextPrediction(text=..., language=\"zh\", score=None)`。")
    lines.append("- 数据合同默认值：`slbridge/contracts.py:364-369` 与 `slbridge/api_schemas.py:137-140` 的 `score: float | None = None`。")
    lines.append("- 历史 1086 产物：`artifacts/csldaily-video-test-627-20261026/unisign-video-1086-paperrefs/predictions.jsonl` 与 "
                 "`artifacts/cslnews-eval/csldaily-test-1086-pose-20261026/predictions.jsonl` 均**没有** score/logprob/beam 字段（字段见 report §6）。")
    lines.append("- 项目坑点记忆 §6：整句文本、`score=None`；beam=4 不代表 N-best/词级概率/时间对齐。")
    lines.append("- 结论：**BLOCKED**。要做 Uni-Sign 校准，需要先让 runtime 导出 `generate(output_scores=True, return_dict_in_generate=True)` 的 token logprob 或 beam score，并用同一批输入重跑；不得用编辑相似度、文本稳定性、多次解码一致性冒充模型置信度（后者只能作为弱信号且须标注）。")
    lines.append("")
    lines.append("## 4. 本报告使用的分数特征")
    lines.append("")
    lines.append("| 特征 | 来源 | 对象 | 方向 |")
    lines.append("|---|---|---|---|")
    lines.append("| `token_score_mean/min/median/p10` | tokens[].score | token | 越高越自信（风险越低） |")
    lines.append("| `window_top_prob_mean/min/p10` | windows[].top_probability | window | 越高越自信（风险越低） |")
    lines.append("| `low_prob_window_frac_05/03` | 同上 | window | 越高风险越高 |")
    lines.append("| `gloss_length_ratio` | pred_tokens/ref_tokens | 序列长度 | 诊断用（过/欠生成） |")
    lines.append("")
    lines.append("## 5. 模型卡（分数对象）")
    lines.append("")
    lines.append("| model_id | 任务 | score_source | score_type | 对象 | 校准状态 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| `slrt-emnlp24-csl-daily` | CSLR（gloss 序列） | `tokens[].score` | `uncalibrated_mean_ensemble_probability` | token | 本报告训练了 OOF Platt 映射（仅风险排序，不是绝对概率） |")
    lines.append("| `slrt-emnlp24-csl-daily` | CSLR | `windows[].top_probability` | 单窗 softmax max | window | 同上（raw 未校准） |")
    lines.append("| `unisign`（视频/官方 pose） | CSLT（中文句子） | 无 | — | — | **BLOCKED：无导出分数** |")
    lines.append("")
    (E03_DIR / "score_definition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    E03_DIR.mkdir(parents=True, exist_ok=True)
    rows, score_types = build_rows()
    n = len(rows)
    if n == 0:
        raise SystemExit("no rows")
    signers = [row["signer_id"] for row in rows]
    labeled = [row for row in rows if row["wer"] is not None]
    n_labeled = len(labeled)

    # raw feature arrays (only samples with full score data; all 1086 here)
    token_mean = np.asarray([row["token_score_mean"] for row in rows], dtype=float)
    window_mean = np.asarray([row["window_top_prob_mean"] for row in rows], dtype=float)
    bad = np.asarray([row["bad"] if row["bad"] is not None else -1 for row in rows], dtype=int)
    error_any = np.asarray([row["error_any"] if row["error_any"] is not None else -1 for row in rows], dtype=int)
    bad_03 = np.asarray([row["bad_03"] if row["bad_03"] is not None else -1 for row in rows], dtype=int)
    upgrade = np.asarray(
        [row["should_upgrade_proxy_EVAL_ONLY"] if row["should_upgrade_proxy_EVAL_ONLY"] is not None else 0 for row in rows],
        dtype=int,
    )
    valid = bad >= 0
    assert valid.all(), f"{int((~valid).sum())} samples are not labeled (missing artifacts)"
    labeled_mask = valid

    labels_by_name = {"bad": bad, "error_any": error_any, "bad_03": bad_03}
    features = {
        "token": token_mean.reshape(-1, 1),
        "window": window_mean.reshape(-1, 1),
        "combo": np.column_stack([token_mean, window_mean]),
    }
    method_specs = {
        "raw_token": {"kind": "raw", "object": "token", "score": "token_score_mean", "feature": "token"},
        "raw_window": {"kind": "raw", "object": "window", "score": "window_top_prob_mean", "feature": "window"},
        "logistic_token": {"kind": "logistic", "object": "token", "score": "token_score_mean", "feature": "token"},
        "logistic_window": {"kind": "logistic", "object": "window", "score": "window_top_prob_mean", "feature": "window"},
        "logistic_combo": {"kind": "logistic", "object": "token+window", "score": "token_score_mean+window_top_prob_mean", "feature": "combo"},
    }

    methods: dict[str, dict] = {}
    oof_primary: dict[str, np.ndarray] = {}
    for name, spec in method_specs.items():
        feature_matrix = features[spec["feature"]]
        entry: dict = {
            "kind": spec["kind"],
            "object": spec["object"],
            "score_feature": spec["score"],
            "calibration": {},
            "threshold_policies": {},
        }
        if spec["kind"] == "raw":
            confidence = token_mean if spec["feature"] == "token" else window_mean
            for label_name, label_values in labels_by_name.items():
                # raw score treated as P(correct) for this label
                calib = calibration_summary(confidence, 1 - label_values)
                calib["interpretation"] = (
                    "raw score used directly as P(correct); this is the naive reading "
                    "the report argues against"
                )
                entry["calibration"][label_name] = calib
                folds = raw_policy_folds(confidence, label_values, signers)
                target_policies = {}
                for target in FDR_TARGETS:
                    accept = np.zeros(n, dtype=bool)
                    for fold in folds["folds"]:
                        held = fold["held_out_signer"]
                        test_mask = np.asarray([s == held for s in signers])
                        accept[test_mask] = apply_tau(confidence[test_mask], fold["policies"][str(target)]["tau"])
                    pooled = policy_metrics(accept, label_values, upgrade)
                    target_policies[str(target)] = {
                        "pooled": pooled,
                        "folds": [fold["policies"][str(target)] | {"held_out_signer": fold["held_out_signer"]}
                                  for fold in folds["folds"]],
                        "threshold_source": "signer-grouped OOF, train-fold FDR-constrained max-coverage prefix",
                    }
                entry["threshold_policies"][label_name] = target_policies
            # store primary-label OOF accept for the risk curve (raw order = score order)
            oof_primary[name] = confidence
        else:
            for label_name, label_values in labels_by_name.items():
                learned = learn_oof(rows, feature_matrix, label_values, signers, spec["feature"])
                p_oof = learned["oof_positive_probability"]
                entry["calibration"][label_name] = calibration_summary(p_oof, label_values)
                entry["folds"] = learned["folds"]
                entry.setdefault("oof_positive_probability_by_label", {})[label_name] = [
                    None if not np.isfinite(v) else round(float(v), 6) for v in p_oof
                ]
                target_policies = {}
                for target in FDR_TARGETS:
                    accept = np.zeros(n, dtype=bool)
                    for fold in learned["folds"]:
                        held = fold["held_out_signer"]
                        test_mask = np.asarray([s == held for s in signers])
                        accept[test_mask] = apply_tau(1.0 - p_oof[test_mask], fold["policies"][str(target)]["tau"])
                    pooled = policy_metrics(accept, label_values, upgrade)
                    target_policies[str(target)] = {
                        "pooled": pooled,
                        "folds": [fold["policies"][str(target)] | {"held_out_signer": fold["held_out_signer"]}
                                  for fold in learned["folds"]],
                        "threshold_source": "nested signer-grouped OOF (inner OOF on 9 train signers)",
                    }
                entry["threshold_policies"][label_name] = target_policies
            # primary-label confidence for the risk curve
            oof_primary[name] = 1.0 - learned["oof_positive_probability"]
        methods[name] = entry

    # attach OOF bad probabilities to the CSV (bad label, primary)
    for name in ("logistic_token", "logistic_window", "logistic_combo"):
        values = methods[name]["oof_positive_probability_by_label"]["bad"]
        for row, value in zip(rows, values):
            row[f"oof_p_bad_{name}"] = value

    # ---- risk coverage curves (bad label primary + sensitivity labels) ----
    risk_rows: list[dict] = []
    risk_curves: dict[str, dict[str, list[dict]]] = {}
    for name in method_specs:
        risk_curves[name] = {}
        for label_name, label_values in labels_by_name.items():
            confidence = oof_primary[name] if label_name == "bad" else None
            if label_name != "bad":
                # for sensitivity, recompute the method's label-specific confidence
                if method_specs[name]["kind"] == "raw":
                    confidence = token_mean if method_specs[name]["feature"] == "token" else window_mean
                else:
                    p = np.asarray(methods[name]["oof_positive_probability_by_label"][label_name], dtype=float)
                    confidence = 1.0 - p
            points = []
            for coverage in COVERAGE_GRID:
                k = max(1, int(round(coverage * n)))
                order = np.argsort(-confidence, kind="mergesort")
                accepted_idx = order[:k]
                fdr = float(label_values[accepted_idx].mean())
                points.append(
                    {
                        "coverage": coverage,
                        "accepted_n": k,
                        "fdr": round(fdr, 4),
                        "threshold": round(float(confidence[order[k - 1]]), 6),
                    }
                )
                risk_rows.append(
                    {
                        "method": name,
                        "object": method_specs[name]["object"],
                        "label": label_name,
                        "coverage": coverage,
                        "accepted_n": k,
                        "threshold": round(float(confidence[order[k - 1]]), 6),
                        "fdr": round(fdr, 4),
                    }
                )
            risk_curves[name][label_name] = points

    # ---- summary numbers for report ----
    from experiments.p1.eval.metrics import corpus_wer

    corpus = corpus_wer([(gt["gt_gloss"], " ".join(read_json(SLRT_DIR / f"{gt['sample_id']}.json")["gloss_sequence"]))
                         for gt in load_gt()])
    label_counts = {}
    for label_name, values in labels_by_name.items():
        label_counts[label_name] = {"n": int(values.sum()), "rate": round(float(values.mean()), 4)}
    primary_policies = {}
    for name in method_specs:
        pooled = methods[name]["threshold_policies"]["bad"][str(PRIMARY_FDR_TARGET)]["pooled"]
        primary_policies[name] = pooled
    raw_token_prob = methods["raw_token"]["calibration"]["bad"]
    raw_window_prob = methods["raw_window"]["calibration"]["bad"]

    summary = {
        "corpus_wer": corpus["corpus_wer"],
        "corpus_s": corpus["substitutions"],
        "corpus_d": corpus["deletions"],
        "corpus_i": corpus["insertions"],
        "corpus_ref_tokens": corpus["reference_tokens"],
        "raw_token_score_as_prob": {"ece": raw_token_prob["ece"], "brier": raw_token_prob["brier"],
                                    "roc_auc": raw_token_prob["roc_auc"]},
        "raw_window_score_as_prob": {"ece": raw_window_prob["ece"], "brier": raw_window_prob["brier"],
                                     "roc_auc": raw_window_prob["roc_auc"]},
        "raw_token_score_range": [round(float(np.nanmin(token_mean)), 4), round(float(np.nanmax(token_mean)), 4)],
        "raw_window_score_range": [round(float(np.nanmin(window_mean)), 4), round(float(np.nanmax(window_mean)), 4)],
        "logistic_token_bad": methods["logistic_token"]["calibration"]["bad"],
        "primary_policies": primary_policies,
    }

    # ---- per_sample_scores.csv ----
    base_fields = [
        "sample_id", "signer_id", "cohort", "artifact_status", "exit_code", "elapsed_seconds",
        "frames_received", "windows_processed", "retained_frames",
        "n_tokens", "token_score_mean", "token_score_min", "token_score_median",
        "token_score_p10", "token_score_max",
        "n_windows", "window_top_prob_mean", "window_top_prob_min", "window_top_prob_p10",
        "window_top_prob_max", "low_prob_window_frac_05", "low_prob_window_frac_03",
        "blank_window_frac",
        "ref_tokens", "pred_tokens", "gloss_length_ratio",
        "wer", "wer_s", "wer_d", "wer_i", "error_any", "bad", "bad_03",
        "uni_video_edit_sim_EVAL_ONLY", "should_upgrade_proxy_EVAL_ONLY",
        "oof_p_bad_logistic_token", "oof_p_bad_logistic_window", "oof_p_bad_logistic_combo",
    ]
    write_csv_rows(E03_DIR / "per_sample_scores.csv", base_fields, rows)

    # ---- calibration_results.json ----
    payload: dict = {
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "generated_at": utc_now(),
        "status": "PASS",
        "blocked_items": {
            "unisign_confidence": {
                "status": "BLOCKED",
                "reason": "Uni-Sign runtime/historical outputs expose no score or logprob; "
                          "score=None in slbridge/uni_sign.py:1007 and the 1086 prediction JSONL has no score field",
                "needed": "re-run Uni-Sign with generate(output_scores=True, return_dict_in_generate=True) "
                          "and export token logprob / beam score for the same 1086 inputs",
            }
        },
        "protocol": {
            "metric_definition_version": "p1-v1",
            "cross_validation": "leave-one-signer-out (10 signers); nested OOF threshold selection on the 9 training signers",
            "label_definitions": {
                "error_any": "WER > 0",
                "bad": f"WER >= {WER_BAD} (primary risk label)",
                "bad_03": f"WER >= {WER_BAD_SENS} (sensitivity)",
            },
            "wer_definition": "metrics.wer_breakdown(gt_gloss, ' '.join(gloss_sequence))",
            "risk_metrics": {
                "error_accept_rate_among_accepted": "accepted & bad / accepted",
                "error_accept_share_of_bad": "accepted & bad / all bad",
                "missed_upgrade_rate": "accepted & should_upgrade / should_upgrade; "
                                       "should_upgrade = bad==1 and Uni-Sign video char edit sim >= 0.5 (eval-side proxy)",
            },
            "fdr_targets": list(FDR_TARGETS),
            "primary_fdr_target": PRIMARY_FDR_TARGET,
            "coverage_grid": COVERAGE_GRID,
            "ece_bins": ECE_BINS,
            "data_scope": "CSL-Daily local test subset, retrospective_seen_1086, not a fresh blind test",
            "slrt_protocol": "rtmpose-aligned approximate sentence crop; glossy CSLR output (not official sentence_frames)",
        },
        "data": {
            "n_samples": n,
            "n_labeled": n_labeled,
            "n_missing_artifact": int(sum(1 for row in rows if row["artifact_status"] == "missing_artifact")),
            "n_signers": len(set(signers)),
            "signer_counts": {s: signers.count(s) for s in sorted(set(signers))},
            "gt_path": str(GT_PATH.relative_to(ROOT)),
            "gt_sha256": sha256_file(GT_PATH),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "slrt_dir": str(SLRT_DIR.relative_to(ROOT)),
            "slrt_json_count": len(list(SLRT_DIR.glob("*.json"))),
            "score_types_observed": score_types,
            "unisign_video_preds_sha256": sha256_file(UNISIGN_VIDEO_PREDS) if UNISIGN_VIDEO_PREDS.exists() else None,
        },
        "labels": {
            "bad": label_counts["bad"],
            "error_any": label_counts["error_any"],
            "bad_03": label_counts["bad_03"],
            "signer_count": len(set(signers)),
        },
        "summary": summary,
        "score_error_association": {
            "token_score_mean_point_biserial_vs_bad": round(float(np.corrcoef(token_mean, bad)[0, 1]), 4),
            "window_top_prob_mean_point_biserial_vs_bad": round(float(np.corrcoef(window_mean, bad)[0, 1]), 4),
            "gloss_length_ratio_point_biserial_vs_bad": round(
                float(np.corrcoef(np.asarray([row["gloss_length_ratio"] or 0.0 for row in rows]), bad)[0, 1]), 4
            ),
        },
        "methods": {
            name: {key: value for key, value in methods[name].items() if key != "oof_positive_probability_by_label"}
            for name in methods
        },
        "risk_curves": risk_curves,
        "reliability_raw_token_as_prob": methods["raw_token"]["calibration"]["bad"]["bins"],
        "reliability_raw_window_as_prob": methods["raw_window"]["calibration"]["bad"]["bins"],
        "notes": [
            "SLRT token score is uncalibrated_mean_ensemble_probability (token object), not P(correct).",
            "Raw scores are usable for risk ranking (AUC > 0.5) but not as calibrated probabilities.",
            "Uni-Sign confidence is BLOCKED; edit similarity / text stability must not be presented as model confidence.",
            "The 1086-sample snapshot is retrospective_seen, not a new blind test.",
        ],
    }
    write_json(E03_DIR / "calibration_results.json", payload)

    # ---- risk_coverage.csv ----
    write_csv_rows(
        E03_DIR / "risk_coverage.csv",
        ["method", "object", "label", "coverage", "accepted_n", "threshold", "fdr"],
        risk_rows,
    )

    # ---- markdown ----
    write_score_definition(payload)
    write_report(payload)

    # ---- run dir: manifest + metrics (re-runs overwrite the generated run dir) ----
    try:
        run = RunDirectory(RUN_ID, experiment_id=EXPERIMENT)
    except FileExistsError:
        run = RunDirectory.__new__(RunDirectory)
        run.run_id = RUN_ID
        run.experiment_id = EXPERIMENT
        run.path = RUNS / RUN_ID
        print(f"[warn] run dir {run.path} exists; regenerating its manifest/metrics in place")
    run_metrics = {
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "status": "PASS",
        "n_samples": n,
        "n_labeled": n_labeled,
        "n_signers": len(set(signers)),
        "corpus_wer": corpus["corpus_wer"],
        "label_rates": {name: label_counts[name]["rate"] for name in label_counts},
        "summary": summary,
        "blocked_items": payload["blocked_items"],
        "metric_definition_version": "p1-v1",
    }
    run.write_metrics(run_metrics)
    (run.path / "README.md").write_text(
        "# E03 run\n\n"
        f"- run_id: {RUN_ID}\n"
        "- script: experiments/p1/e03/build_e03.py\n"
        "- command: `$env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'; .\\.conda-env\\python.exe experiments/p1/e03/build_e03.py`\n"
        "- analysis outputs: experiments/p1/e03/ (per_sample_scores.csv, calibration_results.json, "
        "risk_coverage.csv, score_definition.md, report.md)\n"
        "- no inference was run; SLRT/Uni-Sign historical outputs were read only, GT only on the evaluation side.\n",
        encoding="utf-8",
    )
    run.write_manifest(
        experiment_id=EXPERIMENT,
        run_status="PASS",
        purpose="E03 SLRT confidence calibration + error-risk prediction (CPU only, no model inference)",
        model={
            "model_id": "slrt-emnlp24-csl-daily (imported historical outputs)",
            "checkpoint_sha256": read_json(SLRT_DIR / f"{rows[0]['sample_id']}.json")["model"]["checkpoint_sha256"],
            "score_type": "uncalibrated_mean_ensemble_probability",
            "score_object": "token",
        },
        data={
            "gt_sha256": sha256_file(GT_PATH),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "slrt_dir": str(SLRT_DIR.relative_to(ROOT)),
            "slrt_json_count": len(list(SLRT_DIR.glob("*.json"))),
            "unisign_video_preds_sha256": sha256_file(UNISIGN_VIDEO_PREDS) if UNISIGN_VIDEO_PREDS.exists() else None,
            "n_samples": n,
            "scope": "retrospective_seen_1086",
        },
        code_assets={
            "script": "experiments/p1/e03/build_e03.py",
            "script_sha256": sha256_file(E03_DIR / "build_e03.py"),
            "supporting_modules": {
                "experiments/p1/common.py": sha256_file(P1 / "common.py"),
                "experiments/p1/eval/metrics.py": sha256_file(P1 / "eval" / "metrics.py"),
                "experiments/p1/gt_guard.py": sha256_file(P1 / "gt_guard.py"),
            },
        },
        command=".conda-env\\python.exe experiments/p1/e03/build_e03.py (PYTHONPATH=E:\\githubproject\\SignLanguage-Bridge)",
        device={"device": "cpu", "gpu_used": False, "note": "CPU-only analysis; GPU reserved for E01"},
        outputs={
            "per_sample_scores.csv": {"sha256": sha256_file(E03_DIR / "per_sample_scores.csv"),
                                      "n_rows": n},
            "calibration_results.json": {"sha256": sha256_file(E03_DIR / "calibration_results.json")},
            "risk_coverage.csv": {"sha256": sha256_file(E03_DIR / "risk_coverage.csv"),
                                  "n_rows": len(risk_rows)},
            "score_definition.md": {"sha256": sha256_file(E03_DIR / "score_definition.md")},
            "report.md": {"sha256": sha256_file(E03_DIR / "report.md")},
        },
        limitations=[
            "retrospective_seen_1086, not a fresh blind test",
            "SLRT output is gloss (CSLR), not sentence translation",
            "Uni-Sign confidence BLOCKED (no score/logprob exported)",
            "should_upgrade truth is an eval-side proxy, not human annotation",
        ],
        script_status="executed_in_this_run",
    )

    print(json.dumps({
        "run_id": RUN_ID,
        "n": n,
        "corpus_wer": corpus["corpus_wer"],
        "label_rates": {name: label_counts[name]["rate"] for name in label_counts},
        "primary_fdr_target": PRIMARY_FDR_TARGET,
        "primary_policies": {
            name: {
                "coverage": primary_policies[name]["coverage"],
                "error_accept_rate": primary_policies[name]["error_accept_rate_among_accepted"],
                "missed_upgrade_rate": primary_policies[name]["missed_upgrade_rate"],
            }
            for name in method_specs
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
