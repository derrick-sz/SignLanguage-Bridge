"""Canonical P-1 metrics.

Definitions (metric_definition_version = ``p1-v1``) follow the frozen project
caliber so new runs are comparable with the historical 1086 table:

* BLEU-4: sacreBLEU ``tokenize="13a"`` on *character-separated* text, where both
  hypothesis and reference first go through the upstream mT5-rewrite
  ``' '.join(list(text.replace(' ', '').replace('\\n', '')))``.  Without that
  rewrite the same predictions collapse to ~0 BLEU (see pitfalls R10).
* chrF: sacreBLEU chrF on the raw prediction against the raw upstream reference
  (the reference file itself is character-separated, so spaces are part of the
  strings — identical to the existing evaluator's definition).
* WER: token-level Levenshtein on whitespace-separated gloss tokens; corpus WER
  is ``(ΣS + ΣD + ΣI) / ΣN_reference`` and is only reported for complete runs.
* char ROUGE-L F1 / edit similarity / exact match: on NFKC-normalized,
  whitespace-stripped text.
"""

from __future__ import annotations

import math
import random
import statistics
import unicodedata
from typing import Any, Callable, Iterable, Sequence

METRIC_DEFINITION_VERSION = "p1-v1"


def character_separated(text: str) -> str:
    """Upstream ``fix mt5 tokenizer bug`` rewrite (fine_tuning.py:282-284)."""

    return " ".join(list(str(text).replace(" ", "").replace("\n", "")))


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def lcs_length(a: str, b: str) -> int:
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for j, char_b in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if char_a == char_b else max(current[-1], previous[j]))
        previous = current
    return previous[-1]


def rouge_l_f1_char(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 1.0
    lcs = lcs_length(reference, hypothesis)
    precision = lcs / max(1, len(hypothesis))
    recall = lcs / max(1, len(reference))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def edit_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / max(1, len(a), len(b))


# --------------------------------------------------------------------------
# corpus metrics
# --------------------------------------------------------------------------
def corpus_metrics_cslt(
    pairs: Sequence[tuple[str, str]],
    *,
    alternate_references: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute sentence-translation metrics for ``(reference, hypothesis)`` pairs."""

    import sacrebleu

    refs_upstream = [ref for ref, _ in pairs]
    hyps = [hyp for _, hyp in pairs]
    refs_char = [character_separated(ref) for ref in refs_upstream]
    hyps_char = [character_separated(hyp) for hyp in hyps]

    bleu = sacrebleu.corpus_bleu(hyps_char, [refs_char], tokenize="13a")
    chrf = sacrebleu.corpus_chrf(hyps, [refs_upstream]).score

    normalized = [(normalize_text(ref), normalize_text(hyp)) for ref, hyp in pairs]
    edits = [edit_similarity(a, b) for a, b in normalized]
    rouges = [rouge_l_f1_char(a, b) for a, b in normalized]
    exact = sum(1 for a, b in normalized if a == b)

    signature_metric = sacrebleu.BLEU(tokenize="13a")
    signature_metric.num_refs = 1
    payload: dict[str, Any] = {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "n": len(pairs),
        "bleu4_13a": round(bleu.score, 4),
        "bleu4_13a_precisions": [round(value, 2) for value in bleu.precisions],
        "bleu4_13a_bp": round(bleu.bp, 4),
        "chrf": round(chrf, 4),
        "mean_char_edit_similarity": round(statistics.fmean(edits), 4) if edits else None,
        "char_rogue_l_f1": round(statistics.fmean(rouges), 4) if rouges else None,
        "exact_match_count": exact,
        "sacrebleu_signature": str(signature_metric.get_signature()),
    }
    if alternate_references is not None and len(alternate_references) == len(pairs):
        alt_char = [character_separated(ref) for ref in alternate_references]
        bleu_alt = sacrebleu.corpus_bleu(hyps_char, [alt_char], tokenize="13a")
        payload["bleu4_13a_alternate_reference"] = round(bleu_alt.score, 4)
        payload["bleu4_13a_alternate_reference_note"] = (
            "diagnostic only; alternate caliber (e.g. official annotation punctuation) "
            "cannot be compared to the canonical reference"
        )
    return payload


def _edit_ops(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    """Levenshtein alignment counting substitutions/deletions/insertions."""

    n, m = len(reference), len(hypothesis)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + cost,
            )
    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def gloss_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [token for token in unicodedata.normalize("NFKC", str(text)).split() if token]


def wer_breakdown(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_tokens = gloss_tokens(reference)
    hyp_tokens = gloss_tokens(hypothesis)
    ops = _edit_ops(ref_tokens, hyp_tokens)
    edits = ops["substitutions"] + ops["deletions"] + ops["insertions"]
    return {
        "reference_tokens": len(ref_tokens),
        "predicted_tokens": len(hyp_tokens),
        **ops,
        "edits": edits,
        "wer": (edits / len(ref_tokens)) if ref_tokens else None,
    }


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    total_ref = total_s = total_d = total_i = 0
    for reference, hypothesis in pairs:
        row = wer_breakdown(reference, hypothesis)
        total_ref += row["reference_tokens"]
        total_s += row["substitutions"]
        total_d += row["deletions"]
        total_i += row["insertions"]
    edits = total_s + total_d + total_i
    return {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "n": len(pairs),
        "reference_tokens": total_ref,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "corpus_wer": round(edits / total_ref, 4) if total_ref else None,
    }


# --------------------------------------------------------------------------
# bootstrap confidence intervals (grouped by signer; PLAN/02 §6 asks for them)
# --------------------------------------------------------------------------
def bootstrap_by_group(
    records: Sequence[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    *,
    group_key: str = "signer_id",
    iterations: int = 1000,
    seed: int = 20261026,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster bootstrap: resample signers with replacement, keep whole clusters."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record[group_key]), []).append(record)
    keys = sorted(groups)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample: list[dict[str, Any]] = []
        for _ in range(len(keys)):
            sample.extend(groups[rng.choice(keys)])
        value = statistic(sample)
        if value is not None and not math.isnan(value):
            values.append(float(value))
    point = statistic(list(records))
    if not values:
        return {"point": point, "lower": None, "upper": None, "iterations": iterations, "groups": len(keys)}
    values.sort()
    alpha = (1 - confidence) / 2
    lower = values[int(alpha * (len(values) - 1))]
    upper = values[int((1 - alpha) * (len(values) - 1))]
    return {
        "point": point,
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "iterations": iterations,
        "groups": len(keys),
        "confidence": confidence,
        "method": "cluster_bootstrap_by_signer",
    }
