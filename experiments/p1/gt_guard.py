"""Ground-truth isolation guard.

PLAN.md / docs/02 数据与指标规范.md require that model, fusion and LLM payloads
must never contain the reference sentence, the reference gloss, or any score
derived from them.  This module is the single allow/deny definition used by the
prediction writers and by the tests.

The guard is deliberately blunt: any dictionary key that is known to be
GT-derived is rejected anywhere in a payload, including nested objects.  A
payload that needs one of these keys must be written as an *evaluation record*
under ``experiments/p1/runs/<run_id>/eval_*.jsonl`` by the offline evaluator,
never as an inference input/output.
"""

from __future__ import annotations

from typing import Any, Iterable

# Keys observed in the historical Excel / prediction artifacts that carry or
# imply ground truth.  Matching is case-insensitive and applies to every level.
FORBIDDEN_GT_KEYS: frozenset[str] = frozenset(
    {
        "gt",
        "gt_sentence",
        "gt_sentence_official",
        "gt_gloss",
        "gt_gloss_tokens",
        "ground_truth",
        "reference",
        "references",
        "reference_tokens",
        "ref_sentence",
        "ref_sentence_upstream",
        "ref_upstream",
        "slrt_wer",
        "wer",
        "unisign_edit",
        "unisign_video_edit",
        "unisign_pose_edit",
        "edit_similarity",
        "char_edit_similarity",
        "mean_char_edit_similarity",
        "rouge_l_f1_char",
        "exact_match",
        "diagnostic_class",
        "cohort_class",
    }
)


class GroundTruthLeakError(ValueError):
    """Raised when a payload that must stay GT-free contains a GT-derived key."""


def _walk(node: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else str(key)
            yield key_path, str(key)
            yield from _walk(value, key_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def find_gt_keys(payload: Any) -> list[str]:
    """Return dotted paths of every forbidden key found in ``payload``."""

    found: list[str] = []
    for path, key in _walk(payload, ""):
        if key.lower() in FORBIDDEN_GT_KEYS:
            found.append(path)
    return found


def assert_gt_clean(payload: Any, *, context: str = "payload") -> None:
    """Raise :class:`GroundTruthLeakError` if ``payload`` carries GT keys."""

    leaks = find_gt_keys(payload)
    if leaks:
        raise GroundTruthLeakError(
            f"{context} carries ground-truth derived keys: {', '.join(sorted(leaks))}"
        )


def sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without forbidden keys (only for explicitly allowed writer paths).

    Prefer :func:`assert_gt_clean`; this helper exists for the offline evaluator
    that intentionally drops GT before exporting a prediction-only view.
    """

    if not isinstance(payload, dict):
        raise TypeError("sanitize expects a mapping")
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).lower() in FORBIDDEN_GT_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = sanitize(value)
        else:
            cleaned[key] = value
    assert_gt_clean(cleaned)
    return cleaned
