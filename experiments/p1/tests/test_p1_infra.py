"""P-1 infrastructure tests: GT isolation, manifests, canonical metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p1 import common
from experiments.p1.eval import metrics as metric_lib
from experiments.p1.gt_guard import GroundTruthLeakError, assert_gt_clean, find_gt_keys, sanitize


# ---------------------------------------------------------------------------
# GT isolation
# ---------------------------------------------------------------------------
def test_gt_guard_rejects_top_level_reference():
    with pytest.raises(GroundTruthLeakError):
        assert_gt_clean({"sample_id": "S1", "reference": "他 今 年 四 岁 。"})


def test_gt_guard_rejects_nested_gt_gloss_and_derived_wer():
    payload = {
        "sample_id": "S1",
        "output": {"text": "她今年50岁。", "quality": {"raw_score_value": 0.4}},
        "diagnostics": [{"gt_gloss": "他 今天 年龄 4"}],
    }
    leaks = find_gt_keys(payload)
    assert "diagnostics[0].gt_gloss" in leaks
    with pytest.raises(GroundTruthLeakError):
        assert_gt_clean(payload)


def test_gt_guard_allows_prediction_only_payload():
    payload = {
        "run_id": "r1",
        "sample_id": "S1",
        "model_id": "unisign_rgb",
        "output": {"kind": "TEXT", "text": "她今年50岁。", "gloss_tokens": None, "spoken_language": "zh"},
        "quality": {"raw_score_value": None},
    }
    assert_gt_clean(payload)


def test_sanitize_drops_gt_keys_but_keeps_prediction():
    cleaned = sanitize({"sample_id": "S1", "prediction": "她今年50岁。", "reference": "x", "wer": 0.5})
    assert cleaned == {"sample_id": "S1", "prediction": "她今年50岁。"}


def test_jsonl_writer_blocks_gt(tmp_path: Path):
    path = tmp_path / "model_outputs.jsonl"
    with pytest.raises(GroundTruthLeakError):
        common.write_jsonl(path, [{"sample_id": "S1", "reference": "他 今 年 四 岁 。"}])
    assert not path.exists()


# ---------------------------------------------------------------------------
# manifests and references
# ---------------------------------------------------------------------------
def test_manifest_1086_is_unique_and_gt_free():
    rows = common.read_csv_rows(common.DATASETS / "manifest_1086.csv")
    assert len(rows) == 1086
    assert len({row["sample_id"] for row in rows}) == 1086
    forbidden = {"gt_sentence", "gt_gloss", "reference", "wer", "edit_similarity"}
    assert not forbidden & set(rows[0].keys())
    assert len({row["signer_id"] for row in rows}) == 10


def test_pilot_50_is_subset_of_manifest():
    manifest = {row["sample_id"] for row in common.read_csv_rows(common.DATASETS / "manifest_1086.csv")}
    pilot = common.read_csv_rows(common.DATASETS / "pilot_50.csv")
    assert len(pilot) == 50
    assert len({row["sample_id"] for row in pilot}) == 50
    assert {row["sample_id"] for row in pilot} <= manifest


def test_references_are_eval_only_and_complete():
    rows = common.read_csv_rows(common.REFERENCES / "gt_1086.csv")
    assert len(rows) == 1086
    assert all(row["ref_sentence_upstream"] for row in rows)
    assert all(row["gt_gloss"] for row in rows)
    assert "EVALUATION ONLY" in (common.REFERENCES / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# canonical metrics
# ---------------------------------------------------------------------------
def test_character_separated_matches_upstream_rewrite():
    assert metric_lib.character_separated("他今年四岁。") == "他 今 年 四 岁 。"
    assert metric_lib.character_separated("他 今 年 四 岁 。") == "他 今 年 四 岁 。"


def test_edit_similarity_known_value():
    # historical value in the 1086 comparison table
    reference = metric_lib.normalize_text("他 今 年 四 岁 。")
    hypothesis = metric_lib.normalize_text("她今年50岁。")
    assert metric_lib.edit_similarity(reference, hypothesis) == pytest.approx(0.5714285714285714)


def test_wer_breakdown_counts_sdi():
    row = metric_lib.wer_breakdown("a b c d", "a x c")
    assert row["substitutions"] == 1
    assert row["deletions"] == 1
    assert row["insertions"] == 0
    assert row["reference_tokens"] == 4
    assert row["wer"] == pytest.approx(0.5)


def test_corpus_bleu_reproduces_historical_snapshot():
    """The canonical evaluator must reproduce the frozen 24.8157 baseline.

    This is an integration check of the metric definition against the historical
    imported predictions; it does not re-run any model.
    """

    predictions = common.SHARED_ARTIFACTS / "unisign-video-1086-paperrefs" / "predictions.jsonl"
    references = common.REFERENCES / "gt_1086.csv"
    if not predictions.is_file():
        pytest.skip("historical 1086 prediction snapshot not present")
    from experiments.p1.eval.evaluate import evaluate

    payload, _ = evaluate(predictions, references, task="CSLT")
    assert payload["successful"] == 1086
    assert payload["metrics"]["bleu4_13a"] == pytest.approx(24.8157, abs=0.02)
    assert payload["metrics"]["chrf"] == pytest.approx(21.7974, abs=0.05)
