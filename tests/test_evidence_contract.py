"""T-A0-1 contract tests for the product-side evidence layer.

Scope (task card ``docs/experiments/P-1/P0A_IMPLEMENTATION_TASKS.md`` T-A0-1):

* ``slbridge.evidence.EvidenceCandidate`` must align field-for-field with the
  frozen P-1 schema ``experiments/p1/schemas/evidence_candidate.schema.json``
  (single source of truth chosen by the Lead; API-contract field names are
  mapped, not duplicated: ``model_version``->``model_revision``,
  ``preprocessing_version``->``preprocess_version``, ``task_type``->``task``).
* The GT guard must replicate ``experiments/p1/gt_guard.py`` exactly (the test
  asserts the key sets are identical so the two definitions cannot drift).
* ``from_slbridge_response`` implements the frozen field-mapping decisions
  (sample_id == request_id, seconds->milliseconds window conversion, ADR-P1-006
  calibrated reliability is always null).

Out of scope here (declared, not missing): the "late-arriving output" contract
counter-example from V3.0 section 13 belongs to the job layer (T-A2-2), where
ordering state actually lives; this module is stateless per response.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.p1.gt_guard import FORBIDDEN_GT_KEYS as P1_FORBIDDEN_GT_KEYS
from slbridge.contracts import (ClassificationPrediction, GlossSequencePrediction,
                                GlossToken, RecognitionResult, TextPrediction)
from slbridge.evidence import (EvidenceCandidate, EvidenceOutput, EvidenceQuality,
                               EvidenceWindow, FORBIDDEN_GT_KEYS,
                               GroundTruthLeakError, assert_gt_clean, find_gt_keys,
                               from_slbridge_response)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "experiments" / "p1" / "schemas" / "evidence_candidate.schema.json"


def fake_text_response(**overrides):
    """Uni-Sign shaped InferenceResponse stand-in (CPU-only, no model import)."""
    result = RecognitionResult(
        status="recognized", sample_id="req-0001", model_id="unisign-pose-slt",
        sign_language="csl", prediction=TextPrediction(text="她今年50岁。", language="zh", score=None),
        quality={"pose_ms": 12.0}, task_type="text", score_type="text_probability",
        time_range=(0.5, 3.25))
    value = dict(request_id="req-0001", model_generation=3, result=result,
                 runtime={"backend": "uni_sign_pose_slt"}, timing_ms={"pose_ms": 12.0, "inference_ms": 36.0},
                 session_id="offline-001", segment_id="seg-0001", source_epoch=0,
                 model_id="unisign-pose-slt", model_version="rev-abc123",
                 preprocessing_version="pose-rtmpose-133-v1")
    value.update(overrides)
    return SimpleNamespace(**value)


def fake_gloss_response(**overrides):
    tokens = (GlossToken(5, "我", 0.8, (0.1, 0.4)), GlossToken(12, "赚", 0.6, (0.4, 0.9)),
              GlossToken(3, "钱", 0.7, (0.9, 1.6)))
    result = RecognitionResult(
        status="recognized", sample_id="req-0002", model_id="slrt-cslr",
        sign_language="csl", prediction=GlossSequencePrediction(tokens, "slrt_window_majority"),
        quality={}, task_type="gloss_sequence", score_type="uncalibrated_mean_ensemble_probability",
        time_range=(2.0, 8.5))
    value = dict(request_id="req-0002", model_generation=1, result=result,
                 runtime={"backend": "slrt_online_csl"}, timing_ms={"preprocess_ms": 5600.0, "inference_ms": 3690.0},
                 session_id=None, segment_id=None, source_epoch=None,
                 model_id="slrt-cslr", model_version="rev-def456",
                 preprocessing_version="slrt-rgb-hrnet133-v1")
    value.update(overrides)
    return SimpleNamespace(**value)


def fake_rejected_response(**overrides):
    result = RecognitionResult(
        status="rejected", sample_id="req-0003", model_id="unisign-pose-slt",
        sign_language="csl", prediction=None, quality={}, reason="empty_text_output",
        task_type="text", score_type="text_probability", time_range=(0.0, 2.0))
    value = dict(request_id="req-0003", model_generation=3, result=result,
                 runtime={}, timing_ms={"total_ms": 40.0},
                 session_id=None, segment_id=None, source_epoch=None,
                 model_id="unisign-pose-slt", model_version="rev-abc123",
                 preprocessing_version="pose-rtmpose-133-v1")
    value.update(overrides)
    return SimpleNamespace(**value)


# ---------------------------------------------------------------------------
# Schema alignment (single source of truth: evidence_candidate.schema.json)
# ---------------------------------------------------------------------------

def test_candidate_fields_match_p1_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = set(schema["properties"])
    assert set(EvidenceCandidate.__dataclass_fields__) == properties
    assert set(schema["required"]) <= properties
    # Nested object field sets must match exactly as well.
    window_schema = schema["properties"]["window"]
    output_schema = schema["properties"]["output"]
    quality_schema = schema["properties"]["quality"]
    assert set(EvidenceWindow.__dataclass_fields__) == set(window_schema["properties"])
    assert set(EvidenceOutput.__dataclass_fields__) == set(output_schema["properties"])
    assert set(EvidenceQuality.__dataclass_fields__) == set(quality_schema["properties"])


def test_candidate_enums_match_p1_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    props = schema["properties"]
    assert set(props["task"]["enum"]) == {"CSLR", "CSLT", "ISLR"}
    assert set(props["sign_system"]["enum"]) == {"CSL", "Libras", "Unknown"}
    assert set(props["output"]["properties"]["kind"]["enum"]) == {"GLOSS", "TEXT", "GESTURE_ID", "EMPTY"}
    assert set(props["output"]["properties"]["spoken_language"]["enum"]) == {"zh", "pt", None}
    assert set(props["status"]["enum"]) == {"OK", "FAILED", "TIMEOUT", "SKIPPED", "BLOCKED"}
    # The dataclass must reject values outside those enums (probed below too).
    with pytest.raises(ValueError):
        EvidenceOutput(kind="TRANSCRIPT", gloss_tokens=None, text=None,
                       spoken_language=None, gesture_id=None)


def test_to_dict_covers_all_required_schema_fields():
    candidate = from_slbridge_response(fake_text_response(), run_id="run-1", modality="RGB_video")
    payload = candidate.to_dict()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for name in schema["required"]:
        assert name in payload, f"required schema field missing from to_dict: {name}"
    # additionalProperties:false semantics: no keys outside the schema.
    assert set(payload) <= set(schema["properties"])


# ---------------------------------------------------------------------------
# GT guard replication (must stay identical to experiments/p1/gt_guard.py)
# ---------------------------------------------------------------------------

def test_forbidden_keys_identical_to_p1_gt_guard():
    assert FORBIDDEN_GT_KEYS == P1_FORBIDDEN_GT_KEYS


@pytest.mark.parametrize("key", sorted(P1_FORBIDDEN_GT_KEYS))
def test_gt_guard_rejects_each_forbidden_key(key):
    with pytest.raises(GroundTruthLeakError):
        assert_gt_clean({"sample_id": "S1", key: "anything"})


def test_gt_guard_rejects_nested_and_case_insensitive():
    payload = {"output": {"text": "她今年50岁。"}, "diagnostics": [{"WER": 0.31, "notes": {"Reference": "x"}}]}
    leaks = find_gt_keys(payload)
    assert "diagnostics[0].WER" in leaks
    assert "diagnostics[0].notes.Reference" in leaks
    with pytest.raises(GroundTruthLeakError):
        assert_gt_clean(payload)


def test_gt_guard_allows_clean_candidate_payload():
    candidate = from_slbridge_response(fake_text_response(), run_id="run-1", modality="RGB_video")
    assert_gt_clean(candidate.to_dict(), context="evidence candidate")
    gloss = from_slbridge_response(fake_gloss_response(), run_id="run-1", modality="RGB_video")
    assert_gt_clean(gloss.to_dict(), context="evidence candidate")


# ---------------------------------------------------------------------------
# from_slbridge_response frozen mapping decisions
# ---------------------------------------------------------------------------

def test_text_response_mapping():
    candidate = from_slbridge_response(fake_text_response(), run_id="run-42", modality="RGB_video")
    assert candidate.run_id == "run-42"
    assert candidate.sample_id == "req-0001"          # == InferenceResponse.request_id
    assert candidate.attempt == 1
    assert candidate.window_id == "req-0001-w1"       # P0-A whole-file window derivation
    assert candidate.model_id == "unisign-pose-slt"
    assert candidate.task == "CSLT"                   # task_type text -> CSLT
    assert candidate.sign_system == "CSL"             # csl -> CSL
    assert candidate.modality == "RGB_video"
    assert candidate.model_revision == "rev-abc123"   # == model_version
    assert candidate.preprocess_version == "pose-rtmpose-133-v1"
    assert candidate.status == "OK"
    assert candidate.output.kind == "TEXT"
    assert candidate.output.text == "她今年50岁。"
    assert candidate.output.gloss_tokens is None
    assert candidate.output.spoken_language == "zh"
    # seconds -> integer milliseconds (unit conversion is part of the contract)
    assert candidate.window.start_pts_ms == 500
    assert candidate.window.end_pts_ms == 3250
    assert candidate.window.start_frame == 0
    assert candidate.window.end_frame_exclusive is None


def test_gloss_response_mapping():
    candidate = from_slbridge_response(fake_gloss_response(), run_id="run-42", modality="hrnet_npz")
    assert candidate.sample_id == "req-0002"
    assert candidate.task == "CSLR"
    assert candidate.output.kind == "GLOSS"
    assert candidate.output.gloss_tokens == ("我", "赚", "钱")
    assert candidate.output.text is None
    assert candidate.output.spoken_language is None
    assert candidate.window.start_pts_ms == 2000
    assert candidate.window.end_pts_ms == 8500
    assert candidate.quality.raw_score_type == "uncalibrated_mean_ensemble_probability"


def test_rejected_response_maps_failed_empty():
    candidate = from_slbridge_response(fake_rejected_response(), run_id="run-42", modality="RGB_video")
    assert candidate.status == "FAILED"
    assert candidate.output.kind == "EMPTY"
    assert candidate.output.text is None
    assert candidate.output.gloss_tokens is None
    assert candidate.output.spoken_language is None
    assert candidate.output.gesture_id is None


def test_calibrated_reliability_always_null():
    """ADR-P1-006: no validated calibrator ships in P0-A, so reliability is null."""
    for response in (fake_text_response(), fake_gloss_response(), fake_rejected_response(),
                     fake_text_response(result=RecognitionResult(
                         status="recognized", sample_id="req-0009", model_id="rf",
                         sign_language="csl",
                         prediction=ClassificationPrediction("wave", .9, {"wave": .9}),
                         quality={}, task_type="classification",
                         score_type="uncalibrated_class_probability",
                         time_range=(0.0, 1.0)),
                         model_id="rf", model_version="rev-rf", preprocessing_version="mediapipe225-v1")):
        candidate = from_slbridge_response(response, run_id="r", modality="m")
        assert candidate.quality.calibrated_reliability is None
        assert candidate.quality.confidence_model_revision is None


def test_classification_maps_islr_gesture():
    response = fake_text_response(
        result=RecognitionResult(
            status="recognized", sample_id="req-0009", model_id="rf-demo",
            sign_language="csl", prediction=ClassificationPrediction("wave", .9, {"wave": .9}, {"zh": "挥手"}),
            quality={}, task_type="classification",
            score_type="uncalibrated_class_probability", time_range=(0.0, 1.0)),
        request_id="req-0009", model_id="rf-demo", model_version="rev-rf",
        preprocessing_version="mediapipe225-v1")
    candidate = from_slbridge_response(response, run_id="r", modality="mediapipe225")
    assert candidate.task == "ISLR"
    assert candidate.output.kind == "GESTURE_ID"
    assert candidate.output.gesture_id == "wave"
    assert candidate.quality.raw_score_value == pytest.approx(.9)


def test_time_range_conversion_and_none():
    response = fake_text_response()
    response.result = RecognitionResult(
        status="recognized", sample_id="req-0001", model_id="unisign-pose-slt",
        sign_language="csl", prediction=TextPrediction("她今年50岁。", "zh", None),
        quality={}, task_type="text", score_type="text_probability", time_range=None)
    candidate = from_slbridge_response(response, run_id="r", modality="m")
    assert candidate.window.start_pts_ms is None
    assert candidate.window.end_pts_ms is None
    # Fractional seconds must round to integer ms deterministically.
    response.result = RecognitionResult(
        status="recognized", sample_id="req-0001", model_id="unisign-pose-slt",
        sign_language="csl", prediction=TextPrediction("她今年50岁。", "zh", None),
        quality={}, task_type="text", score_type="text_probability", time_range=(0.2567, 1.0005))
    candidate = from_slbridge_response(response, run_id="r", modality="m")
    assert candidate.window.start_pts_ms == 257
    assert candidate.window.end_pts_ms == 1000


def test_sign_language_normalization():
    def lang(value):
        response = fake_text_response()
        response.result = RecognitionResult(
            status="recognized", sample_id="req-0001", model_id="m", sign_language=value,
            prediction=TextPrediction("文本", "zh", None), quality={}, task_type="text",
            score_type="text_probability", time_range=(0.0, 1.0))
        return from_slbridge_response(response, run_id="r", modality="m").sign_system

    assert lang("csl") == "CSL"
    assert lang("CSL") == "CSL"
    assert lang("libras") == "Libras"
    assert lang(None) == "Unknown"
    with pytest.raises(ValueError):
        lang("bsl")


def test_missing_model_version_or_preprocess_version_rejected():
    with pytest.raises(ValueError):
        from_slbridge_response(fake_text_response(model_version=None), run_id="r", modality="m")
    with pytest.raises(ValueError):
        from_slbridge_response(fake_text_response(preprocessing_version=None), run_id="r", modality="m")


def test_unknown_result_status_rejected():
    with pytest.raises(ValueError):
        from_slbridge_response(fake_text_response(
            result=RecognitionResult("pending", "req-0001", "m", "csl",
                                     TextPrediction("文本", "zh", None), {})), run_id="r", modality="m")


# ---------------------------------------------------------------------------
# V3.0 section 13 contract counter-examples (null confidence / empty strings /
# empty gloss / malformed JSON payloads). Late outputs: see module docstring.
# ---------------------------------------------------------------------------

def test_empty_gloss_rejected():
    response = fake_gloss_response()
    response.result = RecognitionResult(
        status="recognized", sample_id="req-0002", model_id="slrt-cslr", sign_language="csl",
        prediction=GlossSequencePrediction((), "slrt_window_majority"), quality={},
        task_type="gloss_sequence", score_type="uncalibrated_mean_ensemble_probability",
        time_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        from_slbridge_response(response, run_id="r", modality="m")


def test_from_dict_rejects_malformed_payloads():
    good = from_slbridge_response(fake_text_response(), run_id="run-1", modality="RGB_video").to_dict()
    for mutate in (
        lambda row: row.pop("model_id"),
        lambda row: row.update({"unexpected_key": 1}),
        lambda row: row.update({"task": "TRANSLATION"}),
        lambda row: row.update({"sign_system": "ASL"}),
        lambda row: row.update({"status": "ok_with_typo"}),
        lambda row: row.update({"sample_id": ""}),
        lambda row: row["output"].update({"kind": "TEXT", "text": "   "}),
        lambda row: row["output"].update({"kind": "TEXT", "spoken_language": "en"}),
        lambda row: row["quality"].update({"calibrated_reliability": 1.5}),
        lambda row: row.update({"attempt": 0}),
        lambda row: row.update({"window": {"start_frame": -1}}),
    ):
        row = json.loads(json.dumps(good))
        mutate(row)
        with pytest.raises((ValueError, KeyError)):
            EvidenceCandidate.from_dict(row)


def test_roundtrip_to_dict_from_dict():
    for response, modality in ((fake_text_response(), "RGB_video"),
                               (fake_gloss_response(), "hrnet_npz"),
                               (fake_rejected_response(), "RGB_video")):
        candidate = from_slbridge_response(response, run_id="run-1", modality=modality)
        assert EvidenceCandidate.from_dict(candidate.to_dict()) == candidate
