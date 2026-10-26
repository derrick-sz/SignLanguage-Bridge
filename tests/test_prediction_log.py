"""T-A0-1 tests for the product-side prediction recorder.

Scope (task card T-A0-1): ``slbridge.prediction_log.PredictionRecorder`` writes

* one JSONL row per recorded evidence candidate (GT-guarded on append),
* a per-run ``manifest.json`` carrying git state (aligned with
  ``experiments/p1/common.py:git_state`` shape) plus model/preprocess fields.

Acceptance also requires the JSONL to be parseable by
``experiments.p1.common.iter_jsonl`` and to carry the frozen row fields
(``request_id/job_id/model_id/model_version/preprocessing_version/sign_system/
task_type/text|gloss_tokens/timing_ms/error``).  Those slbridge-flavoured names
live beside the P-1 candidate payload as a top-level ``slbridge`` block; the
candidate block stays byte-compatible with the frozen P-1 schema so the same
file can feed ``experiments/p1/eval`` (canonical ``p1-v1`` caliber) unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p1.common import git_state, iter_jsonl
from slbridge.prediction_log import PredictionRecorder, build_manifest


def text_row(**overrides):
    row = {
        "request_id": "req-0001",
        "job_id": None,
        "model_id": "unisign-pose-slt",
        "model_version": "rev-abc123",
        "preprocessing_version": "pose-rtmpose-133-v1",
        "sign_system": "CSL",
        "task_type": "text",
        "text": "她今年50岁。",
        "gloss_tokens": None,
        "timing_ms": {"pose_ms": 12.0, "inference_ms": 36.0, "total_ms": 60.0},
        "error": None,
    }
    row.update(overrides)
    return row


def gloss_row(**overrides):
    row = text_row(request_id="req-0002", model_id="slrt-cslr", model_version="rev-def456",
                   preprocessing_version="slrt-rgb-hrnet133-v1", task_type="gloss_sequence",
                   text=None, gloss_tokens=["我", "赚", "钱"],
                   timing_ms={"preprocess_ms": 5600.0, "inference_ms": 3690.0, "total_ms": 9500.0})
    row.update(overrides)
    return row


def error_row(**overrides):
    row = text_row(request_id="req-0003", task_type="text", text=None,
                   timing_ms={"total_ms": 40.0}, error={"code": "MODEL_UNAVAILABLE", "message": "no route"})
    row.update(overrides)
    return row


def assert_manifest_shape(manifest: dict, run_id: str) -> None:
    assert manifest["run_id"] == run_id
    code = manifest["code"]
    assert code["repository"] == "SignLanguage-Bridge"
    for key in ("git_sha", "branch", "dirty_tree", "dirty_path_count"):
        assert key in code
    # Must mirror experiments/p1/common.py:git_state for the same repo.
    live = git_state()
    assert code["git_sha"] == live["git_sha"]
    assert code["branch"] == live["branch"]
    assert code["dirty_tree"] == live["dirty_tree"]
    for key in ("model_id", "model_version", "preprocess_version", "task", "sign_system"):
        assert key in manifest["models"][0]
    assert manifest["metric_definition_version"] == "p1-v1"


def test_recorder_appends_rows_and_manifest(tmp_path: Path):
    recorder = PredictionRecorder(tmp_path / "run-1", run_id="run-1",
                                  model_id="unisign-pose-slt", model_version="rev-abc123",
                                  preprocess_version="pose-rtmpose-133-v1", task="CSLT",
                                  sign_system="CSL")
    assert recorder.append(text_row()) is True
    assert recorder.count == 1

    rows = list(iter_jsonl(recorder.predictions_path))
    assert len(rows) == 1
    for row in rows:
        for key in ("request_id", "job_id", "model_id", "model_version", "preprocessing_version",
                    "sign_system", "task_type", "text", "gloss_tokens", "timing_ms", "error",
                    "candidate", "schema_version", "recorded_at"):
            assert key in row
        # candidate block must validate as the frozen P-1 EvidenceCandidate
        # (additionalProperties:false => no schema_version key inside).
        assert "schema_version" not in row["candidate"]
        assert row["candidate"]["sign_system"] == "CSL"
    assert rows[0]["recorder_row_version"] == 1
    assert rows[0]["text"] == "她今年50岁。"

    # A second run records the SLRT gloss rows under its own registration.
    slrt = PredictionRecorder(tmp_path / "run-2", run_id="run-2",
                              model_id="slrt-cslr", model_version="rev-def456",
                              preprocess_version="slrt-rgb-hrnet133-v1", task="CSLR",
                              sign_system="CSL")
    assert slrt.append(gloss_row()) is True
    gloss_rows = list(iter_jsonl(slrt.predictions_path))
    assert len(gloss_rows) == 1
    assert gloss_rows[0]["gloss_tokens"] == ["我", "赚", "钱"]
    assert gloss_rows[0]["task_type"] == "gloss_sequence"

    manifest = recorder.write_manifest()
    assert (tmp_path / "run-1" / "manifest.json").is_file()
    assert_manifest_shape(manifest, "run-1")
    on_disk = json.loads((tmp_path / "run-1" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["row_count"] == 1


def test_recorder_rejects_gt_fields(tmp_path: Path):
    recorder = PredictionRecorder(tmp_path / "run-1", run_id="run-1",
                                  model_id="m", model_version="v", preprocess_version="p",
                                  task="CSLT", sign_system="CSL")
    with pytest.raises(Exception) as excinfo:
        recorder.append(text_row(reference="她 今年 50 岁 。"))
    assert "ground-truth" in str(excinfo.value).lower()
    with pytest.raises(Exception):
        recorder.append(gloss_row(wer=0.31))
    with pytest.raises(Exception):
        recorder.append(error_row(gt_sentence="参考"))
    # A rejected row must not be written or counted.
    assert recorder.count == 0
    assert not recorder.predictions_path.exists() or list(iter_jsonl(recorder.predictions_path)) == []


def test_recorder_requires_row_fields(tmp_path: Path):
    recorder = PredictionRecorder(tmp_path / "run-1", run_id="run-1",
                                  model_id="m", model_version="v", preprocess_version="p",
                                  task="CSLT", sign_system="CSL")
    with pytest.raises(ValueError):
        recorder.append({"request_id": "req-1"})  # missing the frozen row fields
    with pytest.raises(ValueError):
        recorder.append(text_row(timing_ms={"total_ms": -5.0}))  # negative timings are malformed


def test_recorder_row_matches_model_registration(tmp_path: Path):
    recorder = PredictionRecorder(tmp_path / "run-1", run_id="run-1",
                                  model_id="unisign-pose-slt", model_version="rev-abc123",
                                  preprocess_version="pose-rtmpose-133-v1", task="CSLT",
                                  sign_system="CSL")
    with pytest.raises(ValueError):
        recorder.append(gloss_row())  # slrt row into a unisign-registered run
    assert recorder.count == 0


def test_error_row_still_records_with_empty_candidate(tmp_path: Path):
    recorder = PredictionRecorder(tmp_path / "run-1", run_id="run-1",
                                  model_id="unisign-pose-slt", model_version="rev-abc123",
                                  preprocess_version="pose-rtmpose-133-v1", task="CSLT",
                                  sign_system="CSL")
    assert recorder.append(error_row()) is True
    rows = list(iter_jsonl(recorder.predictions_path))
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "MODEL_UNAVAILABLE"
    assert rows[0]["text"] is None
    assert rows[0]["candidate"]["status"] == "FAILED"
    assert rows[0]["candidate"]["output"]["kind"] == "EMPTY"

def test_build_manifest_records_schema_version_and_models(tmp_path: Path):
    manifest = build_manifest(run_id="run-x",
                              models=[{"model_id": "slrt-cslr", "model_version": "rev-def456",
                                       "preprocess_version": "slrt-rgb-hrnet133-v1",
                                       "task": "CSLR", "sign_system": "CSL"}],
                              row_count=0)
    assert manifest["models"][0]["model_id"] == "slrt-cslr"
    assert manifest["schema_version"] == 1
    assert manifest["metric_definition_version"] == "p1-v1"
    assert "created_at" in manifest
