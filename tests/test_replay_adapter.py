"""T-A1-1 contract tests for the ReplayAdapter (``replay_jsonl_v1``).

Scope (task card T-A1-1 + task-14):
* register a plain ``ModelAdapter`` that replays GT-free historical predictions
  from ``experiments/p1/e02/candidate_outputs.jsonl`` (text candidates) and
  ``experiments/p1/e02/slrt_gloss_predictions.jsonl`` (SLRT gloss);
* replayed text/gloss must be **verbatim** against the source rows -- a replay
  adapter must never rewrite or "polish" predictions;
* the full protocol path works: ``ModelManager.infer_file`` negotiates,
  executes and ``validate_adapter_result``-validates the replayed result;
* illegal inputs fail loudly (unknown sample, missing fields, GT keys in the
  source, OK-status rows with empty text), and ``limit`` truncates loading;
* evidence conversion + prediction recording reuse T-A0-1 modules verbatim.

Discipline encoded here: the adapter must never read anything below
``experiments/p1/references/`` (GT lives only on the evaluation side); tests
that touch the real 3,258-row file use it strictly as read-only input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slbridge.contracts import GlossSequencePrediction, RecognitionResult, TextPrediction
from slbridge.evidence import GroundTruthLeakError, from_slbridge_response
from slbridge.model_adapters import load_model_adapter, model_adapter_ids
from slbridge.prediction_log import PredictionRecorder
from slbridge.runtime import ModelManager
from slbridge.runtime_contracts import InferenceRequest

REPO = Path(__file__).resolve().parents[1]
REAL_TEXT_SOURCE = REPO / "experiments" / "p1" / "e02" / "candidate_outputs.jsonl"
REAL_GLOSS_SOURCE = REPO / "experiments" / "p1" / "e02" / "slrt_gloss_predictions.jsonl"
PILOT_50 = REPO / "experiments" / "p1" / "datasets" / "pilot_50.csv"


def write_text_source(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def text_row(sample_id: str, text: str, **overrides) -> dict:
    row = {
        "candidate_id": "A",
        "modality": "RGB_video",
        "model_id": "unisign_rgb",
        "model_revision": "unisign-video-1086-paperrefs",
        "output": {"gloss_tokens": None, "kind": "TEXT", "spoken_language": "zh", "text": text},
        "preprocess_version": "imported_historical_snapshot",
        "run_id": "e02-offline-1086-20261026T134508Z",
        "sample_id": sample_id,
        "sign_system": "CSL",
        "status": "OK",
        "task": "CSLT",
        "window": {"end_frame_exclusive": None, "end_pts_ms": None,
                   "needs_future_frames": None, "start_frame": 0, "start_pts_ms": None},
    }
    row.update(overrides)
    return row


def write_gloss_source(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def gloss_row(sample_id: str, glosses: list[str], **overrides) -> dict:
    row = {"error": None, "gloss_sequence": glosses, "sample": sample_id, "status": "final"}
    row.update(overrides)
    return row


def write_manifest(directory: Path, **overrides) -> Path:
    manifest = {
        "backend": "replay_jsonl_v1",
        "source_kind": "text_candidates",
        "candidate_id": "A",
        "data_path": str(directory / "source.jsonl"),
    }
    manifest.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


@pytest.fixture
def text_bundle(tmp_path: Path) -> Path:
    rows = [
        text_row("S000020_P0004_T00", "她今年50岁。"),
        text_row("S000040_P0008_T00", "今天星期五。"),
        text_row("S000054_P0008_T00", "今天我想吃面条。"),
    ]
    write_text_source(tmp_path / "source.jsonl", rows)
    return write_manifest(tmp_path)


@pytest.fixture
def gloss_bundle(tmp_path: Path) -> Path:
    rows = [
        gloss_row("S000020_P0004_T00", ["他", "今天", "年龄", "4"]),
        gloss_row("S000040_P0008_T00", ["今天", "星期", "多少"]),
    ]
    write_gloss_source(tmp_path / "source.jsonl", rows)
    return write_manifest(tmp_path, source_kind="slrt_gloss", candidate_id=None,
                           model_id="slrt-cslr", data_path=str(tmp_path / "source.jsonl"))


def query_file(tmp_path: Path, sample_id: str) -> Path:
    path = tmp_path / "query.json"
    path.write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Registration and capabilities
# ---------------------------------------------------------------------------

def test_replay_backend_registered():
    assert "replay_jsonl_v1" in model_adapter_ids()


def test_text_capabilities(text_bundle: Path):
    adapter = load_model_adapter(text_bundle)
    caps = adapter.capabilities
    assert caps.model_id == "unisign_rgb"
    assert caps.task_type == "text"
    assert caps.sign_language == "csl"
    assert caps.devices == ("cpu",)
    assert caps.modes == ("clip",)
    assert caps.vocabulary == ()
    assert caps.score_type == "replay_imported_text_no_score"
    assert adapter.runtime_info["backend"] == "replay_jsonl_v1"
    assert adapter.runtime_info["replay_source"]["row_count"] == 3
    adapter.close()


def test_gloss_capabilities(gloss_bundle: Path):
    adapter = load_model_adapter(gloss_bundle)
    caps = adapter.capabilities
    assert caps.task_type == "gloss_sequence"
    assert caps.sign_language == "csl"
    assert caps.score_type == "replay_imported_gloss_no_score"
    # Vocabulary is built from the source so GlossToken ids validate.
    for gloss in ("他", "今天", "年龄", "4", "星期", "多少"):
        assert gloss in caps.vocabulary
    adapter.close()


# ---------------------------------------------------------------------------
# Verbatim replay (direct API and full protocol path)
# ---------------------------------------------------------------------------

def test_replay_text_verbatim(text_bundle: Path):
    adapter = load_model_adapter(text_bundle)
    result = adapter.replay("S000040_P0008_T00")
    assert isinstance(result, RecognitionResult)
    assert result.status == "recognized"
    assert result.task_type == "text"
    assert isinstance(result.prediction, TextPrediction)
    assert result.prediction.text == "今天星期五。"   # verbatim, never rewritten
    assert result.prediction.score is None
    assert result.sign_language == "csl"
    adapter.close()


def test_replay_gloss_verbatim(gloss_bundle: Path):
    adapter = load_model_adapter(gloss_bundle)
    result = adapter.replay("S000020_P0004_T00")
    assert result.status == "recognized"
    assert result.task_type == "gloss_sequence"
    prediction = result.prediction
    assert isinstance(prediction, GlossSequencePrediction)
    assert list(prediction.glosses) == ["他", "今天", "年龄", "4"]  # verbatim order
    assert all(token.score is None for token in prediction.tokens)
    adapter.close()


def test_full_protocol_path_via_model_manager(text_bundle: Path, tmp_path: Path):
    adapter = load_model_adapter(text_bundle)
    manager = ModelManager(initial_recognizer=adapter)
    request = InferenceRequest(
        request_id="S000020_P0004_T00",
        representation_id=adapter.capabilities.inputs[0].representation_id,
        sign_language="csl", expected_task_type="text",
        model_id=adapter.capabilities.model_id,
        preprocessing_version=adapter.capabilities.inputs[0].preprocessing_version)
    response = manager.infer_file(query_file(tmp_path, "S000020_P0004_T00"), request)
    assert response.result.prediction.text == "她今年50岁。"
    assert response.model_id == "unisign_rgb"
    assert response.model_version == adapter.capabilities.model_version
    assert response.preprocessing_version == "imported_historical_snapshot"
    assert set(response.timing_ms) >= {"replay_ms", "total_ms"}
    manager.close()


def test_evidence_and_prediction_log_reuse(text_bundle: Path, tmp_path: Path):
    adapter = load_model_adapter(text_bundle)
    manager = ModelManager(initial_recognizer=adapter)
    request = InferenceRequest(request_id="S000054_P0008_T00",
                               sign_language="csl", expected_task_type="text",
                               model_id=adapter.capabilities.model_id)
    response = manager.infer_file(query_file(tmp_path, "S000054_P0008_T00"), request)
    candidate = from_slbridge_response(response, run_id="replay-run-1",
                                       modality="RGB_video")
    assert candidate.output.text == "今天我想吃面条。"
    assert candidate.quality.calibrated_reliability is None

    recorder = PredictionRecorder(tmp_path / "replay-run-1", run_id="replay-run-1",
                                  model_id="unisign_rgb",
                                  model_version=adapter.capabilities.model_version,
                                  preprocess_version="imported_historical_snapshot",
                                  task="CSLT", sign_system="CSL")
    recorder.append_response(response)
    row = json.loads((tmp_path / "replay-run-1" / "predictions.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0])
    assert row["text"] == "今天我想吃面条。"     # recorded verbatim
    assert row["candidate"]["output"]["text"] == "今天我想吃面条。"
    manager.close()


# ---------------------------------------------------------------------------
# Illegal inputs fail loudly
# ---------------------------------------------------------------------------

def test_unknown_sample_rejected(text_bundle: Path):
    adapter = load_model_adapter(text_bundle)
    with pytest.raises(ValueError, match="unknown sample"):
        adapter.replay("S999999_P9999_T00")
    adapter.close()


def test_unknown_candidate_rejected(text_bundle: Path):
    write_manifest(text_bundle, candidate_id="A2")   # source only holds candidate A
    with pytest.raises(ValueError, match="candidate"):
        load_model_adapter(text_bundle)


def test_query_file_missing_sample_id(text_bundle: Path, tmp_path: Path):
    adapter = load_model_adapter(text_bundle)
    manager = ModelManager(initial_recognizer=adapter)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"candidate_id": "A"}), encoding="utf-8")
    request = InferenceRequest(request_id="whatever", sign_language="csl",
                               expected_task_type="text",
                               model_id=adapter.capabilities.model_id)
    with pytest.raises(ValueError):
        manager.infer_file(bad, request)
    manager.close()


def test_gt_keys_in_source_rejected(tmp_path: Path):
    rows = [text_row("S000020_P0004_T00", "她今年50岁。", reference="她 今年 50 岁 。")]
    write_text_source(tmp_path / "source.jsonl", rows)
    write_manifest(tmp_path)
    with pytest.raises(GroundTruthLeakError):
        load_model_adapter(tmp_path)


def test_ok_status_empty_text_rejected(tmp_path: Path):
    rows = [text_row("S000020_P0004_T00", "   ")]
    write_text_source(tmp_path / "source.jsonl", rows)
    write_manifest(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        load_model_adapter(tmp_path)


def test_limit_truncates_rows(tmp_path: Path):
    rows = [text_row("S000020_P0004_T00", "她今年50岁。"),
            text_row("S000040_P0008_T00", "今天星期五。"),
            text_row("S000054_P0008_T00", "今天我想吃面条。")]
    write_text_source(tmp_path / "source.jsonl", rows)
    write_manifest(tmp_path, limit=2)
    adapter = load_model_adapter(tmp_path)
    assert adapter.runtime_info["replay_source"]["row_count"] == 2
    adapter.replay("S000040_P0008_T00")
    with pytest.raises(ValueError, match="unknown sample"):
        adapter.replay("S000054_P0008_T00")
    adapter.close()


def test_non_ok_status_maps_to_rejected(tmp_path: Path):
    rows = [text_row("S000020_P0004_T00", "", status="FAILED")]
    write_text_source(tmp_path / "source.jsonl", rows)
    write_manifest(tmp_path)
    adapter = load_model_adapter(tmp_path)
    result = adapter.replay("S000020_P0004_T00")
    assert result.status == "rejected"
    assert result.reason == "source_status_FAILED"
    assert result.prediction is None
    adapter.close()


# ---------------------------------------------------------------------------
# Acceptance: 50 Pilot-A candidates verbatim against the real frozen source
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_TEXT_SOURCE.exists(), reason="frozen P-1 source not present")
def test_pilot_50_a_candidates_verbatim(tmp_path: Path):
    samples = [line.split(",")[0].strip()
               for line in PILOT_50.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    assert len(samples) == 50
    # Build a replay bundle bound to the real (read-only) frozen source.
    source_rows = {}
    with REAL_TEXT_SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("candidate_id") == "A":
                source_rows[row["sample_id"]] = row
    write_manifest(tmp_path, data_path=str(REAL_TEXT_SOURCE))
    adapter = load_model_adapter(tmp_path)
    matched = 0
    for sample_id in samples:
        row = source_rows.get(sample_id)
        if row is None:
            continue
        result = adapter.replay(sample_id)
        assert result.status == "recognized"
        assert result.prediction.text == row["output"]["text"]   # verbatim
        matched += 1
    assert matched == 50, f"expected 50 A-candidates for the pilot list, matched {matched}"
    adapter.close()


@pytest.mark.skipif(not REAL_GLOSS_SOURCE.exists(), reason="frozen P-1 source not present")
def test_real_gloss_source_first_rows_verbatim(tmp_path: Path):
    expected = []
    with REAL_GLOSS_SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected.append(json.loads(line))
            if len(expected) == 10:
                break
    write_manifest(tmp_path, source_kind="slrt_gloss", candidate_id=None,
                   model_id="slrt-cslr", data_path=str(REAL_GLOSS_SOURCE), limit=10)
    adapter = load_model_adapter(tmp_path)
    for row in expected:
        result = adapter.replay(row["sample"])
        assert result.status == "recognized"
        assert list(result.prediction.glosses) == row["gloss_sequence"]   # verbatim
    adapter.close()
