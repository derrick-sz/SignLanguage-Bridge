"""T-A2-2 tests: async job state machine and runner (replay-driven, CPU-only).

Scope (task card T-A2-2 / task-16):
* 7-state machine ``QUEUED -> PREPROCESSING -> RUNNING -> SUCCEEDED | FAILED |
  TIMED_OUT | CANCELLED`` with illegal transitions rejected;
* ``JobStore``: in-memory registry, idempotent job_id, optional JSONL
  persistence of every state transition;
* ``JobRunner``: drives a replay_jsonl_v1 adapter through the existing
  ``ModelManager.infer_file`` (single-capacity lease, ADR-P1-003 serial),
  records evidence via ``PredictionRecorder`` (T-A0-1), maps failures onto
  ``ContractErrorCodes`` (T-A2-1), retries once on retryable failures
  (pitfalls sec. 7: host-memory OOM masquerades), then FAILED;
* timeout -> TIMED_OUT with INFERENCE_TIMEOUT; cancel -> CANCELLED without
  fabricated results (ADR-P1-004: never fabricate output).

Out of scope (declared): HTTP endpoint wiring (T-A2-3), parallel/adaptive
modes (G2 FAILED; single_model semantics only).

State-transition matrix asserted below (Q=QUEUED, P=PREPROCESSING, R=RUNNING,
S=SUCCEEDED, F=FAILED, T=TIMED_OUT, C=CANCELLED)::

      from\to   Q   P   R   S   F   T   C
      Q         -   ok  no  no  no  no  ok
      P         no  -   ok  no  ok  ok  ok
      R         no  no  -   ok  ok  ok  ok
      S/F/T/C   no  no  no  -   -   -   -   (terminal, immutable)
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from slbridge.api_contract_v1 import ContractErrorCodes
from slbridge.job_runner import (JobRecord, JobRunner, JobState, JobStore,
                                  TRANSITIONS)
from slbridge.model_adapters import load_model_adapter
from slbridge.runtime import ModelManager
from slbridge.runtime_contracts import InferenceRequest

REPO = Path(__file__).resolve().parents[1]
REAL_TEXT_SOURCE = REPO / "experiments" / "p1" / "e02" / "candidate_outputs.jsonl"


def text_row(sample_id: str, text: str) -> dict:
    return {
        "candidate_id": "A", "modality": "RGB_video", "model_id": "unisign_rgb",
        "model_revision": "unisign-video-1086-paperrefs",
        "output": {"gloss_tokens": None, "kind": "TEXT", "spoken_language": "zh", "text": text},
        "preprocess_version": "imported_historical_snapshot",
        "run_id": "e02-offline-1086-20261026T134508Z", "sample_id": sample_id,
        "sign_system": "CSL", "status": "OK", "task": "CSLT",
        "window": {"end_frame_exclusive": None, "end_pts_ms": None,
                   "needs_future_frames": None, "start_frame": 0, "start_pts_ms": None},
    }


def make_bundle(tmp_path: Path, rows: list[dict]) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = tmp_path / "source.jsonl"
    with source.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (bundle / "manifest.json").write_text(json.dumps({
        "backend": "replay_jsonl_v1", "source_kind": "text_candidates",
        "candidate_id": "A", "data_path": str(source)}), encoding="utf-8")
    return bundle


def make_manager(bundle: Path) -> ModelManager:
    adapter = load_model_adapter(bundle)
    return ModelManager(initial_recognizer=adapter)


def query_file(tmp_path: Path, sample_id: str) -> Path:
    path = tmp_path / f"q-{sample_id}.json"
    path.write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
    return path


def adapter_of(manager: ModelManager):
    return manager.recognizer


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def test_transition_matrix_matches_spec():
    assert TRANSITIONS == {
        JobState.QUEUED: {JobState.PREPROCESSING, JobState.CANCELLED},
        JobState.PREPROCESSING: {JobState.RUNNING, JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED},
        JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED},
        JobState.SUCCEEDED: set(), JobState.FAILED: set(),
        JobState.TIMED_OUT: set(), JobState.CANCELLED: set(),
    }


def test_legal_and_illegal_transitions():
    record = JobRecord(job_id="job-1", state=JobState.QUEUED, sign_system="CSL",
                       sample_id="S000020_P0004_T00")
    record.transition(JobState.PREPROCESSING)
    record.transition(JobState.RUNNING)
    record.transition(JobState.SUCCEEDED)
    # Terminal states are immutable.
    for target in (JobState.QUEUED, JobState.RUNNING, JobState.FAILED, JobState.CANCELLED):
        with pytest.raises(ValueError, match="transition"):
            record.transition(target)
    fresh = JobRecord(job_id="job-2", state=JobState.QUEUED, sign_system="CSL", sample_id="x")
    with pytest.raises(ValueError, match="transition"):
        fresh.transition(JobState.SUCCEEDED)   # QUEUED -> SUCCEEDED is illegal
    with pytest.raises(ValueError, match="transition"):
        fresh.transition(JobState.RUNNING)     # QUEUED -> RUNNING skips PREPROCESSING
    fresh.transition(JobState.CANCELLED)       # QUEUED -> CANCELLED is legal


def test_job_record_error_payload_shape():
    record = JobRecord(job_id="job-1", state=JobState.FAILED, sign_system="CSL",
                       sample_id="x", error_code=ContractErrorCodes.MODEL_UNAVAILABLE,
                       error_message="no route")
    payload = record.to_dict()
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "MODEL_UNAVAILABLE"


# ---------------------------------------------------------------------------
# JobStore: idempotency + optional JSONL persistence
# ---------------------------------------------------------------------------

def test_job_store_idempotent_job_id(tmp_path: Path):
    store = JobStore(persistence_path=tmp_path / "jobs.jsonl")
    record = store.create(job_id="job-1", sign_system="CSL", sample_id="S000020_P0004_T00")
    again = store.create(job_id="job-1", sign_system="CSL", sample_id="S000020_P0004_T00")
    assert again is record                       # idempotent: same object back
    other = store.create(job_id="job-2", sign_system="CSL", sample_id="S000040_P0008_T00")
    assert other is not record
    with pytest.raises(ValueError):
        store.create(job_id="job-1", sign_system="LIBRAS", sample_id="other")  # different payload
    assert store.get("job-1") is record
    assert store.get("missing") is None


def test_job_store_persists_transitions(tmp_path: Path):
    path = tmp_path / "jobs.jsonl"
    store = JobStore(persistence_path=path)
    record = store.create(job_id="job-9", sign_system="CSL", sample_id="s")
    # The state machine stays pure; the store journals each observed state.
    for state in (JobState.PREPROCESSING, JobState.RUNNING, JobState.SUCCEEDED):
        record.transition(state)
        store.journal(record)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [entry["status"] for entry in lines] == ["QUEUED", "PREPROCESSING", "RUNNING", "SUCCEEDED"]
    assert lines[-1]["job_id"] == "job-9"
    # Reload replays the latest state per job.
    restored = JobStore(persistence_path=None).restore_from(path)
    assert restored.get("job-9").state == JobState.SUCCEEDED


def test_job_store_capacity_bounded(tmp_path: Path):
    store = JobStore(max_jobs=2)
    store.create(job_id="j1", sign_system="CSL", sample_id="a")
    store.create(job_id="j2", sign_system="CSL", sample_id="b")
    with pytest.raises(RuntimeError, match="capacity"):
        store.create(job_id="j3", sign_system="CSL", sample_id="c")


# ---------------------------------------------------------------------------
# JobRunner end-to-end (replay-driven, CPU-only)
# ---------------------------------------------------------------------------

def test_runner_success_writes_evidence_and_manifest(tmp_path: Path):
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。")])
    manager = make_manager(bundle)
    store = JobStore(persistence_path=tmp_path / "jobs.jsonl")
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    result = runner.run(job_id="job-1", sample_id="S000020_P0004_T00",
                        query_path=query_file(tmp_path, "S000020_P0004_T00"))
    assert result.state == JobState.SUCCEEDED
    assert result.text == "她今年50岁。"          # verbatim through the whole chain
    record = store.get("job-1")
    assert record.state == JobState.SUCCEEDED
    # Prediction log got exactly one GT-free row + a manifest.
    row = json.loads((tmp_path / "run-1" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["request_id"] == "job-1"
    assert row["text"] == "她今年50岁。"
    assert row["candidate"]["output"]["text"] == "她今年50岁。"
    manifest = json.loads((tmp_path / "run-1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    assert manifest["models"][0]["model_id"] == "unisign_rgb"
    manager.close()


def test_runner_unknown_sample_fails_with_model_unavailable(tmp_path: Path):
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。")])
    manager = make_manager(bundle)
    store = JobStore()
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    result = runner.run(job_id="job-2", sample_id="S999999_P9999_T00",
                        query_path=query_file(tmp_path, "S999999_P9999_T00"))
    assert result.state == JobState.FAILED
    assert result.record.error_code == ContractErrorCodes.INVALID_INPUT
    # A failed job must not fabricate output.
    assert result.record.text is None
    manager.close()


def test_runner_timeout_maps_to_timed_out(tmp_path: Path):
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。")])
    manager = make_manager(bundle)
    store = JobStore()

    class Deadline:
        def __init__(self):
            self.calls = 0
        def __call__(self):
            self.calls += 1
            return 0.0        # already expired

    expired = Deadline()
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL", timeout_seconds=None,
                       deadline_factory=expired)
    result = runner.run(job_id="job-3", sample_id="S000020_P0004_T00",
                        query_path=query_file(tmp_path, "S000020_P0004_T00"))
    assert result.state == JobState.TIMED_OUT
    assert result.record.error_code == ContractErrorCodes.INFERENCE_TIMEOUT
    assert result.record.text is None
    manager.close()


def test_runner_cancel_before_start(tmp_path: Path):
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。")])
    manager = make_manager(bundle)
    store = JobStore()
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    runner.cancel("job-4")                     # cancel an unknown id is a no-op
    result = runner.run(job_id="job-4", sample_id="S000020_P0004_T00",
                        query_path=query_file(tmp_path, "S000020_P0004_T00"),
                        cancelled_before_start=True)
    assert result.state == JobState.CANCELLED
    assert result.record.text is None          # never fabricate output
    manager.close()


def test_runner_retries_once_then_fails(tmp_path: Path):
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。")])
    manager = make_manager(bundle)
    store = JobStore()
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    calls = {"n": 0}

    def flaky_infer_file(path, request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("numpy ..._ArrayMemoryError")   # pitfalls sec. 7 fake failure
        return manager_infer(path, request)

    manager_infer = manager.infer_file
    manager.infer_file = flaky_infer_file                     # monkeypatch the lease path
    try:
        result = runner.run(job_id="job-5", sample_id="S000020_P0004_T00",
                            query_path=query_file(tmp_path, "S000020_P0004_T00"))
        assert result.state == JobState.SUCCEEDED             # first attempt retried, second ok
        assert calls["n"] == 2
    finally:
        manager.infer_file = manager_infer
    manager.close()


def test_runner_no_infinite_retry(tmp_path: Path):
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。")])
    manager = make_manager(bundle)
    store = JobStore()
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    calls = {"n": 0}

    def always_fails(path, request):
        calls["n"] += 1
        raise MemoryError("host memory exhausted")

    original = manager.infer_file
    manager.infer_file = always_fails
    try:
        result = runner.run(job_id="job-6", sample_id="S000020_P0004_T00",
                            query_path=query_file(tmp_path, "S000020_P0004_T00"))
        assert result.state == JobState.FAILED
        assert calls["n"] == 2                   # exactly one retry, never more
        assert result.record.error_code == ContractErrorCodes.INTERNAL_ERROR
    finally:
        manager.infer_file = original
    manager.close()


def test_runner_single_capacity_serial(tmp_path: Path):
    """Two jobs on one manager run strictly one-at-a-time (ADR-P1-003)."""
    bundle = make_bundle(tmp_path, [text_row("S000020_P0004_T00", "她今年50岁。"),
                                    text_row("S000040_P0008_T00", "今天星期五。")])
    manager = make_manager(bundle)
    store = JobStore()
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    active = {"now": 0}
    lock = threading.Lock()

    original = manager.infer_file

    def tracking_infer(path, request):
        with lock:
            active["now"] += 1
            peak = active["now"]
            active["now"] -= 1
        assert peak == 1                          # never two at once
        return original(path, request)

    manager.infer_file = tracking_infer
    try:
        threads = []
        for index, sample in enumerate(("S000020_P0004_T00", "S000040_P0008_T00")):
            thread = threading.Thread(
                target=runner.run, kwargs=dict(job_id=f"job-{index}", sample_id=sample,
                                                query_path=query_file(tmp_path, sample)))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert store.get("job-0").state == JobState.SUCCEEDED
        assert store.get("job-1").state == JobState.SUCCEEDED
    finally:
        manager.infer_file = original
    manager.close()


@pytest.mark.skipif(not REAL_TEXT_SOURCE.exists(), reason="frozen P-1 source not present")
def test_runner_real_source_end_to_end(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "backend": "replay_jsonl_v1", "source_kind": "text_candidates",
        "candidate_id": "A", "data_path": str(REAL_TEXT_SOURCE)}), encoding="utf-8")
    manager = make_manager(bundle)
    store = JobStore(persistence_path=tmp_path / "jobs.jsonl")
    runner = JobRunner(manager, store, run_directory=tmp_path / "run-1", run_id="run-1",
                       model_id="unisign_rgb",
                       model_version=adapter_of(manager).capabilities.model_version,
                       preprocess_version="imported_historical_snapshot",
                       task="CSLT", sign_system="CSL")
    result = runner.run(job_id="job-real", sample_id="S000020_P0004_T00",
                        query_path=query_file(tmp_path, "S000020_P0004_T00"))
    assert result.state == JobState.SUCCEEDED
    # Verbatim against the frozen source row.
    expected = None
    with REAL_TEXT_SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("candidate_id") == "A" and row.get("sample_id") == "S000020_P0004_T00":
                expected = row["output"]["text"]
                break
    assert result.text == expected
    manifest = json.loads((tmp_path / "run-1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    manager.close()

