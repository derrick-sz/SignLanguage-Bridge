"""T-A2-3 tests: V1 job HTTP endpoints (POST /v1/videos/infer, GET /v1/jobs).

Runs in the pydantic-enabled environment (.venv-libras: FastAPI TestClient) --
mirrors ``tests/test_api.py``'s importorskip pattern so the .conda-env suite
skips cleanly instead of erroring.

Coverage (task card T-A2-3 / task-17):
* ``POST /v1/videos/infer`` multipart -> 202 + ``{"job_id", "status": "QUEUED"}``;
* polling ``GET /v1/jobs/{job_id}`` to SUCCEEDED with timing metrics and a
  verbatim final text (replay adapter drives the whole chain, CPU-only);
* unknown job -> 404; oversized upload -> 413; auth failure -> 401 (when a
  token is configured); malformed form -> 422;
* parallel_fusion / adaptive_cascade modes are legal enum values but NOT
  enabled in P0-A: they return a 422 with an explicit contract error code,
  never a silent fallback to single_model;
* idempotent re-submission with the same job_id returns the same job state;
* the existing v3 endpoints keep their behaviour (regression probe).
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from slbridge.api import create_app
from slbridge.job_runner import JobState


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


class ReplayManager:
    """A ModelManager stand-in whose infer_file replays verbatim text rows.

    Keeps the endpoint tests free of the adapter/bundle ceremony while
    exercising the real JobRunner wiring through app.state.
    """

    def __init__(self, rows: dict[str, str]):
        self.rows = rows
        self.calls: list[str] = []
        self.lock = threading.RLock()

    def describe(self):
        return {"model_id": "unisign_rgb", "generation": 1, "sign_language": "csl",
                "task_type": "text", "backend": "replay_jsonl_v1"}

    def status(self):
        return {"state": "ready", "generation": 1,
                "active": self.describe(), "previous": None, "error": None}

    def capabilities(self):
        return None

    def infer_file(self, path, request):
        with self.lock:                     # serial single-capacity semantics
            self.calls.append(request.request_id)
        query = json.loads(Path(path).read_text(encoding="utf-8"))
        sample_id = query["sample_id"]
        if sample_id not in self.rows:
            raise ValueError(f"unknown sample for candidate 'A': {sample_id}")
        text = self.rows[sample_id]
        from types import SimpleNamespace
        from slbridge.contracts import RecognitionResult, TextPrediction
        result = RecognitionResult(
            status="recognized", sample_id=request.request_id, model_id="unisign_rgb",
            sign_language="csl", prediction=TextPrediction(text=text, language="zh", score=None),
            quality={"replay": True}, task_type="text", score_type="replay_imported_text_no_score",
            time_range=None)
        return SimpleNamespace(request_id=request.request_id, model_generation=1, result=result,
                               runtime={"backend": "replay_jsonl_v1"},
                               timing_ms={"replay_ms": 1.0, "total_ms": 2.0},
                               session_id=None, segment_id=None, source_epoch=None,
                               model_id="unisign_rgb", model_version="unisign-video-1086-paperrefs",
                               preprocessing_version="imported_historical_snapshot")

    def close(self):
        pass


@pytest.fixture
def app_client(tmp_path: Path):
    manager = ReplayManager({"S000020_P0004_T00": "她今年50岁。",
                             "S000040_P0008_T00": "今天星期五。"})
    app = create_app({}, manager=manager, max_upload_bytes=1024 * 1024,
                     job_run_directory=tmp_path / "runs", job_run_id="api-run-1")
    app.state.replay_rows = manager.rows       # for test-side query files
    with TestClient(app) as client:
        yield client, manager, tmp_path


def submit(client, sample_id: str, job_id: str | None = None, mode: str = "single_model",
           sign_system: str = "CSL", output_zh: bool = True, output_pt_br: bool = True,
           extra_fields: dict | None = None):
    data = {"sign_system": sign_system, "mode": mode,
            "output_zh": str(output_zh).lower(), "output_pt_br": str(output_pt_br).lower()}
    if job_id is not None:
        data["job_id"] = job_id
    if sample_id is not None:
        data["sample_id"] = sample_id          # replay queries carry the target sample
    data.update(extra_fields or {})
    # The uploaded "video" is a replay query document naming the sample.
    document = json.dumps({"sample_id": sample_id} if sample_id else {}).encode("utf-8")
    return client.post("/v1/videos/infer", data=data,
                       files={"video": ("query.json", document, "application/json")})


# ---------------------------------------------------------------------------
# POST /v1/videos/infer
# ---------------------------------------------------------------------------

def test_submit_returns_202_with_queued_job(app_client):
    client, manager, tmp_path = app_client
    response = submit(client, "S000020_P0004_T00", job_id="job-1")
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "QUEUED"


def test_submit_mode_not_enabled_returns_422(app_client):
    client, _, _ = app_client
    for mode in ("parallel_fusion", "adaptive_cascade"):
        response = submit(client, "S000020_P0004_T00", job_id=f"job-{mode}", mode=mode)
        assert response.status_code == 422
        body = response.json()
        # Legacy snake_case codes on the wire; the contract-level mapping to
        # the 8-value enum is map_error_code (T-A2-1) -- asserted separately.
        assert body["error"]["code"] == "invalid_input"
        assert "not enabled" in body["error"]["message"]


def test_submit_bad_sign_system_rejected(app_client):
    client, _, _ = app_client
    response = submit(client, "S000020_P0004_T00", job_id="job-bad", sign_system="ASL")
    assert response.status_code == 422


def test_submit_missing_required_field(app_client):
    client, _, _ = app_client
    response = client.post("/v1/videos/infer",
                           data={"sign_system": "CSL", "mode": "single_model"},
                           files={"video": ("q.json", b"{}", "application/json")})
    assert response.status_code == 422


def test_submit_oversized_upload_413(app_client):
    client, _, _ = app_client
    # A legal file suffix whose bytes exceed the upload limit: _save_upload
    # must reject it with 413 (the suffix check itself is a different path).
    big = b"x" * (1024 * 1024 + 1)
    response = client.post("/v1/videos/infer",
                           data={"sign_system": "CSL", "mode": "single_model", "sample_id": "s"},
                           files={"video": ("big.mp4", big, "video/mp4")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_submit_requires_bearer_token_when_configured(tmp_path: Path):
    manager = ReplayManager({"S000020_P0004_T00": "她今年50岁。"})
    app = create_app({}, manager=manager, bearer_token="secret-token",
                     job_run_directory=tmp_path / "runs", job_run_id="t")
    with TestClient(app) as client:
        response = submit(client, "S000020_P0004_T00", job_id="job-auth")
        assert response.status_code == 401
        authorized = client.post(
            "/v1/videos/infer",
            data={"sign_system": "CSL", "mode": "single_model", "job_id": "job-auth",
                  "sample_id": "S000020_P0004_T00"},
            files={"video": ("q.json", b"{}", "application/json")},
            headers={"Authorization": "Bearer secret-token"})
        assert authorized.status_code == 202


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------

def wait_terminal(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll a job until it reaches a terminal state (async background execution)."""
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in {"SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED"}:
            return body
        if time.monotonic() > deadline:
            return body
        time.sleep(0.02)


def test_poll_to_succeeded_with_verbatim_text(app_client):
    client, manager, tmp_path = app_client
    assert submit(client, "S000020_P0004_T00", job_id="job-ok").status_code == 202
    body = wait_terminal(client, "job-ok")
    assert response_status(client, "job-ok") == 200
    assert body["job_id"] == "job-ok"
    assert body["status"] == "SUCCEEDED"
    assert body["final_event"]["event"] == "subtitle.final"
    assert body["final_event"]["text"]["zh-CN"] == "她今年50岁。"   # verbatim
    assert "metrics" in body and body["metrics"]["replay_ms"] >= 0
    # The prediction log recorded the job run.
    assert (tmp_path / "runs" / "predictions.jsonl").exists()


def response_status(client, job_id: str) -> int:
    return client.get(f"/v1/jobs/{job_id}").status_code


def test_unknown_job_404(app_client):
    client, _, _ = app_client
    response = client.get("/v1/jobs/nope")
    assert response.status_code == 404


def test_failed_job_has_error_code_no_fabricated_text(app_client):
    client, manager, _ = app_client
    assert submit(client, "S999999_P9999_T00", job_id="job-fail").status_code == 202
    body = wait_terminal(client, "job-fail")
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "INVALID_INPUT"
    assert body["final_event"] is None       # never fabricate output


def test_idempotent_resubmission(app_client):
    client, manager, _ = app_client
    assert submit(client, "S000020_P0004_T00", job_id="job-idem").status_code == 202
    first = wait_terminal(client, "job-idem")
    # Re-submitting the identical payload returns the same job, not an error.
    assert submit(client, "S000020_P0004_T00", job_id="job-idem").status_code == 202
    second = wait_terminal(client, "job-idem")
    assert second == first
    # A conflicting payload under the same job_id is rejected.
    conflict = submit(client, "S000040_P0008_T00", job_id="job-idem")
    assert conflict.status_code == 409


def test_existing_v3_health_unchanged(app_client):
    client, manager, _ = app_client
    response = client.get("/v3/health")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 3
    assert "status" in body or "capabilities" in body
