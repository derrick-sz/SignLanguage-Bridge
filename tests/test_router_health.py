"""T-A4 (A4-1 + A4-2 merged) tests: sign_system routing, health, error unification.

Coverage (task-19):
* Routing matrix: every API entry point treats sign_system uniformly --
  ``CSL`` routes normally, ``LIBRAS`` -> 503 MODEL_UNAVAILABLE (never a CSL
  weight, ADR-P1-007), ``UNKNOWN`` -> 422 (P0/P1 require explicit CSL);
  a negative proof asserts ModelRouter never holds a ``libras`` -> CSL-backed
  manager registration;
* Health endpoints per contract: ``/v1/health/live`` (process liveness,
  ``{"status": "alive"}``), ``/v1/health/ready`` (dependency readiness,
  ``ready|not_ready`` + ``workers``; not ready -> 503 + contract code);
* Error-code unification: on the V1 contract endpoints (POST /v1/videos/infer,
  GET /v1/jobs) the response ``error.code`` is ALWAYS the contract uppercase
  enum with a ``legacy_code`` transition field -- the previous mix of
  snake_case (POST) and contract codes (GET) is gone.  Existing v3 endpoints
  keep their exact legacy JSON shape (probed explicitly).
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from slbridge.api import create_app
from slbridge.runtime import ModelRouter

REPO = Path(__file__).resolve().parents[1]


class ReplayManager:
    def __init__(self, rows, *, loaded=True):
        self.rows = rows
        self.loaded = loaded

    def describe(self):
        if not self.loaded:
            return None
        return {"model_id": "unisign_rgb", "generation": 1, "sign_language": "csl",
                "task_type": "text", "backend": "replay_jsonl_v1"}

    def status(self):
        return {"state": "ready" if self.loaded else "empty", "generation": 1,
                "active": self.describe(), "previous": None, "error": None}

    def capabilities(self):
        return None

    def infer_file(self, path, request):
        from types import SimpleNamespace
        from slbridge.contracts import RecognitionResult, TextPrediction
        query = json.loads(Path(path).read_text(encoding="utf-8"))
        sample = query["sample_id"]
        if sample not in self.rows:
            raise ValueError(f"unknown sample for candidate 'A': {sample}")
        text = self.rows[sample]
        result = RecognitionResult(
            status="recognized", sample_id=request.request_id, model_id="unisign_rgb",
            sign_language="csl", prediction=TextPrediction(text=text, language="zh", score=None),
            quality={"replay": True}, task_type="text",
            score_type="replay_imported_text_no_score", time_range=None)
        return SimpleNamespace(request_id=request.request_id, model_generation=1, result=result,
                               runtime={}, timing_ms={"replay_ms": 1.0, "total_ms": 1.5},
                               session_id=None, segment_id=None, source_epoch=None,
                               model_id="unisign_rgb", model_version="rev",
                               preprocessing_version="pv")

    def close(self):
        pass


@pytest.fixture
def client():
    manager = ReplayManager({"S000020_P0004_T00": "她今年50岁。"})
    app = create_app({}, manager=manager, max_upload_bytes=1024 * 1024,
                     job_run_directory=None)
    with TestClient(app) as instance:
        yield instance


def submit_raw(client, sign_system, sample_id="S000020_P0004_T00"):
    document = json.dumps({"sample_id": sample_id}).encode("utf-8")
    return client.post("/v1/videos/infer",
                       data={"sign_system": sign_system, "mode": "single_model",
                             "job_id": f"job-{sign_system.lower()}", "sample_id": sample_id},
                       files={"video": ("query.json", document, "application/json")})


# ---------------------------------------------------------------------------
# 1. Routing matrix (all entry points, uniform behaviour)
# ---------------------------------------------------------------------------

def test_routing_matrix_videos_infer(client):
    csl = submit_raw(client, "CSL")
    assert csl.status_code == 202
    libras = submit_raw(client, "LIBRAS")
    assert libras.status_code == 503
    body = libras.json()
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"
    assert body["error"]["legacy_code"] == "model_unavailable"
    assert "Libras" in body["error"]["message"]
    unknown = submit_raw(client, "UNKNOWN")
    assert unknown.status_code == 422
    body = unknown.json()
    assert body["error"]["code"] == "INVALID_INPUT"
    assert body["error"]["legacy_code"] == "invalid_input"


def test_routing_never_maps_csl_weights_to_libras():
    """ADR-P1-007 negative proof: a libras route can never appear backed by
    the CSL replay manager (registration would require a matching
    sign_language capability; a libras ModelRouter route has no CSL manager)."""

    manager = ReplayManager({"S000020_P0004_T00": "她今年50岁。"})
    router = ModelRouter()
    with pytest.raises(ValueError):
        # The CSL manager's capability says sign_language='csl'; registering it
        # under 'libras' must be refused by ModelRouter itself.
        router.register("libras", manager)
    assert "libras" not in router.capabilities()
    router.close()


def test_existing_v3_endpoints_reject_unknown_language_uniformly(client):
    # v3 surface keeps its legacy shape (snake_case codes) and never silently
    # routes an unknown language to the CSL model: opening a caption session
    # for 'libras' is refused (language/model mismatch on this non-routed app).
    response = client.post("/v3/caption-sessions", json={"sign_language": "libras"})
    assert response.status_code in {404, 422, 503}    # refused, never routed to CSL
    body = response.json()
    assert body["error"]["code"] in {"unknown_language", "invalid_input", "model_unavailable"}
    assert "legacy_code" not in body["error"]          # v3 keeps the old JSON exactly


# ---------------------------------------------------------------------------
# 2. Health endpoints (contract semantics)
# ---------------------------------------------------------------------------

def test_health_live(client):
    response = client.get("/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_live_without_model(tmp_path: Path):
    empty = ReplayManager({}, loaded=False)
    app = create_app({}, manager=empty)
    with TestClient(app) as instance:
        # Liveness is process-only: it stays 200 even with no model loaded.
        assert instance.get("/v1/health/live").status_code == 200


def test_health_ready_with_model(client):
    response = client.get("/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert isinstance(body["workers"], dict) and body["workers"]
    assert all(isinstance(value, str) for value in body["workers"].values())


def test_health_ready_without_model(tmp_path: Path):
    empty = ReplayManager({}, loaded=False)
    app = create_app({}, manager=empty)
    with TestClient(app) as instance:
        response = instance.get("/v1/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"
        assert body["error"]["legacy_code"] == "model_unavailable"


# ---------------------------------------------------------------------------
# 3. Error-code unification on V1 endpoints (contract code + legacy_code)
# ---------------------------------------------------------------------------

def test_videos_infer_errors_use_contract_codes(client):
    cases = {
        "LIBRAS": (503, "MODEL_UNAVAILABLE", "model_unavailable"),
        "UNKNOWN": (422, "INVALID_INPUT", "invalid_input"),
    }
    for sign_system, (expected_status, contract_code, legacy_code) in cases.items():
        response = submit_raw(client, sign_system)
        assert response.status_code == expected_status
        error = response.json()["error"]
        assert error["code"] == contract_code
        assert error["legacy_code"] == legacy_code

    # Mode not enabled: contract code + legacy transition field.
    document = json.dumps({"sample_id": "S000020_P0004_T00"}).encode("utf-8")
    response = client.post("/v1/videos/infer",
                           data={"sign_system": "CSL", "mode": "parallel_fusion",
                                 "job_id": "job-mode", "sample_id": "S000020_P0004_T00"},
                           files={"video": ("query.json", document, "application/json")})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT"
    assert error["legacy_code"] == "invalid_input"

    # Missing required field (FastAPI validation): contract code too.
    response = client.post("/v1/videos/infer",
                           data={"sign_system": "CSL", "mode": "single_model"},
                           files={"video": ("q.json", document, "application/json")})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT"
    assert error.get("legacy_code") in {None, "invalid_request", "invalid_input"}


def test_jobs_endpoint_error_uses_contract_code(client):
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT"
    assert error["legacy_code"] == "unknown_job"

    # A failed job's embedded error keeps contract codes (as since T-A2-2).
    document = json.dumps({"sample_id": "S999999_P9999_T00"}).encode("utf-8")
    assert client.post("/v1/videos/infer",
                       data={"sign_system": "CSL", "mode": "single_model",
                             "job_id": "job-err", "sample_id": "S999999_P9999_T00"},
                       files={"video": ("query.json", document, "application/json")}).status_code == 202
    deadline = time.monotonic() + 10
    body = None
    while time.monotonic() < deadline:
        body = client.get("/v1/jobs/job-err").json()
        if body["status"] == "FAILED":
            break
        time.sleep(0.02)
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "INVALID_INPUT"


def test_v3_error_shape_unchanged(client):
    """The v3 endpoints keep their exact legacy JSON (regression probe)."""
    response = client.get("/v3/health")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 3
    # Legacy error path: unknown caption session (404, snake_case code).
    missing = client.get("/v3/caption-sessions/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "unknown_session"
    assert "legacy_code" not in missing.json()["error"]
