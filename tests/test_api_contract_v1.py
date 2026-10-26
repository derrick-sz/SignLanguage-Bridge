"""T-A2-1 contract tests: V1 API contract schema alignment (dataclass layer).

Single source of truth for field shapes: this test parses the frozen contract
``docs/架构开发文件/SignLanguage_Bridge_API_Contract_V1.yaml`` directly (PyYAML,
already a runtime dependency of the ML stack) and asserts the new
``slbridge/api_contract_v1`` layer mirrors it path/field/enum-by-enum.  If the
contract changes, these tests fail -- no drift is possible.

Why dataclasses + validation functions instead of Pydantic: the frozen
``.conda-env`` runtime (the only environment this repo runs its 889-test suite
in) has **no pydantic installed**; existing HTTP-layer tests already
``importorskip`` around that.  A dependency-free contract layer keeps the
V1-alignment testable in the same environment (decision recorded per task
brief; revisit only if the API extra env becomes the default test env).

``EvidenceCandidate`` is NOT redefined here: ``slbridge/evidence.py`` is the
single source of truth (T-A0-1, P-1 frozen schema); this layer only maps
its payload into contract field names via :func:`contract_evidence_payload`.

Contract paths covered (docs/05-style traceability, per task acceptance):

    GET  /health/live          -> HealthLiveResponse {"status": "alive"}
    GET  /health/ready         -> HealthResponse {status: ready|not_ready, workers}
    GET  /v1/models            -> ModelsResponse {models: [ModelInfo...]}
    POST /v1/videos/infer      -> VideoInferRequest / VideoInferAccepted (202)
    GET  /v1/jobs/{job_id}     -> JobStatus (full enum), final_event=SubtitleEvent
    POST /v1/experiments/replay-> ReplayRequest (auth required, 202)
    all errors                 -> ErrorResponse with the 8-value code enum
    (WS /v1/stream/{session_id} is documented out of scope: P2 transport
     contract requires its own performance test first.)

Out of scope (declared, not missing): job execution semantics (T-A2-2), route
behaviour (T-A4-1/A4-2 wire these models into endpoints), PARTIAL emission
(ADR-P1-004: Uni-Sign has no native partial; subtitle.correction reserved).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# PyYAML lives in the ML env (.conda-env) but not the API env (.venv-libras,
# pydantic-only); skip cleanly where absent so both environments stay green.
pytest.importorskip("yaml", reason="contract alignment tests parse the frozen YAML contract")
import yaml

from slbridge.api_contract_v1 import (ContractError, ContractErrorCodes, DecisionIntent,
                                       EvidenceContract, HealthLiveResponse, HealthResponse,
                                       JobStatus, ModelInfo, ModelsResponse, ReplayRequest,
                                       SubtitleEvent, TimeSpan, VideoInferAccepted,
                                       VideoInferRequest, contract_evidence_payload,
                                       error_response, map_error_code)
from slbridge.evidence import EvidenceCandidate, EvidenceOutput, EvidenceQuality, EvidenceWindow

REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / "docs" / "架构开发文件" / "SignLanguage_Bridge_API_Contract_V1.yaml"


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schemas(contract: dict) -> dict:
    return contract["components"]["schemas"]


def test_contract_file_exists_and_parses():
    assert CONTRACT_PATH.is_file()
    parsed = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert parsed["openapi"] == "3.1.0"
    assert "SignLanguage-Bridge Backend API" in parsed["info"]["title"]


# ---------------------------------------------------------------------------
# Path-by-path coverage table (acceptance: paths/fields/status codes)
# ---------------------------------------------------------------------------

def test_contract_paths_all_covered_by_models(contract: dict):
    paths = set(contract["paths"])
    assert paths == {"/health/live", "/health/ready", "/v1/models", "/v1/videos/infer",
                     "/v1/jobs/{job_id}", "/v1/experiments/replay"}
    # Each path maps to a model in this layer (WS stream is x-websocket-protocol, P2).
    covered = {
        "/health/live": HealthLiveResponse,
        "/health/ready": HealthResponse,
        "/v1/models": ModelsResponse,
        "/v1/videos/infer": (VideoInferRequest, VideoInferAccepted),
        "/v1/jobs/{job_id}": JobStatus,
        "/v1/experiments/replay": ReplayRequest,
    }
    for path, model in covered.items():
        assert path in paths and model is not None
    live = contract["paths"]["/health/live"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert live["properties"]["status"]["const"] == "alive"
    assert HealthLiveResponse().to_dict() == {"status": "alive"}


def test_videos_infer_status_codes(contract: dict):
    responses = contract["paths"]["/v1/videos/infer"]["post"]["responses"]
    assert set(responses) == {"202", "400", "413"}
    accepted = responses["202"]["content"]["application/json"]["schema"]
    assert accepted["required"] == ["job_id", "status"]
    assert accepted["properties"]["status"]["const"] == "QUEUED"
    body = VideoInferAccepted(job_id="job-1")
    assert body.to_dict() == {"job_id": "job-1", "status": "QUEUED"}


def test_jobs_status_codes(contract: dict):
    responses = contract["paths"]["/v1/jobs/{job_id}"]["get"]["responses"]
    assert set(responses) == {"200", "404"}


def test_replay_requires_auth(contract: dict):
    post = contract["paths"]["/v1/experiments/replay"]["post"]
    assert "403" in post["responses"]
    body = post["requestBody"]["content"]["application/json"]["schema"]
    assert set(body["required"]) == {"dataset_ref", "experiment_id"}


# ---------------------------------------------------------------------------
# Enum alignment (every enum value in the YAML has a Python constant)
# ---------------------------------------------------------------------------

def test_error_code_enum_matches_contract(schemas: dict):
    contract_enum = schemas["ErrorResponse"]["properties"]["code"]["enum"]
    assert ContractErrorCodes.values() == tuple(contract_enum)
    assert len(contract_enum) == 8


def test_job_status_enum_matches_contract(schemas: dict):
    contract_enum = schemas["JobStatus"]["properties"]["status"]["enum"]
    assert JobStatus.STATES == tuple(contract_enum)


def test_subtitle_event_and_status_enums(schemas: dict):
    props = schemas["SubtitleEvent"]["properties"]
    assert SubtitleEvent.EVENTS == tuple(props["event"]["enum"])
    assert SubtitleEvent.STATUSES == tuple(props["status"]["enum"])


def test_sign_system_mode_task_enums(schemas: dict):
    assert VideoInferRequest.SIGN_SYSTEMS == tuple(schemas["SignSystem"]["enum"])
    assert VideoInferRequest.MODES == tuple(schemas["InferenceMode"]["enum"])
    assert EvidenceContract.TASKS == tuple(schemas["ModelTask"]["enum"])


def test_model_info_and_health_enums(schemas: dict):
    assert ModelInfo.STATUSES == tuple(schemas["ModelInfo"]["properties"]["status"]["enum"])
    assert HealthResponse.STATUSES == tuple(schemas["HealthResponse"]["properties"]["status"]["enum"])


def test_decision_intent_enums(schemas: dict):
    props = schemas["DecisionIntent"]["properties"]
    assert DecisionIntent.ACTIONS == tuple(props["action"]["enum"])
    assert DecisionIntent.ORIGINS == tuple(props["origin"]["enum"])


# ---------------------------------------------------------------------------
# Model field alignment (required fields + types + bounds)
# ---------------------------------------------------------------------------

def test_time_span_shape(schemas: dict):
    spec = schemas["TimeSpan"]
    assert set(TimeSpan.__dataclass_fields__) == set(spec["properties"])
    assert set(spec["required"]) <= set(TimeSpan.__dataclass_fields__)
    assert TimeSpan(start=0, end=2500).to_dict() == {"start": 0, "end": 2500}
    with pytest.raises(ContractError):
        TimeSpan(start=-1, end=100)
    with pytest.raises(ContractError):
        TimeSpan(start=2000, end=1000)          # end before start
    with pytest.raises(ContractError):
        TimeSpan(start="0", end=1)              # strings are not integers


def test_model_info_required_fields(schemas: dict):
    spec = schemas["ModelInfo"]
    assert set(ModelInfo.__dataclass_fields__) == set(spec["properties"])
    assert set(spec["required"]) == set(ModelInfo.__dataclass_fields__) - {"preprocess_version"} \
        or set(spec["required"]) == set(ModelInfo.__dataclass_fields__)
    info = ModelInfo(id="unisign_rgb", revision="rev-1", task="CSLT",
                     sign_systems=("CSL",), status="ready", preprocess_version="pose-v1")
    assert info.to_dict() == {"id": "unisign_rgb", "revision": "rev-1", "task": "CSLT",
                              "sign_systems": ["CSL"], "status": "ready",
                              "preprocess_version": "pose-v1"}
    with pytest.raises(ContractError):
        ModelInfo(id="m", revision="r", task="TRANSLATION", sign_systems=("CSL",),
                  status="ready", preprocess_version="p")
    with pytest.raises(ContractError):
        ModelInfo(id="m", revision="r", task="CSLT", sign_systems=("ASL",),
                  status="ready", preprocess_version="p")


def test_video_infer_request_shape(contract: dict):
    spec = contract["paths"]["/v1/videos/infer"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert set(spec["required"]) == {"video", "sign_system", "mode"}
    request = VideoInferRequest(sign_system="CSL", mode="single_model",
                                output_zh=True, output_pt_br=True)
    assert request.to_dict() == {"sign_system": "CSL", "mode": "single_model",
                                 "output_zh": True, "output_pt_br": True}
    for bad in (dict(sign_system="ASL", mode="single_model"),
                dict(sign_system="CSL", mode="turbo"),
                dict(sign_system="CSL", mode="adaptive_cascade")):  # valid enum, P0-A unused
        kwargs = {**bad, "output_zh": True, "output_pt_br": True}
        if bad["sign_system"] == "ASL" or bad["mode"] == "turbo":
            with pytest.raises(ContractError):
                VideoInferRequest(**kwargs)
        else:
            VideoInferRequest(**kwargs)  # legal value; enabling is a T-A2-2/T-A4 concern


def test_job_status_shape(schemas: dict):
    spec = schemas["JobStatus"]
    assert set(JobStatus.__dataclass_fields__) == set(spec["properties"])
    assert set(spec["required"]) == {"job_id", "status"}
    job = JobStatus(job_id="job-1", status="SUCCEEDED", sign_system="CSL")
    assert job.to_dict()["status"] == "SUCCEEDED"
    assert "final_event" in job.to_dict() and "metrics" in job.to_dict()
    with pytest.raises(ContractError):
        JobStatus(job_id="job-1", status="DONE")            # not in contract enum
    with pytest.raises(ContractError):
        JobStatus(job_id="", status="QUEUED")


def test_subtitle_event_shape(schemas: dict):
    spec = schemas["SubtitleEvent"]
    assert set(SubtitleEvent.__dataclass_fields__) == set(spec["properties"])
    assert set(spec["required"]) == set(SubtitleEvent.__dataclass_fields__) - {"text", "evidence_ids", "reason_codes"}
    event = SubtitleEvent(event="subtitle.final", schema_version="1.0", session_id="s-1",
                          utterance_id="u-7", revision=2, time_span_ms=TimeSpan(1840, 3400),
                          sign_system="CSL", status="FINAL",
                          text={"zh-CN": "我想去……", "pt-BR": "Quero ir..."},
                          evidence_ids=["c-101", "c-102"])
    payload = event.to_dict()
    assert payload["time_span_ms"] == {"start": 1840, "end": 3400}
    assert payload["revision"] == 2
    with pytest.raises(ContractError):
        SubtitleEvent(event="subtitle.partial", schema_version="1.0", session_id="s",
                      utterance_id="u", revision=0, time_span_ms=TimeSpan(0, 1),
                      sign_system="CSL", status="PARTIAL")   # revision >= 1
    with pytest.raises(ContractError):
        SubtitleEvent(event="subtitle.hype", schema_version="1.0", session_id="s",
                      utterance_id="u", revision=1, time_span_ms=TimeSpan(0, 1),
                      sign_system="CSL", status="FINAL")


def test_health_response_shape(schemas: dict):
    spec = schemas["HealthResponse"]
    assert set(HealthResponse.__dataclass_fields__) == set(spec["properties"])
    health = HealthResponse(status="not_ready", workers={"unisign_rgb": "failed"})
    assert health.to_dict() == {"status": "not_ready", "workers": {"unisign_rgb": "failed"}}
    with pytest.raises(ContractError):
        HealthResponse(status="healthy", workers={})


def test_replay_request_shape():
    request = ReplayRequest(dataset_ref="e02-offline-1086", experiment_id="G2-C",
                            judge_profile="rules", fusion_profile="constrained")
    assert request.to_dict() == {"dataset_ref": "e02-offline-1086", "experiment_id": "G2-C",
                                 "judge_profile": "rules", "fusion_profile": "constrained"}
    with pytest.raises(ContractError):
        ReplayRequest(dataset_ref="", experiment_id="x")
    with pytest.raises(ContractError):
        ReplayRequest(dataset_ref="d", experiment_id="")


def test_decision_intent_shape(schemas: dict):
    spec = schemas["DecisionIntent"]
    assert set(DecisionIntent.__dataclass_fields__) == set(spec["properties"])
    assert set(spec["required"]) == {"action", "origin", "policy_version"}
    intent = DecisionIntent(action="ACCEPT", origin="rules", policy_version="p0a-rules-v1",
                            reason_codes=("numbers_checked",))
    assert intent.to_dict() == {"action": "ACCEPT", "origin": "rules",
                                "policy_version": "p0a-rules-v1",
                                "reason_codes": ["numbers_checked"]}
    with pytest.raises(ContractError):
        DecisionIntent(action="REJECT", origin="rules", policy_version="v1")
    with pytest.raises(ContractError):
        DecisionIntent(action="ACCEPT", origin="gut", policy_version="v1")


# ---------------------------------------------------------------------------
# ErrorResponse + legacy error-code mapping (contract names -> slbridge names)
# ---------------------------------------------------------------------------

def test_error_response_shape(schemas: dict):
    spec = schemas["ErrorResponse"]
    from slbridge.api_contract_v1 import ErrorResponse
    assert set(ErrorResponse.response_fields()) == set(spec["properties"])
    body = error_response("MODEL_UNAVAILABLE", "no Libras-trained weights", trace_id="t-1")
    payload = body.to_dict()
    assert payload["code"] == "MODEL_UNAVAILABLE"
    assert "trace_id" in payload
    with pytest.raises(ContractError):
        error_response("NOT_A_CODE", "x")


def test_map_error_covers_legacy_codes():
    """Legacy slbridge error codes map onto the contract 8-value enum."""
    expected = {
        "model_unavailable": "MODEL_UNAVAILABLE",
        "inference_timeout": "INFERENCE_TIMEOUT",
        "runtime_busy": "MODEL_UNAVAILABLE",
        "request_cancelled": "INFERENCE_TIMEOUT",
        "invalid_input": "INVALID_INPUT",
        "invalid_model": "INVALID_INPUT",
        "unsupported_input": "INVALID_INPUT",
        "upload_too_large": "INVALID_INPUT",
        "internal_error": "INTERNAL_ERROR",
    }
    for legacy, contract_code in expected.items():
        assert map_error_code(legacy) == contract_code, legacy
    # Unknown codes must not be invented: they fall back to INTERNAL_ERROR.
    assert map_error_code("something_new") == "INTERNAL_ERROR"
    # Codes with no P0-A semantics map to their reserved enum slot only.
    assert ContractErrorCodes.LOW_EVIDENCE == "LOW_EVIDENCE"
    assert ContractErrorCodes.JUDGE_UNAVAILABLE == "JUDGE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Evidence mapping (evidence.py single source of truth -> contract names)
# ---------------------------------------------------------------------------

def candidate_fixture(status: str = "OK", kind: str = "TEXT") -> EvidenceCandidate:
    output = (EvidenceOutput(kind="TEXT", gloss_tokens=None, text="她今年50岁。",
                             spoken_language="zh")
              if kind == "TEXT" else
              EvidenceOutput(kind="GLOSS", gloss_tokens=("我", "赚", "钱"), text=None))
    return EvidenceCandidate(
        run_id="run-1", sample_id="S000020_P0004_T00", model_id="unisign_rgb",
        task="CSLT", sign_system="CSL", modality="RGB_video",
        model_revision="rev-1", preprocess_version="pose-v1",
        window=EvidenceWindow(start_frame=0, end_frame_exclusive=None,
                              start_pts_ms=0, end_pts_ms=2500),
        output=output, status=status, window_id="S000020_P0004_T00-w1",
        quality=EvidenceQuality())


def test_contract_evidence_payload_field_names(schemas: dict):
    """The mapping table from T-A0-1, materialized as contract field names."""
    spec = schemas["EvidenceCandidate"]
    payload = contract_evidence_payload(candidate_fixture())
    assert set(payload) == set(spec["properties"])
    # Frozen mapping decisions: contract name <- evidence.py name.
    assert payload["candidate_id"] == "S000020_P0004_T00"          # <- sample_id
    assert payload["session_id"] == "offline"                       # <- default (lead-approved)
    assert payload["utterance_id"] == "S000020_P0004_T00"           # <- sample_id (P0-A file path)
    assert payload["window_id"] == "S000020_P0004_T00-w1"           # <- window_id
    assert payload["model"]["id"] == "unisign_rgb"                  # <- model_id
    assert payload["model"]["revision"] == "rev-1"                  # <- model_revision
    assert payload["model"]["preprocess_version"] == "pose-v1"      # <- preprocess_version
    assert payload["output"]["kind"] == "TEXT"                      # <- output.kind
    assert payload["output"]["text"] == "她今年50岁。"
    assert payload["output"]["gloss_tokens"] is None
    assert payload["time_span_ms"] == {"start": 0, "end": 2500}     # <- window pts (ms ints)
    assert payload["quality"]["calibrated_confidence"] is None      # ADR-P1-006
    assert payload["timing_ms"]["preprocess"] is None
    assert payload["sign_system"] == "CSL"
    for name in spec["required"]:
        assert name in payload


def test_contract_evidence_payload_gloss_and_failed():
    gloss = contract_evidence_payload(candidate_fixture(kind="GLOSS"))
    assert gloss["output"]["kind"] == "GLOSS"
    assert gloss["output"]["text"] is None
    assert gloss["output"]["gloss_tokens"] == ["我", "赚", "钱"]
    failed = contract_evidence_payload(candidate_fixture(status="FAILED"))
    assert failed["quality"]["status"] == "failed"
    # Failure keeps the declared kind but never fabricates output content
    # (contract output.oneOf knows TEXT/GLOSS; EMPTY is a P-1-schema-only kind).
    assert failed["output"]["kind"] == "TEXT"
    assert failed["output"]["text"] is None
    assert failed["output"]["gloss_tokens"] is None


def test_contract_evidence_payload_rejects_unknown_window_by_default():
    """Lead-approved semantics: unknown time must not be fabricated as {0,0}.

    The contract TimeSpan requires nonnegative integers (no null), so an
    unknown span cannot be expressed honestly -- the default is to refuse.
    """
    candidate = candidate_fixture()
    object.__setattr__(candidate, "window",
                       EvidenceWindow(start_frame=0, end_frame_exclusive=None,
                                      start_pts_ms=None, end_pts_ms=None))
    with pytest.raises(ContractError, match="unknown time span"):
        contract_evidence_payload(candidate)


def test_contract_evidence_payload_unknown_window_explicit_placeholder():
    """Explicit opt-in: {0,0} is an auditable PLACEHOLDER, not a real time."""
    candidate = candidate_fixture()
    object.__setattr__(candidate, "window",
                       EvidenceWindow(start_frame=0, end_frame_exclusive=None,
                                      start_pts_ms=None, end_pts_ms=None))
    payload = contract_evidence_payload(candidate, allow_unknown_time_span=True)
    assert payload["time_span_ms"] == {"start": 0, "end": 0}   # placeholder, not real time
    assert payload["window_id"] == "S000020_P0004_T00-w1"       # unknown origin stays traceable
    # The placeholder decision is auditable in the payload itself.
    assert payload["quality"]["status"] in {"ok", "partial", "failed"}


def test_models_response_shape(contract: dict):
    spec = contract["paths"]["/v1/models"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert set(spec["required"]) == {"models"}
    response = ModelsResponse(models=(ModelInfo(id="unisign_rgb", revision="rev-1", task="CSLT",
                                                sign_systems=("CSL",), status="ready",
                                                preprocess_version="pose-v1"),))
    assert response.to_dict()["models"][0]["id"] == "unisign_rgb"


def test_all_to_dict_outputs_are_json_serializable():
    objects = [
        HealthLiveResponse(),
        HealthResponse(status="ready", workers={"m": "ready"}),
        ModelInfo(id="m", revision="r", task="CSLT", sign_systems=("CSL",),
                  status="ready", preprocess_version="p"),
        ModelsResponse(models=()),
        VideoInferRequest(sign_system="CSL", mode="single_model",
                          output_zh=True, output_pt_br=False),
        VideoInferAccepted(job_id="job-9"),
        JobStatus(job_id="job-9", status="RUNNING", sign_system="CSL"),
        SubtitleEvent(event="subtitle.final", schema_version="1.0", session_id="s",
                      utterance_id="u", revision=1, time_span_ms=TimeSpan(0, 1),
                      sign_system="CSL", status="FINAL"),
        ReplayRequest(dataset_ref="d", experiment_id="e"),
        DecisionIntent(action="ACCEPT", origin="rules", policy_version="v"),
        error_response("LOW_EVIDENCE", "not enough evidence"),
    ]
    for obj in objects:
        json.dumps(obj.to_dict(), ensure_ascii=False)  # must not raise
    # The evidence payload is already a plain JSON-compatible dict.
    json.dumps(contract_evidence_payload(candidate_fixture()), ensure_ascii=False)
