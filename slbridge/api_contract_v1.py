"""V1 API contract layer (P0-A phase A2, task T-A2-1 / task-15).

Typed, dependency-free models for the frozen HTTP contract
``docs/架构开发文件/SignLanguage_Bridge_API_Contract_V1.yaml``.  ``tests/test_api_contract_v1.py``
parses that YAML directly and asserts this module mirrors it enum-by-enum and
field-by-field, so the contract file stays the single source of truth.

Implementation decision (per task brief): **dataclasses + validation functions,
not Pydantic**.  The repo's only test environment (``.conda-env``) has no
pydantic installed -- the existing HTTP tests already ``importorskip`` around
that -- and a contract layer must be testable in the same 889-test suite
without new heavyweight dependencies.  Revisit only if the API extra env
becomes the default test environment.

Evidence fields are NOT redefined here: ``slbridge/evidence.py`` is the single
source of truth (T-A0-1, aligned to the P-1 frozen schema).
:func:`contract_evidence_payload` materializes the frozen mapping table into
contract field names only:

    contract field        <- evidence.py field (T-A0-1 docstring)
    --------------------- ---------------------------------------------
    candidate_id          <- sample_id
    session_id            <- "offline" default (lead-approved; recorder-level
                              concern, never inside the candidate block)
    utterance_id          <- sample_id (P0-A whole-file utterance)
    window_id             <- window_id
    time_span_ms          <- window.start_pts_ms/end_pts_ms (already int ms;
                              unknown (null) REFUSES by default -- the contract
                              TimeSpan cannot express "unknown"; pass
                              allow_unknown_time_span=True for an auditable
                              {0, 0} placeholder, never a fabricated time)
    model.id/revision     <- model_id / model_revision
    model.preprocess_version <- preprocess_version
    model.task            <- task ("CSLT"/"CSLR"; "ISLR" cannot appear in the
                              V1 contract enums and is rejected on mapping)
    output.kind/text/...  <- output.kind / output.text / ...
    quality.calibrated_confidence <- quality.calibrated_reliability (null per
                              ADR-P1-006 -- raw scores never masquerade as
                              probabilities)
    quality.status        <- status OK->ok, FAILED->failed, others->partial
                              (contract enum: ok|partial|failed)
    timing_ms.preprocess/inference <- none (the job layer owns real timings;
                              contract allows null)

Error mapping: :func:`map_error_code` translates the legacy slbridge error
codes (``api.py`` ``operation`` mapping) onto the contract 8-value enum;
unknown codes fall back to INTERNAL_ERROR rather than being invented.
``JUDGE_UNAVAILABLE`` and ``LOW_EVIDENCE`` exist as enum slots only -- they
acquire semantics after the G2 track (ADR-P1-005: fusion is not in the main
path; do not emit these codes speculatively).

Out of scope on purpose: endpoint wiring (T-A2-3/T-A4-1/T-A4-2), job execution
semantics (T-A2-2), WebSocket transport (P2, needs its own performance
contract first), PARTIAL emission (ADR-P1-004: Uni-Sign has no native
partial; ``subtitle.partial`` and ``subtitle.correction`` are reserved enum
values, not behaviours).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping

from .evidence import EvidenceCandidate

__all__ = [
    "ContractError", "ContractErrorCodes", "TimeSpan", "HealthLiveResponse",
    "HealthResponse", "ModelInfo", "ModelsResponse", "VideoInferRequest",
    "VideoInferAccepted", "JobStatus", "SubtitleEvent", "ReplayRequest",
    "DecisionIntent", "EvidenceContract", "ErrorResponse", "error_response",
    "map_error_code", "contract_evidence_payload",
]


class ContractError(ValueError):
    """Raised when a payload violates the frozen V1 contract."""


class ContractErrorCodes:
    """The 8-value ErrorResponse.code enum (contract ``ErrorResponse``)."""

    INVALID_INPUT = "INVALID_INPUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INVALID_POSE_SCHEMA = "INVALID_POSE_SCHEMA"
    OUT_OF_VRAM = "OUT_OF_VRAM"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    JUDGE_UNAVAILABLE = "JUDGE_UNAVAILABLE"      # reserved: G2-track semantics only
    LOW_EVIDENCE = "LOW_EVIDENCE"                # reserved: G2-track semantics only
    INTERNAL_ERROR = "INTERNAL_ERROR"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return (cls.INVALID_INPUT, cls.MODEL_UNAVAILABLE, cls.INVALID_POSE_SCHEMA,
                cls.OUT_OF_VRAM, cls.INFERENCE_TIMEOUT, cls.JUDGE_UNAVAILABLE,
                cls.LOW_EVIDENCE, cls.INTERNAL_ERROR)


_LEGACY_ERROR_MAP = {
    "invalid_input": "INVALID_INPUT",
    "invalid_model": "INVALID_INPUT",
    "invalid_request": "INVALID_INPUT",
    "unsupported_input": "INVALID_INPUT",
    "upload_too_large": "INVALID_INPUT",
    "unknown_job": "INVALID_INPUT",
    "request_conflict": "INVALID_INPUT",
    "request_capacity": "INVALID_INPUT",
    "session_conflict": "INVALID_INPUT",
    "model_unavailable": "MODEL_UNAVAILABLE",
    "unknown_model": "MODEL_UNAVAILABLE",
    "unknown_language": "MODEL_UNAVAILABLE",
    "runtime_busy": "MODEL_UNAVAILABLE",
    "inference_timeout": "INFERENCE_TIMEOUT",
    "request_cancelled": "INFERENCE_TIMEOUT",
    "internal_error": "INTERNAL_ERROR",
}


def map_error_code(legacy_code: str) -> str:
    """Map a legacy slbridge error code onto the contract 8-value enum.

    Unknown codes fall back to INTERNAL_ERROR; no code is ever invented.
    """

    if not isinstance(legacy_code, str):
        raise ContractError("legacy error code must be a string")
    return _LEGACY_ERROR_MAP.get(legacy_code, ContractErrorCodes.INTERNAL_ERROR)


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ContractError(f"{name} must be a 1..128 character identifier")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class TimeSpan:
    """Media timeline span in integer milliseconds (contract ``TimeSpan``)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{name} must be a nonnegative integer (ms)")
        if self.end < self.start:
            raise ContractError("end must not precede start")

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class HealthLiveResponse:
    """``GET /health/live`` -- process liveness only."""

    status: str = "alive"

    def __post_init__(self) -> None:
        if self.status != "alive":
            raise ContractError("health live status must be 'alive'")

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status}


@dataclass(frozen=True)
class HealthResponse:
    """``GET /health/ready`` -- dependency readiness (contract ``HealthResponse``)."""

    STATUSES = ("ready", "not_ready")

    status: str
    workers: dict[str, str]

    def __post_init__(self) -> None:
        if self.status not in self.STATUSES:
            raise ContractError(f"health status must be one of {self.STATUSES}")
        if not isinstance(self.workers, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in self.workers.items()):
            raise ContractError("workers must map names to status strings")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "workers": dict(self.workers)}


@dataclass(frozen=True)
class ModelInfo:
    """Contract ``ModelInfo`` (registry snapshot entry)."""

    STATUSES = ("ready", "loading", "unavailable", "failed")

    id: str
    revision: str
    task: str
    sign_systems: tuple[str, ...]
    status: str
    preprocess_version: str

    def __post_init__(self) -> None:
        _identifier(self.id, "id")
        _text(self.revision, "revision")
        _text(self.preprocess_version, "preprocess_version")
        if self.task not in EvidenceContract.TASKS:
            raise ContractError(f"task must be one of {EvidenceContract.TASKS}")
        if (not isinstance(self.sign_systems, (tuple, list)) or not self.sign_systems
                or any(value not in VideoInferRequest.SIGN_SYSTEMS for value in self.sign_systems)):
            raise ContractError(f"sign_systems must be nonempty and within {VideoInferRequest.SIGN_SYSTEMS}")
        if isinstance(self.sign_systems, list):
            object.__setattr__(self, "sign_systems", tuple(self.sign_systems))
        if self.status not in self.STATUSES:
            raise ContractError(f"status must be one of {self.STATUSES}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "revision": self.revision, "task": self.task,
                "sign_systems": list(self.sign_systems), "status": self.status,
                "preprocess_version": self.preprocess_version}


@dataclass(frozen=True)
class ModelsResponse:
    """``GET /v1/models`` response body."""

    models: tuple[ModelInfo, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.models, (tuple, list)) or any(
                not isinstance(entry, ModelInfo) for entry in self.models):
            raise ContractError("models must be ModelInfo entries")
        if isinstance(self.models, list):
            object.__setattr__(self, "models", tuple(self.models))

    def to_dict(self) -> dict[str, Any]:
        return {"models": [entry.to_dict() for entry in self.models]}


@dataclass(frozen=True)
class VideoInferRequest:
    """``POST /v1/videos/infer`` form fields (video bytes travel separately)."""

    SIGN_SYSTEMS = ("CSL", "LIBRAS", "UNKNOWN")
    MODES = ("parallel_fusion", "adaptive_cascade", "single_model")

    sign_system: str
    mode: str
    output_zh: bool = True
    output_pt_br: bool = True

    def __post_init__(self) -> None:
        if self.sign_system not in self.SIGN_SYSTEMS:
            raise ContractError(f"sign_system must be one of {self.SIGN_SYSTEMS}")
        if self.mode not in self.MODES:
            raise ContractError(f"mode must be one of {self.MODES}")
        for name in ("output_zh", "output_pt_br"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"{name} must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"sign_system": self.sign_system, "mode": self.mode,
                "output_zh": self.output_zh, "output_pt_br": self.output_pt_br}


@dataclass(frozen=True)
class VideoInferAccepted:
    """``202`` body of ``POST /v1/videos/infer``."""

    job_id: str
    status: str = "QUEUED"

    def __post_init__(self) -> None:
        _identifier(self.job_id, "job_id")
        if self.status != "QUEUED":
            raise ContractError("an accepted job must be QUEUED")

    def to_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "status": self.status}


@dataclass(frozen=True)
class JobStatus:
    """``GET /v1/jobs/{job_id}`` body (contract ``JobStatus``).

    Execution semantics (state transitions, retry-once, timeout mapping) belong
    to the job layer (T-A2-2); this type only pins the payload shape.
    """

    STATES = ("QUEUED", "PREPROCESSING", "RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED")

    job_id: str
    status: str
    sign_system: str | None = None
    final_event: "SubtitleEvent | None" = None
    error: "ErrorResponse | None" = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.job_id, "job_id")
        if self.status not in self.STATES:
            raise ContractError(f"status must be one of {self.STATES}")
        if self.sign_system is not None and self.sign_system not in VideoInferRequest.SIGN_SYSTEMS:
            raise ContractError(f"sign_system must be one of {VideoInferRequest.SIGN_SYSTEMS}")
        if self.final_event is not None and not isinstance(self.final_event, SubtitleEvent):
            raise ContractError("final_event must be a SubtitleEvent")
        if self.error is not None and not isinstance(self.error, ErrorResponse):
            raise ContractError("error must be an ErrorResponse")
        if not isinstance(self.metrics, dict) or any(
                isinstance(key, bool) or not isinstance(key, str)
                or isinstance(value, bool) or not isinstance(value, (int, float))
                for key, value in self.metrics.items()):
            raise ContractError("metrics must map names to numbers")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_id": self.job_id, "status": self.status}
        if self.sign_system is not None:
            payload["sign_system"] = self.sign_system
        payload["final_event"] = self.final_event.to_dict() if self.final_event is not None else None
        payload["error"] = self.error.to_dict() if self.error is not None else None
        payload["metrics"] = dict(self.metrics)
        return payload


@dataclass(frozen=True)
class SubtitleEvent:
    """Contract ``SubtitleEvent`` (subtitle.partial/.final/.uncertain/.correction).

    P0-A only ever emits ``subtitle.final`` / ``subtitle.uncertain`` (whole
    utterance); PARTIAL would fabricate streaming ability Uni-Sign does not
    have (ADR-P1-004). All four event names stay valid enum values.
    """

    EVENTS = ("subtitle.partial", "subtitle.final", "subtitle.uncertain", "subtitle.correction")
    STATUSES = ("PARTIAL", "FINAL", "UNCERTAIN")

    event: str
    schema_version: str
    session_id: str
    utterance_id: str
    revision: int
    time_span_ms: TimeSpan
    sign_system: str
    status: str
    text: dict[str, str] | None = None
    evidence_ids: tuple[str, ...] | None = None
    reason_codes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.event not in self.EVENTS:
            raise ContractError(f"event must be one of {self.EVENTS}")
        if self.schema_version != "1.0":
            raise ContractError("schema_version must be '1.0'")
        _identifier(self.session_id, "session_id")
        _identifier(self.utterance_id, "utterance_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ContractError("revision must be an integer >= 1")
        if not isinstance(self.time_span_ms, TimeSpan):
            raise ContractError("time_span_ms must be a TimeSpan")
        if self.sign_system not in VideoInferRequest.SIGN_SYSTEMS:
            raise ContractError(f"sign_system must be one of {VideoInferRequest.SIGN_SYSTEMS}")
        if self.status not in self.STATUSES:
            raise ContractError(f"status must be one of {self.STATUSES}")
        if self.text is not None and (not isinstance(self.text, dict) or any(
                key not in {"zh-CN", "pt-BR"} or not isinstance(value, str)
                for key, value in self.text.items())):
            raise ContractError("text must map 'zh-CN'/'pt-BR' to strings")
        for name in ("evidence_ids", "reason_codes"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (tuple, list)) or any(
                        not isinstance(item, str) for item in value):
                    raise ContractError(f"{name} must be a list of strings or null")
                if isinstance(value, list):
                    object.__setattr__(self, name, tuple(value))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": self.event, "schema_version": self.schema_version,
            "session_id": self.session_id, "utterance_id": self.utterance_id,
            "revision": self.revision, "time_span_ms": self.time_span_ms.to_dict(),
            "sign_system": self.sign_system, "status": self.status}
        if self.text is not None:
            payload["text"] = dict(self.text)
        if self.evidence_ids is not None:
            payload["evidence_ids"] = list(self.evidence_ids)
        if self.reason_codes is not None:
            payload["reason_codes"] = list(self.reason_codes)
        return payload


@dataclass(frozen=True)
class ReplayRequest:
    """``POST /v1/experiments/replay`` body (authorized offline replay)."""

    dataset_ref: str
    experiment_id: str
    judge_profile: str = "rules"
    fusion_profile: str = "constrained"

    def __post_init__(self) -> None:
        for name in ("dataset_ref", "experiment_id", "judge_profile", "fusion_profile"):
            _text(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        return {"dataset_ref": self.dataset_ref, "experiment_id": self.experiment_id,
                "judge_profile": self.judge_profile, "fusion_profile": self.fusion_profile}


@dataclass(frozen=True)
class DecisionIntent:
    """Contract ``DecisionIntent`` (ACCEPT|FUSE|RETRY|UNCERTAIN).

    Interface placeholder only: G2 decided fusion does not enter the main path
    (G2_DECISION_MEMO), so no code path constructs this type yet; the shape is
    pinned for the T-A5-1 Protocol.
    """

    ACTIONS = ("ACCEPT", "FUSE", "RETRY", "UNCERTAIN")
    ORIGINS = ("rules", "jev", "timeout_fallback")

    action: str
    origin: str
    policy_version: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in self.ACTIONS:
            raise ContractError(f"action must be one of {self.ACTIONS}")
        if self.origin not in self.ORIGINS:
            raise ContractError(f"origin must be one of {self.ORIGINS}")
        _text(self.policy_version, "policy_version")
        if not isinstance(self.reason_codes, (tuple, list)) or any(
                not isinstance(item, str) for item in self.reason_codes):
            raise ContractError("reason_codes must be a list of strings")
        if isinstance(self.reason_codes, list):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "origin": self.origin,
                "policy_version": self.policy_version,
                "reason_codes": list(self.reason_codes)}


class EvidenceContract:
    """Contract-level enums that only exist as validation sets."""

    TASKS = ("CSLR", "CSLT")


@dataclass(frozen=True)
class ErrorResponse:
    """Contract ``ErrorResponse`` body."""

    code: str
    message: str
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if self.code not in ContractErrorCodes.values():
            raise ContractError(f"code must be one of {ContractErrorCodes.values()}")
        _text(self.message, "message")
        if self.trace_id is not None:
            _identifier(self.trace_id, "trace_id")

    @staticmethod
    def response_fields() -> tuple[str, ...]:
        return tuple(entry.name for entry in fields(ErrorResponse))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        return payload


def error_response(code: str, message: str, *, trace_id: str | None = None) -> ErrorResponse:
    """Build a contract ErrorResponse (validates the code enum)."""

    return ErrorResponse(code=code, message=message, trace_id=trace_id)


def contract_evidence_payload(candidate: EvidenceCandidate, *,
                              session_id: str = "offline",
                              allow_unknown_time_span: bool = False) -> dict[str, Any]:
    """Map an ``EvidenceCandidate`` into contract field names (frozen table).

    See the module docstring for the full mapping; this function only renames
    fields and adapts the two spots where the contract is narrower than the
    P-1 schema (task enum without ISLR; quality.status enum ok|partial|failed;
    integer TimeSpan).

    Unknown time spans (``window.start_pts_ms``/``end_pts_ms`` is null): the
    contract ``TimeSpan`` requires nonnegative integers -- it cannot express
    "unknown".  By default this function therefore **refuses** to fabricate a
    span (``ContractError``; skip the payload or annotate it elsewhere).  With
    ``allow_unknown_time_span=True`` the span becomes ``{start: 0, end: 0}``,
    which is an explicit **placeholder, not a real time** -- the placeholder
    decision must stay auditable at the call site.
    """

    if not isinstance(candidate, EvidenceCandidate):
        raise ContractError("evidence payload requires an EvidenceCandidate")
    if candidate.task not in EvidenceContract.TASKS:
        raise ContractError(
            f"contract task enum has no value for {candidate.task!r} (ISLR is not a V1 model task)")
    if candidate.window.start_pts_ms is None or candidate.window.end_pts_ms is None:
        if not allow_unknown_time_span:
            raise ContractError(
                "unknown time span cannot be expressed as a V1 TimeSpan (integers only); "
                "skip the payload or pass allow_unknown_time_span=True to emit an auditable "
                "{0, 0} placeholder -- never a fabricated real time")
        # Explicit opt-in: placeholder, not a real time.
        start_ms = end_ms = 0
    else:
        start_ms, end_ms = int(candidate.window.start_pts_ms), int(candidate.window.end_pts_ms)
    if end_ms < start_ms:
        raise ContractError("time_span_ms end must not precede start")
    quality_status = {"OK": "ok", "FAILED": "failed"}.get(candidate.status, "partial")
    output = candidate.output
    # Contract output.oneOf only knows TEXT/GLOSS; on failure the payload keeps
    # its original kind with a null body (quality.status=failed carries the
    # failure; never fabricate an output).
    text = output.text if candidate.status == "OK" else None
    gloss_tokens = (list(output.gloss_tokens) if output.gloss_tokens is not None else None) \
        if candidate.status == "OK" else None
    return {
        "schema_version": "1.0",
        "candidate_id": candidate.sample_id,
        "session_id": session_id,
        "utterance_id": candidate.sample_id,
        "window_id": candidate.window_id,
        "time_span_ms": {"start": start_ms, "end": end_ms},
        "sign_system": candidate.sign_system,
        "model": {
            "id": candidate.model_id,
            "revision": candidate.model_revision,
            "task": candidate.task,
            "preprocess_version": candidate.preprocess_version,
        },
        "output": {
            "kind": output.kind,
            "gloss_tokens": gloss_tokens,
            "text": text,
        },
        "quality": {
            "calibrated_confidence": candidate.quality.calibrated_reliability,
            "status": quality_status,
        },
        "timing_ms": {"preprocess": None, "inference": None},
    }
