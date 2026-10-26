"""Async job state machine and runner (P0-A phase A2, task T-A2-2 / task-16).

A job is one queued replay/inference request driven to a terminal state::

    QUEUED -> PREPROCESSING -> RUNNING -> SUCCEEDED | FAILED | TIMED_OUT | CANCELLED

* ``TRANSITIONS`` pins the legal moves; terminal states are immutable and any
  illegal move raises ``ValueError`` (see the matrix in
  ``tests/test_job_runner.py``).
* :class:`JobStore` is an in-memory registry with idempotent ``job_id``
  (re-submitting an identical job returns the same record; a conflicting
  payload raises) and optional JSONL persistence of **every** state
  transition; ``restore_from`` rebuilds the latest state per job.
* :class:`JobRunner` executes jobs **serially** through the existing
  ``ModelManager.infer_file`` single-capacity lease (ADR-P1-003: single-GPU
  serial is the default; no concurrency scheduler in P0-A).  On success it
  records a GT-free prediction row + manifest via T-A0-1's
  ``PredictionRecorder``; failures map onto the contract 8-value error enum
  (T-A2-1) with legacy ``map_error_code``.
* Retry policy: **exactly one retry** for retryable failures (MemoryError /
  OSError allocation-class errors -- pitfalls sec. 7: 32 GB Windows host
  pressure fabricates model failures), never more; non-retryable errors fail
  immediately.
* Timeout -> ``TIMED_OUT`` + ``INFERENCE_TIMEOUT``; cancel -> ``CANCELLED``;
  neither ever fabricates an output (ADR-P1-004 discipline).

Out of scope on purpose: HTTP endpoints (T-A2-3), parallel_fusion /
adaptive_cascade execution (G2 decided fusion stays out of the main path;
``single_model`` semantics only), model loading per job (the runner owns a
ready manager).

Evidence mapping note: the job's whole-file replay has no real media span, so
the recorded ``EvidenceCandidate.window`` timestamps are null (unknown --
lead-approved semantics) and the contract-layer payload would refuse them by
default (``allow_unknown_time_span=False``); the runner therefore stores
evidence via the recorder (P-1 schema, nulls allowed) and never fabricates a
V1 TimeSpan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from .api_contract_v1 import ContractErrorCodes, map_error_code
from .evidence import from_slbridge_response
from .prediction_log import PredictionRecorder

__all__ = ["JobState", "TRANSITIONS", "JobRecord", "JobStore", "JobRunner", "JobResult"]


class JobState:
    """The 7 contract job states (JobStatus.status enum)."""

    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"

    ALL = (QUEUED, PREPROCESSING, RUNNING, SUCCEEDED, FAILED, TIMED_OUT, CANCELLED)
    TERMINAL = (SUCCEEDED, FAILED, TIMED_OUT, CANCELLED)


#: Legal state transitions; empty target sets make terminal states immutable.
TRANSITIONS: dict[str, set[str]] = {
    JobState.QUEUED: {JobState.PREPROCESSING, JobState.CANCELLED},
    JobState.PREPROCESSING: {JobState.RUNNING, JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED},
    JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED},
    JobState.SUCCEEDED: set(),
    JobState.FAILED: set(),
    JobState.TIMED_OUT: set(),
    JobState.CANCELLED: set(),
}

_RETRYABLE_ERRORS = (MemoryError,)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _legacy_code_of(error: BaseException) -> str:
    """Best-effort legacy slbridge code for an exception (then contract-mapped)."""

    name = type(error).__name__
    from .runtime import ModelUnavailable, RuntimeBusy
    from .runtime_contracts import InferenceCancelled, InferenceDeadlineExceeded
    if isinstance(error, InferenceDeadlineExceeded):
        return "inference_timeout"
    if isinstance(error, InferenceCancelled):
        return "request_cancelled"
    if isinstance(error, RuntimeBusy):
        return "runtime_busy"
    if isinstance(error, ModelUnavailable):
        return "model_unavailable"
    if isinstance(error, (ValueError, OSError)):
        return "invalid_input"
    if name in {"TimeoutError", "TimeoutExpired"}:
        return "inference_timeout"
    return "internal_error"


@dataclass
class JobRecord:
    """One job's identity + mutable state under a lock."""

    job_id: str
    state: str
    sign_system: str
    sample_id: str
    request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    text: str | None = None
    gloss_tokens: tuple[str, ...] | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    attempts: int = 0
    media_metadata: dict[str, Any] | None = None       # T-A3-1: ffprobe result (or null)
    media_probe_error: str | None = None               # T-A3-1: honest failure reason
    upload_path: Path | None = None                    # T-A3-1: temp upload to clean up
    upload_expires_at: float | None = None             # T-A3-1: monotonic TTL deadline
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def transition(self, target: str) -> None:
        with self.lock:
            if target not in JobState.ALL:
                raise ValueError(f"unknown job state: {target!r}")
            if target not in TRANSITIONS[self.state]:
                raise ValueError(f"illegal job transition {self.state} -> {target}")
            self.state = target
            self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            payload: dict[str, Any] = {
                "job_id": self.job_id, "status": self.state,
                "sign_system": self.sign_system, "sample_id": self.sample_id,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "attempts": self.attempts, "metrics": dict(self.timing_ms),
            }
            if self.request_id is not None:
                payload["request_id"] = self.request_id
            if self.error_code is not None:
                payload["error"] = {"code": self.error_code,
                                    "message": self.error_message or ""}
            if self.text is not None:
                payload["text"] = self.text
            if self.gloss_tokens is not None:
                payload["gloss_tokens"] = list(self.gloss_tokens)
            payload["media_metadata"] = self.media_metadata
            payload["media_probe_error"] = self.media_probe_error
            return payload


@dataclass
class JobResult:
    """Runner return value: the record plus convenience accessors."""

    record: JobRecord

    @property
    def state(self) -> str:
        return self.record.state

    @property
    def text(self) -> str | None:
        return self.record.text


class JobStore:
    """In-memory job registry; idempotent by job_id; optional JSONL journal."""

    def __init__(self, *, persistence_path: Path | None = None, max_jobs: int = 1024) -> None:
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1:
            raise ValueError("max_jobs must be a positive integer")
        self._records: dict[str, JobRecord] = {}
        self._retired: set[str] = set()
        self._lock = threading.RLock()
        self._persistence = Path(persistence_path) if persistence_path is not None else None
        self._max_jobs = max_jobs

    def create(self, *, job_id: str, sign_system: str, sample_id: str) -> JobRecord:
        key = (sign_system, sample_id)
        with self._lock:
            existing = self._records.get(job_id)
            if existing is not None:
                if (existing.sign_system, existing.sample_id) == key:
                    return existing                      # idempotent re-submit
                raise ValueError(f"job_id {job_id!r} is bound to another payload")
            active = len(self._records) - len(self._retired)
            if active >= self._max_jobs:
                raise RuntimeError(f"job store capacity reached ({self._max_jobs}); retire jobs or raise max_jobs")
            record = JobRecord(job_id=job_id, state=JobState.QUEUED,
                               sign_system=sign_system, sample_id=sample_id)
            self._records[job_id] = record
        self._journal(record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def retire(self, job_id: str) -> bool:
        """Retire a *terminal* job from capacity accounting (record stays).

        The record remains queryable through :meth:`get`; only the bounded
        capacity slot is freed.  Non-terminal or unknown jobs are refused.
        """

        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.state not in JobState.TERMINAL:
                return False
            self._retired.add(job_id)
        self._collect_upload(record)
        return True

    def sweep_uploads(self, *, now: float | None = None) -> list[str]:
        """Delete expired uploads of terminal jobs; returns cleaned job ids.

        Prediction JSONL / manifests are evidence and are never touched --
        only the uploaded temp files (``record.upload_path``) are removed.
        """

        import time as _time
        clock = _time.monotonic() if now is None else now
        cleaned: list[str] = []
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if record.state in JobState.TERMINAL and record.upload_path is not None:
                if record.upload_expires_at is None or clock >= record.upload_expires_at:
                    if self._collect_upload(record):
                        cleaned.append(record.job_id)
        return cleaned

    @staticmethod
    def _collect_upload(record: JobRecord) -> bool:
        with record.lock:
            path = record.upload_path
            record.upload_path = None
            record.upload_expires_at = None
        if path is None:
            return False
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _journal(self, record: JobRecord) -> None:
        if self._persistence is None:
            return
        entry = record.to_dict()
        self._persistence.parent.mkdir(parents=True, exist_ok=True)
        with self._persistence.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def journal(self, record: JobRecord) -> None:
        """Persist the record's current state (called on every transition)."""

        self._journal(record)

    def restore_from(self, path: Path) -> "JobStore":
        """Rebuild the latest state per job from a JSONL journal."""

        store = JobStore()
        if not Path(path).is_file():
            return store
        latest: dict[str, dict[str, Any]] = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                latest[entry["job_id"]] = entry
        for job_id, entry in latest.items():
            record = JobRecord(job_id=job_id, state=entry["status"],
                               sign_system=entry["sign_system"], sample_id=entry["sample_id"],
                               created_at=entry.get("created_at", _utc_now()),
                               updated_at=entry.get("updated_at", _utc_now()),
                               attempts=entry.get("attempts", 0))
            store._records[job_id] = record
        return store


@dataclass
class JobRunner:
    """Serial job executor over one ready ``ModelManager`` (replay or real)."""

    manager: Any
    store: JobStore
    run_directory: Path
    run_id: str
    model_id: str
    model_version: str
    preprocess_version: str
    task: str
    sign_system: str
    timeout_seconds: float | None = 120.0
    retry_once: bool = True
    deadline_factory: Callable[[], float] | None = None   # injectable clock for tests
    _recorder_ready: bool = False

    def __post_init__(self) -> None:
        if self.task not in {"CSLT", "CSLR", "ISLR"}:
            raise ValueError("task must be 'CSLT', 'CSLR' or 'ISLR'")
        if self.sign_system not in {"CSL", "LIBRAS", "UNKNOWN"}:
            raise ValueError("sign_system must be 'CSL', 'LIBRAS' or 'UNKNOWN'")
        if self.timeout_seconds is not None and (
                isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive or null")
        self.run_directory = Path(self.run_directory)
        self._lock = threading.RLock()   # serializes jobs (single capacity)

    def _deadline(self) -> float:
        if self.deadline_factory is not None:
            return self.deadline_factory()
        return time.monotonic() + (self.timeout_seconds or 0.0)

    def _recorder(self) -> PredictionRecorder:
        if not self._recorder_ready:
            self._recorder_instance = PredictionRecorder(
                self.run_directory, run_id=self.run_id, model_id=self.model_id,
                model_version=self.model_version, preprocess_version=self.preprocess_version,
                task=self.task, sign_system=self.sign_system)
            self._recorder_ready = True
        return self._recorder_instance

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job; unknown/terminal ids are a no-op."""

        record = self.store.get(job_id)
        if record is None or record.state in JobState.TERMINAL:
            return False
        try:
            record.transition(JobState.CANCELLED)
            self.store.journal(record)
            return True
        except ValueError:
            return False

    # -- execution -------------------------------------------------------
    def run(self, *, job_id: str, sample_id: str, query_path: Path,
            cancelled_before_start: bool = False) -> JobResult:
        record = self.store.create(job_id=job_id, sign_system=self.sign_system, sample_id=sample_id)
        if cancelled_before_start or record.state == JobState.CANCELLED:
            if record.state != JobState.CANCELLED:
                self._advance(record, JobState.CANCELLED)
            return JobResult(record)
        with self._lock:
            if record.state in JobState.TERMINAL:
                return JobResult(record)          # idempotent replay of a finished job
            self._advance(record, JobState.PREPROCESSING)
            deadline = self._deadline()
            from .runtime_contracts import CancellationToken, InferenceRequest
            token = CancellationToken()
            request = InferenceRequest(
                request_id=job_id, sign_language={"CSL": "csl", "LIBRAS": "libras",
                                                  "UNKNOWN": None}[self.sign_system],
                expected_task_type={"CSLT": "text", "CSLR": "gloss_sequence",
                                    "ISLR": "classification"}[self.task],
                model_id=self.model_id, cancellation=token,
                deadline_monotonic=deadline)
            self._advance(record, JobState.RUNNING)
            response = None
            attempts = 0
            while True:
                attempts += 1
                record.attempts = attempts
                try:
                    response = self.manager.infer_file(query_path, request)
                    break
                except _RETRYABLE_ERRORS as error:              # noqa: PERF203
                    if self.retry_once and attempts == 1:
                        continue                                  # exactly one retry
                    self._fail(record, error)
                    return JobResult(record)
                except Exception as error:
                    from .runtime_contracts import InferenceDeadlineExceeded
                    if isinstance(error, InferenceDeadlineExceeded) or isinstance(error, TimeoutError):
                        self._timeout(record)
                    else:
                        self._fail(record, error)
                    return JobResult(record)
                finally:
                    if token.cancelled:
                        break
            if token.cancelled or record.state != JobState.RUNNING:
                self._cancelled(record)
                return JobResult(record)
            if response is None:
                self._timeout(record)
                return JobResult(record)
            return self._succeed(record, response)

    def _advance(self, record: JobRecord, state: str) -> None:
        record.transition(state)
        self.store.journal(record)

    def _succeed(self, record: JobRecord, response: Any) -> JobResult:
        result = response.result
        if result.status == "recognized":
            prediction = result.prediction
            record.text = getattr(prediction, "text", None)
            tokens = getattr(prediction, "tokens", None)
            if tokens is not None and not record.text:
                record.gloss_tokens = tuple(token.gloss for token in tokens)
        record.timing_ms = {key: value for key, value in (response.timing_ms or {}).items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)}
        record.request_id = response.request_id
        try:
            self._recorder().append_response(response, job_id=record.job_id)
            # T-A3-1: the run manifest carries the job's honest media metadata
            # (null + probe_error when probing failed -- never fabricated).
            self._recorder().write_manifest(
                media_metadata={record.job_id: {
                    "metadata": record.media_metadata,
                    "probe_error": record.media_probe_error,
                }})
        except Exception as error:                   # recording failure fails the job loudly
            self._fail(record, error)
            return JobResult(record)
        self._advance(record, JobState.SUCCEEDED)
        return JobResult(record)

    def _manifest_media(self, record: JobRecord) -> None:
        """Best-effort manifest refresh so failed jobs still expose metadata."""
        try:
            self._recorder().write_manifest(
                media_metadata={record.job_id: {
                    "metadata": record.media_metadata,
                    "probe_error": record.media_probe_error,
                }})
        except Exception:
            pass                                    # metadata must never mask the real failure

    def _fail(self, record: JobRecord, error: BaseException) -> None:
        record.error_code = map_error_code(_legacy_code_of(error))
        record.error_message = f"{type(error).__name__}: {error}"[:512]
        self._manifest_media(record)               # failed jobs keep their metadata
        self._advance(record, JobState.FAILED)

    def _timeout(self, record: JobRecord) -> None:
        record.error_code = ContractErrorCodes.INFERENCE_TIMEOUT
        record.error_message = "job deadline elapsed before a result was produced"
        self._manifest_media(record)
        self._advance(record, JobState.TIMED_OUT)

    def _cancelled(self, record: JobRecord) -> None:
        if record.state in JobState.TERMINAL:
            return
        self._advance(record, JobState.CANCELLED)
