"""Transport-independent requests and typed model responses.

Deadlines use this process's monotonic clock. Native work is cooperatively
cancelled before/after the call; no unsafe native interruption is attempted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import threading
import time
from typing import Any

from .contracts import RecognitionResult


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValueError(f"{name} must be a 1..128 character identifier")


class InferenceCancelled(RuntimeError):
    """The caller cancelled; native work may already have completed."""


class InferenceDeadlineExceeded(TimeoutError):
    """The caller's monotonic deadline elapsed."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    representation_id: str | None = None
    sign_language: str | None = None
    schema_version: int = 1
    expected_task_type: str | None = None
    session_id: str | None = None
    segment_id: str | None = None
    source_epoch: int | None = None
    expected_generation: int | None = None
    model_id: str | None = None
    preprocessing_version: str | None = None
    deadline_monotonic: float | None = None
    cancellation: CancellationToken | None = field(default=None, repr=False, compare=False)
    expected_mode: str = "clip"

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported request schema_version")
        if self.representation_id is not None:
            from .model_adapters import get_input_representation
            get_input_representation(self.representation_id)
        if self.expected_task_type is not None and self.expected_task_type not in {"classification", "gloss_sequence", "text"}:
            raise ValueError("Unsupported declared task type")
        if self.sign_language is not None and (not isinstance(self.sign_language, str)
                                               or not self.sign_language.strip() or len(self.sign_language) > 64):
            raise ValueError("Invalid declared sign language")
        for name in ("session_id", "segment_id", "model_id", "preprocessing_version"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        for name, minimum in (("source_epoch", 0), ("expected_generation", 1)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.deadline_monotonic is not None and (type(self.deadline_monotonic) not in {int, float}
                                                    or not math.isfinite(self.deadline_monotonic)):
            raise ValueError("deadline_monotonic must be finite")
        if self.cancellation is not None and not isinstance(self.cancellation, CancellationToken):
            raise ValueError("cancellation must be a CancellationToken")
        if self.expected_mode not in {"clip", "stream"}:
            raise ValueError("Unsupported inference mode")

    def check_active(self) -> None:
        if self.cancellation is not None and self.cancellation.cancelled:
            raise InferenceCancelled("Inference was cancelled")
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise InferenceDeadlineExceeded("Inference deadline elapsed")


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    model_generation: int
    result: RecognitionResult
    runtime: dict[str, Any]
    timing_ms: dict[str, float]
    schema_version: int = 1
    session_id: str | None = None
    segment_id: str | None = None
    source_epoch: int | None = None
    model_id: str | None = None
    model_version: str | None = None
    preprocessing_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "request_id": self.request_id,
                "session_id": self.session_id, "segment_id": self.segment_id, "source_epoch": self.source_epoch,
                "model_id": self.model_id or self.result.model_id, "model_version": self.model_version,
                "preprocessing_version": self.preprocessing_version,
                "model_generation": self.model_generation, "result": self.result.to_dict(),
                "runtime": dict(self.runtime), "timing_ms": dict(self.timing_ms)}
