"""Transport-independent v1 request and response envelopes."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .contracts import RecognitionResult


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    representation_id: str | None = None
    sign_language: str | None = None
    schema_version: int = 1
    expected_task_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.request_id):
            raise ValueError("request_id must be a 1..128 character identifier")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported request schema_version")
        if self.representation_id is not None and self.representation_id not in {
            "hand63_v1", "pose_hands225_legacy", "mediapipe225_rgb_v1", "tfnet_rgb_v1"
        }:
            raise ValueError("Unsupported declared representation")
        if self.expected_task_type is not None and self.expected_task_type not in {"classification", "gloss_sequence"}:
            raise ValueError("Unsupported declared task type")
        if self.sign_language is not None and (not isinstance(self.sign_language, str)
                                               or not self.sign_language.strip() or len(self.sign_language) > 64):
            raise ValueError("Invalid declared sign language")


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    model_generation: int
    result: RecognitionResult
    runtime: dict[str, Any]
    timing_ms: dict[str, float]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "request_id": self.request_id,
                "model_generation": self.model_generation, "result": self.result.to_dict(),
                "runtime": dict(self.runtime), "timing_ms": dict(self.timing_ms)}
