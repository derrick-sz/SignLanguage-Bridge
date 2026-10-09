"""Public v1 HTTP schemas; paths and model artifacts are never API inputs."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(ApiModel):
    code: str
    message: str


class ErrorResponse(ApiModel):
    schema_version: Literal[1] = 1
    request_id: str
    error: ErrorDetail


class ModelDescriptor(ApiModel):
    model_id: str
    generation: int
    backend: str
    sign_language: str | None
    input_spec: dict[str, Any] | None
    capabilities: list[str]
    labels: dict[str, dict[str, str]]
    runtime: dict[str, Any]
    options: dict[str, Any] = Field(default_factory=dict)
    warmup_ms: float = 0
    task_type: str = "classification"
    score_type: str = "uncalibrated_class_probability"
    vocabulary_size: int | None = None


class RuntimeStatus(ApiModel):
    state: str
    generation: int
    active: ModelDescriptor | None
    previous: ModelDescriptor | None
    error: ErrorDetail | None


class StatusResponse(ApiModel):
    schema_version: Literal[1] = 1
    request_id: str
    status: RuntimeStatus


class CatalogEntry(ApiModel):
    model_id: str


class ModelsResponse(ApiModel):
    schema_version: Literal[1] = 1
    request_id: str
    models: list[CatalogEntry]
    active: ModelDescriptor | None


class SwitchRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    device: Literal["auto", "cpu", "cuda"] | None = None
    device_id: int | None = Field(default=None, ge=0, le=63, strict=True)
    intra_op_threads: int | None = Field(default=None, ge=1, le=64, strict=True)
    gpu_mem_limit_mb: int | None = Field(default=None, ge=64, le=1048576, strict=True)


class ClassificationPrediction(ApiModel):
    label_id: str
    score: float
    scores: dict[str, float]
    display_text: dict[str, str]


class RecognitionResult(ApiModel):
    status: str
    sample_id: str
    model_id: str
    sign_language: str | None
    prediction: ClassificationPrediction | None
    quality: dict[str, Any]
    reason: str | None
    task_type: str
    score_type: str
    time_range: tuple[float, float] | None


class RecognitionResponse(ApiModel):
    schema_version: Literal[1] = 1
    request_id: str
    model_generation: int
    result: RecognitionResult
    runtime: dict[str, Any]
    timing_ms: dict[str, float]


class GlossToken(ApiModel):
    token_id: int
    gloss: str
    score: float | None
    time_range: tuple[float, float] | None


class GlossSequencePrediction(ApiModel):
    tokens: list[GlossToken]
    decoder: str


class SequenceRecognitionResult(ApiModel):
    status: str
    sample_id: str
    model_id: str
    sign_language: str | None
    prediction: GlossSequencePrediction | None
    quality: dict[str, Any]
    reason: str | None
    task_type: Literal["gloss_sequence"]
    score_type: Literal["ctc_token_probability"]
    time_range: tuple[float, float] | None


class SequenceRecognitionResponse(ApiModel):
    schema_version: Literal[2] = 2
    request_id: str
    model_generation: int
    result: SequenceRecognitionResult
    runtime: dict[str, Any]
    timing_ms: dict[str, float]
