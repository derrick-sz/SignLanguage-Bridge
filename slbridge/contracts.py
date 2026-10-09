"""Versioned, JSON-serializable objects shared by data, training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import re
from typing import Any, Mapping

from ._io import canonical_hash


def _strict(cls: type, value: Mapping[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    unknown = set(value) - {entry.name for entry in fields(cls)}
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"Invalid {cls.__name__} fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return dict(value)


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset_id: str
    resource: str
    content_hash: str
    label_id: str
    sign_language: str | None
    input_kind: str
    representation_id: str
    source_id: str | None = None
    derived_from: str | None = None
    groups: dict[str, str] = field(default_factory=dict)
    quality_status: str = "ready"
    quality_issues: tuple[str, ...] = ()
    display_text: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sample_id", "dataset_id", "resource", "label_id", "representation_id"):
            _text(getattr(self, name), name)
        _hash(self.content_hash, "content_hash")
        if self.input_kind not in {"npy", "rgb_video", "landmarks_npz"} or self.quality_status not in {"ready", "quarantined"}:
            raise ValueError("Unsupported input kind or quality status")
        for name in ("sign_language", "source_id", "derived_from"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        for name in ("groups", "display_text"):
            mapping = getattr(self, name)
            if not isinstance(mapping, dict):
                raise ValueError(f"{name} must be an object")
            for key, value in mapping.items():
                _text(key, name)
                _text(value, name)
        if not isinstance(self.quality_issues, tuple) or any(not isinstance(x, str) for x in self.quality_issues):
            raise ValueError("quality_issues must be strings")
        if self.quality_status == "quarantined" and not self.quality_issues:
            raise ValueError("Quarantined samples require an explanation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SampleRecord:
        data = _strict(cls, value, ("sample_id", "dataset_id", "resource", "content_hash", "label_id",
                                    "sign_language", "input_kind", "representation_id"))
        data["quality_issues"] = tuple(data.get("quality_issues", ()))
        return cls(**data)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    root: str
    input_kind: str
    representation_id: str
    sign_language: str | None
    samples: tuple[SampleRecord, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        _text(self.root, "root")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported dataset schema version")
        if self.input_kind not in {"npy", "rgb_video", "landmarks_npz"}:
            raise ValueError("Unsupported dataset input kind")
        _text(self.representation_id, "representation_id")
        if self.sign_language is not None:
            _text(self.sign_language, "sign_language")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("Dataset must contain sample records")
        if any(not isinstance(sample, SampleRecord) for sample in self.samples):
            raise ValueError("Dataset samples must be SampleRecord objects")
        if len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate sample_id")
        if len({s.resource for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate resource")
        for sample in self.samples:
            if (sample.dataset_id, sample.input_kind, sample.representation_id, sample.sign_language) != (
                self.dataset_id, self.input_kind, self.representation_id, self.sign_language
            ):
                raise ValueError("Sample does not match its dataset contract")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetManifest:
        data = _strict(cls, value, ("dataset_id", "root", "input_kind", "representation_id", "sign_language",
                                   "samples", "schema_version"))
        data["samples"] = tuple(SampleRecord.from_dict(sample) for sample in data["samples"])
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        content = self.to_dict()
        content.pop("root")  # Relocating byte-identical source data does not change identity.
        return canonical_hash(content)


@dataclass(frozen=True)
class SplitManifest:
    dataset_hash: str
    seed: int
    protocol: str
    group_fields: tuple[str, ...]
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    excluded: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _hash(self.dataset_hash, "dataset_hash")
        _text(self.protocol, "protocol")
        if type(self.seed) is not int or type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Invalid split seed or schema version")
        for name in ("group_fields", "train", "validation", "test", "excluded"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"{name} must be a sequence of identifiers")
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate entry in {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SplitManifest:
        data = _strict(cls, value, tuple(f.name for f in fields(cls)))
        for name in ("group_fields", "train", "validation", "test", "excluded"):
            data[name] = tuple(data[name])
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class InputSpec:
    representation_id: str
    target_frames: int = 30
    dtype: str = "float32"
    pipeline_id: str = "legacy_sequence_v1"
    normalization: str = "preserve"
    temporal_policy: str = "relative_linear"
    missing_policy: str = "reject"
    minimum_valid_fraction: float = 0.1
    version: int = 1
    extractor: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.representation_id not in {"hand63_v1", "pose_hands225_legacy", "mediapipe225_rgb_v1"}:
            raise ValueError(f"Unsupported representation: {self.representation_id}")
        if type(self.target_frames) is not int or not 2 <= self.target_frames <= 1000:
            raise ValueError("target_frames must be an integer between 2 and 1000")
        is_rgb = self.representation_id == "mediapipe225_rgb_v1"
        expected_missing = "reject" if self.representation_id == "hand63_v1" else "sentinel_to_mask"
        if (self.dtype, self.pipeline_id, self.normalization, self.temporal_policy, self.missing_policy) != (
            "float32", "rgb_landmarks_v1" if is_rgb else "legacy_sequence_v1",
            "shoulder_xy_zero_z" if is_rgb else "preserve", "timestamp_linear" if is_rgb else "relative_linear", expected_missing
        ) or type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported input processing contract")
        if is_rgb:
            if not isinstance(self.extractor, dict) or set(self.extractor) != {"config", "identity"}:
                raise ValueError("RGB landmarks require a recorded extractor config and identity")
            if not isinstance(self.extractor["config"], dict) or not isinstance(self.extractor["identity"], dict):
                raise ValueError("Invalid extractor config/identity")
            canonical_hash(self.extractor)
        elif self.extractor is not None:
            raise ValueError("Legacy input must not declare an RGB extractor")
        if type(self.minimum_valid_fraction) not in {int, float} or not math.isfinite(self.minimum_valid_fraction) or not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must lie in (0, 1]")

    @property
    def width(self) -> int:
        return 63 if self.representation_id == "hand63_v1" else 225

    @property
    def feature_size(self) -> int:
        return self.target_frames * self.width * (2 if self.missing_policy == "sentinel_to_mask" else 1)

    @classmethod
    def for_representation(cls, representation_id: str, target_frames: int = 30,
                           extractor: dict[str, Any] | None = None) -> InputSpec:
        return cls(representation_id=representation_id, target_frames=target_frames,
                   missing_policy="reject" if representation_id == "hand63_v1" else "sentinel_to_mask",
                   pipeline_id="rgb_landmarks_v1" if representation_id == "mediapipe225_rgb_v1" else "legacy_sequence_v1",
                   normalization="shoulder_xy_zero_z" if representation_id == "mediapipe225_rgb_v1" else "preserve",
                   temporal_policy="timestamp_linear" if representation_id == "mediapipe225_rgb_v1" else "relative_linear",
                   extractor=extractor)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InputSpec:
        # The first released legacy spec predates the optional extractor field.
        return cls(**_strict(cls, value, tuple(f.name for f in fields(cls) if f.name != "extractor")))


@dataclass(frozen=True)
class PreparedInput:
    spec: InputSpec
    values: Any
    mask: Any
    source_frames: int
    source_ref: str
    quality: dict[str, Any]
    timestamps: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ModelManifest:
    run_id: str
    backend: str
    input_spec: InputSpec
    labels_hash: str
    dataset_hash: str
    split_hash: str
    artifacts: dict[str, str]
    parameters: dict[str, Any]
    dependencies: dict[str, str]
    provenance: dict[str, Any]
    created_at: str
    sign_language: str | None
    task_type: str = "classification"
    capabilities: tuple[str, ...] = ("clip",)
    score_type: str = "uncalibrated_class_probability"
    preprocessing_state: str = "none"
    schema_version: int = 1
    runtime_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.run_id):
            raise ValueError("Invalid run_id")
        from .backends import get_backend
        get_backend(self.backend)
        if self.task_type != "classification":
            raise ValueError("Unsupported model backend/task")
        if not isinstance(self.input_spec, InputSpec):
            raise ValueError("input_spec must be an InputSpec")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported model schema version")
        if self.capabilities != ("clip",) or self.preprocessing_state != "none":
            raise ValueError("Unsupported model capabilities or processing state")
        if self.score_type != "uncalibrated_class_probability":
            raise ValueError("Unsupported score type")
        for name in ("labels_hash", "dataset_hash", "split_hash"):
            _hash(getattr(self, name), name)
        if not isinstance(self.artifacts, dict) or not self.artifacts:
            raise ValueError("Model artifacts must be a nonempty mapping")
        for name, digest in self.artifacts.items():
            _text(name, "artifact")
            _hash(digest, "artifact hash")
        for name in ("parameters", "dependencies", "provenance"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
        if self.runtime_config is not None and not isinstance(self.runtime_config, dict):
            raise ValueError("runtime_config must be an object or null")
        _text(self.created_at, "created_at")
        if self.sign_language is not None:
            _text(self.sign_language, "sign_language")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.runtime_config is None:
            value.pop("runtime_config")  # Preserve released RF manifest JSON.
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelManifest:
        data = _strict(cls, value, tuple(f.name for f in fields(cls) if f.name != "runtime_config"))
        data["input_spec"] = InputSpec.from_dict(data["input_spec"])
        data["capabilities"] = tuple(data["capabilities"])
        return cls(**data)


@dataclass(frozen=True)
class ClassificationPrediction:
    label_id: str
    score: float
    scores: dict[str, float]
    display_text: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GlossToken:
    token_id: int
    gloss: str
    score: float | None
    time_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if type(self.token_id) is not int or self.token_id <= 0:
            raise ValueError("Gloss token_id must be a positive integer; blank is not a gloss")
        _text(self.gloss, "gloss")
        if self.score is not None and (type(self.score) not in {int, float}
                                       or not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("Gloss score must be a finite token probability or null")
        if self.time_range is not None and (not isinstance(self.time_range, tuple) or len(self.time_range) != 2
                                            or any(type(value) not in {int, float} or not math.isfinite(value)
                                                   for value in self.time_range)
                                            or self.time_range[1] < self.time_range[0]):
            raise ValueError("Gloss time_range must contain finite ordered timestamps or null")


@dataclass(frozen=True)
class GlossSequencePrediction:
    """A CTC gloss transcription; no natural-language translation is implied."""

    tokens: tuple[GlossToken, ...]
    decoder: str

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or any(not isinstance(token, GlossToken) for token in self.tokens):
            raise ValueError("Sequence tokens must be a tuple of GlossToken objects")
        if self.decoder not in {"greedy", "prefix_beam"}:
            raise ValueError("Unsupported CTC decoder")

    @property
    def glosses(self) -> tuple[str, ...]:
        return tuple(token.gloss for token in self.tokens)

    @property
    def text(self) -> str:
        return " ".join(self.glosses)


@dataclass(frozen=True)
class RecognitionResult:
    status: str
    sample_id: str
    model_id: str
    sign_language: str | None
    prediction: ClassificationPrediction | GlossSequencePrediction | None
    quality: dict[str, Any]
    reason: str | None = None
    task_type: str = "classification"
    score_type: str = "uncalibrated_class_probability"
    time_range: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
