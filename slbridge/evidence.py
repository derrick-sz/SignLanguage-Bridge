"""Product-side standard evidence contract (P0-A phase A0, task T-A0-1).

Single source of truth for the ``EvidenceCandidate`` shape is the frozen P-1
schema ``experiments/p1/schemas/evidence_candidate.schema.json`` (Lead decision
2026-10-26/27).  This module implements that shape as immutable dataclasses with
strict validation; the HTTP-facing API-contract field names are *mapped onto*
it (``model_version`` -> ``model_revision``, ``preprocessing_version`` ->
``preprocess_version``, ``task_type`` -> ``task``) rather than duplicated.

Frozen field-mapping decisions (from_slbridge_response) -- do not change
without Lead review; every JSONL row written through
:mod:`slbridge.prediction_log` bakes these in permanently:

==================  ===========================================  ==========================================
contract field      source (slbridge symbol)                    decision
==================  ===========================================  ==========================================
sample_id           ``InferenceResponse.request_id``            ``sample_id == request_id`` (also equals
                                                                ``RecognitionResult.sample_id`` by
                                                                ``validate_adapter_result``).
attempt             --                                          always ``1`` in P0-A (no retries modelled).
window_id           derived                                     ``f"{request_id}-w1"`` -- the whole-file /
                                                                whole-clip window of the P0-A path.
model_id            ``InferenceResponse.model_id``              direct.
task                ``RecognitionResult.task_type``             ``text -> CSLT``, ``gloss_sequence ->
                                                                CSLR``, ``classification -> ISLR``.
sign_system         ``RecognitionResult.sign_language``         ``csl -> CSL``, ``libras -> Libras``,
                                                                ``None -> Unknown``; anything else is
                                                                rejected (no silent guessing).
modality            caller-provided (e.g. ``RGB_video``,        no slbridge equivalent; required by the
                    ``released_pose``, ``hrnet_npz``)          frozen schema.
model_revision      ``InferenceResponse.model_version``         direct; must be nonempty.
preprocess_version  ``InferenceResponse.preprocessing_version`` direct; must be nonempty (V3.0 sec. 4.2).
window              ``RecognitionResult.time_range``           **seconds (float) -> integer ms**
                                                                (``round(x*1000)``); ``None`` stays
                                                                ``None``; ``start_frame`` is 0 for the
                                                                whole-file window, frame end unknown.
output.kind         prediction payload type                     ``TextPrediction -> TEXT``,
                                                                ``GlossSequencePrediction -> GLOSS``,
                                                                ``ClassificationPrediction ->
                                                                GESTURE_ID``, rejection -> ``EMPTY``.
output.text         ``TextPrediction.text``                     nonempty for TEXT.
output.gloss_tokens ``GlossSequencePrediction.glosses``         nonempty tuple for GLOSS.
output.spoken_lang  ``TextPrediction.language``                 normalized ``zh``/``pt``; null otherwise.
output.gesture_id   ``ClassificationPrediction.label_id``     GESTURE_ID only.
status              ``RecognitionResult.status``                ``recognized -> OK``, ``rejected ->
                                                                FAILED``; timeouts/skips are recorded by
                                                                the job layer (T-A2-2), not here.
quality             raw scores                                  ``raw_score_value``/``raw_score_type``
                                                                mirror the adapter score;
                                                                ``calibrated_reliability`` and
                                                                ``confidence_model_revision`` are
                                                                **always null** in P0-A (ADR-P1-006:
                                                                SLRT raw score only ranks, ECE 0.304;
                                                                Uni-Sign emits no score at all).
==================  ===========================================  ==========================================

GT isolation: the guard below replicates ``experiments/p1/gt_guard.py``
deliberately as an independent copy (the product package must not import the
experiments subpackage); ``tests/test_evidence_contract.py`` asserts both key
sets stay identical so the definitions cannot drift silently.

Out of scope on purpose: "late-arriving output" ordering (V3.0 sec. 13) lives
in the job layer where ordering state exists; this module is stateless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from types import SimpleNamespace
from typing import Any, Iterable

__all__ = [
    "EvidenceCandidate", "EvidenceWindow", "EvidenceOutput", "EvidenceQuality",
    "FORBIDDEN_GT_KEYS", "GroundTruthLeakError", "assert_gt_clean", "find_gt_keys",
    "from_slbridge_response", "normalize_sign_system", "task_from_task_type",
]

_SCHEMA_VERSION = "1.0"
_METRIC_DEFINITION_VERSION = "p1-v1"

_SIGN_LANGUAGE_MAP = {"csl": "CSL", "libras": "Libras", "unknown": "Unknown"}
_TASK_MAP = {"text": "CSLT", "gloss_sequence": "CSLR", "classification": "ISLR"}
_STATUS_MAP = {"recognized": "OK", "rejected": "FAILED"}
_SPOKEN_LANGUAGE_MAP = {"zh": "zh", "zh-cn": "zh", "chinese": "zh",
                        "pt": "pt", "pt-br": "pt", "portuguese": "pt"}

# GT guard -- replicated from experiments/p1/gt_guard.py FORBIDDEN_GT_KEYS
# (do not import the experiments package from product code).  The key set must
# stay byte-identical; tests/test_evidence_contract.py enforces that.
FORBIDDEN_GT_KEYS: frozenset[str] = frozenset(
    {
        "gt",
        "gt_sentence",
        "gt_sentence_official",
        "gt_gloss",
        "gt_gloss_tokens",
        "ground_truth",
        "reference",
        "references",
        "reference_tokens",
        "ref_sentence",
        "ref_sentence_upstream",
        "ref_upstream",
        "slrt_wer",
        "wer",
        "unisign_edit",
        "unisign_video_edit",
        "unisign_pose_edit",
        "edit_similarity",
        "char_edit_similarity",
        "mean_char_edit_similarity",
        "rouge_l_f1_char",
        "exact_match",
        "diagnostic_class",
        "cohort_class",
    }
)


class GroundTruthLeakError(ValueError):
    """Raised when a payload that must stay GT-free contains a GT-derived key."""


def _walk(node: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else str(key)
            yield key_path, str(key)
            yield from _walk(value, key_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def find_gt_keys(payload: Any) -> list[str]:
    """Return dotted paths of every forbidden key found in ``payload``."""

    found: list[str] = []
    for path, key in _walk(payload, ""):
        if key.lower() in FORBIDDEN_GT_KEYS:
            found.append(path)
    return found


def assert_gt_clean(payload: Any, *, context: str = "payload") -> None:
    """Raise :class:`GroundTruthLeakError` if ``payload`` carries GT keys."""

    leaks = find_gt_keys(payload)
    if leaks:
        raise GroundTruthLeakError(
            f"{context} carries ground-truth derived keys: {', '.join(sorted(leaks))}"
        )


def normalize_sign_system(sign_language: str | None) -> str:
    """Map a slbridge ``sign_language`` to the frozen schema enum."""

    if sign_language is None:
        return "Unknown"
    if not isinstance(sign_language, str):
        raise ValueError("sign_language must be a string or null")
    mapped = _SIGN_LANGUAGE_MAP.get(sign_language.strip().lower())
    if mapped is None:
        raise ValueError(f"Unsupported sign language for evidence: {sign_language!r}")
    return mapped


def task_from_task_type(task_type: str) -> str:
    """Map a slbridge ``task_type`` to the frozen schema task enum."""

    if not isinstance(task_type, str):
        raise ValueError("task_type must be a string")
    mapped = _TASK_MAP.get(task_type)
    if mapped is None:
        raise ValueError(f"Unsupported task type for evidence: {task_type!r}")
    return mapped


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValueError(f"{name} must be a 1..128 character identifier")


def _milliseconds(seconds: Any, name: str) -> int | None:
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    value = int(round(float(seconds) * 1000))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class EvidenceWindow:
    """Whole-file/whole-clip window of the P0-A path (frozen schema ``window``)."""

    start_frame: int | None = 0
    end_frame_exclusive: int | None = None
    start_pts_ms: int | None = None
    end_pts_ms: int | None = None
    needs_future_frames: bool | None = None

    def __post_init__(self) -> None:
        for name in ("start_frame", "end_frame_exclusive"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or null")
        if self.end_frame_exclusive == 0:
            raise ValueError("end_frame_exclusive must be >= 1 when present")
        for name in ("start_pts_ms", "end_pts_ms"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer (ms) or null")
        if self.start_pts_ms is not None and self.end_pts_ms is not None \
                and self.end_pts_ms < self.start_pts_ms:
            raise ValueError("end_pts_ms must not precede start_pts_ms")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceWindow":
        if not isinstance(value, dict):
            raise ValueError("window must be a JSON object")
        unknown = set(value) - set(EvidenceWindow.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown evidence window fields: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class EvidenceOutput:
    """Typed model output (frozen schema ``output``); kind and payload agree."""

    kind: str
    gloss_tokens: tuple[str, ...] | None
    text: str | None
    spoken_language: str | None = None
    gesture_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"GLOSS", "TEXT", "GESTURE_ID", "EMPTY"}:
            raise ValueError(f"Unsupported evidence output kind: {self.kind!r}")
        if self.kind == "TEXT":
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("TEXT evidence requires nonempty text")
            if self.gloss_tokens is not None or self.gesture_id is not None:
                raise ValueError("TEXT evidence must not carry gloss tokens or gesture id")
        elif self.kind == "GLOSS":
            if (not isinstance(self.gloss_tokens, (tuple, list)) or not self.gloss_tokens
                    or any(not isinstance(token, str) or not token.strip() for token in self.gloss_tokens)):
                raise ValueError("GLOSS evidence requires nonempty gloss tokens")
            if isinstance(self.gloss_tokens, list):
                object.__setattr__(self, "gloss_tokens", tuple(self.gloss_tokens))
            if self.text is not None or self.gesture_id is not None:
                raise ValueError("GLOSS evidence must not carry text or gesture id")
        elif self.kind == "GESTURE_ID":
            if not isinstance(self.gesture_id, str) or not self.gesture_id.strip():
                raise ValueError("GESTURE_ID evidence requires a nonempty gesture id")
            if self.text is not None or self.gloss_tokens is not None:
                raise ValueError("GESTURE_ID evidence must not carry text or gloss tokens")
        else:  # EMPTY
            if self.text is not None or self.gloss_tokens is not None or self.gesture_id is not None:
                raise ValueError("EMPTY evidence must not carry any payload")
        if self.spoken_language is not None and self.spoken_language not in {"zh", "pt"}:
            raise ValueError("spoken_language must be 'zh', 'pt' or null")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gloss_tokens"] = list(self.gloss_tokens) if self.gloss_tokens is not None else None
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceOutput":
        if not isinstance(value, dict):
            raise ValueError("output must be a JSON object")
        unknown = set(value) - set(EvidenceOutput.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown evidence output fields: {sorted(unknown)}")
        gloss = value.get("gloss_tokens")
        if gloss is not None:
            if not isinstance(gloss, list):
                raise ValueError("gloss_tokens must be a list or null")
            gloss = tuple(gloss)
        return cls(kind=value.get("kind"), gloss_tokens=gloss, text=value.get("text"),
                   spoken_language=value.get("spoken_language"), gesture_id=value.get("gesture_id"))


@dataclass(frozen=True)
class EvidenceQuality:
    """Score transparency block (frozen schema ``quality``).

    ``calibrated_reliability`` is always null in P0-A (ADR-P1-006): raw scores
    are uncalibrated and must never masquerade as translation correctness.
    """

    raw_score_value: float | None = None
    raw_score_type: str | None = None
    calibrated_reliability: float | None = None
    confidence_model_revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("raw_score_value", "calibrated_reliability"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{name} must be a number or null")
        if self.calibrated_reliability is not None and not 0 <= self.calibrated_reliability <= 1:
            raise ValueError("calibrated_reliability must lie in [0, 1]")
        for name in ("raw_score_type", "confidence_model_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceQuality":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("quality must be a JSON object or null")
        unknown = set(value) - set(EvidenceQuality.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown evidence quality fields: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class EvidenceCandidate:
    """Standard model evidence, aligned with the frozen P-1 schema.

    Required schema fields are non-defaulted here so construction fails loudly;
    ``attempt``/``window_id``/``quality`` stay optional exactly like the schema.
    """

    run_id: str
    sample_id: str
    model_id: str
    task: str
    sign_system: str
    modality: str
    model_revision: str
    preprocess_version: str
    window: EvidenceWindow
    output: EvidenceOutput
    status: str
    attempt: int = 1
    window_id: str | None = None
    quality: EvidenceQuality = field(default_factory=EvidenceQuality)

    def __post_init__(self) -> None:
        for name in ("run_id", "sample_id", "model_id", "modality",
                     "model_revision", "preprocess_version"):
            _text(getattr(self, name), name)
        _identifier(self.sample_id, "sample_id")
        if self.task not in {"CSLR", "CSLT", "ISLR"}:
            raise ValueError(f"Unsupported evidence task: {self.task!r}")
        if self.sign_system not in {"CSL", "Libras", "Unknown"}:
            raise ValueError(f"Unsupported evidence sign_system: {self.sign_system!r}")
        if self.status not in {"OK", "FAILED", "TIMEOUT", "SKIPPED", "BLOCKED"}:
            raise ValueError(f"Unsupported evidence status: {self.status!r}")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be an integer >= 1")
        if self.window_id is not None and (not isinstance(self.window_id, str) or not self.window_id.strip()):
            raise ValueError("window_id must be a nonempty string or null")
        if not isinstance(self.window, EvidenceWindow):
            if isinstance(self.window, dict):
                object.__setattr__(self, "window", EvidenceWindow.from_dict(self.window))
            else:
                raise ValueError("window must be an EvidenceWindow or a JSON object")
        if not isinstance(self.output, EvidenceOutput):
            if isinstance(self.output, dict):
                object.__setattr__(self, "output", EvidenceOutput.from_dict(self.output))
            else:
                raise ValueError("output must be an EvidenceOutput or a JSON object")
        if not isinstance(self.quality, EvidenceQuality):
            if isinstance(self.quality, dict):
                object.__setattr__(self, "quality", EvidenceQuality.from_dict(self.quality))
            else:
                raise ValueError("quality must be an EvidenceQuality or a JSON object")
        if self.status == "OK" and self.output.kind == "EMPTY":
            raise ValueError("OK evidence must not carry an EMPTY output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "attempt": self.attempt,
            "window_id": self.window_id,
            "model_id": self.model_id,
            "task": self.task,
            "sign_system": self.sign_system,
            "modality": self.modality,
            "model_revision": self.model_revision,
            "preprocess_version": self.preprocess_version,
            "window": self.window.to_dict(),
            "output": self.output.to_dict(),
            "status": self.status,
            "quality": self.quality.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceCandidate":
        if not isinstance(value, dict):
            raise ValueError("EvidenceCandidate must be a JSON object")
        if value.get("schema_version") not in {None, _SCHEMA_VERSION}:
            raise ValueError(f"Unsupported evidence schema_version: {value.get('schema_version')!r}")
        unknown = set(value) - set(EvidenceCandidate.__dataclass_fields__) - {"schema_version"}
        if unknown:
            raise ValueError(f"Unknown evidence fields: {sorted(unknown)}")
        data = {key: item for key, item in value.items() if key != "schema_version"}
        return cls(
            run_id=data["run_id"], sample_id=data["sample_id"], model_id=data["model_id"],
            task=data["task"], sign_system=data["sign_system"], modality=data["modality"],
            model_revision=data["model_revision"], preprocess_version=data["preprocess_version"],
            window=EvidenceWindow.from_dict(data["window"]),
            output=EvidenceOutput.from_dict(data["output"]),
            status=data["status"],
            attempt=data.get("attempt", 1), window_id=data.get("window_id"),
            quality=EvidenceQuality.from_dict(data.get("quality")))


def _spoken_language(language: str | None) -> str | None:
    if language is None:
        return None
    if not isinstance(language, str):
        raise ValueError("prediction language must be a string or null")
    mapped = _SPOKEN_LANGUAGE_MAP.get(language.strip().lower())
    if mapped is None:
        raise ValueError(f"Unsupported spoken language for evidence: {language!r}")
    return mapped


def from_slbridge_response(response: Any, *, run_id: str, modality: str) -> EvidenceCandidate:
    """Convert a slbridge ``InferenceResponse`` into standard evidence.

    ``response`` may be a real :class:`slbridge.runtime_contracts.InferenceResponse`
    or any object with the same attributes (tests inject stand-ins).  The
    mapping decisions are frozen in the module docstring; unit conversion of
    ``time_range`` seconds to integer milliseconds happens here and nowhere else.
    """

    _text(run_id, "run_id")
    _text(modality, "modality")
    result = getattr(response, "result", None)
    if result is None:
        raise ValueError("response.result is required")
    request_id = getattr(response, "request_id", None)
    model_id = getattr(response, "model_id", None) or getattr(result, "model_id", None)
    model_revision = getattr(response, "model_version", None)
    preprocess_version = getattr(response, "preprocessing_version", None)
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("response.request_id is required")
    _text(model_id, "response.model_id")
    _text(model_revision, "response.model_version")
    _text(preprocess_version, "response.preprocessing_version")

    status = getattr(result, "status", None)
    if status not in _STATUS_MAP:
        raise ValueError(f"Unsupported recognition status for evidence: {status!r}")
    evidence_status = _STATUS_MAP[status]
    prediction = getattr(result, "prediction", None)
    reason = getattr(result, "reason", None)
    score_type = getattr(result, "score_type", None)
    raw_score = None

    if prediction is None:
        if status == "recognized":
            raise ValueError("recognized results require a typed prediction")
        output = EvidenceOutput(kind="EMPTY", gloss_tokens=None, text=None,
                                spoken_language=None, gesture_id=None)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rejected results require a nonempty reason")
    elif hasattr(prediction, "glosses") and hasattr(prediction, "tokens"):
        # GlossSequencePrediction (its `.text` property would shadow this branch).
        tokens = prediction.tokens or ()
        glosses = tuple(getattr(token, "gloss", "") for token in tokens)
        if not glosses or any(not gloss.strip() for gloss in glosses):
            raise ValueError("GLOSS evidence requires nonempty gloss tokens")
        output = EvidenceOutput(kind="GLOSS", gloss_tokens=glosses, text=None,
                                spoken_language=None)
        scores = [getattr(token, "score", None) for token in tokens]
        present = [score for score in scores if score is not None]
        raw_score = (sum(present) / len(present)) if present else None
    elif hasattr(prediction, "text") and hasattr(prediction, "language"):
        text = getattr(prediction, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("TEXT evidence requires nonempty text")
        output = EvidenceOutput(kind="TEXT", gloss_tokens=None, text=text,
                                spoken_language=_spoken_language(getattr(prediction, "language", None)))
        raw_score = getattr(prediction, "score", None)
    else:  # ClassificationPrediction-like
        output = EvidenceOutput(kind="GESTURE_ID", gloss_tokens=None, text=None,
                                spoken_language=None, gesture_id=getattr(prediction, "label_id", None))
        raw_score = getattr(prediction, "score", None)

    time_range = getattr(result, "time_range", None)
    start_ms = end_ms = None
    if time_range is not None:
        if (not isinstance(time_range, (tuple, list)) or len(time_range) != 2):
            raise ValueError("time_range must be a pair of seconds or null")
        start_ms = _milliseconds(time_range[0], "time_range start")
        end_ms = _milliseconds(time_range[1], "time_range end")
        if start_ms is not None and end_ms is not None and end_ms < start_ms:
            raise ValueError("time_range end must not precede start")

    quality = EvidenceQuality(
        raw_score_value=(float(raw_score) if isinstance(raw_score, (int, float))
                         and not isinstance(raw_score, bool) else None),
        raw_score_type=(score_type if isinstance(score_type, str) and score_type.strip() else None),
        calibrated_reliability=None,
        confidence_model_revision=None,
    )
    return EvidenceCandidate(
        run_id=run_id,
        sample_id=request_id,
        model_id=model_id,
        task=task_from_task_type(getattr(result, "task_type", None)),
        sign_system=normalize_sign_system(getattr(result, "sign_language", None)),
        modality=modality,
        model_revision=model_revision,
        preprocess_version=preprocess_version,
        window=EvidenceWindow(start_frame=0, end_frame_exclusive=None,
                              start_pts_ms=start_ms, end_pts_ms=end_ms, needs_future_frames=None),
        output=output,
        status=evidence_status,
        attempt=1,
        window_id=f"{request_id}-w1",
        quality=quality,
    )
