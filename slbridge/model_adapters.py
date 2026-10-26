"""Extensible model invocation, with strict representation identities.

The built-in adapters wrap the actual RF/ONNX recognizers and keep their saved
preprocessing. Registering another adapter never implies converting 225-column
features into an unrelated 321-column representation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import threading
from time import perf_counter
from typing import Any, Callable, Protocol, runtime_checkable

from ._io import canonical_hash, read_json
from .contracts import ClassificationPrediction, GlossSequencePrediction, RecognitionResult, TextPrediction
from .runtime_contracts import InferenceRequest, _identifier
from .runtime_options import RuntimeOptions


def _shape(value: Any) -> None:
    if (not isinstance(value, tuple) or not value
            or any(dim is not None and (type(dim) is not int or dim <= 0) for dim in value)):
        raise ValueError("shape must be a nonempty tuple of positive dimensions or null")


@dataclass(frozen=True)
class InputRepresentation:
    representation_id: str
    shape: tuple[int | None, ...]
    dtype: str = "float32"
    layout: str = "T,F"

    def __post_init__(self) -> None:
        _identifier(self.representation_id, "representation_id")
        _shape(self.shape)
        if any(not isinstance(value, str) or not value.strip() for value in (self.dtype, self.layout)):
            raise ValueError("dtype and layout must be nonempty")


_REPRESENTATIONS: dict[str, InputRepresentation] = {}
_REGISTRY_LOCK = threading.RLock()


def register_input_representation(definition: InputRepresentation) -> None:
    if not isinstance(definition, InputRepresentation):
        raise ValueError("An InputRepresentation is required")
    with _REGISTRY_LOCK:
        if definition.representation_id in _REPRESENTATIONS:
            raise ValueError(f"Representation already registered: {definition.representation_id}")
        _REPRESENTATIONS[definition.representation_id] = definition


def get_input_representation(representation_id: str) -> InputRepresentation:
    with _REGISTRY_LOCK:
        try:
            return _REPRESENTATIONS[representation_id]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unsupported declared representation: {representation_id}") from error


for _name, _width in (("hand63_v1", 63), ("pose_hands225_legacy", 225), ("mediapipe225_rgb_v1", 225)):
    register_input_representation(InputRepresentation(_name, (None, _width)))
register_input_representation(InputRepresentation("tfnet_rgb_v1", (1, None, 3, 224, 224), layout="B,L,C,H,W"))


@dataclass(frozen=True)
class ModelInputContract:
    representation_id: str
    shape: tuple[int | None, ...]
    dtype: str
    preprocessing_version: str
    input_kinds: tuple[str, ...] = ("file", "captured_clip", "prepared")

    def __post_init__(self) -> None:
        representation = get_input_representation(self.representation_id)
        _shape(self.shape)
        if (len(self.shape) != len(representation.shape) or self.dtype != representation.dtype
                or any(base is not None and dim != base for base, dim in zip(representation.shape, self.shape))):
            raise ValueError("Input shape/dtype disagree with registered representation")
        _identifier(self.preprocessing_version, "preprocessing_version")
        if (not isinstance(self.input_kinds, tuple) or not self.input_kinds
                or len(set(self.input_kinds)) != len(self.input_kinds)
                or any(kind not in {"file", "captured_clip", "prepared"} for kind in self.input_kinds)):
            raise ValueError("Invalid input kinds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_prepared(self, source: Any, request: InferenceRequest) -> None:
        """Bind serialized specs by full hash, objects by their own version tag.

        Only raw arrays use the request's preprocessing declaration as their
        boundary. A request cannot override a prepared object's actual spec.
        """
        import numpy as np

        values = source if isinstance(source, np.ndarray) else getattr(source, "values", None)
        spec = getattr(source, "spec", None)
        representation = getattr(spec, "representation_id", request.representation_id)
        if representation != self.representation_id:
            raise ValueError("Prepared input representation is incompatible with the model")
        if isinstance(source, np.ndarray):
            preprocessing = request.preprocessing_version
        elif spec is not None and callable(getattr(spec, "to_dict", None)):
            preprocessing = canonical_hash(spec.to_dict())
        else:
            preprocessing = getattr(source, "preprocessing_version", None)
        if preprocessing != self.preprocessing_version:
            raise ValueError("Prepared inputs require a matching preprocessing_version")
        if (not isinstance(values, np.ndarray) or str(values.dtype) != self.dtype
                or values.ndim != len(self.shape)
                or any(dim is not None and values.shape[index] != dim for index, dim in enumerate(self.shape))
                or any(dim == 0 for dim in values.shape) or not np.isfinite(values).all()):
            raise ValueError("Prepared input shape/dtype/finite values violate the model contract")


@dataclass(frozen=True)
class ModelCapabilities:
    model_id: str
    model_version: str
    sign_language: str | None
    task_type: str
    inputs: tuple[ModelInputContract, ...]
    vocabulary: tuple[str, ...]
    runtime: str
    devices: tuple[str, ...]
    modes: tuple[str, ...] = ("clip",)
    score_type: str = "uncalibrated_class_probability"

    def __post_init__(self) -> None:
        for name in ("model_id", "model_version", "runtime", "score_type"):
            _identifier(getattr(self, name), name)
        if self.sign_language is not None and (not isinstance(self.sign_language, str) or not self.sign_language.strip()):
            raise ValueError("Invalid sign language")
        if self.task_type not in {"classification", "gloss_sequence", "text"}:
            raise ValueError("Unsupported declared task type")
        if (not isinstance(self.inputs, tuple) or not self.inputs
                or any(not isinstance(entry, ModelInputContract) for entry in self.inputs)
                or len({entry.representation_id for entry in self.inputs}) != len(self.inputs)):
            raise ValueError("Capabilities require unique typed input contracts")
        if (not isinstance(self.vocabulary, tuple)
                or (self.task_type == "classification" and len(set(self.vocabulary)) != len(self.vocabulary))
                or any(not isinstance(token, str) or not token for token in self.vocabulary)):
            raise ValueError("Vocabulary must contain nonempty strings and unique classification labels")
        for name, allowed in (("devices", {"cpu", "cuda"}), ("modes", {"clip", "stream"})):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values or len(set(values)) != len(values)
                    or any(value not in allowed for value in values)):
                raise ValueError(f"Invalid {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def negotiate(self, request: InferenceRequest, *, input_kind: str) -> ModelInputContract:
        if request.model_id is not None and request.model_id != self.model_id:
            raise ValueError("Declared model_id is incompatible with the model")
        if request.sign_language is not None and request.sign_language != self.sign_language:
            raise ValueError("Declared sign language is incompatible with the model")
        if request.expected_task_type is not None and request.expected_task_type != self.task_type:
            raise ValueError("Declared task type is incompatible with the model")
        if request.expected_mode not in self.modes:
            raise ValueError("Declared inference mode is incompatible with the model")
        matches = [entry for entry in self.inputs if input_kind in entry.input_kinds
                   and (request.representation_id is None or request.representation_id == entry.representation_id)]
        if len(matches) != 1:
            raise ValueError("Declared input representation/kind is incompatible or ambiguous")
        contract = matches[0]
        if request.preprocessing_version is not None and request.preprocessing_version != contract.preprocessing_version:
            raise ValueError("Declared preprocessing_version is incompatible with the model")
        return contract


@runtime_checkable
class ModelAdapter(Protocol):
    capabilities: ModelCapabilities

    @property
    def runtime_info(self) -> dict[str, Any]: ...
    @property
    def recognizer(self) -> Any: ...
    def warmup(self) -> None: ...
    def infer(self, source: Any, request: InferenceRequest, *, input_kind: str) -> tuple[RecognitionResult, dict[str, float]]: ...
    def close(self) -> None: ...


def validate_adapter_result(result: Any, capabilities: ModelCapabilities, request: InferenceRequest) -> None:
    if not isinstance(result, RecognitionResult):
        raise ValueError("Adapter must return a typed RecognitionResult")
    if (result.model_id != capabilities.model_id or result.sample_id != request.request_id
            or result.sign_language != capabilities.sign_language or result.task_type != capabilities.task_type
            or result.score_type != capabilities.score_type or result.status not in {"recognized", "rejected"}):
        raise ValueError("Adapter result identity/task/status disagree with its capabilities")
    expected = {"classification": ClassificationPrediction, "gloss_sequence": GlossSequencePrediction, "text": TextPrediction}
    if result.prediction is not None and not isinstance(result.prediction, expected[result.task_type]):
        raise ValueError("Adapter prediction payload disagrees with its typed task")
    if result.status == "recognized" and result.prediction is None:
        raise ValueError("Recognized results require a typed prediction")
    if result.status == "rejected":
        if not isinstance(result.reason, str) or not result.reason.strip():
            raise ValueError("Rejected results require a nonempty reason")
        # Existing CTC all-blank rejection carries an empty sequence diagnostic.
        # A rejection must never carry an accepted label, text or gloss token.
        if result.prediction is not None and not (isinstance(result.prediction, GlossSequencePrediction)
                                                 and not result.prediction.tokens):
            raise ValueError("Rejected results cannot carry a recognized prediction")
    if not isinstance(result.quality, dict):
        raise ValueError("Adapter quality must be a JSON object")
    try:
        canonical_hash(result.quality)
    except (TypeError, ValueError) as error:
        raise ValueError("Adapter quality must contain finite JSON values") from error
    if result.time_range is not None and (not isinstance(result.time_range, tuple) or len(result.time_range) != 2
                                         or any(type(value) not in {int, float} or not math.isfinite(value)
                                                for value in result.time_range)
                                         or result.time_range[1] < result.time_range[0]):
        raise ValueError("Adapter time_range must contain finite ordered timestamps")
    if isinstance(result.prediction, ClassificationPrediction):
        prediction = result.prediction
        probability = capabilities.score_type in {"uncalibrated_class_probability", "class_probability"}
        if (prediction.label_id not in capabilities.vocabulary or set(prediction.scores) != set(capabilities.vocabulary)
                or any(type(value) not in {int, float} or not math.isfinite(value)
                       or (probability and not 0 <= value <= 1)
                       for value in (*prediction.scores.values(), prediction.score))
                or prediction.score != prediction.scores[prediction.label_id]):
            raise ValueError("Classification payload violates the declared vocabulary/scores")
    if (isinstance(result.prediction, TextPrediction) and result.prediction.score is not None
            and capabilities.score_type in {"text_probability", "calibrated_text_probability"}
            and not 0 <= result.prediction.score <= 1):
        raise ValueError("Text probability must lie in [0, 1]")
    if isinstance(result.prediction, GlossSequencePrediction):
        for token in result.prediction.tokens:
            if token.token_id >= len(capabilities.vocabulary) or capabilities.vocabulary[token.token_id] != token.gloss:
                raise ValueError("Gloss payload violates the ordered vocabulary")


class RecognizerAdapter:
    """A real recognizer wrapper; does not reimplement or mock model inference."""
    def __init__(self, recognizer: Any) -> None:
        self._recognizer = recognizer
        manifest = recognizer.bundle.manifest
        spec = manifest.input_spec
        backend = manifest.backend
        is_rgb = spec.representation_id == "tfnet_rgb_v1"
        shape = (1, None, 3, 224, 224) if is_rgb else (spec.target_frames, spec.width)
        # Hash the complete frozen processing, including extractor identity and
        # bounds; pipeline names alone do not distinguish preprocessing changes.
        preprocessing = canonical_hash(spec.to_dict())
        version = canonical_hash(manifest.to_dict()) if hasattr(manifest, "to_dict") else "injected-v1"
        task_type = getattr(manifest, "task_type", "classification")
        vocabulary = (tuple(recognizer.bundle.vocabulary) if task_type == "gloss_sequence"
                      else tuple(recognizer.bundle.labels))
        self.capabilities = ModelCapabilities(
            manifest.run_id, version, manifest.sign_language, task_type,
            (ModelInputContract(spec.representation_id, shape, spec.dtype, preprocessing,
                                ("file", "captured_clip", "prepared") if spec.representation_id in
                                {"mediapipe225_rgb_v1", "tfnet_rgb_v1"} else ("file", "prepared")),), vocabulary,
            "sklearn" if backend == "random_forest_v1" else "onnxruntime",
            ("cpu",) if backend == "random_forest_v1" else ("cpu", "cuda"),
            tuple(manifest.capabilities), getattr(manifest, "score_type", "uncalibrated_class_probability"))

    @property
    def recognizer(self) -> Any:
        return self._recognizer

    @property
    def runtime_info(self) -> dict[str, Any]:
        return deepcopy(self._recognizer.runtime_info)

    def warmup(self) -> None:
        warmup = getattr(self._recognizer, "warmup", None)
        if warmup is not None:
            warmup()

    def close(self) -> None:
        close = getattr(self._recognizer, "close", None)
        if close is not None:
            close()

    def infer(self, source: Any, request: InferenceRequest, *, input_kind: str) -> tuple[RecognitionResult, dict[str, float]]:
        request.check_active()
        if input_kind == "file":
            output = self._recognizer.recognize_profiled(source, sample_id=request.request_id,
                                                          representation_id=request.representation_id)
        elif input_kind == "captured_clip":
            output = self._recognizer.recognize_clip_profiled(source, sample_id=request.request_id)
        elif input_kind == "prepared":
            started = perf_counter()
            result = self._recognizer.recognize_prepared(source, sample_id=request.request_id)
            elapsed = (perf_counter() - started) * 1000
            output = result, {"preprocess_ms": 0., "inference_ms": elapsed, "total_ms": elapsed}
        else:
            raise ValueError("Unsupported input kind")
        request.check_active()
        return output


@dataclass(frozen=True)
class AdapterDefinition:
    backend_id: str
    load: Callable[[Path, RuntimeOptions], ModelAdapter]
    wrap: Callable[[Any], ModelAdapter] | None = None


_ADAPTERS: dict[str, AdapterDefinition] = {}


def register_model_adapter(definition: AdapterDefinition) -> None:
    if (not isinstance(definition, AdapterDefinition) or not callable(definition.load)
            or (definition.wrap is not None and not callable(definition.wrap))):
        raise ValueError("A concrete AdapterDefinition is required")
    _identifier(definition.backend_id, "backend_id")
    with _REGISTRY_LOCK:
        if definition.backend_id in _ADAPTERS:
            raise ValueError(f"Model adapter already registered: {definition.backend_id}")
        _ADAPTERS[definition.backend_id] = definition


def model_adapter_ids() -> tuple[str, ...]:
    with _REGISTRY_LOCK:
        return tuple(_ADAPTERS)


def _validated_adapter(adapter: Any) -> ModelAdapter:
    if not isinstance(adapter, ModelAdapter) or not isinstance(adapter.capabilities, ModelCapabilities):
        raise ValueError("Model adapter does not implement the typed protocol")
    return adapter


def adapt_recognizer(model: Any) -> ModelAdapter:
    if isinstance(model, ModelAdapter):
        return _validated_adapter(model)
    backend = getattr(getattr(getattr(model, "bundle", None), "manifest", None), "backend", None)
    with _REGISTRY_LOCK:
        definition = _ADAPTERS.get(backend)
    if definition is None or definition.wrap is None:
        raise ValueError(f"No registered adapter for recognizer backend: {backend}")
    return _validated_adapter(definition.wrap(model))


def load_model_adapter(path: str | Path, *, options: RuntimeOptions | None = None) -> ModelAdapter:
    path = Path(path).resolve()
    raw = read_json(path / "manifest.json")
    backend = raw.get("backend") if isinstance(raw, dict) else None
    with _REGISTRY_LOCK:
        definition = _ADAPTERS.get(backend)
    if definition is None:
        raise ValueError(f"Unsupported model adapter backend: {backend}")
    chosen = options or RuntimeOptions()
    if not isinstance(chosen, RuntimeOptions):
        raise ValueError("options must be RuntimeOptions")
    adapter = definition.load(path, chosen)
    try:
        return _validated_adapter(adapter)
    except BaseException:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()
        raise


def _load_classifier(path: Path, options: RuntimeOptions) -> ModelAdapter:
    from .bundle import load_bundle
    from .inference import Recognizer
    recognizer = Recognizer(load_bundle(path, options=options))
    try:
        return RecognizerAdapter(recognizer)
    except BaseException:
        recognizer.close()
        raise


def _load_ctc(path: Path, options: RuntimeOptions) -> ModelAdapter:
    from .sequence import SequenceRecognizer
    recognizer = SequenceRecognizer.load(path, options=options)
    try:
        return RecognizerAdapter(recognizer)
    except BaseException:
        recognizer.close()
        raise


def _load_uni_sign(path: Path, options: RuntimeOptions) -> ModelAdapter:
    from .uni_sign import load_uni_sign_adapter
    return load_uni_sign_adapter(path, options=options)


register_model_adapter(AdapterDefinition("random_forest_v1", _load_classifier, RecognizerAdapter))
register_model_adapter(AdapterDefinition("onnx_classifier_v1", _load_classifier, RecognizerAdapter))
register_model_adapter(AdapterDefinition("onnx_ctc", _load_ctc, RecognizerAdapter))
register_model_adapter(AdapterDefinition("uni_sign_pose_slt", _load_uni_sign))


def _load_slrt(path: Path, options: RuntimeOptions) -> ModelAdapter:
    from .slrt.adapter import load_slrt_adapter
    return load_slrt_adapter(path, options)


register_model_adapter(AdapterDefinition("slrt_online_csl", _load_slrt))


def _load_replay(path: Path, options: RuntimeOptions) -> ModelAdapter:
    from .replay_adapter import load_replay_adapter
    return load_replay_adapter(path, options)


register_model_adapter(AdapterDefinition("replay_jsonl_v1", _load_replay))
