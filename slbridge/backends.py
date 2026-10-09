"""Registry of concrete inference backends and their saved bundle contracts.

The registry itself has no numerical/runtime imports. A backend is constructed
only after the shared bundle loader verifies its complete frozen metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
import math
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from ._io import canonical_hash

if TYPE_CHECKING:
    from .contracts import ModelManifest, SplitManifest
    from .runtime_options import RuntimeOptions


COMMON_ARTIFACTS = frozenset({"labels.json", "resolved_config.json", "splits/dataset.json",
                              "splits/split.json", "metrics.json"})


@dataclass(frozen=True)
class BackendDefinition:
    backend_id: str
    weights_artifact: str
    dependencies: tuple[str, ...]
    validate_config: Callable[..., tuple[dict[str, Any], float]]
    load: Callable[..., Any]

    @property
    def required_artifacts(self) -> frozenset[str]:
        return COMMON_ARTIFACTS | {self.weights_artifact}


_REGISTRY: dict[str, BackendDefinition] = {}


def register_backend(definition: BackendDefinition) -> None:
    """Register an implemented adapter; duplicate ids cannot replace a backend."""
    if not isinstance(definition, BackendDefinition) or not definition.backend_id:
        raise ValueError("A concrete BackendDefinition is required")
    if definition.backend_id in _REGISTRY:
        raise ValueError(f"Backend already registered: {definition.backend_id}")
    if not callable(definition.validate_config) or not callable(definition.load):
        raise ValueError("Backend definition requires validation and loading functions")
    _REGISTRY[definition.backend_id] = definition


def backend_ids() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_backend(backend_id: str) -> BackendDefinition:
    try:
        return _REGISTRY[backend_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unsupported model backend: {backend_id}") from error


def initialize_runtime_libraries() -> dict[str, Any]:
    """Initialize installed ORT before a managed process loads native RF libraries.

    Some Windows numerical DLL builds cannot initialize ORT after sklearn's
    native libraries. Call this explicitly at a mixed-backend application startup,
    never during module import. Pure RF environments do not need ORT installed;
    an unavailable optional runtime is reported while RF remains usable.
    """
    if importlib.util.find_spec("onnxruntime") is None:
        return {"onnxruntime": None, "onnxruntime_error": None}
    try:
        import onnxruntime
        return {"onnxruntime": onnxruntime.__version__, "onnxruntime_error": None}
    except Exception as error:
        return {"onnxruntime": None, "onnxruntime_error": f"{type(error).__name__}: {error}"}


def dependency_version(name: str) -> str:
    """The CPU/GPU ORT wheels implement the same versioned runtime contract."""
    candidates = ("onnxruntime-gpu", "onnxruntime") if name == "onnxruntime" else (name,)
    versions = []
    for candidate in candidates:
        try:
            versions.append(importlib.metadata.version(candidate))
        except importlib.metadata.PackageNotFoundError:
            continue
    if len(set(versions)) > 1:
        raise ValueError("Conflicting ONNX Runtime CPU/GPU package versions; use one recorded runtime environment")
    if versions:
        return versions[0]
    raise ValueError(f"Missing model runtime dependency: {name}")


def validate_dependencies(manifest: ModelManifest, definition: BackendDefinition) -> None:
    for name in definition.dependencies:
        if manifest.dependencies.get(name) != dependency_version(name):
            raise ValueError(f"Model dependency version mismatch for {name}; use its recorded environment")


def _rf_config(manifest: ModelManifest, resolved: Any,
               split: SplitManifest) -> tuple[dict[str, Any], float]:
    from .contracts import InputSpec
    from .training import TrainConfig

    if manifest.runtime_config is not None:
        raise ValueError("Random forest bundle must not declare ONNX runtime bindings")
    if not isinstance(resolved, dict) or set(resolved) != {"training", "input_spec", "rejection_threshold"}:
        raise ValueError("Invalid resolved model config")
    if InputSpec.from_dict(resolved["input_spec"]) != manifest.input_spec:
        raise ValueError("Model and processing InputSpec disagree")
    config = TrainConfig.from_dict(resolved["training"])
    if canonical_hash(config.to_dict()) != canonical_hash(manifest.parameters):
        raise ValueError("Manifest and resolved training parameters disagree")
    if InputSpec.for_representation(config.representation_id, config.target_frames, config.extractor) != manifest.input_spec or config.seed != split.seed:
        raise ValueError("Training parameters disagree with input/split contract")
    threshold = resolved["rejection_threshold"]
    if type(threshold) not in {int, float} or threshold not in config.threshold_candidates:
        raise ValueError("Invalid or unconfigured rejection threshold")
    return {"parameters": dict(config.rf), "seed": config.seed}, float(threshold)


def _load_rf(path: Path, manifest: ModelManifest, labels: tuple[str, ...],
             config: dict[str, Any], options: RuntimeOptions) -> Any:
    from .models import RandomForestBackend

    if options.device == "cuda":
        raise ValueError("random_forest_v1 supports CPU only; strict CUDA was requested")
    backend = RandomForestBackend.load(
        path, expected_classes=labels, expected_feature_size=manifest.input_spec.feature_size,
        expected_input_spec=manifest.input_spec, expected_parameters=config["parameters"],
        expected_seed=config["seed"],
    )
    backend.configure_runtime(options)
    return backend


def _onnx_config(manifest: ModelManifest, resolved: Any,
                 split: SplitManifest) -> tuple[dict[str, Any], float]:
    from .contracts import InputSpec
    from .onnx_backend import OnnxRuntimeConfig

    required = {"schema_version", "backend", "input_spec", "runtime_config", "parameters", "rejection_threshold"}
    if not isinstance(resolved, dict) or set(resolved) != required:
        raise ValueError("Invalid resolved ONNX model config")
    if type(resolved["schema_version"]) is not int or resolved["schema_version"] != 1 or resolved["backend"] != manifest.backend:
        raise ValueError("Unsupported resolved ONNX config version/backend")
    if InputSpec.from_dict(resolved["input_spec"]) != manifest.input_spec:
        raise ValueError("Model and processing InputSpec disagree")
    bindings = OnnxRuntimeConfig.from_dict(resolved["runtime_config"])
    manifest_bindings = OnnxRuntimeConfig.from_dict(manifest.runtime_config)
    if bindings != manifest_bindings:
        raise ValueError("Manifest and resolved ONNX runtime bindings disagree")
    if not isinstance(resolved["parameters"], dict) or canonical_hash(resolved["parameters"]) != canonical_hash(manifest.parameters):
        raise ValueError("Manifest and resolved ONNX parameters disagree")
    threshold = resolved["rejection_threshold"]
    if type(threshold) not in {int, float} or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("ONNX rejection threshold must lie in [0, 1]")
    return bindings.to_dict(), float(threshold)


def _load_onnx(path: Path, manifest: ModelManifest, labels: tuple[str, ...],
               config: dict[str, Any], options: RuntimeOptions) -> Any:
    from .onnx_backend import OnnxClassifierBackend

    return OnnxClassifierBackend.load(path, expected_classes=labels,
                                      expected_input_spec=manifest.input_spec,
                                      runtime_config=config, options=options)


register_backend(BackendDefinition("random_forest_v1", "weights/model.joblib",
                                   ("scikit-learn", "numpy", "joblib"), _rf_config, _load_rf))
register_backend(BackendDefinition("onnx_classifier_v1", "weights/model.onnx",
                                   ("numpy", "onnxruntime"), _onnx_config, _load_onnx))
