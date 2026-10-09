"""The first real model backend: a deterministic landmark random forest.

Runtime dependencies are imported only when a backend is used. Training split
selection, evaluation and model publication belong to the caller.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Sequence

from .contracts import InputSpec, PreparedInput


def _sample_matrix(samples: Sequence[PreparedInput], spec: InputSpec | None = None) -> tuple[Any, InputSpec]:
    import numpy as np
    from .features import feature_vector

    if not samples:
        raise ValueError("At least one PreparedInput is required")
    if any(not isinstance(sample, PreparedInput) for sample in samples):
        raise ValueError("Model samples must be PreparedInput objects")
    expected_spec = samples[0].spec if spec is None else spec
    if not isinstance(expected_spec, InputSpec) or any(sample.spec != expected_spec for sample in samples):
        raise ValueError("Model samples must use the same InputSpec")
    matrix = np.stack([feature_vector(sample) for sample in samples])
    if matrix.shape != (len(samples), expected_spec.feature_size):
        raise ValueError("Prepared feature size does not match InputSpec")
    return matrix, expected_spec


class RandomForestBackend:
    """Own an estimator without exposing its fitting API to external clients."""

    backend_id = "random_forest_v1"

    def __init__(self, estimator: Any, input_spec: InputSpec | None = None) -> None:
        self._estimator = estimator
        self._input_spec = input_spec
        self._runtime_info = {"backend": self.backend_id, "requested_device": "cpu", "device": "cpu",
                              "providers": ["CPU"], "fallback_reason": None}

    def configure_runtime(self, options: Any) -> None:
        from .runtime_options import RuntimeOptions

        if not isinstance(options, RuntimeOptions):
            raise ValueError("options must be RuntimeOptions")
        if options.device == "cuda":
            raise ValueError("random_forest_v1 supports CPU only; strict CUDA was requested")
        self._runtime_info = {"backend": self.backend_id, "requested_device": options.device,
                              "device": "cpu", "providers": ["CPU"],
                              "fallback_reason": "random_forest_v1 supports CPU only" if options.device == "auto" else None,
                              "intra_op_threads": options.intra_op_threads}

    @property
    def runtime_info(self) -> dict[str, Any]:
        return dict(self._runtime_info)

    def warmup(self) -> None:
        if self._input_spec is None:
            raise ValueError("Cannot warm up a model without its complete InputSpec")
        import numpy as np
        from .features import prepare_array

        spec = self._input_spec
        sample = prepare_array(np.zeros((spec.target_frames, spec.width), dtype=np.float32), spec,
                               timestamps=np.arange(spec.target_frames, dtype=np.float64)
                               if spec.temporal_policy == "timestamp_linear" else None)
        self.predict_proba([sample])

    def close(self) -> None:
        self._estimator = None

    @classmethod
    def resolve_parameters(cls, parameters: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameters, dict):
            raise ValueError("Random forest parameters must be an object")
        resolved = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1, "n_jobs": 1}
        unknown = set(parameters) - set(resolved)
        if unknown:
            raise ValueError(f"Unknown random forest parameters: {sorted(unknown)}")
        resolved.update(parameters)
        for name in ("n_estimators", "min_samples_leaf"):
            if type(resolved[name]) is not int or resolved[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        depth = resolved["max_depth"]
        if depth is not None and (type(depth) is not int or depth < 1):
            raise ValueError("max_depth must be a positive integer or null")
        if type(resolved["n_jobs"]) is not int or resolved["n_jobs"] == 0:
            raise ValueError("n_jobs must be a nonzero integer")
        return resolved

    @classmethod
    def train(
        cls,
        samples: Sequence[PreparedInput],
        labels: Sequence[str],
        *,
        seed: int,
        parameters: dict[str, Any],
        augmentation_repeats: int = 0,
        noise_std: float = 0.015,
    ) -> RandomForestBackend:
        resolved = cls.resolve_parameters(parameters)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        if type(augmentation_repeats) is not int or augmentation_repeats < 0:
            raise ValueError("augmentation_repeats must be a nonnegative integer")
        if type(noise_std) not in {int, float} or not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if isinstance(labels, (str, bytes)) or len(labels) != len(samples):
            raise ValueError("Each training sample must have one label")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("Training labels must be nonempty strings")

        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        matrix, spec = _sample_matrix(samples)
        matrices = [matrix]
        rng = np.random.default_rng(seed)
        for _ in range(augmentation_repeats):
            augmented = []
            for sample in samples:
                values = sample.values.copy()
                # Only observed coordinates are perturbed. The bool mask and
                # missing zeros retain exactly their original meaning.
                augmentation_mask = sample.mask.copy()
                if spec.representation_id == "mediapipe225_rgb_v1":
                    augmentation_mask[:, 2::3] = False  # Planar z=0 is part of the schema.
                noise = rng.normal(0, noise_std, size=int(augmentation_mask.sum())).astype(np.float32)
                values[augmentation_mask] += noise
                values.setflags(write=False)
                augmented.append(replace(sample, values=values))
            matrices.append(_sample_matrix(augmented, spec)[0])
        train_matrix = np.concatenate(matrices, axis=0) if augmentation_repeats else matrix
        train_labels = list(labels) * (augmentation_repeats + 1)
        estimator = RandomForestClassifier(random_state=seed, **resolved)
        estimator.fit(train_matrix, train_labels)
        return cls(estimator, spec)

    @property
    def classes(self) -> tuple[str, ...]:
        if self._estimator is None:
            raise RuntimeError("Random forest backend is closed")
        return tuple(str(label) for label in self._estimator.classes_)

    @property
    def feature_size(self) -> int:
        if self._estimator is None:
            raise RuntimeError("Random forest backend is closed")
        return int(self._estimator.n_features_in_)

    def predict_proba(self, samples: Sequence[PreparedInput]) -> Any:
        import numpy as np

        matrix, _ = _sample_matrix(samples, self._input_spec)
        if matrix.shape[1] != self.feature_size:
            raise ValueError("Input feature size is incompatible with this random forest")
        probabilities = np.asarray(self._estimator.predict_proba(matrix), dtype=np.float64)
        if (probabilities.shape != (len(samples), len(self.classes)) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-7)):
            raise ValueError("Random forest returned invalid class probabilities")
        return probabilities

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        if self._input_spec is None:
            raise ValueError("Cannot save a model without its complete InputSpec")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publication is handled by the caller. Refuse accidental reuse even
        # inside its staging directory instead of overwriting existing weights.
        with path.open("xb") as stream:
            joblib.dump({"schema_version": 1, "input_spec": self._input_spec.to_dict(),
                         "estimator": self._estimator}, stream, compress=3)

    @classmethod
    def load(
        cls, path: str | Path, *, expected_classes: tuple[str, ...], expected_feature_size: int,
        expected_input_spec: InputSpec | None = None, expected_parameters: dict[str, Any] | None = None,
        expected_seed: int | None = None,
    ) -> RandomForestBackend:
        if (not isinstance(expected_classes, tuple) or not expected_classes
                or any(not isinstance(label, str) or not label.strip() for label in expected_classes)
                or len(set(expected_classes)) != len(expected_classes)):
            raise ValueError("Expected classes must be a nonempty tuple of unique strings")
        if type(expected_feature_size) is not int or expected_feature_size < 1:
            raise ValueError("Expected feature size must be a positive integer")

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.validation import check_is_fitted

        # This deserializes a trusted local artifact, after the bundle loader
        # has verified all checksums. A checksum alone is not a trust boundary.
        try:
            state = joblib.load(Path(path))
        except Exception as error:
            raise ValueError("Cannot deserialize random forest weights") from error
        if (not isinstance(state, dict) or set(state) != {"schema_version", "input_spec", "estimator"}
                or type(state["schema_version"]) is not int or state["schema_version"] != 1):
            raise ValueError("Weights do not contain a versioned RandomForestClassifier state")
        estimator = state["estimator"]
        input_spec = InputSpec.from_dict(state["input_spec"])
        if input_spec.feature_size != expected_feature_size or (expected_input_spec is not None and input_spec != expected_input_spec):
            raise ValueError("Saved random forest InputSpec disagrees with the model contract")
        if type(estimator) is not RandomForestClassifier:
            raise ValueError("Weights do not contain a RandomForestClassifier")
        try:
            check_is_fitted(estimator, attributes=["estimators_", "classes_", "n_features_in_"])
            if (not isinstance(estimator.classes_, np.ndarray) or estimator.classes_.ndim != 1
                    or not all(isinstance(label, (str, np.str_)) for label in estimator.classes_)
                    or tuple(str(label) for label in estimator.classes_) != expected_classes
                    or estimator.n_features_in_ != expected_feature_size
                    or not estimator.estimators_):
                raise ValueError("Random forest classes/features do not match the model contract")
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Invalid random forest weights or incompatible model contract") from error
        actual_parameters = estimator.get_params()
        if expected_parameters is not None:
            resolved = cls.resolve_parameters(expected_parameters)
            if any(actual_parameters[key] != value for key, value in resolved.items()):
                raise ValueError("Persisted random forest parameters disagree with resolved config")
        if expected_seed is not None and actual_parameters["random_state"] != expected_seed:
            raise ValueError("Persisted random forest seed disagrees with resolved config")
        return cls(estimator, input_spec)
