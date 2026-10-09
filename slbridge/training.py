"""Reproducible classification runs with isolated, atomically published bundles."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from types import MappingProxyType
import uuid

from ._io import canonical_hash, read_json, resolve_resource, sha256_file, write_json_atomic
from .contracts import DatasetManifest, InputSpec, ModelManifest, SplitManifest


@dataclass(frozen=True)
class TrainConfig:
    representation_id: str
    target_frames: int = 30
    seed: int = 42
    rf: dict[str, Any] = field(default_factory=dict)
    augmentation_repeats: int = 0
    noise_std: float = 0.015
    threshold_candidates: tuple[float, ...] = (0.0,)
    minimum_validation_coverage: float = 0.8
    extractor: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        import math
        from .models import RandomForestBackend

        InputSpec.for_representation(self.representation_id, self.target_frames, self.extractor)
        if self.extractor is not None:
            object.__setattr__(self, "extractor", json.loads(json.dumps(self.extractor, allow_nan=False)))
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.augmentation_repeats) is not int or not 0 <= self.augmentation_repeats <= 20:
            raise ValueError("augmentation_repeats must be an integer in [0, 20]")
        if type(self.noise_std) not in {int, float} or not math.isfinite(self.noise_std) or not 0 <= self.noise_std <= 1:
            raise ValueError("noise_std must lie in [0, 1]")
        if not isinstance(self.threshold_candidates, tuple) or not self.threshold_candidates:
            raise ValueError("threshold_candidates must be a nonempty sequence")
        for threshold in self.threshold_candidates:
            if type(threshold) not in {int, float} or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Thresholds must lie in [0, 1]")
        if type(self.minimum_validation_coverage) not in {int, float} or not math.isfinite(self.minimum_validation_coverage) or not 0 < self.minimum_validation_coverage <= 1:
            raise ValueError("minimum_validation_coverage must lie in (0, 1]")
        object.__setattr__(self, "rf", MappingProxyType(RandomForestBackend.resolve_parameters(self.rf)))
        object.__setattr__(self, "threshold_candidates", tuple(sorted(set(self.threshold_candidates))))

    def to_dict(self) -> dict[str, Any]:
        value = {entry.name: getattr(self, entry.name) for entry in fields(self)}
        value["rf"] = dict(self.rf)
        value["threshold_candidates"] = list(self.threshold_candidates)
        if self.extractor is None:
            value.pop("extractor")  # Preserve released legacy bundle parameter hashes.
        else:
            value["extractor"] = json.loads(json.dumps(self.extractor, allow_nan=False))
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainConfig:
        if not isinstance(value, dict):
            raise ValueError("Training config must be a JSON object")
        unknown = set(value) - {entry.name for entry in fields(cls)}
        if unknown or "representation_id" not in value:
            raise ValueError(f"Invalid training config fields: {sorted(unknown)}; representation_id is required")
        data = dict(value)
        if "threshold_candidates" in data:
            if not isinstance(data["threshold_candidates"], list):
                raise ValueError("threshold_candidates must be a JSON array")
            data["threshold_candidates"] = tuple(data["threshold_candidates"])
        return cls(**data)


def evaluate_classification(truth: list[str], probabilities: Any, classes: tuple[str, ...], threshold: float) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    if not truth or probabilities.shape != (len(truth), len(classes)):
        raise ValueError("Evaluation truth and probability shape disagree")
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    accepted = probabilities.max(axis=1) >= threshold
    actual = np.asarray(truth)
    selected_truth, selected_predicted = actual[accepted], predicted[accepted]
    count = int(accepted.sum())
    return {
        "samples": len(truth), "accepted": count, "rejected": len(truth) - count,
        "coverage": count / len(truth), "threshold": threshold,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(classes), average="macro", zero_division=0)),
        "per_class": classification_report(actual, predicted, labels=list(classes), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(classes)).tolist(),
        "class_order": list(classes),
        "accepted_accuracy": float(accuracy_score(selected_truth, selected_predicted)) if count else None,
        "accepted_macro_f1": float(f1_score(selected_truth, selected_predicted, labels=list(classes), average="macro", zero_division=0)) if count else None,
    }


def _provenance() -> dict[str, Any]:
    code = {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    return {"core_code_hash": canonical_hash(code), "core_files": code,
            "python": sys.version.split()[0], "dataset_note": "Legacy label meanings, sign language and global signer identity are unverified unless explicitly supplied."}


def train(manifest: DatasetManifest, split: SplitManifest, config: TrainConfig,
          output_dir: str | Path, run_id: str | None = None) -> Path:
    """Train one real backend; never modify source files or existing model versions."""
    from .data import validate_split, verify_manifest
    from .features import InputRejected, prepare_input
    from .models import RandomForestBackend

    config = TrainConfig.from_dict(config.to_dict())
    if manifest.input_kind not in {"npy", "landmarks_npz"}:
        raise ValueError("Run RGB extraction first; training requires indexed landmark clips")
    if manifest.representation_id != config.representation_id:
        raise ValueError("Training config and dataset representation disagree")
    if config.seed != split.seed:
        raise ValueError("Training seed and frozen split seed disagree")
    validate_split(manifest, split)
    verify_manifest(manifest)
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(Path(manifest.root).resolve()):
        raise ValueError("Model output must be outside the read-only source dataset")
    run_id = run_id or (datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    # Validate run_id before using it in a path.
    if not isinstance(run_id, str):
        raise ValueError("run_id must be a string")
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid run_id")
    final = output_dir / run_id
    if final.exists():
        raise FileExistsError(f"Model version already exists: {final}")
    spec = InputSpec.for_representation(config.representation_id, config.target_frames, config.extractor)
    records = {record.sample_id: record for record in manifest.samples}

    def prepare(ids: tuple[str, ...]) -> tuple[list[Any], list[str]]:
        samples, labels = [], []
        for identifier in ids:
            record = records[identifier]
            try:
                samples.append(prepare_input(resolve_resource(manifest.root, record.resource), spec))
            except InputRejected as error:
                raise ValueError(f"Training sample rejected: {identifier}: {error.reason}; re-index/quarantine and re-freeze the split") from error
            labels.append(record.label_id)
        return samples, labels

    train_samples, train_labels = prepare(split.train)
    validation_samples, validation_labels = prepare(split.validation)
    test_samples, test_labels = prepare(split.test)
    backend = RandomForestBackend.train(train_samples, train_labels, seed=config.seed, parameters=dict(config.rf),
                                        augmentation_repeats=config.augmentation_repeats, noise_std=config.noise_std)
    # Threshold selection sees validation only. Test evaluation happens once,
    # after selection. Probabilities are not presented as calibrated confidence.
    validation_probabilities = backend.predict_proba(validation_samples)
    candidates = [evaluate_classification(validation_labels, validation_probabilities, backend.classes, threshold)
                  for threshold in config.threshold_candidates]
    eligible = [candidate for candidate in candidates if candidate["coverage"] >= config.minimum_validation_coverage]
    if not eligible:
        raise ValueError("No validation threshold satisfies minimum_validation_coverage")
    selected = max(eligible, key=lambda metric: (metric["accepted_accuracy"], metric["coverage"], -metric["threshold"]))
    threshold = selected["threshold"]
    test_metrics = evaluate_classification(test_labels, backend.predict_proba(test_samples), backend.classes, threshold)
    labels = []
    for label in backend.classes:
        displays: dict[str, str] = {}
        for record in manifest.samples:
            if record.label_id == label and record.quality_status == "ready":
                for language, text in record.display_text.items():
                    if language in displays and displays[language] != text:
                        raise ValueError(f"Conflicting display text for label {label}")
                    displays[language] = text
        labels.append({"label_id": label, "display_text": displays})
    label_table = {"schema_version": 1, "labels": labels}
    metrics = {"validation": selected, "test": test_metrics,
               "threshold_selection": {"rule": "validation accepted accuracy, then coverage, then lower threshold", "candidates": candidates},
               "training_original_samples": len(train_samples), "training_augmented_samples": len(train_samples) * config.augmentation_repeats,
               "protocol": split.protocol, "group_fields": list(split.group_fields),
               "limitations": ["Libras experiment is finite source-item classification with local informant groups, not signer-independent evaluation or continuous translation."
                               if manifest.input_kind == "landmarks_npz" else "Results validate framework behaviour; unverified legacy data do not establish CSL/Libras recognition quality.",
                               "No unknown-sign/OOV validation set; threshold rejection is only a low-score heuristic."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f".{run_id}.{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        backend.save(staging / "weights" / "model.joblib")
        write_json_atomic(staging / "labels.json", label_table)
        write_json_atomic(staging / "resolved_config.json", {"training": config.to_dict(), "input_spec": spec.to_dict(),
                                                             "rejection_threshold": threshold})
        write_json_atomic(staging / "splits" / "dataset.json", manifest.to_dict())
        write_json_atomic(staging / "splits" / "split.json", split.to_dict())
        write_json_atomic(staging / "metrics.json", metrics)
        artifacts = {path.relative_to(staging).as_posix(): sha256_file(path)
                     for path in sorted(staging.rglob("*")) if path.is_file()}
        model_manifest = ModelManifest(
            run_id=run_id, backend=backend.backend_id, input_spec=spec, labels_hash=canonical_hash(label_table),
            dataset_hash=manifest.fingerprint, split_hash=split.fingerprint, artifacts=artifacts,
            parameters=config.to_dict(), dependencies={name: importlib.metadata.version(name) for name in
                (("numpy", "scikit-learn", "joblib", "mediapipe", "opencv-contrib-python") if config.extractor else ("numpy", "scikit-learn", "joblib"))},
            provenance=_provenance(), created_at=datetime.now(timezone.utc).isoformat(), sign_language=manifest.sign_language,
        )
        write_json_atomic(staging / "manifest.json", model_manifest.to_dict())
        # Prove the persisted bundle can be loaded before atomically publishing.
        from .bundle import load_bundle
        restored = load_bundle(staging)
        import numpy as np
        np.testing.assert_array_equal(restored.backend.predict_proba(validation_samples), validation_probabilities)
        if final.exists():
            raise FileExistsError(f"Model version already exists: {final}")
        staging.rename(final)
        return final
    finally:
        # Remove only the owned temporary directory, never an existing model.
        if staging.exists():
            shutil.rmtree(staging)
