"""Validate complete local model bundles before loading backend state.

Checksums detect corruption/inconsistent files, not a hostile manifest rewrite.
Joblib bundles must come from a trusted training run; never load uploaded weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._io import canonical_hash, read_json, resolve_resource, sha256_file
from .contracts import DatasetManifest, ModelManifest, SplitManifest
from .backends import get_backend, validate_dependencies


REQUIRED_ARTIFACTS = frozenset({"labels.json", "resolved_config.json", "splits/dataset.json",
                               "splits/split.json", "metrics.json", "weights/model.joblib"})


@dataclass(frozen=True)
class ModelBundle:
    path: Path
    manifest: ModelManifest
    backend: Any
    labels: dict[str, dict[str, str]]
    rejection_threshold: float


def load_bundle(path: str | Path, *, options: Any = None) -> ModelBundle:
    """Load a validated, trusted local bundle without requiring original data."""
    path = Path(path).resolve()
    manifest = ModelManifest.from_dict(read_json(path / "manifest.json"))
    definition = get_backend(manifest.backend)
    required_artifacts = definition.required_artifacts
    if set(manifest.artifacts) != required_artifacts:
        raise ValueError("Model bundle has missing or unexpected artifacts")
    actual_files = {file.relative_to(path).as_posix() for file in path.rglob("*") if file.is_file()}
    if actual_files != required_artifacts | {"manifest.json"}:
        raise ValueError("Incomplete model directory or untracked files")
    for resource, expected in manifest.artifacts.items():
        artifact = resolve_resource(path, resource)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Model artifact checksum mismatch: {resource}")
    validate_dependencies(manifest, definition)
    labels_value = read_json(path / "labels.json")
    if canonical_hash(labels_value) != manifest.labels_hash:
        raise ValueError("Label table does not match model manifest")
    if not isinstance(labels_value, dict) or set(labels_value) != {"schema_version", "labels"} or labels_value["schema_version"] != 1:
        raise ValueError("Invalid label table schema")
    labels: dict[str, dict[str, str]] = {}
    if not isinstance(labels_value["labels"], list) or not labels_value["labels"]:
        raise ValueError("Model label table must be nonempty")
    for label in labels_value["labels"]:
        if not isinstance(label, dict) or set(label) != {"label_id", "display_text"}:
            raise ValueError("Invalid label entry")
        identifier, displays = label["label_id"], label["display_text"]
        if not isinstance(identifier, str) or not identifier or identifier in labels or not isinstance(displays, dict):
            raise ValueError("Duplicate/invalid model label")
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in displays.items()):
            raise ValueError("Invalid label display text")
        labels[identifier] = displays
    dataset = DatasetManifest.from_dict(read_json(path / "splits/dataset.json"))
    split = SplitManifest.from_dict(read_json(path / "splits/split.json"))
    if dataset.fingerprint != manifest.dataset_hash or split.fingerprint != manifest.split_hash:
        raise ValueError("Frozen dataset/split does not match model manifest")
    if dataset.sign_language != manifest.sign_language or dataset.representation_id != manifest.input_spec.representation_id or dataset.input_kind not in {"npy", "landmarks_npz"}:
        raise ValueError("Frozen dataset and model input/language disagree")
    from .data import validate_split
    validate_split(dataset, split)  # Metadata only: inference never needs training source files.
    expected_labels = sorted({sample.label_id for sample in dataset.samples if sample.sample_id in split.train})
    if list(labels) != expected_labels:
        raise ValueError("Label table does not match frozen training labels")
    resolved = read_json(path / "resolved_config.json")
    config, threshold = definition.validate_config(manifest, resolved, split)
    from .runtime_options import RuntimeOptions
    if options is None:
        options = RuntimeOptions()
    if not isinstance(options, RuntimeOptions):
        raise ValueError("options must be RuntimeOptions")
    backend = definition.load(resolve_resource(path, definition.weights_artifact), manifest,
                              tuple(labels), config, options)
    return ModelBundle(path, manifest, backend, labels, threshold)
