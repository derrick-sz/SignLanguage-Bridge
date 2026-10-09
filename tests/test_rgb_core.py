"""F4 contract, source-fold preservation and timestamp processing tests."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from slbridge._io import canonical_hash, read_json, sha256_file
from slbridge.contracts import InputSpec
from slbridge.data import create_split, import_libras, project_split, select_labels, validate_split
from slbridge.features import load_landmark_archive, prepare_array, prepare_input
from slbridge.training import TrainConfig


EXTRACTOR = {"config": {"frames": 3}, "identity": {"test_fixture": "contract-test-only"}}


def write_archive(path: Path, **overrides):
    values = np.zeros((3, 225), dtype=np.float32)
    values[:, ::3] = np.arange(3, dtype=np.float32)[:, None]
    metadata = {"schema_version": 1, "representation_id": "mediapipe225_rgb_v1", "extractor": EXTRACTOR,
                "source_id": "raw-sample", "source_hash": "a" * 64, "quality": {}, "provenance": {}}
    entries = {"values": values, "timestamps": np.array([0.0, 0.1, 0.6], dtype=np.float64),
               "frame_indices": np.array([0, 1, 6], dtype=np.int64), "metadata": np.asarray(json.dumps(metadata))}
    entries.update(overrides)
    np.savez_compressed(path, **entries)
    return values


def test_timestamp_processing_and_archive_parity(tmp_path):
    path = tmp_path / "clip.npz"
    raw = write_archive(path)
    spec = InputSpec.for_representation("mediapipe225_rgb_v1", 3, EXTRACTOR)
    prepared = prepare_input(path, spec)
    direct = prepare_array(raw, spec, timestamps=np.array([0.0, 0.1, 0.6]))
    np.testing.assert_array_equal(prepared.values, direct.values)
    assert prepared.values[1, 0] == pytest.approx(1.4)
    assert prepared.timestamps == (0.0, 0.3, 0.6)
    assert prepared.quality["timestamps_known"] is True
    preserved = prepare_array(raw, spec, timestamps=np.array([0.0, 0.1, 0.6]), quality_metadata={"source_frames": 19})
    assert preserved.quality["source_frames"] == 19 and preserved.quality["landmark_frames"] == 3


@pytest.mark.parametrize("override", [
    {"timestamps": np.array([0.0, 0.0, 1.0], dtype=np.float64)},
    {"timestamps": np.array([0.0, 0.1, 1.0], dtype=np.float32)},
    {"frame_indices": np.array([0, 1, 2], dtype=np.int32)},
    {"frame_indices": np.array([0, 1, 1], dtype=np.int64)},
    {"frame_indices": np.array([0, np.iinfo(np.int64).max, -2], dtype=np.int64)},
    {"frame_indices": np.array([np.iinfo(np.int64).max, np.iinfo(np.int64).min, -2], dtype=np.int64)},
    {"values": np.zeros((3, 225), dtype=np.float64)},
    {"values": np.ones((3, 225), dtype=np.float32)},
    {"metadata": np.asarray('{"schema_version":1,"schema_version":1}')},
])
def test_bad_archive_contracts_refused(tmp_path, override):
    path = tmp_path / "bad.npz"
    write_archive(path, **override)
    with pytest.raises(ValueError):
        load_landmark_archive(path)


def test_archive_identity_and_input_spec_are_explicit(tmp_path):
    path = tmp_path / "clip.npz"
    write_archive(path)
    with pytest.raises(ValueError, match="incompatible"):
        load_landmark_archive(path, expected_extractor={"config": {"frames": 4}, "identity": {}})
    with pytest.raises(ValueError, match="extractor"):
        InputSpec.for_representation("mediapipe225_rgb_v1")
    spec = InputSpec.for_representation("mediapipe225_rgb_v1", 3, EXTRACTOR)
    with pytest.raises(ValueError, match="Timestamp"):
        prepare_array(np.zeros((3, 225), dtype=np.float32), spec)


def fake_source(root: Path):
    for item in (1, 2):
        for group in range(1, 6):
            folder = root / f"{item}_Word {item}" / f"INFORMANTE {group}"
            folder.mkdir(parents=True)
            for repeat in range(2):
                (folder / f"{repeat}.mp4").write_bytes(f"{item}:{group}:{repeat}".encode())
    return import_libras(root)


def fake_derived(source):
    return replace(source, root=str(Path(source.root).parent / "features"), input_kind="landmarks_npz",
                   representation_id="mediapipe225_rgb_v1", samples=tuple(
                       replace(record, input_kind="landmarks_npz", representation_id="mediapipe225_rgb_v1",
                               resource=f"{index}.npz", content_hash=canonical_hash(index),
                               derived_from=record.sample_id, groups={**record.groups, "raw_source_hash": record.content_hash},
                               quality_issues=()) for index, record in enumerate(source.samples)))


def test_frozen_source_fold_membership_survives_derivation(tmp_path):
    source = fake_source(tmp_path / "source")
    split = create_split(source, group_fields=("local_informant",))
    derived = fake_derived(source)
    projected = project_split(source, split, derived)
    for name in ("train", "validation", "test"):
        assert getattr(projected, name) == getattr(split, name)
    assert projected.dataset_hash == derived.fingerprint
    altered = replace(derived, samples=(replace(derived.samples[0], label_id="wrong"),) + derived.samples[1:])
    with pytest.raises(ValueError, match="label/identity"):
        project_split(source, split, altered)


def test_derived_raw_duplicate_hash_cannot_cross_folds(tmp_path):
    source = fake_source(tmp_path / "source")
    derived = fake_derived(source)
    first, second = derived.samples[:2]
    derived = replace(derived, samples=(first, replace(second, groups={**second.groups, "raw_source_hash": first.groups["raw_source_hash"]})) + derived.samples[2:])
    split = create_split(derived)
    positions = {identifier: name for name in ("train", "validation", "test") for identifier in getattr(split, name)}
    assert positions[first.sample_id] == positions[second.sample_id]
    original_part = getattr(split, positions[first.sample_id])
    destination = next(name for name in ("train", "validation", "test") if name != positions[first.sample_id])
    leaked = replace(split, **{positions[first.sample_id]: tuple(x for x in original_part if x != second.sample_id),
                               destination: getattr(split, destination) + (second.sample_id,)})
    with pytest.raises(ValueError, match="crosses"):
        validate_split(derived, leaked)


def test_failed_group_is_excluded_without_reassigning_remaining_sources(tmp_path):
    source = fake_source(tmp_path / "source")
    split = create_split(source, group_fields=("local_informant",))
    # Training has three groups per label, so rejecting one retains coverage.
    damaged_id = split.train[0]
    derived = fake_derived(source)
    damaged = next(record for record in derived.samples if record.sample_id == damaged_id)
    derived = replace(derived, samples=tuple(replace(record, quality_status="quarantined", quality_issues=("quality_rejected",))
                                             if record.sample_id == damaged_id else record for record in derived.samples))
    projected = project_split(source, split, derived)
    expected_excluded = {record.sample_id for record in derived.samples if record.groups["local_informant"] == damaged.groups["local_informant"]}
    assert set(projected.excluded) == expected_excluded
    assert projected.validation == split.validation and projected.test == split.test


def test_selection_preserves_source_identity_and_unknown_labels_fail(tmp_path):
    source = fake_source(tmp_path / "source")
    selected = select_labels(source, ("item:1",))
    assert all(record.label_id == "item:1" for record in selected.samples)
    assert selected.dataset_id == source.dataset_id
    with pytest.raises(ValueError, match="Unknown"):
        select_labels(source, ("item:999",))


def test_legacy_model_bundles_still_load_if_present():
    from slbridge.bundle import load_bundle
    path = Path("artifacts/models/core-hand63-20261009")
    if not path.exists():
        pytest.skip("Local pre-F4 bundle not distributed")
    assert load_bundle(path).manifest.input_spec.extractor is None


def test_train_config_preserves_legacy_parameter_definition():
    config = TrainConfig("hand63_v1")
    assert "extractor" not in config.to_dict()
    config = TrainConfig("mediapipe225_rgb_v1", extractor=EXTRACTOR)
    serialized = config.to_dict()
    serialized["extractor"]["config"]["frames"] = 99
    assert config.extractor["config"]["frames"] == 3


def test_rgb_augmentation_keeps_planar_z_zero(monkeypatch):
    from slbridge.models import RandomForestBackend
    import slbridge.features as features
    spec = InputSpec.for_representation("mediapipe225_rgb_v1", 3, EXTRACTOR)
    raw = np.zeros((3, 225), dtype=np.float32)
    sample = prepare_array(raw, spec, timestamps=np.array([0.0, 0.1, 0.6]))
    original = features.feature_vector
    def check(prepared):
        assert (prepared.values[:, 2::3] == 0).all()
        return original(prepared)
    monkeypatch.setattr(features, "feature_vector", check)
    RandomForestBackend.train([sample, sample], ["A", "B"], seed=42, parameters={"n_estimators": 3}, augmentation_repeats=1)
