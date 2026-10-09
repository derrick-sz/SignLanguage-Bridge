"""Tests use the real RF implementation and persisted weights."""

from dataclasses import replace

import joblib
import numpy as np
import pytest

from slbridge.contracts import InputSpec
from slbridge.features import prepare_array
from slbridge.models import RandomForestBackend


def _samples(representation="hand63_v1"):
    spec = InputSpec.for_representation(representation, target_frames=2)
    return [prepare_array(np.full((2, spec.width), value, dtype=np.float32), spec)
            for value in (-2, -1.5, -1, 1, 1.5, 2)]


def test_real_rf_roundtrip_and_seed_determinism(tmp_path):
    samples = _samples()
    labels = ["A"] * 3 + ["B"] * 3
    options = dict(seed=17, parameters={"n_estimators": 15}, augmentation_repeats=2)
    model = RandomForestBackend.train(samples, labels, **options)
    second = RandomForestBackend.train(samples, labels, **options)
    assert model.classes == ("A", "B")
    assert model.feature_size == 126
    before = model.predict_proba(samples)
    np.testing.assert_array_equal(before, second.predict_proba(samples))
    assert [model.classes[index] for index in before.argmax(axis=1)] == labels
    path = tmp_path / "weights" / "rf.joblib"
    model.save(path)
    restored = RandomForestBackend.load(path, expected_classes=("A", "B"), expected_feature_size=126)
    np.testing.assert_array_equal(before, restored.predict_proba(samples))
    with pytest.raises(FileExistsError):
        model.save(path)


@pytest.mark.parametrize("parameters", [
    {"random_state": 2}, {"n_estimators": True}, {"n_estimators": 0},
    {"max_depth": 0}, {"min_samples_leaf": 0.5}, {"n_jobs": 0},
])
def test_parameters_reject_unknown_or_bad_types(parameters):
    with pytest.raises(ValueError):
        RandomForestBackend.resolve_parameters(parameters)


def test_parameter_defaults_are_resolved_without_mutating_config():
    parameters = {"max_depth": None, "n_jobs": -1}
    assert RandomForestBackend.resolve_parameters(parameters) == {
        "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1,
    }
    assert parameters == {"max_depth": None, "n_jobs": -1}


def test_augmentation_preserves_missing_values_masks_and_source(monkeypatch):
    import slbridge.features as features

    spec = InputSpec.for_representation("pose_hands225_legacy", target_frames=2)
    raw = np.full((2, 225), 0.5, dtype=np.float32)
    raw[:, :63] = -999
    sample = prepare_array(raw, spec)
    original_values = sample.values.copy()
    original_mask = sample.mask.copy()
    observed = []
    original_feature_vector = features.feature_vector

    def capture(prepared):
        observed.append((prepared.values.copy(), prepared.mask.copy()))
        return original_feature_vector(prepared)

    monkeypatch.setattr(features, "feature_vector", capture)
    model = RandomForestBackend.train([sample, sample], ["A", "B"], seed=17,
                                     parameters={"n_estimators": 3}, augmentation_repeats=2)
    assert model.feature_size == 900
    assert len(observed) == 6
    for values, mask in observed:
        np.testing.assert_array_equal(mask, original_mask)
        assert (values[~mask] == 0).all()
    assert any(not np.array_equal(values[mask], original_values[mask]) for values, mask in observed[2:])
    np.testing.assert_array_equal(sample.values, original_values)
    np.testing.assert_array_equal(sample.mask, original_mask)
    assert not sample.values.flags.writeable


def test_weight_loading_checks_classes_feature_count_and_real_backend(tmp_path):
    model = RandomForestBackend.train(_samples(), ["A"] * 3 + ["B"] * 3,
                                     seed=1, parameters={"n_estimators": 3})
    path = tmp_path / "rf.joblib"
    model.save(path)
    for classes, size in [(("B", "A"), 126), (("A", "B"), 63)]:
        with pytest.raises(ValueError):
            RandomForestBackend.load(path, expected_classes=classes, expected_feature_size=size)
    foreign = tmp_path / "foreign.joblib"
    joblib.dump({"classes": ["A", "B"]}, foreign)
    with pytest.raises(ValueError, match="RandomForestClassifier"):
        RandomForestBackend.load(foreign, expected_classes=("A", "B"), expected_feature_size=126)
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"this is not serialized weights")
    with pytest.raises(ValueError, match="deserialize"):
        RandomForestBackend.load(corrupt, expected_classes=("A", "B"), expected_feature_size=126)


def test_prediction_refuses_wrong_contract_or_inconsistent_prepared_input():
    samples = _samples()
    model = RandomForestBackend.train(samples, ["A"] * 3 + ["B"] * 3,
                                     seed=1, parameters={"n_estimators": 3})
    with pytest.raises(ValueError, match="same InputSpec"):
        model.predict_proba(_samples("pose_hands225_legacy"))
    with pytest.raises(ValueError, match="InputSpec"):
        model.predict_proba([replace(samples[0], values=np.ones((2, 62), dtype=np.float32))])


def test_loaded_backend_refuses_different_schema_with_same_feature_size(tmp_path):
    hand_spec = InputSpec.for_representation("hand63_v1", 100)
    pose_spec = InputSpec.for_representation("pose_hands225_legacy", 14)
    assert hand_spec.feature_size == pose_spec.feature_size == 6300
    samples = [prepare_array(np.full((2, 63), value, dtype=np.float32), hand_spec) for value in (-1, 1)]
    model = RandomForestBackend.train(samples, ["A", "B"], seed=1, parameters={"n_estimators": 3})
    path = tmp_path / "rf.joblib"
    model.save(path)
    restored = RandomForestBackend.load(path, expected_classes=("A", "B"), expected_feature_size=6300)
    wrong = prepare_array(np.ones((2, 225), dtype=np.float32), pose_spec)
    with pytest.raises(ValueError, match="InputSpec"):
        restored.predict_proba([wrong])


def test_loaded_backend_checks_actual_resolved_parameters(tmp_path):
    model = RandomForestBackend.train(_samples(), ["A"] * 3 + ["B"] * 3,
                                     seed=1, parameters={"n_estimators": 7})
    path = tmp_path / "rf.joblib"
    model.save(path)
    with pytest.raises(ValueError, match="parameters"):
        RandomForestBackend.load(path, expected_classes=("A", "B"), expected_feature_size=126,
                                 expected_parameters={"n_estimators": 3})
    with pytest.raises(ValueError, match="seed"):
        RandomForestBackend.load(path, expected_classes=("A", "B"), expected_feature_size=126, expected_seed=2)
