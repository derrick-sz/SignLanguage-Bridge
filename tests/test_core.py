"""Acceptance at the same public interfaces used by CLI and applications."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from slbridge._io import read_json, sha256_file, write_json_atomic
from slbridge.bundle import load_bundle
from slbridge.contracts import InputSpec
from slbridge.data import create_split, import_legacy
from slbridge.features import InputRejected, feature_vector, load_sequence, prepare_array, prepare_input
from slbridge.inference import Recognizer
from slbridge.training import TrainConfig, train


def source_dataset(root: Path, width: int = 63) -> Path:
    root.mkdir()
    rng = np.random.default_rng(17)
    for label, value in (("alpha", -1.0), ("beta", 1.0)):
        for index in range(9):
            array = rng.normal(value, 0.03, size=(8 + index, width)).astype(np.float64)
            if width == 225:
                array[:, 99:120] = -999
            np.save(root / f"{label}_{index:03}.npy", array)
    return root


@pytest.fixture(scope="module")
def hand_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("core")
    sources = source_dataset(root / "source")
    manifest = import_legacy(sources, "synthetic", "hand63_v1")
    split = create_split(manifest, seed=42)
    config = TrainConfig("hand63_v1", target_frames=12, rf={"n_estimators": 12}, augmentation_repeats=1,
                         threshold_candidates=(0.0, 0.6), minimum_validation_coverage=0.8)
    bundle = train(manifest, split, config, root / "models", "hand-test")
    return root, sources, manifest, split, config, bundle


def cli(*args: str):
    return subprocess.run([sys.executable, "-m", "slbridge", *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=Path(__file__).resolve().parents[1], timeout=90)


def test_real_cli_index_train_new_process_predict(tmp_path):
    sources = source_dataset(tmp_path / "source")
    manifest_path, split_path = tmp_path / "manifest.json", tmp_path / "split.json"
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, {"representation_id": "hand63_v1", "seed": 42,
                                   "rf": {"n_estimators": 12}, "augmentation_repeats": 1})
    before = {path.name: sha256_file(path) for path in sources.iterdir()}
    commands = [
        ("index", "--source", "legacy", "--root", str(sources), "--dataset-id", "acceptance", "--representation", "hand63_v1", "--output", str(manifest_path)),
        ("split", "--manifest", str(manifest_path), "--output", str(split_path)),
        ("train", "--manifest", str(manifest_path), "--split", str(split_path), "--config", str(config_path), "--output-dir", str(tmp_path / "models"), "--run-id", "cli-run"),
    ]
    for command in commands:
        result = cli(*command)
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)
    model_path = tmp_path / "models" / "cli-run"
    result = cli("predict", "--model", str(model_path), "--input", str(sources / "alpha_000.npy"), "--sample-id", "request-1")
    assert result.returncode == 0, result.stderr
    prediction = json.loads(result.stdout)
    assert prediction["prediction"]["label_id"] == "alpha"
    assert prediction["sample_id"] == "request-1"
    assert prediction["prediction"]["display_text"] == {}  # No translation still recognizes.
    assert prediction["sign_language"] is None
    assert prediction["score_type"] == "uncalibrated_class_probability"
    assert before == {path.name: sha256_file(path) for path in sources.iterdir()}


def test_train_inference_processing_parity(hand_bundle):
    _, sources, _, _, config, path = hand_bundle
    bundle = load_bundle(path)
    source = sources / "alpha_000.npy"
    from_config = prepare_input(source, InputSpec.for_representation(config.representation_id, config.target_frames))
    from_model = prepare_input(source, bundle.manifest.input_spec)
    np.testing.assert_array_equal(feature_vector(from_config), feature_vector(from_model))
    expected = bundle.backend.predict_proba([from_config])[0]
    result = Recognizer(bundle).recognize(source)
    assert result.prediction is not None
    np.testing.assert_array_equal(list(result.prediction.scores.values()), expected)
    assert result.time_range is None and result.quality["timestamps_known"] is False


def test_pose_bundle_retains_missing_mask_and_runs(tmp_path):
    sources = source_dataset(tmp_path / "source", 225)
    manifest = import_legacy(sources, "synthetic-pose", "pose_hands225_legacy")
    split = create_split(manifest)
    config = TrainConfig("pose_hands225_legacy", rf={"n_estimators": 8}, target_frames=10)
    path = train(manifest, split, config, tmp_path / "models", "pose-run")
    recognizer = Recognizer.load(path)
    assert recognizer.bundle.backend.feature_size == 10 * 225 * 2
    assert recognizer.recognize(sources / "beta_000.npy").prediction.label_id == "beta"
    with pytest.raises(ValueError, match="incompatible"):
        recognizer.recognize(sources / "beta_000.npy", representation_id="hand63_v1")


@pytest.mark.parametrize("array", [np.zeros((3, 225), dtype=float), np.zeros((3, 64), dtype=float),
                                   np.zeros((3, 63), dtype=int), np.full((3, 63), np.nan),
                                   np.full((3, 63), np.inf), np.zeros((0, 63), dtype=float),
                                   np.full((3, 63), -999.0), np.full((3, 63), 1e100)])
def test_strict_schema_rejects_invalid_inputs(tmp_path, array):
    path = tmp_path / "input.npy"
    np.save(path, array)
    with pytest.raises(ValueError):
        load_sequence(path, "hand63_v1")


def test_object_npy_cannot_deserialize(tmp_path):
    path = tmp_path / "object.npy"
    np.save(path, np.array([[{}]], dtype=object))
    with pytest.raises(ValueError):
        load_sequence(path, "hand63_v1")


def test_resampling_does_not_interpolate_missing_sentinel():
    array = np.ones((2, 225), dtype=np.float64)
    array[0, :3] = -999
    original = array.copy()
    prepared = prepare_array(array, InputSpec.for_representation("pose_hands225_legacy", 3))
    assert not prepared.mask[1, :3].any()
    assert np.all(prepared.values[1, :3] == 0)
    assert np.all(prepared.values[2, :3] == 1)
    np.testing.assert_array_equal(array, original)


def test_well_formed_low_quality_returns_rejected(hand_bundle):
    root, _, _, _, _, path = hand_bundle
    source = root / "one-frame.npy"
    np.save(source, np.zeros((1, 63), dtype=np.float32))
    result = Recognizer.load(path).recognize(source)
    assert result.status == "rejected" and result.prediction is None
    assert result.reason == "at_least_two_frames_required"
    with pytest.raises(InputRejected):
        prepare_array(np.full((4, 225), -999.0), InputSpec.for_representation("pose_hands225_legacy"))


def test_bundle_load_requires_no_original_training_data(hand_bundle, tmp_path):
    _, _, _, _, _, path = hand_bundle
    moved = tmp_path / "moved"
    shutil.copytree(path, moved)
    # root is deliberately absent. It is provenance, not an inference dependency.
    dataset_path = moved / "splits" / "dataset.json"
    value = read_json(dataset_path)
    value["root"] = str(tmp_path / "not-present")
    write_json_atomic(dataset_path, value)
    manifest_path = moved / "manifest.json"
    metadata = read_json(manifest_path)
    metadata["artifacts"]["splits/dataset.json"] = sha256_file(dataset_path)
    write_json_atomic(manifest_path, metadata)
    assert load_bundle(moved).manifest.run_id == "hand-test"


def test_corrupt_artifact_refused_before_weight_deserialization(hand_bundle, tmp_path, monkeypatch):
    from slbridge.models import RandomForestBackend
    _, _, _, _, _, path = hand_bundle
    moved = tmp_path / "tampered"
    shutil.copytree(path, moved)
    weights = moved / "weights" / "model.joblib"
    weights.write_bytes(weights.read_bytes() + b"corrupted")
    def forbidden_load(*args, **kwargs):
        pytest.fail("Weights must not load before bundle integrity check")
    monkeypatch.setattr(RandomForestBackend, "load", forbidden_load)
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(moved)


def test_mismatched_contract_and_label_order_refused(hand_bundle, tmp_path):
    _, _, _, _, _, path = hand_bundle
    moved = tmp_path / "mismatch"
    shutil.copytree(path, moved)
    config_path = moved / "resolved_config.json"
    value = read_json(config_path)
    value["input_spec"]["target_frames"] = 13
    write_json_atomic(config_path, value)
    manifest_path = moved / "manifest.json"
    metadata = read_json(manifest_path)
    metadata["artifacts"]["resolved_config.json"] = sha256_file(config_path)
    write_json_atomic(manifest_path, metadata)
    with pytest.raises(ValueError, match="InputSpec"):
        load_bundle(moved)


def test_model_version_no_overwrite_or_partial_publication(hand_bundle):
    root, _, manifest, split, config, path = hand_bundle
    before = {file.relative_to(path): sha256_file(file) for file in path.rglob("*") if file.is_file()}
    with pytest.raises(FileExistsError):
        train(manifest, split, config, root / "models", "hand-test")
    assert before == {file.relative_to(path): sha256_file(file) for file in path.rglob("*") if file.is_file()}
    assert not list((root / "models").glob("*.partial"))


def test_failed_training_publish_keeps_existing_versions(hand_bundle, monkeypatch):
    from slbridge.models import RandomForestBackend
    root, _, manifest, split, config, path = hand_bundle
    def failed_save(self, destination):
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(RandomForestBackend, "save", failed_save)
    with pytest.raises(OSError, match="interrupted"):
        train(manifest, split, config, root / "models", "failed-run")
    assert path.is_dir() and not (root / "models" / "failed-run").exists()
    assert not list((root / "models").glob("*.partial"))


def test_unknown_configuration_refused():
    with pytest.raises(ValueError, match="config"):
        TrainConfig.from_dict({"representation_id": "hand63_v1", "typo_seed": 42})
    with pytest.raises(ValueError):
        TrainConfig("hand63_v1", seed=True)


def test_training_config_copies_and_freezes_nested_parameters():
    parameters = {"n_estimators": 7}
    config = TrainConfig("hand63_v1", rf=parameters)
    parameters["n_estimators"] = 3
    assert config.rf["n_estimators"] == 7
    with pytest.raises(TypeError):
        config.rf["n_estimators"] = 3
    serialized = config.to_dict()
    serialized["rf"]["n_estimators"] = 1
    assert config.rf["n_estimators"] == 7


def test_core_import_and_help_do_not_load_runtime_dependencies():
    code = "import sys; import slbridge.cli, slbridge.training, slbridge.inference; assert not set(('numpy','sklearn','joblib','cv2','mediapipe','tkinter','torch')) & set(sys.modules)"
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    process = cli("--help")
    assert process.returncode == 0 and "predict" in process.stdout


def test_incompatible_cli_input_returns_structured_failure(hand_bundle):
    root, _, _, _, _, path = hand_bundle
    wrong = root / "wrong.npy"
    np.save(wrong, np.zeros((30, 225), dtype=np.float64))
    result = cli("predict", "--model", str(path), "--input", str(wrong))
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"] == "ValueError"


def test_cli_emits_portuguese_and_chinese_utf8_on_legacy_windows_encoding(hand_bundle, tmp_path):
    from slbridge._io import canonical_hash
    _, sources, _, _, _, original = hand_bundle
    model = tmp_path / "multilingual"
    shutil.copytree(original, model)
    labels_path = model / "labels.json"
    labels = read_json(labels_path)
    labels["labels"][0]["display_text"] = {"pt-BR": "Árvore", "zh-CN": "树"}
    write_json_atomic(labels_path, labels)
    manifest_path = model / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["labels_hash"] = canonical_hash(labels)
    manifest["artifacts"]["labels.json"] = sha256_file(labels_path)
    write_json_atomic(manifest_path, manifest)
    result = subprocess.run([sys.executable, "-m", "slbridge", "predict", "--model", str(model), "--input", str(sources / "alpha_000.npy")],
                            env={**os.environ, "PYTHONIOENCODING": "gbk"}, capture_output=True,
                            encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    display = json.loads(result.stdout)["prediction"]["display_text"]
    assert display == {"pt-BR": "Árvore", "zh-CN": "树"}
