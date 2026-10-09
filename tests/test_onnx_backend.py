"""Real ONNX graphs validate the adapter; mocks cover provider negotiation only."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

from slbridge._io import canonical_hash, read_json, resolve_resource, sha256_file, write_json_atomic
from slbridge.backends import backend_ids
from slbridge.bundle import load_bundle
from slbridge.contracts import InputSpec, ModelManifest
from slbridge.data import create_split, import_legacy
from slbridge.features import prepare_array
from slbridge.inference import Recognizer
from slbridge.onnx_backend import OnnxClassifierBackend, OnnxRuntimeConfig
from slbridge.runtime_options import RuntimeOptions
from slbridge.training import TrainConfig, train


def _config(layout="feature_vector", kind="logits", *, mask=None, mask_dtype="float32"):
    return {"schema_version": 1, "input_layout": layout, "values_input": "values",
            "mask_input": mask, "mask_dtype": mask_dtype, "output_name": "scores", "output_kind": kind}


def _graph(path, spec=None, *, kind="logits", width=None, sequence=False,
           mask_dtype=None, zipmap=False, invalid_constant=None, fixed_batch=None):
    spec = spec or InputSpec.for_representation("hand63_v1", 2)
    inputs, initializers, nodes = [], [], []
    if sequence:
        shape = [fixed_batch, spec.target_frames, spec.width]
        inputs.append(helper.make_tensor_value_info("values", TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("ReduceMean", ["values"], ["value_mean"], axes=[1, 2], keepdims=0))
        if mask_dtype:
            inputs.append(helper.make_tensor_value_info("mask", TensorProto.BOOL if mask_dtype == "bool" else TensorProto.FLOAT, shape))
            mask = "mask"
            if mask_dtype == "bool":
                nodes.append(helper.make_node("Cast", ["mask"], ["mask_float"], to=TensorProto.FLOAT))
                mask = "mask_float"
            nodes.append(helper.make_node("ReduceMean", [mask], ["second_mean"], axes=[1, 2], keepdims=0))
        else:
            nodes.append(helper.make_node("Neg", ["value_mean"], ["second_mean"]))
        initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axis"))
        nodes.extend([helper.make_node("Unsqueeze", ["value_mean", "axis"], ["first"]),
                      helper.make_node("Unsqueeze", ["second_mean", "axis"], ["second"]),
                      helper.make_node("Concat", ["first", "second"], ["logits"], axis=1)])
    else:
        width = width or spec.feature_size
        inputs.append(helper.make_tensor_value_info("values", TensorProto.FLOAT, [fixed_batch, width]))
        coefficients = np.zeros((width, 2), dtype=np.float32)
        coefficients[0] = [1, -1]
        initializers.append(numpy_helper.from_array(coefficients, "weights"))
        nodes.append(helper.make_node("MatMul", ["values", "weights"], ["logits"]))
    if invalid_constant is not None:
        nodes.append(helper.make_node("Constant", [], ["scores"], value=numpy_helper.from_array(np.asarray([invalid_constant], dtype=np.float32))))
    elif kind == "probabilities":
        nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities" if zipmap else "scores"], axis=1))
    else:
        nodes.append(helper.make_node("Identity", ["logits"], ["scores"]))
    if zipmap:
        nodes.append(helper.make_node("ZipMap", ["probabilities"], ["scores"], domain="ai.onnx.ml", classlabels_strings=["A", "B"]))
        output_type = helper.make_sequence_type_proto(helper.make_map_type_proto(TensorProto.STRING, helper.make_tensor_type_proto(TensorProto.FLOAT, [])))
        output = helper.make_value_info("scores", output_type)
    else:
        output = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [fixed_batch, 2])
    model = helper.make_model(helper.make_graph(nodes, "tiny_classifier", inputs, [output], initializers),
                              opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return spec


def _load(path, spec, config=None, options=None):
    return OnnxClassifierBackend.load(path, expected_classes=("A", "B"), expected_input_spec=spec,
                                      runtime_config=config or _config(), options=options or RuntimeOptions(device="cpu"))


@pytest.mark.parametrize("kind", ["logits", "probabilities"])
def test_real_dense_probabilities_and_stable_logits_reuse_session(tmp_path, monkeypatch, kind):
    path = tmp_path / "model.onnx"
    spec = _graph(path, kind=kind)
    created = []
    factory = ort.InferenceSession

    def record(*args, **kwargs):
        created.append(kwargs)
        return factory(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", record)
    backend = _load(path, spec, _config(kind=kind))
    backend.warmup()
    samples = [prepare_array(np.full((2, 63), value, dtype=np.float32), spec) for value in [10000, -10000, 0]]
    probabilities = backend.predict_proba(samples)
    np.testing.assert_array_equal(probabilities, [[1, 0], [0, 1], [.5, .5]])
    np.testing.assert_array_equal(probabilities, backend.predict_proba(samples))
    assert len(created) == 1
    assert created[0]["sess_options"].intra_op_num_threads == 1
    assert backend.runtime_info["providers"] == ["CPUExecutionProvider"]
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.predict_proba(samples)


@pytest.mark.parametrize("mask_dtype", [None, "float32", "bool"])
def test_real_sequence_ntc_and_explicit_masks(tmp_path, mask_dtype):
    spec = InputSpec.for_representation("pose_hands225_legacy", 3)
    path = tmp_path / "sequence.onnx"
    _graph(path, spec, sequence=True, mask_dtype=mask_dtype)
    backend = _load(path, spec, _config("sequence", mask="mask" if mask_dtype else None,
                                     mask_dtype=mask_dtype or "float32"))
    backend.warmup()
    raw = np.full((3, 225), 2, dtype=np.float32)
    raw[:, :45] = -999
    sample = prepare_array(raw, spec)
    values = backend.predict_proba([sample])
    logits = np.array([1.6, .8 if mask_dtype else -1.6], dtype=np.float64)
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    np.testing.assert_allclose(values[0], expected, atol=1e-7)
    with pytest.raises(ValueError, match="InputSpec"):
        backend.predict_proba([replace(sample, values=np.zeros((3, 224), dtype=np.float32))])


def test_real_graph_refuses_incompatible_input_shape_and_zipmap(tmp_path):
    path = tmp_path / "bad_shape.onnx"
    spec = _graph(path, width=63)
    with pytest.raises(ValueError, match="shape disagrees"):
        _load(path, spec)
    path = tmp_path / "zipmap.onnx"
    _graph(path, kind="probabilities", zipmap=True)
    with pytest.raises(ValueError, match="ZipMap"):
        _load(path, spec, _config(kind="probabilities"))


@pytest.mark.parametrize("sequence", [False, True])
def test_graph_must_support_single_clip_batches(tmp_path, sequence):
    path = tmp_path / "batch2.onnx"
    spec = _graph(path, sequence=sequence, fixed_batch=2)
    with pytest.raises(ValueError, match="single-clip batch size 1"):
        _load(path, spec, _config("sequence" if sequence else "feature_vector"))
    path = tmp_path / "batch1.onnx"
    _graph(path, sequence=sequence, fixed_batch=1)
    backend = _load(path, spec, _config("sequence" if sequence else "feature_vector"))
    backend.warmup()
    sample = prepare_array(np.zeros((2, 63), dtype=np.float32), spec)
    np.testing.assert_array_equal(backend.predict_proba([sample]), [[.5, .5]])
    with pytest.raises(ValueError, match="fixed batch size"):
        backend.predict_proba([sample, sample])


def test_external_unhashed_weights_are_not_read(tmp_path, monkeypatch):
    path = tmp_path / "external.onnx"
    spec = _graph(path)
    model = onnx.load(path)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="untracked_weights.bin", size_threshold=0)
    assert (tmp_path / "untracked_weights.bin").is_file()
    # Serialized bytes are not enough: ORT can resolve this file against CWD.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: pytest.fail("External tensors must be rejected before ORT session creation"))
    with pytest.raises(ValueError, match="Cannot create ONNX.*external tensor data"):
        _load(path, spec)


@pytest.mark.parametrize("location", ["tensor_attribute", "nested_graph", "sparse_values", "sparse_indices",
                                      "function_body", "training_graph"])
@pytest.mark.parametrize("external_flag", [False, True])
def test_external_tensor_references_in_nested_messages_are_rejected_before_ort(tmp_path, monkeypatch, location, external_flag):
    path = tmp_path / "nested_external.onnx"
    spec = _graph(path)
    model = onnx.load(path)
    tensor = onnx.TensorProto()
    tensor.name, tensor.data_type = "unhashed", TensorProto.FLOAT
    tensor.dims.append(1)
    if external_flag:
        tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="external.bin")
    # The file exists where ORT would resolve it. Some deliberately unused
    # messages below need not be executable: every reference must be rejected.
    (tmp_path / "external.bin").write_bytes(np.zeros(1, dtype=np.float32).tobytes())
    if location == "tensor_attribute":
        node = model.graph.node.add(op_type="Constant")
        node.attribute.add(name="value", type=onnx.AttributeProto.TENSOR).t.CopyFrom(tensor)
    elif location == "nested_graph":
        node = model.graph.node.add(op_type="If")
        branch = node.attribute.add(name="then_branch", type=onnx.AttributeProto.GRAPH).g
        branch.initializer.add().CopyFrom(tensor)
    elif location in {"sparse_values", "sparse_indices"}:
        sparse = model.graph.sparse_initializer.add()
        getattr(sparse, "values" if location == "sparse_values" else "indices").CopyFrom(tensor)
    elif location == "function_body":
        function = model.functions.add(name="unused", domain="local")
        node = function.node.add(op_type="Constant")
        node.attribute.add(name="value", type=onnx.AttributeProto.TENSOR).t.CopyFrom(tensor)
    else:
        model.training_info.add().initialization.initializer.add().CopyFrom(tensor)
    path.write_bytes(model.SerializeToString())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: pytest.fail("Nested external tensor reached native ORT"))
    with pytest.raises(ValueError, match="Cannot create ONNX.*external tensor data"):
        _load(path, spec)


def test_external_data_location_alone_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "external_flag_only.onnx"
    spec = _graph(path)
    model = onnx.load(path)
    model.graph.initializer[0].data_location = TensorProto.EXTERNAL
    path.write_bytes(model.SerializeToString())
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: pytest.fail("EXTERNAL flag reached native ORT"))
    with pytest.raises(ValueError, match="Cannot create ONNX.*external tensor data"):
        _load(path, spec)


def test_model_parse_failure_releases_cuda_dll_handles_before_session(tmp_path, monkeypatch):
    from slbridge.onnx_backend import create_onnx_session

    path = tmp_path / "invalid.onnx"
    path.write_bytes(b"\x80")  # Truncated protobuf varint.
    handles = []

    class Handle:
        closed = False

        def close(self):
            self.closed = True

    def add_directory(directory):
        handle = Handle()
        handles.append(handle)
        return handle

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)
    monkeypatch.setattr(os, "add_dll_directory", add_directory, raising=False)
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: pytest.fail("Invalid protobuf reached native ORT"))
    with pytest.raises(ValueError, match="Cannot create ONNX.*invalid model protobuf"):
        create_onnx_session(path, options=RuntimeOptions(cuda_dll_dirs=(str(tmp_path.resolve()),)), backend_id="onnx_ctc")
    assert len(handles) == 1 and handles[0].closed


@pytest.mark.parametrize("constant", [[-1, 2], [.2, .2], [np.nan, 0], [np.inf, 0]])
def test_real_invalid_output_is_refused_during_warmup(tmp_path, constant):
    path = tmp_path / "invalid.onnx"
    spec = _graph(path, invalid_constant=constant)
    backend = _load(path, spec, _config(kind="probabilities"))
    with pytest.raises(ValueError, match="invalid"):
        backend.warmup()


def test_binding_schema_is_complete_and_feature_masks_are_unambiguous():
    with pytest.raises(ValueError, match="complete"):
        OnnxRuntimeConfig.from_dict({"schema_version": 1})
    with pytest.raises(ValueError, match="separate mask"):
        OnnxRuntimeConfig.from_dict(_config(mask="mask"))
    with pytest.raises(ValueError, match="version"):
        OnnxRuntimeConfig.from_dict({**_config(), "schema_version": True})


def test_auto_reports_actual_cpu_fallback_and_strict_cuda_refuses_it(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    spec = _graph(path)
    factory = ort.InferenceSession
    requested = []
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)

    def initialize_cpu(*args, **kwargs):
        requested.append(kwargs["providers"])
        return factory(*args, **{**kwargs, "providers": ["CPUExecutionProvider"]})

    monkeypatch.setattr(ort, "InferenceSession", initialize_cpu)
    backend = _load(path, spec, options=RuntimeOptions(device="auto", device_id=2, gpu_mem_limit_mb=128))
    assert backend.runtime_info["providers"] == ["CPUExecutionProvider"]
    assert "actual ONNX session" in backend.runtime_info["fallback_reason"]
    assert requested[0][0] == ("CUDAExecutionProvider", {"device_id": "2", "gpu_mem_limit": str(128 * 1024 * 1024)})
    backend.warmup()
    with pytest.raises(ValueError, match="actual ONNX session"):
        _load(path, spec, options=RuntimeOptions(device="cuda"))


@pytest.mark.parametrize("backend_id", ["onnx_classifier_v1", "onnx_ctc"])
def test_ctc_disables_cuda_tf32_without_changing_classifier_provider_options(tmp_path, monkeypatch, backend_id):
    from slbridge.onnx_backend import create_onnx_session

    path = tmp_path / "model.onnx"
    _graph(path)
    factory = ort.InferenceSession
    requested = []
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)

    def initialize_cpu(*args, **kwargs):
        requested.append(kwargs["providers"])
        return factory(*args, **{**kwargs, "providers": ["CPUExecutionProvider"]})

    monkeypatch.setattr(ort, "InferenceSession", initialize_cpu)
    session, info, handles = create_onnx_session(path, options=RuntimeOptions(device="auto"), backend_id=backend_id)
    expected_options = {"device_id": "0", "gpu_mem_limit": str(2048 * 1024 * 1024)}
    if backend_id == "onnx_ctc":
        expected_options["use_tf32"] = "0"
        expected_options["arena_extend_strategy"] = "kSameAsRequested"
        assert info["precision"] == "float32_tf32_disabled" and info["use_tf32"] is False
        assert info["cuda_arena_extend_strategy"] is None and info["cuda_arena_shrinkage"] is None
    else:
        assert "precision" not in info and "use_tf32" not in info
    assert requested == [[("CUDAExecutionProvider", expected_options), "CPUExecutionProvider"]]
    assert handles == []


def test_cuda_unavailable_fails_before_session_creation_and_auto_records_reason(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    spec = _graph(path)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    backend = _load(path, spec, options=RuntimeOptions(device="auto"))
    assert "unavailable" in backend.runtime_info["fallback_reason"]
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: pytest.fail("strict CUDA must fail before session creation"))
    with pytest.raises(ValueError, match="unavailable"):
        _load(path, spec, options=RuntimeOptions(device="cuda"))


def test_auto_cuda_initialization_error_is_explicit(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    spec = _graph(path)
    factory = ort.InferenceSession
    calls = []
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)

    def create(*args, **kwargs):
        calls.append(kwargs["providers"])
        if len(calls) == 1:
            raise RuntimeError("simulated missing cuDNN DLL")
        return factory(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", create)
    backend = _load(path, spec, options=RuntimeOptions(device="auto"))
    assert "simulated missing cuDNN" in backend.runtime_info["fallback_reason"]
    assert calls[1] == ["CPUExecutionProvider"]
    backend.warmup()


def test_process_local_dll_directories_are_retained_and_released_without_path_leaks(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    spec = _graph(path)
    factory = ort.InferenceSession
    preloads, handles = [], []
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: preloads.append(kwargs))

    class Handle:
        closed = False

        def close(self):
            self.closed = True

    def add_directory(directory):
        handle = Handle()
        handles.append(handle)
        return handle

    monkeypatch.setattr(os, "add_dll_directory", add_directory, raising=False)
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: factory(*args, **{**kwargs, "providers": ["CPUExecutionProvider"]}))
    options = RuntimeOptions(cuda_dll_dirs=(str(tmp_path.resolve()),))
    backend = _load(path, spec, options=options)
    assert preloads == [{}, {"directory": str(tmp_path.resolve())}]
    assert len(handles) == 1 and not handles[0].closed
    assert backend.runtime_info["cuda_dll_directories_configured"] == 1
    assert str(tmp_path) not in json.dumps(backend.runtime_info)
    backend.close()
    assert handles[0].closed
    with pytest.raises(ValueError, match="existing absolute directory"):
        _load(path, spec, options=RuntimeOptions(cuda_dll_dirs=(str(tmp_path / "missing"),)))


@pytest.fixture(scope="module")
def rf_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("rf-export")
    sources = root / "sources"
    sources.mkdir()
    rng = np.random.default_rng(3)
    for label, center in [("A", -1), ("B", 1)]:
        for i in range(9):
            np.save(sources / f"{label}_{i:03}.npy", rng.normal(center, .05, size=(4, 63)).astype(np.float32))
    dataset = import_legacy(sources, "export-synthetic", "hand63_v1")
    split = create_split(dataset)
    path = train(dataset, split, TrainConfig("hand63_v1", target_frames=2, rf={"n_estimators": 5}), root / "models", "native-rf")
    by_id = {record.sample_id: record for record in dataset.samples}
    return path, [resolve_resource(sources, by_id[identifier].resource) for identifier in split.test], resolve_resource(sources, by_id[split.train[0]].resource)


def test_export_real_rf_heldout_parity_and_backward_manifest(tmp_path, rf_bundle):
    pytest.importorskip("skl2onnx")
    from slbridge.export_onnx import export_rf_bundle

    path, heldout, _ = rf_bundle
    before = {file.relative_to(path): sha256_file(file) for file in path.rglob("*") if file.is_file()}
    assert backend_ids() == ("random_forest_v1", "onnx_classifier_v1")
    old = read_json(path / "manifest.json")
    assert "runtime_config" not in old
    assert ModelManifest.from_dict(old).runtime_config is None
    result_path = export_rf_bundle(path, tmp_path / "onnx-export", "onnx-export", heldout)
    rf = Recognizer.load(path, options=RuntimeOptions(device="cpu"))
    exported = Recognizer.load(result_path, options=RuntimeOptions(device="cpu"))
    exported.warmup()
    for clip in heldout:
        expected = rf.recognize(clip, sample_id="parity")
        result, timing = exported.recognize_profiled(clip, sample_id="parity")
        assert result.status == expected.status
        assert result.prediction.label_id == expected.prediction.label_id
        np.testing.assert_allclose(list(result.prediction.scores.values()), list(expected.prediction.scores.values()), atol=1e-6)
        assert set(timing) == {"preprocess_ms", "inference_ms", "total_ms"}
        assert timing["total_ms"] >= timing["preprocess_ms"] + timing["inference_ms"] >= 0
    manifest = read_json(result_path / "manifest.json")
    assert manifest["provenance"]["onnx_export"]["verification"]["samples"] == len(heldout)
    assert "weights/model.onnx" in manifest["artifacts"] and "weights/model.joblib" not in manifest["artifacts"]
    assert before == {file.relative_to(path): sha256_file(file) for file in path.rglob("*") if file.is_file()}
    with pytest.raises(FileExistsError):
        export_rf_bundle(path, result_path, "onnx-export", heldout)
    exported.close()
    with pytest.raises(RuntimeError, match="closed"):
        exported.warmup()


def test_export_refuses_train_random_and_failed_parity_never_publishes(tmp_path, rf_bundle, monkeypatch):
    pytest.importorskip("skl2onnx")
    from slbridge.export_onnx import export_rf_bundle

    path, heldout, training = rf_bundle
    destination = tmp_path / "invalid-export"
    with pytest.raises(ValueError, match="held-out"):
        export_rf_bundle(path, destination, "bad", [training])
    with pytest.raises(ValueError, match="verification input"):
        export_rf_bundle(path, destination, "bad", [])
    prediction = OnnxClassifierBackend.predict_proba

    def wrong(self, samples):
        actual = prediction(self, samples)
        return actual[:, ::-1] if len(samples) > 1 else actual

    monkeypatch.setattr(OnnxClassifierBackend, "predict_proba", wrong)
    with pytest.raises(ValueError, match="parity failed"):
        export_rf_bundle(path, destination, "bad", heldout)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_onnx_hashes_frozen_metadata_and_bindings_checked_before_loading(tmp_path, rf_bundle, monkeypatch):
    pytest.importorskip("skl2onnx")
    from slbridge.export_onnx import export_rf_bundle

    path, heldout, _ = rf_bundle
    exported = export_rf_bundle(path, tmp_path / "original", "integrity", heldout)
    corrupt = tmp_path / "corrupt"
    shutil.copytree(exported, corrupt)
    weights = corrupt / "weights/model.onnx"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    monkeypatch.setattr(OnnxClassifierBackend, "load", lambda *args, **kwargs: pytest.fail("do not load corrupt weights"))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(corrupt)
    mismatch = tmp_path / "mismatch"
    shutil.copytree(exported, mismatch)
    config = read_json(mismatch / "resolved_config.json")
    config["runtime_config"]["values_input"] = "wrong_name"
    write_json_atomic(mismatch / "resolved_config.json", config)
    manifest = read_json(mismatch / "manifest.json")
    manifest["artifacts"]["resolved_config.json"] = sha256_file(mismatch / "resolved_config.json")
    write_json_atomic(mismatch / "manifest.json", manifest)
    with pytest.raises(ValueError, match="bindings disagree"):
        load_bundle(mismatch)


def test_rf_strict_cuda_fails_before_deserializing(rf_bundle, monkeypatch):
    from slbridge.models import RandomForestBackend

    path, _, _ = rf_bundle
    monkeypatch.setattr(RandomForestBackend, "load", lambda *args, **kwargs: pytest.fail("no RF deserialization for strict CUDA"))
    with pytest.raises(ValueError, match="supports CPU only"):
        Recognizer.load(path, options=RuntimeOptions(device="cuda"))


def test_onnx_modules_and_export_help_remain_optional():
    code = "import sys; import slbridge.inference, slbridge.export_onnx, slbridge.onnx_backend; assert not {'onnxruntime','onnx','skl2onnx','numpy','sklearn'} & set(sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "slbridge.export_onnx", "--help"], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0 and "--verification-manifest" in result.stdout


def test_normal_export_cli_and_export_from_existing_rf_process(tmp_path, rf_bundle):
    pytest.importorskip("skl2onnx")
    path, heldout, _ = rf_bundle
    verification = tmp_path / "verification.json"
    write_json_atomic(verification, {"schema_version": 1, "samples": [{"path": str(clip)} for clip in heldout]})
    destination = tmp_path / "cli-export"
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-m", "slbridge.export_onnx",
                             "--model", str(path), "--output", str(destination), "--run-id", "cli-export",
                             "--verification-manifest", str(verification)], capture_output=True, text=True,
                            encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["backend"] == "onnx_classifier_v1"
    assert load_bundle(destination).manifest.run_id == "cli-export"
    destination = tmp_path / "rf-first-export"
    code = f"""from slbridge.inference import Recognizer
rf = Recognizer.load({str(path)!r})
rf.warmup()
from slbridge.export_onnx import export_rf_bundle
result = export_rf_bundle({str(path)!r}, {str(destination)!r}, 'rf-first-export', {[str(clip) for clip in heldout]!r})
print(result)
rf.close()
"""
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code], capture_output=True,
                            text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stderr
    assert load_bundle(destination).manifest.run_id == "rf-first-export"


def test_mixed_backend_bootstrap_real_rf_then_onnx_in_fresh_process(tmp_path, rf_bundle):
    pytest.importorskip("skl2onnx")
    from slbridge.export_onnx import export_rf_bundle

    path, heldout, _ = rf_bundle
    destination = export_rf_bundle(path, tmp_path / "managed-onnx", "managed-onnx", heldout)
    code = f"""import numpy
from slbridge.backends import initialize_runtime_libraries
assert initialize_runtime_libraries()['onnxruntime_error'] is None
from slbridge.inference import Recognizer
rf = Recognizer.load({str(path)!r})
rf.warmup()
rf.close()
onnx = Recognizer.load({str(destination)!r})
onnx.warmup()
assert onnx.recognize({str(heldout[0])!r}).prediction is not None
onnx.close()
"""
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code], capture_output=True,
                            text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stderr
