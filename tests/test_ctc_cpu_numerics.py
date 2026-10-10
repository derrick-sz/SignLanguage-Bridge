"""Exercise the production session boundary with real subnormal Conv weights."""
from hashlib import sha256

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

from slbridge.onnx_backend import create_onnx_session
from slbridge.runtime_options import RuntimeOptions


def conv_graph(path, *, shared=False):
    # Test signed zero, positive/negative subnormal, and the smallest normal.
    weights = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x3f800000],
                       dtype=np.uint32).view(np.float32).reshape(1, 1, 1, 1, 6)
    nodes = [helper.make_node("Conv", ["video", "weights"], ["output"])]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1, 1])]
    if shared:
        nodes.append(helper.make_node("Identity", ["weights"], ["original_weights"]))
        outputs.append(helper.make_tensor_value_info("original_weights", TensorProto.FLOAT, list(weights.shape)))
    model = helper.make_model(helper.make_graph(nodes, "subnormal_conv", [
        helper.make_tensor_value_info("video", TensorProto.FLOAT, [1, 1, 1, 1, 6])], outputs,
        [numpy_helper.from_array(weights, "weights")]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path)
    return weights


def capture_sessions(monkeypatch):
    factory = ort.InferenceSession
    received = []

    def capture(model, *args, **kwargs):
        received.append(model)
        return factory(model, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", capture)
    return received


def conv_weight_bits(model_bytes):
    model = onnx.load_model_from_string(model_bytes)
    weight_name = next(node.input[1] for node in model.graph.node if node.op_type == "Conv")
    weight = next(value for value in model.graph.initializer if value.name == weight_name)
    return numpy_helper.to_array(weight).view(np.uint32).reshape(-1)


def test_cpu_ctc_flushes_only_subnormal_conv_weights_without_rewriting_bundle(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    original = conv_graph(path)
    before = path.read_bytes()
    received = capture_sessions(monkeypatch)
    session, info, handles = create_onnx_session(path, options=RuntimeOptions(device="cpu"), backend_id="onnx_ctc")
    expected = original.view(np.uint32).reshape(-1).copy()
    expected[2:4] = [0, 0x80000000]
    np.testing.assert_array_equal(conv_weight_bits(received[0]), expected)
    assert path.read_bytes() == before
    numerics = info["cpu_numerics"]
    assert numerics["policy"] == "flush_conv_weights_v1"
    assert numerics["flushed_values"] == 2
    assert numerics["source_sha256"] == sha256(before).hexdigest()
    assert numerics["execution_sha256"] == sha256(received[0]).hexdigest()
    assert handles == []
    output = session.run(None, {"video": np.ones((1, 1, 1, 1, 6), dtype=np.float32)})[0]
    np.testing.assert_array_equal(output, np.ones((1, 1, 1, 1, 1), dtype=np.float32))


def test_shared_initializer_keeps_nonconv_consumer_values(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    weights = conv_graph(path, shared=True)
    received = capture_sessions(monkeypatch)
    session, _, _ = create_onnx_session(path, options=RuntimeOptions(device="cpu"), backend_id="onnx_ctc")
    assert conv_weight_bits(received[0])[2] == 0
    outputs = session.run(None, {"video": np.ones((1, 1, 1, 1, 6), dtype=np.float32)})
    np.testing.assert_array_equal(outputs[1].view(np.uint32), weights.view(np.uint32))


def test_nested_graph_capture_keeps_original_weights(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    weights = conv_graph(path)
    model = onnx.load(path)
    info = helper.make_tensor_value_info("captured", TensorProto.FLOAT, list(weights.shape))
    branch = helper.make_graph([helper.make_node("Identity", ["weights"], ["captured"])],
                               "capture_parent_weight", [], [info])
    model.graph.initializer.append(numpy_helper.from_array(np.array(True), "condition"))
    model.graph.node.append(helper.make_node("If", ["condition"], ["original_weights"],
                                             then_branch=branch, else_branch=branch))
    model.graph.output.append(helper.make_tensor_value_info("original_weights", TensorProto.FLOAT, list(weights.shape)))
    onnx.save(model, path)
    received = capture_sessions(monkeypatch)
    session, _, _ = create_onnx_session(path, options=RuntimeOptions(device="cpu"), backend_id="onnx_ctc")
    assert conv_weight_bits(received[0])[2] == 0
    outputs = session.run(None, {"video": np.ones((1, 1, 1, 1, 6), dtype=np.float32)})
    np.testing.assert_array_equal(outputs[1].view(np.uint32), weights.view(np.uint32))


@pytest.mark.parametrize("backend,policy", [("onnx_classifier_v1", "flush_to_zero"), ("onnx_ctc", "preserve")])
def test_classifier_and_preserve_policy_keep_exact_execution_bytes(tmp_path, monkeypatch, backend, policy):
    path = tmp_path / "model.onnx"
    conv_graph(path)
    received = capture_sessions(monkeypatch)
    _, info, _ = create_onnx_session(path, options=RuntimeOptions(device="cpu", cpu_conv_subnormals=policy),
                                     backend_id=backend)
    assert received == [path.read_bytes()]
    if backend == "onnx_ctc":
        assert info["cpu_numerics"]["policy"] == "preserve"
        assert info["cpu_numerics"]["flushed_values"] == 0


@pytest.mark.parametrize("fallback", ["initialization_error", "implicit_cpu"])
def test_gpu_to_cpu_fallback_uses_sanitized_cpu_execution(tmp_path, monkeypatch, fallback):
    path = tmp_path / "model.onnx"
    conv_graph(path)
    factory = ort.InferenceSession
    received = []
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)

    def initialize(model, *args, **kwargs):
        received.append(model)
        if len(received) == 1 and fallback == "initialization_error":
            raise RuntimeError("test CUDA initialization failed")
        return factory(model, *args, **{**kwargs, "providers": ["CPUExecutionProvider"]})

    monkeypatch.setattr(ort, "InferenceSession", initialize)
    session, info, _ = create_onnx_session(path, options=RuntimeOptions(device="auto"), backend_id="onnx_ctc")
    assert received[0] == path.read_bytes()
    assert conv_weight_bits(received[-1])[2] == 0
    assert info["device"] == "cpu" and info["fallback_reason"]
    assert info["cpu_numerics"]["flushed_values"] == 2
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_actual_cuda_keeps_source_bytes(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    conv_graph(path)
    received = []

    class CudaSession:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def disable_fallback(self):
            pass

    def initialize(model, **kwargs):
        received.append(model)
        return CudaSession()

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)
    monkeypatch.setattr(ort, "InferenceSession", initialize)
    _, info, _ = create_onnx_session(path, options=RuntimeOptions(device="cuda"), backend_id="onnx_ctc")
    assert received == [path.read_bytes()]
    assert "cpu_numerics" not in info


@pytest.mark.parametrize("policy", [None, True, [], "approximate"])
def test_unknown_numerical_policy_is_refused(policy):
    with pytest.raises(ValueError, match="cpu_conv_subnormals"):
        RuntimeOptions(cpu_conv_subnormals=policy)
