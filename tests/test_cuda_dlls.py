"""New cuDNN engines must be available before the first CUDA kernel runs."""
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper

from slbridge import onnx_backend
from slbridge.runtime_options import RuntimeOptions


def identity_graph(path):
    value = helper.make_tensor_value_info("video", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    model = helper.make_model(helper.make_graph([helper.make_node("Identity", ["video"], ["output"])],
                                               "dll-seam", [value], [output]),
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path)


@pytest.mark.parametrize("device,backend_id", [("cuda", "onnx_ctc"), ("cpu", "onnx_ctc"),
                                               ("cuda", "onnx_classifier_v1")])
def test_cuda_kernel_loads_new_cudnn_engine_from_same_runtime(tmp_path, monkeypatch, device, backend_id):
    import ctypes

    directory = tmp_path / "actual-loaded-cudnn"
    directory.mkdir()
    engine = directory / "cudnn_engines_tensor_ir64_9.dll"
    engine.write_bytes(b"synthetic DLL placeholder; ctypes is intercepted")
    (directory / "unrelated.dll").write_bytes(b"must not load")
    path = tmp_path / "model.onnx"
    identity_graph(path)
    loaded = []
    # The actual Windows DLL lookup is covered by the real GPU acceptance;
    # this seam binds extra engines to the already loaded cuDNN version.
    monkeypatch.setattr(onnx_backend, "_loaded_cudnn_directory", lambda: directory, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: loaded.append(Path(name)))
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

        def disable_fallback(self):
            pass

        def run(self):
            if device == "cuda" and backend_id == "onnx_ctc" and engine not in loaded:
                raise RuntimeError("Could not locate cudnn_engines_tensor_ir64_9.dll")
            return "kernel_executed"

    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: Session())
    session, info, _ = onnx_backend.create_onnx_session(path, options=RuntimeOptions(device=device), backend_id=backend_id)
    assert session.run() == "kernel_executed"
    expected = [engine] if device == "cuda" and backend_id == "onnx_ctc" else []
    assert loaded == expected
    assert info["cudnn_engine_libraries_preloaded"] == len(expected)


def test_no_loaded_cudnn_does_not_load_unrelated_installations(monkeypatch):
    import ctypes

    monkeypatch.setattr(onnx_backend, "_loaded_cudnn_directory", lambda: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: pytest.fail("No selected cuDNN distribution"))
    assert onnx_backend._preload_cudnn_engines() == 0


@pytest.mark.parametrize("device", ["cuda", "auto"])
def test_engine_load_failure_is_explicit_and_releases_search_directories(tmp_path, monkeypatch, device):
    import os

    path = tmp_path / "model.onnx"
    identity_graph(path)
    handles = []

    class Handle:
        closed = False

        def close(self):
            self.closed = True

    def add_directory(directory):
        handle = Handle()
        handles.append(handle)
        return handle

    def fail_engines():
        raise OSError("new cuDNN engine dependency missing")

    monkeypatch.setattr(os, "add_dll_directory", add_directory, raising=False)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: None)
    monkeypatch.setattr(onnx_backend, "_preload_cudnn_engines", fail_engines)
    options = RuntimeOptions(device=device, cuda_dll_dirs=(str(tmp_path.resolve()),))
    if device == "cuda":
        monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: pytest.fail("Strict CUDA must reject missing engine"))
        with pytest.raises(ValueError, match="Cannot preload CUDA/cuDNN.*engine dependency missing"):
            onnx_backend.create_onnx_session(path, options=options, backend_id="onnx_ctc")
    else:
        session, info, dll_handles = onnx_backend.create_onnx_session(path, options=options, backend_id="onnx_ctc")
        assert session.get_providers() == ["CPUExecutionProvider"]
        assert "engine dependency missing" in info["fallback_reason"]
        assert info["cudnn_engine_libraries_preloaded"] == 0
        assert dll_handles == []
    assert len(handles) == 1 and handles[0].closed
