"""HTTP boundary tests, independent of model downloads and native runtimes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from slbridge import api, api_cli
from slbridge.api import create_app
from slbridge.contracts import ClassificationPrediction, InputSpec, RecognitionResult
from slbridge.runtime import InferenceResponse, ModelManager, ModelUnavailable, RuntimeBusy
from slbridge.runtime_options import RuntimeOptions


def descriptor(model_id="model-a", generation=1):
    return {"model_id": model_id, "generation": generation, "backend": "random_forest_v1",
            "sign_language": "Libras", "input_spec": {"representation_id": "hand63_v1"},
            "capabilities": ["clip"], "labels": {"hello": {"pt": "Olá"}},
            "runtime": {"device": "cpu", "providers": []},
            "options": {key: value for key, value in RuntimeOptions().to_dict().items() if key != "cuda_dll_dirs"},
            "warmup_ms": 0}


class FakeManager:
    def __init__(self):
        self.active = descriptor()
        self.previous = None
        self.generation = 1
        self.closed = 0
        self.paths = []
        self.requests = []
        self.switches = []
        self.failure = None
        self.switch_failure = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self._gate = threading.Lock()
        self.native_saw_event_loop = None

    def status(self):
        return {"state": "ready" if self.active else "empty", "generation": self.generation,
                "active": self.active, "previous": self.previous, "error": None}

    def describe(self):
        if self.active is None:
            raise ModelUnavailable("No model is loaded")
        return self.active

    def switch(self, path, *, options=None):
        if not self._gate.acquire(blocking=False):
            raise RuntimeBusy("Runtime is busy")
        try:
            if self.switch_failure:
                raise self.switch_failure
            self.switches.append((Path(path), options))
            self.previous = self.active
            self.generation += 1
            self.active = descriptor(Path(path).name, self.generation)
            return self.status()
        finally:
            self._gate.release()

    def rollback(self):
        if not self._gate.acquire(blocking=False):
            raise RuntimeBusy("Runtime is busy")
        try:
            if self.previous is None:
                raise ModelUnavailable("No previous model is available")
            self.active, self.previous = self.previous, self.active
            self.generation += 1
            self.active = {**self.active, "generation": self.generation}
            return self.status()
        finally:
            self._gate.release()

    def infer_file(self, path, request):
        if not self._gate.acquire(blocking=False):
            raise RuntimeBusy("Runtime is busy")
        try:
            try:
                asyncio.get_running_loop()
                self.native_saw_event_loop = True
            except RuntimeError:
                self.native_saw_event_loop = False
            self.paths.append(Path(path))
            self.requests.append(request)
            assert Path(path).read_bytes() == b"input-bytes"
            self.started.set()
            assert self.release.wait(5)
            if self.failure:
                raise self.failure
            if self.active is None:
                raise ModelUnavailable("No model is loaded")
            result = RecognitionResult(
                "recognized", request.request_id, self.active["model_id"], "Libras",
                ClassificationPrediction("hello", .9, {"hello": .9}, {"pt": "Olá"}), {},
            )
            return InferenceResponse(request_id=request.request_id, model_generation=self.generation,
                                     result=result, runtime=self.active["runtime"], timing_ms={"total_ms": .25})
        finally:
            self._gate.release()

    def close(self):
        self.closed += 1


@pytest.fixture
def manager():
    return FakeManager()


@pytest.fixture
def app(tmp_path, manager):
    return create_app({"public-a": tmp_path / "model-a", "public-b": tmp_path / "model-b"}, manager=manager)


def upload(client, **kwargs):
    return client.post("/v1/recognitions", files={"file": ("source.npy", b"input-bytes", "application/octet-stream")},
                       **kwargs)


def error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"schema_version", "request_id", "error"}
    assert body["schema_version"] == 1 and body["request_id"]
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    return body


def test_lifespan_loads_once_owns_manager_and_uses_server_defaults(tmp_path, manager):
    options = RuntimeOptions(device="cpu", intra_op_threads=2)
    app = create_app({"a": tmp_path / "model-a"}, initial_model="a", options=options, manager=manager)
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/health").status_code == 200
        assert manager.switches == [((tmp_path / "model-a").resolve(), options)]
        assert manager.closed == 0
    assert manager.closed == 1


def test_startup_failure_closes_manager(tmp_path, manager):
    manager.switch_failure = ValueError("invalid bundle")
    app = create_app({"a": tmp_path}, initial_model="a", manager=manager)
    with pytest.raises(ValueError, match="invalid bundle"):
        with TestClient(app):
            pass
    assert manager.closed == 1


def test_catalog_exposes_only_aliases_and_typed_openapi(app, manager, tmp_path):
    with TestClient(app) as client:
        body = client.get("/v1/models").json()
        assert body["models"] == [{"model_id": "public-a"}, {"model_id": "public-b"}]
        assert body["active"] == json.loads(json.dumps(manager.active))
        assert str(tmp_path) not in json.dumps(body)
        schema = client.get("/openapi.json").json()
        assert "SwitchRequest" in schema["components"]["schemas"]
        assert "RecognitionResponse" in schema["components"]["schemas"]
        assert schema["paths"]["/v1/recognitions"]["post"]["responses"]["422"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"}
        assert client.get("/docs").status_code == 200


def test_inference_preserves_contract_and_removes_generated_temp_file(app, manager):
    with TestClient(app) as client:
        response = client.post("/v1/recognitions",
                               files={"file": ("../../caller-secret.npy", b"input-bytes")},
                               data={"request_id": "request-123", "representation_id": "hand63_v1",
                                     "sign_language": "Libras"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["request_id"] == "request-123" == body["result"]["sample_id"]
        assert body["model_generation"] == 1 and body["timing_ms"] == {"total_ms": .25}
        assert body["result"]["prediction"]["label_id"] == "hello"
        assert manager.requests[0].representation_id == "hand63_v1"
        assert manager.requests[0].sign_language == "Libras"
        assert manager.paths[0].name.startswith("slbridge-input-")
        assert manager.paths[0].suffix == ".npy" and "caller-secret" not in str(manager.paths[0])
        assert not manager.paths[0].exists()
        assert manager.native_saw_event_loop is False


def test_missing_request_id_is_generated(app, manager):
    with TestClient(app) as client:
        body = upload(client).json()
        assert body["request_id"] == manager.requests[0].request_id
        assert len(body["request_id"]) == 36


@pytest.mark.parametrize("failure,status,code", [
    (ValueError("representation mismatch"), 422, "invalid_input"),
    (ModelUnavailable("No model is loaded"), 503, "model_unavailable"),
    (RuntimeBusy("Runtime is busy"), 409, "runtime_busy"),
    (RuntimeError("private model path C:/secret"), 500, "internal_error"),
])
def test_inference_errors_are_uniform_and_cleanup_inputs(app, manager, failure, status, code):
    manager.failure = failure
    with TestClient(app, raise_server_exceptions=False) as client:
        body = error(upload(client, data={"request_id": "failed-123"}), status, code)
        assert body["request_id"] == "failed-123"
        assert all(not path.exists() for path in manager.paths)
        if status == 500:
            assert "secret" not in json.dumps(body)


def test_switch_uses_alias_defaults_and_supports_rollback(tmp_path, manager):
    defaults = RuntimeOptions(device="cpu", device_id=0, intra_op_threads=3, gpu_mem_limit_mb=1024,
                              cuda_dll_dirs=(str(tmp_path / "private-cuda-directory"),))
    app = create_app({"b": tmp_path / "model-b"}, options=defaults, manager=manager)
    with TestClient(app) as client:
        switched = client.post("/v1/models/switch", json={"model_id": "b", "device": "cuda", "device_id": 2})
        assert switched.status_code == 200, switched.text
        selected = manager.switches[0][1]
        assert selected == RuntimeOptions(device="cuda", device_id=2, intra_op_threads=3, gpu_mem_limit_mb=1024,
                                          cuda_dll_dirs=defaults.cuda_dll_dirs)
        assert manager.switches[0][0] == (tmp_path / "model-b").resolve()
        rolled = client.post("/v1/models/rollback")
        assert rolled.status_code == 200, rolled.text
        assert rolled.json()["status"]["active"]["model_id"] == "model-a"
        assert rolled.json()["status"]["generation"] == 3


def test_switch_passes_explicit_cpu_numerical_policy_to_manager(tmp_path, manager):
    app = create_app({"b": tmp_path / "model-b"}, manager=manager)
    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model_id": "b", "cpu_conv_subnormals": "preserve"})
        assert response.status_code == 200, response.text
        assert manager.switches[0][1].cpu_conv_subnormals == "preserve"
        invalid = client.post("/v1/models/switch", json={"model_id": "b", "cpu_conv_subnormals": "anything"})
        assert invalid.status_code == 422
        assert len(manager.switches) == 1


def test_unknown_alias_invalid_options_and_arbitrary_paths_are_rejected(app, manager):
    with TestClient(app) as client:
        error(client.post("/v1/models/switch", json={"model_id": "absent"}), 404, "unknown_model")
        for body in ({"model_id": "public-a", "path": "C:/arbitrary/model"},
                     {"model_id": "C:/arbitrary/model"},
                     {"model_id": "public-a", "device": "tpu"},
                     {"model_id": "public-a", "intra_op_threads": 0},
                     {"model_id": "public-a", "device_id": True},
                     {"model_id": "public-a", "device_id": 64},
                     {"model_id": "public-a", "gpu_mem_limit_mb": 1},
                     {"model_id": "public-a", "cuda_dll_dirs": ["C:/arbitrary"]}):
            error(client.post("/v1/models/switch", json=body), 422, "invalid_request")
        error(client.post("/v1/recognitions", json={"path": "C:/arbitrary/source.npy"}), 422, "invalid_request")
        assert manager.switches == [] and manager.paths == []


def test_failed_switch_preserves_active_and_rollback_without_previous_is_unavailable(app, manager):
    manager.switch_failure = ValueError("invalid model contract")
    before = dict(manager.active)
    with TestClient(app) as client:
        error(client.post("/v1/models/switch", json={"model_id": "public-b"}), 422, "invalid_model")
        assert manager.active == before
        error(client.post("/v1/models/rollback"), 503, "model_unavailable")


def test_busy_inference_does_not_block_health_and_rejects_second_inference_and_switch(app, manager):
    manager.release.clear()
    outcome = []
    with TestClient(app) as client:
        worker = threading.Thread(target=lambda: outcome.append(upload(client)))
        worker.start()
        try:
            assert manager.started.wait(2)
            assert client.get("/v1/health").status_code == 200
            error(upload(client), 409, "runtime_busy")
            error(client.post("/v1/models/switch", json={"model_id": "public-b"}), 409, "runtime_busy")
            assert manager.active["model_id"] == "model-a"
        finally:
            manager.release.set()
            worker.join(3)
        assert not worker.is_alive() and outcome[0].status_code == 200
        assert len(manager.paths) == 1 and not manager.paths[0].exists()


def test_validation_malformed_body_and_missing_routes_have_error_envelope(app):
    with TestClient(app) as client:
        error(client.post("/v1/models/switch", content="not json", headers={"Content-Type": "application/json"}),
              422, "invalid_request")
        error(client.get("/does-not-exist"), 404, "invalid_request")
        error(client.get("/v1/recognitions"), 405, "invalid_request")
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"input-bytes")},
                          data={"request_id": " "}), 422, "invalid_request")
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"input-bytes")},
                          data={"request_id": "x" * 129}), 422, "invalid_request")
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"input-bytes")},
                          data={"representation_id": "unknown"}), 422, "invalid_request")
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"input-bytes")},
                          data={"request_id": "has spaces"}), 422, "invalid_request")
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"input-bytes")},
                          data={"sign_language": "x" * 65}), 422, "invalid_request")


@pytest.mark.parametrize("filename,data,code", [("weights.pkl", b"weights", "unsupported_input"),
                                                ("x.npy", b"", "invalid_input")])
def test_model_weights_and_empty_inputs_are_rejected(app, manager, filename, data, code):
    with TestClient(app) as client:
        error(client.post("/v1/recognitions", files={"file": (filename, data)}), 422, code)
        assert not manager.paths


def test_per_file_limit_removes_partially_written_input(tmp_path, manager, monkeypatch):
    original = api.tempfile.NamedTemporaryFile
    def temporary(*args, **kwargs):
        return original(*args, **kwargs, dir=tmp_path)
    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", temporary)
    app = create_app({}, manager=manager, max_upload_bytes=8)
    with TestClient(app) as client:
        error(upload(client), 413, "upload_too_large")
    assert not list(tmp_path.glob("slbridge-input-*")) and not manager.paths


def test_content_length_limit_rejects_before_upload_save(manager, monkeypatch):
    saved = []
    monkeypatch.setattr(api, "_save_upload", lambda *args: saved.append(args))
    app = create_app({}, manager=manager, max_upload_bytes=8)
    with TestClient(app) as client:
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"x" * (70 * 1024))}),
              413, "upload_too_large")
    assert not saved and not manager.paths


def test_streamed_body_limit_stops_parser_and_closes_spool(manager, monkeypatch):
    import starlette.formparsers
    spools = []
    original = starlette.formparsers.SpooledTemporaryFile
    def spool(*args, **kwargs):
        value = original(*args, **kwargs)
        spools.append(value)
        return value
    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", spool)
    app = create_app({}, manager=manager, max_upload_bytes=16)
    boundary = "test-boundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.npy\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + b"x" * (150 * 1024) + f"\r\n--{boundary}--\r\n".encode()
    emitted = 0
    async def chunks():
        nonlocal emitted
        for offset in range(0, len(body), 4096):
            emitted += min(4096, len(body) - offset)
            yield body[offset:offset + 4096]
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/recognitions", content=chunks(),
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            error(response, 413, "upload_too_large")
    with TestClient(app):
        asyncio.run(scenario())
    assert emitted < len(body) and not manager.paths
    assert spools and all(value.closed for value in spools)


def test_multipart_cardinality_limits_and_form_size(manager, monkeypatch):
    import starlette.formparsers
    spools = []
    original = starlette.formparsers.SpooledTemporaryFile
    def spool(*args, **kwargs):
        value = original(*args, **kwargs)
        spools.append(value)
        return value
    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", spool)
    app = create_app({}, manager=manager)
    with TestClient(app) as client:
        files = [("file", ("x.npy", b"input-bytes")), ("extra", ("weights.npy", b"input-bytes"))]
        error(client.post("/v1/recognitions", files=files), 400, "invalid_request")
        error(client.post("/v1/recognitions", files={"file": ("x.npy", b"input-bytes")},
                          data={"request_id": "a" * 5000}), 400, "invalid_request")
        error(client.post("/v1/recognitions", files={"unknown": ("x.npy", b"input-bytes")}), 422, "invalid_request")
    assert not manager.paths
    assert spools and all(value.closed for value in spools)


def test_no_model_status_and_401_envelope_without_secret(tmp_path, manager):
    manager.active = None
    app = create_app({"a": tmp_path}, manager=manager, bearer_token="test-token")
    with TestClient(app) as client:
        response = client.get("/v1/health")
        error(response, 401, "unauthorized")
        assert response.headers["WWW-Authenticate"] == "Bearer"
        error(client.get("/v1/health", headers={"Authorization": "Bearer incorrect"}), 401, "unauthorized")
        good = client.get("/v1/health", headers={"Authorization": "Bearer test-token"})
        assert good.status_code == 200 and good.json()["status"]["active"] is None
        schema = client.get("/openapi.json", headers={"Authorization": "Bearer test-token"}).json()
        assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"


def test_catalog_strictness_and_relative_resolution(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema_version": 1, "models": {"rf": "../bundle"}}), encoding="utf-8")
    assert api_cli.load_catalog(path) == {"rf": (tmp_path / "../bundle").resolve()}
    invalid = [
        '{"schema_version":1,"models":{"a":"one","a":"two"}}',
        '{"schema_version":true,"models":{"a":"one"}}',
        '{"schema_version":2,"models":{"a":"one"}}',
        '{"schema_version":1,"models":{"../escape":"one"}}',
        '{"schema_version":1,"models":{"a":1}}',
        '{"schema_version":1,"models":{},"extra":true}',
    ]
    for value in invalid:
        path.write_text(value, encoding="utf-8")
        with pytest.raises(ValueError):
            api_cli.load_catalog(path)


def test_cli_non_loopback_requires_env_token(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SLBRIDGE_API_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        api_cli.main(["--catalog", str(tmp_path / "models.json"), "--host", "0.0.0.0"])
    assert exc.value.code == 2
    assert "bearer token" in capsys.readouterr().err


def test_cli_default_bind_single_worker_and_env_token(tmp_path, monkeypatch):
    import uvicorn
    path = tmp_path / "models.json"
    path.write_text('{"schema_version":1,"models":{"rf":"model"}}', encoding="utf-8")
    monkeypatch.setenv("SLBRIDGE_API_TOKEN", "test-token")
    called = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: called.append((app, kwargs)))
    assert api_cli.main(["--catalog", str(path), "--model", "rf", "--device", "cpu"]) == 0
    assert called[0][1] == {"host": "127.0.0.1", "port": 8765, "workers": 1}


def test_factory_rejects_bad_server_configuration(tmp_path):
    for kwargs in ({"model_catalog": {"../path": tmp_path}},
                   {"model_catalog": {"a": tmp_path}, "initial_model": "missing"},
                   {"model_catalog": {}, "max_upload_bytes": 0},
                   {"model_catalog": {}, "bearer_token": " "}):
        with pytest.raises(ValueError):
            create_app(**kwargs)


def test_real_manager_descriptor_status_and_input_contract_errors(tmp_path):
    class Model:
        def __init__(self):
            self.bundle = SimpleNamespace(
                manifest=SimpleNamespace(run_id="actual-model", backend="random_forest_v1",
                                         sign_language="Libras", capabilities=("clip",),
                                         input_spec=InputSpec.for_representation("hand63_v1")),
                labels={"hello": {"pt": "Olá"}},
            )
            self.runtime_info = {"device": "cpu"}
            self.closed = False
        def recognize_profiled(self, path, *, sample_id, representation_id=None):
            assert Path(path).read_bytes() == b"input-bytes"
            result = RecognitionResult("recognized", sample_id, "actual-model", "Libras",
                                       ClassificationPrediction("hello", 1., {"hello": 1.}, {"pt": "Olá"}), {})
            return result, {"preprocess_ms": .1, "inference_ms": .2}
        def close(self):
            self.closed = True
    model = Model()
    manager = ModelManager(initial_recognizer=model)
    app = create_app({"missing": tmp_path / "missing-bundle"}, manager=manager)
    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"]["active"]["options"] == {
            key: value for key, value in RuntimeOptions().to_dict().items() if key != "cuda_dll_dirs"}
        recognized = upload(client)
        assert recognized.status_code == 200, recognized.text
        assert set(recognized.json()["timing_ms"]) == {"preprocess_ms", "inference_ms", "total_ms"}
        error(upload(client, data={"sign_language": "ASL"}), 422, "invalid_input")
        error(client.post("/v1/models/switch", json={"model_id": "missing"}), 422, "invalid_model")
        failed = client.get("/v1/health")
        assert failed.status_code == 200, failed.text
        assert failed.json()["status"]["error"]["code"]
        assert failed.json()["status"]["active"]["model_id"] == "actual-model"
    assert model.closed


def test_server_cuda_paths_are_preserved_internally_and_not_returned(tmp_path, manager):
    private_dir = str(tmp_path / "private-server-dir")
    manager.active["options"]["cuda_dll_dirs"] = [private_dir]
    app = create_app({"a": tmp_path}, manager=manager)
    with TestClient(app) as client:
        for route in ("/v1/health", "/v1/models"):
            response = client.get(route)
            assert response.status_code == 200, response.text
            assert private_dir not in response.text and "cuda_dll_dirs" not in response.text
        assert manager.active["options"]["cuda_dll_dirs"] == [private_dir]


def test_known_catalog_dll_and_upload_paths_are_redacted_in_errors(tmp_path, manager):
    bundle = tmp_path / "private-model-bundle"
    dll_dir = tmp_path / "private-cuda-directory"
    manager.switch_failure = ValueError(f"Failed loading {bundle.as_posix()}/model.onnx: DLL directory {dll_dir}")
    options = RuntimeOptions(cuda_dll_dirs=(str(dll_dir),))
    app = create_app({"a": bundle}, options=options, manager=manager)
    with TestClient(app) as client:
        switched = client.post("/v1/models/switch", json={"model_id": "a"})
        body = error(switched, 422, "invalid_model")
        assert "Failed loading [server path]/model.onnx" in body["error"]["message"]
        assert "private-model-bundle" not in switched.text and "private-cuda-directory" not in switched.text
        manager.failure = ValueError("placeholder")
        original = manager.infer_file
        def inference(path, request):
            manager.failure = ValueError(f"Failed input at {path}: format mismatch")
            return original(path, request)
        manager.infer_file = inference
        failed = upload(client)
        body = error(failed, 422, "invalid_input")
        assert "[server path]: format mismatch" in body["error"]["message"]
        assert "slbridge-input-" not in failed.text


def test_runtime_failure_telemetry_redacts_paths_and_keeps_provider_reason(tmp_path, manager):
    dll_dir = str(tmp_path / "private-cuda-directory")
    manager.active["runtime"].update({"cuda_dll_dirs": [dll_dir],
                                      "fallback_reason": f"Missing CUDA DLL in {dll_dir}"})
    app = create_app({}, manager=manager)
    with TestClient(app) as client:
        for response in (client.get("/v1/health"), upload(client)):
            assert response.status_code == 200, response.text
            assert "private-cuda-directory" not in response.text and "cuda_dll_dirs" not in response.text
            assert "Missing CUDA DLL in [server path]" in response.text
        assert manager.active["runtime"]["cuda_dll_dirs"] == [dll_dir]


def test_malformed_parser_body_closes_already_opened_spool(manager, monkeypatch):
    import starlette.formparsers
    spools = []
    original = starlette.formparsers.SpooledTemporaryFile
    def spool(*args, **kwargs):
        value = original(*args, **kwargs)
        spools.append(value)
        return value
    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", spool)
    body = (b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.npy"\r\n\r\n'
            b'input-bytes\r\n--boundary\r\nInvalid Header\r\n\r\n')
    app = create_app({}, manager=manager)
    with TestClient(app) as client:
        error(client.post("/v1/recognitions", content=body,
                          headers={"Content-Type": "multipart/form-data; boundary=boundary"}), 400, "invalid_request")
    assert not manager.paths
    assert spools and all(value.closed for value in spools)
