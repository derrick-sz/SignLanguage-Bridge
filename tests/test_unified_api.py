"""Synthetic transport/lifecycle tests; these make no recognition-quality claim."""
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from slbridge.api import create_app
from slbridge.contracts import (ClassificationPrediction, GlossSequencePrediction, GlossToken,
                                RecognitionResult, TextPrediction)
from slbridge.model_adapters import (AdapterDefinition, InputRepresentation, ModelCapabilities, ModelInputContract,
                                     register_input_representation, register_model_adapter)
from slbridge.runtime import ModelManager, ModelRouter


register_input_representation(InputRepresentation("http_test321_v1", (None, 321)))


class BoundaryAdapter:
    """Deterministic typed adapter, solely to isolate the HTTP boundary."""
    def __init__(self, language="Libras", task="classification"):
        self.capabilities = ModelCapabilities(
            f"boundary-{language}", "test-v1", language, task,
            (ModelInputContract("http_test321_v1", (None, 321), "float32", "test-pre-v1", ("file",)),),
            ("blank", "hello"), "test-runtime", ("cpu",), score_type="test_score")
        self.bundle = SimpleNamespace(manifest=SimpleNamespace(
            run_id=self.capabilities.model_id, backend="test-boundary", sign_language=language,
            task_type=task, score_type="test_score", input_spec=SimpleNamespace(representation_id="http_test321_v1"),
            capabilities=("clip",)), labels={"hello": {"pt": "Olá"}})
        self.runtime_info = {"device": "cpu"}
        self.calls = 0
        self.paths = []
        self.closed = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    @property
    def recognizer(self):
        return self

    def warmup(self):
        pass

    def infer(self, source, request, *, input_kind):
        assert input_kind == "file"
        self.paths.append(Path(source))
        assert Path(source).read_bytes() == b"boundary-input"
        self.calls += 1
        self.started.set()
        assert self.release.wait(5)
        task = self.capabilities.task_type
        prediction = (ClassificationPrediction("hello", .9, {"blank": .1, "hello": .9}, {"pt": "Olá"})
                      if task == "classification" else
                      GlossSequencePrediction((GlossToken(1, "hello", .9), GlossToken(1, "hello", .8)), "greedy")
                      if task == "gloss_sequence" else TextPrediction("Hello", "en", .9))
        return RecognitionResult("recognized", request.request_id, self.capabilities.model_id,
                                 self.capabilities.sign_language, prediction, {},
                                 task_type=task, score_type="test_score"), {"inference_ms": 1.}

    def close(self):
        self.closed = True


def make_app(adapter=None, **kwargs):
    adapter = adapter or BoundaryAdapter()
    return create_app({}, manager=ModelManager(initial_recognizer=adapter), **kwargs), adapter


def upload(client, data=None, content=b"boundary-input"):
    return client.post("/v3/recognitions", files={"file": ("clip.npy", content)}, data=data or {})


def segment(request_id="r1", segment_id="s1", sequence=1, **kwargs):
    return dict(request_id=request_id, session_id="session", sign_language="Libras", segment_id=segment_id,
                sequence=sequence, source_epoch=0, expected_generation=1, start_time=0., end_time=1.,
                source_id="camera", **kwargs)


def open_session(client, session_id="session", language="Libras"):
    response = client.post("/v3/caption-sessions", json={"session_id": session_id, "sign_language": language})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("task", ["classification", "gloss_sequence", "text"])
def test_unified_api_uses_registered_representation_and_typed_task(task):
    app, adapter = make_app(BoundaryAdapter(task=task))
    with TestClient(app) as client:
        capabilities = client.get("/v3/models").json()["capabilities"]
        assert capabilities["inputs"][0]["representation_id"] == "http_test321_v1"
        negotiated = client.post("/v3/models/negotiate", json={"representation_id": "http_test321_v1", "task_type": task})
        assert negotiated.status_code == 200, negotiated.text
        response = upload(client, {"representation_id": "http_test321_v1", "task_type": task, "expected_generation": 1})
        assert response.status_code == 200, response.text
        assert response.json()["schema_version"] == 3
        assert response.json()["result"]["task_type"] == task
        assert response.json()["model_version"] == "test-v1"
        assert response.json()["preprocessing_version"] == "test-pre-v1"
        if task == "gloss_sequence":
            assert len(response.json()["result"]["prediction"]["tokens"]) == 2
        assert not adapter.paths[0].exists()
    assert adapter.closed


def test_generation_task_and_preprocessing_mismatches_do_not_invoke_adapter():
    app, adapter = make_app()
    with TestClient(app) as client:
        for data, code, status in (({"expected_generation": 2}, "model_generation_conflict", 409),
                                   ({"task_type": "text"}, "invalid_input", 422),
                                   ({"preprocessing_version": "wrong"}, "invalid_input", 422),
                                   ({"representation_id": "tfnet_rgb_v1"}, "invalid_input", 422)):
            response = upload(client, data)
            assert response.status_code == status, response.text
            assert response.json()["error"]["code"] == code
        assert adapter.calls == 0


def test_caption_dedupe_repeated_text_and_idempotent_undo_clear_close():
    app, adapter = make_app()
    with TestClient(app) as client:
        open_session(client)
        first = upload(client, segment())
        assert first.status_code == 200, first.text
        assert first.json()["caption_event_id"] == "session:s1"
        assert first.json()["caption_committed"]
        duplicate = upload(client, segment())
        assert duplicate.json() == first.json() and adapter.calls == 1
        conflict = upload(client, segment(), content=b"changed")
        assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "request_conflict"
        second = upload(client, segment("r2", "s2", 2))
        assert second.status_code == 200, second.text
        events = client.get("/v3/caption-sessions/session").json()["snapshot"]["events"]
        assert len(events) == 2 and events[0]["caption"] == events[1]["caption"]
        action = {"action_id": "undo1", "action": "undo"}
        result = client.post("/v3/caption-sessions/session/actions", json=action)
        assert len(result.json()["snapshot"]["events"]) == 1
        assert client.post("/v3/caption-sessions/session/actions", json=action).json() == result.json()
        assert client.post("/v3/caption-sessions/session/actions", json={**action, "action": "clear"}).status_code == 409
        cleared = client.post("/v3/caption-sessions/session/actions", json={"action_id": "clear1", "action": "clear"})
        assert cleared.json()["snapshot"]["events"] == []
        closed = client.post("/v3/caption-sessions/session/actions", json={"action_id": "close1", "action": "close"})
        assert closed.json()["closed"]
        assert upload(client, segment("r3", "s3", 3)).status_code == 409
        assert client.delete("/v3/caption-sessions/session").status_code == 200
        assert client.delete("/v3/caption-sessions/session").status_code == 200
        assert client.get("/v3/caption-sessions/session").status_code == 404


def test_session_request_and_action_capacities_are_bounded():
    app, adapter = make_app(max_sessions=1, max_session_requests=1, max_session_actions=1)
    with TestClient(app) as client:
        open_session(client)
        assert client.post("/v3/caption-sessions", json={"session_id": "another", "sign_language": "Libras"}).status_code == 429
        assert upload(client, segment()).status_code == 200
        assert upload(client, segment("r2", "s2", 2)).status_code == 429
        assert client.post("/v3/caption-sessions/session/actions", json={"action_id": "a1", "action": "undo"}).status_code == 200
        assert client.post("/v3/caption-sessions/session/actions", json={"action_id": "a2", "action": "undo"}).status_code == 429
        client.delete("/v3/caption-sessions/session")
        open_session(client, "replacement")
    assert adapter.calls == 1


def test_timeout_discards_late_caption_and_keeps_file_until_native_returns():
    app, adapter = make_app(inference_timeout_seconds=.06)
    adapter.release.clear()
    with TestClient(app) as client:
        open_session(client)
        response = upload(client, segment())
        assert response.status_code == 504, response.text
        assert adapter.started.is_set() and adapter.paths[0].exists()
        assert client.get("/v3/caption-sessions/session").json()["snapshot"]["events"] == []
        adapter.release.set()
        deadline = time.monotonic() + 2
        while adapter.paths[0].exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert not adapter.paths[0].exists()
        assert client.get("/v3/caption-sessions/session").json()["snapshot"]["events"] == []
        retry = upload(client, segment())
        assert retry.status_code == 504 and adapter.calls == 1


def test_cancel_source_invalidation_and_inflight_dedup_are_effective():
    app, adapter = make_app()
    adapter.release.clear()
    outcomes = []
    with TestClient(app) as client:
        open_session(client)
        worker = threading.Thread(target=lambda: outcomes.append(upload(client, segment())))
        worker.start()
        try:
            assert adapter.started.wait(2)
            duplicate = upload(client, segment())
            assert duplicate.status_code == 409 and duplicate.json()["error"]["code"] == "request_inflight"
            action = client.post("/v3/caption-sessions/session/actions", json={"action_id": "a1", "action": "invalidate_source", "source_epoch": 1})
            assert action.status_code == 200, action.text
            worker.join(2)
            assert outcomes[0].status_code == 409
            assert outcomes[0].json()["error"]["code"] == "request_cancelled"
            assert upload(client, segment("r2", "s2", 2)).json()["error"]["code"] == "stale_source"
        finally:
            adapter.release.set()
            worker.join(2)
        assert client.get("/v3/caption-sessions/session").json()["snapshot"]["events"] == []


def test_language_routes_use_independent_capacity_and_legacy_task_compatibility():
    libras, csl = BoundaryAdapter(), BoundaryAdapter("csl", "gloss_sequence")
    router = ModelRouter()
    router.register("Libras", ModelManager(initial_recognizer=libras))
    router.register("csl", ModelManager(initial_recognizer=csl))
    app = create_app({}, router=router)
    libras.release.clear()
    outcomes = []
    with TestClient(app) as client:
        worker = threading.Thread(target=lambda: outcomes.append(upload(client, {"sign_language": "Libras"})))
        worker.start()
        try:
            assert libras.started.wait(2)
            sequence = upload(client, {"sign_language": "csl", "task_type": "gloss_sequence"})
            assert sequence.status_code == 200, sequence.text
            assert csl.calls == 1 and libras.calls == 1
            assert upload(client).json()["error"]["code"] == "language_required"
            assert upload(client, {"sign_language": "missing"}).status_code == 404
        finally:
            libras.release.set()
            worker.join(2)
        assert outcomes[0].status_code == 200
        legacy = client.post("/v1/recognitions", files={"file": ("clip.npy", b"boundary-input")}, data={"sign_language": "Libras"})
        assert legacy.status_code == 200 and legacy.json()["schema_version"] == 1
        assert "session_id" not in legacy.json()
    assert libras.closed and csl.closed


def test_unknown_duplicate_and_partial_segment_fields_are_rejected():
    app, adapter = make_app()
    with TestClient(app) as client:
        assert upload(client, {"path": "C:/private"}).status_code == 422
        assert upload(client, {"segment_id": "s1"}).status_code == 422
        assert upload(client, {"session_id": "session"}).status_code == 422
        assert client.post("/v3/models/negotiate", json={"representation_id": "unknown"}).status_code == 422
        response = client.post("/v3/recognitions", files=[("file", ("clip.npy", b"boundary-input")),
            ("request_id", (None, "r1")), ("request_id", (None, "r2"))])
        assert response.status_code == 422
        assert adapter.calls == 0


def test_route_startup_switch_and_rollback_preserve_language_identity(tmp_path):
    def load_boundary(path, options):
        return BoundaryAdapter(json.loads((path / "manifest.json").read_text())["language"])
    register_model_adapter(AdapterDefinition("http-boundary-loader-v1", load_boundary))
    paths = {}
    for alias, language in (("libras-a", "Libras"), ("libras-b", "Libras"), ("csl-a", "csl")):
        path = tmp_path / alias
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"backend": "http-boundary-loader-v1", "language": language}))
        paths[alias] = path
    app = create_app(paths, routes={"Libras": "libras-a", "csl": "csl-a"})
    with TestClient(app) as client:
        initial = client.get("/v3/health").json()["status"]
        assert initial["Libras"]["generation"] == initial["csl"]["generation"] == 1
        wrong = client.post("/v3/models/switch", json={"sign_language": "Libras", "model_id": "csl-a"})
        assert wrong.status_code == 422, wrong.text
        assert client.get("/v3/health").json()["status"]["Libras"]["active"]["sign_language"] == "Libras"
        good = client.post("/v3/models/switch", json={"sign_language": "Libras", "model_id": "libras-b"})
        assert good.status_code == 200, good.text
        status = client.get("/v3/health").json()["status"]
        assert status["Libras"]["generation"] == 2 and status["csl"]["generation"] == 1
        rollback = client.post("/v3/models/rollback", params={"sign_language": "Libras"})
        assert rollback.status_code == 200 and rollback.json()["status"]["generation"] == 3
        assert client.get("/v1/health").status_code == 409


def test_snapshot_removes_private_cuda_and_temporary_paths(tmp_path):
    app, adapter = make_app()
    private = str(tmp_path / "private-cuda")
    adapter.runtime_info = {"cuda_dll_dirs": [private], "fallback_reason": f"missing DLL in {private}"}
    original = adapter.infer
    def native(source, request, *, input_kind):
        result, timing = original(source, request, input_kind=input_kind)
        result.quality["source_ref"] = str(source)
        return result, timing
    adapter.infer = native
    with TestClient(app) as client:
        open_session(client)
        response = upload(client, segment())
        assert response.status_code == 200, response.text
        snapshot = client.get("/v3/caption-sessions/session")
        assert "private-cuda" not in response.text and "private-cuda" not in snapshot.text
        assert "cuda_dll_dirs" not in response.text and "cuda_dll_dirs" not in snapshot.text
        assert "slbridge-input-" not in response.text and "slbridge-input-" not in snapshot.text


def test_simultaneous_request_id_retry_cannot_replace_the_original_record():
    app, adapter = make_app()
    adapter.release.clear()
    outcomes = []
    barrier = threading.Barrier(2)
    with TestClient(app) as client:
        open_session(client)
        manager = app.state.manager
        original = manager.negotiate
        def negotiated(request, *, input_kind):
            result = original(request, input_kind=input_kind)
            barrier.wait(2)
            return result
        manager.negotiate = negotiated
        workers = [threading.Thread(target=lambda: outcomes.append(upload(client, segment()))) for _ in range(2)]
        for worker in workers:
            worker.start()
        try:
            assert adapter.started.wait(2)
            deadline = time.monotonic() + 2
            while not outcomes and time.monotonic() < deadline:
                time.sleep(.01)
            assert len(outcomes) == 1 and outcomes[0].status_code == 409
            assert outcomes[0].json()["error"]["code"] == "request_inflight"
        finally:
            adapter.release.set()
            for worker in workers:
                worker.join(2)
        assert sorted(response.status_code for response in outcomes) == [200, 409]
        assert adapter.calls == 1
        assert len(client.get("/v3/caption-sessions/session").json()["snapshot"]["events"]) == 1


def test_simultaneous_distinct_ids_cannot_exceed_request_history_capacity():
    app, adapter = make_app(max_session_requests=1)
    adapter.release.clear()
    outcomes = []
    barrier = threading.Barrier(2)
    with TestClient(app) as client:
        open_session(client)
        manager = app.state.manager
        original = manager.negotiate
        def negotiated(request, *, input_kind):
            result = original(request, input_kind=input_kind)
            barrier.wait(2)
            if request.request_id == "r2":
                assert adapter.started.wait(2)
            return result
        manager.negotiate = negotiated
        workers = [threading.Thread(target=lambda n=n: outcomes.append(upload(client, segment(f"r{n}", f"s{n}", n))))
                   for n in (1, 2)]
        for worker in workers:
            worker.start()
        try:
            assert adapter.started.wait(2)
            deadline = time.monotonic() + 2
            while not outcomes and time.monotonic() < deadline:
                time.sleep(.01)
            assert len(outcomes) == 1 and outcomes[0].status_code == 429
            assert outcomes[0].json()["error"]["code"] == "request_capacity"
            assert len(app.state.sessions["session"].records) == 1
        finally:
            adapter.release.set()
            for worker in workers:
                worker.join(2)
        assert sorted(response.status_code for response in outcomes) == [200, 429]
        assert adapter.calls == 1
        assert len(app.state.sessions["session"].records) == 1


def test_routed_switch_supports_an_adapter_without_a_recognizer_bundle(tmp_path):
    loaded = []
    def load_unbundled(path, options):
        value = BoundaryAdapter(json.loads((path / "manifest.json").read_text())["language"])
        del value.bundle
        loaded.append(value)
        return value
    register_model_adapter(AdapterDefinition("http-no-bundle-adapter-v1", load_unbundled))
    paths = {}
    for alias, language in (("a", "Libras"), ("b", "Libras"), ("wrong", "csl")):
        path = tmp_path / alias
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"backend": "http-no-bundle-adapter-v1", "language": language}))
        paths[alias] = path
    with TestClient(create_app(paths, routes={"Libras": "a"})) as client:
        switched = client.post("/v3/models/switch", json={"sign_language": "Libras", "model_id": "b"})
        assert switched.status_code == 200, switched.text
        assert switched.json()["status"]["active"]["model_id"] == "boundary-Libras"
        inferred = upload(client, {"sign_language": "Libras", "expected_generation": 2})
        assert inferred.status_code == 200, inferred.text
        wrong = client.post("/v3/models/switch", json={"sign_language": "Libras", "model_id": "wrong"})
        assert wrong.status_code == 422, wrong.text
        assert client.get("/v3/health").json()["status"]["Libras"]["generation"] == 2
    assert all(value.closed for value in loaded)


def test_request_and_action_capacities_can_be_configured_independently():
    app, _ = make_app(max_session_requests=1, max_session_actions=3)
    with TestClient(app) as client:
        open_session(client)
        assert upload(client, segment()).status_code == 200
        for action_id, action in (("a1", "undo"), ("a2", "clear"), ("a3", "undo")):
            response = client.post("/v3/caption-sessions/session/actions", json={"action_id": action_id, "action": action})
            assert response.status_code == 200, response.text
        full = client.post("/v3/caption-sessions/session/actions", json={"action_id": "a4", "action": "clear"})
        assert full.status_code == 429 and full.json()["error"]["code"] == "action_capacity"
