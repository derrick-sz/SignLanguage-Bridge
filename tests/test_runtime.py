from types import SimpleNamespace
import threading

import pytest

from slbridge.contracts import InputSpec, RecognitionResult
from slbridge.runtime import InferenceRequest, ModelManager, ModelUnavailable, RuntimeBusy
from slbridge.runtime_options import RuntimeOptions


class Model:
    def __init__(self, name, *, fail_warmup=False):
        self.bundle = SimpleNamespace(manifest=SimpleNamespace(
            run_id=name, backend="random_forest_v1", sign_language="libras",
            input_spec=InputSpec.for_representation("hand63_v1"), capabilities=("clip",)),
            labels={"a": {"pt": "A"}})
        self.runtime_info = {"providers": ["CPUExecutionProvider"]}
        self.fail_warmup = fail_warmup
        self.closed = 0
        self.warmed = 0

    def warmup(self):
        self.warmed += 1
        if self.fail_warmup:
            raise ValueError("bad graph")

    def close(self):
        self.closed += 1

    def recognize_profiled(self, path, *, sample_id, representation_id):
        return (RecognitionResult("rejected", sample_id, self.bundle.manifest.run_id,
                                  "libras", None, {}, "test_rejection"),
                {"preprocess_ms": 1., "inference_ms": 2., "total_ms": 3.})


def test_switch_failure_preserves_active_and_closes_candidate():
    old, bad = Model("old"), Model("bad", fail_warmup=True)
    manager = ModelManager(initial_recognizer=old)
    with pytest.raises(ValueError, match="bad graph"):
        manager.switch("bad", loader=lambda _: bad)
    assert manager.recognizer is old
    assert manager.generation == 1 and old.closed == 0 and bad.closed == 1
    assert manager.status()["state"] == "ready"
    response = manager.infer_file("unused.npy", InferenceRequest("req-1"))
    assert response.result.model_id == "old" and response.model_generation == 1
    manager.close()
    assert old.closed == 1


def test_switch_and_rollback_generations_and_two_instance_budget():
    a, b, c = Model("a"), Model("b"), Model("c")
    manager = ModelManager(initial_recognizer=a)
    manager.switch("b", loader=lambda _: b)
    assert manager.generation == 2 and b.warmed == 1
    manager.rollback()
    assert manager.recognizer is a and manager.generation == 3
    manager.switch("c", loader=lambda _: c)
    assert b.closed == 1
    assert manager.status()["previous"]["model_id"] == "a"
    manager.close()
    manager.close()
    assert a.closed == c.closed == 1


def test_candidate_is_not_active_until_commit_and_can_be_discarded():
    a, b = Model("a"), Model("b")
    manager = ModelManager(initial_recognizer=a)
    candidate = manager.prepare("b", loader=lambda _: b)
    assert manager.recognizer is a and manager.status()["state"] == "loading"
    with pytest.raises(RuntimeBusy):
        manager.infer_file("unused", InferenceRequest("req"))
    manager.discard(candidate)
    assert b.closed == 1 and manager.recognizer is a and manager.generation == 1
    with pytest.raises(ModelUnavailable):
        manager.activate(candidate)
    manager.close()


def test_inflight_blocks_switch_and_shutdown_defers_release():
    model = Model("a")
    manager = ModelManager(initial_recognizer=model)
    entered, release = threading.Event(), threading.Event()
    def operation(_):
        entered.set()
        assert release.wait(3)
        return "result"
    worker = threading.Thread(target=lambda: manager.run(operation))
    worker.start()
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeBusy):
            manager.switch("b", loader=lambda _: Model("b"))
        with pytest.raises(RuntimeBusy):
            manager.run(lambda _: None)
        manager.close()
        assert model.closed == 0
        with pytest.raises(ModelUnavailable):
            manager.run(lambda _: None)
    finally:
        release.set()
        worker.join(3)
    assert model.closed == 1


def test_close_during_load_closes_uncommitted_model():
    model = Model("b")
    manager = ModelManager()
    entered, release = threading.Event(), threading.Event()
    errors = []
    def loader(_):
        entered.set()
        assert release.wait(3)
        return model
    def work():
        try:
            manager.switch("b", loader=loader)
        except ModelUnavailable as error:
            errors.append(error)
    worker = threading.Thread(target=work)
    worker.start()
    assert entered.wait(2)
    manager.close()
    release.set()
    worker.join(3)
    assert len(errors) == 1 and model.closed == 1 and manager.recognizer is None


def test_response_contract_declared_input_language_and_no_file_path_leak():
    manager = ModelManager(initial_recognizer=Model("a"))
    request = InferenceRequest("client-123", representation_id="hand63_v1", sign_language="libras")
    response = manager.infer_file("C:/temp/private-upload.npy", request).to_dict()
    assert response["request_id"] == response["result"]["sample_id"] == "client-123"
    assert response["schema_version"] == 1 and response["model_generation"] == 1
    assert response["timing_ms"]["inference_ms"] == 2.
    with pytest.raises(ValueError, match="language"):
        manager.infer_file("unused", InferenceRequest("id", sign_language="csl"))
    with pytest.raises(ValueError, match="representation"):
        manager.infer_file("unused", InferenceRequest("id", representation_id="pose_hands225_legacy"))
    manager.close()


@pytest.mark.parametrize("kwargs", [{"request_id": ""}, {"request_id": "id", "schema_version": 2},
                                   {"request_id": "id", "representation_id": "guess"}])
def test_request_validation(kwargs):
    with pytest.raises(ValueError):
        InferenceRequest(**kwargs)


@pytest.mark.parametrize("kwargs", [{"device": "gpu"}, {"device_id": True}, {"intra_op_threads": 0},
                                   {"gpu_mem_limit_mb": 1}])
def test_runtime_options_validation(kwargs):
    with pytest.raises(ValueError):
        RuntimeOptions(**kwargs)
