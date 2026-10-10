"""Adapter extensibility, strict contracts and real registered native backends."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.contracts import RecognitionResult, TextPrediction
import slbridge.model_adapters as adapters
from slbridge.model_adapters import (AdapterDefinition, InputRepresentation, ModelCapabilities,
                                    ModelInputContract, register_input_representation,
                                    register_model_adapter)
from slbridge.runtime import ModelGenerationConflict, ModelManager, ModelRouter, ModelUnavailable
from slbridge.runtime_contracts import (CancellationToken, InferenceCancelled,
                                      InferenceDeadlineExceeded, InferenceRequest)
from slbridge.runtime_options import RuntimeOptions


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    monkeypatch.setattr(adapters, "_REPRESENTATIONS", dict(adapters._REPRESENTATIONS))
    monkeypatch.setattr(adapters, "_ADAPTERS", dict(adapters._ADAPTERS))
    register_input_representation(InputRepresentation("full321_test_v1", (None, 321)))


class TextAdapter:
    """An explicit deterministic test adapter; no sign-recognition claim."""
    def __init__(self, model_id="text-model", language="libras"):
        self.capabilities = ModelCapabilities(
            model_id, "test-v1", language, "text",
            (ModelInputContract("full321_test_v1", (2, 321), "float32", "test-preprocess-v1"),),
            (), "numpy", ("cpu",), score_type="none")
        self.closed = 0
        self.calls = 0
        self.entered = None
        self.release = None
        self.output_change = None
        self.fail_warmup = False

    @property
    def recognizer(self):
        return self

    @property
    def runtime_info(self):
        return {"runtime": "numpy", "device": "cpu"}

    def warmup(self):
        if self.fail_warmup:
            raise ValueError("example warmup failed")

    def close(self):
        self.closed += 1

    def infer(self, source, request, *, input_kind):
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(3)
        result = RecognitionResult("recognized", request.request_id, self.capabilities.model_id,
                                   self.capabilities.sign_language, TextPrediction("test output", "en"),
                                   {"synthetic": True}, task_type="text", score_type="none")
        if self.output_change is not None:
            result = self.output_change(result)
        return result, {"inference_ms": 1.0}


def request(identifier="request-1", **kwargs):
    values = {"representation_id": "full321_test_v1", "preprocessing_version": "test-preprocess-v1",
              "sign_language": "libras", "expected_task_type": "text"}
    values.update(kwargs)
    return InferenceRequest(identifier, **values)


def test_new_adapter_registry_loads_and_emits_bound_text_without_caller_branch(tmp_path):
    created = []
    def load(path, options):
        assert path == tmp_path and options.device == "cpu"
        model = TextAdapter()
        created.append(model)
        return model
    register_model_adapter(AdapterDefinition("numpy_text_test", load))
    (tmp_path / "manifest.json").write_text(json.dumps({"backend": "numpy_text_test"}), encoding="utf-8")
    manager = ModelManager(options=RuntimeOptions(device="cpu"))
    try:
        manager.switch(tmp_path)
        result = manager.infer(np.ones((2, 321), dtype=np.float32), request(
            session_id="session-1", segment_id="segment-1", source_epoch=7,
            expected_generation=1, model_id="text-model"))
        envelope = result.to_dict()
        assert isinstance(result.result.prediction, TextPrediction)
        assert envelope["session_id"] == "session-1" and envelope["segment_id"] == "segment-1"
        assert envelope["source_epoch"] == 7 and envelope["model_generation"] == 1
        assert envelope["model_id"] == envelope["result"]["model_id"] == "text-model"
        assert envelope["model_version"] == "test-v1" and envelope["preprocessing_version"] == "test-preprocess-v1"
        assert envelope["runtime"]["device"] == "cpu" and envelope["timing_ms"]["total_ms"] >= 0
        assert manager.capabilities()["inputs"][0]["shape"] == (2, 321)
        assert manager.describe()["model_id"] == "text-model"
        assert manager.describe()["task_type"] == "text"
    finally:
        manager.close()
    assert created[0].closed == 1


def test_wrapped_adapter_status_reports_actual_adapter_device_without_legacy_model_fields():
    class WrappedTextAdapter(TextAdapter):
        @property
        def recognizer(self):
            return self.wrapped

    adapter = WrappedTextAdapter()
    adapter.wrapped = SimpleNamespace()
    manager = ModelManager(initial_recognizer=adapter)
    try:
        response = manager.infer_file("unused", request())
        assert manager.describe()["runtime"] == response.runtime == adapter.runtime_info
        assert manager.status()["active"]["runtime"] == adapter.runtime_info
    finally:
        manager.close()
    assert adapter.closed == 1


@pytest.mark.parametrize("changes,match", [
    ({"representation_id": "mediapipe225_rgb_v1"}, "representation"),
    ({"preprocessing_version": "unrelated-v1"}, "preprocessing"),
    ({"expected_task_type": "classification"}, "task type"),
    ({"sign_language": "csl"}, "language"),
    ({"expected_mode": "stream"}, "mode"),
    ({"expected_generation": 2}, "generation"),
    ({"model_id": "wrong-model"}, "model_id"),
])
def test_capability_mismatches_reject_before_invocation(changes, match):
    model = TextAdapter()
    manager = ModelManager(initial_recognizer=model)
    try:
        with pytest.raises((ValueError, ModelGenerationConflict), match=match):
            manager.infer_file("unused", request(**changes))
        assert model.calls == 0
    finally:
        manager.close()


@pytest.mark.parametrize("values", [np.ones((2, 225), dtype=np.float32), np.ones((2, 321), dtype=np.float64),
                                    np.full((2, 321), np.nan, dtype=np.float32)])
def test_225_cannot_be_padded_or_implicitly_converted_to_321(values):
    model = TextAdapter()
    manager = ModelManager(initial_recognizer=model)
    try:
        with pytest.raises(ValueError, match="shape/dtype/finite"):
            manager.infer(values, request())
        assert model.calls == 0
    finally:
        manager.close()


def test_duplicate_ids_and_shape_overrides_cannot_replace_existing_contracts():
    with pytest.raises(ValueError, match="already registered"):
        register_input_representation(InputRepresentation("mediapipe225_rgb_v1", (None, 321)))
    with pytest.raises(ValueError, match="shape/dtype"):
        ModelInputContract("mediapipe225_rgb_v1", (2, 321), "float32", "v1")
    with pytest.raises(ValueError, match="already registered"):
        register_model_adapter(AdapterDefinition("random_forest_v1", lambda *_: TextAdapter()))


@pytest.mark.parametrize("changed", [{"pipeline": "online-v2"}, {"config": {"scale": 2.}}])
def test_same_representation_and_shape_cannot_override_complete_preprocessing_hash(changed):
    from slbridge._io import canonical_hash

    training_spec = {"representation_id": "full321_test_v1", "pipeline": "training-v1", "config": {"scale": 1.}}
    version = canonical_hash(training_spec)
    model = TextAdapter()
    model.capabilities = replace(model.capabilities, inputs=(replace(model.capabilities.inputs[0], preprocessing_version=version),))
    manager = ModelManager(initial_recognizer=model)
    values = np.ones((2, 321), dtype=np.float32)
    actual_spec = {**training_spec, **changed}
    incompatible = SimpleNamespace(values=values, spec=SimpleNamespace(
        representation_id="full321_test_v1", to_dict=lambda: actual_spec))
    matching = SimpleNamespace(values=values, spec=SimpleNamespace(
        representation_id="full321_test_v1", to_dict=lambda: training_spec))
    try:
        # Even a matching request version cannot override a different actual spec.
        with pytest.raises(ValueError, match="preprocessing_version"):
            manager.infer(incompatible, request(preprocessing_version=version))
        assert model.calls == 0
        assert manager.infer(matching, request(preprocessing_version=version)).result.status == "recognized"
    finally:
        manager.close()


def test_serialized_spec_cannot_use_named_version_to_bypass_hash_check():
    model = TextAdapter()
    manager = ModelManager(initial_recognizer=model)
    source = SimpleNamespace(values=np.ones((2, 321), dtype=np.float32), spec=SimpleNamespace(
        representation_id="full321_test_v1", to_dict=lambda: {"pipeline": "different"}))
    try:
        with pytest.raises(ValueError, match="preprocessing_version"):
            manager.infer(source, request())
        assert model.calls == 0
    finally:
        manager.close()


@pytest.mark.parametrize("version", [None, "other-preprocess-v1", "test-preprocess-v1"])
def test_prepared_objects_without_serialized_spec_require_their_own_version(version):
    model = TextAdapter()
    manager = ModelManager(initial_recognizer=model)
    source = SimpleNamespace(values=np.ones((2, 321), dtype=np.float32), spec=SimpleNamespace(
        representation_id="full321_test_v1"))
    if version is not None:
        source.preprocessing_version = version
    try:
        if version == "test-preprocess-v1":
            assert manager.infer(source, request()).result.status == "recognized"
        else:
            with pytest.raises(ValueError, match="preprocessing_version"):
                manager.infer(source, request())
            assert model.calls == 0
    finally:
        manager.close()


def test_text_raw_scores_are_finite_and_probability_scores_have_declared_bounds():
    model = TextAdapter()
    model.capabilities = replace(model.capabilities, score_type="text_log_probability")
    model.output_change = lambda result: replace(result, prediction=TextPrediction("text", "en", -4.5),
                                                 score_type="text_log_probability")
    manager = ModelManager(initial_recognizer=model)
    try:
        assert manager.infer_file("unused", request()).result.prediction.score == -4.5
        model.capabilities = replace(model.capabilities, score_type="text_probability")
        model.output_change = lambda result: replace(result, prediction=TextPrediction("text", "en", -4.5),
                                                     score_type="text_probability")
        with pytest.raises(ValueError, match="Text probability"):
            manager.infer_file("unused", request())
        with pytest.raises(ValueError, match="finite"):
            TextPrediction("text", "en", float("nan"))
    finally:
        manager.close()


@pytest.mark.parametrize("change", [lambda result: replace(result, model_id="spoofed"),
                                   lambda result: replace(result, sample_id="wrong-request"),
                                   lambda result: replace(result, task_type="classification"),
                                   lambda result: replace(result, prediction=None)])
def test_adapter_response_is_validated_against_capability_and_request(change):
    model = TextAdapter()
    model.output_change = change
    manager = ModelManager(initial_recognizer=model)
    try:
        with pytest.raises(ValueError, match="Adapter|Recognized"):
            manager.infer_file("unused", request())
    finally:
        manager.close()


@pytest.mark.parametrize("change,match", [
    (lambda result: replace(result, status="rejected", prediction=None), "reason"),
    (lambda result: replace(result, status="rejected", reason="quality_failure"), "recognized prediction"),
    (lambda result: replace(result, time_range=(float("nan"), 1.)), "time_range"),
    (lambda result: replace(result, time_range=(2., 1.)), "time_range"),
    (lambda result: replace(result, quality={"nested": {"confidence": float("inf")}}), "finite JSON"),
])
def test_adapter_rejection_semantics_and_finite_metadata_are_checked(change, match):
    model = TextAdapter()
    model.output_change = change
    manager = ModelManager(initial_recognizer=model)
    try:
        with pytest.raises(ValueError, match=match):
            manager.infer_file("unused", request())
    finally:
        manager.close()


def test_empty_ctc_rejection_remains_valid_but_nonempty_rejected_gloss_is_refused():
    from slbridge.contracts import GlossSequencePrediction, GlossToken
    from slbridge.model_adapters import validate_adapter_result

    caps = replace(TextAdapter().capabilities, task_type="gloss_sequence", vocabulary=(" ", "HELLO"),
                   score_type="ctc_token_probability")
    inference_request = request(expected_task_type="gloss_sequence")
    blank = RecognitionResult("rejected", inference_request.request_id, caps.model_id, "libras",
                              GlossSequencePrediction((), "greedy"), {}, "ctc_all_blank",
                              task_type="gloss_sequence", score_type="ctc_token_probability")
    validate_adapter_result(blank, caps, inference_request)
    nonempty = replace(blank, prediction=GlossSequencePrediction((GlossToken(1, "HELLO", .9),), "greedy"))
    with pytest.raises(ValueError, match="recognized prediction"):
        validate_adapter_result(nonempty, caps, inference_request)


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("reason", ["cancel", "deadline"])
def test_cooperative_cancel_or_deadline_never_delivers_late_result(when, reason):
    model = TextAdapter()
    manager = ModelManager(initial_recognizer=model)
    token = CancellationToken()
    deadline = (time.monotonic() - 1) if when == "before" and reason == "deadline" else None
    if when == "before" and reason == "cancel":
        token.cancel()
    if when == "during":
        model.entered, model.release = threading.Event(), threading.Event()
        if reason == "deadline":
            deadline = time.monotonic() + .08
    work_request = request(cancellation=token, deadline_monotonic=deadline)
    errors, delivered = [], []
    def work():
        try:
            delivered.append(manager.infer_file("unused", work_request))
        except (InferenceCancelled, InferenceDeadlineExceeded) as error:
            errors.append(error)
    try:
        if when == "before":
            work()
            assert model.calls == 0
        else:
            worker = threading.Thread(target=work)
            worker.start()
            assert model.entered.wait(2)
            if reason == "cancel":
                token.cancel()
            else:
                time.sleep(.09)
            assert model.closed == 0 and manager.status()["state"] == "inferencing"
            model.release.set()
            worker.join(3)
            assert not worker.is_alive()
        assert not delivered and len(errors) == 1
        assert isinstance(errors[0], InferenceCancelled if reason == "cancel" else InferenceDeadlineExceeded)
        assert manager.status()["state"] == "ready"
    finally:
        if model.release is not None:
            model.release.set()
        manager.close()


def test_router_has_independent_language_capacity_and_generations():
    left, right = TextAdapter(language="libras"), TextAdapter(model_id="csl-text", language="csl")
    left.entered, left.release = threading.Event(), threading.Event()
    libras, csl = ModelManager(initial_recognizer=left), ModelManager(initial_recognizer=right)
    router = ModelRouter()
    router.register("libras", libras)
    router.register("csl", csl)
    outputs = []
    worker = threading.Thread(target=lambda: outputs.append(router.infer_file("unused", request())))
    worker.start()
    try:
        assert left.entered.wait(2)
        chinese = router.infer_file("unused", request("csl-request", sign_language="csl"))
        assert chinese.model_id == "csl-text" and chinese.result.sign_language == "csl"
        csl.switch("unused", loader=lambda _: TextAdapter("csl-new", "csl"))
        assert csl.generation == 2 and libras.generation == 1
        left.release.set()
        worker.join(3)
        assert outputs[0].model_generation == 1
        with pytest.raises(ValueError, match="independent"):
            router.register("third-language", libras)
        with pytest.raises(ModelUnavailable, match="No model route"):
            router.manager_for("missing")
    finally:
        left.release.set()
        worker.join(3)
        router.close()
    assert left.closed == right.closed == 1


def test_bad_adapter_warmup_releases_candidate_and_keeps_active():
    old, bad = TextAdapter("old"), TextAdapter("bad")
    bad.fail_warmup = True
    manager = ModelManager(initial_recognizer=old)
    try:
        with pytest.raises(ValueError, match="warmup failed"):
            manager.switch("unused", loader=lambda _: bad)
        assert manager.recognizer is old and manager.generation == 1
        assert old.closed == 0 and bad.closed == 1
    finally:
        manager.close()


def test_prepare_negotiates_new_captured_representation_without_consumer_allowlist():
    adapter = TextAdapter()
    manager = ModelManager()
    try:
        candidate = manager.prepare("unused", loader=lambda _: adapter, required_input_kind="captured_clip")
        manager.activate(candidate)
        assert manager.supports_typed_inference
        assert manager.capabilities()["inputs"][0]["representation_id"] == "full321_test_v1"
        assert manager.infer_clip("synthetic", request()).model_id == "text-model"
        file_only = TextAdapter("file-only")
        file_only.capabilities = replace(file_only.capabilities, inputs=(replace(
            file_only.capabilities.inputs[0], input_kinds=("file",)),))
        with pytest.raises(ValueError, match="required input kind"):
            manager.prepare("unused", loader=lambda _: file_only, required_input_kind="captured_clip")
        assert file_only.closed == 1 and manager.recognizer is adapter and manager.generation == 1
    finally:
        manager.close()


def test_run_only_injection_marker_and_legacy_rgb_loader_compatibility():
    model = SimpleNamespace(bundle=SimpleNamespace(manifest=SimpleNamespace(
        input_spec=SimpleNamespace(representation_id="mediapipe225_rgb_v1"), capabilities=("clip",))))
    manager = ModelManager()
    try:
        candidate = manager.prepare("unused", loader=lambda _: model, required_input_kind="captured_clip")
        manager.activate(candidate)
        assert not manager.supports_typed_inference and manager.capabilities() is None
        assert manager.run(lambda injected: injected is model)
    finally:
        manager.close()


def test_registered_adapter_can_declare_stream_without_claiming_clip_support():
    model = TextAdapter()
    model.capabilities = replace(model.capabilities, modes=("stream",))
    manager = ModelManager()
    try:
        manager.switch("unused", loader=lambda _: model)
        with pytest.raises(ValueError, match="mode"):
            manager.infer_file("unused", request())
        result = manager.infer_file("unused", request(expected_mode="stream"))
        assert result.model_id == "text-model"
    finally:
        manager.close()


def test_documented_adapter_example_runs(capsys):
    document = Path(__file__).resolve().parents[1] / "docs" / "model-adapter-example.md"
    code = document.read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(code, str(document), "exec"), {})
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["result"]["task_type"] == "text"
    assert envelope["result"]["prediction"]["text"] == "sum=3.0"
    assert envelope["session_id"] == "example-session" and envelope["model_generation"] == 1


def test_registered_real_rf_and_exported_onnx_bundle_share_typed_contract(tmp_path):
    pytest.importorskip("skl2onnx")
    from slbridge._io import resolve_resource
    from slbridge.data import create_split, import_legacy
    from slbridge.training import TrainConfig, train
    from slbridge.export_onnx import export_rf_bundle

    # Initialize ORT before native sklearn, matching managed Windows startup.
    manager = ModelManager(options=RuntimeOptions(device="cpu"))
    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(31)
    for label, value in (("A", -1.), ("B", 1.)):
        for index in range(9):
            np.save(source / f"{label}_{index:03}.npy", rng.normal(value, .03, size=(4, 63)).astype(np.float32))
    dataset = import_legacy(source, "adapter-real", "hand63_v1")
    split = create_split(dataset)
    rf = train(dataset, split, TrainConfig("hand63_v1", target_frames=2, rf={"n_estimators": 5}),
               tmp_path / "models", "rf-adapter-real")
    samples = {sample.sample_id: sample for sample in dataset.samples}
    heldout = [resolve_resource(source, samples[identifier].resource) for identifier in split.test]
    try:
        manager.switch(rf)
        rf_response = manager.infer_file(heldout[0], InferenceRequest("rf-request", expected_task_type="classification"))
        onnx = export_rf_bundle(rf, tmp_path / "onnx", "onnx-adapter-real", heldout)
        manager.switch(onnx)
        onnx_response = manager.infer_file(heldout[0], InferenceRequest("onnx-request", expected_task_type="classification"))
        assert rf_response.result.prediction.label_id == onnx_response.result.prediction.label_id
        np.testing.assert_allclose(list(rf_response.result.prediction.scores.values()),
                                   list(onnx_response.result.prediction.scores.values()), atol=1e-6)
        assert rf_response.preprocessing_version == onnx_response.preprocessing_version
        assert rf_response.model_version != onnx_response.model_version
        assert onnx_response.runtime["providers"] == ["CPUExecutionProvider"]
        assert manager.capabilities()["runtime"] == "onnxruntime"
    finally:
        manager.close()


def test_registered_real_ctc_bundle_preserves_ordered_gloss_payload(tmp_path):
    # Reuse the established tiny real ONNX graph/bundle fixture builders.
    from test_sequence import _graph, _manifest, _prepared
    from slbridge.contracts import GlossSequencePrediction

    _graph(tmp_path / "model.onnx")
    _manifest(tmp_path)
    manager = ModelManager(options=RuntimeOptions(device="cpu"))
    try:
        manager.switch(tmp_path)
        response = manager.infer(_prepared(), InferenceRequest("ctc-request", expected_task_type="gloss_sequence",
                                                               representation_id="tfnet_rgb_v1", sign_language="csl"))
        assert isinstance(response.result.prediction, GlossSequencePrediction)
        assert response.result.prediction.glosses == ("Z",)
        assert manager.capabilities()["vocabulary"] == (" ", "Z", "A")
        assert response.runtime["providers"] == ["CPUExecutionProvider"]
    finally:
        manager.close()
