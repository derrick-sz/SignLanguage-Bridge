"""CTC path semantics, bundle integrity, and real ONNX sequence contracts."""
from dataclasses import replace
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge._io import canonical_hash, read_json, sha256_file, write_json_atomic
from slbridge.cecsl import RGBInputSpec
from slbridge.contracts import GlossSequencePrediction, GlossToken
from slbridge.inference import Recognizer
from slbridge.runtime import ModelManager
from slbridge.runtime_contracts import InferenceRequest
from slbridge.runtime_options import RuntimeOptions
from slbridge.sequence import (CTCDecoderConfig, CTCOnnxConfig, OnnxCTCBackend, SequenceBundle,
                               SequenceManifest, SequenceRecognizer, greedy_decode,
                               load_sequence_bundle, prefix_beam_decode)


VOCABULARY = (" ", "Z", "A")


def _logits(path, size=3):
    values = np.full((len(path), size), -10., dtype=np.float32)
    values[np.arange(len(path)), path] = 10.
    return values


def _prepared(frames=5, spec=None):
    spec = spec or RGBInputSpec()
    return SimpleNamespace(spec=spec, source_frames=frames, source_ref="prepared",
                           quality={"full_decode_verified": True}, time_range=(0., .2),
                           values=np.zeros((1, ((frames + 3) // 4) * 4 + 12, 3, 224, 224), dtype=np.float32))


def test_greedy_merges_repeats_before_removing_blank_and_preserves_order():
    prediction = greedy_decode(_logits([1, 1, 0, 1, 2, 2, 0]), VOCABULARY)
    assert prediction.glosses == ("Z", "Z", "A")
    assert prediction.text == "Z Z A"
    assert [token.token_id for token in prediction.tokens] == [1, 1, 2]
    assert all(.99 < token.score <= 1 and token.time_range is None for token in prediction.tokens)
    assert prediction.decoder == "greedy"


@pytest.mark.parametrize("decoder", [greedy_decode, prefix_beam_decode])
def test_all_blank_is_empty_and_blank_never_becomes_a_gloss(decoder):
    prediction = decoder(_logits([0, 0, 0, 0]), VOCABULARY)
    assert prediction.tokens == () and prediction.glosses == () and prediction.text == ""


def test_prefix_beam_marginalizes_collapsed_paths_against_exhaustive_oracle():
    probabilities = np.array([[.4, .35, .25], [.4, .35, .25], [.4, .35, .25]])
    totals = {}
    for path in itertools.product(range(3), repeat=3):
        collapsed = tuple(token for index, token in enumerate(path)
                          if token != 0 and (index == 0 or token != path[index - 1]))
        totals[collapsed] = totals.get(collapsed, 0.) + np.prod([probabilities[time, token] for time, token in enumerate(path)])
    expected = max(totals, key=totals.get)
    assert expected == (1,)
    # The maximum frame path is all blank, but several CTC paths support Z.
    assert greedy_decode(np.log(probabilities), VOCABULARY).tokens == ()
    beam = prefix_beam_decode(np.log(probabilities), VOCABULARY, beam_width=100, token_top_k=2)
    assert tuple(token.token_id for token in beam.tokens) == expected
    assert all(token.score is None and token.time_range is None for token in beam.tokens)


def test_prefix_beam_retains_blank_separated_repeated_tokens():
    prediction = prefix_beam_decode(_logits([1, 1, 0, 1, 1]), VOCABULARY)
    assert prediction.glosses == ("Z", "Z")


@pytest.mark.parametrize("values", [np.array([[np.nan, 0., 1.]]), np.array([[np.inf, 0., 1.]]),
                                    np.zeros((2, 1, 3)), np.zeros((2, 2)), np.zeros((0, 3)),
                                    np.zeros((2, 3), dtype=np.int64)])
def test_decoder_refuses_nonfinite_or_invalid_vocabulary_axis(values):
    with pytest.raises(ValueError, match="finite T,V"):
        greedy_decode(values, VOCABULARY)


@pytest.mark.parametrize("changes", [{"beam_width": 0}, {"beam_width": True}, {"token_top_k": 101},
                                    {"blank_id": 1}, {"decoder": "upstream_beam"}])
def test_decoder_parameters_are_explicit_and_bounded(changes):
    with pytest.raises(ValueError):
        CTCDecoderConfig(**changes)


def _manifest(path, spec=None, **changes):
    spec = spec or RGBInputSpec()
    vocabulary = {"schema_version": 1, "blank_id": 0, "tokens": list(VOCABULARY),
                  "provenance": {"source": "synthetic-test"}}
    provenance = {"source": "synthetic-test", "accuracy_validated": False}
    write_json_atomic(path / "vocabulary.json", vocabulary)
    write_json_atomic(path / "preprocessing.json", spec.to_dict())
    write_json_atomic(path / "provenance.json", provenance)
    from slbridge.backends import dependency_version
    manifest = SequenceManifest("ctc-test", spec, canonical_hash(vocabulary),
                                {name: sha256_file(path / name) for name in ("model.onnx", "vocabulary.json", "preprocessing.json", "provenance.json")},
                                CTCDecoderConfig().to_dict(),
                                {name: dependency_version(name) for name in ("numpy", "onnxruntime")},
                                provenance, "2026-10-09T00:00:00Z", "csl", CTCOnnxConfig().to_dict())
    if changes:
        manifest = replace(manifest, **changes)
    write_json_atomic(path / "manifest.json", manifest.to_dict())
    return manifest


def _graph(path, *, invalid_length=False, nonfinite=False, output_layout="TBV", bad_vocabulary=False):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    constants = {"axis1": np.array(1, dtype=np.int64), "padding": np.array(8 if invalid_length else 12, dtype=np.int64),
                 "stride": np.array(4, dtype=np.int64), "axis0": np.array([0], dtype=np.int64),
                 "tail": np.array([1, 3], dtype=np.int64), "head": np.array([1], dtype=np.int64),
                 "vocab": np.array([3], dtype=np.int64),
                 "seed_logits": np.array([[[0., float('nan') if nonfinite else 2., 1.]]], dtype=np.float32)}
    nodes = [helper.make_node("Shape", ["video"], ["video_shape"]),
             helper.make_node("Gather", ["video_shape", "axis1"], ["length"]),
             helper.make_node("Sub", ["length", "padding"], ["unpadded"]),
             helper.make_node("Div", ["unpadded", "stride"], ["time_scalar"]),
             helper.make_node("Unsqueeze", ["time_scalar", "axis0"], ["time"])]
    nodes.append(helper.make_node("Concat", ["time", "tail"] if output_layout == "TBV" else ["head", "time", "vocab"],
                                  ["out_shape"], axis=0))
    nodes.append(helper.make_node("Expand", ["seed_logits", "out_shape"], ["logits"]))
    shape = ["T", 1, 2 if bad_vocabulary else 3] if output_layout == "TBV" else [1, "T", 3]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, shape),
               helper.make_tensor_value_info("time", TensorProto.INT64, [1])]
    graph = helper.make_graph(nodes, "synthetic_ctc", [helper.make_tensor_value_info("video", TensorProto.FLOAT, [1, "L", 3, 224, 224])],
                              outputs, [numpy_helper.from_array(value, name) for name, value in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path)


@pytest.mark.parametrize("layout", ["TBV", "BTV"])
def test_real_onnx_ctc_dynamic_time_strict_shape_warmup_and_close(tmp_path, layout):
    path = tmp_path / "model.onnx"
    _graph(path, output_layout=layout)
    backend = OnnxCTCBackend.load(path, input_spec=RGBInputSpec(), vocabulary=VOCABULARY,
                                  config=CTCOnnxConfig(output_layout=layout), options=RuntimeOptions(device="cpu"))
    backend.warmup()
    assert backend.predict_logits(_prepared(frames=5)).shape == (2, 3)
    assert backend.predict_logits(_prepared(frames=9)).shape == (3, 3)
    assert backend.runtime_info["device"] == "cpu"
    with pytest.raises(ValueError, match="Prepared RGB"):
        backend.predict_logits(_prepared(spec=RGBInputSpec(max_frames=10)))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.warmup()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_only_actual_cuda_ctc_runs_shrink_the_selected_device_arena(tmp_path, device):
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / "model.onnx"
    _graph(path)
    inner = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    observed = []

    class Session:
        def get_inputs(self):
            return inner.get_inputs()

        def get_outputs(self):
            return inner.get_outputs()

        def run(self, names, feeds, **kwargs):
            observed.append(kwargs)
            # The surrogate graph executes on CPU; capture the CUDA-specific
            # RunOptions without submitting a nonexistent GPU arena to it.
            return inner.run(names, feeds)

    backend = OnnxCTCBackend(Session(), RGBInputSpec(), VOCABULARY, CTCOnnxConfig(),
                             {"device": device, "device_id": 2})
    backend.warmup()
    backend.predict_logits(_prepared())
    assert len(observed) == 2
    if device == "cuda":
        for kwargs in observed:
            assert set(kwargs) == {"run_options"}
            assert kwargs["run_options"].get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:2"
    else:
        assert observed == [{}, {}]
    backend.close()
    assert backend._run_options is None


@pytest.mark.parametrize("bad", ["length", "nonfinite", "vocabulary"])
def test_real_ctc_rejects_wrong_temporal_length_nonfinite_and_class_axis(tmp_path, bad):
    path = tmp_path / "model.onnx"
    _graph(path, invalid_length=bad == "length", nonfinite=bad == "nonfinite", bad_vocabulary=bad == "vocabulary")
    if bad == "vocabulary":
        with pytest.raises(ValueError, match="vocabulary"):
            OnnxCTCBackend.load(path, input_spec=RGBInputSpec(), vocabulary=VOCABULARY,
                                config=CTCOnnxConfig(), options=RuntimeOptions(device="cpu"))
    else:
        backend = OnnxCTCBackend.load(path, input_spec=RGBInputSpec(), vocabulary=VOCABULARY,
                                      config=CTCOnnxConfig(), options=RuntimeOptions(device="cpu"))
        with pytest.raises(ValueError, match="invalid CTC logits"):
            backend.warmup()


def test_v2_load_dispatch_preserves_vocabulary_order_and_manager_lifecycle(tmp_path, monkeypatch):
    _graph(tmp_path / "model.onnx")
    manifest = _manifest(tmp_path)
    recognizer = Recognizer.load(tmp_path, options=RuntimeOptions(device="cpu"))
    assert isinstance(recognizer, SequenceRecognizer)
    assert recognizer.bundle.vocabulary == VOCABULARY  # Never alphabetical A,Z.
    result = recognizer.recognize_prepared(_prepared(), sample_id="sample-5")
    assert result.task_type == "gloss_sequence" and result.prediction.glosses == ("Z",)
    assert result.score_type == "ctc_token_probability" and result.time_range == (0., .2)
    assert "score" not in result.to_dict()["prediction"]  # No invented sentence confidence.
    import slbridge.cecsl as cecsl
    monkeypatch.setattr(cecsl, "prepare_rgb_file", lambda path, spec: _prepared(spec=spec))
    manager = ModelManager(initial_recognizer=recognizer, options=RuntimeOptions(device="cpu"))
    described = manager.describe()
    assert described["task_type"] == "gloss_sequence" and described["vocabulary_size"] == 3
    assert described["labels"] == {} and "vocabulary" not in described
    with pytest.raises(ValueError, match="task type"):
        manager.infer_file("unused", InferenceRequest("v1", expected_task_type="classification"))
    response = manager.infer_file("unused", InferenceRequest("v2", representation_id="tfnet_rgb_v1", expected_task_type="gloss_sequence"))
    assert response.result.prediction.tokens[0].gloss == "Z" and response.result.sample_id == "v2"
    assert response.timing_ms["model_ms"] >= 0 and response.timing_ms["decode_ms"] >= 0
    assert response.timing_ms["inference_ms"] >= response.timing_ms["model_ms"] + response.timing_ms["decode_ms"]
    candidate = manager.prepare(tmp_path, required_representation=("mediapipe225_rgb_v1", "tfnet_rgb_v1"))
    manager.activate(candidate)
    assert manager.generation == 2
    manager.rollback()
    assert manager.generation == 3
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        recognizer.warmup()


def test_blank_result_is_explicit_with_no_sentence_probability():
    spec = RGBInputSpec()
    manifest = SimpleNamespace(parameters=CTCDecoderConfig().to_dict(), run_id="blank-model", sign_language="csl",
                               task_type="gloss_sequence", score_type="ctc_token_probability", input_spec=spec)
    backend = SimpleNamespace(predict_logits=lambda prepared: _logits([0, 0]))
    result = SequenceRecognizer(SequenceBundle(None, manifest, backend, VOCABULARY)).recognize_prepared(_prepared())
    assert result.status == "rejected" and result.reason == "ctc_all_blank"
    assert result.prediction.tokens == ()
    assert set(result.to_dict()["prediction"]) == {"tokens", "decoder"}


@pytest.mark.parametrize("artifact", ["model.onnx", "vocabulary.json", "preprocessing.json", "provenance.json"])
def test_all_artifact_checksums_are_verified_before_native_loading(tmp_path, monkeypatch, artifact):
    _graph(tmp_path / "model.onnx")
    _manifest(tmp_path)
    file = tmp_path / artifact
    file.write_bytes(file.read_bytes() + b"corrupt")
    monkeypatch.setattr(OnnxCTCBackend, "load", lambda *args, **kwargs: pytest.fail("No native load for corrupt artifacts"))
    with pytest.raises(ValueError, match="checksum"):
        load_sequence_bundle(tmp_path)


def test_rehashed_vocabulary_reorder_is_refused_and_untracked_weights_are_rejected(tmp_path, monkeypatch):
    _graph(tmp_path / "model.onnx")
    _manifest(tmp_path)
    vocabulary = read_json(tmp_path / "vocabulary.json")
    vocabulary["tokens"] = [" ", "A", "Z"]
    write_json_atomic(tmp_path / "vocabulary.json", vocabulary)
    manifest = read_json(tmp_path / "manifest.json")
    manifest["artifacts"]["vocabulary.json"] = sha256_file(tmp_path / "vocabulary.json")
    write_json_atomic(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(OnnxCTCBackend, "load", lambda *args, **kwargs: pytest.fail("No native load for inconsistent vocabulary"))
    with pytest.raises(ValueError, match="vocabulary"):
        load_sequence_bundle(tmp_path)
    _manifest(tmp_path)
    (tmp_path / "unhashed.bin").write_bytes(b"external model tensors")
    with pytest.raises(ValueError, match="untracked"):
        load_sequence_bundle(tmp_path)


@pytest.mark.parametrize("changes", [{"schema_version": 1}, {"schema_version": True}, {"task_type": "classification"},
                                    {"backend": "onnx_classifier_v1"}, {"score_type": "sentence_probability"},
                                    {"dataset_hash": "bad"}])
def test_sequence_manifest_never_weakens_classification_schema(tmp_path, changes):
    _graph(tmp_path / "model.onnx")
    manifest = _manifest(tmp_path).to_dict()
    manifest.update(changes)
    with pytest.raises(ValueError):
        SequenceManifest.from_dict(manifest)


def test_manifest_path_escape_and_archives_are_refused(tmp_path):
    _graph(tmp_path / "model.onnx")
    manifest = _manifest(tmp_path).to_dict()
    manifest["frozen_evidence"] = {"dataset": "../outside.json"}
    manifest["dataset_hash"] = "a" * 64
    manifest["artifacts"]["../outside.json"] = "a" * 64
    with pytest.raises(ValueError, match="relative resource"):
        SequenceManifest.from_dict(manifest)
    archive = tmp_path / "unsafe.zip"
    archive.write_bytes(b"archive content")
    with pytest.raises(ValueError, match="archives are unsupported"):
        load_sequence_bundle(archive)
