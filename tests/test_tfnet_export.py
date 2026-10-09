"""Math/security and surrogate-graph checks; these do not benchmark TFNet."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.tfnet_export import (
    _fp32_export_precision, _input_record, build_export_wrapper, channel_dft_matrices,
    compare_tfnet_outputs, export_tfnet,
)
from slbridge.tfnet_native import (
    TFNET_CHECKPOINT_SHA256, TFNET_SOURCE_COMMIT, TFNetNativeRunner, TFNetOutput, _load_checkpoint, _load_source_model,
    validate_tfnet_input, verify_tfnet_source,
)


def test_dft_is_normalized_last_channel_axis_not_time():
    generator = np.random.default_rng(24)
    features = generator.normal(size=(1, 7, 512)).astype(np.float32)
    cosine, sine = channel_dft_matrices()
    magnitude = np.sqrt(np.square(features @ cosine) + np.square(features @ sine))
    reference = np.abs(np.fft.fft(features, axis=-1, norm="forward"))
    np.testing.assert_allclose(magnitude, reference, rtol=2e-4, atol=2e-7)
    constant = np.ones((1, 3, 512), np.float32)
    real, imaginary = constant @ cosine, constant @ sine
    np.testing.assert_allclose(real[..., 0], 1.0, atol=1e-7)
    np.testing.assert_allclose(real[..., 1:], 0.0, atol=1e-7)
    np.testing.assert_allclose(imaginary, 0.0, atol=1e-7)
    np.testing.assert_allclose(magnitude[:, 2], np.abs(np.fft.fft(features[:, 2], norm="forward")), rtol=2e-4, atol=2e-7)


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_dft_rejects_invalid_width(value):
    with pytest.raises(ValueError):
        channel_dft_matrices(value)


@pytest.mark.parametrize("matmul,cudnn", [(False, False), (False, True), (True, False), (True, True)])
def test_export_precision_scope_forces_fp32_and_restores_after_exception(matmul, cudnn):
    fake_torch = SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=matmul)),
        cudnn=SimpleNamespace(allow_tf32=cudnn)))
    with pytest.raises(RuntimeError, match="synthetic early failure"):
        with _fp32_export_precision(fake_torch) as precision:
            assert fake_torch.backends.cuda.matmul.allow_tf32 is False
            assert fake_torch.backends.cudnn.allow_tf32 is False
            assert precision["caller_flags_before_export"] == {"cuda_matmul_allow_tf32": matmul, "cudnn_allow_tf32": cudnn}
            raise RuntimeError("synthetic early failure")
    assert fake_torch.backends.cuda.matmul.allow_tf32 is matmul
    assert fake_torch.backends.cudnn.allow_tf32 is cudnn


def test_native_input_requires_exact_normalized_blchw_and_sorted_lengths():
    valid = np.zeros((1, 16, 3, 224, 224), np.float32)
    values, lengths = validate_tfnet_input(valid)
    assert values.shape == valid.shape
    assert lengths.tolist() == [16]
    assert values.flags.c_contiguous
    assert values is valid
    readonly = valid.copy()
    readonly.setflags(write=False)
    copied, _ = validate_tfnet_input(readonly)
    assert copied.flags.writeable
    assert not np.shares_memory(copied, readonly)
    np.testing.assert_array_equal(copied, readonly)
    for invalid in (valid.astype(np.float64), valid[:, :15], valid[:, :, :, :223], np.full_like(valid, np.nan), np.full_like(valid, 2.0)):
        with pytest.raises(ValueError):
            validate_tfnet_input(invalid)
    for invalid_lengths in ([15], [20], [16.0], []):
        with pytest.raises(ValueError):
            validate_tfnet_input(valid, invalid_lengths)
    with pytest.raises(ValueError, match="descending"):
        validate_tfnet_input(np.zeros((2, 20, 3, 224, 224), np.float32), [16, 20])


def test_source_verification_rejects_changed_fixed_files(tmp_path, monkeypatch):
    from slbridge import tfnet_native

    for name in tfnet_native._SOURCE_MODULES:
        (tmp_path / f"{name}.py").write_text("# synthetic source identity test\n", encoding="utf-8")
    monkeypatch.setattr(tfnet_native, "_git", lambda source, *args: TFNET_SOURCE_COMMIT if args[0] == "rev-parse" else "")
    provenance = verify_tfnet_source(tmp_path)
    assert provenance["source_commit"] == TFNET_SOURCE_COMMIT
    assert len(provenance["source_files_sha256"]) == 5
    monkeypatch.setattr(tfnet_native, "_git", lambda source, *args: TFNET_SOURCE_COMMIT if args[0] == "rev-parse" else "Net.py")
    with pytest.raises(ValueError, match="modified"):
        verify_tfnet_source(tmp_path)


def test_source_constructor_avoids_download_bytecode_and_restores_import_names(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    original_module = ModuleType("Module")
    monkeypatch.setitem(sys.modules, "Module", original_module)
    (tmp_path / "Module.py").write_text(
        "class BasicBlock: pass\n"
        "def ResNet34MAM(block, layers, **kwargs): return (block, layers)\n"
        "def resnet34MAM(**kwargs): raise AssertionError('download constructor called')\n",
        encoding="utf-8",
    )
    for name in ("BiLSTM", "Transformer", "SEN"):
        (tmp_path / f"{name}.py").write_text("# own synthetic source\n", encoding="utf-8")
    (tmp_path / "Net.py").write_text(
        "import Module\n"
        "class moduleNet:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.backbone = Module.resnet34MAM()\n"
        "        self.options = kwargs\n", encoding="utf-8",
    )
    model = _load_source_model(tmp_path, "cpu")
    assert model.backbone[1] == [3, 4, 6, 3]
    assert model.options["moduleChoice"] == "TFNet"
    assert model.options["wordSetNum"] == 3516
    assert sys.modules["Module"] is original_module
    assert not (tmp_path / "__pycache__").exists()


def test_weights_only_load_allows_only_audited_numpy_float64_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "synthetic_checkpoint.pth"
    torch.save({"moduleNet_state_dict": {"test": torch.ones(1)}, "bestWerScore": np.float64(33.0)}, path)
    loaded = _load_checkpoint(path, torch)
    assert loaded["bestWerScore"] == 33.0
    assert torch.equal(loaded["moduleNet_state_dict"]["test"], torch.ones(1))
    torch.save({"moduleNet_state_dict": {}, "unexpected": Path("unsafe-extra-object")}, path)
    with pytest.raises(ValueError, match="weights-only"):
        _load_checkpoint(path, torch)


def test_runner_uses_upstream_collated_lengths_and_explicit_fused_inference():
    torch = pytest.importorskip("torch")
    import threading

    class ContractModel(torch.nn.Module):
        def forward(self, values, lengths, isTrain=True):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert isTrain is False
            assert isinstance(lengths, list)
            assert lengths[0].shape == (1,)
            assert lengths[0].device.type == "cpu"
            assert lengths[0].tolist() == [16]
            scores = torch.zeros((1, 1, 3516))
            return scores, None, None, None, scores, torch.tensor([1]), [], [], []

    runner = TFNetNativeRunner.__new__(TFNetNativeRunner)
    runner._torch = torch
    runner.device = torch.device("cpu")
    runner._lock = threading.RLock()
    runner.model = ContractModel().train()
    output = runner.predict(np.zeros((1, 16, 3, 224, 224), np.float32))
    assert output.scores.shape == (1, 1, 3516)
    assert output.lengths.tolist() == [1]
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.predict(np.zeros((1, 16, 3, 224, 224), np.float32))


def test_native_provenance_snapshots_caller_precision_without_changing_flags(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from slbridge import tfnet_native

    fixture = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "synthetic-identity-fixture.pth"
    checkpoint.write_bytes(b"fixture mocked state load, not a TFNet checkpoint")
    monkeypatch.setattr(tfnet_native, "verify_tfnet_source", lambda path: {"source_dir": str(tmp_path), "source_commit": TFNET_SOURCE_COMMIT})
    monkeypatch.setattr(tfnet_native, "_sha256", lambda path: TFNET_CHECKPOINT_SHA256)
    monkeypatch.setattr(tfnet_native, "_load_source_model", lambda path, device: fixture)
    monkeypatch.setattr(tfnet_native, "_load_checkpoint", lambda path, tensor_module: {"moduleNet_state_dict": fixture.state_dict()})
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    runner = TFNetNativeRunner(checkpoint, tmp_path, device="cpu")
    assert runner.provenance["precision"]["cuda_matmul_allow_tf32"] == before[0]
    assert runner.provenance["precision"]["cudnn_allow_tf32"] == before[1]
    assert runner.provenance["precision"]["captured_at"] == "model_load"
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
    runner.close()


def _scores(path, classes=4):
    scores = np.full((len(path), 1, classes), -5.0, np.float32)
    for index, token in enumerate(path):
        scores[index, 0, token] = 5.0
    return scores


def test_parity_retains_blank_separated_repeated_glosses():
    scores = _scores([1, 1, 0, 1, 2, 0])
    report = compare_tfnet_outputs(scores, scores.copy(), np.array([6]), np.array([6]))
    assert report["reference_greedy_ids"] == [1, 1, 2]
    assert report["greedy_ctc_equal"] is True
    assert report["max_absolute_error"] == 0.0


def test_parity_rejects_logits_lengths_and_decoded_changes():
    scores = _scores([1, 0, 1])
    with pytest.raises(ValueError, match="gate failed"):
        compare_tfnet_outputs(scores, scores + 1.0, np.array([3]), np.array([3]))
    with pytest.raises(ValueError, match="lengths"):
        compare_tfnet_outputs(scores, scores, np.array([3]), np.array([2]))
    # A tied score can change the decode despite passing a loose allclose gate.
    expected = np.zeros((1, 1, 4), np.float32)
    actual = expected.copy()
    actual[0, 0, 1] = 1e-7
    with pytest.raises(ValueError, match="greedy_ctc_equal"):
        compare_tfnet_outputs(expected, actual, np.array([1]), np.array([1]))


def test_input_metadata_records_original_modulo_four():
    values = np.zeros((1, 20, 3, 224, 224), np.float32)
    _, record = _input_record(SimpleNamespace(values=values, source_frames=5, source_ref="heldout.mp4"))
    assert record["source_frames_modulo4"] == 1
    assert record["source_ref"] == "heldout.mp4"
    assert len(record["input_tensor_sha256"]) == 64
    with pytest.raises(ValueError, match="contradicts"):
        _input_record(SimpleNamespace(values=values, source_frames=3))


def _surrogate_model(torch):
    """Own tiny modules exercise the wrapper, not the external TFNet backbone."""
    class Visual(torch.nn.Module):
        def forward(self, video):
            features = video.mean(dim=(1, 3, 4)).unsqueeze(-1).expand(-1, -1, 512)
            return features.reshape(-1, 512), [], [], []

    class Surrogate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d = Visual()
            self.conv1d = torch.nn.Module()
            self.conv1d1 = torch.nn.Module()
            for branch in (self.conv1d, self.conv1d1):
                branch.temporal_conv = torch.nn.Sequential(
                    torch.nn.Conv1d(512, 4, 5), torch.nn.MaxPool1d(2),
                    torch.nn.Conv1d(4, 4, 5), torch.nn.MaxPool1d(2),
                )
            self.temporal_model = torch.nn.Module()
            self.temporal_model1 = torch.nn.Module()
            for branch in (self.temporal_model, self.temporal_model1):
                branch.rnn = torch.nn.LSTM(4, 2, num_layers=2, bidirectional=True)
            self.classifier55 = torch.nn.Linear(4, 4)

        def packed_reference(self, video):
            features, _, _, _ = self.conv2d(video.transpose(1, 2))
            features = features.reshape(1, video.shape[1], 512)
            frequency = torch.fft.fft(features.float(), dim=-1, norm="forward").abs()
            first = self.conv1d.temporal_conv(features.transpose(1, 2)).permute(2, 0, 1)
            second = self.conv1d1.temporal_conv(frequency.transpose(1, 2)).permute(2, 0, 1)
            length = first.shape[0]
            first_packed = torch.nn.utils.rnn.pack_padded_sequence(first, [length])
            second_packed = torch.nn.utils.rnn.pack_padded_sequence(second, [length])
            first = torch.nn.utils.rnn.pad_packed_sequence(self.temporal_model.rnn(first_packed)[0])[0]
            second = torch.nn.utils.rnn.pad_packed_sequence(self.temporal_model1.rnn(second_packed)[0])[0]
            return self.classifier55(first + second), torch.tensor([length], dtype=torch.int64)

    torch.manual_seed(19)
    return Surrogate().eval()


@pytest.mark.parametrize("length", [16, 20, 24, 28])
def test_wrapper_channel_fft_and_full_length_dense_lstm_match_packed_surrogate(length):
    torch = pytest.importorskip("torch")
    model = _surrogate_model(torch)
    wrapper = build_export_wrapper(model)
    channel_features = torch.randn(1, 7, 512)
    with torch.inference_mode():
        frequency = wrapper.frequency_magnitude(channel_features)
        torch.testing.assert_close(frequency, torch.fft.fft(channel_features, dim=-1, norm="forward").abs(), rtol=2e-4, atol=2e-7)
    values = torch.linspace(-1, 1, length).reshape(1, length, 1, 1, 1).expand(-1, -1, 3, 2, 2)
    with torch.inference_mode():
        expected, expected_lengths = model.packed_reference(values)
        actual, actual_lengths = wrapper(values)
    report = compare_tfnet_outputs(expected.numpy(), actual.numpy(), expected_lengths.numpy(), actual_lengths.numpy())
    assert report["output_lengths"] == [(length - 12) // 4]


def test_surrogate_export_verifies_dynamic_time_and_preserves_destination(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    from slbridge import tfnet_export

    model = _surrogate_model(torch)

    class SurrogateRunner:
        def __init__(self, *args, **kwargs):
            self.model = model
            self.device = torch.device("cpu")
            self.provenance = {"synthetic_surrogate": True}

        def predict(self, values):
            with torch.inference_mode():
                scores, lengths = self.model.packed_reference(torch.from_numpy(values))
            return TFNetOutput(scores.numpy(), lengths.numpy(), 0.0)

        def close(self):
            self.model = None

    monkeypatch.setattr(tfnet_export, "TFNetNativeRunner", SurrogateRunner)
    values = [np.broadcast_to(np.linspace(-1, 1, length, dtype=np.float32).reshape(1, length, 1, 1, 1), (1, length, 3, 224, 224)).copy() for length in (16, 20, 28)]
    output = tmp_path / "surrogate.onnx"
    metadata = export_tfnet("synthetic-checkpoint", "synthetic-source", output, values)
    assert metadata["parity_passed"] is True
    assert metadata["runtime_intra_op_threads"] == 4
    assert metadata["precision"]["reference_and_export"] == {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    assert len(metadata["parity_inputs"]) == 3
    assert output.is_file()
    assert Path(str(output) + ".metadata.json").is_file()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "video"
    assert session.get_inputs()[0].shape[1] == "padded_frames"
    assert session.get_outputs()[0].shape == ["output_steps", 1, 4]
    unseen = np.zeros((1, 24, 3, 224, 224), np.float32)
    scores, lengths = session.run(["logits", "lengths"], {"video": unseen})
    assert scores.shape == (3, 1, 4)
    assert lengths.tolist() == [3]
    graph_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_tfnet("synthetic-checkpoint", "synthetic-source", output, values)
    assert output.read_bytes() == graph_bytes


def test_export_validates_destination_and_options_before_dependencies(tmp_path):
    values = [np.zeros((1, 16, 3, 224, 224), np.float32)]
    with pytest.raises(ValueError, match="At least one"):
        export_tfnet("unused", "unused", tmp_path / "test.onnx", [])
    with pytest.raises(ValueError, match="opset"):
        export_tfnet("unused", "unused", tmp_path / "test.onnx", values, opset_version=12)
    with pytest.raises(ValueError, match="tolerances"):
        export_tfnet("unused", "unused", tmp_path / "test.onnx", values, atol=float("inf"))
    with pytest.raises(ValueError, match="end in"):
        export_tfnet("unused", "unused", tmp_path / "test.pth", values)
    with pytest.raises(ValueError, match="intra_op_threads"):
        export_tfnet("unused", "unused", tmp_path / "test.onnx", values, intra_op_threads=0)


def test_precision_is_restored_when_runner_constructor_fails(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    from slbridge import tfnet_export

    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    def fail(*args, **kwargs):
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        raise RuntimeError("synthetic constructor failure")
    monkeypatch.setattr(tfnet_export, "TFNetNativeRunner", fail)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError, match="synthetic constructor failure"):
            export_tfnet("unused", "unused", tmp_path / "test.onnx", [np.zeros((1,16,3,224,224), np.float32)])
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original
