"""Evaluation protocol and recoverable journal tests without executing weights."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge._io import read_json
from slbridge.cecsl import freeze_cecsl_dataset
from slbridge.cecsl_cli import (edit_distance, evaluate_selected, load_frozen_cecsl,
                               main, package_sequence, select_official_samples, wer_metrics)


def _frozen(tmp_path):
    root = tmp_path / "data"
    rows = {"train": [["train-one", "A", "a", "a", ""]],
            "dev": [["dev-one", "A", "b", "b", ""]],
            "test": [["test-one", "A", "a", "a", ""],
                     ["test-two", "B", "bbb", "b/b/b", ""]]}
    (root / "label").mkdir(parents=True)
    for split, records in rows.items():
        with (root / "label" / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Number", "Translator", "Chinese Sentences", "Gloss", "Note"])
            writer.writerows(records)
        for identifier, signer, *_ in records:
            video = root / "video" / split / signer / f"{identifier}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"fake-video-for-hash")
    out = tmp_path / "frozen"
    freeze_cecsl_dataset(root, out, expected_vocab_size=3)
    return load_frozen_cecsl(out)


def _prepared(monkeypatch):
    import slbridge.cecsl_cli as cli
    monkeypatch.setattr(cli, "prepare_rgb_file", lambda path, spec: SimpleNamespace(
        source_frames=4, values=np.zeros((1, 16, 3, 224, 224), dtype=np.float32), spec=spec))


def _predict_a(prepared):
    logits = np.array([[-20, 20, -20]], dtype=np.float32)
    return logits, 0.1


@pytest.mark.parametrize("reference,hypothesis,expected", [
    (["a", "b"], ["a", "b"], 0), (["a", "b"], ["a"], 1),
    (["a"], ["a", "b"], 1), (["a"], ["b"], 1), (["a", "a"], [], 2),
    ([], ["a"], 1), ([], [], 0), (["a", "b", "c"], ["c", "a", "c"], 2),
])
def test_edit_distance(reference, hypothesis, expected):
    assert edit_distance(reference, hypothesis) == expected


def test_two_wer_aggregations_are_distinct():
    metrics = wer_metrics([{"edit_distance": 1, "reference_tokens": 1},
                           {"edit_distance": 1, "reference_tokens": 3}])
    assert metrics["utterance_mean_wer_percent"] == pytest.approx(100 * 2 / 3)
    assert metrics["corpus_wer_percent"] == 50


def test_frozen_membership_selection_and_checksum(tmp_path):
    frozen = _frozen(tmp_path)
    assert [row["sample_id"] for row in select_official_samples(frozen, "test")] == ["test-one", "test-two"]
    assert [row["sample_id"] for row in select_official_samples(frozen, "test", offset=1, limit=1)] == ["test-two"]
    with pytest.raises(ValueError, match="empty"):
        select_official_samples(frozen, "test", offset=2)
    path = frozen["root"] / "vocabulary.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_cecsl(frozen["root"])


def test_evaluation_records_all_blank_as_deletions_and_keeps_official_scope(tmp_path, monkeypatch):
    frozen = _frozen(tmp_path)
    _prepared(monkeypatch)
    selected = select_official_samples(frozen, "test")
    def all_blank(_):
        return np.array([[20, -20, -20]], dtype=np.float32), 0.2
    report = evaluate_selected(frozen, selected, all_blank, tmp_path / "report.json", split="test",
                               backend_identity={"test": "synthetic-only"})
    assert report["scope"] == "entire_official_split"
    assert report["metrics"]["utterance_mean_wer_percent"] == 100
    assert report["metrics"]["corpus_wer_percent"] == 100
    assert [row["edit_distance"] for row in report["records"]] == [1, 3]
    assert all(row["hypothesis"] == [] for row in report["records"])
    assert not report["upstream_advertised_wer_verified"]
    assert read_json(tmp_path / "report.json.progress.json")["status"] == "complete"


def test_interrupted_evaluation_resumes_exact_frozen_prefix_without_repeating(tmp_path, monkeypatch):
    frozen = _frozen(tmp_path)
    _prepared(monkeypatch)
    selected = select_official_samples(frozen, "test")
    output = tmp_path / "report.json"
    calls = []
    def interrupted(prepared):
        calls.append(prepared)
        if len(calls) == 2:
            raise KeyboardInterrupt("synthetic stop")
        return _predict_a(prepared)
    with pytest.raises(KeyboardInterrupt):
        evaluate_selected(frozen, selected, interrupted, output, split="test", backend_identity={"test": "same"})
    assert len(output.with_name(output.name + ".samples.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert read_json(str(output) + ".progress.json")["status"] == "interrupted"
    resumed_calls = []
    def resumed(prepared):
        resumed_calls.append(prepared)
        return _predict_a(prepared)
    report = evaluate_selected(frozen, selected, resumed, output, split="test",
                               backend_identity={"test": "same"}, resume=True)
    assert len(resumed_calls) == 1 and report["metrics"]["samples"] == 2
    with pytest.raises(FileExistsError):
        evaluate_selected(frozen, selected, resumed, output, split="test", backend_identity={"test": "same"})


def test_resume_refuses_changed_backend_or_video(tmp_path, monkeypatch):
    frozen = _frozen(tmp_path)
    _prepared(monkeypatch)
    selected = select_official_samples(frozen, "test")
    output = tmp_path / "resume.json"
    def progress(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        evaluate_selected(frozen, selected, _predict_a, output, split="test",
                          backend_identity={"test": "same"}, progress=progress)
    with pytest.raises(ValueError, match="identity differs"):
        evaluate_selected(frozen, selected, _predict_a, output, split="test",
                          backend_identity={"test": "changed"}, resume=True)
    path = Path(frozen["index"]["provenance"]["dataset_root"]) / selected[0]["resource"]
    path.write_bytes(b"FAKE-video-for-hash")  # same byte count, different content
    with pytest.raises(ValueError, match="bytes changed"):
        evaluate_selected(frozen, selected, _predict_a, output, split="test",
                          backend_identity={"test": "same"}, resume=True)


def test_new_evaluation_refuses_video_size_changes_without_skipping(tmp_path, monkeypatch):
    frozen = _frozen(tmp_path)
    _prepared(monkeypatch)
    selected = select_official_samples(frozen, "test", limit=1)
    path = Path(frozen["index"]["provenance"]["dataset_root"]) / selected[0]["resource"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="byte size changed"):
        evaluate_selected(frozen, selected, _predict_a, tmp_path / "failed.json", split="test",
                          backend_identity={"test": "same"})
    assert read_json(tmp_path / "failed.json.progress.json")["status"] == "failed"
    assert not (tmp_path / "failed.json").exists()


def test_package_is_bound_to_verified_export_and_writes_sequence_contract(tmp_path, monkeypatch):
    from slbridge._io import sha256_file, write_json_atomic
    from slbridge.sequence import SequenceManifest
    frozen = _frozen(tmp_path)
    graph = tmp_path / "test.onnx"
    graph.write_bytes(b"package fixture, no real ONNX execution")
    metadata = {"parity_passed": True, "onnx_sha256": sha256_file(graph),
                "cecsl_frozen_artifacts": frozen["identity"], "cecsl_input_spec": frozen["spec"].to_dict()}
    write_json_atomic(str(graph) + ".metadata.json", metadata)
    monkeypatch.setattr("slbridge.cecsl_cli.importlib.metadata.version", lambda name: "1.23.2")
    # This test checks bundle construction with a deliberately synthetic graph,
    # rather than the separately tested production checkpoint identity gate.
    monkeypatch.setattr("slbridge.cecsl_cli._require_checkpoint_vocabulary", lambda frozen: None)
    output = tmp_path / "bundle"
    result = package_sequence(graph, frozen["root"], output, run_id="synthetic-test")
    manifest = SequenceManifest.from_dict(result["manifest"])
    assert manifest.schema_version == 2 and manifest.backend == "onnx_ctc"
    assert manifest.sign_language == "csl"
    assert read_json(output / "preprocessing.json") == frozen["spec"].to_dict()
    assert read_json(output / "provenance.json")["frozen_preprocessing"] == frozen["preprocessing"]
    assert set(path.name for path in output.iterdir()) == set(manifest.artifacts) | {"manifest.json"}
    with pytest.raises(FileExistsError):
        package_sequence(graph, frozen["root"], output, run_id="synthetic-test")
    metadata["cecsl_frozen_artifacts"] = {}
    write_json_atomic(str(graph) + ".metadata.json", metadata)
    with pytest.raises(ValueError, match="frozen CE-CSL identity"):
        package_sequence(graph, frozen["root"], tmp_path / "bad", run_id="synthetic-test")


def test_cli_help_is_available_without_native_dependencies():
    with pytest.raises(SystemExit) as error:
        main(["native", "--help"])
    assert error.value.code == 0


def _interrupt_after_first(tmp_path, monkeypatch):
    frozen = _frozen(tmp_path)
    _prepared(monkeypatch)
    selected = select_official_samples(frozen, "test")
    output = tmp_path / "recover.json"
    def progress(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        evaluate_selected(frozen, selected, _predict_a, output, split="test",
                          backend_identity={"test": "same"}, progress=progress)
    return frozen, selected, output


def test_resume_recovers_only_truncated_final_append_after_validating_prefix(tmp_path, monkeypatch):
    frozen, selected, output = _interrupt_after_first(tmp_path, monkeypatch)
    journal = Path(str(output) + ".samples.jsonl")
    valid = journal.read_bytes()
    with journal.open("ab") as stream:
        stream.write(b'{"sample_id":"test-two","ref')
    report = evaluate_selected(frozen, selected, _predict_a, output, split="test",
                               backend_identity={"test": "same"}, resume=True)
    assert journal.read_bytes().startswith(valid)
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["test-one", "test-two"]
    assert report["metrics"]["samples"] == 2


@pytest.mark.parametrize("field,bad", [("reference", ["b"]), ("reference_tokens", 99),
                                      ("edit_distance", 99), ("hypothesis_ids", [0]),
                                      ("hypothesis", ["b"]), ("source_frames", 601),
                                      ("output_steps", 2), ("input_shape", [1, 999, 3, 224, 224])])
def test_resume_recomputes_and_validates_all_persisted_wer_inputs(tmp_path, monkeypatch, field, bad):
    frozen, selected, output = _interrupt_after_first(tmp_path, monkeypatch)
    journal = Path(str(output) + ".samples.jsonl")
    row = json.loads(journal.read_text(encoding="utf-8"))
    row[field] = bad
    complete = (json.dumps(row) + "\n").encode()
    original = complete + b'{"incomplete"'
    journal.write_bytes(original)
    with pytest.raises(ValueError, match="journal"):
        evaluate_selected(frozen, selected, _predict_a, output, split="test",
                          backend_identity={"test": "same"}, resume=True)
    # Rejecting a corrupt complete row must not truncate its later suffix.
    assert journal.read_bytes() == original


def test_output_lock_blocks_parallel_state_writes_and_os_releases_on_termination(tmp_path, monkeypatch):
    frozen = _frozen(tmp_path)
    _prepared(monkeypatch)
    selected = select_official_samples(frozen, "test")
    output = tmp_path / "locked.json"
    state = Path(str(output) + ".progress.json")
    state.write_bytes(b'{"external":"must remain unchanged"}')
    code = ("from slbridge.cecsl_cli import _evaluation_lock\n"
            "import sys\n"
            "with _evaluation_lock(sys.argv[1]):\n"
            "    print('locked', flush=True)\n"
            "    sys.stdin.readline()\n")
    process = subprocess.Popen([sys.executable, "-c", code, str(output)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               cwd=Path(__file__).resolve().parents[1])
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(FileExistsError, match="Another evaluator"):
            evaluate_selected(frozen, selected, _predict_a, output, split="test", backend_identity={"test": "same"})
        assert state.read_bytes() == b'{"external":"must remain unchanged"}'
    finally:
        process.terminate()
        process.wait(timeout=10)
    state.unlink()
    report = evaluate_selected(frozen, selected, _predict_a, output, split="test", backend_identity={"test": "same"})
    assert report["metrics"]["samples"] == 2
    assert Path(str(output) + ".lock").is_file()


def test_checkpoint_vocabulary_requires_exact_mapping_not_just_dimension(tmp_path):
    from slbridge.cecsl_cli import _require_checkpoint_vocabulary
    with pytest.raises(ValueError, match="token order"):
        _require_checkpoint_vocabulary(_frozen(tmp_path))


def test_bundle_evaluation_rejects_explicit_limit_changes_and_uses_frozen_defaults(tmp_path, monkeypatch):
    from slbridge._io import canonical_hash
    from slbridge.cecsl import RGBInputSpec
    from slbridge.cecsl_cli import evaluate_onnx
    frozen = _frozen(tmp_path)
    bundle_spec = RGBInputSpec(max_seconds=30)
    model = tmp_path / "mock-bundle"
    model.mkdir()
    prepared_specs = []
    backend = SimpleNamespace(runtime_info={"device": "cpu"}, closed=False)
    backend.close = lambda: setattr(backend, "closed", True)
    backend.predict_logits = lambda prepared: _predict_a(prepared)[0]
    manifest = SimpleNamespace(vocabulary_hash=canonical_hash(frozen["vocabulary"]),
                               artifacts={"model.onnx": "a" * 64}, input_spec=bundle_spec)
    monkeypatch.setattr("slbridge.sequence.load_sequence_bundle",
                        lambda path, options: SimpleNamespace(manifest=manifest, backend=backend))
    monkeypatch.setattr("slbridge.cecsl_cli._require_checkpoint_vocabulary", lambda frozen: None)
    def prepare(path, spec):
        prepared_specs.append(spec)
        return SimpleNamespace(source_frames=4, values=np.zeros((1, 16, 3, 224, 224), np.float32), spec=spec)
    monkeypatch.setattr("slbridge.cecsl_cli.prepare_rgb_file", prepare)
    with pytest.raises(ValueError, match="Explicit RGB limits"):
        evaluate_onnx(frozen["root"], model, tmp_path / "mismatch.json", limit=1, spec=RGBInputSpec())
    assert backend.closed and not prepared_specs
    assert not (tmp_path / "mismatch.json").exists()
    backend.closed = False
    report = evaluate_onnx(frozen["root"], model, tmp_path / "defaults.json", limit=1)
    assert report["metrics"]["samples"] == 1 and prepared_specs == [bundle_spec]
    assert report["identity"]["input_spec"] == bundle_spec.to_dict()


def test_evaluate_cli_only_passes_rgb_limits_when_explicitly_requested(tmp_path, monkeypatch):
    from slbridge.cecsl import RGBInputSpec
    frozen = _frozen(tmp_path)
    calls = []
    def evaluate(frozen_dir, model, output, **kwargs):
        calls.append(kwargs)
        return {"status": "complete", "scope": "selected_official_split_subset", "metrics": {},
                "sample_journal": "synthetic"}
    monkeypatch.setattr("slbridge.cecsl_cli.evaluate_onnx", evaluate)
    arguments = ["evaluate", "--frozen-dir", str(frozen["root"]), "--model", "mock",
                 "--output", str(tmp_path / "out.json"), "--limit", "1"]
    assert main(arguments) == 0
    assert "spec" not in calls[0]
    assert calls[0]["options"].intra_op_threads == 4 and calls[0]["options"].gpu_mem_limit_mb == 8192
    assert main(arguments + ["--max-seconds", "10", "--gpu-mem-limit-mb", "12288"]) == 0
    assert calls[1]["spec"] == RGBInputSpec(max_seconds=10)
    assert calls[1]["options"].gpu_mem_limit_mb == 12288
