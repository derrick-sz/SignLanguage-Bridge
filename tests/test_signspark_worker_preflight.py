"""Worker preflight must fail before model construction for invalid assets."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def worker():
    script = Path(__file__).resolve().parents[1] / "scripts/animation/generate_custom_signspark.py"
    spec = importlib.util.spec_from_file_location("signspark_worker_preflight", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_base_archive_fails_before_source_or_model_work(worker, tmp_path, monkeypatch, capsys):
    wrong = tmp_path / "base.zip"
    wrong.write_bytes(b"unverified avatar archive")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid base archive reached source/model work")

    monkeypatch.setattr(worker, "verify_source", unexpected)
    monkeypatch.setattr(worker, "cold_model", unexpected)
    with pytest.raises(SystemExit) as error:
        worker.main(["--text", "今天去图书馆。", "--base-archive", str(wrong),
                     "--output", str(tmp_path / "out")])
    assert error.value.code == 2
    assert "checksum" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_empty_self_declared_source_manifest_is_not_author_evidence(worker, tmp_path):
    receipt = tmp_path / "source-bundle.json"
    receipt.write_text(json.dumps({"commit": worker.SOURCE_COMMIT, "files": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.*SHA256"):
        worker.verify_source(tmp_path, receipt)


def test_rewritten_source_and_manifest_cannot_redefine_pinned_commit(worker, tmp_path):
    source = tmp_path / "unet.py"
    source.write_bytes(b"not the official model")
    raw = source.read_bytes()
    receipt = tmp_path / "source-bundle.json"
    receipt.write_text(json.dumps({"repository":"JianHe0628/SignSparK","commit": worker.SOURCE_COMMIT, "files": [
        {"path": source.name,"size":len(raw), "blob_sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.*SHA256"):
        worker.verify_source(tmp_path, receipt)


def test_authenticated_source_receipt_checks_each_file(worker, tmp_path, monkeypatch):
    raw = b"# fixture source\n"
    source = tmp_path / "fixture.py"
    source.write_bytes(raw)
    receipt = tmp_path / "source-bundle.json"
    receipt.write_text(json.dumps({"repository":"JianHe0628/SignSparK","commit": worker.SOURCE_COMMIT, "files": [
        {"path": source.name,"size":len(raw), "blob_sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}
    ]}), encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "SOURCE_BUNDLE_SHA256", digest)
    assert worker.verify_source(tmp_path, receipt) == digest
    source.write_bytes(b"# modified source\n")
    with pytest.raises(ValueError, match="source file changed"):
        worker.verify_source(tmp_path, receipt)


def test_text_assets_cannot_skip_tokenizer_verification_with_empty_receipt(worker, tmp_path):
    (tmp_path / "manifest.json").write_text('{"files": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="text asset manifest.*SHA256"):
        worker.verify_text_assets(tmp_path)


def test_raw_stream_is_saved_independently_with_exact_values_and_identity(worker,tmp_path):
    raw=np.arange(16*60,dtype=np.float32).reshape(16,60)/100
    record=worker.save_raw_stream(tmp_path,0,'body',raw)
    path=tmp_path/record['path']
    assert path.name=='raw-stream-0-body.npz'
    assert record['request_index']==0 and record['role']=='body'
    assert record['shape']==[16,60] and record['dtype']=='float32'
    assert record['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['values'],raw)
    original=path.read_bytes()
    with pytest.raises(ValueError,match='already exists'):
        worker.save_raw_stream(tmp_path,0,'body',raw+1)
    assert path.read_bytes()==original
