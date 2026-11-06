"""Invalid target assets must not waste a pretrained GPU inference run."""
import importlib.util
from pathlib import Path

import pytest


def test_bad_presentation_fails_before_checkpoint_or_gpu_work(tmp_path,monkeypatch,capsys):
    path=Path(__file__).resolve().parents[1]/'scripts/animation/generate_custom_signspark.py'
    spec=importlib.util.spec_from_file_location('signspark_presentation_cli',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    for name in ('verify_base_archive','verify_source','verify_text_assets'):
        monkeypatch.setattr(worker,name,lambda *args:'verified-by-earlier-contract')
    def unexpected(*args,**kwargs):
        pytest.fail('Invalid presentation reached large-model allocation')
    monkeypatch.setattr(worker,'cold_model',unexpected)
    with pytest.raises(SystemExit) as error:
        worker.main(['--text','新句。','--source-presentation',str(tmp_path/'absent-avatar'),
                     '--smplx-model',str(tmp_path/'absent-template.npz'),
                     '--output',str(tmp_path/'output')])
    assert error.value.code==2
    assert 'manifest.json' in capsys.readouterr().err
    assert not (tmp_path/'output').exists()
