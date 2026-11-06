"""Preflight failures must precede GPU/model allocation and output publication."""
from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest


SCRIPT=Path(__file__).resolve().parents[1]/'scripts/animation/generate_custom_signspark.py'


def test_invalid_new_text_rejected_before_any_source_or_model_access(tmp_path):
    result=subprocess.run([sys.executable,str(SCRIPT),'--text','','--source',str(tmp_path/'absent'),
                           '--models',str(tmp_path/'models'),'--output',str(tmp_path/'result')],capture_output=True,text=True)
    assert result.returncode==2
    assert 'Invalid text' in result.stderr
    assert not (tmp_path/'result').exists()


def test_missing_pinned_source_manifest_is_a_clean_preflight_failure(tmp_path,monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('custom_signspark_cli_fixture',SCRIPT)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    # This case targets source preflight, independently of any machine's E: ZIP.
    monkeypatch.setattr(worker,'verify_base_archive',lambda path:'0'*64)
    def forbidden_model(*args,**kwargs):
        pytest.fail('Missing source reached model allocation')
    monkeypatch.setattr(worker,'cold_model',forbidden_model)
    with pytest.raises(SystemExit) as error:
        worker.main(['--text','明天我去医院。','--source',str(tmp_path/'source'),
                     '--source-manifest',str(tmp_path/'absent.json'),
                     '--base-archive',str(tmp_path/'unused-fixture.zip'),
                     '--models',str(tmp_path/'models'),'--output',str(tmp_path/'result')])
    assert error.value.code==2
    assert 'source manifest' in capsys.readouterr().err.lower()
    assert not (tmp_path/'result').exists()
