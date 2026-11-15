"""Mechanism tests for the independent probe; no checkpoints or GPU required."""
from types import SimpleNamespace
import weakref

import pytest

from scripts.probe_unisign_scores import auc, generation_capture, score_metadata, segments, verify_checkpoint
from scripts.validate_realtime_pipeline import sha
from scripts import probe_unisign_scores as probe


def test_actual_weight_bytes_are_verified_before_model_loading(tmp_path):
    checkpoint=tmp_path/'checkpoint.bin';checkpoint.write_bytes(b'weight-a')
    (tmp_path/'manifest.json').write_text('{}')
    config=SimpleNamespace(checkpoint=checkpoint,checkpoint_bytes=8,checkpoint_sha256=sha(checkpoint),bundle_dir=tmp_path)
    identity=verify_checkpoint(config);assert identity['actual_sha256']==config.checkpoint_sha256
    checkpoint.write_bytes(b'weight-b')
    with pytest.raises(ValueError,match='checkpoint'):verify_checkpoint(config)


@pytest.mark.parametrize('during_load',[True,False])
def test_initialization_failure_closes_the_diagnostic_runtime(tmp_path,monkeypatch,during_load):
    from slbridge import uni_sign
    closed=[]
    class Runtime:
        def __init__(self,*a,**kw):pass
        def load(self):
            if during_load:raise RuntimeError('fixture load failure')
        @property
        def runtime_info(self):raise RuntimeError('fixture metadata failure')
        def close(self):closed.append(True)
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(probe,'memory_preflight',lambda *a:None)
    monkeypatch.setattr(probe,'read',lambda path:dict(source_manifest_sha256=sha(tmp_path/'manifest.json'),cases=[]))
    monkeypatch.setattr(probe,'verify_checkpoint',lambda config:{})
    monkeypatch.setattr(uni_sign.UniSignConfig,'from_bundle',lambda path:SimpleNamespace())
    monkeypatch.setattr(uni_sign,'TorchUniSignRuntime',Runtime)
    with pytest.raises(RuntimeError,match='fixture'):probe.infer(tmp_path,repeats=1)
    assert closed==[True] and not (tmp_path/'summary.json').exists()


def test_frozen_boundaries_require_completed_model_outputs_and_inclusive_frame_indices():
    row=dict(request_status='ok',events=[dict(type='word'),dict(type='sentence',status='pending'),
        dict(type='sentence',status='committed',utterance_id=0,frame_start=7,frame_end=12,text='机器输出',reason='pause')])
    assert segments(row)==[dict(utterance_id=0,first=7,last=12,historical_text='机器输出',reason='pause')]
    with pytest.raises(ValueError):segments({**row,'request_status':'error'})
    for first,last in [(True,2),(-1,2),(7,6)]:
        with pytest.raises(ValueError):segments(dict(request_status='ok',events=[{**row['events'][-1],'frame_start':first,'frame_end':last}]))


def test_beam_score_capture_preserves_sequences_and_releases_full_scores_and_override():
    import torch
    retained=[]
    class Inner:
        generation_config=SimpleNamespace(length_penalty=1.)
        def generate(self,**kwargs):
            value=torch.tensor([[0,12,1]])
            if not kwargs.get('output_scores'):return value
            assert kwargs['return_dict_in_generate'] is True
            scores=(torch.zeros((4,32)),)*4
            retained.append(weakref.ref(scores[0]))
            return SimpleNamespace(sequences=value,sequences_scores=torch.tensor([-.25]),scores=scores)
    model=SimpleNamespace(mt5_model=Inner())
    with generation_capture(model,scored=False) as baseline:
        first=model.mt5_model.generate(max_new_tokens=100,num_beams=4)
    with generation_capture(model,scored=True) as scored:
        second=model.mt5_model.generate(max_new_tokens=100,num_beams=4)
    assert torch.equal(first,second) and baseline['tokens']==scored['tokens']
    assert scored['beam_score']==-.25 and scored['generated_tokens']==2 and scored['score_steps']==4
    assert scored['temporary_score_bytes']==4*4*32*4
    assert retained[0]() is None
    assert 'generate' not in model.mt5_model.__dict__
    previous=lambda **kw:(_ for _ in ()).throw(RuntimeError('generation failure'))
    model.mt5_model.generate=previous
    with pytest.raises(RuntimeError,match='generation failure'):
        with generation_capture(model,scored=True):model.mt5_model.generate()
    assert model.mt5_model.generate is previous


def test_nonfinite_or_multiple_sequence_scores_are_not_reported_as_valid():
    import torch
    for values in [[float('nan')],[float('inf')],[.1],[-.1,-.2]]:
        result=SimpleNamespace(sequences=torch.tensor([[0,1]]),sequences_scores=torch.tensor(values),scores=(torch.zeros(2,3),))
        with pytest.raises(ValueError):score_metadata(result,length_penalty=1)


def test_risk_auc_counts_ties_and_keeps_missing_class_explicit():
    assert auc([1.,2.],[0.])==1.
    assert auc([0.],[1.,2.])==0.
    assert auc([1.],[1.])==.5
    assert auc([], [1.]) is None and auc([1.],[]) is None
