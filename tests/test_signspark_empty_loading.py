"""Small actual Torch models exercise parameter-only meta loading, without CUDA."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def worker():
    path=Path(__file__).resolve().parents[1]/'scripts/animation/generate_custom_signspark.py'
    spec=importlib.util.spec_from_file_location('signspark_empty_worker',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def checkpoint(tmp_path,torch,state):
    path=tmp_path/'state.pt';torch.save(state,path)
    return path,{'expected_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'expected_bytes':path.stat().st_size}


def test_empty_wrapper_keeps_actual_xlmr_buffers_and_strict_assign_forward(worker,tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('accelerate')
    pytest.importorskip('transformers')
    from transformers import AutoModel,XLMRobertaConfig
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    config=XLMRobertaConfig(vocab_size=97,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                           intermediate_size=32,max_position_embeddings=32)
    def factory(*args):return AutoModel.from_config(config,add_pooling_layer=True),object()
    normal,_=factory();normal.eval().requires_grad_(False)
    path,identity=checkpoint(tmp_path,torch,normal.state_dict())
    monkeypatch.setattr(worker,'cold_model',factory)
    assert hasattr(worker,'empty_model'),'parameter-only constructor missing'
    empty,_=worker.empty_model('face',tmp_path)
    assert all(p.is_meta for p in empty.parameters())
    assert all(b.device.type=='cpu' for b in empty.buffers())
    assert torch.equal(empty.embeddings.position_ids,normal.embeddings.position_ids)
    assert torch.equal(empty.embeddings.token_type_ids,normal.embeddings.token_type_ids)
    report=load_strict_checkpoint(empty,path,**identity,assign=True)
    assert report['missing_keys']==report['unexpected_keys']==[]
    assert all(not t.is_meta for t in [*empty.parameters(),*empty.buffers()])
    empty.eval().requires_grad_(False).to('cpu')
    with torch.no_grad():
        ids=torch.tensor([[0,4,5,2]])
        torch.testing.assert_close(normal(ids).last_hidden_state,empty(ids).last_hidden_state,rtol=0,atol=0)
    assert not torch.cuda.is_initialized()


def test_empty_wrapper_rejects_nonmeta_parameters_and_meta_buffers(worker,monkeypatch,tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('accelerate')
    normal=torch.nn.Linear(2,1)
    monkeypatch.setattr(worker,'cold_model',lambda *args:(normal,None))
    assert hasattr(worker,'empty_model'),'parameter-only constructor missing'
    with pytest.raises(ValueError,match='meta parameter'):worker.empty_model('face',tmp_path)
    def wrong_buffers(*args):
        model=torch.nn.Linear(2,1)
        model.register_buffer('position_ids',torch.empty(2,device='meta'),persistent=False)
        return model,None
    monkeypatch.setattr(worker,'cold_model',wrong_buffers)
    with pytest.raises(ValueError,match='buffer'):worker.empty_model('face',tmp_path)


def test_meta_without_assign_fails_before_deserialization(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path,identity=checkpoint(tmp_path,torch,torch.nn.Linear(2,1).state_dict())
    with torch.device('meta'):model=torch.nn.Linear(2,1)
    def forbidden(*args,**kwargs):pytest.fail('meta assign=False reached deserialization')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='assign=True'):load_strict_checkpoint(model,path,**identity)


def test_strict_assign_rejects_unmaterialized_nonpersistent_buffer(tmp_path):
    torch=pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path,identity=checkpoint(tmp_path,torch,torch.nn.Linear(2,1).state_dict())
    with torch.device('meta'):
        model=torch.nn.Linear(2,1)
        model.register_buffer('position_ids',torch.arange(2),persistent=False)
    with pytest.raises(ValueError,match='meta.*buffer'):
        load_strict_checkpoint(model,path,**identity,assign=True)


def test_memory_snapshot_does_not_initialize_cuda_and_reads_windows_bytes(worker):
    class Cuda:
        def is_initialized(self):return False
        def __getattr__(self,name):pytest.fail('snapshot initialized or queried unavailable CUDA: '+name)
    assert hasattr(worker,'memory_snapshot'),'stage memory observation missing'
    value=worker.memory_snapshot(SimpleNamespace(cuda=Cuda()))
    assert value['cuda']=={'initialized':False}
    assert value['pid']>0 and value['time_unix']>0
    if sys.platform=='win32':
        assert value['process']['rss_bytes']>0
        assert value['process']['peak_rss_bytes']>=value['process']['rss_bytes']
        assert value['process']['private_bytes']>0
        assert value['host']['commit_limit_bytes']>=value['host']['committed_bytes']>0
        assert value['host']['available_bytes']>0
    json.dumps(value,allow_nan=False)


def test_failed_cuda_transfer_preserves_after_load_and_failure_snapshots(worker,tmp_path,monkeypatch):
    for name in ('verify_base_archive','verify_source','verify_text_assets','verify_presentation_inputs'):
        monkeypatch.setattr(worker,name,lambda *args,**kwargs:'verified')
    monkeypatch.setattr(worker,'MODEL_FILES',{'face':(4,'a'*64),'body':(4,'b'*64),'hand':(4,'c'*64)})
    models=tmp_path/'models'
    for name in worker.MODEL_FILES:
        folder=models/name;folder.mkdir(parents=True);(folder/'ema_0.9999_200000.pt').write_bytes(b'fake')
    class Model:
        def eval(self):return self
        def requires_grad_(self,value):return self
        def to(self,device):raise RuntimeError('fixture allocation refusal')
    monkeypatch.setattr(worker,'verified_file',lambda *args:None)
    assert hasattr(worker,'empty_model') and hasattr(worker,'memory_snapshot')
    monkeypatch.setattr(worker,'empty_model',lambda *args:(Model(),None))
    monkeypatch.setattr(worker,'load_strict_checkpoint',lambda *args,**kwargs:{'tensors':3,'missing_keys':[],'unexpected_keys':[]})
    monkeypatch.setattr(worker,'memory_snapshot',lambda *args,**kwargs:{'pid':123,'cuda':{'initialized':True,'allocated_bytes':42}})
    fake=SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace()),
                         cuda=SimpleNamespace(is_available=lambda:True,empty_cache=lambda:None,reset_peak_memory_stats=lambda:None))
    monkeypatch.setitem(sys.modules,'torch',fake)
    output=tmp_path/'out'
    with pytest.raises(RuntimeError,match='fixture allocation refusal'):
        worker.main(['--text','新句。','--models',str(models),'--output',str(output)])
    manifest=json.loads((output/'generation-manifest.json').read_text(encoding='utf8'))
    assert manifest['state']=='failed'
    assert [x['stage'] for x in manifest['memory_snapshots']]==[
        'before_construct','after_construct','after_load','before_cuda','failure']
    assert manifest['memory_snapshots'][-1]['failed_after']=='before_cuda'
    assert manifest['checkpoint_loads']['body']['tensors']==3


@pytest.mark.parametrize('stderr_fails',[False,True])
def test_failure_snapshot_write_and_stderr_errors_never_replace_primary_error(worker,tmp_path,monkeypatch,capsys,stderr_fails):
    for name in ('verify_base_archive','verify_source','verify_text_assets','verify_presentation_inputs'):
        monkeypatch.setattr(worker,name,lambda *args,**kwargs:'verified')
    monkeypatch.setattr(worker,'MODEL_FILES',{'body':(4,'a'*64),'hand':(4,'b'*64),'face':(4,'c'*64)})
    models=tmp_path/'models'
    for name in worker.MODEL_FILES:
        folder=models/name;folder.mkdir(parents=True);(folder/'ema_0.9999_200000.pt').write_bytes(b'fake')
    primary=RuntimeError('original CUDA allocation refusal')
    class Model:
        def eval(self):return self
        def requires_grad_(self,value):return self
        def to(self,device):raise primary
    monkeypatch.setattr(worker,'verified_file',lambda *args:None)
    monkeypatch.setattr(worker,'empty_model',lambda *args:(Model(),None))
    monkeypatch.setattr(worker,'load_strict_checkpoint',lambda *args,**kwargs:{'tensors':3,'missing_keys':[],'unexpected_keys':[]})
    monkeypatch.setattr(worker,'memory_snapshot',lambda *args,**kwargs:{'pid':123,'cuda':{'initialized':True}})
    fake=SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace()),
        cuda=SimpleNamespace(is_available=lambda:True,empty_cache=lambda:None,reset_peak_memory_stats=lambda:None))
    monkeypatch.setitem(sys.modules,'torch',fake)
    original_write=worker.json_write;failed_manifests=[]
    def denied_failure_write(path,manifest):
        if manifest['state']=='failed':
            failed_manifests.append(manifest)
            raise OSError('failure snapshot write denied')
        original_write(path,manifest)
    monkeypatch.setattr(worker,'json_write',denied_failure_write)
    if stderr_fails:
        def denied_stderr(*args,**kwargs):raise OSError('stderr write denied')
        monkeypatch.setattr(worker,'print',denied_stderr,raising=False)
    output=tmp_path/'out'
    with pytest.raises(RuntimeError) as raised:
        worker.main(['--text','新句。','--models',str(models),'--output',str(output)])
    assert raised.value is primary
    assert failed_manifests and failed_manifests[-1]['state']=='failed'
    assert 'original CUDA allocation refusal' in failed_manifests[-1]['error']
    persisted=json.loads((output/'generation-manifest.json').read_text(encoding='utf8'))
    assert persisted['state']=='running','failed diagnostic write must not be reported as persisted'
    if not stderr_fails:
        assert 'failure snapshot write denied' in capsys.readouterr().err
