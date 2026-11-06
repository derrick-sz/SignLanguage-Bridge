"""New-text requests must never silently become dataset/target-pose sampling."""
import hashlib

import numpy as np
import pytest


def test_new_text_condition_uses_only_explicit_length_and_zero_observations():
    from slbridge.animation.signspark_generation import GenerationRequest, source_free_condition
    request=GenerationRequest('明天下午我去医院。', frames=96, steps=20, seed=7)
    condition=source_free_condition(request,'body')
    assert condition['y']['text']==['<Chinese> 明天下午我去医院。']
    assert condition['y']['lengths'].tolist()==[96]
    assert condition['y']['mask'].shape==(1,1,304)
    assert condition['y']['mask'].sum()==96
    assert condition['obs_x0'].shape==(1,60,304)
    assert not condition['obs_x0'].any() and not condition['obs_mask'].any()
    assert condition['obs_mask'].dtype==np.bool_


@pytest.mark.parametrize('values',[{'text':''},{'text':None},{'text':'a'*201},{'text':'我\x00去'},
    {'frames':15},{'frames':305},{'frames':True},{'steps':0},{'steps':101},{'seed':-1},{'seed':2**32}])
def test_unbounded_or_invalid_generation_is_rejected_before_model_load(values):
    from slbridge.animation.signspark_generation import GenerationRequest
    args={'text':'我去医院。'}|values
    with pytest.raises(ValueError):GenerationRequest(**args)


@pytest.mark.parametrize('stream,width',[('hand',90),('body',60),('face',56)])
def test_stream_dimensions_follow_author_protocol(stream,width):
    from slbridge.animation.signspark_generation import GenerationRequest,source_free_condition
    condition=source_free_condition(GenerationRequest('你好。'),stream)
    assert condition['obs_x0'].shape==(1,width,304)


def test_target_pose_cannot_be_passed_to_new_text_builder():
    from slbridge.animation.signspark_generation import GenerationRequest,source_free_condition
    with pytest.raises(TypeError):source_free_condition(GenerationRequest('你好。'),'body',target_poses=np.ones((3,60)))


@pytest.mark.parametrize('assign',[False,True])
def test_verified_checkpoint_loads_strictly_and_rejects_missing_keys(tmp_path,assign):
    torch=pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    model=torch.nn.Linear(2,1)
    path=tmp_path/'weights.pt'
    torch.save({'weight':torch.tensor([[2.,3.]]),'bias':torch.tensor([4.])},path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    load_strict_checkpoint(model,path,expected_sha256=digest,expected_bytes=path.stat().st_size,assign=assign)
    assert model(torch.tensor([[1.,1.]])).item()==9.
    path=tmp_path/'incomplete.pt'  # Never overwrite a checkpoint still mapped by a model.
    torch.save({'weight':torch.tensor([[2.,3.]])},path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='Missing key'):load_strict_checkpoint(model,path,expected_sha256=digest,expected_bytes=path.stat().st_size,assign=assign)


def test_checkpoint_bytes_must_match_before_deserializing(tmp_path):
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path=tmp_path/'bad.pt';path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='SHA|checksum'):
        load_strict_checkpoint(None,path,expected_sha256='0'*64,expected_bytes=path.stat().st_size)


def test_assign_loader_shares_real_loaded_state_storage_and_keeps_frozen_flags(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    model=torch.nn.Linear(2,1).requires_grad_(False)
    old_weight=model.weight
    path=tmp_path/'weights.pt'
    torch.save({'weight':torch.tensor([[2.,3.]]),'bias':torch.tensor([4.])},path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    original_load=torch.load
    observed={}

    def capture_real_load(*args,**kwargs):
        state=original_load(*args,**kwargs)
        observed.update(state=state,options=kwargs)
        return state

    monkeypatch.setattr(torch,'load',capture_real_load)
    load_strict_checkpoint(model,path,expected_sha256=digest,expected_bytes=path.stat().st_size,assign=True)
    assert observed['options']['mmap'] is True and observed['options']['weights_only'] is True
    assert model.weight.data_ptr()==observed['state']['weight'].data_ptr()
    assert model.bias.data_ptr()==observed['state']['bias'].data_ptr()
    assert model.weight.data_ptr()!=old_weight.data_ptr()
    assert not model.weight.requires_grad and not model.bias.requires_grad
    assert model(torch.tensor([[1.,1.]])).item()==9.
