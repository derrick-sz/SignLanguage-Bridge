"""Fixture-first publication and explicit nonmanual protocol contracts."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import struct

import pytest


def helper():
    assert importlib.util.find_spec('slbridge.animation.signspark_presentation') is not None, 'presentation helper missing'
    return importlib.import_module('slbridge.animation.signspark_presentation')


@pytest.fixture
def source(tmp_path,monkeypatch):
    module=helper()
    folder=tmp_path/'verified-source';folder.mkdir()
    payload=json.dumps({'asset':{'version':'2.0'},'scenes':[{'nodes':[]}]}).encode()
    payload+=b' '*((-len(payload))%4)
    avatar=struct.pack('<4sII',b'glTF',2,20+len(payload))+struct.pack('<II',len(payload),0x4e4f534a)+payload
    (folder/'avatar-clean.glb').write_bytes(avatar)
    old=b'old baked motion must not be read or copied';(folder/'old.glb').write_bytes(old)
    avatar_sha=hashlib.sha256(avatar).hexdigest()
    manifest={'schema_version':1,'avatar':'avatar-clean.glb','files':{'avatar-clean.glb':avatar_sha,'old.glb':hashlib.sha256(old).hexdigest()},
              'attribution':'Fixture attributed derivative','rig_profile':{'id':'godette-v1','bones':{'jaw':['Jaw_4']}},
              'pilots':[{'asset_id':'old','source_motion_sha256':'a'*64,'export_file':'old.glb','language_review':'approved'}]}
    raw=json.dumps(manifest).encode();(folder/'manifest.json').write_bytes(raw)
    template=tmp_path/'template.npz';template.write_bytes(b'fixture numeric template')
    candidate=deepcopy(module.JAW_CANDIDATE)
    candidate.update(source_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),target_avatar_sha256=avatar_sha)
    monkeypatch.setattr(module,'JAW_CANDIDATE',candidate)
    monkeypatch.setattr(module,'SOURCE_MANIFEST_SHA256',hashlib.sha256(raw).hexdigest())
    return module,folder,template,manifest


def test_builder_copies_only_verified_avatar_and_new_source_bound_pilots(source,tmp_path):
    module,folder,_,original=source
    (folder/'old.glb').unlink()  # old baked tracks are not an input dependency
    output=tmp_path/'generated'/'presentation'
    assets=[{'id':'generated-csl-new','sha256':'b'*64,'text':'新句','export_file':'old.glb','language_review':'approved'}]
    manifest=module.build_avatar_presentation(output,assets=assets,source_presentation=folder)
    assert {p.name for p in output.iterdir()}=={'avatar-clean.glb','manifest.json'}
    assert set(manifest['files'])=={'avatar-clean.glb'}
    assert manifest['rig_profile']==original['rig_profile'] and manifest['attribution']==original['attribution']
    assert manifest['export_supported'] is False and manifest['status']=='research_candidate'
    from slbridge.animation.presentation import PresentationBundle
    bundle=PresentationBundle(output,assets)
    pilot=bundle.catalog()['pilots'][0]
    assert pilot['asset_id']=='generated-csl-new' and 'export_file' not in pilot
    assert pilot['language_review']==pilot['motion_review']==pilot['expression_review']=='unverified'
    assert bundle.resource('avatar-clean.glb')[0]==(folder/'avatar-clean.glb').read_bytes()
    with pytest.raises(KeyError):bundle.resource('old.glb')


def generated_clip():
    poses=[[[0.,0.,0.] for _ in range(55)] for _ in range(2)]
    poses[0][22]=[.08,.01,0.];poses[1][22]=[.09,.01,0.]
    poses[0][25]=[.4,.2,.1]
    return {'id':'generated-csl-new','text':'新句','fps':30.,'frames':2,'poses':poses,
            'source_kind':'pretrained_text_conditioned_generation','generation_codec':'signspark-row6d-flat-hand-v1',
            'expression_channels_retained':50,'expression':[[.01*i for i in range(50)] for _ in range(2)],
            'unmapped_target_channels':['expression','jaw','eyes'],'language_review':'approved','human_approved':True}


def test_explicit_jaw_opt_in_authenticates_both_files_and_preserves_flat_hand_and_expression50(source):
    module,folder,template,_=source
    clip=generated_clip();before=deepcopy(clip)
    assert hasattr(module,'opt_in_generated_jaw')
    result=module.opt_in_generated_jaw(clip,smplx_model=template,source_presentation=folder)
    assert result is not clip and clip==before
    assert result['poses']==clip['poses'] and result['expression']==clip['expression']
    assert result['poses'][0][22]==[.08,.01,0.] and result['poses'][0][25]==[.4,.2,.1]
    assert result['expression_channels_retained']==50
    assert result['unmapped_target_channels']==['expression','jaw_swing','eyes']
    assert result['nonmanual_candidate']==module.JAW_CANDIDATE
    assert result['template_sha256']==hashlib.sha256(template.read_bytes()).hexdigest()
    assert result['presentation_status']=='research_candidate' and result['export_supported'] is False
    assert result['human_approved'] is False
    assert result['language_review']==result['motion_review']==result['expression_review']=='unverified'
    template.write_bytes(b'wrong template')
    with pytest.raises(ValueError,match='SHA256'):
        module.opt_in_generated_jaw(clip,smplx_model=template,source_presentation=folder)
    assert clip==before


@pytest.mark.parametrize('fault',['avatar','manifest','avatar-and-declared-hash'])
def test_changed_source_cannot_publish_an_avatar_package_or_opt_in(source,tmp_path,fault):
    module,folder,template,manifest=source
    if fault in ('avatar','avatar-and-declared-hash'):(folder/'avatar-clean.glb').write_bytes(b'replaced avatar')
    if fault in ('manifest','avatar-and-declared-hash'):
        if fault=='manifest':manifest['rig_profile']['bones']['jaw']=['UnverifiedJaw']
        else:manifest['files']['avatar-clean.glb']=hashlib.sha256(b'replaced avatar').hexdigest()
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    output=tmp_path/'not-published'
    with pytest.raises(ValueError,match='SHA256'):
        module.build_avatar_presentation(output,source_presentation=folder)
    assert not output.exists()
    clip=generated_clip();before=deepcopy(clip)
    with pytest.raises(ValueError,match='SHA256'):
        module.opt_in_generated_jaw(clip,smplx_model=template,source_presentation=folder)
    assert clip==before


@pytest.mark.parametrize('patch',[
    {'generation_codec':'soke-axis-angle-plus-handmean'},
    {'expression_channels_retained':10},
    {'unmapped_target_channels':['jaw']},
    {'template_sha256':'0'*64},
    {'nonmanual_candidate':{'profile':'other'}},
])
def test_unverified_protocols_are_not_silently_relabelled_as_signspark(source,patch):
    module,folder,template,_=source
    clip=generated_clip()|patch;before=deepcopy(clip)
    with pytest.raises(ValueError):module.opt_in_generated_jaw(clip,smplx_model=template,source_presentation=folder)
    assert clip==before


def test_existing_output_is_never_replaced_and_empty_pilots_are_supported(source,tmp_path):
    module,folder,_,_=source;output=tmp_path/'presentation'
    result=module.build_avatar_presentation(output,source_presentation=folder)
    before={p.name:p.read_bytes() for p in output.iterdir()}
    assert result['pilots']==[]
    with pytest.raises(ValueError,match='new directory'):
        module.build_avatar_presentation(output,source_presentation=folder)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before


def test_bad_new_pilot_is_rejected_before_publication(source,tmp_path):
    module,folder,_,_=source;output=tmp_path/'presentation'
    with pytest.raises(ValueError):
        module.build_avatar_presentation(output,assets=[{'id':'new','sha256':'not-a-digest'}],source_presentation=folder)
    assert not output.exists()
    assert not list(tmp_path.glob('.presentation-*'))


def test_real_verified_avatar_and_template_optional_integration(tmp_path):
    import os
    if os.environ.get('SIGNBRIDGE_REAL_PRESENTATION_TEST')!='1':pytest.skip('explicit real-asset integration opt-in')
    module=helper()
    template=Path('G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz')
    result=module.opt_in_generated_jaw(generated_clip(),smplx_model=template)
    output=tmp_path/'presentation'
    manifest=module.build_avatar_presentation(output)
    from slbridge.animation.presentation import PresentationBundle
    bundle=PresentationBundle(output,[])
    assert bundle.catalog()['avatar_sha256']==result['nonmanual_candidate']['target_avatar_sha256']
    assert len(manifest['files'])==1 and len(list(output.glob('*.glb')))==1
