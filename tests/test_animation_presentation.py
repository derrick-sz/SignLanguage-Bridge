import hashlib
import importlib
import json
import struct

import pytest


def bundle_class():
    try:
        module = importlib.import_module('slbridge.animation.presentation')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'presentation bundle is not implemented'
    return module.PresentationBundle


def make_bundle(tmp_path, *, name='avatar.glb', bad_hash=False, source='a'*64):
    data = b'glTF-test-asset'
    (tmp_path / 'avatar.glb').write_bytes(data)
    manifest = {'schema_version':1, 'avatar':name, 'attribution':'Godette CC BY 4.0; modified',
                'files':{name:('0'*64 if bad_hash else hashlib.sha256(data).hexdigest())},
                'pilots':[{'asset_id':'hello', 'source_motion_sha256':source}],
                'processing_version':'source-wrist-ik-v1'}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    return tmp_path


def test_presentation_snapshot_does_not_expose_disk_paths_or_inherit_language_approval(tmp_path):
    bundle = bundle_class()(make_bundle(tmp_path), [{'id':'hello','sha256':'a'*64}])
    info = bundle.catalog()
    assert str(tmp_path) not in json.dumps(info)
    assert info['language_review'] == 'unverified'
    assert info['avatar'] == 'avatar.glb'
    (tmp_path/'avatar.glb').write_bytes(b'changed after load')
    assert bundle.resource('avatar.glb')[0] == b'glTF-test-asset'
    with pytest.raises(KeyError):
        bundle.resource('manifest.json')


@pytest.mark.parametrize('name', ['../avatar.glb', '/avatar.glb', 'C:/avatar.glb', 'folder\\avatar.glb'])
def test_presentation_rejects_paths_outside_the_explicit_bundle(tmp_path, name):
    with pytest.raises(ValueError):
        bundle_class()(make_bundle(tmp_path,name=name), [{'id':'hello','sha256':'a'*64}])


def test_presentation_rejects_corrupt_file_and_wrong_source_motion(tmp_path):
    cls = bundle_class()
    with pytest.raises(ValueError, match='checksum'):
        cls(make_bundle(tmp_path,bad_hash=True), [{'id':'hello','sha256':'a'*64}])
    with pytest.raises(ValueError, match='source'):
        cls(make_bundle(tmp_path,source='b'*64), [{'id':'hello','sha256':'a'*64}])


def test_optional_presentation_is_served_only_through_its_allowlist(tmp_path):
    from fastapi.testclient import TestClient
    from slbridge.api import create_app
    from slbridge.animation.registry import MotionRegistry
    from test_animation_registry import make_archive
    cls = bundle_class()
    registry = MotionRegistry(make_archive(tmp_path))
    assets = registry.catalog()['assets']
    identity = next(a['sha256'] for a in assets if a['id']=='hello')
    registry.presentation = cls(make_bundle(tmp_path,source=identity), assets)
    with TestClient(create_app({},animation_registry=registry)) as client:
        assert client.get('/v1/animation/presentation').json()['avatar'] == 'avatar.glb'
        assert client.get('/animation/presentation/avatar.glb').content == b'glTF-test-asset'
        assert client.get('/animation/presentation/manifest.json').status_code == 404
        for name in ['motion-geometry.mjs','expression.mjs']:
            assert client.get('/animation/static/'+name).status_code == 200


def test_explicit_rig_profile_is_exposed_and_invalid_binding_is_rejected(tmp_path):
    cls=bundle_class()
    path=make_bundle(tmp_path)
    manifest=json.loads((path/'manifest.json').read_text())
    manifest['rig_profile']={'id':'custom-1','bones':{'leftHand':['Hand.L']},'corrections':{'leftHand':[0,0.1,0]}}
    (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    assert cls(path,[{'id':'hello','sha256':'a'*64}]).catalog()['rig_profile']['id']=='custom-1'
    manifest['rig_profile']['corrections']['leftHand']=[0,20,0]
    (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    with pytest.raises(ValueError,match='profile'):
        cls(path,[{'id':'hello','sha256':'a'*64}])


def glb_export(identity, source, avatar, *, binding=True, temporal_profile=None):
    document={'asset':{'version':'2.0'},'animations':[{'name':identity}]}
    if binding:
        document['asset']['extras']={'slbridge_motion':{'asset_id':identity,'source_motion_sha256':source,
            'avatar_sha256':avatar,'processing_version':'source-wrist-ik-v1'}}
        if temporal_profile is not None:
            document['asset']['extras']['slbridge_motion']['temporal_profile']=temporal_profile
    payload=json.dumps(document).encode();payload+=b' '*((-len(payload))%4)
    return struct.pack('<4sII',b'glTF',2,20+len(payload))+struct.pack('<II',len(payload),0x4e4f534a)+payload


def test_unknown_pilot_or_missing_source_digest_is_rejected(tmp_path):
    cls=bundle_class();path=make_bundle(tmp_path)
    manifest=json.loads((path/'manifest.json').read_text())
    for pilot in [{'asset_id':'not-in-registry'},{'asset_id':'hello'},
                  {'asset_id':'not-in-registry','source_motion_sha256':'a'*64}]:
        manifest['pilots']=[pilot];(path/'manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='source'):
            cls(path,[{'id':'hello','sha256':'a'*64}])


def test_pilot_review_flags_are_reset_and_private_fields_are_not_published(tmp_path):
    cls=bundle_class();path=make_bundle(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    manifest['pilots'][0].update(language_review='approved',motion_review='approved',expression_review='approved',local_path='private/path')
    (path/'manifest.json').write_text(json.dumps(manifest))
    pilot=cls(path,[{'id':'hello','sha256':'a'*64}]).catalog()['pilots'][0]
    assert all(pilot[k]=='unverified' for k in ('language_review','motion_review','expression_review'))
    assert 'local_path' not in pilot


def test_valid_hashes_cannot_bind_another_pilots_animation_or_another_avatar(tmp_path):
    cls=bundle_class();path=make_bundle(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    avatar=manifest['files']['avatar.glb']
    for name,identity,source in [('hello.glb','hello','a'*64),('second.glb','second','b'*64)]:
        data=glb_export(identity,source,avatar);(path/name).write_bytes(data);manifest['files'][name]=hashlib.sha256(data).hexdigest()
    manifest['pilots'][0]['export_file']='hello.glb'
    (path/'manifest.json').write_text(json.dumps(manifest))
    assert cls(path,[{'id':'hello','sha256':'a'*64}]).catalog()['pilots'][0]['export_file']=='hello.glb'
    manifest['pilots'][0]['export_file']='second.glb'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='binding'):
        cls(path,[{'id':'hello','sha256':'a'*64}])
    manifest['pilots'][0]['export_file']='hello.glb'
    wrong=glb_export('hello','a'*64,'c'*64);(path/'hello.glb').write_bytes(wrong)
    manifest['files']['hello.glb']=hashlib.sha256(wrong).hexdigest();(path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='binding'):
        cls(path,[{'id':'hello','sha256':'a'*64}])


def test_build_export_embeds_identity_without_changing_animation(tmp_path):
    module=importlib.import_module('scripts.animation.build_presentation')
    assert hasattr(module,'bind_export'), 'producer does not embed export binding'
    path=tmp_path/'hello.glb';path.write_bytes(glb_export('hello','a'*64,'b'*64,binding=False))
    binding={'asset_id':'hello','source_motion_sha256':'a'*64,'avatar_sha256':'b'*64,'processing_version':'source-wrist-ik-v1'}
    module.bind_export(path,binding)
    data=path.read_bytes();length,kind=struct.unpack_from('<II',data,12)
    document=json.loads(data[20:20+length])
    assert document['asset']['extras']['slbridge_motion']==binding
    assert document['animations']==[{'name':'hello'}]
    assert struct.unpack_from('<I',data,8)[0]==len(data)


def test_temporal_repair_profile_is_explicit_and_bound_to_the_export(tmp_path):
    cls=bundle_class();path=make_bundle(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    profile='source-quaternion-temporal-v1'
    manifest['pilots'][0].update(temporal_profile=profile,export_file='hello.glb')
    data=glb_export('hello','a'*64,manifest['files']['avatar.glb'],temporal_profile=profile)
    (path/'hello.glb').write_bytes(data);manifest['files']['hello.glb']=hashlib.sha256(data).hexdigest()
    (path/'manifest.json').write_text(json.dumps(manifest))
    assert cls(path,[{'id':'hello','sha256':'a'*64}]).catalog()['pilots'][0]['temporal_profile']==profile
    del manifest['pilots'][0]['temporal_profile']
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='binding'):
        cls(path,[{'id':'hello','sha256':'a'*64}])


def test_unknown_temporal_repair_profile_is_rejected(tmp_path):
    cls=bundle_class();path=make_bundle(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    manifest['pilots'][0]['temporal_profile']='unverified-arbitrary-filter'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='temporal'):
        cls(path,[{'id':'hello','sha256':'a'*64}])
