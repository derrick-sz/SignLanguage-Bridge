import importlib
import json
import struct

import pytest


def baked_clip(folder, identity):
    binary=struct.pack('<8f',0,0.04,0,0,0,0,0,0)
    document={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
              'nodes':[{'name':'Hand'}],'buffers':[{'byteLength':len(binary)}],
              'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':8},
                             {'buffer':0,'byteOffset':8,'byteLength':24}],
              'accessors':[{'bufferView':0,'componentType':5126,'type':'SCALAR','count':2},
                           {'bufferView':1,'componentType':5126,'type':'VEC3','count':2}],
              'animations':[{'samplers':[{'input':0,'output':1}],
                             'channels':[{'sampler':0,'target':{'node':0,'path':'translation'}}]}]}
    payload=json.dumps(document).encode();payload+=b' '*((-len(payload))%4)
    data=struct.pack('<4sII',b'glTF',2,28+len(payload)+len(binary))+struct.pack('<II',len(payload),0x4e4f534a)+payload+struct.pack('<II',len(binary),0x004e4942)+binary
    (folder/(identity+'.glb')).write_bytes(data)
    matrix=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
    track={'asset_id':identity,'frames':2,'fps':25,'boneNames':{'hand':'Hand'},
           'worldFrames':[{'hand':matrix},{'hand':matrix}]}
    (folder/(identity+'.tracks.json')).write_text(json.dumps(track))


def test_export_verifier_uses_declared_pilots_instead_of_fixed_five(tmp_path):
    module=importlib.import_module('scripts.animation.verify_exports')
    for identity in ['first','second']:
        baked_clip(tmp_path,identity)
    (tmp_path/'manifest.json').write_text(json.dumps({'pilots':[{'asset_id':'first'},{'asset_id':'second'}]}))
    reports=module.verify(tmp_path)
    assert {r['asset_id'] for r in reports}=={'first','second'}
    assert all(r['frames_checked']==2 and r['maximum_joint_position_difference']==0 and r['passed'] for r in reports)


@pytest.mark.parametrize('declared',[['first','second','third','fourth','missing'],['first','second','third','fourth']])
def test_export_verifier_rejects_missing_or_unlisted_clip_identities(tmp_path,declared):
    module=importlib.import_module('scripts.animation.verify_exports')
    for identity in ['first','second','third','fourth','fifth']:
        baked_clip(tmp_path,identity)
    (tmp_path/'manifest.json').write_text(json.dumps({'pilots':[{'asset_id':identity} for identity in declared]}))
    with pytest.raises(AssertionError,match='pilot'):
        module.verify(tmp_path)
