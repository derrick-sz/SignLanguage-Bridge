import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def library(tmp_path, name, entries, split='train'):
    root = tmp_path/name
    root.mkdir()
    (root/'units').mkdir()
    units=[]
    for identity,label,value in entries:
        path=f'units/{identity}.npz'
        poses=np.zeros((3,55,3),dtype=np.float32)
        poses[:,25,2]=value
        np.savez_compressed(root/path,poses=poses,expression=np.zeros((3,50),dtype=np.float32))
        units.append({'id':identity,'label':label,'path':path,'sha256':sha((root/path).read_bytes()),
                      'frames':3,'fps':30,'source':{'video_file':identity,'start':4,'end':7,'split':split}})
    manifest={'schema':'signbridge-word-library-v1','bundle_id':name,'base_archive_sha256':'c'*64,
              'source':{'split':split,'lmdb_sha256':('a' if split=='train' else 'd')*64,
                        'annotations_sha256':('b' if split=='train' else 'e')*64,
                        'codec':'signspark-row6d-flat-hand-v1','left_hand_convention':'smplx-left'},
              'skeleton':{'parents':[-1]+list(range(54)),'restJoints':[[0,0,0]]*55},
              'preview_fps':30,'units':units,'usage_role':'product-demo','evaluation_eligible':False}
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return root


def write_manifest(path, mutate):
    value=json.loads((path/'manifest.json').read_text())
    mutate(value)
    (path/'manifest.json').write_text(json.dumps(value),encoding='utf-8')


def test_old_ids_metadata_and_bytes_survive_while_sources_round_robin(tmp_path):
    from slbridge.animation.word_library_merge import merge_word_libraries
    base=library(tmp_path,'base',[(f'b{i}','我',i*.1) for i in range(3)])
    train=library(tmp_path,'train8',[(f't{i}','我',.5+i*.1) for i in range(8)])
    dev=library(tmp_path,'dev8',[('d1','我',1.4),('new','成都',1.5)],'dev')
    test=library(tmp_path,'test8',[('e1','我',1.6)],'test')
    before=(base/'manifest.json').read_bytes()
    output=tmp_path/'merged'
    merged=merge_word_libraries(base,[train,dev,test],output)
    assert merged['usage_role']=='product-demo' and merged['evaluation_eligible'] is False
    assert merged['source']['split']=='mixed'
    assert merged['source']['evaluation_eligible'] is False
    chosen=[u for u in merged['units'] if u['label']=='我']
    assert len(chosen)==8 and [u['id'] for u in chosen[:3]]==['b0','b1','b2']
    assert [u['id'] for u in chosen[3:6]]==['t0','d1','e1']
    assert merged['statistics']['labels_added']==1
    assert merged['statistics']['base_units_preserved']==3
    sources={s['library_bundle_id']:s for s in merged['source']['sources']}
    for unit in merged['units']:
        origin=sources[unit['source']['library_bundle_id']]
        assert unit['source']['library_manifest_sha256']==origin['manifest_sha256']
        assert unit['source']['split']==origin['split']
        assert unit['source']['lmdb_sha256']==origin['lmdb_sha256']
    for unit in chosen[:3]:
        assert (output/unit['path']).read_bytes()==(base/unit['path']).read_bytes()
    assert (base/'manifest.json').read_bytes()==before


def test_duplicate_identical_unit_is_deduplicated_and_old_origin_retained(tmp_path):
    from slbridge.animation.word_library_merge import merge_word_libraries
    base=library(tmp_path,'base',[('same','我',.1)])
    addition=library(tmp_path,'addition',[('same','我',.1)])
    result=merge_word_libraries(base,[addition],tmp_path/'out')
    assert len(result['units'])==1
    assert result['units'][0]['source']['library_bundle_id']=='base'
    assert result['statistics']['duplicate_units']==1


@pytest.mark.parametrize('fault',['id_conflict','skeleton','fps','sha','path','source_split','product_flag','shape'])
def test_conflicting_or_unsafe_sources_rejected_before_output(tmp_path,fault):
    from slbridge.animation.word_library_merge import merge_word_libraries
    base=library(tmp_path,'base',[('same','我',.1)])
    addition=library(tmp_path,'addition',[('other','你',.2)],'dev')
    if fault=='id_conflict':
        write_manifest(addition,lambda m:m['units'][0].update(id='same'))
    elif fault=='skeleton':
        write_manifest(addition,lambda m:m['skeleton']['restJoints'][1].__setitem__(0,3))
    elif fault=='fps':
        write_manifest(addition,lambda m:m['units'][0].update(fps=20))
    elif fault=='sha':
        (addition/'units/other.npz').write_bytes(b'tampered')
    elif fault=='path':
        write_manifest(addition,lambda m:m['units'][0].update(path='../outside.npz'))
    elif fault=='source_split':
        write_manifest(addition,lambda m:m['units'][0]['source'].update(split='train'))
    elif fault=='product_flag':
        write_manifest(addition,lambda m:m.update(evaluation_eligible=True))
    elif fault=='shape':
        path=addition/'units/other.npz'
        np.savez_compressed(path,poses=np.zeros((3,54,3),dtype=np.float32),expression=np.zeros((3,50),dtype=np.float32))
        write_manifest(addition,lambda m:m['units'][0].update(sha256=sha(path.read_bytes())))
    output=tmp_path/'out'
    with pytest.raises(ValueError):
        merge_word_libraries(base,[addition],output)
    assert not output.exists()


def test_merged_product_library_can_be_extended_again(tmp_path):
    from slbridge.animation.word_library_merge import merge_word_libraries
    base=library(tmp_path,'base',[('a','我',.1)])
    dev=library(tmp_path,'dev',[('b','你',.2)],'dev')
    first=merge_word_libraries(base,[dev],tmp_path/'first')
    train=library(tmp_path,'train',[('c','医院',.3)])
    second=merge_word_libraries(tmp_path/'first',[train],tmp_path/'second')
    assert [u['id'] for u in second['units'][:len(first['units'])]]==[u['id'] for u in first['units']]
    for unit in first['units']:
        assert (tmp_path/'first'/unit['path']).read_bytes()==(tmp_path/'second'/unit['path']).read_bytes()


def test_existing_output_and_a_cap_below_base_count_never_overwrite(tmp_path):
    from slbridge.animation.word_library_merge import merge_word_libraries
    base=library(tmp_path,'base',[('a','我',.1),('b','我',.2)])
    output=tmp_path/'out'
    with pytest.raises(ValueError):
        merge_word_libraries(base,[],output,max_candidates=1)
    assert not output.exists()
    output.mkdir()
    sentinel=output/'do-not-change'
    sentinel.write_text('keep')
    with pytest.raises(ValueError):
        merge_word_libraries(base,[],output)
    assert sentinel.read_text()=='keep'


def test_more_than_sixteen_source_origins_fail_before_output(tmp_path):
    from slbridge.animation.word_library_merge import merge_word_libraries
    base=library(tmp_path,'base',[('base-unit','我',.1)])
    additions=[library(tmp_path,f'lib-{i}',[(f'unit-{i}',f'词{i}',.2)]) for i in range(16)]
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='source|origin|16'):
        merge_word_libraries(base,additions,output)
    assert not output.exists()
