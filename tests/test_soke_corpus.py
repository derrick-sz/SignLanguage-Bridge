import io
import pickle
from pathlib import Path
import zipfile

import numpy as np
import pytest

from slbridge.animation.soke_pickle import parse_pickle, Opaque
from slbridge.animation.soke import decode_frame, effective_poses, scan_archive, read_member, load_sequence, write_candidate_archive, video_binding


def frame():
    return {k:np.zeros(n,dtype=np.float32) for k,n in {
        'smplx_root_pose':3,'smplx_body_pose':63,'smplx_lhand_pose':45,
        'smplx_rhand_pose':45,'smplx_jaw_pose':3,'smplx_shape':10,
        'smplx_expr':10,'cam_trans':3}.items()}


def test_numeric_pickle_without_invoking_numpy_constructors():
    source=frame();source['smplx_body_pose']=np.arange(63,dtype=np.float32)/100
    decoded=decode_frame(pickle.dumps(source,protocol=4))
    for key in source:np.testing.assert_array_equal(decoded[key],source[key])


def test_annotation_tensor_reduce_is_inert():
    raw=b'cbuiltins\neval\n(V__import__("os").getcwd()\ntR.'
    assert isinstance(parse_pickle(raw),Opaque)


def test_object_arrays_are_rejected():
    with pytest.raises(ValueError,match='dtype'):
        parse_pickle(pickle.dumps(np.array([None],dtype=object),protocol=4))


@pytest.mark.parametrize('args', [b')', b']', b'N', b'K\x01\x85'])
def test_invalid_numpy_dtype_arguments_raise_value_error(args):
    raw = (b'cnumpy.core.multiarray\n_reconstruct\n)R(K\x01K\x01\x85'
           b'cnumpy\ndtype\n' + args + b'R\x89C\x04\x00\x00\x00\x00tb.')
    with pytest.raises(ValueError, match='dtype'):
        parse_pickle(raw)


@pytest.mark.parametrize('change',[lambda x:x.pop('cam_trans'),lambda x:x.update(smplx_lhand_pose=np.zeros(12,dtype=np.float32)),lambda x:x['smplx_body_pose'].__setitem__(0,np.nan)])
def test_invalid_frame_rejected(change):
    source=frame();change(source)
    with pytest.raises(ValueError):decode_frame(pickle.dumps(source,protocol=4))


def test_hand_mean_added_once_and_root_lower_body_canonicalized():
    source={k:np.stack([v,v+0.1]) for k,v in frame().items()}
    means=np.arange(90,dtype=np.float32).reshape(2,45)/100
    before={k:v.copy() for k,v in source.items()}
    result=effective_poses(source,means)
    assert result.shape==(2,55,3)
    np.testing.assert_array_equal(result[:,:12],0)
    np.testing.assert_allclose(result[:,25:40].reshape(2,45),source['smplx_lhand_pose']+means[0])
    np.testing.assert_allclose(result[:,40:55].reshape(2,45),source['smplx_rhand_pose']+means[1])
    np.testing.assert_array_equal(result[:,12:22].reshape(2,30),source['smplx_body_pose'][:,33:])
    for key in source:np.testing.assert_array_equal(source[key],before[key])


def make_archive(path,numbers):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for number in numbers:
            archive.writestr(f'csl-daily_pose/S000000_P0000_T00/{number:06}.pkl',pickle.dumps(frame(),protocol=4))


def test_archive_inventory_sorts_frame_numbers_and_detects_gaps(tmp_path):
    path=tmp_path/'poses.zip';make_archive(path,[3,0,1])
    inventory,selected=scan_archive(path,{'S000000_P0000_T00'})
    row=inventory['sequences']['S000000_P0000_T00']
    assert row['frames']==3 and row['missing_frame_numbers']==[2]
    assert [m.frame_number for m in selected['S000000_P0000_T00']]==[0,1,3]
    assert decode_frame(read_member(path,selected['S000000_P0000_T00'][0]))['smplx_body_pose'].shape==(63,)


def test_duplicate_frame_members_rejected(tmp_path):
    path=tmp_path/'poses.zip'
    with pytest.warns(UserWarning):make_archive(path,[0,0])
    with pytest.raises(ValueError,match='Duplicate'):scan_archive(path,set())


def test_member_crc_mismatch_rejected(tmp_path):
    path=tmp_path/'poses.zip';make_archive(path,[0])
    _,selected=scan_archive(path,{'S000000_P0000_T00'});member=selected['S000000_P0000_T00'][0]
    member.crc^=1
    with pytest.raises(ValueError,match='CRC'):read_member(path,member)


def test_pickle_trailing_payload_rejected():
    with pytest.raises(ValueError):parse_pickle(pickle.dumps({'x':1},protocol=4)+b'ignored')


def test_incomplete_sequence_cannot_be_silently_filled(tmp_path):
    path=tmp_path/'poses.zip';make_archive(path,[0,2])
    _,selected=scan_archive(path,{'S000000_P0000_T00'})
    with pytest.raises(ValueError,match='Incomplete'):load_sequence(path,selected['S000000_P0000_T00'],3)


def test_complete_sequence_keeps_raw_fields_and_identity(tmp_path):
    path=tmp_path/'poses.zip';make_archive(path,[1,0])
    _,selected=scan_archive(path,{'S000000_P0000_T00'})
    parameters,identity=load_sequence(path,selected['S000000_P0000_T00'],2)
    assert parameters['smplx_lhand_pose'].shape==(2,45)
    assert len(identity)==64


def test_candidate_bundle_has_no_inherited_approvals(tmp_path):
    import json,hashlib
    from slbridge.animation.registry import MotionRegistry
    source=tmp_path/'base.zip';skeleton={'parents':[-1]+list(range(54)),'restJoints':[[0,0,0]]*55}
    with zipfile.ZipFile(source,'w') as archive:
        archive.writestr('demo/experimental-index.json',json.dumps(skeleton))
        for name in ('avatar/model.glb','avatar/three.module.js','avatar/jsm/loaders/GLTFLoader.js','avatar/jsm/utils/BufferGeometryUtils.js'):archive.writestr(name,b'fixture')
    clip={'id':'soke-S000000_P0000_T00','text':'你好','frames':2,'fps':25,'poses':np.zeros((2,55,3)).tolist(),'human_approved':True,'motion_review_status':'approved'}
    result=tmp_path/'candidate.zip'
    write_candidate_archive(result,source,[clip],'SOKE CSL-Daily research candidates')
    registry=MotionRegistry(result)
    asset=registry.catalog()['assets'][0]
    assert asset['language_review']==asset['motion_review']=='unverified'
    assert registry.motion(clip['id'])['clip']['human_approved'] is False
    assert registry.catalog()['attribution']['motion']=='SOKE CSL-Daily research candidates'


def test_uncropped_original_never_claims_temporal_alignment():
    binding=video_binding({'fps':30.0,'frames':168},52,30.0)
    assert binding['frame_count_match'] is False
    assert binding['temporal_alignment']=='unverified'
    assert binding['pose_frames']==52 and binding['original_video_frames']==168
    assert video_binding({'fps':30.0,'frames':52},52,30.0)['temporal_alignment']=='unverified'


def test_bake_limit_rejected_before_reading_assets(tmp_path,capsys):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/animation/build_soke_corpus.py'
    spec=importlib.util.spec_from_file_location('soke_build_cli_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=[]
    for name in ('source-archive','indices','videos','base-archive','smplx-model','output','ffprobe','blender','avatar'):
        args.extend(['--'+name,str(tmp_path/name)])
    with pytest.raises(SystemExit) as error:module.main(args+['--count','101'])
    assert error.value.code==2
    assert '100' in capsys.readouterr().err


def test_expansion_preserves_seed_identity_and_rejects_stale_text():
    from slbridge.animation.soke import select_seed_rows
    rows=[{'name':'a','text':'你好','num_frames':10},{'name':'b','text':'开会','num_frames':20}]
    seed=[dict(rows[1])]
    assert select_seed_rows(rows,seed,100)==[rows[1]]
    seed[0]['text']='另一个意思'
    with pytest.raises(ValueError):select_seed_rows(rows,seed,100)
    with pytest.raises(ValueError):select_seed_rows(rows,rows,1)
