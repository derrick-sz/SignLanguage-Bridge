"""Small real LMDB/ISO fixtures for the native, source-preserving word library."""
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slbridge.animation.word_library_build import build_word_library
from test_animation_registry import make_archive

lmdb = pytest.importorskip('lmdb')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def streams(frames=12):
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    body = np.tile(identity, (frames, 21))
    left = np.tile(identity, (frames, 15))
    angles = np.linspace(.1, .7, frames)
    matrices = Rotation.from_rotvec(np.column_stack((np.zeros(frames), np.zeros(frames), angles))).as_matrix()
    left[:, :6] = matrices[:, :2, :].reshape(frames, 6)
    right = np.tile(identity, (frames, 15))
    face = np.zeros((frames, 56), dtype=np.float32)
    face[:, :6] = identity
    face[:, 6:] = np.arange(frames, dtype=np.float32)[:, None]
    return dict(body_features=body, left_features=left, right_features=right, face_features=face,
                translation=np.array([{'must_not_unpickle': True}], dtype=object))


def fixture(tmp_path, entries=None, values=None):
    database = tmp_path / 'train.lmdb'
    env = lmdb.open(str(database), map_size=16*1024*1024)
    data = values or {'S000000_P0000_T00': streams()}
    with env.begin(write=True) as txn:
        txn.put(b'__meta__', pickle.dumps({'clip_ids': list(data)}))
        for name, fields in data.items():
            blob = io.BytesIO()
            np.savez_compressed(blob, **fields)
            txn.put(name.encode(), blob.getvalue())
    env.close()
    entries = entries if entries is not None else [
        dict(video_file='S000000_P0000_T00', label='你好', start=2, end=8, seq_len=6)]
    iso = tmp_path / 'csl_iso.train'
    iso.write_bytes(pickle.dumps(entries, protocol=4))
    archive = make_archive(tmp_path)
    return database, iso, archive


def build(tmp_path, fixture_args=None, **kwargs):
    database, iso, archive = fixture_args or fixture(tmp_path)
    return build_word_library(database, iso, archive, tmp_path / 'words',
                              expected_lmdb_sha256=sha(database/'data.mdb'),
                              expected_annotations_sha256=sha(iso), **kwargs)


def test_standalone_export_keeps_whole_native_take_and_excludes_multiword_context(tmp_path):
    from slbridge.animation.signspark_codec import decode_signspark_streams
    entries=[
        dict(video_file='single',label='谢谢',start=2,end=6,seq_len=4),
        dict(video_file='context',label='谢谢',start=2,end=6,seq_len=4),
        dict(video_file='context',label='医生',start=7,end=11,seq_len=4),
    ]
    fields=streams()
    args=fixture(tmp_path,entries=entries,values={'single':fields,'context':streams()})
    manifest=build(tmp_path,args,standalone_only=True,labels=['谢谢'])
    unit,=manifest['units']
    assert manifest['standalone_expressions'] is True
    assert manifest['source']['selection_scope']=='whole-single-gloss-utterance-v1'
    assert unit['complete_expression'] is True
    assert unit['source']['video_file']=='single'
    assert unit['source']['start']==0 and unit['source']['end']==unit['frames']==12
    assert unit['source']['annotation_frame_range']==[2,6]
    assert unit['source']['source_label_count']==1
    assert len(unit['source']['native_blob_sha256'])==64
    expected=decode_signspark_streams(fields['body_features'][:,66:],fields['left_features'],
                                      fields['right_features'],fields['face_features'],left_hand_convention='smplx-left')
    with np.load(tmp_path/'words'/unit['path'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['poses'],expected.poses)
        np.testing.assert_array_equal(saved['expression'],expected.expression)


def test_label_filter_cannot_turn_a_multiword_source_into_standalone(tmp_path):
    entries=[dict(video_file='source',label=label,start=start,end=start+3,seq_len=3)
             for label,start in [('谢谢',1),('医生',7)]]
    args=fixture(tmp_path,entries=entries,values={'source':streams()})
    with pytest.raises(ValueError,match='No valid'):
        build(tmp_path,args,standalone_only=True,labels=['谢谢'])
    assert not (tmp_path/'words'/'manifest.json').exists()


def test_native_word_boundaries_left_convention_and_expression_are_preserved(tmp_path):
    from slbridge.animation.signspark_codec import decode_signspark_streams
    manifest = build(tmp_path)
    assert manifest['schema'] == 'signbridge-word-library-v1'
    assert manifest['preview_fps'] == 30
    assert manifest['fps_basis'] == 'explicit product preview clock; native LMDB timestamps not independently verified'
    assert len(manifest['skeleton']['parents']) == len(manifest['skeleton']['restJoints']) == 55
    unit, = manifest['units']
    assert unit['label'] == '你好' and unit['frames'] == 6 and unit['fps'] == 30
    assert unit['source'] == dict(video_file='S000000_P0000_T00', start=2, end=8, split='train')
    assert not Path(unit['path']).is_absolute() and '..' not in Path(unit['path']).parts
    path = tmp_path / 'words' / unit['path']
    assert sha(path) == unit['sha256']
    source = streams()
    expected = decode_signspark_streams(source['body_features'][2:8, 66:], source['left_features'][2:8],
                                       source['right_features'][2:8], source['face_features'][2:8],
                                       left_hand_convention='smplx-left')
    with np.load(path, allow_pickle=False) as clip:
        assert set(clip.files) == {'poses', 'expression'}
        assert clip['poses'].dtype == clip['expression'].dtype == np.float32
        np.testing.assert_array_equal(clip['poses'], expected.poses)
        np.testing.assert_array_equal(clip['expression'], expected.expression)
        assert np.all(clip['poses'][:, 25, 2] > 0)  # must not flip native left a second time
    assert json.loads((tmp_path/'words/manifest.json').read_text(encoding='utf-8')) == manifest
    assert manifest['language_review'] == 'NOT_RUN'


def test_median_length_selection_is_stable_and_maximum_candidates_is_bounded(tmp_path):
    entries = [dict(video_file='S000000_P0000_T00', label='我', start=0, end=end, seq_len=end)
               for end in [3, 5, 7, 9, 11]]
    manifest = build(tmp_path, fixture(tmp_path, entries))
    assert len(manifest['units']) == 3
    assert [unit['frames'] for unit in manifest['units']] == [7, 5, 9]


def test_eight_candidates_are_allowed_without_changing_default_first_three(tmp_path):
    entries = [dict(video_file='S000000_P0000_T00', label='我', start=0, end=end, seq_len=end)
               for end in range(3, 12)]
    data = fixture(tmp_path, entries)
    baseline = build(tmp_path, data)
    expanded = build_word_library(*data, tmp_path/'eight', expected_lmdb_sha256=sha(data[0]/'data.mdb'),
                                 expected_annotations_sha256=sha(data[1]), max_candidates=8)
    assert len(expanded['units']) == 8
    assert baseline['units'] == expanded['units'][:3]
    for unit in baseline['units']:
        assert (tmp_path/'words'/unit['path']).read_bytes() == (tmp_path/'eight'/unit['path']).read_bytes()


@pytest.mark.parametrize('split', ['dev', 'test'])
def test_nontrain_sources_retain_actual_split_and_product_only_role(tmp_path, split):
    manifest = build(tmp_path, split=split)
    assert manifest['source']['split'] == manifest['units'][0]['source']['split'] == split
    assert manifest['usage_role'] == 'product-demo' and manifest['evaluation_eligible'] is False
    assert manifest['source']['dataset_revision'] is None


@pytest.mark.parametrize('kwargs', [{'split':'validation'}, {'max_candidates':9}, {'max_candidates':True}])
def test_invalid_split_and_candidate_cap_fail_before_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build(tmp_path, **kwargs)
    assert not (tmp_path/'words').exists()


def test_known_dev_source_cannot_be_mislabeled_train(tmp_path, monkeypatch):
    import slbridge.animation.word_library_build as builder
    data = fixture(tmp_path)
    pair = (sha(data[0]/'data.mdb'), sha(data[1]))
    monkeypatch.setitem(builder.SOURCE_IDENTITIES, 'dev', pair)
    with pytest.raises(ValueError, match='split'):
        build(tmp_path, data)
    assert not (tmp_path/'words').exists()


@pytest.mark.parametrize('fault', ['range', 'length', 'object', 'nan', 'missing', 'degenerate', 'float32_overflow', 'extra_body_columns'])
def test_invalid_candidate_is_isolated_and_next_valid_word_is_retained(tmp_path, fault):
    bad = dict(video_file='S000001_P0000_T00', label='坏', start=1, end=5, seq_len=4)
    values = {'S000000_P0000_T00': streams(), 'S000001_P0000_T00': streams()}
    if fault == 'range': bad.update(start=10, end=14)
    elif fault == 'length': bad['seq_len'] = 99
    elif fault == 'object': values[bad['video_file']]['left_features'] = np.zeros((12,90), dtype=object)
    elif fault == 'nan': values[bad['video_file']]['face_features'][2,-1] = np.nan
    elif fault == 'missing': del values[bad['video_file']]
    elif fault == 'float32_overflow':
        values[bad['video_file']]['face_features'] = values[bad['video_file']]['face_features'].astype(np.float64)
        values[bad['video_file']]['face_features'][2, -1] = 1e300
    elif fault == 'extra_body_columns':
        values[bad['video_file']]['body_features'] = np.concatenate(
            (values[bad['video_file']]['body_features'], values[bad['video_file']]['body_features'][:, :6]), axis=1)
    else: values[bad['video_file']]['right_features'][:] = 0
    good = dict(video_file='S000000_P0000_T00', label='好', start=2, end=8, seq_len=6)
    manifest = build(tmp_path, fixture(tmp_path, [bad, good], values))
    assert [unit['label'] for unit in manifest['units']] == ['好']
    assert manifest['statistics']['failed_candidates'] + manifest['statistics']['excluded_annotations'] >= 1


def test_global_pickle_and_persistent_ids_are_rejected_before_output(tmp_path):
    database, iso, archive = fixture(tmp_path)
    for payload in (pickle.dumps(Path('unsafe')), b'Pforbidden\n.'):
        iso.write_bytes(payload)
        with pytest.raises(ValueError, match='annotation|pickle|class|persistent'):
            build_word_library(database, iso, archive, tmp_path/'words', expected_lmdb_sha256=sha(database/'data.mdb'),
                               expected_annotations_sha256=sha(iso))
        assert not (tmp_path/'words').exists()


@pytest.mark.parametrize('which', ['lmdb', 'annotations'])
def test_full_identity_mismatch_fails_before_any_output(tmp_path, which):
    database, iso, archive = fixture(tmp_path)
    with pytest.raises(ValueError, match='SHA|identity|checksum'):
        build_word_library(database, iso, archive, tmp_path/'words',
                           expected_lmdb_sha256='0'*64 if which == 'lmdb' else sha(database/'data.mdb'),
                           expected_annotations_sha256='0'*64 if which == 'annotations' else sha(iso))
    assert not (tmp_path/'words').exists()


def test_existing_output_is_never_reused_or_overwritten(tmp_path):
    build(tmp_path)
    original = sha(tmp_path/'words/manifest.json')
    with pytest.raises(ValueError, match='new|exist'):
        database, iso, archive = tmp_path/'train.lmdb', tmp_path/'csl_iso.train', tmp_path/'motion.zip'
        build_word_library(database, iso, archive, tmp_path/'words', expected_lmdb_sha256=sha(database/'data.mdb'),
                           expected_annotations_sha256=sha(iso))
    assert sha(tmp_path/'words/manifest.json') == original


def test_only_original_lmdb_timebase_is_sliced_and_sources_are_unchanged(tmp_path):
    database, iso, archive = fixture(tmp_path)
    before = [sha(path) for path in (database/'data.mdb', iso, archive)]
    first = build(tmp_path, (database, iso, archive))
    second = build_word_library(database, iso, archive, tmp_path/'second',
                                expected_lmdb_sha256=before[0], expected_annotations_sha256=before[1])
    assert first['bundle_id'] == second['bundle_id']
    assert first['units'] == second['units']
    assert before == [sha(path) for path in (database/'data.mdb', iso, archive)]
    assert first['source']['left_hand_convention'] == 'smplx-left'
    assert first['source']['dataset_revision'] is None  # do not mislabel a custom fixture as official


def test_npz_expansion_limit_prevents_loading_large_arrays(tmp_path, monkeypatch):
    import slbridge.animation.word_library_build as builder
    database, iso, archive = fixture(tmp_path)
    monkeypatch.setattr(builder, '_MAX_EXPANDED', 4)
    with pytest.raises(ValueError, match='No valid native word'):
        build(tmp_path, (database, iso, archive))
    assert not (tmp_path/'words/manifest.json').exists()
