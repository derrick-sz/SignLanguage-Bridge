"""Real SQLite/NPZ consumer tests: no full-library or rendering dependency."""
import hashlib
import importlib
import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from slbridge.api import create_app
from slbridge.animation.registry import MotionRegistry
from test_animation_registry import make_archive


PREFIX = '/v1/animation/corpus'
SAMPLE = 'S000001_P0000_T00'


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def update_row(root, identity=SAMPLE, **values):
    with sqlite3.connect(root/'corpus.sqlite3') as db:
        db.execute('UPDATE samples SET '+','.join(k+'=?' for k in values)+' WHERE id=?', [*values.values(), identity])


@pytest.fixture
def corpus_root(tmp_path):
    root = tmp_path/'corpus'; root.mkdir()
    with sqlite3.connect(root/'corpus.sqlite3') as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        db.execute('CREATE TABLE samples(id TEXT PRIMARY KEY,text TEXT,gloss TEXT,split TEXT,signer INTEGER,expected_frames INTEGER,actual_frames INTEGER,state TEXT,parameter_file TEXT,parameter_sha256 TEXT,source_parameters_sha256 TEXT,metrics_json TEXT,error TEXT)')
        build = dict(schema='soke-full-library-v1', source_archive_sha256='a'*64, template_sha256='b'*64,
                     index_sha256={'train':'c'*64}, processing_version='soke-full-raw-upper133-v1', refined=False)
        skeleton = dict(parents=[-1]+list(range(54)), restJoints=[[0, i*.01, 0] for i in range(55)],
                        representation='smplx-axis-angle', smplx_rest_model_sha256='b'*64)
        db.executemany('INSERT INTO metadata VALUES(?,?)', [('build',encode(build)), ('skeleton',encode(skeleton))])
        cases = [
            (SAMPLE, '今天开会', ['今天', '会议'], 'train', 'converted_candidate', 70),
            ('S000002_P0001_T00', '明天工作', ['明天', '劳动'], 'test', 'converted_candidate', 5),
            ('S000003_P0002_T00', '今天开会', ['会议'], 'val', 'pending', 0),
            ('S000004_P0003_T00', '没有动作', ['缺少'], 'train', 'incomplete_frames', 0),
            ('S000005_P0004_T00', '百分之100%', ['百分比'], 'train', 'converted_candidate', 0),
        ]
        for identity,text,gloss,split,state,step in cases:
            relative = 'sequences/'+identity[1:4]+'/'+identity+'.npz'
            path = root/relative; path.parent.mkdir(exist_ok=True,parents=True)
            poses = np.zeros((5,55,3),dtype=np.float32); poses[2,25,0] = .15
            np.savez_compressed(path,effective_poses=poses,frame_numbers=np.arange(5,dtype=np.int32))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (identity,text,encode(gloss),split,0,5,5,state,relative,digest,'d'*64,
                 encode({'mapped_max_step_degrees':step}),None))
    return root


def make_client(root, tmp_path):
    spec = importlib.util.find_spec('slbridge.animation.corpus_store')
    assert spec is not None, 'SQLite corpus reader is not implemented'
    store = importlib.import_module('slbridge.animation.corpus_store').CorpusStore(root)
    registry = MotionRegistry(make_archive(tmp_path))
    registry.corpus = store
    return TestClient(create_app({},animation_registry=registry)), store


def test_absent_corpus_is_optional_and_old_catalog_survives(tmp_path):
    with TestClient(create_app({},animation_registry=MotionRegistry(make_archive(tmp_path)))) as client:
        response = client.get(PREFIX)
        assert response.status_code == 200
        assert response.json()['enabled'] is False
        assert len(client.get('/v1/animation/assets').json()['assets']) == 3
        assert client.get(PREFIX+'/assets').status_code == 404


def test_paging_text_gloss_and_filters_use_only_metadata(corpus_root,tmp_path):
    client,store = make_client(corpus_root,tmp_path)
    for path in corpus_root.rglob('*.npz'): path.unlink()
    with client:
        first = client.get(PREFIX+'/assets',params={'limit':2}).json()
        assert [a['sample_id'] for a in first['assets']] == [SAMPLE,'S000002_P0001_T00']
        assert first['total'] == 3 and first['has_more']
        second = client.get(PREFIX+'/assets',params={'limit':2,'offset':2}).json()
        assert [a['sample_id'] for a in second['assets']] == ['S000005_P0004_T00']
        assert not second['has_more']
        assert client.get(PREFIX+'/assets',params={'q':'会议'}).json()['total'] == 1
        assert client.get(PREFIX+'/assets',params={'q':'工作','split':'test','quality':'quiet'}).json()['total'] == 1
        assert client.get(PREFIX+'/assets',params={'quality':'warning'}).json()['total'] == 1
        assert client.get(PREFIX+'/assets',params={'q':'%'}).json()['total'] == 1
        assert client.get(PREFIX+'/assets',params={'q':"' OR 1=1 --"}).json()['total'] == 0
        assert client.get(PREFIX+'/assets',params={'state':'all'}).json()['total'] == 5
        for params in ({'limit':101},{'offset':-1},{'split':'other'},{'quality':'approved'}):
            assert client.get(PREFIX+'/assets',params=params).status_code == 422
        status = client.get(PREFIX).json()
        assert status['counts']['converted_candidate'] == 3
        assert len(client.get('/v1/animation/assets').json()['assets']) == 3
        assert str(corpus_root) not in encode(first)


@pytest.mark.parametrize('offset,status,expected_ids',[
    (0,200,[SAMPLE,'S000002_P0001_T00']),
    (2,200,['S000005_P0004_T00']),
    (2**63-1,200,[]),
    (2**63,422,None),
    ('9'*1000,422,None),
    ('9'*10000,422,None),
],ids=['zero','normal-page','max-int64','int64-overflow','1000-digits','10000-digits'])
def test_offset_boundaries_return_pages_or_parameter_errors(corpus_root,tmp_path,offset,status,expected_ids):
    client,_=make_client(corpus_root,tmp_path)
    with client:
        response=client.get(PREFIX+'/assets',params={'limit':2,'offset':offset})
        assert response.status_code==status,response.text
        if status==200:
            page=response.json()
            assert [a['sample_id'] for a in page['assets']]==expected_ids
            assert page['offset']==offset and page['total']==3
            assert page['has_more']==(offset==0)


def test_direct_corpus_search_rejects_offset_outside_sqlite_integer_range(corpus_root,tmp_path):
    _,store=make_client(corpus_root,tmp_path)
    with pytest.raises(ValueError,match='search parameters'):
        store.search(offset=2**63)


def test_unconverted_detail_is_visible_but_never_playable(corpus_root,tmp_path):
    client,_ = make_client(corpus_root,tmp_path)
    with client:
        identity = 'S000003_P0002_T00'
        detail = client.get(PREFIX+'/assets/'+identity).json()
        assert detail['state'] == 'pending' and not detail['playable']
        assert client.get(PREFIX+'/motions/'+identity).status_code == 409
        assert client.get(PREFIX+'/assets/S999999_P0000_T00').status_code == 404
        assert client.get(PREFIX+'/motions/not-a-sample').status_code == 404


def test_plan_requires_exact_text_confirmation_and_research_preview(corpus_root,tmp_path):
    client,_ = make_client(corpus_root,tmp_path)
    with client:
        body = dict(sample_id=SAMPLE,text='今天开会')
        assert client.post(PREFIX+'/plans',json=body).json()['status'] == 'needs_confirmation'
        body['confirmed'] = True
        assert client.post(PREFIX+'/plans',json=body).json()['status'] == 'needs_language_review'
        body.update(research_preview=True,text='不要今天开会')
        assert client.post(PREFIX+'/plans',json=body).json()['can_play'] is False
        body['text'] = '今天，开会。'
        plan = client.post(PREFIX+'/plans',json=body).json()
        assert plan['can_play'] and plan['language_review'] == 'unverified'
        assert plan['source_archive_sha256'] == 'a'*64
        assert len(plan['items']) == 1 and plan['items'][0]['sample_id'] == SAMPLE
        assert client.post(PREFIX+'/plans',json=body|{'path':'/secret'}).status_code == 422


def test_lazy_motion_preserves_raw_binds_derived_content_and_rejects_stale_sha(corpus_root,tmp_path):
    client,_ = make_client(corpus_root,tmp_path)
    with client:
        plan = client.post(PREFIX+'/plans',json=dict(sample_id=SAMPLE,text='今天开会',confirmed=True,research_preview=True)).json()
        item = plan['items'][0]
        response = client.get(PREFIX+'/motions/'+SAMPLE,params={'expected_sha256':item['sha256']})
        assert response.status_code == 200, response.text
        motion = response.json(); clip = motion['clip']
        assert hashlib.sha256(encode(clip).encode()).hexdigest() == item['sha256'] == motion['sha256']
        assert clip['poses'][2][25][0] == pytest.approx(.15)
        assert clip['refined_poses'][2][25][0] < clip['poses'][2][25][0]
        assert clip['gesture_refinement']['profile'] == 'soke-gesture-refinement-v1'
        assert clip['human_approved'] is False and clip['language_review'] == clip['motion_review'] == 'unverified'
        assert clip['source_video_binding']['temporal_alignment'] == 'unverified'
        assert 'source_sha256' not in clip, 'no invented original-video digest'
        assert clip['id'] == item['asset_id'] and clip['source_id'] == SAMPLE
        assert client.get(PREFIX+'/motions/'+SAMPLE,params={'expected_sha256':'0'*64}).status_code == 409


@pytest.mark.parametrize('fault',['hash','path','frames','numeric','oversize','gap'])
def test_untrusted_parameter_files_fail_closed(corpus_root,tmp_path,fault):
    client,_ = make_client(corpus_root,tmp_path)
    path = corpus_root/'sequences/000'/f'{SAMPLE}.npz'
    if fault == 'hash': path.write_bytes(b'corrupt')
    elif fault == 'path': update_row(corpus_root,parameter_file='../outside.npz')
    elif fault == 'frames': update_row(corpus_root,actual_frames=3001,expected_frames=3001)
    elif fault == 'oversize':
        with path.open('wb') as stream: stream.truncate(32*1024*1024+1)
    else:
        poses = np.zeros((5,55,3)); numbers = np.arange(5)
        if fault == 'numeric': poses[0,0,0] = np.nan
        else: numbers[3] = 10
        np.savez_compressed(path,effective_poses=poses,frame_numbers=numbers)
        update_row(corpus_root,parameter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with client:
        response = client.get(PREFIX+'/motions/'+SAMPLE)
        assert response.status_code == 409, response.text
        assert str(corpus_root) not in response.text


def test_reader_sees_committed_writer_updates_and_ids_follow_content(corpus_root,tmp_path):
    client,_ = make_client(corpus_root,tmp_path)
    with client:
        before = client.get(PREFIX+'/assets/'+SAMPLE).json()['id']
        update_row(corpus_root,parameter_sha256='e'*64)
        assert client.get(PREFIX+'/assets/'+SAMPLE).json()['id'] != before
        update_row(corpus_root,identity='S000003_P0002_T00',state='converted_candidate')
        assert client.get(PREFIX+'/assets').json()['total'] == 4


def test_bad_source_parameter_identity_is_rejected(corpus_root,tmp_path):
    client,_ = make_client(corpus_root,tmp_path)
    update_row(corpus_root,source_parameters_sha256='not-a-digest')
    with client:
        assert client.get(PREFIX+'/motions/'+SAMPLE).status_code == 409


def test_npy_header_cannot_request_a_huge_allocation(corpus_root,tmp_path):
    from io import BytesIO
    import zipfile
    client,_ = make_client(corpus_root,tmp_path)
    data = BytesIO()
    np.lib.format.write_array_header_1_0(data, {'descr':'<f8','fortran_order':False,'shape':(10**10,55,3)})
    path = corpus_root/'sequences/000'/f'{SAMPLE}.npz'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('effective_poses.npy',data.getvalue())
    update_row(corpus_root,parameter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with client:
        assert client.get(PREFIX+'/motions/'+SAMPLE).status_code == 409


def test_cli_missing_corpus_exits_with_argument_error(tmp_path,capfd):
    from slbridge.animation.cli import main
    with pytest.raises(SystemExit) as failure:
        main(['--archive',str(make_archive(tmp_path)),'--corpus',str(tmp_path/'missing')])
    assert failure.value.code == 2
    assert 'error:' in capfd.readouterr().err


def test_cli_mounts_reader_without_replacing_original_registry(corpus_root,tmp_path,monkeypatch):
    from slbridge.animation.cli import main
    def inspect_app(app,**kwargs):
        with TestClient(app) as client:
            assert client.get(PREFIX).json()['enabled']
            assert client.get(PREFIX+'/assets').json()['total'] == 3
            assert len(client.get('/v1/animation/assets').json()['assets']) == 3
    monkeypatch.setattr('uvicorn.run',inspect_app)
    assert main(['--archive',str(make_archive(tmp_path)),'--corpus',str(corpus_root)]) == 0
