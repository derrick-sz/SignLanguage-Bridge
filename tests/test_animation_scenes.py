"""Scene matching uses real read-only SQLite metadata and verified source NPZs."""
from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from slbridge.animation.corpus_store import CorpusStore, CorpusIntegrityError
from slbridge.animation.scenes import SceneLibrary
from test_animation_corpus import corpus_root, update_row, SAMPLE


@pytest.fixture
def library(corpus_root, tmp_path):
    definitions = tmp_path/'scenes.json'
    definitions.write_text(json.dumps({'schema_version': 1, 'scenes': [
        {'id': 'work', 'title': '工作', 'keywords': ['今天', '工作'], 'targets': [
            {'id': 'a', 'text': '今天开会。'}, {'id': 'b', 'text': '今天工作'},
            {'id': 'c', 'text': '今天开会。百分之100%'}]},
        {'id': 'number', 'title': '数字', 'keywords': ['100'], 'targets': []},
    ]}, ensure_ascii=False), encoding='utf-8')
    return SceneLibrary(CorpusStore(corpus_root), definitions)


def add(corpus_root, number, text, frames=5):
    identity = f'S{number:06d}_P0000_T00'
    relative = f'sequences/{identity}.npz'
    poses = np.full((frames, 55, 3), number/100, dtype=np.float32)
    np.savez_compressed(corpus_root/relative, effective_poses=poses, frame_numbers=np.arange(frames, dtype=np.int32))
    with sqlite3.connect(corpus_root/'corpus.sqlite3') as db:
        db.execute('INSERT INTO samples SELECT ?,?,gloss,split,signer,?,?,state,?,?,source_parameters_sha256,metrics_json,error FROM samples WHERE id=?',
                   (identity, text, frames, frames, relative, hashlib.sha256((corpus_root/relative).read_bytes()).hexdigest(), SAMPLE))
    return identity, poses


def ready(library, text):
    return library.plan(text, confirmed=True, research_preview=True)


def test_plan_input_budget_matches_http_500(library):
    with pytest.raises(ValueError):
        ready(library, '字' * 501)


def test_released_json_string_gloss_tokens_are_counted_and_searchable(corpus_root, library):
    update_row(corpus_root, SAMPLE, gloss=json.dumps('苹果 医院', ensure_ascii=False))
    library = SceneLibrary(CorpusStore(corpus_root), library.definitions_path)
    assert library.coverage()['gloss_entries'] >= 2
    page = library.catalog(q='苹果')
    assert len(page['assets']) == 1
    assert page['assets'][0]['gloss'] == ['苹果', '医院']


def test_catalog_train_complete_dedup_paging_and_coverage(corpus_root, library):
    add(corpus_root, 10, json.dumps('今天开会', ensure_ascii=False))
    add(corpus_root, 11, '坏帧')
    update_row(corpus_root, 'S000011_P0000_T00', actual_frames=4)
    library = SceneLibrary(library.store, library.definitions_path)
    page = library.catalog(limit=1)
    assert page['total'] == 2 and page['has_more']
    assert page['assets'][0]['text'] == '今天开会'
    assert library.catalog(offset=1)['assets'][0]['text'] == '百分之100%'
    assert library.catalog(scene='work')['total'] == 1
    assert library.catalog(q='100')['total'] == 1
    assert library.catalog(q='明天工作')['total'] == 0
    scene = library.catalog()['scenes'][0]
    assert scene['candidate_texts'] == 1
    assert scene['target_exact'] == 1 and scene['target_total'] == 3
    report = library.coverage()
    assert report['target_exact'] == 1 and report['target_composed'] == 1 and report['target_blocked'] == 1
    assert report['language_review'] == 'NOT_RUN'
    assert report['instance_count'] == 3 and report['distinct_texts'] == 2


@pytest.mark.parametrize('text', ['今天开会？', '今天开会!', '今天不开会', '今天开会小王', '百分之10%', '今天开会。不存在'])
def test_no_semantic_deletion_or_partial_playback(library, text):
    plan = ready(library, text)
    assert not plan['can_play'] and plan['status'] == 'blocked' and plan['unmatched']
    with pytest.raises((ValueError, CorpusIntegrityError)):
        library.motion(plan)


def test_confirmation_and_original_caption(library):
    assert not library.plan('今天开会。')['can_play']
    assert not library.plan('今天开会。', confirmed=True)['can_play']
    plan = ready(library, '今天开会。')
    assert plan['can_play'] and plan['mode'] == 'existing_expression'
    assert plan['items'][0]['text'] == '今天开会'
    motion = library.motion(plan)
    assert motion['clip']['text'] == '今天开会'
    assert np.array_equal(motion['clip']['poses'], library.store._parameters(library.store._row(SAMPLE)))


def test_decimal_and_question_punctuation(corpus_root, library):
    add(corpus_root, 10, '价格3.5元？')
    library = SceneLibrary(library.store, library.definitions_path)
    assert ready(library, '价格3.5元?')['can_play']
    assert ready(library, '价格3.5元？今天开会。')['can_play']
    for text in ['价格35元？', '价格3.5元', '价格3.5元！']:
        assert not ready(library, text)['can_play']


@pytest.mark.parametrize('separator', [';', '；', '\n', '。', '.'])
def test_explicit_sentence_separators(library, separator):
    assert ready(library, f'今天开会{separator}百分之100%')['can_play']


def test_segments_budget(library, corpus_root):
    assert ready(library, '。'.join(['今天开会']*6))['can_play']
    assert not ready(library, '。'.join(['今天开会']*7))['can_play']
    add(corpus_root, 10, '长动作', frames=900)
    library = SceneLibrary(library.store, library.definitions_path)
    assert not ready(library, '长动作。长动作')['can_play']


def test_featured_source_order_and_coverage_independence(library, corpus_root):
    identity, _ = add(corpus_root, 10, '新的真实原句')
    definitions = json.loads(library.definitions_path.read_text(encoding='utf-8'))
    definitions['scenes'][0]['featured_sample_ids'] = [identity]
    library.definitions_path.write_text(json.dumps(definitions, ensure_ascii=False), encoding='utf-8')
    library = SceneLibrary(library.store, library.definitions_path)
    page = library.catalog(scene='work')
    assert page['assets'][0]['sample_id'] == identity
    assert page['scenes'][0]['target_exact'] == 1


def test_newline_is_a_separator_not_a_deleted_character(corpus_root, library):
    add(corpus_root, 10, '今天开会百分之100%')
    library = SceneLibrary(library.store, library.definitions_path)
    plan = ready(library, '今天开会\n百分之100%')
    assert plan['mode'] == 'expression_sequence' and len(plan['items']) == 2


def test_slerp_short_arc_source_frames_and_provenance(library, corpus_root):
    a = library.store._parameters(library.store._row(SAMPLE))
    a[-1, 25, 0] = np.deg2rad(179)
    path = corpus_root/library.store._row(SAMPLE)['parameter_file']
    np.savez_compressed(path, effective_poses=a, frame_numbers=np.arange(5, dtype=np.int32))
    update_row(corpus_root, parameter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    identity, b = add(corpus_root, 10, '再见')
    b[0, 25, 0] = np.deg2rad(-179)
    path_b = corpus_root/library.store._row(identity)['parameter_file']
    np.savez_compressed(path_b, effective_poses=b, frame_numbers=np.arange(5, dtype=np.int32))
    update_row(corpus_root, identity, parameter_sha256=hashlib.sha256(path_b.read_bytes()).hexdigest())
    library = SceneLibrary(library.store, library.definitions_path)
    plan = ready(library, '今天开会。再见')
    motion = library.motion(plan)
    poses = np.asarray(motion['clip']['poses'])
    assert len(poses) == 19 and np.array_equal(poses[:5], a) and np.array_equal(poses[14:], b)
    assert np.all(np.abs(poses[5:14,25,0]) > 3)
    assert plan['transitions'][0]['output_frame_range'] == [5,14]
    assert plan['items'][1]['output_frame_range'] == [14,19]
    assert motion['clip']['composition']['source_frames_preserved']
    assert motion['clip']['text'] == '今天开会。再见'


@pytest.mark.parametrize('kind', ['npz', 'row', 'template', 'plan'])
def test_identity_drift_and_tampering_refused(library, corpus_root, kind):
    plan = ready(library, '今天开会')
    if kind == 'npz':
        (corpus_root/library.store._row(SAMPLE)['parameter_file']).write_bytes(b'changed')
    elif kind == 'row':
        update_row(corpus_root, text='字幕改变')
    elif kind == 'template':
        with sqlite3.connect(corpus_root/'corpus.sqlite3') as db:
            build = deepcopy(library.store.build); build['template_sha256'] = 'c'*64
            db.execute("UPDATE metadata SET value=? WHERE key='build'", (json.dumps(build),))
    else:
        plan['items'][0]['text'] = '伪造'
    with pytest.raises(CorpusIntegrityError):
        library.motion(plan)


@pytest.mark.parametrize('params', [{'limit':0}, {'offset':-1}, {'scene':'unknown'}, {'limit':True}, {'q':None}])
def test_catalog_parameter_validation(library, params):
    with pytest.raises(ValueError):
        library.catalog(**params)
