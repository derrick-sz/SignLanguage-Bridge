import hashlib
import json

import numpy as np
import pytest


@pytest.fixture
def word_bundle(tmp_path):
    root = tmp_path / 'library'
    root.mkdir()
    units = []
    for identity, label, angles in [('a', '我', [.1, .2]), ('b', '去', [2., 2.1]),
                                    ('c', '去', [.3, .4]), ('d', '医院', [.5, .6])]:
        pose = np.zeros((2, 55, 3), dtype=np.float32)
        pose[:, 25, 2] = angles
        path = root / f'{identity}.npz'
        np.savez_compressed(path, poses=pose, expression=np.zeros((2, 50), dtype=np.float32))
        units.append({'id': identity, 'label': label, 'path': path.name,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'frames': 2, 'fps': 30,
                      'source': {'video_file': identity, 'start': 10, 'end': 12, 'split': 'train'}})
    manifest = {'schema': 'signbridge-word-library-v1', 'bundle_id': 'f' * 64,
                'source': {'split': 'train', 'lmdb_sha256': 'a' * 64, 'annotations_sha256': 'b' * 64},
                'skeleton': {'parents': [-1] + list(range(54)), 'restJoints': [[0, 0, 0]] * 55},
                'units': units}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root, manifest


def test_real_word_sources_preserved_and_nearer_context_candidate_selected(word_bundle):
    from slbridge.animation.word_composition import WordLibrary
    root, _ = word_bundle
    result = WordLibrary(root).compose('我去医院。', ['我', '去', '医院'])
    assert result['status'] == 'ready'
    motion = result['motion']
    clip = motion['clip']
    assert clip['text'] == '我去医院。' and clip['source_kind'] == 'retrieved_word_composition'
    assert clip['frames'] == 18 and clip['fps'] == 30
    assert clip['composition']['source_frames_preserved'] is True
    items = clip['composition']['items']
    assert [item['unit_id'] for item in items] == ['a', 'c', 'd']
    assert [item['source_frame_range'] for item in items] == [[10, 12]] * 3
    poses = np.array(clip['poses'])
    for item in items:
        with np.load(root / (item['unit_id'] + '.npz'), allow_pickle=False) as saved:
            start, end = item['output_frame_range']
            np.testing.assert_array_equal(poses[start:end], saved['poses'])
    # A known same-axis rotation must increase through the six bridge frames.
    assert np.all(np.diff(poses[1:9, 25, 2]) > 0)
    assert clip['language_review'] == 'NOT_RUN'
    digest = hashlib.sha256(json.dumps(clip, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert motion['sha256'] == digest


def test_missing_word_blocks_entire_sentence_and_never_changes_gloss(word_bundle):
    from slbridge.animation.word_composition import WordLibrary
    result = WordLibrary(word_bundle[0]).compose('我不去医院。', ['我', '不', '去', '医院'])
    assert result['status'] == 'blocked' and result['missing_glosses'] == ['不']
    assert result['glosses'] == ['我', '不', '去', '医院'] and 'motion' not in result


def test_file_change_is_rejected_even_after_cache_warm(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, WordLibraryError
    root, _ = word_bundle
    library = WordLibrary(root)
    library.compose('我', ['我'])
    (root / 'a.npz').write_bytes(b'tampered')
    with pytest.raises(WordLibraryError):
        library.compose('我', ['我'])


def test_manifest_change_cannot_publish_old_index(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, WordLibraryError
    root, manifest = word_bundle
    library = WordLibrary(root)
    manifest['bundle_id'] = 'e' * 64
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(WordLibraryError):
        library.compose('我', ['我'])


@pytest.mark.parametrize('glosses', [[], '我', [''], [None], ['我'] * 33])
def test_invalid_or_over_budget_plans_rejected(word_bundle, glosses):
    from slbridge.animation.word_composition import WordLibrary
    with pytest.raises(ValueError):
        WordLibrary(word_bundle[0]).compose('我', glosses)


def test_unsafe_manifest_path_is_rejected(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, WordLibraryError
    root, manifest = word_bundle
    manifest['units'][0]['path'] = '../outside.npz'
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(WordLibraryError):
        WordLibrary(root)


def test_invalid_pose_and_frame_budget_fail_closed(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, WordLibraryError
    root, manifest = word_bundle
    np.savez_compressed(root / 'a.npz', poses=np.full((2, 55, 3), np.nan), expression=np.zeros((2, 50)))
    manifest['units'][0]['sha256'] = hashlib.sha256((root / 'a.npz').read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(WordLibraryError):
        WordLibrary(root).compose('我', ['我'])


def test_service_retains_input_and_normalizes_newline_without_modifying_negation(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, CompositionService
    class Planner:
        def plan(self, text):
            assert text == '我 去医院。'
            return {'glosses': ['我', '去', '医院'], 'model_identity': {'revision': 'fixture'}}
        def status(self):
            return {'state': 'ready'}
        def close(self):
            pass
    result = CompositionService(WordLibrary(word_bundle[0]), Planner()).compose('我\n去医院。')
    assert result['input_text'] == '我\n去医院。' and result['generation_text'] == '我 去医院。'
    assert result['motion']['clip']['text'] == result['generation_text']
    assert result['timings_ms']['total_ms'] >= 0


def test_total_frame_budget_rejects_long_repeated_word_before_publication(word_bundle):
    from slbridge.animation.word_composition import WordLibrary
    root, manifest = word_bundle
    path = root / 'a.npz'
    np.savez_compressed(path, poses=np.zeros((180, 55, 3), dtype=np.float32),
                        expression=np.zeros((180, 50), dtype=np.float32))
    manifest['units'][0].update(frames=180, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest['units'][0]['source']['end'] = 190
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='1800'):
        WordLibrary(root).compose('我' * 11, ['我'] * 11)


def test_service_does_not_queue_or_overlap_planner_requests(word_bundle):
    import threading
    from slbridge.animation.word_composition import WordLibrary, CompositionService, CompositionBusyError
    entered, release = threading.Event(), threading.Event()
    class Planner:
        def plan(self, text):
            entered.set()
            assert release.wait(5)
            return {'glosses': ['我']}
    service = CompositionService(WordLibrary(word_bundle[0]), Planner())
    errors = []
    def first():
        try:
            service.compose('我')
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(CompositionBusyError):
            service.compose('去')
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors


def test_known_destination_omission_blocks_before_any_motion_is_loaded(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, CompositionService
    class Planner:
        def plan(self, text):
            return {'glosses': ['我', '去']}
    result = CompositionService(WordLibrary(word_bundle[0]), Planner()).compose('我去医院。')
    assert result['status'] == 'needs_review' and result['omitted_terms'] == ['医院']
    assert result['glosses'] == ['我', '去'] and 'motion' not in result


def test_anchor_omission_check_is_conservative_and_not_a_translation_claim(word_bundle):
    from slbridge.animation.word_composition import WordLibrary, CompositionService
    class Planner:
        def plan(self, text):
            return {'glosses': ['我', '去', '医院']}
    result = CompositionService(WordLibrary(word_bundle[0]), Planner()).compose('我去医院。')
    assert result['status'] == 'ready' and result['language_review'] == 'NOT_RUN'


def product_binding(manifest):
    from copy import deepcopy
    source = deepcopy(manifest['source'])
    binding = {'library_bundle_id': manifest['bundle_id'], 'manifest_sha256': 'c'*64, **source}
    manifest.update(usage_role='product-demo',evaluation_eligible=False)
    manifest['source'] = {'split':'mixed','usage_role':'product-demo','evaluation_eligible':False,'sources':[binding]}
    for unit in manifest['units']:
        unit['source'].update(library_bundle_id=binding['library_bundle_id'],
                              library_manifest_sha256=binding['manifest_sha256'],
                              lmdb_sha256=binding['lmdb_sha256'], annotations_sha256=binding['annotations_sha256'])
    return binding


def test_eight_authenticated_candidates_and_true_product_dev_source(word_bundle):
    from copy import deepcopy
    from slbridge.animation.word_composition import WordLibrary
    root,manifest = word_bundle
    binding=product_binding(manifest)
    for index in range(7):
        item=deepcopy(manifest['units'][0]);item['id']=f'extra{index}'
        manifest['units'].append(item)
    dev={**binding,'library_bundle_id':'dev-library','split':'dev','manifest_sha256':'d'*64,'lmdb_sha256':'e'*64}
    manifest['source']['sources'].append(dev)
    unit=manifest['units'][-1]
    unit['source'].update(split='dev',library_bundle_id='dev-library',library_manifest_sha256='d'*64,lmdb_sha256='e'*64)
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    library=WordLibrary(root)
    assert len(library._words['我'])==8
    assert library.compose('我去医院。',['我','去','医院'])['status']=='ready'


@pytest.mark.parametrize('fault',['missing_product_role','missing_eval_flag','mismatched_split','missing_origin','changed_origin_sha','ninth_candidate'])
def test_expanded_source_boundaries_fail_closed(word_bundle,fault):
    from copy import deepcopy
    from slbridge.animation.word_composition import WordLibrary,WordLibraryError
    root,manifest=word_bundle
    binding=product_binding(manifest)
    dev={**binding,'library_bundle_id':'dev-library','split':'dev','manifest_sha256':'d'*64,'lmdb_sha256':'e'*64}
    manifest['source']['sources'].append(dev)
    manifest['units'][0]['source'].update(split='dev',library_bundle_id='dev-library',library_manifest_sha256='d'*64,lmdb_sha256='e'*64)
    if fault=='missing_product_role':manifest.pop('usage_role')
    elif fault=='missing_eval_flag':manifest['evaluation_eligible']=True
    elif fault=='mismatched_split':manifest['units'][0]['source']['split']='train'
    elif fault=='missing_origin':manifest['units'][0]['source'].pop('library_bundle_id')
    elif fault=='changed_origin_sha':manifest['units'][0]['source']['lmdb_sha256']='9'*64
    else:
        for index in range(8):
            unit=deepcopy(manifest['units'][0]);unit['id']=f'additional{index}';manifest['units'].append(unit)
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(WordLibraryError):WordLibrary(root)


def test_official_release_pair_cannot_be_relabelled_even_when_binding_matches(word_bundle):
    from slbridge.animation.word_composition import WordLibrary,WordLibraryError
    from slbridge.animation.word_library_build import SOURCE_IDENTITIES
    root,manifest=word_bundle
    lmdb_sha,annotations_sha=SOURCE_IDENTITIES['train']
    manifest['source'].update(lmdb_sha256=lmdb_sha,annotations_sha256=annotations_sha)
    binding=product_binding(manifest)
    binding['split']='dev'
    for unit in manifest['units']:unit['source']['split']='dev'
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(WordLibraryError):WordLibrary(root)
