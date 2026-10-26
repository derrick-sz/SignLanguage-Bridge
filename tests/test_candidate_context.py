"""Candidate evidence survives display coalescing without becoming a caption."""
from dataclasses import replace
import json

import pytest

from slbridge.captions import CaptionCatalog, CaptionEntry
from slbridge.continuous import ContinuousUpdate
from slbridge.contracts import (
    ClassificationPrediction, GlossSequencePrediction, GlossToken, RecognitionResult,
)
from slbridge.runtime_contracts import InferenceResponse
from test_continuous_session import setup, stop, wait, WindowModel


def ledger(limit=200):
    from slbridge.candidate_context import CandidateLedger
    return CandidateLedger('test-session', max_history=limit)


def update(number=1, *, language='libras', gloss=False):
    prediction = (GlossSequencePrediction((GlossToken(1, '我', .7, (.1, .2)),
                    GlossToken(2, '水', .6, (.3, .4))), 'greedy') if gloss else
        ClassificationPrediction('yes', .6, {'yes': .6, 'tree': .3, 'water': .1}, {'pt-BR': 'Sim'}))
    result = RecognitionResult('recognized', f'sample-{number}', f'model-{language}', language,
        prediction, {'observed': .9}, task_type='gloss_sequence' if gloss else 'classification',
        score_type='ctc_token_probability' if gloss else 'uncalibrated_class_probability')
    response = InferenceResponse(f'request-{number}', 1, result, {}, {'inference_ms': 4.},
        session_id='test-session', source_epoch=1, model_version='version-1', preprocessing_version='prep-1')
    return ContinuousUpdate(response, number*.5, number*.5+2., 100.+number, .1)


def append(store, value, *, epoch=1, generation=1, window_revision=1):
    return store.append(value, source_id='camera-test', source_epoch=epoch,
        model_generation=generation, window_revision=window_revision, catalog=CaptionCatalog())


def test_ordered_history_keeps_revisions_alternatives_and_overlapping_window_evidence():
    store = ledger()
    first = update()
    append(store, first)
    append(store, update(2))
    snapshot = store.snapshot()
    assert [row['sequence'] for row in snapshot['candidates']] == [1, 2]
    a, b = snapshot['candidates']
    assert b['overlaps_candidate_id'] == a['candidate_id']
    assert a['status'] == b['status'] == 'unconfirmed'
    assert a['result']['prediction']['scores'] == {'yes': .6, 'tree': .3, 'water': .1}
    first.response.result.prediction.scores['yes'] = .01
    snapshot['candidates'][0]['result']['prediction']['scores']['tree'] = 1.
    assert store.snapshot()['candidates'][0]['result']['prediction']['scores']['yes'] == .6
    assert store.snapshot()['candidates'][0]['result']['prediction']['scores']['tree'] == .3
    assert store.snapshot(after_revision=snapshot['revision']) is None


def test_capacity_clear_and_cross_source_language_boundaries():
    store = ledger(3)
    for number in range(1, 5):
        append(store, update(number))
    assert [r['sequence'] for r in store.snapshot()['candidates']] == [2, 3, 4]
    assert store.snapshot()['dropped_count'] == 1
    # A new binding is retained but never represented as the same overlapping action.
    value = update(5, language='csl', gloss=True)
    value = replace(value, response=replace(value.response, source_epoch=2, model_generation=2))
    append(store, value, epoch=2, generation=2, window_revision=2)
    row = store.snapshot()['candidates'][-1]
    assert row['overlaps_candidate_id'] is None
    assert row['result']['prediction']['tokens'][0]['time_range'] == (.1, .2)
    assert row['token_time_origin'] == 'window_start'
    store.set_limit(1)
    assert len(store.snapshot()['candidates']) == 1 and store.snapshot()['dropped_count'] == 4
    before = store.snapshot()['revision']
    store.clear()
    assert store.snapshot()['revision'] > before
    assert store.snapshot()['candidates'] == [] and store.snapshot()['dropped_count'] == 0
    append(store, update(6))
    assert store.snapshot()['candidates'][0]['sequence'] == 6


def test_candidate_caption_is_frozen_with_catalog_review_provenance():
    store = ledger()
    catalog = CaptionCatalog({'yes': CaptionEntry({'zh-CN': '是 / 是的'}, 'draft')},
                             sign_language='libras', catalog_id='dictionary-v1')
    store.append(update(), source_id='test', source_epoch=1, model_generation=1,
                 window_revision=1, catalog=catalog)
    row = store.snapshot()['candidates'][0]
    assert row['caption']['zh'] == '是 / 是的'
    assert row['catalog_id'] == 'dictionary-v1'
    assert row['catalog_review_status'] == 'draft'


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, 1001])
def test_invalid_capacity_is_rejected(limit):
    with pytest.raises(ValueError):
        ledger(limit)


def test_history_archives_every_completed_window_even_without_ui_polling():
    session, source = setup()
    try:
        assert callable(getattr(session, 'candidate_context', None)), 'Session candidate snapshot API is missing'
        source.gate.set()
        wait(lambda: session.continuous_status()['completed_windows'] >= 4)
        with session._lock:
            snapshot = session.candidate_context()
            assert len(snapshot['candidates']) == session.continuous_status()['completed_windows']
            assert len([e for e in session.poll() if e.kind == 'continuous_result']) == 1
        assert not session.captions.snapshot()['events']
        session.disable_continuous()
        session.stop_source()
        assert session.candidate_context()['candidates'] == snapshot['candidates']
    finally:
        stop(session, source)


def test_clear_candidates_invalidates_native_call_but_keeps_stream_enabled(tmp_path):
    model = WindowModel(block=True)
    session, source = setup(model)
    try:
        assert callable(getattr(session, 'clear_candidates', None)), 'Candidate reset API is missing'
        source.gate.set()
        assert model.entered.wait(2)
        source.gate.clear()
        session.clear_candidates()
        model.release.set()
        wait(lambda: not session.busy)
        assert not session.candidate_context()['candidates']
        assert session.continuous and session.source_active
        assert not any(e.kind == 'continuous_result' for e in session.poll())
        destination = tmp_path/'context.json'
        session.export_candidate_context(destination)
        exported = json.loads(destination.read_text(encoding='utf-8'))
        assert exported['kind'] == 'recognition_candidate_context'
        assert exported['candidates'] == []
    finally:
        model.release.set()
        stop(session, source)


@pytest.mark.parametrize('action', ['cancel', 'stop_source', 'pause_source', 'disable_continuous'])
def test_invalidated_calls_never_enter_candidate_context(action):
    model = WindowModel(block=True)
    session, source = setup(model, kind='video')
    try:
        assert callable(getattr(session, 'candidate_context', None)), 'Candidate snapshot API is missing'
        source.gate.set()
        assert model.entered.wait(2)
        source.count = source.index
        getattr(session, action)(True) if action == 'pause_source' else getattr(session, action)()
        model.release.set()
        wait(lambda: not session.busy)
        assert not session.candidate_context()['candidates']
    finally:
        model.release.set()
        stop(session, source)


def test_other_sessions_rejections_and_changed_generation_are_not_evidence():
    store = ledger()
    value = update()
    for response in (
        replace(value.response, session_id='someone-else'),
        replace(value.response, source_epoch=2),
        replace(value.response, model_generation=2),
        replace(value.response, result=replace(value.response.result, status='rejected', prediction=None)),
    ):
        assert not append(store, replace(value, response=response))
    assert store.snapshot()['candidates'] == []


def test_history_rejects_late_native_result_after_age_expiry():
    model = WindowModel(block=True)
    session, source = setup(model, max_result_age_seconds=.1)
    try:
        source.gate.set()
        assert model.entered.wait(2)
        source.gate.clear()
        import time
        time.sleep(.15)
        model.release.set()
        wait(lambda: not session.busy)
        assert session.continuous_status()['stale_results'] > 0
        assert session.candidate_context()['candidates'] == []
    finally:
        model.release.set()
        stop(session, source)
