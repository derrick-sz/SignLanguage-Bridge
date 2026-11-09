"""The persistent planner is exercised through real tiny JSONL subprocesses."""
import hashlib
from importlib import import_module
import json
from pathlib import Path
import runpy
import sys
import threading
import time

import pytest


@pytest.fixture
def module():
    return import_module('slbridge.animation.text_gloss_planner')


@pytest.fixture
def planner_factory(tmp_path, monkeypatch, module):
    assets = {}
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    for name in ('model.safetensors', 'config.json', 'tokenizer_config.json',
                 'generation_config.json', 'sentencepiece.bpe.model',
                 'signspark_t2g.json', 'special_tokens_map.json'):
        raw = ('fixture-' + name).encode()
        (model_dir / name).write_bytes(raw)
        assets[name] = (len(raw), hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(module, 'ASSET_FILES', assets)
    script = tmp_path / 'worker.py'
    script.write_text('''import argparse, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--model-dir'); p.add_argument('--device'); p.add_argument('--identity-json')
a=p.parse_args(); identity=json.loads(a.identity_json)
mode=Path(a.model_dir).parent.joinpath('mode').read_text()
if mode=='startup_timeout': time.sleep(5)
if mode=='wrong_identity': identity['model_sha256']='0'*64
if mode=='startup_error':
 print(json.dumps({'type':'error','error':'model_unavailable'}),flush=True); sys.exit(1)
print(json.dumps({'type':'ready','protocol':'signbridge-t2g-jsonl-v1','model_identity':identity,
 'startup_metrics':{'load_ms':1.0,'warmup_ms':2.5}}),flush=True)
for line in sys.stdin:
 r=json.loads(line); Path(a.model_dir).joinpath('received').write_text(r['text'],encoding='utf-8')
 if r['text']=='slow' or mode=='timeout': time.sleep(.5)
 if mode=='exit': sys.exit(4)
 if mode=='malformed': print('not-json',flush=True); continue
 if mode=='huge': print('x'*40000,flush=True); continue
 if mode=='request_error' and r['text']=='bad':
  print(json.dumps({'type':'error','id':r['id'],'error':'output_limit'}),flush=True); continue
 result={'type':'result','id':r['id'],'glosses':['我','陌生词'], 'model_identity':identity,'elapsed_ms':1.25}
 if mode=='wrong_id': result['id']='old'
 if mode=='empty': result['glosses']=[]
 if mode=='too_many': result['glosses']=['词']*33
 if mode=='bad_gloss': result['glosses']=['有 空格']
 if mode=='negative_time': result['elapsed_ms']=-1
 if mode=='changed_identity': result['model_identity']={**identity,'num_beams':1}
 print(json.dumps(result,ensure_ascii=False),flush=True)
''', encoding='utf-8')
    (tmp_path / 'mode').write_text('normal')
    created = []

    def factory(mode='normal', **overrides):
        (tmp_path / 'mode').write_text(mode)
        config = {'python': sys.executable, 'script': str(script), 'model_dir': str(model_dir),
                  'device': 'cpu', 'startup_timeout': 2.0, 'request_timeout': 1.0}
        config.update(overrides)
        planner = module.TextGlossPlanner(config)
        created.append(planner)
        return planner, model_dir

    yield factory
    for planner in created:
        planner.close()


def settled(planner, wanted='ready', timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = planner.status()
        if state['state'] != 'starting':
            assert state['state'] == wanted, state
            return state
        time.sleep(.01)
    pytest.fail('planner did not leave starting')


def test_real_jsonl_worker_stays_loaded_and_preserves_predicted_glosses(planner_factory, module):
    planner, model_dir = planner_factory()
    status = settled(planner)
    result = planner.plan('  我\n想\t学习。  ')
    assert result['glosses'] == ['我', '陌生词']  # no dictionary/nearest-word substitution
    assert result['elapsed_ms'] == 1.25
    assert result['model_identity'] == status['model_identity']
    assert result['model_identity']['num_beams'] == 4
    assert status['startup_metrics'] == {'load_ms':1.0,'warmup_ms':2.5}
    assert 'model_dir' not in json.dumps(result)
    assert model_dir.joinpath('received').read_text(encoding='utf-8') == '我 想 学习。'
    assert planner.plan('第二句')['model_identity'] == result['model_identity']
    planner.close()
    assert planner.status()['state'] == 'failed'
    with pytest.raises(module.PlannerUnavailableError):
        planner.plan('已经关闭')


@pytest.mark.parametrize('text', ['', '  ', '字'*201, 'a\x00b', 'a\ud800b', None, 1])
def test_invalid_input_never_reaches_worker(planner_factory, text):
    planner, model_dir = planner_factory()
    settled(planner)
    with pytest.raises(ValueError):
        planner.plan(text)
    assert not (model_dir / 'received').exists()


@pytest.mark.parametrize('mode', ['wrong_id', 'malformed', 'empty', 'too_many',
                                  'bad_gloss', 'negative_time', 'changed_identity', 'huge', 'exit'])
def test_invalid_or_stale_response_fails_closed(planner_factory, module, mode):
    planner, _ = planner_factory(mode)
    settled(planner)
    with pytest.raises(module.PlannerUnavailableError):
        planner.plan('新的完整句')
    assert planner.status()['state'] == 'failed'


@pytest.mark.parametrize('mode', ['wrong_identity', 'startup_error', 'startup_timeout'])
def test_bad_startup_never_becomes_available(planner_factory, module, mode):
    planner, _ = planner_factory(mode, startup_timeout=.1)
    settled(planner, 'failed')
    with pytest.raises(module.PlannerUnavailableError):
        planner.plan('尚未就绪')


def test_request_timeout_invalidates_owned_process(planner_factory, module):
    planner, _ = planner_factory('timeout', request_timeout=.05)
    settled(planner)
    with pytest.raises(module.PlannerUnavailableError):
        planner.plan('超时')
    assert planner.status()['state'] == 'failed'
    assert planner._process.poll() is not None


def test_busy_request_is_rejected_without_queuing(planner_factory, module):
    planner, model_dir = planner_factory()
    settled(planner)
    results = []
    active = threading.Thread(target=lambda: results.append(planner.plan('slow')))
    active.start()
    deadline = time.monotonic()+1
    while not (model_dir / 'received').exists() and time.monotonic() < deadline:
        time.sleep(.005)
    with pytest.raises(module.PlannerBusyError):
        planner.plan('第二条')
    active.join(2)
    assert len(results) == 1 and planner.status()['state'] == 'ready'


def test_request_level_limit_error_does_not_poison_next_request(planner_factory):
    planner, _ = planner_factory('request_error')
    settled(planner)
    with pytest.raises(ValueError):
        planner.plan('bad')
    assert planner.status()['state'] == 'ready'
    assert planner.plan('下一条')['glosses'] == ['我', '陌生词']


def test_asset_hash_rejected_before_process_start(planner_factory, module):
    planner, model_dir = planner_factory('startup_timeout', startup_timeout=.5)
    # This direct verifier also proves same-size corruption is not accepted.
    settled(planner, 'failed')
    model_dir.joinpath('model.safetensors').write_bytes(b'x'*module.ASSET_FILES['model.safetensors'][0])
    with pytest.raises(ValueError, match='identity'):
        module.verify_assets(model_dir)


def test_missing_asset_path_is_sanitized(planner_factory):
    planner, _ = planner_factory(model_dir='C:/private/absent/secret')
    status = settled(planner, 'failed')
    assert 'private' not in json.dumps(status)


def test_changed_asset_is_rejected_before_next_request(planner_factory, module):
    planner, model_dir = planner_factory()
    settled(planner)
    model_dir.joinpath('config.json').write_bytes(b'changed')
    with pytest.raises(module.PlannerUnavailableError):
        planner.plan('不能沿用已变更的资产身份')
    assert planner.status()['state'] == 'failed'
    assert not model_dir.joinpath('received').exists()


def test_unverified_tokenizer_override_is_not_loaded(planner_factory, module):
    planner, model_dir = planner_factory()
    settled(planner)
    model_dir.joinpath('added_tokens.json').write_text('{}')
    with pytest.raises(ValueError, match='identity'):
        module.verify_assets(model_dir)


def test_close_during_startup_does_not_publish_ready_or_leak_log(planner_factory, module, tmp_path):
    planner, _ = planner_factory('startup_timeout', stderr_log=str(tmp_path/'worker.log'))
    planner.close()
    planner._starter.join(2)
    assert planner.status()['state'] == 'failed'
    assert planner._stderr is None
    assert planner._process is None or planner._process.poll() is not None
    with pytest.raises(module.PlannerUnavailableError):
        planner.plan('已关闭')


def test_close_during_request_rejects_late_result(planner_factory, module):
    planner, model_dir = planner_factory()
    settled(planner)
    errors = []
    def run():
        try:
            planner.plan('slow')
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic()+1
    while not model_dir.joinpath('received').exists() and time.monotonic() < deadline:
        time.sleep(.005)
    planner.close()
    thread.join(2)
    assert len(errors) == 1 and isinstance(errors[0], module.PlannerUnavailableError)
    assert planner.status()['state'] == 'failed'


def test_worker_prompt_and_no_truncation_contract():
    worker = runpy.run_path(str(Path(__file__).parents[1]/'scripts/animation/text_gloss_worker.py'), run_name='worker_fixture')
    assert worker['build_prompt']('明天学习。') == 'Translate Chinese Sentence to Chinese Sign Language Gloss: 明天学习。\nGloss:'
    class Tokenizer:
        eos_token_id=2
        unk_token_id=3
        def decode(self, ids, **kwargs):
            return '我 学习'
    assert worker['decode_glosses']([6034, 10, 11, 2], Tokenizer()) == ['我','学习']
    for bad in ([6034,10,11], [6034,3,2], [6034]+[10]*149+[2]):
        with pytest.raises(ValueError):
            worker['decode_glosses'](bad, Tokenizer())


@pytest.mark.parametrize('unknown_user_text,encoded_length,expected_error',
                         [(False, 31, None), (True, 31, 'unknown_token'), (False, 1025, 'input_limit')])
def test_worker_checks_user_vocabulary_without_rejecting_fixed_prompt_unknowns(
        unknown_user_text, encoded_length, expected_error):
    worker = runpy.run_path(str(Path(__file__).parents[1]/'scripts/animation/text_gloss_worker.py'), run_name='worker_fixture')
    calls = []
    class EncodedIds:
        shape=(1, encoded_length)
    class Tokenizer:
        unk_token_id=3
        def __call__(self, value, **kwargs):
            calls.append((value, kwargs))
            if isinstance(value, str):
                return {'input_ids':[3] if unknown_user_text else [10,11]}
            return {'input_ids':EncodedIds()}  # the author's English prompt has known UNKs
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            worker['encode_prompt'](Tokenizer(), '我想喝水。', 1024)
    else:
        encoded = worker['encode_prompt'](Tokenizer(), '我想喝水。', 1024)
        assert encoded['input_ids'].shape == (1, 31)
        assert calls[1][0] == ['Translate Chinese Sentence to Chinese Sign Language Gloss: 我想喝水。\nGloss:']
    assert all(call[1]['truncation'] is False for call in calls)
