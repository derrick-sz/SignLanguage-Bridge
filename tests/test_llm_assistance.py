import asyncio
import json
import threading
import time

import pytest

from slbridge import llm_assistance as a


def reply(text):return {'status':'suggestion','text':text,'used_token_indices':[0]}


def wait(queue,job):
    deadline=time.monotonic()+3
    while queue.get(job)['status'] in {'pending','running'}:
        assert time.monotonic()<deadline;time.sleep(.005)
    return queue.get(job)


def submit(queue,utterance=0,revision=1):
    return queue.submit(session_id='session',source_epoch=1,utterance_id=utterance,
                        revision=revision,glosses=['老师','重复','重点'],text='老师重复说重点。')


def test_changes_to_negation_person_quantity_or_time_are_advisory_risks():
    out=a.validate_advice(reply('他不是去见同学。'),['不是','他','去','见面','同学'],'不是，他是去见他同学。')
    assert out['status']=='risk' and out['text']!='他不是去见同学。'
    assert a.validate_advice(reply('我明天去学校。'),['我','昨天','学校'],'我昨天去学校。')['status']=='risk'


def test_suggestion_cannot_introduce_an_unobserved_content_word_or_invalid_evidence():
    assert a.validate_advice(reply('老师重复说医院。'),['老师','重复','重点'],'老师重复说重点。')['status']=='risk'
    with pytest.raises(ValueError):
        a.validate_advice(dict(reply('老师说重点。'),used_token_indices=[99]),['老师'],'老师重复说重点。')


def test_disabled_backend_does_not_access_credentials_or_network():
    q=a.AdviceQueue(a.LLMConfig(enabled=False),backend=lambda *x:pytest.fail('Disabled backend called'))
    assert q.get(submit(q))['status']=='unavailable';q.close()


def test_slow_advice_does_not_block_submit_or_overwrite_another_sentence():
    gate=threading.Event();started=threading.Event()
    async def backend(glosses,text):
        started.set()
        while not gate.is_set():await asyncio.sleep(.005)
        return reply(text)
    q=a.AdviceQueue(a.LLMConfig(enabled=True),backend=backend)
    first=submit(q);assert started.wait(1)
    second=submit(q,1);third=submit(q,2)
    assert q.get(second)['status']=='pending' and q.get(third)['status']=='busy'
    gate.set();assert wait(q,first)['status']=='suggestion';assert wait(q,second)['status']=='suggestion'
    assert q.close()


def test_new_revision_invalidates_a_late_old_result():
    gate=threading.Event();started=threading.Event()
    async def backend(glosses,text):
        started.set()
        while not gate.is_set():await asyncio.sleep(.005)
        return reply(text)
    q=a.AdviceQueue(a.LLMConfig(enabled=True),backend=backend)
    old=submit(q);assert started.wait(1);new=submit(q,revision=2);gate.set()
    assert wait(q,old)['status']=='superseded' and wait(q,new)['status']=='suggestion';q.close()


def test_backend_exceptions_never_return_credentials_in_public_jobs():
    async def backend(*args):raise RuntimeError('Bearer very-private-key')
    q=a.AdviceQueue(a.LLMConfig(enabled=True),backend=backend)
    result=wait(q,submit(q));assert result['status']=='error'
    assert 'very-private-key' not in json.dumps(result);q.close()


def test_remote_or_credential_in_url_configuration_is_rejected():
    for url in ['https://remote.invalid/v1','http://secret@127.0.0.1:8788/v1','http://127.0.0.1:8788/v1?token=x']:
        with pytest.raises(ValueError):a.LLMConfig(base_url=url)


def test_whole_job_deadline_cancels_slow_work_and_allows_next_sentence():
    started=threading.Event();calls=0
    async def backend(glosses,text):
        nonlocal calls
        calls+=1;started.set()
        if calls==1:await asyncio.sleep(10)
        return reply(text)
    q=a.AdviceQueue(a.LLMConfig(enabled=True,timeout_seconds=1),backend=backend)
    first=submit(q);assert started.wait(1);second=submit(q,1)
    result=wait(q,first)
    assert result['status']=='timeout' and result['latency_ms']<1800
    assert wait(q,second)['status']=='suggestion';assert q.close()


def test_shutdown_cancels_active_advice_within_bound():
    started=threading.Event()
    async def backend(*args):started.set();await asyncio.sleep(30)
    q=a.AdviceQueue(a.LLMConfig(enabled=True),backend=backend)
    job=submit(q);assert started.wait(1);start=time.monotonic()
    assert q.close() and time.monotonic()-start<1
    assert q.get(job)['status']=='cancelled'


def test_blocking_backend_is_rejected_instead_of_leaking_replacement_threads():
    with pytest.raises(TypeError,match='async'):
        a.AdviceQueue(a.LLMConfig(enabled=True),backend=lambda *args:time.sleep(10))


@pytest.mark.parametrize('drip,expected',[(True,'timeout'),(False,'error')])
def test_http_body_has_total_time_and_size_bounds(monkeypatch,drip,expected):
    import httpx
    class Response:
        def raise_for_status(self):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def aiter_bytes(self,chunk_size):
            for _ in range(20):
                if drip:await asyncio.sleep(.2)
                yield b'x'*4096
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def stream(self,*args,**kwargs):return Response()
    monkeypatch.setattr(a,'_credentials',lambda config:'private')
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    q=a.AdviceQueue(a.LLMConfig(enabled=True,timeout_seconds=1))
    result=wait(q,submit(q));assert result['status']==expected
    assert result['latency_ms']<1800;assert q.close()
