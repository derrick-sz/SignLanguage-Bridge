import importlib.util
import json
from pathlib import Path
import threading

from experiments.p1.e02.llm_fusion.backends.base import BackendResult

spec=importlib.util.spec_from_file_location('scene_validation',Path(__file__).parents[1]/'scripts/validate_slrt_llm_scenes.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Backend:
    def __init__(self): self.calls=[]
    def generate(self,messages,**kwargs):
        tokens=json.loads(messages[-1]['content'])['tokens']
        self.calls.append(tokens)
        return BackendResult(json.dumps({'status':'draft','text':' '.join(tokens),
            'used_token_indices':list(range(len(tokens)))}),latency_ms=1,extra={})


def test_empty_visual_evidence_does_not_call_llm_or_invent_a_sentence():
    backend=Backend()
    result=module.organize_tokens(backend,[])
    assert not backend.calls
    assert result['status']=='insufficient' and result['text']=='' and not result['called']


def test_organizer_receives_only_actual_words_without_reference_or_scene_answer():
    backend=Backend()
    result=module.organize_tokens(backend,['我','想','睡觉'])
    assert backend.calls==[['我','想','睡觉']]
    assert result['status']=='draft' and result['used_token_indices']==[0,1,2]


def test_late_version_is_discarded_and_same_eof_version_is_not_requested_twice():
    entered=threading.Event(); release=threading.Event()
    class BlockingBackend(Backend):
        def generate(self,messages,**kwargs):
            if not self.calls:
                entered.set()
                assert release.wait(5)
            return super().generate(messages,**kwargs)
    backend=BlockingBackend()
    import time
    organizer=module.IncrementalOrganizer(backend,time.perf_counter())
    organizer.offer(['我','想'])
    assert entered.wait(5)
    organizer.offer(['我','想','喝','水'])
    organizer.offer(['我','想','喝','水'],eof=True)
    release.set(); organizer.finish(['我','想','喝','水'])
    assert backend.calls==[['我','想'],['我','想','喝','水']]
    assert organizer.calls[0]['stale']
    assert len(organizer.published)==1
    assert organizer.published[0]['tokens']==['我','想','喝','水']


def test_invalid_token_evidence_is_a_format_failure():
    class InvalidBackend(Backend):
        def generate(self,*args,**kwargs):
            return BackendResult('{"status":"draft","text":"一句话","used_token_indices":[99]}',
                latency_ms=1,extra={})
    result=module.organize_tokens(InvalidBackend(),['绿色'])
    assert result['status']=='error' and result['text']==''


def test_async_provider_exception_is_recorded_without_sensitive_error_text():
    class RaisingBackend:
        def generate(self,*args,**kwargs):
            raise RuntimeError('sensitive-provider-detail')
    import time
    organizer=module.IncrementalOrganizer(RaisingBackend(),time.perf_counter())
    organizer.finish(['我','想','睡觉'])
    assert len(organizer.calls)==1 and not organizer.thread.is_alive()
    failure=organizer.calls[0]['result']
    assert failure['status']=='error' and failure['error_type']=='RuntimeError'
    assert failure['called'] is None
    assert 'sensitive-provider-detail' not in json.dumps(organizer.calls)


def test_nonempty_draft_without_declared_token_evidence_is_rejected():
    class NoEvidenceBackend(Backend):
        def generate(self,*args,**kwargs):
            return BackendResult('{"status":"draft","text":"我想睡觉","used_token_indices":[]}',
                latency_ms=1,extra={})
    result=module.organize_tokens(NoEvidenceBackend(),['绿色'])
    assert result['status']=='error' and result['text']==''


def test_cancel_does_not_flush_partial_tail_as_natural_eof():
    backend=Backend()
    import time
    organizer=module.IncrementalOrganizer(backend,time.perf_counter())
    organizer.finish(['我','想','睡觉'],natural_eof=False)
    assert backend.calls==[] and organizer.calls==[] and organizer.published==[]
