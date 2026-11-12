from types import SimpleNamespace
import numpy as np

from slbridge import realtime_engine as m
from slbridge.realtime_protocol import DecodedFrame
from slbridge.runtime_contracts import CancellationToken
from slbridge.slrt.assets import SLRTConfig
from slbridge.sources import FramePacket
from tests.test_realtime_segment import pose


class Runtime:
    vocabulary=('<blank>','我')
    info={'test_double':True}
    def predict(self,*args):
        return np.array([.01,.99],np.float32),dict(preprocess_ms=0,inference_ms=0,window_total_ms=0)


def setup(moving=True):
    events=[];translations=[]
    session=SimpleNamespace(session_id='test',source_epoch=3,cancelled=False,cancellation=CancellationToken(),
                            emit=lambda kind,**data:events.append(dict(data,type=kind)))
    class Frontend:
        info={'fixture':True}
        def extract(self,image,**kwargs):return pose(min(int(image[0,0,0]),6)*10 if moving else 0)
        def close(self):pass
    def translate(points,scores,**kwargs):
        translations.append((points.copy(),scores.copy()));return '我想睡觉。',{'shared_pose':True}
    engine=m.RealtimeEngine(session,Runtime(),Frontend(),SLRTConfig(stride=2,vote_size=1),
                            translator=translate,fps=10,pause_seconds=.8)
    return engine,session,events,translations


def feed(engine,index,image_index=None):
    frame=FramePacket(np.full((32,32,3),index if image_index is None else image_index,np.uint8),index/10,index,'test')
    engine.feed(DecodedFrame(frame,index*100))


def test_words_arrive_before_sentence_and_pose_is_reused_at_boundary():
    engine,session,events,translations=setup()
    for i in range(25):feed(engine,i)
    engine.finish();engine.close()
    kinds=[e['type'] for e in events]
    assert kinds.index('word')<kinds.index('sentence')
    sentences=[e for e in events if e['type']=='sentence' and e['status']=='committed']
    assert len(sentences)==1 and sentences[0]['text']=='我想睡觉。'
    assert len(translations)==1 and translations[0][0].shape[1:]==(133,2)


def test_static_person_never_reaches_translation_even_if_classifier_guesses_a_word():
    engine,session,events,translations=setup(moving=False)
    for i in range(25):feed(engine,i)
    engine.finish();engine.close()
    assert not translations
    assert not any(e['type'] in {'word','sentence'} for e in events)


def test_interrupted_input_is_not_flushed_as_a_complete_sentence():
    engine,session,events,translations=setup()
    for i in range(10):feed(engine,i)
    feed(engine,20);engine.cancel();engine.close()
    assert not translations
    assert any(e['type']=='utterance_reset' for e in events)


def test_advice_failure_does_not_interrupt_words_or_the_original_translation():
    engine,session,events,translations=setup()
    def bad_advice(**kwargs):raise RuntimeError('fixture')
    engine.assister=SimpleNamespace(submit=bad_advice)
    for i in range(25):feed(engine,i)
    engine.finish();engine.close()
    assert any(e['type']=='sentence' and e['status']=='committed' for e in events)
    assert any(e['type']=='assistance' and e['status']=='unavailable' for e in events)


def test_sentence_translation_waits_for_input_debt_without_losing_previous_sentence():
    engine,session,events,translations=setup();session.pending_frames=8
    for i in range(25):feed(engine,i)
    assert not translations
    assert any(e['type']=='sentence' and e['status']=='pending' for e in events)
    pending_id=next(e['utterance_id'] for e in events if e['type']=='sentence')
    # Input resumes in the next utterance while the completed previous one waits.
    seen=[];native=engine.translator
    def record(*args,**kwargs):seen.append(engine.frames);return native(*args,**kwargs)
    engine.translator=record
    session.pending_frames=0;feed(engine,25)
    assert len(translations)==1
    assert seen==[26]  # Process the already-dequeued frame before lower-priority translation.
    assert any(e['type']=='sentence' and e['status']=='committed' and e['utterance_id']==pending_id for e in events)
    engine.finish();engine.close()


def test_waiting_translation_deadline_and_cancel_do_not_manufacture_a_sentence(monkeypatch):
    clock=[100.];monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    engine,session,events,translations=setup();session.pending_frames=8
    for i in range(25):feed(engine,i)
    clock[0]+=2;feed(engine,25)
    assert not translations
    assert any(e['type']=='sentence' and e['status']=='rejected' and e.get('reason')=='translation_wait_limit' for e in events)
    assert engine.translation_wait_max_ms>=2000
    engine.cancel();session.pending_frames=0;engine.finish();engine.close()
    assert not translations


def test_natural_eof_drains_deferred_translation_and_cancel_discards_it():
    for cancel in [False,True]:
        engine,session,events,translations=setup();session.pending_frames=8
        for i in range(25):feed(engine,i)
        assert not translations
        if cancel:engine.cancel()
        engine.finish();engine.close()
        assert len(translations)==(0 if cancel else 1)


def test_bounded_translation_queue_preserves_earlier_utterances_and_runs_when_idle(monkeypatch):
    monkeypatch.setattr(m.time,'monotonic',lambda:100.)
    engine,session,events,translations=setup();session.pending_frames=8
    for cycle in range(3):
        for i in range(25):feed(engine,cycle*25+i,image_index=i)
        engine.boundary()
    assert not translations and engine.translation_queue_peak==2
    assert any(e['type']=='sentence' and e.get('reason')=='translation_queue_limit' for e in events)
    pending=[e['utterance_id'] for e in events if e['type']=='sentence' and e['status']=='pending']
    session.pending_frames=0;engine.idle()
    committed=[e['utterance_id'] for e in events if e['type']=='sentence' and e['status']=='committed']
    assert committed==pending[:2] and len(translations)==2
    engine.cancel();engine.close()
