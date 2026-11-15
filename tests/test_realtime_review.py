"""Evidence hints describe uncertainty; they must preserve the model output."""
import numpy as np
from types import SimpleNamespace

from slbridge.realtime_engine import RealtimeEngine
from slbridge.slrt.assets import SLRTConfig
from slbridge.runtime_contracts import CancellationToken
from tests.test_realtime_engine import feed
from tests.test_realtime_segment import pose


def engine_with_one_word(word, text):
    events = []
    class Runtime:
        vocabulary = ('<blank>', word)
        info = {'test_double': True}
        def predict(self, *args):
            return np.array([.01,.99],np.float32),dict(preprocess_ms=0,inference_ms=0,window_total_ms=0)
    class Frontend:
        info = {'test_double': True}
        def extract(self, image, **kwargs): return pose(min(int(image[0,0,0]),6)*10)
        def close(self): pass
    session=SimpleNamespace(session_id='review',source_epoch=1,cancelled=False,cancellation=CancellationToken(),
                            emit=lambda kind,**data:events.append(dict(type=kind,**data)))
    engine=RealtimeEngine(session,Runtime(),Frontend(),SLRTConfig(stride=2,vote_size=1),fps=10,
                          translator=lambda *args,**kwargs:(text,{'test_double':True}))
    for i in range(25): feed(engine,i)
    engine.finish();engine.close()
    return events


def test_one_word_large_expansion_is_marked_without_changing_or_hiding_native_text():
    events=engine_with_one_word('没有','没钱，我出去玩。')
    sentence=next(e for e in events if e['type']=='sentence' and e['status']=='committed')
    assert sentence['text']=='没钱，我出去玩。'
    assert sentence['predicted_gloss']==['没有']
    assert sentence['review_hint']['reason']=='single_word_expansion'
    assert sentence['review_hint']['gloss_count']==1
    assert sentence['review_hint']['text_units']>sentence['review_hint']['gloss_units']*2


def test_legitimate_single_word_greetings_remain_unflagged_and_are_not_replaced():
    for word,text in [('谢谢','谢谢。'),('你好','你好！')]:
        sentence=next(e for e in engine_with_one_word(word,text)
                      if e['type']=='sentence' and e['status']=='committed')
        assert sentence['text']==text
        assert sentence['review_hint'] is None
