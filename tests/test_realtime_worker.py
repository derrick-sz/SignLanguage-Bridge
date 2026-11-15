from contextlib import contextmanager
import threading
import time

import numpy as np

from slbridge import realtime_worker as w
from slbridge.realtime_protocol import DecodedFrame
from slbridge.sources import FramePacket


def packet(index):
    return DecodedFrame(FramePacket(np.zeros((32,32,3),np.uint8),index/30,index,'test'),100+index)


def wait(predicate):
    deadline=time.monotonic()+3
    while not predicate():
        assert time.monotonic()<deadline
        time.sleep(.005)


class Engine:
    info={'model_id':'fixture'}
    def __init__(self, session, gate=None):
        self.session=session;self.gate=gate;self.finished=False;self.closed=False;self.seen=[]
    def feed(self, frame):
        self.seen.append(frame.packet.frame_index)
        if self.gate:self.gate.wait(2)
        self.session.emit('word',status='partial',gloss='A')
    def finish(self):self.finished=True;self.session.emit('sentence',text='A')
    def cancel(self):pass


def session(gate=None, **kwargs):
    engines=[]
    @contextmanager
    def factory(s):
        e=Engine(s,gate);engines.append(e)
        try:yield e
        finally:e.closed=True
    s=w.LiveSession(source_epoch=2,engine_factory=factory,**kwargs)
    s.start();wait(lambda:bool(engines))
    return s,engines


def test_finish_drains_received_frames_and_flushes_only_at_natural_end():
    s,engines=session();s.submit(packet(0));s.submit(packet(1));s.finish()
    assert s.join(2)
    events=s.poll()
    assert engines[0].seen==[0,1] and engines[0].finished and engines[0].closed
    assert any(e['type']=='sentence' for e in events)
    assert events[-1]['type']=='finished'
    assert all(e['source_epoch']==2 and e['session_id']==s.session_id for e in events)


def test_cancel_discards_late_native_output_without_closing_running_engine():
    gate=threading.Event();s,engines=session(gate)
    s.poll();s.submit(packet(0));wait(lambda:engines[0].seen)
    s.cancel('changed_source')
    assert not s.join(.01) and not engines[0].closed
    gate.set();assert s.join(2)
    assert not engines[0].finished and engines[0].closed
    assert not any(e['type'] in {'word','sentence','ack'} for e in s.poll())


def test_overload_is_explicit_and_cannot_manufacture_a_final_sentence():
    gate=threading.Event();s,engines=session(gate,queue_size=1)
    s.poll();s.submit(packet(0));wait(lambda:engines[0].seen)
    assert s.submit(packet(1))
    assert not s.submit(packet(2))
    gate.set();assert s.join(2)
    events=s.poll()
    assert any(e['type']=='error' and e['code']=='overload' for e in events)
    assert not engines[0].finished


def test_manual_boundary_is_ordered_between_received_frames():
    s,engines=session();engines[0].boundary=lambda:engines[0].seen.append('boundary')
    s.submit(packet(0));assert s.boundary();s.submit(packet(1));s.finish()
    assert s.join(2)
    assert engines[0].seen==[0,'boundary',1]
