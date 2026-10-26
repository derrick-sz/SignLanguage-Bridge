"""Real thread/queue lifecycle checks, with bounded deterministic input doubles."""
import time
import threading
from dataclasses import replace
import numpy as np
import pytest

from slbridge.slrt.assets import SLRTConfig
from slbridge.slrt.capture import SLRTCaptureSession
from slbridge.slrt.stream import SLRTStream
from slbridge.sources import FramePacket


class Runtime:
    vocabulary=('<blank>','A')
    info={'test_double':True}
    def predict(self,*args):
        time.sleep(.002)
        return np.array([.05,.95],np.float32),dict(preprocess_ms=0.,inference_ms=2.,window_total_ms=2.)


class Frontend:
    info={'test_double':True}
    closed=False
    def extract(self,*args,**kwargs):
        return np.zeros((133,3),np.float32)
    def close(self):self.closed=True


class Backend:
    config=SLRTConfig(queue_size=4,vote_size=1,stride=4)
    def new_stream(self,**kwargs):
        self.frontend=Frontend()
        return SLRTStream(Runtime(),self.frontend,self.config,event_sink=kwargs.get('event_sink'),window_sink=kwargs.get('window_sink'))


class Source:
    def __init__(self,count=120,delay=0):
        self.i=0;self.count=count;self.delay=delay;self.closed=False
    def open(self):return self
    def read(self):
        if self.i==self.count:return None
        time.sleep(self.delay)
        p=FramePacket(np.zeros((512,512,3),np.uint8),self.i/30,self.i,'queue-test');self.i+=1
        return p
    def close(self):self.closed=True


def wait_for(predicate,timeout=5):
    end=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>end:pytest.fail('Timed out waiting for capture lifecycle')
        time.sleep(.01)


def test_queue_is_bounded_and_natural_eof_releases_source():
    b=Backend();source=Source()
    s=SLRTCaptureSession(b,source_factory=lambda:source)
    s.start();wait_for(lambda:not s.is_running)
    r=s.result()
    assert r['error'] is None and r['max_queue_depth']<=4
    assert r['frames_captured']==120 and r['queue_dropped']>0
    assert r['retained_frames']<=16 and r['status']=='final'
    assert source.closed and b.frontend.closed
    s.stop();s.stop()


def test_pause_resume_reset_and_stop_are_nonblocking():
    b=Backend();source=Source(count=10000,delay=.01)
    s=SLRTCaptureSession(b,source_factory=lambda:source)
    s.start();wait_for(lambda:s.result().get('frames_received',0)>=12)
    s.pause();time.sleep(.1);r=s.result()
    before=r.get('frames_received',0)
    time.sleep(.1)
    assert s.result().get('frames_received',0)==before
    s.resume();wait_for(lambda:s.result().get('frames_received',0)>before)
    s.reset();wait_for(lambda:s.result().get('frames_received',10000)<before)
    s.stop()
    assert not s.is_running and source.closed and b.frontend.closed


def test_camera_disconnect_is_an_error_and_does_not_flush_a_tail():
    class BrokenSource(Source):
        def read(self):
            if self.i>=12:raise RuntimeError('camera disconnected')
            return super().read()
    b=Backend();source=BrokenSource(delay=.01)
    s=SLRTCaptureSession(b,source_factory=lambda:source)
    s.start();wait_for(lambda:not s.is_running)
    assert 'camera disconnected' in s.error
    assert s.result()['gloss_sequence']==[]
    assert source.closed


def test_pause_resume_and_reset_cannot_be_lost_during_native_inference():
    entered,release=threading.Event(),threading.Event()
    class BlockedRuntime(Runtime):
        def predict(self,*args):
            if not entered.is_set():
                entered.set();assert release.wait(3)
            return super().predict(*args)
    class RecordingBackend(Backend):
        config=SLRTConfig(queue_size=32,vote_size=1)
        def new_stream(self,**kwargs):
            self.frontend=Frontend()
            self.stream=SLRTStream(BlockedRuntime(),self.frontend,self.config,event_sink=kwargs['event_sink'],window_sink=kwargs['window_sink'])
            return self.stream
    b=RecordingBackend();source=Source(count=1000,delay=.01)
    windows=[]
    s=SLRTCaptureSession(b,source_factory=lambda:source,window_sink=windows.append)
    s.start();assert entered.wait(3)
    s.pause();s.resume();s.reset();s.pause();s.resume()
    boundary=source.i
    release.set()
    wait_for(lambda:s.result().get('windows_processed',0)>=2)
    s.stop()
    assert all(min(w['source_frame_indices'])>=boundary-1 for w in windows)
    assert all(t['time_range'][0]>=max(0,boundary-1)/30 for t in s.result().get('tokens',[]))


def test_native_capture_pending_is_never_reported_as_released():
    release,entered=threading.Event(),threading.Event()
    class BlockedSource(Source):
        def read(self):
            entered.set();release.wait(30)
            return None
    b=Backend();source=BlockedSource()
    s=SLRTCaptureSession(b,source_factory=lambda:source)
    s.start();assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError,match='releasing'):
            s.stop(timeout=6)
        assert s.is_running and not source.closed
        original=s._worker
        s.start()
        assert s._worker is original
    finally:
        release.set();s.stop()
    assert source.closed and not s.is_running


def test_reset_immediately_followed_by_stop_cannot_restore_old_gloss():
    entered,release=threading.Event(),threading.Event()
    class StepRuntime(Runtime):
        calls=0
        def predict(self,*args):
            self.calls+=1
            if self.calls==3:
                entered.set();assert release.wait(3)
            p=np.array([.05,.95] if self.calls==1 else [.95,.05],np.float32)
            return p,dict(preprocess_ms=0.,inference_ms=0.,window_total_ms=0.)
    class StepBackend(Backend):
        config=SLRTConfig(queue_size=32,vote_size=1)
        def new_stream(self,**kwargs):
            self.frontend=Frontend()
            return SLRTStream(StepRuntime(),self.frontend,self.config,event_sink=kwargs['event_sink'],window_sink=kwargs['window_sink'])
    b=StepBackend();source=Source(count=1000,delay=.01)
    s=SLRTCaptureSession(b,source_factory=lambda:source)
    s.start();assert entered.wait(3)
    assert s.result()['gloss_sequence']==['A']
    s.reset();s.request_stop();release.set();s.stop()
    assert s.result().get('gloss_sequence',[])==[]
    assert s.result().get('partial') is None
