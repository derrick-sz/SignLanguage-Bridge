import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from slbridge.desktop import SignLanguageApp
from slbridge.desktop_widgets import PreviewWidget
from slbridge.segmentation import ActivityObservation
from slbridge.session import RecognitionSession
from slbridge.sources import FramePacket
from test_automatic_session import ReplaySource
from test_continuous_session import WindowModel, wait


def raw_landmarks():
    pose = [SimpleNamespace(x=.5,y=.5,visibility=0.) for _ in range(33)]
    pose[11],pose[12] = SimpleNamespace(x=.25,y=.5,visibility=1.),SimpleNamespace(x=.75,y=.5,visibility=1.)
    hand = [SimpleNamespace(x=.25+i*.005,y=.3) for i in range(21)]
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pose),
        left_hand_landmarks=SimpleNamespace(landmark=hand),right_hand_landmarks=None)


class OverlayObserver:
    instances = []
    def __init__(self, config):
        self.calls = 0
        self.closed = False
        self.instances.append(self)
    def observe(self, packet):
        self.calls += 1
        return ActivityObservation(packet.timestamp,True,True,.8,True)
    def observe_landmarks(self, packet):
        return self.observe(packet),raw_landmarks()
    def close(self):
        self.closed = True


def test_main_screen_has_independent_keypoint_display_switch():
    app = QApplication.instance() or QApplication([])
    window = SignLanguageApp()
    try:
        assert hasattr(window,'keypoints_check'), 'Recognition keypoint display switch is missing'
        assert window.keypoints_check.text() == '显示关键点'
        assert not window.keypoints_check.isChecked()
        window.keypoints_check.click()
        assert window.session.show_keypoints
        assert not window.session.continuous and not window.session.automatic
    finally:
        window.close()
        app.processEvents()


def test_keypoint_display_shares_activity_observer_and_clears_on_disable():
    OverlayObserver.instances.clear()
    source = ReplaySource(count=400)
    session = RecognitionSession(WindowModel(),observer_factory=OverlayObserver)
    try:
        assert callable(getattr(session,'set_show_keypoints',None)), 'Keypoint display API is missing'
        session.open_source(source,kind='camera')
        wait(lambda:session.source_active)
        session.enable_continuous()
        session.set_show_keypoints(True)
        source.gate.set()
        saved=[]
        def ready():
            pair=session.take_landmark_preview()
            if pair is not None: saved.append(pair)
            return bool(saved)
        wait(ready)
        packet,frame=saved[-1]
        assert (packet.source_id,packet.frame_index)==(frame.source_id,frame.frame_index)
        assert frame.source_epoch==session.epoch
        assert len(frame.image_xy)==75 and frame.point_observed[33]
        assert len(OverlayObserver.instances)==1
        assert np.max(packet.image)==0
        session.set_show_keypoints(False)
        assert session.take_landmark_preview() is None
        assert session.continuous
    finally:
        source.gate.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())


def test_preview_overlay_uses_mirror_and_clears_with_placeholder():
    from slbridge.preview_landmarks import PreviewLandmarks
    app=QApplication.instance() or QApplication([])
    widget=PreviewWidget()
    widget.resize(400,240)
    packet=FramePacket(np.zeros((20,40,3),np.uint8),0.,0,'sample')
    frame=PreviewLandmarks.from_result(packet,raw_landmarks(),source_epoch=1)
    image=QImage(40,20,QImage.Format.Format_RGB888)
    image.fill(0)
    try:
        widget.show()
        widget.set_image(image,landmarks=frame,mirrored=False)
        app.processEvents()
        left=widget.grab().toImage()
        def pixel(image,x,y):
            return image.pixelColor(round(x*image.devicePixelRatio()),round(y*image.devicePixelRatio()))
        assert pixel(left,100,80).blue()>100
        assert pixel(left,300,80).blue()<30
        widget.set_image(image,landmarks=frame,mirrored=True)
        app.processEvents()
        right=widget.grab().toImage()
        assert pixel(right,300,80).blue()>100
        assert pixel(right,100,80).blue()<30
        widget.set_placeholder('closed','')
        assert widget.landmarks is None
    finally:
        widget.close()


def pump(predicate,seconds=4):
    deadline=time.perf_counter()+seconds
    while time.perf_counter()<deadline:
        QApplication.instance().processEvents()
        if predicate(): return
        time.sleep(.005)
    raise AssertionError('Keypoint UI did not reach expected state')


def test_display_switch_during_recording_keeps_raw_recording_and_result():
    app=QApplication.instance() or QApplication([])
    model=WindowModel()
    source=ReplaySource(count=400)
    session=RecognitionSession(model,observer_factory=OverlayObserver)
    window=SignLanguageApp(session=session)
    try:
        window.show()
        session.open_source(source,kind='camera')
        source.gate.set()
        pump(lambda:session.source_active and session.latest_frame is not None)
        window.start_recording()
        started=window._record_started
        window.keypoints_check.click()
        pump(lambda:window.preview.landmarks is not None)
        assert session.state=='RECORDING' and window._record_started==started
        window.keypoints_check.click()
        assert window.preview.landmarks is None and session.state=='RECORDING'
        pump(lambda:len(session.buffer._frames)>=3)
        window.finish_recording()
        assert not window._error_detail,window._error_detail
        pump(lambda:window._last_result is not None)
        assert len(model.clips)==1
        assert all(np.max(frame.image)==0 for frame in model.clips[0].frames)
        assert not session.captions.snapshot()['events']
        window.stop_source()
        window.mirror_check.click()
        assert window.preview.image is None and window.preview.landmarks is None
    finally:
        source.gate.set()
        window.close()
        app.processEvents()


@pytest.mark.parametrize('action',['stop','pause','disable'])
def test_inflight_landmarks_do_not_survive_invalidation(action):
    entered,release=threading.Event(),threading.Event()
    class Blocked(OverlayObserver):
        def observe_landmarks(self,packet):
            entered.set()
            assert release.wait(3)
            return super().observe_landmarks(packet)
    source=ReplaySource(count=400)
    session=RecognitionSession(observer_factory=Blocked)
    try:
        session.open_source(source,kind='camera')
        wait(lambda:session.source_active)
        session.set_show_keypoints(True)
        source.gate.set()
        assert entered.wait(2)
        source.gate.clear()
        if action=='stop': session.stop_source()
        elif action=='pause': session.pause_source(True)
        else: session.set_show_keypoints(False)
        release.set()
        time.sleep(.1)
        assert session.take_landmark_preview() is None
    finally:
        release.set()
        source.gate.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())


def test_display_off_does_not_start_detection_and_stale_detection_is_not_published():
    OverlayObserver.instances.clear()
    entered,release=threading.Event(),threading.Event()
    class Slow(OverlayObserver):
        def observe_landmarks(self,packet):
            entered.set()
            assert release.wait(3)
            return super().observe_landmarks(packet)
    source=ReplaySource(count=400)
    session=RecognitionSession(observer_factory=Slow)
    try:
        session.open_source(source,kind='camera')
        source.gate.set()
        wait(lambda:source.index>10)
        assert not OverlayObserver.instances
        session.set_show_keypoints(True)
        assert entered.wait(2)
        source.gate.clear()
        time.sleep(.6)
        release.set()
        time.sleep(.1)
        assert session.take_landmark_preview() is None
        assert session.source_active and session.take_preview() is not None
    finally:
        release.set()
        source.gate.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())


def test_overlay_failure_does_not_abort_manual_recording():
    class Broken(OverlayObserver):
        def observe_landmarks(self,packet): raise RuntimeError('detector fixture failed')
    source=ReplaySource(count=400)
    session=RecognitionSession(WindowModel(),observer_factory=Broken)
    try:
        session.open_source(source,kind='camera')
        source.gate.set()
        wait(lambda:session.latest_frame is not None)
        session.start_recording()
        session.set_show_keypoints(True)
        wait(lambda:not session.show_keypoints)
        assert session.state=='RECORDING' and session.source_active
        assert any(event.kind=='keypoints_error' for event in session.poll())
    finally:
        source.gate.set()
        session.close()


def test_overlay_masks_missing_or_invalid_landmarks():
    from slbridge.preview_landmarks import PreviewLandmarks
    packet=FramePacket(np.zeros((20,40,3),np.uint8),0.,0,'sample')
    raw=raw_landmarks()
    raw.left_hand_landmarks.landmark[0].x=float('nan')
    frame=PreviewLandmarks.from_result(packet,raw,source_epoch=3)
    assert not frame.point_observed[0] and not frame.point_observed[33]
    assert not any(frame.point_observed[54:])
    assert frame.image_xy[33]==(0.,0.) and frame.point_observed[34]


def test_rapid_off_on_does_not_publish_the_old_detection():
    entered,release,source_end=threading.Event(),threading.Event(),threading.Event()
    done=threading.Event()
    class SingleSource:
        reads=0
        def open(self): return self
        def read(self):
            self.reads+=1
            if self.reads==1: return FramePacket(np.zeros((20,40,3),np.uint8),1.,0,'single')
            source_end.wait(3)
            return None
        def close(self): pass
    class Blocked(OverlayObserver):
        def observe_landmarks(self,packet):
            entered.set()
            assert release.wait(3)
            response=super().observe_landmarks(packet)
            done.set()
            return response
    session=RecognitionSession(observer_factory=Blocked)
    try:
        session.set_show_keypoints(True)
        session.open_source(SingleSource(),kind='camera')
        assert entered.wait(2)
        session.set_show_keypoints(False)
        session.set_show_keypoints(True)
        release.set()
        assert done.wait(2)
        time.sleep(.05)
        assert session.take_landmark_preview() is None
    finally:
        release.set()
        source_end.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())


def test_rapid_off_on_ignores_an_old_display_failure():
    entered,release,source_end=threading.Event(),threading.Event(),threading.Event()
    class SingleSource:
        reads=0
        def open(self): return self
        def read(self):
            self.reads+=1
            if self.reads==1: return FramePacket(np.zeros((20,40,3),np.uint8),1.,0,'single')
            source_end.wait(3)
            return None
        def close(self): pass
    class Broken(OverlayObserver):
        def observe_landmarks(self,packet):
            entered.set()
            assert release.wait(3)
            raise RuntimeError('obsolete display failure')
    session=RecognitionSession(observer_factory=Broken)
    try:
        session.set_show_keypoints(True)
        session.open_source(SingleSource(),kind='camera')
        assert entered.wait(2)
        session.set_show_keypoints(False)
        session.set_show_keypoints(True)
        release.set()
        time.sleep(.1)
        assert session.show_keypoints
        assert not any(event.kind=='keypoints_error' for event in session.poll())
    finally:
        release.set()
        source_end.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())


@pytest.mark.parametrize('action',['toggle','source','continuous'])
def test_old_observer_close_error_cannot_disable_a_new_selection(action):
    closing,release=threading.Event(),threading.Event()
    class BrokenClose(OverlayObserver):
        def close(self):
            closing.set()
            assert release.wait(3)
            self.closed=True
            raise RuntimeError('obsolete close failure')
    first,second=ReplaySource(count=400),ReplaySource(count=400)
    session=RecognitionSession(WindowModel(),observer_factory=BrokenClose)
    try:
        session.open_source(first,kind='camera')
        session.set_show_keypoints(True)
        first.gate.set()
        wait(lambda:session._landmark_preview is not None)
        session.set_show_keypoints(False)
        assert closing.wait(2)
        if action=='source':
            session.open_source(second,kind='camera')
            second.gate.set()
        elif action=='continuous':
            session.enable_continuous()
        session.set_show_keypoints(True)
        release.set()
        time.sleep(.1)
        assert session.show_keypoints
        assert not any(event.kind in ('error','keypoints_error') for event in session.poll())
        if action=='continuous': assert session.continuous
    finally:
        release.set()
        first.gate.set()
        second.gate.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())


def test_toggle_between_cleanup_decision_and_native_close_is_preserved():
    scheduled,release=threading.Event(),threading.Event()
    class BrokenClose(OverlayObserver):
        def close(self):
            self.closed=True
            raise RuntimeError('retired cleanup failure')
    class DelayedCleanupSession(RecognitionSession):
        def _close_observer(self,*args,**kwargs):
            if not scheduled.is_set():
                scheduled.set()
                assert release.wait(3)
            return super()._close_observer(*args,**kwargs)
    source=ReplaySource(count=400)
    session=DelayedCleanupSession(observer_factory=BrokenClose)
    try:
        session.open_source(source,kind='camera')
        session.set_show_keypoints(True)
        source.gate.set()
        wait(lambda:session._landmark_preview is not None)
        session.set_show_keypoints(False)
        assert scheduled.wait(2)
        session.set_show_keypoints(True)
        release.set()
        time.sleep(.1)
        assert session.show_keypoints
        assert not any(event.kind=='keypoints_error' for event in session.poll())
    finally:
        release.set()
        source.gate.set()
        session.close()
        wait(lambda:not session._capture_thread.is_alive() and not session._observer_worker.is_alive())
