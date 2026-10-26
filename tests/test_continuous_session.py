import threading
import time

import pytest

from slbridge.segmentation import ActivityObservation
from slbridge.session import RecognitionSession
from test_automatic_session import ClipModel, ReplaySource


def wait(predicate, seconds=4):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Continuous recognition did not reach expected state')


class MotionObserver:
    visible = True

    def __init__(self, config):
        self.closed = False

    def observe(self, packet):
        return ActivityObservation(packet.timestamp, self.visible, self.visible, .8, self.visible)

    def close(self):
        self.closed = True


class WindowModel(ClipModel):
    def __init__(self, block=False):
        super().__init__(block=block)
        self.clips = []

    def recognize_clip_profiled(self, clip, *, sample_id):
        self.clips.append(clip)
        return super().recognize_clip_profiled(clip, sample_id=sample_id)


def setup(model=None, *, kind='camera', observer=MotionObserver, **config):
    from slbridge.continuous import ContinuousConfig
    MotionObserver.visible = True
    source = ReplaySource(count=300)
    session = RecognitionSession(model or WindowModel(), observer_factory=observer)
    session.open_source(source, kind=kind)
    wait(lambda: session.source_active)
    defaults = dict(minimum_seconds=.1, window_seconds=.35, update_interval_seconds=.1,
                    max_result_age_seconds=2.)
    defaults.update(config)
    session.enable_continuous(ContinuousConfig(**defaults))
    return session, source


def stop(session, source):
    source.gate.set()
    if hasattr(session.recognizer, 'release'):
        session.recognizer.release.set()
    session.close()
    wait(lambda: not session._worker.is_alive() and not session._observer_worker.is_alive()
         and not session._capture_thread.is_alive())


def test_continuous_api_does_not_require_pause_mode():
    session = RecognitionSession()
    try:
        assert callable(getattr(session, 'enable_continuous', None)), 'Continuous stream recognition is missing'
    finally:
        session.close()


def test_motion_only_video_stream_returns_updates_without_committing_captions():
    session, source = setup()
    try:
        source.gate.set()
        wait(lambda: session.continuous_status()['completed_windows'] >= 3)
        events = session.poll()
        assert any(e.kind == 'continuous_result' for e in events)
        assert len([e for e in events if e.kind == 'continuous_result']) == 1
        assert not session.automatic
        assert not session.captions.snapshot()['events']
        assert all(.1 - 1e-9 <= c.frames[-1].timestamp <= .35 + 1e-9 for c in session.recognizer.clips)
    finally:
        stop(session, source)


def test_slow_inference_discards_old_result_and_next_call_uses_latest_window():
    model = WindowModel(block=True)
    session, source = setup(model, max_result_age_seconds=.3)
    try:
        source.gate.set()
        assert model.entered.wait(2)
        first_end = model.clips[0].frames[-1].frame_index
        wait(lambda: source.index > first_end + 30)
        assert len(model.clips) == 1
        assert session._tasks.qsize() == 0
        model.release.set()
        wait(lambda: len(model.clips) > 1)
        assert model.clips[1].frames[-1].frame_index > first_end + 20
        assert session.continuous_status()['stale_results'] >= 1
        assert session.continuous_status()['buffered_frames'] <= 8
    finally:
        model.release.set()
        stop(session, source)


@pytest.mark.parametrize('action', ['disable_continuous', 'cancel', 'stop_source', 'clear_history'])
def test_invalidation_discards_inflight_window(action):
    model = WindowModel(block=True)
    session, source = setup(model)
    try:
        source.gate.set()
        assert model.entered.wait(2)
        source.count = source.index
        getattr(session, action)()
        model.release.set()
        wait(lambda: not session.busy)
        assert not any(e.kind == 'continuous_result' for e in session.poll())
        assert not session.captions.snapshot()['events']
    finally:
        model.release.set()
        stop(session, source)


def test_quality_loss_clears_candidate_and_does_not_infer_without_hands():
    session, source = setup()
    try:
        source.gate.set()
        wait(lambda: session.continuous_status()['completed_windows'] > 0)
        MotionObserver.visible = False
        wait(lambda: not session.continuous_status()['quality_ok'])
        calls = len(session.recognizer.clips)
        time.sleep(.06)
        assert len(session.recognizer.clips) <= calls + 1
        assert any(e.kind == 'continuous_clear' for e in session.poll())
    finally:
        MotionObserver.visible = True
        stop(session, source)


def test_automatic_and_continuous_modes_are_exclusive():
    session, source = setup()
    try:
        assert session.continuous and not session.automatic
        session.enable_automatic()
        assert session.automatic and not session.continuous
        session.disable_automatic()
        session.enable_continuous()
        assert session.continuous and not session.automatic
    finally:
        stop(session, source)


def test_pause_invalidates_inflight_window_and_resume_starts_fresh():
    model = WindowModel(block=True)
    session, source = setup(model, kind='video')
    try:
        source.gate.set()
        assert model.entered.wait(2)
        session.pause_source(True)
        model.release.set()
        wait(lambda: not session.busy)
        assert not any(e.kind == 'continuous_result' for e in session.poll())
        assert session.continuous
        session.pause_source(False)
        wait(lambda: session.continuous_status()['completed_windows'] > 0)
    finally:
        model.release.set()
        stop(session, source)


def test_observer_failure_stops_recognition_with_a_recognition_error():
    class BrokenObserver(MotionObserver):
        def observe(self, packet):
            raise RuntimeError('observation failed')
    session, source = setup(observer=BrokenObserver)
    try:
        source.gate.set()
        wait(lambda: not session.continuous)
        errors = [e for e in session.poll() if e.kind == 'error']
        assert errors and errors[-1].task_kind == 'continuous'
        assert session.source is source and session.source_active
    finally:
        stop(session, source)


def test_source_gap_invalidates_inflight_result_before_new_window_is_ready():
    from dataclasses import replace
    model = WindowModel(block=True)
    session, source = setup(model)
    original_read = source.read
    jumped = threading.Event()
    def read():
        packet = original_read()
        return replace(packet, timestamp=packet.timestamp+1.) if packet is not None and jumped.is_set() else packet
    source.read = read
    try:
        source.gate.set()
        assert model.entered.wait(2)
        token = session._continuous_token
        jumped.set()
        wait(lambda: token.cancelled)
        source.count = source.index
        model.release.set()
        wait(lambda: not session.busy)
        assert not any(e.kind == 'continuous_result' for e in session.poll())
    finally:
        model.release.set()
        stop(session, source)


def test_eof_cancels_inflight_window_and_stops_continuous_mode():
    model = WindowModel(block=True)
    session, source = setup(model)
    try:
        source.gate.set()
        assert model.entered.wait(2)
        token = session._continuous_token
        source.count = source.index
        wait(lambda: source.closed)
        assert token.cancelled
        assert not session.continuous
        model.release.set()
        wait(lambda: not session.busy)
        assert not any(e.kind == 'continuous_result' for e in session.poll())
    finally:
        model.release.set()
        stop(session, source)
