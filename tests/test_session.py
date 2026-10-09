from types import SimpleNamespace
import threading
import time
import numpy as np
import pytest
from slbridge.session import RecognitionSession
from slbridge.sources import FramePacket, ClipBuffer, SourceError


def wait_for(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('background operation did not complete')


class FakeSource:
    def __init__(self, *, fail=False, eof=None):
        self.closed = False
        self.opened = False
        self.fail = fail
        self.eof = eof
        self.index = 0
    def open(self):
        self.opened = True
        return self
    def read(self):
        time.sleep(.005)
        if self.fail:
            raise SourceError('camera disconnected')
        if self.closed or self.eof is not None and self.index >= self.eof:
            return None
        i = self.index
        self.index += 1
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[:, 0, 0] = 200
        return FramePacket(image, i * .01, i, 'fake')
    def close(self):
        self.closed = True


class FakeRecognizer:
    def __init__(self, block=False, error=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.error = error
        self.calls = 0
        self.clip = None
        self.path = None
        if not block:
            self.release.set()
    def _run(self):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(3)
        if self.error:
            raise ValueError('model failed')
        return SimpleNamespace(status='recognized')
    def recognize(self, path):
        self.path = path
        return self._run()
    def recognize_clip(self, clip, **kwargs):
        self.clip = clip
        return self._run()


def test_cancel_discards_late_result_and_keeps_single_worker():
    model = FakeRecognizer(block=True)
    session = RecognitionSession(model)
    source = FakeSource(eof=2)
    try:
        session.open_source(source, kind='video', video_path='whole.mp4')
        first = session.recognize_file()
        assert model.entered.wait(2)
        session.cancel()
        assert session.request_id > first
        for _ in range(8):
            with pytest.raises(RuntimeError, match='模型忙'):
                session.recognize_file()
        assert model.calls == 1 and session._tasks.qsize() == 0
        model.release.set()
        wait_for(lambda: not session.busy)
        assert not any(e.kind == 'result' for e in session.poll())
        session.recognize_file()
        wait_for(lambda: not session.busy)
        assert any(e.kind == 'result' for e in session.poll())
        assert str(model.path) == 'whole.mp4'
    finally:
        model.release.set()
        session.close()
    wait_for(lambda: source.closed)


def test_switch_source_discards_late_error_and_releases_old_device():
    model = FakeRecognizer(block=True, error=True)
    session = RecognitionSession(model)
    old, new = FakeSource(), FakeSource()
    try:
        session.open_source(old, kind='video', video_path='a.mp4')
        wait_for(lambda: old.opened)
        session.recognize_file()
        assert model.entered.wait(2)
        session.open_source(new, kind='camera')
        wait_for(lambda: old.closed)
        model.release.set()
        wait_for(lambda: not session.busy)
        assert session.state == 'PREVIEW'
        assert not any(e.kind in {'error', 'result'} for e in session.poll())
    finally:
        session.close()
    wait_for(lambda: new.closed)


def test_recording_keeps_original_frames_and_recovers_after_error():
    model = FakeRecognizer()
    session = RecognitionSession(model)
    source = FakeSource()
    try:
        session.open_source(source, kind='camera')
        wait_for(lambda: source.index >= 2)
        session.start_recording()
        wait_for(lambda: source.index >= 8)
        session.finish_recording()
        wait_for(lambda: not session.busy)
        assert model.clip.frames[0].timestamp == 0
        assert model.clip.frames[0].image[0, 0, 0] == 200
        assert model.clip.frames[0].image[0, -1, 0] == 0
        session.start_recording()
        session.cancel()
        assert session.state == 'PREVIEW'
        with pytest.raises(ValueError):
            session.buffer.finish()
    finally:
        session.close()


def test_disconnect_clears_buffer_and_preview_and_releases():
    session = RecognitionSession(FakeRecognizer())
    source = FakeSource(fail=True)
    try:
        session.open_source(source, kind='camera')
        wait_for(lambda: session.state == 'ERROR')
        wait_for(lambda: source.closed)
        assert session.take_preview() is None
        assert any('disconnected' in e.value for e in session.poll() if e.kind == 'error')
        with pytest.raises(ValueError):
            session.buffer.finish()
        replacement = FakeSource()
        session.open_source(replacement, kind='camera')
        wait_for(lambda: replacement.index > 1)
        assert session.state == 'PREVIEW'
    finally:
        session.close()


def test_model_errors_recover_and_close_is_idempotent():
    model = FakeRecognizer(error=True)
    session = RecognitionSession(model)
    source = FakeSource(eof=2)
    session.open_source(source, kind='video', video_path='a.mp4')
    session.recognize_file()
    wait_for(lambda: not session.busy)
    assert session.state == 'ERROR'
    assert any(e.kind == 'error' for e in session.poll())
    model.error = False
    session.recognize_file()
    wait_for(lambda: not session.busy)
    assert session.state == 'RESULT'
    session.close()
    session.close()
    wait_for(lambda: source.closed)
    wait_for(lambda: not session._worker.is_alive())


def test_preview_latest_slot_and_eof_does_not_change_full_file():
    session = RecognitionSession(FakeRecognizer())
    source = FakeSource(eof=4)
    try:
        session.open_source(source, kind='video', video_path='complete.mp4')
        wait_for(lambda: source.closed)
        assert session.take_preview().frame_index == 3
        assert session.take_preview() is None
        assert any(e.kind == 'eof' for e in session.poll())
        session.paused = True
        session.recognize_file()
        wait_for(lambda: not session.busy)
        assert str(session.recognizer.path) == 'complete.mp4'
    finally:
        session.close()


def test_incompatible_model_rejected_without_replacing_previous():
    original = FakeRecognizer()
    session = RecognitionSession(original)
    bad = SimpleNamespace(bundle=SimpleNamespace(manifest=SimpleNamespace(input_spec=SimpleNamespace(representation_id='hand63_v1'), capabilities=('clip',))))
    try:
        session.load_model('bad', loader=lambda _: bad)
        wait_for(lambda: not session.busy)
        assert session.recognizer is original
        assert any('mediapipe225_rgb_v1' in e.value for e in session.poll() if e.kind == 'error')
    finally:
        session.close()


def test_switch_during_recording_preserves_clip_buffer_and_request():
    session = RecognitionSession(FakeRecognizer())
    source = FakeSource()
    try:
        session.open_source(source, kind='camera')
        wait_for(lambda: session.source_active)
        session.start_recording()
        wait_for(lambda: source.index >= 8)
        request = session.request_id
        with pytest.raises(RuntimeError, match='结束或取消当前录制'):
            session.load_model('unused', loader=lambda _: pytest.fail('must not load'))
        assert session.state == 'RECORDING' and session.request_id == request
        session.finish_recording()
        wait_for(lambda: not session.busy)
        assert len(session.recognizer.clip.frames) >= 2
    finally:
        session.close()


def test_cancel_during_warmup_discards_candidate_and_keeps_active():
    original = FakeRecognizer()
    session = RecognitionSession(original)
    entered, release = threading.Event(), threading.Event()
    closed = []
    def warmup():
        entered.set()
        assert release.wait(3)
    candidate = SimpleNamespace(
        bundle=SimpleNamespace(manifest=SimpleNamespace(
            input_spec=SimpleNamespace(representation_id='mediapipe225_rgb_v1'), capabilities=('clip',))),
        warmup=warmup, close=lambda: closed.append(True))
    try:
        session.load_model('candidate', loader=lambda _: candidate)
        assert entered.wait(2)
        session.cancel()
        release.set()
        wait_for(lambda: not session.busy)
        assert session.recognizer is original and session.model_manager.generation == 1
        assert closed == [True]
        assert not any(e.kind == 'model' for e in session.poll())
    finally:
        release.set()
        session.close()


def test_runtime_closed_before_activation_does_not_kill_model_worker(monkeypatch):
    session = RecognitionSession()
    model = SimpleNamespace(bundle=SimpleNamespace(manifest=SimpleNamespace(
        input_spec=SimpleNamespace(representation_id='mediapipe225_rgb_v1'), capabilities=('clip',))))
    original_prepare = session.model_manager.prepare
    def prepare_then_close(*args, **kwargs):
        candidate = original_prepare(*args, **kwargs)
        session.model_manager.close()
        return candidate
    monkeypatch.setattr(session.model_manager, 'prepare', prepare_then_close)
    try:
        session.load_model('fake', loader=lambda _: model)
        wait_for(lambda: not session.busy)
        assert session._worker.is_alive() and session.state == 'ERROR'
        session.load_model('fake', loader=lambda _: model)
        wait_for(lambda: not session.busy)
        assert session._worker.is_alive()
    finally:
        session.close()


def test_recording_limit_aborts_and_releases_camera():
    session = RecognitionSession(FakeRecognizer(), buffer=ClipBuffer(max_frames=2))
    source = FakeSource()
    try:
        session.open_source(source, kind='camera')
        wait_for(lambda: session.source_active)
        session.start_recording()
        wait_for(lambda: session.state == 'ERROR')
        wait_for(lambda: source.closed)
        with pytest.raises(ValueError):
            session.buffer.finish()
        assert any(e.kind == 'error' for e in session.poll())
    finally:
        session.close()


def test_capture_has_one_owner_and_rapid_switches_remain_bounded():
    entered, release = threading.Event(), threading.Event()
    operations = []
    class BlockingSource(FakeSource):
        def open(self):
            operations.append(('open', threading.get_ident()))
            return super().open()
        def read(self):
            operations.append(('read-enter', threading.get_ident()))
            entered.set()
            assert release.wait(3)
            operations.append(('read-exit', threading.get_ident()))
            return None
        def close(self):
            operations.append(('close', threading.get_ident()))
            super().close()
    session = RecognitionSession(FakeRecognizer())
    first = BlockingSource()
    replacements = [FakeSource(eof=1) for _ in range(50)]
    try:
        session.open_source(first, kind='camera')
        assert entered.wait(2)
        owner = session._capture_thread
        for replacement in replacements:
            session.open_source(replacement, kind='camera')
            assert session._capture_thread is owner
        assert not first.closed
        assert all(not s.opened for s in replacements)
        release.set()
        wait_for(lambda: replacements[-1].closed)
        assert first.closed
        assert all(not s.opened for s in replacements[:-1])
        assert len({thread_id for _, thread_id in operations}) == 1
        assert [name for name, _ in operations] == ['open', 'read-enter', 'read-exit', 'close']
    finally:
        release.set()
        session.close()
    wait_for(lambda: not session._capture_thread.is_alive())


def test_close_during_capture_does_not_release_on_ui_thread():
    entered, release = threading.Event(), threading.Event()
    class BlockingSource(FakeSource):
        def read(self):
            entered.set()
            assert release.wait(3)
            assert not self.closed
            return None
    source = BlockingSource()
    session = RecognitionSession(FakeRecognizer())
    session.open_source(source, kind='camera')
    assert entered.wait(2)
    start = time.monotonic()
    session.close()
    assert time.monotonic() - start < .2
    assert not source.closed
    release.set()
    wait_for(lambda: source.closed and not session._capture_thread.is_alive())
