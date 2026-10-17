import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.keypoint_session import KeypointSession
from slbridge.sources import FramePacket


def wait(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("keypoint session did not reach expected behavior")


class Source:
    def __init__(self, name="sample", interval=.01, count=None):
        self.name, self.interval, self.count = name, interval, count
        self.index = 0
        self.closed = False
        self.owners = []

    def open(self):
        self.owners.append(threading.get_ident())

    def read(self):
        self.owners.append(threading.get_ident())
        time.sleep(self.interval)
        if self.count is not None and self.index >= self.count:
            return None
        n = self.index
        self.index += 1
        return FramePacket(np.zeros((16, 24, 3), np.uint8), n * .025, n, self.name)

    def close(self):
        self.owners.append(threading.get_ident())
        self.closed = True


class Graph:
    def process(self, image):
        point = lambda x, y: SimpleNamespace(x=x, y=y, visibility=1.)
        pose = [point(.5, .5) for _ in range(33)]
        pose[11], pose[12] = point(.3, .3), point(.7, .3)
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pose),
                               left_hand_landmarks=None, right_hand_landmarks=None)

    def close(self):
        pass


class Factory:
    def __init__(self, block_after=None):
        self.instances = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_after = block_after
        if block_after is None:
            self.release.set()

    def __call__(self, **kwargs):
        from slbridge.keypoints import KeypointExtractor
        base = KeypointExtractor(holistic_factory=lambda **options: Graph(), **kwargs)
        factory = self

        class Extractor:
            calls = 0
            closed = False

            def process(self, packet, **options):
                self.calls += 1
                if factory.block_after is not None and len(factory.instances) == 1 and self.calls > factory.block_after:
                    factory.entered.set()
                    assert factory.release.wait(3)
                return base.process(packet, **options)

            def metadata(self):
                return base.metadata()

            def reset(self, **options):
                base.reset(**options)

            def close(self):
                self.closed = True
                base.close()

        result = Extractor()
        self.instances.append(result)
        return result


def close(session, factory=None):
    if factory:
        factory.release.set()
    session.close()
    wait(lambda: not session._capture_thread.is_alive() and not session._extract_thread.is_alive())


def test_capture_without_model_records_and_reads_same_processed_observations(tmp_path):
    from slbridge.keypoint_recording import KeypointRecording
    factory, source = Factory(), Source()
    session = KeypointSession(extractor_factory=factory)
    path = tmp_path / "captured.slkp.jsonl"
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.status()["processed_frames"] >= 2)
        session.start_recording(path, metadata={"label": "example", "review_status": "approved"})
        wait(lambda: session.status()["recorded_frames"] >= 4)
        saved = session.finish_recording()
        recording = KeypointRecording(saved)
        frames = list(recording.iter_frames())
        assert len(frames) == session.status()["recorded_frames"]
        assert frames[0].source_id == "sample" and frames[-1].timestamp > frames[0].timestamp
        assert recording.metadata["annotations"]["review_status"] == "draft"
        assert recording.metadata["source_timing_mode"] == "application_monotonic"
        preview, points = session.take_view()
        assert preview.frame_index == points.frame_index and preview.source_id == points.source_id
        assert session.status()["receipt_age_ms"] >= session.status()["extraction_ms"] >= 0
    finally:
        close(session, factory)
    assert source.closed and len(set(source.owners)) == 1


def test_slow_detector_discards_old_frames_and_source_switch_discards_late_output():
    factory = Factory(block_after=0)
    old, new = Source("old", interval=.002), Source("new", interval=.02)
    session = KeypointSession(extractor_factory=factory)
    try:
        session.open_source(old, kind="camera")
        assert factory.entered.wait(2)
        wait(lambda: session.status()["dropped_frames"] > 3)
        assert session.status()["queue_size"] <= 2
        session.open_source(new, kind="camera")
        wait(lambda: old.closed and session.source_active)
        factory.release.set()
        wait(lambda: session.status()["processed_frames"] >= 2)
        _, points = session.take_view()
        assert points.source_id == "new" and points.source_epoch == session.epoch
        assert factory.instances[0].closed
        assert not any(e.kind == "error" for e in session.poll())
    finally:
        close(session, factory)


def test_reset_invalidates_inflight_output_and_preserves_recording_identity(tmp_path):
    from slbridge.keypoint_recording import KeypointRecording
    factory, source = Factory(block_after=1), Source(interval=.02)
    session = KeypointSession(extractor_factory=factory)
    path = tmp_path / "reset.slkp.jsonl"
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.status()["processed_frames"] == 1)
        session.start_recording(path)
        assert factory.entered.wait(2)
        session.reset_tracking()
        epoch = session.epoch
        factory.release.set()
        wait(lambda: session.status()["recorded_frames"] >= 2)
        frames = list(KeypointRecording(session.finish_recording()).iter_frames())
        assert all(f.source_epoch == epoch and f.tracking_epoch >= 1 for f in frames)
        assert frames[0].reset_reason is not None and frames[0].motion_valid is False
    finally:
        close(session, factory)


def test_pause_freezes_output_and_stopping_cancels_partial_recording(tmp_path):
    factory, source = Factory(), Source(interval=.02)
    session = KeypointSession(extractor_factory=factory)
    path = tmp_path / "cancelled.slkp.jsonl"
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.status()["processed_frames"] >= 2)
        session.start_recording(path)
        wait(lambda: session.status()["recorded_frames"] >= 2)
        session.pause()
        count = session.status()["processed_frames"]
        time.sleep(.08)
        assert session.status()["processed_frames"] == count
        session.resume()
        wait(lambda: session.status()["processed_frames"] > count)
        session.stop_source()
        assert not path.exists() and not session.status()["recording"]
        wait(lambda: source.closed)
        assert not list(tmp_path.glob("*.partial"))
    finally:
        close(session, factory)


def test_archive_replay_uses_saved_points_without_creating_an_extractor(tmp_path):
    from slbridge.keypoint_recording import KeypointRecording
    factory, source = Factory(), Source(interval=.02)
    capture = KeypointSession(extractor_factory=factory)
    path = tmp_path / "replay.slkp.jsonl"
    try:
        capture.open_source(source, kind="camera")
        wait(lambda: capture.status()["processed_frames"] >= 1)
        capture.start_recording(path)
        wait(lambda: capture.status()["recorded_frames"] >= 3)
        capture.finish_recording()
    finally:
        close(capture, factory)
    original = list(KeypointRecording(path).iter_frames())
    def forbidden(**kwargs):
        raise AssertionError("replay must not construct a detector")
    replay = KeypointSession(extractor_factory=forbidden)
    try:
        replay.open_recording(path)
        wait(lambda: replay.status()["processed_frames"] == len(original))
        preview, frame = replay.take_view()
        assert preview is None and frame.to_dict() == original[-1].to_dict()
        assert replay.status()["receipt_age_ms"] is None and replay.status()["extraction_ms"] is None
        with pytest.raises(RuntimeError):
            replay.start_recording(tmp_path / "copy.slkp.jsonl")
    finally:
        close(replay)


@pytest.mark.parametrize("pause_seconds", [.25, 1.1])
def test_pause_inside_long_video_wait_preserves_frame_and_shifts_playback_clock(tmp_path, pause_seconds):
    from slbridge.keypoint_recording import KeypointRecording
    class SparseVideo(Source):
        timestamp_mode = "fps_fallback"
        def read(self):
            packet = super().read()
            return (None if packet is None else
                    FramePacket(packet.image, float(packet.frame_index), packet.frame_index, packet.source_id))
    factory, source = Factory(), SparseVideo(interval=.001, count=2)
    session = KeypointSession(extractor_factory=factory)
    path = tmp_path / "sparse.slkp.jsonl"
    try:
        session.open_source(source, kind="video")
        wait(lambda: session.status()["processed_frames"] == 1)
        first_at = time.perf_counter()
        session.start_recording(path)
        time.sleep(.12)
        session.pause()
        time.sleep(pause_seconds)
        assert session.status()["processed_frames"] == 1
        session.resume()
        wait(lambda: session.status()["processed_frames"] == 2)
        assert time.perf_counter() - first_at >= 1. + pause_seconds - .08
        recording = KeypointRecording(session.finish_recording())
        frames = list(recording.iter_frames())
        assert [f.frame_index for f in frames] == [1]
        assert frames[0].timestamp == 1.
        assert recording.metadata["source_timing_mode"] == "fps_fallback"
    finally:
        close(session, factory)


def test_tracking_reset_cannot_hide_a_backwards_source_clock():
    gate = threading.Event()
    class BadClock(Source):
        def read(self):
            if self.index:
                assert gate.wait(3)
            packet = super().read()
            return FramePacket(packet.image, .5 if packet.frame_index == 0 else .4,
                               packet.frame_index, packet.source_id)
    factory, source = Factory(), BadClock(interval=.001)
    session = KeypointSession(extractor_factory=factory)
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.status()["processed_frames"] == 1)
        session.reset_tracking()
        gate.set()
        wait(lambda: bool(session.status()["error"]))
        assert "timestamps must be strictly increasing" in session.status()["error"]
        wait(lambda: source.closed)
        session.open_source(Source("recovered", interval=.005), kind="camera")
        wait(lambda: session.status()["processed_frames"] >= 2)
        assert session.take_view()[1].source_id == "recovered"
    finally:
        gate.set()
        close(session, factory)


def test_recording_write_and_cleanup_errors_do_not_kill_capture_workers():
    class BrokenRecorder:
        def append(self, frame):
            raise OSError("disk full")
        def abort(self):
            raise PermissionError("partial locked")
    factory, source = Factory(), Source(interval=.01)
    session = KeypointSession(extractor_factory=factory)
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.status()["processed_frames"] >= 1)
        with session._lock:
            session._recorder = BrokenRecorder()
        wait(lambda: "disk full" in session.status()["error"])
        assert "partial locked" in session.status()["error"]
        assert not session.status()["recording"]
        count = session.status()["processed_frames"]
        wait(lambda: session.status()["processed_frames"] > count)
        session.open_source(Source("recovered", interval=.01), kind="camera")
        wait(lambda: session.status()["processed_frames"] >= 2)
        assert session.take_view()[1].source_id == "recovered"
        assert session.status()["error"] == ""
    finally:
        close(session, factory)


def test_published_recording_cleanup_warning_preserves_saved_archive(tmp_path, monkeypatch):
    from pathlib import Path
    from slbridge.keypoint_recording import KeypointRecording
    factory, source = Factory(), Source(interval=.01)
    session = KeypointSession(extractor_factory=factory)
    path = tmp_path / "published.slkp.jsonl"
    partial = None
    original_unlink = Path.unlink
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.status()["processed_frames"] >= 1)
        session.start_recording(path)
        wait(lambda: session.status()["recorded_frames"] >= 3)
        session.pause()
        partial = session._recorder._partial
        def locked(candidate, *args, **kwargs):
            if candidate == partial:
                raise PermissionError("partial locked after publish")
            return original_unlink(candidate, *args, **kwargs)
        monkeypatch.setattr(Path, "unlink", locked)
        assert session.finish_recording() == path
        assert KeypointRecording(path).summary()["count"] >= 3
        kinds = [e.kind for e in session.poll()]
        assert "recording_saved" in kinds and "recording_cleanup_warning" in kinds
        assert not session.status()["recording"] and session.status()["error"] == ""
        assert partial.exists()
    finally:
        monkeypatch.setattr(Path, "unlink", original_unlink)
        close(session, factory)
        if partial is not None:
            partial.unlink(missing_ok=True)
