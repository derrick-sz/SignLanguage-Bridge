from pathlib import Path

import cv2
import numpy as np
import pytest
import time

from slbridge.sources import (
    CameraSource, CapturedClip, ClipBuffer, ClipLimitError, FramePacket,
    SourceError, VideoSource,
)


class FakeCapture:
    def __init__(self, images, times=None, fps=25, opened=True, count=None):
        self.images = images
        self.times = times or [0] * len(images)
        self.fps = fps
        self.opened = opened
        self.count = len(images) if count is None else count
        self.position = 0
        self.releases = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.position >= len(self.images):
            return False, None
        result = self.images[self.position]
        self.position += 1
        return True, result

    def get(self, prop):
        return {cv2.CAP_PROP_POS_MSEC: self.times[max(0, self.position - 1)],
                cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count}.get(prop, 0)

    def release(self):
        self.releases += 1

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES and value == 0:
            self.position = 0
            return True
        return False


def picture(width=8, height=4):
    return np.tile(np.arange(width, dtype=np.uint8)[None, :, None], (height, 1, 3))


def packet(t=0, index=0, source="test", image=None):
    return FramePacket(picture() if image is None else image, t, index, source)


def test_video_media_time_eof_and_idempotent_release():
    cap = FakeCapture([picture(), picture()], times=[500, 540])
    with VideoSource(Path("sample.mp4"), capture_factory=lambda path: cap) as source:
        first, second = source.read(), source.read()
        assert (first.timestamp, second.timestamp) == (0.5, 0.54)
        assert (first.frame_index, second.frame_index) == (0, 1)
        np.testing.assert_array_equal(first.image, picture())
        assert source.read() is None
        assert source.read() is None
    source.close()
    assert cap.releases == 1


def test_video_fps_fallback_is_explicit_and_unavailable_time_fails():
    cap = FakeCapture([picture()] * 3)
    with VideoSource("x", capture_factory=lambda path: cap) as source:
        assert [source.read().timestamp for _ in range(3)] == [0, 0.04, 0.08]
        assert source.timestamp_mode == "fps_fallback"
    cap = FakeCapture([picture()] * 2, fps=0)
    with pytest.raises(SourceError, match="time"):
        VideoSource("x", capture_factory=lambda path: cap).open()
    assert cap.releases == 1


@pytest.mark.parametrize("opened,count,reads", [(False, 0, 0), (True, 3, 2)])
def test_unopenable_or_truncated_video_releases(opened, count, reads):
    cap = FakeCapture([picture()] * 2, times=[0, 40], opened=opened, count=count)
    source = VideoSource("x", capture_factory=lambda path: cap)
    with pytest.raises(SourceError):
        source.open()
        for _ in range(reads):
            source.read()
        source.read()
    assert cap.releases == 1


def test_camera_monotonic_time_disconnect_and_release():
    cap = FakeCapture([picture()] * 2)
    ticks = iter([30.0, 30.125])
    source = CameraSource(capture_factory=lambda index: cap, clock=lambda: next(ticks)).open()
    assert [source.read().timestamp for _ in range(2)] == [30, 30.125]
    with pytest.raises(SourceError, match="disconnect"):
        source.read()
    source.close()
    assert cap.releases == 1


def test_default_camera_clock_uses_high_resolution_perf_counter(monkeypatch):
    cap = FakeCapture([picture()] * 2)
    ticks = iter([30.0, 30.000001])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(time, "monotonic", lambda: 30.0)
    with CameraSource(capture_factory=lambda index: cap) as source:
        assert [source.read().timestamp for _ in range(2)] == [30.0, 30.000001]
    assert cap.releases == 1


def test_fps_fallback_uses_whole_clip_index_time():
    cap = FakeCapture([picture()] * 3, times=[500, 500, 0])
    with VideoSource("x", capture_factory=lambda path: cap) as source:
        assert [source.read().timestamp for _ in range(3)] == [0, 0.04, 0.08]
        assert source.timestamp_mode == "fps_fallback"


@pytest.mark.parametrize("times", [[0, 20, 20], [0, 100, 100]])
def test_late_invalid_media_times_match_saved_file_fallback(times, monkeypatch, tmp_path):
    from slbridge.media import ExtractionConfig, decode_video

    images = [picture()] * 3
    cap = FakeCapture(images, times=times)
    with VideoSource("x", capture_factory=lambda path: cap) as source:
        timestamps = [source.read().timestamp for _ in range(3)]
        assert source.timestamp_mode == "fps_fallback"
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture(images, times=times))
    saved = decode_video(path, ExtractionConfig(frames=3))
    np.testing.assert_array_equal(timestamps, saved.timestamps)
    assert timestamps == [0, 0.04, 0.08]


@pytest.mark.parametrize("count", [2, 4, float("nan"), 3.5, 1, 1_000_001])
def test_invalid_declared_count_matches_saved_file_rejection(count, monkeypatch, tmp_path):
    from slbridge.media import VideoDecodeError, decode_video

    images = [picture()] * 3
    cap = FakeCapture(images, times=[0, 40, 80], count=count)
    with pytest.raises(SourceError):
        VideoSource("x", capture_factory=lambda path: cap).open()
    assert cap.releases == 1
    path = tmp_path / "fake.mp4"
    path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture(images, times=[0, 40, 80], count=count))
    with pytest.raises(VideoDecodeError):
        decode_video(path)


def test_failed_seek_reopens_and_releases_both_captures():
    class NoSeekCapture(FakeCapture):
        def set(self, prop, value):
            return False

    first = NoSeekCapture([picture()] * 2, times=[0, 40])
    second = FakeCapture([picture()] * 2, times=[0, 40])
    queue = iter([first, second])
    with VideoSource("x", capture_factory=lambda path: next(queue)) as source:
        assert [source.read().timestamp for _ in range(2)] == [0, 0.04]
        assert source.read() is None
    assert [first.releases, second.releases] == [1, 1]


@pytest.mark.parametrize("second_count", [1, 3])
def test_reopened_decoder_count_mismatch_is_rejected(second_count):
    class NoSeekCapture(FakeCapture):
        def set(self, prop, value):
            return False

    first = NoSeekCapture([picture()] * 2, times=[0, 40])
    second = FakeCapture([picture()] * second_count, times=[0, 40, 80], count=2)
    queue = iter([first, second])
    with pytest.raises(SourceError):
        with VideoSource("x", capture_factory=lambda path: next(queue)) as source:
            while source.read() is not None:
                pass
    assert [first.releases, second.releases] == [1, 1]


@pytest.mark.parametrize("times", [[1, 1], [2, 1], [1, float("nan")]])
def test_camera_invalid_clock_releases(times):
    cap = FakeCapture([picture()] * 2)
    ticks = iter(times)
    source = CameraSource(capture_factory=lambda index: cap, clock=lambda: next(ticks)).open()
    source.read()
    with pytest.raises(SourceError):
        source.read()
    assert cap.releases == 1


def test_capture_exception_and_invalid_image_release():
    class BrokenCapture(FakeCapture):
        def read(self):
            raise RuntimeError("capture backend failed")

    for cap in [BrokenCapture([]), FakeCapture([np.zeros((3, 3), dtype=np.uint8)])]:
        with pytest.raises(SourceError):
            with CameraSource(capture_factory=lambda index: cap) as source:
                source.read()
        assert cap.releases == 1


def test_reopen_restarts_video_and_context_exception_releases():
    captures = [FakeCapture([picture()] * 2, times=[0, 40]) for _ in range(2)]
    queue = iter(captures)
    source = VideoSource("x", capture_factory=lambda path: next(queue))
    with source:
        assert source.read().frame_index == 0
    with pytest.raises(RuntimeError):
        with source:
            assert source.read().frame_index == 0
            raise RuntimeError("consumer failed")
    assert [cap.releases for cap in captures] == [1, 1]


def test_buffer_copies_resizes_and_rebases_complete_clip():
    buffer = ClipBuffer(max_width=4)
    original = picture()
    buffer.append(packet(100, 4, image=original))
    original[:] = 255
    buffer.append(packet(100.25, 7))
    clip = buffer.finish()
    assert isinstance(clip.frames, tuple)
    assert [frame.timestamp for frame in clip.frames] == [0, 0.25]
    assert [frame.frame_index for frame in clip.frames] == [4, 7]
    assert clip.frames[0].image.shape == (2, 4, 3)
    assert clip.frames[0].image.max() < 255
    with pytest.raises(ValueError, match="two"):
        buffer.finish()


@pytest.mark.parametrize("limits", [{"max_frames": 2}, {"max_seconds": 0.05}, {"max_bytes": 2 * 96}])
def test_overflow_aborts_and_clears_whole_clip(limits):
    buffer = ClipBuffer(**limits)
    buffer.append(packet())
    buffer.append(packet(0.04, 1))
    with pytest.raises(ClipLimitError):
        buffer.append(packet(0.08, 2))
    with pytest.raises(ValueError, match="two"):
        buffer.finish()


@pytest.mark.parametrize("frames", [
    (packet(), packet(0, 1)),
    (packet(), packet(0.1, 0)),
    (packet(), packet(0.1, 1, "other")),
])
def test_captured_clip_validates_sequence(frames):
    with pytest.raises(ValueError):
        CapturedClip(frames, "test")


@pytest.mark.parametrize("kwargs", [
    {"timestamp": float("nan")}, {"timestamp": float("inf")},
    {"image": np.zeros((3, 4), dtype=np.uint8)},
    {"image": np.zeros((3, 4, 3), dtype=np.float32)},
    {"frame_index": -1}, {"source_id": ""},
])
def test_packet_rejects_invalid_values(kwargs):
    values = dict(image=picture(), timestamp=0, frame_index=0, source_id="test")
    values.update(kwargs)
    with pytest.raises(ValueError):
        FramePacket(**values)


def test_clear_allows_new_source_and_source_requires_open():
    buffer = ClipBuffer()
    buffer.append(packet())
    buffer.clear()
    buffer.append(packet(2, 1, "other"))
    buffer.append(packet(3, 2, "other"))
    assert buffer.finish().source_id == "other"
    with pytest.raises(SourceError, match="open"):
        CameraSource().read()
