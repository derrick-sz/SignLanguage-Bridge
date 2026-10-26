"""A frame-count Uni-Sign window must become ready at ordinary source FPS."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.continuous import ContinuousConfig, RollingWindow
from slbridge.sources import FramePacket


def feed_frame(window, index, fps):
    window.append(FramePacket(np.zeros((8, 12, 3), dtype=np.uint8), index / fps, index, 'video'))


@pytest.mark.parametrize('fps', [25., 29.97, 30., 31.26, 60.])
def test_cli_frame_window_is_ready_at_requested_count(tmp_path, monkeypatch, fps):
    from slbridge.desktop_cli import main
    received = {}

    class FakeQt:
        @classmethod
        def instance(cls):
            return cls()

        def exec(self):
            return 0

    class FakeApp:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def load_catalog(self, path):
            pass

        def show(self):
            pass

    monkeypatch.setitem(sys.modules, 'PySide6.QtWidgets', SimpleNamespace(QApplication=FakeQt))
    monkeypatch.setitem(sys.modules, 'slbridge.desktop', SimpleNamespace(SignLanguageApp=FakeApp))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text('{}', encoding='utf-8')
    assert main(['--unisign', '--unisign-model', str(bundle), '--continuous',
                 '--catalog', str(tmp_path / 'catalog.json')]) == 0
    window = RollingWindow(received['continuous_config'])
    for n in range(47):
        feed_frame(window, n, fps)
        assert window.snapshot() is None
    feed_frame(window, 47, fps)
    first = window.snapshot()
    assert first is not None, f'48 frames at {fps} FPS never became ready'
    assert len(first.clip.frames) == 48
    assert first.start_time == 0 and first.end_time == pytest.approx(47/fps)
    for n in range(48, 360):
        feed_frame(window, n, fps)
        snapshot = window.snapshot()
        assert snapshot is not None and len(snapshot.clip.frames) == 48
        assert snapshot.clip.frames[-1].frame_index == n
        assert window.buffered_frames <= 48
    assert received['continuous_config'].update_interval_seconds == pytest.approx(16/30)


def test_frame_readiness_waits_for_count_even_after_time_minimum():
    window = RollingWindow(ContinuousConfig(minimum_seconds=.1, window_seconds=8,
                                            minimum_frames=48, max_frames=48))
    for n in range(47):
        feed_frame(window, n, 30.)
        assert window.snapshot() is None
    feed_frame(window, 47, 30.)
    assert len(window.snapshot().clip.frames) == 48


@pytest.mark.parametrize('count', [True, 1, 49, 2.5])
def test_frame_readiness_rejects_invalid_count(count):
    with pytest.raises(ValueError):
        ContinuousConfig(max_frames=48, minimum_frames=count)


def test_default_duration_window_keeps_its_existing_readiness():
    config = ContinuousConfig()
    assert config.minimum_frames is None
    window = RollingWindow(config)
    for n in range(36):
        feed_frame(window, n, 30.)
    assert window.snapshot() is None
    feed_frame(window, 36, 30.)
    assert window.snapshot() is not None
