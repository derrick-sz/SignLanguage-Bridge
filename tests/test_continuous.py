from dataclasses import replace
import importlib.util

import numpy as np
import pytest

from slbridge.sources import FramePacket


def api():
    assert importlib.util.find_spec('slbridge.continuous') is not None, 'Rolling video window is missing'
    from slbridge.continuous import ContinuousConfig, RollingWindow
    return ContinuousConfig, RollingWindow


def packet(n, *, source='camera', width=12):
    return FramePacket(np.full((8, width, 3), n % 256, np.uint8), n * .1, n, source)


def test_window_has_explicit_bounded_defaults_and_waits_for_minimum_duration():
    Config, Window = api()
    config = Config()
    assert (config.minimum_seconds, config.window_seconds, config.update_interval_seconds) == (1.2, 2.4, .5)
    assert (config.max_frames, config.max_bytes, config.max_width) == (180, 96*1024*1024, 640)
    window = Window(config)
    for n in range(5):
        window.append(packet(n))
    assert window.snapshot() is None
    for n in range(5, 50):
        window.append(packet(n))
    snapshot = window.snapshot()
    assert snapshot is not None and snapshot.end_time == 4.9
    assert 1.2 <= snapshot.end_time - snapshot.start_time <= 2.4 + 1e-9
    assert snapshot.clip.frames[0].timestamp == 0
    assert window.buffered_frames <= 25


def test_window_copies_source_memory_and_snapshot_survives_eviction():
    Config, Window = api()
    window = Window(Config(minimum_seconds=.1, window_seconds=.3, update_interval_seconds=.1))
    first = packet(0)
    window.append(first)
    first.image[:] = 200
    window.append(packet(1))
    saved = window.snapshot()
    assert saved.clip.frames[0].image[0, 0, 0] == 0
    assert not saved.clip.frames[0].image.flags.writeable
    for n in range(2, 20):
        window.append(packet(n))
    assert saved.clip.frames[0].frame_index == 0
    assert window.snapshot().clip.frames[-1].frame_index == 19


def test_frame_byte_and_gap_limits_never_build_an_old_unbounded_window():
    Config, Window = api()
    window = Window(Config(minimum_seconds=.1, window_seconds=.5, max_frames=3, max_bytes=900))
    for n in range(20):
        window.append(packet(n))
        assert window.buffered_frames <= 3 and window.buffered_bytes <= 900
    snapshot = window.snapshot()
    assert snapshot.clip.frames[-1].frame_index == 19
    window.append(packet(50))
    assert window.snapshot() is None and window.buffered_frames == 1


@pytest.mark.parametrize('fault', ['source', 'time', 'shape'])
def test_window_rejects_identity_errors(fault):
    Config, Window = api()
    window = Window(Config())
    window.append(packet(0))
    invalid = {'source': packet(1, source='other'), 'time': packet(0), 'shape': packet(1, width=14)}[fault]
    with pytest.raises(ValueError):
        window.append(invalid)


@pytest.mark.parametrize('values', [dict(window_seconds=0), dict(minimum_seconds=3),
                                  dict(update_interval_seconds=0), dict(max_frames=1),
                                  dict(max_result_age_seconds=float('nan'))])
def test_config_rejects_invalid_limits(values):
    Config, _ = api()
    with pytest.raises(ValueError):
        Config(**values)
