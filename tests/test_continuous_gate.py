"""The MediaPipe hands/quality gate must not drop known-good file replay.

A live camera should wait until someone is actually signing; an explicitly
replayed file already contains signing, and at 204x170 the MediaPipe gate never
opened, so no caption was ever produced (measured 2026-10-25).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from slbridge.continuous import ContinuousConfig
from slbridge.session import RecognitionSession
from slbridge.sources import FramePacket


def wait_for(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("background operation did not complete")


class FakeSource:
    def __init__(self, eof=2):
        self.eof = eof
        self.index = 0
        self.closed = False

    def open(self):
        return self

    def read(self):
        time.sleep(.005)
        if self.closed or self.index >= self.eof:
            return None
        index = self.index
        self.index += 1
        return FramePacket(np.zeros((8, 12, 3), dtype=np.uint8), index * .01, index, "fake")

    def close(self):
        self.closed = True


class FakeRecognizer:
    """Only what ``enable_continuous`` inspects: capabilities + the two calls."""

    def capabilities(self):
        return {"modes": ("clip",), "inputs": ({"input_kinds": ("captured_clip",)},)}

    def recognize_clip(self, clip, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(status="recognized")

    def recognize(self, path):
        from types import SimpleNamespace
        return SimpleNamespace(status="recognized")


def ready_session(kind):
    session = RecognitionSession(FakeRecognizer())
    # The gate only needs the clip contract; a full typed adapter is out of scope.
    session.model_manager.capabilities = lambda: {
        "modes": ("clip",), "inputs": ({"input_kinds": ("captured_clip",)},)}
    session.open_source(FakeSource(eof=6), kind=kind,
                        **({"video_path": "clip.mp4"} if kind == "video" else {}))
    wait_for(lambda: session.source_active and not session.busy)
    return session


def gate_for(kind, **config_kwargs):
    session = ready_session(kind)
    try:
        session.enable_continuous(ContinuousConfig(
            minimum_seconds=.2, window_seconds=1.0, update_interval_seconds=.1,
            max_frames=8, minimum_frames=8, **config_kwargs))
        return session._continuous_require_hands
    finally:
        session.disable_continuous()
        session.close()


def test_config_defaults_to_auto_and_rejects_non_boolean():
    assert ContinuousConfig().require_hands_visible is None
    assert ContinuousConfig(require_hands_visible=True).require_hands_visible is True
    assert ContinuousConfig(require_hands_visible=False).require_hands_visible is False
    with pytest.raises(ValueError, match="require_hands_visible"):
        ContinuousConfig(require_hands_visible="yes")


def test_file_replay_does_not_require_hands_by_default():
    assert gate_for("video") is False


def test_camera_keeps_requiring_hands_by_default():
    assert gate_for("camera") is True


def test_explicit_override_wins_over_source_kind():
    assert gate_for("video", require_hands_visible=True) is True
    assert gate_for("camera", require_hands_visible=False) is False


