from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from slbridge.contracts import InputSpec, RecognitionResult, ClassificationPrediction
from slbridge.segmentation import ActivityObservation, SegmentConfig
from slbridge.session import RecognitionSession
from slbridge.sources import FramePacket


def wait(predicate, seconds=4):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Automatic session did not reach expected behavior")


class ClipModel:
    def __init__(self, block=False):
        self.bundle = SimpleNamespace(manifest=SimpleNamespace(
            backend="random_forest_v1", run_id="behavior-fixture", task_type="classification",
            score_type="uncalibrated_class_probability", sign_language="libras",
            capabilities=("clip",), input_spec=InputSpec.for_representation("mediapipe225_rgb_v1",
                                                  extractor={"config": {}, "identity": {}})),
            labels={"word": {"pt-BR": "Olá"}})
        self.runtime_info = {"providers": ["CPUExecutionProvider"], "device": "cpu"}
        self.entered, self.release = threading.Event(), threading.Event()
        if not block:
            self.release.set()
        self.closed = False

    def recognize_clip_profiled(self, clip, *, sample_id):
        self.entered.set()
        assert self.release.wait(4)
        result = RecognitionResult("recognized", sample_id, "behavior-fixture", "libras",
                    ClassificationPrediction("word", .8, {"word": .8}, {"pt-BR": "Olá"}), {})
        return result, {"preprocess_ms": 0., "inference_ms": 0.}

    def close(self):
        self.closed = True


class ReplaySource:
    def __init__(self, count=60):
        self.gate = threading.Event()
        self.index, self.count = 0, count
        self.closed = False

    def open(self):
        return self

    def read(self):
        self.gate.wait(3)
        time.sleep(.005)
        if self.index >= self.count:
            return None
        index = self.index
        self.index += 1
        return FramePacket(np.zeros((8, 12, 3), np.uint8), index * .05, index, "replay")

    def close(self):
        self.closed = True


class Observer:
    instances = []
    def __init__(self, config):
        self.closed = False
        self.instances.append(self)

    def observe(self, packet):
        n = packet.frame_index
        hands = not (26 <= n <= 30)
        motion = .8 if n <= 15 or 31 <= n <= 45 else .05
        return ActivityObservation(packet.timestamp, hands, True, motion, hands)

    def close(self):
        self.closed = True


CONFIG = SegmentConfig(start_candidate_seconds=.1, end_candidate_seconds=.25,
                       pre_buffer_seconds=.1, post_buffer_seconds=.05,
                       min_segment_seconds=.2, rearm_seconds=.1)


def started(model=None, **kwargs):
    source = ReplaySource()
    session = RecognitionSession(model or ClipModel(), observer_factory=Observer, **kwargs)
    session.open_source(source, kind="camera")
    wait(lambda: session.source_active)
    session.enable_automatic(CONFIG)
    return session, source


def stop(session, source):
    source.gate.set()
    session.close()
    wait(lambda: not session._observer_worker.is_alive() and not session._capture_thread.is_alive()
         and not session._worker.is_alive())


def test_continuous_capture_commits_two_identical_words_and_history_survives_source_stop():
    session, source = started()
    try:
        source.gate.set()
        wait(lambda: len(session.captions.snapshot()["events"]) == 2)
        snapshot = session.captions.snapshot()
        assert [e["result"]["prediction"]["label_id"] for e in snapshot["events"]] == ["word", "word"]
        assert len({e["event_id"] for e in snapshot["events"]}) == 2
        assert session._tasks.maxsize == 1 and session._auto_frames.maxsize == 2
        session.stop_source()
        assert len(session.captions.snapshot()["events"]) == 2
        session.undo_caption()
        assert len(session.captions.snapshot()["events"]) == 1
        session.clear_history()
        assert not session.captions.snapshot()["events"]
        assert not any(e.kind == "auto_result" for e in session.poll())
    finally:
        stop(session, source)


@pytest.mark.parametrize("action", ["cancel", "stop_source", "clear_history"])
def test_invalidated_native_result_never_appends(action):
    model = ClipModel(block=True)
    session, source = started(model)
    try:
        source.gate.set()
        assert model.entered.wait(3)
        getattr(session, action)()
        model.release.set()
        wait(lambda: not session.busy)
        assert not session.captions.snapshot()["events"]
    finally:
        model.release.set()
        stop(session, source)


def test_deadline_drops_native_result_and_keeps_workers_usable():
    model = ClipModel(block=True)
    session, source = started(model, automatic_timeout=.03)
    try:
        source.count = 26
        source.gate.set()
        assert model.entered.wait(3)
        time.sleep(.05)
        model.release.set()
        wait(lambda: not session.busy)
        assert not session.captions.snapshot()["events"]
        assert any(e.kind == "candidate" and "deadline" in e.value.get("reason", "") for e in session.poll())
        assert session._worker.is_alive()
    finally:
        model.release.set()
        stop(session, source)


def test_busy_capture_applies_backpressure_with_bounded_candidates():
    model = ClipModel(block=True)
    session, source = started(model)
    try:
        source.gate.set()
        assert model.entered.wait(3)
        wait(lambda: source.closed)
        assert len(session._auto_pending) <= 1
        assert len(session.captions.snapshot()["candidates"]) <= 2
        session.disable_automatic()
        model.release.set()
        wait(lambda: not session.busy)
    finally:
        model.release.set()
        stop(session, source)


def test_eof_releases_activity_graph_and_close_error_keeps_worker_alive():
    session, source = started()
    try:
        source.gate.set()
        wait(lambda: source.closed and Observer.instances[-1].closed)
        assert session._observer_worker.is_alive()
    finally:
        stop(session, source)

    class BrokenClose(Observer):
        def close(self):
            super().close()
            raise RuntimeError("graph close fixture error")

    source = ReplaySource()
    session = RecognitionSession(ClipModel(), observer_factory=BrokenClose)
    try:
        session.open_source(source, kind="camera")
        wait(lambda: session.source_active)
        session.enable_automatic(CONFIG)
        source.gate.set()
        wait(lambda: source.closed and not session.automatic)
        assert session._observer_worker.is_alive()
        assert any(e.kind == "error" and "graph close" in e.value for e in session.poll())
    finally:
        stop(session, source)
