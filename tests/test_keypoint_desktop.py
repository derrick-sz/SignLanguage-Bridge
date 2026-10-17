"""Qt controls plus real archive readback; no camera or accuracy acceptance."""
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog
import pytest

from slbridge.keypoint_desktop import KeypointPreview, KeypointWorkbench
from slbridge.keypoint_recording import KeypointRecorder, KeypointRecording
from slbridge.keypoints import KeypointExtractor, LandmarkFrame, SCHEMA_ID
from slbridge.sources import FramePacket


def landmark_frame(index=0, source_id="sample-source", source_epoch=1):
    coordinates = [(0., 0.)] * 75
    coordinates[11], coordinates[12] = (.25, .5), (.75, .5)
    coordinates[33], coordinates[34] = (.25, .3), (.4, .3)
    observed = tuple(i in (11, 12, 33, 34) for i in range(75))
    visibility = tuple(1. if i in (11, 12) else None for i in range(33))
    return LandmarkFrame(
        schema_id=SCHEMA_ID, session_id="sample-session", source_id=source_id,
        source_epoch=source_epoch, tracking_epoch=0, frame_index=index,
        timestamp=index / 10, time_domain="media_seconds", received_monotonic=1.,
        extracted_monotonic=1.005, image_size=(40, 20), image_xy=tuple(coordinates),
        normalized_xy=tuple(((x - .5) * 2, y - .5) if observed[i] else (0., 0.)
                            for i, (x, y) in enumerate(coordinates)), point_observed=observed,
        normalized_observed=observed, anchor_valid=True, pose_visibility=visibility,
        left_present=True, right_present=False, motion=0., motion_valid=False,
        dt=None if index == 0 else .1, frames_skipped=0, reset_reason=None,
    )


class FakeSource:
    def __init__(self, target):
        self.target = target
        self.closed = False

    def close(self):
        self.closed = True


class FakeSession:
    """Use actual recorder/reader while replacing capture and extraction."""

    def __init__(self):
        self.epoch = 1
        self.closed = False
        self.source = None
        self.calls = []
        self.events = []
        self.view = (None, None)
        self.recorder = None
        self.data = dict(source_kind=None, source_active=False, paused=False,
                         recording=False, recorded_frames=0, processed_frames=0,
                         dropped_frames=0, queue_size=0, latest_dt=None,
                         latest_gap_seconds=None, extraction_ms=None, receipt_age_ms=None,
                         time_domain=None, error=None)

    def status(self):
        return dict(self.data)

    def _event(self, kind, value=None):
        self.events.append(SimpleNamespace(kind=kind, value=value, request_id=0, epoch=self.epoch))

    def open_source(self, source, *, kind, video_path=None):
        self.calls.append(("open_source", kind, source.target, video_path))
        if self.source is not None:
            self.source.close()
        self.source = source
        self.data.update(source_kind=kind, source_active=True, paused=False)
        self._event("opened")

    def open_recording(self, path):
        recording = KeypointRecording(path)
        self.calls.append(("open_recording", Path(path)))
        self.data.update(source_kind="recording", source_active=True, paused=False,
                         extraction_ms=None, receipt_age_ms=None)
        self.view = None, next(recording.iter_frames())
        self._event("opened")

    def toggle_pause(self):
        self.data["paused"] = not self.data["paused"]
        self._event("paused", self.data["paused"])

    def stop_source(self):
        self.calls.append(("stop_source",))
        self.cancel_recording()
        if self.source is not None:
            self.source.close()
        self.data["source_active"] = False
        self._event("stopped")

    def reset_tracking(self):
        self.calls.append(("reset_tracking",))
        self.view = None, None
        self._event("tracking_reset")

    def start_recording(self, path, *, metadata=None):
        self.calls.append(("start_recording", Path(path), dict(metadata)))
        extractor = KeypointExtractor(session_id="sample-session", source_epoch=1)
        self.recorder = KeypointRecorder(path, metadata={"extractor": extractor.metadata(), "annotations": metadata})
        extractor.close()
        self.data.update(recording=True, recorded_frames=0)
        self._event("recording_started")

    def publish(self, frame):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        image[:, 0, 2] = 255
        self.view = FramePacket(image, frame.timestamp, frame.frame_index, frame.source_id), frame
        self.data.update(processed_frames=self.data["processed_frames"] + 1,
                         extraction_ms=5., receipt_age_ms=8., latest_dt=frame.dt,
                         latest_gap_seconds=0.)
        if self.recorder is not None:
            self.recorder.append(frame)
            self.data["recorded_frames"] += 1

    def finish_recording(self):
        recorder, self.recorder = self.recorder, None
        self.data["recording"] = False
        path = recorder.finish()
        self._event("recording_saved", str(path))
        return path

    def cancel_recording(self):
        if self.recorder is not None:
            self.recorder.abort()
            self.recorder = None
        self.data["recording"] = False
        self._event("recording_cancelled", "用户取消录制")

    def poll(self):
        events, self.events = self.events, []
        return events

    def take_view(self):
        result, self.view = self.view, (None, None)
        return result

    def close(self):
        self.calls.append(("close",))
        self.cancel_recording()
        self.stop_source()
        self.closed = True


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication(["keypoint-test"])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def workspace(qt_app, monkeypatch):
    monkeypatch.setattr("slbridge.keypoint_desktop.VideoSource", FakeSource)
    monkeypatch.setattr("slbridge.keypoint_desktop.CameraSource", FakeSource)
    session = FakeSession()
    window = KeypointWorkbench(session=session)
    window.show()
    qt_app.processEvents()
    yield window, session
    window.close()
    qt_app.processEvents()


def test_mounted_record_controls_save_readback_then_skeleton_replay(workspace, monkeypatch, tmp_path, qt_app):
    window, session = workspace
    destination = tmp_path / "待核验样本.slkp.jsonl"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("源视频.mp4", ""))
    window.video_button.click()
    window._poll()
    assert session.calls[0] == ("open_source", "video", "源视频.mp4", "源视频.mp4")
    assert not window.record_button.isEnabled(), "wait for an extracted frame before recording"
    session.publish(landmark_frame(0))
    window._poll()
    assert window.record_button.isEnabled()
    window.label_edit.setText("bom-dia")
    window.language_combo.setCurrentIndex(1)
    window.signer_edit.setText("participant-03")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(destination), ""))
    window.record_button.click()
    assert not window.video_button.isEnabled() and window.finish_button.isEnabled()
    session.publish(landmark_frame(1))
    window._poll()
    session.publish(landmark_frame(2))
    window._poll()
    assert window.preview.image.pixelColor(0, 0).red() == 255
    assert "左手 2/21" in window.quality_label.text()
    assert "2 帧" in window.recording_label.text()
    session.data["source_active"] = False
    window._poll()
    assert window.finish_button.isEnabled(), "EOF must not disable saving captured samples"
    window.finish_button.click()
    window._poll()
    recording = KeypointRecording(destination)
    assert recording.metadata["annotations"] == {
        "label_id": "bom-dia", "sign_language": "libras",
        "signer_id": "participant-03", "review_status": "draft",
    }
    frames = list(recording.iter_frames())
    assert len(frames) == 2 and frames[1].timestamp == .2
    assert frames[0].image_xy == landmark_frame().image_xy
    assert window._saved_path == destination
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(destination), ""))
    window.replay_button.click()
    window._poll()
    qt_app.processEvents()
    assert window.preview.image is None and window.preview.landmarks is not None
    assert window.skeleton_only_check.isChecked() and not window.skeleton_only_check.isEnabled()
    assert not window.record_button.isEnabled()
    assert "提取 — ms" in window.metrics_label.text()
    assert "尚未接入手语识别" in window.scope_label.text()


def test_camera_pause_stop_reset_cancel_and_close_release(workspace, monkeypatch, tmp_path):
    window, session = workspace
    window.camera_index.setValue(2)
    window.camera_button.click()
    window._poll()
    assert session.calls[-1] == ("open_source", "camera", 2, None)
    source = session.source
    session.publish(landmark_frame())
    window._poll()
    window.pause_button.click()
    assert session.data["paused"] and window.pause_button.text() == "继续"
    assert not window.record_button.isEnabled()
    window.pause_button.click()
    assert not session.data["paused"]
    window.reset_button.click()
    assert ("reset_tracking",) in session.calls
    path = tmp_path / "cancelled.slkp.jsonl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window.record_button.click()
    assert not window.stop_button.isEnabled() and window.reset_button.isEnabled()
    session.publish(landmark_frame())
    window.cancel_button.click()
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
    assert window.record_button.isEnabled()
    window.stop_button.click()
    assert source.closed and not window.record_button.isEnabled()
    window.close()
    assert session.closed and not window._timer.isActive()


def test_skeleton_pixels_use_image_coordinates_and_suppress_mismatched_overlay(qt_app):
    preview = KeypointPreview()
    preview.resize(400, 220)
    preview.show()
    frame = landmark_frame()
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    preview.set_packet(FramePacket(pixels, 0., 0, frame.source_id))
    preview.set_landmarks(frame)
    qt_app.processEvents()
    target = preview._target()
    point = preview._point(frame.image_xy[33], target)

    def sampled_color():
        # QWidget positions are logical pixels. A Windows display at 125%
        # produces a 1.25-DPR grab whose image pixels use physical positions.
        pixmap = preview.grab()
        scale = pixmap.devicePixelRatio()
        return pixmap.toImage().pixelColor(round(point.x() * scale), round(point.y() * scale))

    shown = sampled_color()
    assert shown.blue() > 100 and shown.red() < 50
    # A newer raw image must not be presented with the old frame's skeleton.
    preview.set_packet(FramePacket(pixels, .1, 1, frame.source_id))
    qt_app.processEvents()
    shown = sampled_color()
    assert shown.red() == shown.green() == shown.blue() == 0
    preview.set_skeleton_only(True)
    qt_app.processEvents()
    shown = sampled_color()
    assert shown.blue() > 100 and shown.red() < 50
    preview.close()


def test_main_workspace_opens_owned_session_and_closes_it(qt_app, monkeypatch):
    from slbridge.desktop import SignLanguageApp

    created = []
    def factory(parent):
        session = FakeSession()
        workbench = KeypointWorkbench(parent, session=session)
        created.append((workbench, session))
        return workbench

    monkeypatch.setattr("slbridge.keypoint_desktop.KeypointWorkbench", factory)
    main = SignLanguageApp()
    stop_calls = []
    original_stop = main.session.stop_source
    def stop():
        stop_calls.append(True)
        original_stop()
    monkeypatch.setattr(main.session, "stop_source", stop)
    main.show()
    qt_app.processEvents()
    main.keypoint_button.click()
    qt_app.processEvents()
    workbench, session = created[0]
    assert stop_calls == [True]
    assert workbench.isVisible() and workbench.parent() is main
    assert workbench.windowModality() == Qt.WindowModality.WindowModal
    assert main.session.recognizer is None
    assert workbench.session is not main.session
    main.close()
    assert session.closed and main._closed


def test_operation_error_keeps_recovery_and_no_partial_output(workspace, monkeypatch, tmp_path):
    window, session = workspace
    window.camera_button.click()
    session.publish(landmark_frame())
    window._poll()
    path = tmp_path / "empty.slkp.jsonl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window.record_button.click()
    window.finish_button.click()
    assert "empty recording" in window.status_label.text()
    assert not path.exists()
    assert window.record_button.isEnabled()


def test_cleanup_warning_after_publication_keeps_saved_archive_status(workspace, monkeypatch, tmp_path):
    window, session = workspace
    window.camera_button.click()
    session.publish(landmark_frame())
    window._poll()
    path = tmp_path / "published.slkp.jsonl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window.record_button.click()
    session.publish(landmark_frame(1))
    window.finish_button.click()
    session._event("recording_cleanup_warning", "partial locked")
    window._poll()
    assert KeypointRecording(path).summary()["count"] == 1
    assert window._saved_path == path
    assert window.recording_label.text().startswith("已保存")
    assert "关键点已保存；临时文件未能清理" in window.status_label.text()
    assert "partial locked" in window.status_label.text()


def test_qt_controls_with_real_async_session_record_and_replay(qt_app, monkeypatch, tmp_path):
    from slbridge.keypoint_session import KeypointSession

    class Source:
        index = 0
        closed = False
        def open(self):
            return self
        def read(self):
            time.sleep(.02)
            if self.closed:
                return None
            index = self.index
            self.index += 1
            return FramePacket(np.zeros((16, 24, 3), np.uint8), index * .025, index, "injected-camera")
        def close(self):
            self.closed = True

    graphs = []
    class Graph:
        closed = False
        def process(self, image):
            point = lambda x, y: SimpleNamespace(x=x, y=y, visibility=1.)
            pose = [point(.5, .5) for _ in range(33)]
            pose[11], pose[12] = point(.3, .3), point(.7, .3)
            hand = [point(.2 + i * .01, .4) for i in range(21)]
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pose),
                                   left_hand_landmarks=SimpleNamespace(landmark=hand),
                                   right_hand_landmarks=None)
        def close(self):
            self.closed = True

    def graph_factory(**options):
        graph = Graph()
        graphs.append(graph)
        return graph

    source = Source()
    session = KeypointSession(extractor_factory=lambda **kwargs: KeypointExtractor(
        holistic_factory=graph_factory, **kwargs))
    window = KeypointWorkbench(session=session)
    monkeypatch.setattr("slbridge.keypoint_desktop.CameraSource", lambda index: source)
    destination = tmp_path / "async.slkp.jsonl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(destination), ""))

    def pump(condition, seconds=4):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            qt_app.processEvents()
            window._poll()
            if condition():
                return
            time.sleep(.005)
        raise AssertionError("real session and Qt controls did not reach the expected state")

    try:
        window.show()
        window.camera_button.click()
        pump(lambda: window.record_button.isEnabled() and window.preview.landmarks is not None)
        assert "左手 21/21" in window.quality_label.text()
        window.record_button.click()
        pump(lambda: session.status()["recorded_frames"] >= 3)
        window.pause_button.click()
        assert session.status()["paused"]
        window.finish_button.click()
        pump(lambda: destination.exists())
        recording = KeypointRecording(destination)
        assert recording.summary()["count"] >= 3
        first = next(recording.iter_frames())
        window.stop_button.click()
        pump(lambda: source.closed)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(destination), ""))
        window.replay_button.click()
        pump(lambda: window.preview.landmarks is not None)
        assert window.preview.landmarks.source_id == first.source_id
        assert window.preview.image is None
        assert not window.record_button.isEnabled()
        assert len(graphs) == 1, "archive playback must not re-extract or load a recognizer"
    finally:
        window.close()
    pump(lambda: all(graph.closed for graph in graphs))
    assert session.closed
