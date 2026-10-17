"""Exercise real video -> Qt controls -> recording -> skeleton replay.

Only file dialogs are supplied by the harness. The source, MediaPipe graph,
session, recorder and reader are production implementations. This is Qt widget
acceptance; it does not attest to a physical camera or desktop interaction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog
    from slbridge.keypoint_desktop import KeypointWorkbench
    from slbridge.keypoint_recording import KeypointRecording

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    destination = out / "qt-real-video.slkp.jsonl"
    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = KeypointWorkbench()
    session = window.session
    window.show()

    def pump(condition, timeout=30.):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            app.processEvents()
            if session.status()["error"]:
                raise AssertionError(session.status()["error"])
            if condition():
                return
            time.sleep(.005)
        raise AssertionError("Qt flow timed out: " + repr(session.status()))

    def click(button):
        assert button.isEnabled(), button.objectName() + " disabled"
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        app.processEvents()

    try:
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(args.video), "")):
            click(window.video_button)
        pump(lambda: window.record_button.isEnabled() and window.preview.landmarks is not None)
        window.label_edit.setText("validation-sample-001")
        window.language_combo.setCurrentIndex(2)
        window.signer_edit.setText("dataset-test-A")
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(destination), "")):
            click(window.record_button)
        pump(lambda: session.status()["recorded_frames"] >= 10)
        click(window.reset_button)
        pump(lambda: session.status()["recorded_frames"] >= 20 and window.preview.landmarks is not None)
        app.processEvents()
        assert window.grab().save(str(out / "capture.png"))
        click(window.pause_button)
        paused_count = session.status()["recorded_frames"]
        until = time.perf_counter() + .3
        while time.perf_counter() < until:
            app.processEvents()
            time.sleep(.005)
        assert session.status()["recorded_frames"] == paused_count
        click(window.pause_button)
        pump(lambda: session.status()["recorded_frames"] >= paused_count + 3)
        click(window.finish_button)
        assert destination.is_file() and window._saved_path == destination
        recording = KeypointRecording(destination)
        frames = list(recording.iter_frames())
        assert len(frames) == session.status()["recorded_frames"]
        assert recording.metadata["annotations"] == dict(
            label_id="validation-sample-001", sign_language="csl",
            signer_id="dataset-test-A", review_status="draft")
        assert recording.metadata["source_timing_mode"] in {"pos_msec", "fps_fallback"}
        assert any(frame.reset_reason for frame in frames)
        capture_status = session.status()
        click(window.stop_button)
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(destination), "")):
            click(window.replay_button)
        pump(lambda: (session.status()["processed_frames"] == len(frames)
                      and window.preview.landmarks is not None
                      and window.preview.landmarks.frame_index == frames[-1].frame_index),
             timeout=frames[-1].timestamp - frames[0].timestamp + 15.)
        assert window.preview.landmarks.to_dict() == frames[-1].to_dict()
        assert window.preview.image is None and window.skeleton_only_check.isChecked()
        assert not window.record_button.isEnabled()
        assert session.status()["receipt_age_ms"] is None
        assert session.status()["extraction_ms"] is None
        assert window.grab().save(str(out / "replay.png"))
        report = dict(
            passed=True, video=str(args.video.resolve()), qt_platform=app.platformName(),
            source_timing_mode=recording.metadata["source_timing_mode"],
            recording=recording.summary(), capture_status=capture_status,
            replay_status=session.status(),
            left_hand_frames=sum(f.left_present for f in frames),
            right_hand_frames=sum(f.right_present for f in frames),
            shoulder_anchor_frames=sum(f.anchor_valid for f in frames),
            reset_frames=[dict(frame_index=f.frame_index, reason=f.reset_reason,
                               tracking_epoch=f.tracking_epoch) for f in frames if f.reset_reason],
            controls=["open_video", "record", "reset", "pause", "resume", "finish", "stop", "replay"],
            boundaries=["file dialogs supplied by harness", "Qt widget acceptance",
                        "physical camera untested", "no sign meaning recognizer or new algorithm"],
        )
    finally:
        window.close()
        deadline = time.perf_counter() + 10.
        while (session._capture_thread.is_alive() or session._extract_thread.is_alive()) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not session._capture_thread.is_alive() and not session._extract_thread.is_alive()
    report["workers_closed"] = True
    (out / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
