"""Exercise mounted Qt controls with a real saved model and video.

Optional camera validation checks capture and task lifecycle only. It saves no
camera pixels and makes no hardware recognition accuracy claim. Evidence is
saved under the caller's ignored artifacts directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from slbridge._io import configure_utf8_streams, write_json_atomic


def validate(model: Path, video: Path, catalog: Path, output: Path, *,
             camera: int | None = None, camera_seconds: float = 2.0) -> dict:
    from PySide6.QtWidgets import QApplication, QFileDialog, QFrame

    from slbridge.desktop import SignLanguageApp

    if camera is not None and camera < 0:
        raise ValueError("Camera device index must be nonnegative")
    if not 0.1 <= camera_seconds <= 10:
        raise ValueError("Camera recording duration must be between 0.1 and 10 seconds")
    output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    app = SignLanguageApp(model=model)
    app.show()
    started = time.monotonic()
    updates = 0

    def pump(condition, timeout=60):
        nonlocal updates
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            application.processEvents()
            updates += 1
            if app.session.state == "ERROR":
                raise AssertionError(app.status_label.text())
            if condition():
                return
            time.sleep(.01)
        raise TimeoutError(f"Desktop did not finish: {app.status_label.text()}")

    def click(button):
        application.processEvents()
        assert button.isEnabled(), f"Disabled control: {button.text()}"
        button.click()
        application.processEvents()

    try:
        app.load_catalog(catalog)
        pump(lambda: app.session.recognizer is not None and not app.session.busy)
        # The dialog response is controlled; source creation and decoding are real.
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(video), "")):
            click(app.video_button)
        pump(lambda: app._photo is not None)
        click(app.recognize_button)
        processing_updates = updates
        pump(lambda: app.caption is not None)
        inference_updates = updates - processing_updates
        first = app.caption.to_dict()
        assert first["status"] == "recognized", first
        assert app.zh_label.text() == first["zh"]
        assert app.pt_label.text() == first["pt"]
        assert "待核验" in app.notice_label.text()
        assert inference_updates > 0
        assert app.isVisible() and app.zh_label.isVisible() and app.pt_label.isVisible()

        # Select a real editable file, save its changed contents, then invoke the
        # dedicated reload button. This proves save -> reload -> mounted subtitle.
        data = json.loads(catalog.read_text(encoding="utf-8"))
        edited = output / "edited-caption-catalog.json"
        write_json_atomic(edited, data)
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(edited), "")):
            click(app.catalog_button)
        entry = next(item for item in data["labels"] if item["label_id"] == first["label_id"])
        entry["display_text"]["zh-CN"] = "验收编辑：" + first["zh"]
        write_json_atomic(edited, data)
        click(app.reload_catalog_button)
        assert app.zh_label.text() == entry["display_text"]["zh-CN"]
        app.load_catalog(catalog)
        application.processEvents()
        assert app.zh_label.text() == first["zh"]

        # Dictionary reload and text/style changes post delayed LayoutRequest
        # events. Keep pumping them before inspecting geometry or taking a shot.
        app.raise_()
        app.activateWindow()
        settling_started = time.monotonic()
        pump(lambda: time.monotonic() - settling_started >= .15, timeout=5)
        divider = app.findChild(QFrame, "divider")
        assert divider is not None, "Result divider is missing"

        def label_geometry(label):
            rect = label.geometry()
            return {
                "text": label.text(),
                "rect": [rect.x(), rect.y(), rect.width(), rect.height()],
                "font_height": label.fontMetrics().height(),
                "minimum_height": label.minimumHeight(),
                "visible": label.isVisible(),
            }

        divider_rect = divider.geometry()
        layout_geometry = {
            "window_size": [app.width(), app.height()],
            "device_pixel_ratio": app.devicePixelRatioF(),
            "zh": label_geometry(app.zh_label),
            "pt": label_geometry(app.pt_label),
            "divider_rect": [divider_rect.x(), divider_rect.y(),
                             divider_rect.width(), divider_rect.height()],
            "zh_above_divider": app.zh_label.geometry().bottom() < divider_rect.y(),
        }
        # Preserve real geometry even when an assertion stops the acceptance.
        write_json_atomic(output / "desktop-layout.json", layout_geometry)
        assert app.zh_label.height() >= app.zh_label.fontMetrics().height(), layout_geometry
        assert app.pt_label.height() >= app.pt_label.fontMetrics().height(), layout_geometry
        assert layout_geometry["zh_above_divider"], layout_geometry

        # Capture only the file-validation stage. QWidget.grab() follows Qt's
        # device-pixel scaling and does not crop bottom captions on Windows.
        screenshot = output / "desktop-file-caption.png"
        try:
            if not app.grab().save(str(screenshot)):
                raise OSError("Qt could not save the desktop screenshot")
        except (OSError, RuntimeError) as error:
            screenshot = None
            screenshot_error = str(error)
        else:
            screenshot_error = None

        click(app.cancel_button)
        assert app.caption is None and app.pt_label.text() == ""
        click(app.stop_button)
        pump(lambda: not app.session.source_active)

        camera_report = None
        if camera is not None:
            # No camera screenshot or frames are written. A model result here
            # proves task delivery/rejection behavior, never label correctness.
            app.open_camera(camera)
            pump(lambda: app.session.source_kind == "camera"
                 and app.session.source_active and app._photo is not None, timeout=20)
            click(app.record_button)
            assert app.session.state == "RECORDING"
            recording_started = time.monotonic()
            pump(lambda: time.monotonic() - recording_started >= camera_seconds,
                 timeout=camera_seconds + 5)
            click(app.finish_button)
            camera_processing_updates = updates
            pump(lambda: app.caption is not None)
            camera_caption = app.caption.to_dict()
            assert camera_caption["status"] in {"recognized", "rejected"}
            assert app.zh_label.text() == camera_caption["zh"]
            assert app.pt_label.text() == camera_caption["pt"]
            if camera_caption["status"] == "rejected":
                assert app.pt_label.text() == ""
            click(app.cancel_button)
            assert app.caption is None and app.pt_label.text() == ""
            click(app.stop_button)
            pump(lambda: not app.session.source_active and app.session.source_kind is None)
            camera_report = {
                "device_index": camera, "preview_opened": True,
                "record_seconds": camera_seconds, "recording_submitted": True,
                "caption": camera_caption, "cancel_clears_caption": True,
                "source_released": True, "camera_pixels_saved": False,
                "accuracy_validation": False,
                "updates_during_inference": updates - camera_processing_updates,
            }

        report = {
            "stage": "complete", "framework": "PySide6",
            "model": str(model.resolve()), "video": str(video.resolve()),
            "file_caption": first, "mounted_caption_labels": True,
            "saved_dictionary_edit_changes_display": True,
            "dedicated_reload_button_verified": True,
            "cancel_clears_caption": True,
            "updates_during_inference": inference_updates,
            "seconds": round(time.monotonic() - started, 3),
            "screenshot": str(screenshot.resolve()) if screenshot else None,
            "screenshot_error": screenshot_error,
            "layout_geometry": layout_geometry,
            "camera_hardware": camera_report if camera_report is not None else "not requested",
        }
    finally:
        app.on_close()
        application.processEvents()
        app.session._capture_thread.join(5)
        app.session._worker.join(5)
    assert not app.session._capture_thread.is_alive()
    assert not app.session._worker.is_alive()
    report["workers_released"] = True
    write_json_atomic(output / "desktop-validation.json", report)
    return report


def main(argv=None):
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description="Validate actual Qt desktop controls with a real video/model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/libras_caption_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/desktop-redesign-acceptance-20261009"))
    parser.add_argument("--camera", type=int, help="Optional hardware camera index; lifecycle validation only")
    parser.add_argument("--camera-seconds", type=float, default=2.0, help="Camera recording duration (0.1 to 10 seconds)")
    args = parser.parse_args(argv)
    report = validate(args.model, args.video, args.catalog, args.output,
                      camera=args.camera, camera_seconds=args.camera_seconds)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
