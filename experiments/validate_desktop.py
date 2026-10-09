"""Exercise mounted Tk controls with a real saved model and video.

Camera replay checks are engineering fixtures, never hardware accuracy claims.
All evidence is saved under the caller's ignored artifacts directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from slbridge._io import configure_utf8_streams, write_json_atomic


def validate(model: Path, video: Path, catalog: Path, output: Path) -> dict:
    import tkinter as tk

    from slbridge.desktop import SignLanguageApp

    output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    app = SignLanguageApp(root, model=model)
    started = time.monotonic()
    updates = 0

    def pump(condition, timeout=60):
        nonlocal updates
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            root.update()
            updates += 1
            if condition():
                return
            if app.session.state == "ERROR":
                raise AssertionError(app.status_var.get())
            time.sleep(.01)
        raise TimeoutError(f"Desktop did not finish: {app.status_var.get()}")

    try:
        app.load_catalog(catalog)
        pump(lambda: app.session.recognizer is not None and not app.session.busy)
        with patch("slbridge.desktop.filedialog.askopenfilename", return_value=str(video)):
            app.video_button.invoke()
        pump(lambda: app._photo is not None)
        app.recognize_button.invoke()
        processing_updates = updates
        pump(lambda: app.caption is not None)
        first = app.caption.to_dict()
        assert first["status"] == "recognized", first
        assert app.zh_label.cget("text") == first["zh"]
        assert app.pt_label.cget("text") == first["pt"]
        assert "草案" in app.notice_var.get()
        assert updates > processing_updates
        assert app.zh_label.winfo_ismapped() and app.pt_label.winfo_ismapped()

        # Save an edit and route it through the actual reload control.
        data = json.loads(catalog.read_text(encoding="utf-8"))
        entry = next(item for item in data["labels"] if item["label_id"] == first["label_id"])
        entry["display_text"]["zh-CN"] = "验收编辑：" + first["zh"]
        edited = output / "edited-caption-catalog.json"
        write_json_atomic(edited, data)
        with patch("slbridge.desktop.filedialog.askopenfilename", return_value=str(edited)):
            app.catalog_button.invoke()
        root.update()
        assert app.zh_label.cget("text") == entry["display_text"]["zh-CN"]
        app.load_catalog(catalog)
        root.update()

        screenshot = None
        try:
            from PIL import ImageGrab
            # Use a native window handle rather than logical Tk coordinates:
            # Windows display scaling otherwise crops the bottom captions.
            root.lift()
            root.update()
            shot = ImageGrab.grab(window=int(root.wm_frame(), 0))
            screenshot = output / "desktop-file-caption.png"
            shot.save(screenshot)
        except (OSError, ImportError, ValueError) as error:
            screenshot_error = str(error)
        else:
            screenshot_error = None

        app.cancel_button.invoke()
        root.update()
        assert app.caption is None and app.pt_var.get() == ""
        app.stop_button.invoke()
        pump(lambda: not app.session.source_active)
        report = {
            "stage": "complete", "model": str(model.resolve()), "video": str(video.resolve()),
            "file_caption": first, "mounted_caption_labels": True,
            "saved_dictionary_edit_changes_display": True, "cancel_clears_caption": True,
            "updates_during_inference": updates - processing_updates,
            "seconds": round(time.monotonic() - started, 3),
            "screenshot": str(screenshot.resolve()) if screenshot else None,
            "screenshot_error": screenshot_error,
            "camera_hardware": "not tested by this file acceptance",
        }
    finally:
        app.on_close()
        app.session._capture_thread.join(5)
        app.session._worker.join(5)
    assert not app.session._capture_thread.is_alive()
    assert not app.session._worker.is_alive()
    report["workers_released"] = True
    write_json_atomic(output / "desktop-validation.json", report)
    return report


def main(argv=None):
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description="Validate actual desktop controls with a real video/model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/libras_caption_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/desktop-acceptance-20261009"))
    args = parser.parse_args(argv)
    report = validate(args.model, args.video, args.catalog, args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
