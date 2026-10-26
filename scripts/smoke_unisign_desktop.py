"""Offscreen real Qt smoke test for the optional Uni-Sign desktop path."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pump(app, predicate, seconds):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(.02)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real Qt desktop path with Uni-Sign")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--recognize-timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not (args.bundle / "manifest.json").is_file():
        parser.error(f"Uni-Sign bundle is missing: {args.bundle}")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from slbridge.desktop import SignLanguageApp
    from slbridge.runtime_options import RuntimeOptions

    app = QApplication.instance() or QApplication([])
    window = SignLanguageApp(
        model=args.bundle, video=args.video, unisign_model=args.bundle,
        runtime_options=RuntimeOptions(device=args.device), defer_startup_input=True)
    window.show()
    loaded = pump(app, lambda: window.session.recognizer is not None and not window.session.busy,
                  args.model_timeout)
    source_active = pump(app, lambda: window.session.source_active, 20.0) if loaded else False
    result_received = False
    if loaded and source_active and not window._error_detail:
        window.recognize_video()
        result_received = pump(
            app, lambda: window._last_result is not None and not window.session.busy,
            args.recognize_timeout)
    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "bundle": str(args.bundle.resolve()),
        "video": str(args.video.resolve()),
        "device": args.device,
        "loaded": loaded,
        "source_active": source_active,
        "result_received": result_received,
        "title": window.result_title.text(),
        "zh": window.zh_label.text(),
        "pt": window.pt_label.text(),
        "status": window.status_label.text(),
        "language_label": window.language_label.text(),
        "model_path": window._model_path,
        "error_detail": window._error_detail,
        "details": window._result_details,
    }
    window.close()
    app.processEvents()
    output_path = (args.output or ROOT / "artifacts/unisign-desktop" /
                   f"desktop-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **{key: output[key] for key in (
        "loaded", "source_active", "result_received", "zh", "error_detail")}}, ensure_ascii=False, indent=2))
    return 0 if all(output[key] for key in ("loaded", "source_active", "result_received")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
