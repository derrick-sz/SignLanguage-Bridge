"""Bounded real camera lifecycle probe; no camera pixels saved or accuracy claim."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from slbridge._io import configure_utf8_streams, write_json_atomic
from slbridge.session import RecognitionSession
from slbridge.sources import CameraSource
from slbridge.runtime_options import RuntimeOptions


def main():
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/auto-camera-probe.json"))
    args = parser.parse_args()
    if not .5 <= args.seconds <= 10 or args.camera < 0:
        parser.error("seconds must be .5..10 and camera >=0")
    session = RecognitionSession(options=RuntimeOptions(device="cpu"))
    evidence = {"camera_pixels_saved": False, "accuracy_validation": False, "action_ground_truth": None}
    try:
        session.load_model(args.model)
        limit = time.monotonic() + 30
        while session.busy and time.monotonic() < limit:
            time.sleep(.01)
        if session.busy or session.recognizer is None:
            raise RuntimeError("Camera model could not finish loading")
        session.open_source(CameraSource(args.camera), kind="camera")
        limit = time.monotonic() + 10
        while not session.source_active and time.monotonic() < limit:
            time.sleep(.01)
        if not session.source_active:
            raise RuntimeError("Camera unavailable")
        session.enable_automatic()
        deadline = time.monotonic() + args.seconds
        events = []
        while time.monotonic() < deadline:
            for event in session.poll():
                # Candidate transitions only; do not save images or identities.
                if event.kind in {"candidate", "error"}:
                    events.append({"kind": event.kind, "value": event.value})
            time.sleep(.02)
        evidence.update(status=session.automatic_status(), transitions=events,
                        committed_count=len(session.captions.snapshot()["events"]), model=session.model_manager.describe())
        if evidence["status"]["observed_frames"] < 2:
            raise AssertionError("Actual camera observer did not process two frames")
        if any(event["kind"] == "error" for event in events):
            raise AssertionError(events)
        evidence["lifecycle"] = "capture_and_actual_observer_passed"
    finally:
        session.close()
        limit = time.monotonic() + 10
        while any(worker.is_alive() for worker in (session._capture_thread, session._observer_worker, session._worker)) and time.monotonic() < limit:
            time.sleep(.01)
        evidence["workers_released"] = not any(w.is_alive() for w in (session._capture_thread, session._observer_worker, session._worker))
        write_json_atomic(args.output, evidence)
    if not evidence["workers_released"]:
        raise RuntimeError("Camera/model workers did not finish within release bound")
    print("Camera lifecycle probe saved to", args.output.resolve())


if __name__ == "__main__":
    main()
