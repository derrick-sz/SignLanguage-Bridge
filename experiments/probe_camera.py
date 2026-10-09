"""Read two frames and release a local camera; never save camera pixels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from slbridge._io import configure_utf8_streams, write_json_atomic


def main(argv=None):
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description="Probe local camera capture/release without saving images")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="artifacts/desktop-acceptance-20261009/camera-probe.json")
    args = parser.parse_args(argv)
    if args.worker:
        from slbridge.sources import CameraSource
        try:
            with CameraSource(args.camera) as source:
                frames = [source.read(), source.read()]
                report = {"status": "available", "device": args.camera, "frames": len(frames),
                          "shape": list(frames[0].image.shape),
                          "timestamps_increase": frames[1].timestamp > frames[0].timestamp}
            report["released"] = True
        except Exception as error:
            report = {"status": "unavailable", "device": args.camera, "reason": str(error)}
        print(json.dumps(report, ensure_ascii=False))
        return 0
    try:
        child = subprocess.run([sys.executable, "-m", "experiments.probe_camera", "--camera", str(args.camera),
                                "--worker"], capture_output=True, text=True, encoding="utf-8", timeout=15)
        if child.returncode:
            report = {"status": "error", "device": args.camera, "reason": child.stderr[-2000:]}
        else:
            report = json.loads(child.stdout)
    except subprocess.TimeoutExpired:
        report = {"status": "timeout", "device": args.camera, "reason": "camera open/read exceeded 15 seconds"}
    report["accuracy_validation"] = False
    report["camera_pixels_saved"] = False
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
