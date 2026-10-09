"""Real TCP/video acceptance; starts and stops its own local API process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    import httpx

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [sys.executable, "-m", "slbridge.api_cli", "--catalog", str(args.catalog.resolve()),
               "--model", "libras-rf", "--device", "cpu", "--port", str(port)]
    result = {"transport": "TCP HTTP", "input": str(args.input.resolve()), "steps": []}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=flags)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("API process exited; inspect server.log")
                    try:
                        health = client.get("/v1/health")
                        health.raise_for_status()
                        break
                    except httpx.ConnectError:
                        if time.monotonic() > deadline:
                            raise RuntimeError("API startup timed out")
                        time.sleep(.1)
                result["health"] = health.json()
                schema = client.get("/openapi.json")
                schema.raise_for_status()
                assert "/v1/recognitions" in schema.json()["paths"]
                for alias in ("libras-rf", "libras-onnx"):
                    if alias == "libras-onnx":
                        switched = client.post("/v1/models/switch", json={"model_id": alias, "device": "cpu"})
                        switched.raise_for_status()
                        result["steps"].append({"switch": switched.json()})
                    started = time.perf_counter()
                    with args.input.open("rb") as source:
                        response = client.post("/v1/recognitions", files={"file": (args.input.name, source, "video/mp4")},
                                               data={"request_id": f"tcp-{alias}", "sign_language": "libras",
                                                     "representation_id": "mediapipe225_rgb_v1"})
                    response.raise_for_status()
                    content = response.json()
                    assert content["request_id"] == content["result"]["sample_id"] == f"tcp-{alias}"
                    result["steps"].append({"alias": alias, "http_wall_ms": (time.perf_counter() - started) * 1000,
                                            "response": content})
                a, b = [step["response"]["result"] for step in result["steps"] if "response" in step]
                assert a["status"] == b["status"]
                if a["prediction"] is not None:
                    assert a["prediction"]["label_id"] == b["prediction"]["label_id"]
                rollback = client.post("/v1/models/rollback")
                rollback.raise_for_status()
                assert rollback.json()["status"]["active"]["backend"] == "random_forest_v1"
                result["steps"].append({"rollback": rollback.json()})
                failed = client.post("/v1/models/switch", json={"model_id": "libras-rf", "device": "cuda"})
                assert failed.status_code == 422
                state = client.get("/v1/health").json()["status"]
                assert state["active"]["backend"] == "random_forest_v1" and state["generation"] == 3
                result["steps"].append({"strict_cuda_rf_error": failed.json(), "preserved_generation": 3})
                result["passed"] = True
        finally:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)
    (args.output / "api_video_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": True, "artifact": str(args.output / "api_video_validation.json"),
                      "timing": [step["response"]["timing_ms"] for step in result["steps"] if "response" in step]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
