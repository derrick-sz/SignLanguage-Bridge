"""Probe installed providers and real saved models without training/downloads."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from slbridge._io import configure_utf8_streams, write_json_atomic
from slbridge.runtime import ModelManager, ModelRouter
from slbridge.runtime_contracts import InferenceRequest
from slbridge.runtime_options import RuntimeOptions


def main():
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libras-model", type=Path, required=True)
    parser.add_argument("--libras-video", type=Path, required=True)
    parser.add_argument("--csl-model", type=Path)
    parser.add_argument("--csl-video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/unified-runtime-probe.json"))
    args = parser.parse_args()
    import onnxruntime as ort

    report = {"onnxruntime_version": ort.__version__, "available_providers": ort.get_available_providers()}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        report["nvidia_smi"] = {"returncode": gpu.returncode, "stdout": gpu.stdout.strip(), "stderr": gpu.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi"] = {"error": str(error)}
    router = ModelRouter()
    try:
        libras = ModelManager(options=RuntimeOptions(device="auto"))
        try:
            libras.switch(args.libras_model)
            router.register("libras", libras)
        except BaseException:
            libras.close()
            raise
        if args.csl_model:
            csl = ModelManager(options=RuntimeOptions(device="cpu"))
            try:
                csl.switch(args.csl_model)
                router.register("csl", csl)
            except BaseException:
                csl.close()
                raise
        report["capabilities"] = router.capabilities()
        report["stage"] = "models_loaded"
        write_json_atomic(args.output, report)
        report["libras"] = router.infer_file(args.libras_video, InferenceRequest(
            "provider-probe-libras", sign_language="libras", expected_generation=libras.generation)).to_dict()
        report["stage"] = "libras_completed"
        write_json_atomic(args.output, report)
        if args.csl_video:
            report["stage"] = "csl_inferencing"
            write_json_atomic(args.output, report)
            report["csl"] = router.infer_file(args.csl_video, InferenceRequest(
                "provider-probe-csl", sign_language="csl", expected_task_type="gloss_sequence",
                expected_generation=router.manager_for("csl").generation)).to_dict()
        explicit_cuda = ModelManager(options=RuntimeOptions(device="cuda"))
        try:
            explicit_cuda.switch(args.libras_model)
            report["explicit_cuda"] = {"status": "loaded", "description": explicit_cuda.describe()}
        except Exception as error:
            report["explicit_cuda"] = {"status": "incompatible", "code": type(error).__name__, "message": str(error)}
        finally:
            explicit_cuda.close()
        report["stage"] = "complete"
    finally:
        router.close()
        write_json_atomic(args.output, report)
    print("Runtime probe saved to", args.output.resolve())


if __name__ == "__main__":
    main()
