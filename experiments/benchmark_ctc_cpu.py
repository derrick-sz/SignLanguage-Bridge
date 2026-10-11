"""Measure complete CTC file inference on CPU/CUDA and retain full logits."""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import platform
import sys
import time

from slbridge._io import configure_utf8_streams, sha256_file, write_json_atomic
from slbridge.runtime import ModelManager
from slbridge.runtime_contracts import InferenceRequest
from slbridge.runtime_options import RuntimeOptions


def main():
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=RuntimeOptions().intra_op_threads)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=RuntimeOptions().gpu_mem_limit_mb)
    parser.add_argument("--cuda-dll-dir", type=Path, action="append", default=[])
    parser.add_argument("--cpu-conv-subnormals", choices=("flush_to_zero", "preserve"), default="flush_to_zero")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reference-logits", type=Path)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error("runs must be 1..10")
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    options = RuntimeOptions(device=args.device, device_id=args.device_id,
                             intra_op_threads=args.threads, gpu_mem_limit_mb=args.gpu_mem_limit_mb,
                             cpu_conv_subnormals=args.cpu_conv_subnormals,
                             cuda_dll_dirs=tuple(str(path.resolve()) for path in args.cuda_dll_dir))
    report = {"model": str(args.model.resolve()), "video": str(args.video.resolve()),
              "video_sha256": sha256_file(args.video), "options": options.to_dict(),
              "stage": "loading", "runs": []}
    packages = {}
    for name in ("numpy", "opencv-contrib-python", "opencv-python-headless", "onnx", "protobuf", "onnxruntime",
                 "onnxruntime-gpu", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report["environment"] = {"python": sys.executable, "python_version": platform.python_version(),
                             "platform": platform.platform(), "packages": packages}
    path = args.output / "benchmark.json"
    write_json_atomic(path, report)
    manager = ModelManager(options=options)
    try:
        began = time.perf_counter()
        manager.switch(args.model)
        if manager.capabilities()["task_type"] != "gloss_sequence":
            raise ValueError("This benchmark requires an actual CTC sequence model")
        report.update(load_and_warmup_ms=(time.perf_counter() - began) * 1000,
                      model_description=manager.describe(), stage="ready")
        if args.device == "cuda" and report["model_description"]["runtime"]["device"] != "cuda":
            raise AssertionError("Strict CUDA benchmark did not activate CUDAExecutionProvider")
        backend = manager.recognizer.bundle.backend
        report.update(model_sha256=sha256_file(args.model / "model.onnx"),
                      manifest_sha256=sha256_file(args.model / "manifest.json"),
                      provider_options=backend._session.get_provider_options())
        if args.reference_logits:
            report["reference_logits"] = {"path": str(args.reference_logits.resolve()),
                                          "sha256": sha256_file(args.reference_logits)}
        write_json_atomic(path, report)
        original_predict = backend.predict_logits
        logits = []

        def capture(prepared):
            value = original_predict(prepared)
            logits[:] = [value.copy()]
            return value

        backend.predict_logits = capture
        try:
            for index in range(args.runs):
                report["stage"] = f"inference_{index + 1}"
                write_json_atomic(path, report)
                response = manager.infer_file(args.video, InferenceRequest(
                    f"ctc-{args.device}-{index + 1}", expected_task_type="gloss_sequence",
                    expected_generation=manager.generation))
                if response.result.status != "recognized":
                    raise AssertionError(response.result.to_dict())
                values = logits[0]
                np.save(args.output / f"logits-{index + 1}.npy", values)
                entry = response.to_dict()
                report["runs"].append(entry)
                if args.reference_logits:
                    from slbridge.sequence import CTCDecoderConfig, greedy_decode, prefix_beam_decode

                    reference = np.load(args.reference_logits, allow_pickle=False)
                    entry["parity"] = {"shape": list(values.shape), "reference_shape": list(reference.shape)}
                    assert values.shape == reference.shape, "Full sequence logits shape changed"
                    difference = np.abs(values.astype(np.float64) - reference.astype(np.float64))
                    config = CTCDecoderConfig.from_dict(manager.recognizer.bundle.manifest.parameters)
                    vocabulary = manager.recognizer.bundle.vocabulary
                    decoded = (greedy_decode(reference, vocabulary) if config.decoder == "greedy" else
                               prefix_beam_decode(reference, vocabulary, beam_width=config.beam_width,
                                                  token_top_k=config.token_top_k))
                    parity = {"shape": list(values.shape), "rtol": 1e-4, "atol": 1e-4,
                              "max_absolute_error": float(difference.max(initial=0)),
                              "allclose": bool(np.allclose(values, reference, rtol=1e-4, atol=1e-4)),
                              "argmax_frame_equal": bool(np.array_equal(values.argmax(-1), reference.argmax(-1))),
                              "decoded_token_ids_equal": [token.token_id for token in response.result.prediction.tokens]
                                  == [token.token_id for token in decoded.tokens]}
                    entry["parity"] = parity
                    if not all(parity[key] for key in ("allclose", "argmax_frame_equal", "decoded_token_ids_equal")):
                        raise AssertionError(parity)
                write_json_atomic(path, report)
                print(f"Run {index+1}: {response.timing_ms}", flush=True)
        finally:
            backend.predict_logits = original_predict
        report["stage"] = "complete"
    except BaseException as error:
        report["stage"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manager.close()
        write_json_atomic(path, report)
    print("CTC runtime benchmark:", path.resolve())


if __name__ == "__main__":
    main()
