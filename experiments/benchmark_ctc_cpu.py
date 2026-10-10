"""Measure real, complete CTC file inference and save full logits for parity."""
from __future__ import annotations

import argparse
from pathlib import Path
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
    parser.add_argument("--cpu-conv-subnormals", choices=("flush_to_zero", "preserve"), default="flush_to_zero")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reference-logits", type=Path)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error("runs must be 1..10")
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    options = RuntimeOptions(device="cpu", intra_op_threads=args.threads,
                             cpu_conv_subnormals=args.cpu_conv_subnormals)
    report = {"model": str(args.model.resolve()), "video": str(args.video.resolve()),
              "video_sha256": sha256_file(args.video), "options": options.to_dict(),
              "stage": "loading", "runs": []}
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
        write_json_atomic(path, report)
        backend = manager.recognizer.bundle.backend
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
                    f"ctc-cpu-{index + 1}", expected_task_type="gloss_sequence",
                    expected_generation=manager.generation))
                if response.result.status != "recognized":
                    raise AssertionError(response.result.to_dict())
                values = logits[0]
                np.save(args.output / f"logits-{index + 1}.npy", values)
                entry = response.to_dict()
                if args.reference_logits:
                    reference = np.load(args.reference_logits, allow_pickle=False)
                    assert values.shape == reference.shape, "Full sequence logits shape changed"
                    difference = np.abs(values.astype(np.float64) - reference.astype(np.float64))
                    parity = {"shape": list(values.shape), "rtol": 1e-4, "atol": 1e-4,
                              "max_absolute_error": float(difference.max(initial=0)),
                              "allclose": bool(np.allclose(values, reference, rtol=1e-4, atol=1e-4)),
                              "argmax_frame_equal": bool(np.array_equal(values.argmax(-1), reference.argmax(-1)))}
                    entry["parity"] = parity
                    if not parity["allclose"] or not parity["argmax_frame_equal"]:
                        raise AssertionError(parity)
                report["runs"].append(entry)
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
    print("CTC CPU benchmark:", path.resolve())


if __name__ == "__main__":
    main()
