"""Verify frozen RF/ONNX parity and warmed, single-request runtime latency.

This acceptance run uses the native bundle's existing test split and cached
landmark archives. It reports runtime parity and timings, not a new accuracy
estimate. No training, export, video extraction, or model publication occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import platform
import sys
import time
from typing import Any

from slbridge._io import configure_utf8_streams, read_json, resolve_resource, sha256_file, write_json_atomic
from slbridge.backends import dependency_version
from slbridge.contracts import DatasetManifest, SplitManifest
from slbridge.data import validate_split
from slbridge.features import prepare_input
from slbridge.inference import Recognizer
from slbridge.runtime_options import RuntimeOptions


def percentiles(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    import numpy as np

    if not rows:
        raise ValueError("Cannot summarize empty timing observations")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Profiling phase names changed during acceptance")
    return {key: {"p50": float(np.percentile([row[key] for row in rows], 50)),
                  "p95": float(np.percentile([row[key] for row in rows], 95)),
                  "min": float(min(row[key] for row in rows)),
                  "max": float(max(row[key] for row in rows))}
            for key in sorted(keys)}


def result_signature(value: Any) -> dict:
    prediction = value.prediction
    return {"status": value.status, "sample_id": value.sample_id,
            "sign_language": value.sign_language, "reason": value.reason,
            "quality": value.quality, "time_range": value.time_range,
            "task_type": value.task_type, "score_type": value.score_type,
            "label_id": prediction.label_id if prediction else None,
            "display_text": prediction.display_text if prediction else None}


def load_timed(path: Path, options: RuntimeOptions) -> tuple[Recognizer, float]:
    started = time.perf_counter_ns()
    result = Recognizer.load(path, options=options)
    return result, (time.perf_counter_ns() - started) / 1e6


def verify(native_path: Path, onnx_path: Path, output: Path, *, device: str = "cpu",
           repeats: int = 3, warmup_runs: int = 5, tolerance: float = 1e-6,
           cuda_dll_dirs: tuple[str, ...] = ()) -> dict:
    import numpy as np
    # This acceptance explicitly requires ORT. Import it before sklearn's
    # native DLLs; the Windows Anaconda host otherwise fails DLL initialization.
    import onnxruntime

    native_path, onnx_path, output = native_path.resolve(), onnx_path.resolve(), output.resolve()
    if type(repeats) is not int or repeats < 1 or type(warmup_runs) is not int or warmup_runs < 1:
        raise ValueError("repeats and warmup_runs must be positive integers")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)
    native, native_load_ms = load_timed(native_path, RuntimeOptions(device="cpu", intra_op_threads=1))
    converted, onnx_load_ms = load_timed(onnx_path, RuntimeOptions(device=device, intra_op_threads=1,
                                                                cuda_dll_dirs=cuda_dll_dirs))
    native_manifest, onnx_manifest = native.bundle.manifest, converted.bundle.manifest
    if native_manifest.backend != "random_forest_v1" or onnx_manifest.backend != "onnx_classifier_v1":
        raise ValueError("Acceptance requires native RF and exported ONNX bundles")
    if (native_manifest.input_spec != onnx_manifest.input_spec
            or native_manifest.dataset_hash != onnx_manifest.dataset_hash
            or native_manifest.split_hash != onnx_manifest.split_hash
            or native.bundle.labels != converted.bundle.labels
            or native.bundle.rejection_threshold != converted.bundle.rejection_threshold
            or native.bundle.backend.classes != converted.bundle.backend.classes):
        raise ValueError("Export changed preprocessing, frozen split, labels, threshold, or class order")
    dataset = DatasetManifest.from_dict(read_json(native_path / "splits/dataset.json"))
    split = SplitManifest.from_dict(read_json(native_path / "splits/split.json"))
    validate_split(dataset, split)
    if not split.test:
        raise ValueError("Frozen test split is empty")
    by_id = {record.sample_id: record for record in dataset.samples}
    inputs = []
    for identifier in split.test:
        record = by_id[identifier]
        path = resolve_resource(dataset.root, record.resource)
        if sha256_file(path) != record.content_hash:
            raise ValueError(f"Frozen held-out archive hash mismatch: {identifier}")
        inputs.append((identifier, path, record.content_hash))
    print(f"Verify export probabilities and recognition decisions on {len(inputs)} frozen test archives; this is not a new accuracy estimate", flush=True)
    prepared = [prepare_input(path, native_manifest.input_spec) for _, path, _ in inputs]
    native_probabilities = native.bundle.backend.predict_proba(prepared)
    converted_probabilities = converted.bundle.backend.predict_proba(prepared)
    numeric_error = float(np.max(np.abs(native_probabilities - converted_probabilities)))
    np.testing.assert_allclose(converted_probabilities, native_probabilities, rtol=0, atol=tolerance)
    mismatches = []
    recognition_counts = Counter()
    for identifier, path, _ in inputs:
        native_result = native.recognize(path, sample_id=identifier)
        converted_result = converted.recognize(path, sample_id=identifier)
        recognition_counts[native_result.status] += 1
        if result_signature(native_result) != result_signature(converted_result):
            mismatches.append({"sample_id": identifier, "native": result_signature(native_result),
                               "onnx": result_signature(converted_result)})
        if native_result.prediction is not None and converted_result.prediction is not None:
            if abs(native_result.prediction.score - converted_result.prediction.score) > tolerance:
                mismatches.append({"sample_id": identifier, "reason": "selected_score_mismatch"})
    if mismatches:
        write_json_atomic(output / "parity_mismatches.json", {"mismatches": mismatches})
        raise AssertionError(f"Recognition decisions changed for {len(mismatches)} frozen inputs")
    # A separate synthetic quality case exercises rejection semantics; it never
    # enters export verification, frozen evaluation, or latency statistics.
    rejected_path = output / "synthetic_missing_coordinates.npz"
    if inputs[0][1].suffix.lower() == ".npz":
        with np.load(inputs[0][1], allow_pickle=False) as archive:
            rejected_archive = {key: archive[key].copy() for key in archive.files}
        rejected_archive["values"].fill(-999)
        np.savez_compressed(rejected_path, **rejected_archive)
    else:
        raise ValueError("Quality-rejection acceptance currently requires timestamped .npz archives")
    native_rejected = native.recognize(rejected_path, sample_id="synthetic:all_missing")
    onnx_rejected = converted.recognize(rejected_path, sample_id="synthetic:all_missing")
    if native_rejected.status != "rejected" or result_signature(native_rejected) != result_signature(onnx_rejected):
        raise AssertionError("Quality rejection behavior changed after export")
    warmup_ms = {}
    for name, recognizer in (("native_rf", native), ("onnx", converted)):
        started = time.perf_counter_ns()
        recognizer.warmup()
        for _ in range(warmup_runs):
            recognizer.recognize(inputs[0][1], sample_id=inputs[0][0])
        warmup_ms[name] = (time.perf_counter_ns() - started) / 1e6
    observations: dict[str, list[dict[str, float]]] = {"native_rf": [], "onnx": []}
    print(f"Measure {repeats} passes over all {len(inputs)} cached inputs after warmup, with one request at a time and one inference thread", flush=True)
    # Alternate execution order to reduce drift from machine background load.
    for repeat in range(repeats):
        order = (("native_rf", native), ("onnx", converted))
        if repeat % 2:
            order = order[::-1]
        for identifier, path, _ in inputs:
            for name, recognizer in order:
                result, phases = recognizer.recognize_profiled(path, sample_id=identifier)
                if any(not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0 for value in phases.values()):
                    raise ValueError("Runtime returned invalid profiling phases")
                observations[name].append(dict(phases))
        print(f"Completed timing pass {repeat + 1}/{repeats}", flush=True)
    summary = {"status": "passed", "scope": "Export numerical parity, prediction/rejection parity, cached-input runtime latency",
               "accuracy_claim": None, "native_model": str(native_path), "onnx_model": str(onnx_path),
               "native_model_manifest_sha256": sha256_file(native_path / "manifest.json"),
               "onnx_model_manifest_sha256": sha256_file(onnx_path / "manifest.json"),
               "dataset_hash": native_manifest.dataset_hash, "split_hash": native_manifest.split_hash,
               "test_input_count": len(inputs), "all_test_input_hashes_verified": True,
               "probability_max_absolute_error": numeric_error, "absolute_tolerance": tolerance,
               "prediction_or_rejection_mismatches": len(mismatches), "native_decision_counts": dict(recognition_counts),
               "quality_rejection": result_signature(native_rejected),
               "runtime_info": {"native_rf": native.runtime_info, "onnx": converted.runtime_info},
               "load_ms": {"native_rf": native_load_ms, "onnx": onnx_load_ms}, "warmup_ms": warmup_ms,
               "benchmark": {"input_kind": dataset.input_kind, "cache_state": "existing archives and warmed OS file cache",
                             "includes_rgb_video_decode_and_landmark_extraction": False,
                             "batch_size": 1, "concurrency": 1, "intra_op_threads": 1,
                             "repeats": repeats, "warmup_recognitions_per_backend": warmup_runs,
                             "observations_per_backend": len(observations["native_rf"]),
                             "timing_ms": {key: percentiles(rows) for key, rows in observations.items()}},
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "onnxruntime_preloaded_before_native_model": True,
                               "dependencies": {name: dependency_version(name) for name in
                                                ("numpy", "scikit-learn", "joblib", "onnxruntime")}},
               "inputs": [{"sample_id": identifier, "path": str(path), "sha256": digest}
                          for identifier, path, digest in inputs]}
    write_json_atomic(output / "runtime_validation.json", summary)
    write_json_atomic(output / "timing_observations.json", observations)
    print(f"RF/ONNX parity passed on {len(inputs)} frozen test archives; maximum probability error {numeric_error:.3g}", flush=True)
    native.close()
    converted.close()
    return summary


def main() -> None:
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-dll-dir", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        verify(args.native_model, args.onnx_model, args.output, device=args.device,
               repeats=args.repeats, warmup_runs=args.warmup_runs, tolerance=args.absolute_tolerance,
               cuda_dll_dirs=tuple(str(path.resolve()) for path in args.cuda_dll_dir))
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output / "runtime_validation_failure.json",
                          {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
