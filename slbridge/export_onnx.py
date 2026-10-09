"""Optional RF-to-ONNX export, verified on hash-pinned frozen holdout clips.

The source bundle and its original trained weights are never overwritten. The
destination is published only after dense-output parity and bundle readback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
from typing import Sequence
import uuid

from ._io import configure_utf8_streams, read_json, resolve_resource, sha256_file, write_json_atomic
from .contracts import DatasetManifest, ModelManifest, SplitManifest


def _isolated_export(model_path: str | Path, output_path: str | Path, run_id: str,
                     verification_inputs: Sequence[str | Path]) -> Path:
    """Avoid a Windows native DLL conflict if a caller already initialized RF."""
    payload = {"model_path": str(Path(model_path).resolve()), "output_path": str(Path(output_path).resolve()),
               "run_id": run_id, "verification_inputs": [str(Path(path).resolve()) for path in verification_inputs]}
    bootstrap = """import json, sys
try:
    import onnx, onnxruntime, skl2onnx
    from slbridge.export_onnx import export_rf_bundle
    args = json.load(sys.stdin)
    path = export_rf_bundle(**args)
    result = {"path": str(path)}
except Exception as error:
    result = {"error": type(error).__name__, "message": str(error)}
print("SLBRIDGE_EXPORT_RESULT=" + json.dumps(result))
"""
    process = subprocess.run([sys.executable, "-c", bootstrap], input=json.dumps(payload),
                             text=True, encoding="utf-8", capture_output=True,
                             cwd=Path(__file__).resolve().parents[1])
    result_lines = [line for line in process.stdout.splitlines() if line.startswith("SLBRIDGE_EXPORT_RESULT=")]
    if process.returncode != 0 or not result_lines:
        raise ValueError(f"Isolated ONNX export failed: {process.stderr.strip() or process.stdout.strip()}")
    result = json.loads(result_lines[-1].split("=", 1)[1])
    if "error" in result:
        exception = {"FileExistsError": FileExistsError, "OSError": OSError, "RuntimeError": RuntimeError}.get(result["error"], ValueError)
        raise exception(result["message"])
    return Path(result["path"])


def export_rf_bundle(model_path: str | Path, output_path: str | Path, run_id: str,
                     verification_inputs: Sequence[str | Path]) -> Path:
    """Export to the exact output_path using real frozen validation/test inputs.

    Verification clips may be relocated but their bytes must match the source
    bundle's frozen holdout records. Synthetic/random inputs and training clips
    cannot qualify a published export. This proves parity on the supplied subset,
    not improved accuracy or GPU operator placement.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid run_id")
    if isinstance(verification_inputs, (str, bytes)) or not verification_inputs:
        raise ValueError("At least one frozen held-out verification input is required")
    source_path, final = Path(model_path).resolve(), Path(output_path).resolve()
    if final.exists():
        raise FileExistsError(f"Model version already exists: {final}")
    if final.is_relative_to(source_path) or source_path.is_relative_to(final):
        raise ValueError("Export output must be separate from its source model")
    if (os.name == "nt" and "sklearn" in sys.modules
            and ("onnx" not in sys.modules or "onnxruntime" not in sys.modules)):
        return _isolated_export(source_path, final, run_id, verification_inputs)
    try:
        # ONNX must initialize before skl2onnx imports sklearn on Windows;
        # ORT must likewise initialize before native sklearn/OpenMP libraries.
        import onnx
        import onnxruntime
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as error:
        raise ValueError("RF export requires the optional skl2onnx export dependency") from error
    from .backends import COMMON_ARTIFACTS, dependency_version
    from .bundle import load_bundle
    from .features import prepare_input
    from .onnx_backend import OnnxClassifierBackend
    from .runtime_options import RuntimeOptions
    source = load_bundle(source_path, options=RuntimeOptions(device="cpu"))
    if source.manifest.backend != "random_forest_v1":
        source.backend.close()
        raise ValueError("RF export requires a random_forest_v1 source bundle")
    staging: Path | None = None
    converted_backend = None
    restored = None
    try:
        dataset = DatasetManifest.from_dict(read_json(source_path / "splits/dataset.json"))
        split = SplitManifest.from_dict(read_json(source_path / "splits/split.json"))
        if final.is_relative_to(Path(dataset.root).resolve()):
            raise ValueError("Export output must be outside the read-only source dataset")
        heldout_ids = set(split.validation) | set(split.test)
        heldout = {record.content_hash: record for record in dataset.samples if record.sample_id in heldout_ids}
        samples, records, seen_hashes = [], [], set()
        for input_path in verification_inputs:
            input_path = Path(input_path).resolve()
            digest = sha256_file(input_path)
            if digest not in heldout:
                raise ValueError(f"Verification input is not a hash-pinned frozen held-out clip: {input_path}")
            if digest in seen_hashes:
                raise ValueError("Duplicate held-out verification input")
            seen_hashes.add(digest)
            record = heldout[digest]
            samples.append(prepare_input(input_path, source.manifest.input_spec))
            records.append({"sample_id": record.sample_id, "content_hash": digest,
                            "split": "validation" if record.sample_id in split.validation else "test"})
        import numpy as np

        model = convert_sklearn(
            source.backend._estimator,
            initial_types=[("features", FloatTensorType([None, source.manifest.input_spec.feature_size]))],
            options={id(source.backend._estimator): {"zipmap": False}}, target_opset=17,
        )
        if len(model.graph.output) != 2:
            raise ValueError("RF converter did not produce labels and a dense probability tensor")
        bindings = {"schema_version": 1, "input_layout": "feature_vector", "values_input": "features",
                    "mask_input": None, "mask_dtype": "float32", "output_name": model.graph.output[1].name,
                    "output_kind": "probabilities"}
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.parent / f".{final.name}.{uuid.uuid4().hex}.partial"
        staging.mkdir()
        weight_path = staging / "weights/model.onnx"
        weight_path.parent.mkdir()
        with weight_path.open("xb") as stream:
            stream.write(model.SerializeToString())
        converted_backend = OnnxClassifierBackend.load(
            weight_path, expected_classes=source.backend.classes,
            expected_input_spec=source.manifest.input_spec, runtime_config=bindings,
            options=RuntimeOptions(device="cpu"),
        )
        converted_backend.warmup()
        expected = source.backend.predict_proba(samples)
        actual = converted_backend.predict_proba(samples)
        absolute_tolerance, relative_tolerance = 1e-6, 1e-5
        if not np.allclose(expected, actual, rtol=relative_tolerance, atol=absolute_tolerance):
            raise ValueError(f"ONNX held-out probability parity failed: max_absolute_error={float(np.abs(expected - actual).max())}")
        if not np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1)):
            raise ValueError("ONNX held-out label parity failed")
        threshold = source.rejection_threshold
        if not np.array_equal(expected.max(axis=1) >= threshold, actual.max(axis=1) >= threshold):
            raise ValueError("ONNX held-out rejection-threshold parity failed")
        for resource in COMMON_ARTIFACTS - {"resolved_config.json"}:
            destination = staging / resource
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolve_resource(source_path, resource), destination)
        parameters = {"source_backend": source.manifest.backend, "source_parameters": source.manifest.parameters,
                      "converter": {"name": "skl2onnx", "version": importlib.metadata.version("skl2onnx"),
                                    "target_opset": 17, "zipmap": False}}
        write_json_atomic(staging / "resolved_config.json", {
            "schema_version": 1, "backend": "onnx_classifier_v1", "input_spec": source.manifest.input_spec.to_dict(),
            "runtime_config": bindings, "parameters": parameters, "rejection_threshold": threshold,
        })
        verification = {"samples": len(samples), "inputs": records,
                        "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
                        "max_absolute_error": float(np.abs(expected - actual).max()),
                        "label_agreement": True, "threshold_decision_agreement": True}
        provenance = dict(source.manifest.provenance)
        provenance["onnx_export"] = {"source_model_id": source.manifest.run_id,
                                     "source_manifest_hash": sha256_file(source_path / "manifest.json"),
                                     "exporter_hash": sha256_file(__file__), "verification": verification,
                                     "note": "Original evaluation metrics are copied; parity covers only the recorded holdout inputs. CUDA session availability does not prove GPU execution of RF operators."}
        artifacts = {path.relative_to(staging).as_posix(): sha256_file(path)
                     for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = ModelManifest(
            run_id=run_id, backend="onnx_classifier_v1", input_spec=source.manifest.input_spec,
            labels_hash=source.manifest.labels_hash, dataset_hash=source.manifest.dataset_hash,
            split_hash=source.manifest.split_hash, artifacts=artifacts, parameters=parameters,
            dependencies={name: dependency_version(name) for name in ("numpy", "onnxruntime")},
            provenance=provenance, created_at=datetime.now(timezone.utc).isoformat(),
            sign_language=source.manifest.sign_language, runtime_config=bindings,
        )
        write_json_atomic(staging / "manifest.json", manifest.to_dict())
        restored = load_bundle(staging, options=RuntimeOptions(device="cpu"))
        restored.backend.warmup()
        np.testing.assert_array_equal(actual, restored.backend.predict_proba(samples))
        if final.exists():
            raise FileExistsError(f"Model version already exists: {final}")
        staging.rename(final)
        return final
    finally:
        source.backend.close()
        if converted_backend is not None:
            converted_backend.close()
        if restored is not None:
            restored.backend.close()
        if staging is not None and staging.exists():
            # Only the exact temporary directory created above is owned here.
            shutil.rmtree(staging)


def _verification_paths(path: str | Path) -> list[Path]:
    """Read a verification file listing raw clip paths; bundle hashes are authoritative."""
    manifest_path = Path(path).resolve()
    value = read_json(manifest_path)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or not isinstance(value.get("samples"), list) or not value["samples"]):
        raise ValueError("Verification manifest requires schema_version=1 and nonempty samples")
    paths = []
    for sample in value["samples"]:
        if not isinstance(sample, dict) or not isinstance(sample.get("path"), str) or not sample["path"]:
            raise ValueError("Each verification manifest sample requires a path")
        candidate = Path(sample["path"])
        paths.append(candidate if candidate.is_absolute() else manifest_path.parent / candidate)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Validated source RF bundle")
    parser.add_argument("--output", required=True, help="Exact new ONNX bundle directory")
    parser.add_argument("--run-id", required=True)
    verification = parser.add_mutually_exclusive_group(required=True)
    verification.add_argument("--verification-manifest", help="JSON v1 samples with path entries")
    verification.add_argument("--verification-input", action="append", help="Frozen validation/test NPY or NPZ clip; may repeat")
    args = parser.parse_args(argv)
    try:
        inputs = _verification_paths(args.verification_manifest) if args.verification_manifest else args.verification_input
        result = export_rf_bundle(args.model, args.output, args.run_id, inputs)
        print(json.dumps({"model_path": str(result), "backend": "onnx_classifier_v1"}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
