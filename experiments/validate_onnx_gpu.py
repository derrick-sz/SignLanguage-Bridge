"""Prove production-adapter CUDA node placement using an untrained fixture."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path, action="append", default=[])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 1000:
        parser.error("iterations must be in [1, 1000]")
    # Establish native library order before importing other numerical backends.
    import onnx
    import onnxruntime as ort
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from slbridge.contracts import InputSpec
    from slbridge.features import prepare_array
    from slbridge.onnx_backend import OnnxClassifierBackend
    from slbridge.runtime_options import RuntimeOptions

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = InputSpec.for_representation("hand63_v1", target_frames=2)
    weights = np.random.default_rng(3).normal(0, .01, size=(spec.feature_size, 2)).astype(np.float32)
    graph = helper.make_graph([helper.make_node("MatMul", ["features", "weights"], ["logits"])],
                              "untrained-runtime-fixture",
                              [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", spec.feature_size])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 2])],
                              [numpy_helper.from_array(weights, "weights")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    model_path = output / "untrained-fixture.onnx"
    onnx.save(model, model_path)
    config = {"schema_version": 1, "input_layout": "feature_vector", "values_input": "features",
              "mask_input": None, "mask_dtype": "float32", "output_name": "logits", "output_kind": "logits"}
    options = RuntimeOptions(device="cuda", cuda_dll_dirs=tuple(str(path.resolve()) for path in args.cuda_dll_dir))
    original_options = ort.SessionOptions
    def profiled_options():
        value = original_options()
        value.enable_profiling = True
        value.profile_file_prefix = str(output / "production-adapter")
        return value
    with patch.object(ort, "SessionOptions", profiled_options):
        gpu = OnnxClassifierBackend.load(model_path, expected_classes=("alpha", "beta"),
                                         expected_input_spec=spec, runtime_config=config, options=options)
    cpu = None
    try:
        sample = prepare_array(np.ones((2, 63), dtype=np.float32), spec)
        gpu.warmup()
        for _ in range(args.iterations):
            result = gpu.predict_proba([sample])
        profile_path = Path(gpu._session.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        nodes = [event.get("args", {}) for event in events if event.get("cat") == "Node"]
        counts = Counter(node.get("provider", "unknown") for node in nodes)
        real_gpu_compute = sum(node.get("provider") == "CUDAExecutionProvider"
                               and node.get("op_name") in {"MatMul", "Gemm"} for node in nodes)
        if real_gpu_compute < args.iterations:
            raise RuntimeError("CUDA session exists but the fixture's compute nodes were not executed on CUDA")
        cpu = OnnxClassifierBackend.load(model_path, expected_classes=("alpha", "beta"),
                                         expected_input_spec=spec, runtime_config=config,
                                         options=RuntimeOptions(device="cpu"))
        np.testing.assert_allclose(result, cpu.predict_proba([sample]), atol=1e-6, rtol=1e-5)
        report = {"passed": True, "accuracy_claim": None, "fixture": "untrained MatMul classifier",
                  "runtime": gpu.runtime_info, "onnxruntime": ort.__version__, "iterations": args.iterations,
                  "node_provider_counts": dict(counts), "cuda_compute_events": real_gpu_compute,
                  "cpu_gpu_output_agreement": True, "profile": str(profile_path)}
        report_path = output / "gpu_validation.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report))
    finally:
        gpu.close()
        if cpu is not None:
            cpu.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
