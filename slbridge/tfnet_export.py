"""Numerically gated, batch-one dynamic-time CE-CSL TFNet ONNX export.

Only the exporter needs PyTorch and the separately supplied trusted upstream
checkout. The resulting graph uses ordinary ONNX operators, with raw fused TBV
logits and int64 lengths. It is a complete-clip model, not causal streaming.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import threading
from typing import Any, Sequence
import uuid

import numpy as np

from .tfnet_native import TFNetNativeRunner, _sha256, validate_tfnet_input

_PRECISION_LOCK = threading.RLock()


@contextmanager
def _fp32_export_precision(torch: Any):
    """Scope process-wide Torch flags across every reference/export operation."""
    with _PRECISION_LOCK:
        previous = {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield {"reference_and_export": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
                   "caller_flags_before_export": previous, "caller_flags_restored_on_exit": True,
                   "dtype": "float32", "tolerance_policy": "unchanged_explicit_rtol_atol",
                   "native_baseline_comparison": "Parity reference is recomputed with explicit FP32; earlier native baselines using caller defaults may permit cuDNN TF32"}
        finally:
            try:
                torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
            finally:
                torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]


def channel_dft_matrices(channels: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Real/imaginary bases of fft(x, dim=-1, norm='forward')."""
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
        raise ValueError("channels must be a positive integer")
    indices = np.arange(channels, dtype=np.float64)
    phase = (2.0 * np.pi / channels) * np.outer(indices, indices)
    return (np.cos(phase) / channels).astype(np.float32), (-np.sin(phase) / channels).astype(np.float32)


def build_export_wrapper(model: Any) -> Any:
    """Reuse loaded modules, lowering FFT and B1 full-length packed recurrence.

    No model parameters are changed. Packed and dense bidirectional LSTM outputs
    coincide for a single complete sequence; export_tfnet still proves this on
    every supplied input before exporting, then checks ONNX Runtime parity.
    """
    import torch

    class ExportTFNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = model.conv2d
            self.temporal = model.conv1d.temporal_conv
            self.frequency_temporal = model.conv1d1.temporal_conv
            self.recurrent = model.temporal_model.rnn
            self.frequency_recurrent = model.temporal_model1.rnn
            self.fused_classifier = model.classifier55
            cosine, sine = channel_dft_matrices(512)
            parameter = next(model.parameters())
            self.register_buffer("dft_cosine", torch.from_numpy(cosine).to(parameter.device))
            self.register_buffer("dft_sine", torch.from_numpy(sine).to(parameter.device))

        def frequency_magnitude(self, features: Any) -> Any:
            # features are B,L,512. This must never transform the temporal axis.
            features = features.float()
            real = torch.matmul(features, self.dft_cosine)
            imaginary = torch.matmul(features, self.dft_sine)
            return torch.sqrt(real * real + imaginary * imaginary)

        def forward(self, video: Any) -> tuple[Any, Any]:
            features, _, _, _ = self.visual(video.transpose(1, 2))
            features = features.reshape(video.shape[0], video.shape[1], 512)
            frequency = self.frequency_magnitude(features).transpose(1, 2)
            temporal = self.temporal(features.transpose(1, 2)).permute(2, 0, 1)
            frequency = self.frequency_temporal(frequency).permute(2, 0, 1)
            recurrent = self.recurrent(temporal)[0]
            frequency_recurrent = self.frequency_recurrent(frequency)[0]
            logits = self.fused_classifier(recurrent + frequency_recurrent)
            lengths = torch._shape_as_tensor(logits)[0].reshape(1).to(torch.int64)
            return logits, lengths

    return ExportTFNet().eval()


def _ctc_ids(scores: np.ndarray, length: int) -> list[int]:
    path = np.argmax(scores[:length, 0, :], axis=-1)
    result, previous = [], None
    for raw in path:
        token = int(raw)
        if token != previous and token != 0:
            result.append(token)
        previous = token
    return result


def compare_tfnet_outputs(
    expected: np.ndarray, actual: np.ndarray, expected_lengths: np.ndarray,
    actual_lengths: np.ndarray, *, rtol: float = 1e-4, atol: float = 1e-4,
) -> dict[str, Any]:
    """Dense-score, full-length and corrected greedy CTC equality gate."""
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Parity tolerances must be finite and nonnegative")
    if expected.shape != actual.shape or expected.ndim != 3 or expected.shape[1] != 1:
        raise ValueError("TFNet parity requires equal batch-one TBV score shapes")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("TFNet parity scores must be finite")
    if (expected_lengths.shape != (1,) or actual_lengths.shape != (1,)
            or not np.array_equal(expected_lengths, actual_lengths)
            or int(expected_lengths[0]) != expected.shape[0]):
        raise ValueError("TFNet parity output lengths differ or do not cover the full sequence")
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    reference_tokens = _ctc_ids(expected, int(expected_lengths[0]))
    actual_tokens = _ctc_ids(actual, int(actual_lengths[0]))
    scores_equal = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    tokens_equal = reference_tokens == actual_tokens
    report = {
        "score_shape": list(expected.shape), "output_lengths": expected_lengths.tolist(),
        "max_absolute_error": float(difference.max(initial=0.0)),
        "max_relative_error": float((difference / np.maximum(np.abs(expected), 1e-6)).max(initial=0.0)),
        "argmax_frame_agreement": float(np.mean(expected.argmax(-1) == actual.argmax(-1))),
        "allclose": scores_equal, "rtol": rtol, "atol": atol,
        "greedy_ctc_equal": tokens_equal, "reference_greedy_ids": reference_tokens,
        "actual_greedy_ids": actual_tokens,
    }
    if not scores_equal or not tokens_equal:
        raise ValueError("TFNet parity gate failed: " + json.dumps(report))
    return report


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _input_record(value: Any) -> tuple[np.ndarray, dict[str, Any]]:
    values = value.values if hasattr(value, "values") and not isinstance(value, np.ndarray) else value
    array, _ = validate_tfnet_input(values)
    if array.shape[0] != 1:
        raise ValueError("TFNet ONNX export is restricted to batch one")
    source_frames = getattr(value, "source_frames", None)
    if source_frames is not None and (not isinstance(source_frames, int) or source_frames < 1
                                      or 4 * ((source_frames + 3) // 4) + 12 != array.shape[1]):
        raise ValueError("Verification source frame count contradicts padded input length")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return array, {
        "input_shape": list(array.shape), "input_tensor_sha256": digest.hexdigest(),
        "source_frames": source_frames,
        "source_frames_modulo4": None if source_frames is None else source_frames % 4,
        "source_ref": str(getattr(value, "source_ref", "preprocessed_array")),
    }


def export_tfnet(
    checkpoint_path: str | Path, source_dir: str | Path, output_path: str | Path,
    verification_inputs: Sequence[Any], *, device: str = "cpu", opset_version: int = 17,
    rtol: float = 1e-4, atol: float = 1e-4,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    intra_op_threads: int = 4,
) -> dict[str, Any]:
    """Publish a new standalone ONNX file only after both parity gates pass.

    Inputs may be PreparedRGB objects or normalized BLCHW numpy arrays. Real
    held-out input selection and benchmark protocol belong to the caller; this
    function records exact input hashes without claiming WER or unseen accuracy.
    Existing destinations are never overwritten. Successful metadata is saved
    beside the graph at ``<output_path>.metadata.json``.
    """
    if isinstance(verification_inputs, (str, bytes)) or not verification_inputs:
        raise ValueError("At least one preprocessed verification input is required")
    if not isinstance(opset_version, int) or isinstance(opset_version, bool) or not 17 <= opset_version <= 20:
        raise ValueError("TFNet export supports explicit ONNX opset 17 through 20")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Parity tolerances must be finite and nonnegative")
    if type(intra_op_threads) is not int or not 1 <= intra_op_threads <= 64:
        raise ValueError("intra_op_threads must be an integer in [1,64]")
    output = Path(output_path).resolve()
    metadata_path = Path(str(output) + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"TFNet export destination already exists: {output}")
    if output.suffix.lower() != ".onnx":
        raise ValueError("TFNet output_path must end in .onnx")
    prepared = [_input_record(value) for value in verification_inputs]
    if not providers or isinstance(providers, str):
        raise ValueError("Explicit ONNX Runtime providers are required")
    try:
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as error:
        raise ValueError("TFNet export requires optional torch, onnx and onnxruntime dependencies") from error
    unavailable = set(providers) - set(ort.get_available_providers())
    if unavailable:
        raise ValueError(f"ONNX Runtime providers unavailable: {sorted(unavailable)}")
    with _fp32_export_precision(torch) as precision:
        return _export_tfnet_under_fp32(
            checkpoint_path, source_dir, output, metadata_path, prepared, device=device,
            opset_version=opset_version, rtol=rtol, atol=atol, providers=providers,
            intra_op_threads=intra_op_threads, torch=torch, onnx=onnx, ort=ort, precision=precision,
        )


def _export_tfnet_under_fp32(
    checkpoint_path: str | Path, source_dir: str | Path, output: Path, metadata_path: Path,
    prepared: list[tuple[np.ndarray, dict[str, Any]]], *, device: str, opset_version: int,
    rtol: float, atol: float, providers: Sequence[str], intra_op_threads: int,
    torch: Any, onnx: Any, ort: Any, precision: dict[str, Any],
) -> dict[str, Any]:
    runner = TFNetNativeRunner(checkpoint_path, source_dir, device=device)
    wrapper = build_export_wrapper(runner.model)
    reports: list[dict[str, Any]] = []
    reference_outputs = []
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.stem}.{uuid.uuid4().hex}.partial.onnx"
    staging_metadata = Path(str(staging) + ".metadata.json")
    session = None
    try:
        for array, record in prepared:
            native = runner.predict(array)
            with torch.inference_mode():
                converted, lengths = wrapper(torch.from_numpy(array).to(runner.device))
                converted = converted.detach().float().cpu().numpy()
                lengths = lengths.detach().cpu().numpy()
            rewrite = compare_tfnet_outputs(native.scores, converted, native.lengths, lengths, rtol=rtol, atol=atol)
            reports.append({**record, "native_seconds": native.elapsed_seconds, "native_to_rewrite": rewrite})
            reference_outputs.append(native)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, (torch.from_numpy(prepared[0][0]).to(runner.device),), str(staging),
                input_names=["video"], output_names=["logits", "lengths"],
                dynamic_axes={"video": {1: "padded_frames"}, "logits": {0: "output_steps"}},
                opset_version=opset_version, dynamo=False, export_params=True,
                keep_initializers_as_inputs=False, do_constant_folding=True, external_data=False,
            )
        onnx.checker.check_model(str(staging))
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        runtime_providers = [(name, {"use_tf32": 0}) if name == "CUDAExecutionProvider" else name
                             for name in providers]
        session = ort.InferenceSession(str(staging), sess_options=options, providers=runtime_providers)
        missing_providers = set(providers) - set(session.get_providers())
        if missing_providers:
            raise ValueError(f"ONNX Runtime failed to enable requested providers: {sorted(missing_providers)}")
        input_metadata = session.get_inputs()
        output_metadata = {item.name: item for item in session.get_outputs()}
        if (len(input_metadata) != 1 or input_metadata[0].name != "video"
                or input_metadata[0].shape != [1, "padded_frames", 3, 224, 224]
                or output_metadata["logits"].shape != ["output_steps", 1, reference_outputs[0].scores.shape[2]]):
            raise ValueError("ONNX graph metadata must expose batch1 dynamic-time video and fixed-vocabulary TBV logits")
        for (array, _), native, record in zip(prepared, reference_outputs, reports):
            start = time.perf_counter()
            scores, lengths = session.run(["logits", "lengths"], {"video": array})
            record["onnx_seconds"] = time.perf_counter() - start
            record["native_to_onnx"] = compare_tfnet_outputs(
                native.scores, scores, native.lengths, lengths, rtol=rtol, atol=atol,
            )
        metadata = {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": "CE-CSL_TFNet", "provenance": runner.provenance,
            "onnx_sha256": _sha256(staging), "onnx_path": str(output),
            "opset_version": opset_version, "exporter": "torch.onnx_TorchScript_dynamo_false",
            "versions": {name: _version(name) for name in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "onnxruntime-gpu")},
            "runtime_providers_requested": list(providers), "runtime_providers_actual": session.get_providers(),
            "runtime_intra_op_threads": intra_op_threads,
            "precision": precision,
            "runtime_cuda_use_tf32": 0 if "CUDAExecutionProvider" in providers else None,
            "bindings": {"values_input": "video", "output_name": "logits", "lengths_output": "lengths",
                         "input_layout": "BLCHW", "output_layout": "TBV", "output_kind": "logits",
                         "batch_size": 1, "dynamic_time": True, "input_dtype": "float32"},
            "fft": {"axis": "feature_channels", "channels": 512, "normalization": "forward", "lowering": "real_cosine_sine_MatMul_magnitude"},
            "recurrence": {"lowering": "dense_LSTM", "restriction": "batch1_full_effective_length", "native_parity_verified": True},
            "parity_inputs": reports, "parity_passed": True,
            "source_frame_modulo4_coverage": sorted({record["source_frames_modulo4"] for record in reports if record["source_frames_modulo4"] is not None}),
            "limitations": ["Complete-clip noncausal gloss model", "Parity only covers the recorded inputs", "No WER claim or benchmark extrapolation"],
        }
        staging_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Drop ORT's file handle before publishing on Windows.
        session = None
        if output.exists() or metadata_path.exists():
            raise FileExistsError(f"TFNet export destination appeared during export: {output}")
        staging.rename(output)
        try:
            staging_metadata.rename(metadata_path)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return metadata
    finally:
        session = None
        staging.unlink(missing_ok=True)
        staging_metadata.unlink(missing_ok=True)
        runner.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-dir", required=True, help="Explicit trusted upstream checkout at the audited commit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verification-npy", action="append", required=True, help="Float32 preprocessed BLCHW numpy input; repeat for varied lengths")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--provider", action="append")
    parser.add_argument("--intra-op-threads", type=int, default=4)
    args = parser.parse_args(argv)
    metadata = export_tfnet(
        args.checkpoint, args.source_dir, args.output,
        [np.load(path, allow_pickle=False) for path in args.verification_npy],
        device=args.device, opset_version=args.opset, rtol=args.rtol, atol=args.atol,
        providers=args.provider or ["CPUExecutionProvider"],
        intra_op_threads=args.intra_op_threads,
    )
    print(json.dumps({"onnx_path": metadata["onnx_path"], "onnx_sha256": metadata["onnx_sha256"], "parity_passed": metadata["parity_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
