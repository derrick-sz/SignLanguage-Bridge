"""Dense ONNX classification with explicit landmark tensor bindings.

This is an inference adapter, not a trained temporal model. ONNX Runtime is an
optional dependency and is imported only when an ONNX artifact is loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
from typing import Any, Sequence

from .contracts import InputSpec, PreparedInput


@dataclass(frozen=True)
class OnnxRuntimeConfig:
    schema_version: int
    input_layout: str
    values_input: str
    mask_input: str | None
    mask_dtype: str
    output_name: str
    output_kind: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported ONNX runtime config version")
        if self.input_layout not in {"feature_vector", "sequence"}:
            raise ValueError("ONNX input_layout must be feature_vector (NC) or sequence (NTC)")
        for name in ("values_input", "output_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"ONNX {name} must be a nonempty tensor name")
        if self.mask_input is not None and (not isinstance(self.mask_input, str) or not self.mask_input.strip()):
            raise ValueError("ONNX mask_input must be a tensor name or null")
        if self.mask_input == self.values_input:
            raise ValueError("ONNX values and masks require different tensor names")
        if self.input_layout == "feature_vector" and self.mask_input is not None:
            raise ValueError("feature_vector includes the InputSpec mask; separate mask_input is unsupported")
        if self.mask_dtype not in {"float32", "bool"}:
            raise ValueError("ONNX mask_dtype must be float32 or bool")
        if self.output_kind not in {"probabilities", "logits"}:
            raise ValueError("ONNX output_kind must be probabilities or logits")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> OnnxRuntimeConfig:
        if not isinstance(value, dict) or set(value) != {entry.name for entry in fields(cls)}:
            raise ValueError("ONNX requires a complete versioned runtime_config with explicit tensor bindings")
        return cls(**value)


def _validate_shape(node: Any, dimensions: tuple[int, ...], name: str) -> None:
    shape = node.shape
    if not isinstance(shape, (tuple, list)) or len(shape) != len(dimensions):
        raise ValueError(f"ONNX {name} must have shape {dimensions}; got {shape}")
    # Only the batch dimension may be dynamic; preprocessing fixes T and C.
    batch = shape[0]
    if batch is not None and not isinstance(batch, str) and (type(batch) is not int or batch != 1):
        raise ValueError(f"ONNX {name} must support single-clip batch size 1; got batch dimension {batch}")
    if tuple(shape[1:]) != dimensions[1:]:
        raise ValueError(f"ONNX {name} shape disagrees with InputSpec/classes: {shape}; expected {dimensions}")


def _validate_self_contained_onnx(model_bytes: bytes) -> str:
    """Reject external tensor references anywhere before ORT can read a file.

    A session created from bytes can still resolve an external tensor relative
    to the working directory. Parse bytes without external-data loading and
    traverse every populated protobuf message field, including nested graphs,
    sparse tensors, function bodies and training graphs.
    """
    try:
        import onnx
    except ImportError as error:
        raise ValueError("Cannot create ONNX session: self-contained artifact validation requires the ONNX parser dependency") from error
    try:
        model = onnx.load_model_from_string(model_bytes)
    except Exception as error:
        raise ValueError(f"Cannot create ONNX session: invalid model protobuf: {error}") from error
    pending = [model]
    while pending:
        message = pending.pop()
        if isinstance(message, onnx.TensorProto):
            if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
                raise ValueError("Cannot create ONNX session: external tensor data is unsupported; the model must be one self-contained hashed artifact")
            # TensorProto has no tensor/graph-valued children. Avoid materializing
            # raw_data again merely to inspect its primitive protobuf fields.
            continue
        for descriptor, value in message.ListFields():
            if descriptor.type == descriptor.TYPE_MESSAGE:
                repeated = getattr(descriptor, "is_repeated", None)
                if repeated is None:  # Protobuf 5 exposes label instead.
                    repeated = descriptor.label == descriptor.LABEL_REPEATED
                if repeated:
                    pending.extend(value)
                else:
                    pending.append(value)
    return onnx.__version__


def create_onnx_session(path: str | Path, *, options: Any, backend_id: str) -> tuple[Any, dict[str, Any], list[Any]]:
    """Create a self-contained ONNX session with verified provider selection.

    Shared by classification and CTC adapters; tensor validation belongs to each
    adapter and remains task specific. DLL handles must be retained until close.
    """
    from .runtime_options import RuntimeOptions

    if options is None:
        options = RuntimeOptions()
    if not isinstance(options, RuntimeOptions):
        raise ValueError("options must be RuntimeOptions")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ValueError(f"Cannot import ONNX Runtime; install the ONNX extra and verify native DLL compatibility: {error}") from error

    available = tuple(ort.get_available_providers())
    cuda_available = "CUDAExecutionProvider" in available
    if options.device == "cuda" and not cuda_available:
        raise ValueError("Strict CUDA requested, but CUDAExecutionProvider is unavailable; install a compatible ONNX Runtime GPU environment")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = options.intra_op_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    use_cuda = options.device != "cpu" and cuda_available
    dll_handles: list[Any] = []
    dll_dirs = tuple(getattr(options, "cuda_dll_dirs", ()))
    for directory in dll_dirs:
        candidate = Path(directory)
        if not candidate.is_absolute() or not candidate.is_dir():
            raise ValueError(f"CUDA DLL directory must be an existing absolute directory: {directory}")

    def release_dll_handles() -> None:
        for handle in reversed(dll_handles):
            handle.close()
        dll_handles.clear()

    if use_cuda:
        try:
            if hasattr(os, "add_dll_directory"):
                for directory in dll_dirs:
                    dll_handles.append(os.add_dll_directory(directory))
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
                for directory in dll_dirs:
                    ort.preload_dlls(directory=directory)
        except Exception as error:
            release_dll_handles()
            if options.device == "cuda":
                raise ValueError(f"Cannot preload CUDA/cuDNN runtime libraries: {error}") from error
            use_cuda = False
            preload_error = f"CUDA/cuDNN library preloading failed: {type(error).__name__}: {error}"
        else:
            preload_error = None
    else:
        preload_error = None
    providers: list[Any] = ["CPUExecutionProvider"]
    fallback_reason = None
    if use_cuda:
        cuda_options = {"device_id": str(options.device_id),
                        "gpu_mem_limit": str(options.gpu_mem_limit_mb * 1024 * 1024)}
        if backend_id == "onnx_ctc":
            cuda_options["use_tf32"] = "0"
            cuda_options["arena_extend_strategy"] = "kSameAsRequested"
        providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
    elif options.device == "auto":
        fallback_reason = preload_error or "CUDAExecutionProvider is unavailable in this ONNX Runtime environment"
    # Validate the same bytes subsequently passed to ORT. Bytes alone do not
    # prevent ORT from resolving external tensor references against the CWD.
    try:
        model_bytes = Path(path).read_bytes()
        parser_version = _validate_self_contained_onnx(model_bytes)
    except (OSError, ValueError):
        release_dll_handles()
        raise
    try:
        session = ort.InferenceSession(model_bytes, sess_options=session_options, providers=providers)
    except Exception as error:
        if options.device != "auto" or not use_cuda:
            release_dll_handles()
            raise ValueError(f"Cannot create ONNX {options.device} session: {error}") from error
        fallback_reason = f"CUDA session initialization failed: {type(error).__name__}: {error}"
        try:
            session = ort.InferenceSession(model_bytes, sess_options=session_options,
                                          providers=["CPUExecutionProvider"])
        except Exception as cpu_error:
            release_dll_handles()
            raise ValueError(f"Cannot create ONNX CPU fallback session: {cpu_error}") from cpu_error
    # ORT can itself retry creation using CPU. Verify the constructed
    # session, not the requested provider list or merely the installed wheel.
    actual = list(session.get_providers())
    session.disable_fallback()  # Prevent a later run from silently replacing its providers.
    active_cuda = "CUDAExecutionProvider" in actual
    if options.device == "cuda" and not active_cuda:
        release_dll_handles()
        raise ValueError("Strict CUDA requested, but the actual ONNX session did not activate CUDAExecutionProvider (check CUDA/cuDNN DLL compatibility)")
    if options.device == "auto" and use_cuda and not active_cuda:
        fallback_reason = fallback_reason or "CUDAExecutionProvider was requested but the actual ONNX session activated CPU only (check CUDA/cuDNN compatibility)"
    if options.device == "cpu" and actual != ["CPUExecutionProvider"]:
        release_dll_handles()
        raise ValueError(f"ONNX CPU request created unexpected execution providers: {actual}")
    if not active_cuda and "CPUExecutionProvider" not in actual:
        release_dll_handles()
        raise ValueError(f"ONNX session has no supported execution provider: {actual}")
    info = {"backend": backend_id, "requested_device": options.device,
            "device": "cuda" if active_cuda else "cpu", "providers": actual,
            "available_providers": list(available), "fallback_reason": fallback_reason,
            "device_id": options.device_id, "intra_op_threads": options.intra_op_threads,
            "gpu_mem_limit_mb": options.gpu_mem_limit_mb,
            "cuda_dll_directories_configured": len(dll_dirs),
            "onnx_parser_version": parser_version,
            "provider_scope": "session; individual operators may execute on CPU"}
    if backend_id == "onnx_ctc":
        info.update(precision="float32_tf32_disabled", use_tf32=False,
                    cuda_arena_extend_strategy="kSameAsRequested" if active_cuda else None,
                    cuda_arena_shrinkage=f"gpu:{options.device_id}" if active_cuda else None)
    return session, info, dll_handles


class OnnxClassifierBackend:
    backend_id = "onnx_classifier_v1"

    def __init__(self, session: Any, input_spec: InputSpec, classes: tuple[str, ...],
                 config: OnnxRuntimeConfig, runtime_info: dict[str, Any], dll_handles: list[Any] | None = None) -> None:
        self._session = session
        self._input_spec = input_spec
        self._classes = classes
        self._config = config
        self._runtime_info = runtime_info
        self._dll_handles = dll_handles or []
        try:
            self._validate_session()
        except Exception:
            self.close()
            raise

    @classmethod
    def load(cls, path: str | Path, *, expected_classes: tuple[str, ...],
             expected_input_spec: InputSpec, runtime_config: dict[str, Any],
             options: Any = None) -> OnnxClassifierBackend:
        from .runtime_options import RuntimeOptions

        if options is None:
            options = RuntimeOptions()
        if not isinstance(options, RuntimeOptions):
            raise ValueError("options must be RuntimeOptions")
        if not isinstance(expected_input_spec, InputSpec):
            raise ValueError("Expected ONNX input contract must be InputSpec")
        if (not isinstance(expected_classes, tuple) or not expected_classes
                or any(not isinstance(label, str) or not label.strip() for label in expected_classes)
                or len(set(expected_classes)) != len(expected_classes)):
            raise ValueError("Expected classes must be a nonempty tuple of unique strings")
        config = OnnxRuntimeConfig.from_dict(runtime_config)
        session, info, dll_handles = create_onnx_session(path, options=options, backend_id=cls.backend_id)
        return cls(session, expected_input_spec, expected_classes, config, info, dll_handles)

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def feature_size(self) -> int:
        return self._input_spec.feature_size

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, list) else value
                for key, value in self._runtime_info.items()}

    def _validate_session(self) -> None:
        spec, config = self._input_spec, self._config
        inputs = {node.name: node for node in self._session.get_inputs()}
        required = {config.values_input} | ({config.mask_input} if config.mask_input else set())
        if set(inputs) != required:
            raise ValueError(f"ONNX input names disagree with runtime bindings: {sorted(inputs)}; expected {sorted(required)}")
        dimensions = ((1, spec.feature_size) if config.input_layout == "feature_vector"
                      else (1, spec.target_frames, spec.width))
        values = inputs[config.values_input]
        if values.type != "tensor(float)":
            raise ValueError("ONNX values input must be tensor(float) / float32")
        _validate_shape(values, dimensions, "values input")
        if config.mask_input:
            mask = inputs[config.mask_input]
            required_type = "tensor(bool)" if config.mask_dtype == "bool" else "tensor(float)"
            if mask.type != required_type:
                raise ValueError(f"ONNX mask input must be {required_type}")
            _validate_shape(mask, dimensions, "mask input")
            if (type(mask.shape[0]) is int and type(values.shape[0]) is int
                    and mask.shape[0] != values.shape[0]):
                raise ValueError("ONNX values and masks have inconsistent batch dimensions")
        outputs = {node.name: node for node in self._session.get_outputs()}
        if config.output_name not in outputs:
            raise ValueError("ONNX configured classification output does not exist")
        output = outputs[config.output_name]
        if output.type not in {"tensor(float)", "tensor(double)"}:
            raise ValueError("ONNX classification output must be a dense float tensor; ZipMap is unsupported")
        _validate_shape(output, (1, len(self.classes)), "classification output")
        if (type(output.shape[0]) is int and type(values.shape[0]) is int
                and output.shape[0] != values.shape[0]):
            raise ValueError("ONNX input and output have inconsistent batch dimensions")
        self._fixed_batch = values.shape[0] if type(values.shape[0]) is int else None

    def _input_feed(self, samples: Sequence[PreparedInput]) -> dict[str, Any]:
        import numpy as np
        from .features import feature_vector

        if not samples or any(not isinstance(sample, PreparedInput) for sample in samples):
            raise ValueError("At least one PreparedInput is required for ONNX inference")
        if any(sample.spec != self._input_spec for sample in samples):
            raise ValueError("Model samples must use the same InputSpec")
        if self._fixed_batch is not None and len(samples) != self._fixed_batch:
            raise ValueError(f"ONNX graph requires a fixed batch size of {self._fixed_batch}")
        # Reuse validation of finite coordinates, zero missing encoding and mask
        # shape/dtype even when a sequence graph receives separate tensors.
        vectors = [feature_vector(sample) for sample in samples]
        values = (np.stack(vectors) if self._config.input_layout == "feature_vector"
                  else np.stack([sample.values for sample in samples]))
        feed = {self._config.values_input: np.ascontiguousarray(values, dtype=np.float32)}
        if self._config.mask_input:
            feed[self._config.mask_input] = np.ascontiguousarray(
                np.stack([sample.mask for sample in samples]),
                dtype=np.bool_ if self._config.mask_dtype == "bool" else np.float32)
        return feed

    def predict_proba(self, samples: Sequence[PreparedInput]) -> Any:
        import numpy as np

        if self._session is None:
            raise RuntimeError("ONNX backend is closed")
        feed = self._input_feed(samples)
        output = self._session.run([self._config.output_name], feed)[0]
        if (not isinstance(output, np.ndarray) or output.dtype.kind != "f"
                or output.shape != (len(samples), len(self.classes)) or not np.isfinite(output).all()):
            raise ValueError("ONNX returned an invalid dense class output")
        probabilities = np.asarray(output, dtype=np.float64)
        if self._config.output_kind == "logits":
            shifted = probabilities - probabilities.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        if ((probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)):
            raise ValueError("ONNX returned invalid class probabilities")
        return probabilities

    def warmup(self) -> None:
        import numpy as np
        from .features import prepare_array

        spec = self._input_spec
        sample = prepare_array(np.zeros((spec.target_frames, spec.width), dtype=np.float32), spec,
                               timestamps=np.arange(spec.target_frames, dtype=np.float64)
                               if spec.temporal_policy == "timestamp_linear" else None)
        self.predict_proba([sample] * (self._fixed_batch or 1))

    def close(self) -> None:
        self._session = None
        for handle in reversed(self._dll_handles):
            handle.close()
        self._dll_handles.clear()
