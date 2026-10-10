"""Self-contained ONNX CTC bundles and continuous gloss transcription.

Vocabulary order is the model's class order. Token scores are frame-posterior
means for greedy runs, and null for beam search without an alignment. Neither
decoder estimates a calibrated sentence probability or a spoken translation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from ._io import canonical_hash, read_json, resolve_resource, sha256_file
from .contracts import GlossSequencePrediction, GlossToken, RecognitionResult


REQUIRED_ARTIFACTS = frozenset({"model.onnx", "vocabulary.json", "preprocessing.json", "provenance.json"})


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True)
class CTCDecoderConfig:
    decoder: str = "greedy"
    beam_width: int = 10
    blank_id: int = 0
    token_top_k: int = 20

    def __post_init__(self) -> None:
        if self.decoder not in {"greedy", "prefix_beam"}:
            raise ValueError("CTC decoder must be greedy or prefix_beam")
        if type(self.blank_id) is not int or self.blank_id != 0:
            raise ValueError("CTC blank_id must be 0")
        if type(self.beam_width) is not int or not 1 <= self.beam_width <= 100:
            raise ValueError("CTC beam_width must be an integer in [1, 100]")
        if type(self.token_top_k) is not int or not 1 <= self.token_top_k <= 100:
            raise ValueError("CTC token_top_k must be an integer in [1, 100]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> CTCDecoderConfig:
        if not isinstance(value, dict) or set(value) != {entry.name for entry in fields(cls)}:
            raise ValueError("CTC requires complete decoder parameters")
        return cls(**value)


@dataclass(frozen=True)
class CTCOnnxConfig:
    schema_version: int = 2
    values_input: str = "video"
    output_name: str = "logits"
    output_layout: str = "TBV"
    output_kind: str = "logits"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("Unsupported CTC runtime config version")
        for name in ("values_input", "output_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"CTC {name} must be a nonempty tensor name")
        if self.output_layout not in {"TBV", "BTV"} or self.output_kind != "logits":
            raise ValueError("CTC outputs require explicitly bound TBV or BTV logits")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> CTCOnnxConfig:
        if not isinstance(value, dict) or set(value) != {entry.name for entry in fields(cls)}:
            raise ValueError("CTC requires complete versioned runtime bindings")
        return cls(**value)


@dataclass(frozen=True)
class SequenceManifest:
    run_id: str
    input_spec: Any
    vocabulary_hash: str
    artifacts: dict[str, str]
    parameters: dict[str, Any]
    dependencies: dict[str, str]
    provenance: dict[str, Any]
    created_at: str
    sign_language: str | None
    runtime_config: dict[str, Any]
    backend: str = "onnx_ctc"
    task_type: str = "gloss_sequence"
    capabilities: tuple[str, ...] = ("clip",)
    score_type: str = "ctc_token_probability"
    schema_version: int = 2
    dataset_hash: str | None = None
    split_hash: str | None = None
    frozen_evidence: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .cecsl import RGBInputSpec

        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.run_id):
            raise ValueError("Invalid sequence run_id")
        if (type(self.schema_version) is not int or self.schema_version != 2
                or self.task_type != "gloss_sequence" or self.backend != "onnx_ctc"):
            raise ValueError("Unsupported sequence schema/backend/task")
        if not isinstance(self.input_spec, RGBInputSpec) or self.input_spec.representation_id != "tfnet_rgb_v1":
            raise ValueError("Sequence input_spec must be the tfnet_rgb_v1 RGBInputSpec")
        if self.capabilities != ("clip",) or self.score_type != "ctc_token_probability":
            raise ValueError("Unsupported sequence capability/score type")
        _sha(self.vocabulary_hash, "vocabulary_hash")
        for name in ("dataset_hash", "split_hash"):
            if getattr(self, name) is not None:
                _sha(getattr(self, name), name)
        if not isinstance(self.frozen_evidence, dict) or set(self.frozen_evidence) - {"dataset", "split"}:
            raise ValueError("Invalid frozen evidence references")
        if any(not isinstance(value, str) for value in self.frozen_evidence.values()):
            raise ValueError("Frozen evidence references must be artifact paths")
        if len(set(self.frozen_evidence.values())) != len(self.frozen_evidence):
            raise ValueError("Frozen evidence references must be distinct")
        if set(self.frozen_evidence.values()) & REQUIRED_ARTIFACTS:
            raise ValueError("Frozen evidence references cannot replace required artifacts")
        for name in self.frozen_evidence:
            if getattr(self, f"{name}_hash") is None:
                raise ValueError(f"Frozen {name} evidence requires its SHA256")
        if not isinstance(self.artifacts, dict) or set(self.artifacts) != REQUIRED_ARTIFACTS | set(self.frozen_evidence.values()):
            raise ValueError("Sequence bundle has missing or unexpected artifacts")
        for resource, digest in self.artifacts.items():
            resolve_resource(Path.cwd(), resource)
            _sha(digest, "artifact hash")
        for name in ("parameters", "dependencies", "provenance"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
            canonical_hash(getattr(self, name))
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.dependencies.items()):
            raise ValueError("dependencies must map names to version strings")
        CTCDecoderConfig.from_dict(self.parameters)
        CTCOnnxConfig.from_dict(self.runtime_config)
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be nonempty")
        if self.sign_language is not None and (not isinstance(self.sign_language, str) or not self.sign_language.strip()):
            raise ValueError("sign_language must be nonempty or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SequenceManifest:
        from .cecsl import RGBInputSpec

        required = {"schema_version", "run_id", "task_type", "backend", "input_spec", "vocabulary_hash",
                    "artifacts", "parameters", "dependencies", "provenance", "created_at", "sign_language",
                    "capabilities", "score_type", "runtime_config"}
        allowed = {entry.name for entry in fields(cls)}
        if not isinstance(value, dict) or set(value) - allowed or required - set(value):
            raise ValueError("Invalid sequence manifest fields")
        data = dict(value)
        if not isinstance(data["input_spec"], dict) or set(data["input_spec"]) != {entry.name for entry in fields(RGBInputSpec)}:
            raise ValueError("Sequence manifest requires a complete explicit RGBInputSpec")
        data["input_spec"] = RGBInputSpec.from_dict(data["input_spec"])
        if not isinstance(data["capabilities"], (list, tuple)):
            raise ValueError("Sequence capabilities must be an array")
        data["capabilities"] = tuple(data["capabilities"])
        return cls(**data)


def _vocabulary(value: Any) -> tuple[str, ...]:
    required = {"schema_version", "blank_id", "tokens"}
    if (not isinstance(value, dict) or required - set(value) or set(value) - required - {"provenance"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or type(value["blank_id"]) is not int or value["blank_id"] != 0):
        raise ValueError("Invalid ordered CTC vocabulary schema")
    tokens = value["tokens"]
    if (not isinstance(tokens, list) or len(tokens) < 2
            or any(not isinstance(token, str) or not token for token in tokens)
            or any(not token.strip() for token in tokens[1:]) or len(set(tokens)) != len(tokens)):
        raise ValueError("CTC vocabulary requires unique ordered tokens and blank at index 0")
    if "provenance" in value and not isinstance(value["provenance"], dict):
        raise ValueError("Vocabulary provenance must be an object")
    return tuple(tokens)


@dataclass(frozen=True)
class SequenceBundle:
    path: Path
    manifest: SequenceManifest
    backend: Any
    vocabulary: tuple[str, ...]


def load_sequence_bundle(path: str | Path, *, options: Any = None) -> SequenceBundle:
    """Validate every artifact before creating a native session; never unpickle."""
    from .backends import dependency_version

    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Sequence bundles must be complete local directories; archives are unsupported")
    manifest = SequenceManifest.from_dict(read_json(resolve_resource(path, "manifest.json")))
    actual = set()
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError("Sequence bundle symlinks are unsupported")
        if entry.is_file():
            actual.add(entry.relative_to(path).as_posix())
    if actual != set(manifest.artifacts) | {"manifest.json"}:
        raise ValueError("Incomplete sequence directory or untracked files")
    for resource, expected in manifest.artifacts.items():
        artifact = resolve_resource(path, resource)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Sequence artifact checksum mismatch: {resource}")
    raw_vocabulary = read_json(resolve_resource(path, "vocabulary.json"))
    if canonical_hash(raw_vocabulary) != manifest.vocabulary_hash:
        raise ValueError("Ordered vocabulary does not match sequence manifest")
    vocabulary = _vocabulary(raw_vocabulary)
    if read_json(resolve_resource(path, "preprocessing.json")) != manifest.input_spec.to_dict():
        raise ValueError("Sequence model and preprocessing contract disagree")
    if read_json(resolve_resource(path, "provenance.json")) != manifest.provenance:
        raise ValueError("Sequence model and provenance artifact disagree")
    for name, resource in manifest.frozen_evidence.items():
        if manifest.artifacts[resource] != getattr(manifest, f"{name}_hash"):
            raise ValueError(f"Frozen {name} evidence does not match its SHA256")
        read_json(resolve_resource(path, resource))
    # NumPy is used only for validated float32/float64 array operations. The
    # exporter version is recorded as provenance, without coupling a portable
    # ONNX graph to that version. ORT retains its recorded runtime contract.
    for name in ("onnxruntime",):
        if manifest.dependencies.get(name) != dependency_version(name):
            raise ValueError(f"Model dependency version mismatch for {name}; use its recorded environment")
    backend = OnnxCTCBackend.load(resolve_resource(path, "model.onnx"), input_spec=manifest.input_spec,
                                  vocabulary=vocabulary, config=CTCOnnxConfig.from_dict(manifest.runtime_config),
                                  options=options)
    return SequenceBundle(path, manifest, backend, vocabulary)


def _posteriors(logits: Any, vocabulary_size: int) -> Any:
    import numpy as np

    if (not isinstance(logits, np.ndarray) or logits.dtype.kind != "f" or logits.ndim != 2
            or logits.shape[0] < 1 or logits.shape[1] != vocabulary_size or not np.isfinite(logits).all()):
        raise ValueError("CTC logits must be a finite T,V float tensor matching the vocabulary")
    values = np.asarray(logits, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - values.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def greedy_decode(logits: Any, vocabulary: tuple[str, ...], *, blank_id: int = 0) -> GlossSequencePrediction:
    """Merge consecutive repeats before removing blanks, preserving blank-separated repeats."""
    if blank_id != 0:
        raise ValueError("CTC blank_id must be 0")
    probabilities = _posteriors(logits, len(vocabulary))
    path = probabilities.argmax(axis=1)
    tokens = []
    start = 0
    while start < len(path):
        identifier = int(path[start])
        end = start + 1
        while end < len(path) and path[end] == identifier:
            end += 1
        if identifier != blank_id:
            tokens.append(GlossToken(identifier, vocabulary[identifier], float(probabilities[start:end, identifier].mean())))
        start = end
    return GlossSequencePrediction(tuple(tokens), "greedy")


def prefix_beam_decode(logits: Any, vocabulary: tuple[str, ...], *, beam_width: int = 10,
                       token_top_k: int = 20, blank_id: int = 0) -> GlossSequencePrediction:
    """Bounded, token-pruned CTC prefix beam search, separate from upstream ctcdecode.

    Last-prefix symbols and blank are always retained. Scores are null because
    marginalizing paths does not supply per-token alignment/posterior estimates.
    """
    import numpy as np

    config = CTCDecoderConfig("prefix_beam", beam_width, blank_id, token_top_k)
    probabilities = _posteriors(logits, len(vocabulary))
    with np.errstate(divide="ignore"):
        log_probs = np.log(probabilities)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0., -math.inf)}
    for row in log_probs:
        count = min(config.token_top_k, len(vocabulary) - 1)
        candidates = {int(index) + 1 for index in np.argpartition(row[1:], -count)[-count:]}
        updated: dict[tuple[int, ...], list[float]] = {}

        def add(prefix: tuple[int, ...], index: int, score: float) -> None:
            if score != -math.inf:
                pair = updated.setdefault(prefix, [-math.inf, -math.inf])
                pair[index] = float(np.logaddexp(pair[index], score))

        for prefix, (blank, nonblank) in beams.items():
            total = float(np.logaddexp(blank, nonblank))
            add(prefix, 0, total + float(row[0]))
            last = prefix[-1] if prefix else None
            for token in candidates | ({last} if last is not None else set()):
                score = float(row[token])
                if token == last:
                    add(prefix, 1, nonblank + score)
                    add(prefix + (token,), 1, blank + score)
                else:
                    add(prefix + (token,), 1, total + score)
        ordered = sorted(updated.items(), key=lambda item: (-float(np.logaddexp(*item[1])), item[0]))
        beams = {prefix: tuple(pair) for prefix, pair in ordered[:config.beam_width]}
    best = max(beams, key=lambda prefix: float(np.logaddexp(*beams[prefix])))
    return GlossSequencePrediction(tuple(GlossToken(identifier, vocabulary[identifier], None) for identifier in best), "prefix_beam")


class OnnxCTCBackend:
    backend_id = "onnx_ctc"

    def __init__(self, session: Any, input_spec: Any, vocabulary: tuple[str, ...], config: CTCOnnxConfig,
                 runtime_info: dict[str, Any], dll_handles: list[Any] | None = None) -> None:
        self._session, self._input_spec, self.vocabulary = session, input_spec, vocabulary
        self.config, self._runtime_info, self._dll_handles = config, runtime_info, dll_handles or []
        self._run_options = None
        try:
            self._validate_session()
            if runtime_info.get("device") == "cuda":
                import onnxruntime as ort

                self._run_options = ort.RunOptions()
                self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage",
                                                       f"gpu:{runtime_info.get('device_id', 0)}")
        except Exception:
            self.close()
            raise

    @classmethod
    def load(cls, path: str | Path, *, input_spec: Any, vocabulary: tuple[str, ...],
             config: CTCOnnxConfig, options: Any = None) -> OnnxCTCBackend:
        from .cecsl import RGBInputSpec
        from .onnx_backend import create_onnx_session

        if not isinstance(input_spec, RGBInputSpec) or not isinstance(config, CTCOnnxConfig):
            raise ValueError("CTC load requires RGBInputSpec and explicit ONNX bindings")
        _vocabulary({"schema_version": 1, "blank_id": 0, "tokens": list(vocabulary)})
        session, info, handles = create_onnx_session(path, options=options, backend_id=cls.backend_id)
        return cls(session, input_spec, vocabulary, config, info, handles)

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, list) else value for key, value in self._runtime_info.items()}

    def _validate_session(self) -> None:
        config = self.config
        inputs = {node.name: node for node in self._session.get_inputs()}
        if set(inputs) != {config.values_input}:
            raise ValueError("ONNX CTC input names disagree with runtime bindings")
        node = inputs[config.values_input]
        shape = node.shape
        if node.type != "tensor(float)" or not isinstance(shape, (tuple, list)) or len(shape) != 5:
            raise ValueError("ONNX CTC input must be a float32 B,L,3,224,224 tensor")
        if shape[0] not in (None, 1) and not isinstance(shape[0], str):
            raise ValueError("ONNX CTC input must support single-clip batch size 1")
        if tuple(shape[2:]) != (3, 224, 224) or not (shape[1] is None or isinstance(shape[1], str)):
            raise ValueError("ONNX CTC input must have dynamic L and fixed RGB 3,224,224 dimensions")
        outputs = {node.name: node for node in self._session.get_outputs()}
        if config.output_name not in outputs:
            raise ValueError("ONNX configured CTC logits output does not exist")
        node = outputs[config.output_name]
        shape = node.shape
        if node.type not in {"tensor(float)", "tensor(double)"} or not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError("ONNX CTC output must be dense float TBV/BTV logits")
        time_axis, batch_axis = (0, 1) if config.output_layout == "TBV" else (1, 0)
        if (shape[batch_axis] not in (None, 1) and not isinstance(shape[batch_axis], str)
                or shape[2] != len(self.vocabulary)
                or not (shape[time_axis] is None or isinstance(shape[time_axis], str))):
            raise ValueError("ONNX CTC output shape disagrees with dynamic time/batch/vocabulary contract")

    def predict_logits(self, prepared: Any) -> Any:
        import numpy as np

        if self._session is None:
            raise RuntimeError("ONNX CTC backend is closed")
        if getattr(prepared, "spec", None) != self._input_spec:
            raise ValueError("Prepared RGB input is incompatible with the sequence model")
        frames = prepared.source_frames
        if type(frames) is not int or not 1 <= frames <= self._input_spec.max_frames:
            raise ValueError("Prepared RGB source_frames exceeds the input contract")
        expected_t = (frames + 3) // 4
        values = prepared.values
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                or values.shape != (1, expected_t * 4 + 12, 3, 224, 224)
                or not np.isfinite(values).all() or values.min() < -1 or values.max() > 1):
            raise ValueError("Prepared RGB values violate the finite float32 B,L,3,224,224 input contract")
        feeds = {self.config.values_input: np.ascontiguousarray(values)}
        if self._run_options is None:
            raw = self._session.run([self.config.output_name], feeds)[0]
        else:
            raw = self._session.run([self.config.output_name], feeds, run_options=self._run_options)[0]
        expected = (expected_t, 1, len(self.vocabulary)) if self.config.output_layout == "TBV" else (1, expected_t, len(self.vocabulary))
        if not isinstance(raw, np.ndarray) or raw.dtype.kind != "f" or raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError("ONNX returned invalid CTC logits or temporal length; expected ceil(source_frames/4)")
        return raw[:, 0, :] if self.config.output_layout == "TBV" else raw[0, :, :]

    def warmup(self) -> None:
        import numpy as np
        from types import SimpleNamespace

        self.predict_logits(SimpleNamespace(spec=self._input_spec, source_frames=1,
                                            values=np.zeros((1, 16, 3, 224, 224), dtype=np.float32)))

    def close(self) -> None:
        self._session = None
        self._run_options = None
        for handle in reversed(self._dll_handles):
            handle.close()
        self._dll_handles.clear()


class SequenceRecognizer:
    def __init__(self, bundle: SequenceBundle) -> None:
        self.bundle = bundle
        self._closed = False

    @classmethod
    def load(cls, path: str | Path, *, options: Any = None) -> SequenceRecognizer:
        return cls(load_sequence_bundle(path, options=options))

    @property
    def runtime_info(self) -> dict[str, Any]:
        return self.bundle.backend.runtime_info

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SequenceRecognizer is closed")

    def warmup(self) -> None:
        self._ensure_open()
        self.bundle.backend.warmup()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.bundle.backend.close()

    def recognize(self, path: str | Path, *, sample_id: str | None = None,
                  representation_id: str | None = None) -> RecognitionResult:
        return self.recognize_profiled(path, sample_id=sample_id, representation_id=representation_id)[0]

    def recognize_profiled(self, path: str | Path, *, sample_id: str | None = None,
                           representation_id: str | None = None) -> tuple[RecognitionResult, dict[str, float]]:
        from .cecsl import prepare_rgb_file

        self._ensure_open()
        if representation_id is not None and representation_id != self.bundle.manifest.input_spec.representation_id:
            raise ValueError("Declared input representation is incompatible with the model")
        return self._recognize_profiled(lambda: prepare_rgb_file(path, self.bundle.manifest.input_spec),
                                        sample_id or str(Path(path).resolve()))

    def recognize_clip(self, clip: Any, *, sample_id: str | None = None) -> RecognitionResult:
        return self.recognize_clip_profiled(clip, sample_id=sample_id)[0]

    def recognize_clip_profiled(self, clip: Any, *, sample_id: str | None = None) -> tuple[RecognitionResult, dict[str, float]]:
        from .cecsl import prepare_rgb_clip

        self._ensure_open()
        return self._recognize_profiled(lambda: prepare_rgb_clip(clip, self.bundle.manifest.input_spec),
                                        sample_id or clip.source_id)

    def recognize_prepared(self, prepared: Any, *, sample_id: str | None = None) -> RecognitionResult:
        self._ensure_open()
        return self._predict(prepared, sample_id or prepared.source_ref)

    def _recognize_profiled(self, prepare: Any, identifier: str) -> tuple[RecognitionResult, dict[str, float]]:
        from .features import InputRejected

        started = perf_counter()
        try:
            prepared = prepare()
        except InputRejected as error:
            elapsed = (perf_counter() - started) * 1000
            manifest = self.bundle.manifest
            result = RecognitionResult("rejected", identifier, manifest.run_id, manifest.sign_language,
                                       None, error.quality, error.reason, task_type=manifest.task_type,
                                       score_type=manifest.score_type)
            return result, {"preprocess_ms": elapsed, "inference_ms": 0., "model_ms": 0.,
                            "decode_ms": 0., "total_ms": elapsed}
        preprocessing = (perf_counter() - started) * 1000
        infer_start = perf_counter()
        result, stages = self._predict_profiled(prepared, identifier)
        inference = (perf_counter() - infer_start) * 1000
        return result, {"preprocess_ms": preprocessing, "inference_ms": inference, **stages,
                        "total_ms": (perf_counter() - started) * 1000}

    def _predict(self, prepared: Any, identifier: str) -> RecognitionResult:
        return self._predict_profiled(prepared, identifier)[0]

    def _predict_profiled(self, prepared: Any, identifier: str) -> tuple[RecognitionResult, dict[str, float]]:
        started = perf_counter()
        logits = self.bundle.backend.predict_logits(prepared)
        model_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        config = CTCDecoderConfig.from_dict(self.bundle.manifest.parameters)
        prediction = (greedy_decode(logits, self.bundle.vocabulary) if config.decoder == "greedy" else
                      prefix_beam_decode(logits, self.bundle.vocabulary, beam_width=config.beam_width,
                                         token_top_k=config.token_top_k))
        manifest = self.bundle.manifest
        empty = not prediction.tokens
        result = RecognitionResult("rejected" if empty else "recognized", identifier, manifest.run_id,
                                 manifest.sign_language, prediction, prepared.quality,
                                 "ctc_all_blank" if empty else None, task_type=manifest.task_type,
                                 score_type=manifest.score_type, time_range=getattr(prepared, "time_range", None))
        return result, {"model_ms": model_ms, "decode_ms": (perf_counter() - started) * 1000}

    def begin_stream(self) -> None:
        raise NotImplementedError("capability unavailable: this sequence model supports completed clips only")
