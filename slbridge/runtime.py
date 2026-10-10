"""Single-capacity inference and transactional model switching, independent of UI/HTTP.

Native calls cannot be force-cancelled safely. A lease pins its model until it
returns, and shutdown releases that instance afterwards. Switching pauses new
inference; this first implementation does not promise uninterrupted service.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any

from .runtime_contracts import InferenceRequest, InferenceResponse
from .runtime_options import RuntimeOptions
from .model_adapters import (ModelAdapter, ModelInputContract, adapt_recognizer,
                             load_model_adapter, validate_adapter_result)


class RuntimeBusy(RuntimeError):
    """Capacity is occupied or a model transaction is in progress."""


class ModelUnavailable(RuntimeError):
    """No active model, no rollback target, or runtime is closed."""


class ModelGenerationConflict(ModelUnavailable):
    """The requested generation/model binding is no longer active."""


@dataclass
class ModelCandidate:
    recognizer: Any
    options: RuntimeOptions
    warmup_ms: float
    consumed: bool = False
    adapter: ModelAdapter | None = None


@dataclass(frozen=True)
class _Version:
    recognizer: Any
    generation: int
    options: RuntimeOptions
    warmup_ms: float = 0.
    adapter: ModelAdapter | None = None


def _release(model) -> None:
    close = getattr(model, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            logging.getLogger(__name__).exception("Model resource release failed")


def _owned(version) -> Any:
    return version.adapter if version.adapter is not None else version.recognizer


def _injected_adapter(model) -> ModelAdapter | None:
    # Old local clients/tests inject run-only objects without a bundle backend.
    # Those remain usable through run(); typed inference requires an adapter.
    # File-backed loading never uses this compatibility path.
    manifest = getattr(getattr(model, "bundle", None), "manifest", None)
    backend = getattr(manifest, "backend", None)
    spec = getattr(manifest, "input_spec", None)
    if isinstance(model, ModelAdapter) or (backend is not None and callable(getattr(spec, "to_dict", None))):
        return adapt_recognizer(model)
    return None


def _describe(version: _Version | None) -> dict | None:
    if version is None:
        return None
    model = version.recognizer
    bundle = getattr(model, "bundle", None)
    manifest = getattr(bundle, "manifest", None)
    spec = getattr(manifest, "input_spec", None)
    capabilities = version.adapter.capabilities if version.adapter is not None else None
    input_spec = (spec.to_dict() if hasattr(spec, "to_dict") else
                  {"representation_id": spec.representation_id} if spec is not None else
                  capabilities.inputs[0].to_dict() if capabilities is not None else None)
    return deepcopy({"model_id": getattr(manifest, "run_id", getattr(capabilities, "model_id", "injected-model")),
                     "generation": version.generation,
                     "backend": getattr(manifest, "backend", getattr(capabilities, "runtime", "unknown")),
                     "task_type": getattr(manifest, "task_type", getattr(capabilities, "task_type", "classification")),
                     "score_type": getattr(manifest, "score_type", getattr(capabilities, "score_type", "uncalibrated_class_probability")),
                     "sign_language": getattr(manifest, "sign_language", getattr(capabilities, "sign_language", None)),
                     "input_spec": input_spec,
                     "capabilities": list(getattr(manifest, "capabilities", getattr(capabilities, "modes", ()))),
                     "labels": getattr(bundle, "labels", {}),
                     **({"vocabulary_size": len(bundle.vocabulary)} if hasattr(bundle, "vocabulary") else {}),
                     "runtime": (version.adapter.runtime_info if version.adapter is not None else
                                 getattr(model, "runtime_info", {})),
                     "options": version.options.to_dict(), "warmup_ms": version.warmup_ms})


class ModelManager:
    def __init__(self, *, options: RuntimeOptions | None = None, initial_recognizer=None):
        self.options = options or RuntimeOptions()
        if not isinstance(self.options, RuntimeOptions):
            raise ValueError("options must be RuntimeOptions")
        # Windows native libraries must be initialized in a stable order before
        # the first sklearn model. This is explicit runtime startup, never a
        # package/module import side effect; missing optional ORT leaves RF usable.
        from .backends import initialize_runtime_libraries
        self.runtime_libraries = initialize_runtime_libraries()
        self._lock = threading.RLock()
        self._generation = 0
        self._active = None
        self._previous = None
        self._pending = None
        self._transition = False
        self._inflight = False
        self._closed = False
        self._deferred = []
        self._error = None
        if initial_recognizer is not None:
            self.set_initial(initial_recognizer)

    @property
    def recognizer(self):
        with self._lock:
            return self._active.recognizer if self._active is not None else None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def supports_typed_inference(self) -> bool:
        with self._lock:
            return self._active is not None and self._active.adapter is not None

    def set_initial(self, recognizer) -> None:
        """Inject an already-constructed model before first use (local clients/tests).

        Subsequent replacement must pass prepare/activate, never this shortcut.
        File-backed clients always use switch, which validates and warms up.
        """
        with self._lock:
            self._check_open()
            if self._generation or self._transition or self._inflight:
                raise RuntimeBusy("Initial model can only be installed before first use")
            if recognizer is not None:
                adapter = _injected_adapter(recognizer)
                self._generation = 1
                self._active = _Version(adapter.recognizer if adapter is not None else recognizer,
                                        1, self.options, adapter=adapter)

    def _check_open(self) -> None:
        if self._closed:
            raise ModelUnavailable("Model runtime is closed")

    def describe(self) -> dict | None:
        with self._lock:
            return _describe(self._active)

    def capabilities(self) -> dict | None:
        """Full adapter contract; legacy describe/status retain their HTTP shape."""
        with self._lock:
            return (deepcopy(self._active.adapter.capabilities.to_dict())
                    if self._active is not None and self._active.adapter is not None else None)

    def _negotiate(self, version: _Version, request: InferenceRequest, *, input_kind: str) -> ModelInputContract:
        if not isinstance(request, InferenceRequest):
            raise ValueError("Inference requires a versioned InferenceRequest")
        request.check_active()
        if version.adapter is None:
            raise ValueError("Run-only injected recognizer has no typed model adapter")
        if request.expected_generation is not None and request.expected_generation != version.generation:
            raise ModelGenerationConflict("The request's model generation is no longer active")
        if request.model_id is not None and request.model_id != version.adapter.capabilities.model_id:
            raise ModelGenerationConflict("The request's model_id is no longer active")
        return version.adapter.capabilities.negotiate(request, input_kind=input_kind)

    def negotiate(self, request: InferenceRequest, *, input_kind: str = "file") -> ModelInputContract:
        """Check compatibility without reserving capacity; infer checks again in its lease."""
        with self._lock:
            self._check_open()
            if self._active is None:
                raise ModelUnavailable("No model is loaded")
            return self._negotiate(self._active, request, input_kind=input_kind)

    def status(self) -> dict:
        with self._lock:
            state = ("closed" if self._closed else "loading" if self._transition else
                     "inferencing" if self._inflight else "ready" if self._active else "empty")
            return {"state": state, "generation": self._generation,
                    "active": _describe(self._active), "previous": _describe(self._previous),
                    "error": deepcopy(self._error)}

    def prepare(self, path: str | Path, *, options: RuntimeOptions | None = None,
                loader=None, required_representation: str | tuple[str, ...] | None = None,
                required_input_kind: str | None = None) -> ModelCandidate:
        if required_input_kind is not None and required_input_kind not in {"file", "captured_clip", "prepared"}:
            raise ValueError("Unsupported required_input_kind")
        with self._lock:
            self._check_open()
            if self._inflight or self._transition:
                raise RuntimeBusy("Finish the current clip/task before switching models")
            self._transition = True
            self._error = None
            previous, self._previous = self._previous, None
        # Keep at most active + candidate. Release older rollback before loading.
        if previous is not None:
            _release(_owned(previous))
        model = None
        adapter = None
        chosen_options = options or self.options
        try:
            if loader is None:
                adapter = load_model_adapter(path, options=chosen_options)
                model = adapter.recognizer
            else:
                model = loader(path)
                adapter = _injected_adapter(model)
                if adapter is not None:
                    model = adapter.recognizer
            if not isinstance(chosen_options, RuntimeOptions):
                raise ValueError("options must be RuntimeOptions")
            if adapter is not None and chosen_options.device != "auto" and chosen_options.device not in adapter.capabilities.devices:
                raise ValueError("Requested device is unsupported by the model adapter")
            manifest = getattr(getattr(model, "bundle", None), "manifest", None)
            modes = adapter.capabilities.modes if adapter is not None else getattr(manifest, "capabilities", ())
            if adapter is None and "clip" not in modes:
                raise ValueError("Model does not support clip inference")
            if required_input_kind is not None:
                if adapter is not None:
                    supported = any(required_input_kind in entry.input_kinds for entry in adapter.capabilities.inputs)
                else:
                    # Preserve explicit legacy loader/test injection only. Real
                    # packages always negotiate registered adapter contracts.
                    representation = getattr(getattr(manifest, "input_spec", None), "representation_id", None)
                    supported = (required_input_kind == "file" or
                                 required_input_kind == "captured_clip" and representation in
                                 {"mediapipe225_rgb_v1", "tfnet_rgb_v1"} or
                                 required_input_kind == "prepared" and callable(getattr(model, "recognize_prepared", None)))
                if not supported:
                    raise ValueError(f"Model does not support required input kind: {required_input_kind}")
            if required_representation is not None:
                if "clip" not in modes:
                    raise ValueError("Model does not support clip inference")
                allowed = (required_representation,) if isinstance(required_representation, str) else required_representation
                if (not isinstance(allowed, tuple) or not allowed or any(not isinstance(value, str) for value in allowed)):
                    raise ValueError("required_representation must be a string or nonempty tuple of strings")
                representations = (tuple(entry.representation_id for entry in adapter.capabilities.inputs)
                                   if adapter is not None else (getattr(getattr(manifest, "input_spec", None), "representation_id", None),))
                if not any(value in allowed for value in representations):
                    raise ValueError(f"需要 {required_representation} 且支持 clip 的模型包")
            start = time.perf_counter()
            warmup = getattr(adapter if adapter is not None else model, "warmup", None)
            if warmup is not None:
                warmup()
            if (adapter is not None and chosen_options.device != "auto"
                    and adapter.runtime_info.get("device", chosen_options.device) != chosen_options.device):
                raise ValueError("Model runtime did not honor the requested device")
            candidate = ModelCandidate(model, chosen_options, (time.perf_counter() - start) * 1000, adapter=adapter)
            with self._lock:
                self._check_open()
                self._pending = candidate
            return candidate
        except BaseException as error:
            if adapter is not None or model is not None:
                _release(adapter if adapter is not None else model)
            with self._lock:
                self._transition = False
                self._error = {"code": type(error).__name__, "message": str(error)}
            raise

    def activate(self, candidate: ModelCandidate) -> dict:
        with self._lock:
            self._check_open()
            if candidate is not self._pending or candidate.consumed:
                raise ModelUnavailable("Candidate is no longer valid")
            if self._inflight:
                raise RuntimeBusy("An inference still owns the active model")
            self._generation += 1
            self._previous = self._active
            self._active = _Version(candidate.recognizer, self._generation,
                                    candidate.options, candidate.warmup_ms, candidate.adapter)
            candidate.consumed = True
            self._pending = None
            self._transition = False
            self._error = None
            return self.status()

    def discard(self, candidate: ModelCandidate) -> None:
        with self._lock:
            if candidate.consumed:
                return
            if candidate is not self._pending:
                raise ModelUnavailable("Candidate does not belong to this runtime")
            candidate.consumed = True
            self._pending = None
            self._transition = False
        _release(_owned(candidate))

    def switch(self, path: str | Path, *, options: RuntimeOptions | None = None, loader=None) -> dict:
        candidate = self.prepare(path, options=options, loader=loader)
        try:
            return self.activate(candidate)
        finally:
            self.discard(candidate)

    def rollback(self) -> dict:
        with self._lock:
            self._check_open()
            if self._inflight or self._transition:
                raise RuntimeBusy("Finish the current clip/task before rolling back")
            if self._previous is None:
                raise ModelUnavailable("No previous model is retained")
            self._generation += 1
            previous = self._previous
            self._previous = self._active
            self._active = _Version(previous.recognizer, self._generation, previous.options,
                                    previous.warmup_ms, previous.adapter)
            self._error = None
            return self.status()

    @contextmanager
    def _lease(self, expected_model=None):
        with self._lock:
            self._check_open()
            if self._transition or self._inflight:
                raise RuntimeBusy("Model runtime is busy; retry after the current task")
            if self._active is None:
                raise ModelUnavailable("No model is loaded")
            version = self._active
            if expected_model is not None and expected_model is not version.recognizer:
                raise ModelGenerationConflict("The request's model version is no longer active")
            self._inflight = True
        try:
            yield version
        finally:
            with self._lock:
                self._inflight = False
                deferred, self._deferred = self._deferred, []
            for model in deferred:
                _release(model)

    def run(self, operation, *, expected_model=None):
        """Local session adapter; pins the same model throughout a native call."""
        with self._lease(expected_model) as version:
            return operation(version.recognizer)

    def infer(self, source, request: InferenceRequest, *, input_kind: str = "prepared") -> InferenceResponse:
        if not isinstance(request, InferenceRequest):
            raise ValueError("Inference requires a versioned InferenceRequest")
        with self._lease() as version:
            start = time.perf_counter()
            adapter = version.adapter
            contract = self._negotiate(version, request, input_kind=input_kind)
            if input_kind == "prepared":
                contract.validate_prepared(source, request)
            result, timing = adapter.infer(source, request, input_kind=input_kind)
            request.check_active()
            validate_adapter_result(result, adapter.capabilities, request)
            timing = dict(timing)
            if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
                   for value in timing.values()):
                raise ValueError("Adapter timings must be finite nonnegative milliseconds")
            timing["total_ms"] = (time.perf_counter() - start) * 1000
            response = InferenceResponse(request.request_id, version.generation, result,
                                         deepcopy(adapter.runtime_info), timing,
                                         session_id=request.session_id, segment_id=request.segment_id,
                                         source_epoch=request.source_epoch, model_id=adapter.capabilities.model_id,
                                         model_version=adapter.capabilities.model_version,
                                         preprocessing_version=contract.preprocessing_version)
            request.check_active()
            return response

    def infer_file(self, path: str | Path, request: InferenceRequest) -> InferenceResponse:
        return self.infer(path, request, input_kind="file")

    def infer_clip(self, clip, request: InferenceRequest) -> InferenceResponse:
        return self.infer(clip, request, input_kind="captured_clip")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active, previous, pending = self._active, self._previous, self._pending
            self._active = self._previous = self._pending = None
            self._transition = False
            if active is not None and self._inflight:
                self._deferred.append(_owned(active))
                active = None
            if pending is not None:
                pending.consumed = True
        for model in (_owned(active) if active else None, _owned(previous) if previous else None,
                      _owned(pending) if pending else None):
            if model is not None:
                _release(model)


class ModelRouter:
    """Independent language runtimes; switching one route cannot switch another.

    Each manager retains its own single-request capacity. GPU providers may
    internally serialize work; this router promises identity isolation, not
    physical concurrent GPU execution. Registered managers are owned by router.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._routes: dict[str, ModelManager] = {}
        self._closed = False

    def register(self, sign_language: str, manager: ModelManager) -> None:
        if not isinstance(sign_language, str) or not sign_language.strip() or len(sign_language) > 64:
            raise ValueError("A sign language route is required")
        if not isinstance(manager, ModelManager):
            raise ValueError("A route requires a ModelManager")
        with self._lock:
            if self._closed:
                raise ModelUnavailable("Model router is closed")
            if sign_language in self._routes or manager in self._routes.values():
                raise ValueError("Language routes must have independent managers")
            capabilities = manager.capabilities()
            if capabilities is not None and capabilities["sign_language"] != sign_language:
                raise ValueError("Language route disagrees with the model")
            self._routes[sign_language] = manager

    def manager_for(self, sign_language: str) -> ModelManager:
        with self._lock:
            if self._closed:
                raise ModelUnavailable("Model router is closed")
            try:
                return self._routes[sign_language]
            except (KeyError, TypeError) as error:
                raise ModelUnavailable(f"No model route for sign language: {sign_language}") from error

    def _manager(self, request: InferenceRequest) -> ModelManager:
        if not isinstance(request, InferenceRequest) or request.sign_language is None:
            raise ValueError("Routed inference requires an explicit sign_language")
        return self.manager_for(request.sign_language)

    def infer_file(self, path, request: InferenceRequest) -> InferenceResponse:
        return self._manager(request).infer_file(path, request)

    def infer_clip(self, clip, request: InferenceRequest) -> InferenceResponse:
        return self._manager(request).infer_clip(clip, request)

    def infer(self, source, request: InferenceRequest, *, input_kind: str = "prepared") -> InferenceResponse:
        return self._manager(request).infer(source, request, input_kind=input_kind)

    def capabilities(self) -> dict[str, dict | None]:
        with self._lock:
            return {language: manager.capabilities() for language, manager in self._routes.items()}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            managers = list(self._routes.values())
            self._routes.clear()
        for manager in managers:
            manager.close()
