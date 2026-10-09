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
from pathlib import Path
import threading
import time
from typing import Any

from .runtime_contracts import InferenceRequest, InferenceResponse
from .runtime_options import RuntimeOptions


class RuntimeBusy(RuntimeError):
    """Capacity is occupied or a model transaction is in progress."""


class ModelUnavailable(RuntimeError):
    """No active model, no rollback target, or runtime is closed."""


@dataclass
class ModelCandidate:
    recognizer: Any
    options: RuntimeOptions
    warmup_ms: float
    consumed: bool = False


@dataclass(frozen=True)
class _Version:
    recognizer: Any
    generation: int
    options: RuntimeOptions
    warmup_ms: float = 0.


def _release(model) -> None:
    close = getattr(model, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            logging.getLogger(__name__).exception("Model resource release failed")


def _describe(version: _Version | None) -> dict | None:
    if version is None:
        return None
    model = version.recognizer
    bundle = getattr(model, "bundle", None)
    manifest = getattr(bundle, "manifest", None)
    spec = getattr(manifest, "input_spec", None)
    return deepcopy({"model_id": getattr(manifest, "run_id", "injected-model"),
                     "generation": version.generation,
                     "backend": getattr(manifest, "backend", "unknown"),
                     "task_type": getattr(manifest, "task_type", "classification"),
                     "score_type": getattr(manifest, "score_type", "uncalibrated_class_probability"),
                     "sign_language": getattr(manifest, "sign_language", None),
                     "input_spec": (spec.to_dict() if hasattr(spec, "to_dict") else
                                    {"representation_id": spec.representation_id} if spec is not None else None),
                     "capabilities": list(getattr(manifest, "capabilities", ())),
                     "labels": getattr(bundle, "labels", {}),
                     **({"vocabulary_size": len(bundle.vocabulary)} if hasattr(bundle, "vocabulary") else {}),
                     "runtime": getattr(model, "runtime_info", {}),
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
                self._generation = 1
                self._active = _Version(recognizer, 1, self.options)

    def _check_open(self) -> None:
        if self._closed:
            raise ModelUnavailable("Model runtime is closed")

    def describe(self) -> dict | None:
        with self._lock:
            return _describe(self._active)

    def status(self) -> dict:
        with self._lock:
            state = ("closed" if self._closed else "loading" if self._transition else
                     "inferencing" if self._inflight else "ready" if self._active else "empty")
            return {"state": state, "generation": self._generation,
                    "active": _describe(self._active), "previous": _describe(self._previous),
                    "error": deepcopy(self._error)}

    def prepare(self, path: str | Path, *, options: RuntimeOptions | None = None,
                loader=None, required_representation: str | tuple[str, ...] | None = None) -> ModelCandidate:
        with self._lock:
            self._check_open()
            if self._inflight or self._transition:
                raise RuntimeBusy("Finish the current clip/task before switching models")
            self._transition = True
            self._error = None
            previous, self._previous = self._previous, None
        # Keep at most active + candidate. Release older rollback before loading.
        if previous is not None:
            _release(previous.recognizer)
        model = None
        chosen_options = options or self.options
        try:
            if loader is None:
                from .inference import Recognizer
                model = Recognizer.load(path, options=chosen_options)
            else:
                model = loader(path)
            manifest = model.bundle.manifest
            if "clip" not in manifest.capabilities:
                raise ValueError("Model does not support clip inference")
            if required_representation is not None:
                allowed = (required_representation,) if isinstance(required_representation, str) else required_representation
                if (not isinstance(allowed, tuple) or not allowed or any(not isinstance(value, str) for value in allowed)):
                    raise ValueError("required_representation must be a string or nonempty tuple of strings")
                if manifest.input_spec.representation_id not in allowed:
                    raise ValueError(f"需要 {required_representation} 且支持 clip 的模型包")
            start = time.perf_counter()
            warmup = getattr(model, "warmup", None)
            if warmup is not None:
                warmup()
            candidate = ModelCandidate(model, chosen_options, (time.perf_counter() - start) * 1000)
            with self._lock:
                self._check_open()
                self._pending = candidate
            return candidate
        except BaseException as error:
            if model is not None:
                _release(model)
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
                                    candidate.options, candidate.warmup_ms)
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
        _release(candidate.recognizer)

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
            self._active = _Version(previous.recognizer, self._generation, previous.options, previous.warmup_ms)
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
                raise ModelUnavailable("The request's model version is no longer active")
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

    def _infer(self, source, request: InferenceRequest, *, captured: bool) -> InferenceResponse:
        if not isinstance(request, InferenceRequest):
            raise ValueError("Inference requires a versioned InferenceRequest")
        with self._lease() as version:
            model = version.recognizer
            manifest = model.bundle.manifest
            if request.expected_task_type is not None and request.expected_task_type != getattr(manifest, "task_type", "classification"):
                raise ValueError("Declared task type is incompatible with the model")
            if request.sign_language is not None and request.sign_language != manifest.sign_language:
                raise ValueError("Declared sign language is incompatible with the model")
            if request.representation_id is not None and request.representation_id != manifest.input_spec.representation_id:
                raise ValueError("Declared input representation is incompatible with the model")
            start = time.perf_counter()
            if captured:
                result, timing = model.recognize_clip_profiled(source, sample_id=request.request_id)
            else:
                result, timing = model.recognize_profiled(source, sample_id=request.request_id,
                                                          representation_id=request.representation_id)
            timing = dict(timing)
            timing["total_ms"] = (time.perf_counter() - start) * 1000
            return InferenceResponse(request.request_id, version.generation, result,
                                     deepcopy(model.runtime_info), timing)

    def infer_file(self, path: str | Path, request: InferenceRequest) -> InferenceResponse:
        return self._infer(path, request, captured=False)

    def infer_clip(self, clip, request: InferenceRequest) -> InferenceResponse:
        return self._infer(clip, request, captured=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active, previous, pending = self._active, self._previous, self._pending
            self._active = self._previous = self._pending = None
            self._transition = False
            if active is not None and self._inflight:
                self._deferred.append(active.recognizer)
                active = None
            if pending is not None:
                pending.consumed = True
        for model in (active.recognizer if active else None, previous.recognizer if previous else None,
                      pending.recognizer if pending else None):
            if model is not None:
                _release(model)
