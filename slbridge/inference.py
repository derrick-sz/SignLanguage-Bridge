"""Clip recognition independent of CLI, GUI and training source directories."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, TYPE_CHECKING

from .bundle import ModelBundle, load_bundle
from .contracts import ClassificationPrediction, PreparedInput, RecognitionResult
from .features import InputRejected, prepare_input

if TYPE_CHECKING:
    from .sources import CapturedClip


class Recognizer:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._closed = False

    @classmethod
    def load(cls, path: str | Path, *, options: Any = None) -> Recognizer:
        return cls(load_bundle(path, options=options))

    @property
    def runtime_info(self) -> dict[str, Any]:
        return self.bundle.backend.runtime_info

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Recognizer is closed")

    def warmup(self) -> None:
        """Prove a concrete model input/output can run; this is not accuracy validation."""
        self._ensure_open()
        self.bundle.backend.warmup()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.bundle.backend.close()

    def recognize(self, path: str | Path, *, sample_id: str | None = None,
                  representation_id: str | None = None) -> RecognitionResult:
        """Use the bundle's saved processing; optional schema declaration must agree.

        Format/contract errors raise ValueError. Valid but low-quality inputs
        produce a rejected result. Text displays never act as a label allowlist.
        """
        return self.recognize_profiled(path, sample_id=sample_id, representation_id=representation_id)[0]

    def recognize_profiled(self, path: str | Path, *, sample_id: str | None = None,
                           representation_id: str | None = None) -> tuple[RecognitionResult, dict[str, float]]:
        self._ensure_open()
        started = perf_counter()
        manifest = self.bundle.manifest
        if representation_id is not None and representation_id != manifest.input_spec.representation_id:
            raise ValueError("Declared input representation is incompatible with the model")
        identifier = sample_id or str(Path(path).resolve())
        try:
            prepared = prepare_input(path, manifest.input_spec)
        except InputRejected as error:
            elapsed = (perf_counter() - started) * 1000
            return self._rejected(identifier, error), {"preprocess_ms": elapsed, "inference_ms": 0.0, "total_ms": elapsed}
        preprocessing = (perf_counter() - started) * 1000
        result, inference = self._predict_profiled(prepared, identifier)
        return result, {"preprocess_ms": preprocessing, "inference_ms": inference,
                        "total_ms": (perf_counter() - started) * 1000}

    def recognize_clip(self, clip: CapturedClip, *, sample_id: str | None = None) -> RecognitionResult:
        """Recognize one completed captured clip with the saved RGB contract."""
        return self.recognize_clip_profiled(clip, sample_id=sample_id)[0]

    def recognize_clip_profiled(self, clip: CapturedClip, *, sample_id: str | None = None) -> tuple[RecognitionResult, dict[str, float]]:
        from .live import prepare_clip

        self._ensure_open()
        started = perf_counter()
        identifier = sample_id or clip.source_id
        try:
            prepared = prepare_clip(clip, self.bundle.manifest.input_spec)
        except InputRejected as error:
            elapsed = (perf_counter() - started) * 1000
            return self._rejected(identifier, error), {"preprocess_ms": elapsed, "inference_ms": 0.0, "total_ms": elapsed}
        preprocessing = (perf_counter() - started) * 1000
        result, inference = self._predict_profiled(prepared, identifier)
        return result, {"preprocess_ms": preprocessing, "inference_ms": inference,
                        "total_ms": (perf_counter() - started) * 1000}

    def recognize_prepared(self, prepared: PreparedInput, *, sample_id: str | None = None) -> RecognitionResult:
        self._ensure_open()
        if not isinstance(prepared, PreparedInput) or prepared.spec != self.bundle.manifest.input_spec:
            raise ValueError("Prepared input is incompatible with the model InputSpec")
        return self._predict(prepared, sample_id or prepared.source_ref)

    def _rejected(self, identifier: str, error: InputRejected) -> RecognitionResult:
        manifest = self.bundle.manifest
        return RecognitionResult("rejected", identifier, manifest.run_id, manifest.sign_language,
                                 None, error.quality, error.reason)

    def _predict(self, prepared: PreparedInput, identifier: str) -> RecognitionResult:
        return self._predict_profiled(prepared, identifier)[0]

    def _predict_profiled(self, prepared: PreparedInput, identifier: str) -> tuple[RecognitionResult, float]:
        started = perf_counter()
        probabilities = self.bundle.backend.predict_proba([prepared])[0]
        inference = (perf_counter() - started) * 1000
        return self._prediction_result(prepared, identifier, probabilities), inference

    def _prediction_result(self, prepared: PreparedInput, identifier: str, probabilities: Any) -> RecognitionResult:
        manifest = self.bundle.manifest
        scores = {label: float(value) for label, value in zip(self.bundle.backend.classes, probabilities, strict=True)}
        label = max(scores, key=scores.get)
        if scores[label] < self.bundle.rejection_threshold:
            return RecognitionResult("rejected", identifier, manifest.run_id, manifest.sign_language,
                                     None, prepared.quality, "below_validation_selected_score_threshold")
        prediction = ClassificationPrediction(label, scores[label], scores, self.bundle.labels[label])
        time_range = (prepared.timestamps[0], prepared.timestamps[-1]) if prepared.timestamps else None
        return RecognitionResult("recognized", identifier, manifest.run_id, manifest.sign_language,
                                 prediction, prepared.quality, time_range=time_range)

    def begin_stream(self) -> None:
        raise NotImplementedError("capability unavailable: this model supports clip inference only")
