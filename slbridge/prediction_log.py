"""Product-side JSONL prediction recorder (P0-A phase A0, task T-A0-1).

Each row written by :class:`PredictionRecorder` has two blocks:

* ``candidate`` -- a frozen-schema :class:`slbridge.evidence.EvidenceCandidate`
  payload (byte-compatible with ``experiments/p1/schemas/evidence_candidate.schema.json``,
  so the same JSONL can feed the canonical ``p1-v1`` evaluator in
  ``experiments/p1/eval/metrics.py`` unchanged; text post-processing such as
  ``character_separated`` belongs to the evaluator, the recorder stores raw text).
* top-level slbridge-flavoured bookkeeping fields required by the task card:
  ``request_id/job_id/model_id/model_version/preprocessing_version/sign_system/
  task_type/text|gloss_tokens/timing_ms/error``.

Writes are GT-guarded on every append (same rule set as
``experiments/p1/gt_guard.py``): a row carrying ``reference``/``gt_*``/``wer``/
``edit_similarity``/... raises before anything touches the file, so failed
rows are never partially written.  ``manifest.json`` is generated per run with
git state aligned to ``experiments/p1/common.py:git_state`` plus the model /
preprocessing registration that produced the run.

Predeclared-but-unused calibration fields are intentionally absent here; the
task card reserves ``policy_version``/``prompt_version`` slots for the G2 track,
so ``build_manifest`` accepts and records them verbatim when provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .evidence import (EvidenceCandidate, EvidenceOutput, EvidenceQuality,
                       EvidenceWindow, assert_gt_clean, from_slbridge_response)

__all__ = ["PredictionRecorder", "build_manifest", "RECORDER_ROW_VERSION"]

RECORDER_ROW_VERSION = 1
_SCHEMA_VERSION = "1.0"

_ROW_FIELDS = ("request_id", "job_id", "model_id", "model_version",
               "preprocessing_version", "sign_system", "task_type",
               "text", "gloss_tokens", "timing_ms", "error")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _check_timing(timing: Any) -> None:
    if not isinstance(timing, dict):
        raise ValueError("timing_ms must be a JSON object")
    for key, value in timing.items():
        _text(key, "timing_ms key")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f"timing_ms.{key} must be a finite nonnegative number")


def _check_error(error: Any) -> None:
    if error is None:
        return
    if not isinstance(error, dict):
        raise ValueError("error must be a JSON object or null")
    unknown = set(error) - {"code", "message"}
    if unknown:
        raise ValueError(f"Unknown error fields: {sorted(unknown)}")
    _text(error.get("code"), "error.code")
    message = error.get("message")
    if message is not None and not isinstance(message, str):
        raise ValueError("error.message must be a string or null")


def _row_to_candidate(row: Mapping[str, Any], run_id: str, modality: str) -> EvidenceCandidate:
    """Build the frozen-schema candidate payload from a recorder row."""

    request_id = row.get("request_id")
    _text(request_id, "request_id")
    task_type = row.get("task_type")
    if task_type not in {"text", "gloss_sequence", "classification"}:
        raise ValueError("task_type must be 'text', 'gloss_sequence' or 'classification'")
    sign_system = row.get("sign_system")
    if sign_system not in {"CSL", "Libras", "Unknown"}:
        raise ValueError("sign_system must be 'CSL', 'Libras' or 'Unknown'")
    text = row.get("text")
    gloss_tokens = row.get("gloss_tokens")
    error = row.get("error")
    if task_type == "text":
        if error is None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("text rows require nonempty text")
        if text is not None and not (isinstance(text, str) and text.strip()):
            raise ValueError("text must be nonempty when present")
        if gloss_tokens is not None:
            raise ValueError("text rows must not carry gloss_tokens")
        output = (EvidenceOutput(kind="TEXT", gloss_tokens=None, text=text, spoken_language="zh")
                  if error is None else
                  EvidenceOutput(kind="EMPTY", gloss_tokens=None, text=None))
    elif task_type == "gloss_sequence":
        if (not isinstance(gloss_tokens, list) or not gloss_tokens
                or any(not isinstance(token, str) or not token.strip() for token in gloss_tokens)):
            raise ValueError("gloss_sequence rows require nonempty gloss_tokens")
        if text is not None:
            raise ValueError("gloss rows must not carry text")
        output = EvidenceOutput(kind="GLOSS", gloss_tokens=tuple(gloss_tokens), text=None)
    else:
        raise ValueError("classification rows are not recorded by the P0-A recorder yet")

    status = "FAILED" if error is not None else "OK"
    if error is not None and text is not None:
        raise ValueError("error rows must not carry text")
    window = EvidenceWindow(start_frame=0, end_frame_exclusive=None,
                            start_pts_ms=None, end_pts_ms=None, needs_future_frames=None)
    return EvidenceCandidate(
        run_id=run_id, sample_id=request_id, model_id=row.get("model_id"),
        task={"text": "CSLT", "gloss_sequence": "CSLR", "classification": "ISLR"}[task_type],
        sign_system=sign_system, modality=modality,
        model_revision=row.get("model_version"), preprocess_version=row.get("preprocessing_version"),
        window=window, output=output, status=status,
        attempt=1, window_id=f"{request_id}-w1", quality=EvidenceQuality())


def git_state() -> dict[str, Any]:
    """Return repository identity aligned with ``experiments/p1/common.py``.

    Deliberately an independent copy (the product package must not import the
    experiments subpackage); shape and semantics mirror the P-1 helper so runs
    stay comparable.  Never raises.
    """

    root = Path(__file__).resolve().parents[1]

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(root), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance must not break the run
            return ""

    status = run("status", "--porcelain")
    return {
        "repository": "SignLanguage-Bridge",
        "git_sha": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty_tree": bool(status),
        "dirty_path_count": len([line for line in status.splitlines() if line.strip()]),
    }


def build_manifest(*, run_id: str, models: list[dict[str, Any]], row_count: int,
                   purpose: str = "p0a-prediction-recording",
                   policy_version: str | None = None,
                   prompt_version: str | None = None) -> dict[str, Any]:
    """Assemble a per-run manifest (git state + model/preprocess registration)."""

    _text(run_id, "run_id")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list of registration records")
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("each model entry must be a JSON object")
        for key in ("model_id", "model_version", "preprocess_version", "task", "sign_system"):
            _text(model.get(key), f"models[].{key}")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a nonnegative integer")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_status": "RUNNING",
        "created_at": _utc_now(),
        "purpose": purpose,
        "metric_definition_version": "p1-v1",
        "code": {"repository": "SignLanguage-Bridge", **git_state()},
        "models": [dict(model) for model in models],
        "row_count": row_count,
        "privacy_and_licensing": {"gt_free_prediction_log": True},
    }
    # Reserved G2 slots: recorded verbatim when provided, never consumed here.
    if policy_version is not None:
        manifest["policy_version"] = policy_version
    if prompt_version is not None:
        manifest["prompt_version"] = prompt_version
    return manifest


@dataclass
class PredictionRecorder:
    """Append-only GT-free JSONL prediction log for one run.

    The recorder is bound to a single model registration (task/sign system/
    preprocess version) on construction; rows that disagree with that
    registration are rejected so one run can never mix calibers.
    """

    directory: Path
    run_id: str
    model_id: str
    model_version: str
    preprocess_version: str
    task: str
    sign_system: str

    def __post_init__(self) -> None:
        for name in ("run_id", "model_id", "model_version", "preprocess_version"):
            _text(getattr(self, name), name)
        if self.task not in {"CSLR", "CSLT", "ISLR"}:
            raise ValueError("task must be 'CSLR', 'CSLT' or 'ISLR'")
        if self.sign_system not in {"CSL", "Libras", "Unknown"}:
            raise ValueError("sign_system must be 'CSL', 'Libras' or 'Unknown'")
        if "/" in self.run_id or "\\" in self.run_id or self.run_id.strip() != self.run_id:
            raise ValueError("run_id must be a plain directory name")
        self.directory = Path(self.directory)
        self.predictions_path = self.directory / "predictions.jsonl"
        self.count = 0
        self.modality = self.preprocess_version  # default modality label

    def append(self, row: Mapping[str, Any]) -> bool:
        """Validate, GT-guard and append one prediction row.

        Returns ``True`` when the row was written.  Any invalid or leaking row
        raises before the file is touched, so the log never contains partial
        or rejected writes.
        """

        if not isinstance(row, dict):
            raise ValueError("row must be a JSON object")
        # GT guard first: a leaking row must fail with a ground-truth error,
        # before any structural complaint about its (forbidden) extra fields.
        assert_gt_clean(row, context="prediction row")
        missing = [name for name in _ROW_FIELDS if name not in row]
        if missing:
            raise ValueError(f"Prediction row is missing fields: {missing}")
        unknown = set(row) - set(_ROW_FIELDS)
        if unknown:
            raise ValueError(f"Unknown prediction row fields: {sorted(unknown)}")
        _check_timing(row["timing_ms"])
        _check_error(row["error"])
        if row["model_id"] != self.model_id or row["model_version"] != self.model_version \
                or row["preprocessing_version"] != self.preprocess_version:
            raise ValueError("Row model registration disagrees with the recorder run")
        if row["sign_system"] != self.sign_system:
            raise ValueError("Row sign_system disagrees with the recorder run")
        expected_task = {"CSLT": "text", "CSLR": "gloss_sequence", "ISLR": "classification"}[self.task]
        if row["task_type"] != expected_task:
            raise ValueError("Row task_type disagrees with the recorder run")

        candidate = _row_to_candidate(row, run_id=self.run_id, modality=self.modality)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "recorder_row_version": RECORDER_ROW_VERSION,
            "run_id": self.run_id,
            "recorded_at": _utc_now(),
            **{name: row[name] for name in _ROW_FIELDS},
            "candidate": candidate.to_dict(),
        }
        assert_gt_clean(payload, context=f"prediction record for {self.predictions_path.name}")
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.predictions_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        self.count += 1
        return True

    def append_response(self, response: Any, *, job_id: str | None = None) -> EvidenceCandidate:
        """Convert and append a slbridge ``InferenceResponse`` in one step."""

        candidate = from_slbridge_response(response, run_id=self.run_id, modality=self.modality)
        timing = getattr(response, "timing_ms", None) or {}
        task_type = {"CSLT": "text", "CSLR": "gloss_sequence", "ISLR": "classification"}[self.task]
        self.append({
            "request_id": candidate.sample_id,
            "job_id": job_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "preprocessing_version": self.preprocess_version,
            "sign_system": self.sign_system,
            "task_type": task_type,
            "text": (candidate.output.text if candidate.output.kind == "TEXT" else None),
            "gloss_tokens": (list(candidate.output.gloss_tokens)
                             if candidate.output.kind == "GLOSS" else None),
            "timing_ms": {key: value for key, value in timing.items()
                          if isinstance(value, (int, float)) and not isinstance(value, bool)},
            "error": None,
        })
        return candidate

    def write_manifest(self, **extra: Any) -> dict[str, Any]:
        """Write ``manifest.json`` for this run and return it."""

        manifest = build_manifest(
            run_id=self.run_id,
            models=[{
                "model_id": self.model_id,
                "model_version": self.model_version,
                "preprocess_version": self.preprocess_version,
                "task": self.task,
                "sign_system": self.sign_system,
            }],
            row_count=self.count,
        )
        manifest.update(extra)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.predictions_path.with_name("manifest.json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.directory / "manifest.json")
        return manifest
