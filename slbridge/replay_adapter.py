"""ReplayAdapter (``replay_jsonl_v1``): GT-free historical-prediction replay.

P0-A phase A1, task T-A1-1.  This adapter replays **frozen, GT-free**
historical predictions as if they were model outputs, so the whole
"query -> inference -> evidence -> prediction log" chain runs without a GPU:

* text candidates come from ``experiments/p1/e02/candidate_outputs.jsonl``
  (rows shaped like the frozen P-1 ``EvidenceCandidate`` schema);
* SLRT gloss sequences come from ``experiments/p1/e02/slrt_gloss_predictions.jsonl``.

Replay contract -- verbatim, never rewritten:

* replayed text/gloss must be byte-identical to the source row ("a replay
  adapter must never polish predictions"); tests assert this against the
  real frozen sources (Pilot-50 A candidates: 50/50 verbatim);
* illegal inputs fail loudly: unknown sample_id, unknown candidate_id,
  missing query fields, OK-status rows with empty text, and any GT-derived
  key in the source (the T-A0-1 guard runs over every loaded row);
* ``limit`` (manifest field) truncates how many rows are loaded, so partial
  replay fixtures stay cheap.

Discipline: this module never reads anything below ``experiments/p1/references/``
(GT lives only on the evaluation side) and never touches the verified inference
paths of ``uni_sign.py`` / ``slrt/``.  It registers as a plain ModelAdapter
through the existing backend registry; construction validates data eagerly,
Torch/GPU loading is never involved.

Model bundle layout (the "model directory" of a replay backend)::

    <bundle>/manifest.json
    {
      "backend": "replay_jsonl_v1",
      "source_kind": "text_candidates" | "slrt_gloss",
      "candidate_id": "A",              # text_candidates only
      "model_id": "unisign_rgb",        # optional override / sanity label
      "data_path": "experiments/p1/e02/candidate_outputs.jsonl",  # abs or repo-relative
      "limit": null                     # optional max rows to load
    }

Minimal bundle setup (PowerShell, repo root)::

    $b = "artifacts/replay-bundles/unisign-a"   # any directory OUTSIDE git
    New-Item -ItemType Directory -Force $b | Out-Null
    @{ backend = "replay_jsonl_v1"; source_kind = "text_candidates";
       candidate_id = "A";
       data_path = "E:/githubproject/SignLanguage-Bridge/experiments/p1/e02/candidate_outputs.jsonl" } |
      ConvertTo-Json | Set-Content "$b/manifest.json" -Encoding UTF8
    # Serve it like any other catalog model:
    #   python -m slbridge.api_cli --catalog <catalog.json> --model <alias>
    # where catalog.json maps the alias to this bundle directory (see
    # slbridge/api_cli.py:load_catalog for the strict v1 catalog format).
    # A bundle binds its source by path only; the loader re-runs the GT guard
    # and row validation on every start, so a swapped-in source carrying GT
    # keys or corrupt rows fails loudly instead of replaying silently.

Input queries are small JSON files (``{"sample_id": "..."}``) handed to
``ModelManager.infer_file`` exactly like an uploaded video would be; the
adapter resolves the sample against its loaded rows and returns a typed
``RecognitionResult`` that passes ``validate_adapter_result``.  A direct
``replay(sample_id)`` API is also provided for the job/evidence layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .contracts import GlossSequencePrediction, GlossToken, RecognitionResult, TextPrediction
from .evidence import assert_gt_clean
from .model_adapters import (ModelAdapter, ModelCapabilities, ModelInputContract,
                             InputRepresentation, get_input_representation,
                             register_input_representation)

__all__ = ["ReplayAdapter", "load_replay_adapter", "REPLAY_BACKEND_ID"]

REPLAY_BACKEND_ID = "replay_jsonl_v1"
_REPRESENTATION = "replay_query_v1"
_SCORE_TYPES = {"text_candidates": "replay_imported_text_no_score",
                "slrt_gloss": "replay_imported_gloss_no_score"}
_SOURCE_KINDS = {"text_candidates", "slrt_gloss"}


def _register_representation() -> None:
    try:
        get_input_representation(_REPRESENTATION)
    except ValueError:
        # A JSON query document has no tensor shape; the contract is nominal.
        register_input_representation(InputRepresentation(_REPRESENTATION, (None,), dtype="str"))


@dataclass(frozen=True)
class _SourceSpec:
    source_kind: str
    candidate_id: str | None
    model_id: str
    data_path: Path
    limit: int | None


def _read_manifest(path: Path) -> _SourceSpec:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Replay bundle requires manifest.json: {path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("backend") != REPLAY_BACKEND_ID:
        raise ValueError(f"manifest.json backend must be {REPLAY_BACKEND_ID}")
    source_kind = raw.get("source_kind")
    if source_kind not in _SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(_SOURCE_KINDS)}")
    candidate_id = raw.get("candidate_id")
    if source_kind == "text_candidates":
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("text_candidates replay requires a nonempty candidate_id")
    elif candidate_id is not None:
        raise ValueError("slrt_gloss replay must not declare candidate_id")
    data_path = raw.get("data_path")
    if not isinstance(data_path, str) or not data_path.strip():
        raise ValueError("data_path is required")
    resolved = Path(data_path)
    if not resolved.is_absolute():
        resolved = (Path(__file__).resolve().parents[1] / resolved)
    resolved = resolved.resolve()
    if "references" in resolved.parts:
        raise ValueError(f"Replay sources must never point into references/: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Replay source file not found: {resolved}")
    limit = raw.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or null")
    model_id = raw.get("model_id")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ValueError("model_id must be a nonempty string or null")
    return _SourceSpec(source_kind=source_kind, candidate_id=candidate_id,
                       model_id=model_id, data_path=resolved, limit=limit)


def _rows_from_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Replay source rows must be JSON objects: {path}")
            # GT guard runs over every loaded row, at load time, once.
            assert_gt_clean(row, context=f"replay source row {len(rows)} of {path.name}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"Replay source contains no rows: {path}")
    return rows


def _load_text_candidates(spec: _SourceSpec) -> dict[str, dict[str, Any]]:
    """Load one candidate's rows, keyed by sample_id (later rows win)."""
    index: dict[str, dict[str, Any]] = {}
    for row in _rows_from_jsonl(spec.data_path, spec.limit):
        if row.get("candidate_id") != spec.candidate_id:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("text_candidates rows require nonempty sample_id")
        status = row.get("status")
        if status not in {"OK", "FAILED", "TIMEOUT", "SKIPPED", "BLOCKED"}:
            raise ValueError(f"Unsupported replay row status: {status!r}")
        output = row.get("output")
        if not isinstance(output, dict) or output.get("kind") not in {"TEXT", "GLOSS", "EMPTY"}:
            raise ValueError("text_candidates rows require a typed output object")
        if status == "OK":
            text = output.get("text")
            if output.get("kind") == "TEXT" and (not isinstance(text, str) or not text.strip()):
                raise ValueError(
                    f"OK-status replay row has empty text (corrupt source): {sample_id}")
        index[sample_id] = row
    if not index:
        raise ValueError(
            f"No rows for candidate {spec.candidate_id!r} in {spec.data_path}")
    return index


def _load_gloss_rows(spec: _SourceSpec) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """Load SLRT gloss rows keyed by sample; build the token vocabulary."""
    index: dict[str, dict[str, Any]] = {}
    vocabulary: set[str] = set()
    for row in _rows_from_jsonl(spec.data_path, spec.limit):
        sample = row.get("sample")
        if not isinstance(sample, str) or not sample.strip():
            raise ValueError("slrt_gloss rows require nonempty sample")
        status = row.get("status")
        if status != "final":
            raise ValueError(f"Unsupported slrt_gloss row status: {status!r}")
        if row.get("error") is not None:
            raise ValueError(f"slrt_gloss rows must carry error=null: {sample}")
        glosses = row.get("gloss_sequence")
        if (not isinstance(glosses, list) or not glosses
                or any(not isinstance(token, str) or not token.strip() for token in glosses)):
            raise ValueError(f"slrt_gloss rows require nonempty gloss_sequence: {sample}")
        vocabulary.update(glosses)
        index[sample] = row
    # GlossToken reserves id 0 for blank; real glosses occupy ids 1..n so
    # validate_adapter_result's vocabulary indexing holds for every token.
    return index, ("<blank>",) + tuple(sorted(vocabulary - {"<blank>"}))


class ReplayAdapter:
    """A plain ModelAdapter that replays frozen GT-free predictions."""

    def __init__(self, spec: _SourceSpec) -> None:
        _register_representation()
        self._spec = spec
        self._closed = False
        if spec.source_kind == "text_candidates":
            self._index = _load_text_candidates(spec)
            self._vocabulary: tuple[str, ...] = ()
            task_type = "text"
            first = next(iter(self._index.values()))
            model_id = spec.model_id or first["model_id"]
            model_version = str(first.get("model_revision") or "replay-imported-v1")
            preprocess_version = str(first.get("preprocess_version") or "replay-imported-v1")
            self._modality = str(first.get("modality") or "replay")
        else:
            self._index, self._vocabulary = _load_gloss_rows(spec)
            task_type = "gloss_sequence"
            model_id = spec.model_id or "slrt-cslr"
            model_version = "slrt-1086-replay"
            preprocess_version = "slrt-rgb-hrnet133-replay"
            self._modality = "hrnet_npz_replay"
        self._task_type = task_type
        self._model_version = model_version
        self._preprocess_version = preprocess_version
        self.capabilities = ModelCapabilities(
            model_id, model_version, "csl", task_type,
            (ModelInputContract(_REPRESENTATION, (None,), "str",
                                preprocess_version, ("file", "captured_clip")),),
            self._vocabulary, REPLAY_BACKEND_ID, ("cpu",), ("clip",),
            _SCORE_TYPES[spec.source_kind])

    # -- ModelAdapter protocol ------------------------------------------
    @property
    def recognizer(self) -> "ReplayAdapter":
        return self

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": REPLAY_BACKEND_ID,
            "replay_source": {
                "source_kind": self._spec.source_kind,
                "candidate_id": self._spec.candidate_id,
                "data_path": str(self._spec.data_path),
                "row_count": len(self._index),
                "limit": self._spec.limit,
                "verbatim_replay": True,
            },
            "device": None,   # CPU-only bookkeeping; no inference runtime
        }

    def warmup(self) -> None:
        self._check_open()   # data was validated eagerly at construction

    def close(self) -> None:
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Replay adapter is closed")

    # -- replay ----------------------------------------------------------
    def replay(self, sample_id: str, *, request_id: str | None = None) -> RecognitionResult:
        """Replay one historical prediction, verbatim, as a typed result."""
        self._check_open()
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        row = self._index.get(sample_id)
        if row is None:
            raise ValueError(f"unknown sample for candidate {self._spec.candidate_id!r}: {sample_id}")
        return self._result_from_row(row, request_id or sample_id)

    def _result_from_row(self, row: dict[str, Any], request_id: str) -> RecognitionResult:
        quality = {"replay": True,
                   "replay_run_id": row.get("run_id"),
                   "replay_source_kind": self._spec.source_kind,
                   "replay_candidate_id": self._spec.candidate_id,
                   "replay_sample_id": row.get("sample_id") or row.get("sample"),
                   "replay_modality": self._modality}
        if self._task_type == "text":
            status = row.get("status")
            output = row.get("output", {})
            if status != "OK":
                return RecognitionResult(
                    status="rejected", sample_id=request_id, model_id=self.capabilities.model_id,
                    sign_language="csl", prediction=None, quality=quality,
                    reason=f"source_status_{status}", task_type="text",
                    score_type=self.capabilities.score_type, time_range=None)
            text = output.get("text")
            window = row.get("window") or {}
            time_range = None
            if isinstance(window.get("start_pts_ms"), (int, float)) \
                    and isinstance(window.get("end_pts_ms"), (int, float)):
                time_range = (float(window["start_pts_ms"]) / 1000.0,
                              float(window["end_pts_ms"]) / 1000.0)
            return RecognitionResult(
                status="recognized", sample_id=request_id, model_id=self.capabilities.model_id,
                sign_language="csl",
                prediction=TextPrediction(text=text, language="zh", score=None),
                quality=quality, task_type="text",
                score_type=self.capabilities.score_type, time_range=time_range)
        glosses = row.get("gloss_sequence")
        tokens = tuple(GlossToken(self._vocabulary.index(gloss), gloss, None)
                       for gloss in glosses)
        return RecognitionResult(
            status="recognized", sample_id=request_id, model_id=self.capabilities.model_id,
            sign_language="csl",
            # The imported rows were produced by SLRT's window-majority decoder.
            prediction=GlossSequencePrediction(tokens, "slrt_window_majority"),
            quality=quality, task_type="gloss_sequence",
            score_type=self.capabilities.score_type, time_range=None)

    # -- ModelAdapter.infer ----------------------------------------------
    def infer(self, source: Any, request: Any, *, input_kind: str) -> tuple[RecognitionResult, dict[str, float]]:
        self._check_open()
        self.capabilities.negotiate(request, input_kind=input_kind)
        request.check_active()
        if input_kind not in {"file", "captured_clip"}:
            raise ValueError("ReplayAdapter supports file and captured_clip inputs only")
        started = perf_counter()
        sample_id = self._resolve_query(source)
        result = self.replay(sample_id, request_id=request.request_id)
        elapsed = (perf_counter() - started) * 1000.0
        return result, {"replay_ms": elapsed}

    def _resolve_query(self, source: Any) -> str:
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Replay query file not found: {path}")
        try:
            query = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"Replay query must be a JSON document: {path}") from error
        if not isinstance(query, dict):
            raise ValueError("Replay query must be a JSON object with sample_id")
        sample_id = query.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Replay query requires a nonempty sample_id")
        return sample_id


def load_replay_adapter(path: str | Path, options: Any = None) -> ReplayAdapter:
    spec = _read_manifest(Path(path).resolve())
    return ReplayAdapter(spec)
