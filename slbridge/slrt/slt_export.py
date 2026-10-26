"""Convert official online-CSLR predictions into the annotation pickles used by
``Online/SLT``.

``prediction_slide.py`` writes ``{split}_results.pkl`` (a name -> result dict).
``Online/SLT`` expects ``csl_pred.{dev,test}`` pickles whose items carry the
predicted ``gloss`` (G2T source), the original ``text`` target, ``name`` and
``num_frames``. The upstream repository does not ship that converter.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

DEFAULT_DECODE_KEY = "window_greedy_7_gls_hyp"
FALLBACK_DECODE_KEYS = ("window_greedy_7_gls_hyp", "naive_greedy_gls_hyp")


class SltExportError(ValueError):
    pass


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)


def _annotation_list(raw: Any, path: Path) -> list[dict]:
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise SltExportError(f"annotation file must contain a list/dict of dicts: {path}")
    return raw


def _pick_prediction(result: dict, decode_key: str | None) -> tuple[str, str]:
    keys = []
    if decode_key:
        keys.append(decode_key)
    keys.extend(key for key in FALLBACK_DECODE_KEYS if key not in keys)
    for key in keys:
        value = result.get(key)
        if isinstance(value, str):
            return value, key
    raise SltExportError(f"prediction result has none of {keys!r}")


def convert_results_to_annotations(
    results: dict,
    annotations: list[dict] | dict,
    *,
    decode_key: str | None = DEFAULT_DECODE_KEY,
    allow_missing: bool = False,
) -> tuple[list[dict], dict]:
    """Return (annotations with predicted gloss, summary)."""
    items = _annotation_list(annotations, Path("<annotations>"))
    if not isinstance(results, dict):
        raise SltExportError("prediction results must be a mapping name -> result dict")
    output = []
    missing = []
    empty = []
    used_keys = set()
    seen = set()
    for item in items:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise SltExportError("every annotation needs a non-empty string 'name'")
        if name in seen:
            raise SltExportError(f"duplicate annotation name: {name}")
        seen.add(name)
        result = results.get(name)
        if result is None:
            missing.append(name)
            if not allow_missing:
                continue
            prediction, key = "", "missing"
        else:
            if not isinstance(result, dict):
                raise SltExportError(f"result for {name} is not a dict")
            prediction, key = _pick_prediction(result, decode_key)
        if not prediction.strip():
            empty.append(name)
        new_item = dict(item)
        new_item["gls_ref"] = item.get("gloss", "")
        new_item["gloss"] = prediction.strip()
        new_item["prediction_decode_key"] = key
        output.append(new_item)
        used_keys.add(key)
    if missing and not allow_missing:
        raise SltExportError(
            f"{len(missing)} annotation names are absent from the prediction results; "
            f"first={missing[:5]}; rerun with --allow-missing only for diagnostics"
        )
    summary = {
        "annotations": len(items),
        "written": len(output),
        "missing_results": len(missing),
        "empty_predictions": len(empty),
        "decode_keys_used": sorted(used_keys),
        "missing_names": missing[:50],
        "empty_names": empty[:50],
    }
    return output, summary


def write_annotation_pickle(path: str | Path, items: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(items, handle, protocol=4)
