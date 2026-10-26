"""Merge Uni-Sign prediction runs and summarize them on a fixed selection.

Two jobs share the same code path:

* **subset** — take an existing run's ``predictions.jsonl`` (for example the
  complete 1,176-sample released-pose run) and restrict it to the samples of a
  selection file, recomputing the text metrics on exactly that subset;
* **merge** — combine several ``predictions.jsonl`` files produced by earlier
  runs on disjoint (or partially overlapping) sample sets, keep the last record
  per sample, and summarize the union.

The per-sample metric definitions match
``scripts/evaluate_unisign_csldaily_video.py``: SacreBLEU 13a on
character-separated text after the upstream ``' '.join(list(text.replace(' ','')))``
rewrite, chrF on raw text, char-level ROUGE-L, char edit similarity and exact
match on whitespace-stripped NFKC text.

Outputs ``predictions.jsonl`` (one record per selected sample) and
``summary.json`` (metrics + per_sample) in ``--output-dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_unisign_cecsl import edit_similarity, normalize_text, rouge_l_f1_char  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text) -> str:
    return "".join(str(text).replace("\n", "").split())


def character_separated(text) -> str:
    return " ".join(list(clean_text(text)))


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * (q / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def load_predictions(paths):
    """Last record per sample wins; records are returned as-is."""
    merged = {}
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("sample"):
                merged[row["sample"]] = row
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="Uni-Sign predictions")
    parser.add_argument("--references", type=Path, default=None,
                        help="optional upstream reference file ('sample: X, ground-truth: ...'); "
                             "overrides the per-row reference for scoring and display")
    parser.add_argument("--strict", action="store_true",
                        help="fail if any selected sample is missing from the predictions")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    samples = selection["samples"]
    merged = load_predictions(args.predictions)
    reference_map = {}
    if args.references is not None:
        marker = ", ground-truth: "
        for line in args.references.read_text(encoding="utf-8").splitlines():
            if marker not in line:
                continue
            head, reference = line.split(marker, 1)
            reference_map[head.split("sample: ", 1)[-1].strip()] = reference.strip()

    rows = []
    missing = []
    for item in samples:
        name = item["sample"]
        row = merged.get(name)
        if row is None:
            missing.append(name)
            rows.append({
                "sample": name, "frames": item.get("num_frames"),
                "reference": reference_map.get(name, item.get("reference", "")), "prediction": "",
                "status": "missing", "error": "not present in any predictions file",
                "char_edit_similarity": None, "rouge_l_f1_char": None, "exact_match": False,
            })
            continue
        reference = reference_map.get(name) or row.get("reference") or item.get("reference", "")
        prediction = row.get("prediction") or ""
        status = row.get("status", "ok")
        rows.append({
            "sample": name, "frames": row.get("frames", item.get("num_frames")),
            "reference": reference, "prediction": prediction, "status": status,
            "error": row.get("error"), "gloss": row.get("gloss", item.get("gloss")),
            "char_edit_similarity": row.get("char_edit_similarity"),
            "rouge_l_f1_char": row.get("rouge_l_f1_char"),
            "exact_match": row.get("exact_match"),
            "e2e_seconds": row.get("e2e_seconds"),
            "pose_ms": row.get("pose_ms"),
            "model_total_ms": row.get("model_total_ms"),
        })
    if missing and args.strict:
        raise SystemExit(f"{len(missing)} selected samples missing, e.g. {missing[:5]}")

    successful = [row for row in rows if row["status"] == "ok"]
    for row in rows:
        reference = normalize_text(row["reference"])
        prediction = normalize_text(row["prediction"])
        if row["status"] == "ok":
            row["char_edit_similarity"] = edit_similarity(reference, prediction)
            row["rouge_l_f1_char"] = rouge_l_f1_char(reference, prediction)
            row["exact_match"] = reference == prediction
        else:
            row["char_edit_similarity"] = None
            row["rouge_l_f1_char"] = None
            row["exact_match"] = False

    import sacrebleu

    refs_raw = [character_separated(row["reference"]) for row in successful]
    hyps_raw = [character_separated(row["prediction"]) for row in successful]
    refs_norm = [normalize_text(row["reference"]) for row in successful]
    hyps_norm = [normalize_text(row["prediction"]) for row in successful]
    bleu = sacrebleu.corpus_bleu(hyps_raw, [refs_raw], tokenize="13a") if successful else None
    bleu_raw = sacrebleu.corpus_bleu([row["prediction"] for row in successful],
                                     [[row["reference"] for row in successful]], tokenize="13a") if successful else None
    chrf = (sacrebleu.corpus_chrf([row["prediction"] for row in successful],
                                  [[row["reference"] for row in successful]]).score
            if successful else None)
    edits = [edit_similarity(a, b) for a, b in zip(refs_norm, hyps_norm)]
    rouges = [rouge_l_f1_char(a, b) for a, b in zip(refs_norm, hyps_norm)]
    exact = sum(a == b for a, b in zip(refs_norm, hyps_norm))
    e2e = [row["e2e_seconds"] for row in successful if row.get("e2e_seconds") is not None]
    pose_ms = [row["pose_ms"] for row in successful if row.get("pose_ms") is not None]
    model_ms = [row["model_total_ms"] for row in successful if row.get("model_total_ms") is not None]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "label": args.label,
        "scope": f"{args.label}: merged {len(args.predictions)} prediction file(s), restricted to the "
                 f"{len(samples)} samples of {Path(args.selection).name}",
        "inputs": {
            "selection": str(args.selection.resolve()),
            "selection_sha256": sha256(args.selection),
            "references": (str(args.references.resolve()) if args.references else None),
            "references_sha256": (sha256(args.references) if args.references else None),
            "predictions": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in args.predictions],
        },
        "reference_source": (str(args.references.resolve()) if args.references
                             else "per-row reference (evaluation run)"),
        "attempted": len(rows), "successful": len(successful),
        "failed_or_empty": len(rows) - len(successful),
        "missing_samples": missing,
        "failed_samples": [{"sample": row["sample"], "status": row["status"], "error": row["error"]}
                           for row in rows if row["status"] != "ok"],
        "metrics": {
            "n": len(successful),
            "bleu4_13a": (bleu.score if bleu else None),
            "bleu4_13a_raw_no_mt5_fix": (bleu_raw.score if bleu_raw else None),
            "bleu4_precisions": ([round(v, 2) for v in bleu.precisions] if bleu else None),
            "bp": (round(bleu.bp, 4) if bleu else None),
            "chrf": chrf,
            "rouge_l_f1_char": statistics.fmean(rouges) if rouges else None,
            "mean_char_edit_similarity": statistics.fmean(edits) if edits else None,
            "edit_similarity_p50": percentile(edits, 50),
            "exact_match_count": exact,
            "prediction_chars_total": sum(len(text) for text in hyps_norm),
            "reference_chars_total": sum(len(text) for text in refs_norm),
        },
        "latency": {
            "e2e_seconds_mean": statistics.fmean(e2e) if e2e else None,
            "e2e_seconds_p50": percentile(e2e, 50),
            "e2e_seconds_p95": percentile(e2e, 95),
            "pose_ms_mean": statistics.fmean(pose_ms) if pose_ms else None,
            "model_total_ms_mean": statistics.fmean(model_ms) if model_ms else None,
            "note": "Latency rows come from the original runs; values may include GPU contention.",
        },
        "per_sample": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "attempted": len(rows),
                      "successful": len(successful), "missing": len(missing),
                      "metrics": summary["metrics"], "latency": summary["latency"]},
                     ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
