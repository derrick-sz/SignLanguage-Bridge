"""Build a per-sample Uni-Sign prediction/reference comparison table."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unisign_cecsl import edit_similarity, normalize_text  # noqa: E402


def clean(text: str) -> str:
    return "".join(str(text).split())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--selection", type=Path, help="optional selection.json to preserve sample order")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_name = {row["sample"]: row for row in predictions}
    order = sorted(by_name)
    if args.selection and args.selection.is_file():
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        order = [row["sample"] for row in selection.get("samples", []) if row["sample"] in by_name]
        order += [name for name in sorted(by_name) if name not in order]
    rows = []
    for index, name in enumerate(order, 1):
        row = by_name[name]
        reference = clean(row.get("reference", ""))
        prediction = clean(row.get("prediction", ""))
        status = row.get("status", "unknown")
        if status == "ok" and reference and prediction:
            similarity = edit_similarity(normalize_text(reference), normalize_text(prediction))
            exact = normalize_text(reference) == normalize_text(prediction)
        else:
            similarity = None
            exact = False
        rows.append({
            "index": index,
            "sample": name,
            "frames": row.get("frames_total"),
            "ground_truth": reference,
            "prediction": prediction,
            "char_edit_similarity": similarity,
            "exact_match": exact,
            "status": status,
            "error": row.get("error"),
        })
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Uni-Sign 52-test-subset predictions vs ground truth",
        "",
        f"- attempted: {len(rows)}, ok: {sum(1 for r in rows if r['status'] == 'ok')}, "
        f"error/empty: {sum(1 for r in rows if r['status'] != 'ok')}",
        f"- BLEU-4 (13a): {summary.get('metrics', {}).get('bleu4_13a')}",
        f"- chrF: {summary.get('metrics', {}).get('chrf')}",
        f"- char ROUGE-L F1: {summary.get('metrics', {}).get('rouge_l_f1_char')}",
        f"- mean char edit similarity: {summary.get('metrics', {}).get('mean_char_edit_similarity')}",
        "",
        "| # | sample | ground truth | model translation | char edit sim | exact | status |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for row in rows:
        sim = "" if row["char_edit_similarity"] is None else f"{row['char_edit_similarity']:.4f}"
        lines.append(
            f"| {row['index']} | {row['sample']} | {row['ground_truth']} | {row['prediction']} | "
            f"{sim} | {'Y' if row['exact_match'] else 'N'} | {row['status']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "csv": str(args.csv.resolve()),
        "markdown": str(args.markdown.resolve()),
        "rows": len(rows),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "mean_edit_similarity": statistics.mean(
            r["char_edit_similarity"] for r in rows if r["char_edit_similarity"] is not None
        ) if any(r["char_edit_similarity"] is not None for r in rows) else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
