"""Convert ``prediction_slide.py`` results to the annotation pickles used by SLT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slbridge.slrt.slt_export import (  # noqa: E402
    DEFAULT_DECODE_KEY,
    SltExportError,
    convert_results_to_annotations,
    load_pickle,
    write_annotation_pickle,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="*_results.pkl from prediction_slide.py")
    p.add_argument("--sentence-meta", type=Path, required=True, help="csl.dev or csl.test sentence annotation pickle")
    p.add_argument("--output", type=Path, required=True, help="csl_pred.dev / csl_pred.test")
    p.add_argument("--decode-key", default=DEFAULT_DECODE_KEY)
    p.add_argument("--allow-missing", action="store_true", help="write empty glosses for missing names (diagnostics only)")
    p.add_argument("--summary", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = load_pickle(args.results)
    annotations = load_pickle(args.sentence_meta)
    try:
        items, summary = convert_results_to_annotations(
            results, annotations, decode_key=args.decode_key, allow_missing=args.allow_missing
        )
    except SltExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_annotation_pickle(args.output, items)
    summary.update({"results": str(args.results), "sentence_meta": str(args.sentence_meta), "output": str(args.output)})
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
