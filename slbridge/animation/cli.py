"""Run the local animation workspace, optionally with existing recognition models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local CSL animation research workspace")
    parser.add_argument("--archive", type=Path, required=True, help="Server-owned SignBridge motion ZIP")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--presentation", type=Path, help="Optional checked presentation bundle directory")
    parser.add_argument("--corpus", type=Path, help="Optional full SOKE library directory containing corpus.sqlite3")
    parser.add_argument("--catalog", type=Path, help="Existing strict v1 recognition model catalog")
    parser.add_argument("--model", help="Optional initial recognition model alias")
    parser.add_argument("--port", type=int, default=8874)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        from .registry import MotionRegistry
        checksum = args.expected_sha256
        sidecar = Path(str(args.archive) + ".sha256")
        if checksum is None and sidecar.is_file():
            checksum = sidecar.read_text(encoding="utf-8-sig").split()[0]
        registry = MotionRegistry(args.archive, checksum)
        if args.corpus:
            from .corpus_store import CorpusStore
            registry.corpus = CorpusStore(args.corpus)
        if args.presentation:
            from .presentation import PresentationBundle
            registry.presentation = PresentationBundle(args.presentation, registry.catalog()["assets"])
        from ..api_cli import load_catalog
        from ..api import create_app
        import uvicorn
        catalog = load_catalog(args.catalog) if args.catalog else {}
        app = create_app(catalog, initial_model=args.model, animation_registry=registry)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(f"Animation workspace: http://127.0.0.1:{args.port}/animation/ "
          f"({len(registry.catalog()['assets'])} research assets)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
