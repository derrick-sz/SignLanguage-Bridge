"""Compatibility launcher for running the desktop directly from a checkout."""
from __future__ import annotations
from pathlib import Path
import sys

# Direct script execution places app/, not the project root, on sys.path.
base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

from slbridge.desktop_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
