"""Build an independent frozen v11 presentation; no Blender or corpus writes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Existing server-owned motion ZIP (read-only)')
    parser.add_argument('--expected-sha256', help='Optional expected archive checksum; otherwise uses ZIP.sha256 if present')
    parser.add_argument('--avatar', type=Path, required=True, help='Frozen godette-reference-v11.glb, verified by SHA256')
    parser.add_argument('--output', type=Path, required=True, help='New presentation directory; existing directories are refused')
    args = parser.parse_args(argv)
    try:
        from slbridge.animation.registry import MotionRegistry
        from slbridge.animation.godette_v11_presentation import build_godette_v11_presentation
        checksum = args.expected_sha256
        sidecar = Path(str(args.archive) + '.sha256')
        if checksum is None and sidecar.is_file():
            words = sidecar.read_text(encoding='utf-8-sig').split()
            if not words:
                raise ValueError('Empty archive checksum sidecar')
            checksum = words[0]
        registry = MotionRegistry(args.archive, checksum)
        build_godette_v11_presentation(args.output, avatar=args.avatar, catalog=registry.catalog())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f'Godette v11 presentation: {args.output.resolve()} (6 source-bound pilots; research only)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
