"""Preserve an existing word library and add provenance-bound product candidates."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slbridge.animation.word_library_merge import merge_word_libraries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--addition', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-candidates', type=int, default=8)
    args = parser.parse_args(argv)
    try:
        result = merge_word_libraries(args.base, args.addition, args.output, args.max_candidates)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({'bundle_id': result['bundle_id'], 'statistics': result['statistics']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
