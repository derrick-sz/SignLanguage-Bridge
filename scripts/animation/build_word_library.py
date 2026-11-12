"""Export authenticated native SignSparK word spans for fast composition."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slbridge.animation.word_library_build import build_word_library


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmdb', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--base-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-lmdb-sha256', required=True)
    parser.add_argument('--expected-annotations-sha256', required=True)
    parser.add_argument('--max-candidates', type=int, default=3)
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='train')
    parser.add_argument('--standalone-only', action='store_true', help='Keep full takes with exactly one source gloss')
    parser.add_argument('--label', action='append', dest='labels', help='Exact label to include; repeat as needed')
    args = parser.parse_args(argv)
    try:
        manifest = build_word_library(args.lmdb, args.annotations, args.base_archive, args.output,
                                      expected_lmdb_sha256=args.expected_lmdb_sha256,
                                      expected_annotations_sha256=args.expected_annotations_sha256,
                                      max_candidates=args.max_candidates, split=args.split,
                                      standalone_only=args.standalone_only, labels=args.labels)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({'bundle_id': manifest['bundle_id'], 'statistics': manifest['statistics']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
