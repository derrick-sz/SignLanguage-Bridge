"""Ingest the full SOKE ZIP into a resumable SQLite/NPZ sentence library."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slbridge.animation.full_corpus import build_library
from slbridge.animation.registry import MotionRegistry
from build_soke_corpus import annotations, BASE_SHA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive', 'indices', 'base-archive', 'smplx-model', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Optional bounded smoke run; omitted means all complete sequences')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    registry = MotionRegistry(args.base_archive, BASE_SHA)
    source = registry.motion(registry.catalog()['assets'][0]['id'])
    skeleton = {k: source[k] for k in ('parents', 'restJoints', 'representation', 'smplx_rest_model_sha256') if k in source}
    with np.load(args.smplx_model, allow_pickle=False) as model:
        mean = np.stack([model['hands_meanl'].reshape(45), model['hands_meanr'].reshape(45)]).astype(np.float32)
        rest = np.asarray(model['J_regressor'] @ model['v_template'])
    if not np.allclose(rest, skeleton['restJoints'], atol=2e-6, rtol=0):
        parser.error('Verified template differs from playback skeleton')
    rows, hashes = annotations(args.indices)
    from slbridge.animation.full_corpus import sha256_file

    def progress(done, total):
        if total is None:
            print(json.dumps(done, ensure_ascii=False), flush=True)
        else:
            print(f'Indexing ZIP directory {done}/{total}', flush=True)

    status = build_library(args.source_archive, rows, mean, skeleton, args.output,
                           template_sha256=sha256_file(args.smplx_model), index_sha256=hashes,
                           limit=args.limit, retry_failed=args.retry_failed, progress=progress)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 1 if status['states'].get('failed', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
