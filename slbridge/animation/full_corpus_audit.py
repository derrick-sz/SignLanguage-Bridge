"""Read-only, bounded artifact audit; partial conversion never becomes a full pass."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import zipfile

import numpy as np

from .full_corpus import sha256_file
from .soke import FIELDS


def _verify_arrays(path, count, mean):
    if path.stat().st_size > 32*1024*1024:
        raise ValueError('Parameter file exceeds budget')
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        if len(entries) > 32 or len({i.filename for i in entries}) != len(entries) or sum(info.file_size for info in entries) > 32*1024*1024:
            raise ValueError('Expanded parameter file exceeds budget')
        for entry in entries:
            with z.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported numeric array version')
                size = math.prod(shape)*dtype.itemsize
                if dtype.hasobject or dtype.kind not in 'fibu' or size > 32*1024*1024 or stream.tell()+size != entry.file_size:
                    raise ValueError('Numeric array payload exceeds budget or differs from header')
    with np.load(path, allow_pickle=False) as data:
        for key, size in FIELDS.items():
            array = data[key]
            if array.shape != (count, size) or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError('Invalid raw parameter '+key)
        poses = data['effective_poses']
        if poses.shape != (count, 55, 3) or not np.isfinite(poses).all():
            raise ValueError('Invalid effective pose dimensions')
        if not np.array_equal(data['frame_numbers'], np.arange(count)):
            raise ValueError('Invalid frame identity')
        if np.any(poses[:, :12]) or np.any(poses[:, 23:25]):
            raise ValueError('Canonical root/lower-body/eyes changed')
        expected = [
            (poses[:, 12:22].reshape(count, 30), data['smplx_body_pose'][:, 33:]),
            (poses[:, 25:40].reshape(count, 45), data['smplx_lhand_pose']+mean[0]),
            (poses[:, 40:55].reshape(count, 45), data['smplx_rhand_pose']+mean[1]),
            (poses[:, 22], data['smplx_jaw_pose']),
        ]
        for actual, wanted in expected:
            if not np.array_equal(actual, wanted):
                raise ValueError('Effective pose differs from retained raw fields/hand mean')
        features = np.concatenate([data['smplx_body_pose'][:, 33:], data['smplx_lhand_pose'],
                                   data['smplx_rhand_pose'], data['smplx_jaw_pose'], data['smplx_expr']], axis=-1)
        if not np.array_equal(data['features133_raw'], features):
            raise ValueError('Raw 133D feature assembly changed')


def audit_library(root, progress=None):
    root = Path(root).resolve()
    db = sqlite3.connect((root/'corpus.sqlite3').as_uri()+'?mode=ro', uri=True)
    db.execute('BEGIN')  # consistent snapshot even if the producer is still running
    try:
        meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        mean = np.asarray(meta['hand_mean'], dtype=np.float32)
        states = dict(db.execute('SELECT state,count(*) FROM samples GROUP BY state'))
        checked = frames = invalid = 0
        errors = []
        for identity, count, relative, expected in db.execute('SELECT id,expected_frames,parameter_file,parameter_sha256 FROM samples WHERE state="converted_candidate" ORDER BY id'):
            try:
                path = (root/relative).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError('Missing/escaping parameter file')
                if sha256_file(path) != expected:
                    raise ValueError('Parameter file SHA differs')
                _verify_arrays(path, count, mean)
                checked += 1
                frames += count
            except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
                invalid += 1
                if len(errors) < 100:
                    errors.append({'id': identity, 'error': str(exc)})
            if progress and (checked+invalid) % 1000 == 0:
                progress(checked, invalid)
        metrics = [json.loads(row[0]) for row in db.execute('SELECT metrics_json FROM samples WHERE state="converted_candidate"')]
        return {
            'schema': 'soke-full-library-artifact-audit-v1',
            'snapshot_time': datetime.now(timezone.utc).isoformat(),
            'passed': invalid == 0, 'errors': errors, 'invalid_sequences': invalid,
            'states': states, 'verified_sequences': checked, 'verified_frames': frames,
            'full_numeric_conversion_complete': invalid == 0 and checked > 0
                and states.get('pending', 0) == 0 and states.get('failed', 0) == 0,
            'source_archive_sha256': meta['build']['source_archive_sha256'],
            'unchecked_archive_frame_files': meta['inventory']['actual_frame_files']-frames,
            'unique_texts_converted': db.execute('SELECT count(DISTINCT text) FROM samples WHERE state="converted_candidate"').fetchone()[0],
            'signers_converted': db.execute('SELECT count(DISTINCT signer) FROM samples WHERE state="converted_candidate"').fetchone()[0],
            'quality_diagnostics': {
                'samples_with_mapped_steps_over30': sum(m['mapped_max_step_degrees'] > 30 for m in metrics),
                'mapped_step_events_over60': sum(m['mapped_step_events_over60'] for m in metrics),
                'maximum_mapped_step_degrees': max((m['mapped_max_step_degrees'] for m in metrics), default=0),
            },
            'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN',
            'original_video_temporal_alignment': 'unverified',
        }
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_library(args.library, progress=lambda n, bad: print(f'Verified {n}, invalid {bad}', flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
