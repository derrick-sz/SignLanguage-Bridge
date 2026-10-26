"""Calibrate an explicit, scoped CE-CSL direction map using dev samples only."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_unisign_cecsl import (
    build_pose_protocol, compute_metrics, edit_similarity, file_sha256, load_pose,
    normalize_text, protocol_digest, save_pose, write_json,
)
from slbridge.runtime_options import RuntimeOptions
from slbridge.uni_sign import UniSignBackend, UniSignConfig
from slbridge.uni_sign_preprocess import orient_pose_window


def choose_directions(records, signers, margin=.02):
    """Choose once from calibration rows; validation never changes a choice."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Calibration margin must be finite and nonnegative")
    if any(not r['number'].startswith('dev-') for r in records):
        raise ValueError("Direction calibration accepts dev records only")
    directions, details = {}, {}
    for signer in signers:
        rows = [r for r in records if r['translator'] == signer and r['phase'] == 'calibration']
        if len(rows) < 3:
            raise ValueError(f"At least three dev calibration samples required for {signer}")
        original = statistics.mean(r['scores']['preserve'] for r in rows)
        mirrored = statistics.mean(r['scores']['mirror'] for r in rows)
        if not all(math.isfinite(v) for v in [original, mirrored]):
            raise ValueError("Calibration scores must be finite")
        directions[signer] = 'mirror' if mirrored - original > margin else 'preserve'
        details[signer] = {'n': len(rows), 'preserve_mean_edit_similarity': original,
                           'mirror_mean_edit_similarity': mirrored,
                           'direction': directions[signer], 'margin': margin}
    return directions, details


def prepare_calibration_run(output_dir, identity):
    """Admit only rows bound to the same manifest and selected dev samples."""
    manifest = output_dir / 'calibration-manifest.json'
    predictions = output_dir / 'predictions.jsonl'
    if predictions.exists() and not manifest.exists():
        raise ValueError('Existing calibration predictions lack a manifest; use a new output directory')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != identity:
        raise ValueError('Existing calibration protocol differs; use a new output directory')
    expected = {r['number']: r for r in identity['selected']}
    existing = {}
    protocol_id = protocol_digest(identity)
    if predictions.exists():
        for line in predictions.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            number = row.get('number')
            if number in existing or number not in expected or row.get('calibration_protocol_id') != protocol_id:
                raise ValueError('Calibration record protocol or sample identity differs')
            if any(row.get(key) != value for key, value in expected[number].items()):
                raise ValueError('Calibration record reference/phase/video differs from manifest')
            for direction in ['preserve', 'mirror']:
                prediction = row.get('predictions', {}).get(direction)
                score = row.get('scores', {}).get(direction)
                if not isinstance(prediction, str) or not isinstance(score, (float, int)) or not math.isfinite(score):
                    raise ValueError('Incomplete calibration prediction/score')
                actual = edit_similarity(normalize_text(row['reference']), normalize_text(prediction))
                if not math.isclose(actual, score, abs_tol=1e-12):
                    raise ValueError('Calibration score does not match reference/prediction')
            existing[number] = row
    write_json(manifest, identity)
    return existing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-labels', type=Path, default=Path(r'D:\Zenodo\CE-CSL\CE-CSL\label\dev.csv'))
    parser.add_argument('--video-root', type=Path, default=Path(r'D:\Zenodo\CE-CSL\CE-CSL\video\dev'))
    parser.add_argument('--bundle', type=Path, default=ROOT / 'models/external/uni_sign/model_bundle')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/unisign-cecsl-eval/repair-20261024/dev-calibration')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--pose-cache-dir', type=Path)
    parser.add_argument('--pose-detect-stride', type=int,
                        help='Reuse the detected person box for N frames (1 = detect every frame)')
    parser.add_argument('--pose-sample-only', action='store_true',
                        help='Extract poses only on the frames the model window keeps')
    args = parser.parse_args(argv)
    labels = list(csv.DictReader(args.dev_labels.open(encoding='utf-8-sig', newline='')))
    if (len(labels) != 515 or len({r['Number'] for r in labels}) != 515
            or any(not re.fullmatch(r'dev-\d{5}', r['Number']) or r['Translator'] not in list('ABCDEFGHIJKL') for r in labels)):
        parser.error('Expected the official 515 CE-CSL dev records; test inputs are rejected')
    groups = collections.defaultdict(list)
    for r in labels:
        groups[r['Translator']].append(r)
    if set(groups) != set('ABCDEFGHIJKL'):
        parser.error('Expected all twelve CE-CSL dev signers A-L')
    selected = []
    for signer, group in sorted(groups.items()):
        group.sort(key=lambda r: r['Number'])
        indices = np.rint(np.linspace(0, len(group) - 1, 5)).astype(int)
        if len(set(indices)) != 5:
            parser.error(f'Insufficient distinct samples for {signer}')
        for index, offset in enumerate(indices):
            r = group[int(offset)]
            path = (args.video_root / signer / (r['Number'] + '.mp4')).resolve()
            selected.append({'number': r['Number'], 'translator': signer,
                             'reference': r['Chinese Sentences'], 'video': str(path),
                             'video_size': path.stat().st_size, 'video_mtime_ns': path.stat().st_mtime_ns,
                             'video_sha256': file_sha256(path),
                             'phase': 'calibration' if index % 2 == 0 else 'validation'})
    selected.sort(key=lambda r: (r['phase'] != 'calibration', r['number']))
    config = UniSignConfig.from_bundle(args.bundle)
    config.runtime['pose_orientation'] = 'preserve'
    if args.pose_detect_stride is not None:
        config.runtime['pose_detect_stride'] = args.pose_detect_stride
    if args.pose_sample_only:
        config.runtime['pose_sample_only'] = True
    backend = UniSignBackend(config, RuntimeOptions(device=args.device))
    protocol = build_pose_protocol(config, pose_device=backend.pose_device)
    # The calibration's pose protocol must match the evaluation protocol exactly,
    # including the pose-affecting knobs (only non-default values are recorded).
    if config.runtime.get('pose_sample_only'):
        protocol['pose_sample_only'] = True
    if int(config.runtime.get('pose_detect_stride', 1)) > 1:
        protocol['pose_detect_stride'] = int(config.runtime['pose_detect_stride'])
    identity = {'schema_version': 2, 'pose_protocol': protocol,
                'checkpoint_sha256': config.checkpoint_sha256, 'mt5_revision': config.mt5_revision,
                'num_beams': 4, 'max_new_tokens': 100,
                'dtype': backend.dtype, 'device': backend.device,
                'selection_method': 'dev anchors, mean edit similarity, margin 0.02',
                'labels_sha256': hashlib.sha256(args.dev_labels.read_bytes()).hexdigest(), 'selected': selected}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    existing = prepare_calibration_run(out, identity)
    records_path = out / 'predictions.jsonl'
    cache = (args.pose_cache_dir or out / 'pose-cache') / protocol_digest(protocol)[:16]
    direction_path = out / 'direction-map.json'
    locked_digest = None
    try:
        if any(r['number'] not in existing for r in selected):
            backend.load_model()
        for i, r in enumerate(selected):
            if r['phase'] == 'validation' and locked_digest is None:
                directions, details = choose_directions(list(existing.values()), sorted(groups))
                direction_map = {'schema_version': 1, 'dataset': 'CE-CSL', 'calibration_split': 'dev',
                                 'scope': 'Explicit CE-CSL recording/signers map; not an automatic camera mirror detector',
                                 'checkpoint_sha256': config.checkpoint_sha256, 'pose_protocol': protocol,
                                 'calibration_manifest_sha256': protocol_digest(identity),
                                 'selection_method': 'Three dev anchors per signer; mean edit similarity margin 0.02; ties preserve',
                                 'directions': directions, 'calibration': details,
                                 'calibration_ids': [x['number'] for x in selected if x['phase'] == 'calibration']}
                if direction_path.exists() and json.loads(direction_path.read_text(encoding='utf-8')) != direction_map:
                    raise ValueError('A locked direction map differs; use a new output directory')
                write_json(direction_path, direction_map)
                locked_digest = hashlib.sha256(direction_path.read_bytes()).hexdigest()
                print(json.dumps({'event': 'directions_locked_before_validation', 'directions': directions,
                                  'sha256': locked_digest}), flush=True)
            if r['number'] in existing:
                continue
            video = Path(r['video'])
            pose_path = cache / (r['number'] + '.npz')
            if pose_path.exists():
                pose = load_pose(pose_path, video, protocol['pose_normalization'], pose_protocol=protocol)
                pose_source = 'disk_cache'
            else:
                pose, _, _ = backend._video_pose_window(video)
                save_pose(pose_path, pose, video, protocol['pose_normalization'], pose_protocol=protocol)
                pose_source = 'video_extraction'
            row = {**r, 'predictions': {}, 'scores': {}, 'pose_source': pose_source,
                   'calibration_protocol_id': protocol_digest(identity)}
            for direction in ['preserve', 'mirror']:
                prediction, _ = backend._runtime.infer(orient_pose_window(pose, direction), num_beams=4, max_new_tokens=100)
                row['predictions'][direction] = prediction
                row['scores'][direction] = edit_similarity(normalize_text(r['reference']), normalize_text(prediction))
            with records_path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            existing[r['number']] = row
            print(json.dumps({'event': 'sample', 'done': i+1, 'total': len(selected), **row}, ensure_ascii=False), flush=True)
        assert locked_digest == hashlib.sha256(direction_path.read_bytes()).hexdigest()
        direction_map = json.loads(direction_path.read_text(encoding='utf-8'))
        validation = [existing[r['number']] for r in selected if r['phase'] == 'validation']
        baseline = compute_metrics([{**r, 'prediction': r['predictions']['preserve']} for r in validation], split='dev')
        corrected = compute_metrics([{**r, 'prediction': r['predictions'][direction_map['directions'][r['translator']]]} for r in validation], split='dev')
        summary = {'schema_version': 1, 'calibration_n': sum(r['phase'] == 'calibration' for r in selected), 'validation_n': len(validation),
                   'direction_map_sha256': locked_digest, 'directions': direction_map['directions'],
                   'validation_original': baseline, 'validation_locked_directions': corrected,
                   'boundary': 'Dev calibration plus disjoint dev validation; no test scores used for map selection. Not human semantic acceptance.'}
        write_json(out / 'summary.json', summary)
        print(json.dumps({'event': 'done', 'summary': summary}, ensure_ascii=False), flush=True)
        return 0
    finally:
        backend.close()


if __name__ == '__main__':
    raise SystemExit(main())

