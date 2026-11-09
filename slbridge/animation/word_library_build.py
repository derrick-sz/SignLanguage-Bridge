"""Offline native SignSparK word spans; no Torch, model or SOKE time mapping."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import statistics
import zipfile

import numpy as np

from .registry import MotionRegistry
from .signspark_codec import CODEC_VERSION, decode_signspark_streams


SCHEMA = 'signbridge-word-library-v1'
LMDB_SHA256 = '7dbc5dc932c6e34ea1d7d47976e8db8fb90ba4c2aff3eb05a6991c7008366222'
ANNOTATIONS_SHA256 = '92cabbe0fcc97770abd491802be00cb35883a2237cabb7f0c5de847845355f2c'
DATASET_REVISION = '9fe7e04ef59204c787cd6094c747f889efff6ba0'
ANNOTATIONS_REVISION = 'a883bc2d13a5d7f85e64591b32bae4cb48950b97'
SOURCE_IDENTITIES = {
    'train': (LMDB_SHA256, ANNOTATIONS_SHA256),
    'dev': ('c01bded1a737ba0405d7deedd13de3c6d8173584daadc47682b83449b93bd6d5',
            '93ba7e1757ef83a65878c3371ec340099854bd87ea2c4046e327538db1378ec5'),
    'test': ('c9a851b879925880eb6f87cbbb6495361c86d158dbede2ccf0f31a2eceb4319f',
             '901ac0ac89755c0500f029e0235dccc2149ebfaae2ce068ffd52838abc9975a0'),
}
_MAX_BLOB = 64 * 1024 * 1024
_MAX_EXPANDED = 128 * 1024 * 1024


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _verified(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('Expected an explicit SHA256 identity')
    if _digest(path) != expected:
        raise ValueError('Source SHA256 checksum mismatch')


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('annotation class resolution forbidden')

    def persistent_load(self, identity):
        raise pickle.UnpicklingError('annotation persistent ID forbidden')


def _annotations(path):
    if Path(path).stat().st_size > 32 * 1024 * 1024:
        raise ValueError('Annotation size exceeds limit')
    try:
        with Path(path).open('rb') as stream:
            entries = _PrimitiveUnpickler(stream).load()
            if stream.read(1):
                raise ValueError('Trailing annotation pickle data')
    except (pickle.UnpicklingError, EOFError, TypeError, AttributeError) as error:
        raise ValueError('Invalid or unsafe annotation pickle') from error
    if not isinstance(entries, list) or not 0 < len(entries) <= 500_000:
        raise ValueError('Expected a bounded annotation list')
    return entries


def _entry(value):
    if not isinstance(value, dict):
        return None
    label, video = value.get('label'), value.get('video_file')
    start, end, length = (value.get(key) for key in ('start', 'end', 'seq_len'))
    if (not isinstance(label, str) or not 0 < len(label) <= 128 or label != label.strip()
            or any(ord(c) < 32 for c in label) or not isinstance(video, str)
            or not 0 < len(video) <= 256 or any(ord(c) < 32 for c in video)
            or any(type(v) is not int for v in (start, end, length))
            or start < 0 or end <= start or end - start != length or not 3 <= length <= 180):
        return None
    return {'label': label, 'video_file': video, 'start': start, 'end': end, 'seq_len': length}


def _native_streams(blob):
    """Read only numeric fields; never access object translations or metadata."""
    if len(blob) > _MAX_BLOB:
        raise ValueError('LMDB clip blob exceeds limit')
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
            members = zipped.infolist()
            if (len(members) > 32 or len({m.filename for m in members}) != len(members)
                    or sum(m.file_size for m in members) > _MAX_EXPANDED
                    or any(m.file_size > _MAX_BLOB for m in members)):
                raise ValueError('Expanded NPZ size or member count exceeds limit')
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            names = ('body_features', 'left_features', 'right_features', 'face_features')
            streams = [archive[name] for name in names]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError('Invalid numeric LMDB NPZ') from error
    for value, width in zip(streams, (126, 90, 90, 56)):
        if (value.ndim != 2 or value.shape[1] < width or value.shape[1] > width + 60
                or not 1 <= len(value) <= 20_000 or value.dtype.kind not in 'fiu'
                or value.dtype.itemsize > 8 or not np.isfinite(value).all()):
            raise ValueError('Invalid finite native stream shape or dtype')
    body, left, right, face = streams
    if any(len(value) != len(body) for value in streams):
        raise ValueError('Native streams disagree on frame count')
    if face.shape[1] != 56:
        raise ValueError('Face stream must retain exactly 56 channels')
    if body.shape[1] != 126:
        raise ValueError('Body stream must match fixed 126-channel protocol')
    return body[:, 66:], left[:, :90], right[:, :90], face


def build_word_library(lmdb_path, annotations, base_archive, output, *,
                       expected_lmdb_sha256, expected_annotations_sha256, max_candidates=3, split='train'):
    """Authenticate paired sources then export at most eight native spans per label.

    ISO frame ranges are used only against their paired LMDB, never against
    SOKE fittings. Invalid annotations or individual clips are counted and
    isolated; a source-identity mismatch fails before creating the output.
    """
    output, database, annotations = Path(output), Path(lmdb_path), Path(annotations)
    if output.exists():
        raise ValueError('Use a new output directory; existing output is preserved')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 8:
        raise ValueError('Use one to eight candidates per label')
    if split not in SOURCE_IDENTITIES:
        raise ValueError('Use an explicit train, dev or test source split')
    pair = (expected_lmdb_sha256, expected_annotations_sha256)
    known_splits = [name for name, hashes in SOURCE_IDENTITIES.items() if pair == hashes]
    if known_splits and known_splits != [split]:
        raise ValueError('Declared split differs from authenticated source split')
    for name, hashes in SOURCE_IDENTITIES.items():
        if (expected_lmdb_sha256 == hashes[0] or expected_annotations_sha256 == hashes[1]) and pair != hashes:
            raise ValueError('Official LMDB and annotation identity pair differs')
    data_file = database / 'data.mdb' if database.is_dir() else database
    _verified(data_file, expected_lmdb_sha256)
    _verified(annotations, expected_annotations_sha256)
    entries = _annotations(annotations)
    registry = MotionRegistry(base_archive)
    first = registry.catalog()['assets'][0]['id']
    skeleton_motion = registry.motion(first)
    skeleton = {key: skeleton_motion[key] for key in ('parents', 'restJoints')}
    official = pair == SOURCE_IDENTITIES[split]
    source = {'lmdb_sha256': expected_lmdb_sha256, 'annotations_sha256': expected_annotations_sha256,
              'split': split, 'codec': CODEC_VERSION, 'left_hand_convention': 'smplx-left',
              'dataset_repo': 'LionelLow/SignSparK_data' if official else None,
              'dataset_revision': DATASET_REVISION if official else None,
              'annotations_repo': 'LionelLow/SignSparK_Retrieval' if official else None,
              'annotations_revision': ANNOTATIONS_REVISION if official else None}
    groups, seen = defaultdict(list), set()
    excluded = 0
    for raw in entries:
        value = _entry(raw)
        if value is None:
            excluded += 1
            continue
        identity = (value['label'], value['video_file'], value['start'], value['end'])
        if identity in seen:
            excluded += 1
            continue
        seen.add(identity)
        groups[value['label']].append(value)
    import lmdb
    env = lmdb.open(str(database), subdir=database.is_dir(), readonly=True, lock=False,
                    readahead=False, meminit=False, max_readers=1)
    output.mkdir(parents=True)
    (output/'units').mkdir()
    units, failed, reasons, failed_labels = [], 0, Counter(), []
    try:
        with env.begin() as transaction:
            for label, candidates in sorted(groups.items()):
                median = statistics.median(value['seq_len'] for value in candidates)
                candidates.sort(key=lambda value: (abs(value['seq_len']-median), value['seq_len'],
                                                    value['video_file'], value['start'], value['end']))
                count = 0
                for value in candidates:
                    try:
                        blob = transaction.get(value['video_file'].encode('utf-8'))
                        if blob is None:
                            raise ValueError('Missing native LMDB clip')
                        streams = _native_streams(blob)
                        start, end = value['start'], value['end']
                        if end > len(streams[0]):
                            raise ValueError('Annotation range exceeds native LMDB frames')
                        motion = decode_signspark_streams(*(v[start:end] for v in streams),
                                                          left_hand_convention='smplx-left')
                        with np.errstate(over='ignore', invalid='ignore'):
                            poses = motion.poses.astype(np.float32, copy=False)
                            expression = motion.expression.astype(np.float32, copy=False)
                        if not np.isfinite(poses).all() or not np.isfinite(expression).all():
                            raise ValueError('Decoded word cannot be represented as finite float32')
                    except (ValueError, TypeError, UnicodeError) as error:
                        failed += 1
                        reasons[str(error)] += 1
                        continue
                    fingerprint = json.dumps([source, label, value['video_file'], start, end],
                                             sort_keys=True, ensure_ascii=False).encode('utf-8')
                    identity = 'word-' + hashlib.sha256(fingerprint).hexdigest()[:24]
                    relative = f'units/{identity}.npz'
                    np.savez_compressed(output/relative, poses=poses, expression=expression)
                    units.append({'id': identity, 'label': label, 'path': relative,
                                  'sha256': _digest(output/relative), 'frames': end-start, 'fps': 30,
                                  'source': {key: value[key] for key in ('video_file', 'start', 'end')}
                                            | {'split': split}})
                    count += 1
                    if count == max_candidates:
                        break
                if count == 0:
                    failed_labels.append(label)
    finally:
        env.close()
    if not units:
        raise ValueError('No valid native word units; partial output preserved for diagnosis')
    identity_input = json.dumps({'source': source, 'base_archive_sha256': registry.sha256,
                                'units': units}, sort_keys=True, ensure_ascii=False).encode('utf-8')
    manifest = {'schema': SCHEMA, 'bundle_id': 'csl-words-' + hashlib.sha256(identity_input).hexdigest()[:24],
                'source': source, 'base_archive_sha256': registry.sha256, 'skeleton': skeleton,
                'preview_fps': 30,
                'usage_role': 'product-demo', 'evaluation_eligible': False,
                'fps_basis': 'explicit product preview clock; native LMDB timestamps not independently verified',
                'units': units, 'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN',
                'statistics': {'input_annotations': len(entries), 'eligible_annotations': len(seen),
                               'excluded_annotations': excluded, 'candidate_labels': len(groups),
                               'labels': len({u['label'] for u in units}), 'units': len(units),
                               'failed_candidates': failed, 'failure_reasons': dict(reasons),
                               'labels_without_valid_units': failed_labels}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
