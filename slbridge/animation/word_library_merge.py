"""Merge authenticated word assets into a provenance-preserving product pool.

No motion is synthesized or retimed here. A mixed product pool is explicitly
ineligible as an independent model evaluation corpus.
"""
from collections import Counter, OrderedDict, defaultdict, deque
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

from .registry import _skeleton
from .signspark_codec import CODEC_VERSION
from .word_library_build import SOURCE_IDENTITIES

_SHA = re.compile(r'[0-9a-f]{64}\Z')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
SCHEMA = 'signbridge-word-library-v1'
_MAX_INDEX = 16 * 1024 * 1024
_MAX_ASSET = 8 * 1024 * 1024
_SOURCE_FIELDS = ('library_bundle_id', 'manifest_sha256', 'split', 'lmdb_sha256',
                  'annotations_sha256', 'dataset_repo', 'dataset_revision',
                  'annotations_repo', 'annotations_revision')


def _bytes_digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _safe_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe word asset path')
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or value.suffix != '.npz':
        raise ValueError('Unsafe word asset path')
    path = (root/value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Word asset path escaped source')
    return path


def _source(row):
    if (not isinstance(row, dict) or not isinstance(row.get('library_bundle_id'), str)
            or not _ID.fullmatch(row['library_bundle_id'])
            or row.get('split') not in ('train', 'dev', 'test')
            or any(not isinstance(row.get(key), str) or not _SHA.fullmatch(row[key])
                   for key in ('manifest_sha256', 'lmdb_sha256', 'annotations_sha256'))):
        raise ValueError('Invalid source library identity')
    pair = (row['lmdb_sha256'], row['annotations_sha256'])
    for split, hashes in SOURCE_IDENTITIES.items():
        if ((pair[0] == hashes[0] or pair[1] == hashes[1])
                and (pair != hashes or row['split'] != split)):
            raise ValueError('Source split or paired official identities differ')
    return {key: row.get(key) for key in _SOURCE_FIELDS}


def _motion_bytes(path, unit):
    if path.stat().st_size > _MAX_ASSET:
        raise ValueError('Word asset size exceeds bound')
    raw = path.read_bytes()
    if _bytes_digest(raw) != unit['sha256']:
        raise ValueError('Word asset SHA256 differs')
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            members = archive.infolist()
            if (len(members) != 2 or len({i.filename for i in members}) != 2
                    or sum(i.file_size for i in members) > _MAX_ASSET):
                raise ValueError('Expanded word asset exceeds bound')
        with np.load(BytesIO(raw), allow_pickle=False) as stored:
            if set(stored.files) != {'poses', 'expression'}:
                raise ValueError('Unexpected word parameter fields')
            poses, expression = stored['poses'], stored['expression']
        if (poses.shape != (unit['frames'], 55, 3) or poses.dtype != np.float32
                or expression.shape != (unit['frames'], 50) or expression.dtype != np.float32
                or not np.isfinite(poses).all() or not np.isfinite(expression).all()):
            raise ValueError('Invalid word parameter shape or finite dtype')
    except (OSError, zipfile.BadZipFile, KeyError, TypeError) as error:
        raise ValueError('Invalid word parameter archive') from error
    return raw


def _library(path):
    root = Path(path).resolve()
    manifest_path = root/'manifest.json'
    if manifest_path.stat().st_size > _MAX_INDEX:
        raise ValueError('Word index exceeds bound')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get('schema') != SCHEMA or not _ID.fullmatch(manifest.get('bundle_id', ''))
            or not isinstance(manifest.get('source'), dict)
            or not isinstance(manifest.get('units'), list) or not 1 <= len(manifest['units']) <= 20000
            or not isinstance(manifest.get('base_archive_sha256'), str)
            or not _SHA.fullmatch(manifest['base_archive_sha256']) or manifest.get('preview_fps') != 30):
        raise ValueError('Invalid source word library')
    skeleton = manifest.get('skeleton')
    _skeleton(skeleton)
    source = manifest['source']
    if source.get('codec') != CODEC_VERSION or source.get('left_hand_convention') != 'smplx-left':
        raise ValueError('Native word codec or hand convention differs')
    digest = _bytes_digest(raw)
    if source.get('split') == 'mixed':
        if (manifest.get('usage_role') != 'product-demo' or manifest.get('evaluation_eligible') is not False
                or source.get('usage_role') != 'product-demo' or source.get('evaluation_eligible') is not False
                or not isinstance(source.get('sources'), list) or not source['sources']):
            raise ValueError('Mixed libraries must remain product-only with source identities')
        rows = [_source(row) for row in source['sources']]
    else:
        if source.get('split') not in ('train', 'dev', 'test'):
            raise ValueError('Invalid source split')
        if source['split'] != 'train' and (manifest.get('usage_role') != 'product-demo'
                                              or manifest.get('evaluation_eligible') is not False):
            raise ValueError('Nontrain library cannot claim independent evaluation')
        rows = [_source(dict(source, library_bundle_id=manifest['bundle_id'], manifest_sha256=digest))]
    origins = {}
    for row in rows:
        previous = origins.setdefault(row['library_bundle_id'], row)
        if previous != row:
            raise ValueError('Conflicting source library identity')
    units, ids = [], set()
    for raw_unit in manifest['units']:
        unit = deepcopy(raw_unit)
        label, identity = unit.get('label'), unit.get('id')
        record = unit.get('source')
        if (not isinstance(identity, str) or not _ID.fullmatch(identity) or identity in ids
                or not isinstance(label, str) or not label.strip() or len(label) > 100
                or type(unit.get('frames')) is not int or not 1 <= unit['frames'] <= 3000
                or unit.get('fps') != 30 or not _SHA.fullmatch(unit.get('sha256', ''))
                or not isinstance(record, dict) or record.get('split') not in ('train', 'dev', 'test')
                or not isinstance(record.get('video_file'), str) or not record['video_file']
                or type(record.get('start')) is not int or type(record.get('end')) is not int
                or record['start'] < 0 or record['end'] - record['start'] != unit['frames']):
            raise ValueError('Invalid word identity or source range')
        ids.add(identity)
        if source['split'] == 'mixed':
            origin = origins.get(record.get('library_bundle_id'))
            if (origin is None or record.get('library_manifest_sha256') != origin['manifest_sha256']
                    or record['split'] != origin['split']
                    or record.get('lmdb_sha256') != origin['lmdb_sha256']
                    or record.get('annotations_sha256') != origin['annotations_sha256']):
                raise ValueError('Word source binding differs from declared origin')
        else:
            origin = rows[0]
            if record['split'] != origin['split']:
                raise ValueError('Unit split differs from authenticated library split')
            record.update(library_bundle_id=origin['library_bundle_id'],
                          library_manifest_sha256=origin['manifest_sha256'],
                          lmdb_sha256=origin['lmdb_sha256'], annotations_sha256=origin['annotations_sha256'])
        unit['path'] = f'units/{identity}.npz'
        actual_path = _safe_path(root, raw_unit['path'])
        _motion_bytes(actual_path, unit)
        units.append((unit, actual_path))
    return {'root': root, 'manifest_sha256': digest, 'bundle_id': manifest['bundle_id'],
            'source_rows': rows, 'skeleton': deepcopy(skeleton), 'base_archive_sha256': manifest['base_archive_sha256'],
            'units': units}


def _unit_identity(unit):
    # The same native unit may be present in train3 and train8 manifests.
    # Its origin-library envelope is different, while its actual bytes and
    # original data/interval/semantic label must remain identical.
    source = {k: v for k, v in unit['source'].items()
              if k not in ('library_bundle_id', 'library_manifest_sha256')}
    return _json({key: unit[key] for key in ('id', 'label', 'sha256', 'frames', 'fps')} | {'source': source})


def _span_identity(unit):
    source = unit['source']
    return (unit['label'], source['lmdb_sha256'], source['video_file'], source['start'], source['end'])


def merge_word_libraries(base, additions, output, max_candidates=8):
    """Keep every base unit, then round-robin unique native sources per word.

    Input files are never written. Validation completes before a new output is
    created; interrupted copies retain an unpublished directory for diagnosis.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing output is preserved')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 8:
        raise ValueError('Use one to eight candidates per label')
    if not isinstance(additions, (list, tuple)):
        raise ValueError('Expected an explicit list of additional libraries')
    libraries = [_library(path) for path in [base, *additions]]
    baseline = libraries[0]
    if any(lib['skeleton'] != baseline['skeleton'] or lib['base_archive_sha256'] != baseline['base_archive_sha256']
           for lib in libraries[1:]):
        raise ValueError('Source skeleton or base archive identity differs')
    sources = OrderedDict()
    identities, spans = {}, {}
    duplicate_units = 0
    pools = []
    for lib in libraries:
        for row in lib['source_rows']:
            old = sources.setdefault(row['library_bundle_id'], row)
            if old != row:
                raise ValueError('Conflicting source library manifests')
        groups = defaultdict(list)
        for unit, path in lib['units']:
            identity = _unit_identity(unit)
            old = identities.setdefault(unit['id'], identity)
            if old != identity:
                raise ValueError('Duplicate unit ID has conflicting identity')
            span = _span_identity(unit)
            if span in spans:
                if spans[span]['sha256'] != unit['sha256'] or spans[span]['frames'] != unit['frames']:
                    raise ValueError('Duplicate native span has conflicting bytes')
                duplicate_units += 1
                continue
            spans[span] = unit
            groups[unit['label']].append((unit, path))
        pools.append(groups)
    if len(sources) > 16:
        raise ValueError('Merged source origins exceed the runtime limit of 16')
    chosen = list(baseline['units'])
    counts = Counter(unit['label'] for unit, _ in chosen)
    if max(counts.values()) > max_candidates:
        raise ValueError('Candidate cap cannot discard an existing base unit')
    old_labels = set(counts)
    labels = sorted(set().union(*(set(pool) for pool in pools)))
    for label in labels:
        queues = [deque(pool.get(label, [])) for pool in pools[1:]]
        while counts[label] < max_candidates and any(queues):
            for candidates in queues:
                if candidates and counts[label] < max_candidates:
                    chosen.append(candidates.popleft())
                    counts[label] += 1
    if len(chosen) > 20000:
        raise ValueError('Merged unit count exceeds runtime bound')
    manifest = {'schema': SCHEMA, 'usage_role': 'product-demo', 'evaluation_eligible': False,
                'base_archive_sha256': baseline['base_archive_sha256'], 'skeleton': baseline['skeleton'],
                'preview_fps': 30,
                'fps_basis': 'explicit product preview clock; native LMDB timestamps not independently verified',
                'source': {'split': 'mixed', 'usage_role': 'product-demo', 'evaluation_eligible': False,
                           'codec': CODEC_VERSION, 'left_hand_convention': 'smplx-left',
                           'sources': list(sources.values())},
                'units': [unit for unit, _ in chosen], 'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN',
                'statistics': {'base_labels': len(old_labels), 'base_units': len(baseline['units']),
                               'base_units_preserved': len(baseline['units']), 'labels': len(counts),
                               'units': len(chosen), 'labels_added': len(set(counts)-old_labels),
                               'units_added': len(chosen)-len(baseline['units']), 'duplicate_units': duplicate_units,
                               'max_candidates': max_candidates,
                               'units_by_split': dict(Counter(u['source']['split'] for u, _ in chosen))},
                'input_libraries': [{'library_bundle_id': lib['bundle_id'],
                                    'manifest_sha256': lib['manifest_sha256']} for lib in libraries]}
    manifest['bundle_id'] = 'csl-product-' + _bytes_digest(_json(manifest).encode('utf-8'))[:24]
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    if len(encoded) > _MAX_INDEX:
        raise ValueError('Merged manifest exceeds runtime bound')
    output.mkdir(parents=True)
    (output/'units').mkdir()
    for unit, source_path in chosen:
        raw = _motion_bytes(source_path, unit)
        (output/unit['path']).write_bytes(raw)
    for lib in libraries:
        if _bytes_digest((lib['root']/'manifest.json').read_bytes()) != lib['manifest_sha256']:
            raise ValueError('Source library changed during merge; output remains unpublished')
    (output/'manifest.json').write_bytes(encoded)
    return manifest
