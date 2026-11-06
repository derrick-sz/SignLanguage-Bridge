"""Read-only, bounded audit of a completed two-request SignSparK run.

No model imports, checkpoint hashing, inference, service access or source edits.
The report must be a NEW file outside every audited run directory. Codec replay
checks artifact consistency only, not an independent proof of pose semantics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCE_COMMIT = '916402d0ecd6322fb4a9e5f513229a94a57bf4b6'
MODEL_REVISION = 'b907f25f9bb9f5a0adcaf131a23a155dc471d9a8'
SOURCE_RECEIPTS = {'880b46ee3e46aba550e4975eaa746ad0d18ac82a2a2bd7d05180fa5e6cffbad6',
                   '0883d912770c8ebc3d59cb66c82c6a228b95b31a250752d90205a963f93b252d'}
TEXT_RECEIPTS = {'3f7c359cbf06363e8054bcc1a901d09fbe6b4b2c8bb7e01309e1f8a383267f2b',
                 'e27b5393be1d295c36ba805b118712c7c85b4b9a8d951b933bcba09801e058d7'}
BASE_SHA = '4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5'
SOURCE_PRESENTATION = 'a9fa82f145861c731d6aadc63957df549e7008fb701cb282a5f8dcd7143d80c8'
TEMPLATE_SHA = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
AVATAR_SHA = 'b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c'
CODEC = 'signspark-row6d-flat-hand-v1'
MODEL_FILES = {
    'body': (5580532361, 'bf95bb02802b1c46f40afb4424f8f521be6c8641ad8acfe406a666b463e3a380'),
    'hand': (5581699849, '5bb8e407ba2c106a22c3b0c16fd1b0fc0f412a58d235143325fdb67efe5916c2'),
    'face': (5580376713, '841e6ce5c929b142ad8ea3cc152fa67f57e7a8841954bf068df42a5d986ead53'),
}
WIDTHS = {'body': 60, 'left': 90, 'right': 90, 'face': 56}
JAW = {'profile': 'smplx-jaw-godette-calibrated-v1', 'source_joint': 22,
       'source_template_sha256': TEMPLATE_SHA, 'target_avatar_sha256': AVATAR_SHA,
       'status': 'research_candidate', 'export_supported': False}


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def reject_constant(value):
    raise AuditFailure('Nonfinite JSON constant: ' + value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.receipts = {}

    def path(self, relative):
        require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Expected relative artifact path')
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root) and path != self.root, 'Artifact path leaves run directory')
        return path

    def read(self, relative, limit=8 * 1024 * 1024, expected=None):
        path = self.path(relative)
        require(path.stat().st_size <= limit, 'Artifact byte budget exceeded: ' + relative)
        with path.open('rb') as handle:
            data = handle.read(limit + 1)
        require(len(data) <= limit, 'Artifact grew beyond byte budget: ' + relative)
        actual = digest(data)
        if expected is not None:
            require(actual == expected, 'Artifact SHA256 mismatch: ' + relative)
        self.receipts[relative] = {'bytes': len(data), 'sha256': actual}
        return data

    def json(self, relative, limit=8 * 1024 * 1024, expected=None):
        return json.loads(self.read(relative, limit, expected), parse_constant=reject_constant)

    def npz(self, relative, keys, expected=None):
        import numpy as np
        data = self.read(relative, expected=expected)
        with zipfile.ZipFile(BytesIO(data)) as archive:
            items = archive.infolist()
            require(len(items) == len(keys) and {i.filename for i in items} == {k + '.npy' for k in keys},
                    'Unexpected/duplicate NPZ members: ' + relative)
            require(sum(i.file_size for i in items) <= 16 * 1024 * 1024, 'NPZ expansion budget exceeded')
            for item in items:
                with archive.open(item) as handle:
                    version = np.lib.format.read_magic(handle)
                    require(version in ((1, 0), (2, 0)), 'Unsupported numeric NPY header version')
                    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = header(handle, max_header_size=10000)
                    require(len(shape) <= 4 and all(type(n) is int and 0 <= n <= 100000 for n in shape), 'NPY dimensions exceed budget')
                    require(dtype.kind in 'fbiu' and dtype.itemsize <= 8 and math.prod(shape) * dtype.itemsize <= 16 * 1024 * 1024,
                            'NPY allocation exceeds budget or uses unsafe dtype')
        with np.load(BytesIO(data), allow_pickle=False) as source:
            arrays = {key: source[key] for key in keys}
        for key, value in arrays.items():
            require(value.dtype.kind in 'fbiu' and np.isfinite(value).all(), 'Nonfinite/non-numeric NPZ field: ' + key)
        return arrays


def identity_checks(manifest, complete=True):
    require(manifest.get('schema') == 'custom-csl-signspark-generation-v1', 'Unexpected generation schema')
    fixed = {'source_commit': SOURCE_COMMIT, 'model_revision': MODEL_REVISION,
             'base_archive_sha256': BASE_SHA, 'codec': CODEC, 'target_motion': None,
             'observed_keyframes': 0, 'length_source': 'explicit request', 'text_guidance_scale': -1,
             'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN'}
    for key, expected in fixed.items():
        require(key in manifest and manifest[key] == expected, 'Generation identity/protocol mismatch: ' + key)
    require(manifest.get('source_manifest_sha256') in SOURCE_RECEIPTS, 'Unknown authenticated source receipt')
    require(manifest.get('text_assets_manifest_sha256') in TEXT_RECEIPTS, 'Unknown authenticated text receipt')
    require(type(manifest['observed_keyframes']) is int, 'Invalid keyframe count type')
    requests = manifest.get('requests')
    require(isinstance(requests, list) and len(requests) == 2, 'This audit requires exactly two requests')
    for request in requests:
        require(isinstance(request, dict) and set(request) == {'text', 'frames', 'steps', 'seed'}, 'Unexpected request contract')
        require(isinstance(request['text'], str) and 0 < len(request['text'].strip()) <= 200, 'Invalid request text')
        require(type(request['frames']) is int and 16 <= request['frames'] <= 304, 'Invalid explicit frame count')
        require(type(request['steps']) is int and 1 <= request['steps'] <= 100, 'Invalid sampling steps')
        require(type(request['seed']) is int and 0 <= request['seed'] < 2**32, 'Invalid seed')
    require(requests[0]['text'].strip() != requests[1]['text'].strip(), 'Requests must have different texts')
    require(all(requests[0][k] == requests[1][k] for k in ('frames', 'steps', 'seed')),
            'Text-condition comparison needs equal explicit frames, steps and seed')
    logs = manifest.get('streams')
    expected_streams = {'body', 'hand', 'face'} if complete else {'body', 'hand'}
    require(isinstance(logs, dict) and set(logs) == expected_streams, 'Missing/unexpected completed stream load receipts')
    for name, row in logs.items():
        size, sha = MODEL_FILES[name]
        require(row.get('sha256') == sha and row.get('bytes') == size, 'Wrong checkpoint receipt: ' + name)
        require(row.get('tensors') == 859 and row.get('missing_keys') == [] and row.get('unexpected_keys') == [],
                'Strict 859-tensor load not established: ' + name)
        require(row.get('assign') is True and row.get('mmap') is True, 'Unexpected loader protocol: ' + name)
    return requests


def raw_arrays(reader, manifest, requests, roles):
    import numpy as np
    rows = manifest.get('raw_streams')
    expected = {(i, role) for i in range(2) for role in roles}
    require(isinstance(rows, list) and len(rows) == len(expected), 'Wrong raw stream count')
    result = {}; seen = set()
    for row in rows:
        i, role = row.get('request_index'), row.get('role')
        require(type(i) is int and (i, role) in expected and (i, role) not in seen, 'Duplicate/unknown raw stream identity')
        seen.add((i, role))
        name = f'raw-stream-{i}-{role}.npz'
        shape = (requests[i]['frames'], WIDTHS[role])
        require(row.get('path') == name and row.get('shape') == list(shape) and row.get('dtype') == 'float32',
                'Raw receipt identity/shape mismatch: ' + name)
        require(isinstance(row.get('sha256'), str) and len(row['sha256']) == 64, 'Missing raw SHA256')
        values = reader.npz(name, {'values'}, row['sha256'])['values']
        require(values.shape == shape and values.dtype == np.dtype('float32'), 'Raw array shape/dtype mismatch: ' + name)
        result[i, role] = values
    require(seen == expected, 'Incomplete raw role coverage')
    return result


def reviews(clip):
    require(clip.get('human_approved') is False and clip.get('training_ready') is False, 'Unexpected approval/training claim')
    for key in ('language_review', 'motion_review', 'expression_review'):
        require(clip.get(key) == 'unverified', 'Review was promoted: ' + key)
    require('motion_approval_id' not in clip and 'review_attestation_sha256' not in clip, 'Inherited approval provenance')
    require(clip.get('nonmanual_candidate') == JAW, 'Nonmanual explicit opt-in profile mismatch')
    require(clip.get('template_sha256') == TEMPLATE_SHA and clip.get('expression_channels_retained') == 50,
            'Source template/native expression contract mismatch')
    require(clip.get('unmapped_target_channels') == ['expression', 'jaw_swing'], 'Unmapped expression/jaw_swing status changed')
    require(clip.get('export_supported') is False and clip.get('presentation_status') == 'research_candidate',
            'Unexpected export or presentation acceptance claim')


def audit_complete(reader, manifest, report):
    import numpy as np
    from slbridge.animation.registry import MotionRegistry
    from slbridge.animation.presentation import PresentationBundle
    from slbridge.animation.signspark_codec import decode_signspark_streams
    from slbridge.animation.gesture import motion_metrics

    requests = identity_checks(manifest)
    report['checks'].append('fixed identities, two requests, all three strict 859-tensor load receipts')
    raw = raw_arrays(reader, manifest, requests, WIDTHS)
    report['checks'].append('eight raw streams: SHA256, identity, shape, float32, finite')
    variations = {str(i): float(np.max(np.abs(np.diff(raw[i, 'body'].astype(np.float64), axis=0)))) for i in range(2)}
    require(all(v > 1e-5 for v in variations.values()), 'Static raw body output')
    text_difference = float(np.max(np.abs(raw[0, 'body'].astype(np.float64) - raw[1, 'body'])))
    require(text_difference > 0, 'Different texts produced identical same-seed raw body output')
    declared = manifest.get('different_text_body_max_difference')
    require(isinstance(declared, (int, float)) and np.isfinite(declared) and np.isclose(declared, text_difference, rtol=1e-6, atol=1e-7),
            'Reported text-condition difference disagrees with raw arrays')
    control = manifest.get('masked_observation_control_max_difference')
    require(isinstance(control, (int, float)) and np.isfinite(control) and 0 <= control <= 1e-5,
            'Masked observation control receipt missing/failed')
    report['raw_body_temporal_max_difference'] = variations
    report['different_text_body_max_difference'] = text_difference
    report['masked_observation_control'] = {'recorded_max_difference': control, 'verification': 'manifest receipt only; inference not reexecuted'}

    archive_name = 'candidate-motion.zip'
    archive_sha = manifest.get('candidate_archive_sha256')
    require(isinstance(archive_sha, str) and len(archive_sha) == 64, 'Missing candidate archive SHA256')
    # Only the bounded generated archive is read; no original checkpoints/template are opened.
    archive_data = reader.read(archive_name, 64 * 1024 * 1024, archive_sha)
    del archive_data
    registry = MotionRegistry(reader.path(archive_name), archive_sha)
    assets = registry.catalog()['assets']
    require(len(assets) == 2, 'Candidate archive must contain only the two generated requests')
    by_id = {asset['id']: asset for asset in assets}
    ids = []; candidate_results = []
    keys = set(WIDTHS) | {'effective_poses', 'expression', 'refined_poses', 'gesture_repair_mask'}
    for i, request in enumerate(requests):
        request_sha = digest(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
        identity = 'generated-csl-' + request_sha[:16]; ids.append(identity)
        require(identity in by_id, 'Generated request is missing from registry')
        asset = by_id[identity]
        clip = reader.json(identity + '.json')
        require(clip.get('id') == identity and clip.get('text') == request['text'] and clip.get('frames') == request['frames'],
                'Final clip request identity mismatch')
        required = {'source_kind': 'pretrained_text_conditioned_generation', 'source_sha256': request_sha,
                    'source_model_sha256': {key: value[1] for key, value in MODEL_FILES.items()},
                    'model_revision': MODEL_REVISION, 'generation_source_commit': SOURCE_COMMIT,
                    'generation_codec': CODEC, 'target_motion': None, 'observed_keyframes': 0,
                    'explicit_output_frames': request['frames'], 'fps': 30.0}
        for key, value in required.items():
            require(key in clip and clip[key] == value, 'Final clip protocol mismatch: ' + key)
        reviews(clip)
        parameters_sha = clip.get('generated_parameters_sha256')
        require(isinstance(parameters_sha, str) and len(parameters_sha) == 64, 'Missing final NPZ SHA256')
        arrays = reader.npz(identity + '.npz', keys, parameters_sha)
        for role in WIDTHS:
            require(arrays[role].dtype == raw[i, role].dtype and np.array_equal(arrays[role], raw[i, role]),
                    'Final NPZ raw values differ from saved raw stream: ' + role)
        frames = request['frames']
        # The codec retains float32; gesture._validated deliberately converts
        # to float64, and refine_gestures returns source.copy() in that dtype.
        for field, shape, dtype in (('effective_poses', (frames, 55, 3), 'float32'),
                                    ('refined_poses', (frames, 55, 3), 'float64'),
                                    ('expression', (frames, 50), 'float32')):
            require(arrays[field].shape == shape and arrays[field].dtype == np.dtype(dtype), 'Final NPZ array contract mismatch: ' + field)
        require(arrays['gesture_repair_mask'].shape == (frames, 55) and arrays['gesture_repair_mask'].dtype == np.dtype('bool'),
                'Invalid repair-mask contract')
        decoded = decode_signspark_streams(*(raw[i, k] for k in ('body', 'left', 'right', 'face')), left_hand_convention='model-right')
        codec_error = float(np.max(np.abs(decoded.poses.astype(np.float64) - arrays['effective_poses'])))
        require(codec_error <= 1e-6 and np.array_equal(decoded.expression, arrays['expression']), 'Codec replay disagrees with saved artifact')
        require(float(np.max(np.abs(np.diff(arrays['effective_poses'].astype(np.float64), axis=0)))) > 1e-5, 'Static effective pose output')
        require(np.array_equal(np.asarray(clip.get('poses')), arrays['effective_poses'].astype(np.float64)), 'Final JSON raw poses differ from NPZ')
        require(np.array_equal(np.asarray(clip.get('refined_poses')), arrays['refined_poses'].astype(np.float64)), 'Final JSON refined poses differ from NPZ')
        require(clip.get('gesture_refinement', {}).get('profile') == 'soke-gesture-refinement-v1', 'Unknown refinement profile')
        require(clip['gesture_refinement'].get('language_review') == clip['gesture_refinement'].get('motion_review') == 'NOT_RUN', 'Refinement claims unearned review')
        after = motion_metrics(arrays['refined_poses'], clip['fps'])
        recorded_after = clip['gesture_refinement'].get('after', {})
        for key, value in after.items():
            recorded = recorded_after.get(key)
            require(isinstance(recorded, (int, float)) and np.isfinite(recorded) and
                    (recorded == value if isinstance(value, int) else np.isclose(recorded, value, rtol=1e-7, atol=1e-7)),
                    'Recorded after-motion metric differs from actual refined NPZ: ' + key)
        expected_warning = after['mapped_step_events_over30'] > 0
        clip_warning = clip.get('quality_warning', False)
        registry_warning = asset.get('quality_warning')
        quality = {'actual_after': after, 'expected_quality_warning': expected_warning,
                   'clip_quality_warning_present': 'quality_warning' in clip,
                   'clip_quality_warning': clip_warning, 'registry_quality_warning': registry_warning}
        if (type(clip_warning) is not bool or clip_warning is not expected_warning or
                type(registry_warning) is not bool or registry_warning is not expected_warning):
            # Continue other integrity checks and run01 comparison, but retain
            # REJECTED status so this warning cannot be hidden by valid hashes.
            report['errors'].append(f'QualityWarningMismatch: {identity}: actual after>30 events='
                                    f"{after['mapped_step_events_over30']}, expected={expected_warning}, "
                                    f'clip={clip_warning}, registry={registry_warning}')
        packaged = registry.motion(identity)['clip']
        require(all(packaged.get(key) == value for key, value in clip.items()), 'Registry clip differs from final standalone JSON')
        reviews(packaged)
        require(asset.get('group') == 'generated' and asset.get('language_review') == asset.get('motion_review') == 'unverified',
                'Registry misclassifies/reviews generated source')
        with zipfile.ZipFile(reader.path(archive_name)) as archive:
            payload = archive.read('demo/experimental/' + identity + '.json')
        require(digest(payload) == asset['sha256'], 'Registry motion SHA does not match packaged JSON')
        candidate_results.append({'id': identity, 'request_index': i, 'parameters_sha256': parameters_sha,
                                  'registry_motion_sha256': asset['sha256'], 'codec_max_abs_difference': codec_error,
                                  'quality': quality})
    require(manifest.get('asset_ids') == ids and set(by_id) == set(ids), 'Manifest asset ID set/order differs from requests')
    report['checks'].append('final NPZ/raw/codec/JSON/registry provenance and values consistent')
    report['candidates'] = candidate_results

    inputs = {'source_template_sha256': TEMPLATE_SHA, 'target_avatar_sha256': AVATAR_SHA,
              'source_presentation_manifest_sha256': SOURCE_PRESENTATION}
    require(manifest.get('presentation_inputs') == inputs, 'Presentation input identity mismatch')
    require(manifest.get('nonmanual_profile') == JAW['profile'] and manifest.get('mouth_glb_export_supported') is False,
            'Unexpected generation nonmanual/export status')
    require(manifest.get('target_avatar_sha256') == AVATAR_SHA, 'Generation target avatar mismatch')
    p = reader.json('presentation/manifest.json', 128 * 1024, manifest.get('presentation_manifest_sha256'))
    require(p.get('files') == {'avatar-clean.glb': AVATAR_SHA} and p.get('avatar') == 'avatar-clean.glb', 'Presentation is not avatar-only')
    require({path.name for path in reader.path('presentation').iterdir()} == {'manifest.json', 'avatar-clean.glb'}, 'Unexpected presentation files')
    require(p.get('source_presentation_manifest_sha256') == SOURCE_PRESENTATION and p.get('processing_version') == 'signspark-avatar-jaw-candidate-v1',
            'Presentation version/source mismatch')
    require(p.get('status') == 'research_candidate' and p.get('export_supported') is False, 'Presentation acceptance/export promoted')
    require(all(p.get(key) == 'unverified' for key in ('language_review', 'motion_review', 'expression_review')), 'Presentation review promoted')
    pilots = p.get('pilots', [])
    require(len(pilots) == 2 and {pilot.get('asset_id') for pilot in pilots} == set(ids), 'Pilots are not the new candidate IDs')
    for pilot in pilots:
        require('export_file' not in pilot and pilot.get('source_motion_sha256') == by_id[pilot['asset_id']]['sha256'], 'Pilot export/source SHA mismatch')
        require(pilot.get('text') == by_id[pilot['asset_id']]['text'], 'Pilot text differs from its bound generated asset')
        require(all(pilot.get(key) == 'unverified' for key in ('language_review', 'motion_review', 'expression_review')), 'Pilot review promoted')
    bundle = PresentationBundle(reader.path('presentation'), assets)
    require(bundle.catalog()['manifest_sha256'] == manifest['presentation_manifest_sha256'], 'PresentationBundle manifest SHA differs')
    report['checks'].append('avatar-only PresentationBundle, two new SHA-bound pilots, explicit nonmanual profile, no export')
    report['presentation'] = {'manifest_sha256': bundle.catalog()['manifest_sha256'], 'avatar_sha256': bundle.catalog()['avatar_sha256']}
    return requests, raw


def compare_failed(reader, requests, raw):
    import numpy as np
    other = reader.json('generation-manifest.json')
    initial_sha = reader.receipts['generation-manifest.json']['sha256']
    require(other.get('state') == 'failed', 'Comparison reference must remain a failed run')
    old_requests = identity_checks(other, complete=False)
    require(old_requests == requests, 'Comparison requests/explicit lengths/seeds differ')
    old = raw_arrays(reader, other, old_requests, ('body', 'left', 'right'))
    rows = []
    for i in range(2):
        for role in ('body', 'left', 'right'):
            a, b = old[i, role], raw[i, role]
            rows.append({'request_index': i, 'role': role, 'bitwise_equal': a.tobytes() == b.tobytes(),
                         'max_abs_difference': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))})
    reader.read('generation-manifest.json', expected=initial_sha)
    return {'reference_run': str(reader.root), 'reference_state': 'failed', 'reference_error': other.get('error'),
            'reference_is_not_a_complete_positive_control': True, 'all_bitwise_equal': all(row['bitwise_equal'] for row in rows),
            'streams': rows, 'face': 'NOT_COMPARED: no completed face streams in failed reference', 'files_verified': reader.receipts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New JSON report outside audited run directories')
    parser.add_argument('--compare-run', type=Path, help='Optional failed run01 with six completed body/hand streams')
    args = parser.parse_args(argv)
    run = args.run_dir.resolve(); output = args.output.resolve()
    compared = args.compare_run.resolve() if args.compare_run else None
    if output.exists() or output.is_relative_to(run) or (compared is not None and output.is_relative_to(compared)):
        parser.error('Use a new output report outside all audited run directories')
    reader = Reader(run)
    report = {'schema': 'signspark-generated-artifact-audit-v1', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'run_dir': str(run), 'status': 'REJECTED', 'checks': [], 'errors': [],
              'scope': 'bounded artifact consistency only; no inference or checkpoint hashing',
              'checkpoint_identity_check': 'manifest load receipts compared with fixed identities; large checkpoint bytes not rehashed',
              'novel_text_membership': 'NOT_RECHECKED; no corpus-wide scan',
              'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN', 'expression_review': 'NOT_RUN',
              'browser_playback': 'NOT_RUN', 'codec_replay': 'consistency check, not independent protocol proof'}
    try:
        manifest = reader.json('generation-manifest.json')
        initial_sha = reader.receipts['generation-manifest.json']['sha256']
        report.update(observed_state=manifest.get('state'), generation_manifest_sha256=initial_sha,
                      recorded_raw_stream_count=len(manifest.get('raw_streams', [])), recorded_error=manifest.get('error'))
        require(manifest.get('state') == 'complete', f"Run state is {manifest.get('state')!r}; only complete runs can pass")
        require('error' not in manifest, 'Complete run retains an error field')
        report['checks'].append('generation manifest is complete')
        requests, raw = audit_complete(reader, manifest, report)
        if compared is not None:
            report['comparison'] = compare_failed(Reader(compared), requests, raw)
            if not report['comparison']['all_bitwise_equal']:
                report['comparison_note'] = 'Completed stream differences are reported; equality is not assumed and failed run remains failed'
        reader.read('generation-manifest.json', expected=initial_sha)
        report['status'] = 'PASS' if not report['errors'] else 'REJECTED'
    except (AuditFailure, OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['files_verified'] = reader.receipts
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'observed_state': report.get('observed_state'),
                      'output': str(output), 'errors': report['errors']}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
