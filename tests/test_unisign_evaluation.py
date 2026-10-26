import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_unisign_cecsl import load_pose, save_pose
from slbridge.uni_sign_preprocess import PoseWindow


def test_pose_cache_rejects_different_frontend_and_direction_protocols(tmp_path):
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'a fixed video fixture')
    values = [np.zeros((3, n, 3), dtype=np.float32) for n in [9, 21, 21, 18]]
    pose = PoseWindow(*values, sampled_indices=(0, 1, 2),
                      quality={'sampled_indices': [0, 1, 2], 'pose_orientation': 'preserve'})
    cache = tmp_path / 'sample.npz'
    protocol = {'pose_mode': 'lightweight', 'max_frames': 256, 'pose_orientation': 'preserve'}
    save_pose(cache, pose, video, 'isotropic_short_side', pose_protocol=protocol)
    restored = load_pose(cache, video, 'isotropic_short_side', pose_protocol=protocol)
    assert np.array_equal(restored.left, pose.left)
    assert not restored.left.flags.writeable
    for key, value in [('pose_mode', 'performance'), ('max_frames', 128), ('pose_orientation', 'mirror')]:
        with pytest.raises(ValueError, match='protocol'):
            load_pose(cache, video, 'isotropic_short_side', pose_protocol={**protocol, key: value})


def test_pose_cache_rejects_changed_video_bytes_even_with_same_size_and_mtime(tmp_path):
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'video A')
    stat = video.stat()
    pose = PoseWindow(*(np.zeros((2, n, 3), np.float32) for n in [9, 21, 21, 18]),
                      sampled_indices=(0, 1), quality={'sampled_indices': [0, 1]})
    path = tmp_path / 'cache.npz'
    save_pose(path, pose, video, 'isotropic_short_side', pose_protocol={'pose_orientation': 'preserve'})
    video.write_bytes(b'video B')
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='content'):
        load_pose(path, video, 'isotropic_short_side', pose_protocol={'pose_orientation': 'preserve'})


def test_direction_calibration_uses_dev_calibration_rows_only():
    from scripts.calibrate_unisign_cecsl_direction import choose_directions

    rows = [
        {'number': f'dev-{i:05d}', 'translator': signer, 'phase': 'calibration',
         'scores': {'preserve': original, 'mirror': reflected}}
        for signer, original, reflected, offset in [('A', .4, .1, 0), ('B', .1, .5, 10)]
        for i in range(offset + 1, offset + 4)
    ]
    rows.append({'number': 'dev-00050', 'translator': 'A', 'phase': 'validation',
                 'scores': {'preserve': 0, 'mirror': 1}})
    directions, details = choose_directions(rows, ['A', 'B'])
    assert directions == {'A': 'preserve', 'B': 'mirror'}
    assert details['A']['n'] == 3
    rows[0]['number'] = 'test-00001'
    with pytest.raises(ValueError, match='dev'):
        choose_directions(rows, ['A', 'B'])


def test_resume_refuses_changed_decoding_or_directions_before_overwriting(tmp_path):
    from scripts.evaluate_unisign_cecsl import ensure_run_protocol

    protocol = {'num_beams': 4, 'directions': {'A': 'preserve'}}
    identity = ensure_run_protocol(tmp_path, protocol)
    assert identity == ensure_run_protocol(tmp_path, protocol)
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text('existing evidence\n', encoding='utf-8')
    for changed in [{'num_beams': 2, 'directions': {'A': 'preserve'}},
                    {'num_beams': 4, 'directions': {'A': 'mirror'}}]:
        with pytest.raises(ValueError, match='protocol'):
            ensure_run_protocol(tmp_path, changed)
    assert predictions.read_text(encoding='utf-8') == 'existing evidence\n'


def test_direction_map_rejects_test_calibration_and_uncovered_signers(tmp_path):
    from scripts.evaluate_unisign_cecsl import load_direction_map

    path = tmp_path / 'direction-map.json'
    protocol = {'pose_orientation': 'preserve'}
    config = SimpleNamespace(checkpoint_sha256='a' * 64)
    data = {'schema_version': 1, 'dataset': 'CE-CSL', 'calibration_split': 'dev',
            'checkpoint_sha256': 'a' * 64, 'pose_protocol': protocol,
            'calibration_ids': ['dev-00001'], 'directions': {'A': 'mirror'}}
    path.write_text(json.dumps(data), encoding='utf-8')
    assert load_direction_map(path, config, protocol, ['A']) == {'A': 'mirror'}
    with pytest.raises(ValueError, match='signer'):
        load_direction_map(path, config, protocol, ['A', 'B'])
    data['calibration_split'] = 'test'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='train or dev'):
        load_direction_map(path, config, protocol, ['A'])


def test_calibration_refuses_orphaned_prediction_rows_without_claiming_a_manifest(tmp_path):
    from scripts.calibrate_unisign_cecsl_direction import prepare_calibration_run

    (tmp_path / 'predictions.jsonl').write_text('{"number":"dev-00001"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='manifest'):
        prepare_calibration_run(tmp_path, {'selected': []})
    assert not (tmp_path / 'calibration-manifest.json').exists()


def test_pose_protocol_binds_the_resolved_extractor_device():
    from scripts.evaluate_unisign_cecsl import build_pose_protocol, protocol_digest

    cfg = SimpleNamespace(runtime={'device': 'cuda', 'pose_device': 'cpu'}, source_commit='fixed')
    cpu = build_pose_protocol(cfg)
    cfg.runtime['pose_device'] = 'cuda'
    cuda = build_pose_protocol(cfg)
    assert cpu['pose_device'] == 'cpu'
    assert cuda['pose_device'] == 'cuda'
    assert protocol_digest(cpu) != protocol_digest(cuda)


def test_calibration_resume_rejects_changed_phase_and_reference(tmp_path):
    from scripts.calibrate_unisign_cecsl_direction import prepare_calibration_run
    from scripts.evaluate_unisign_cecsl import protocol_digest

    selected = {'number': 'dev-00001', 'translator': 'A', 'reference': '你好', 'phase': 'calibration'}
    identity = {'selected': [selected], 'dtype': 'float32'}
    assert prepare_calibration_run(tmp_path, identity) == {}
    row = {**selected, 'calibration_protocol_id': protocol_digest(identity),
           'predictions': {'preserve': '你好', 'mirror': '再见'}, 'scores': {'preserve': 1., 'mirror': 0.}}
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    assert list(prepare_calibration_run(tmp_path, identity)) == ['dev-00001']
    for key, bad in [('phase', 'validation'), ('reference', 'STALE_REFERENCE')]:
        path.write_text(json.dumps({**row, key: bad}) + '\n', encoding='utf-8')
        with pytest.raises(ValueError, match='reference/phase/video'):
            prepare_calibration_run(tmp_path, identity)
