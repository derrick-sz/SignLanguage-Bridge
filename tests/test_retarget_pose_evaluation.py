import numpy as np
import pytest
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from slbridge.animation.retarget_pose_evaluation import evaluate_pose_samples, load_bound_pose_sample


def sample(sample_id, signer, offset):
    human = np.zeros((3, 133, 2), dtype=np.float64)
    avatar = np.zeros((3, 133, 3), dtype=np.float64)
    human[:, 5] = avatar[:, 5, :2] = [40, 50]
    human[:, 6] = avatar[:, 6, :2] = [60, 50]
    human[:, 9] = [[46, 55], [48, 56], [50, 57]]
    human[:, 10] = [[54, 55], [52, 56], [50, 57]]
    avatar[:, [9, 10], :2] = human[:, [9, 10]] - offset
    avatar[:, :, 2] = 1
    return dict(sample_id=sample_id, signer_id=signer, human_xy=human,
                human_score=np.ones((3, 133)), avatar=avatar, valid=np.ones(3, bool))


def test_offsets_fit_only_training_signer_and_keep_reverse_control():
    train = sample('S1_P1', 'P1', np.array([2, 4]))
    holdout = sample('S2_P2', 'P2', np.array([2, 4]))
    # A low confidence outlier must not influence the fit or its denominator.
    train['human_xy'][0, 9] = [9999, 9999]
    train['human_score'][0, 9] = .01
    result = evaluate_pose_samples([train, holdout], fit_signers={'P1'}, holdout_signers={'P2'})
    assert result['fit']['point_pairs'] == 5
    assert result['holdout']['point_pairs'] == 6
    np.testing.assert_allclose(result['offsets']['left'], [.1, .2], atol=1e-7)
    assert result['holdout']['before_mean'] == pytest.approx(np.sqrt(.05))
    assert result['holdout']['after_mean'] == pytest.approx(0, abs=1e-7)
    assert result['holdout']['reverse_mean'] > result['holdout']['forward_mean']


def test_rejects_overlapping_signers_and_source_sentences():
    first = sample('S1_P1', 'P1', np.array([2, 4]))
    second = sample('S1_P2', 'P2', np.array([2, 4]))
    with pytest.raises(ValueError, match='signer'):
        evaluate_pose_samples([first, second], fit_signers={'P1'}, holdout_signers={'P1', 'P2'})
    with pytest.raises(ValueError, match='source'):
        evaluate_pose_samples([first, second], fit_signers={'P1'}, holdout_signers={'P2'})


def test_rejects_missing_confident_shoulders_instead_of_reporting_zero_error():
    first = sample('S1_P1', 'P1', np.array([2, 4]))
    second = sample('S2_P2', 'P2', np.array([2, 4]))
    second['human_score'][:, 5] = 0
    with pytest.raises(ValueError, match='holdout'):
        evaluate_pose_samples([first, second], fit_signers={'P1'}, holdout_signers={'P2'})


def test_rejects_misbound_pose_and_out_of_range_frames(tmp_path):
    data = sample('S1_P1', 'P1', np.array([2, 4]))
    path = tmp_path / 'pose.npz'
    binding = dict(sample_id='S1_P1', signer_id='P1', human_pose_sha256='a'*64,
                   avatar_video_sha256='b'*64, source_motion_sha256='c'*64,
                   crop_start=10, crop_end_exclusive=13, avatar_frames=20, motion_frames=3, avatar_pre_frames=2)
    arrays = {k: data[k] for k in ['human_xy', 'human_score', 'avatar', 'valid']}
    arrays.update({k: np.array(binding[k]) for k in ['sample_id', 'signer_id', 'human_pose_sha256',
                                                       'avatar_video_sha256', 'source_motion_sha256']})
    arrays.update(human_indices=np.array([10, 11, 12]), avatar_indices=np.array([2, 3, 4]), clock=np.array([0., .5, 1.]))
    np.savez(path, **arrays)
    assert load_bound_pose_sample(path, binding)['sample_id'] == 'S1_P1'
    arrays['sample_id'] = np.array('S2_P1')
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='identity'):
        load_bound_pose_sample(path, binding)
    arrays['sample_id'] = np.array('S1_P1')
    arrays['human_indices'] = np.array([10, 11, 13])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='frame'):
        load_bound_pose_sample(path, binding)


def test_cli_requires_extraction_cache_digest_before_reporting(tmp_path):
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    def write_json(path, data):
        path.write_text(json.dumps(data), encoding='utf8')
    pose_dir, source_dir = tmp_path/'poses', tmp_path/'source'
    pose_dir.mkdir(); source_dir.mkdir()
    native, videos, humans, records = [], [], [], []
    for index, signer in enumerate(['P1', 'P2']):
        ident = f'S{index+1}_{signer}'
        motion, video, human_video, pose = [tmp_path/f'{name}-{index}' for name in ('motion', 'video', 'human', 'pose')]
        motion.write_bytes(f'motion-{index}'.encode()); video.write_bytes(f'video-{index}'.encode())
        human_video.write_bytes(f'human-{index}'.encode()); pose.write_bytes(f'pose-{index}'.encode())
        pose.rename(source_dir/f'{ident}.pkl'); pose = source_dir/f'{ident}.pkl'
        d = sample(ident, signer, np.array([2, 4]))
        cache = pose_dir/f'{index:03d}.npz'
        np.savez(cache, **{k:d[k] for k in ('human_xy', 'human_score', 'avatar', 'valid')},
                 sample_id=np.array(ident), signer_id=np.array(signer), human_pose_sha256=np.array(digest(pose)),
                 avatar_video_sha256=np.array(digest(video)), source_motion_sha256=np.array(digest(motion)),
                 human_indices=np.array([10, 11, 12]), avatar_indices=np.array([2, 3, 4]), clock=np.array([0., .5, 1.]))
        native.append(dict(index=index, sample_id=ident, signer_id=signer, source_split='train', frames=3,
                           motion_path=str(motion), motion_sha256=digest(motion), human_video_sha256=digest(human_video)))
        videos.append(dict(index=index, sample_id=ident, video=str(video), video_sha256=digest(video),
                           source_motion_sha256=digest(motion), frames=20))
        humans.append(dict(sample_id=ident, signer_id=signer, kind='positive', source_split='train',
                           video=str(human_video), video_sha256=digest(human_video), pose_sha256=digest(pose),
                           crop_start=10, crop_end_exclusive=13))
        records.append(dict(index=index, sample_id=ident, cache_sha256=digest(cache), human_pose_sha256=digest(pose),
                            avatar_video_sha256=digest(video), source_motion_sha256=digest(motion)))
    native_file, videos_file, human_file, extraction_file = [tmp_path/f'{name}.json' for name in ('native', 'videos', 'human', 'extraction')]
    write_json(native_file, {'samples':native})
    write_json(videos_file, {'samples':videos, 'source_manifest_sha256':digest(native_file)})
    write_json(human_file, {'samples':humans})
    write_json(extraction_file, {'samples_processed':2, 'records':records, 'native_manifest_sha256':digest(native_file),
                                  'recording_manifest_sha256':digest(videos_file)})
    args = SimpleNamespace(native_manifest=native_file, recording_manifest=videos_file, human_manifest=human_file,
                           extraction_record=extraction_file, human_pose_dir=source_dir, poses=pose_dir,
                           fit_signers=['P1'], holdout_signers=['P2'], avatar_pre_frames=2,
                           min_confidence=.3, output=tmp_path/'result.json')
    run = runpy.run_path(str(Path(__file__).parents[1]/'scripts/animation/evaluate_retarget_pose.py'))['run']
    cache = pose_dir/'000.npz'; original = cache.read_bytes()
    with np.load(cache, allow_pickle=False) as data:
        altered = {key:data[key].copy() for key in data.files}
    altered['human_xy'][0, 9, 0] += 10
    np.savez(cache, **altered)
    with pytest.raises(ValueError, match='cache digest'):
        run(args)
    assert not args.output.exists()
    cache.write_bytes(original)
    run(args)
    result = json.loads(args.output.read_text(encoding='utf8'))
    assert result['fit']['samples'] == result['holdout']['samples'] == 1
    with pytest.raises(ValueError, match='new output'):
        run(args)
