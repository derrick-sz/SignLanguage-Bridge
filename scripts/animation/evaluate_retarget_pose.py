"""Audit paired train-source avatar/human poses with explicit signer holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slbridge.animation.retarget_pose_evaluation import evaluate_pose_samples, load_bound_pose_sample


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def run(args):
    if args.output.exists():
        raise ValueError('use a new output path; do not overwrite earlier evidence')
    native, recordings, human, extraction = map(read, (args.native_manifest, args.recording_manifest,
                                                        args.human_manifest, args.extraction_record))
    if (extraction['native_manifest_sha256'] != digest(args.native_manifest)
            or extraction['recording_manifest_sha256'] != digest(args.recording_manifest)
            or recordings['source_manifest_sha256'] != digest(args.native_manifest)):
        raise ValueError('manifest or extraction identity mismatch')
    native_samples, recorded_samples = native['samples'], recordings['samples']
    if len(native_samples) != len(recorded_samples) or extraction['samples_processed'] != len(native_samples):
        raise ValueError('incomplete extraction is not a complete run')
    extracted = extraction.get('records', [])
    if len(extracted) != len(native_samples):
        raise ValueError('extraction lacks per-sample integrity records')
    humans = {sample['sample_id']: sample for sample in human['samples'] if sample.get('kind') == 'positive'}
    samples, bindings = [], []
    for motion, video, saved in zip(native_samples, recorded_samples, extracted):
        identity = motion['sample_id']
        if (video['sample_id'] != identity or motion['source_split'] != 'train'
                or video['index'] != motion['index'] or video['source_motion_sha256'] != motion['motion_sha256']
                or saved['index'] != motion['index'] or saved['sample_id'] != identity or identity not in humans):
            raise ValueError('source sample identity mismatch')
        original = humans[identity]
        if (original.get('source_split') != 'train' or original.get('signer_id') != motion['signer_id']
                or original.get('video_sha256') != motion.get('human_video_sha256')):
            raise ValueError('human train source binding mismatch')
        pose_path = args.human_pose_dir / f'{identity}.pkl'
        paths = [(motion['motion_path'], motion['motion_sha256']), (video['video'], video['video_sha256']),
                 (original['video'], original['video_sha256']), (pose_path, original['pose_sha256'])]
        if any(digest(path) != expected for path, expected in paths):
            raise ValueError('source file digest mismatch')
        binding = dict(sample_id=identity, signer_id=motion['signer_id'],
                       human_pose_sha256=original['pose_sha256'], avatar_video_sha256=video['video_sha256'],
                       source_motion_sha256=motion['motion_sha256'], crop_start=original['crop_start'],
                       crop_end_exclusive=original['crop_end_exclusive'], avatar_frames=video['frames'],
                       motion_frames=motion['frames'], avatar_pre_frames=args.avatar_pre_frames)
        path = args.poses / f"{motion['index']:03d}.npz"
        if (saved['cache_sha256'] != digest(path) or saved['human_pose_sha256'] != binding['human_pose_sha256']
                or saved['avatar_video_sha256'] != binding['avatar_video_sha256']
                or saved['source_motion_sha256'] != binding['source_motion_sha256']):
            raise ValueError('pose cache digest or extraction identity mismatch')
        samples.append(load_bound_pose_sample(path, binding))
        bindings.append(dict(sample_id=identity, pose_cache_sha256=digest(path), **{
            key: binding[key] for key in ('human_pose_sha256', 'avatar_video_sha256', 'source_motion_sha256')}))
    result = evaluate_pose_samples(samples, fit_signers=args.fit_signers, holdout_signers=args.holdout_signers,
                                   min_confidence=args.min_confidence)
    result['provenance'] = dict(native_manifest_sha256=digest(args.native_manifest),
                                recording_manifest_sha256=digest(args.recording_manifest),
                                human_manifest_sha256=digest(args.human_manifest),
                                extraction_record_sha256=digest(args.extraction_record),
                                fit_signers=sorted(args.fit_signers), holdout_signers=sorted(args.holdout_signers),
                                pose_bindings=bindings,
                                warning='linear time correspondence is provisional; 2D poses are estimates, not sign-language approval')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('fit', 'holdout')}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--recording-manifest', type=Path, required=True)
    parser.add_argument('--human-manifest', type=Path, required=True)
    parser.add_argument('--extraction-record', type=Path, required=True)
    parser.add_argument('--human-pose-dir', type=Path, required=True)
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--fit-signers', nargs='+', required=True)
    parser.add_argument('--holdout-signers', nargs='+', required=True)
    parser.add_argument('--avatar-pre-frames', type=int, default=6)
    parser.add_argument('--min-confidence', type=float, default=.3)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
