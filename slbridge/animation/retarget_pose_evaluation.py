"""Confidence-filtered 2D retarget diagnostics, never semantic approval.

Inputs must already have an explicitly documented temporal correspondence.
This module does not estimate timestamps, modify motion, or accept recognition
labels. Offsets are fitted only on the designated signers and are diagnostic
screen-space values rather than a production animation correction.
"""
from __future__ import annotations

import numpy as np


def load_bound_pose_sample(path, binding):
    """Load data-only NPZ and require matching source/video/frame identities."""
    identity = ('sample_id', 'signer_id', 'human_pose_sha256', 'avatar_video_sha256', 'source_motion_sha256')
    with np.load(path, allow_pickle=False) as data:
        if any(key not in data or np.asarray(data[key]).shape != () or str(data[key].item()) != binding[key]
               for key in identity):
            raise ValueError('pose identity does not match its bound inputs')
        values = {key: data[key].copy() for key in ('human_xy', 'human_score', 'avatar', 'valid',
                                                     'human_indices', 'avatar_indices', 'clock')}
    clock, hidx, aidx = values['clock'], values['human_indices'], values['avatar_indices']
    if (clock.ndim != 1 or len(clock) < 2 or hidx.shape != clock.shape or aidx.shape != clock.shape
            or hidx.dtype.kind not in 'iu' or aidx.dtype.kind not in 'iu' or not np.isfinite(clock).all()
            or clock[0] != 0 or clock[-1] != 1 or np.any(np.diff(clock) < 0)
            or np.any(hidx < binding['crop_start']) or np.any(hidx >= binding['crop_end_exclusive'])
            or np.any(aidx < 0) or np.any(aidx >= binding['avatar_frames'])
            or not np.array_equal(hidx, np.rint(binding['crop_start'] + clock *
                                                   (binding['crop_end_exclusive'] - binding['crop_start'] - 1)).astype(int))
            or not np.array_equal(aidx, np.rint(binding['avatar_pre_frames'] + clock *
                                                   (binding['motion_frames'] - 1)).astype(int))):
        raise ValueError('pose frame indices do not match the declared alignment')
    return {'sample_id': binding['sample_id'], 'signer_id': binding['signer_id'],
            **{key: values[key] for key in ('human_xy', 'human_score', 'avatar', 'valid')}}


def evaluate_pose_samples(samples, *, fit_signers, holdout_signers, min_confidence=.3):
    """Compare shoulder-normalized wrists with a signer-disjoint holdout.

    Each sample supplies same-length COCO-133 human ``human_xy`` and
    ``human_score`` arrays, detected avatar ``avatar`` x/y/confidence, and
    ``valid`` frame mask. The caller owns temporal and source identity checks.
    """
    fit_signers, holdout_signers = set(fit_signers), set(holdout_signers)
    if not fit_signers or not holdout_signers or fit_signers & holdout_signers:
        raise ValueError('fit and holdout signer sets must be nonempty and disjoint')
    if not 0 <= min_confidence <= 1:
        raise ValueError('invalid confidence threshold')
    rows, seen, source_groups = [], set(), {}
    for sample in samples:
        sample_id, signer = sample['sample_id'], sample['signer_id']
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError('duplicate or empty sample id')
        seen.add(sample_id)
        if signer in fit_signers:
            group = 'fit'
        elif signer in holdout_signers:
            group = 'holdout'
        else:
            raise ValueError('sample signer missing from declared split')
        source = sample_id.split('_', 1)[0]
        if source in source_groups and source_groups[source] != group:
            raise ValueError('source sentence crosses fit and holdout')
        source_groups[source] = group
        human = np.asarray(sample['human_xy'], dtype=np.float64)
        scores = np.asarray(sample['human_score'], dtype=np.float64)
        avatar = np.asarray(sample['avatar'], dtype=np.float64)
        valid = np.asarray(sample['valid'], dtype=bool)
        if (human.ndim != 3 or human.shape[1:] != (133, 2) or len(human) < 2
                or scores.shape != human.shape[:2] or avatar.shape != (*human.shape[:2], 3)
                or valid.shape != (len(human),)):
            raise ValueError('invalid pose sample shape')

        def normalize(points):
            center = (points[:, 5, :2] + points[:, 6, :2]) / 2
            scale = np.linalg.norm(points[:, 5, :2] - points[:, 6, :2], axis=1)
            safe = np.where(np.isfinite(scale) & (scale > 5), scale, np.nan)
            return (points[:, :, :2] - center[:, None, :]) / safe[:, None, None], scale

        human_n, human_scale = normalize(human)
        avatar_n, avatar_scale = normalize(avatar)
        shoulders = (valid & np.isfinite(human_scale) & (human_scale > 5)
                     & np.isfinite(avatar_scale) & (avatar_scale > 5)
                     & (scores[:, 5] >= min_confidence) & (scores[:, 6] >= min_confidence)
                     & (avatar[:, 5, 2] >= min_confidence) & (avatar[:, 6, 2] >= min_confidence))
        for side, joint in (('left', 9), ('right', 10)):
            for tick in range(len(human)):
                if (shoulders[tick] and scores[tick, joint] >= min_confidence
                        and avatar[tick, joint, 2] >= min_confidence
                        and np.isfinite(human_n[tick, joint]).all()
                        and np.isfinite(avatar_n[tick, joint]).all()):
                    rows.append(dict(sample_id=sample_id, signer_id=signer, group=group,
                                     side=side, tick=tick, ticks=len(human),
                                     human=human_n[tick, joint], avatar=avatar_n[tick, joint]))
    offsets = {}
    for side in ('left', 'right'):
        fitted = [r['human'] - r['avatar'] for r in rows if r['group'] == 'fit' and r['side'] == side]
        if not fitted:
            raise ValueError(f'fit has no confident {side} wrist points')
        offsets[side] = np.mean(fitted, axis=0)
    result = {'offsets': {side: v.tolist() for side, v in offsets.items()},
              'min_confidence': min_confidence,
              'alignment': 'caller-provided; semantic correctness not assessed'}
    for group in ('fit', 'holdout'):
        items = [r for r in rows if r['group'] == group]
        if not items:
            raise ValueError(f'{group} has no confident wrist points')
        before = [np.linalg.norm(r['human'] - r['avatar']) for r in items]
        after = [np.linalg.norm(r['human'] - r['avatar'] - offsets[r['side']]) for r in items]
        lookup = {(r['sample_id'], r['side'], r['tick']): r for r in items}
        pairs = [(r, lookup[(r['sample_id'], r['side'], r['ticks'] - 1 - r['tick'])])
                 for r in items if (r['sample_id'], r['side'], r['ticks'] - 1 - r['tick']) in lookup]
        result[group] = dict(samples=len({r['sample_id'] for r in items}), point_pairs=len(items),
                             before_mean=float(np.mean(before)), after_mean=float(np.mean(after)),
                             forward_mean=float(np.mean([np.linalg.norm(a['human'] - a['avatar']) for a, _ in pairs])) if pairs else None,
                             reverse_mean=float(np.mean([np.linalg.norm(b['human'] - a['avatar']) for a, b in pairs])) if pairs else None,
                             reverse_control_pairs=len(pairs))
    return result
