import numpy as np
import pytest

from slbridge.uni_sign_preprocess import (
    BODY_INDICES,
    FACE_INDICES,
    LEFT_HAND_INDICES,
    RIGHT_HAND_INDICES,
    deterministic_frame_indices,
    normalize_rtmpose_pixels,
    prepare_pose_window,
)


def synthetic_pose(frames=6):
    points = np.zeros((frames, 1, 133, 2), dtype=np.float32)
    scores = np.ones((frames, 1, 133), dtype=np.float32)
    for frame in range(frames):
        base = np.array([0.2 + frame * .01, 0.3], dtype=np.float32)
        for offset, index in enumerate(BODY_INDICES):
            points[frame, 0, index] = base + np.array([offset * .02, offset * .01])
        for indices in (LEFT_HAND_INDICES, RIGHT_HAND_INDICES):
            for offset, index in enumerate(indices):
                points[frame, 0, index] = np.array([.3 + offset * .01, .4 + offset * .005])
        for offset, index in enumerate(FACE_INDICES):
            points[frame, 0, index] = np.array([.5 + offset * .005, .6 + offset * .003])
    return points, scores


def test_preprocess_shapes_and_confidence_channel():
    points, scores = synthetic_pose()
    scores[..., 5] = .1
    window = prepare_pose_window(points, scores)
    assert window.body.shape == (6, 9, 3)
    assert window.left.shape == (6, 21, 3)
    assert window.right.shape == (6, 21, 3)
    assert window.face_all.shape == (6, 18, 3)
    assert np.allclose(window.body[:, :, 2], np.where(scores[0, 0, BODY_INDICES] > .3,
                                                      scores[0, 0, BODY_INDICES], 0))
    assert np.allclose(window.left[:, 0, :2], 0)
    assert np.allclose(window.right[:, 0, :2], 0)
    assert np.allclose(window.face_all[:, -1, :2], 0)
    assert window.quality['confidence_is_channel_2'] is True
    assert not window.body.flags.writeable


def test_confidence_is_not_replaced_by_z_coordinate():
    points, scores = synthetic_pose()
    points[..., 1] = 0.75
    scores[:] = 0.8
    window = prepare_pose_window(points, scores)
    assert np.allclose(window.body[..., 2][np.abs(window.body).sum(-1) > 0], 0.8)


def test_deterministic_sampling_preserves_endpoints():
    assert deterministic_frame_indices(10, 256) == tuple(range(10))
    sampled = deterministic_frame_indices(300, 256)
    assert len(sampled) == 256 and sampled[0] == 0 and sampled[-1] == 299
    assert list(sampled) == sorted(set(sampled))


def test_isotropic_pose_normalization_preserves_geometry_for_widescreen_video():
    pixels = np.array([[[0.0, 0.0], [1080.0, 0.0], [0.0, 1080.0]]], dtype=np.float32)
    normalized = normalize_rtmpose_pixels(
        pixels, width=1920, height=1080, mode='isotropic_short_side')
    assert np.allclose(normalized, [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    horizontal = np.linalg.norm(normalized[0, 1] - normalized[0, 0])
    vertical = np.linalg.norm(normalized[0, 2] - normalized[0, 0])
    assert np.isclose(horizontal, vertical)


def test_pose_normalization_keeps_legacy_wh_mode_and_validates_inputs():
    pixels = np.array([[[960.0, 540.0]]], dtype=np.float32)
    normalized = normalize_rtmpose_pixels(pixels, width=1920, height=1080, mode='wh')
    assert np.allclose(normalized, [[[0.5, 0.5]]])
    with pytest.raises(ValueError, match='mode'):
        normalize_rtmpose_pixels(pixels, width=1920, height=1080, mode='unsupported')
    with pytest.raises(ValueError, match='width'):
        normalize_rtmpose_pixels(pixels, width=0, height=1080)


def test_invalid_pose_is_rejected_and_insufficient_body_has_zero_scale():
    with pytest.raises(ValueError):
        prepare_pose_window(np.zeros((3, 132, 2), np.float32), np.ones((3, 132), np.float32))
    bad = np.zeros((3, 1, 133, 2), np.float32)
    bad[..., 0] = np.nan
    with pytest.raises(ValueError):
        prepare_pose_window(bad, np.zeros((3, 1, 133), np.float32))
    points, scores = synthetic_pose()
    scores[:] = 0
    window = prepare_pose_window(points, scores)
    assert window.quality['body_scale'] == 0
    assert window.quality['valid_body_fraction'] == 0


def test_direction_correction_keeps_anatomical_points_and_confidence_together():
    from slbridge.uni_sign_preprocess import PoseWindow, orient_pose_window

    body = np.zeros((2, 9, 3), dtype=np.float32)
    left = np.zeros((2, 21, 3), dtype=np.float32)
    right = np.zeros((2, 21, 3), dtype=np.float32)
    face = np.zeros((2, 18, 3), dtype=np.float32)
    body[:, 0] = [.1, -.3, .73]  # nose
    body[:, 7] = [.4, .3, .81]   # anatomical left wrist
    body[:, 8] = [-.2, .5, .92]  # anatomical right wrist
    left[:, 4] = [.05, .04, .6]
    right[:, 4] = [-.04, .08, .9]
    face[:, 0] = [-.3, .2, .6]
    face[:, 8] = [.2, .3, .7]
    pose = PoseWindow(body, left, right, face, (0, 1),
                      {'left_hand_presence': .6, 'right_hand_presence': .8})
    mirrored = orient_pose_window(pose, 'mirror')
    assert np.allclose(mirrored.body[0, [0, 7, 8]],
                       [[-.1, -.3, .73], [.2, .5, .92], [-.4, .3, .81]])
    assert np.allclose(mirrored.left[0, 4], [.04, .08, .9])
    assert np.allclose(mirrored.right[0, 4], [-.05, .04, .6])
    assert np.allclose(mirrored.face_all[0, 0], [-.2, .3, .7])
    assert mirrored.quality['left_hand_presence'] == .8
    assert mirrored.quality['right_hand_presence'] == .6
    assert mirrored.quality['pose_orientation'] == 'mirror'
    assert not mirrored.left.flags.writeable
    assert np.allclose(pose.body[0, 7], [.4, .3, .81])
    assert orient_pose_window(mirrored, 'mirror') is mirrored
    restored = orient_pose_window(mirrored, 'preserve')
    for name in pose.parts:
        assert np.array_equal(pose.parts[name], restored.parts[name])
    with pytest.raises(ValueError, match='pose_orientation'):
        orient_pose_window(pose, 'auto')
