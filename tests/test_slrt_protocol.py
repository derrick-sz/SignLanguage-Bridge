"""Contract tests: these do not claim recognition accuracy."""
import json
from pathlib import Path

import numpy as np
import pytest

from slbridge.slrt.assets import SLRTConfig, SLRTResourceError, load_vocabulary
from slbridge.slrt.preprocess import select_keypoints, prepare_inputs
from slbridge.slrt.decoder import IncrementalGlossDecoder


def test_vocab_order_and_blank_are_strict(tmp_path):
    p = tmp_path / 'vocab.json'
    p.write_text(json.dumps(['<blank>', 'A', 'B']), encoding='utf-8')
    assert load_vocabulary(p, expected_classes=3) == ('<blank>', 'A', 'B')
    p.write_text(json.dumps(['A', '<blank>', 'B']), encoding='utf-8')
    with pytest.raises(SLRTResourceError, match='blank'):
        load_vocabulary(p, expected_classes=3)


def test_reject_mediapipe_and_wrong_confidence():
    with pytest.raises(ValueError, match='133'):
        select_keypoints(np.zeros((16, 225), np.float32))
    x = np.zeros((16, 133, 3), np.float32)
    x[..., 2] = -1
    with pytest.raises(ValueError, match='confidence'):
        select_keypoints(x)


def test_point_order_has_hands_mouth_and_upper_body():
    x = np.zeros((16, 133, 3), np.float32)
    x[..., 0] = np.arange(133)
    assert select_keypoints(x)[0, :, 0].tolist() == list(range(91, 133)) + list(range(71, 91, 2)) + list(range(11))


def test_official_color_geometry_and_heatmap():
    torch = pytest.importorskip('torch')
    bgr = np.zeros((16, 320, 320, 3), np.uint8)
    bgr[..., 0] = 255
    kp = np.zeros((16, 133, 3), np.float32)
    kp[..., :2] = 256
    # Official threshold=0 does not multiply heatmaps by confidence.
    rgb, hm = prepare_inputs(bgr, kp, device='cpu')
    assert rgb.shape == (1, 3, 16, 224, 224)
    assert hm.shape == (1, 63, 16, 112, 112)
    assert torch.allclose(rgb[:, 0], torch.ones_like(rgb[:, 0]), atol=1e-6)
    assert torch.allclose(rgb[:, 2], -torch.ones_like(rgb[:, 2]), atol=1e-6)
    assert hm.max() > .9 and hm.min() >= -1


def test_blank_separated_repeat_is_preserved_and_finalized():
    d = IncrementalGlossDecoder(('<blank>', 'A', 'B'), vote_size=1)
    events = []
    for i, k in enumerate([1, 1, 0, 1, 2, 2]):
        p = np.full(3, .05, np.float32); p[k] = .9
        events += d.push(p, anchor_time=i / 30, available_time=(i + 8) / 30,
                         evidence_range=(max(0, i - 7) / 30, (i + 8) / 30))
    events += d.flush()
    final = [e for e in events if e['status'] == 'final']
    assert [e['gloss'] for e in final] == ['A', 'A', 'B']
    assert all(e['score'] == pytest.approx(.9) for e in final)
    assert all(e['boundary_source'] == 'sliding_window_anchor_estimate' for e in final)
    assert d.flush() == []
    d.reset()
    assert d.flush() == []


def test_majority_vote_waits_for_future_predictions_and_ties_blank():
    d = IncrementalGlossDecoder(('<blank>', 'A', 'B'), vote_size=3)
    def push(k, i):
        p = np.full(3, .05); p[k] = .9
        return d.push(p, anchor_time=i, available_time=i + 8, evidence_range=(i, i + 8))
    assert push(1, 0) == []
    events = push(2, 1)
    assert events[0]['gloss'] == 'A' and events[0]['available_time'] == 9
    push(0, 2)
    assert any(e['status'] == 'final' for e in d.flush()) is False


def test_config_rejects_unsupported_language_and_invalid_windows():
    with pytest.raises(ValueError):
        SLRTConfig(window_size=15)
    with pytest.raises(ValueError):
        SLRTConfig(stride=0)
    with pytest.raises(ValueError):
        SLRTConfig(sign_language='libras')


def test_vote_evidence_includes_left_and_right_context():
    d=IncrementalGlossDecoder(('<blank>','A'),vote_size=7)
    events=[]
    for i,k in enumerate([0]*15+[1]*10+[0]*10):
        p=np.array([.05,.05]);p[k]=.95
        events+=d.push(p,anchor_time=i/30,available_time=(i+8)/30,
                       evidence_range=(max(0,i-7)/30,(i+8)/30))
    first=next(e for e in events if e['gloss']=='A')
    assert first['time_range'][0]==15/30
    assert first['evidence_time_range'][0]==5/30
    assert first['available_time']==26/30


def test_mirror_swaps_points_and_is_involutive():
    from slbridge.slrt.preprocess import mirror_keypoints_133
    kp = np.zeros((4, 133, 3), np.float32)
    kp[..., 0] = np.arange(133, dtype=np.float32) * 2
    kp[..., 2] = np.linspace(.1, .9, 133, dtype=np.float32)
    once = mirror_keypoints_133(kp)
    assert np.array_equal(mirror_keypoints_133(once), kp)
    for a, b in ((5, 6), (17, 20), (91, 112), (91 + 7, 112 + 7), (71, 77)):
        assert np.allclose(once[:, a, 0], 512 - kp[:, b, 0]) and np.allclose(once[:, b, 0], 512 - kp[:, a, 0])
        assert np.allclose(once[:, a, 1:], kp[:, b, 1:]) and np.allclose(once[:, b, 1:], kp[:, a, 1:])
    from slbridge.slrt.preprocess import MIRROR_133
    assert MIRROR_133[0] == 0 and MIRROR_133[56] == 56  # body nose and face nose tip map to themselves


def test_mirror_rejects_wrong_layout_and_width():
    from slbridge.slrt.preprocess import mirror_keypoints_133
    with pytest.raises(ValueError, match='133'):
        mirror_keypoints_133(np.zeros((4, 69, 3), np.float32))
    with pytest.raises(ValueError, match='reference_width'):
        mirror_keypoints_133(np.zeros((4, 133, 3), np.float32), reference_width=1)


def test_mirror_heatmap_equals_flipped_heatmap_grid():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(20261024)
    frames = rng.integers(0, 255, size=(16, 320, 320, 3), dtype=np.uint8)
    kp = np.zeros((16, 133, 3), np.float32)
    kp[..., 0] = rng.uniform(30, 480, size=(16, 133))
    kp[..., 1] = rng.uniform(30, 480, size=(16, 133))
    kp[..., 2] = rng.uniform(.3, 1, size=(16, 133))
    from slbridge.slrt.preprocess import KEYPOINT_INDICES, MIRROR_133, mirror_keypoints_133
    rgb, hm = prepare_inputs(frames, kp, device='cpu')
    mirrored_rgb, mirrored_hm = prepare_inputs(np.ascontiguousarray(frames[:, :, ::-1, :]),
                                               mirror_keypoints_133(kp), device='cpu')
    # Mirroring relabels left/right, so heatmap channels permute as well: the
    # mirrored sample equals the flipped sample with hands/mouth/body left-right
    # channels swapped to their mirrored counterparts.
    position = {g: c for c, g in enumerate(KEYPOINT_INDICES)}
    channels = [position[MIRROR_133[g]] for g in KEYPOINT_INDICES]
    flipped = hm.flip(-1)[:, channels]
    assert torch.allclose(mirrored_rgb, rgb.flip(-1), atol=1e-4)
    assert torch.allclose(mirrored_hm, flipped, atol=1e-4)
    assert channels[:21] == list(range(21, 42))  # the two hands exchange blocks
