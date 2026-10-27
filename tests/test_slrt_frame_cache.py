"""TDD equivalence tests for the SLRT per-frame preprocessing cache (task-24).

The optimisation may only remove duplicate work: every cached window tensor must
be **bit-identical** to the batched ``prepare_inputs`` call, and the production
runtime (cache enabled vs disabled) must produce the same Gloss on the same
sample.  The cache is stride-agnostic but its benefit comes from stride < window
overlap: at stride=1 every frame is otherwise prepared ~window_size times.

The integration test loads the real checkpoint/video and is opt-in because it
needs dataset assets, CUDA and a few seconds per arm:

    SLRT_FRAME_CACHE_INTEGRATION=1 pytest tests/test_slrt_frame_cache.py -q
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from slbridge.slrt.preprocess import FramePreparationCache, prepare_inputs


def _window(frames: int = 16, seed: int = 20261027):
    rng = np.random.default_rng(seed)
    bgr = rng.integers(0, 255, size=(frames, 320, 320, 3), dtype=np.uint8)
    kp = np.zeros((frames, 133, 3), np.float32)
    kp[..., 0] = rng.uniform(20, 500, size=(frames, 133))
    kp[..., 1] = rng.uniform(20, 500, size=(frames, 133))
    kp[..., 2] = rng.uniform(.2, 1, size=(frames, 133))
    return bgr, kp


def _assert_bitwise_equal(left, right):
    assert left.shape == right.shape
    assert torch.equal(left, right), (
        f'max abs diff {float((left - right).abs().max()):.3e}')


def test_cached_window_is_bit_identical_to_batched_prepare_inputs():
    bgr, kp = _window()
    cache = FramePreparationCache()
    cached_rgb, cached_hm = cache.prepare(bgr, kp, device='cpu')
    reference_rgb, reference_hm = prepare_inputs(bgr, kp, device='cpu')
    _assert_bitwise_equal(cached_rgb, reference_rgb)
    _assert_bitwise_equal(cached_hm, reference_hm)
    assert cached_rgb.shape == (1, 3, 16, 224, 224)
    assert cached_hm.shape == (1, 63, 16, 112, 112)
    assert (cache.misses, cache.hits) == (16, 0)  # all random frames are distinct


def test_repeated_frames_hit_the_cache_and_stay_bit_identical():
    bgr, kp = _window(frames=1)
    bgr = np.repeat(bgr, 16, axis=0)
    kp = np.repeat(kp, 16, axis=0)
    cache = FramePreparationCache()
    first_rgb, first_hm = cache.prepare(bgr, kp, device='cpu')
    assert (cache.misses, cache.hits) == (1, 15)
    second_rgb, second_hm = cache.prepare(bgr, kp, device='cpu')
    assert (cache.misses, cache.hits) == (1, 31)
    reference_rgb, reference_hm = prepare_inputs(bgr, kp, device='cpu')
    for candidate in (first_rgb, second_rgb):
        _assert_bitwise_equal(candidate, reference_rgb)
    for candidate in (first_hm, second_hm):
        _assert_bitwise_equal(candidate, reference_hm)


def test_stride_one_overlapping_windows_are_bit_identical_and_each_frame_prepared_once():
    """Mirrors the production stream: window=16, stride=1 -> 16x duplicate work."""
    frames = 32
    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 255, size=(frames, 320, 320, 3), dtype=np.uint8)
    kp = np.zeros((frames, 133, 3), np.float32)
    kp[..., 0] = rng.uniform(20, 500, size=(frames, 133))
    kp[..., 1] = rng.uniform(20, 500, size=(frames, 133))
    kp[..., 2] = rng.uniform(.2, 1, size=(frames, 133))
    cache = FramePreparationCache()
    starts = list(range(frames - 16 + 1))
    windows = 0
    for start in starts:
        window_bgr = np.ascontiguousarray(bgr[start:start + 16])
        window_kp = np.ascontiguousarray(kp[start:start + 16])
        cached_rgb, cached_hm = cache.prepare(window_bgr, window_kp, device='cpu')
        reference_rgb, reference_hm = prepare_inputs(window_bgr, window_kp, device='cpu')
        _assert_bitwise_equal(cached_rgb, reference_rgb)
        _assert_bitwise_equal(cached_hm, reference_hm)
        windows += 1
    assert windows == len(starts)
    # Every source frame is prepared exactly once across all overlapping windows.
    assert cache.misses == frames, f'expected {frames} unique frames, got {cache.misses}'
    assert cache.hits == windows * 16 - frames


def test_cache_limit_bounds_entries_without_changing_values():
    frames = 32
    rng = np.random.default_rng(11)
    bgr = rng.integers(0, 255, size=(frames, 320, 320, 3), dtype=np.uint8)
    kp = np.zeros((frames, 133, 3), np.float32)
    kp[..., 0] = rng.uniform(20, 500, size=(frames, 133))
    kp[..., 1] = rng.uniform(20, 500, size=(frames, 133))
    kp[..., 2] = rng.uniform(.2, 1, size=(frames, 133))
    cache = FramePreparationCache(limit=4)
    for start in range(frames - 16 + 1):
        window_bgr = np.ascontiguousarray(bgr[start:start + 16])
        window_kp = np.ascontiguousarray(kp[start:start + 16])
        cached_rgb, cached_hm = cache.prepare(window_bgr, window_kp, device='cpu')
        reference_rgb, reference_hm = prepare_inputs(window_bgr, window_kp, device='cpu')
        _assert_bitwise_equal(cached_rgb, reference_rgb)
        _assert_bitwise_equal(cached_hm, reference_hm)
        assert cache.size <= 4


def test_cache_preserves_window_validation_errors():
    cache = FramePreparationCache()
    bgr, kp = _window(frames=4)
    with pytest.raises(ValueError, match='320 x 320 x 3 uint8'):
        cache.prepare(np.zeros((4, 32, 32, 3), np.uint8), kp, device='cpu')
    bad_confidence = kp.copy()
    bad_confidence[..., 2] = -1
    with pytest.raises(ValueError, match='confidence'):
        cache.prepare(bgr, bad_confidence, device='cpu')
    with pytest.raises(ValueError, match='same frame indices'):
        cache.prepare(bgr, kp[:2], device='cpu')
    assert cache.size == 0  # failures must not populate the cache


@pytest.mark.skipif(
    os.environ.get('SLRT_FRAME_CACHE_INTEGRATION') != '1',
    reason='opt-in integration test (real checkpoint + video + CUDA)',
)
def test_gloss_is_identical_with_and_without_cache_on_real_sample():
    from pathlib import Path

    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend
    import torch as _torch

    if not _torch.cuda.is_available():
        pytest.skip('CUDA is required for the integration timing test')
    root = Path(__file__).resolve().parents[1]
    config_path = root / 'configs' / 'slrt-csl-hrnet-npz.example.json'
    trimmed = root / 'artifacts' / 'csldaily-video-test-627-20261026' / 'trimmed' / 'S000502_P0000_T00.mp4'
    pose_npz = root / 'artifacts' / 'csldaily-video-test-627-20261026' / 'pose' / 'S000502_P0000_T00.npz'
    if not (config_path.is_file() and trimmed.is_file() and pose_npz.is_file()):
        pytest.skip('SLRT/CSL-Daily assets are not available in this environment')
    config = SLRTConfig.from_file(config_path)
    results = {}
    for label, cache_enabled in (('cached', True), ('batched', False)):
        backend = SLRTBackend(config, frame_cache_enabled=cache_enabled)
        try:
            results[label] = backend.recognize_video(trimmed, pose_npz=pose_npz)
        finally:
            backend.close()
    assert results['cached']['gloss_sequence'] == results['batched']['gloss_sequence']
    assert results['cached']['windows_processed'] == results['batched']['windows_processed']
