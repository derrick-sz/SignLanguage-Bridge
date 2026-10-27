"""CSL-Daily official inference geometry, without training augmentations."""
from __future__ import annotations

import numpy as np

KEYPOINT_INDICES = tuple(range(91, 133)) + tuple(range(71, 91, 2)) + tuple(range(11))

# Mirror pairs over the raw COCO-WholeBody133 layout: body left/right including
# feet, the 68-face bilateral landmarks (bridge centre, nose tip and the upper
# inner lip centres stay fixed), and the 21-point left/right hands in order.
_FACE_MIRROR_PAIRS = ((0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
                      (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
                      (31, 35), (32, 34),
                      (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
                      (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
                      (60, 64), (61, 63), (65, 67))
_BODY_MIRROR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                      (17, 20), (18, 21), (19, 22))


def _mirror_table_133():
    table = list(range(133))

    def pair(a, b):
        table[a], table[b] = b, a

    for a, b in _BODY_MIRROR_PAIRS:
        pair(a, b)
    for a, b in _FACE_MIRROR_PAIRS:
        pair(23 + a, 23 + b)
    for i in range(21):
        pair(91 + i, 112 + i)
    return tuple(table)


MIRROR_133 = _mirror_table_133()


def mirror_keypoints_133(keypoints, *, reference_width=512):
    """Reflect x about the vertical centre line and swap left/right points.

    Diagnostic support for explicit direction experiments; production inference
    keeps the recorded orientation. ``reference_width - x`` (rather than the
    half-pixel ``reference_width - 1 - x``) so the reflected heatmap grid
    commutes exactly with the 170-column Gaussian grid. Confidence moves with
    its point; applying the mirror twice restores the input exactly.
    """
    x = np.asarray(keypoints)
    if x.ndim != 3 or x.shape[1:] != (133, 3):
        raise ValueError('mirror_keypoints_133 requires T x 133 x 3 COCO-WholeBody points')
    if not isinstance(reference_width, int) or reference_width <= 1:
        raise ValueError('reference_width must be an integer greater than 1')
    out = x[:, MIRROR_133, :].copy()
    out[..., 0] = reference_width - out[..., 0]
    return out


def select_keypoints(keypoints):
    x = np.asarray(keypoints)
    if x.ndim != 3 or x.shape[1:] != (133, 3) or x.shape[0] == 0:
        raise ValueError('SLRT requires T x 133 x 3 COCO-WholeBody x/y/confidence; MediaPipe225 is incompatible')
    if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError('Keypoint coordinates must be finite floating-point values')
    if (x[..., 2] < 0).any() or (x[..., 2] > 1).any():
        raise ValueError('Keypoint confidence must lie in [0,1]; channel 3 is not z')
    return np.ascontiguousarray(x[:, KEYPOINT_INDICES], dtype=np.float32)


def canonical_frame(bgr):
    """PIL bilinear matches official Image.resize on CSL's 512-square frames."""
    from PIL import Image
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('SLRT video input must be a uint8 BGR image')
    # Store 320 pixels, not the original high-resolution camera frame.
    rgb = Image.fromarray(bgr[..., ::-1]).resize((320, 320), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(rgb)[..., ::-1])


def prepare_inputs(bgr, keypoints, *, device='cpu'):
    """Frames are BGR 320-square; keypoints are in 512-square reference pixels."""
    import torch
    from torchvision.transforms import functional as tvf
    x = np.asarray(bgr)
    if x.ndim != 4 or x.shape[1:] != (320, 320, 3) or x.dtype != np.uint8:
        raise ValueError('Expected T x 320 x 320 x 3 uint8 BGR frames')
    kp = select_keypoints(keypoints)
    if len(x) != len(kp):
        raise ValueError('RGB and keypoints must have exactly the same frame indices')
    # BGR is deliberate: official code receives RGB and reverses it before S3D.
    rgb = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255
    rgb = tvf.resize(tvf.center_crop(rgb, [270, 270]), [224, 224], antialias=True)
    rgb = (rgb - .5) / .5
    coords = torch.from_numpy(kp[..., :2]).to(device) * (170 / 512)
    yy, xx = torch.meshgrid(torch.arange(170, device=device), torch.arange(170, device=device), indexing='ij')
    # Bound peak allocation to four frames rather than all T x63 maps at once.
    heatmaps = []
    for c in coords.split(4):
        d2 = (xx - c[..., 0, None, None]) ** 2 + (yy - c[..., 1, None, None]) ** 2
        h = torch.exp(-d2 / (2 * 8 ** 2))
        h = tvf.resize(tvf.center_crop(h, [143, 143]), [112, 112], antialias=True)
        heatmaps.append((h - .5) / .5)
    hm = torch.cat(heatmaps)
    return rgb.permute(1, 0, 2, 3).unsqueeze(0), hm.permute(1, 0, 2, 3).unsqueeze(0)


# Windows with stride < window_size prepare the same frame many times (at
# stride=1 and window=16, ~16x).  This cache removes that duplicate work without
# changing arithmetic: each frame is prepared by the unmodified
# ``prepare_inputs`` on a single T=1 window, cached by exact input bytes and
# concatenated back to the batch layout.  A cache hit therefore returns exactly
# the tensor the batched call would have produced for that frame
# (tests/test_slrt_frame_cache.py asserts bit-identical output).
FRAME_CACHE_LIMIT = 128  # frames; bounded per runtime (~0.5 GB at full occupancy)


class FramePreparationCache:
    """Bounded per-frame RGB+heatmap cache for overlapping sliding windows.

    ``limit`` is a hard cap on stored frames (the whole cache is dropped when the
    cap is reached, which keeps VRAM/RAM bounded); ``limit=0`` keeps the per-frame
    code path but stores nothing.  The cache is not thread-safe by itself; the
    SLRT design runs one inference consumer per runtime.
    """

    def __init__(self, limit: int = FRAME_CACHE_LIMIT):
        if type(limit) is not int or limit < 0:
            raise ValueError('frame cache limit must be a non-negative integer')
        self.limit = limit
        self._frames: dict[tuple[int, int], tuple[object, object]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def info(self) -> dict:
        return dict(enabled=self.limit > 0, limit=self.limit, size=self.size,
                    hits=self.hits, misses=self.misses)

    def prepare(self, bgr, keypoints, *, device='cpu'):
        import torch
        x = np.asarray(bgr)
        if x.ndim != 4 or x.shape[1:] != (320, 320, 3) or x.dtype != np.uint8:
            raise ValueError('Expected T x 320 x 320 x 3 uint8 BGR frames')
        kp = select_keypoints(keypoints)  # validate exactly like prepare_inputs
        if len(x) != len(kp):
            raise ValueError('RGB and keypoints must have exactly the same frame indices')
        raw = np.asarray(keypoints)
        rgb_parts, hm_parts = [], []
        for index in range(len(x)):
            key = (hash(x[index].tobytes()), hash(raw[index].tobytes()))
            item = self._frames.get(key)
            if item is None:
                if self.limit and len(self._frames) >= self.limit:
                    self._frames.clear()
                item = prepare_inputs(x[index:index + 1], raw[index:index + 1], device=device)
                self.misses += 1
                if self.limit:
                    self._frames[key] = item
            else:
                self.hits += 1
            rgb_parts.append(item[0])
            hm_parts.append(item[1])
        return torch.cat(rgb_parts, dim=2), torch.cat(hm_parts, dim=2)
