"""Uni-Sign RTMPose-133 preprocessing.

The upstream online demo expects Wholebody output with 133 keypoints. It forms
four graph parts and writes the per-keypoint confidence into channel 2. This
module ports that preprocessing without using MediaPipe's 225-column protocol or
substituting a z coordinate for confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


RTMPOSE_KEYPOINT_COUNT = 133
CONFIDENCE_THRESHOLD = 0.3
BODY_INDICES = tuple([0] + list(range(3, 11)))
LEFT_HAND_INDICES = tuple(range(91, 112))
RIGHT_HAND_INDICES = tuple(range(112, 133))
FACE_INDICES = tuple(list(range(23, 40, 2)) + list(range(83, 91)) + [53])
# COCO Wholebody anatomical swap pairs, restricted to the selected parts.
# https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/configs/_base_/datasets/coco_wholebody.py
BODY_MIRROR_INDICES = (0, 2, 1, 4, 3, 6, 5, 8, 7)
FACE_MIRROR_INDICES = (8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 12, 11, 10, 9, 16, 15, 14, 17)


@dataclass(frozen=True)
class PoseWindow:
    body: np.ndarray
    left: np.ndarray
    right: np.ndarray
    face_all: np.ndarray
    sampled_indices: tuple[int, ...]
    quality: dict[str, Any]

    @property
    def parts(self) -> dict[str, np.ndarray]:
        return {"body": self.body, "left": self.left, "right": self.right, "face_all": self.face_all}


def orient_pose_window(pose: PoseWindow, pose_orientation: str) -> PoseWindow:
    """Apply an explicit direction to canonical, body/root-normalized poses.

    No direction is inferred from a prediction, signer name or reference text.
    Coordinates and confidences move together; reflecting twice restores the
    original parts. Reapplying the requested direction is a no-op.
    """
    if pose_orientation not in {"preserve", "mirror"}:
        raise ValueError("pose_orientation must be preserve or mirror")
    current = pose.quality.get("pose_orientation", "preserve")
    if current not in {"preserve", "mirror"}:
        raise ValueError("Invalid cached pose_orientation")
    if current == pose_orientation:
        return pose
    parts = {
        "body": pose.body[:, BODY_MIRROR_INDICES, :].copy(),
        "left": pose.right.copy(),
        "right": pose.left.copy(),
        "face_all": pose.face_all[:, FACE_MIRROR_INDICES, :].copy(),
    }
    for part in parts.values():
        part[..., 0] *= -1
        part.setflags(write=False)
    quality = dict(pose.quality, pose_orientation=pose_orientation)
    if "left_hand_presence" in quality and "right_hand_presence" in quality:
        quality["left_hand_presence"], quality["right_hand_presence"] = (
            quality["right_hand_presence"], quality["left_hand_presence"])
    return PoseWindow(**parts, sampled_indices=pose.sampled_indices, quality=quality)


def normalize_rtmpose_pixels(points: Any, width: int, height: int, *,
                               mode: str = "isotropic_short_side") -> np.ndarray:
    """Normalize RTMPose pixel coordinates without distorting source geometry.

    CSL-Daily pose files were produced on square frames. CE-CSL and camera
    frames are commonly widescreen, so dividing x by width and y by height
    would compress the horizontal axis. The default short-side normalization
    uses one common pixel scale for both axes and preserves the aspect ratio.
    The historical ``wh`` mode remains available for explicit compatibility.
    """
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise ValueError("height must be a positive integer")
    values = np.asarray(points, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("points must have a final coordinate dimension of 2")
    if mode == "isotropic_short_side":
        scale = float(min(width, height))
        divisor = np.asarray([scale, scale], dtype=np.float32)
    elif mode == "wh":
        divisor = np.asarray([width, height], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported pose_normalization mode: {mode}")
    return values / divisor


def deterministic_frame_indices(frame_count: int, max_frames: int = 256) -> tuple[int, ...]:
    """Return sorted, unique indices without random frame loss across windows."""
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("frame_count must be a nonnegative integer")
    if type(max_frames) is not int or max_frames < 2:
        raise ValueError("max_frames must be an integer >= 2")
    if frame_count <= max_frames:
        return tuple(range(frame_count))
    indices = np.rint(np.linspace(0, frame_count - 1, num=max_frames)).astype(np.int64)
    # ``linspace`` cannot duplicate for frame_count > max_frames, but retaining
    # an explicit uniqueness check makes the contract stable if implementation
    # details change in a future NumPy release.
    unique = tuple(int(value) for value in np.unique(indices))
    if len(unique) != max_frames:
        raise RuntimeError("Deterministic sampling did not produce the requested frame count")
    return unique


def _coerce_pose_arrays(keypoints: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32)
    if points.ndim == 3 and points.shape[-2:] == (RTMPOSE_KEYPOINT_COUNT, 2):
        points = points[:, None, :, :]
    elif points.ndim != 4 or points.shape[-2:] != (RTMPOSE_KEYPOINT_COUNT, 2):
        raise ValueError("keypoints must have shape (T,133,2), (T,1,133,2), or (T,N,133,2)")
    if confidence.ndim == 2 and confidence.shape[-1] == RTMPOSE_KEYPOINT_COUNT:
        confidence = confidence[:, None, :]
    elif confidence.ndim != 3 or confidence.shape[-1] != RTMPOSE_KEYPOINT_COUNT:
        raise ValueError("scores must have shape (T,133), (T,1,133), or (T,N,133)")
    if points.shape[0] != confidence.shape[0] or points.shape[1] != confidence.shape[1]:
        raise ValueError("keypoints and scores disagree in frame/person dimensions")
    if points.shape[0] < 2:
        raise ValueError("Uni-Sign requires at least two frames")
    if points.shape[1] < 1:
        raise ValueError("Uni-Sign requires at least one detected person per frame")
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise ValueError("keypoints and scores must be finite")
    # The official loader selects instance zero. Keep that explicit for videos
    # containing more than one detected person.
    return points[:, 0, :, :], confidence[:, 0, :]


def _crop_scale(motion: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Port of the upstream body crop/scale normalization."""
    result = motion.copy()
    observed = motion[..., 2] > CONFIDENCE_THRESHOLD
    valid_coordinates = motion[observed][:, :2]
    if len(valid_coordinates) < 4:
        zeros = np.zeros(motion.shape, dtype=np.float32)
        return zeros, 0.0, {"valid_body_frames": 0, "valid_body_fraction": 0.0}
    xmin = float(valid_coordinates[:, 0].min())
    xmax = float(valid_coordinates[:, 0].max())
    ymin = float(valid_coordinates[:, 1].min())
    ymax = float(valid_coordinates[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)
    if not math.isfinite(scale) or scale <= 0:
        zeros = np.zeros(motion.shape, dtype=np.float32)
        return zeros, 0.0, {"valid_body_frames": int(observed.any(axis=1).sum()),
                            "valid_body_fraction": float(observed.any(axis=1).mean())}
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1).astype(np.float32)
    result[result[..., 2] <= CONFIDENCE_THRESHOLD] = 0
    frame_valid = observed.any(axis=1)
    return result, scale, {"valid_body_frames": int(frame_valid.sum()),
                           "valid_body_fraction": float(frame_valid.mean())}


def _part_from_indices(points: np.ndarray, confidence: np.ndarray,
                       indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    part = points[:, indices, :].copy()
    scores = confidence[:, indices].copy()
    if indices in (LEFT_HAND_INDICES, RIGHT_HAND_INDICES):
        part = part - part[:, :1, :]
    elif indices == FACE_INDICES:
        part = part - part[:, -1:, :]
    return part, scores


def prepare_pose_window(keypoints: Any, scores: Any, *, max_frames: int = 256) -> PoseWindow:
    """Create the four B,T,V,3 tensors expected by Uni-Sign.

    Coordinates must already be normalized. The runtime default uses one
    common short-side pixel scale to preserve wide-screen geometry; legacy
    ``wh`` callers may provide separate width/height normalization.
    The third channel is always the original keypoint confidence.
    """
    points, confidence = _coerce_pose_arrays(keypoints, scores)
    source_frame_count = int(points.shape[0])
    sampled = deterministic_frame_indices(points.shape[0], max_frames=max_frames)
    points = points[list(sampled)]
    confidence = confidence[list(sampled)]

    body_xy, body_scores = _part_from_indices(points, confidence, BODY_INDICES)
    body_motion = np.concatenate([body_xy, body_scores[..., None]], axis=-1).astype(np.float32)
    body, scale, body_quality = _crop_scale(body_motion)

    parts = {"body": body}
    presence: dict[str, float] = {}
    for name, indices in (("left", LEFT_HAND_INDICES), ("right", RIGHT_HAND_INDICES), ("face_all", FACE_INDICES)):
        part_xy, part_scores = _part_from_indices(points, confidence, indices)
        part = np.concatenate([part_xy, part_scores[..., None]], axis=-1).astype(np.float32)
        if scale == 0:
            part = np.zeros(part.shape, dtype=np.float32)
        else:
            part[..., :2] = part[..., :2] / scale
            part = np.clip(part, -1, 1)
            part[part[..., 2] <= CONFIDENCE_THRESHOLD] = 0
        parts[name] = part.astype(np.float32, copy=False)
        presence[name] = float((part_scores > CONFIDENCE_THRESHOLD).any(axis=1).mean())

    expected_shapes = {"body": (len(sampled), 9, 3), "left": (len(sampled), 21, 3),
                       "right": (len(sampled), 21, 3), "face_all": (len(sampled), 18, 3)}
    for name, part in parts.items():
        if part.shape != expected_shapes[name]:
            raise RuntimeError(f"Unexpected Uni-Sign {name} shape: {part.shape}")
        part.setflags(write=False)

    quality = {
        "source_frames": source_frame_count,
        "sampled_frames": len(sampled),
        "sampled_indices": list(sampled),
        "body_scale": float(scale),
        "valid_body_frames": int(body_quality["valid_body_frames"]),
        "valid_body_fraction": float(body_quality["valid_body_fraction"]),
        "left_hand_presence": presence["left"],
        "right_hand_presence": presence["right"],
        "face_presence": presence["face_all"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "confidence_is_channel_2": True,
        "keypoint_count": RTMPOSE_KEYPOINT_COUNT,
        "pose_orientation": "preserve",
    }
    return PoseWindow(body=parts["body"], left=parts["left"], right=parts["right"],
                      face_all=parts["face_all"], sampled_indices=sampled, quality=quality)

