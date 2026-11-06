"""Decode native SignSparK streams to the bridge's effective SMPL-X rotations.

Protocol source: JianHe0628/SignSparK at
916402d0ecd6322fb4a9e5f513229a94a57bf4b6, DATA.md,
signspark/pose_datasets_lmdb.py and tools/visualize.py. The renderer uses
flat_hand_mean=True: these rotations must NOT receive SOKE's hand mean.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation


CODEC_VERSION = "signspark-row6d-flat-hand-v1"


@dataclass(frozen=True)
class SignSparKMotion:
    """Effective local rotations plus all 50 native expression coefficients.

    ``poses`` is float32 [T,55,3] axis-angle in radians. Root, the first eleven
    body joints and both eyes are unpredicted and neutral, not inferred motion.
    ``expression`` is a separate [T,50] copy without scaling or truncation.
    The bridge avatar need not support all native expression coefficients.
    """

    poses: np.ndarray
    expression: np.ndarray


def _rotation_matrices(features, name):
    rows = np.asarray(features, dtype=np.float64).reshape(-1, 6)

    def reject(mask):
        bad = np.flatnonzero(mask)
        if len(bad):
            joints = features.shape[1] // 6
            frame, joint = divmod(int(bad[0]), joints)
            raise ValueError(f"{name} has degenerate 6D rotation at frame {frame}, joint {joint}")

    def direction(vector):
        scale = np.max(np.abs(vector), axis=-1, keepdims=True)
        reject(scale[:, 0] <= 1e-8)
        # Scale before squaring so large finite decoder values cannot overflow.
        unit = vector / scale
        return unit / np.linalg.norm(unit, axis=-1, keepdims=True)

    first, second = direction(rows[:, :3]), direction(rows[:, 3:])
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    length = np.linalg.norm(second, axis=-1, keepdims=True)
    reject(length[:, 0] <= 1e-6)
    second = second / length
    return np.stack([first, second, np.cross(first, second)], axis=-2)


def decode_signspark_streams(
    body: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    face: np.ndarray,
    *,
    left_hand_convention: Literal["model-right", "smplx-left"],
) -> SignSparKMotion:
    """Decode unnormalized body[T,60], hands[T,90], face[T,56].

    All streams must describe the same frames. ``model-right`` is the trained
    hand model's convention; its left output is conjugated by diag(1,-1,-1).
    ``smplx-left`` is the released LMDB's native left convention and is already
    anatomical. The required keyword prevents guessing which boundary is used.
    No mean, standard deviation, translation or hand-mean correction is added.
    Inputs are neither modified nor retained by reference.

    Malformed, nonfinite, empty, mismatched or degenerate streams raise
    ValueError. Rotation rows with max-abs <= 1e-8 or relative orthogonal
    component <= 1e-6 are rejected, never replaced by identity.
    """
    if not isinstance(left_hand_convention, str) or left_hand_convention not in ("model-right", "smplx-left"):
        raise ValueError("left_hand_convention must be model-right or smplx-left")
    for name, value, width in (("body", body, 60), ("left", left, 90), ("right", right, 90), ("face", face, 56)):
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [T,{width}]")
        if value.dtype.kind not in "fiu" or value.dtype.itemsize > 8 or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite real numeric values")
    count = len(body)
    if count == 0 or any(len(value) != count for value in (left, right, face)):
        raise ValueError("All stream frames must match and be nonempty")
    poses = np.zeros((count, 55, 3), dtype=np.float32)
    poses[:, 12:22] = Rotation.from_matrix(_rotation_matrices(body, "body")).as_rotvec().reshape(count, 10, 3)
    left_matrices = _rotation_matrices(left, "left")
    if left_hand_convention == "model-right":
        flip = np.diag([1.0, -1.0, -1.0])
        left_matrices = flip @ left_matrices @ flip
    poses[:, 25:40] = Rotation.from_matrix(left_matrices).as_rotvec().reshape(count, 15, 3)
    poses[:, 40:55] = Rotation.from_matrix(_rotation_matrices(right, "right")).as_rotvec().reshape(count, 15, 3)
    poses[:, 22] = Rotation.from_matrix(_rotation_matrices(face[:, :6], "face")).as_rotvec()
    return SignSparKMotion(poses=poses, expression=face[:, 6:].copy())
