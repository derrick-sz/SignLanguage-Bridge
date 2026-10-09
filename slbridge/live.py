"""Captured RGB clips processed with the model's recorded video contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._io import canonical_hash
from .contracts import InputSpec, PreparedInput
from .features import InputRejected, prepare_array

if TYPE_CHECKING:
    from .sources import CapturedClip


def prepare_clip(clip: CapturedClip, spec: InputSpec) -> PreparedInput:
    """Uniformly sample a complete, unmirrored clip and reset tracking per call.

    Format and environment errors raise ValueError; insufficient observable
    content raises InputRejected, exactly as the file preparation pipeline does.
    """
    if spec.representation_id != "mediapipe225_rgb_v1":
        raise ValueError("Captured RGB clips require a mediapipe225_rgb_v1 model")
    from .sources import CapturedClip

    if not isinstance(clip, CapturedClip):
        raise TypeError("clip must be CapturedClip")
    import cv2
    import mediapipe as mp
    import numpy as np

    from .media import ExtractionConfig, extractor_identity, normalize_landmarks
    config = ExtractionConfig.from_dict(spec.extractor["config"])
    identity = extractor_identity(config)
    if canonical_hash(identity) != canonical_hash(spec.extractor["identity"]):
        raise ValueError("Video extractor runtime differs from the model's saved identity")
    frames = clip.frames
    quality = {"source_frames": len(frames), "timestamps_known": True,
               "timestamp_source": "captured_frame_timestamp"}
    if len(frames) < 2:
        raise InputRejected("at_least_two_frames_required", quality)
    times = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    if not np.isfinite(times).all() or times[0] < 0 or not (np.diff(times) > 0).all():
        raise ValueError("Captured timestamps must be finite, nonnegative and strictly increasing")
    indices = [frame.frame_index for frame in frames]
    if any(type(index) is not int or index < 0 for index in indices) or any(
            current <= previous for previous, current in zip(indices, indices[1:])):
        raise ValueError("Captured frame indices must be strictly increasing")
    first = frames[0].image
    if not isinstance(first, np.ndarray):
        raise ValueError("Captured frames must be consistent uint8 BGR images")
    shape = first.shape
    for frame in frames:
        if frame.source_id != clip.source_id:
            raise ValueError("Captured frames must belong to the clip source")
        if (not isinstance(frame.image, np.ndarray) or frame.image.dtype != np.uint8
                or frame.image.ndim != 3 or frame.image.shape[2] != 3
                or frame.image.shape != shape or min(frame.image.shape[:2]) < 1):
            raise ValueError("Captured frames must be consistent uint8 BGR images")
    selected = np.rint(np.linspace(0, len(frames) - 1, min(config.frames, len(frames)))).astype(np.int64)
    sequence, observations = [], []
    # Public Holistic API with the same recorded options as media.py. Opening a
    # fresh graph also isolates every call; reset explicitly matches file clips.
    with mp.solutions.holistic.Holistic(
            model_complexity=config.model_complexity, **identity["holistic_options"]) as holistic:
        holistic.reset()
        for index in selected:
            image = frames[int(index)].image
            if image.shape[1] > config.max_width:
                height = max(1, int(round(image.shape[0] * config.max_width / image.shape[1])))
                image = cv2.resize(image, (config.max_width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.setflags(write=False)
            values, observed = normalize_landmarks(
                holistic.process(rgb), image.shape[1], image.shape[0], config.pose_visibility)
            sequence.append(values)
            observations.append(observed)
    values = np.ascontiguousarray(sequence, dtype=np.float32)
    quality.update({
        "sampled_frames": len(selected), "source_width": shape[1], "source_height": shape[0],
        "extraction_width": image.shape[1], "extraction_height": image.shape[0],
        "duration_seconds": float(times[-1] - times[0]),
        "duration_definition": "last_minus_first_decoded_timestamp",
        "anchor_frame_fraction": sum(item["anchor"] for item in observations) / len(observations),
        "hand_frame_fraction": sum(item["left_hand"] or item["right_hand"] for item in observations) / len(observations),
        "left_hand_frame_fraction": sum(item["left_hand"] for item in observations) / len(observations),
        "right_hand_frame_fraction": sum(item["right_hand"] for item in observations) / len(observations),
        "valid_fraction": float((values != -999).mean()),
        "missing_policy": "unobserved_point_xyz_all_minus999",
    })
    if quality["anchor_frame_fraction"] < config.minimum_anchor_frame_fraction:
        raise InputRejected("insufficient_anchor_frames", quality)
    if quality["hand_frame_fraction"] < config.minimum_hand_frame_fraction:
        raise InputRejected("insufficient_hand_frames", quality)
    return prepare_array(values, spec, clip.source_id, timestamps=times[selected], quality_metadata=quality)
