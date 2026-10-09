"""Lazy, versioned RGB decoding and real MediaPipe Holistic extraction.

The new representation contains shoulder-relative 2D coordinates. Its third
coordinate is deliberately zero: Holistic's pose and hand z values do not share
a calibrated reference, so they cannot be combined as a 3D skeleton.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

from ._io import sha256_file
from .features import InputRejected


REPRESENTATION_ID = "mediapipe225_rgb_v1"


class VideoDecodeError(ValueError):
    """The source video cannot satisfy the declared clip decoding contract."""


@dataclass(frozen=True)
class ExtractionConfig:
    frames: int = 30
    max_width: int = 640
    model_complexity: int = 1
    pose_visibility: float = 0.5
    minimum_hand_frame_fraction: float = 0.1
    minimum_anchor_frame_fraction: float = 0.8
    mirror: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.frames) is not int or not 2 <= self.frames <= 1000:
            raise ValueError("frames must be an integer between 2 and 1000")
        if type(self.max_width) is not int or not 64 <= self.max_width <= 4096:
            raise ValueError("max_width must be an integer between 64 and 4096")
        if type(self.model_complexity) is not int or self.model_complexity != 1:
            raise ValueError("Only bundled model_complexity=1 is supported; model downloads are disabled")
        if type(self.mirror) is not bool or self.mirror:
            raise ValueError("RGB extraction preserves anatomical left/right and does not mirror")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported extraction version")
        for name in ("pose_visibility", "minimum_hand_frame_fraction", "minimum_anchor_frame_fraction"):
            value = getattr(self, name)
            if type(value) not in {float, int} or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExtractionConfig:
        if not isinstance(value, dict):
            raise ValueError("ExtractionConfig must be an object")
        unknown = set(value) - {entry.name for entry in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown extraction config fields: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class ExtractedClip:
    values: Any
    timestamps: Any
    frame_indices: Any
    quality: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DecodedClip:
    """Uniformly selected images after checking the entire decoded clip."""

    images: tuple[Any, ...]
    timestamps: Any
    frame_indices: Any
    quality: dict[str, Any]


@lru_cache(maxsize=1)
def _runtime_identity() -> dict[str, Any]:
    import cv2
    import mediapipe as mp
    import numpy as np

    if mp.__version__ != "0.10.14":
        raise RuntimeError("mediapipe225_rgb_v1 requires mediapipe==0.10.14")
    package = Path(mp.__file__).resolve().parent
    required = (
        "modules/holistic_landmark/holistic_landmark_cpu.binarypb",
        "modules/pose_landmark/pose_landmark_full.tflite",
        "modules/pose_detection/pose_detection.tflite",
        "modules/hand_landmark/hand_landmark_full.tflite",
        "modules/holistic_landmark/hand_recrop.tflite",
        "modules/face_detection/face_detection_short_range.tflite",
        "modules/face_landmark/face_landmark.tflite",
    )
    missing = [name for name in required if not (package / name).is_file()]
    if missing:
        raise RuntimeError(f"Bundled MediaPipe resources are missing: {missing}")
    # Include every bundled graph and model, so graph-referenced resources cannot
    # silently change while the extraction cache remains valid.
    resources = sorted(path for path in (package / "modules").rglob("*")
                       if path.is_file() and path.suffix in {".tflite", ".binarypb"})
    return {
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "mediapipe": mp.__version__},
        "resources": {path.relative_to(package).as_posix(): sha256_file(path) for path in resources},
    }


def extractor_identity(config: ExtractionConfig | None = None) -> dict[str, Any]:
    config = config or ExtractionConfig()
    if not isinstance(config, ExtractionConfig):
        raise TypeError("config must be ExtractionConfig")
    runtime = _runtime_identity()
    return {
        "representation_id": REPRESENTATION_ID,
        "config": config.to_dict(),
        "versions": dict(runtime["versions"]),
        "resources": dict(runtime["resources"]),
        "extractor_code_hash": sha256_file(Path(__file__)),
        "holistic_options": {"static_image_mode": False, "smooth_landmarks": True,
                             "enable_segmentation": False, "smooth_segmentation": False,
                             "refine_face_landmarks": False, "min_detection_confidence": 0.5,
                             "min_tracking_confidence": 0.5},
        "coordinates": "aspect_corrected_xy_shoulder_midpoint_scale_z_zero",
        "point_order": "pose33_anatomical_left_hand21_anatomical_right_hand21",
        "temporal_policy": "uniform_selected_frames_after_full_decode",
        "tracking_reset": "before_each_clip",
    }


def decode_video(path: str | Path, config: ExtractionConfig | None = None) -> DecodedClip:
    """Decode every frame; fail if metadata and actual decoding disagree.

    POS_MSEC is preferred only when all decoded timestamps are finite and
    strictly increasing. The fallback uses the video's reported fps and records
    that choice explicitly; no default fps is fabricated.
    """
    import cv2
    import numpy as np

    config = config or ExtractionConfig()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoDecodeError(f"Cannot open video: {path}")
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(raw_count) or raw_count < 2 or abs(raw_count - round(raw_count)) > 0.01:
            raise VideoDecodeError("Video must declare a reliable frame count of at least two")
        expected_count = int(round(raw_count))
        if expected_count > 1_000_000:
            raise VideoDecodeError("Video frame count exceeds the clip extraction limit")
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        indices = np.rint(np.linspace(0, expected_count - 1, min(config.frames, expected_count))).astype(np.int64)
        selected = set(indices.tolist())
        images, decoded_timestamps = [], []
        count, source_shape = 0, None
        while True:
            success, image = capture.read()
            if not success:
                break
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise VideoDecodeError(f"Invalid decoded image at frame {count}")
            if source_shape is None:
                source_shape = tuple(image.shape)
            elif tuple(image.shape) != source_shape:
                raise VideoDecodeError("Video changes image dimensions during decoding")
            decoded_timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            if count in selected:
                if image.shape[1] > config.max_width:
                    height = max(1, int(round(image.shape[0] * config.max_width / image.shape[1])))
                    image = cv2.resize(image, (config.max_width, height), interpolation=cv2.INTER_AREA)
                images.append(image)
            count += 1
            if count > expected_count:
                raise VideoDecodeError(f"Decoded more frames than declared ({expected_count})")
        if count != expected_count:
            raise VideoDecodeError(f"Incomplete video decoding: expected {expected_count} frames, decoded {count}")
        all_times = np.asarray(decoded_timestamps, dtype=np.float64)
        if np.isfinite(all_times).all() and (all_times >= 0).all() and (np.diff(all_times) > 0).all():
            timestamp_source = "decoder_pos_msec"
        elif fps is not None:
            all_times = np.arange(count, dtype=np.float64) / fps
            timestamp_source = "frame_index_fps"
        else:
            raise VideoDecodeError("Video has neither reliable decoder timestamps nor a usable fps")
        timestamps = np.ascontiguousarray(all_times[indices], dtype=np.float64)
        timestamps.setflags(write=False)
        indices.setflags(write=False)
        return DecodedClip(tuple(images), timestamps, indices, {
            "source_frames": count,
            "sampled_frames": len(indices),
            "source_fps": fps,
            "source_width": source_shape[1],
            "source_height": source_shape[0],
            "extraction_width": images[0].shape[1],
            "extraction_height": images[0].shape[0],
            "timestamps_known": True,
            "timestamp_source": timestamp_source,
            "duration_seconds": float(all_times[-1] - all_times[0]),
            "duration_definition": "last_minus_first_decoded_timestamp",
            "full_decode_verified": True,
        })
    finally:
        capture.release()


def normalize_landmarks(results: Any, image_width: int, image_height: int,
                        pose_visibility: float = 0.5) -> tuple[Any, dict[str, bool]]:
    """Normalize Holistic results while preserving missing anatomical points."""
    import numpy as np

    values = np.full((75, 3), -999.0, dtype=np.float32)
    observed = {"anchor": False, "left_hand": False, "right_hand": False}
    aspect = image_width / image_height

    def valid(point: Any, check_visibility: bool = False) -> bool:
        x, y = float(point.x), float(point.y)
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            return False
        if check_visibility:
            visibility = float(point.visibility)
            return math.isfinite(visibility) and visibility >= pose_visibility
        return True

    pose_result = getattr(results, "pose_landmarks", None)
    if pose_result is None or len(pose_result.landmark) != 33:
        return values.reshape(-1), observed
    pose = pose_result.landmark
    if not valid(pose[11], True) or not valid(pose[12], True):
        return values.reshape(-1), observed
    left = np.asarray([pose[11].x * aspect, pose[11].y], dtype=np.float64)
    right = np.asarray([pose[12].x * aspect, pose[12].y], dtype=np.float64)
    scale = float(np.linalg.norm(left - right))
    if not math.isfinite(scale) or scale < 1e-6:
        return values.reshape(-1), observed
    center = (left + right) / 2
    observed["anchor"] = True

    def put(point: Any, index: int, check_visibility: bool = False) -> None:
        if valid(point, check_visibility):
            xy = (np.asarray([point.x * aspect, point.y], dtype=np.float64) - center) / scale
            values[index] = (xy[0], xy[1], 0.0)

    for index, point in enumerate(pose):
        put(point, index, True)
    for name, offset in (("left_hand", 33), ("right_hand", 54)):
        hand = getattr(results, f"{name}_landmarks", None)
        if hand is not None and len(hand.landmark) == 21:
            for index, point in enumerate(hand.landmark):
                put(point, offset + index)
            observed[name] = bool((values[offset:offset + 21, 0] != -999).any())
    return values.reshape(-1), observed


class HolisticExtractor:
    """Reuse the graph safely across clips, resetting all tracking each time."""

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()
        if not isinstance(self.config, ExtractionConfig):
            raise TypeError("config must be ExtractionConfig")
        self._holistic = None
        self._identity = None

    def __enter__(self) -> HolisticExtractor:
        import mediapipe as mp

        if self._holistic is not None:
            raise RuntimeError("Extractor is already open")
        self._identity = extractor_identity(self.config)
        self._holistic = mp.solutions.holistic.Holistic(
            model_complexity=self.config.model_complexity, **self._identity["holistic_options"])
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    def extract(self, path: str | Path) -> ExtractedClip:
        import cv2
        import numpy as np

        if self._holistic is None:
            raise RuntimeError("Use HolisticExtractor as a context manager")
        decoded = decode_video(path, self.config)
        self._holistic.reset()
        sequence, observation = [], []
        for image in decoded.images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.setflags(write=False)
            result = self._holistic.process(rgb)
            values, observed = normalize_landmarks(result, image.shape[1], image.shape[0], self.config.pose_visibility)
            sequence.append(values)
            observation.append(observed)
        values = np.ascontiguousarray(sequence, dtype=np.float32)
        quality = dict(decoded.quality)
        quality.update({
            "anchor_frame_fraction": sum(item["anchor"] for item in observation) / len(observation),
            "hand_frame_fraction": sum(item["left_hand"] or item["right_hand"] for item in observation) / len(observation),
            "left_hand_frame_fraction": sum(item["left_hand"] for item in observation) / len(observation),
            "right_hand_frame_fraction": sum(item["right_hand"] for item in observation) / len(observation),
            "valid_fraction": float((values != -999).mean()),
            "missing_policy": "unobserved_point_xyz_all_minus999",
        })
        if quality["anchor_frame_fraction"] < self.config.minimum_anchor_frame_fraction:
            raise InputRejected("insufficient_anchor_frames", quality)
        if quality["hand_frame_fraction"] < self.config.minimum_hand_frame_fraction:
            raise InputRejected("insufficient_hand_frames", quality)
        values.setflags(write=False)
        return ExtractedClip(values, decoded.timestamps, decoded.frame_indices, quality, self._identity)


def extract_video(path: str | Path, config: ExtractionConfig | None = None) -> ExtractedClip:
    with HolisticExtractor(config) as extractor:
        return extractor.extract(path)
