"""Versioned two-dimensional landmarks from one persistent extraction graph.

The stream contains observations, timing and missing-point masks. It does not
recognize gestures, assign word meanings or implement the older 225/321 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from copy import deepcopy
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from .segmentation import MediaPipeActivityObserver, SegmentConfig
from .sources import FramePacket


SCHEMA_ID = "pose_hands_xy75_stream_v1"
EXTRACTOR_VERSION = 1
POINT_COUNT = 75
TIME_DOMAINS = ("media_seconds", "application_monotonic")
_ZERO = (0.0, 0.0)
_TUPLE_FIELDS = {"image_size", "image_xy", "normalized_xy", "point_observed",
                 "normalized_observed", "pose_visibility"}


def _finite(value: Any, name: str, *, minimum: float = 0.0) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")


def _nonnegative_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _text(value: Any, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    """An immutable, strict observation. Coordinates contain no depth channel.

    Point order is pose 0..32, anatomical left hand 0..20, right hand 0..20.
    image_xy shares the input image's unmirrored [0, 1] coordinate system.
    normalized_xy uses shoulder centre/width, after correcting image aspect.
    Hand presence means a complete detector group exists; individual points
    may still be unobserved. Missing coordinates are exactly (0, 0).
    """

    schema_id: str
    session_id: str
    source_id: str
    source_epoch: int
    tracking_epoch: int
    frame_index: int
    timestamp: float
    time_domain: str
    received_monotonic: float
    extracted_monotonic: float
    image_size: tuple[int, int]
    image_xy: tuple[tuple[float, float], ...]
    normalized_xy: tuple[tuple[float, float], ...]
    point_observed: tuple[bool, ...]
    normalized_observed: tuple[bool, ...]
    anchor_valid: bool
    pose_visibility: tuple[float | None, ...]
    left_present: bool
    right_present: bool
    motion: float
    motion_valid: bool
    dt: float | None
    frames_skipped: int
    reset_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_id != SCHEMA_ID or type(self.schema_id) is not str:
            raise ValueError("Unsupported landmark schema_id")
        for name in ("session_id", "source_id"):
            _text(getattr(self, name), name)
        for name in ("source_epoch", "tracking_epoch", "frame_index", "frames_skipped"):
            _nonnegative_integer(getattr(self, name), name)
        if type(self.time_domain) is not str or self.time_domain not in TIME_DOMAINS:
            raise ValueError("Unsupported landmark time_domain")
        for name in ("timestamp", "received_monotonic", "extracted_monotonic", "motion"):
            _finite(getattr(self, name), name)
        if self.extracted_monotonic < self.received_monotonic:
            raise ValueError("extracted_monotonic must not precede received_monotonic")
        if self.dt is not None:
            _finite(self.dt, "dt")
            if self.dt <= 0:
                raise ValueError("dt must be positive or None")
        if self.reset_reason is not None:
            _text(self.reset_reason, "reset_reason")
        for name in ("anchor_valid", "left_present", "right_present", "motion_valid"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.motion_valid and self.dt is None:
            raise ValueError("valid motion requires dt")
        if not self.motion_valid and self.motion != 0:
            raise ValueError("invalid motion must be zero")
        if type(self.image_size) is not tuple or len(self.image_size) != 2 or any(
                type(value) is not int or value <= 0 for value in self.image_size):
            raise ValueError("image_size must be an immutable positive (width, height)")
        for name in ("point_observed", "normalized_observed"):
            mask = getattr(self, name)
            if type(mask) is not tuple or len(mask) != POINT_COUNT or any(
                    type(value) is not bool for value in mask):
                raise ValueError(f"{name} must contain exactly 75 immutable booleans")
        for name in ("image_xy", "normalized_xy"):
            points = getattr(self, name)
            if type(points) is not tuple or len(points) != POINT_COUNT:
                raise ValueError(f"{name} must contain exactly 75 immutable coordinate pairs")
            mask = self.point_observed if name == "image_xy" else self.normalized_observed
            for index, point in enumerate(points):
                if type(point) is not tuple or len(point) != 2 or any(
                        type(value) not in (int, float) or not math.isfinite(value)
                        for value in point):
                    raise ValueError(f"{name}[{index}] must be an immutable finite xy pair")
                if not mask[index] and point != _ZERO:
                    raise ValueError(f"unobserved {name}[{index}] must be zero")
                if name == "image_xy" and mask[index] and any(
                        value < 0 or value > 1 for value in point):
                    raise ValueError("observed image_xy must lie in [0, 1]")
        if type(self.pose_visibility) is not tuple or len(self.pose_visibility) != 33:
            raise ValueError("pose_visibility must contain exactly 33 immutable values")
        for index, value in enumerate(self.pose_visibility):
            if value is not None:
                _finite(value, f"pose_visibility[{index}]")
                if value > 1:
                    raise ValueError("pose_visibility must lie in [0, 1] or be None")
            elif self.point_observed[index]:
                raise ValueError("an observed pose point requires a visibility value")
        expected = tuple(value and self.anchor_valid for value in self.point_observed)
        if self.normalized_observed != expected:
            raise ValueError("normalized_observed must agree with point mask and anchor_valid")
        if self.motion_valid and (not any(self.normalized_observed[33:]) or self.reset_reason is not None):
            raise ValueError("valid motion requires observed hand points and uninterrupted tracking")
        for present, start in ((self.left_present, 33), (self.right_present, 54)):
            if not present and any(self.point_observed[start:start + 21]):
                raise ValueError("an absent hand cannot contain observed points")
        if self.anchor_valid:
            if not self.point_observed[11] or not self.point_observed[12]:
                raise ValueError("valid anchors require both observed shoulders")
            aspect = self.image_size[0] / self.image_size[1]
            left, right = self.image_xy[11], self.image_xy[12]
            scale = math.hypot((left[0] - right[0]) * aspect, left[1] - right[1])
            if scale <= 1e-6:
                raise ValueError("valid anchors require distinct shoulders")

    def to_dict(self) -> dict[str, Any]:
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in _TUPLE_FIELDS:
            value = payload[name]
            payload[name] = [list(point) for point in value] if name in (
                "image_xy", "normalized_xy") else list(value)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LandmarkFrame:
        if type(payload) is not dict:
            raise ValueError("a landmark frame must be a JSON object")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("landmark fields are missing or unsupported")
        owned = dict(payload)
        for name in _TUPLE_FIELDS:
            value = owned[name]
            if type(value) is not list:
                raise ValueError(f"{name} must be a JSON array")
            if name in ("image_xy", "normalized_xy"):
                if any(type(point) is not list for point in value):
                    raise ValueError(f"{name} must contain JSON coordinate arrays")
                owned[name] = tuple(tuple(point) for point in value)
            else:
                owned[name] = tuple(value)
        return cls(**owned)


class KeypointExtractor:
    """Extract each accepted packet exactly once, preserving its capture time.

    reset() restarts tracking, not the externally managed source epoch. Manual
    resets clear the local sequence baseline; recording/session owners retain
    responsibility for source ordering across such resets. Automatic gap and
    size resets preserve the baseline and report the measured source dt.
    """

    def __init__(self, config: SegmentConfig | None = None, session_id: str | None = None,
                 source_epoch: int = 0, time_domain: str = "media_seconds",
                 holistic_factory: Callable[..., Any] | None = None) -> None:
        self.config = SegmentConfig() if config is None else config
        if not isinstance(self.config, SegmentConfig):
            raise TypeError("config must be SegmentConfig")
        self.session_id = uuid4().hex if session_id is None else session_id
        _text(self.session_id, "session_id")
        _nonnegative_integer(source_epoch, "source_epoch")
        if type(time_domain) is not str or time_domain not in TIME_DOMAINS:
            raise ValueError("Unsupported landmark time_domain")
        if holistic_factory is not None and not callable(holistic_factory):
            raise TypeError("holistic_factory must be callable or None")
        self.source_epoch = source_epoch
        self.time_domain = time_domain
        self._factory = holistic_factory
        self._tracking_epoch = 0
        self._observer = self._new_observer()
        self._previous: LandmarkFrame | None = None
        self._previous_packet: tuple[str, float, int, tuple[int, int]] | None = None
        self._pending_reset_reason: str | None = "initial"
        self._closed = False
        self._protocol = None

    def _new_observer(self) -> MediaPipeActivityObserver:
        return MediaPipeActivityObserver(self.config, holistic_factory=self._factory)

    def _reset_tracking(self, reason: str, *, clear_baseline: bool) -> None:
        self._observer.close()
        self._observer = self._new_observer()
        self._tracking_epoch += 1
        self._previous = None
        if clear_baseline:
            self._previous_packet = None
        self._pending_reset_reason = reason

    def reset(self, reason: str = "manual") -> None:
        if self._closed:
            raise RuntimeError("keypoint extractor is closed")
        _text(reason, "reset reason")
        self._reset_tracking(reason, clear_baseline=True)

    @staticmethod
    def _group(result: Any, name: str, count: int) -> tuple[Any, ...]:
        group = getattr(result, name, None)
        values = getattr(group, "landmark", None)
        if values is None:
            return ()
        try:
            return tuple(values) if len(values) == count else ()
        except TypeError:
            return ()

    @staticmethod
    def _xy(point: Any) -> tuple[float, float] | None:
        try:
            x, y = float(point.x), float(point.y)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return x, y

    @staticmethod
    def _visibility(point: Any) -> float | None:
        try:
            value = float(point.visibility)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and 0 <= value <= 1 else None

    def _coordinates(self, result: Any, image_size: tuple[int, int]) -> dict[str, Any]:
        pose = self._group(result, "pose_landmarks", 33)
        left = self._group(result, "left_hand_landmarks", 21)
        right = self._group(result, "right_hand_landmarks", 21)
        visibility = tuple(self._visibility(point) for point in pose) if pose else (None,) * 33
        points: list[tuple[float, float]] = []
        mask: list[bool] = []
        for group, offset, count in ((pose, 0, 33), (left, 33, 21), (right, 54, 21)):
            for index in range(count):
                point = self._xy(group[index]) if group else None
                if offset == 0 and (visibility[index] is None or
                                    visibility[index] < self.config.pose_visibility):
                    point = None
                points.append(point if point is not None else _ZERO)
                mask.append(point is not None)
        aspect = image_size[0] / image_size[1]
        shoulders = [(points[index][0] * aspect, points[index][1]) for index in (11, 12)]
        scale = math.dist(*shoulders) if mask[11] and mask[12] else 0.0
        anchor_valid = bool(scale > 1e-6)
        normalized = [_ZERO] * POINT_COUNT
        if anchor_valid:
            cx, cy = tuple((a + b) / 2 for a, b in zip(*shoulders))
            normalized = [((point[0] * aspect - cx) / scale, (point[1] - cy) / scale)
                          if observed else _ZERO for point, observed in zip(points, mask)]
        return {"image_xy": tuple(points), "normalized_xy": tuple(normalized),
                "point_observed": tuple(mask),
                "normalized_observed": tuple(observed and anchor_valid for observed in mask),
                "anchor_valid": anchor_valid, "pose_visibility": visibility,
                "left_present": bool(left), "right_present": bool(right)}

    def process(self, packet: FramePacket, received_monotonic: float | None = None) -> LandmarkFrame:
        if self._closed:
            raise RuntimeError("keypoint extractor is closed")
        if not isinstance(packet, FramePacket):
            raise TypeError("process requires FramePacket")
        packet.__post_init__()
        _text(packet.source_id, "source_id")
        now = time.perf_counter()
        received = now if received_monotonic is None else received_monotonic
        _finite(received, "received_monotonic")
        if received > now:
            raise ValueError("received_monotonic cannot be in the future")
        image_size = (packet.image.shape[1], packet.image.shape[0])
        previous_packet = self._previous_packet
        dt = None
        skipped = 0
        if previous_packet is not None:
            source_id, timestamp, index, size = previous_packet
            if packet.source_id != source_id:
                raise ValueError("source changed; use a new extractor or explicitly reset first")
            if packet.timestamp <= timestamp or packet.frame_index <= index:
                raise ValueError("timestamps and frame indices must be strictly increasing")
            dt = packet.timestamp - timestamp
            skipped = packet.frame_index - index - 1
            reason = "image_size_changed" if image_size != size else (
                "timestamp_gap" if dt > self.config.max_observation_gap_seconds else None)
            if reason is not None:
                self._reset_tracking(reason, clear_baseline=False)
        try:
            _, result = self._observer.observe_landmarks(packet)
            coordinates = self._coordinates(result, image_size)
            previous = self._previous
            motion = 0.0
            motion_valid = False
            if previous is not None and dt is not None:
                changes = [math.dist(coordinates["normalized_xy"][index], previous.normalized_xy[index])
                           for index in range(33, POINT_COUNT)
                           if coordinates["normalized_observed"][index]
                           and previous.normalized_observed[index]]
                if changes:
                    motion = sum(changes) / len(changes) / dt
                    motion_valid = math.isfinite(motion)
                    if not motion_valid:
                        motion = 0.0
            frame = LandmarkFrame(
                schema_id=SCHEMA_ID, session_id=self.session_id, source_id=packet.source_id,
                source_epoch=self.source_epoch, tracking_epoch=self._tracking_epoch,
                frame_index=packet.frame_index, timestamp=packet.timestamp, time_domain=self.time_domain,
                received_monotonic=received, extracted_monotonic=time.perf_counter(),
                image_size=image_size, motion=motion, motion_valid=motion_valid,
                dt=dt, frames_skipped=skipped, reset_reason=self._pending_reset_reason, **coordinates)
        except Exception:
            # An extraction failure may have mutated the native graph. Reopen
            # it before retrying, but retain the last accepted source identity.
            self._reset_tracking("extraction_error", clear_baseline=False)
            raise
        self._previous = frame
        self._previous_packet = (packet.source_id, packet.timestamp, packet.frame_index, image_size)
        self._pending_reset_reason = None
        return frame

    def metadata(self) -> dict[str, Any]:
        if self._protocol is None:
            self._protocol = self._build_protocol()
        protocol = deepcopy(self._protocol)
        encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return {"schema_id": SCHEMA_ID, "extractor_version": EXTRACTOR_VERSION,
                "point_count": POINT_COUNT, "coordinate_dimensions": 2,
                "session_id": self.session_id,
                "source_id": self._previous_packet[0] if self._previous_packet is not None else None,
                "source_epoch": self.source_epoch, "tracking_epoch": self._tracking_epoch,
                "time_domain": self.time_domain, "protocol": protocol,
                "protocol_hash": hashlib.sha256(encoded).hexdigest()}

    def _build_protocol(self) -> dict[str, Any]:
        try:
            version = package_metadata.version("mediapipe")
        except package_metadata.PackageNotFoundError:
            version = None
        protocol = {
            "schema_id": SCHEMA_ID, "extractor_version": EXTRACTOR_VERSION,
            "point_count": POINT_COUNT, "coordinate_dimensions": 2,
            "point_order": [{"group": "pose", "start": 0, "count": 33},
                            {"group": "left_hand", "start": 33, "count": 21},
                            {"group": "right_hand", "start": 54, "count": 21}],
            "image_coordinates": "unmirrored common image xy normalized by width and height",
            "body_normalization": "aspect-corrected shoulder centre and shoulder width; no depth",
            "missing_policy": "zero xy plus explicit per-point masks; no imputation",
            "quality": {"image_xy_range": [0, 1], "pose_visibility": self.config.pose_visibility,
                        "shoulder_min_width": 1e-6,
                        "hand_presence": "detector returned a structurally complete 21-point group",
                        "normalized_mask": "point_observed AND anchor_valid",
                        "motion": "mean common observed hand-point displacement per source second"},
            "tracking": {"backend": "mediapipe.solutions.holistic", "backend_version": version,
                         "factory_injected": self._factory is not None,
                         "model_complexity": 1, "static_image_mode": False, "smooth_landmarks": True,
                         "enable_segmentation": False, "smooth_segmentation": False,
                         "refine_face_landmarks": False, "min_detection_confidence": 0.5,
                         "min_tracking_confidence": 0.5,
                         "max_observation_gap_seconds": self.config.max_observation_gap_seconds,
                         "reset_on_image_size_change": True},
            "sampling": {"resample": False, "max_width": self.config.max_width,
                         "timestamps": "preserve source capture timestamps",
                         "dt": "source delta; None after manual reset; preserved on automatic reset",
                         "frames_skipped": "source frame-index gap, excluding the current frame"},
            "processing_clock": "time.perf_counter seconds",
            "implementation": {name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("keypoints.py", "segmentation.py")},
            "compatibility": "new stream protocol; not interchangeable with 225/321 model features",
        }
        return protocol

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._observer.close()
            self._previous = None
            self._previous_packet = None

    def __enter__(self) -> KeypointExtractor:
        if self._closed:
            raise RuntimeError("keypoint extractor is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
