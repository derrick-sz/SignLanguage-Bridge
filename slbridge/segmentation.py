"""Timestamp based action boundaries, independent of any clip classifier.

Activity is an observation heuristic, not a prediction or unknown-sign gate.
Every completed action still goes through the model's existing clip contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable
from uuid import uuid4

from .sources import CapturedClip, FramePacket


@dataclass(frozen=True)
class SegmentConfig:
    config_version: int = 1
    start_candidate_seconds: float = 0.12
    end_candidate_seconds: float = 0.65
    pre_buffer_seconds: float = 0.25
    post_buffer_seconds: float = 0.10
    min_segment_seconds: float = 0.25
    max_segment_seconds: float = 8.0
    quality_gap_seconds: float = 0.20
    max_observation_gap_seconds: float = 0.50
    rearm_seconds: float = 0.20
    start_motion_threshold: float = 0.35
    end_motion_threshold: float = 0.15
    rearm_motion_threshold: float = 0.35
    max_buffer_seconds: float = 10.0
    max_frames: int = 600
    max_bytes: int = 128 * 1024 * 1024
    max_width: int = 640
    pose_visibility: float = 0.5
    static_active_y: float = 0.85
    static_active_x: float = 1.8

    def __post_init__(self) -> None:
        if type(self.config_version) is not int or self.config_version != 1:
            raise ValueError("Unsupported segmentation config_version")
        positive = ("start_candidate_seconds", "end_candidate_seconds",
                    "min_segment_seconds", "max_segment_seconds", "quality_gap_seconds",
                    "max_observation_gap_seconds", "rearm_seconds", "start_motion_threshold",
                    "rearm_motion_threshold", "max_buffer_seconds", "static_active_x")
        nonnegative = ("pre_buffer_seconds", "post_buffer_seconds", "end_motion_threshold")
        for name in positive + nonnegative + ("pose_visibility", "static_active_y"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in positive and value <= 0 or name in nonnegative and value < 0:
                raise ValueError(f"{name} has an invalid range")
        for name in ("max_frames", "max_bytes", "max_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_frames < 2:
            raise ValueError("max_frames must allow at least two frames")
        if not 0 < self.pose_visibility <= 1:
            raise ValueError("pose_visibility must lie in (0, 1]")
        if self.min_segment_seconds >= self.max_segment_seconds:
            raise ValueError("min_segment_seconds must be below max_segment_seconds")
        if self.end_motion_threshold >= min(self.start_motion_threshold, self.rearm_motion_threshold):
            raise ValueError("end motion threshold must be below start and rearm thresholds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivityObservation:
    timestamp: float
    hands_visible: bool
    quality_ok: bool
    motion: float
    pose_active: bool = False
    left_hand_visible: bool | None = None
    right_hand_visible: bool | None = None

    def __post_init__(self) -> None:
        for name in ("timestamp", "motion"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("hands_visible", "quality_ok", "pose_active"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("left_hand_visible", "right_hand_visible"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean or None")
        if self.left_hand_visible is not None and self.right_hand_visible is not None:
            if self.hands_visible != (self.left_hand_visible or self.right_hand_visible):
                raise ValueError("hands_visible must agree with left/right hand visibility")


@dataclass(frozen=True)
class AutoSegment:
    segment_id: str
    clip: CapturedClip
    start_time: float
    end_time: float
    action_end_time: float
    source_id: str
    sequence: int


@dataclass(frozen=True)
class SegmentNotice:
    kind: str
    state: str
    reason: str
    segment: AutoSegment | None = None


@dataclass(frozen=True)
class _FrameIdentity:
    timestamp: float
    frame_index: int
    source_id: str
    shape: tuple[int, ...]

    @classmethod
    def of(cls, packet: FramePacket) -> _FrameIdentity:
        return cls(packet.timestamp, packet.frame_index, packet.source_id, packet.image.shape)


def _elapsed(now: float, began: float, duration: float) -> bool:
    return now - began + 1e-9 >= duration


class AutoSegmenter:
    """Collect a whole action; emit it only after a visible, good-quality pause.

    Feed one source in strictly increasing timestamp/frame-index order. Images
    are copied into a bounded ring. Active clips are rejected instead of being
    truncated when a bound is reached. reset() starts a new source timeline.
    """

    def __init__(self, config: SegmentConfig | None = None) -> None:
        self.config = config or SegmentConfig()
        if not isinstance(self.config, SegmentConfig):
            raise TypeError("config must be SegmentConfig")
        self._instance_id = uuid4().hex
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        # Keep sequence increasing across reset so stale callbacks cannot reuse
        # a completed segment ID within this segmenter's session.
        self.state = "WAITING"
        self._frames: deque[FramePacket] = deque()
        self._bytes = 0
        self._previous: _FrameIdentity | None = None
        self._action_start: float | None = None
        self._end_start: float | None = None
        self._quality_start: float | None = None
        self._rearm_start: float | None = None
        self._rearm_mode: str | None = None
        self._motion_rearm_allowed = True
        self._expected_hands: set[str] = set()
        self._had_motion = False

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    @property
    def buffered_bytes(self) -> int:
        return self._bytes

    @property
    def buffered_seconds(self) -> float:
        return self._frames[-1].timestamp - self._frames[0].timestamp if self._frames else 0.0

    def _state(self, state: str, reason: str, notices: list[SegmentNotice]) -> None:
        if self.state != state:
            self.state = state
            notices.append(SegmentNotice("state", state, reason))

    def _clear_frames(self) -> None:
        self._frames.clear()
        self._bytes = 0

    def _drop_before(self, timestamp: float) -> None:
        while self._frames and self._frames[0].timestamp < timestamp:
            self._bytes -= self._frames.popleft().image.nbytes

    def _hand_sides(self, observation: ActivityObservation) -> set[str]:
        return {side for side in ("left", "right")
                if getattr(observation, f"{side}_hand_visible") is True}

    def _usable(self, observation: ActivityObservation) -> bool:
        return (observation.quality_ok and observation.hands_visible
                and all(getattr(observation, f"{side}_hand_visible") is not False
                        for side in self._expected_hands))

    def _reject(self, reason: str, notices: list[SegmentNotice]) -> None:
        self._clear_frames()
        self._action_start = self._end_start = self._quality_start = self._rearm_start = None
        self._rearm_mode = None
        self._expected_hands.clear()
        self._motion_rearm_allowed = False
        self._state("REARM", reason, notices)
        notices.append(SegmentNotice("rejected", self.state, reason))

    def _append(self, packet: FramePacket, notices: list[SegmentNotice]) -> bool:
        import cv2

        image = packet.image
        if image.shape[1] > self.config.max_width:
            height = max(1, round(image.shape[0] * self.config.max_width / image.shape[1]))
            image = cv2.resize(image, (self.config.max_width, height), interpolation=cv2.INTER_AREA)
        protected = self.state in {"START_CANDIDATE", "CAPTURING", "END_CANDIDATE"}
        if protected:
            self._drop_before(self._action_start - self.config.pre_buffer_seconds)
        else:
            self._drop_before(packet.timestamp - self.config.max_buffer_seconds)
        if not protected:
            while self._frames and (len(self._frames) + 1 > self.config.max_frames
                                   or self._bytes + image.nbytes > self.config.max_bytes):
                self._bytes -= self._frames.popleft().image.nbytes
        reason = None
        if image.nbytes + self._bytes > self.config.max_bytes:
            reason = "buffer_byte_limit"
        elif len(self._frames) + 1 > self.config.max_frames:
            reason = "buffer_frame_limit"
        elif self._frames and packet.timestamp - self._frames[0].timestamp > self.config.max_buffer_seconds + 1e-9:
            reason = "buffer_time_limit"
        if reason is not None:
            self._reject(reason, notices)
            return False
        owned = image.copy()
        owned.setflags(write=False)
        self._frames.append(replace(packet, image=owned))
        self._bytes += owned.nbytes
        return True

    def _begin(self, timestamp: float, observation: ActivityObservation,
               notices: list[SegmentNotice]) -> None:
        self._action_start = timestamp
        self._expected_hands = self._hand_sides(observation)
        self._had_motion = observation.motion > self.config.end_motion_threshold
        self._quality_start = self._end_start = None
        self._state("START_CANDIDATE", "activity_candidate", notices)

    def _finish(self, notices: list[SegmentNotice]) -> None:
        action_end = self._end_start
        selected = tuple(frame for frame in self._frames
                         if frame.timestamp <= action_end + self.config.post_buffer_seconds + 1e-9)
        if len(selected) < 2:
            self._reject("insufficient_frames", notices)
            return
        if not _elapsed(action_end, self._action_start, self.config.min_segment_seconds):
            self._reject("segment_too_short", notices)
            return
        origin = selected[0].timestamp
        clip = CapturedClip(tuple(replace(frame, timestamp=frame.timestamp - origin)
                                  for frame in selected), selected[0].source_id)
        self._sequence += 1
        segment = AutoSegment(f"segment:{self._instance_id}:{self._sequence}", clip, origin,
                              selected[-1].timestamp, action_end, clip.source_id, self._sequence)
        self._clear_frames()
        self._action_start = self._end_start = self._quality_start = self._rearm_start = None
        self._rearm_mode = None
        self._expected_hands.clear()
        self._motion_rearm_allowed = True
        self._state("REARM", "end_confirmed", notices)
        notices.append(SegmentNotice("ready", self.state, "end_confirmed", segment))

    def feed(self, packet: FramePacket, observation: ActivityObservation) -> tuple[SegmentNotice, ...]:
        if not isinstance(packet, FramePacket) or not isinstance(observation, ActivityObservation):
            raise TypeError("feed requires FramePacket and ActivityObservation")
        packet.__post_init__()
        observation.__post_init__()
        if observation.timestamp != packet.timestamp:
            raise ValueError("observation timestamp must match the frame timestamp")
        previous = self._previous
        if previous is not None:
            if packet.source_id != previous.source_id:
                raise ValueError("source changed; reset the segmenter first")
            if packet.timestamp <= previous.timestamp or packet.frame_index <= previous.frame_index:
                raise ValueError("timestamps and frame indices must be strictly increasing")
            if packet.image.shape != previous.shape:
                raise ValueError("source image dimensions changed; reset the segmenter first")
        self._previous = _FrameIdentity.of(packet)
        notices: list[SegmentNotice] = []
        timestamp = packet.timestamp
        if previous is not None and timestamp - previous.timestamp > self.config.max_observation_gap_seconds:
            if self.state in {"CAPTURING", "END_CANDIDATE"}:
                self._reject("timestamp_gap", notices)
            elif self.state == "START_CANDIDATE":
                self._action_start = None
                self._expected_hands.clear()
                self._clear_frames()
                self._state("WAITING", "timestamp_gap", notices)
            self._rearm_start = self._rearm_mode = None
        usable = self._usable(observation)
        starting = usable and (observation.pose_active or observation.motion >= self.config.start_motion_threshold)

        if self.state == "REARM":
            release = (not observation.hands_visible or
                       observation.quality_ok and not observation.pose_active
                       and observation.motion <= self.config.end_motion_threshold)
            renewed = (self._motion_rearm_allowed and usable
                       and observation.motion >= self.config.rearm_motion_threshold)
            mode = "motion" if renewed else "release" if release else None
            if mode is None:
                self._rearm_start = self._rearm_mode = None
            elif mode != self._rearm_mode:
                self._rearm_start, self._rearm_mode = timestamp, mode
            elif _elapsed(timestamp, self._rearm_start, self.config.rearm_seconds):
                began = self._rearm_start
                self._rearm_start = self._rearm_mode = None
                if renewed:
                    self._begin(began, observation, notices)
                else:
                    self._state("WAITING", "released", notices)
            if self.state == "REARM":
                self._append(packet, notices)
                return tuple(notices)

        if self.state == "WAITING" and starting:
            self._begin(timestamp, observation, notices)
        elif self.state == "START_CANDIDATE":
            if not starting:
                self._action_start = None
                self._expected_hands.clear()
                self._state("WAITING", "start_cancelled", notices)
            elif _elapsed(timestamp, self._action_start, self.config.start_candidate_seconds):
                self._state("CAPTURING", "start_confirmed", notices)

        if self.state in {"START_CANDIDATE", "CAPTURING", "END_CANDIDATE"} and usable:
            self._had_motion = self._had_motion or observation.motion > self.config.end_motion_threshold

        if not self._append(packet, notices):
            return tuple(notices)
        if self.state not in {"CAPTURING", "END_CANDIDATE"}:
            return tuple(notices)

        if self._quality_start is not None and _elapsed(timestamp, self._quality_start,
                                                       self.config.quality_gap_seconds):
            self._reject("quality_lost", notices)
            return tuple(notices)
        if not usable:
            if self._quality_start is None:
                self._quality_start = timestamp
            self._end_start = None
            self._state("CAPTURING", "quality_interrupted_end", notices)
            if _elapsed(timestamp, self._quality_start, self.config.quality_gap_seconds):
                self._reject("quality_lost", notices)
            elif timestamp - self._action_start > self.config.max_segment_seconds + 1e-9:
                self._reject("segment_too_long", notices)
            return tuple(notices)
        self._quality_start = None
        self._expected_hands.update(self._hand_sides(observation))
        if observation.motion > self.config.end_motion_threshold:
            self._end_start = None
            self._state("CAPTURING", "activity_resumed", notices)
        elif self._end_start is None and (self._had_motion or _elapsed(
                timestamp, self._action_start, self.config.min_segment_seconds)):
            self._end_start = timestamp
            self._state("END_CANDIDATE", "pause_candidate", notices)

        action_duration = (self._end_start if self._end_start is not None else timestamp) - self._action_start
        if action_duration > self.config.max_segment_seconds + 1e-9:
            self._reject("segment_too_long", notices)
        elif self._end_start is not None and _elapsed(timestamp, self._end_start, max(
                self.config.end_candidate_seconds, self.config.post_buffer_seconds)):
            self._finish(notices)
        return tuple(notices)

    def flush(self) -> tuple[SegmentNotice, ...]:
        """EOF never supplies the missing evidence of a finished action."""
        notices: list[SegmentNotice] = []
        if self.state in {"START_CANDIDATE", "CAPTURING", "END_CANDIDATE"}:
            self._reject("incomplete_at_eof", notices)
        self._clear_frames()
        return tuple(notices)


class MediaPipeActivityObserver:
    """Own a separate Holistic graph for per-frame activity observations.

    Motion is mean hand-landmark displacement per timestamp second in shoulder
    widths. A raised/in-front hand may be active without any motion. The
    position thresholds are unvalidated heuristics, not a sign vocabulary.
    """

    def __init__(self, config: SegmentConfig | None = None, *,
                 holistic_factory: Callable[..., Any] | None = None) -> None:
        self.config = config or SegmentConfig()
        if not isinstance(self.config, SegmentConfig):
            raise TypeError("config must be SegmentConfig")
        self._factory = holistic_factory
        self._graph = None
        self._closed = False
        self._previous_packet: _FrameIdentity | None = None
        self._previous_hands: dict[str, Any] = {}

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("activity observer is closed")
        if self._graph is None:
            if self._factory is None:
                import mediapipe as mp

                factory = mp.solutions.holistic.Holistic
            else:
                factory = self._factory
            self._graph = factory(model_complexity=1, static_image_mode=False,
                                  smooth_landmarks=True, enable_segmentation=False,
                                  smooth_segmentation=False, refine_face_landmarks=False,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def observe(self, packet: FramePacket) -> ActivityObservation:
        return self.observe_landmarks(packet)[0]

    def observe_landmarks(self, packet: FramePacket) -> tuple[ActivityObservation, Any]:
        """Return activity and its raw landmarks from one persistent graph call.

        This is an extraction outlet, not a new activity-quality contract.
        Callers must consume the result before the next graph observation.
        """
        import cv2
        import numpy as np

        if not isinstance(packet, FramePacket):
            raise TypeError("observe requires FramePacket")
        packet.__post_init__()
        previous = self._previous_packet
        if previous is not None and (packet.source_id != previous.source_id
                                     or packet.timestamp <= previous.timestamp
                                     or packet.frame_index <= previous.frame_index):
            raise ValueError("observer requires one source with strictly increasing timestamps and indices")
        self._open()
        image = packet.image
        if image.shape[1] > self.config.max_width:
            height = max(1, round(image.shape[0] * self.config.max_width / image.shape[1]))
            image = cv2.resize(image, (self.config.max_width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.setflags(write=False)
        result = self._graph.process(rgb)
        aspect = image.shape[1] / image.shape[0]

        def xy(point: Any, *, visibility: bool = False):
            x, y = float(point.x), float(point.y)
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                return None
            if visibility:
                value = float(point.visibility)
                if not math.isfinite(value) or value < self.config.pose_visibility:
                    return None
            return np.asarray([x * aspect, y], dtype=np.float64)

        pose = getattr(result, "pose_landmarks", None)
        anchors = [xy(pose.landmark[i], visibility=True) for i in (11, 12)] if (
            pose is not None and len(pose.landmark) == 33) else [None, None]
        anchor_ok = all(value is not None for value in anchors)
        scale = float(np.linalg.norm(anchors[0] - anchors[1])) if anchor_ok else 0.0
        anchor_ok = anchor_ok and scale > 1e-6
        center = (anchors[0] + anchors[1]) / 2 if anchor_ok else None
        hands: dict[str, Any] = {}
        visible: dict[str, bool] = {}
        invalid_hand = False
        for side in ("left", "right"):
            group = getattr(result, f"{side}_hand_landmarks", None)
            points = [xy(point) for point in group.landmark] if (
                group is not None and len(group.landmark) == 21) else []
            visible[side] = len(points) == 21 and all(value is not None for value in points)
            invalid_hand = invalid_hand or group is not None and not visible[side]
            if anchor_ok and visible[side]:
                hands[side] = (np.asarray(points) - center) / scale
        hands_visible = any(visible.values())
        quality = bool(anchor_ok and hands_visible and not invalid_hand)
        motion = 0.0
        if quality and previous is not None:
            elapsed = packet.timestamp - previous.timestamp
            if elapsed <= self.config.max_observation_gap_seconds:
                changes = [np.linalg.norm(points - self._previous_hands[side], axis=1)
                           for side, points in hands.items() if side in self._previous_hands]
                if changes:
                    motion = float(np.concatenate(changes).mean() / elapsed)
        active = quality and any(points[0, 1] <= self.config.static_active_y
                                 and abs(points[0, 0]) <= self.config.static_active_x
                                 for points in hands.values())
        self._previous_packet = _FrameIdentity.of(packet)
        self._previous_hands = hands if quality else {}
        observation = ActivityObservation(packet.timestamp, bool(hands_visible), quality, motion,
                                          bool(active), visible["left"], visible["right"])
        return observation, result

    def close(self) -> None:
        graph, self._graph = self._graph, None
        self._closed = True
        self._previous_packet = None
        self._previous_hands.clear()
        if graph is not None:
            graph.close()

    def __enter__(self) -> MediaPipeActivityObserver:
        self._open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
