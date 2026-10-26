"""Bounded explicit RGB windows; rolling inference is not native model streaming."""
from collections import deque
from dataclasses import dataclass, replace
import math
import time

from .sources import CapturedClip, FramePacket


@dataclass(frozen=True)
class ContinuousConfig:
    minimum_seconds: float = 1.2
    window_seconds: float = 2.4
    update_interval_seconds: float = .5
    max_result_age_seconds: float = 3.
    max_observation_gap_seconds: float = .5
    max_frames: int = 180
    max_bytes: int = 96 * 1024 * 1024
    max_width: int = 640
    minimum_frames: int | None = None
    # None = auto: file replay does not require the MediaPipe hands/quality gate
    # (the footage is known to contain signing), live capture keeps it.
    # True = always require, False = never require.
    require_hands_visible: bool | None = None

    def __post_init__(self):
        for name in ('minimum_seconds', 'window_seconds', 'update_interval_seconds',
                     'max_result_age_seconds', 'max_observation_gap_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.minimum_seconds > self.window_seconds or self.window_seconds > 8:
            raise ValueError('Continuous window must fit the minimum and 8-second clip bound')
        for name in ('max_frames', 'max_bytes', 'max_width'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not 2 <= self.max_frames <= 600 or self.max_bytes > 128 * 1024 * 1024:
            raise ValueError('Continuous frame/byte limits exceed the clip contract')
        if self.minimum_frames is not None and (type(self.minimum_frames) is not int
                or not 2 <= self.minimum_frames <= self.max_frames):
            raise ValueError('minimum_frames must be an integer between 2 and max_frames')
        if self.require_hands_visible is not None and not isinstance(self.require_hands_visible, bool):
            raise ValueError('require_hands_visible must be a boolean or None')


@dataclass(frozen=True)
class WindowSnapshot:
    clip: CapturedClip
    start_time: float
    end_time: float
    received_at: float


@dataclass(frozen=True)
class ContinuousUpdate:
    response: object
    start_time: float
    end_time: float
    completed_at: float
    age_seconds: float


class RollingWindow:
    def __init__(self, config: ContinuousConfig | None = None):
        self.config = config or ContinuousConfig()
        self.dropped_frames = 0
        self.clear()

    def clear(self):
        self._frames = deque()
        self._bytes = 0
        self._previous = None
        self._received_at = None

    @property
    def buffered_frames(self):
        return len(self._frames)

    @property
    def buffered_bytes(self):
        return self._bytes

    @property
    def latest_time(self):
        return self._frames[-1].timestamp if self._frames else None

    def append(self, packet: FramePacket):
        packet.__post_init__()
        identity = (packet.source_id, packet.frame_index, packet.timestamp, packet.image.shape)
        if self._previous is not None:
            source, index, stamp, shape = self._previous
            if (packet.source_id != source or packet.frame_index <= index
                    or packet.timestamp <= stamp or packet.image.shape != shape):
                raise ValueError('Rolling window requires one ordered, consistent source')
            if packet.timestamp - stamp > self.config.max_observation_gap_seconds:
                self.clear()
        self._previous = identity
        image = packet.image
        if image.shape[1] > self.config.max_width:
            import cv2
            image = cv2.resize(image, (self.config.max_width,
                max(1, round(image.shape[0] * self.config.max_width / image.shape[1]))),
                interpolation=cv2.INTER_AREA)
        image = image.copy()
        image.setflags(write=False)
        self._frames.append(replace(packet, image=image))
        self._bytes += image.nbytes
        self._received_at = time.perf_counter()
        while self._frames and (len(self._frames) > self.config.max_frames
                or self._bytes > self.config.max_bytes
                or packet.timestamp-self._frames[0].timestamp > self.config.window_seconds+1e-9):
            self._bytes -= self._frames.popleft().image.nbytes
            self.dropped_frames += 1

    def snapshot(self):
        if len(self._frames) < 2:
            return None
        start, end = self._frames[0].timestamp, self._frames[-1].timestamp
        if self.config.minimum_frames is not None:
            if len(self._frames) < self.config.minimum_frames:
                return None
        elif end-start+1e-9 < self.config.minimum_seconds:
            return None
        clip = CapturedClip(tuple(replace(frame, timestamp=frame.timestamp-start) for frame in self._frames),
                            self._frames[0].source_id)
        return WindowSnapshot(clip, start, end, self._received_at)

