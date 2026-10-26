"""Unmirrored BGR frame sources and bounded, complete action recordings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from array import array
import math
from pathlib import Path
import time
from typing import Any, Callable


class SourceError(ValueError):
    """A source could not open, decode a frame, or provide valid timing."""


class ClipLimitError(ValueError):
    """A whole recording was aborted because a configured limit was exceeded."""


@dataclass(frozen=True)
class FramePacket:
    image: Any
    timestamp: float
    frame_index: int
    source_id: str

    def __post_init__(self) -> None:
        import numpy as np

        if (not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8
                or self.image.ndim != 3 or self.image.shape[2] != 3
                or min(self.image.shape[:2]) < 1):
            raise ValueError("image must be a nonempty uint8 BGR image")
        if (type(self.timestamp) not in (int, float) or not math.isfinite(self.timestamp)
                or self.timestamp < 0):
            raise ValueError("timestamp must be finite and nonnegative")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a nonempty string")


def _check_next(previous: FramePacket, current: FramePacket) -> None:
    if current.source_id != previous.source_id:
        raise ValueError("frames must come from a single source")
    if current.timestamp <= previous.timestamp:
        raise ValueError("frame timestamps must be strictly increasing")
    if current.frame_index <= previous.frame_index:
        raise ValueError("frame indices must be strictly increasing")


@dataclass(frozen=True)
class CapturedClip:
    frames: tuple[FramePacket, ...]
    source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or len(self.frames) < 2:
            raise ValueError("a clip requires at least two frames in a tuple")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a nonempty string")
        previous = None
        for packet in self.frames:
            if not isinstance(packet, FramePacket):
                raise ValueError("frames must contain FramePacket objects")
            packet.__post_init__()
            if packet.source_id != self.source_id:
                raise ValueError("frames must come from a single source")
            if previous is not None:
                _check_next(previous, packet)
            previous = packet


class _CaptureSource:
    def __init__(self, target: str | int, source_id: str,
                 capture_factory: Callable | None = None) -> None:
        self.source_id = source_id
        self._target = target
        self._factory = capture_factory
        self._capture = None
        self._index = 0
        self._previous = None
        self._eof = False

    def open(self):
        if self._capture is not None:
            return self
        import cv2

        self._index, self._previous, self._eof = 0, None, False
        try:
            self._capture = (self._factory or cv2.VideoCapture)(self._target)
            if not self._capture.isOpened():
                raise SourceError(f"cannot open source {self.source_id}")
            self._opened()
        except Exception as exc:
            self.close()
            if isinstance(exc, SourceError):
                raise
            raise SourceError(f"cannot open source {self.source_id}: {exc}") from exc
        return self

    def _opened(self) -> None:
        pass

    def read(self) -> FramePacket | None:
        if self._eof:
            return None
        if self._capture is None:
            raise SourceError("source must be open before reading")
        try:
            ok, image = self._capture.read()
            if not ok:
                return self._read_failed()
            packet = FramePacket(image, self._timestamp(), self._index, self.source_id)
            if self._previous is not None:
                _check_next(self._previous, packet)
            self._previous = packet
            self._index += 1
            return packet
        except Exception as exc:
            self.close()
            if isinstance(exc, SourceError):
                raise
            raise SourceError(f"invalid frame from {self.source_id}: {exc}") from exc

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self):
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()


class VideoSource(_CaptureSource):
    def __init__(self, path: str | Path, *, capture_factory: Callable | None = None):
        self.path = Path(path)
        self.timestamp_mode = "pos_msec"
        super().__init__(str(path), f"video:{self.path.resolve()}", capture_factory)

    def _opened(self) -> None:
        import cv2

        self.timestamp_mode = "pos_msec"
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(raw_count) or raw_count < 2 or abs(raw_count - round(raw_count)) > 0.01:
            raise SourceError("video must declare a reliable frame count of at least two")
        self._frame_count = int(round(raw_count))
        if self._frame_count > 1_000_000:
            raise SourceError("video frame count exceeds the clip extraction limit")
        # Preflight retains at most 8 MB of timestamps, never a clip of RGB pixels.
        # Decide one timeline before returning any preview or recognition packet.
        self._times = array("d")
        reliable_times = True
        self._shape = None
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            if len(self._times) >= self._frame_count:
                raise SourceError(f"video decoded more frames than declared ({self._frame_count})")
            FramePacket(image, 0, len(self._times), self.source_id)
            if self._shape is None:
                self._shape = image.shape
            elif image.shape != self._shape:
                raise SourceError("video changes image dimensions during decoding")
            timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            reliable_times = reliable_times and (
                math.isfinite(timestamp) and timestamp >= 0
                and (not self._times or timestamp > self._times[-1]))
            self._times.append(timestamp)
        if len(self._times) != self._frame_count:
            raise SourceError(f"incomplete video decoding: expected {self._frame_count} frames, decoded {len(self._times)}")
        if not reliable_times:
            if not math.isfinite(self._fps) or self._fps <= 0:
                raise SourceError("video has no reliable media time or FPS fallback")
            self.timestamp_mode = "fps_fallback"
            self._times = array("d", (index / self._fps for index in range(self._frame_count)))
        # Seeking must report success and a zero frame position. Otherwise reopen
        # a fresh decoder rather than silently continue from the preflight EOF.
        try:
            rewound = self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            position = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
            rewound = rewound and math.isfinite(position) and abs(position) <= 0.01
        except Exception:
            rewound = False
        if not rewound:
            self.close()
            self._capture = (self._factory or cv2.VideoCapture)(self._target)
            if not self._capture.isOpened():
                raise SourceError("cannot reopen video after timeline preflight")

    def _timestamp(self) -> float:
        if self._index >= self._frame_count:
            raise SourceError(f"video decoded more frames than declared ({self._frame_count})")
        return self._times[self._index]

    def read(self) -> FramePacket | None:
        packet = super().read()
        if packet is not None and packet.image.shape != self._shape:
            self.close()
            raise SourceError("video changes image dimensions during decoding")
        return packet

    def _read_failed(self) -> None:
        if self._index == 0:
            raise SourceError("video contains no decodable frames")
        if self._frame_count != self._index:
            raise SourceError("video ended before its declared frame count; decode failed")
        self._eof = True
        self.close()
        return None


class CameraSource(_CaptureSource):
    def __init__(self, index: int = 0, *, capture_factory: Callable | None = None,
                 clock: Callable[[], float] | None = None):
        if type(index) is not int or index < 0:
            raise ValueError("camera index must be a nonnegative integer")
        # On Python 3.12 Windows, monotonic may repeat for reads within a coarse
        # clock tick; perf_counter provides actual high-resolution monotonic time.
        self.index = index
        self._clock = time.perf_counter if clock is None else clock
        self._first_packet = None
        super().__init__(index, f"camera:{index}", capture_factory)

    def _opened(self) -> None:
        # A Windows camera handle may open while its USB stream cannot deliver
        # any image. Do not report source readiness until one valid frame exists.
        self._first_packet = super().read()

    def read(self) -> FramePacket | None:
        if self._first_packet is not None:
            packet, self._first_packet = self._first_packet, None
            return packet
        return super().read()

    def close(self) -> None:
        self._first_packet = None
        super().close()

    def _timestamp(self) -> float:
        return self._clock()

    def _read_failed(self):
        if self._index == 0:
            raise SourceError("camera opened but could not read the first frame; check the USB connection and camera access")
        raise SourceError("camera disconnected or frame capture failed")


class ClipBuffer:
    def __init__(self, *, max_seconds: float = 15, max_frames: int = 600,
                 max_bytes: int = 128 * 1024 * 1024, max_width: int = 640):
        if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        for name, value in (("max_frames", max_frames), ("max_bytes", max_bytes),
                            ("max_width", max_width)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_seconds, self.max_frames = max_seconds, max_frames
        self.max_bytes, self.max_width = max_bytes, max_width
        self.clear()

    def clear(self) -> None:
        self._frames: list[FramePacket] = []
        self._bytes = 0

    def append(self, packet: FramePacket) -> None:
        import cv2

        if not isinstance(packet, FramePacket):
            raise ValueError("append requires a FramePacket")
        packet.__post_init__()
        if self._frames:
            _check_next(self._frames[-1], packet)
        image = packet.image
        height, width = image.shape[:2]
        if width > self.max_width:
            image = cv2.resize(image, (self.max_width, max(1, round(height * self.max_width / width))),
                               interpolation=cv2.INTER_AREA)
        duration = packet.timestamp - self._frames[0].timestamp if self._frames else 0
        if (len(self._frames) + 1 > self.max_frames or duration > self.max_seconds
                or self._bytes + image.nbytes > self.max_bytes):
            self.clear()
            raise ClipLimitError("recording exceeds duration, frame count, or byte limit; recording cleared")
        image = image.copy()
        self._frames.append(replace(packet, image=image))
        self._bytes += image.nbytes

    def finish(self) -> CapturedClip:
        if len(self._frames) < 2:
            self.clear()
            raise ValueError("recording requires at least two frames")
        origin = self._frames[0].timestamp
        clip = CapturedClip(tuple(replace(packet, timestamp=packet.timestamp - origin)
                                  for packet in self._frames), self._frames[0].source_id)
        self.clear()
        return clip
