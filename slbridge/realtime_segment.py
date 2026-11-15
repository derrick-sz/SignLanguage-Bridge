"""Bounded pose-only utterance buffer with conservative pause boundaries."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PoseSample:
    index: int
    timestamp: float
    keypoints: np.ndarray


@dataclass(frozen=True)
class PoseSegment:
    reason: str
    samples: tuple[PoseSample, ...]


class UtteranceBuffer:
    @property
    def active(self):
        return self._samples is not None and not self._blocked

    def __init__(self, *, fps=30, pause_seconds=.8, max_frames=256, max_seconds=8,
                 motion_threshold=.018):
        if (not math.isfinite(fps) or not 10 <= fps <= 60 or not .4 <= pause_seconds <= 2
                or type(max_frames) is not int or not 16 <= max_frames <= 256 or not 1 <= max_seconds <= 20):
            raise ValueError('Invalid utterance buffer limits')
        self.fps, self.pause_seconds = fps, pause_seconds
        self.max_frames, self.max_seconds = max_frames, max_seconds
        self.motion_threshold = motion_threshold
        self.min_frames = max(4, int(fps*.25))
        self._pre = deque(maxlen=max(2, int(fps*.2)))
        self.reset()

    def reset(self):
        self._pre.clear()
        self._samples = None
        self._previous = None
        self._last_index = self._last_stamp = None
        self._last_motion = self._last_word = -math.inf
        self._moving_run = 0
        self._has_words = False
        self._blocked = False
        self.overflowed = False

    def observe(self, packet, keypoints, *, has_words, word_changed, blank):
        if self._last_index is not None and (packet.frame_index != self._last_index+1 or
                                             packet.timestamp-self._last_stamp > .5):
            self.reset()
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (133,3) or not np.isfinite(points).all():
            self.reset()
            return None
        scale = max(float(np.linalg.norm(points[5,:2]-points[6,:2])), 64.)
        moving = False
        if self._previous is not None:
            valid = (points[91:,2]>.2) & (self._previous[91:,2]>.2)
            if valid.sum() >= 8:
                distance = np.linalg.norm(points[91:,:2]-self._previous[91:,:2], axis=-1)
                moving = float(np.median(distance[valid]))/scale > self.motion_threshold
        self._previous = points.copy()
        self._last_index, self._last_stamp = packet.frame_index, packet.timestamp
        if moving:
            self._last_motion = packet.timestamp
        if self._blocked:
            if not moving and packet.timestamp-self._last_motion >= self.pause_seconds:
                self.reset()
            return None
        if word_changed:
            self._last_word = packet.timestamp
        self._has_words = has_words
        points = points.copy(); points.setflags(write=False)
        sample = PoseSample(packet.frame_index, packet.timestamp, points)
        if self._samples is None:
            self._pre.append(sample)
            self._moving_run = self._moving_run+1 if moving else 0
            if self._moving_run >= 2:
                self._samples = list(self._pre)
        else:
            self._samples.append(sample)
        if self._samples is None:
            return None
        duration = packet.timestamp-self._samples[0].timestamp
        if len(self._samples) > self.max_frames or duration > self.max_seconds:
            self.reset()
            self._blocked = self.overflowed = True
            self._last_motion = packet.timestamp
            self._previous = points
            self._last_index, self._last_stamp = packet.frame_index, packet.timestamp
            return None
        hands = points[91:]; visible = hands[:,2]>.2
        rest = (visible.sum() >= 8 and float(points[5:7,2].min())>.2 and
                float(np.median(hands[visible,1])) > float(points[5:7,1].mean())+.8*scale)
        if (has_words and len(self._samples) >= self.min_frames and
                packet.timestamp-self._last_motion >= self.pause_seconds and
                packet.timestamp-self._last_word >= .35 and (blank or rest)):
            return self.take('pause')
        return None

    def take(self, reason):
        samples = self._samples
        valid = not self._blocked and self._has_words and samples is not None and len(samples) >= self.min_frames
        result = PoseSegment(reason, tuple(samples)) if valid else None
        self.reset()
        return result
