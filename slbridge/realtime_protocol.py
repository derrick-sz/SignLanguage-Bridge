"""Versioned, bounded input contract for the CSL live session."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import json
import math

from .sources import FramePacket

PROTOCOL = 'csl-realtime-v1'
MAX_MESSAGE_BYTES = 256 * 1024
MAX_JPEG_BYTES = 180 * 1024
MAX_SESSION_SECONDS = 1800


def _number(value, minimum, maximum):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def parse_message(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_MESSAGE_BYTES:
        raise ValueError('Live message exceeds size limit')
    def unique(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError('Duplicate live message field')
            data[key] = value
        return data
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON value')
    try:
        data = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError('Invalid live message') from error
    if not isinstance(data, dict):
        raise ValueError('Live message must be an object')
    return data


def validate_start(data):
    required = {'type', 'protocol', 'sign_language', 'source_epoch', 'fps'}
    if not isinstance(data, dict) or not required <= set(data) or set(data)-required-{'assistance'}:
        raise ValueError('Invalid live start fields')
    if data['type'] != 'start' or data['protocol'] != PROTOCOL or data['sign_language'] != 'csl':
        raise ValueError('Unsupported live protocol or language')
    if type(data['source_epoch']) is not int or not 0 <= data['source_epoch'] <= 2**31-1:
        raise ValueError('Invalid source epoch')
    if not _number(data['fps'], 10, 60):
        raise ValueError('Capture FPS must be between 10 and 60')
    if 'assistance' in data and type(data['assistance']) is not bool:
        raise ValueError('assistance must be an explicit boolean')
    return dict(data)


@dataclass(frozen=True)
class DecodedFrame:
    packet: FramePacket
    capture_ms: float


class FrameDecoder:
    def __init__(self, *, source_epoch, source_id):
        self.source_epoch = source_epoch
        self.source_id = source_id
        self._previous = None

    def decode(self, data):
        required = {'type', 'index', 'timestamp', 'source_epoch', 'capture_ms', 'jpeg'}
        if not isinstance(data, dict) or set(data) != required or data['type'] != 'frame':
            raise ValueError('Invalid frame fields')
        if type(data['source_epoch']) is not int or data['source_epoch'] != self.source_epoch:
            raise ValueError('Frame belongs to a different source epoch')
        index, stamp = data['index'], data['timestamp']
        if type(index) is not int or not 0 <= index < 2**31 or not _number(stamp, 0, MAX_SESSION_SECONDS):
            raise ValueError('Invalid frame index/time')
        if not _number(data['capture_ms'], 0, 1e12):
            raise ValueError('Invalid client capture clock')
        if self._previous and (index <= self._previous[0] or stamp <= self._previous[1]):
            raise ValueError('Frames must have strictly increasing index/time')
        encoded = data['jpeg']
        if not isinstance(encoded, str) or len(encoded) > (MAX_JPEG_BYTES*4//3+4):
            raise ValueError('Frame exceeds compressed size limit')
        from PIL import Image
        try:
            raw = base64.b64decode(encoded, validate=True)
            if not raw or len(raw) > MAX_JPEG_BYTES:
                raise ValueError('Frame exceeds compressed size limit')
            import numpy as np
            with Image.open(BytesIO(raw)) as image:
                # Inspect header dimensions before allocating decoded pixels.
                if image.format != 'JPEG' or not all(16 <= n <= 512 for n in image.size):
                    raise ValueError('Live frames require JPEG dimensions in [16,512]')
                pixels = np.asarray(image.convert('RGB'), dtype=np.uint8)[..., ::-1].copy()
        except (binascii.Error, OSError, Image.DecompressionBombError) as error:
            raise ValueError('Invalid live JPEG frame') from error
        packet = FramePacket(pixels, float(stamp), index, self.source_id)
        self._previous = index, float(stamp)
        return DecodedFrame(packet, float(data['capture_ms']))
