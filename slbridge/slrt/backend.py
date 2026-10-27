"""SLRT model ownership and ordinary-video inference."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import math
import threading

from ..sources import FramePacket, SourceError
from .assets import SLRTResourceError, sha256
from .pose import make_frontend
from .runtime import SLRTRuntime
from .stream import SLRTStream


def video_packets(path, *, cancellation=None, metadata=None):
    """Sequential decode, preserving reliable POS_MSEC; explicit FPS fallback.

    Unlike the existing clip preflight, this never decodes the future video
    before emitting its first frame. No video frames are resampled.
    """
    import cv2
    path = Path(path).resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise SourceError(f'Cannot open video: {path}')
    info = metadata if metadata is not None else {}
    index, previous = 0, -1.
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        declared = int(round(raw_count)) if math.isfinite(raw_count) and raw_count > 0 else 0
        info.update(path=str(path),original_fps=fps if math.isfinite(fps) else None,
                    declared_frame_count=declared,timestamp_fallback_frames=0)
        while True:
            if cancellation:
                cancellation()
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            if not math.isfinite(timestamp) or timestamp <= previous:
                if not math.isfinite(fps) or fps <= 0:
                    raise SourceError('Video has neither monotonic timestamps nor reliable FPS')
                timestamp = index / fps
                info['timestamp_fallback_frames'] += 1
            if timestamp <= previous or timestamp < 0:
                raise SourceError('Video timestamp fallback is inconsistent with its preceding timeline')
            previous = timestamp
            yield FramePacket(frame,timestamp,index,f'video:{path}')
            index += 1
        if index == 0:
            raise SourceError('Video contains no decoded frames')
        declared = info['declared_frame_count']
        if declared > 0 and index != declared:
            raise SourceError(f'Video decoded frame count mismatch: decoded {index}, expected {declared}')
        info.update(decoded_frames=index,last_timestamp=previous,timestamp_source='pos_msec_with_explicit_fps_fallback',
                    completeness_validation='declared_count_exact' if declared else 'decoder_eof_only')
    finally:
        capture.release()


class SLRTBackend:
    def __init__(self, config, *, frame_cache_enabled: bool = True):
        self.config = config
        self.runtime = SLRTRuntime(config, frame_cache_enabled=frame_cache_enabled)
        self._lock = threading.RLock()
        self._closed = False

    def new_stream(self,*,pose_npz=None,video_path=None,event_sink=None,window_sink=None):
        if self._closed:
            raise RuntimeError('SLRT backend is closed')
        # Each stream is one clip/source: drop prepared-frame tensors from the
        # previous clip before starting.  (Within a stream the cache removes the
        # stride=1 duplicate work; values are bit-identical to the batched path.)
        self.runtime.clear_frame_cache()
        frontend = make_frontend(self.config,pose_npz=pose_npz,video_path=video_path)
        return SLRTStream(self.runtime,frontend,self.config,event_sink=event_sink,window_sink=window_sink)

    def recognize_video(self,path,*,pose_npz=None,cancellation=None,event_sink=None,window_sink=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise SLRTResourceError(f'Input video missing: {path}')
        with self._lock:
            start = perf_counter()
            stream = self.new_stream(pose_npz=pose_npz,video_path=path,event_sink=event_sink,window_sink=window_sink)
            metadata = dict(video_sha256=sha256(path))
            decode_ms = 0.
            packets = video_packets(path,cancellation=cancellation,metadata=metadata)
            try:
                while True:
                    begin = perf_counter()
                    try:
                        packet = next(packets)
                    except StopIteration:
                        break
                    decode_ms += (perf_counter()-begin)*1000
                    stream.feed(packet)
                if cancellation:
                    cancellation()
                stream.flush()
                result = stream.result()
                elapsed = perf_counter()-start
                result.update(source=metadata,video_decode_ms=decode_ms,wall_seconds=elapsed,
                              processing_fps=stream.frames_received/elapsed if elapsed else None,
                              first_stable_wall_latency_ms=None)
                # See the CLI event sink for wall-clock first-final latency.
                return result
            finally:
                packets.close()
                stream.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.runtime.close()
