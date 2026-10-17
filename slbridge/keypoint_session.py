"""Model-free, bounded keypoint capture and recorded-stream playback."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
import queue
import threading
import time
import uuid

from .session import SessionEvent


class _RecordingSource:
    def __init__(self, path):
        self.path = Path(path)
        self.recording = None
        self._iterator = None

    def open(self):
        from .keypoint_recording import KeypointRecording
        self.recording = KeypointRecording(self.path)
        self._iterator = iter(self.recording.iter_frames())
        return self

    def read(self):
        return next(self._iterator, None)

    def close(self):
        if self._iterator is not None:
            self._iterator.close()
        self._iterator = None


class KeypointSession:
    """Two native-resource owners, a two-frame queue and two latest slots.

    No recognizer is constructed. Cancellation invalidates delivery; an already
    running detector returns on its owning worker before it can be closed.
    Recording saves only successfully processed observations, with their gaps.
    """

    def __init__(self, *, extractor_factory=None):
        self.session_id = uuid.uuid4().hex
        self.epoch = 0
        self.closed = False
        self.source = None
        self.source_kind = None
        self.source_active = False
        self.video_path = None
        self.paused = False
        self._pause_started = None
        self._paused_duration = 0.
        self._source_timing_mode = None
        self._factory = extractor_factory
        self._lock = threading.RLock()
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._frames = queue.Queue(maxsize=2)
        self._events = deque(maxlen=48)
        self._tracking_version = 0
        self._eof = None
        self._preview = None
        self._keypoints = None
        self._metadata = None
        self._recorder = None
        self._recorded_frames = 0
        self._processed = 0
        self._dropped = 0
        self._latest_dt = None
        self._extraction_ms = None
        self._receipt_age_ms = None
        self._time_domain = None
        self._error = ""
        self._capture_thread = threading.Thread(target=self._capture_loop,
                                               name="slbridge-keypoint-capture", daemon=True)
        self._extract_thread = threading.Thread(target=self._extract_loop,
                                               name="slbridge-keypoint-extract", daemon=True)
        self._capture_thread.start()
        self._extract_thread.start()

    def _event(self, kind, value=None):
        self._events.append(SessionEvent(kind, value, 0, self.epoch))

    def _clear_queue(self):
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _cancel_recording(self, reason):
        recorder, self._recorder = self._recorder, None
        cleanup_error = None
        if recorder is not None:
            try:
                recorder.abort()
            except Exception as exc:
                cleanup_error = "临时文件清理失败：" + str(exc)
                self._error = cleanup_error
                self._event("recording_error", cleanup_error)
            finally:
                self._event("recording_cancelled", reason + ("；" + cleanup_error if cleanup_error else ""))
        return cleanup_error

    def stop_source(self):
        with self._lock:
            self.epoch += 1
            self._tracking_version += 1
            self._stop.set()
            self._clear_queue()
            self._events.clear()
            self._cancel_recording("来源已关闭，未完成录制已取消")
            self.source = None
            self.source_kind = None
            self.video_path = None
            self.source_active = False
            self.paused = False
            self._pause_started = None
            self._paused_duration = 0.
            self._source_timing_mode = None
            self._eof = None
            self._preview = self._keypoints = self._metadata = None
            self._latest_dt = self._extraction_ms = self._receipt_age_ms = None
            self._time_domain = None
            self._event("stopped", "来源已停止")
            self._changed.set()

    def open_source(self, source, *, kind="video", video_path=None):
        if kind not in {"video", "camera"}:
            raise ValueError("source kind must be video or camera")
        self._open(source, kind, video_path)

    def open_recording(self, path):
        self._open(_RecordingSource(path), "recording", path)

    def _open(self, source, kind, path):
        with self._lock:
            if self.closed:
                raise RuntimeError("关键点会话已关闭")
            self.stop_source()
            self.source = source
            self.source_kind = kind
            self.video_path = Path(path) if path is not None else None
            self._processed = self._dropped = self._recorded_frames = 0
            self._error = ""
            self._stop = threading.Event()
            self._changed.set()

    def pause(self):
        with self._lock:
            if not self.source_active:
                raise RuntimeError("当前没有活动来源")
            if not self.paused:
                self.paused = True
                self._pause_started = time.perf_counter()
                self._tracking_version += 1
                self._clear_queue()
                self._event("paused", True)

    def resume(self):
        with self._lock:
            if not self.source_active:
                raise RuntimeError("当前没有活动来源")
            if self.paused:
                self._paused_duration += time.perf_counter() - self._pause_started
                self._pause_started = None
                self.paused = False
                self._event("paused", False)

    def toggle_pause(self):
        self.resume() if self.paused else self.pause()

    def reset_tracking(self):
        with self._lock:
            if self.source_kind == "recording" or not self.source_active:
                raise RuntimeError("只有活动采集来源可以重置跟踪")
            self._tracking_version += 1
            self._clear_queue()
            self._preview = self._keypoints = None
            self._latest_dt = None
            self._event("tracking_reset", "下一帧重新建立跟踪，录制继续保留重置标记")

    def start_recording(self, path, metadata=None):
        from .keypoint_recording import KeypointRecorder
        with self._lock:
            if self.closed or not self.source_active or self.source_kind == "recording" or self.paused:
                raise RuntimeError("请先启动视频或摄像头采集")
            if self._recorder is not None:
                raise RuntimeError("已在录制关键点")
            if self._metadata is None:
                raise RuntimeError("正在等待第一帧关键点，请稍后重试")
            annotations = deepcopy(metadata or {})
            if not isinstance(annotations, dict):
                raise TypeError("recording metadata must be a mapping")
            annotations["review_status"] = "draft"
            self._recorder = KeypointRecorder(path, metadata={
                "extractor": deepcopy(self._metadata), "source_kind": self.source_kind,
                "source_path": str(self.video_path) if self.video_path else None,
                "source_timing_mode": self._source_timing_mode,
                "annotations": annotations,
            })
            self._recorded_frames = 0
            self._event("recording_started", str(Path(path)))

    def finish_recording(self):
        with self._lock:
            if self._recorder is None:
                raise RuntimeError("尚未开始录制关键点")
            recorder, self._recorder = self._recorder, None
            try:
                path = recorder.finish(reason="stopped")
            except Exception as exc:
                self._error = str(exc)
                try:
                    recorder.abort()
                except Exception as cleanup:
                    self._error += "；临时文件清理失败：" + str(cleanup)
                self._event("recording_error", self._error)
                raise
            self._event("recording_saved", str(path))
            if getattr(recorder, "cleanup_error", None):
                self._event("recording_cleanup_warning", recorder.cleanup_error)
            return path

    def cancel_recording(self):
        with self._lock:
            self._cancel_recording("用户取消录制")

    def poll(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def take_view(self):
        with self._lock:
            value = self._preview, self._keypoints
            self._preview = self._keypoints = None
            return value

    def take_preview(self):
        with self._lock:
            value, self._preview = self._preview, None
            return value

    def take_keypoints(self):
        with self._lock:
            value, self._keypoints = self._keypoints, None
            return value

    def status(self):
        with self._lock:
            return dict(source_kind=self.source_kind, source_active=self.source_active,
                        paused=self.paused, recording=self._recorder is not None,
                        recorded_frames=self._recorded_frames, processed_frames=self._processed,
                        dropped_frames=self._dropped, queue_size=self._frames.qsize(),
                        latest_dt=self._latest_dt, latest_gap_seconds=self._latest_dt,
                        extraction_ms=self._extraction_ms, receipt_age_ms=self._receipt_age_ms,
                        time_domain=self._time_domain, error=self._error)

    def _matches(self, epoch, version=None):
        return not self.closed and epoch == self.epoch and (
            version is None or version == self._tracking_version)

    def _fail(self, error, epoch):
        with self._lock:
            if self._matches(epoch):
                self._stop.set()
                self._tracking_version += 1
                self._clear_queue()
                cleanup_error = self._cancel_recording("采集失败，未完成录制已取消")
                self.source_active = False
                self._preview = self._keypoints = None
                self._metadata = None
                self._error = str(error) + ("；" + cleanup_error if cleanup_error else "")
                self._event("error", self._error)

    def _capture_loop(self):
        while True:
            self._changed.wait()
            self._changed.clear()
            with self._lock:
                if self.closed:
                    return
                source, epoch, stop, kind = self.source, self.epoch, self._stop, self.source_kind
            if source is not None:
                self._capture(source, epoch, stop, kind)

    def _capture(self, source, epoch, stop, kind):
        try:
            source.open()
            with self._lock:
                if not self._matches(epoch) or stop.is_set():
                    return
                self.source_active = True
                self._source_timing_mode = ("application_monotonic" if kind == "camera"
                                            else getattr(source, "timestamp_mode", "unspecified"))
                self._event("opened", kind)
            started, first_time = time.perf_counter(), None
            previous_packet = None
            while not stop.is_set():
                if not self._wait_capture(epoch, stop, started):
                    break
                packet = source.read()
                received = time.perf_counter()
                if packet is None:
                    with self._lock:
                        if self._matches(epoch):
                            self._eof = epoch
                            if kind == "recording":
                                self._event("eof", "关键点回放结束")
                    break
                if kind != "recording":
                    # Track source order independently of detector resets and
                    # dropped observations. A reset cannot excuse a bad clock.
                    from .sources import FramePacket, _check_next
                    if not isinstance(packet, FramePacket):
                        raise ValueError("capture requires FramePacket observations")
                    packet.__post_init__()
                    if previous_packet is not None:
                        _check_next(previous_packet, packet)
                    previous_packet = packet
                media_offset = None
                if kind in {"video", "recording"}:
                    if first_time is None:
                        first_time = packet.timestamp
                    media_offset = packet.timestamp - first_time
                delivered = False
                while self._wait_capture(epoch, stop, started, media_offset):
                    with self._lock:
                        if not self._matches(epoch) or stop.is_set():
                            break
                        # A pause can race the ready check. Retain the decoded
                        # frame and recheck the shifted clock under this lock.
                        if self.paused or (media_offset is not None and
                                started + self._paused_duration + media_offset > time.perf_counter()):
                            continue
                        if kind == "recording":
                            self._keypoints = packet
                            self._preview = None
                            self._processed += 1
                            self._latest_dt = packet.dt
                            self._time_domain = packet.time_domain
                            self._metadata = deepcopy(source.recording.metadata)
                        else:
                            task = packet, epoch, self._tracking_version, received, kind
                            try:
                                self._frames.put_nowait(task)
                            except queue.Full:
                                try:
                                    self._frames.get_nowait()
                                    self._dropped += 1
                                except queue.Empty:
                                    pass
                                self._frames.put_nowait(task)
                        delivered = True
                        break
                if not delivered:
                    break
        except Exception as exc:
            self._fail(exc, epoch)
        finally:
            try:
                source.close()
            except Exception as exc:
                self._fail(exc, epoch)
            with self._lock:
                if self._matches(epoch):
                    self.source_active = False

    def _wait_capture(self, epoch, stop, started, media_offset=None):
        """Pause the playback clock even inside a long inter-frame wait."""
        while not stop.is_set():
            with self._lock:
                if not self._matches(epoch):
                    return False
                paused = self.paused
                remaining = (0. if media_offset is None else
                             started + self._paused_duration + media_offset - time.perf_counter())
                if not paused and remaining <= 0:
                    return True
            if stop.wait(.02 if paused else min(.02, max(0., remaining))):
                return False
        return False

    def _extract_loop(self):
        extractor, binding = None, None
        try:
            while True:
                try:
                    task = self._frames.get(timeout=.05)
                except queue.Empty:
                    with self._lock:
                        if self.closed:
                            return
                        done = self._eof == self.epoch
                        obsolete = self.source is None or self._stop.is_set() or (
                            binding is not None and binding[0] != self.epoch)
                        if done and self.source_kind != "recording":
                            self._eof = None
                            self._event("eof", "采集回放结束，可完成保存当前录制")
                    if (done or obsolete) and extractor is not None:
                        self._close_extractor(extractor, binding[0])
                        extractor, binding = None, None
                    continue
                packet, epoch, version, received, kind = task
                with self._lock:
                    if not self._matches(epoch, version) or self._stop.is_set() or self.paused:
                        continue
                try:
                    if binding is None or binding[0] != epoch:
                        if extractor is not None:
                            self._close_extractor(extractor, binding[0])
                        from .keypoints import KeypointExtractor
                        extractor = (self._factory or KeypointExtractor)(
                            session_id=self.session_id, source_epoch=epoch,
                            time_domain="media_seconds" if kind == "video" else "application_monotonic")
                    elif binding[1] != version:
                        extractor.reset(reason="session_reset")
                    binding = epoch, version
                    started = time.perf_counter()
                    frame = extractor.process(packet, received_monotonic=received)
                    ended = time.perf_counter()
                    metadata = extractor.metadata()
                    with self._lock:
                        if not self._matches(epoch, version) or self._stop.is_set() or self.paused:
                            continue
                        self._preview, self._keypoints = packet, frame
                        self._metadata = deepcopy(metadata)
                        self._processed += 1
                        self._latest_dt = frame.dt
                        self._extraction_ms = max(0., (ended - started) * 1000)
                        self._receipt_age_ms = max(0., (ended - received) * 1000)
                        self._time_domain = frame.time_domain
                        if self._recorder is not None:
                            try:
                                self._recorder.append(frame)
                                self._recorded_frames = self._recorder.count
                            except Exception as exc:
                                cleanup_error = self._cancel_recording("录制写入失败")
                                self._error = str(exc) + ("；" + cleanup_error if cleanup_error else "")
                                self._event("recording_error", self._error)
                except Exception as exc:
                    with self._lock:
                        current = self._matches(epoch, version)
                    if current:
                        self._fail(exc, epoch)
        finally:
            if extractor is not None:
                self._close_extractor(extractor, binding[0])

    def _close_extractor(self, extractor, epoch):
        try:
            extractor.close()
        except Exception as exc:
            self._fail(exc, epoch)

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.stop_source()
            self.closed = True
            self._changed.set()
