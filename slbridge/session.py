"""Bounded background capture and clip recognition; no GUI dependencies."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import Any

from .sources import ClipBuffer


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    value: Any
    request_id: int
    epoch: int


class RecognitionSession:
    """One model worker and one latest preview slot, with explicit cancellation.

    Cancellation invalidates delivery, not a running native model call. Until
    that call returns another inference cannot be queued.
    """

    def __init__(self, recognizer=None, *, buffer=None):
        self.recognizer = recognizer
        self.buffer = buffer if buffer is not None else ClipBuffer()
        self.state = "IDLE"
        self.epoch = 0
        self.request_id = 0
        self.source = None
        self.source_kind = None
        self.source_active = False
        self.video_path = None
        self.latest_frame = None
        self.paused = False
        self.closed = False
        self._busy = False
        self._lock = threading.RLock()
        self._events = deque(maxlen=32)
        self._tasks = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._source_changed = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_worker, name="slbridge-capture", daemon=True)
        self._capture_thread.start()
        self._worker = threading.Thread(target=self._run_worker, name="slbridge-model", daemon=True)
        self._worker.start()

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def _event(self, kind, value=None):
        self._events.append(SessionEvent(kind, value, self.request_id, self.epoch))

    def _invalidate(self):
        self.request_id += 1
        self.buffer.clear()
        self._events.clear()
        self._event("clear")

    def poll(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def take_preview(self):
        with self._lock:
            frame = self.latest_frame
            self.latest_frame = None
            return frame

    def stop_source(self):
        with self._lock:
            self.epoch += 1
            self._invalidate()
            self._stop.set()
            self.source = None
            self.source_kind = None
            self.source_active = False
            self.video_path = None
            self.latest_frame = None
            self.state = "IDLE"
            self._source_changed.set()

    def open_source(self, source, *, kind, video_path=None):
        if self.closed:
            raise RuntimeError("会话已关闭")
        if kind not in {"camera", "video"}:
            raise ValueError("Unknown source kind")
        self.stop_source()
        with self._lock:
            self.source = source
            self.source_kind = kind
            self.video_path = Path(video_path) if video_path is not None else None
            self.paused = False
            self._stop = stop = threading.Event()
            self.state = "PREVIEW"
            self._source_changed.set()

    def _capture_worker(self):
        # Source open/read/close always share one owner. Rapid switches replace
        # the pending source rather than creating more capture threads.
        while True:
            self._source_changed.wait()
            self._source_changed.clear()
            with self._lock:
                if self.closed:
                    return
                source, stop, epoch, kind = self.source, self._stop, self.epoch, self.source_kind
            if source is not None:
                self._capture(source, stop, epoch, kind)

    def _capture(self, source, stop, epoch, kind):
        try:
            source.open()
            with self._lock:
                if epoch == self.epoch and not stop.is_set() and not self.closed:
                    self.source_active = True
                    self._event("opened", f"{kind} 来源已打开")
            started = time.monotonic()
            first_timestamp = None
            paused_at = None
            while not stop.is_set():
                if kind == "video" and self.paused:
                    if paused_at is None:
                        paused_at = time.monotonic()
                    stop.wait(.02)
                    continue
                if paused_at is not None:
                    started += time.monotonic() - paused_at
                    paused_at = None
                packet = source.read()
                if packet is None:
                    with self._lock:
                        if epoch == self.epoch and not self.closed:
                            self._event("eof", "文件预览结束；仍可识别完整文件")
                    break
                if kind == "video":
                    if first_timestamp is None:
                        first_timestamp = packet.timestamp
                    if stop.wait(max(0, started + packet.timestamp - first_timestamp - time.monotonic())):
                        break
                with self._lock:
                    if epoch != self.epoch or self.closed:
                        break
                    self.latest_frame = packet
                    if self.state == "RECORDING":
                        self.buffer.append(packet)
        except Exception as error:
            with self._lock:
                if epoch == self.epoch and not self.closed:
                    self._invalidate()
                    self.latest_frame = None
                    self.state = "ERROR"
                    self._event("error", str(error))
        finally:
            try:
                source.close()
            except Exception as error:
                with self._lock:
                    if epoch == self.epoch and not self.closed:
                        self.state = "ERROR"
                        self._event("error", str(error))
            with self._lock:
                if epoch == self.epoch:
                    self.source_active = False
                    if not self.closed:
                        self._event("source_closed", "来源已释放" if kind == "camera" else "文件预览已结束并释放；仍可识别完整文件")

    def start_recording(self):
        with self._lock:
            if self.closed or self.source_kind != "camera" or not self.source_active:
                raise RuntimeError("请先打开摄像头")
            if self._busy:
                raise RuntimeError("模型任务仍在结束，请稍后重试")
            self._invalidate()
            self.state = "RECORDING"

    def finish_recording(self):
        with self._lock:
            if self.state != "RECORDING":
                raise RuntimeError("尚未开始录制")
            try:
                clip = self.buffer.finish()
            except Exception:
                self._invalidate()
                self.state = "ERROR"
                raise
            return self._submit(lambda recognizer: recognizer.recognize_clip(clip, sample_id=clip.source_id))

    def recognize_file(self):
        with self._lock:
            if self.video_path is None:
                raise RuntimeError("请先打开视频文件")
            path = self.video_path
            return self._submit(lambda recognizer: recognizer.recognize(path))

    def _submit(self, operation, *, kind="result", require_model=True):
        with self._lock:
            if self.closed:
                raise RuntimeError("会话已关闭")
            if self._busy:
                raise RuntimeError("模型忙；取消后须等待当前任务结束")
            if require_model and self.recognizer is None:
                raise RuntimeError("请先加载 RGB 片段模型")
            self._invalidate()
            self.state = "PROCESSING"
            self._busy = True
            self._tasks.put_nowait((self.request_id, self.epoch, operation, kind, self.recognizer))
            return self.request_id

    def load_model(self, path, loader=None):
        def load(_):
            if loader is None:
                from ._io import read_json
                metadata = read_json(Path(path) / "manifest.json")
                if (metadata.get("input_spec", {}).get("representation_id") != "mediapipe225_rgb_v1"
                        or "clip" not in metadata.get("capabilities", ())):
                    raise ValueError("需要 mediapipe225_rgb_v1 且支持 clip 的模型包")
                from .inference import Recognizer
                model = Recognizer.load(path)
            else:
                model = loader(path)
            manifest = model.bundle.manifest
            if manifest.input_spec.representation_id != "mediapipe225_rgb_v1" or "clip" not in manifest.capabilities:
                raise ValueError("需要 mediapipe225_rgb_v1 且支持 clip 的模型包")
            return model

        return self._submit(load, kind="model", require_model=False)

    def _run_worker(self):
        while True:
            task = self._tasks.get()
            if task is None:
                return
            with self._lock:
                if self.closed:
                    self._busy = False
                    return
            request_id, epoch, operation, kind, model = task
            try:
                value = operation(model)
                error = None
            except Exception as exc:
                value, error = None, str(exc)
            with self._lock:
                self._busy = False
                if self.closed:
                    return
                if (request_id, epoch) != (self.request_id, self.epoch):
                    continue
                if error is not None:
                    self.buffer.clear()
                    self.state = "ERROR"
                    self._event("error", error)
                else:
                    if kind == "model":
                        self.recognizer = value
                        self.state = "PREVIEW" if self.source is not None else "IDLE"
                    else:
                        self.state = "RESULT" if value.status == "recognized" else "REJECTED"
                    self._event(kind, value)

    def cancel(self):
        with self._lock:
            self._invalidate()
            self.state = "PREVIEW" if self.source is not None else "IDLE"

    def close(self):
        if self.closed:
            return
        self.stop_source()
        with self._lock:
            self.closed = True
            self._source_changed.set()
            # A running task already left the queue, so this remains bounded.
            try:
                self._tasks.put_nowait(None)
            except queue.Full:
                # The queued task wakes the worker, which observes closed and
                # exits without running it. Do not race its concurrent dequeue.
                pass
