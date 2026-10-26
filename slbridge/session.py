"""Bounded background capture and clip recognition; no GUI dependencies."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
import math
from typing import Any

from .sources import ClipBuffer
from .runtime import ModelManager
from .runtime_contracts import InferenceRequest, CancellationToken
from .caption_events import CaptionLedger
from .candidate_context import CandidateLedger


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    value: Any
    request_id: int
    epoch: int
    task_kind: str | None = None


class RecognitionSession:
    """One model worker and one latest preview slot, with explicit cancellation.

    Cancellation invalidates delivery, not a running native model call. Until
    that call returns another inference cannot be queued.
    """

    def __init__(self, recognizer=None, *, buffer=None, options=None, caption_catalog=None,
                 observer_factory=None, automatic_timeout=10.):
        if type(automatic_timeout) not in (int, float) or not math.isfinite(automatic_timeout) or automatic_timeout <= 0:
            raise ValueError("Automatic inference timeout must be finite and positive")
        self.model_manager = ModelManager(initial_recognizer=recognizer, options=options)
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
        self.caption_catalog = caption_catalog
        self.captions = CaptionLedger(max_pending=2)
        self._candidates = CandidateLedger(self.captions.session_id)
        self.automatic = False
        self.continuous = False
        self.show_keypoints = False
        self._keypoints_version = 0
        self._landmark_preview = None
        self._continuous_window = None
        self._continuous_version = 0
        self._continuous_sequence = 0
        self._continuous_last_submitted = None
        self._continuous_token = None
        self._continuous_quality_ok = False
        self._continuous_require_hands = True
        self._continuous_observation = None
        self._continuous_completed = 0
        self._continuous_stale = 0
        self._auto_version = 0
        self._auto_sequence = 0
        self._auto_segmenter = None
        self._observer_factory = observer_factory
        self._auto_frames = queue.Queue(maxsize=2)
        self._auto_pending = deque()
        self._automatic_timeout = automatic_timeout
        self._auto_token = None
        self._auto_eof = None
        self._auto_dropped_frames = 0
        self._auto_dropped_segments = 0
        self._auto_observed_frames = 0
        self._lock = threading.RLock()
        self._events = deque(maxlen=32)
        self._tasks = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._source_changed = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_worker, name="slbridge-capture", daemon=True)
        self._capture_thread.start()
        self._worker = threading.Thread(target=self._run_worker, name="slbridge-model", daemon=True)
        self._worker.start()
        self._observer_worker = threading.Thread(target=self._observe_loop, name="slbridge-activity", daemon=True)
        self._observer_worker.start()

    @property
    def recognizer(self):
        return self.model_manager.recognizer

    @recognizer.setter
    def recognizer(self, model):
        # Preserve pre-use dependency injection; live replacements use load_model.
        self.model_manager.set_initial(model)

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def _event(self, kind, value=None, *, task_kind=None):
        if kind == "continuous_result":
            self._events = deque((event for event in self._events if event.kind != kind), maxlen=32)
        self._events.append(SessionEvent(kind, value, self.request_id, self.epoch, task_kind))

    def _invalidate(self):
        self.request_id += 1
        self.buffer.clear()
        self._events.clear()
        self._event("clear")
        self._reset_automatic(disable=True)
        self._reset_continuous(disable=True)

    def _observing(self):
        return self.automatic or self.continuous or self.show_keypoints

    def _clear_landmarks(self):
        self._keypoints_version += 1
        self._landmark_preview = None

    def set_show_keypoints(self, enabled):
        if type(enabled) is not bool:
            raise TypeError('Keypoint visibility must be a boolean')
        with self._lock:
            if self.closed:
                raise RuntimeError('会话已关闭')
            self.show_keypoints = enabled
            self._clear_landmarks()

    def take_landmark_preview(self):
        from .preview_landmarks import MAX_PREVIEW_AGE
        with self._lock:
            pair,self._landmark_preview = self._landmark_preview,None
            if (pair is not None and self.show_keypoints and self.source_active and not self.paused
                    and pair[1].source_epoch==self.epoch
                    and time.perf_counter()-pair[1].completed_at<=MAX_PREVIEW_AGE):
                return pair
            return None

    def _reset_continuous(self, *, disable=False):
        had_continuous = self.continuous or self._continuous_window is not None
        self._continuous_version += 1
        if self._continuous_token is not None:
            self._continuous_token.cancel()
        self._continuous_token = None
        self._continuous_last_submitted = None
        self._continuous_quality_ok = False
        self._continuous_observation = None
        if self._continuous_window is not None:
            self._continuous_window.clear()
        if disable:
            self.continuous = False
            self._continuous_window = None
        if had_continuous:
            self._events = deque((event for event in self._events
                                  if not event.kind.startswith("continuous_")), maxlen=32)
            self._event("continuous_clear")

    def enable_continuous(self, config=None):
        from .continuous import ContinuousConfig, RollingWindow
        from .segmentation import SegmentConfig
        config = config or ContinuousConfig()
        if not isinstance(config, ContinuousConfig):
            raise TypeError("ContinuousConfig is required")
        with self._lock:
            if self.closed or not self.source_active:
                raise RuntimeError("请先打开摄像头或视频回放")
            if self._busy or self.state == "RECORDING":
                raise RuntimeError("请先完成当前任务")
            capabilities = self.model_manager.capabilities()
            if capabilities is None or "clip" not in capabilities["modes"] or not any(
                "captured_clip" in entry["input_kinds"] for entry in capabilities["inputs"]):
                raise RuntimeError("当前模型未声明视频窗口输入能力")
            self._invalidate()
            self._continuous_window = RollingWindow(config)
            self._continuous_activity_config = SegmentConfig(max_width=config.max_width)
            # The hands/quality gate exists so a live camera waits until someone is
            # signing. An explicitly replayed file already contains signing, so the
            # gate only drops usable windows there (measured: 0 candidates on
            # 204x170 news clips). ``require_hands_visible`` overrides the default.
            requested = config.require_hands_visible
            if requested is None:
                requested = self.source_kind != "video"
            self._continuous_require_hands = bool(requested)
            self._continuous_completed = self._continuous_stale = 0
            self.continuous = True
            self.state = "CONTINUOUS"
            self._event("continuous_status", "正在积累画面；动作过程中持续更新，无需停顿。")

    def disable_continuous(self):
        with self._lock:
            self._reset_automatic(disable=True)
            self._reset_continuous(disable=True)
            self.state = "PREVIEW" if self.source is not None else "IDLE"

    def continuous_status(self):
        with self._lock:
            window = self._continuous_window
            return dict(enabled=self.continuous, quality_ok=self._continuous_quality_ok,
                        completed_windows=self._continuous_completed, stale_results=self._continuous_stale,
                        buffered_frames=window.buffered_frames if window else 0,
                        buffered_bytes=window.buffered_bytes if window else 0,
                        max_result_age_seconds=window.config.max_result_age_seconds if window else None)

    def pause_source(self, paused):
        with self._lock:
            self.paused = bool(paused)
            self._clear_landmarks()
            if self.continuous:
                self._reset_continuous()
                self._event("continuous_status", "预览已暂停。" if paused else "继续积累最新画面。")

    def _dispatch_continuous(self):
        if (not self.continuous or self._busy or not self._continuous_quality_ok
                or self.source is None or self.source_kind == "video" and self.paused):
            return
        window = self._continuous_window
        config = window.config
        observation = self._continuous_observation
        if (window.latest_time is None or observation is None
                or abs(window.latest_time-observation.timestamp) > config.max_observation_gap_seconds):
            return
        if (self._continuous_last_submitted is not None
                and window.latest_time-self._continuous_last_submitted+1e-9 < config.update_interval_seconds):
            return
        snapshot = window.snapshot()
        if snapshot is None or time.perf_counter()-snapshot.received_at > config.max_result_age_seconds:
            return
        self._continuous_sequence += 1
        token = CancellationToken()
        request = InferenceRequest(f"continuous-{self.captions.session_id}-{self._continuous_sequence}",
            session_id=self.captions.session_id, source_epoch=self.epoch,
            expected_generation=self.model_manager.generation, cancellation=token,
            deadline_monotonic=time.monotonic()+self._automatic_timeout)
        self._continuous_token = token
        self._continuous_last_submitted = snapshot.end_time
        self._busy = True
        self._tasks.put_nowait((self.request_id, self.epoch,
            lambda _: self.model_manager.infer_clip(snapshot.clip, request), "continuous_result",
            (self._continuous_version, snapshot)))

    def _reset_automatic(self, *, disable=False):
        self._auto_version += 1
        self._clear_landmarks()
        if disable:
            self.automatic = False
        self._auto_pending.clear()
        self._auto_eof = None
        if self._auto_token is not None:
            self._auto_token.cancel()
        while True:
            try:
                self._auto_frames.get_nowait()
            except queue.Empty:
                break
        if self._auto_segmenter is not None:
            self._auto_segmenter.reset()
        self.captions.cancel()

    def enable_automatic(self, config=None):
        from .segmentation import AutoSegmenter
        with self._lock:
            if self.closed or not self.source_active:
                raise RuntimeError("请先打开摄像头或视频回放")
            if self._busy or self.state == "RECORDING":
                raise RuntimeError("请先完成当前任务")
            capabilities = self.model_manager.capabilities()
            if capabilities is None or not any(
                "captured_clip" in entry["input_kinds"] for entry in capabilities["inputs"]
            ) or "clip" not in capabilities["modes"]:
                raise RuntimeError("当前模型未声明摄像头片段输入能力")
            self._reset_automatic(disable=True)
            self._reset_continuous(disable=True)
            self._auto_segmenter = AutoSegmenter(config)
            self.automatic = True
            self.captions.bind(self.epoch, self.model_manager.generation)
            self.state = "AUTO"
            self._event("automatic", "自动分段已启动；完成动作后稍作停顿")

    def disable_automatic(self):
        with self._lock:
            self._reset_automatic(disable=True)
            self.state = "PREVIEW" if self.source is not None else "IDLE"
            self._event("automatic", "自动分段已停止；可使用手动录制")

    def undo_caption(self):
        with self._lock:
            self.captions.undo()
            self._event("captions", self.captions.snapshot())

    def clear_history(self):
        with self._lock:
            # Reset the action boundary too, so an old partial action cannot refill.
            self.request_id += 1
            self._events.clear()
            self._reset_automatic()
            self._reset_continuous()
            self.captions.clear()
            self._event("captions", self.captions.snapshot())

    def automatic_status(self):
        with self._lock:
            return dict(enabled=self.automatic, observation_queue=self._auto_frames.qsize(),
                        pending_segments=len(self._auto_pending), dropped_frames=self._auto_dropped_frames,
                        dropped_segments=self._auto_dropped_segments,
                        observed_frames=self._auto_observed_frames,
                        segmentation_state=getattr(self._auto_segmenter, "state", "WAITING"))

    def candidate_context(self, *, after_revision=None):
        """Return detached candidate evidence; never treat it as committed text."""
        with self._lock:
            return self._candidates.snapshot(after_revision=after_revision)

    def set_candidate_limit(self, limit):
        with self._lock:
            self._candidates.set_limit(limit)

    def clear_candidates(self):
        with self._lock:
            # A pre-clear native call cannot refill the newly cleared context.
            self._reset_continuous()
            self._candidates.clear()

    def export_candidate_context(self, path):
        from ._io import write_json_atomic
        snapshot = self.candidate_context()
        write_json_atomic(path, snapshot)
        return snapshot

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
                            self._clear_landmarks()
                            if self.continuous:
                                self._reset_automatic(disable=True)
                                self._reset_continuous(disable=True)
                                self.state = "PREVIEW"
                                self._event("continuous_status", "视频播放已结束，连续识别已停止。")
                            elif self.automatic:
                                self._auto_eof = (self.epoch, self._auto_version)
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
                    if self.continuous and self._continuous_quality_ok:
                        previous = self._continuous_window.latest_time
                        if previous is not None and packet.timestamp-previous > self._continuous_window.config.max_observation_gap_seconds:
                            self._reset_continuous()
                            self._event("continuous_status", "画面中断，正在重新积累最新画面。")
                        else:
                            self._continuous_window.append(packet)
                            self._dispatch_continuous()
                    if self.state == "RECORDING":
                        self.buffer.append(packet)
                    if self._observing():
                        self._offer_auto_frame(packet)
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

    def _offer_auto_frame(self, packet):
        task = (packet, self.epoch, self._auto_version, self.model_manager.generation,
                self._keypoints_version, time.perf_counter())
        try:
            self._auto_frames.put_nowait(task)
        except queue.Full:
            # Latest two observations, no unbounded capture backlog.
            try:
                self._auto_frames.get_nowait()
            except queue.Empty:
                pass
            self._auto_dropped_frames += 1
            self._auto_frames.put_nowait(task)

    def _observe_loop(self):
        observer, binding = None, None
        try:
            while True:
                try:
                    task = self._auto_frames.get(timeout=.1)
                except queue.Empty:
                    if self.closed:
                        return
                    with self._lock:
                        eof_closed = False
                        if self._auto_eof == (self.epoch, self._auto_version) and (self.automatic or self.continuous):
                            if self.automatic:
                                for notice in self._auto_segmenter.flush():
                                    self._event("candidate", dict(state=notice.state, reason=notice.reason,
                                                                  kind=notice.kind))
                            self._auto_eof = None
                            eof_closed = True
                            self._event("automatic" if self.automatic else "continuous_status",
                                        "回放已结束；字幕记录保留。")
                        should_close = not self._observing() or not self.source_active or eof_closed
                        cleanup_keypoints_version = self._keypoints_version
                    if should_close and observer is not None:
                        self._close_observer(observer,binding,cleanup_keypoints_version)
                        observer, binding = None, None
                    continue
                packet, epoch, version, generation, keypoints_version, received = task
                with self._lock:
                    if self.closed:
                        return
                    if (not self._observing() or (not self.source_active or self.paused) and not (self.automatic or self.continuous)
                            or (epoch, version, generation) != (
                        self.epoch, self._auto_version, self.model_manager.generation
                    )):
                        continue
                    from .segmentation import SegmentConfig
                    config = self._auto_segmenter.config if self.automatic else self._continuous_activity_config if self.continuous else SegmentConfig()
                    show_keypoints = self.show_keypoints
                try:
                    if binding != (epoch, version):
                        if observer is not None:
                            self._close_observer(observer,binding,keypoints_version)
                            observer = None
                        if not self._observing():
                            continue
                        from .segmentation import MediaPipeActivityObserver
                        factory = self._observer_factory or MediaPipeActivityObserver
                        observer = factory(config)
                        binding = (epoch, version)
                    landmark_frame = None
                    if show_keypoints:
                        from .preview_landmarks import PreviewLandmarks, MAX_PREVIEW_AGE
                        observation,raw = observer.observe_landmarks(packet)
                        if time.perf_counter()-received<=MAX_PREVIEW_AGE:
                            landmark_frame = PreviewLandmarks.from_result(packet,raw,source_epoch=epoch,
                                pose_visibility=config.pose_visibility)
                    else:
                        observation = observer.observe(packet)
                    with self._lock:
                        if (not self._observing() or (not self.source_active or self.paused) and not (self.automatic or self.continuous)
                                or (epoch, version, generation) != (
                            self.epoch, self._auto_version, self.model_manager.generation
                        )):
                            continue
                        self._auto_observed_frames += 1
                        if (landmark_frame is not None and self.show_keypoints
                                and self.source_active and not self.paused
                                and keypoints_version==self._keypoints_version):
                            self._landmark_preview = packet,landmark_frame
                        if self.continuous:
                            quality_ok = (observation.quality_ok and observation.hands_visible
                                          if self._continuous_require_hands else True)
                            if not quality_ok and self._continuous_quality_ok:
                                self._reset_continuous()
                                self._event("continuous_status", "等待双手清晰进入画面。")
                            self._continuous_quality_ok = quality_ok
                            self._continuous_observation = observation
                            self._dispatch_continuous()
                            continue
                        if not self.automatic:
                            continue
                        for notice in self._auto_segmenter.feed(packet, observation):
                            if notice.kind == "ready":
                                self._accept_auto_segment(notice.segment)
                            else:
                                self._event("candidate", dict(state=notice.state, reason=notice.reason,
                                                              kind=notice.kind))
                except Exception as error:
                    with self._lock:
                        if (epoch, version) == (self.epoch, self._auto_version):
                            if not self.automatic and not self.continuous:
                                if keypoints_version != self._keypoints_version:
                                    # Retire the failed graph on its owner before
                                    # the next packet, without undoing a new toggle.
                                    binding = None
                                    continue
                                self.show_keypoints = False
                                self._clear_landmarks()
                                self._event('keypoints_error','关键点显示已关闭：'+str(error))
                                continue
                            continuous_error = self.continuous
                            self.show_keypoints = False
                            self._reset_automatic(disable=True)
                            self._reset_continuous(disable=True)
                            self.state = "ERROR"
                            self._event("error", "自动观察中止；可手动录制。" + str(error),
                                        task_kind="continuous" if continuous_error else None)
                            self._event('keypoints_error','关键点显示已关闭：'+str(error))
        finally:
            if observer is not None:
                self._close_observer(observer,binding,keypoints_version)

    def _close_observer(self, observer, binding, keypoints_version):
        try:
            observer.close()
        except Exception as error:
            with self._lock:
                if (not self.closed and binding==(self.epoch,self._auto_version)
                        and keypoints_version==self._keypoints_version):
                    if not self.automatic and not self.continuous:
                        self.show_keypoints = False
                        self._clear_landmarks()
                        self._event('keypoints_error','关键点显示已关闭：'+str(error))
                        return
                    continuous_error = self.continuous
                    self.show_keypoints = False
                    self._reset_automatic(disable=True)
                    self._reset_continuous(disable=True)
                    self.state = "ERROR"
                    self._event("error", "活动观察器释放失败；自动模式已停止。" + str(error),
                                task_kind="continuous" if continuous_error else None)

    def _accept_auto_segment(self, segment):
        if len(self._auto_pending) >= 1:
            self._auto_dropped_segments += 1
            self._event("candidate", dict(state="REARM", kind="rejected", reason="inference_backpressure"))
            return
        self._auto_sequence += 1
        generation = self.model_manager.generation
        request = InferenceRequest(
            request_id=f"auto-{self.captions.session_id}-{self._auto_sequence}",
            session_id=self.captions.session_id, segment_id=segment.segment_id,
            source_epoch=self.epoch, expected_generation=generation,
            deadline_monotonic=time.monotonic() + self._automatic_timeout,
            cancellation=CancellationToken())
        self.captions.reserve(segment_id=segment.segment_id, sequence=self._auto_sequence,
                              request_id=request.request_id, source_epoch=self.epoch,
                              model_generation=generation, start_time=segment.start_time,
                              end_time=segment.end_time, source_id=segment.source_id)
        self._auto_pending.append((segment, request, self._auto_version, time.monotonic()))
        self._event("captions", self.captions.snapshot())
        self._dispatch_auto()

    def _dispatch_auto(self):
        if self._busy or not self._auto_pending or not self.automatic:
            return
        segment, request, version, queued_at = self._auto_pending.popleft()
        if time.monotonic() >= request.deadline_monotonic:
            self.captions.reject(segment.segment_id, "queue_timeout")
            self._event("captions", self.captions.snapshot())
            return
        self._busy = True
        self._auto_token = request.cancellation
        def infer(_):
            queue_ms = (time.monotonic() - queued_at) * 1000
            response = self.model_manager.infer_clip(segment.clip, request)
            response.timing_ms["queue_ms"] = queue_ms
            return response
        self._tasks.put_nowait((self.request_id, self.epoch, infer, "auto_result",
                                (version, segment.segment_id)))

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
            if self.model_manager.supports_typed_inference:
                request = self._manual_request()
                return self._submit(lambda _: self.model_manager.infer_clip(clip, request), kind="typed_result")
            return self._submit(lambda recognizer: recognizer.recognize_clip(clip, sample_id=clip.source_id))

    def recognize_file(self):
        with self._lock:
            if self.video_path is None:
                raise RuntimeError("请先打开视频文件")
            path = self.video_path
            if self.model_manager.supports_typed_inference:
                request = self._manual_request()
                return self._submit(lambda _: self.model_manager.infer_file(path, request), kind="typed_result")
            return self._submit(lambda recognizer: recognizer.recognize(path))

    def _manual_request(self):
        from uuid import uuid4
        return InferenceRequest(request_id="manual-" + uuid4().hex, session_id=self.captions.session_id,
                                source_epoch=self.epoch, expected_generation=self.model_manager.generation)

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
            return self.model_manager.prepare(path, loader=loader,
                                              required_input_kind="captured_clip")

        with self._lock:
            if self.state == "RECORDING":
                raise RuntimeError("请先结束或取消当前录制，再更换识别模型")
            return self._submit(load, kind="model", require_model=False)

    def rollback_model(self):
        with self._lock:
            if self.closed or self._busy or self.state == "RECORDING":
                raise RuntimeError("请先结束当前录制或识别，再恢复模型")
            self.model_manager.rollback()
            self._invalidate()
            self.state = "PREVIEW" if self.source is not None else "IDLE"
            self._event("model", self.recognizer)

    def _run_worker(self):
        try:
            self._work_loop()
        finally:
            self.model_manager.close()

    def _work_loop(self):
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
                value = (operation(model) if kind in {"model", "auto_result", "typed_result", "continuous_result"} else
                         self.model_manager.run(operation, expected_model=model))
                error = None
            except Exception as exc:
                value, error = None, str(exc)
            with self._lock:
                self._busy = False
                if self.closed:
                    if kind == "model" and value is not None:
                        self.model_manager.discard(value)
                    return
                if (request_id, epoch) != (self.request_id, self.epoch):
                    if kind == "model" and value is not None:
                        self.model_manager.discard(value)
                    self._dispatch_auto()
                    self._dispatch_continuous()
                    continue
                if kind == "continuous_result":
                    from .continuous import ContinuousUpdate
                    version, snapshot = model
                    if self.continuous and version == self._continuous_version:
                        config = self._continuous_window.config
                        age = time.perf_counter()-snapshot.received_at
                        latest = self._continuous_window.latest_time
                        stale = (age > config.max_result_age_seconds or latest is None
                                 or latest-snapshot.end_time > config.max_result_age_seconds
                                 or self._continuous_observation is None
                                 or latest-self._continuous_observation.timestamp > config.max_observation_gap_seconds
                                 or self.source_kind == "video" and self.paused)
                        if stale:
                            self._continuous_stale += 1
                            self._event("continuous_clear")
                            self._event("continuous_status", "已跳过过期结果，正在处理最新画面。")
                        elif error:
                            self._reset_automatic(disable=True)
                            self._reset_continuous(disable=True)
                            self._event("error", "连续识别已停止："+error, task_kind="continuous")
                        else:
                            self._continuous_completed += 1
                            if value.result.status == "recognized":
                                value.timing_ms['window_age_ms'] = age*1000
                                update = ContinuousUpdate(value, snapshot.start_time,
                                    snapshot.end_time, time.perf_counter(), age)
                                # Store before UI event coalescing, on the same validity boundary.
                                self._candidates.append(update, source_id=snapshot.clip.source_id,
                                    source_epoch=epoch, model_generation=self.model_manager.generation,
                                    window_revision=version, catalog=self.caption_catalog)
                                self._event("continuous_result", update)
                            else:
                                self._event("continuous_clear")
                                self._event("continuous_status", "当前窗口不足以识别，继续观察。")
                    self._dispatch_continuous()
                    continue
                if kind == "auto_result":
                    if model[0] == self._auto_version and self.automatic:
                        if error:
                            self.captions.reject(model[1], error)
                            self._event("candidate", dict(state="REARM", kind="rejected", reason=error))
                        else:
                            self.captions.complete(value, self.caption_catalog)
                            self._event("auto_result", value)
                        self._event("captions", self.captions.snapshot())
                    self._dispatch_auto()
                    continue
                if kind == "typed_result" and error is None:
                    value = value.result
                    kind = "result"
                if error is None and kind == "model":
                    try:
                        self.model_manager.activate(value)
                    except Exception as exc:
                        self.model_manager.discard(value)
                        value, error = None, str(exc)
                if error is not None:
                    self.buffer.clear()
                    self.state = "ERROR"
                    self._event("error", error, task_kind=kind)
                else:
                    if kind == "model":
                        value = value.recognizer
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
