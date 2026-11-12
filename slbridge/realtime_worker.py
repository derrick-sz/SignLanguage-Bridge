"""Bounded live-session worker; the worker alone owns native engine lifetime."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from copy import deepcopy
from uuid import uuid4

from .realtime_protocol import MAX_SESSION_SECONDS, PROTOCOL, DecodedFrame
from .runtime_contracts import CancellationToken

_BOUNDARY = object()


class LiveSession:
    def __init__(self, *, source_epoch, engine_factory, queue_size=16, event_limit=256, on_done=None):
        if type(queue_size) is not int or not 1 <= queue_size <= 32:
            raise ValueError('Live queue_size must be in [1,32]')
        if type(event_limit) is not int or not 16 <= event_limit <= 1024:
            raise ValueError('Invalid live event limit')
        self.session_id = uuid4().hex
        self.source_epoch = source_epoch
        self._factory, self._on_done = engine_factory, on_done
        self._frames = queue.Queue(maxsize=queue_size)
        self._events = queue.Queue(maxsize=event_limit)
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self.cancellation = CancellationToken()
        self._done = threading.Event()
        self._finishing = False
        self._thread = None
        self.reason = None
        self._started = time.perf_counter()
        self._sequence = 0
        self._queue_ms = deque(maxlen=512)
        self._processing_ms = deque(maxlen=512)
        self._frames_completed = 0
        self._prepare_ms = None
        self._native_report = None
        self._report_at = 0.

    def performance(self):
        def summary(values):
            rows = sorted(values)
            def percentile(fraction):
                if not rows: return None
                position = (len(rows)-1)*fraction
                left = int(position); right = min(left+1,len(rows)-1)
                return rows[left]+(rows[right]-rows[left])*(position-left)
            return dict(count=len(rows), p50=percentile(.5), p95=percentile(.95),
                        max=max(rows) if rows else None)
        with self._lock:
            return dict(session_id=self.session_id, source_epoch=self.source_epoch,
                        reason=self.reason, running=self.is_running, prepare_ms=self._prepare_ms,
                        frames_completed=self._frames_completed, pending_frames=self.pending_frames,
                        sample_limit=512, percentile_scope='last_512_completed_frames',
                        queue_ms=summary(self._queue_ms), processing_ms=summary(self._processing_ms),
                        native=deepcopy(self._native_report))

    def _snapshot_native(self, engine):
        # Metrics must never invalidate a good inference or conceal its failure.
        try:
            report = engine.performance() if callable(getattr(engine,'performance',None)) else None
            with self._lock: self._native_report = report
        except Exception:
            pass
        self._report_at = time.perf_counter()

    @property
    def is_running(self):
        return self._thread is not None and not self._done.is_set()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    @property
    def pending_frames(self):
        return self._frames.qsize()

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Live session cannot be restarted')
        self._thread = threading.Thread(target=self._run, name='csl-live-session', daemon=True)
        self._thread.start()

    def emit(self, kind, **data):
        with self._lock:
            if self._cancelled.is_set() and kind not in {'error', 'stopped'}:
                return
            event = dict(data, type=kind, protocol=PROTOCOL, session_id=self.session_id,
                         source_epoch=self.source_epoch, sequence=self._sequence,
                         server_elapsed_ms=(time.perf_counter()-self._started)*1000)
            self._sequence += 1
            try:
                self._events.put_nowait(event)
            except queue.Full:
                self._cancelled.set()
                self.cancellation.cancel()
                self.reason = 'event_backpressure'
                self._clear(self._events)
                self._events.put_nowait(dict(event, type='error', code=self.reason))

    @staticmethod
    def _clear(q):
        while True:
            try:q.get_nowait()
            except queue.Empty:return

    def submit(self, frame):
        if not isinstance(frame, DecodedFrame):
            raise ValueError('Live input requires a validated frame')
        with self._lock:
            if self._finishing or self._cancelled.is_set() or self._done.is_set():
                return False
            try:
                self._frames.put_nowait((frame, time.perf_counter()))
                return True
            except queue.Full:
                self.cancel('overload')
                self.emit('error', code='overload', message='识别暂时跟不上采集，请暂停后重试。')
                return False

    def finish(self):
        with self._lock:
            self._finishing = True

    def boundary(self):
        with self._lock:
            if self._finishing or self._cancelled.is_set() or self._done.is_set():
                return False
            try:
                self._frames.put_nowait((_BOUNDARY, time.perf_counter()))
                return True
            except queue.Full:
                self.cancel('overload')
                self.emit('error', code='overload')
                return False

    def cancel(self, reason='cancelled'):
        with self._lock:
            if self._cancelled.is_set():
                return
            self.reason = reason
            self._cancelled.set()
            self.cancellation.cancel()
            self._clear(self._frames)
            self._clear(self._events)

    def poll(self):
        events = []
        with self._lock:
            while len(events) < 128:
                try:events.append(self._events.get_nowait())
                except queue.Empty:break
        return events

    def join(self, timeout=5):
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_running

    def _run(self):
        natural = False
        try:
            with self._factory(self) as engine:
                try:
                    with self._lock: self._prepare_ms = (time.perf_counter()-self._started)*1000
                    self.emit('ready', capabilities=engine.info)
                    while not self._cancelled.is_set():
                        if time.perf_counter()-self._started > MAX_SESSION_SECONDS:
                            self.cancel('session_limit')
                            break
                        try:
                            frame, received = self._frames.get(timeout=.05)
                        except queue.Empty:
                            if self._finishing:
                                engine.finish()
                                natural = not self._cancelled.is_set()
                                break
                            if hasattr(engine,'idle'): engine.idle()
                            continue
                        if self._cancelled.is_set(): break
                        started = time.perf_counter()
                        delay = (started-received)*1000
                        if frame is _BOUNDARY:
                            engine.boundary()
                            continue
                        engine.feed(frame)
                        elapsed = (time.perf_counter()-started)*1000
                        with self._lock:
                            self._queue_ms.append(delay); self._processing_ms.append(elapsed)
                            self._frames_completed += 1
                        self.emit('ack', frame_index=frame.packet.frame_index, capture_ms=frame.capture_ms,
                                  source_timestamp=frame.packet.timestamp, queue_ms=delay,
                                  processing_ms=elapsed, pending_frames=self._frames.qsize())
                        if time.perf_counter()-self._report_at >= 1: self._snapshot_native(engine)
                    if not natural: engine.cancel()
                finally:
                    # Still inside the native lease, before the factory closes it.
                    self._snapshot_native(engine)
        except Exception as error:
            self.cancel('engine_error')
            self.emit('error', code='engine_error', error_type=type(error).__name__,
                      message='实时识别未能完成，请检查服务后重试。')
        finally:
            self.emit('finished' if natural else 'stopped', reason=self.reason)
            self._done.set()
            if self._on_done:
                self._on_done(self)
