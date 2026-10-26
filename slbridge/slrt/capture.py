"""Bounded background capture/inference shared by the CLI and optional Qt UI."""
from __future__ import annotations

from collections import deque
import queue
import threading
from time import perf_counter

from ..sources import CameraSource, SourceError
from .backend import video_packets


class SLRTCaptureSession:
    def __init__(self,backend,*,camera=0,video=None,pose_npz=None,event_sink=None,window_sink=None,source_factory=None):
        self.backend = backend
        self.config = backend.config
        self.camera, self.video, self.pose_npz = camera,video,pose_npz
        self._event_sink,self._window_sink = event_sink,window_sink
        self._source_factory = source_factory
        self._frames = queue.Queue(maxsize=self.config.queue_size)
        self._events = deque(maxlen=128)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self._paused = threading.Event()
        self._worker = self._capture = None
        self._epoch = self._processing_epoch = 0
        self._reset_pending = False
        self.state = 'idle'
        self.error = None
        self.queue_dropped = 0
        self.events_dropped = 0
        self.frames_captured = 0
        self.max_queue_depth = 0
        self._preview = None
        self._result = None
        self._stream = None
        self._source_opened = self._source_released = False
        self._started_at = self._first_final_wall_ms = None

    @property
    def is_running(self):
        return any(t is not None and t.is_alive() for t in (self._worker,self._capture))

    def start(self):
        with self._lock:
            if self.is_running:
                return
            self._stop.clear();self._capture_done.clear();self._paused.clear()
            self.error = None
            self._result = None
            self._preview = None
            self.frames_captured = self.queue_dropped = self.events_dropped = self.max_queue_depth = 0
            self._first_final_wall_ms = None
            self._reset_pending = False
            self._capture = None
            self._source_opened = self._source_released = False
            self._drain_frames()
            self.state = 'loading'
            self._started_at = perf_counter()
            self._worker = threading.Thread(target=self._run,name='slrt-inference',daemon=True)
            self._worker.start()

    def _publish(self,event):
        with self._lock:
            if self._processing_epoch != self._epoch:
                return
            event = dict(event,epoch=self._epoch,wall_elapsed_ms=(perf_counter()-self._started_at)*1000)
            if event['status'] == 'final' and self._first_final_wall_ms is None:
                self._first_final_wall_ms = event['wall_elapsed_ms']
            if len(self._events) == self._events.maxlen:
                self.events_dropped += 1
            self._events.append(event)
            if self._event_sink:
                self._event_sink(event)

    def _publish_window(self,value):
        with self._lock:
            if self._processing_epoch == self._epoch and self._window_sink:
                self._window_sink(dict(value,epoch=self._epoch))

    def _capture_loop(self):
        source = packets = None
        try:
            if self._source_factory is not None:
                source = self._source_factory()
                source.open()
                self._source_opened = True
            elif self.video is None:
                source = CameraSource(self.camera)
                source.open()
                self._source_opened = True
            else:
                packets = video_packets(self.video)
            first_ts = first_wall = None
            while not self._stop.is_set():
                with self._lock:
                    read_epoch = self._epoch
                try:
                    packet = next(packets) if packets is not None else source.read()
                except StopIteration:
                    break
                if packet is None:
                    break
                self._source_opened = True
                if first_ts is None:
                    first_ts,first_wall = packet.timestamp,perf_counter()
                if packets is not None and self._stop.wait(max(0,first_wall+packet.timestamp-first_ts-perf_counter())):
                    break
                from .preprocess import canonical_frame
                # Small preview is independent of the raw inference frame queue.
                preview = canonical_frame(packet.image)
                with self._lock:
                    self._preview = preview
                    self.frames_captured += 1
                    epoch = read_epoch
                if self._paused.is_set():
                    continue
                # Bound pixels in the queue as well as packet count; 512-square
                # retains the official HRNet input reference without camera 4K.
                if packet.image.shape[:2] != (512,512):
                    from PIL import Image
                    from ..sources import FramePacket
                    import numpy as np
                    frame = np.asarray(Image.fromarray(packet.image[...,::-1]).resize((512,512),Image.Resampling.BILINEAR))[...,::-1].copy()
                    packet = FramePacket(frame,packet.timestamp,packet.frame_index,packet.source_id)
                try:
                    self._frames.put_nowait((epoch,packet))
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                        self.queue_dropped += 1
                    except queue.Empty:
                        pass
                    try:
                        self._frames.put_nowait((epoch,packet))
                    except queue.Full:
                        self.queue_dropped += 1
                self.max_queue_depth = max(self.max_queue_depth,self._frames.qsize())
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            self._stop.set()
        finally:
            try:
                if source is not None:
                    source.close()
                if packets is not None:
                    packets.close()
                self._source_released = True
            except Exception as error:
                self.error = f'Capture release failed: {type(error).__name__}: {error}'
                self._stop.set()
            self._capture_done.set()

    def _run(self):
        stream = None
        try:
            stream = self.backend.new_stream(pose_npz=self.pose_npz,video_path=self.video,
                                            event_sink=self._publish,window_sink=self._publish_window)
            self._stream = stream
            if self._stop.is_set():
                return
            self._capture = threading.Thread(target=self._capture_loop,name='slrt-capture',daemon=True)
            self._capture.start()
            self.state = 'paused' if self._paused.is_set() else 'running'
            applied_epoch = self._epoch
            while not self._stop.is_set():
                with self._lock:
                    epoch = self._epoch
                    reset_pending,self._reset_pending = self._reset_pending,False
                self._processing_epoch = epoch
                if epoch != applied_epoch or reset_pending:
                    # Even pause+resume arriving during one native call creates
                    # a discontinuity. A reset cannot be overwritten by them.
                    if reset_pending:
                        stream.reset()
                        self._first_final_wall_ms = None
                    else:
                        stream.interrupt('control_boundary')
                    if self._paused.is_set():
                        stream.pause()
                    else:
                        stream.resume()
                    applied_epoch = epoch
                    with self._lock:
                        if epoch == self._epoch:
                            self._result = stream.result()
                if self._paused.is_set():
                    self._stop.wait(.03)
                    continue
                try:
                    packet_epoch,packet = self._frames.get(timeout=.05)
                except queue.Empty:
                    if self._capture_done.is_set():
                        stream.flush()
                        break
                    continue
                if packet_epoch != epoch:
                    continue
                stream.feed(packet)
                with self._lock:
                    if self._processing_epoch == self._epoch:
                        self._result = stream.result()
            self.state = 'error' if self.error else 'stopped'
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            self.state = 'error'
        finally:
            self._stop.set()
            if self._capture is not None:
                self._capture.join(timeout=5)
                if self._capture.is_alive():
                    self.error = 'Camera read did not return within 5 seconds; native capture is still releasing'
                    self.state = 'error'
            if stream is not None:
                # Manual stop cancels the active tail. Only natural source EOF
                # runs EOS padding/flush; errors never manufacture final tokens.
                if self._stop.is_set() and not stream._eos:
                    stream.interrupt('stopped')
                with self._lock:
                    if self._reset_pending:
                        # reset followed immediately by stop must not resurrect
                        # the previous session's finalized history.
                        stream.reset()
                        self._reset_pending = False
                        self._first_final_wall_ms = None
                        self._result = stream.result()
                    elif self._processing_epoch == self._epoch:
                        self._result = stream.result()
                    elif self._result is not None:
                        # Preserve only the last published generation. A native
                        # call completed after stop/pause cannot commit its tail.
                        self._result['partial'] = None
                    else:
                        stream.reset()
                        self._result = stream.result()
                stream.close()
            self._stream = None
            self._drain_frames()
            if self.error or self.state not in {'error','stopped'}:
                self.state = 'error' if self.error else 'stopped'

    def _drain_frames(self):
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def _control(self,command):
        with self._lock:
            self._epoch += 1
            if command == 'reset':
                self._reset_pending = True
            self._events.clear()
            self._drain_frames()

    def pause(self):
        self._paused.set();self._control('pause');self.state = 'paused'

    def resume(self):
        self._control('resume');self._paused.clear();self.state = 'running'

    def reset(self):
        self._control('reset')
        with self._lock:
            self._result = None

    def request_stop(self):
        with self._lock:
            if not self._stop.is_set():
                self._epoch += 1
                if self._result is not None:
                    self._result['partial'] = None
                self._stop.set()

    def stop(self,timeout=30):
        self.request_stop()
        if threading.current_thread() in (self._worker,self._capture):
            return
        deadline = perf_counter()+timeout
        for thread in (self._worker,self._capture):
            if thread is not None:
                thread.join(timeout=max(0,deadline-perf_counter()))
        if self.is_running:
            raise RuntimeError('SLRT capture/inference is still releasing; keep the backend alive until native calls return')

    def preview(self):
        with self._lock:
            return None if self._preview is None else self._preview.copy()

    def poll(self):
        with self._lock:
            out = list(self._events);self._events.clear()
            return out

    def result(self):
        with self._lock:
            from copy import deepcopy
            return dict(deepcopy(self._result or {}),state=self.state,error=self.error,
                        frames_captured=self.frames_captured,queue_dropped=self.queue_dropped,
                        max_queue_depth=self.max_queue_depth,event_queue_dropped=self.events_dropped,
                        first_stable_wall_latency_ms=self._first_final_wall_ms,
                        capture_source_opened=self._source_opened,source_release_confirmed=self._source_released,
                        capture_thread_alive=self._capture is not None and self._capture.is_alive(),
                        inference_thread_alive=self._worker is not None and self._worker.is_alive(),
                        queue_capacity=self.config.queue_size,epoch=self._epoch)
