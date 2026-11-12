"""A bounded, timestamped online window state machine, independent of capture/UI."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4
import numpy as np

from ..sources import FramePacket
from .assets import PROTOCOL
from .decoder import IncrementalGlossDecoder
from .preprocess import canonical_frame


@dataclass
class _Frame:
    offset: int
    packet: FramePacket
    bgr: np.ndarray
    keypoints: np.ndarray


class SLRTStream:
    """Feed only arrived frames. Complete-window S3D, not recurrent feature caching.

    The trained 16-frame protocol has 7 frames of left and 8 of right context.
    Right edge replication is allowed only by explicit flush at source EOF.
    """
    def __init__(self, runtime, frontend, config, *, event_sink=None, window_sink=None):
        self.runtime, self.frontend, self.config = runtime, frontend, config
        self.event_sink, self.window_sink = event_sink, window_sink
        self.session_id = uuid4().hex
        self._frames = deque(maxlen=config.window_size)
        self._history = deque(maxlen=config.max_gloss_history)
        self._timings = {k: deque(maxlen=4096) for k in ('pose_ms','frame_preprocess_ms','preprocess_ms','inference_ms','window_total_ms')}
        self._timing_totals = {k: 0. for k in self._timings}
        self._timing_counts = {k: 0 for k in self._timings}
        self.decoder = IncrementalGlossDecoder(runtime.vocabulary, vote_size=config.vote_size)
        self.frames_received = self.windows_processed = self.quality_rejected_frames = self.dropped_frames = 0
        self._segment = -1
        self._last_packet = None
        self._closed = self._eos = self._paused = False
        self.first_output_latency_ms = None
        self._first_source_time = None
        self._partial = None
        self._history_truncated = 0
        self.canonical_frames_reused = 0
        self._reset_window()

    def _reset_window(self):
        self._frames.clear()
        self._received = self._next_anchor = 0
        self._segment += 1
        self.decoder.reset()

    def _measure(self, name, value):
        self._timings[name].append(value)
        self._timing_totals[name] += value
        self._timing_counts[name] += 1

    def _emit(self, events):
        out = []
        for e in events:
            e = dict(e, session_id=self.session_id, segment_id=self._segment,
                     model_id=self.config.model_id, sign_language='csl',
                     available_time_source='latest_required_input_timestamp_not_wall_delivery',
                     event_id=f'{self._segment}:{e["event_id"]}')
            if e['status'] == 'partial':
                self._partial = e
            elif e['status'] == 'final':
                self._partial = None
                if len(self._history) == self._history.maxlen:
                    self._history_truncated += 1
                self._history.append(e)
                if self.first_output_latency_ms is None and self._first_source_time is not None:
                    self.first_output_latency_ms = (e['available_time'] - self._first_source_time) * 1000
            if self.event_sink:
                self.event_sink(e)
            out.append(e)
        return out

    def interrupt(self, reason):
        events = []
        if self._partial is not None:
            e = dict(self._partial, status='retracted', reason=reason)
            events.append(e)
            if self.event_sink:
                self.event_sink(e)
            self._partial = None
        self._reset_window()
        return events

    def feed(self, packet: FramePacket):
        if self._closed or self._eos:
            raise RuntimeError('SLRT stream is closed/EOS; reset before feeding')
        if self._paused:
            return []
        if not isinstance(packet, FramePacket):
            raise ValueError('SLRT feed requires a timestamped FramePacket')
        events = []
        previous = self._last_packet
        if previous is not None:
            if packet.source_id != previous.source_id:
                raise ValueError('Source changed; reset the stream explicitly')
            if packet.timestamp <= previous.timestamp or packet.frame_index <= previous.frame_index:
                raise ValueError('SLRT frames/timestamps must strictly increase')
            skipped = packet.frame_index - previous.frame_index - 1
            if skipped or packet.timestamp - previous.timestamp > self.config.max_gap_seconds:
                self.dropped_frames += skipped
                # Never silently reinterpret sparsely sampled frames as a dense clip.
                events += self.interrupt('input_discontinuity')
        self._last_packet = packet
        if self._first_source_time is None:
            self._first_source_time = packet.timestamp
        self.frames_received += 1
        started = perf_counter()
        prepared = None
        if callable(getattr(self.frontend,'extract_prepared',None)):
            prepared,kp = self.frontend.extract_prepared(packet.image,frame_index=packet.frame_index,timestamp=packet.timestamp)
        else:
            kp = self.frontend.extract(packet.image, frame_index=packet.frame_index, timestamp=packet.timestamp)
        self._measure('pose_ms', (perf_counter()-started)*1000)
        if kp is None:
            self.quality_rejected_frames += 1
            return events + self.interrupt('no_reliable_single_signer')
        started = perf_counter()
        if prepared is None:
            bgr = canonical_frame(packet.image)
        else:
            if (not isinstance(prepared,np.ndarray) or prepared.dtype!=np.uint8 or
                    prepared.shape!=(320,320,3) or not prepared.flags.c_contiguous):
                raise ValueError('Invalid prepared canonical frame')
            bgr=prepared
            bgr.setflags(write=False)
            self.canonical_frames_reused+=1
        self._measure('frame_preprocess_ms',(perf_counter()-started)*1000)
        # Drop original high-resolution pixels from the retained frame window.
        compact_packet = FramePacket(bgr,packet.timestamp,packet.frame_index,packet.source_id)
        self._frames.append(_Frame(self._received,compact_packet,bgr,np.array(kp,dtype=np.float32,copy=True)))
        self._received += 1
        right = self.config.window_size // 2
        while self._next_anchor + right < self._received:
            events += self._predict(eos=False)
        return events

    def _predict(self, *, eos):
        left = self.config.window_size // 2 - 1
        start, stop = self._next_anchor-left, self._next_anchor-left+self.config.window_size
        by_offset = {f.offset:f for f in self._frames}
        indices = [min(max(i,0),self._received-1) for i in range(start,stop)]
        if not eos and stop > self._received:
            raise RuntimeError('Attempted to access future video frames')
        try:
            frames = [by_offset[i] for i in indices]
        except KeyError as e:
            raise RuntimeError('SLRT frame buffer lost required context') from e
        anchor = by_offset[self._next_anchor]
        p,timings = self.runtime.predict(np.stack([f.bgr for f in frames]), np.stack([f.keypoints for f in frames]))
        for k,v in timings.items():
            self._measure(k,v)
        self.windows_processed += 1
        available = self._frames[-1].packet.timestamp
        evidence = (frames[0].packet.timestamp, frames[-1].packet.timestamp)
        if self.window_sink:
            self.window_sink(dict(segment_id=self._segment,anchor_frame=anchor.packet.frame_index,
                                  anchor_time=anchor.packet.timestamp,available_time=available,
                                  source_frame_indices=[f.packet.frame_index for f in frames],
                                  left_edge_repeats=max(0,-start),right_edge_repeats=max(0,stop-self._received),
                                  eos_padding=eos,top_id=int(p.argmax()),top_probability=float(p.max()),**timings))
        self._next_anchor += self.config.stride
        return self._emit(self.decoder.push(p,anchor_time=anchor.packet.timestamp,available_time=available,evidence_range=evidence))

    def flush(self):
        if self._closed or self._eos:
            return []
        events = []
        while self._frames and self._next_anchor < self._received:
            events += self._predict(eos=True)
        events += self._emit(self.decoder.flush())
        self._eos = True
        return events

    def pause(self):
        self._paused = True
        return self.interrupt('paused')

    def resume(self):
        self._paused = False
        self._last_packet = None

    def reset(self):
        if self._closed:
            raise RuntimeError('SLRT stream is closed')
        self.interrupt('reset')
        self.session_id = uuid4().hex
        self._history.clear()
        self._history_truncated = 0
        self._last_packet = self._first_source_time = None
        self._eos = self._paused = False
        self.first_output_latency_ms = None
        self.frames_received = self.windows_processed = self.quality_rejected_frames = self.dropped_frames = 0
        self.canonical_frames_reused = 0
        for k in self._timings:
            self._timings[k].clear(); self._timing_totals[k] = 0.; self._timing_counts[k] = 0

    def result(self):
        summary = {}
        for k,values in self._timings.items():
            summary[k] = dict(total=self._timing_totals[k],count=self._timing_counts[k],
                              mean=self._timing_totals[k]/self._timing_counts[k] if self._timing_counts[k] else None,
                              p50=float(np.median(values)) if values else None,
                              p95=float(np.percentile(values,95)) if values else None,
                              percentile_scope='last_4096_observations')
        return dict(schema_version=1,session_id=self.session_id,sign_language='csl',model_name=self.config.model_id,
                    status='final' if self._eos else 'partial',gloss_sequence=[t['gloss'] for t in self._history],
                    tokens=list(self._history),partial=self._partial,translation=None,
                    frames_received=self.frames_received,windows_processed=self.windows_processed,
                    quality_rejected_frames=self.quality_rejected_frames,dropped_frames=self.dropped_frames,
                    canonical_frames_reused=self.canonical_frames_reused,
                    retained_frames=len(self._frames),history_truncated=self._history_truncated,
                    first_stable_required_context_ms=self.first_output_latency_ms,
                    preprocessing_protocol=PROTOCOL,config=self.config.to_dict(),
                    model=self.runtime.info,pose=self.frontend.info,timings=summary,
                    temporal_protocol='trained_16_frames' if self.config.window_size == 16 else 'experimental_window_length',
                    time_boundary_source='sliding_window_anchor_estimate',
                    padding_protocol='fixed_left_context_eos_right_repeat',
                    official_stride1_padding_equivalent=self.config.stride == 1)

    def close(self):
        if self._closed:
            return
        self.interrupt('closed')
        self._closed = True
        self.frontend.close()
        self._frames.clear()
