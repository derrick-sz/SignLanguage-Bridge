"""SLRT live words and Uni-Sign sentence translation over one pose stream."""
from __future__ import annotations

import time
from collections import deque
import numpy as np

from .runtime_contracts import InferenceRequest
from .slrt.stream import SLRTStream
from .realtime_segment import UtteranceBuffer
from .realtime_review import expansion_hint


class _PoseTap:
    def __init__(self, frontend):
        self.inner = frontend
        self.info = frontend.info
        self.last = None
        if callable(getattr(frontend,'extract_prepared',None)):
            self.extract_prepared=self._extract_prepared
    def _extract_prepared(self,*args,**kwargs):
        frame,self.last=self.inner.extract_prepared(*args,**kwargs)
        return frame,self.last
    def extract(self, *args, **kwargs):
        self.last = self.inner.extract(*args, **kwargs)
        return self.last
    def close(self):
        self.inner.close()


class RealtimeEngine:
    def __init__(self, session, runtime, frontend, config, *, translator=None, fps=30,
                 pause_seconds=.8, max_seconds=8, assister=None, model_info=None):
        self.session, self.translator, self.assister = session, translator, assister
        self.frontend = _PoseTap(frontend)
        self.tracker = UtteranceBuffer(fps=fps, pause_seconds=pause_seconds, max_seconds=max_seconds)
        self.stream = SLRTStream(runtime, self.frontend, config,
                                 event_sink=self._word, window_sink=self._window)
        self.info = dict(model_info or {}, sign_language='csl', task='incremental_gloss_with_sentence_translation',
                         input_protocol='shared_rtmpose133_square320_ref512_v1',
                         sentence_translation=translator is not None)
        self.utterance_id = 0
        self._words = {}
        self._current = self._previous = None
        self._changed = self._blank = self._flushing = self._quality_lost = False
        self._overflow_noticed = False
        self.frames = self.gaps = self.sentences = 0
        self.review_hints = 0
        self._timings = {}
        self._timing_cursor = None
        self.canonical_frames_reused = 0
        self._translations = deque()
        self.translation_deferred = self.translation_rejected = 0
        self.translation_queue_peak = 0
        self.translation_wait_max_ms = 0.

    def _word(self, event):
        key = str(event['event_id'])
        if event['status'] == 'retracted' and key not in self._words:
            return
        if not self.tracker.active and not self._flushing and event['status'] != 'retracted':
            return
        if event['status'] == 'retracted':
            self._words.pop(key, None)
        else:
            self._changed |= key not in self._words
            self._words[key] = dict(event)
        self.session.emit('word', utterance_id=self.utterance_id,
                          input_frame_index=self._current.packet.frame_index if self._current else None,
                          client_capture_ms=self._current.capture_ms if self._current else None, **event)

    def _window(self, value):
        self._blank = value['top_id'] == 0

    def _collect_timings(self):
        current = self.stream.result()
        previous = self._timing_cursor
        if previous is None or previous['session_id'] != current['session_id']:
            previous = {'timings': {}, 'canonical_frames_reused': 0}
        for name, row in current['timings'].items():
            total, count = self._timings.get(name, (0.,0))
            old = previous['timings'].get(name, {'total': 0., 'count': 0})
            self._timings[name] = total+row['total']-old['total'], count+row['count']-old['count']
        self.canonical_frames_reused += current['canonical_frames_reused']-previous['canonical_frames_reused']
        self._timing_cursor = {k: current[k] for k in ('session_id','timings','canonical_frames_reused')}

    def performance(self):
        """Worker-owned snapshot; repeated reads must not add the same timings."""
        self._collect_timings()
        return dict(frames_processed=self.frames, dropped_input_frames=self.gaps,
             translation_deferred=self.translation_deferred, translation_rejected=self.translation_rejected,
             translation_queue_peak=self.translation_queue_peak, translation_wait_max_ms=self.translation_wait_max_ms,
             canonical_frames_reused=self.canonical_frames_reused,
             review_hints=self.review_hints,
             sentences=self.sentences, timings={k: dict(total_ms=v[0], count=v[1],
             mean_ms=v[0]/v[1] if v[1] else None) for k,v in self._timings.items()})

    def _reset(self, reason, message):
        self._collect_timings()
        self.stream.reset()
        self.tracker.reset()
        self._words.clear()
        self._blank = False
        self.session.emit('utterance_reset', utterance_id=self.utterance_id, reason=reason, message=message)
        self.utterance_id += 1

    def feed(self, decoded):
        packet = decoded.packet
        self._current = decoded
        self.frames += 1
        if self._previous is not None and (packet.frame_index != self._previous.frame_index+1 or
                                           packet.timestamp-self._previous.timestamp > .5):
            self.gaps += max(0, packet.frame_index-self._previous.frame_index-1)
            self._reset('input_discontinuity', '画面帧出现中断，当前未完成句已作废。')
        self._previous = packet
        self._changed = False
        self.stream.feed(packet)
        points = self.frontend.last
        if points is None:
            if not self._quality_lost:
                self._reset('pose_quality', '暂未看清单个手语者的双手，正在重新积累动作。')
            self._quality_lost = True
            self._drain_translations()
            return
        self._quality_lost = False
        segment = self.tracker.observe(packet, points, has_words=bool(self._words),
                                       word_changed=self._changed, blank=self._blank)
        if self.tracker.overflowed and not self._overflow_noticed:
            # Do not translate a truncated prefix or restart in its middle.
            self._overflow_noticed = True
            self.stream.interrupt('utterance_limit')
            self._words.clear()
            self.session.emit('utterance_reset', utterance_id=self.utterance_id, reason='utterance_limit',
                              message='这句较长，未自动补译。请分句表达或点击结束这句。')
            self.utterance_id += 1
        if not self.tracker.overflowed:
            self._overflow_noticed = False
        if segment is not None:
            self._translate(segment)
        self._drain_translations()

    def _translate(self, segment):
        if self.session.cancelled:
            return
        # A declared utterance boundary is explicit, distinct from cancellation.
        self._flushing = True
        try:
            self.stream.flush()
        finally:
            self._flushing = False
        evidence = [x['gloss'] for x in self._words.values() if x['status'] != 'retracted']
        utterance = self.utterance_id
        first, last = segment.samples[0], segment.samples[-1]
        if evidence and self.translator is not None:
            self.session.emit('sentence', utterance_id=utterance, revision=1, status='pending', text='',
                              reason=segment.reason, start_seconds=first.timestamp, end_seconds=last.timestamp)
            job=dict(utterance=utterance,segment=segment,evidence=evidence,queued=time.monotonic(),
                     points=np.stack([s.keypoints for s in segment.samples]))
            if len(self._translations)>=2:
                self._reject_translation(job,'translation_queue_limit')
            else:
                self._translations.append(job)
                self.translation_queue_peak=max(self.translation_queue_peak,len(self._translations))
                self.translation_deferred+=getattr(self.session,'pending_frames',0)>0
        self._collect_timings()
        self.stream.reset()
        self._words.clear()
        self.utterance_id += 1
        self._blank = False
        self._drain_translations()

    def _reject_translation(self,job,reason):
        self.translation_rejected+=1
        self.session.emit('sentence',utterance_id=job['utterance'],revision=1,status='rejected',text='',
                          reason=reason,message='本句翻译未及时完成，请核对实时词或分句重试。')

    def _drain_translations(self,*,force=False):
        while self._translations and not self.session.cancelled:
            job=self._translations[0]
            wait_ms=(time.monotonic()-job['queued'])*1000
            self.translation_wait_max_ms=max(self.translation_wait_max_ms,wait_ms)
            if wait_ms>=1500:
                self._translations.popleft();self._reject_translation(job,'translation_wait_limit');continue
            # Keep one native/GPU owner. Service accumulated input before spending
            # the next burst on Uni-Sign; do not enlarge the input queue to hide debt.
            if not force and getattr(self.session,'pending_frames',0)>0:
                return
            self._translations.popleft()
            utterance,segment,evidence=job['utterance'],job['segment'],job['evidence']
            first,last=segment.samples[0],segment.samples[-1]
            request = InferenceRequest(f'live-{self.session.session_id}-{utterance}', sign_language='csl',
                 session_id=self.session.session_id, source_epoch=self.session.source_epoch,
                 expected_task_type='text', cancellation=self.session.cancellation,
                 deadline_monotonic=time.monotonic()+10)
            points = job['points']
            started = time.perf_counter()
            text, details = self.translator(points[:,:,:2], points[:,:,2], width=512, height=512, request=request)
            request.check_active()
            hint = expansion_hint(evidence,text)
            self.session.emit('sentence', utterance_id=utterance, revision=1,
                 status='committed' if text else 'rejected', text=text, predicted_gloss=evidence,
                 review_hint=hint,
                 reason=segment.reason, start_seconds=first.timestamp, end_seconds=last.timestamp,
                 frame_start=first.index, frame_end=last.index,
                 translation_wait_ms=wait_ms,
                 translation_ms=(time.perf_counter()-started)*1000,
                 input_protocol=self.info['input_protocol'], details=details)
            self.sentences += bool(text)
            self.review_hints += bool(hint)
            if self.assister is not None and text:
                try:
                    self.assister.submit(utterance_id=utterance, revision=1, glosses=evidence, text=text)
                except Exception:
                    self.session.emit('assistance',utterance_id=utterance,revision=1,status='unavailable',
                                      text='本次辅助核对未完成，原译文已保留。')

    def boundary(self):
        segment = self.tracker.take('manual')
        if segment is not None:
            self._translate(segment)
        else:
            self._reset('manual_empty', '本句未积累到有效动作，已重新开始。')

    def idle(self):
        self._drain_translations()

    def finish(self):
        segment = self.tracker.take('natural_eof')
        if segment is not None:
            self._translate(segment)
        self._drain_translations(force=True)
        self.session.emit('summary', **self.performance())

    def cancel(self):
        self._translations.clear()
        self.tracker.reset()
        self.stream.interrupt('cancelled')
        self._words.clear()

    def close(self):
        self._translations.clear()
        self.stream.close()
