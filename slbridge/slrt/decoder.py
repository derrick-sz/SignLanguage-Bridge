"""Incremental majority vote and run compression, with bounded working state."""
from __future__ import annotations

from collections import deque
import math
import numpy as np


class IncrementalGlossDecoder:
    """Center vote requires future *arrived* windows; only EOS repeats the edge.

    final means the raw run has ended, never human semantic confirmation.
    Scores are the mean ensemble probability of the voted class, uncalibrated.
    """
    def __init__(self, vocabulary, *, vote_size=7):
        if not vocabulary or vocabulary[0] != '<blank>':
            raise ValueError('Decoder requires blank at index 0')
        if type(vote_size) is not int or vote_size < 1 or vote_size % 2 == 0:
            raise ValueError('vote_size must be positive and odd')
        self.vocabulary = tuple(vocabulary)
        self.vote_size = vote_size
        self.reset()

    def reset(self):
        self._pending = deque()
        self._left = deque(maxlen=self.vote_size // 2)
        self._first = self._last = self._active = None
        self._next_id = 0
        self._flushed = False

    def push(self, probabilities, *, anchor_time, available_time, evidence_range):
        if self._flushed:
            raise ValueError('reset decoder before feeding after EOS')
        p = np.asarray(probabilities, dtype=np.float32)
        if (p.shape != (len(self.vocabulary),) or not np.isfinite(p).all() or
                (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1, atol=1e-4)):
            raise ValueError('Expected actual normalized class probabilities')
        if not all(math.isfinite(t) for t in (anchor_time, available_time, *evidence_range)) or available_time < anchor_time:
            raise ValueError('Prediction timestamps must be finite and ordered')
        if self._last is not None and anchor_time <= self._last['time']:
            raise ValueError('Prediction anchors must strictly increase')
        item = dict(prob=p.copy(), token=int(p.argmax()), time=float(anchor_time), available=float(available_time), evidence=tuple(evidence_range))
        self._first = self._first or item
        self._last = item
        self._pending.append(item)
        return self._drain(eos=False)

    def _drain(self, *, eos):
        events = []
        half = self.vote_size // 2
        while self._pending and (eos or len(self._pending) > half):
            center = self._pending[0]
            left = list(self._left)
            left = [self._first] * (half - len(left)) + left
            right = list(self._pending)[:half+1]
            right += [self._last] * (half + 1 - len(right))
            window = left + right
            counts = np.bincount([r['token'] for r in window], minlength=len(self.vocabulary))
            k = int(counts.argmax()) if counts.max() > self.vote_size // 2 else 0
            p = float(np.mean([r['prob'][k] for r in window]))
            available = max(r['available'] for r in window)
            if self._active is not None and self._active['token_id'] != k:
                events.append(self._event('final', available, 'run_ended'))
                self._active = None
            if k != 0:
                if self._active is None:
                    self._active = dict(event_id=self._next_id, token_id=k, gloss=self.vocabulary[k],
                                        time_range=[center['time'], center['time']],
                                        evidence_time_range=[min(r['evidence'][0] for r in window),max(r['evidence'][1] for r in window)], score_sum=0., count=0)
                    self._next_id += 1
                self._active['time_range'][1] = center['time']
                self._active['evidence_time_range'][0] = min(self._active['evidence_time_range'][0],min(r['evidence'][0] for r in window))
                self._active['evidence_time_range'][1] = max(self._active['evidence_time_range'][1], max(r['evidence'][1] for r in window))
                self._active['score_sum'] += p
                self._active['count'] += 1
                events.append(self._event('partial', available, 'vote'))
            self._left.append(self._pending.popleft())
        return events

    def _event(self, status, available, reason):
        a = self._active
        return {k: a[k].copy() if isinstance(a[k], list) else a[k] for k in ('event_id','token_id','gloss','time_range','evidence_time_range')} | {
            'status': status, 'available_time': available, 'score': a['score_sum'] / a['count'],
            'score_type': 'uncalibrated_mean_ensemble_probability',
            'boundary_source': 'sliding_window_anchor_estimate', 'reason': reason,
        }

    def flush(self):
        if self._flushed:
            return []
        events = self._drain(eos=True)
        if self._active is not None:
            events.append(self._event('final', self._last['available'], 'eos'))
            self._active = None
        self._flushed = True
        return events
