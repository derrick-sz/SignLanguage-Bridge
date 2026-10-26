"""Bounded, unconfirmed rolling-window evidence for later semantic assistance.

This is a sequence of model revisions, not a segmented utterance. It contains
text and metadata only. The owning session serializes access with its lock.
"""
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import math

from .captions import CaptionCatalog, caption_from_result


class CandidateLedger:
    def __init__(self, session_id, *, max_history=200):
        self._validate_limit(max_history)
        self.session_id = session_id
        self.max_history = max_history
        self.revision = 0
        self._sequence = 0
        self._candidates = deque()
        self.dropped_count = 0

    @staticmethod
    def _validate_limit(value):
        if type(value) is not int or not 1 <= value <= 1000:
            raise ValueError('Candidate retention must be an integer from 1 to 1000')

    def _trim(self):
        while len(self._candidates) > self.max_history:
            self._candidates.popleft()
            self.dropped_count += 1

    def set_limit(self, value):
        self._validate_limit(value)
        if value != self.max_history:
            self.max_history = value
            self._trim()
            self.revision += 1

    def clear(self):
        self._candidates.clear()
        self.dropped_count = 0
        self.revision += 1

    def append(self, update, *, source_id, source_epoch, model_generation,
               window_revision, catalog=None):
        response = update.response
        if (response.session_id != self.session_id or response.source_epoch != source_epoch
                or response.model_generation != model_generation
                or response.result.status != 'recognized'):
            return False
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in (update.start_time, update.end_time))
                or update.end_time < update.start_time):
            raise ValueError('Candidate window times must be finite and ordered')
        catalog = catalog or CaptionCatalog()
        result = response.result
        previous = self._candidates[-1] if self._candidates else None
        overlaps = None
        if previous is not None:
            binding = (source_id, source_epoch, model_generation, window_revision, result.sign_language)
            old_binding = tuple(previous[k] for k in
                ('source_id', 'source_epoch', 'model_generation', 'window_revision', 'sign_language'))
            if binding == old_binding and update.start_time < previous['end_time']:
                overlaps = previous['candidate_id']
        self._sequence += 1
        caption = caption_from_result(result, catalog, epoch=source_epoch)
        entry = (catalog.labels.get(caption.label_id)
                 if catalog.sign_language == result.sign_language else None)
        item = dict(candidate_id=f'{self.session_id}:candidate-{self._sequence}',
            sequence=self._sequence, status='unconfirmed', source_id=source_id,
            source_epoch=source_epoch, window_revision=window_revision,
            start_time=update.start_time, end_time=update.end_time,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            overlaps_candidate_id=overlaps, sign_language=result.sign_language,
            token_time_origin='window_start', catalog_id=catalog.catalog_id,
            catalog_review_status=entry.review_status if entry else 'unmapped',
            caption=caption.to_dict(), window_age_ms=update.age_seconds*1000,
            **{key: value for key, value in response.to_dict().items()
               if key not in {'schema_version', 'session_id', 'source_epoch'}})
        self._candidates.append(deepcopy(item))
        self._trim()
        self.revision += 1
        return True

    def snapshot(self, *, after_revision=None):
        if after_revision == self.revision:
            return None
        return dict(schema_version=1, kind='recognition_candidate_context',
            session_id=self.session_id, revision=self.revision, max_history=self.max_history,
            dropped_count=self.dropped_count,
            evidence_policy=dict(status='unconfirmed', overlapping_windows=True,
                sentence_boundaries='unknown', scores='model_specific_not_calibrated_confidence',
                token_time_origin='window_start', automatic_translation=False,
                omitted_windows='rejected_cancelled_stale_or_not_scheduled'),
            candidates=deepcopy(list(self._candidates)))
