"""Ordered, bounded caption events. Identity belongs to an action, never its text."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import threading
from uuid import uuid4
from urllib.parse import quote

from .captions import CaptionCatalog, caption_from_result


class CaptionLedger:
    def __init__(self, session_id=None, *, max_history=200, max_pending=2, max_seen=1024):
        for value in (max_history, max_pending, max_seen):
            if type(value) is not int or value < 1:
                raise ValueError("Caption capacities must be positive integers")
        if session_id is not None and (not isinstance(session_id, str) or not session_id or len(session_id) > 128):
            raise ValueError("Invalid caption session ID")
        self.session_id = session_id or uuid4().hex
        self.max_history, self.max_pending, self.max_seen = max_history, max_pending, max_seen
        self.source_epoch = 0
        self.model_generation = 0
        self.revision = 0
        self.last_sequence = 0
        self._pending = OrderedDict()
        self._seen = OrderedDict()
        self._actions = OrderedDict()
        self._events = []
        self._lock = threading.RLock()

    def bind(self, source_epoch, model_generation):
        if any(type(v) is not int or v < 0 for v in (source_epoch, model_generation)):
            raise ValueError("Invalid caption binding")
        with self._lock:
            changed = (source_epoch, model_generation) != (self.source_epoch, self.model_generation)
            if changed:
                self.cancel()
                self.source_epoch, self.model_generation = source_epoch, model_generation
            return changed

    def lookup(self, segment_id):
        with self._lock:
            value = self._pending.get(segment_id) or self._seen.get(segment_id)
            return deepcopy(value) if value else None

    def reserve(self, *, segment_id, sequence, request_id, source_epoch, model_generation,
                start_time, end_time, source_id):
        for value in (segment_id, request_id, source_id):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("Caption identifiers must be nonempty and bounded")
        if type(sequence) is not int or sequence < 1:
            raise ValueError("Caption sequence must be a positive integer")
        if any(type(v) is not int or v < 0 for v in (source_epoch, model_generation)):
            raise ValueError("Invalid caption binding")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
               for v in (start_time, end_time)) or end_time < start_time:
            raise ValueError("Caption time boundaries must be finite and ordered")
        metadata = dict(segment_id=segment_id, sequence=sequence, request_id=request_id,
                        source_epoch=source_epoch, model_generation=model_generation,
                        start_time=start_time, end_time=end_time, source_id=source_id)
        with self._lock:
            existing = self._pending.get(segment_id) or self._seen.get(segment_id)
            if existing:
                if any(existing[k] != v for k, v in metadata.items()):
                    raise ValueError("Segment ID already belongs to another action")
                return existing["event_id"]
            if (source_epoch, model_generation) != (self.source_epoch, self.model_generation):
                raise ValueError("Caption binding is stale")
            if sequence <= self.last_sequence:
                raise ValueError("Segment sequence has expired or is out of order")
            if len(self._pending) >= self.max_pending:
                raise RuntimeError("Caption candidate capacity reached; retry the action")
            self.last_sequence = sequence
            event_id = f"{quote(self.session_id, safe='')}:{quote(segment_id, safe='')}"
            self._pending[segment_id] = dict(metadata, event_id=event_id, status="pending")
            return event_id

    def complete(self, response, catalog: CaptionCatalog | None = None):
        with self._lock:
            item = self._pending.get(response.segment_id)
            if not item or item["status"] != "pending":
                return False
            if (response.session_id != self.session_id or response.request_id != item["request_id"]
                    or response.source_epoch != item["source_epoch"]
                    or response.model_generation != item["model_generation"]
                    or (item["source_epoch"], item["model_generation"]) !=
                    (self.source_epoch, self.model_generation)):
                return False
            result = response.result
            if result.status != "recognized":
                return self.reject(response.segment_id, result.reason or "rejected")
            caption = caption_from_result(result, catalog, epoch=item["source_epoch"])
            item.update(status="completed", result=result.to_dict(), caption=caption.to_dict(),
                        model_id=response.model_id, model_version=response.model_version,
                        preprocessing_version=response.preprocessing_version,
                        runtime=deepcopy(response.runtime), timing_ms=dict(response.timing_ms))
            self._drain()
            return True

    def reject(self, segment_id, reason):
        with self._lock:
            item = self._pending.get(segment_id)
            if not item or item["status"] != "pending":
                return False
            item.update(status="rejected", reason=reason)
            self._drain()
            return True

    def _remember(self, item):
        self._seen[item["segment_id"]] = deepcopy(item)
        while len(self._seen) > self.max_seen:
            self._seen.popitem(last=False)

    def _drain(self):
        while self._pending:
            key, item = next(iter(self._pending.items()))
            if item["status"] == "pending":
                break
            self._pending.pop(key)
            if item["status"] == "completed":
                item["status"] = "committed"
                self._events.append(deepcopy(item))
                self._events = self._events[-self.max_history:]
            self._remember(item)

    def cancel(self):
        with self._lock:
            # Completed but order-blocked candidates are also invalidated.
            for item in self._pending.values():
                item.update(status="cancelled", reason="cancelled")
            self._drain()

    def _action(self, action_id, kind):
        if action_id is None:
            return False
        if not isinstance(action_id, str) or not action_id or len(action_id) > 128:
            raise ValueError("Invalid action ID")
        if action_id in self._actions:
            if self._actions[action_id] != kind:
                raise ValueError("Action ID reused for a different operation")
            return True
        if len(self._actions) >= self.max_seen:
            raise RuntimeError("Caption action capacity reached; create a new session")
        self._actions[action_id] = kind
        return False

    def undo(self, action_id=None):
        with self._lock:
            if self._action(action_id, "undo"):
                return None
            return deepcopy(self._events.pop()) if self._events else None

    def clear(self, action_id=None):
        with self._lock:
            if self._action(action_id, "clear"):
                return
            self.cancel()
            self._events.clear()
            self.revision += 1

    def snapshot(self):
        with self._lock:
            return deepcopy(dict(session_id=self.session_id, source_epoch=self.source_epoch,
                                 model_generation=self.model_generation, revision=self.revision,
                                 last_sequence=self.last_sequence, events=self._events,
                                 candidates=list(self._pending.values())))
