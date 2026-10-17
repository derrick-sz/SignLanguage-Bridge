"""Bounded, lossless keypoint recordings; no sign recognition is performed.

Each recording is a header, complete landmark packets, and a verified footer.
An interrupted or rejected recording never becomes a completed output file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .keypoints import EXTRACTOR_VERSION, LandmarkFrame, POINT_COUNT, SCHEMA_ID, TIME_DOMAINS


FORMAT_ID = "slbridge.keypoints.jsonl"
FORMAT_VERSION = 1
_IDENTITY_FIELDS = ("schema_id", "session_id", "source_id", "source_epoch", "time_domain")


class RecordingError(ValueError):
    """A recording is incomplete, malformed, incompatible, or over its limit."""


class RecordingLimitError(RecordingError):
    """A whole recording was rejected instead of being silently truncated."""


@dataclass(frozen=True)
class RecordingLimits:
    max_frames: int = 100_000
    max_bytes: int = 256 * 1024 * 1024
    max_line_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_frames", "max_bytes", "max_line_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _json_value(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise RecordingError("JSON metadata exceeds the nesting limit")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item, depth + 1)
        return
    raise RecordingError("metadata must contain finite JSON values and string keys")


def _encode(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RecordingError(f"cannot serialize recording JSON: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecordingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               RecordingError(f"non-finite JSON number: {value}")))
    except (UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        raise RecordingError(f"invalid recording JSON: {exc}") from exc
    if type(value) is not dict:
        raise RecordingError("each recording line must be a JSON object")
    return value


def _identity(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {name: packet[name] for name in _IDENTITY_FIELDS}


def _check_next(previous: LandmarkFrame | None, current: LandmarkFrame,
                identity: dict[str, Any]) -> None:
    if _identity(current.to_dict()) != identity:
        raise RecordingError("recording source, session, epoch, clock, or schema changed")
    if previous is not None:
        if current.timestamp <= previous.timestamp or current.frame_index <= previous.frame_index:
            raise RecordingError("recording timestamps and frame indices must strictly increase")
        if current.tracking_epoch < previous.tracking_epoch:
            raise RecordingError("tracking_epoch must not decrease")


def _metadata_copy(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise RecordingError("metadata must be a JSON object")
    data = dict(metadata)
    _json_value(data)
    copied = _decode(_encode(data))
    annotations = copied.get("annotations")
    if annotations is not None:
        if type(annotations) is not dict:
            raise RecordingError("annotations must be a JSON object")
        # Collection labels are drafts. This format provides no review workflow.
        annotations["review_status"] = "draft"
    _check_extractor(copied)
    return copied


def _check_extractor(metadata: dict[str, Any], identity: dict[str, Any] | None = None) -> None:
    extractor = metadata.get("extractor")
    required = {"schema_id", "extractor_version", "point_count", "coordinate_dimensions",
                "session_id", "source_id", "source_epoch", "tracking_epoch", "time_domain",
                "protocol", "protocol_hash"}
    if type(extractor) is not dict or not required.issubset(extractor):
        raise RecordingError("metadata must contain the extractor identity and protocol")
    if (extractor.get("schema_id") != SCHEMA_ID
            or type(extractor.get("extractor_version")) is not int
            or extractor["extractor_version"] != EXTRACTOR_VERSION
            or type(extractor.get("point_count")) is not int or extractor["point_count"] != POINT_COUNT
            or type(extractor.get("coordinate_dimensions")) is not int
            or extractor["coordinate_dimensions"] != 2
            or type(extractor.get("session_id")) is not str or not extractor["session_id"].strip()
            or type(extractor.get("source_epoch")) is not int or extractor["source_epoch"] < 0
            or type(extractor.get("tracking_epoch")) is not int or extractor["tracking_epoch"] < 0
            or extractor.get("time_domain") not in TIME_DOMAINS
            or (extractor.get("source_id") is not None and (
                type(extractor["source_id"]) is not str or not extractor["source_id"].strip()))):
        raise RecordingError("unsupported or malformed extractor identity")
    protocol = extractor.get("protocol")
    if type(protocol) is not dict:
        raise RecordingError("extractor protocol must be a JSON object")
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    if extractor.get("protocol_hash") != hashlib.sha256(canonical).hexdigest():
        raise RecordingError("extractor protocol hash does not match its protocol")
    if identity is not None and any(extractor.get(name) != identity[name] for name in _IDENTITY_FIELDS):
        raise RecordingError("extractor metadata disagrees with recording identity")


class KeypointRecorder:
    """Write one source timeline, committing only after a complete footer.

    Files are first written to a unique partial in the destination directory.
    An atomic hard link publishes the result without replacing an existing file.
    """

    def __init__(self, path: str | Path, *, metadata: Mapping[str, Any],
                 max_frames: int = 100_000, max_bytes: int = 256 * 1024 * 1024):
        self.path = Path(path)
        self.limits = RecordingLimits(max_frames=max_frames, max_bytes=max_bytes)
        self._metadata = _metadata_copy(metadata)
        if len(_encode(self._metadata)) > self.limits.max_line_bytes:
            raise RecordingLimitError("recording metadata exceeds the line limit")
        if self.path.exists():
            raise FileExistsError(f"recording output already exists: {self.path}")
        self._partial = self.path.with_name(f".{self.path.name}.{uuid4().hex}.partial")
        self._stream = self._partial.open("xb")
        self._closed = False
        self._count = 0
        self._bytes = 0
        self._hash = hashlib.sha256()
        self._previous: LandmarkFrame | None = None
        self._identity: dict[str, Any] | None = None
        self._cleanup_error: str | None = None

    @property
    def count(self) -> int:
        return self._count

    @property
    def cleanup_error(self) -> str | None:
        """Report a cleanup failure, including a residual partial after publish."""
        return self._cleanup_error

    def _abort_after_failure(self, failure: Exception) -> None:
        try:
            self.abort()
        except Exception as cleanup:
            failure.keypoint_cleanup_error = self._cleanup_error or str(cleanup)
            if hasattr(failure, "add_note"):
                failure.add_note(f"Recording cleanup also failed: {failure.keypoint_cleanup_error}")
            raise failure from cleanup

    def _write(self, raw: bytes) -> None:
        if len(raw) > self.limits.max_line_bytes:
            raise RecordingLimitError("recording line exceeds the line limit")
        if self._bytes + len(raw) > self.limits.max_bytes:
            raise RecordingLimitError("recording exceeds the byte limit")
        self._stream.write(raw)
        self._bytes += len(raw)

    def append(self, frame: LandmarkFrame) -> None:
        if self._closed:
            raise RecordingError("recording is closed")
        try:
            if not isinstance(frame, LandmarkFrame):
                raise RecordingError("append requires a LandmarkFrame")
            # Revalidate even if a frozen instance was modified through low-level code.
            packet = frame.to_dict()
            validated = LandmarkFrame.from_dict(packet)
            if self._count >= self.limits.max_frames:
                raise RecordingLimitError("recording exceeds the frame limit")
            if self._identity is None:
                self._identity = _identity(packet)
                extractor = self._metadata["extractor"]
                if extractor["source_id"] is None:
                    extractor["source_id"] = frame.source_id
                if extractor["tracking_epoch"] > frame.tracking_epoch:
                    raise RecordingError("extractor metadata tracking epoch is ahead of its first frame")
                extractor["tracking_epoch"] = frame.tracking_epoch
                _check_extractor(self._metadata, self._identity)
                header = {"type": "header", "format": FORMAT_ID, "version": FORMAT_VERSION,
                          "schema_id": SCHEMA_ID, "identity": self._identity,
                          "metadata": self._metadata}
                self._write(_encode(header))
            _check_next(self._previous, validated, self._identity)
            raw = _encode({"type": "frame", "packet": packet})
            self._write(raw)
            self._hash.update(raw)
            self._count += 1
            self._previous = validated
        except Exception as failure:
            self._abort_after_failure(failure)
            raise

    def finish(self, reason: str = "stopped") -> Path:
        if self._closed:
            raise RecordingError("recording is closed")
        try:
            if self._count == 0:
                raise RecordingError("cannot finish an empty recording")
            if type(reason) is not str or not reason or len(reason) > 1024:
                raise RecordingError("finish reason must be a nonempty short string")
            footer = {"type": "footer", "count": self._count,
                      "frames_sha256": self._hash.hexdigest(), "reason": reason}
            self._write(_encode(footer))
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            # Same-directory hard linking is atomic and never overwrites a target.
            os.link(self._partial, self.path)
        except Exception as failure:
            self._abort_after_failure(failure)
            raise
        # Publication succeeded. Cleanup cannot turn this complete output into
        # an unsaved result, and must never delete or overwrite its target.
        self._closed = True
        try:
            self._partial.unlink(missing_ok=True)
        except Exception as cleanup:
            self._cleanup_error = (
                f"Complete recording was saved; temporary file remains at {self._partial}: "
                f"{type(cleanup).__name__}: {cleanup}")
        return self.path

    def abort(self) -> None:
        self._closed = True
        close_failure = None
        try:
            if not self._stream.closed:
                self._stream.close()
        except Exception as exc:
            close_failure = exc
        try:
            self._partial.unlink(missing_ok=True)
        except Exception as unlink_failure:
            self._cleanup_error = f"Cannot remove temporary file {self._partial}: {unlink_failure}"
            if close_failure is not None:
                self._cleanup_error = f"Cannot close recording: {close_failure}; {self._cleanup_error}"
                raise close_failure from unlink_failure
            raise
        if close_failure is not None:
            self._cleanup_error = f"Cannot close recording: {close_failure}"
            raise close_failure
        self._cleanup_error = None


class KeypointRecording:
    """Verify a complete recording, then replay its bounded packets as a stream."""

    def __init__(self, path: str | Path, *, limits: RecordingLimits | None = None):
        self.path = Path(path)
        self.limits = limits or RecordingLimits()
        if not isinstance(self.limits, RecordingLimits):
            raise TypeError("limits must be RecordingLimits")
        self._header: dict[str, Any] | None = None
        self._footer: dict[str, Any] | None = None
        self._summary: dict[str, Any] = {}
        for _ in self._scan():
            pass

    @property
    def metadata(self) -> dict[str, Any]:
        return _decode(_encode(self._header["metadata"]))

    def summary(self) -> dict[str, Any]:
        return _decode(_encode(self._summary))

    def iter_frames(self) -> Iterator[LandmarkFrame]:
        return self._scan()

    def _scan(self) -> Iterator[LandmarkFrame]:
        if self.path.stat().st_size > self.limits.max_bytes:
            raise RecordingLimitError("recording exceeds the byte limit")
        total_bytes = 0
        count = 0
        header = None
        footer = None
        previous = None
        digest = hashlib.sha256()
        first_time = last_time = None
        valid_anchors = observed_points = left_present = right_present = skipped = 0
        epochs: set[int] = set()
        with self.path.open("rb") as stream:
            while True:
                raw = stream.readline(self.limits.max_line_bytes + 1)
                if not raw:
                    break
                total_bytes += len(raw)
                if total_bytes > self.limits.max_bytes:
                    raise RecordingLimitError("recording exceeds the byte limit")
                if len(raw) > self.limits.max_line_bytes:
                    raise RecordingLimitError("recording line exceeds the line limit")
                if not raw.endswith(b"\n"):
                    raise RecordingError("recording contains an incomplete JSONL line")
                item = _decode(raw)
                kind = item.get("type")
                if header is None:
                    expected = {"type", "format", "version", "schema_id", "identity", "metadata"}
                    if (set(item) != expected or kind != "header" or item["format"] != FORMAT_ID
                            or type(item["version"]) is not int or item["version"] != FORMAT_VERSION
                            or item["schema_id"] != SCHEMA_ID):
                        raise RecordingError("unsupported or malformed recording header")
                    if type(item["identity"]) is not dict or set(item["identity"]) != set(_IDENTITY_FIELDS):
                        raise RecordingError("malformed recording identity")
                    identity = item["identity"]
                    if (type(identity["schema_id"]) is not str or identity["schema_id"] != SCHEMA_ID
                            or any(type(identity[name]) is not str or not identity[name].strip()
                                   for name in ("session_id", "source_id"))
                            or type(identity["source_epoch"]) is not int or identity["source_epoch"] < 0
                            or type(identity["time_domain"]) is not str
                            or identity["time_domain"] not in TIME_DOMAINS):
                        raise RecordingError("malformed recording identity values")
                    if type(item["metadata"]) is not dict:
                        raise RecordingError("recording metadata must be a JSON object")
                    _json_value(item["metadata"])
                    _check_extractor(item["metadata"], item["identity"])
                    annotations = item["metadata"].get("annotations")
                    if annotations is not None and (type(annotations) is not dict
                                                   or annotations.get("review_status") != "draft"):
                        raise RecordingError("recording annotations must remain draft")
                    header = item
                    if self._header is not None and header != self._header:
                        raise RecordingError("recording changed since validation")
                    continue
                if footer is not None:
                    raise RecordingError("recording has data after its footer")
                if kind == "footer":
                    if (set(item) != {"type", "count", "frames_sha256", "reason"}
                            or type(item["count"]) is not int or item["count"] != count or count == 0
                            or type(item["frames_sha256"]) is not str
                            or item["frames_sha256"] != digest.hexdigest()
                            or type(item["reason"]) is not str or not item["reason"]
                            or len(item["reason"]) > 1024):
                        raise RecordingError("recording footer count, checksum, or reason is invalid")
                    footer = item
                    continue
                if kind != "frame" or set(item) != {"type", "packet"}:
                    raise RecordingError("expected a complete landmark frame or footer")
                if count >= self.limits.max_frames:
                    raise RecordingLimitError("recording exceeds the frame limit")
                try:
                    frame = LandmarkFrame.from_dict(item["packet"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise RecordingError(f"invalid landmark packet: {exc}") from exc
                _check_next(previous, frame, header["identity"])
                if previous is None and frame.tracking_epoch != header["metadata"]["extractor"]["tracking_epoch"]:
                    raise RecordingError("first frame tracking epoch disagrees with extractor metadata")
                digest.update(raw)
                count += 1
                first_time = frame.timestamp if first_time is None else first_time
                last_time = frame.timestamp
                valid_anchors += int(frame.anchor_valid)
                observed_points += sum(frame.point_observed)
                left_present += int(frame.left_present)
                right_present += int(frame.right_present)
                skipped += frame.frames_skipped
                epochs.add(frame.tracking_epoch)
                previous = frame
                yield frame
        if header is None or footer is None:
            raise RecordingError("recording is truncated or has no complete header/footer")
        if self._footer is not None and footer != self._footer:
            raise RecordingError("recording changed since validation")
        self._header, self._footer = header, footer
        self._summary = {"format": FORMAT_ID, "schema_id": SCHEMA_ID,
                         "path": str(self.path.resolve()), "metadata": header["metadata"],
                         "identity": header["identity"], "count": count, "bytes": total_bytes,
                         "first_timestamp": first_time, "last_timestamp": last_time,
                         "duration_seconds": last_time - first_time,
                         "frames_sha256": footer["frames_sha256"], "reason": footer["reason"],
                         "quality": {"anchor_valid_frames": valid_anchors,
                                     "observed_points": observed_points,
                                     "left_present_frames": left_present,
                                     "right_present_frames": right_present,
                                     "frames_skipped": skipped,
                                     "tracking_epochs": sorted(epochs)}}
