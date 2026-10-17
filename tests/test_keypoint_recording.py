"""Persistence integrity, source ownership and whole-recording limit checks."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from slbridge.keypoint_recording import (
    KeypointRecorder, KeypointRecording, RecordingError, RecordingLimitError, RecordingLimits,
)
from slbridge.keypoints import KeypointExtractor, LandmarkFrame, SCHEMA_ID


def frame(index=0, **changes):
    value = LandmarkFrame(
        schema_id=SCHEMA_ID, session_id="session", source_id="source", source_epoch=1,
        tracking_epoch=0, frame_index=index, timestamp=index / 10, time_domain="media_seconds",
        received_monotonic=10 + index, extracted_monotonic=10 + index + 0.01,
        image_size=(96, 72), image_xy=((0.0, 0.0),) * 75,
        normalized_xy=((0.0, 0.0),) * 75, point_observed=(False,) * 75,
        normalized_observed=(False,) * 75, anchor_valid=False, pose_visibility=(None,) * 33,
        left_present=False, right_present=False, motion=0.0, motion_valid=False,
        dt=None if index == 0 else 0.1, frames_skipped=0, reset_reason=None)
    return replace(value, **changes)


def metadata(**extra):
    extractor = KeypointExtractor(session_id="session", source_epoch=1)
    result = {"extractor": extractor.metadata(), "annotations": {"label": "测试", "review_status": "approved"}}
    extractor.close()
    return {**result, **extra}


def save(path: Path, frames=None):
    recorder = KeypointRecorder(path, metadata=metadata())
    for value in frames if frames is not None else (frame(0), frame(1)):
        recorder.append(value)
    return recorder.finish("eof")


def rewrite(path, mutate, *, recompute_hash=False):
    rows = [json.loads(raw) for raw in path.read_bytes().splitlines()]
    mutate(rows)
    lines = [(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rows]
    if recompute_hash and rows[-1].get("type") == "footer":
        digest = hashlib.sha256()
        for raw, row in zip(lines, rows):
            if row.get("type") == "frame":
                digest.update(raw)
        rows[-1]["frames_sha256"] = digest.hexdigest()
        lines[-1] = (json.dumps(rows[-1], separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(b"".join(lines))


def test_roundtrip_complete_packets_draft_labels_and_no_image_pixels(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    supplied = metadata()
    recorder = KeypointRecorder(path, metadata=supplied)
    supplied["annotations"]["label"] = "changed"
    values = (frame(0), frame(1), frame(2, tracking_epoch=1, reset_reason="manual"))
    for value in values:
        recorder.append(value)
    assert recorder.count == 3 and not path.exists()
    assert len(list(tmp_path.glob("*.partial"))) == 1
    assert recorder.finish("manual_stop") == path
    assert not list(tmp_path.glob("*.partial"))
    recording = KeypointRecording(path)
    assert tuple(recording.iter_frames()) == values
    assert recording.metadata["annotations"] == {"label": "测试", "review_status": "draft"}
    assert recording.metadata["extractor"]["source_id"] == "source"
    returned = recording.metadata
    returned["annotations"]["label"] = "mutated"
    assert recording.metadata["annotations"]["label"] == "测试"
    summary = recording.summary()
    assert summary["count"] == 3 and summary["duration_seconds"] == pytest.approx(0.2)
    assert summary["quality"]["tracking_epochs"] == [0, 1]
    assert summary["reason"] == "manual_stop"
    assert '"image":' not in path.read_text(encoding="utf-8")
    with pytest.raises(RecordingError, match="closed"):
        recorder.append(frame(3))


def test_abort_and_empty_finish_remove_unique_partial(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.abort()
    recorder.abort()
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
    recorder = KeypointRecorder(path, metadata=metadata())
    with pytest.raises(RecordingError, match="empty"):
        recorder.finish()
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("changes", [
    {"session_id": "different"}, {"source_id": "different"}, {"source_epoch": 2},
    {"time_domain": "application_monotonic"}, {"timestamp": 0}, {"frame_index": 0},
])
def test_cross_source_and_nonincreasing_packets_abort_whole_recording(tmp_path, changes):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.append(frame(0))
    with pytest.raises(RecordingError):
        recorder.append(frame(1, **changes))
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


def test_tracking_epoch_regression_is_rejected(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.append(frame(0, tracking_epoch=2))
    with pytest.raises(RecordingError, match="tracking_epoch"):
        recorder.append(frame(1, tracking_epoch=1))
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


def test_frame_and_byte_limits_abort_instead_of_truncating(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata(), max_frames=1)
    recorder.append(frame(0))
    with pytest.raises(RecordingLimitError, match="frame limit"):
        recorder.append(frame(1))
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
    recorder = KeypointRecorder(path, metadata=metadata(), max_bytes=20)
    with pytest.raises(RecordingLimitError, match="byte limit"):
        recorder.append(frame(0))
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


def test_footer_is_included_in_byte_limit(tmp_path):
    full = save(tmp_path / "full.slkp.jsonl")
    path = tmp_path / "limited.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata(), max_bytes=full.stat().st_size - 1)
    recorder.append(frame(0))
    recorder.append(frame(1))
    with pytest.raises(RecordingLimitError, match="byte limit"):
        recorder.finish("eof")
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


def test_existing_output_and_commit_race_never_overwrite(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        KeypointRecorder(path, metadata=metadata())
    path.unlink()
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.append(frame(0))
    path.write_text("another writer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        recorder.finish()
    assert path.read_text(encoding="utf-8") == "another writer"
    assert not list(tmp_path.glob("*.partial"))


def test_commit_failure_removes_partial_without_complete_output(tmp_path, monkeypatch):
    from slbridge import keypoint_recording

    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.append(frame(0))

    def fail(*args):
        raise OSError("fake unsupported hard link")

    monkeypatch.setattr(keypoint_recording.os, "link", fail)
    with pytest.raises(OSError, match="hard link"):
        recorder.finish()
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


def test_extractor_metadata_must_match_first_frame_identity(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    with pytest.raises(RecordingError, match="metadata disagrees"):
        recorder.append(frame(0, session_id="other"))
    assert not path.exists() and not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("operation", ["append", "finish"])
def test_internal_cleanup_failure_preserves_primary_error_and_reports_cause(tmp_path, monkeypatch, operation):
    from slbridge import keypoint_recording

    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.append(frame(0))
    original_unlink = Path.unlink

    def fail_unlink(value, *args, **kwargs):
        if value.name.endswith(".partial"):
            raise PermissionError("fake partial locked")
        return original_unlink(value, *args, **kwargs)

    def fail_link(*args):
        raise OSError("fake publish failed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        if operation == "append":
            expected, message = RecordingError, "LandmarkFrame"
            action = lambda: recorder.append({})
        else:
            patch.setattr(keypoint_recording.os, "link", fail_link)
            expected, message = OSError, "fake publish failed"
            action = recorder.finish
        with pytest.raises(expected, match=message) as error:
            action()
        assert isinstance(error.value.__cause__, PermissionError)
        assert "fake partial locked" in error.value.keypoint_cleanup_error
        assert "fake partial locked" in recorder.cleanup_error
        assert recorder._partial.exists() and not path.exists()
    recorder.abort()
    assert recorder.cleanup_error is None and not recorder._partial.exists()


def test_publish_success_with_cleanup_failure_returns_valid_output_and_warning(tmp_path, monkeypatch):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    recorder.append(frame(0))
    original_unlink = Path.unlink

    def fail_unlink(value, *args, **kwargs):
        if value == recorder._partial:
            raise PermissionError("fake partial locked")
        return original_unlink(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        assert recorder.finish() == path
        assert path.exists() and recorder._partial.exists()
        assert tuple(KeypointRecording(path).iter_frames()) == (frame(0),)
        assert "Complete recording was saved" in recorder.cleanup_error
        assert "fake partial locked" in recorder.cleanup_error
    recorder.abort()
    assert path.exists() and not recorder._partial.exists() and recorder.cleanup_error is None
    assert KeypointRecording(path).summary()["count"] == 1


def test_abort_preserves_close_failure_when_unlink_also_fails(tmp_path, monkeypatch):
    path = tmp_path / "sample.slkp.jsonl"
    recorder = KeypointRecorder(path, metadata=metadata())
    stream = recorder._stream

    class CloseFailure:
        @property
        def closed(self):
            return stream.closed

        def close(self):
            stream.close()
            raise OSError("fake close failure")

    recorder._stream = CloseFailure()
    original_unlink = Path.unlink

    def fail_unlink(value, *args, **kwargs):
        if value == recorder._partial:
            raise PermissionError("fake unlink failure")
        return original_unlink(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        with pytest.raises(OSError, match="fake close failure") as error:
            recorder.abort()
        assert isinstance(error.value.__cause__, PermissionError)
        assert "fake close failure" in recorder.cleanup_error
        assert "fake unlink failure" in recorder.cleanup_error
    recorder.abort()
    assert not recorder._partial.exists()


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].update(schema_id="future_schema"),
    lambda rows: rows[0]["identity"].update(source_epoch=True),
    lambda rows: rows[0]["metadata"]["extractor"].update(protocol_hash="0" * 64),
    lambda rows: rows[0]["metadata"]["annotations"].update(review_status="approved"),
    lambda rows: rows[1]["packet"].update(timestamp=0.001),
    lambda rows: rows[1]["packet"].update(image_xy=[[0, 0]] * 74),
    lambda rows: rows[1]["packet"].update(point_observed=[0] * 75),
    lambda rows: rows[1]["packet"].update(unknown_field=1),
    lambda rows: rows[1]["packet"].update(source_id="other"),
    lambda rows: rows[-1].update(count=900),
    lambda rows: rows.append({"type": "unexpected"}),
    lambda rows: rows.pop(),
])
def test_tampered_truncated_and_incompatible_files_rejected(tmp_path, mutation):
    path = save(tmp_path / "sample.slkp.jsonl")
    rewrite(path, mutation)
    with pytest.raises(RecordingError):
        KeypointRecording(path)


def test_recomputed_hash_cannot_hide_order_or_ownership_fault(tmp_path):
    path = save(tmp_path / "sample.slkp.jsonl")
    rewrite(path, lambda rows: rows[2]["packet"].update(timestamp=0), recompute_hash=True)
    with pytest.raises(RecordingError, match="strictly increase"):
        KeypointRecording(path)


def test_duplicate_json_keys_nonfinite_and_incomplete_line_rejected(tmp_path):
    path = save(tmp_path / "sample.slkp.jsonl")
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'{"type":"header"', b'{"type":"header","type":"header"', 1))
    with pytest.raises(RecordingError, match="duplicate"):
        KeypointRecording(path)
    path.write_bytes(raw.replace(b'"timestamp":0.0', b'"timestamp":NaN', 1))
    with pytest.raises(RecordingError, match="non-finite"):
        KeypointRecording(path)
    path.write_bytes(raw[:-1])
    with pytest.raises(RecordingError, match="incomplete"):
        KeypointRecording(path)


def test_reader_limits_and_replay_readback_check(tmp_path):
    path = save(tmp_path / "sample.slkp.jsonl")
    with pytest.raises(RecordingLimitError, match="frame limit"):
        KeypointRecording(path, limits=RecordingLimits(max_frames=1))
    with pytest.raises(RecordingLimitError, match="byte limit"):
        KeypointRecording(path, limits=RecordingLimits(max_bytes=20))
    with pytest.raises(RecordingLimitError, match="line limit"):
        KeypointRecording(path, limits=RecordingLimits(max_line_bytes=20))
    recording = KeypointRecording(path)
    rewrite(path, lambda rows: rows[-1].update(reason="changed"))
    with pytest.raises(RecordingError, match="changed"):
        list(recording.iter_frames())


def test_bad_metadata_and_bad_frame_cleanup(tmp_path):
    path = tmp_path / "sample.slkp.jsonl"
    with pytest.raises(RecordingError, match="extractor"):
        KeypointRecorder(path, metadata={})
    bad = metadata()
    bad["value"] = float("inf")
    with pytest.raises(RecordingError, match="finite"):
        KeypointRecorder(path, metadata=bad)
    recorder = KeypointRecorder(path, metadata=metadata())
    with pytest.raises(RecordingError, match="LandmarkFrame"):
        recorder.append({})
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
