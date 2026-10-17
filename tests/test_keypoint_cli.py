"""CLI extraction uses observation packets, preserves source timing and aborts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge import keypoint_cli
from slbridge.keypoint_recording import KeypointRecording
from slbridge.keypoints import KeypointExtractor
from slbridge.sources import FramePacket, SourceError


@pytest.fixture
def extraction(monkeypatch):
    sources, extractors, graphs = [], [], []

    class Source:
        def __init__(self, path):
            self.source_id = f"video:{path.resolve()}"
            self.timestamp_mode = "fps_fallback"
            self.index = 0
            self.closed = False
            self.times = (0.0, 0.08, 0.17)
            sources.append(self)

        def open(self):
            return self

        def read(self):
            if self.index == len(self.times):
                return None
            index = self.index
            self.index += 1
            return FramePacket(np.full((72, 96, 3), index, dtype=np.uint8),
                               self.times[index], index, self.source_id)

        def close(self):
            self.closed = True

    class Graph:
        def __init__(self, **options):
            self.calls = 0
            self.closed = False
            self.fail_at = None
            graphs.append(self)

        def process(self, image):
            self.calls += 1
            assert not image.flags.writeable
            if self.fail_at == self.calls:
                raise RuntimeError("fake detector failed")
            return SimpleNamespace()

        def close(self):
            self.closed = True

    class Extractor(KeypointExtractor):
        def __init__(self, **kwargs):
            super().__init__(session_id="cli-test", holistic_factory=Graph, **kwargs)
            extractors.append(self)

    monkeypatch.setattr(keypoint_cli, "VideoSource", Source)
    monkeypatch.setattr(keypoint_cli, "KeypointExtractor", Extractor)
    return sources, extractors, graphs


def test_extract_inspect_preserves_all_frames_irregular_clock_and_draft_annotations(
        tmp_path, capsys, extraction):
    source_path = tmp_path / "input.mp4"
    output_path = tmp_path / "sample.slkp.jsonl"
    result = keypoint_cli.main(["extract", "--video", str(source_path), "--output", str(output_path),
                               "--label", "你好", "--language", "CSL", "--performer", "local-1"])
    assert result == 0
    captured = capsys.readouterr()
    assert not captured.err
    summary = json.loads(captured.out)
    assert summary["count"] == 3
    assert summary["identity"]["time_domain"] == "media_seconds"
    assert summary["metadata"]["source_timing_mode"] == "fps_fallback"
    assert summary["metadata"]["annotations"] == {
        "review_status": "draft", "label": "你好", "language": "CSL", "performer": "local-1"}
    frames = tuple(KeypointRecording(output_path).iter_frames())
    assert [frame.timestamp for frame in frames] == [0.0, 0.08, 0.17]
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert all(frame.extracted_monotonic >= frame.received_monotonic for frame in frames)
    sources, extractors, graphs = extraction
    assert len(graphs) == 1 and graphs[0].calls == 3 and graphs[0].closed
    assert sources[0].closed and extractors[0]._closed
    assert keypoint_cli.main(["inspect", str(output_path)]) == 0
    assert json.loads(capsys.readouterr().out) == summary


def test_extract_hard_frame_limit_rejects_complete_output(tmp_path, capsys, extraction):
    output_path = tmp_path / "sample.slkp.jsonl"
    result = keypoint_cli.main(["extract", "--video", str(tmp_path / "input.mp4"),
                               "--output", str(output_path), "--max-frames", "1"])
    assert result == 1 and not output_path.exists()
    assert not list(tmp_path.glob("*.partial"))
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error_type"] == "RecordingLimitError"
    sources, extractors, graphs = extraction
    assert sources[0].closed and extractors[0]._closed and graphs[0].closed
    assert graphs[0].calls == 1


def test_extraction_failure_cleans_partial_and_closes_resources(tmp_path, capsys, extraction, monkeypatch):
    original = keypoint_cli.KeypointExtractor.process

    def process(self, packet, **kwargs):
        if packet.frame_index == 1:
            raise RuntimeError("fake extraction failed")
        return original(self, packet, **kwargs)

    monkeypatch.setattr(keypoint_cli.KeypointExtractor, "process", process)
    output_path = tmp_path / "sample.slkp.jsonl"
    assert keypoint_cli.main(["extract", "--video", str(tmp_path / "input.mp4"),
                              "--output", str(output_path)]) == 1
    assert not output_path.exists() and not list(tmp_path.glob("*.partial"))
    captured = capsys.readouterr()
    assert not captured.out and "fake extraction failed" in json.loads(captured.err)["error"]
    sources, extractors, graphs = extraction
    assert sources[0].closed and extractors[0]._closed and graphs[0].closed


def test_existing_output_not_replaced(tmp_path, capsys, extraction):
    output_path = tmp_path / "sample.slkp.jsonl"
    output_path.write_text("keep me", encoding="utf-8")
    assert keypoint_cli.main(["extract", "--video", str(tmp_path / "input.mp4"),
                              "--output", str(output_path)]) == 1
    assert output_path.read_text(encoding="utf-8") == "keep me"
    assert json.loads(capsys.readouterr().err)["error_type"] == "FileExistsError"
    sources, extractors, graphs = extraction
    assert sources[0].closed and extractors[0]._closed and not graphs


def test_source_open_failure_closes_source_without_partial(tmp_path, capsys, extraction, monkeypatch):
    def fail(self):
        raise SourceError("fake unreadable video")

    monkeypatch.setattr(keypoint_cli.VideoSource, "open", fail)
    output_path = tmp_path / "sample.slkp.jsonl"
    assert keypoint_cli.main(["extract", "--video", str(tmp_path / "input.mp4"),
                              "--output", str(output_path)]) == 1
    assert not output_path.exists() and not list(tmp_path.glob("*.partial"))
    assert json.loads(capsys.readouterr().err)["error_type"] == "SourceError"
    sources, extractors, graphs = extraction
    assert sources[0].closed and not extractors and not graphs


def test_inspect_corrupt_file_and_argument_validation(tmp_path, capsys):
    path = tmp_path / "corrupt.slkp.jsonl"
    path.write_bytes(b'{"type":"header"}\n')
    assert keypoint_cli.main(["inspect", str(path)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error_type"] == "RecordingError"
    with pytest.raises(SystemExit) as error:
        keypoint_cli.main(["inspect", str(path), "--max-frames", "0"])
    assert error.value.code == 2


def test_cli_reports_saved_file_and_temporary_cleanup_warning(tmp_path, capsys, extraction, monkeypatch):
    output_path = tmp_path / "sample.slkp.jsonl"
    original_unlink = Path.unlink

    def fail_unlink(path, *args, **kwargs):
        if path.name.endswith(".partial"):
            raise PermissionError("fake cleanup blocked")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        assert keypoint_cli.main(["extract", "--video", str(tmp_path / "input.mp4"),
                                  "--output", str(output_path)]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    summary = json.loads(captured.out)
    assert summary["count"] == 3 and "fake cleanup blocked" in summary["cleanup_warning"]
    assert KeypointRecording(output_path).summary()["count"] == 3
    partials = list(tmp_path.glob("*.partial"))
    assert len(partials) == 1
    partials[0].unlink()


def test_cli_preserves_primary_failure_if_abort_cleanup_also_fails(tmp_path, capsys, extraction, monkeypatch):
    output_path = tmp_path / "sample.slkp.jsonl"
    original_unlink = Path.unlink

    def fail_unlink(path, *args, **kwargs):
        if path.name.endswith(".partial"):
            raise PermissionError("fake cleanup blocked")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        assert keypoint_cli.main(["extract", "--video", str(tmp_path / "input.mp4"),
                                  "--output", str(output_path), "--max-frames", "1"]) == 1
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert not captured.out and not output_path.exists()
    assert error["error_type"] == "RecordingLimitError" and "frame limit" in error["error"]
    assert "fake cleanup blocked" in error["cleanup_error"]
    partials = list(tmp_path.glob("*.partial"))
    assert len(partials) == 1
    partials[0].unlink()
