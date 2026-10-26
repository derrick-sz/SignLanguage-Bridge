"""T-A3-1 tests: media metadata probing + job lifecycle cleanup.

Coverage (task card T-A3-1 / task-18):
* ``probe_media``: ffprobe-based metadata (container/codec/fps/frames/size/
  duration) for real video files; non-media uploads (the current replay-query
  JSON) must yield ``metadata=None`` plus a recorded ``probe_error`` reason --
  never fabricated fields;
* job records and the run manifest carry the metadata;
* lifecycle: terminal jobs release their upload temp file after a TTL;
  failures leave no orphan files; the store's capacity accounting frees slots
  for finished jobs; the prediction JSONL (evidence!) is never deleted.

Runs wherever fastapi is importable (.venv-libras); the .conda-env suite
skips cleanly.  Real-video probes use an existing trimmed artifact strictly
read-only (nothing is written into the repo).
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from slbridge.api import create_app
from slbridge.job_runner import JobStore
from slbridge.media_probe import probe_media

REPO = Path(__file__).resolve().parents[1]
REAL_VIDEO = REPO / "artifacts" / "csldaily-video-test-627-20261026" / "trimmed" / "S000020_P0004_T00.mp4"


# ---------------------------------------------------------------------------
# probe_media
# ---------------------------------------------------------------------------

def test_probe_media_json_returns_null_metadata_with_reason(tmp_path: Path):
    query = tmp_path / "q.json"
    query.write_text(json.dumps({"sample_id": "S000020_P0004_T00"}), encoding="utf-8")
    result = probe_media(query)
    assert result["metadata"] is None            # never fabricated
    assert isinstance(result["probe_error"], str) and result["probe_error"]
    assert "ffprobe" in result["probe_error"].lower() or "probe" in result["probe_error"].lower()


def test_probe_media_missing_file(tmp_path: Path):
    result = probe_media(tmp_path / "nope.mp4")
    assert result["metadata"] is None
    assert isinstance(result["probe_error"], str) and result["probe_error"]


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="trimmed artifact not present locally")
def test_probe_media_real_video_complete(tmp_path: Path):
    result = probe_media(REAL_VIDEO)
    assert result["probe_error"] is None
    metadata = result["metadata"]
    assert metadata["container"] == "mov,mp4,m4a,3gp,3g2,mj2"
    assert metadata["video_codec"] in {"h264", "mpeg4"}   # trimmed clips re-encode per batch
    assert metadata["width"] > 0 and metadata["height"] > 0
    assert 20 <= metadata["fps"] <= 35            # CSL-Daily clips are ~30fps
    assert metadata["frames"] > 0
    assert metadata["duration_seconds"] > 0
    # Every field is a JSON primitive (goes straight into job records/manifest).
    json.dumps(metadata)


def test_probe_media_injectable_ffprobe(tmp_path: Path):
    """A fake ffprobe runner lets us exercise parse paths without ffmpeg."""
    from slbridge.media_probe import probe_media_with_runner

    query = tmp_path / "fake.mp4"
    query.write_bytes(b"\x00" * 16)

    def fake_ok(path):
        return 0, json.dumps({
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "2.500"},
            "streams": [{"codec_type": "video", "codec_name": "h264",
                         "width": 1280, "height": 1280, "r_frame_rate": "30/1",
                         "nb_frames": "75"}]}), ""

    result = probe_media_with_runner(query, fake_ok)
    assert result["probe_error"] is None
    assert result["metadata"]["fps"] == 30.0
    assert result["metadata"]["frames"] == 75
    assert result["metadata"]["duration_seconds"] == 2.5

    def fake_fail(path):
        return 1, "", "Invalid data found when processing input"

    result = probe_media_with_runner(query, fake_fail)
    assert result["metadata"] is None
    assert "Invalid data" in result["probe_error"]


# ---------------------------------------------------------------------------
# Job-side integration: metadata lands on the record + manifest
# ---------------------------------------------------------------------------

class ReplayManager:
    def __init__(self, rows):
        self.rows = rows

    def describe(self):
        return {"model_id": "unisign_rgb", "generation": 1, "sign_language": "csl",
                "task_type": "text", "backend": "replay_jsonl_v1"}

    def status(self):
        return {"state": "ready", "generation": 1, "active": self.describe(),
                "previous": None, "error": None}

    def capabilities(self):
        return None

    def infer_file(self, path, request):
        from types import SimpleNamespace
        from slbridge.contracts import RecognitionResult, TextPrediction
        query = json.loads(Path(path).read_text(encoding="utf-8"))
        text = self.rows[query["sample_id"]]
        result = RecognitionResult(
            status="recognized", sample_id=request.request_id, model_id="unisign_rgb",
            sign_language="csl", prediction=TextPrediction(text=text, language="zh", score=None),
            quality={"replay": True}, task_type="text",
            score_type="replay_imported_text_no_score", time_range=None)
        return SimpleNamespace(request_id=request.request_id, model_generation=1, result=result,
                               runtime={}, timing_ms={"replay_ms": 1.0, "total_ms": 1.5},
                               session_id=None, segment_id=None, source_epoch=None,
                               model_id="unisign_rgb", model_version="rev",
                               preprocessing_version="pv")

    def close(self):
        pass


def wait_terminal(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(f"/v1/jobs/{job_id}")
        body = response.json()
        if body["status"] in {"SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED"}:
            return body
        if time.monotonic() > deadline:
            return body
        time.sleep(0.02)


def submit(client, sample_id: str, job_id: str, document: bytes, name: str = "query.json"):
    return client.post("/v1/videos/infer",
                       data={"sign_system": "CSL", "mode": "single_model",
                             "job_id": job_id, "sample_id": sample_id},
                       files={"video": (name, document, "application/json")})


@pytest.fixture
def app_client(tmp_path: Path):
    manager = ReplayManager({"S000020_P0004_T00": "她今年50岁。"})
    app = create_app({}, manager=manager, max_upload_bytes=1024 * 1024,
                     job_run_directory=tmp_path / "runs", job_run_id="media-run-1",
                     job_upload_ttl_seconds=0.0)      # clean up immediately in tests
    with TestClient(app) as client:
        yield client, tmp_path


def test_job_metadata_null_for_json_upload(app_client):
    client, tmp_path = app_client
    document = json.dumps({"sample_id": "S000020_P0004_T00"}).encode("utf-8")
    assert submit(client, "S000020_P0004_T00", "job-json", document).status_code == 202
    body = wait_terminal(client, "job-json")
    assert body["status"] == "SUCCEEDED"
    record = client.app.state.job_store.get("job-json")
    assert record.media_metadata is None
    assert isinstance(record.media_probe_error, str) and record.media_probe_error


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="trimmed artifact not present locally")
def test_job_metadata_real_video(app_client):
    client, tmp_path = app_client
    # Upload a *copy* of the read-only artifact through the endpoint; the
    # original on disk is never touched.
    document = REAL_VIDEO.read_bytes()
    response = client.post("/v1/videos/infer",
                           data={"sign_system": "CSL", "mode": "single_model",
                                 "job_id": "job-video", "sample_id": "S000020_P0004_T00"},
                           files={"video": ("clip.mp4", document, "video/mp4")})
    assert response.status_code == 202
    body = wait_terminal(client, "job-video")
    record = client.app.state.job_store.get("job-video")
    # The replay manager reads the file as JSON: a real video makes it fail.
    assert body["status"] == "FAILED"
    # But the media metadata was still probed and recorded before execution.
    assert record.media_metadata is not None
    assert record.media_metadata["video_codec"] in {"h264", "mpeg4"}
    assert record.media_metadata["width"] == 1280
    assert record.media_metadata["frames"] > 0
    assert record.media_probe_error is None
    # Manifest exposes the same metadata for the run.
    manifest = json.loads((tmp_path / "runs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["media_metadata"]["job-video"]["metadata"]["width"] == 1280


def test_uploads_cleaned_after_terminal_state(app_client, tmp_path: Path):
    client, _ = app_client
    document = json.dumps({"sample_id": "S000020_P0004_T00"}).encode("utf-8")
    assert submit(client, "S000020_P0004_T00", "job-clean", document).status_code == 202
    wait_terminal(client, "job-clean")
    upload_dir = Path(tempfile.gettempdir())
    leftovers = [p for p in upload_dir.glob("slbridge-input-*") if p.stat().st_mtime > time.time() - 60]
    assert leftovers == []                     # no orphan uploads from this run


def test_failed_job_leaves_no_orphan(app_client):
    client, _ = app_client
    document = json.dumps({"sample_id": "S999999_P9999_T00"}).encode("utf-8")
    assert submit(client, "S999999_P9999_T00", "job-fail-orphan", document).status_code == 202
    body = wait_terminal(client, "job-fail-orphan")
    assert body["status"] == "FAILED"
    leftovers = [p for p in Path(tempfile.gettempdir()).glob("slbridge-input-*")
                 if p.stat().st_mtime > time.time() - 60]
    assert leftovers == []


def test_prediction_log_survives_cleanup(app_client, tmp_path: Path):
    client, _ = app_client
    document = json.dumps({"sample_id": "S000020_P0004_T00"}).encode("utf-8")
    assert submit(client, "S000020_P0004_T00", "job-keep", document).status_code == 202
    wait_terminal(client, "job-keep")
    # Cleanup removes uploads, never evidence.
    assert (tmp_path / "runs" / "predictions.jsonl").is_file()
    lines = (tmp_path / "runs" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["text"] == "她今年50岁。"


def test_job_store_capacity_frees_finished_slots(tmp_path: Path):
    store = JobStore(max_jobs=2)
    first = store.create(job_id="j1", sign_system="CSL", sample_id="a")
    second = store.create(job_id="j2", sign_system="CSL", sample_id="b")
    with pytest.raises(RuntimeError):
        store.create(job_id="j3", sign_system="CSL", sample_id="c")
    # Legal terminal path: QUEUED -> PREPROCESSING -> RUNNING -> SUCCEEDED.
    first.transition("PREPROCESSING")
    first.transition("RUNNING")
    first.transition("SUCCEEDED")
    store.retire("j1")                          # frees the slot, keeps the record
    third = store.create(job_id="j3", sign_system="CSL", sample_id="c")
    assert third is not None
    assert store.get("j1") is first             # the record itself is still queryable


import tempfile  # noqa: E402  (used above in cleanup assertions)
