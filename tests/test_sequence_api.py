"""Transport task isolation and readable sequence results through the real manager."""
from types import SimpleNamespace

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient
from slbridge.api import create_app
from slbridge.cecsl import RGBInputSpec
from slbridge.contracts import GlossToken, GlossSequencePrediction, RecognitionResult
from slbridge.runtime import ModelManager


class SequenceRecognizer:
    runtime_info = {"providers": ["CPUExecutionProvider"]}

    def __init__(self, task="gloss_sequence"):
        self.bundle = SimpleNamespace(manifest=SimpleNamespace(
            run_id="test-sequence", backend="onnx_ctc", sign_language="csl",
            task_type=task, score_type="ctc_token_probability",
            input_spec=RGBInputSpec(), capabilities=("clip",)), labels={},
            vocabulary=(" ", "我", "我"))
        self.calls = 0
        self.closed = False

    def recognize_profiled(self, path, *, sample_id, representation_id=None):
        self.calls += 1
        assert path.read_bytes() == b"synthetic-video"
        return RecognitionResult("recognized", sample_id, "test-sequence", "csl",
            GlossSequencePrediction((GlossToken(1, "我", .8), GlossToken(1, "我", .7)), "greedy"),
            {"source_frames": 20}, task_type="gloss_sequence", score_type="ctc_token_probability"), {"inference_ms": 1.}

    def close(self):
        self.closed = True


def client_for(recognizer):
    return TestClient(create_app({}, manager=ModelManager(initial_recognizer=recognizer)))


def test_sequence_api_keeps_repeated_tokens_and_explicit_unknown_times():
    recognizer = SequenceRecognizer()
    with client_for(recognizer) as client:
        response = client.post("/v2/recognitions", files={"file": ("clip.mp4", b"synthetic-video")},
                               data={"request_id": "sequence-1", "representation_id": "tfnet_rgb_v1", "sign_language": "csl"})
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == 2 and body["request_id"] == "sequence-1"
        tokens = body["result"]["prediction"]["tokens"]
        assert [token["gloss"] for token in tokens] == ["我", "我"]
        assert all(token["time_range"] is None for token in tokens)
        assert body["result"]["task_type"] == "gloss_sequence"
        assert client.get("/v1/models").json()["active"]["vocabulary_size"] == 3
    assert recognizer.closed


@pytest.mark.parametrize("route,task", [("/v1/recognitions", "gloss_sequence"),
                                       ("/v2/recognitions", "classification")])
def test_incompatible_api_task_rejected_under_model_lease(route, task):
    recognizer = SequenceRecognizer(task)
    with client_for(recognizer) as client:
        response = client.post(route, files={"file": ("clip.mp4", b"synthetic-video")})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"
        assert recognizer.calls == 0


def test_sequence_api_rejects_landmark_file_before_inference():
    recognizer = SequenceRecognizer()
    with client_for(recognizer) as client:
        response = client.post("/v2/recognitions", files={"file": ("clip.npy", b"synthetic-video")})
        assert response.status_code == 422 and recognizer.calls == 0


def test_sequence_api_whitespace_language_is_client_error():
    recognizer = SequenceRecognizer()
    with client_for(recognizer) as client:
        response = client.post("/v2/recognitions", files={"file": ("clip.mp4", b"synthetic-video")},
                               data={"sign_language": " "})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert recognizer.calls == 0
