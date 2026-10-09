"""Captured RGB processing contracts and opt-in real model parity acceptance."""

from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge import live, media
from slbridge.contracts import InputSpec
from slbridge.features import InputRejected, prepare_input
from slbridge.inference import Recognizer
from slbridge.sources import CapturedClip, ClipBuffer, FramePacket, VideoSource


def clip(count=7, width=96):
    return CapturedClip(tuple(FramePacket(np.full((72, width, 3), index, dtype=np.uint8),
                                         index / 25, index, "test") for index in range(count)), "test")


@pytest.fixture
def extraction(monkeypatch):
    import mediapipe as mp

    config = media.ExtractionConfig(frames=5, max_width=64)
    identity = {"holistic_options": {"static_image_mode": False}, "marker": "saved"}
    monkeypatch.setattr(media, "extractor_identity", lambda config: identity)
    graphs = []

    class Graph:
        def __init__(self, **options):
            self.options, self.images, self.resets, self.closed = options, [], 0, False
            graphs.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def reset(self):
            self.resets += 1

        def process(self, image):
            assert not image.flags.writeable
            self.images.append(image.copy())
            return image

    def normalized(result, width, height, visibility):
        values = np.zeros(225, dtype=np.float32)
        values[::3] = float(result[0, 0, 0])
        return values, {"anchor": True, "left_hand": True, "right_hand": False}

    monkeypatch.setattr(mp.solutions.holistic, "Holistic", Graph)
    monkeypatch.setattr(media, "normalize_landmarks", normalized)
    spec = InputSpec.for_representation(media.REPRESENTATION_ID,
                                      extractor={"config": config.to_dict(), "identity": identity})
    return spec, graphs


def test_uniform_sampling_resize_quality_and_tracking_isolation(extraction):
    spec, graphs = extraction
    first = live.prepare_clip(clip(), spec)
    second = live.prepare_clip(clip(), spec)
    assert first.quality["source_frames"] == 7
    assert first.quality["sampled_frames"] == 5
    assert first.quality["landmark_frames"] == 5
    assert first.quality["hand_frame_fraction"] == 1
    assert first.quality["right_hand_frame_fraction"] == 0
    assert first.timestamps[0] == 0 and first.timestamps[-1] == 6 / 25
    assert [int(image[0, 0, 0]) for image in graphs[0].images] == [0, 2, 3, 4, 6]
    assert all(image.shape == (48, 64, 3) for image in graphs[0].images)
    assert len(graphs) == 2 and all(graph.resets == 1 and graph.closed for graph in graphs)
    assert graphs[0].options == {"model_complexity": 1, "static_image_mode": False}
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_bgr_to_rgb_preserves_orientation(extraction):
    spec, graphs = extraction
    value = clip(2, 64)
    for packet in value.frames:
        packet.image[:, :32] = [11, 22, 33]
        packet.image[:, 32:] = [44, 55, 66]
    live.prepare_clip(value, spec)
    np.testing.assert_array_equal(graphs[0].images[0][0, 0], [33, 22, 11])
    np.testing.assert_array_equal(graphs[0].images[0][0, -1], [66, 55, 44])


def test_saved_identity_mismatch_and_legacy_model_rejected(extraction):
    spec, graphs = extraction
    mismatched = replace(spec, extractor={**spec.extractor, "identity": {"wrong": True}})
    with pytest.raises(ValueError, match="saved identity"):
        live.prepare_clip(clip(), mismatched)
    with pytest.raises(ValueError, match="RGB clips require"):
        live.prepare_clip(clip(), InputSpec.for_representation("hand63_v1"))
    assert not graphs


def test_empty_observations_return_structured_rejection(extraction, monkeypatch):
    spec, graphs = extraction
    monkeypatch.setattr(media, "normalize_landmarks", lambda *args: (
        np.full(225, -999, dtype=np.float32), {"anchor": False, "left_hand": False, "right_hand": False}))
    recognizer = Recognizer(SimpleNamespace(manifest=SimpleNamespace(
        input_spec=spec, run_id="model", sign_language="Libras")))
    result = recognizer.recognize_clip(clip(), sample_id="recording")
    assert result.status == "rejected" and result.reason == "insufficient_anchor_frames"
    assert result.sample_id == "recording" and result.prediction is None
    assert result.quality["source_frames"] == 7 and graphs[0].closed
    with pytest.raises(NotImplementedError, match="clip inference only"):
        recognizer.begin_stream()


def test_shared_prediction_and_threshold(extraction, monkeypatch):
    spec, _ = extraction
    prepared = live.prepare_clip(clip(), spec)
    monkeypatch.setattr("slbridge.inference.prepare_input", lambda *args: prepared)
    backend = SimpleNamespace(classes=("a", "b"), predict_proba=lambda inputs: [[0.8, 0.2]])
    bundle = SimpleNamespace(manifest=SimpleNamespace(input_spec=spec, run_id="model", sign_language="Libras"),
                             backend=backend, rejection_threshold=0.7, labels={"a": {"pt-BR": "A"}})
    recognizer = Recognizer(bundle)
    assert recognizer.recognize("file.mp4", sample_id="x") == recognizer.recognize_clip(clip(), sample_id="x")
    bundle.rejection_threshold = 0.9
    result = recognizer.recognize_clip(clip())
    assert result.status == "rejected" and result.reason == "below_validation_selected_score_threshold"


def test_observed_anchor_without_hands_rejected(extraction, monkeypatch):
    spec, graphs = extraction
    monkeypatch.setattr(media, "normalize_landmarks", lambda *args: (
        np.zeros(225, dtype=np.float32), {"anchor": True, "left_hand": False, "right_hand": False}))
    with pytest.raises(InputRejected, match="insufficient_hand_frames") as failure:
        live.prepare_clip(clip(), spec)
    assert failure.value.quality["hand_frame_fraction"] == 0
    assert failure.value.quality["anchor_frame_fraction"] == 1
    assert graphs[0].closed


@pytest.mark.parametrize("fault", ["single", "time", "source", "index", "shape"])
def test_defensive_validation_of_mutated_clip(extraction, fault):
    spec, graphs = extraction
    value = clip()
    if fault == "single":
        object.__setattr__(value, "frames", value.frames[:1])
        with pytest.raises(InputRejected, match="at_least_two"):
            live.prepare_clip(value, spec)
    else:
        frame = value.frames[-1]
        attribute, invalid = {"time": ("timestamp", float("nan")), "source": ("source_id", "other"),
                              "index": ("frame_index", 0), "shape": ("image", np.zeros((3, 3, 3), np.uint8))}[fault]
        object.__setattr__(frame, attribute, invalid)
        with pytest.raises(ValueError):
            live.prepare_clip(value, spec)
    assert not graphs


@pytest.mark.skipif(not (os.environ.get("SLBRIDGE_LIVE_VIDEO") and os.environ.get("SLBRIDGE_LIVE_MODEL")),
                    reason="set SLBRIDGE_LIVE_VIDEO and SLBRIDGE_LIVE_MODEL for real saved-model acceptance")
def test_real_video_and_saved_model_cross_path_parity():
    video, model = os.environ["SLBRIDGE_LIVE_VIDEO"], os.environ["SLBRIDGE_LIVE_MODEL"]
    recognizer = Recognizer.load(model)
    buffer = ClipBuffer()
    with VideoSource(video) as source:
        while (packet := source.read()) is not None:
            buffer.append(packet)
    captured = buffer.finish()
    spec = recognizer.bundle.manifest.input_spec
    old = prepare_input(video, spec)
    new = live.prepare_clip(captured, spec)
    np.testing.assert_array_equal(old.values, new.values)
    np.testing.assert_array_equal(old.mask, new.mask)
    np.testing.assert_array_equal(old.timestamps, new.timestamps)
    np.testing.assert_array_equal(recognizer.bundle.backend.predict_proba([old]),
                                  recognizer.bundle.backend.predict_proba([new]))
    old_result = recognizer.recognize(video, sample_id="parity")
    new_result = recognizer.recognize_clip(captured, sample_id="parity")
    assert old_result.status == new_result.status
    assert old_result.prediction == new_result.prediction
    assert old_result.reason == new_result.reason
    assert old_result.time_range == new_result.time_range
