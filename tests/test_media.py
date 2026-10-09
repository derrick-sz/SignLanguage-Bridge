"""RGB input contracts, normalization and real video decoding acceptance."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.features import InputRejected
from slbridge.media import ExtractionConfig, HolisticExtractor, VideoDecodeError, decode_video, normalize_landmarks


def point(x=0.5, y=0.5, visibility=1.0, z=7.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def landmarks(points):
    return SimpleNamespace(landmark=points)


def holistic_result():
    pose = [point() for _ in range(33)]
    pose[11], pose[12] = point(0.4, 0.4), point(0.6, 0.4)
    return SimpleNamespace(pose_landmarks=landmarks(pose),
                           left_hand_landmarks=landmarks([point(0.3, 0.5) for _ in range(21)]),
                           right_hand_landmarks=landmarks([point(0.7, 0.6) for _ in range(21)]))


def test_import_does_not_load_media_or_array_dependencies():
    code = "import sys; import slbridge.media; assert not ({'numpy','cv2','mediapipe'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=20)
    assert result.returncode == 0, result.stderr


def test_config_round_trip_and_unknown_fields():
    config = ExtractionConfig(frames=16, max_width=320)
    assert ExtractionConfig.from_dict(config.to_dict()) == config
    assert ExtractionConfig.from_dict({}) == ExtractionConfig()
    with pytest.raises(ValueError, match="Unknown"):
        ExtractionConfig.from_dict({"fake": 1})


@pytest.mark.parametrize("values", [
    {"frames": True}, {"frames": 1}, {"max_width": 0}, {"max_width": 320.0},
    {"model_complexity": 0}, {"model_complexity": 2}, {"model_complexity": True},
    {"mirror": True}, {"mirror": 0}, {"version": 2},
    {"pose_visibility": float("nan")}, {"pose_visibility": True},
    {"minimum_hand_frame_fraction": 0}, {"minimum_anchor_frame_fraction": 1.1},
])
def test_config_rejects_ambiguous_or_download_enabling_options(values):
    with pytest.raises(ValueError):
        ExtractionConfig(**values)


def test_common_shoulder_reference_preserves_anatomical_hands():
    values, observed = normalize_landmarks(holistic_result(), 200, 100)
    points = values.reshape(75, 3)
    np.testing.assert_allclose(points[11], (-0.5, 0, 0), atol=1e-6)
    np.testing.assert_allclose(points[12], (0.5, 0, 0), atol=1e-6)
    np.testing.assert_allclose(points[33], (-1.0, 0.25, 0), atol=1e-6)
    np.testing.assert_allclose(points[54], (1.0, 0.5, 0), atol=1e-6)
    assert values.dtype == np.float32
    assert observed == {"anchor": True, "left_hand": True, "right_hand": True}
    assert (points[:, 2] == 0).all()  # Discard incompatible Holistic z references.


def test_normalization_is_invariant_to_shared_image_translation_and_scale():
    original = holistic_result()
    transformed = holistic_result()
    for group in (transformed.pose_landmarks, transformed.left_hand_landmarks, transformed.right_hand_landmarks):
        for item in group.landmark:
            item.x = item.x * 0.5 + 0.2
            item.y = item.y * 0.5 + 0.1
    expected, _ = normalize_landmarks(original, 200, 100)
    actual, _ = normalize_landmarks(transformed, 200, 100)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_low_visibility_and_out_of_image_points_remain_missing():
    result = holistic_result()
    result.pose_landmarks.landmark[0].visibility = 0.3
    result.left_hand_landmarks.landmark[3].x = -0.1
    result.right_hand_landmarks.landmark[7].y = float("nan")
    values, observed = normalize_landmarks(result, 200, 100)
    points = values.reshape(75, 3)
    for index in (0, 36, 61):
        assert (points[index] == -999).all()
    assert observed["anchor"]
    assert (points[1] != -999).all()


@pytest.mark.parametrize("failure", ["no_pose", "missing_shoulder", "coincident_shoulders"])
def test_missing_anchor_never_fabricates_hand_coordinates(failure):
    result = holistic_result()
    if failure == "no_pose":
        result.pose_landmarks = None
    elif failure == "missing_shoulder":
        result.pose_landmarks.landmark[11].visibility = 0.2
    else:
        result.pose_landmarks.landmark[12] = result.pose_landmarks.landmark[11]
    values, observed = normalize_landmarks(result, 200, 100)
    assert (values == -999).all()
    assert not any(observed.values())


@pytest.fixture
def generated_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (96, 72))
    assert writer.isOpened(), "Test environment has no MJPG encoder"
    for index in range(7):
        writer.write(np.full((72, 96, 3), index * 20, dtype=np.uint8))
    writer.release()
    return path


def test_real_decoder_uniform_selection_and_actual_time(generated_video):
    result = decode_video(generated_video, ExtractionConfig(frames=5, max_width=64))
    np.testing.assert_array_equal(result.frame_indices, [0, 2, 3, 4, 6])
    np.testing.assert_allclose(result.timestamps, np.asarray([0, 2, 3, 4, 6]) / 25, atol=1e-6)
    assert all(image.shape == (48, 64, 3) for image in result.images)
    assert result.quality["source_frames"] == 7
    assert result.quality["source_fps"] == 25
    assert result.quality["full_decode_verified"]
    assert result.timestamps.dtype == np.float64 and result.frame_indices.dtype == np.int64
    assert not result.timestamps.flags.writeable


def test_real_decoder_short_clip_is_not_padded_with_invented_frames(generated_video):
    result = decode_video(generated_video)
    np.testing.assert_array_equal(result.frame_indices, np.arange(7))
    assert len(result.images) == 7


def test_decoder_fallback_records_actual_fps_instead_of_inventing_time(generated_video, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    capture_type = cv2.VideoCapture

    class CaptureWithoutTimestamps:
        def __init__(self, path):
            self.capture = capture_type(path)

        def get(self, property_id):
            return 0.0 if property_id == cv2.CAP_PROP_POS_MSEC else self.capture.get(property_id)

        def __getattr__(self, name):
            return getattr(self.capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", CaptureWithoutTimestamps)
    result = decode_video(generated_video)
    assert result.quality["timestamp_source"] == "frame_index_fps"
    np.testing.assert_allclose(result.timestamps, np.arange(7) / 25)


def test_decoder_refuses_incomplete_clip_against_declared_count(generated_video, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    capture_type = cv2.VideoCapture
    captures = []

    class CaptureWithLongerDeclaredClip:
        def __init__(self, path):
            self.capture = capture_type(path)
            self.released = False
            captures.append(self)

        def get(self, property_id):
            value = self.capture.get(property_id)
            return value + 1 if property_id == cv2.CAP_PROP_FRAME_COUNT else value

        def release(self):
            self.released = True
            self.capture.release()

        def __getattr__(self, name):
            return getattr(self.capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", CaptureWithLongerDeclaredClip)
    with pytest.raises(VideoDecodeError, match="expected 8 frames, decoded 7"):
        decode_video(generated_video)
    assert captures[0].released


def test_invalid_video_fails_decode_without_claiming_quality_success(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(VideoDecodeError):
        decode_video(path)


def test_missing_file_remains_a_source_error(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError):
        decode_video(tmp_path / "missing.mp4")


def test_actual_holistic_rejects_blank_video_and_resets_between_clips(generated_video):
    pytest.importorskip("mediapipe")
    with HolisticExtractor(ExtractionConfig(frames=3)) as extractor:
        quality = []
        for _ in range(2):
            with pytest.raises(InputRejected) as failure:
                extractor.extract(generated_video)
            assert failure.value.reason == "insufficient_anchor_frames"
            quality.append(failure.value.quality)
        assert quality[0] == quality[1]
        assert quality[0]["full_decode_verified"]
        assert quality[0]["anchor_frame_fraction"] == 0
    with pytest.raises(RuntimeError, match="context manager"):
        extractor.extract(generated_video)
