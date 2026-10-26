"""Tests for the ``pose_sample_only`` file-inference optimisation.

The optimisation extracts poses only on the frames that ``prepare_pose_window``
keeps. These tests use a fake ``cv2.VideoCapture`` whose frames encode their own
index, so the resulting pose tensors prove *which* frames were fed to the
extractor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from slbridge.runtime_options import RuntimeOptions
from slbridge.uni_sign import UniSignBackend, UniSignConfig
from slbridge.uni_sign_assets import (
    MT5_REVISION,
    UNISIGN_CHECKPOINT_BYTES,
    UNISIGN_CHECKPOINT_SHA256,
    UNISIGN_REVISION,
    UNISIGN_SOURCE_COMMIT,
)
from slbridge.uni_sign_preprocess import deterministic_frame_indices

ZERO_SHA = "0" * 64


def build_config(tmp_path, **runtime_overrides):
    root = tmp_path / "uni_sign"
    bundle = root / "model_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    runtime = {
        "device": "cpu", "dtype": "float32", "max_frames": 256,
        "num_beams": 4, "max_new_tokens": 20, "pose_mode": "lightweight",
        "pose_backend": "onnxruntime", "pose_device": "cpu",
        "rtmlib_cache_dir": "../rtmlib-cache", "stabilize_after": 2,
        "max_video_frames": 600,
    }
    runtime.update(runtime_overrides)
    return UniSignConfig(
        bundle_dir=bundle,
        checkpoint=root / "csl_daily_pose_only_slt.pth",
        checkpoint_bytes=UNISIGN_CHECKPOINT_BYTES,
        checkpoint_sha256=UNISIGN_CHECKPOINT_SHA256,
        mt5_dir=root / "mt5-base",
        source_dir=tmp_path / "uni_sign_source",
        source_commit=UNISIGN_SOURCE_COMMIT,
        source_zip_sha256=ZERO_SHA,
        model_revision=UNISIGN_REVISION,
        mt5_revision=MT5_REVISION,
        run_id="uni-sign-csl-daily-pose-only-182913ca",
        display_name="Uni-Sign Pose-only（CSL-Daily，实验）",
        runtime=runtime,
        license={"model": "CC-BY-NC-4.0"},
    )


class FakeCapture:
    """Minimal OpenCV capture: frames carry their own index in two pixels."""

    instances: list["FakeCapture"] = []

    def __init__(self, path, *, frames, reported=None, fps=30.0, width=64, height=48):
        self.path = path
        self.frames = frames
        self.reported = frames if reported is None else reported
        self.fps = fps
        self.width = width
        self.height = height
        self.position = 0
        self.held = -1
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.reported
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0.0

    def _image(self, index):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        frame[0, 0, 0] = index // 256
        frame[0, 0, 1] = index % 256
        return frame

    def read(self):
        if self.position >= self.frames:
            return False, None
        self.held = self.position
        self.position += 1
        return True, self._image(self.held)

    def grab(self):
        if self.position >= self.frames:
            return False
        self.held = self.position
        self.position += 1
        return True

    def retrieve(self):
        if self.held < 0:
            return False, None
        return True, self._image(self.held)

    def release(self):
        self.released = True


class IndexExtractor:
    """Produces frame-dependent poses so tensor equality proves frame selection."""

    def __init__(self):
        self.calls = 0
        self.indices = []

    def extract(self, frame):
        index = int(frame[0, 0, 0]) * 256 + int(frame[0, 0, 1])
        self.calls += 1
        self.indices.append(index)
        points = np.zeros((1, 133, 2), dtype=np.float32)
        scores = np.ones((1, 133), dtype=np.float32)
        phase = (index % 11) / 11.0
        points[0, 0] = [0.30 + 0.10 * phase, 0.30]
        points[0, 3:11, 0] = np.linspace(0.20, 0.80, 8) + 0.02 * phase
        points[0, 3:11, 1] = np.linspace(0.25, 0.75, 8)
        points[0, 91:112, 0] = np.linspace(0.30, 0.60, 21) + 0.01 * phase
        points[0, 91:112, 1] = np.linspace(0.35, 0.65, 21)
        points[0, 112:133, 0] = np.linspace(0.40, 0.70, 21) + 0.01 * phase
        points[0, 112:133, 1] = np.linspace(0.45, 0.75, 21)
        points[0, 23:40:2] = np.linspace(0.50, 0.70, 18).reshape(9, 2)
        points[0, 83:91] = np.linspace(0.60, 0.80, 16).reshape(8, 2)
        points[0, 53] = [0.50, 0.50]
        return points, scores

    @property
    def runtime_info(self):
        return {"pose_device": "cpu"}

    def close(self):
        pass


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake video container")
    return path


def backend_for(tmp_path, **overrides):
    backend = UniSignBackend(build_config(tmp_path, **overrides), RuntimeOptions(device="cpu"))
    extractor = IndexExtractor()
    backend._extractor = extractor
    backend._loaded = True
    return backend, extractor


def install_capture(monkeypatch, **capture_kwargs):
    FakeCapture.instances.clear()

    def factory(path):
        return FakeCapture(path, **capture_kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", factory)


def test_sample_only_extracts_only_the_sampled_frames(tmp_path, video_file, monkeypatch):
    install_capture(monkeypatch, frames=500)
    backend, extractor = backend_for(tmp_path, pose_sample_only=True)
    pose, metrics, time_range = backend._video_pose_window(video_file)

    planned = deterministic_frame_indices(500, max_frames=256)
    assert extractor.calls == len(planned) == 256
    assert extractor.indices == list(planned)
    assert tuple(pose.sampled_indices) == planned
    assert pose.quality["source_frames"] == 500
    assert pose.quality["pose_sample_only"] is True
    assert pose.quality["skipped_frames"] == 500 - 256
    assert metrics["pose_sample_only"] is True
    assert metrics["pose_skipped_frames"] == 244
    assert metrics["window_frames"] == 500
    assert metrics["sampled_frames"] == 256
    assert time_range == pytest.approx((0.0, 499 / 30.0))


def test_sample_only_matches_full_extraction_tensors(tmp_path, video_file, monkeypatch):
    install_capture(monkeypatch, frames=400)
    full_backend, full_extractor = backend_for(tmp_path, pose_sample_only=False)
    full_pose, full_metrics, _ = full_backend._video_pose_window(video_file)
    assert full_extractor.calls == 400
    assert full_metrics["pose_sample_only"] is False

    sampled_backend, sampled_extractor = backend_for(tmp_path, pose_sample_only=True)
    sampled_pose, sampled_metrics, _ = sampled_backend._video_pose_window(video_file)
    assert sampled_extractor.calls == 256

    for part in ("body", "left", "right", "face_all"):
        assert np.array_equal(getattr(full_pose, part), getattr(sampled_pose, part)), part
    assert tuple(sampled_pose.sampled_indices) == tuple(full_pose.sampled_indices)
    assert sampled_pose.quality["source_frames"] == full_pose.quality["source_frames"] == 400


def test_short_video_is_unaffected_by_the_flag(tmp_path, video_file, monkeypatch):
    install_capture(monkeypatch, frames=100)
    backend, extractor = backend_for(tmp_path, pose_sample_only=True)
    pose, metrics, _ = backend._video_pose_window(video_file)
    assert extractor.calls == 100
    assert metrics["pose_sample_only"] is False
    assert metrics["pose_skipped_frames"] == 0
    assert len(pose.sampled_indices) == 100


def test_unknown_frame_count_falls_back_to_reading_every_frame(tmp_path, video_file, monkeypatch):
    install_capture(monkeypatch, frames=300, reported=0)
    backend, extractor = backend_for(tmp_path, pose_sample_only=True)
    pose, metrics, _ = backend._video_pose_window(video_file)
    assert extractor.calls == 300
    assert metrics["pose_sample_only"] is False
    assert len(pose.sampled_indices) == 256


def test_overreported_frame_count_truncates_the_plan(tmp_path, video_file, monkeypatch):
    install_capture(monkeypatch, frames=300, reported=500)
    backend, extractor = backend_for(tmp_path, pose_sample_only=True)
    pose, metrics, _ = backend._video_pose_window(video_file)
    assert extractor.calls == len(pose.sampled_indices)
    assert metrics["window_frames"] == 300
    assert pose.quality["source_frames"] == 300
    assert tuple(pose.sampled_indices) == deterministic_frame_indices(500, max_frames=256)[:len(pose.sampled_indices)]


def test_flag_is_reported_and_must_be_boolean(tmp_path):
    backend, _ = backend_for(tmp_path)
    assert backend.runtime_info["pose_sample_only"] is False
    with pytest.raises(ValueError, match="pose_sample_only"):
        backend_for(tmp_path, pose_sample_only="true")


def test_video_longer_than_the_hard_limit_still_fails(tmp_path, video_file, monkeypatch):
    install_capture(monkeypatch, frames=601)
    backend, extractor = backend_for(tmp_path, pose_sample_only=True)
    with pytest.raises(ValueError, match="limit is 600"):
        backend._video_pose_window(video_file)
    assert extractor.calls == 0

