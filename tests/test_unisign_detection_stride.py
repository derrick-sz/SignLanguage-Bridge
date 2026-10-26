"""Tests for opt-in cross-frame detection reuse (``pose_detect_stride``).

The extractor is built without running ``__init__`` so no ONNX session or model
download is involved; only the stride bookkeeping is exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

from slbridge.runtime_options import RuntimeOptions
from slbridge.uni_sign import RtmposeExtractor, UniSignBackend, UniSignConfig
from slbridge.uni_sign_assets import (
    MT5_REVISION,
    UNISIGN_CHECKPOINT_BYTES,
    UNISIGN_CHECKPOINT_SHA256,
    UNISIGN_REVISION,
    UNISIGN_SOURCE_COMMIT,
)

ZERO_SHA = "0" * 64


class FakeWholebody:
    def __init__(self):
        self.det_calls = 0
        self.pose_calls = 0
        self.full_calls = 0
        self.boxes_used = []
        self.det_model = self._detect
        self.pose_model = self._pose

    def __call__(self, frame):
        # Counted separately from det/pose so the stride-1 path is distinguishable.
        self.full_calls += 1
        return (np.zeros((1, 133, 2), dtype=np.float32),
                np.ones((1, 133), dtype=np.float32))

    def _detect(self, frame):
        self.det_calls += 1
        return np.array([[0.0, 0.0, 10.0, 10.0, 0.9]], dtype=np.float32)

    def _pose(self, frame, *, bboxes):
        self.pose_calls += 1
        self.boxes_used.append(None if bboxes is None else np.array(bboxes, copy=True))
        return (np.zeros((1, 133, 2), dtype=np.float32),
                np.ones((1, 133), dtype=np.float32))


def make_extractor(stride):
    extractor = RtmposeExtractor.__new__(RtmposeExtractor)
    extractor._wholebody = FakeWholebody()
    extractor._detect_stride = stride
    extractor._frames_seen = 0
    extractor._last_boxes = None
    return extractor


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


def test_default_stride_calls_wholebody_once_per_frame():
    extractor = make_extractor(1)
    for _ in range(5):
        extractor.extract(np.zeros((8, 8, 3), np.uint8))
    assert extractor._wholebody.full_calls == 5
    assert extractor._wholebody.det_calls == 0
    assert extractor._wholebody.pose_calls == 0


def test_stride_reuses_the_last_box_between_detections():
    extractor = make_extractor(5)
    for _ in range(12):
        extractor.extract(np.zeros((8, 8, 3), np.uint8))
    assert extractor._wholebody.pose_calls == 12
    assert extractor._wholebody.det_calls == 3          # frames 1, 6, 11
    assert extractor._wholebody.full_calls == 0
    assert all(box is not None for box in extractor._wholebody.boxes_used)


def test_reset_detection_forces_a_new_detection():
    extractor = make_extractor(10)
    extractor.extract(np.zeros((8, 8, 3), np.uint8))
    extractor.extract(np.zeros((8, 8, 3), np.uint8))
    assert extractor._wholebody.det_calls == 1
    extractor.reset_detection()
    extractor.extract(np.zeros((8, 8, 3), np.uint8))
    assert extractor._wholebody.det_calls == 2
    assert extractor._frames_seen == 1


def test_stride_range_is_validated_before_loading_anything(tmp_path):
    with pytest.raises(ValueError, match="detect_stride"):
        RtmposeExtractor(mode="lightweight", backend="onnxruntime", device="cpu",
                         cache_dir=tmp_path, detect_stride=0)
    with pytest.raises(ValueError, match="detect_stride"):
        RtmposeExtractor(mode="lightweight", backend="onnxruntime", device="cpu",
                         cache_dir=tmp_path, detect_stride=31)


def test_backend_reports_and_validates_the_stride(tmp_path):
    backend = UniSignBackend(build_config(tmp_path), RuntimeOptions(device="cpu"))
    assert backend.runtime_info["pose_detect_stride"] == 1
    configured = UniSignBackend(build_config(tmp_path, pose_detect_stride=3),
                                RuntimeOptions(device="cpu"))
    assert configured.runtime_info["pose_detect_stride"] == 3
    with pytest.raises(ValueError, match="pose_detect_stride"):
        UniSignBackend(build_config(tmp_path, pose_detect_stride=0), RuntimeOptions(device="cpu"))


class ResettingExtractor:
    def __init__(self):
        self.resets = 0

    def reset_detection(self):
        self.resets += 1


def test_source_change_resets_the_reused_box(tmp_path):
    backend = UniSignBackend(build_config(tmp_path, pose_detect_stride=3),
                             RuntimeOptions(device="cpu"))
    extractor = ResettingExtractor()
    backend._extractor = extractor
    # The first call establishes the source, so it already counts as a reset.
    backend._clear_for_source("camera:0", 0)
    assert extractor.resets == 1
    backend._clear_for_source("camera:0", 1)
    assert extractor.resets == 1
    backend._clear_for_source("camera:1", 0)
    assert extractor.resets == 2


def test_missing_reset_hook_is_tolerated(tmp_path):
    backend = UniSignBackend(build_config(tmp_path), RuntimeOptions(device="cpu"))
    backend._extractor = object()
    backend._clear_for_source("camera:0", 0)
    backend._clear_for_source("camera:1", 0)

