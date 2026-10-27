"""P1-RGBPOSE-ADAPTER tests (task-32): Uni-Sign RGB+pose adapter.

CPU-testable parts run everywhere; the GPU integration test is opt-in via
``SLBRIDGE_RGB_POSE_GPU=1`` (needs the pinned checkpoints + CUDA, reproduces
one frozen prediction verbatim).

Frozen baseline this adapter must reproduce (official_archive pose source):
``experiments/p1/e01/runs/unisign-rgb-pose-1086-20261027/`` -- BLEU-4 26.3026,
protocol.json pins every inference parameter.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from slbridge.evidence import assert_gt_clean
from slbridge.model_adapters import get_input_representation, model_adapter_ids
from slbridge.runtime_contracts import InferenceRequest

REPO = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL = REPO / "experiments" / "p1" / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "protocol.json"
REAL_VIDEO = REPO / "artifacts" / "csldaily-video-test-627-20261026" / "trimmed" / "S000020_P0004_T00.mp4"


@pytest.fixture(scope="module")
def module():
    return __import__("slbridge.uni_sign_rgb_pose", fromlist=[""])


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Registration & capabilities
# ---------------------------------------------------------------------------

def test_adapter_registered(module):
    assert "unisign_rgb_pose_slt" in model_adapter_ids()


def test_bundle_loads_through_registry(tmp_path: Path):
    """The canonical lazy loader in model_adapters resolves a bundle manifest."""
    from slbridge.model_adapters import load_model_adapter
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "backend": "unisign_rgb_pose_slt",
        "pose_source": "rtmpose",
        "video_dir": str(tmp_path),
        "enable_rgb": True,
        "fallback_to_pose_only": True,
        "device": "cpu",
    }), encoding="utf-8")
    adapter = load_model_adapter(bundle)
    assert adapter.capabilities.model_id == "unisign_rgb_pose"
    adapter.close()


def test_bundle_rejects_json_alignment_directory(tmp_path: Path):
    """official_archive + a directory without npz files must fail loudly."""
    from slbridge.model_adapters import load_model_adapter
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "backend": "unisign_rgb_pose_slt",
        "pose_source": "official_archive",
        "pose_dir": str(tmp_path),
        "alignment_dir": str(tmp_path / "alignment-json"),   # wrong shape
        "fallback_to_pose_only": False,
    }), encoding="utf-8")
    (tmp_path / "alignment-json").mkdir()
    (tmp_path / "alignment-json" / "S000020_P0004_T00.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="alignment_dir"):
        load_model_adapter(bundle)


def test_capabilities_reuse_uni_sign_representation(module):
    from slbridge.uni_sign import REPRESENTATION_ID
    adapter = _make_adapter(module, pose_source="rtmpose")
    caps = adapter.capabilities
    assert caps.model_id == "unisign_rgb_pose"
    assert caps.task_type == "text"
    assert caps.sign_language == "csl"
    assert caps.runtime == "pytorch_unisign_rgb_pose"
    assert caps.score_type == "uncalibrated_text"
    assert caps.modes == ("clip",)
    assert len(caps.inputs) == 1                       # video/file only, per lead constraint 1
    contract = caps.inputs[0]
    assert contract.representation_id == REPRESENTATION_ID   # reused, not re-registered
    assert get_input_representation(contract.representation_id).shape == (None, None, None, 3)
    assert contract.input_kinds == ("file",)
    adapter.close()


def test_model_version_is_protocol_hash(module, frozen):
    adapter = _make_adapter(module, pose_source="rtmpose")
    version = adapter.capabilities.model_version
    assert isinstance(version, str) and len(version) == 64      # canonical sha256 hex
    # The verbatim identity fields ride on runtime_info / result metadata.
    info = adapter.runtime_info
    assert info["checkpoint_sha256"] == frozen["checkpoints"]["rgb_pose"]["sha256"]
    assert info["source_commit"] == frozen["source"]["commit"]
    adapter.close()


# ---------------------------------------------------------------------------
# Frozen-parameter consistency (protocol.json value-by-value)
# ---------------------------------------------------------------------------

def test_frozen_constants_match_protocol(module, frozen):
    assert module.IMAGE_SIZE == 112                    # hand crops (EfficientNet-B0 in ckpt)
    assert module.MAX_LENGTH == frozen["input_protocol"]["max_length"] == 256
    assert module.NUM_BEAMS == frozen["decoding"]["num_beams"] == 4
    assert module.MAX_NEW_TOKENS == frozen["decoding"]["max_new_tokens"] == 100
    assert module.DTYPE_NAME == frozen["decoding"]["dtype"] == "bfloat16"
    assert module.CHECKPOINT_SHA256 == frozen["checkpoints"]["rgb_pose"]["sha256"]
    assert module.SOURCE_COMMIT == frozen["source"]["commit"]
    assert module.SOURCE_COMMIT == "eed438bcb49e30405cd6ccdfcccca330c134e830"
    assert module.PROTOCOL_ID == "unisign-rgb-pose-v1"
    assert module.FRAME_SAMPLING == "deterministic_linspace"
    assert module.HAND_SAMPLING == "low_confidence_weighted_fixed_seed"


def test_preprocess_version_is_parameter_hash(module):
    adapter = _make_adapter(module, pose_source="rtmpose")
    version = adapter.capabilities.inputs[0].preprocessing_version
    assert isinstance(version, str) and len(version) == 64      # canonical sha256 hex
    adapter.close()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_bad_pose_source(module):
    with pytest.raises(ValueError, match="pose_source"):
        module.UniSignRGBPoseConfig(pose_source="youtube")


def test_config_official_archive_requires_pose_dir(module, tmp_path: Path):
    with pytest.raises(ValueError, match="pose_dir"):
        module.UniSignRGBPoseConfig(pose_source="official_archive", pose_dir=None,
                                    alignment_dir=tmp_path)


def test_config_fallback_forbidden_in_official_mode(module, tmp_path: Path):
    with pytest.raises(ValueError, match="fallback"):
        module.UniSignRGBPoseConfig(pose_source="official_archive",
                                    pose_dir=tmp_path, alignment_dir=tmp_path,
                                    fallback_to_pose_only=True)   # regression mode must fail loudly


# ---------------------------------------------------------------------------
# Hand-crop geometry (bbox_4hands / sample_hand_indices)
# ---------------------------------------------------------------------------

def test_bbox_4hands_geometry(module):
    # One frame, 21 keypoints per hand in normalized coords; the box maths
    # must mirror upstream exactly (centre +- box_hw/2, 1.2 factor, even size).
    left = np.full((1, 21, 2), 0.4)
    right = np.full((1, 21, 2), 0.6)
    left_boxes, right_boxes, box_hw = module.bbox_4hands(left, right, (1000, 1000))
    # A single point per hand collapses the bbox -> box_hw 0 (upstream keeps it).
    assert box_hw >= 0 and box_hw % 2 == 0
    assert left_boxes.shape == (1, 4) and right_boxes.shape == (1, 4)
    # A spread-out hand yields a real box, centred on the mean.
    spread = np.tile(np.linspace(0.3, 0.5, 21)[:, None], (1, 2))[None, :, :]  # [1,21,2]
    left_boxes2, right_boxes2, box_hw2 = module.bbox_4hands(spread, spread, (1000, 1000))
    assert box_hw2 > 0 and box_hw2 % 2 == 0
    # Degenerate point clouds collapse to the minimum box (zeros).
    zero_boxes_l, zero_boxes_r, zero_hw = module.bbox_4hands(None, None, (480, 640))
    assert zero_hw == 0


def test_sample_hand_indices_deterministic_and_weighted(module):
    # Upstream indexing expects confs shaped [T, 133] (per-frame slices 91:112).
    frames = 40
    confs = np.full((frames, 133), 0.9, dtype=np.float32)
    confs[5, 91:112] = 0.1               # one very low-confidence left-hand frame
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    chosen_a = module.sample_hand_indices(confs, "left", rng_a)
    chosen_b = module.sample_hand_indices(confs, "left", rng_b)
    assert chosen_a is not None and np.array_equal(chosen_a, chosen_b)   # deterministic
    assert len(chosen_a) == int(np.ceil(0.1 * frames))                   # upstream 10% budget
    # Weighted sampling prefers low-confidence fingertips (key_ids 95/99/103/107/111):
    # drive those to zero and the sampler must still return a valid subset.
    confs[:, [95, 99, 103, 107, 111]] = 0.05
    chosen_low = module.sample_hand_indices(confs, "left", np.random.default_rng(7))
    assert chosen_low is not None and len(chosen_low) == int(np.ceil(0.1 * frames))
    # No usable frames -> None (not an exception).
    dead = np.zeros((frames, 133), dtype=np.float32)
    assert module.sample_hand_indices(dead, "left", np.random.default_rng(0)) is None


def test_crop_scale_matches_upstream_semantics(module):
    motion = np.zeros((5, 21, 3))
    motion[..., :2] = 0.5
    motion[..., 2] = 0.9
    result, scale, centre = module.crop_scale(motion, 0.3)
    assert scale == 0                                    # a point cloud collapses
    motion2 = np.zeros((5, 21, 3))
    motion2[:, :, 0] = np.linspace(0.2, 0.8, 21)
    motion2[:, :, 1] = np.linspace(0.3, 0.7, 21)
    motion2[..., 2] = 0.9
    result2, scale2, _ = module.crop_scale(motion2, 0.3)
    assert scale2 > 0
    assert np.all(np.abs(result2) <= 1.0)                # clipped to [-1, 1]


# ---------------------------------------------------------------------------
# Determinism of input construction (CPU, official archive fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="trimmed artifact not present")
def test_build_inputs_deterministic_cpu(module, tmp_path: Path):
    """Same sample -> byte-identical tensors twice (linspace + fixed seed)."""
    pytest.importorskip("torch", reason="input construction builds torch tensors")
    pose_dir = REPO / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
    npz_dir = REPO / "artifacts" / "csldaily-video-test-627-20261026" / "pose"
    config = module.UniSignRGBPoseConfig(pose_source="official_archive",
                                         pose_dir=pose_dir, alignment_dir=npz_dir,
                                         fallback_to_pose_only=False)
    sample = "S000020_P0004_T00"
    first = module.build_sample_inputs(config, sample)
    second = module.build_sample_inputs(config, sample)
    assert first["frames"] == second["frames"]
    for part in ("body", "left", "right", "face_all"):
        assert torch_equal(first["parts"][part], second["parts"][part])
    for side in ("left", "right"):
        assert torch_equal(first["rgb"][f"{side}_hands"], second["rgb"][f"{side}_hands"])
        assert torch_equal(first["rgb"][f"{side}_skeletons_norm"],
                           second["rgb"][f"{side}_skeletons_norm"])
        assert torch_equal(first["rgb"][f"{side}_sampled_indices"],
                           second["rgb"][f"{side}_sampled_indices"])
    # The source dict carries the exact model-facing keys.
    source = module.source_from_sample(first, with_rgb=True)
    for key in ("body", "left", "right", "face_all", "attention_mask",
                "src_length_batch", "left_hands", "right_hands",
                "left_skeletons_norm", "right_skeletons_norm",
                "left_sampled_indices", "right_sampled_indices",
                "left_rgb_len", "right_rgb_len"):
        assert key in source, key


def torch_equal(a, b) -> bool:
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and bool(torch.equal(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Fallback semantics (mocked runtime; no GPU)
# ---------------------------------------------------------------------------

def _make_adapter(module, **overrides):
    config = module.UniSignRGBPoseConfig(**{"pose_source": "rtmpose", **overrides})
    return module.UniSignRGBPoseAdapter(config)


def test_fallback_on_rgb_failure_in_rtmpose_mode(module):
    """RGB branch fails -> pose-only result with fallback metadata (no silent success)."""
    config = module.UniSignRGBPoseConfig(pose_source="rtmpose", fallback_to_pose_only=True)
    adapter = module.UniSignRGBPoseAdapter.__new__(module.UniSignRGBPoseAdapter)
    adapter.config = config
    adapter._runtime = None
    adapter._pose_only_runtime = _FakeRuntime("fallback-text")
    adapter._extractor = None
    adapter._rgb_failures = []
    adapter._pose_only_source_provider = lambda: {"name_batch": ["pose-only"]}
    result = adapter._recognize_with_fallback(
        _FakeInputs(), request=_request(), rgb_error=RuntimeError("cannot open video"))
    assert result.status == "recognized"
    assert result.prediction.text == "fallback-text"
    assert result.quality["fallback_reason"] == "rgb_unavailable"
    assert result.quality["fallback_model_id"] == "unisign_pose_only"
    assert result.quality["rgb_failure"] == "cannot open video"
    assert adapter._rgb_failures                                        # failure recorded for runtime_info


def test_no_fallback_when_disabled(module):
    config = module.UniSignRGBPoseConfig(pose_source="rtmpose", fallback_to_pose_only=False)
    adapter = module.UniSignRGBPoseAdapter.__new__(module.UniSignRGBPoseAdapter)
    adapter.config = config
    with pytest.raises(RuntimeError, match="cannot open video"):
        adapter._recognize_with_fallback(
            _FakeInputs(), request=_request(), rgb_error=RuntimeError("cannot open video"))


def test_official_mode_never_falls_back(module):
    """official_archive is the regression mode: failures raise, never fall back."""
    config = module.UniSignRGBPoseConfig(pose_source="official_archive",
                                         pose_dir=Path("."), alignment_dir=Path("."),
                                         fallback_to_pose_only=False)
    adapter = module.UniSignRGBPoseAdapter.__new__(module.UniSignRGBPoseAdapter)
    adapter.config = config
    adapter._pose_only_runtime = _FakeRuntime("should-not-appear")
    with pytest.raises(RuntimeError, match="regression"):
        adapter._recognize_with_fallback(
            _FakeInputs(), request=_request(), rgb_error=RuntimeError("decode failed"))


class _FakeRuntime:
    def __init__(self, text):
        self.text = text

    def infer(self, source, *, num_beams, max_new_tokens):
        return self.text, {"model_total_ms": 1.0}


class _FakeInputs:
    pass


def _request():
    return InferenceRequest(request_id="req-1", sign_language="csl",
                            expected_task_type="text")


# ---------------------------------------------------------------------------
# GT-free discipline
# ---------------------------------------------------------------------------

def test_result_payload_is_gt_free(module):
    from slbridge.contracts import RecognitionResult, TextPrediction
    result = RecognitionResult(
        status="recognized", sample_id="req-1", model_id="unisign_rgb_pose",
        sign_language="csl",
        prediction=TextPrediction(text="她今年50岁。", language="zh", score=None),
        quality={"protocol_id": module.PROTOCOL_ID,
                 "checkpoint_sha256": module.CHECKPOINT_SHA256,
                 "source_commit": module.SOURCE_COMMIT,
                 "pose_source": "official_archive",
                 "fallback_reason": None},
        task_type="text", score_type="uncalibrated_text")
    assert_gt_clean(result.to_dict(), context="rgb+pose result")


# ---------------------------------------------------------------------------
# Opt-in GPU integration: one frozen sample, verbatim
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="trimmed artifact not present")
def test_gpu_one_sample_verbatim_vs_frozen(module, frozen):
    if __import__("os").environ.get("SLBRIDGE_RGB_POSE_GPU") != "1":
        pytest.skip("opt-in GPU test; set SLBRIDGE_RGB_POSE_GPU=1")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pose_dir = REPO / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
    npz_dir = REPO / "artifacts" / "csldaily-video-test-627-20261026" / "pose"
    config = module.UniSignRGBPoseConfig(pose_source="official_archive",
                                         pose_dir=pose_dir, alignment_dir=npz_dir,
                                         fallback_to_pose_only=False,
                                         device="cuda", dtype="bfloat16")
    adapter = module.UniSignRGBPoseAdapter(config)
    adapter.warmup()
    try:
        sample = "S000020_P0004_T00"
        result = adapter.recognize_file(REAL_VIDEO, _request_with_sample(sample))
        frozen_text = _frozen_prediction_text("S000020_P0004_T00")
        assert result.status == "recognized"
        assert result.prediction.text == frozen_text            # verbatim
        assert result.quality["checkpoint_sha256"] == module.CHECKPOINT_SHA256
        assert result.quality["fallback_reason"] is None
    finally:
        adapter.close()


def _request_with_sample(sample: str):
    # official_archive mode resolves the sample from the video filename stem.
    return InferenceRequest(request_id=sample, sign_language="csl",
                            expected_task_type="text")


def _frozen_prediction_text(sample: str) -> str:
    path = REPO / "experiments" / "p1" / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "predictions_rgb_pose.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["sample_id"] == sample and row.get("status") == "ok":
            return row["text"]
    raise AssertionError(f"frozen prediction missing for {sample}")
