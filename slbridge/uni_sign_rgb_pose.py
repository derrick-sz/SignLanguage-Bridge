"""Uni-Sign RGB+pose SLT adapter (P1-RGBPOSE, task-32).

Engineering integration of the frozen RGB+pose evaluation path -- **not** a
replacement for the production pose-only default (``uni_sign_pose_slt``).

Single source of truth: the inference core below is lifted verbatim from the
frozen script ``experiments/p1/e01/eval_unisign_rgb_pose.py`` (BLEU-4 26.3026
on the full 1086; protocol pinned in
``experiments/p1/e01/runs/unisign-rgb-pose-1086-20261027/protocol.json``).
That script now *imports* this module, so adapter and regression share one
implementation and any drift is attributable to the integration layer only.

Frozen protocol (value-by-value, asserted by tests):
    checkpoint  csl_daily_rgb_pose_slt.pth
                sha256 c2ba5d7ae32f3fa40f7c6807b5004042be81fca192abe7d8ed7182c0d13470f6
    source      uni_sign_source @ eed438bcb49e30405cd6ccdfcccca330c134e830
    hand crops  112x112 (EfficientNet-B0 branch, weights inside the ckpt)
    pose        official released CSL-Daily pose archive (normalized 133 kps)
    max_length  256, deterministic linspace frame sampling
    hands       low-confidence weighted sampling, per-sample fixed RNG seed
    decoding    num_beams=4, max_new_tokens=100, dtype=bfloat16

Pose sources (lead constraint 2):
    * ``official_archive`` -- regression mode: reads the official pose pkl +
      the npz alignment metadata; **the only mode that reproduces the frozen
      predictions verbatim**.  Failures raise; fallback is forbidden.
    * ``rtmpose`` -- production mode: extracts 133 keypoints from the video
      with ``slbridge.uni_sign.RtmposeExtractor`` (no official archive).  If
      the RGB branch is unavailable (decode failure / empty hand crops /
      OOM-after-retry) and ``fallback_to_pose_only=True`` (default), the
      adapter falls back to the pose-only checkpoint and records
      ``fallback_reason``/``fallback_model_id`` in the result metadata --
      never silently: the metadata always says which model produced the text.

Input contract reuses ``slbridge.uni_sign.REPRESENTATION_ID``
("uni_sign_rgb_v1", video frames T,H,W,C uint8): one ModelInputContract,
file input kind; the pose source is configuration, not a contract input.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from types import SimpleNamespace

import numpy as np

from . import _io
from .contracts import RecognitionResult, TextPrediction
from .model_adapters import ModelCapabilities, ModelInputContract
from .runtime_contracts import InferenceRequest

# ---------------------------------------------------------------------------
# frozen protocol constants (protocol.json, value-by-value)
# ---------------------------------------------------------------------------
PROTOCOL_ID = "unisign-rgb-pose-v1"
IMAGE_SIZE = 112
MAX_LENGTH = 256
NUM_BEAMS = 4
MAX_NEW_TOKENS = 100
DTYPE_NAME = "bfloat16"
SEED = 20261027
FRAME_SAMPLING = "deterministic_linspace"
HAND_SAMPLING = "low_confidence_weighted_fixed_seed"
CHECKPOINT_SHA256 = "c2ba5d7ae32f3fa40f7c6807b5004042be81fca192abe7d8ed7182c0d13470f6"
SOURCE_COMMIT = "eed438bcb49e30405cd6ccdfcccca330c134e830"
POSE_ONLY_SHA256 = "73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c"
SCORE_TYPE = "uncalibrated_text"
MODEL_ID = "unisign_rgb_pose"
FALLBACK_MODEL_ID = "unisign_pose_only"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_POSE_DIR = _REPO_ROOT / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
_DEFAULT_VIDEO_DIR = _REPO_ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "trimmed"
_DEFAULT_ALIGNMENT_DIR = _REPO_ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "pose"
_DEFAULT_CHECKPOINT = _REPO_ROOT / "models" / "external" / "uni_sign" / "csl_daily_rgb_pose_slt.pth"
_DEFAULT_POSE_ONLY_CHECKPOINT = _REPO_ROOT / "models" / "external" / "uni_sign" / "csl_daily_pose_only_slt.pth"
_DEFAULT_SOURCE_DIR = _REPO_ROOT / "models" / "external" / "uni_sign_source"
_DEFAULT_MT5_DIR = _REPO_ROOT / "models" / "external" / "uni_sign" / "mt5-base"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# upstream model import (isolated module scope; verbatim from the frozen script)
# ---------------------------------------------------------------------------
def load_upstream_class(source_dir: Path | None = None, mt5_dir: Path | None = None):
    saved: dict[str, object] = {}
    names = {"config", "models", "datasets", "utils", "stgcn_layers", "deformable_attention_2d"}
    for name in list(sys.modules):
        if name in names or name.startswith("stgcn_layers."):
            saved[name] = sys.modules.pop(name)
    source_dir = Path(source_dir or _DEFAULT_SOURCE_DIR)
    mt5_dir = Path(mt5_dir or _DEFAULT_MT5_DIR)
    sys.path.insert(0, str(source_dir))
    module_name = f"_p1_upstream_uni_sign_{hashlib.sha256(str(source_dir).encode()).hexdigest()[:12]}"
    try:
        import importlib.util

        import config as upstream_config

        upstream_config.mt5_path = str(mt5_dir)
        spec = importlib.util.spec_from_file_location(module_name, source_dir / "models.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {source_dir / 'models.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.Uni_Sign
    finally:
        try:
            sys.path.remove(str(source_dir))
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == module_name or name in names or name.startswith("stgcn_layers."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def _disable_efficientnet_pretrained() -> None:
    """Checkpoint carries all EfficientNet-B0 weights; avoid a network download."""

    import torchvision.models

    if getattr(torchvision.models.efficientnet_b0, "_p1_no_pretrained", False):
        return
    original = torchvision.models.efficientnet_b0

    def _no_pretrained(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original(*args, **kwargs)

    _no_pretrained._p1_no_pretrained = True  # type: ignore[attr-defined]
    torchvision.models.efficientnet_b0 = _no_pretrained


# ---------------------------------------------------------------------------
# runtime (verbatim from the frozen script's UniSignRuntime)
# ---------------------------------------------------------------------------
class UniSignRGBPoseRuntime:
    """Load one pinned checkpoint and run it on pre-built inputs."""

    def __init__(self, checkpoint: Path, *, rgb_support: bool,
                 device: str = "cuda", dtype: str = "bfloat16",
                 source_dir: Path | None = None, mt5_dir: Path | None = None):
        self.checkpoint = Path(checkpoint)
        self.rgb_support = bool(rgb_support)
        self.device_name = device
        self.dtype_name = dtype
        self._source_dir = source_dir
        self._mt5_dir = mt5_dir
        self.model = None
        self.tokenizer = None
        self.load_ms = None
        self.param_count = None
        self.checkpoint_sha256 = sha256_file(self.checkpoint)

    def load(self) -> None:
        import torch
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype_name]
        device = self.device_name
        if sys.version_info < (3, 11):
            import typing
            typing.Self = typing.TypeVar("Self")  # type: ignore[attr-defined]
        _disable_efficientnet_pretrained()
        started = time.perf_counter()
        if device == "cuda":
            torch.cuda.synchronize()
        cls = load_upstream_class(self._source_dir, self._mt5_dir)
        args = SimpleNamespace(dataset="CSL_Daily", task="SLT",
                               rgb_support=self.rgb_support, hidden_dim=256,
                               label_smoothing=0.2, max_length=MAX_LENGTH)
        model = cls(args=args)
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain a model state dict")
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        loaded = model.load_state_dict(state, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(
                "checkpoint does not match the pinned model definition: "
                f"missing={loaded.missing_keys[:8]}, unexpected={loaded.unexpected_keys[:8]}"
            )
        model.eval()
        model = model.to(device).to(dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        self.model = model
        self.tokenizer = model.mt5_tokenizer
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.param_count = sum(parameter.numel() for parameter in model.parameters())
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        import gc

        gc.collect()
        import torch
        if self.device_name == "cuda":
            torch.cuda.empty_cache()

    def peak_memory_mb(self) -> dict[str, float]:
        if self.device_name != "cuda":
            return {}
        import torch
        return {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
        }

    def infer(self, source: dict, *, num_beams: int, max_new_tokens: int) -> tuple[str, dict[str, float]]:
        import torch
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype_name]
        device = self.device_name
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        moved = {}
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    moved[key] = value.to(device=device, dtype=dtype)
                elif key == "attention_mask":
                    moved[key] = value.to(device=device, dtype=torch.long)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        if device == "cuda":
            torch.cuda.synchronize()
        data_ready = time.perf_counter()
        target = {"gt_sentence": ["."], "gt_gloss": [""]}  # non-data sentinel (loss unused)
        with torch.inference_mode():
            stack_out = self.model(moved, target)
            if device == "cuda":
                torch.cuda.synchronize()
            forward_done = time.perf_counter()
            generated = self.model.generate(stack_out, max_new_tokens=max_new_tokens, num_beams=num_beams)
            if device == "cuda":
                torch.cuda.synchronize()
            generation_done = time.perf_counter()
            text = self.tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)[0].strip()
            decoded = time.perf_counter()
        return text, {
            "transfer_ms": (data_ready - started) * 1000.0,
            "forward_ms": (forward_done - data_ready) * 1000.0,
            "generation_ms": (generation_done - forward_done) * 1000.0,
            "decode_ms": (decoded - generation_done) * 1000.0,
            "model_total_ms": (decoded - started) * 1000.0,
        }


# ---------------------------------------------------------------------------
# upstream preprocessing (vendored from pinned commit eed438b datasets.py)
# ---------------------------------------------------------------------------
def crop_scale(motion, thr):
    import copy

    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


def load_part_kp(skeletons, confs, force_ok=False):
    import torch

    thr = 0.3
    kps_with_scores = {}
    scale = None
    for part in ["body", "left", "right", "face_all"]:
        kps = []
        confidences = []
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            if part == "body":
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == "left":
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == "right":
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == "face_all":
                hand_kp2d = skeleton[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]
            else:
                raise NotImplementedError
            kps.append(hand_kp2d)
            confidences.append(confidence)
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        if part == "body":
            result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)
        else:
            assert scale is not None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= thr] = 0
        kps_with_scores[part] = torch.tensor(result)
    return kps_with_scores


def bbox_4hands(left_keypoints, right_keypoints, hw):
    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)
        return (max_x + min_x) / 2, (max_y + min_y) / 2, (max_x - min_x), (max_y - min_y)

    H, W = hw
    left_keypoints = np.zeros([1, 21, 2]) if left_keypoints is None else left_keypoints
    right_keypoints = np.zeros([1, 21, 2]) if right_keypoints is None else right_keypoints
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x, left_mean_y = W * left_mean_x, H * left_mean_y
    left_diff_x, left_diff_y = W * left_diff_x, H * left_diff_y
    left_box_hw = max(max(left_diff_x), max(left_diff_y))

    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x, right_mean_y = W * right_mean_x, H * right_mean_y
    right_diff_x, right_diff_y = W * right_diff_x, H * right_diff_y
    right_box_hw = max(max(right_diff_x), max(right_diff_y))

    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)
    left_new_box = np.stack([left_mean_x - box_hw / 2, left_mean_y - box_hw / 2,
                             left_mean_x + box_hw / 2, left_mean_y + box_hw / 2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw / 2, right_mean_y - box_hw / 2,
                               right_mean_x + box_hw / 2, right_mean_y + box_hw / 2]).astype(np.int16)
    return left_new_box.transpose(1, 0), right_new_box.transpose(1, 0), box_hw


# ---------------------------------------------------------------------------
# RGB frames / hand crops (verbatim)
# ---------------------------------------------------------------------------
def read_video_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


HAND_CONF_LAYOUTS = ("legacy", "squeeze", "auto")


def resolve_hand_confs(confs: np.ndarray, layout: str) -> tuple[np.ndarray, str]:
    """Resolve the confidence-array layout for hand sampling; return (confs, effective).

    Frozen-protocol defect (verified by the lead, 2026-10-27): the official pose
    pkl stores ``scores`` as ``(T, 1, 133)`` while upstream ``sample_hand_indices``
    indexes ``confs[:, 91:112]``, which on that shape yields an empty selection --
    so the frozen 26.3026 baseline ran the fusion model with **zero RGB support**
    (a coincidence-level match to the paper's 26.36, not the paper protocol).

    * ``legacy`` (default): pass the array through unchanged -> reproduces the
      frozen behaviour exactly (hands None, zero support tensors).
    * ``squeeze``: ``(T, 1, 133)`` -> ``confs[:, 0, :]`` so hand sampling sees
      real frames (the corrected variant).
    * ``auto``: squeeze when ``ndim == 3 and shape[1] == 1``, else passthrough.
    """

    if layout not in HAND_CONF_LAYOUTS:
        raise ValueError(f"hand_conf_layout must be one of {HAND_CONF_LAYOUTS}, got {layout!r}")
    if layout == "legacy":
        return confs, "legacy"
    squeezable = confs.ndim == 3 and confs.shape[1] == 1
    if layout == "squeeze":
        if not squeezable:
            raise ValueError(
                f"hand_conf_layout='squeeze' requires confs shaped (T, 1, K), got {confs.shape}")
        return confs[:, 0, :], "squeeze"
    # auto
    if squeezable:
        return confs[:, 0, :], "squeeze"
    return confs, "legacy"


def sample_hand_indices(confs: np.ndarray, side: str, rng: np.random.Generator) -> np.ndarray | None:
    """Upstream low-confidence weighted hand-frame sampling for one hand.

    ``confs`` must be laid out ``(T, K)``; use :func:`resolve_hand_confs` first
    (the frozen script passed ``(T, 1, 133)`` straight in, which yields None for
    both hands -- the legacy default reproduces that verbatim).
    """

    sl = slice(91, 112) if side == "left" else slice(112, 133)
    if side == "left":
        key_ids = [95, 99, 103, 107, 111]
    else:
        key_ids = [95 + 21, 99 + 21, 103 + 21, 107 + 21, 111 + 21]
    filter_values = confs[:, sl].mean(-1)
    good = np.where(filter_values > 0.3)[0]
    if len(good) == 0:
        return None
    sub = confs[good][:, key_ids].min(-1)
    weights = np.max(sub) - sub + 1e-5
    probabilities = weights / np.sum(weights)
    sample_size = int(np.ceil(0.1 * len(good)))
    chosen = rng.choice(good, size=sample_size, replace=False, p=probabilities)
    return np.sort(chosen)


def build_support_rgb(skeletons: np.ndarray, confs: np.ndarray, frame_map: np.ndarray,
                      video_path: Path, rng: np.random.Generator,
                      hand_conf_layout: str = "legacy") -> dict:
    """Hand-crop RGB support dict (adapted from upstream load_support_rgb_dict).

    ``hand_conf_layout`` routes the confidence array through
    :func:`resolve_hand_confs` first; the returned dict additionally carries
    ``hand_conf_layout_effective`` and per-hand frame statistics so results stay
    auditable (legacy zero-support vs squeeze real crops).
    """

    import torch
    from PIL import Image
    from torchvision import transforms

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    resolved_confs, layout_effective = resolve_hand_confs(np.asarray(confs), hand_conf_layout)
    support = {"hand_conf_layout_effective": layout_effective,
               "hand_frames_left": 0, "hand_frames_right": 0}
    sampled = {}
    for side in ("left", "right"):
        chosen = sample_hand_indices(resolved_confs, side, rng)
        if chosen is not None:
            chosen = chosen[frame_map[chosen] >= 0]
        if chosen is None or len(chosen) == 0:
            support[f"{side}_sampled_indices"] = torch.tensor([-1])
            support[f"{side}_hands"] = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            support[f"{side}_skeletons_norm"] = torch.zeros(1, 21, 2)
            sampled[side] = None
        else:
            support[f"{side}_sampled_indices"] = torch.tensor(chosen)
            support[f"hand_frames_{side}"] = int(len(chosen))
            sampled[side] = chosen
    if sampled["left"] is None and sampled["right"] is None:
        return support
    if sampled["left"] is not None:
        left_sk = skeletons[sampled["left"], 0, 91:112]
    else:
        left_sk = None
    if sampled["right"] is not None:
        right_sk = skeletons[sampled["right"], 0, 112:133]
    else:
        right_sk = None
    # frame size from the local video (first readable mapped frame)
    probe_indices = []
    for side in ("left", "right"):
        if sampled[side] is not None:
            probe_indices.extend(int(i) for i in frame_map[sampled[side]])
    first_frame = read_video_frames(video_path, [min(probe_indices)])[0]
    H, W = first_frame.shape[:2]
    left_boxes, right_boxes, box_hw = bbox_4hands(left_sk, right_sk, (H, W))
    if box_hw == 0:
        for side in ("left", "right"):
            support[f"{side}_sampled_indices"] = torch.tensor([-1])
            support[f"{side}_hands"] = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            support[f"{side}_skeletons_norm"] = torch.zeros(1, 21, 2)
        return support
    factor = IMAGE_SIZE / box_hw
    for side, keypoints, boxes in (("left", left_sk, left_boxes), ("right", right_sk, right_boxes)):
        chosen = sampled[side]
        if chosen is None:
            support[f"{side}_hands"] = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            support[f"{side}_skeletons_norm"] = torch.zeros(1, 21, 2)
            continue
        local_indices = [int(i) for i in frame_map[chosen]]
        frames = read_video_frames(video_path, local_indices)
        hands = torch.zeros(len(chosen), 3, IMAGE_SIZE, IMAGE_SIZE)
        skeletons_norm = keypoints * np.array([W, H]) - boxes[:, None, [0, 1]]
        skeletons_norm = skeletons_norm / box_hw
        skeletons_norm = np.clip(skeletons_norm, 0, 1)
        for index, frame in enumerate(frames):
            box = boxes[index]
            crop = frame[box[1]:box[3], box[0]:box[2], :]
            crop = np.pad(crop, ((0, max(0, box_hw - crop.shape[0])),
                                 (0, max(0, box_hw - crop.shape[1])), (0, 0)),
                          mode="constant", constant_values=0)
            image = Image.fromarray(crop).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            hands[index] = data_transform(image)
        support[f"{side}_hands"] = hands
        support[f"{side}_skeletons_norm"] = torch.tensor(skeletons_norm)
    return support


# ---------------------------------------------------------------------------
# per-sample input construction (official archive / regression mode)
# ---------------------------------------------------------------------------
def load_official_pose(pose_dir: Path, sample: str) -> dict:
    import pickle

    with (Path(pose_dir) / f"{sample}.pkl").open("rb") as stream:
        pose = pickle.load(stream)
    keypoints = pose["keypoints"]
    scores = pose["scores"]
    start, end = int(pose["start"]), int(pose["end"])
    if end <= start:
        raise ValueError(f"invalid trim range for {sample}: {start}..{end}")
    return {"keypoints": keypoints, "scores": scores, "start": start, "end": end,
            "w_h": pose.get("w_h"), "frames_total": len(keypoints)}


def alignment_offset(alignment_dir: Path, sample: str, start: int) -> int:
    """local frame = pose frame + offset, using local RTMPose alignment metadata.

    ``alignment_dir`` MUST be the directory containing ``<sample>.npz`` files
    whose ``metadata_json`` carries ``alignment.best_start`` (i.e.
    ``artifacts/csldaily-video-test-627-20261026/pose``).  The standalone
    ``alignment/*.json`` directory is NOT supported here -- pointing at it
    silently yields offset 0 for every sample and breaks verbatim
    reproduction, so the loader validates the directory shape up front.
    """

    path = Path(alignment_dir) / f"{sample}.npz"
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"].item()))
    best_start = metadata.get("alignment", {}).get("best_start")
    if best_start is None:
        return 0
    return int(start) - int(best_start)


def build_sample_inputs(config: "UniSignRGBPoseConfig", sample: str) -> dict:
    """Build one sample's model inputs under the frozen protocol (CPU)."""

    pose = load_official_pose(config.pose_dir, sample)
    start, end = pose["start"], pose["end"]
    duration = end - start
    frames = list(range(duration))
    if duration > MAX_LENGTH:
        frames = sorted({int(round(value)) for value in np.linspace(0, duration - 1, MAX_LENGTH)})
    skeletons = [np.asarray(pose["keypoints"][start + index], dtype=np.float64) for index in frames]
    confs = [np.asarray(pose["scores"][start + index], dtype=np.float32) for index in frames]
    parts = load_part_kp(skeletons, confs)
    offset = alignment_offset(config.alignment_dir, sample, start)
    video_path = Path(config.video_dir) / f"{sample}.mp4"
    video_frames = None
    if video_path.exists():
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    frame_map = np.array([index + offset for index in frames], dtype=np.int64)
    frame_map[(frame_map < 0) | ((video_frames is not None) & (frame_map >= video_frames))] = -1
    rng = np.random.default_rng(SEED + int(hashlib.sha256(sample.encode("utf-8")).hexdigest()[:8], 16) % 100000)
    rgb = {}
    if video_path.exists():
        rgb = build_support_rgb(np.asarray(skeletons), np.asarray(confs), frame_map,
                                video_path, rng,
                                hand_conf_layout=config.hand_conf_layout)
    return {"sample": sample, "frames_total": duration, "frames": frames,
            "parts": parts, "rgb": rgb, "offset": offset,
            "video_frames": video_frames, "w_h": pose["w_h"],
            "hand_conf_layout": config.hand_conf_layout,
            "hand_conf_layout_effective": rgb.get("hand_conf_layout_effective"),
            "hand_frames_left": rgb.get("hand_frames_left", 0),
            "hand_frames_right": rgb.get("hand_frames_right", 0)}


def source_from_sample(inputs: dict, *, with_rgb: bool) -> dict:
    import torch

    source: dict[str, object] = {}
    length = len(inputs["frames"])
    for part, tensor in inputs["parts"].items():
        source[part] = tensor.unsqueeze(0)
    source["attention_mask"] = torch.ones((1, length), dtype=torch.long)
    source["src_length_batch"] = torch.tensor([length], dtype=torch.long)
    source["name_batch"] = [inputs["sample"]]
    if with_rgb:
        rgb = inputs["rgb"]
        for side in ("left", "right"):
            sampled = rgb[f"{side}_sampled_indices"]
            source[f"{side}_sampled_indices"] = sampled
            source[f"{side}_hands"] = rgb[f"{side}_hands"]
            source[f"{side}_skeletons_norm"] = rgb[f"{side}_skeletons_norm"]
            source[f"{side}_rgb_len"] = [int(sampled.numel())]
    return source


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class UniSignRGBPoseConfig:
    """Adapter configuration (lead constraints 1/2/5)."""

    pose_source: str = "rtmpose"
    pose_dir: Path | None = None
    alignment_dir: Path | None = None
    video_dir: Path = _DEFAULT_VIDEO_DIR
    checkpoint: Path = _DEFAULT_CHECKPOINT
    pose_only_checkpoint: Path = _DEFAULT_POSE_ONLY_CHECKPOINT
    source_dir: Path | None = None
    mt5_dir: Path | None = None
    enable_rgb: bool = True
    fallback_to_pose_only: bool = True
    hand_conf_layout: str = "legacy"
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_beams: int = NUM_BEAMS
    max_new_tokens: int = MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        if self.pose_source not in {"official_archive", "rtmpose"}:
            raise ValueError(f"pose_source must be 'official_archive' or 'rtmpose', got {self.pose_source!r}")
        if self.hand_conf_layout not in {"legacy", "squeeze", "auto"}:
            raise ValueError(
                f"hand_conf_layout must be 'legacy', 'squeeze' or 'auto', got {self.hand_conf_layout!r}")
        if self.pose_source == "official_archive":
            if self.pose_dir is None:
                raise ValueError("pose_source='official_archive' requires pose_dir (regression mode)")
            if self.alignment_dir is None:
                raise ValueError("pose_source='official_archive' requires alignment_dir (regression mode)")
            if self.fallback_to_pose_only:
                raise ValueError(
                    "fallback_to_pose_only is forbidden in official_archive (regression) mode: "
                    "failures must raise, never silently fall back")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        for name in ("pose_dir", "alignment_dir", "video_dir", "checkpoint", "pose_only_checkpoint"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if self.enable_rgb is False and self.fallback_to_pose_only is False and self.pose_source == "rtmpose":
            # pose-only without fallback is just the existing adapter; refuse the
            # degenerate combination so callers pick uni_sign_pose_slt directly.
            raise ValueError("enable_rgb=False with fallback disabled duplicates uni_sign_pose_slt")

    def protocol_parameters(self) -> dict:
        return {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "image_size": IMAGE_SIZE,
            "max_length": MAX_LENGTH,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "dtype": self.dtype,
            "frame_sampling": FRAME_SAMPLING,
            "hand_sampling": HAND_SAMPLING,
            "pose_source": self.pose_source,
            "enable_rgb": self.enable_rgb,
            "fallback_to_pose_only": self.fallback_to_pose_only,
            "hand_conf_layout": self.hand_conf_layout,
        }

    def preprocess_parameters(self) -> dict:
        return {
            "protocol_id": PROTOCOL_ID,
            "image_size": IMAGE_SIZE,
            "max_length": MAX_LENGTH,
            "frame_sampling": FRAME_SAMPLING,
            "hand_sampling": HAND_SAMPLING,
            "seed": SEED,
            "pose_source": self.pose_source,
            "hand_conf_layout": self.hand_conf_layout,
            "pose_dir": (str(self.pose_dir) if self.pose_dir is not None else None),
            "alignment_dir": (str(self.alignment_dir) if self.alignment_dir is not None else None),
            "video_dir": str(self.video_dir),
        }


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------
class UniSignRGBPoseAdapter:
    """Typed ModelAdapter over the frozen RGB+pose runtime (see module docstring)."""

    def __init__(self, config: UniSignRGBPoseConfig):
        if not isinstance(config, UniSignRGBPoseConfig):
            raise ValueError("UniSignRGBPoseAdapter requires a UniSignRGBPoseConfig")
        self.config = config
        self._runtime: UniSignRGBPoseRuntime | None = None
        self._pose_only_runtime: UniSignRGBPoseRuntime | None = None
        self._extractor = None                       # rtmpose mode, lazy
        self._rgb_failures: list[str] = []
        from .uni_sign import REPRESENTATION_ID, _register_representation
        _register_representation()
        self._representation_id = REPRESENTATION_ID
        self.capabilities = ModelCapabilities(
            MODEL_ID, _io.canonical_hash(self.config.protocol_parameters()), "csl", "text",
            (ModelInputContract(REPRESENTATION_ID, (None, None, None, 3), "uint8",
                                _io.canonical_hash(self.config.preprocess_parameters()),
                                ("file",)),),
            (), "pytorch_unisign_rgb_pose", ("cpu", "cuda"), ("clip",), SCORE_TYPE)

    # -- ModelAdapter protocol ------------------------------------------
    @property
    def recognizer(self) -> "UniSignRGBPoseAdapter":
        return self

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": "pytorch_unisign_rgb_pose",
            "protocol_id": PROTOCOL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "pose_source": self.config.pose_source,
            "enable_rgb": self.config.enable_rgb,
            "fallback_to_pose_only": self.config.fallback_to_pose_only,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "model_loaded": self._runtime is not None,
            "pose_only_loaded": self._pose_only_runtime is not None,
            "rgb_failures": list(self._rgb_failures),
        }

    def warmup(self) -> None:
        runtime = self._ensure_runtime()
        runtime.load()

    def close(self) -> None:
        for runtime in (self._runtime, self._pose_only_runtime):
            if runtime is not None and runtime.model is not None:
                runtime.close()
        self._runtime = None
        self._pose_only_runtime = None
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None

    def infer(self, source: Any, request: InferenceRequest, *,
              input_kind: str) -> tuple[RecognitionResult, dict[str, float]]:
        if input_kind != "file":
            raise ValueError("Uni-Sign RGB+pose supports file inputs only")
        self.capabilities.negotiate(request, input_kind=input_kind)
        request.check_active()
        started = time.perf_counter()
        result = self.recognize_file(source, request)
        total_ms = (time.perf_counter() - started) * 1000.0
        timing = {
            "preprocess_ms": float(result.quality.get("preprocess_ms", 0.0)),
            "inference_ms": float(result.quality.get("model_total_ms", 0.0)),
            "total_ms": total_ms,
        }
        return result, timing

    # -- recognition ------------------------------------------------------
    def recognize_file(self, path: Path | str, request: InferenceRequest) -> RecognitionResult:
        request.check_active()
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"input video not found: {path.name}")
        sample = path.stem
        request_started = time.perf_counter()
        try:
            if self.config.pose_source == "official_archive":
                inputs = build_sample_inputs(self.config, sample)
                source = source_from_sample(inputs, with_rgb=self.config.enable_rgb)
                preprocess_ms = (time.perf_counter() - request_started) * 1000.0
                return self._recognize_with_fallback(
                    inputs, request, rgb_error=None,
                    source=source, preprocess_ms=preprocess_ms)
            inputs, source, preprocess_ms = self._rtmpose_inputs(path, sample)
            return self._recognize_with_fallback(
                inputs, request, rgb_error=None,
                source=source, preprocess_ms=preprocess_ms)
        except _RGBUnavailable as error:
            return self._recognize_with_fallback(None, request, rgb_error=error)
        except RuntimeError as error:
            if self._is_rgb_failure(error):
                return self._recognize_with_fallback(None, request, rgb_error=error)
            raise

    def _recognize_with_fallback(self, inputs, request: InferenceRequest, *,
                                 rgb_error: Exception | None,
                                 source: dict | None = None,
                                 preprocess_ms: float = 0.0) -> RecognitionResult:
        """Run the RGB+pose runtime; fall back to pose-only per lead constraint 5."""

        if rgb_error is not None:
            if self.config.pose_source == "official_archive":
                # Regression mode: never fall back, never fabricate.
                raise RuntimeError(
                    f"RGB branch failed in official_archive (regression) mode: {rgb_error}") from rgb_error
            if not self.config.fallback_to_pose_only:
                raise RuntimeError(f"RGB branch failed and fallback is disabled: {rgb_error}") from rgb_error
            return self._pose_only_result(request, rgb_error, preprocess_ms)
        runtime = self._ensure_runtime()
        text, timings = runtime.infer(source, num_beams=self.config.num_beams,
                                      max_new_tokens=self.config.max_new_tokens)
        return self._make_result(request, text, timings, preprocess_ms,
                                 fallback_reason=None, fallback_model_id=None)

    def _pose_only_result(self, request: InferenceRequest, rgb_error: Exception,
                          preprocess_ms: float) -> RecognitionResult:
        runtime = self._ensure_pose_only_runtime()
        # Pose-only needs the pose parts without the RGB support dict.
        text, timings = runtime.infer(self._pose_only_source(), num_beams=self.config.num_beams,
                                      max_new_tokens=self.config.max_new_tokens)
        self._rgb_failures.append(f"{type(rgb_error).__name__}: {rgb_error}"[:240])
        return self._make_result(request, text, timings, preprocess_ms,
                                 fallback_reason="rgb_unavailable",
                                 fallback_model_id=FALLBACK_MODEL_ID,
                                 rgb_failure=str(rgb_error)[:240])

    def _make_result(self, request: InferenceRequest, text: str, timings: dict,
                     preprocess_ms: float, *, fallback_reason: str | None,
                     fallback_model_id: str | None,
                     rgb_failure: str | None = None) -> RecognitionResult:
        quality = {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "pose_source": self.config.pose_source,
            "enable_rgb": self.config.enable_rgb,
            "frame_sampling": FRAME_SAMPLING,
            "hand_sampling": HAND_SAMPLING,
            "preprocess_ms": round(preprocess_ms, 3),
            **{key: round(value, 3) for key, value in timings.items()},
            "fallback_reason": fallback_reason,
            "fallback_model_id": fallback_model_id,
        }
        if rgb_failure is not None:
            quality["rgb_failure"] = rgb_failure
        model_id = MODEL_ID if fallback_model_id is None else FALLBACK_MODEL_ID
        if not text or not text.strip():
            return RecognitionResult(
                status="rejected", sample_id=request.request_id, model_id=model_id,
                sign_language="csl", prediction=None, quality=quality,
                reason="empty_text_output", task_type="text",
                score_type=SCORE_TYPE, time_range=None)
        return RecognitionResult(
            status="recognized", sample_id=request.request_id, model_id=model_id,
            sign_language="csl",
            prediction=TextPrediction(text=text.strip(), language="zh", score=None),
            quality=quality, task_type="text", score_type=SCORE_TYPE, time_range=None)

    # -- rtmpose (production) input path ----------------------------------
    def _rtmpose_inputs(self, path: Path, sample: str):
        """Extract 133 keypoints with RtmposeExtractor and reuse the frozen pipeline."""
        extractor = self._ensure_extractor()
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        keypoints, scores = [], []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_keypoints, frame_scores = extractor.extract(frame)
            if frame_keypoints is None or len(frame_keypoints) == 0:
                raise _RGBUnavailable("rtmpose produced no keypoints")
            keypoints.append(np.asarray(frame_keypoints[0], dtype=np.float64))
            scores.append(np.asarray(frame_scores[0], dtype=np.float32))
        capture.release()
        if not keypoints:
            raise _RGBUnavailable(f"no decodable frames in {path}")
        height = frame.shape[0]
        width = frame.shape[1]
        skeletons = [np.stack([kp], axis=0) for kp in keypoints]      # [T,1,133,3]
        confs = [np.stack([sc], axis=0) for sc in scores]             # [T,1,133]
        # The frozen pipeline expects normalized [T,1,133,2] + [T,1,133];
        # rtmpose emits pixel coords, so normalize per axis like Uni-Sign does.
        normalized = [np.stack([kp[:133, :2] / np.array([width, height],
                                                        dtype=np.float64)], axis=0) for kp in keypoints]
        parts = load_part_kp(normalized, confs)
        duration = len(normalized)
        frames = list(range(duration))
        if duration > MAX_LENGTH:
            frames = sorted({int(round(value)) for value in np.linspace(0, duration - 1, MAX_LENGTH)})
        frame_map = np.array(frames, dtype=np.int64)
        rng = np.random.default_rng(SEED + int(hashlib.sha256(sample.encode("utf-8")).hexdigest()[:8], 16) % 100000)
        skeletons_array = np.asarray(normalized)
        confs_array = np.asarray([confs[index] for index in frames])
        rgb = build_support_rgb(skeletons_array, confs_array, frame_map, path, rng)
        inputs = {"sample": sample, "frames_total": duration, "frames": frames,
                  "parts": parts, "rgb": rgb, "offset": 0,
                  "video_frames": duration, "w_h": [width, height]}
        source = source_from_sample(inputs, with_rgb=self.config.enable_rgb)
        return inputs, source, 0.0

    def _pose_only_source(self) -> dict:
        # Fallback pose-only source: an injectable provider (tests, or a
        # production wiring around uni_sign_pose_slt); without one the
        # fallback path cannot fabricate inputs, so it fails loudly.
        provider = self.__dict__.get("_pose_only_source_provider")
        if callable(provider):
            return provider()
        raise RuntimeError("pose-only fallback requires a source provider "
                           "(wire uni_sign_pose_slt or inject _pose_only_source_provider)")

    # -- helpers ----------------------------------------------------------
    def _ensure_runtime(self) -> UniSignRGBPoseRuntime:
        if self._runtime is None:
            self._runtime = UniSignRGBPoseRuntime(
                self.config.checkpoint, rgb_support=True, device=self.config.device,
                dtype=self.config.dtype, source_dir=self.config.source_dir,
                mt5_dir=self.config.mt5_dir)
        return self._runtime

    def _ensure_pose_only_runtime(self) -> UniSignRGBPoseRuntime:
        if self._pose_only_runtime is None:
            self._pose_only_runtime = UniSignRGBPoseRuntime(
                self.config.pose_only_checkpoint, rgb_support=False,
                device=self.config.device, dtype=self.config.dtype,
                source_dir=self.config.source_dir, mt5_dir=self.config.mt5_dir)
        return self._pose_only_runtime

    def _ensure_extractor(self):
        if self._extractor is None:
            from .uni_sign import RtmposeExtractor
            self._extractor = RtmposeExtractor(
                mode=self.config.pose_source and "lightweight", backend="rtmpose",
                device=self.config.device if self.config.device != "cuda" else "cuda",
                cache_dir=Path(self.config.video_dir).parent / "rgb-pose-cache")
        return self._extractor

    @staticmethod
    def _is_rgb_failure(error: RuntimeError) -> bool:
        text = str(error).lower()
        return ("cannot open video" in text or "cannot read frame" in text
                or "out of memory" in text or "empty hand" in text)


class _RGBUnavailable(RuntimeError):
    """Internal: RGB branch cannot provide crops (fallback eligible)."""


def load_unisign_rgb_pose_adapter(path: str | Path, options: Any = None) -> UniSignRGBPoseAdapter:
    """Bundle loader: manifest.json carries the UniSignRGBPoseConfig fields.

    Registration lives in ``slbridge/model_adapters.py`` (lazy loader, backend
    id ``unisign_rgb_pose_slt``); this module deliberately does NOT register
    on import to avoid a duplicate-id error against that canonical entry.
    """

    manifest_path = Path(path) / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("backend") != "unisign_rgb_pose_slt":
        raise ValueError(f"manifest backend must be unisign_rgb_pose_slt: {manifest_path}")
    fields = {key: value for key, value in raw.items() if key != "backend"}
    for key in ("pose_dir", "alignment_dir", "video_dir", "checkpoint",
                "pose_only_checkpoint", "source_dir", "mt5_dir"):
        if fields.get(key) is not None:
            fields[key] = str(Path(path) / fields[key]) if not Path(fields[key]).is_absolute() \
                else fields[key]
    config = UniSignRGBPoseConfig(**fields)
    if config.pose_source == "official_archive":
        # Guard against the classic misconfiguration: a directory without
        # <sample>.npz files silently produces offset 0 everywhere and breaks
        # verbatim reproduction.  Fail loudly at load time instead.
        npz_files = list(Path(config.alignment_dir).glob("*.npz"))
        if not npz_files:
            raise ValueError(
                "alignment_dir must contain <sample>.npz files with "
                "metadata_json.alignment.best_start (the pose/ npz directory); "
                f"the JSON alignment directory is not supported: {config.alignment_dir}")
    requested_device = getattr(options, "device", None) if options is not None else None
    if requested_device in {"cpu", "cuda"}:      # "auto" keeps the manifest's device
        config = replace(config, device=requested_device)
    return UniSignRGBPoseAdapter(config)
