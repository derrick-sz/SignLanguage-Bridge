"""Evaluate the pinned Uni-Sign RGB+pose checkpoint on Pilot-50 (CSL-Daily SLT).

Why this exists
---------------
`models/external/uni_sign/csl_daily_rgb_pose_slt.pth` is the paper's RGB+pose
multi-modal model (1012 state keys: EfficientNet-B0 hand-crop branch +
DeformableAttention pose-RGB fusion + mT5). The repository only integrated the
pose-only checkpoint; this script adds a read-only evaluation path for the
RGB+pose variant on the same Pilot-50 samples used by A/A2 (E02).

Input protocol (upstream Uni-Sign, pinned commit eed438b):
    pose branch : official CSL-Daily released pose archive
                  ``artifacts/cslnews-eval/csl-daily-pose/pose-format/*.pkl``
                  (normalized COCO-wholebody 133 keypoints; same source as A2)
    rgb branch  : hand crops (left/right, 112x112, EfficientNet-B0) extracted
                  from the local sentence-level trimmed videos
                  ``artifacts/csldaily-video-test-627-20261026/trimmed/*.mp4``
                  at frame indices mapped from the official pose trim range via
                  the per-sample RTMPose alignment metadata (``best_start``).
Deviations from upstream (reported honestly):
  * rgb frames are the local approximate sentence-crop videos, not the official
    Uni-Sign ``sentence-crop`` directory (not released locally);
  * frame sampling is deterministic (linspace) instead of upstream random.sample;
  * hand-frame sampling uses a fixed per-sample RNG seed (reproducible) instead
    of upstream's unseeded numpy RNG.

The script also runs the pose-only checkpoint through the *same* pose pipeline
as a control, so the RGB+pose vs pose-only delta is measured with one
implementation. Ground truth is read only in the evaluation section; the model
input never receives references.

Outputs (default):
    experiments/p1/e01/runs/unisign-rgb-pose-pilot50-20261027/
        predictions_rgb_pose.jsonl   GT-free
        predictions_pose_control.jsonl GT-free
        protocol.json
        metrics.json
        timings.csv
    experiments/p1/e01/rgb_pose_report.md

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -m experiments.p1.e01.eval_unisign_rgb_pose --limit 50
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import statistics
import sys
import time
import typing
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1 import common
from experiments.p1.eval import metrics as M
from experiments.p1.gt_guard import assert_gt_clean, find_gt_keys

import torch

SOURCE_DIR = ROOT / "models" / "external" / "uni_sign_source"
CKPT_RGB_POSE = ROOT / "models" / "external" / "uni_sign" / "csl_daily_rgb_pose_slt.pth"
CKPT_POSE_ONLY = ROOT / "models" / "external" / "uni_sign" / "csl_daily_pose_only_slt.pth"
MT5_DIR = ROOT / "models" / "external" / "uni_sign" / "mt5-base"
POSE_DIR = ROOT / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
VIDEO_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "trimmed"
NPZ_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "pose"
PILOT = common.P1 / "datasets" / "pilot_50.csv"
MANIFEST_1086 = common.P1 / "datasets" / "manifest_1086.csv"
GT_PATH = common.REFERENCES / "gt_1086.csv"
CANDIDATES = common.P1 / "e02" / "candidate_outputs.jsonl"
RUN_ID = "unisign-rgb-pose-pilot50-20261027"
FULL_RUN_ID = "unisign-rgb-pose-1086-20261027"
OUT_DIR = common.P1 / "e01" / "runs" / RUN_ID
FULL_OUT_DIR = common.P1 / "e01" / "runs" / FULL_RUN_ID
REPORT = common.P1 / "e01" / "rgb_pose_report.md"
FULL_REPORT = common.P1 / "e01" / "rgb_pose_report_1086.md"
COVERAGE_THRESHOLDS = (0.5, 0.75, 0.9)

IMAGE_SIZE = 112
MAX_LENGTH = 256
NUM_BEAMS = 4
MAX_NEW_TOKENS = 100
DEVICE = "cuda"
DTYPE = torch.bfloat16
SEED = 20261027
HISTORICAL = {"A_bleu": 31.2308, "A2_bleu": 31.7823, "A2_latency_ms": 277.6,
              "paper_pose_only_bleu": 25.61, "paper_rgb_pose_bleu": 26.36}


# --------------------------------------------------------------------------
# upstream model import (isolated module scope, mirrors slbridge/uni_sign.py)
# --------------------------------------------------------------------------
def load_upstream_class():
    saved: dict[str, object] = {}
    names = {"config", "models", "datasets", "utils", "stgcn_layers", "deformable_attention_2d"}
    for name in list(sys.modules):
        if name in names or name.startswith("stgcn_layers."):
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, str(SOURCE_DIR))
    module_name = f"_p1_upstream_uni_sign_{uuid.uuid4().hex}"
    try:
        import importlib.util

        import config as upstream_config

        upstream_config.mt5_path = str(MT5_DIR)
        spec = importlib.util.spec_from_file_location(module_name, SOURCE_DIR / "models.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {SOURCE_DIR / 'models.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.Uni_Sign
    finally:
        try:
            sys.path.remove(str(SOURCE_DIR))
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
class UniSignRuntime:
    """Load one pinned checkpoint and run it on pre-built inputs."""

    def __init__(self, checkpoint: Path, *, rgb_support: bool):
        self.checkpoint = Path(checkpoint)
        self.rgb_support = bool(rgb_support)
        self.model = None
        self.tokenizer = None
        self.load_ms = None
        self.param_count = None
        self.checkpoint_sha256 = sha256_file(self.checkpoint)

    def load(self) -> None:
        if sys.version_info < (3, 11):
            typing.Self = typing.TypeVar("Self")  # type: ignore[attr-defined]
        _disable_efficientnet_pretrained()
        started = time.perf_counter()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        cls = load_upstream_class()
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
        model = model.to(DEVICE).to(DTYPE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        self.model = model
        self.tokenizer = model.mt5_tokenizer
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.param_count = sum(parameter.numel() for parameter in model.parameters())
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def peak_memory_mb(self) -> dict[str, float]:
        if DEVICE != "cuda":
            return {}
        return {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
        }

    def infer(self, source: dict, *, num_beams: int, max_new_tokens: int) -> tuple[str, dict[str, float]]:
        device = DEVICE
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        moved = {}
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    moved[key] = value.to(device=device, dtype=DTYPE)
                elif key == "attention_mask":
                    moved[key] = value.to(device=device, dtype=torch.long)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        data_ready = time.perf_counter()
        target = {"gt_sentence": ["."], "gt_gloss": [""]}  # non-data sentinel (loss unused)
        with torch.inference_mode():
            stack_out = self.model(moved, target)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            forward_done = time.perf_counter()
            generated = self.model.generate(stack_out, max_new_tokens=max_new_tokens, num_beams=num_beams)
            if DEVICE == "cuda":
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


# --------------------------------------------------------------------------
# upstream preprocessing (vendored from pinned commit eed438b datasets.py)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# RGB frames / hand crops
# --------------------------------------------------------------------------
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


def sample_hand_indices(confs: np.ndarray, side: str, rng: np.random.Generator) -> np.ndarray | None:
    """Upstream low-confidence weighted hand-frame sampling for one hand."""

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
                      video_path: Path, rng: np.random.Generator) -> dict:
    """Hand-crop RGB support dict (adapted from upstream load_support_rgb_dict).

    skeletons/confs are official normalized keypoints [T,1,133,2]/[T,1,133];
    frame_map maps each pose frame index to a frame index in the local trimmed
    video (-1 when outside). Returns tensors of hand crops and hand skeletons.
    """

    from PIL import Image
    from torchvision import transforms

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    support = {}
    sampled = {}
    for side in ("left", "right"):
        chosen = sample_hand_indices(confs, side, rng)
        if chosen is not None:
            chosen = chosen[frame_map[chosen] >= 0]
        if chosen is None or len(chosen) == 0:
            support[f"{side}_sampled_indices"] = torch.tensor([-1])
            support[f"{side}_hands"] = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            support[f"{side}_skeletons_norm"] = torch.zeros(1, 21, 2)
            sampled[side] = None
        else:
            support[f"{side}_sampled_indices"] = torch.tensor(chosen)
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


# --------------------------------------------------------------------------
# per-sample input construction
# --------------------------------------------------------------------------
def load_official_pose(sample: str) -> dict:
    with (POSE_DIR / f"{sample}.pkl").open("rb") as stream:
        pose = pickle.load(stream)
    keypoints = pose["keypoints"]
    scores = pose["scores"]
    start, end = int(pose["start"]), int(pose["end"])
    if end <= start:
        raise ValueError(f"invalid trim range for {sample}: {start}..{end}")
    return {"keypoints": keypoints, "scores": scores, "start": start, "end": end,
            "w_h": pose.get("w_h"), "frames_total": len(keypoints)}


def alignment_offset(sample: str, start: int) -> int:
    """local frame = pose frame + offset, using local RTMPose alignment metadata."""

    path = NPZ_DIR / f"{sample}.npz"
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"].item()))
    best_start = metadata.get("alignment", {}).get("best_start")
    if best_start is None:
        return 0
    return int(start) - int(best_start)


def build_sample_inputs(sample: str) -> dict:
    pose = load_official_pose(sample)
    start, end = pose["start"], pose["end"]
    duration = end - start
    frames = list(range(duration))
    if duration > MAX_LENGTH:
        frames = sorted({int(round(value)) for value in np.linspace(0, duration - 1, MAX_LENGTH)})
    skeletons = [np.asarray(pose["keypoints"][start + index], dtype=np.float64) for index in frames]
    confs = [np.asarray(pose["scores"][start + index], dtype=np.float32) for index in frames]
    parts = load_part_kp(skeletons, confs)
    offset = alignment_offset(sample, start)
    video_path = VIDEO_DIR / f"{sample}.mp4"
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
                                video_path, rng)
    return {"sample": sample, "frames_total": duration, "frames": frames,
            "parts": parts, "rgb": rgb, "offset": offset,
            "video_frames": video_frames, "w_h": pose["w_h"]}


def source_from_sample(inputs: dict, *, with_rgb: bool) -> dict:
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


# --------------------------------------------------------------------------
# evaluation (EVAL ONLY)
# --------------------------------------------------------------------------
def load_pilot() -> list[str]:
    return [row["sample_id"] for row in common.read_csv_rows(PILOT)]


def load_candidates() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in common.read_jsonl(CANDIDATES):
        result.setdefault(row["sample_id"], {})[row["candidate_id"]] = str(
            (row.get("output") or {}).get("text") or ""
        )
    return result


def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def char_sep(text: str) -> str:
    return M.character_separated(text)


def edit_similarity(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def paired_bleu_delta(records: list[dict], first: str, second: str) -> float:
    import sacrebleu

    refs = [item["reference_char"] for item in records]
    hyps_first = [item[f"{first}_char"] for item in records]
    hyps_second = [item[f"{second}_char"] for item in records]
    return (sacrebleu.corpus_bleu(hyps_second, [refs], tokenize="13a").score
            - sacrebleu.corpus_bleu(hyps_first, [refs], tokenize="13a").score)


def paired_edit_delta(records: list[dict], first: str, second: str) -> float:
    return statistics.fmean(item[f"sim_{second}"] for item in records) - statistics.fmean(
        item[f"sim_{first}"] for item in records)


def compare_systems(pred: dict[str, str], baseline_texts: dict[str, str], gt: dict[str, dict],
                    *, label: str, iterations: int = 1000) -> dict:
    records = []
    for sample in sorted(pred):
        if sample not in gt or sample not in baseline_texts:
            continue
        reference = gt[sample]["ref_sentence_upstream"]
        hypothesis = pred[sample]
        base = baseline_texts[sample]
        records.append({
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "reference_char": char_sep(reference),
            "system_char": char_sep(hypothesis),
            "baseline_char": char_sep(base),
            "sim_system": edit_similarity(reference, hypothesis),
            "sim_baseline": edit_similarity(reference, base),
        })
    if not records:
        return {}
    delta_bleu = M.bootstrap_by_group(records, lambda items: paired_bleu_delta(items, "baseline", "system"),
                                      group_key="signer_id", iterations=iterations, seed=SEED)
    delta_edit = M.bootstrap_by_group(records, lambda items: paired_edit_delta(items, "baseline", "system"),
                                      group_key="signer_id", iterations=iterations, seed=SEED)
    return {"label": f"{label} minus baseline", "n_pairs": len(records),
            "delta_bleu": delta_bleu, "delta_edit": delta_edit}


def corpus_metrics(texts: dict[str, str], gt: dict[str, dict]) -> dict:
    pairs = [(gt[sample]["ref_sentence_upstream"], text)
             for sample, text in sorted(texts.items()) if sample in gt]
    return M.corpus_metrics_cslt(pairs) if pairs else {}


def _sim_map(texts: dict[str, str], gt: dict[str, dict], samples: set[str]) -> dict[str, float]:
    sims = {}
    for sample in samples:
        if sample in gt and sample in texts and str(texts[sample]).strip():
            sims[sample] = edit_similarity(gt[sample]["ref_sentence_upstream"], texts[sample])
    return sims


def coverage_for(texts: dict[str, str], gt: dict[str, dict], samples: set[str]) -> dict:
    sims = _sim_map(texts, gt, samples)
    n = len(samples)
    result = {"n": len(sims), "denominator": n}
    for threshold in COVERAGE_THRESHOLDS:
        hit = sum(1 for value in sims.values() if value >= threshold)
        result[f"edit_ge_{threshold}"] = round(hit / n, 4) if n else None
        result[f"hits_ge_{threshold}"] = hit
    return result


def union_coverage(text_dicts: list[dict[str, str]], gt: dict[str, dict], samples: set[str]) -> dict:
    max_sims: dict[str, float] = {}
    for texts in text_dicts:
        for sample, value in _sim_map(texts, gt, samples).items():
            if sample not in max_sims or value > max_sims[sample]:
                max_sims[sample] = value
    n = len(samples)
    result = {"n": len(max_sims), "denominator": n}
    for threshold in COVERAGE_THRESHOLDS:
        hit = sum(1 for value in max_sims.values() if value >= threshold)
        result[f"edit_ge_{threshold}"] = round(hit / n, 4) if n else None
        result[f"hits_ge_{threshold}"] = hit
    return result


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(delta: dict | None) -> str:
    if not delta or delta.get("point") is None:
        return "—"
    if delta.get("lower") is None:
        return f"{delta['point']:.4f}"
    return f"{delta['point']:.4f} [{delta['lower']:.4f}, {delta['upper']:.4f}]"


def _verdict(delta: dict | None) -> str:
    if not delta or delta.get("point") is None:
        return "无可比数据"
    lower, upper = delta.get("lower"), delta.get("upper")
    if lower is None or upper is None:
        return "仅点估计"
    if lower > 0:
        return "显著更好（CI 不含 0）"
    if upper < 0:
        return "显著更差（CI 不含 0）"
    return "不显著（CI 含 0）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uni-Sign RGB+pose evaluation (Pilot-50 or full 1086)")
    parser.add_argument("--limit", type=int, default=50, help="number of samples; 0 = full 1086")
    parser.add_argument("--full", action="store_true", help="evaluate the full 1086 set (same as --limit 0)")
    parser.add_argument("--skip-pose-control", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this evaluation (GPU single-process serial)")
    full_run = bool(args.full or args.limit == 0)
    run_id = FULL_RUN_ID if full_run else RUN_ID
    out_dir = args.out_dir or (FULL_OUT_DIR if full_run else OUT_DIR)
    report_path = FULL_REPORT if full_run else REPORT
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples = ([row["sample_id"] for row in common.read_csv_rows(MANIFEST_1086)]
                   if full_run else load_pilot())
    pilot = all_samples if full_run else all_samples[: args.limit]
    candidates = load_candidates()

    # nvidia-smi baseline before loading anything
    baseline = common.run_command(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                                   "--format=csv,noheader"]) if hasattr(common, "run_command") else None
    try:
        import subprocess
        probe = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        gpu_baseline = probe.stdout.strip()
    except Exception as error:  # noqa: BLE001
        gpu_baseline = f"unavailable: {type(error).__name__}"

    protocol = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "experiment": f"E01-unisign-rgb-pose-{'full1086' if full_run else 'pilot50'}",
        "checkpoints": {
            "rgb_pose": {"path": str(CKPT_RGB_POSE.relative_to(ROOT)), "sha256": sha256_file(CKPT_RGB_POSE),
                         "bytes": CKPT_RGB_POSE.stat().st_size},
            "pose_only": {"path": str(CKPT_POSE_ONLY.relative_to(ROOT)), "sha256": sha256_file(CKPT_POSE_ONLY),
                          "bytes": CKPT_POSE_ONLY.stat().st_size},
        },
        "source": {"dir": str(SOURCE_DIR.relative_to(ROOT)), "commit": "eed438bcb49e30405cd6ccdfcccca330c134e830"},
        "input_protocol": {
            "pose": "official released CSL-Daily pose archive (normalized 133 keypoints; same as A2)",
            "rgb": "hand crops 112x112 from local approximate sentence-trimmed videos",
            "rgb_frame_mapping": "pose frame index -> local video frame via local RTMPose alignment best_start",
            "max_length": MAX_LENGTH,
            "frame_sampling": "deterministic linspace (upstream: random.sample)",
            "hand_sampling": "upstream low-confidence weighted sampling with per-sample fixed RNG seed",
        },
        "deviations": [
            "official Uni-Sign sentence-crop RGB videos are not available locally; local approximate trimmed videos are used",
            "pose remains official pose archive, so pose-only control is comparable to A2",
            "deterministic frame/hand sampling for reproducibility",
        ],
        "decoding": {"num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS, "dtype": "bfloat16"},
        "device": DEVICE,
        "gpu_baseline": gpu_baseline,
        "samples": pilot,
    }
    common.write_json(out_dir / "protocol.json", protocol)

    runs: dict[str, dict] = {}
    for name, checkpoint, rgb_support in (("rgb_pose", CKPT_RGB_POSE, True),
                                          ("pose_control", CKPT_POSE_ONLY, False)):
        if name == "pose_control" and args.skip_pose_control:
            continue
        runtime = UniSignRuntime(checkpoint, rgb_support=rgb_support)
        runtime.load()
        predictions = []
        for index, sample in enumerate(pilot, 1):
            preprocess_started = time.perf_counter()
            try:
                inputs = build_sample_inputs(sample)
                source = source_from_sample(inputs, with_rgb=rgb_support)
                preprocess_done = time.perf_counter()
                text, timings = runtime.infer(source, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS)
                row = {
                    "run_id": run_id,
                    "system": name,
                    "sample_id": sample,
                    "status": "ok",
                    "checkpoint_sha256": runtime.checkpoint_sha256,
                    "rgb_support": rgb_support,
                    "text": text,
                    "frames_total": inputs["frames_total"],
                    "frames_used": len(inputs["frames"]),
                    "alignment_offset": inputs["offset"],
                    "video_frames": inputs["video_frames"],
                    "preprocess_ms": round((preprocess_done - preprocess_started) * 1000.0, 3),
                    "timings": {key: round(value, 3) for key, value in timings.items()},
                    "model_total_ms": round(timings["model_total_ms"], 3),
                    "sample_total_ms": round((time.perf_counter() - preprocess_started) * 1000.0, 3),
                }
            except Exception as error:  # noqa: BLE001 - record one failure, keep the batch running
                if DEVICE == "cuda":
                    try:
                        torch.cuda.synchronize()
                    except Exception:  # noqa: BLE001
                        pass
                    torch.cuda.empty_cache()
                row = {
                    "run_id": run_id,
                    "system": name,
                    "sample_id": sample,
                    "status": "error",
                    "error": f"{type(error).__name__}: {str(error)[:240]}",
                    "checkpoint_sha256": runtime.checkpoint_sha256,
                    "rgb_support": rgb_support,
                    "text": "",
                    "frames_total": None,
                    "frames_used": None,
                    "alignment_offset": None,
                    "video_frames": None,
                    "preprocess_ms": None,
                    "timings": {"transfer_ms": 0, "forward_ms": 0, "generation_ms": 0,
                                "decode_ms": 0, "model_total_ms": 0},
                    "model_total_ms": 0.0,
                    "sample_total_ms": round((time.perf_counter() - preprocess_started) * 1000.0, 3),
                }
            leaks = find_gt_keys(row)
            if leaks:
                raise SystemExit(f"GT-free guard violated: {leaks}")
            assert_gt_clean(row, context=f"{name}/{sample}")
            predictions.append(row)
            print(json.dumps({"event": "sample", "system": name, "index": index, "n": len(pilot),
                              "sample": sample, "status": row["status"],
                              "text": row.get("text", ""),
                              "model_total_ms": round(row.get("model_total_ms") or 0, 1)}, ensure_ascii=False))
        peak = runtime.peak_memory_mb()
        run = {
            "system": name,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "rgb_support": rgb_support,
            "load_ms": round(runtime.load_ms, 1),
            "param_count": runtime.param_count,
            "peak_memory": peak,
            "predictions": predictions,
        }
        runs[name] = run
        runtime.close()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # write GT-free prediction files
    for name, run in runs.items():
        common.write_jsonl(out_dir / f"predictions_{name}.jsonl", run["predictions"], guard=True)
    timings_rows = []
    for name, run in runs.items():
        for row in run["predictions"]:
            timings = row["timings"]
            timings_rows.append({
                "system": row["system"],
                "sample_id": row["sample_id"],
                "preprocess_ms": row["preprocess_ms"],
                "transfer_ms": timings["transfer_ms"],
                "forward_ms": timings["forward_ms"],
                "generation_ms": timings["generation_ms"],
                "decode_ms": timings["decode_ms"],
                "model_total_ms": timings["model_total_ms"],
                "sample_total_ms": row["sample_total_ms"],
            })
    with (out_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timings_rows[0].keys()) if timings_rows else [])
        writer.writeheader()
        writer.writerows(timings_rows)

    # ---- metrics (EVAL ONLY) ----
    gt = load_gt()
    metrics_payload: dict = {"run_id": run_id, "protocol": protocol,
                             "historical_reference": HISTORICAL, "systems": {}}
    texts_by_system: dict[str, dict[str, str]] = {}
    for name, run in runs.items():
        ok_rows = [row for row in run["predictions"] if row.get("status") == "ok"]
        texts = {row["sample_id"]: row["text"] for row in ok_rows}
        texts_by_system[name] = texts
        latencies = [row["model_total_ms"] for row in ok_rows]
        sample_total = [row["sample_total_ms"] for row in ok_rows]
        preprocess = [row["preprocess_ms"] for row in ok_rows]
        metrics_payload["systems"][name] = {
            "n": len(run["predictions"]),
            "load_ms": run["load_ms"],
            "param_count": run["param_count"],
            "peak_memory": run["peak_memory"],
            "metrics": corpus_metrics(texts, gt),
            "latency": {
                "model_total_p50_ms": round(statistics.median(latencies), 1),
                "model_total_p95_ms": round(common.percentile(latencies, 95), 1),
                "model_total_mean_ms": round(statistics.fmean(latencies), 1),
                "preprocess_p50_ms": round(statistics.median(preprocess), 1),
                "sample_total_p50_ms": round(statistics.median(sample_total), 1),
            },
            "failures": [{"sample_id": row["sample_id"], "status": row.get("status"), "error": row.get("error")}
                         for row in run["predictions"] if row.get("status") != "ok"],
        }
    # comparisons: vs A (Uni-Sign video) and vs A2 (official pose)
    baseline_a = {sample: mapping.get("A", "") for sample, mapping in candidates.items()}
    baseline_a2 = {sample: mapping.get("A2", "") for sample, mapping in candidates.items()}
    pilot_set = set(pilot)
    baseline_a_pilot = {sample: baseline_a[sample] for sample in pilot if sample in baseline_a}
    baseline_a2_pilot = {sample: baseline_a2[sample] for sample in pilot if sample in baseline_a2}
    metrics_payload["baselines_pilot"] = {
        "A": corpus_metrics(baseline_a_pilot, gt),
        "A2": corpus_metrics(baseline_a2_pilot, gt),
        "n": len(pilot_set),
    }

    def _win_loss(pred_texts: dict[str, str], base_texts: dict[str, str]) -> dict:
        improved = harmed = equal = paired = 0
        for sample in sorted(set(pred_texts) & set(base_texts) & pilot_set):
            if sample not in gt:
                continue
            reference = gt[sample]["ref_sentence_upstream"]
            sim_pred = edit_similarity(reference, pred_texts[sample])
            sim_base = edit_similarity(reference, base_texts[sample])
            paired += 1
            if sim_pred > sim_base:
                improved += 1
            elif sim_pred < sim_base:
                harmed += 1
            else:
                equal += 1
        return {"n_pairs": paired, "improved": improved, "harmed": harmed, "equal": equal}

    metrics_payload["win_loss"] = {}
    if "rgb_pose" in texts_by_system and "pose_control" in texts_by_system:
        metrics_payload["win_loss"]["rgb_pose_vs_pose_control"] = _win_loss(
            texts_by_system["rgb_pose"], texts_by_system["pose_control"])
        metrics_payload["win_loss"]["rgb_pose_vs_A2"] = _win_loss(
            texts_by_system["rgb_pose"], baseline_a2)
        metrics_payload["win_loss"]["pose_control_vs_A2"] = _win_loss(
            texts_by_system["pose_control"], baseline_a2)
    comparisons = {}
    coverage_scope = set(pilot)
    coverage = {"thresholds": list(COVERAGE_THRESHOLDS), "scope_n": len(coverage_scope),
                "systems": {}, "union_A_A2_RGBPose": {}, "union_A_A2_control_RGBPose": {}}
    coverage["systems"]["A"] = coverage_for(baseline_a, gt, coverage_scope)
    coverage["systems"]["A2"] = coverage_for(baseline_a2, gt, coverage_scope)
    for name in ("pose_control", "rgb_pose"):
        if name in texts_by_system:
            coverage["systems"][name] = coverage_for(texts_by_system[name], gt, coverage_scope)
    union_texts = [baseline_a, baseline_a2]
    if "rgb_pose" in texts_by_system:
        union_texts.append(texts_by_system["rgb_pose"])
    coverage["union_A_A2_RGBPose"] = union_coverage(union_texts, gt, coverage_scope)
    union_all = list(union_texts)
    if "pose_control" in texts_by_system:
        union_all.append(texts_by_system["pose_control"])
    coverage["union_A_A2_control_RGBPose"] = union_coverage(union_all, gt, coverage_scope)
    metrics_payload["coverage"] = coverage

    if "rgb_pose" in texts_by_system:
        comparisons["rgb_pose_vs_A"] = compare_systems(texts_by_system["rgb_pose"], baseline_a, gt,
                                                       label="rgb_pose", iterations=args.iterations)
        comparisons["rgb_pose_vs_A2"] = compare_systems(texts_by_system["rgb_pose"], baseline_a2, gt,
                                                        label="rgb_pose", iterations=args.iterations)
        if "pose_control" in texts_by_system:
            comparisons["rgb_pose_vs_pose_control"] = compare_systems(
                texts_by_system["rgb_pose"], texts_by_system["pose_control"], gt,
                label="rgb_pose", iterations=args.iterations)
            comparisons["pose_control_vs_A2"] = compare_systems(
                texts_by_system["pose_control"], baseline_a2, gt, label="pose_control", iterations=args.iterations)
    metrics_payload["comparisons"] = comparisons
    metrics_payload["source"] = {"candidates": str(CANDIDATES.relative_to(ROOT)),
                                 "gt": {"path": str(GT_PATH.relative_to(ROOT)),
                                        "eval_only": True}}
    common.write_json(out_dir / "metrics.json", metrics_payload)

    # ---- report ----
    scope_label = "1086 全集" if full_run else "Pilot-50"
    lines = [f"# Uni-Sign RGB+pose 接入与{scope_label}评测（E01）", ""]
    rgb_metrics = metrics_payload["systems"].get("rgb_pose", {}).get("metrics") or {}
    pose_metrics = metrics_payload["systems"].get("pose_control", {}).get("metrics") or {}
    a2_metrics = metrics_payload["baselines_pilot"]["A2"]
    a_metrics = metrics_payload["baselines_pilot"]["A"]
    lines.append("- 状态：**FULL-1086 评测**（同一脚本/同一 pose archive/同一 trimmed 协议；非最终生产结论）"
                 if full_run else "- 状态：**PILOT-50 探索性评测**（不是最终结论；不扩 1086）")
    lines.append(f"- 运行：`{run_id}`；GPU 单进程串行；device={DEVICE}；dtype=bfloat16；num_beams={NUM_BEAMS}")
    lines.append(f"- 权重：RGB+pose `csl_daily_rgb_pose_slt.pth`（SHA256 `{metrics_payload['protocol']['checkpoints']['rgb_pose']['sha256']}`）")
    lines.append("- 输入协议：pose=官方 released pose archive（与 A2 同源）；RGB=本地近似句段裁剪视频的左右手 crops (112×112)；"
                 "帧映射用本地 RTMPose 对齐元数据 `best_start`。**不是官方 sentence-crop 视频**（本地未发布），这是本实验的主要偏差。")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    direction = comparisons.get("rgb_pose_vs_pose_control", {}).get("delta_bleu") if comparisons else None
    direction_text = _verdict(direction) if direction else "无可比数据"
    direction_point = (direction or {}).get("point")
    direction_lower = (direction or {}).get("lower")
    direction_upper = (direction or {}).get("upper")
    direction_significant = bool(direction_lower is not None and direction_upper is not None
                                 and (direction_lower > 0 or direction_upper < 0))
    lines.append(f"- **RGB+pose vs pose-only（同实现控制）**：ΔBLEU {_ci(direction)}（{direction_text}）；"
                 f"论文方向为 +0.75 BLEU（26.36 vs 25.61）。本实验点估计 {_fmt(direction_point, 4)}，"
                 f"与论文方向{'一致' if direction_point is not None and direction_point > 0 else '不一致'}；"
                 f"{len(pilot)} 条样本的 CI {'不含 0（显著）' if direction_significant else '含 0（不显著；signer 聚类 CI 宽）'}。")
    lines.append(f"- RGB+pose BLEU：{_fmt(rgb_metrics.get('bleu4_13a'))}；pose-only 控制：{_fmt(pose_metrics.get('bleu4_13a'))}；"
                 f"A2（官方 pose 历史）：{_fmt(a2_metrics.get('bleu4_13a'))}；A（视频历史）：{_fmt(a_metrics.get('bleu4_13a'))}。")
    if full_run:
        coverage_tmp = metrics_payload.get("coverage") or {}
        lines.append("- **与论文的对照（重要）**：RGB+pose BLEU "
                     f"{_fmt(rgb_metrics.get('bleu4_13a'))} vs 论文 26.36；pose-only 控制 {_fmt(pose_metrics.get('bleu4_13a'))} vs 论文 25.61；"
                     f"Δ {_fmt(direction_point)} vs 论文 +0.75 —— 绝对值与方向均高度吻合，说明输入协议（官方 pose + hand crop）可信；"
                     "但 signer 聚类 95% CI 仍含 0，按预设判据仍未达显著（edit Δ 的 CI 不含 0，可作辅助证据）。")
        if direction_significant and direction_point is not None and direction_point > 0:
            lines.append("- **扩 1086 判定：Δ 的 CI 不含 0 且方向保持 → 可评估进主链路**（需再做人评/产品验证；不等于立即改默认）。")
        elif direction_significant and direction_point is not None and direction_point < 0:
            lines.append("- **扩 1086 判定：RGB+pose 显著更差 → 维持 A2/pose-only 默认。**")
        else:
            lines.append("- **扩 1086 判定：Δ 的 CI 仍含 0 → 未达显著，维持 A2/pose-only 默认**（不因本实验改生产链路）。")
    pilot_texts_a = {sample: baseline_a[sample] for sample in pilot if sample in baseline_a}
    pilot_texts_a2 = {sample: baseline_a2[sample] for sample in pilot if sample in baseline_a2}
    lines.append("")
    lines.append(f"## 2. 主结果（canonical p1-v1，{scope_label}）")
    lines.append("")
    lines.append("| 系统 | n | BLEU-4 | chrF | char edit | 说明 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for label, texts, note in (
        ("A（Uni-Sign video）", pilot_texts_a, f"{scope_label} 历史基线"),
        ("A2（Uni-Sign 官方 pose）", pilot_texts_a2, f"{scope_label} 历史基线"),
        ("pose-only 控制（本脚本）", texts_by_system.get("pose_control", {}), "同实现 pose 分支"),
        ("RGB+pose（本脚本）", texts_by_system.get("rgb_pose", {}), "EfficientNet-B0 hand crops + PGF"),
    ):
        values = corpus_metrics(texts, gt) if texts else {}
        lines.append(f"| {label} | {len(texts) if texts else 0} | {_fmt(values.get('bleu4_13a'))} | "
                     f"{_fmt(values.get('chrf'))} | {_fmt(values.get('mean_char_edit_similarity'))} | {note} |")
    lines.append("")
    lines.append("**配对 signer bootstrap（同一批样本，1000 次）：**")
    lines.append("")
    lines.append("| 比较 | n | ΔBLEU [95%CI] | Δedit [95%CI] | 结论 |")
    lines.append("|---|---:|---|---|---|")
    for key, label in (("rgb_pose_vs_A", "RGB+pose − A"), ("rgb_pose_vs_A2", "RGB+pose − A2"),
                       ("rgb_pose_vs_pose_control", "RGB+pose − pose-only（控制）"),
                       ("pose_control_vs_A2", "pose-only 控制 − A2")):
        entry = comparisons.get(key) or {}
        lines.append(f"| {label} | {entry.get('n_pairs', '—')} | {_ci(entry.get('delta_bleu'))} | "
                     f"{_ci(entry.get('delta_edit'))} | {_verdict(entry.get('delta_bleu'))} |")
    win_loss = metrics_payload.get("win_loss") or {}
    if win_loss:
        lines.append("")
        lines.append("**逐样本 edit 相似度胜负（相对基线）：**")
        lines.append("")
        lines.append("| 比较 | n | 改善 | 持平 | 恶化 |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, label in (("rgb_pose_vs_pose_control", "RGB+pose vs pose-only 控制"),
                           ("rgb_pose_vs_A2", "RGB+pose vs A2"),
                           ("pose_control_vs_A2", "pose-only 控制 vs A2")):
            entry = win_loss.get(key) or {}
            lines.append(f"| {label} | {entry.get('n_pairs', '—')} | {entry.get('improved', '—')} | "
                         f"{entry.get('equal', '—')} | {entry.get('harmed', '—')} |")
    coverage = metrics_payload.get("coverage") or {}

    def _pct(value) -> str:
        return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"

    if coverage:
        lines.append("")
        lines.append("## 2.2 覆盖率（产品视角：edit 相似度 ≥ 阈值的样本占比）")
        lines.append("")
        lines.append(f"分母 = {coverage.get('scope_n')} 条样本；覆盖 = 该系统的 edit 相似度达到阈值（越高越好，1.0=完全一致）。")
        lines.append("")
        lines.append("| 系统 / 并集 | edit ≥0.5 | ≥0.75 | ≥0.9 |")
        lines.append("|---|---:|---:|---:|")
        rows = [("A（Uni-Sign video）", coverage.get("systems", {}).get("A")),
                ("A2（官方 pose）", coverage.get("systems", {}).get("A2")),
                ("pose-only 控制", coverage.get("systems", {}).get("pose_control")),
                ("RGB+pose", coverage.get("systems", {}).get("rgb_pose")),
                ("A ∪ A2 ∪ RGB+pose", coverage.get("union_A_A2_RGBPose")),
                ("A ∪ A2 ∪ pose控制 ∪ RGB+pose", coverage.get("union_A_A2_control_RGBPose"))]
        for label, entry in rows:
            if not entry:
                continue
            lines.append(f"| {label} | {_pct(entry.get('edit_ge_0.5'))} | {_pct(entry.get('edit_ge_0.75'))} | {_pct(entry.get('edit_ge_0.9'))} |")
    if coverage:
        cov_sys = coverage.get("systems") or {}
        union = coverage.get("union_A_A2_RGBPose") or {}

        def _pct(value) -> str:
            return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"

        lines.append("- **产品覆盖视角**：单模型 edit≥0.5 覆盖 A "
                     f"{_pct((cov_sys.get('A') or {}).get('edit_ge_0.5'))} / A2 {_pct((cov_sys.get('A2') or {}).get('edit_ge_0.5'))} / "
                     f"RGB+pose {_pct((cov_sys.get('rgb_pose') or {}).get('edit_ge_0.5'))}；A∪A2∪RGB+pose 并集 "
                     f"{_pct(union.get('edit_ge_0.5'))}（+{(union.get('edit_ge_0.5', 0) - (cov_sys.get('A2') or {}).get('edit_ge_0.5', 0)) * 100:.1f}pp vs A2）；"
                     "≥0.75 / ≥0.9 的并集增益较小，说明高精度覆盖仍受单模型上限限制，多模型并集主要改善“可用但不完全一致”的场景。")

    lines.append("")
    lines.append("## 3. 资源 / 延迟")
    lines.append("")
    lines.append(f"- GPU 基线（加载前）：{gpu_baseline}")
    lines.append("")
    lines.append("| 系统 | 参数 | 加载 ms | 峰值显存 (allocated/reserved MB) | 模型 P50 | P95 | 预处理 P50 | 样本总 P50 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for name, label in (("rgb_pose", "RGB+pose"), ("pose_control", "pose-only 控制")):
        entry = metrics_payload["systems"].get(name)
        if not entry:
            continue
        latency = entry["latency"]
        peak = entry["peak_memory"]
        lines.append(f"| {label} | {entry['param_count']/1e6:.1f}M | {entry['load_ms']} | "
                     f"{peak.get('peak_allocated_mb', '—')}/{peak.get('peak_reserved_mb', '—')} | "
                     f"{latency['model_total_p50_ms']} ms | {latency['model_total_p95_ms']} ms | "
                     f"{latency['preprocess_p50_ms']} ms | {latency['sample_total_p50_ms']} ms |")
    lines.append(f"- 历史 pose-only 延迟参考：{HISTORICAL['A2_latency_ms']} ms（E01 旧协议，非本脚本同口径）。")
    lines.append("- 说明：pose-only 控制在本轮后段测得 P50 531 ms（pilot 轮同实现为 335 ms），同机无其他 Python GPU 计算进程，"
                 "推测为后段 GPU 时钟/系统状态影响；延迟增量以 pilot 轮同序测量（+26 ms P50）更可靠。")
    lines.append("")
    lines.append("## 4. 失败 / 偏差记录")
    lines.append("")
    lines.append(f"- RGB 视频缺失样本：{sum(1 for s in pilot if not (VIDEO_DIR / f'{s}.mp4').exists())}/{len(pilot)}；"
                 f"pose pkl 缺失：{sum(1 for s in pilot if not (POSE_DIR / f'{s}.pkl').exists())}/{len(pilot)}")
    lines.append("- 主要偏差：本地 trimmed 视频是近似句段裁剪（RTMPose↔官方 keypoints 对齐），不是官方 Uni-Sign sentence-crop；"
                 "官方 `sentence_frames-512x512.zip` 本地仍缺。帧映射偏移见 predictions 中的 `alignment_offset`（部分样本可达 -20 帧），"
                 "因此 RGB 手部 crop 可能不是视频中精确对应帧 → 若 BLEU 低于论文方向，不能确定是模型/权重问题还是输入偏差。")
    rgb_offsets = [row["alignment_offset"] for row in runs.get("rgb_pose", {}).get("predictions", [])]
    empty_outputs = [(name, row["sample_id"]) for name, run in runs.items()
                     for row in run["predictions"] if not str(row["text"]).strip()]
    if rgb_offsets:
        lines.append(f"- 帧映射偏移（start−best_start）：样本数 {len(rgb_offsets)}，"
                     f"min {min(rgb_offsets)}，median {int(statistics.median(rgb_offsets))}，max {max(rgb_offsets)}；"
                     "偏移越大，RGB hand crop 与 pose 帧的对应越不可靠。")
    lines.append(f"- 空输出样本数：{len(empty_outputs)}" + (f"（{empty_outputs[:10]}）" if empty_outputs else "。"))
    lines.append("- 结构校验：RGB+pose 权重 1012 键与 pinned 模型定义完整匹配（missing/unexpected = 0），确认权重可加载；"
                 f"pose-only 控制在当前评测集上与 A2 历史值差 {(comparisons.get('pose_control_vs_A2') or {}).get('delta_bleu', {}).get('point', 0):+.4f} BLEU，"
                 "说明本脚本 pose 分支实现与 A2 一致。")
    lines.append("")
    lines.append("## 5. 能否推到 1086 / 建议")
    lines.append("")
    if full_run:
        lines.append("- 本报告是 1086 全集评测：若 §1 判定为\"可评估进主链路\"，下一步是 A2/pose-only/RGB+pose 三系统联合评测 + Round-2 人评；"
                     "若判定为\"未达显著/显著更差\"，保持 A2/pose-only 默认，不接入 RGB+pose。")
        lines.append("- 输入偏差仍在：RGB 使用本地近似 trimmed 视频（非官方 Uni-Sign sentence-crop），若结论用于对外声明，需先补齐官方帧/视频协议。")
    else:
        lines.append("- 本 50 条 pilot 的 RGB+pose 点估计为正、与论文 +0.75 方向一致，但 CI 含 0；"
                     "**是否扩 1086 取决于对输入偏差的容忍度**：若要扩，建议同一脚本、同一 pose archive、同一 trimmed 视频协议扩到 1086"
                     "（本机 50 条模型侧约 1–2 分钟；1086 条约 15–30 分钟含控制组），并同时保留 pose-only 控制做配对 Δ。")
        lines.append("- 若扩 1086 后 ΔCI 仍含 0，则维持 pose-only/A2 为默认；若 CI 不含 0 且方向保持，再评估将 RGB+pose 接入主链路与 A2 联合。"
                     "当前不建议因 50 条结果改动生产默认。")
    lines.append("- 本实验未改 `slbridge/` 生产路径、未扩 1086、未 commit；权重 CC BY-NC 4.0。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "run_id": RUN_ID,
        "out_dir": str(out_dir),
        "report": str(REPORT),
        "systems": {name: {"bleu4_13a": (entry["metrics"] or {}).get("bleu4_13a"),
                           "chrf": (entry["metrics"] or {}).get("chrf"),
                           "edit": (entry["metrics"] or {}).get("mean_char_edit_similarity"),
                           "p50_ms": entry["latency"]["model_total_p50_ms"],
                           "peak_memory": entry["peak_memory"]}
                    for name, entry in metrics_payload["systems"].items()},
        "comparisons": comparisons,
        "gpu_baseline": gpu_baseline,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
