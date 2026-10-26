"""Extract official HRNet-W48 DARK COCO-WholeBody133 keypoints into SLRT NPZ.

Run this script inside the legacy OpenMMLab environment (mmpose 0.29 + mmdet 2.25
+ mmcv-full 1.7, CPU or CUDA), not inside the project's Torch 2.9 environment.
The output is consumable by:

    python -m slbridge.slrt.cli video --video ... --pose-backend hrnet-npz --pose-npz ...

The implementation follows the asset path used by the upstream ``gen_pose.py``:
Faster-RCNN person detector, HRNet-W48 wholebody DARK pose estimator, 133 points
in a 512-square reference frame, x/y/confidence unchanged, and RGB image order
(the upstream reader is PIL-based). The project's in-process ``HRNetFrontend``
is a separate, unverified path and is not used here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

COORDINATE_SYSTEM = "coco_wholebody_133_xyc_512_v1"
FRONTEND = "hrnet_w48_dark"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DET_CONFIG = PROJECT_ROOT / "models/external/legacy_openmmlab/mmdet-v2.25.3/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
DEFAULT_DET_CKPT = PROJECT_ROOT / "models/external/uni_sign/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
DEFAULT_POSE_CONFIG = PROJECT_ROOT / "models/external/legacy_openmmlab/mmpose-v0.29.0/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
DEFAULT_POSE_CKPT = PROJECT_ROOT / "models/external/uni_sign/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="source video")
    parser.add_argument("--output", type=Path, required=True, help="output .npz for --pose-backend hrnet-npz")
    parser.add_argument("--summary", type=Path, default=None, help="optional JSON sidecar; defaults next to the npz")
    parser.add_argument("--det-config", type=Path, default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-checkpoint", type=Path, default=DEFAULT_DET_CKPT)
    parser.add_argument("--hrnet-config", type=Path, default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--hrnet-checkpoint", type=Path, default=DEFAULT_POSE_CKPT)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; the legacy CPU environment only has CPU Torch")
    parser.add_argument("--resize", type=int, default=512, help="square resize before detection/pose (official CSL-Daily reference is 512)")
    parser.add_argument("--color-order", choices=("rgb", "bgr"), default="rgb",
                        help="image order passed to OpenMMLab; upstream gen_pose reads RGB via PIL")
    parser.add_argument("--det-score-thr", type=float, default=0.5)
    parser.add_argument("--det-area-thr", type=float, default=1600.0)
    parser.add_argument("--max-persons", type=int, default=1, help="keep at most N highest-score person boxes for pose inference")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--fps", type=float, default=0.0, help="override FPS; 0 reads the container")
    parser.add_argument("--allow-missing", action="store_true",
                        help="write zero keypoints for frames without a person; default fails instead of hiding gaps")
    parser.add_argument("--metadata-extra", default=None, help="JSON object merged into metadata")
    return parser.parse_args(argv)


def _load_frame(capture, resize: int, color_order: str = "rgb"):
    import cv2
    from PIL import Image
    ok, frame = capture.read()
    if not ok:
        return None
    rgb = Image.fromarray(frame[..., ::-1]).resize((resize, resize), Image.Resampling.BILINEAR)
    image = np.asarray(rgb)
    if color_order == "bgr":
        image = image[..., ::-1].copy()
    return image


def _make_detector(args):
    import torch  # noqa: F401  (load native Torch DLLs before OpenMMLab CV stack)
    from mmdet.apis import init_detector
    return init_detector(str(args.det_config), str(args.det_checkpoint), device=args.device)


def _make_pose_model(args):
    from mmpose.apis import init_pose_model
    return init_pose_model(str(args.hrnet_config), str(args.hrnet_checkpoint), device=args.device)


def _detect_person(detector, image, args):
    from mmdet.apis import inference_detector
    result = inference_detector(detector, image)
    boxes = np.asarray(result[0], dtype=np.float32)
    if boxes.size == 0:
        return None
    keep = (boxes[:, 4] >= args.det_score_thr) & ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= args.det_area_thr)
    boxes = boxes[keep]
    if len(boxes) == 0:
        return None
    # Upstream sorts by bbox score and uses the last (highest-score) person.
    boxes = boxes[np.argsort(boxes[:, 4])][-max(1, args.max_persons):]
    return boxes[-1]


def _pose(detector, pose_model, image, box, args):
    from mmpose.apis import inference_top_down_pose_model
    if box is None:
        return None
    poses = inference_top_down_pose_model(pose_model, image, [dict(bbox=box)], format="xyxy")[0]
    if not poses:
        return None
    poses = sorted(poses, key=lambda item: float(item["bbox"][-1]))
    return np.asarray(poses[-1]["keypoints"], dtype=np.float32)


def main(argv=None) -> int:
    args = parse_args(argv)
    import cv2
    import torch
    import mmcv
    import mmdet
    import mmpose

    for name in ("det_config", "det_checkpoint", "hrnet_config", "hrnet_checkpoint", "video"):
        path = getattr(args, name)
        if not Path(path).is_file():
            raise SystemExit(f"missing {name}: {path}")
    if args.color_order != "rgb":
        print("warning: upstream gen_pose reads RGB frames; --color-order bgr is an experiment", file=sys.stderr)
    if args.resize <= 0:
        raise SystemExit("--resize must be positive")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".json")

    started = time.perf_counter()
    detector = _make_detector(args)
    pose_model = _make_pose_model(args)
    if str(detector.CLASSES[0]).lower() != "person":
        raise SystemExit("detector must have 'person' as COCO class 0")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = args.fps if args.fps > 0 else float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if not np.isfinite(fps) or fps <= 0:
        raise SystemExit("invalid FPS; pass --fps explicitly")
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    keypoints = []
    missing = []
    t0 = time.perf_counter()
    while True:
        if args.max_frames and len(keypoints) >= args.max_frames:
            break
        image = _load_frame(capture, args.resize, args.color_order)
        if image is None:
            break
        index = len(keypoints)
        box = _detect_person(detector, image, args)
        pose = _pose(detector, pose_model, image, box, args)
        if pose is None:
            missing.append(index)
            if not args.allow_missing:
                raise SystemExit(f"no reliable person/pose at frame {index}; rerun with --allow-missing only for diagnostics")
            pose = np.zeros((133, 3), dtype=np.float32)
        if pose.shape != (133, 3):
            raise SystemExit(f"unexpected pose shape at frame {index}: {pose.shape}")
        if not np.isfinite(pose).all():
            raise SystemExit(f"non-finite keypoints at frame {index}")
        if (pose[:, 2] < 0).any() or (pose[:, 2] > 1).any():
            pose = pose.copy()
            pose[:, 2] = np.clip(pose[:, 2], 0.0, 1.0)
        keypoints.append(pose)
    capture.release()
    elapsed = time.perf_counter() - t0
    if not keypoints:
        raise SystemExit("no frames decoded")
    keypoints = np.stack(keypoints).astype(np.float32)
    timestamps = (np.arange(len(keypoints), dtype=np.float64) / fps).astype(np.float64)
    frame_indices = np.arange(len(keypoints), dtype=np.int64)

    extra = {}
    if args.metadata_extra:
        extra = json.loads(args.metadata_extra)
        if not isinstance(extra, dict):
            raise SystemExit("--metadata-extra must be a JSON object")
    metadata = {
        "schema_version": 1,
        "video_path": str(args.video.resolve()),
        "video_sha256": sha256(args.video),
        "source_frames": source_frames,
        "frames": int(len(keypoints)),
        "fps": fps,
        "frontend": FRONTEND,
        "coordinate_system": COORDINATE_SYSTEM,
        "color_order": args.color_order,
        "resize": [args.resize, args.resize],
        "detector": {
            "config": str(args.det_config.resolve()),
            "config_sha256": sha256(args.det_config),
            "checkpoint": str(args.det_checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.det_checkpoint),
            "score_threshold": args.det_score_thr,
            "area_threshold": args.det_area_thr,
            "max_persons": args.max_persons,
        },
        "pose": {
            "config": str(args.hrnet_config.resolve()),
            "config_sha256": sha256(args.hrnet_config),
            "checkpoint": str(args.hrnet_checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.hrnet_checkpoint),
        },
        "missing_frames": missing,
        "source": "scripts/extract_slrt_hrnet_pose.py",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mmcv": mmcv.__version__,
            "mmdet": mmdet.__version__,
            "mmpose": mmpose.__version__,
        },
        "extra": extra,
    }
    np.savez_compressed(
        args.output,
        keypoints=keypoints,
        frame_indices=frame_indices,
        timestamps=timestamps,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "summary": str(summary_path.resolve()),
        "frames": len(keypoints),
        "missing_frames": len(missing),
        "fps": fps,
        "elapsed_seconds": round(elapsed, 3),
        "torch": torch.__version__,
        "mmdet": mmdet.__version__,
        "mmpose": mmpose.__version__,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
