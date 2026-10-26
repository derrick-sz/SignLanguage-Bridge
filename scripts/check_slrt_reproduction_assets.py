"""Report which SLRT Online reproduction assets are present, missing or blocked.

This is a read-only inventory. It intentionally treats CSL-Daily data as
external input: absence is reported, never filled with placeholders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_status(path: Path, expected_sha: str | None = None) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    actual = sha256(path)
    result = {"status": "present", "path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    if expected_sha:
        result["sha256_expected"] = expected_sha
        result["sha256_ok"] = actual == expected_sha
        result["status"] = "present" if actual == expected_sha else "hash-mismatch"
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--legacy-env", type=Path, default=Path(r"F:\DSH-envs\slrt-legacy"))
    args = p.parse_args(argv)

    asset = ROOT / "models/external/EMNLP24_Online/CSL-Daily"
    online = ROOT / "models/external/SLRT_Online"
    upstream = ROOT / "Paper-algorithm/SLRT-main"
    data_dir = upstream / "data/csl-daily"
    legacy_python = args.legacy_env / "python.exe"
    legacy_mmcv = args.legacy_env / "Lib/site-packages/mmcv"

    weights = {
        "cslr_best.ckpt": file_status(asset / "ckpts/cslr_best.ckpt", "e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d"),
        "csl_iso_with_blank.vocab": file_status(asset / "meta/csl_iso_with_blank.vocab", "0c4984f0dd3e7897b8c8d9e6e99a848b60814a5d9cf7a49baf7cc9b9cd124c76"),
        "phoenix cslr_best.ckpt": file_status(asset.parent / "Phoenix-2014T/ckpts/cslr_best.ckpt"),
        "phoenix slt_best.ckpt": file_status(asset.parent / "Phoenix-2014T/ckpts/slt_best.ckpt"),
        "K400 S3D": file_status(online / "pretrained_models/s3ds_actioncls_ckpt"),
        "base mBART": file_status(online / "pretrained_models/mBart/pytorch_model.bin"),
        "CSL gloss assets": file_status(online / "pretrained_models/mBART_zh_gloss_assets/gloss2ids.pkl"),
        "full-vocab mBART_zh fallback": file_status(online / "pretrained_models/mBART_zh_fullvocab/pytorch_model.bin"),
        "HRNet-W48 DARK": file_status(ROOT / "models/external/uni_sign/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"),
        "Faster-RCNN person": file_status(ROOT / "models/external/uni_sign/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"),
    }
    layouts = {
        "official best.ckpt": file_status(upstream / "Online/CSLR/results/csl-daily_ISLR/ckpts/best.ckpt"),
        "official vocab": file_status(data_dir / "csl_iso_with_blank.vocab"),
        "ISLR train meta": file_status(data_dir / "csl_iso_center_label_bag2items_halfblk.train"),
        "ISLR dev meta": file_status(data_dir / "csl_iso_with_blank.dev"),
        "ISLR test meta": file_status(data_dir / "csl_iso_with_blank.test"),
        "S2G gloss ids": file_status(data_dir / "csl_s2g_gloss2ids.pkl"),
        "TwoStream S2G ckpt": file_status(upstream / "Online/CTC_fusion/results/csl-daily_s2g/ckpts/best.ckpt"),
    }
    data_required = (
        "sentence_frames-512x512.zip",
        "keypoints_hrnet_dark_coco_wholebody_iso.pkl",
        "keypoints_hrnet_dark_coco_wholebody.pkl",
        "csl.train",
        "csl.dev",
        "csl.test",
    )
    data = {name: file_status(data_dir / name) for name in data_required}
    configs = {
        "mmpose HRNet config": file_status(ROOT / "models/external/legacy_openmmlab/mmpose-v0.29.0/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"),
        "mmdet Faster-RCNN config": file_status(ROOT / "models/external/legacy_openmmlab/mmdet-v2.25.3/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"),
    }
    tools = {
        "legacy python": file_status(legacy_python),
        "legacy mmcv": {"status": "present" if legacy_mmcv.is_dir() else "missing", "path": str(legacy_mmcv)},
        "HRNet extraction script": file_status(ROOT / "scripts/extract_slrt_hrnet_pose.py"),
        "SLT prediction converter": file_status(ROOT / "scripts/convert_slrt_predictions_to_slt.py"),
        "official layout script": file_status(ROOT / "scripts/setup_slrt_official_layout.py"),
    }
    missing = [name for name, value in {**weights, **layouts, **data, **configs, **tools}.items() if value["status"] != "present"]
    report = {
        "schema_version": 1,
        "weights": weights,
        "official_layout": layouts,
        "data": data,
        "configs": configs,
        "tools": tools,
        "missing": missing,
        "note": "Data absence is expected until the official CSL-Daily release is placed under Paper-algorithm/SLRT-main/data/csl-daily.",
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("SLRT Online asset inventory")
        for section in ("weights", "official_layout", "configs", "tools", "data"):
            print(f"[{section}]")
            for name, value in report[section].items():
                print(f"  {value['status']:14s} {name}")
        print(f"missing: {len(missing)}")
        for name in missing:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
