"""Build the upstream SLRT ``Online/CSLR`` asset layout from local files.

The upstream entrypoints resolve everything relative to ``Online/CSLR`` (or the
repository root): ``results/{model_dir}/ckpts/best.ckpt``, ``data/csl-daily/...``
and ``pretrained_models/...``. This module only creates links and an explicit
manifest; it never fabricates videos, keypoints or annotation files.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

CSL_DAILY = "csl-daily"
CSL_DAILY_MODEL_DIR = "results/csl-daily_ISLR"
CSL_DAILY_DATA_FILES = (
    "csl.train",
    "csl.dev",
    "csl.test",
    "csl_iso_with_blank.vocab",
    "csl_s2g_gloss2ids.pkl",
    "keypoints_hrnet_dark_coco_wholebody_iso.pkl",
    "keypoints_hrnet_dark_coco_wholebody.pkl",
    "sentence_frames-512x512.zip",
    "csl_iso_center_label_bag2items_halfblk.train",
    "csl_iso_with_blank.dev",
    "csl_iso_with_blank.test",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: str | Path, target: str | Path) -> str:
    """Create a hardlink when possible and a regular copy otherwise.

    Returns ``"hardlink"`` or ``"copy"``. Existing targets are kept only when
    their SHA256 equals the source; a different target is a hard error.
    """
    source = Path(source).resolve()
    target = Path(target).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source missing: {source}")
    source_hash = sha256(source)
    if target.exists():
        if not target.is_file():
            raise ValueError(f"target exists but is not a file: {target}")
        if sha256(target) != source_hash:
            raise ValueError(f"target exists with different content: {target}")
        try:
            same_inode = target.stat().st_ino == source.stat().st_ino and target.drive.lower() == source.drive.lower()
        except OSError:
            same_inode = False
        return "hardlink" if same_inode else "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def expected_data_markdown(data_dir: Path) -> str:
    lines = [
        "# CSL-Daily data expected by SLRT Online",
        "",
        "Put the official CSL-Daily files here; never fabricate them. The upstream",
        "`slide_csl-daily.yaml` and ISLR/fusion configs resolve `../../data/csl-daily`.",
        "",
        "Sources: the dataset page is <http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/>",
        "(also referenced by `TwoStreamNetwork/download.sh`); the authors' processed",
        "meta/keypoint SharePoint links in `Online/CSLR/README.md` returned 404 on",
        "2026-10-25 and must be rechecked with the authors.",
        "",
        "## Required for online CSLR evaluation (`slide_csl-daily.yaml`)",
        "",
        "- `sentence_frames-512x512.zip` — 512×512 JPEG frames, one directory per video (`sentence_frames-512x512/{video}/{frame:06d}.jpg`).",
        "- `keypoints_hrnet_dark_coco_wholebody_iso.pkl` — dict name → `[T,133,3]` x/y/confidence in 512-pixel coordinates.",
        "- `csl.train`, `csl.dev`, `csl.test` — sentence-level annotation pickles with `name`, `gloss`, `text`, `num_frames`.",
        "- `csl_iso_with_blank.vocab` — 2001-class JSON list, `<blank>` at index 0 (already linked by this tool).",
        "",
        "## Required for ISLR training/segmentation and CTC fusion",
        "",
        "- `csl_iso_center_label_bag2items_halfblk.train`",
        "- `csl_iso_with_blank.dev`, `csl_iso_with_blank.test`",
        "- `csl_s2g_gloss2ids.pkl`",
        "- `keypoints_hrnet_dark_coco_wholebody.pkl` (non-iso variant used by fusion)",
        "",
        "Expected files:",
        "",
    ]
    lines.extend(f"- `{name}`" for name in CSL_DAILY_DATA_FILES)
    lines.append("")
    return "\n".join(lines)


def build_layout(
    upstream_root: str | Path,
    asset_root: str | Path,
    *,
    pose_asset_root: str | Path | None = None,
    pretrained_root: str | Path | None = None,
    data_source: str | Path | None = None,
    two_stream_checkpoint: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    """Create the documented links and return a JSON-serialisable manifest."""
    upstream_root = Path(upstream_root).resolve()
    asset_root = Path(asset_root).resolve()
    online = upstream_root / "Online" / "CSLR"
    if not online.is_dir():
        raise FileNotFoundError(f"upstream Online/CSLR missing: {online}")

    data_dir = upstream_root / "data" / CSL_DAILY
    data_dir.mkdir(parents=True, exist_ok=True)
    actions = {}
    ckpt_source = asset_root / "ckpts" / "cslr_best.ckpt"
    ckpt_target = upstream_root / "Online" / "CSLR" / CSL_DAILY_MODEL_DIR / "ckpts" / "best.ckpt"
    actions[str(ckpt_target)] = link_or_copy(ckpt_source, ckpt_target)

    vocab_source = asset_root / "meta" / "csl_iso_with_blank.vocab"
    vocab_target = upstream_root / "data" / CSL_DAILY / "csl_iso_with_blank.vocab"
    actions[str(vocab_target)] = link_or_copy(vocab_source, vocab_target)

    # The released CSL-Daily meta package already contains word-level ISLR
    # annotations (and the vocabulary). Link them when present; do not pretend
    # they are the sentence-level csl.train/dev/test files.
    for name in (
        "csl_iso_center_label_bag2items_halfblk.train",
        "csl_iso_with_blank.dev",
        "csl_iso_with_blank.test",
    ):
        source = asset_root / "meta" / name
        if source.is_file():
            actions[str(upstream_root / "data" / CSL_DAILY / name)] = link_or_copy(
                source, upstream_root / "data" / CSL_DAILY / name
            )

    # TwoStream-SLR's CSL gloss dictionary is shipped with the upstream mirror
    # and is the vocabulary used by CTC_fusion (4 specials + 2000 glosses).
    two_stream_gloss2ids = upstream_root / "TwoStreamNetwork" / "data" / "csl-daily" / "gloss2ids.pkl"
    if two_stream_gloss2ids.is_file():
        actions[str(upstream_root / "data" / CSL_DAILY / "csl_s2g_gloss2ids.pkl")] = link_or_copy(
            two_stream_gloss2ids, upstream_root / "data" / CSL_DAILY / "csl_s2g_gloss2ids.pkl"
        )

    # Optional: the author-provided continuous CSL-Daily package uses
    # `csl-daily.*` / `csl-daily-keypoints.pkl`. Create the aliases expected by
    # the Online configs. On Windows use a directory junction to keep the 1.9 GB
    # keypoint file on the data drive instead of duplicating it.
    if data_source:
        source_dir = Path(data_source).resolve()
        alias_pairs = (
            ("csl-daily.train", "csl.train"),
            ("csl-daily.dev", "csl.dev"),
            ("csl-daily.test", "csl.test"),
            ("csl-daily-keypoints.pkl", "keypoints_hrnet_dark_coco_wholebody.pkl"),
            ("csl-daily-keypoints.pkl", "keypoints_hrnet_dark_coco_wholebody_iso.pkl"),
            ("gloss2ids.pkl", "csl_s2g_gloss2ids.pkl"),
        )
        for source_name, target_name in alias_pairs:
            source = source_dir / source_name
            if source.is_file():
                actions[str(data_dir / target_name)] = link_or_copy(source, data_dir / target_name)

    if two_stream_checkpoint:
        target = upstream_root / "Online" / "CTC_fusion" / "results" / "csl-daily_s2g" / "ckpts" / "best.ckpt"
        actions[str(target)] = link_or_copy(Path(two_stream_checkpoint), target)

    data_dir = upstream_root / "data" / CSL_DAILY
    expected_note = data_dir / "EXPECTED_FILES.md"
    if not expected_note.exists():
        expected_note.write_text(expected_data_markdown(data_dir), encoding="utf-8")
        actions[str(expected_note)] = "written"

    pretrained = upstream_root / "pretrained_models"
    if pose_asset_root:
        pose_root = Path(pose_asset_root).resolve()
        for name in (
            "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
            "faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth",
        ):
            source = pose_root / name
            if source.is_file():
                actions[str(pretrained / name)] = link_or_copy(source, pretrained / name)
    if pretrained_root:
        pretrained_root = Path(pretrained_root).resolve()
        k400 = pretrained_root / "s3ds_actioncls_ckpt"
        if k400.is_file():
            actions[str(pretrained / "s3ds_actioncls_ckpt")] = link_or_copy(k400, pretrained / "s3ds_actioncls_ckpt")
        mbart = pretrained_root / "mBart"
        if mbart.is_dir():
            # The official configs use `pretrained_models/mBart` for the
            # notebook-style pruning step, not as a ready Gloss2Text model.
            link = pretrained / "mBart"
            if not link.exists():
                try:
                    os.symlink(mbart, link, target_is_directory=True)
                    actions[str(link)] = "symlink"
                except OSError:
                    actions[str(link)] = "directory-present-elsewhere"

    present = {name: (data_dir / name).is_file() for name in CSL_DAILY_DATA_FILES}
    manifest = {
        "schema_version": 1,
        "upstream_root": str(upstream_root),
        "asset_root": str(asset_root),
        "links": actions,
        "data_dir": str(data_dir),
        "data_files": present,
        "missing_data_files": [name for name, exists in present.items() if not exists],
        "note": "Links only. Missing CSL-Daily data is reported, never synthesised or padded.",
    }
    if manifest_path is None:
        manifest_path = upstream_root / "SLRT_ASSET_LAYOUT.json"
    manifest_path = Path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
