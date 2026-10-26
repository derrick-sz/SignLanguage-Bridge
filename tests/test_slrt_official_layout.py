import json
import os

import pytest

from slbridge.slrt.official_layout import (
    CSL_DAILY_DATA_FILES,
    build_layout,
    link_or_copy,
    sha256,
)


def _write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fixture(tmp_path):
    upstream = tmp_path / "upstream"
    (upstream / "Online" / "CSLR").mkdir(parents=True)
    asset = tmp_path / "assets"
    ckpt = _write(asset / "ckpts" / "cslr_best.ckpt", b"checkpoint")
    vocab = _write(asset / "meta" / "csl_iso_with_blank.vocab", b"vocab")
    pose = tmp_path / "pose"
    hrnet = _write(pose / "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth", b"hrnet")
    detector = _write(pose / "faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth", b"detector")
    pretrained = tmp_path / "pretrained"
    k400 = _write(pretrained / "s3ds_actioncls_ckpt", b"k400")
    return upstream, asset, pose, pretrained, ckpt, vocab, hrnet, detector, k400


def test_build_layout_links_core_assets_and_reports_data(tmp_path):
    upstream, asset, pose, pretrained, ckpt, vocab, hrnet, detector, k400 = _fixture(tmp_path)
    manifest_path = tmp_path / "layout.json"
    manifest = build_layout(
        upstream,
        asset,
        pose_asset_root=pose,
        pretrained_root=pretrained,
        manifest_path=manifest_path,
    )
    target_ckpt = upstream / "Online/CSLR/results/csl-daily_ISLR/ckpts/best.ckpt"
    assert target_ckpt.is_file()
    assert target_ckpt.stat().st_size == ckpt.stat().st_size
    if os.name == "nt":
        assert target_ckpt.stat().st_ino == ckpt.stat().st_ino
    assert (upstream / "data/csl-daily/csl_iso_with_blank.vocab").is_file()
    assert (upstream / "pretrained_models/s3ds_actioncls_ckpt").is_file()
    assert (upstream / "pretrained_models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth").is_file()
    assert (upstream / "pretrained_models/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth").is_file()
    assert (upstream / "data/csl-daily/EXPECTED_FILES.md").is_file()
    assert set(manifest["missing_data_files"]) == set(CSL_DAILY_DATA_FILES) - {"csl_iso_with_blank.vocab"}
    assert manifest["data_files"]["csl_iso_with_blank.vocab"] is True
    assert manifest["data_files"]["csl.test"] is False
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_repeated_layout_is_idempotent(tmp_path):
    upstream, asset, pose, pretrained, ckpt, vocab, hrnet, detector, k400 = _fixture(tmp_path)
    build_layout(upstream, asset, pose_asset_root=pose, pretrained_root=pretrained, manifest_path=tmp_path / "a.json")
    manifest = build_layout(upstream, asset, pose_asset_root=pose, pretrained_root=pretrained, manifest_path=tmp_path / "b.json")
    target = upstream / "Online/CSLR/results/csl-daily_ISLR/ckpts/best.ckpt"
    assert sha256(target) == sha256(ckpt)
    assert manifest["links"][str(target)] in {"hardlink", "existing"}


def test_link_or_copy_refuses_different_existing_target(tmp_path):
    source = _write(tmp_path / "source", b"one")
    target = _write(tmp_path / "target", b"two")
    with pytest.raises(ValueError):
        link_or_copy(source, target)


def test_link_or_copy_accepts_same_content(tmp_path):
    source = _write(tmp_path / "source", b"same")
    target = _write(tmp_path / "target", b"same")
    assert link_or_copy(source, target) == "existing"
