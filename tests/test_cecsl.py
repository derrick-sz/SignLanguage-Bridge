"""CE-CSL class identity, official folds and exact TFNet RGB evaluation input."""

import ast
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from slbridge._io import read_json
from slbridge.cecsl import (RGBInputSpec, build_cecsl_vocabulary, freeze_cecsl_dataset,
                           normalize_gloss, padded_rgb_length, prepare_rgb_clip,
                           prepare_rgb_file, preprocessing_manifest)
from slbridge.sources import CapturedClip, FramePacket


@pytest.mark.parametrize("raw,expected", [
    ("退1", "退"), ("词12", "词1"), ("001", "1"), ("2年1", "2年1"),
    ("词(注)1", "词"), ("词{注}[注]（注）2", "词"), (",", "，"),
    ("?词", "？词"), ("。", "。"), (" 词 ", " 词 "),
    ("词(外[内]尾)2", "词尾)"),  # upstream flag is deliberately non-nested
])
def test_exact_annotation_rules(raw, expected):
    assert normalize_gloss(raw) == expected


@pytest.mark.parametrize("raw", ["", "(仅注)", "²", None])
def test_invalid_normalization_is_explicit(raw):
    with pytest.raises(ValueError):
        normalize_gloss(raw)


def _dataset(root):
    rows = {"train": [["train-1", "A", "甲。", "甲1/。", ""]],
            "dev": [["dev-1", "A", "乙？", "乙(注)/?", ""]],
            "test": [["test-1", "B", "甲，", "甲/,", ""]]}
    for split, records in rows.items():
        labels = root / "label"
        labels.mkdir(parents=True, exist_ok=True)
        with (labels / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Number", "Translator", "Chinese Sentences", "Gloss", "Note"])
            writer.writerows(records)
        for identifier, signer, *_ in records:
            video = root / "video" / split / signer / f"{identifier}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"index fixture, not decodable video")
    return root


def test_freeze_exact_order_official_folds_and_signer_overlap(tmp_path):
    root = _dataset(tmp_path / "dataset")
    vocab = build_cecsl_vocabulary(root)
    assert vocab["tokens"] == [" "] + sorted({"甲", "乙", "。", "，", "？"})
    assert vocab["blank_id"] == 0
    out = tmp_path / "frozen"
    summary = freeze_cecsl_dataset(root, out, expected_vocab_size=6)
    assert summary["rows"] == {"train": 1, "dev": 1, "test": 1}
    assert summary["signer_overlap"] == {"train_dev": ["A"], "train_test": [], "dev_test": []}
    splits = read_json(out / "official_splits.json")
    assert splits["train"] == ["train-1"] and splits["dev"] == ["dev-1"] and splits["test"] == ["test-1"]
    index = read_json(out / "dataset_index.json")
    assert index["provenance"]["classification_label_coverage_required"] is False
    assert index["provenance"]["signer_disjoint"] is False
    assert [record["sample_id"] for record in index["samples"]] == ["train-1", "dev-1", "test-1"]
    assert all(record["video_exists"] for record in index["samples"])
    assert freeze_cecsl_dataset(root, out, expected_vocab_size=6) == summary
    assert preprocessing_manifest()["training_augmentation_applied"] is False


def test_freeze_refuses_missing_media_and_mismatched_existing_artifact(tmp_path):
    root = _dataset(tmp_path / "dataset")
    (root / "video/dev/A/dev-1.mp4").unlink()
    with pytest.raises(ValueError, match="Missing 1"):
        freeze_cecsl_dataset(root, tmp_path / "missing", expected_vocab_size=6)
    out = tmp_path / "frozen"
    summary = freeze_cecsl_dataset(root, out, expected_vocab_size=6, require_all_videos=False)
    assert len(summary["videos_missing"]) == 1
    path = out / "vocabulary.json"
    incompatible = json.loads(path.read_text(encoding="utf-8"))
    incompatible["tokens"][1] = "错误"
    path.write_text(json.dumps(incompatible), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        freeze_cecsl_dataset(root, out, expected_vocab_size=6, require_all_videos=False)
    assert path.read_bytes() == original


def _clip(count, interval=0.04):
    # Deliberately asymmetric colours and non-square input expose RGB mistakes
    # and JPEG-before/after-resize differences.
    rng = np.random.default_rng(41)
    return CapturedClip(tuple(FramePacket(rng.integers(0, 256, (67, 83, 3), dtype=np.uint8),
                                         index * interval, index, "synthetic")
                              for index in range(count)), "synthetic")


def _upstream_frame_reference(image, path):
    cv2 = pytest.importorskip("cv2")
    # Converter receives RGB from imageio, swaps to BGR, resizes, writes JPEG;
    # MyDataset then reads BGR, converts RGB, resizes, and calls CenterCrop.
    converter_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    converter_bgr = cv2.cvtColor(converter_rgb, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(converter_bgr, (256, 256))
    cv2.imencode(".jpg", resized)[1].tofile(path)
    loader = cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), (256, 256))
    return loader[16:240, 16:240].transpose(2, 0, 1).astype(np.float32) / np.float32(127.5) - np.float32(1)


@pytest.mark.parametrize("count", [4, 5, 6, 7])
def test_rgb_pixel_reference_and_modulo_four_padding(tmp_path, count):
    pytest.importorskip("cv2")
    clip = _clip(count)
    prepared = prepare_rgb_clip(clip)
    assert prepared.values.shape == (1, padded_rgb_length(count), 3, 224, 224)
    assert prepared.values.dtype == np.float32 and prepared.values.flags.c_contiguous
    assert not prepared.values.flags.writeable
    assert prepared.source_frames == count and prepared.source_ref == "synthetic"
    assert prepared.time_range == (0.0, (count - 1) * 0.04)
    for index, packet in enumerate(clip.frames):
        expected = _upstream_frame_reference(packet.image, tmp_path / f"{index:05d}.jpg")
        np.testing.assert_array_equal(prepared.values[0, index + 6], expected)
    np.testing.assert_array_equal(prepared.values[0, :6], np.repeat(prepared.values[0, 6:7], 6, axis=0))
    right = 6 + (-count) % 4
    np.testing.assert_array_equal(prepared.values[0, -right:], np.repeat(prepared.values[0, 6 + count - 1:6 + count], right, axis=0))
    assert prepared.quality["right_padding"] == right and not prepared.quality["truncated"]


def test_captured_limits_reject_whole_clip_before_preprocessing(monkeypatch):
    import slbridge.cecsl as cecsl

    def should_not_run(*args):
        raise AssertionError("limit must reject before processing any frame")

    monkeypatch.setattr(cecsl, "_frame_values", should_not_run)
    with pytest.raises(ValueError, match="frame limit"):
        prepare_rgb_clip(_clip(7), RGBInputSpec(max_frames=6))
    with pytest.raises(ValueError, match="duration limit"):
        prepare_rgb_clip(_clip(2, 16))
    with pytest.raises(ValueError, match="resource byte"):
        prepare_rgb_clip(_clip(2), RGBInputSpec(max_resource_bytes=1))


def test_spec_semantics_locked_and_trusted_limits_bounded():
    assert RGBInputSpec.from_dict(RGBInputSpec().to_dict()) == RGBInputSpec()
    assert replace(RGBInputSpec(), max_frames=1200, max_seconds=60).max_frames == 1200
    for override in ({"jpeg_quality": 90}, {"crop_size": 200}, {"max_frames": 10001},
                     {"max_seconds": 301}, {"max_resource_bytes": 2**31 + 1},
                     {"max_frames": True}, {"unknown": "bad"}):
        with pytest.raises(ValueError):
            RGBInputSpec.from_dict({**RGBInputSpec().to_dict(), **override})


def test_video_full_decode_and_file_clip_pixel_parity(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (84, 68))
    if not writer.isOpened():
        pytest.skip("Local OpenCV build lacks MJPG writer")
    rng = np.random.default_rng(5)
    for _ in range(7):
        writer.write(rng.integers(0, 256, (68, 84, 3), dtype=np.uint8))
    writer.release()
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(FramePacket(image, len(frames) / 25, len(frames), "file-reference"))
    capture.release()
    actual = prepare_rgb_file(path)
    expected = prepare_rgb_clip(CapturedClip(tuple(frames), "file-reference"))
    np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.source_frames == 7 and actual.quality["full_decode_verified"]
    with pytest.raises(ValueError, match="frame limit"):
        prepare_rgb_file(path, RGBInputSpec(max_frames=6))
    with pytest.raises(ValueError, match="resource byte"):
        prepare_rgb_file(path, RGBInputSpec(max_resource_bytes=1))


def test_implementation_matches_pinned_upstream_normalizer_when_available():
    source = Path("artifacts/cecsl-tfnet-audit-20261009/TFNet/DataProcessMoudle.py")
    if not source.is_file():
        pytest.skip("Pinned upstream audit source is local evidence")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "PreWords")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    raw = []
    for split in ("train", "dev", "test"):
        with (source.parent / "data/CE-CSL" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            raw.extend(token for row in reader for token in row[3].split("/"))
    expected = namespace["PreWords"](raw.copy())
    assert [normalize_gloss(token) for token in raw] == expected
    tokens = [" "] + sorted(set(expected))
    assert len(tokens) == 3516 and tokens[77] == "。" and tokens[3514:] == ["，", "？"]
