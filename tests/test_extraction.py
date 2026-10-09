from dataclasses import replace
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from slbridge._io import read_json, sha256_file
from slbridge.data import create_split, import_libras, load_manifest, save_manifest, verify_manifest
from slbridge.extraction import extract_dataset
from slbridge.features import InputRejected, load_landmark_archive
from slbridge import media


def rgb_manifest(root: Path, count: int = 4):
    for item in (1, 2):
        for informant in range(1, count + 1):
            folder = root / f"{item}_Sinal {item}" / f"INFORMANTE {informant}"
            folder.mkdir(parents=True)
            (folder / "IMG.mp4").write_bytes(f"item:{item}:group:{informant}".encode())
    return import_libras(root)


@pytest.fixture
def fake_media(monkeypatch):
    state = {"calls": [], "opened": 0, "closed": 0, "failures": {}, "revision": "fake-v1"}

    def identity(config=None):
        return {"representation_id": "mediapipe225_rgb_v1", "version": state["revision"],
                "config": (config or media.ExtractionConfig()).to_dict()}

    class FakeExtractor:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            state["opened"] += 1
            return self

        def __exit__(self, *args):
            state["closed"] += 1

        def extract(self, path):
            state["calls"].append(Path(path))
            failure = state["failures"].get(Path(path))
            if failure is not None:
                raise failure
            count = self.config.frames
            values = np.zeros((count, 225), dtype=np.float32)
            values[:, 0::3] = int(sha256_file(path)[:2], 16) / 256
            values[:, 99:162] = -999
            times = np.linspace(0, 2, count, dtype=np.float64)
            indices = np.arange(count, dtype=np.int64) * 3
            return media.ExtractedClip(values, times, indices,
                {"anchor_frame_fraction": 1.0, "hand_frame_fraction": 1.0,
                 "source_frames": count * 3, "timestamps_known": True,
                 "timestamp_source": "frame_index_fps", "full_decode_verified": True}, identity(self.config))

    monkeypatch.setattr(media, "extractor_identity", identity)
    monkeypatch.setattr(media, "HolisticExtractor", FakeExtractor)
    return state


def test_extraction_archive_contract_resume_and_source_immutability(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    before = {sample.resource: (Path(manifest.root) / sample.resource).read_bytes() for sample in manifest.samples}
    progress = []
    output = tmp_path / "cache"
    derived = extract_dataset(manifest, output, progress=progress.append)
    assert derived.input_kind == "landmarks_npz"
    assert derived.representation_id == "mediapipe225_rgb_v1"
    assert (fake_media["opened"], fake_media["closed"]) == (1, 1)
    assert len(fake_media["calls"]) == len(manifest.samples)
    assert [event["completed"] for event in progress] == [0, 5, 8]
    assert progress[-1]["ready"] == 8
    source_by_id = {sample.sample_id: sample for sample in manifest.samples}
    for sample in derived.samples:
        source = source_by_id[sample.sample_id]
        assert sample.dataset_id == source.dataset_id
        assert sample.source_id == source.source_id
        assert sample.derived_from == source.sample_id
        assert sample.label_id == source.label_id
        assert sample.sign_language == source.sign_language
        assert sample.display_text == source.display_text
        assert sample.groups == {**source.groups, "raw_source_hash": source.content_hash}
        assert sample.quality_issues == ()
        values, timestamps, indices, metadata = load_landmark_archive(output / sample.resource)
        assert values.dtype == np.float32 and values.shape == (30, 225)
        assert timestamps.dtype == np.float64 and indices.dtype == np.int64
        assert metadata["source_hash"] == source.content_hash
        assert metadata["source_id"] == source.sample_id
    verify_manifest(derived)
    manifest_path = tmp_path / "derived.json"
    save_manifest(derived, manifest_path)
    assert load_manifest(manifest_path) == derived
    stats = {sample.resource: (output / sample.resource).stat().st_mtime_ns for sample in derived.samples}
    resumed = extract_dataset(manifest, output, progress=progress.append)
    assert resumed == derived
    assert len(fake_media["calls"]) == 8
    assert progress[-1]["cache_hits"] == 8
    assert stats == {sample.resource: (output / sample.resource).stat().st_mtime_ns for sample in resumed.samples}
    assert before == {sample.resource: (Path(manifest.root) / sample.resource).read_bytes() for sample in manifest.samples}


def test_config_and_extractor_identity_create_distinct_cache_versions(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    output = tmp_path / "cache"
    first = extract_dataset(manifest, output)
    second = extract_dataset(manifest, output, media.ExtractionConfig(frames=15))
    fake_media["revision"] = "fake-v2"
    third = extract_dataset(manifest, output, media.ExtractionConfig(frames=15))
    assert len({first.fingerprint, second.fingerprint, third.fingerprint}) == 3
    assert len(set(s.resource for m in (first, second, third) for s in m.samples)) == 24
    assert [s.sample_id for s in first.samples] == [s.sample_id for s in third.samples]
    assert load_landmark_archive(output / second.samples[0].resource)[0].shape == (15, 225)


def test_corrupt_cache_repaired_atomically_and_changed_source_refused(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    output = tmp_path / "cache"
    first = extract_dataset(manifest, output)
    (output / first.samples[0].resource).write_bytes(b"truncated npz")
    repaired = extract_dataset(manifest, output)
    verify_manifest(repaired)
    assert repaired == first
    assert len(fake_media["calls"]) == 9
    assert not list(output.rglob(".clip-*"))
    cache_before = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    (Path(manifest.root) / manifest.samples[0].resource).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        extract_dataset(manifest, output)
    assert cache_before == {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_invalid_compressed_payload_is_repaired(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    output = tmp_path / "cache"
    first = extract_dataset(manifest, output)
    path = output / first.samples[0].resource
    with zipfile.ZipFile(path) as archive:
        offset = archive.getinfo("values.npy").header_offset
    data = bytearray(path.read_bytes())
    filename_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
    compressed_start = offset + 30 + filename_length + extra_length
    data[compressed_start:compressed_start + 4] = b"\xff" * 4
    path.write_bytes(data)
    restored = extract_dataset(manifest, output)
    assert restored == first
    assert len(fake_media["calls"]) == 9
    verify_manifest(restored)


def test_video_failures_and_source_quarantine_are_explicit_and_retryable(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    sources = manifest.samples
    manifest = replace(manifest, samples=(replace(sources[0], quality_status="quarantined",
                        quality_issues=("duplicate_label_conflict",)),) + sources[1:])
    root = Path(manifest.root)
    fake_media["failures"][root / sources[1].resource] = media.VideoDecodeError("decoder failed")
    fake_media["failures"][root / sources[2].resource] = InputRejected("no_hand", {"hand_frame_fraction": 0.0})
    events = []
    derived = extract_dataset(manifest, tmp_path / "cache", progress=events.append)
    assert len(derived.samples) == 8
    assert len(fake_media["calls"]) == 7
    assert [s.quality_status for s in derived.samples[:3]] == ["quarantined"] * 3
    assert derived.samples[0].quality_issues == ("duplicate_label_conflict",)
    assert derived.samples[1].quality_issues == ("rgb_decode_failed:decoder failed",)
    assert derived.samples[2].quality_issues == ("no_hand",)
    verify_manifest(derived)
    assert events[-1]["rejected"] == 3 and events[-1]["errors"] == 1
    assert read_json(tmp_path / "cache" / derived.samples[2].resource)["quality"] == {"hand_frame_fraction": 0.0}
    fake_media["failures"].clear()
    resumed = extract_dataset(manifest, tmp_path / "cache")
    assert len(fake_media["calls"]) == 9
    assert [s.quality_status for s in resumed.samples[:3]] == ["quarantined", "ready", "ready"]


def test_program_errors_abort_without_becoming_video_failures(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    fake_media["failures"][Path(manifest.root) / manifest.samples[0].resource] = RuntimeError("tracking runtime broke")
    with pytest.raises(RuntimeError, match="tracking runtime"):
        extract_dataset(manifest, tmp_path / "cache")
    assert fake_media["closed"] == 1
    assert not list((tmp_path / "cache" / "failures").iterdir())
    assert not (tmp_path / "cache" / "summary.json").exists()


def test_interrupted_batch_keeps_completed_clip_and_resumes_remainder(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    root = Path(manifest.root)
    fake_media["failures"][root / manifest.samples[1].resource] = RuntimeError("interrupted")
    output = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        extract_dataset(manifest, output)
    completed = list((output / "clips").glob("*.npz"))
    assert len(completed) == 1
    before = completed[0].read_bytes(), completed[0].stat().st_mtime_ns
    fake_media["failures"].clear()
    events = []
    resumed = extract_dataset(manifest, output, progress=events.append)
    verify_manifest(resumed)
    assert events[-1]["cache_hits"] == 1
    assert len(fake_media["calls"]) == 9
    assert before == (completed[0].read_bytes(), completed[0].stat().st_mtime_ns)


def test_missing_dependencies_fail_before_any_cache_write(tmp_path, fake_media, monkeypatch):
    manifest = rgb_manifest(tmp_path / "rgb")

    def unavailable(config):
        raise ModuleNotFoundError("mediapipe is missing")

    monkeypatch.setattr(media, "extractor_identity", unavailable)
    with pytest.raises(ModuleNotFoundError, match="missing"):
        extract_dataset(manifest, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_cache_output_overlap_and_unowned_nonempty_directory_refused(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    for output in (Path(manifest.root), Path(manifest.root) / "features", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            extract_dataset(manifest, output)
    output = tmp_path / "other"
    output.mkdir()
    existing = output / "notes.txt"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="ownership"):
        extract_dataset(manifest, output)
    assert existing.read_text(encoding="utf-8") == "keep"
    assert list(output.iterdir()) == [existing]


def test_raw_byte_duplicates_keep_split_link_after_per_sample_archiving(tmp_path, fake_media):
    manifest = rgb_manifest(tmp_path / "rgb")
    first, second = manifest.samples[:2]
    (Path(manifest.root) / second.resource).write_bytes((Path(manifest.root) / first.resource).read_bytes())
    manifest = import_libras(manifest.root)
    derived = extract_dataset(manifest, tmp_path / "cache")
    assert len({s.resource for s in derived.samples}) == 8
    assert derived.samples[0].content_hash != derived.samples[1].content_hash
    assert derived.samples[0].groups["raw_source_hash"] == derived.samples[1].groups["raw_source_hash"]
    split = create_split(derived)
    locations = {s: name for name in ("train", "validation", "test", "excluded") for s in getattr(split, name)}
    assert locations[first.sample_id] == locations[second.sample_id]


def test_wrong_extractor_array_type_does_not_publish_invalid_archive(tmp_path, fake_media, monkeypatch):
    manifest = rgb_manifest(tmp_path / "rgb")
    original = media.HolisticExtractor.extract

    def invalid(self, path):
        clip = original(self, path)
        return replace(clip, timestamps=clip.timestamps.astype(np.float32))

    monkeypatch.setattr(media.HolisticExtractor, "extract", invalid)
    with pytest.raises(ValueError, match="float64"):
        extract_dataset(manifest, tmp_path / "cache")
    assert not list((tmp_path / "cache" / "clips").iterdir())
    assert not list((tmp_path / "cache" / "failures").iterdir())
