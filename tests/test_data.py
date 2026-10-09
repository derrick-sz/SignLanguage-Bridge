from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from slbridge.data import (
    create_split, import_legacy, import_libras, load_manifest,
    save_manifest, validate_split, verify_manifest,
)


def make_legacy(root: Path, count: int = 6):
    root.mkdir()
    for label_index, label in enumerate(("A", "B")):
        for index in range(count):
            np.save(root / f"{label}_{index:03d}.npy", np.full((30, 63), label_index * 100 + index, dtype=np.float32))
    return import_legacy(root, "synthetic", "hand63_v1")


def membership(split):
    return {identifier: name for name in ("train", "validation", "test", "excluded")
            for identifier in getattr(split, name)}


def test_import_is_deterministic_atomic_and_relocatable(tmp_path):
    root = tmp_path / "source"
    first = make_legacy(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    assert first.sign_language is None
    assert all(not sample.groups and sample.sign_language is None for sample in first.samples)
    assert first == import_legacy(root, "synthetic", "hand63_v1")
    manifest_path = tmp_path / "manifest.json"
    save_manifest(first, manifest_path)
    saved_stat = manifest_path.stat().st_mtime_ns
    save_manifest(first, manifest_path)
    assert manifest_path.stat().st_mtime_ns == saved_stat
    assert load_manifest(manifest_path) == first
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    second = import_legacy(relocated, "synthetic", "hand63_v1")
    assert first.fingerprint == second.fingerprint
    assert [s.sample_id for s in first.samples] == [s.sample_id for s in second.samples]
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


def test_malformed_and_conflicting_samples_quarantined(tmp_path):
    root = tmp_path / "source"
    make_legacy(root)
    shutil.copyfile(root / "A_000.npy", root / "A_008.npy")
    shutil.copyfile(root / "A_000.npy", root / "B_008.npy")
    np.save(root / "A_009.npy", np.ones((30, 225), dtype=np.float32))
    (root / "B_009.npy").write_bytes(b"not a numpy file")
    manifest = import_legacy(root, "synthetic", "hand63_v1")
    conflicts = [sample for sample in manifest.samples if "duplicate_label_conflict" in sample.quality_issues]
    assert len(conflicts) == 3
    assert all(sample.quality_status == "quarantined" for sample in conflicts)
    malformed = [sample for sample in manifest.samples if sample.resource.endswith("009.npy")]
    assert len(malformed) == 2
    assert all(sample.quality_status == "quarantined" for sample in malformed)
    split = create_split(manifest)
    assert set(split.excluded) == {s.sample_id for s in manifest.samples if s.quality_status != "ready"}
    validate_split(manifest, split)


def test_duplicate_content_and_declared_groups_never_cross(tmp_path):
    root = tmp_path / "source"
    manifest = make_legacy(root, count=9)
    shutil.copyfile(root / "A_000.npy", root / "A_010.npy")
    manifest = import_legacy(root, "synthetic", "hand63_v1")
    records = []
    for sample in manifest.samples:
        repetition = int(Path(sample.resource).stem.rsplit("_", 1)[1])
        # Three sessions, with each session containing both labels.
        session = 0 if repetition == 10 else repetition // 3
        records.append(replace(sample, groups={"session_id": f"session:{session}"}))
    manifest = replace(manifest, samples=tuple(records))
    split = create_split(manifest, group_fields=("session_id",), seed=9)
    assert split == create_split(manifest, group_fields=("session_id",), seed=9)
    validate_split(manifest, split)
    positions = membership(split)
    for session in {s.groups["session_id"] for s in manifest.samples}:
        assert len({positions[s.sample_id] for s in manifest.samples if s.groups["session_id"] == session}) == 1
    for part in (split.train, split.validation, split.test):
        assert {s.label_id for s in manifest.samples if s.sample_id in part} == {"A", "B"}


def test_transitive_source_relationships_are_indivisible(tmp_path):
    manifest = make_legacy(tmp_path / "source", count=9)
    records = list(manifest.samples)
    a = [index for index, sample in enumerate(records) if sample.label_id == "A"]
    records[a[1]] = replace(records[a[1]], derived_from=records[a[0]].sample_id)
    records[a[2]] = replace(records[a[2]], source_id=records[a[1]].sample_id)
    manifest = replace(manifest, samples=tuple(records))
    split = create_split(manifest)
    positions = membership(split)
    assert len({positions[records[index].sample_id] for index in a[:3]}) == 1


def test_unknown_signers_and_impossible_constraints_raise(tmp_path):
    manifest = make_legacy(tmp_path / "source")
    with pytest.raises(ValueError, match="unknown"):
        create_split(manifest, group_fields=("signer_id",))
    all_one = replace(manifest, samples=tuple(replace(sample, groups={"session_id": "same"}) for sample in manifest.samples))
    with pytest.raises(ValueError, match="infeasible"):
        create_split(all_one, group_fields=("session_id",))
    for fractions in ((0.0, 0.2), (0.6, 0.5), (-0.1, 0.2), (float("nan"), 0.2)):
        with pytest.raises(ValueError, match="fractions"):
            create_split(manifest, validation_fraction=fractions[0], test_fraction=fractions[1])


def test_integrity_and_path_escape_refused(tmp_path):
    root = tmp_path / "source"
    manifest = make_legacy(root)
    verify_manifest(manifest)
    escaped = replace(manifest, samples=(replace(manifest.samples[0], resource="../outside.npy"),) + manifest.samples[1:])
    with pytest.raises(ValueError, match="resource|escape"):
        verify_manifest(escaped)
    (root / manifest.samples[0].resource).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        verify_manifest(manifest)


def test_public_manifest_save_cannot_overwrite_source(tmp_path):
    root = tmp_path / "source"
    manifest = make_legacy(root)
    original = root / manifest.samples[0].resource
    before = original.read_bytes()
    for output in (original, root / "manifest.json", tmp_path / "not-json.npy"):
        with pytest.raises(ValueError, match="read-only"):
            save_manifest(manifest, output)
    assert original.read_bytes() == before


def test_manual_split_cannot_hide_ready_labels_or_leak(tmp_path):
    manifest = make_legacy(tmp_path / "source")
    split = create_split(manifest)
    hidden = replace(split, train=split.train[1:], excluded=(split.train[0],))
    with pytest.raises(ValueError, match="Excluded"):
        validate_split(manifest, hidden)
    moved = replace(split, train=split.train + split.test[:1])
    with pytest.raises(ValueError, match="multiple"):
        validate_split(manifest, moved)


def test_libras_identity_local_groups_and_conflicts(tmp_path):
    root = tmp_path / "rgb"
    for item in (1, 2):
        for informant in (1, 2, 3):
            folder = root / f"{item}_Mesmo texto" / f"INFORMANTE {informant}"
            folder.mkdir(parents=True)
            (folder / "IMG_001.mp4").write_bytes(f"item:{item}:group:{informant}".encode())
    manifest = import_libras(root)
    assert len({s.sample_id for s in manifest.samples}) == 6
    assert {s.label_id for s in manifest.samples} == {"item:1", "item:2"}
    assert {s.display_text["pt-BR"] for s in manifest.samples} == {"Mesmo texto"}
    assert len({s.groups["local_informant"] for s in manifest.samples}) == 6
    assert all("signer_id" not in s.groups for s in manifest.samples)
    assert all("rgb_decode_not_verified" in s.quality_issues for s in manifest.samples)
    create_split(manifest, group_fields=("local_informant",))
    shutil.copyfile(root / "1_Mesmo texto/INFORMANTE 1/IMG_001.mp4",
                    root / "2_Mesmo texto/INFORMANTE 1/IMG_001.mp4")
    manifest = import_libras(root)
    assert len([s for s in manifest.samples if s.quality_status == "quarantined"]) == 2
    with pytest.raises(ValueError, match="infeasible"):
        create_split(manifest, group_fields=("local_informant",))
