"""Read-only dataset indexing, integrity checks and leakage-safe splits.

RGB import validates references and hashes; decoding and feature extraction are
deliberately a separate, currently unavailable capability.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import random
import re
from typing import Iterable

from ._io import canonical_hash, read_json, resolve_resource, sha256_file, write_json_atomic
from .contracts import DatasetManifest, SampleRecord, SplitManifest
from .features import load_landmark_archive, load_sequence


_LEGACY_REPRESENTATIONS = {"hand63_v1", "pose_hands225_legacy"}
_LEGACY_NAME = re.compile(r"^(.+)_(\d+)$")
_LIBRAS_ITEM = re.compile(r"^(\d+)_(.+)$")
_LOCAL_INFORMANT = re.compile(r"^INFORMANTE\s*(\d+)$", re.IGNORECASE)


def _dataset_root(root: str | Path) -> Path:
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Dataset root is not a directory: {resolved}")
    return resolved


def _sample_id(dataset_id: str, resource: str) -> str:
    return f"{dataset_id}:{canonical_hash(resource)[:24]}"


def _files(root: Path, suffix: str) -> Iterable[tuple[Path, str]]:
    # Case-insensitive extensions and normalized, root-relative references work
    # on Windows and retain their identities if the dataset is relocated.
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file() and path.suffix.lower() == suffix:
            resource = path.relative_to(root).as_posix()
            yield resolve_resource(root, resource), resource


def _quarantine_conflicts(samples: Iterable[SampleRecord]) -> tuple[SampleRecord, ...]:
    records = tuple(samples)
    labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        labels[record.content_hash].add(record.label_id)
    conflicts = {digest for digest, values in labels.items() if len(values) > 1}
    return tuple(
        replace(
            record,
            quality_status="quarantined",
            quality_issues=tuple(sorted(set(record.quality_issues + ("duplicate_label_conflict",)))),
        )
        if record.content_hash in conflicts else record
        for record in records
    )


def import_legacy(
    root: str | Path,
    dataset_id: str,
    representation_id: str,
    sign_language: str | None = None,
) -> DatasetManifest:
    """Index legacy NPY files without modifying or silently reinterpreting them."""
    if representation_id not in _LEGACY_REPRESENTATIONS:
        raise ValueError(f"Unsupported legacy representation: {representation_id}")
    dataset_root = _dataset_root(root)
    samples: list[SampleRecord] = []
    for path, resource in _files(dataset_root, ".npy"):
        parsed = _LEGACY_NAME.fullmatch(path.stem)
        label_id = parsed.group(1) if parsed else "unknown"
        issues: list[str] = []
        if parsed is None:
            issues.append("invalid_legacy_filename")
        try:
            load_sequence(path, representation_id)
        except (ValueError, TypeError, OSError, EOFError) as error:
            issues.append("invalid_sequence:" + str(error).replace(str(path), resource))
        identifier = _sample_id(dataset_id, resource)
        samples.append(SampleRecord(
            sample_id=identifier,
            dataset_id=dataset_id,
            resource=resource,
            content_hash=sha256_file(path),
            label_id=label_id,
            sign_language=sign_language,
            input_kind="npy",
            representation_id=representation_id,
            source_id=identifier,
            quality_status="quarantined" if issues else "ready",
            quality_issues=tuple(issues),
        ))
    if not samples:
        raise ValueError(f"No NPY samples found in {dataset_root}")
    return DatasetManifest(
        dataset_id=dataset_id,
        root=str(dataset_root),
        input_kind="npy",
        representation_id=representation_id,
        sign_language=sign_language,
        samples=_quarantine_conflicts(samples),
    )


def import_libras(
    root: str | Path,
    dataset_id: str = "libras-eqt-uece-rgb",
) -> DatasetManifest:
    """Index Libras RGB assets; local informant numbers are not signer IDs."""
    dataset_root = _dataset_root(root)
    samples: list[SampleRecord] = []
    for path, resource in _files(dataset_root, ".mp4"):
        parts = Path(resource).parts
        item = _LIBRAS_ITEM.fullmatch(parts[0]) if len(parts) >= 3 else None
        informant = _LOCAL_INFORMANT.fullmatch(parts[1]) if len(parts) >= 3 else None
        issues: list[str] = []
        groups: dict[str, str] = {}
        display_text: dict[str, str] = {}
        label_id = "unknown"
        if item is None:
            issues.append("invalid_libras_item_directory")
        else:
            item_id = str(int(item.group(1)))
            label_id = f"item:{item_id}"
            display_text["pt-BR"] = item.group(2)
            if informant is not None:
                # Scope the group to its item. INFORMANTE 1 in two different
                # items does not establish that the same person signed both.
                groups["local_informant"] = f"{label_id}:informant:{int(informant.group(1))}"
        if informant is None:
            issues.append("missing_local_informant")
        if path.stat().st_size == 0:
            issues.append("empty_resource")
        identifier = _sample_id(dataset_id, resource)
        samples.append(SampleRecord(
            sample_id=identifier,
            dataset_id=dataset_id,
            resource=resource,
            content_hash=sha256_file(path),
            label_id=label_id,
            sign_language="libras",
            input_kind="rgb_video",
            representation_id="rgb_video_v1",
            source_id=identifier,
            groups=groups,
            quality_status="quarantined" if issues else "ready",
            quality_issues=tuple(issues + ["rgb_decode_not_verified"]),
            display_text=display_text,
        ))
    if not samples:
        raise ValueError(f"No MP4 samples found in {dataset_root}")
    return DatasetManifest(
        dataset_id=dataset_id,
        root=str(dataset_root),
        input_kind="rgb_video",
        representation_id="rgb_video_v1",
        sign_language="libras",
        samples=_quarantine_conflicts(samples),
    )


def _validate_structure(manifest: DatasetManifest) -> None:
    identifiers: set[str] = set()
    resources: set[str] = set()
    if not manifest.samples:
        raise ValueError("Manifest has no samples")
    for sample in manifest.samples:
        if sample.sample_id in identifiers:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        if sample.resource in resources:
            raise ValueError(f"Duplicate resource: {sample.resource}")
        identifiers.add(sample.sample_id)
        resources.add(sample.resource)
        if sample.dataset_id != manifest.dataset_id:
            raise ValueError(f"Dataset ID mismatch: {sample.sample_id}")
        if (sample.input_kind, sample.representation_id, sample.sign_language) != (
            manifest.input_kind, manifest.representation_id, manifest.sign_language
        ):
            raise ValueError(f"Sample contract differs from manifest: {sample.sample_id}")
        resolve_resource(manifest.root, sample.resource)
        if not re.fullmatch(r"[0-9a-f]{64}", sample.content_hash):
            raise ValueError(f"Invalid content hash: {sample.sample_id}")
        if not sample.label_id:
            raise ValueError(f"Missing label: {sample.sample_id}")
        if sample.quality_status not in {"ready", "quarantined"}:
            raise ValueError(f"Unknown quality status: {sample.sample_id}")
        if sample.quality_status == "quarantined" and not sample.quality_issues:
            raise ValueError(f"Quarantined sample has no reason: {sample.sample_id}")
    labels: dict[str, set[str]] = defaultdict(set)
    for sample in manifest.samples:
        labels[sample.content_hash].add(sample.label_id)
    for sample in manifest.samples:
        if len(labels[sample.content_hash]) > 1 and sample.quality_status != "quarantined":
            raise ValueError(f"Unquarantined duplicate label conflict: {sample.sample_id}")


def save_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    _validate_structure(manifest)
    path = Path(path)
    if path.suffix.lower() != ".json" or path.resolve().is_relative_to(Path(manifest.root).resolve()):
        raise ValueError("Manifest output must be a .json file outside the read-only dataset")
    write_json_atomic(path, manifest.to_dict())


def load_manifest(path: str | Path) -> DatasetManifest:
    manifest = DatasetManifest.from_dict(read_json(path))
    _validate_structure(manifest)
    return manifest


def verify_manifest(manifest: DatasetManifest) -> None:
    """Verify all indexed bytes, including quarantined assets, are unchanged."""
    _validate_structure(manifest)
    for sample in manifest.samples:
        path = resolve_resource(manifest.root, sample.resource)
        if not path.is_file():
            raise ValueError(f"Missing resource: {sample.resource}")
        if sha256_file(path) != sample.content_hash:
            raise ValueError(f"Content hash changed: {sample.resource}")
        if sample.quality_status == "ready" and manifest.input_kind == "npy":
            load_sequence(path, manifest.representation_id)
        elif sample.quality_status == "ready" and manifest.input_kind == "landmarks_npz":
            _, _, _, metadata = load_landmark_archive(path)
            if metadata["source_id"] != sample.derived_from or metadata["source_hash"] != sample.groups.get("raw_source_hash"):
                raise ValueError(f"Derived archive source provenance mismatch: {sample.sample_id}")


def _components(
    manifest: DatasetManifest,
    group_fields: tuple[str, ...],
) -> list[tuple[SampleRecord, ...]]:
    records = manifest.samples
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    seen: dict[tuple[str, str], int] = {}
    by_id = {record.sample_id: index for index, record in enumerate(records)}
    for index, record in enumerate(records):
        keys = [("content_hash", record.content_hash)]
        if record.groups.get("raw_source_hash"):
            # Archives for two raw copies include different sample IDs. Their
            # bytes differ, but original content must still never cross folds.
            keys.append(("raw_source_hash", record.groups["raw_source_hash"]))
        if record.source_id:
            keys.append(("source_id", record.source_id))
            if record.source_id in by_id:
                union(index, by_id[record.source_id])
        if record.derived_from:
            keys.append(("source_id", record.derived_from))
            if record.derived_from in by_id:
                union(index, by_id[record.derived_from])
        for field in group_fields:
            value = record.groups.get(field)
            if not value and record.quality_status == "ready":
                raise ValueError(f"Requested group {field!r} is unknown for {record.sample_id}")
            if value:
                keys.append(("group:" + field, value))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)
    return sorted(
        (tuple(sorted(values, key=lambda value: value.sample_id)) for values in grouped.values()),
        key=lambda values: values[0].sample_id,
    )


def _assign_components(
    components: list[tuple[SampleRecord, ...]],
    seed: int,
    fractions: tuple[float, float, float],
) -> list[int]:
    counts = [Counter(sample.label_id for sample in values) for values in components]
    total = sum(counts, Counter())
    label_components: dict[str, list[int]] = {
        label: [index for index, values in enumerate(counts) if label in values]
        for label in sorted(total)
    }
    for label, indices in label_components.items():
        if len(indices) < 3:
            raise ValueError(
                f"Split infeasible: label {label!r} has {len(indices)} independent components; "
                "train, validation and test each require at least one"
            )
    targets = [{label: total[label] * fraction for label in total} for fraction in fractions]
    total_size = sum(total.values())
    # A bounded deterministic heuristic handles real datasets without importing
    # an optimization package. It never relaxes grouping or class coverage.
    for attempt in range(128):
        rng = random.Random(seed + attempt * 1000003)
        assignment = [-1] * len(components)
        assigned = [Counter(), Counter(), Counter()]
        labels = sorted(total, key=lambda label: (len(label_components[label]), rng.random(), label))
        failed = False
        for label in labels:
            splits = list(range(3))
            rng.shuffle(splits)
            for split in splits:
                if assigned[split][label]:
                    continue
                candidates = [index for index in label_components[label] if assignment[index] < 0]
                if not candidates:
                    failed = True
                    break
                rng.shuffle(candidates)
                chosen = max(candidates, key=lambda index: (
                    sum(1.0 / len(label_components[value]) for value in counts[index] if not assigned[split][value]),
                    -sum(counts[index].values()),
                ))
                assignment[chosen] = split
                assigned[split].update(counts[chosen])
            if failed:
                break
        if failed:
            continue
        remaining = [index for index, split in enumerate(assignment) if split < 0]
        rng.shuffle(remaining)
        remaining.sort(key=lambda index: -sum(counts[index].values()))
        for index in remaining:
            def cost(split: int) -> float:
                change = 0.0
                for label, addition in counts[index].items():
                    target = targets[split][label]
                    before = assigned[split][label] - target
                    change += ((before + addition) ** 2 - before ** 2) / max(target, 1.0)
                current_size = sum(assigned[split].values())
                addition = sum(counts[index].values())
                target_size = total_size * fractions[split]
                before = current_size - target_size
                return change + 0.25 * ((before + addition) ** 2 - before ** 2) / max(target_size, 1.0)
            split = min(range(3), key=lambda value: (cost(value), value))
            assignment[index] = split
            assigned[split].update(counts[index])
        if all(set(values) == set(total) for values in assigned):
            return assignment
    raise ValueError(
        "Could not construct a split with all labels in all three sets under the declared constraints "
        "after 128 deterministic attempts; inspect connected components or supply more independent samples"
    )


def create_split(
    manifest: DatasetManifest,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    group_fields: tuple[str, ...] = (),
) -> SplitManifest:
    _validate_structure(manifest)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in (validation_fraction, test_fraction)) or not (
        0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1
    ):
        raise ValueError("All three split fractions must be positive and sum to one")
    if isinstance(group_fields, str):
        raise ValueError("group_fields must be a sequence of group names, not one string")
    group_fields = tuple(group_fields)
    if any(not isinstance(field, str) or not field for field in group_fields) or len(set(group_fields)) != len(group_fields):
        raise ValueError("group_fields must contain unique nonempty names")
    components = _components(manifest, group_fields)
    eligible = [values for values in components if all(record.quality_status == "ready" for record in values)]
    excluded = sorted(
        record.sample_id for values in components
        if any(member.quality_status != "ready" for member in values) for record in values
    )
    if not eligible:
        raise ValueError("No eligible samples remain after quarantine and source grouping")
    assignment = _assign_components(
        eligible, seed, (1.0 - validation_fraction - test_fraction, validation_fraction, test_fraction)
    )
    sets = [[], [], []]
    for values, split in zip(eligible, assignment):
        sets[split].extend(record.sample_id for record in values)
    if group_fields:
        protocol = "grouped:" + ",".join(group_fields)
    elif manifest.input_kind == "npy":
        protocol = "legacy_sample_level_demo_signer_unknown"
    else:
        protocol = "sample_level_signer_unknown"
    split = SplitManifest(
        dataset_hash=manifest.fingerprint,
        seed=seed,
        protocol=protocol,
        group_fields=group_fields,
        train=tuple(sorted(sets[0])),
        validation=tuple(sorted(sets[1])),
        test=tuple(sorted(sets[2])),
        excluded=tuple(excluded),
    )
    validate_split(manifest, split)
    return split


def validate_split(manifest: DatasetManifest, split: SplitManifest) -> None:
    _validate_structure(manifest)
    if split.dataset_hash != manifest.fingerprint:
        raise ValueError("Split refers to a different dataset manifest")
    named_sets = (split.train, split.validation, split.test, split.excluded)
    locations: dict[str, int] = {}
    by_id = {record.sample_id: record for record in manifest.samples}
    for position, values in enumerate(named_sets):
        for identifier in values:
            if identifier not in by_id:
                raise ValueError(f"Unknown split sample: {identifier}")
            if identifier in locations:
                raise ValueError(f"Sample appears multiple times in split: {identifier}")
            locations[identifier] = position
            if position != 3 and by_id[identifier].quality_status != "ready":
                raise ValueError(f"Non-ready sample is not excluded: {identifier}")
    if set(locations) != set(by_id):
        raise ValueError("Split does not account for every manifest sample")
    components = _components(manifest, tuple(split.group_fields))
    expected_excluded = {
        record.sample_id for component in components
        if any(member.quality_status != "ready" for member in component) for record in component
    }
    if set(split.excluded) != expected_excluded:
        raise ValueError("Excluded samples must be exactly the quarantined source components")
    for component in components:
        if len({locations[record.sample_id] for record in component}) > 1:
            raise ValueError("Related source, duplicate content or declared group crosses split sets")
    eligible_labels = {by_id[identifier].label_id for identifier, location in locations.items() if location != 3}
    if not eligible_labels:
        raise ValueError("Split has no eligible labels")
    for name, values in zip(("train", "validation", "test"), named_sets[:3]):
        labels = {by_id[identifier].label_id for identifier in values}
        if labels != eligible_labels:
            raise ValueError(f"{name} lacks required labels: {sorted(eligible_labels - labels)}")


def select_labels(manifest: DatasetManifest, label_ids: tuple[str, ...]) -> DatasetManifest:
    """Create an explicit source subset without altering sample identities."""
    _validate_structure(manifest)
    if not label_ids or len(set(label_ids)) != len(label_ids):
        raise ValueError("Select a nonempty set of unique label IDs")
    known = {sample.label_id for sample in manifest.samples}
    missing = set(label_ids) - known
    if missing:
        raise ValueError(f"Unknown dataset labels: {sorted(missing)}")
    return replace(manifest, samples=tuple(sample for sample in manifest.samples if sample.label_id in label_ids))


def project_split(source: DatasetManifest, split: SplitManifest, derived: DatasetManifest) -> SplitManifest:
    """Keep frozen source-fold membership; exclude failed related components."""
    validate_split(source, split)
    source_records = {record.sample_id: record for record in source.samples}
    if set(source_records) != {record.sample_id for record in derived.samples}:
        raise ValueError("Derived dataset must account for every frozen source sample")
    for record in derived.samples:
        original = source_records[record.sample_id]
        if (record.dataset_id, record.label_id, record.sign_language, record.derived_from,
                record.groups.get("raw_source_hash")) != (
                original.dataset_id, original.label_id, original.sign_language, original.sample_id, original.content_hash):
            raise ValueError("Derived record changes frozen source label/identity")
        if any(record.groups.get(key) != value for key, value in original.groups.items()):
            raise ValueError("Derived record changes source grouping")
    excluded = {record.sample_id for component in _components(derived, split.group_fields)
                if any(member.quality_status != "ready" for member in component) for record in component}
    result = SplitManifest(dataset_hash=derived.fingerprint, seed=split.seed,
                           protocol="derived_from_frozen:" + split.protocol, group_fields=split.group_fields,
                           train=tuple(identifier for identifier in split.train if identifier not in excluded),
                           validation=tuple(identifier for identifier in split.validation if identifier not in excluded),
                           test=tuple(identifier for identifier in split.test if identifier not in excluded),
                           excluded=tuple(sorted(excluded)))
    validate_split(derived, result)
    return result
