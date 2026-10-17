"""Read-only bounded audit of updated Libras annotations and frame directories.

Object NPY files are read only with an explicit NumPy reconstruction allowlist.
No dataset content is written, and no archive is extracted.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import statistics
import sys
import unicodedata
import zipfile

import numpy as np


class AnnotationUnpickler(pickle.Unpickler):
    def __init__(self, file):
        super().__init__(file)
        self.globals_seen = []

    def find_class(self, module, name):
        self.globals_seen.append([module, name])
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            return getattr(np.core.multiarray, name)
        raise pickle.UnpicklingError(f"Denied global: {module}.{name}")


def load_annotation(path: Path):
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Annotation file exceeds 10 MiB safety bound")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        header_end = stream.tell()
        if not dtype.hasobject:
            raise ValueError("Expected object NPY annotation, numeric format found")
        if shape != ():
            raise ValueError(f"Expected scalar object array, found {shape}")
        unpickler = AnnotationUnpickler(stream)
        value = unpickler.load()
        if stream.read(1):
            raise ValueError("Trailing bytes after annotation pickle")
    if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != np.dtype("O"):
        raise ValueError("Unexpected reconstructed annotation root")
    value = value.item()
    if not isinstance(value, dict):
        raise ValueError("Annotation payload must be a dictionary")
    return value, {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "npy_version": version, "shape": shape, "fortran_order": fortran,
        "dtype": str(dtype), "header_bytes": header_end,
        "allowed_globals_observed": unpickler.globals_seen,
    }


def scalar_summary(values):
    if not values:
        return None
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "mean": statistics.mean(values), "max": max(values)}


def normalized_frame_id(value):
    return value.replace("\r\n", "_").replace("\r", "_").replace("\n", "_")


def normalized_label(value):
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w\s]|_", " ", value).split())


@functools.lru_cache(maxsize=64)
def ordered_frame_hash(folder):
    digest = hashlib.sha256()
    files = sorted(path for path in folder.iterdir() if path.is_file())
    byte_count = 0
    for path in files:
        content = path.read_bytes()
        byte_count += len(content)
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return {"sha256": digest.hexdigest(), "files": len(files), "content_bytes": byte_count, "definition": "SHA256 over sorted filename-length/filename/content-length/SHA256(content) records"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/Zenodo/Libras-EQT-UECE/Libras-EQT-UECE")
    parser.add_argument("--output", default="artifacts/libras-updated-data-audit-20261010/annotation-audit.json")
    parser.add_argument("--raw-root", default="Libras-EQT-UECE (RGB)")
    args = parser.parse_args()
    root = Path(args.root)
    frame_root = root / "features/fullFrame-256x256px"
    report = {
        "dataset_root": str(root),
        "loading_policy": "NPY header checked first; <=10 MiB each; scalar object arrays only; restricted unpickler allowing numpy.ndarray, numpy.dtype, numpy(_core|core).multiarray._reconstruct/scalar; all other globals denied",
        "scan_policy": "All annotation samples and frame directory entry counts; at most three images decoded per split; all PNG contents hashed only for cross-split same-ID overlaps; no videos opened; archives central directory only",
        "splits": {}, "cross_split": {}, "archives": {},
    }
    keys_by_split, signers_by_split, labels_by_split = {}, {}, {}
    sequence_signatures = {}
    annotation_payloads = {}
    raw_root = Path(args.raw_root).resolve()
    raw_index = collections.defaultdict(list)
    raw_class_files = collections.Counter()
    raw_directory_names = sorted(item.name for item in os.scandir(raw_root) if item.is_dir())
    for video in raw_root.rglob("*.mp4"):
        key = (video.parent.name.strip(), video.stem)
        raw_index[key].append(video)
        raw_class_files[video.parent.parent.name] += 1
    report["raw_rgb_inventory"] = {
        "root": str(raw_root), "all_class_directory_names": raw_directory_names,
        "active_class_directory_count": len(raw_class_files), "active_class_video_counts": dict(sorted(raw_class_files.items())),
        "video_files": sum(raw_class_files.values()),
        "signer_stem_ambiguous_keys": [{"key": key, "paths": [str(path) for path in values]} for key, values in raw_index.items() if len(values) != 1],
    }
    class_labels = collections.defaultdict(set)
    ambiguous_class_labels = collections.defaultdict(set)
    for split in ("train", "dev", "test"):
        payload, file_info = load_annotation(root / "annotations" / f"{split}.npy")
        metadata = {key: value for key, value in payload.items() if not isinstance(value, dict)}
        samples = {key: value for key, value in payload.items() if isinstance(value, dict)}
        annotation_payloads[split] = samples
        field_counts = collections.Counter()
        signer_counts = collections.Counter()
        label_values = collections.defaultdict(collections.Counter)
        frame_counts, annotation_lengths = [], []
        frame_extension_counts = collections.Counter()
        non_contiguous_frame_names = []
        missing, mismatched, empties = [], [], []
        literal_missing, path_resolutions = [], []
        source_unmatched, source_ambiguous = [], []
        source_class_counts = collections.Counter()
        source_class_label_resolutions = []
        examples, image_checks = [], []
        split_dir_names = {item.name for item in os.scandir(frame_root / split) if item.is_dir()}
        signatures = {}
        for index, (sample_id, sample) in enumerate(samples.items()):
            field_counts.update(sample.keys())
            signer_counts[str(sample.get("signer"))] += 1
            if len(examples) < 4:
                examples.append({"sample_id": sample_id, "annotation": sample})
            for field in ("label_gloss", "label_char", "label_text", "text", "gloss"):
                if field in sample:
                    label_values[field][json.dumps(sample[field], ensure_ascii=False, sort_keys=True)] += 1
            for item in sample.get("original_info", []):
                if isinstance(item, dict):
                    for field in ("label_gloss", "label_char", "label_text", "text", "gloss"):
                        if field in item:
                            label_values[f"original_info.{field}"][json.dumps(item[field], ensure_ascii=False, sort_keys=True)] += 1
            folder = frame_root / split / sample_id
            if not folder.exists():
                literal_missing.append(sample_id)
                normalized_id = normalized_frame_id(sample_id)
                candidates = [name for name in split_dir_names if name == normalized_id]
                path_resolutions.append({"annotation_sample_id": sample_id, "normalized_frame_id": normalized_id, "candidate_count": len(candidates), "candidates": candidates})
                if len(candidates) == 1:
                    folder = frame_root / split / candidates[0]
            stem_match = re.search(r"_(IMG_\d+)$", sample_id)
            source_candidates = raw_index.get((sample.get("signer", "").strip(), stem_match.group(1)), []) if stem_match else []
            if len(source_candidates) > 1:
                labels = ["".join(item.get("label_char", [])) for item in sample.get("original_info", [])]
                class_matched = [path for path in source_candidates if normalized_label(path.parent.parent.name.partition("_")[2]) in {normalized_label(label) for label in labels}]
                if len(class_matched) == 1:
                    source_class_label_resolutions.append({"sample_id": sample_id, "annotation_text": labels, "candidate_paths": [str(path) for path in source_candidates], "resolved_path": str(class_matched[0])})
                    source_candidates = class_matched
            if len(source_candidates) == 1:
                class_name = source_candidates[0].parent.parent.name
                source_class_counts[class_name] += 1
                for entry in sample.get("original_info", []):
                    class_labels[class_name].add(json.dumps(entry.get("label_char"), ensure_ascii=False))
            elif not source_candidates:
                source_unmatched.append(sample_id)
            else:
                source_ambiguous.append({"sample_id": sample_id, "candidates": [str(path) for path in source_candidates]})
                for path in source_candidates:
                    for entry in sample.get("original_info", []):
                        ambiguous_class_labels[path.parent.parent.name].add(json.dumps(entry.get("label_char"), ensure_ascii=False))
            if not folder.exists():
                missing.append(sample_id)
                continue
            image_entries = sorted((item for item in os.scandir(folder) if item.is_file()), key=lambda item: item.name)
            frame_counts.append(len(image_entries))
            frame_extension_counts.update(Path(item.name).suffix.lower() for item in image_entries)
            expected_names = [f"{index:04d}.png" for index in range(1, len(image_entries) + 1)]
            if [item.name for item in image_entries] != expected_names:
                non_contiguous_frame_names.append(sample_id)
            signatures[sample_id] = {"names": [item.name for item in image_entries], "sizes": [item.stat().st_size for item in image_entries]}
            for field in ("num_frames", "frame_count", "length", "seq_len"):
                if field in sample:
                    annotation_lengths.append(sample[field])
                    if sample[field] != len(image_entries):
                        mismatched.append({"sample_id": sample_id, "field": field, "annotation": sample[field], "files": len(image_entries)})
            if not image_entries:
                empties.append(sample_id)
            if len(image_checks) < 3 and image_entries:
                import cv2
                path = Path(image_entries[0].path)
                pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                image_checks.append({"path": str(path), "decoded_shape": list(pixels.shape) if pixels is not None else None, "frame_names_first_last": [image_entries[0].name, image_entries[-1].name]})
            if (index + 1) % 500 == 0:
                print(f"{split}: scanned {index + 1}/{len(samples)} sample folders", flush=True)
        sample_keys = set(samples)
        resolved_keys = {normalized_frame_id(value) for value in sample_keys}
        normalized_collisions = collections.defaultdict(list)
        for sample_id in sample_keys:
            normalized_collisions[normalized_frame_id(sample_id)].append(sample_id)
        keys_by_split[split] = sample_keys
        signers_by_split[split] = set(signer_counts)
        labels_by_split[split] = {field: set(values) for field, values in label_values.items()}
        sequence_signatures[split] = signatures
        report["splits"][split] = {
            "annotation_file": file_info, "metadata": metadata,
            "sample_count": len(samples), "top_level_field_counts": dict(field_counts),
            "signer_counts": dict(sorted(signer_counts.items())),
            "label_field_counts": {field: {"distinct": len(values), "most_common": values.most_common(20), "all_values": dict(sorted(values.items()))} for field, values in label_values.items()},
            "frame_folder_count": len(split_dir_names),
            "annotation_missing_frame_folders": missing,
            "literal_annotation_missing_frame_folders": literal_missing,
            "newline_path_resolutions": path_resolutions,
            "normalization_collisions": {key: values for key, values in normalized_collisions.items() if len(values) > 1},
            "frame_folders_without_annotation": sorted(split_dir_names - resolved_keys),
            "empty_frame_folders": empties,
            "file_count_per_sample": scalar_summary(frame_counts),
            "total_frame_files": sum(frame_counts),
            "frame_extension_counts": dict(frame_extension_counts),
            "samples_without_exact_0001_to_N_png_names": non_contiguous_frame_names,
            "annotation_length_values": scalar_summary(annotation_lengths),
            "frame_count_mismatches": mismatched,
            "examples": examples, "sample_image_checks": image_checks,
            "source_rgb_mapping": {
                "policy": "Unique signer.strip()+IMG stem; ambiguous keys disambiguated only when normalized source class phrase uniquely matches joined label_char (NFKD, casefold, combining marks removed, punctuation/underscores treated as spaces); source videos never decoded",
                "matched_samples": sum(source_class_counts.values()), "unmatched_sample_ids": source_unmatched,
                "ambiguous_samples": source_ambiguous,
                "class_label_disambiguations": source_class_label_resolutions,
                "source_class_counts": dict(sorted(source_class_counts.items())),
                "source_classes_missing_from_split": sorted(set(raw_class_files) - set(source_class_counts)),
            },
        }
    canonical_groups = collections.defaultdict(list)
    all_label_sequences = set()
    for split, samples in annotation_payloads.items():
        for sample_id, sample in samples.items():
            stem = re.search(r"_(IMG_\d+)$", sample_id).group(1)
            labels = [item.get("label_char", []) for item in sample["original_info"]]
            signature = (sample["signer"].strip(), stem, json.dumps(labels, ensure_ascii=False))
            canonical_groups[signature].append((split, sample_id))
            all_label_sequences.update(json.dumps(label, ensure_ascii=False) for label in labels)
    report["label_char_semantics"] = {
        "distinct_sequences": len(all_label_sequences),
        "joined_labels_with_one_whitespace_word": sum(len("".join(json.loads(label)).split()) == 1 for label in all_label_sequences),
        "joined_labels_with_multiple_whitespace_words": sum(len("".join(json.loads(label)).split()) > 1 for label in all_label_sequences),
        "element_types": "Lists of Portuguese word/punctuation-bearing strings interspersed with literal spaces; neither demonstrated gloss nor individual Unicode characters",
    }
    report["canonical_label_signer_stem_duplicates"] = [{
        "canonical_key": signature, "samples": [{"split": split, "sample_id": sample_id, "frames": ordered_frame_hash(frame_root / split / normalized_frame_id(sample_id))} for split, sample_id in entries],
    } for signature, entries in canonical_groups.items() if len(entries) > 1]
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        common = sorted(keys_by_split[left] & keys_by_split[right])
        report["cross_split"][f"{left}__{right}"] = {
            "shared_sample_id_count": len(common), "shared_sample_ids": common,
            "shared_signer_labels": sorted(signers_by_split[left] & signers_by_split[right]),
            "shared_label_counts": {field: len(values & labels_by_split[right].get(field, set())) for field, values in labels_by_split[left].items()},
            "overlap_content_hashes": [{
                "sample_id": sample_id,
                "annotation_identical": annotation_payloads[left][sample_id] == annotation_payloads[right][sample_id],
                left: ordered_frame_hash(frame_root / left / normalized_frame_id(sample_id)),
                right: ordered_frame_hash(frame_root / right / normalized_frame_id(sample_id)),
            } for sample_id in common],
        }
    labels_to_classes = collections.defaultdict(list)
    all_class_labels = {key: set(class_labels[key]) | set(ambiguous_class_labels[key]) for key in set(class_labels) | set(ambiguous_class_labels)}
    for class_name, values in all_class_labels.items():
        for value in values:
            labels_to_classes[value].append(class_name)
    report["raw_rgb_inventory"]["source_class_to_annotation_label_char"] = {key: sorted(values) for key, values in sorted(class_labels.items())}
    report["raw_rgb_inventory"]["source_class_to_ambiguous_possible_label_char"] = {key: sorted(values) for key, values in sorted(ambiguous_class_labels.items())}
    report["raw_rgb_inventory"]["label_char_shared_by_source_classes"] = {key: sorted(values) for key, values in labels_to_classes.items() if len(values) > 1}
    for archive in sorted(root.parent.rglob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            report["archives"][str(archive)] = {
                "archive_bytes": archive.stat().st_size, "members": len(members),
                "uncompressed_bytes": sum(item.file_size for item in members),
                "directory_sample": [item.filename for item in members[:20]],
                "npy_json_csv_txt_members": [item.filename for item in members if Path(item.filename).suffix.lower() in {".npy", ".json", ".csv", ".txt"}][:50],
                "png_files_by_split": {split: sum(f"/fullFrame-256x256px/{split}/" in item.filename and item.filename.endswith(".png") for item in members) for split in ("train", "dev", "test")},
            }
    report["limits"] = [
        "Signers are dataset-provided labels; global identity has not been independently verified.",
        "Cross-split sample-ID overlaps have complete ordered PNG content hashes; duplicates under renamed sample IDs are not checked.",
        "Image dimensions are decoded for three samples per split, not all frame files.",
        "No exact FPS or source-video duration is inferred from frame directory names.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "splits": {key: {field: value[field] for field in ("sample_count", "signer_counts", "total_frame_files", "file_count_per_sample")} for key, value in report["splits"].items()}, "cross_split": report["cross_split"], "archives": report["archives"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
