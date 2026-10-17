"""Read-only, pickle-free audit of the locally supplied Libras landmark arrays.

The audit reports storage facts; dimensions alone do not establish landmark order,
coordinate semantics, extractor version, timestamps, or global signer identities.
Numeric range inspection samples three deterministic files per folder-level class;
nonfinite/zero/sentinel counts and paired slice equality are checked for every file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().casefold()


def structured_key(path: Path, root: Path) -> str | None:
    """Match numeric category ID, local informant directory, and basename only."""
    relative = path.relative_to(root)
    if len(relative.parts) != 3:
        return None
    match = re.match(r"^(\d+)_", relative.parts[0])
    if match is None:
        return None
    return "/".join([str(int(match.group(1))), relative.parts[1].casefold(), path.stem.casefold()])


def quantiles(values: list[int] | np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    expected_bytes = None if dtype.hasobject else int(math.prod(shape) * dtype.itemsize)
    actual_bytes = path.stat().st_size
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "has_object_dtype": bool(dtype.hasobject),
        "fortran_order": bool(fortran_order),
        "npy_version": list(version),
        "header_bytes": offset,
        "file_bytes": actual_bytes,
        "expected_numeric_payload_bytes": expected_bytes,
        "payload_size_matches_header": None if expected_bytes is None else actual_bytes == offset + expected_bytes,
    }


def inspect_numeric(path: Path, header: dict) -> dict:
    result = {"path": str(path), "shape": header["shape"], "dtype": header["dtype"]}
    if header["has_object_dtype"]:
        return {**result, "status": "object_dtype_rejected_without_unpickling"}
    if len(header["shape"]) != 2:
        return {**result, "status": "not_2d_skipped"}
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if not np.issubdtype(array.dtype, np.number):
        return {**result, "status": "not_numeric_skipped"}
    finite = np.isfinite(array)
    zeros = array == 0
    sentinel = array == -999
    usable = finite & ~zeros & ~sentinel
    result.update(
        status="inspected",
        numeric_count=int(array.size),
        nan_count=int(np.isnan(array).sum()),
        positive_infinity_count=int(np.isposinf(array).sum()),
        negative_infinity_count=int(np.isneginf(array).sum()),
        zero_count=int(zeros.sum()),
        minus999_count=int(sentinel.sum()),
        all_zero_rows=int(np.all(zeros, axis=1).sum()),
        all_minus999_rows=int(np.all(sentinel, axis=1).sum()),
        constant_columns=int(np.count_nonzero(np.all(array == array[:1], axis=0))),
        finite_nonzero_nonsentinel_values=quantiles(array[usable]),
    )
    # Column modulo three is a storage observation, not an asserted xyz contract.
    if array.shape[1] % 3 == 0:
        result["column_modulo_3_numeric_ranges_not_semantic_xyz"] = []
        for column_mod in range(3):
            block = array[:, column_mod::3]
            good = np.isfinite(block) & (block != 0) & (block != -999)
            result["column_modulo_3_numeric_ranges_not_semantic_xyz"].append(quantiles(block[good]))
    if array.shape[1] == 126:
        result["63_column_storage_block_zero_rows"] = [
            int(np.all(array[:, :63] == 0, axis=1).sum()),
            int(np.all(array[:, 63:] == 0, axis=1).sum()),
        ]
    del array
    return result


def numeric_counts(array: np.ndarray) -> dict:
    result = {
        "files": 1,
        "frames": int(array.shape[0]),
        "numeric_count": int(array.size),
        "nan_count": int(np.isnan(array).sum()),
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "negative_infinity_count": int(np.isneginf(array).sum()),
        "zero_count": int(np.count_nonzero(array == 0)),
        "minus999_count": int(np.count_nonzero(array == -999)),
        "all_zero_rows": int(np.all(array == 0, axis=1).sum()),
        "all_minus999_rows": int(np.all(array == -999, axis=1).sum()),
        "entire_array_zero_files": int(np.all(array == 0)),
    }
    if array.shape[1] == 126:
        result["first63_storage_block_all_zero_rows"] = int(np.all(array[:, :63] == 0, axis=1).sum())
        result["last63_storage_block_all_zero_rows"] = int(np.all(array[:, 63:] == 0, axis=1).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, default=Path("D:/Zenodo/Libras-EQT-UECE/Landmarks"))
    parser.add_argument("--rgb", type=Path, default=Path("E:/githubproject/SignLanguage-Bridge/Libras-EQT-UECE (RGB)"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/libras-updated-data-audit-20261010/landmark-audit.json"))
    args = parser.parse_args()
    report = {
        "schema": "libras-landmark-storage-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "landmark_root": str(args.landmarks.resolve()),
        "rgb_root": str(args.rgb.resolve()),
        "safety": {"allow_pickle": False, "object_dtype_policy": "header_only_reject", "training": False, "gpu": False, "video_decode": False},
        "numeric_sampling": "three deterministic files per class: first, middle, last; each selected numeric array scanned in full independently",
        "semantic_limitations": [
            "Dimensions do not establish landmark ordering, xyz units, or MediaPipe version.",
            "INFORMANTE numbers are local directory labels and are not asserted global signer identities.",
            "Zero values are counted as observed; absent metadata does not prove which zero blocks mean missing detections.",
            "Path pairing does not independently prove frame synchronization, timestamps, or identical extractor configuration.",
        ],
    }
    rgb_paths = sorted(p for p in args.rgb.rglob("*") if p.is_file() and p.suffix.casefold() in {".mp4", ".avi", ".mov", ".mkv"}) if args.rgb.exists() else []
    rgb_map = {relative_key(p, args.rgb): p for p in rgb_paths}
    rgb_structured = defaultdict(list)
    for path in rgb_paths:
        key = structured_key(path, args.rgb)
        if key is not None:
            rgb_structured[key].append(path)
    report["rgb_video_count"] = len(rgb_paths)
    report["rgb_structured_key_collision_count"] = sum(len(v) != 1 for v in rgb_structured.values())
    roots = sorted(p for p in args.landmarks.iterdir() if p.is_dir())
    inventories = {}
    samples = {}
    report["datasets"] = {}
    for root in roots:
        all_files = sorted(p for p in root.rglob("*") if p.is_file())
        paths = [p for p in all_files if p.suffix.casefold() == ".npy"]
        headers = {}
        errors = []
        by_class = defaultdict(list)
        class_informants = defaultdict(Counter)
        shape_counts, dtype_counts, widths, sequence_lengths = Counter(), Counter(), Counter(), []
        for path in paths:
            relative = path.relative_to(root)
            key = relative_key(path, root)
            try:
                header = read_header(path)
            except Exception as exc:
                errors.append({"path": str(path), "error_type": type(exc).__name__, "error": str(exc)})
                continue
            header.update(relative_path=relative.as_posix(), rgb_path=str(rgb_map[key]) if key in rgb_map else None)
            source_key = structured_key(path, root)
            structured_matches = rgb_structured.get(source_key, [])
            header["rgb_by_classid_localinformant_stem"] = str(structured_matches[0]) if len(structured_matches) == 1 else None
            headers[key] = header
            shape_counts[str(tuple(header["shape"]))] += 1
            dtype_counts[header["dtype"]] += 1
            if len(header["shape"]) == 2:
                sequence_lengths.append(header["shape"][0])
                widths[str(header["shape"][1])] += 1
            category = relative.parts[0] if len(relative.parts) > 1 else "__root__"
            informant = relative.parts[1] if len(relative.parts) > 2 else "__none__"
            by_class[category].append(path)
            class_informants[category][informant] += 1
        selected = []
        for paths_in_class in by_class.values():
            for index in sorted({0, len(paths_in_class) // 2, len(paths_in_class) - 1}):
                selected.append(paths_in_class[index])
        numeric = [inspect_numeric(p, headers[relative_key(p, root)]) for p in selected]
        samples[root.name] = selected
        inventories[root.name] = (root, headers)
        totals = Counter()
        for result in numeric:
            if result["status"] == "inspected":
                for field in ("numeric_count", "nan_count", "positive_infinity_count", "negative_infinity_count", "zero_count", "minus999_count", "all_zero_rows", "all_minus999_rows"):
                    totals[field] += result[field]
        report["datasets"][root.name] = {
            "root": str(root),
            "npy_count": len(paths),
            "header_success_count": len(headers),
            "header_errors": errors,
            "class_directory_count": len(by_class),
            "class_counts": {k: len(v) for k, v in sorted(by_class.items())},
            "class_informant_local_directory_counts": {k: dict(sorted(v.items())) for k, v in sorted(class_informants.items())},
            "files_per_class_distribution": dict(sorted(Counter(len(v) for v in by_class.values()).items())),
            "file_extension_counts": dict(Counter(p.suffix.casefold() for p in all_files)),
            "non_npy_files": [str(p.relative_to(root)) for p in all_files if p.suffix.casefold() != ".npy"],
            "dtype_counts": dict(dtype_counts),
            "width_counts": dict(widths),
            "shape_counts": dict(shape_counts),
            "sequence_length_quantiles": quantiles(sequence_lengths),
            "sequence_length_histogram": dict(sorted(Counter(sequence_lengths).items())),
            "zero_length_sequence_count": sum(length == 0 for length in sequence_lengths),
            "object_dtype_count": sum(h["has_object_dtype"] for h in headers.values()),
            "payload_size_mismatch_count": sum(h["payload_size_matches_header"] is False for h in headers.values()),
            "bytes": sum(h["file_bytes"] for h in headers.values()),
            "rgb_relative_path_match_count": sum(h["rgb_path"] is not None for h in headers.values()),
            "rgb_relative_path_missing": [h["relative_path"] for h in headers.values() if h["rgb_path"] is None],
            "rgb_classid_localinformant_stem_match_count": sum(h["rgb_by_classid_localinformant_stem"] is not None for h in headers.values()),
            "rgb_classid_localinformant_stem_missing": [h["relative_path"] for h in headers.values() if h["rgb_by_classid_localinformant_stem"] is None],
            "rgb_class_directory_name_differences": sorted({(h["relative_path"].split("/")[0], Path(h["rgb_by_classid_localinformant_stem"]).relative_to(args.rgb).parts[0]) for h in headers.values() if h["rgb_path"] is None and h["rgb_by_classid_localinformant_stem"] is not None}),
            "numeric_sample_count": len(numeric),
            "numeric_sample_totals": dict(totals),
            "numeric_samples": numeric,
            "headers": headers,
        }
        print(root.name, len(paths), len(by_class), dict(widths), quantiles(sequence_lengths), flush=True)
    hand_name, full_name = "Libras-EQT-UECE (Hand Landmarks)", "Libras-EQT-UECE (Landmarks)"
    if hand_name in inventories and full_name in inventories:
        hand_root, hand_headers = inventories[hand_name]
        full_root, full_headers = inventories[full_name]
        common = sorted(set(hand_headers) & set(full_headers))
        mismatched = [key for key in common if hand_headers[key]["shape"][0] != full_headers[key]["shape"][0]]
        pair_numeric = []
        for hand_path in samples[hand_name]:
            key = relative_key(hand_path, hand_root)
            if key not in full_headers or hand_headers[key]["has_object_dtype"] or full_headers[key]["has_object_dtype"]:
                continue
            full_path = full_root / full_headers[key]["relative_path"]
            hand = np.load(hand_path, allow_pickle=False, mmap_mode="r")
            full = np.load(full_path, allow_pickle=False, mmap_mode="r")
            exact_offsets = []
            if hand.ndim == 2 and full.ndim == 2 and hand.shape[0] == full.shape[0] and hand.shape[1] <= full.shape[1]:
                # Search all possible contiguous offsets; no order assumption.
                for offset in range(full.shape[1] - hand.shape[1] + 1):
                    if np.array_equal(hand, full[:, offset:offset + hand.shape[1]]):
                        exact_offsets.append(offset)
            pair_numeric.append({"relative_path": hand_headers[key]["relative_path"], "hand_shape": list(hand.shape), "full_shape": list(full.shape), "exact_contiguous_full_offsets": exact_offsets})
            del hand, full
        report["pairing"] = {
            "common_relative_video_key_count": len(common),
            "hand_only_relative_keys": sorted(set(hand_headers) - set(full_headers)),
            "full_only_relative_keys": sorted(set(full_headers) - set(hand_headers)),
            "sequence_length_mismatch_count": len(mismatched),
            "sequence_length_mismatches": mismatched,
            "numeric_comparison_sample_count": len(pair_numeric),
            "numeric_comparison_exact_offset_histogram": dict(Counter(str(p["exact_contiguous_full_offsets"]) for p in pair_numeric)),
            "numeric_comparisons": pair_numeric,
        }
        # Bounded resident memory: inspect one corresponding video pair at a time.
        full_scan = {"scope": "every numeric 2d array, streamed one pair at a time; no pickle or RGB decode", "datasets": {}, "pair_count": 0, "equal_hand_to_full_192_318_count": 0, "nonmatching_relative_paths": [], "errors": []}
        counts = {hand_name: Counter(), full_name: Counter()}
        hand_missingness = []
        for key, header in hand_headers.items():
            try:
                hand = np.load(hand_root / header["relative_path"], allow_pickle=False, mmap_mode="r")
                counts[hand_name].update(numeric_counts(hand))
                allzero_rows = np.all(hand == 0, axis=1)
                hand_missingness.append({"relative_path": header["relative_path"], "frames": int(hand.shape[0]), "entire_row_zero_frames": int(allzero_rows.sum()), "entire_row_zero_fraction": float(allzero_rows.mean())})
                if key in full_headers:
                    full = np.load(full_root / full_headers[key]["relative_path"], allow_pickle=False, mmap_mode="r")
                    counts[full_name].update(numeric_counts(full))
                    full_scan["pair_count"] += 1
                    if full.ndim == 2 and full.shape[1] == 318 and np.array_equal(hand, full[:, 192:318]):
                        full_scan["equal_hand_to_full_192_318_count"] += 1
                    else:
                        full_scan["nonmatching_relative_paths"].append(header["relative_path"])
                    del full
                del hand
            except Exception as exc:
                full_scan["errors"].append({"relative_path": header["relative_path"], "error_type": type(exc).__name__, "error": str(exc)})
        for key in set(full_headers) - set(hand_headers):
            try:
                full = np.load(full_root / full_headers[key]["relative_path"], allow_pickle=False, mmap_mode="r")
                counts[full_name].update(numeric_counts(full))
                del full
            except Exception as exc:
                full_scan["errors"].append({"relative_path": full_headers[key]["relative_path"], "error_type": type(exc).__name__, "error": str(exc)})
        full_scan["datasets"] = {name: dict(values) for name, values in counts.items()}
        full_scan["hand_per_file_zero_row_statistics"] = {
            "zero_row_fraction_quantiles": quantiles([r["entire_row_zero_fraction"] for r in hand_missingness]),
            "files_at_least_half_rows_zero": sum(r["entire_row_zero_fraction"] >= 0.5 for r in hand_missingness),
            "files_all_rows_zero": [r["relative_path"] for r in hand_missingness if r["entire_row_zero_fraction"] == 1.0],
            "per_file": hand_missingness,
        }
        report["full_numeric_stream_scan"] = full_scan
        print("Full numeric scan:", {k: v for k, v in full_scan.items() if k not in {"datasets", "hand_per_file_zero_row_statistics"}}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Audit saved:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
