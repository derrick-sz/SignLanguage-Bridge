"""Read-only, deterministic association audit for the updated Libras assets.

This does not import dataset rows into a production representation. Annotations
use the sibling audit's restricted NumPy unpickler; landmarks are header-only;
RGB videos and PNG pixels are never opened. Only the audit JSON is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np

from audit_libras_annotations_20261010 import load_annotation


FILE_ID = re.compile(r"^(?P<caption>.+)_(?P<informant>INFORMANTE\s+\d+\s*)_(?P<stem>IMG_\d+)$", re.IGNORECASE | re.DOTALL)
ITEM = re.compile(r"^(?P<item_id>\d+)_(?P<caption>.*)$", re.DOTALL)


def normalize_caption(value: str) -> str:
    """Keep accents/punctuation/words; fold case and whitespace/underscores."""
    return " ".join(re.sub(r"[\s_]+", " ", unicodedata.normalize("NFC", value)).casefold().split())


def compare_caption(left: str, right: str) -> dict:
    if left == right:
        status, rule = "matched", "literal_exact"
    elif normalize_caption(left) == normalize_caption(right):
        status, rule = "matched", "NFC_casefold_whitespace_underscore_equivalence"
    else:
        status, rule = "mismatch", "no_allowed_equivalence"
    return {"status": status, "rule": rule, "left": left, "right": right,
            "normalized_left": normalize_caption(left), "normalized_right": normalize_caption(right),
            "actual_newline_present": "\n" in left or "\r" in left or "\n" in right or "\r" in right}


def normalize_informant(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def review_caption(value: str) -> str:
    """Lossy caption key for review candidates only, never frozen provenance."""
    text = unicodedata.normalize("NFKD", value).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


def index_assets(root: Path, suffix: str) -> tuple[dict, dict]:
    indexed = defaultdict(list)
    invalid, item_counts = [], Counter()
    for path in sorted(root.rglob(f"*{suffix}")):
        relative = path.relative_to(root)
        parts = relative.parts
        item = ITEM.fullmatch(parts[0]) if len(parts) == 3 else None
        if item is None or not re.fullmatch(r"INFORMANTE\s+\d+", parts[1], re.IGNORECASE):
            invalid.append(str(relative))
            continue
        asset = {"resource": str(relative), "path": str(path), "item_directory": parts[0],
                 "item_id": str(int(item.group("item_id"))), "caption": item.group("caption"),
                 "local_informant": parts[1], "stem": path.stem}
        key = (normalize_informant(parts[1]), path.stem.casefold())
        indexed[key].append(asset)
        item_counts[parts[0]] += 1
    return indexed, {"root": str(root), "structured_files": sum(item_counts.values()),
                     "item_count": len(item_counts), "files_per_item": dict(sorted(item_counts.items())),
                     "invalid_structure": invalid,
                     "nonunique_local_informant_stem_keys": [
                         {"key": list(key), "resources": [asset["resource"] for asset in assets]}
                         for key, assets in sorted(indexed.items()) if len(assets) > 1]}


def header(path: Path, cache: dict) -> dict:
    resource = str(path)
    if resource in cache:
        return cache[resource]
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            header_bytes = stream.tell()
        result = {"shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran),
                  "header_bytes": header_bytes, "file_bytes": path.stat().st_size,
                  "object_dtype": bool(dtype.hasobject), "npy_version": list(version),
                  "frame_count": shape[0] if len(shape) == 2 and not dtype.hasobject else None,
                  "numeric_payload_read": False, "error": None}
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}", "numeric_payload_read": False,
                  "frame_count": None}
    cache[resource] = result
    return result


def asset_candidates(index: dict, key: tuple | None, source: dict | None) -> dict:
    candidates = index.get(key, []) if key is not None else []
    # A known source item ID is structural provenance, not a global signer ID.
    if source is not None:
        candidates = [asset for asset in candidates if asset["item_id"] == source["item_id"]]
        rule = "source_item_id_plus_local_informant_plus_IMG_stem"
    else:
        rule = "local_informant_plus_IMG_stem_only_source_unresolved"
    status = "matched" if len(candidates) == 1 else "missing" if not candidates else "ambiguous"
    return {"status": status, "rule": rule, "candidate_count": len(candidates), "candidates": candidates}


def frame_folder_index(root: Path) -> tuple[dict, dict]:
    exact, normalized = {}, defaultdict(list)
    for entry in sorted(os.scandir(root), key=lambda entry: entry.name):
        if entry.is_dir():
            exact[entry.name] = entry.path
            normalized[normalize_caption(entry.name)].append(entry.name)
    return exact, normalized


def resolve_frames(fileid: str, exact: dict, normalized: dict, cache: dict) -> dict:
    if fileid in exact:
        names, rule = [fileid], "literal_exact"
    else:
        names = normalized.get(normalize_caption(fileid), [])
        rule = "NFC_casefold_whitespace_underscore_equivalence_including_actual_newlines"
    status = "matched" if len(names) == 1 else "missing" if not names else "ambiguous"
    result = {"status": status, "rule": rule, "candidate_count": len(names), "candidates": names,
              "actual_newline_present_in_fileid": "\n" in fileid or "\r" in fileid,
              "png_count": None, "other_file_count": None}
    if len(names) == 1:
        path = exact[names[0]]
        if path not in cache:
            suffixes = Counter(Path(entry.name).suffix.casefold() for entry in os.scandir(path) if entry.is_file())
            cache[path] = {"png_count": suffixes[".png"], "other_file_count": sum(suffixes.values()) - suffixes[".png"]}
        result.update(cache[path])
        result["path"] = path
    return result


def joined_labels(sample: dict) -> list[str]:
    values = []
    for entry in sample.get("original_info", []):
        tokens = entry.get("label_char", []) if isinstance(entry, dict) else []
        if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
            values.append("".join(tokens))
    return values


def run(args) -> dict:
    raw_root, landmark_root, t_root = (Path(value).resolve() for value in (args.raw_root, args.landmark_root, args.t_root))
    body_root = landmark_root / "Libras-EQT-UECE (Landmarks)"
    hand_root = landmark_root / "Libras-EQT-UECE (Hand Landmarks)"
    raw, raw_inventory = index_assets(raw_root, ".mp4")
    body, body_inventory = index_assets(body_root, ".npy")
    hand, hand_inventory = index_assets(hand_root, ".npy")
    report = {"schema": "libras_asset_join_audit_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "roots": {"raw": str(raw_root), "body": str(body_root), "hand": str(hand_root), "T": str(t_root)},
              "policy": {"source_key": "fileid local INFORMANTE label (strip/collapse whitespace, retain original) + IMG stem; duplicate items remain ambiguous",
                         "informant_semantics": "Local label within an item only; not a verified global signer identity",
                         "caption_equivalence": "literal exact first; otherwise Unicode NFC + casefold + runs of whitespace (including actual CR/LF) and underscore treated as one space; accents, punctuation and words retained",
                         "label_equivalence": "joined original_info.label_char uses the same bounded caption normalization; no Portuguese tokens are declared gloss labels",
                         "review_candidate_equivalence": "NFKD + combining accents removed + casefold + punctuation/underscore treated as whitespace + whitespace collapsed; compare original_info.label_char to source caption; unique candidates only; not provenance confirmation",
                         "safety": "restricted annotation unpickler; NPY header only; filename counts only; no numeric NPY payload, PNG decoding/hashing or RGB decoding/hashing; no writes under source roots"},
              "inventory": {"raw": raw_inventory, "body": body_inventory, "hand": hand_inventory},
              "splits": {}, "records": [], "cross_split_source_groups": [],
              "cross_split_same_fileid_groups": [], "cross_split_possible_source_groups": []}
    header_cache, frame_cache = {}, {}
    source_groups = defaultdict(list)
    fileid_groups, possible_source_groups = defaultdict(list), defaultdict(list)
    for split in ("train", "dev", "test"):
        payload, annotation_info = load_annotation(t_root / "annotations" / f"{split}.npy")
        samples = {key: value for key, value in payload.items() if isinstance(value, dict)}
        exact, normalized = frame_folder_index(t_root / "features" / "fullFrame-256x256px" / split)
        counts = Counter()
        items = Counter()
        for sample_id, sample in samples.items():
            counts["records"] += 1
            fileid = sample.get("fileid")
            if not isinstance(fileid, str):
                fileid = sample_id
            match = FILE_ID.fullmatch(fileid)
            key = (normalize_informant(match.group("informant")), match.group("stem").casefold()) if match else None
            sources = raw.get(key, []) if key is not None else []
            source_status = "matched" if len(sources) == 1 else "missing" if not sources else "ambiguous"
            source = sources[0] if len(sources) == 1 else None
            counts[f"source_{source_status}"] += 1
            whitespace_alias = bool(match and match.group("informant") != match.group("informant").strip())
            counts["fileid_local_informant_trailing_whitespace_alias"] += int(whitespace_alias)
            occurrence = {"split": split, "sample_id": sample_id, "fileid": fileid,
                          "source_status": source_status}
            fileid_groups[fileid].append(occurrence)
            for candidate in sources:
                possible_source_groups[candidate["resource"]].append(occurrence)
            row = {"split": split, "sample_id": sample_id, "fileid": fileid,
                   "sample_id_equals_fileid": sample_id == fileid,
                   "fileid_structure_valid": match is not None,
                   "local_informant": match.group("informant") if match else None,
                   "local_informant_canonical": normalize_informant(match.group("informant")) if match else None,
                   "local_informant_trailing_whitespace_alias": whitespace_alias,
                   "annotation_signer": sample.get("signer"),
                   "annotation_signer_matches_fileid": bool(match and isinstance(sample.get("signer"), str)
                       and normalize_informant(sample["signer"]) == normalize_informant(match.group("informant"))),
                   "source": {"status": source_status, "rule": "unique_local_informant_plus_IMG_stem_no_caption_disambiguation",
                              "candidate_count": len(sources), "candidates": sources},
                   "body": asset_candidates(body, key, source), "hand": asset_candidates(hand, key, source),
                   "frames": resolve_frames(fileid, exact, normalized, frame_cache),
                   "annotation_label_char_texts": joined_labels(sample), "caption_match": None,
                   "label_match": None, "body_hand_frame_pairing": "unresolved"}
            counts[f"frames_{row['frames']['status']}"] += 1
            if source is not None:
                items[source["item_directory"]] += 1
                source_groups[source["resource"]].append({"split": split, "sample_id": sample_id, "fileid": fileid})
                row["caption_match"] = compare_caption(match.group("caption"), source["caption"])
                counts[f"caption_{row['caption_match']['status']}"] += 1
                comparisons = [compare_caption(value, source["caption"]) for value in row["annotation_label_char_texts"]]
                row["label_match"] = {"status": "matched" if any(value["status"] == "matched" for value in comparisons)
                                      else "missing" if not comparisons else "mismatch", "comparisons": comparisons}
                counts[f"label_{row['label_match']['status']}"] += 1
            for kind in ("body", "hand"):
                asset = row[kind]
                counts[f"{kind}_{asset['status']}"] += 1
                if asset["status"] == "matched":
                    selected = asset["candidates"][0]
                    asset["header"] = header(Path(selected["path"]), header_cache)
                    asset["caption_match_to_source"] = compare_caption(selected["caption"], source["caption"]) if source else None
                    count = asset["header"]["frame_count"]
                    png_count = row["frames"]["png_count"]
                    same = count == png_count if count is not None and png_count is not None else None
                    asset["npy_frames_equal_png_count"] = same
                    counts[f"{kind}_png_length_{'matched' if same is True else 'mismatch' if same is False else 'unknown'}"] += 1
            if row["body"]["status"] == row["hand"]["status"] == "matched":
                left, right = row["body"]["header"]["frame_count"], row["hand"]["header"]["frame_count"]
                row["body_hand_frame_pairing"] = "matched" if left is not None and left == right else "mismatch" if left is not None and right is not None else "unknown"
            counts[f"body_hand_{row['body_hand_frame_pairing']}"] += 1
            row["bounded_source_caption_and_label_confirmed"] = bool(source is not None
                and row["caption_match"]["status"] == "matched" and row["label_match"]["status"] == "matched"
                and row["annotation_signer_matches_fileid"] and row["sample_id_equals_fileid"])
            counts["bounded_source_caption_and_label_confirmed"] += int(row["bounded_source_caption_and_label_confirmed"])
            review_labels = {review_caption(value) for value in row["annotation_label_char_texts"]}
            review_sources = [candidate for candidate in sources if review_caption(candidate["caption"]) in review_labels]
            review_status = "candidate" if len(review_sources) == 1 else "missing" if not review_sources else "ambiguous"
            review_source = review_sources[0] if len(review_sources) == 1 else None
            review = {"status": review_status, "rule": "label_char_NFKD_deaccent_casefold_punctuation_whitespace_unique_source_candidate",
                      "is_frozen_provenance": False, "candidate_count": len(review_sources), "candidates": review_sources,
                      "normalized_annotation_labels": sorted(review_labels),
                      "body": asset_candidates(body, key, review_source), "hand": asset_candidates(hand, key, review_source),
                      "body_hand_frame_pairing": "unresolved"}
            counts[f"review_source_{review_status}"] += 1
            if review_source is not None:
                review["flat_caption_candidate_equal"] = review_caption(match.group("caption")) == review_caption(review_source["caption"])
                counts["review_flat_caption_candidate_equal"] += int(review["flat_caption_candidate_equal"])
                for kind in ("body", "hand"):
                    asset = review[kind]
                    counts[f"review_{kind}_{asset['status']}"] += 1
                    if asset["status"] == "matched":
                        selected = asset["candidates"][0]
                        asset["header"] = header(Path(selected["path"]), header_cache)
                        frame_count = asset["header"]["frame_count"]
                        png_count = row["frames"]["png_count"]
                        same = frame_count == png_count if frame_count is not None and png_count is not None else None
                        asset["npy_frames_equal_png_count"] = same
                        counts[f"review_{kind}_png_length_{'matched' if same is True else 'mismatch' if same is False else 'unknown'}"] += 1
                if review["body"]["status"] == review["hand"]["status"] == "matched":
                    left, right = review["body"]["header"]["frame_count"], review["hand"]["header"]["frame_count"]
                    review["body_hand_frame_pairing"] = "matched" if left is not None and left == right else "mismatch" if left is not None and right is not None else "unknown"
                counts[f"review_body_hand_{review['body_hand_frame_pairing']}"] += 1
            row["review_candidate"] = review
            report["records"].append(row)
        report["splits"][split] = {"annotation": annotation_info, "counts": dict(sorted(counts.items())),
                                    "mapped_source_item_counts": dict(sorted(items.items())),
                                    "frame_directory_count": len(exact)}
        print(f"{split}: {len(samples)} annotation rows joined; {counts['source_matched']} unique RGB sources", flush=True)
    for resource, occurrences in sorted(source_groups.items()):
        splits = sorted({row["split"] for row in occurrences})
        if len(splits) > 1:
            report["cross_split_source_groups"].append({"raw_resource": resource, "splits": splits,
                                                        "occurrences": occurrences,
                                                        "evidence": "same uniquely resolved raw RGB resource; no content hashes performed"})
    for fileid, occurrences in sorted(fileid_groups.items()):
        splits = sorted({row["split"] for row in occurrences})
        if len(splits) > 1:
            report["cross_split_same_fileid_groups"].append({"fileid": fileid, "splits": splits,
                                                           "occurrences": occurrences,
                                                           "evidence": "same annotation fileid across partitions; source may remain ambiguous; content equality requires the separate hash audit"})
    for resource, occurrences in sorted(possible_source_groups.items()):
        splits = sorted({row["split"] for row in occurrences})
        if len(splits) > 1:
            report["cross_split_possible_source_groups"].append({"raw_resource": resource, "splits": splits,
                                                               "occurrences": occurrences,
                                                               "evidence": "raw resource occurs among source candidates across partitions; ambiguous association is not confirmed"})
    report["summary"] = {"annotation_rows": len(report["records"]),
                         "headers_read": len(header_cache), "frame_directories_counted": len(frame_cache),
                         "cross_split_source_group_count": len(report["cross_split_source_groups"]),
                         "cross_split_source_group_scope": "uniquely resolved raw resources only; does not cover ambiguous source candidates",
                         "cross_split_same_fileid_group_count": len(report["cross_split_same_fileid_groups"]),
                         "cross_split_possible_source_group_count": len(report["cross_split_possible_source_groups"]),
                         "counts": dict(sorted(sum((Counter(split["counts"]) for split in report["splits"].values()), Counter()).items()))}
    report["limits"] = ["Header shape and file counts do not prove numeric quality, pixel integrity or temporal alignment.",
                        "A unique INFORMANTE+IMG candidate is an association candidate; caption/label mismatches remain unconfirmed.",
                        "Strict source/caption fields use no punctuation/accent/word deletion, fuzzy matching or ambiguous-item disambiguation.",
                        "review_candidate uses separately declared lossy normalization and cannot be counted as confirmed provenance.",
                        "Only identical resolved source paths are grouped; renamed or byte-identical sources require a separate hash audit.",
                        "No global signer identity, FPS, joint topology, extraction provenance, gloss labels or streaming compatibility is inferred.",
                        "A corrected evaluation protocol must be a separate immutable experiment from the supplied original partitions."]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="E:/githubproject/SignLanguage-Bridge/Libras-EQT-UECE (RGB)")
    parser.add_argument("--landmark-root", default="D:/Zenodo/Libras-EQT-UECE/Landmarks")
    parser.add_argument("--t-root", default="D:/Zenodo/Libras-EQT-UECE/Libras-EQT-UECE")
    parser.add_argument("--output", default="artifacts/libras-updated-data-audit-20261010/asset-join-audit.json")
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output).resolve()
    roots = [Path(value).resolve() for value in (args.raw_root, args.landmark_root, args.t_root)]
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("Audit output must be outside all read-only source roots")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
