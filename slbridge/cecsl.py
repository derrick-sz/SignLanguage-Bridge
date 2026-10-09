"""Frozen CE-CSL annotations and deterministic, whole-clip TFNet RGB input.

This is a continuous gloss corpus: its official folds are retained verbatim.
The RGB representation reproduces the upstream evaluation converter and loader,
including the intermediate JPEG. No sampling or temporal truncation is applied.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ._io import canonical_hash, resolve_resource, sha256_file, write_json_atomic

if TYPE_CHECKING:
    from .sources import CapturedClip


REPRESENTATION_ID = "tfnet_rgb_v1"
TFNET_SOURCE_COMMIT = "ee7cb6a253a711e8f3699a1e8e0943179268060c"
NORMALIZER_ID = "tfnet_prewords_v1"
OFFICIAL_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class RGBInputSpec:
    representation_id: str = REPRESENTATION_ID
    input_layout: str = "B,L,C,H,W"
    dtype: str = "float32"
    channels: int = 3
    resize_size: int = 256
    crop_size: int = 224
    left_padding: int = 6
    temporal_stride: int = 4
    normalization: str = "pixels/127.5-1"
    jpeg_quality: int = 95
    max_frames: int = 600
    max_seconds: float = 15.0
    max_resource_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        fixed = {"representation_id": REPRESENTATION_ID, "input_layout": "B,L,C,H,W",
                 "dtype": "float32", "channels": 3, "resize_size": 256, "crop_size": 224,
                 "left_padding": 6, "temporal_stride": 4,
                 "normalization": "pixels/127.5-1", "jpeg_quality": 95}
        for name, expected in fixed.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"{name} must equal {expected!r} for {REPRESENTATION_ID}")
        # Raised limits are intended for trusted offline evaluation; they remain
        # bounded and are saved in the model/input manifest.
        if type(self.max_frames) is not int or not 2 <= self.max_frames <= 10000:
            raise ValueError("max_frames must be an integer between 2 and 10000")
        if (type(self.max_seconds) not in {int, float} or not math.isfinite(self.max_seconds)
                or not 0 < self.max_seconds <= 300):
            raise ValueError("max_seconds must lie in (0, 300]")
        if (type(self.max_resource_bytes) is not int
                or not 1 <= self.max_resource_bytes <= 2 * 1024**3):
            raise ValueError("max_resource_bytes must be an integer between 1 and 2 GiB")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RGBInputSpec:
        if not isinstance(value, dict):
            raise ValueError("RGBInputSpec must be an object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown RGB input fields: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class PreparedRGB:
    values: Any
    source_frames: int
    source_ref: str
    quality: dict[str, Any]
    time_range: tuple[float, float]
    spec: RGBInputSpec


def normalize_gloss(token: str) -> str:
    """Reproduce PreWords, including its one-digit and non-nested rules.

    Empty/invalid annotations are rejected explicitly; the upstream equivalent
    crashes while indexing the empty string. Whitespace is never stripped.
    """
    if not isinstance(token, str) or not token:
        raise ValueError("CE-CSL gloss tokens must be nonempty strings")
    kept, in_annotation = [], False
    for character in token:
        if character in "({[（":
            in_annotation = True
        if not in_annotation:
            kept.append(character)
        if character in ")}]）":
            in_annotation = False
    word = "".join(kept)
    if not word:
        raise ValueError(f"CE-CSL annotation normalizes to an empty gloss: {token!r}")
    if word[-1].isdigit() and not word[0].isdigit():
        word = word[:-1]
    if word[0] in ",，":
        word = "，" + word[1:]
    if word[0] in "?？":
        word = "？" + word[1:]
    if word.isdigit():
        try:
            word = str(int(word))
        except ValueError as error:
            raise ValueError(f"Numeric gloss cannot be canonicalized: {token!r}") from error
    return word


def normalize_glosses(gloss: str) -> tuple[str, ...]:
    if not isinstance(gloss, str):
        raise ValueError("CE-CSL Gloss column must be a string")
    return tuple(normalize_gloss(token) for token in gloss.split("/"))


def _read_annotations(dataset_root: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    annotations, csv_identity = {}, {}
    for split in OFFICIAL_SPLITS:
        path = dataset_root / "label" / f"{split}.csv"
        records = []
        seen = set()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None or len(header) < 4 or header[:4] != [
                    "Number", "Translator", "Chinese Sentences", "Gloss"]:
                raise ValueError(f"Invalid CE-CSL CSV header: {path}")
            for row_index, row in enumerate(reader, start=2):
                if len(row) != len(header) or not row[0] or not row[1]:
                    raise ValueError(f"Invalid CE-CSL annotation row: {split}:{row_index}")
                if row[0] in seen:
                    raise ValueError(f"Duplicate CE-CSL sample ID in {split}: {row[0]}")
                seen.add(row[0])
                # Resolve later via the safe-resource helper as these fields
                # come from annotation files rather than trusted code.
                resource = f"video/{split}/{row[1]}/{row[0]}.mp4"
                resolve_resource(dataset_root, resource)
                records.append({"sample_id": row[0], "signer": row[1], "sentence": row[2],
                                "raw_gloss": row[3], "glosses": list(normalize_glosses(row[3])),
                                "source_split": split, "csv_row": row_index,
                                "resource": resource, "note": row[4] if len(row) > 4 else ""})
        annotations[split] = records
        csv_identity[split] = {"resource": f"label/{split}.csv", "sha256": sha256_file(path),
                               "rows": len(records), "header": header}
    return annotations, csv_identity


def build_cecsl_vocabulary(dataset_root: str | Path) -> dict[str, Any]:
    """Freeze all official split glosses in the checkpoint's exact class order."""
    annotations, csv_identity = _read_annotations(Path(dataset_root).resolve())
    tokens = [" "] + sorted({token for records in annotations.values()
                             for record in records for token in record["glosses"]})
    if len(tokens) != len(set(tokens)):
        raise ValueError("CE-CSL labels contain the reserved CTC blank token")
    return {"schema_version": 1, "blank_id": 0, "tokens": tokens,
            "provenance": {"dataset_id": "CE-CSL", "normalizer_id": NORMALIZER_ID,
                           "source_commit": TFNET_SOURCE_COMMIT,
                           "token_sources": list(OFFICIAL_SPLITS), "csv": csv_identity,
                           "ordering": "blank_space_then_python_sorted_normalized_unique_glosses",
                           "punctuation_retained": True}}


def preprocessing_manifest(spec: RGBInputSpec | None = None) -> dict[str, Any]:
    spec = _spec(spec)
    return {"schema_version": 1, "input_spec": spec.to_dict(),
            "source_commit": TFNET_SOURCE_COMMIT,
            "temporal_policy": "all_original_decoded_frames_in_order_no_resampling_no_truncation",
            "frame_pipeline": ["opencv_decode_bgr_uint8", "resize_256x256_INTER_LINEAR",
                               "jpeg_encode_quality95_then_decode_bgr", "BGR_to_RGB",
                               "resize_256x256_INTER_LINEAR", "center_crop_rows16:240_cols16:240",
                               "transpose_C_H_W_float32", "divide_127.5_subtract_1"],
            "padding": {"left": "6_copies_first_frame", "right": "6+(-source_frames)%4_copies_last_frame",
                        "single_clip_length": "4*ceil(source_frames/4)+12",
                        "model_output_steps": "ceil(source_frames/4)"},
            "resource_limit_definition": "compressed_file_bytes_or_total_captured_BGR_bytes",
            "duration_definition": "file_frame_count/fps_or_captured_last_minus_first_timestamp",
            "training_augmentation_applied": False}


def freeze_cecsl_dataset(dataset_root: str | Path, output_dir: str | Path,
                        *, upstream_root: str | Path | None = None,
                        expected_vocab_size: int | None = 3516,
                        spec: RGBInputSpec | None = None,
                        require_all_videos: bool = True) -> dict[str, Any]:
    """Write reproducible vocab, official index/folds, and preprocessing evidence.

    Video existence and byte size are indexed; whole-video hashes are deliberately
    not computed. This avoids scanning a large corpus merely to freeze CSV order.
    """
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    annotations, csv_identity = _read_annotations(root)
    vocabulary = build_cecsl_vocabulary(root)
    tokens = vocabulary["tokens"]
    if expected_vocab_size is not None and len(tokens) != expected_vocab_size:
        raise ValueError(f"Expected {expected_vocab_size} CE-CSL classes including blank, got {len(tokens)}")
    word_to_id = {word: index for index, word in enumerate(tokens)}
    samples, missing = [], []
    all_ids = set()
    for split in OFFICIAL_SPLITS:
        for original in annotations[split]:
            record = dict(original)
            if record["sample_id"] in all_ids:
                raise ValueError(f"CE-CSL sample ID crosses official splits: {record['sample_id']}")
            all_ids.add(record["sample_id"])
            path = resolve_resource(root, record["resource"])
            exists = path.is_file()
            record.update({"gloss_ids": [word_to_id[word] for word in record["glosses"]],
                           "video_exists": exists, "video_bytes": path.stat().st_size if exists else None})
            if not exists:
                missing.append(record["resource"])
            samples.append(record)
    if require_all_videos and missing:
        raise ValueError(f"Missing {len(missing)} CE-CSL videos; first: {missing[0]}")
    signers = {split: sorted({record["signer"] for record in annotations[split]}) for split in OFFICIAL_SPLITS}
    overlap = {f"{left}_{right}": sorted(set(signers[left]) & set(signers[right]))
               for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    source_hashes = {}
    if upstream_root is not None:
        upstream = Path(upstream_root)
        for name in ("DataProcessMoudle.py", "videoAugmentation.py", "CE-CSLDataPreProcess.py"):
            source_hashes[name] = sha256_file(upstream / name)
    provenance = {"dataset_root": str(root), "dataset_id": "CE-CSL",
                  "task_type": "continuous_gloss_ctc", "csv": csv_identity,
                  "source_commit": TFNET_SOURCE_COMMIT, "source_file_hashes": source_hashes,
                  "video_verification": "exists_and_size_only_not_hashed_or_decoded",
                  "split_policy": "official_csv_membership_and_row_order_preserved",
                  "signer_disjoint": not any(overlap.values()), "signers": signers,
                  "signer_overlap": overlap,
                  "signer_overlap_note": "Official splits may share signers; no independent-signer claim is made.",
                  "classification_label_coverage_required": False}
    index = {"schema_version": 1, "provenance": provenance, "samples": samples}
    splits = {"schema_version": 1, "dataset_index_hash": canonical_hash(index),
              "policy": "official_cecsl_continuous_gloss_splits",
              **{split: [record["sample_id"] for record in annotations[split]] for split in OFFICIAL_SPLITS},
              "signers": signers, "signer_overlap": overlap}
    outputs = {"vocabulary.json": vocabulary, "preprocessing.json": preprocessing_manifest(spec),
               "dataset_index.json": index, "official_splits.json": splits}
    # Frozen files cannot silently change identity on a repeated invocation.
    # Preflight all four before writing, so an incompatible directory stays intact.
    from ._io import read_json
    for name, value in outputs.items():
        existing = output / name
        if existing.exists() and (not existing.is_file() or read_json(existing) != value):
            raise ValueError(f"Refusing to overwrite a different frozen CE-CSL artifact: {existing}")
    for name, value in outputs.items():
        write_json_atomic(output / name, value)
    summary = {"schema_version": 1, "vocabulary_size": len(tokens), "blank_id": 0,
               "normalized_gloss_count": len(tokens) - 1,
               "rows": {split: len(annotations[split]) for split in OFFICIAL_SPLITS},
               "videos_present": len(samples) - len(missing), "videos_missing": missing,
               "signers": signers, "signer_overlap": overlap,
               "punctuation_ids": {token: word_to_id[token] for token in ("。", "，", "？") if token in word_to_id},
               "artifacts": {name: {"path": str(output / name), "sha256": sha256_file(output / name),
                                    "canonical_hash": canonical_hash(value)} for name, value in outputs.items()}}
    existing_summary = output / "freeze_summary.json"
    if existing_summary.exists() and (not existing_summary.is_file() or read_json(existing_summary) != summary):
        raise ValueError(f"Refusing to overwrite a different frozen CE-CSL summary: {existing_summary}")
    write_json_atomic(existing_summary, summary)
    return summary


def _spec(spec: RGBInputSpec | None) -> RGBInputSpec:
    if spec is None:
        return RGBInputSpec()
    if not isinstance(spec, RGBInputSpec):
        raise TypeError("spec must be RGBInputSpec")
    return spec


def padded_rgb_length(source_frames: int) -> int:
    if type(source_frames) is not int or source_frames < 1:
        raise ValueError("source_frames must be a positive integer")
    return ((source_frames + 3) // 4) * 4 + 12


def _check_count(count: int, spec: RGBInputSpec) -> None:
    if count < 2:
        raise ValueError("At least two original RGB frames are required")
    if count > spec.max_frames:
        raise ValueError(f"RGB frame limit exceeded ({count}>{spec.max_frames}); clip is never truncated")


def _check_duration(seconds: float, spec: RGBInputSpec) -> None:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("RGB clip duration must be finite and nonnegative")
    if seconds > spec.max_seconds:
        raise ValueError(f"RGB duration limit exceeded ({seconds:.6g}>{spec.max_seconds}); clip is never truncated")


def _frame_values(image: Any, spec: RGBInputSpec) -> Any:
    import cv2
    import numpy as np

    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("RGB preprocessing requires nonempty uint8 BGR frames")
    if image.nbytes > spec.max_resource_bytes:
        raise ValueError("Decoded RGB frame exceeds the resource byte limit")
    bgr = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality])
    if not ok:
        raise ValueError("RGB reference JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("RGB reference JPEG decoding failed")
    rgb = cv2.resize(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_LINEAR)
    values = np.ascontiguousarray(rgb[16:240, 16:240].transpose(2, 0, 1), dtype=np.float32)
    values /= np.float32(127.5)
    values -= np.float32(1.0)
    return values


def _finish(values: Any, count: int, source_ref: str, quality: dict[str, Any],
            time_range: tuple[float, float], spec: RGBInputSpec) -> PreparedRGB:
    values[0, :6] = values[0, 6]
    values[0, 6 + count:] = values[0, 6 + count - 1]
    quality.update({"source_frames": count, "sampled_frames": count, "padded_frames": values.shape[1],
                    "left_padding": 6, "right_padding": values.shape[1] - count - 6,
                    "temporal_resampling": False, "truncated": False,
                    "representation_id": REPRESENTATION_ID, "jpeg_quality": 95,
                    "preprocessing": "upstream_resize256_jpeg95_rgb_center224_float32"})
    values.setflags(write=False)
    return PreparedRGB(values, count, source_ref, quality, time_range, spec)


def prepare_rgb_clip(clip: CapturedClip, spec: RGBInputSpec | None = None) -> PreparedRGB:
    """Prepare all original BGR packets of an explicitly ended recording."""
    from .sources import CapturedClip
    import numpy as np

    spec = _spec(spec)
    if not isinstance(clip, CapturedClip):
        raise TypeError("clip must be CapturedClip")
    # Frames are mutable arrays even though packets are frozen dataclasses.
    clip.__post_init__()
    count = len(clip.frames)
    _check_count(count, spec)
    first, last = float(clip.frames[0].timestamp), float(clip.frames[-1].timestamp)
    _check_duration(last - first, spec)
    if sum(packet.image.nbytes for packet in clip.frames) > spec.max_resource_bytes:
        raise ValueError("Captured RGB clip exceeds the resource byte limit")
    shape = clip.frames[0].image.shape
    if any(packet.image.shape != shape for packet in clip.frames):
        raise ValueError("RGB clip changes image dimensions")
    values = np.empty((1, padded_rgb_length(count), 3, 224, 224), dtype=np.float32)
    for index, packet in enumerate(clip.frames):
        values[0, index + 6] = _frame_values(packet.image, spec)
    return _finish(values, count, clip.source_id,
                   {"source_width": shape[1], "source_height": shape[0], "timestamps_known": True,
                    "timestamp_source": "captured_frame_timestamp", "duration_seconds": last - first,
                    "duration_definition": "last_minus_first_captured_timestamp",
                    "full_decode_verified": True}, (first, last), spec)


def prepare_rgb_file(path: str | Path, spec: RGBInputSpec | None = None) -> PreparedRGB:
    """Decode the entire bounded clip and reject unreliable/incomplete media."""
    import cv2
    import numpy as np

    spec, path = _spec(spec), Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > spec.max_resource_bytes:
        raise ValueError("RGB video exceeds the resource byte limit")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open RGB video: {path}")
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (not math.isfinite(raw_count) or raw_count < 2
                or abs(raw_count - round(raw_count)) > 0.01):
            raise ValueError("RGB video must declare a reliable frame count of at least two")
        expected = int(round(raw_count))
        _check_count(expected, spec)
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        if fps is not None:
            _check_duration(expected / fps, spec)
        values = np.empty((1, padded_rgb_length(expected), 3, 224, 224), dtype=np.float32)
        timestamps, shape, count = [], None, 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if count >= expected:
                raise ValueError(f"RGB video decoded more frames than declared ({expected}); clip is never truncated")
            if image is None:
                raise ValueError(f"RGB video has an invalid decoded frame at {count}")
            if shape is None:
                shape = image.shape
            elif image.shape != shape:
                raise ValueError("RGB video changes image dimensions")
            values[0, count + 6] = _frame_values(image, spec)
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            count += 1
        if count != expected:
            raise ValueError(f"Incomplete RGB video decoding: expected {expected} frames, decoded {count}")
        times = np.asarray(timestamps, dtype=np.float64)
        if np.isfinite(times).all() and (times >= 0).all() and (np.diff(times) > 0).all():
            timestamp_source = "decoder_pos_msec"
        elif fps is not None:
            times = np.arange(count, dtype=np.float64) / fps
            timestamp_source = "frame_index_fps"
        else:
            raise ValueError("RGB video has neither reliable timestamps nor a usable fps")
        first, last = float(times[0]), float(times[-1])
        _check_duration(last - first, spec)
        return _finish(values, count, str(path.resolve()),
                       {"source_width": shape[1], "source_height": shape[0], "source_fps": fps,
                        "source_file_bytes": path.stat().st_size, "timestamps_known": True,
                        "timestamp_source": timestamp_source, "duration_seconds": last - first,
                        "duration_definition": "last_minus_first_decoded_timestamp",
                        "media_duration_seconds": count / fps if fps is not None else None,
                        "full_decode_verified": True}, (first, last), spec)
    finally:
        capture.release()
