"""Shared deterministic processing for legacy landmark clips.

The historical clips have no trustworthy timestamps. Resampling uses relative
frame position, never a fabricated fps. Existing coordinates are preserved.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from ._io import canonical_hash
from .contracts import InputSpec, PreparedInput


class InputRejected(ValueError):
    """Well-formed input with insufficient observable content."""

    def __init__(self, reason: str, quality: dict[str, Any]) -> None:
        super().__init__(reason)
        self.quality = quality
        self.reason = reason


def validate_sequence(array: Any, representation_id: str) -> Any:
    import numpy as np

    widths = {"hand63_v1": 63, "pose_hands225_legacy": 225, "mediapipe225_rgb_v1": 225}
    if representation_id not in widths:
        raise ValueError(f"Unsupported NPY representation: {representation_id}")
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape[1] != widths[representation_id]:
        raise ValueError(f"Expected (frames, {widths[representation_id]}) for {representation_id}; got {getattr(array, 'shape', None)}")
    if array.dtype.kind != "f" or array.dtype.itemsize not in {4, 8}:
        raise ValueError(f"Expected float32/float64 coordinates; got {array.dtype}")
    if not 1 <= array.shape[0] <= 10000:
        raise ValueError("Sequence must contain 1..10000 frames")
    if not np.isfinite(array).all():
        raise ValueError("Sequence contains NaN/Inf")
    if representation_id == "hand63_v1" and (array == -999).any():
        raise ValueError("hand63_v1 does not permit the -999 missing sentinel")
    if np.abs(array[array != -999]).max(initial=0) > np.finfo(np.float32).max:
        raise ValueError("Coordinates overflow float32")
    if representation_id == "mediapipe225_rgb_v1":
        triples = array.reshape(len(array), 75, 3)
        observed = triples != -999
        if (observed.any(axis=2) != observed.all(axis=2)).any():
            raise ValueError("RGB landmark missingness must apply to entire xyz triples")
        if (triples[:, :, 2][observed[:, :, 2]] != 0).any():
            raise ValueError("mediapipe225_rgb_v1 declares planar xy with observed z=0")
    return array


def load_sequence(path: str | Path, representation_id: str) -> Any:
    import numpy as np

    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Landmark input must be a .npy file")
    if representation_id == "mediapipe225_rgb_v1":
        raise ValueError("RGB landmarks require a timestamped .npz archive, not legacy .npy")
    from .input_limits import check_landmark_file
    check_landmark_file(path, representation_id)
    # Pickled/object arrays are never accepted from source data.
    return validate_sequence(np.load(path, allow_pickle=False), representation_id)


def load_landmark_archive(path: str | Path, expected_extractor: dict[str, Any] | None = None) -> tuple[Any, Any, Any, dict[str, Any]]:
    import zipfile
    import zlib
    try:
        return _load_landmark_archive(path, expected_extractor)
    except (zipfile.BadZipFile, zlib.error, EOFError) as error:
        raise ValueError("Invalid landmark ZIP archive payload") from error


def _load_landmark_archive(path: str | Path, expected_extractor: dict[str, Any] | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np

    if Path(path).suffix.lower() != ".npz":
        raise ValueError("RGB landmark archive must be .npz")
    from .input_limits import check_landmark_file
    check_landmark_file(path, "mediapipe225_rgb_v1")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"values", "timestamps", "frame_indices", "metadata"}:
            raise ValueError("Unexpected landmark archive entries")
        values = validate_sequence(archive["values"], "mediapipe225_rgb_v1").copy()
        timestamps, indices, encoded = archive["timestamps"].copy(), archive["frame_indices"].copy(), archive["metadata"]
        if values.dtype != np.float32 or timestamps.dtype != np.float64 or indices.dtype != np.int64:
            raise ValueError("Landmark archive dtypes must be float32, float64 and int64")
        if encoded.ndim != 0 or encoded.dtype.kind != "U":
            raise ValueError("Landmark archive metadata must be scalar JSON text")
        def reject_constant(value: str) -> None:
            raise ValueError(f"Invalid non-finite archive metadata: {value}")
        def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate archive metadata key: {key}")
                result[key] = value
            return result
        metadata = json.loads(str(encoded), parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1 or metadata.get("representation_id") != "mediapipe225_rgb_v1":
        raise ValueError("Invalid landmark archive schema")
    if set(metadata) != {"schema_version", "representation_id", "extractor", "source_id", "source_hash", "quality", "provenance"}:
        raise ValueError("Unexpected landmark archive metadata fields")
    if (not isinstance(metadata["source_id"], str) or not metadata["source_id"]
            or not isinstance(metadata["source_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["source_hash"])
            or not isinstance(metadata["quality"], dict) or not isinstance(metadata["provenance"], dict)):
        raise ValueError("Invalid landmark archive provenance")
    if not isinstance(metadata["extractor"], dict) or set(metadata["extractor"]) != {"config", "identity"}:
        raise ValueError("Landmark archive has no extractor contract")
    if expected_extractor is not None and canonical_hash(metadata["extractor"]) != canonical_hash(expected_extractor):
        raise ValueError("Landmark archive extractor is incompatible with this model")
    if (timestamps.ndim != 1 or timestamps.dtype.kind != "f" or len(timestamps) != len(values)
            or len(timestamps) < 2 or not np.isfinite(timestamps).all() or timestamps[0] < 0 or not (np.diff(timestamps) > 0).all()):
        raise ValueError("Landmark timestamps must be finite and strictly increasing")
    if (indices.ndim != 1 or indices.dtype != np.int64 or len(indices) != len(values)
            or not (indices >= 0).all() or not (indices[1:] > indices[:-1]).all()):
        raise ValueError("Landmark frame indices must be strictly increasing")
    return values, timestamps, indices, metadata


def prepare_array(array: Any, spec: InputSpec, source_ref: str = "memory", *,
                  timestamps: Any = None, quality_metadata: dict[str, Any] | None = None) -> PreparedInput:
    import numpy as np

    validate_sequence(array, spec.representation_id)
    values = array.astype(np.float32, copy=True)
    mask = values != -999 if spec.missing_policy == "sentinel_to_mask" else np.ones_like(values, dtype=bool)
    valid_fraction = float(mask.mean())
    quality = dict(quality_metadata or {})
    quality.setdefault("source_frames", int(len(values)))
    quality.update({"landmark_frames": int(len(values)), "valid_fraction": valid_fraction, "timestamps_known": timestamps is not None})
    if len(values) < 2:
        raise InputRejected("at_least_two_frames_required", quality)
    if valid_fraction < spec.minimum_valid_fraction:
        raise InputRejected("insufficient_observed_coordinates", quality)
    values[~mask] = 0
    # Interpolate only when both bordering values are observed. Missing spans
    # remain masked; zeros/-999 cannot contaminate neighbouring coordinates.
    if spec.temporal_policy == "timestamp_linear":
        times = np.asarray(timestamps) if timestamps is not None else np.array([])
        if times.ndim != 1 or len(times) != len(values) or times.dtype.kind != "f" or not np.isfinite(times).all() or times[0] < 0 or not (np.diff(times) > 0).all():
            raise ValueError("Timestamp-based processing requires finite, increasing source times")
        positions = np.linspace(float(times[0]), float(times[-1]), spec.target_frames, dtype=np.float64)
        right = np.searchsorted(times, positions, side="left").clip(0, len(times) - 1)
        left = np.maximum(0, right - 1)
        left[positions == times[right]] = right[positions == times[right]]
        denominator = times[right] - times[left]
        weight = np.divide(positions - times[left], denominator, out=np.zeros_like(positions), where=denominator != 0).astype(np.float32)[:, None]
        prepared_timestamps = tuple(float(value) for value in positions)
    else:
        if timestamps is not None:
            raise ValueError("Legacy relative-frame schema does not declare real timestamps")
        positions = np.linspace(0, len(values) - 1, spec.target_frames, dtype=np.float64)
        left = np.floor(positions).astype(int)
        right = np.ceil(positions).astype(int)
        weight = (positions - left).astype(np.float32)[:, None]
        prepared_timestamps = None
    result_mask = mask[left] & mask[right]
    result = values[left] * (1 - weight) + values[right] * weight
    result[~result_mask] = 0
    result = np.ascontiguousarray(result, dtype=np.float32)
    result_mask = np.ascontiguousarray(result_mask, dtype=bool)
    quality["prepared_valid_fraction"] = float(result_mask.mean())
    if quality["prepared_valid_fraction"] < spec.minimum_valid_fraction:
        raise InputRejected("insufficient_coordinates_after_resampling", quality)
    result.setflags(write=False)
    result_mask.setflags(write=False)
    return PreparedInput(spec, result, result_mask, len(values), source_ref, quality, prepared_timestamps)


def prepare_input(path: str | Path, spec: InputSpec) -> PreparedInput:
    if spec.representation_id == "mediapipe225_rgb_v1":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            values, timestamps, _, metadata = load_landmark_archive(path, spec.extractor)
            return prepare_array(values, spec, str(path.resolve()), timestamps=timestamps, quality_metadata=metadata["quality"])
        if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise ValueError("RGB model input must be a video or compatible landmark archive")
        from .media import ExtractionConfig, extract_video, extractor_identity
        config = ExtractionConfig.from_dict(spec.extractor["config"])
        if canonical_hash(extractor_identity(config)) != canonical_hash(spec.extractor["identity"]):
            raise ValueError("Video extractor runtime differs from the model's saved identity")
        clip = extract_video(path, config)
        return prepare_array(clip.values, spec, str(path.resolve()), timestamps=clip.timestamps, quality_metadata=clip.quality)
    return prepare_array(load_sequence(path, spec.representation_id), spec, str(Path(path).resolve()))


def feature_vector(prepared: PreparedInput) -> Any:
    import numpy as np

    expected = (prepared.spec.target_frames, prepared.spec.width)
    if prepared.values.shape != expected or prepared.values.dtype != np.float32:
        raise ValueError("Prepared values do not satisfy InputSpec")
    if prepared.mask.shape != expected or prepared.mask.dtype != bool:
        raise ValueError("Prepared mask does not satisfy InputSpec")
    if not np.isfinite(prepared.values).all() or (prepared.values[~prepared.mask] != 0).any():
        raise ValueError("Invalid prepared coordinates or missing encoding")
    if prepared.spec.missing_policy == "reject" and not prepared.mask.all():
        raise ValueError("Missing coordinates incompatible with this InputSpec")
    vector = prepared.values.reshape(-1)
    if prepared.spec.missing_policy == "sentinel_to_mask":
        vector = np.concatenate((vector, prepared.mask.reshape(-1).astype(np.float32)))
    return np.ascontiguousarray(vector, dtype=np.float32)
