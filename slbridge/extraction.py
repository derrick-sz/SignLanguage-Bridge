"""Resumable, read-only RGB extraction into versioned landmark archives.

Only generated resources in a separately owned cache directory are written.
Every archive combines coordinates, source timestamps and provenance in one
atomic publication. The caller saves the returned manifest outside its root.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
from zipfile import BadZipFile
from zlib import error as ZlibError
from typing import Any, Callable, TYPE_CHECKING

from ._io import canonical_hash, read_json, resolve_resource, sha256_file, write_json_atomic
from .contracts import DatasetManifest, SampleRecord
from .data import verify_manifest
from .features import InputRejected

if TYPE_CHECKING:
    from .media import ExtractionConfig


REPRESENTATION_ID = "mediapipe225_rgb_v1"
_OWNER = {"schema_version": 1, "format": "slbridge_rgb_extraction_cache"}


def _cache_root(source_root: str | Path, output_dir: str | Path) -> Path:
    source = Path(source_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Extraction output must be separate from the read-only RGB dataset")
    if output.exists() and not output.is_dir():
        raise ValueError("Extraction output must be a directory")
    marker = resolve_resource(output, ".slbridge-extraction.json")
    if output.exists() and any(output.iterdir()) and not marker.is_file():
        raise ValueError("Refusing to write to a nonempty directory without a cache ownership marker")
    if marker.exists() and read_json(marker) != _OWNER:
        raise ValueError("Invalid extraction cache ownership marker")
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(marker, _OWNER)
    # Check any pre-existing links before creating or writing these directories.
    resolve_resource(output, "clips").mkdir(exist_ok=True)
    resolve_resource(output, "failures").mkdir(exist_ok=True)
    return output


def _publish_archive(path: Path, clip: Any, metadata: dict[str, Any]) -> None:
    import numpy as np
    from .features import load_landmark_archive

    if (not isinstance(clip.values, np.ndarray) or clip.values.dtype != np.float32
            or not isinstance(clip.timestamps, np.ndarray) or clip.timestamps.dtype != np.float64
            or not isinstance(clip.frame_indices, np.ndarray) or clip.frame_indices.dtype != np.int64):
        raise ValueError("Extractor must return float32 values, float64 timestamps and int64 frame indices")
    serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".clip-",
                                         suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, values=clip.values, timestamps=clip.timestamps,
                                frame_indices=clip.frame_indices, metadata=np.asarray(serialized))
            stream.flush()
            os.fsync(stream.fileno())
        # Programmer or extractor-contract errors fail here before publication.
        load_landmark_archive(temporary, expected_extractor=metadata["extractor"])
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cached_metadata(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    import numpy as np
    from .features import load_landmark_archive

    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Cache archive is not a regular file: {path}")
    try:
        values, timestamps, indices, metadata = load_landmark_archive(path, expected_extractor=expected["extractor"])
        if values.dtype != np.float32 or timestamps.dtype != np.float64 or indices.dtype != np.int64:
            return None
        for field in ("schema_version", "representation_id", "source_id", "source_hash"):
            if metadata[field] != expected[field]:
                return None
        return metadata
    except (ValueError, TypeError, KeyError, EOFError, OSError, BadZipFile, ZlibError):
        # The owned deterministic cache file can be replaced after extraction.
        return None


def _derived_record(source: SampleRecord, root: Path, resource: str, *,
                    issues: tuple[str, ...] = ()) -> SampleRecord:
    groups = dict(source.groups)
    # Retain exact source-byte identity even though each archive includes its
    # own sample provenance. The split layer treats this as an invariant link.
    groups["raw_source_hash"] = source.content_hash
    return replace(source, resource=resource,
                   content_hash=sha256_file(resolve_resource(root, resource)),
                   input_kind="landmarks_npz", representation_id=REPRESENTATION_ID,
                   source_id=source.source_id or source.sample_id,
                   derived_from=source.sample_id, groups=groups,
                   quality_status="quarantined" if issues else "ready", quality_issues=issues)


def extract_dataset(
    manifest: DatasetManifest,
    output_dir: str | Path,
    config: ExtractionConfig | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> DatasetManifest:
    """Extract clips sequentially, resume valid archives and quarantine bad video.

    Missing dependencies, tracking failures and invalid extractor outputs are
    program/environment errors and abort. Video decoding and declared quality
    rejection affect only that sample. Failed samples are retried on a later
    call; successful archives are always validated before being reused.
    """
    from .media import ExtractionConfig, HolisticExtractor, VideoDecodeError, extractor_identity

    if not isinstance(manifest, DatasetManifest) or (
        manifest.input_kind, manifest.representation_id
    ) != ("rgb_video", "rgb_video_v1"):
        raise ValueError("Extraction requires a validated rgb_video_v1 manifest")
    if config is None:
        config = ExtractionConfig()
    if not isinstance(config, ExtractionConfig):
        raise TypeError("config must be ExtractionConfig")
    # Check every indexed source, including quarantined sources, before writes.
    verify_manifest(manifest)
    extractor_spec = {"config": config.to_dict(), "identity": extractor_identity(config)}
    canonical_hash(extractor_spec)  # Require finite JSON-serializable identity.
    output = _cache_root(manifest.root, output_dir)
    started = time.monotonic()
    total = len(manifest.samples)
    stats = {"completed": 0, "total": total, "ready": 0, "cache_hits": 0,
             "rejected": 0, "errors": 0}
    records: list[SampleRecord] = []
    extractor = None

    def report() -> None:
        if progress is not None:
            progress({**stats, "elapsed_seconds": round(time.monotonic() - started, 3)})

    report()
    with ExitStack() as stack:
        for source in manifest.samples:
            source_path = resolve_resource(manifest.root, source.resource)
            # Do not trust a stale manifest or a success cache if source bytes
            # change after the initial whole-manifest verification.
            if sha256_file(source_path) != source.content_hash:
                raise ValueError(f"Content hash changed: {source.resource}")
            key = canonical_hash({"source_hash": source.content_hash,
                                  "extractor": extractor_spec, "sample_id": source.sample_id})
            metadata = {"schema_version": 1, "representation_id": REPRESENTATION_ID,
                        "extractor": extractor_spec,
                        "source_id": source.sample_id,
                        "source_hash": source.content_hash}
            if source.quality_status != "ready":
                issues = source.quality_issues
                resource = f"failures/{key}.json"
                write_json_atomic(resolve_resource(output, resource), {
                    **metadata, "status": "quarantined", "quality_issues": list(issues),
                    "source_resource": source.resource, "quality": {}, "provenance": {},
                })
                records.append(_derived_record(source, output, resource, issues=issues))
                stats["rejected"] += 1
            else:
                resource = f"clips/{key}.npz"
                archive = resolve_resource(output, resource)
                cached = _cached_metadata(archive, metadata)
                if cached is not None:
                    records.append(_derived_record(source, output, resource))
                    stats["cache_hits"] += 1
                    stats["ready"] += 1
                else:
                    if extractor is None:
                        extractor = stack.enter_context(HolisticExtractor(config))
                    try:
                        clip = extractor.extract(source_path)
                    except (VideoDecodeError, InputRejected) as error:
                        if sha256_file(source_path) != source.content_hash:
                            raise ValueError(f"Content hash changed during extraction: {source.resource}") from error
                        reason = (error.reason if isinstance(error, InputRejected)
                                  else "rgb_decode_failed:" + str(error).replace(str(source_path), source.resource))
                        issues = (reason,)
                        failed_resource = f"failures/{key}.json"
                        write_json_atomic(resolve_resource(output, failed_resource), {
                            **metadata, "status": "quarantined", "quality_issues": list(issues),
                            "source_resource": source.resource,
                            "quality": error.quality if isinstance(error, InputRejected) else {},
                            "provenance": {},
                        })
                        records.append(_derived_record(source, output, failed_resource, issues=issues))
                        stats["rejected"] += 1
                        stats["errors"] += int(isinstance(error, VideoDecodeError))
                    else:
                        if sha256_file(source_path) != source.content_hash:
                            raise ValueError(f"Content hash changed during extraction: {source.resource}")
                        _publish_archive(archive, clip, {
                            **metadata, "quality": clip.quality, "provenance": clip.provenance,
                        })
                        records.append(_derived_record(source, output, resource))
                        stats["ready"] += 1
            stats["completed"] += 1
            if stats["completed"] % 5 == 0 or stats["completed"] == total:
                report()
    derived = DatasetManifest(dataset_id=manifest.dataset_id, root=str(output),
                              input_kind="landmarks_npz", representation_id=REPRESENTATION_ID,
                              sign_language=manifest.sign_language, samples=tuple(records))
    write_json_atomic(resolve_resource(output, "summary.json"), {
        **stats, "schema_version": 1, "source_manifest_hash": manifest.fingerprint,
        "derived_manifest_hash": derived.fingerprint, "extractor": extractor_spec,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    return derived
