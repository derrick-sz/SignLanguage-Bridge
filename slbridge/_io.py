"""Deterministic JSON, checksums and safe local resource references."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import sys
from typing import Any


def configure_utf8_streams() -> None:
    """Chinese/Portuguese CLI JSON must be representable on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A repeat import produces exactly the same bytes and does not touch an existing file.
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_resource(root: str | Path, resource: str) -> Path:
    """Resolve a portable relative reference; forbid escape, including symlinks."""
    if not isinstance(resource, str) or not resource or "\\" in resource or ":" in resource:
        raise ValueError(f"Invalid relative resource: {resource!r}")
    relative = PurePosixPath(resource)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != resource:
        raise ValueError(f"Invalid relative resource: {resource!r}")
    root_path = Path(root).resolve()
    result = root_path.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root_path) or result == root_path:
        raise ValueError(f"Resource escapes root: {resource}")
    return result
