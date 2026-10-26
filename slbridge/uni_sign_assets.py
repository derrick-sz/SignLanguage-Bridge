"""Pinned Uni-Sign assets, integrity checks, and local bundle metadata.

This module intentionally has no PyTorch dependency. Download and compatibility
checks can therefore run before the isolated Uni-Sign environment is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


UNISIGN_REPO = "ZechengLi19/Uni-Sign"
UNISIGN_REVISION = "182913ca7a8b7dd9837d7d6f10717353f4408f80"
UNISIGN_CHECKPOINT = "csl_daily_pose_only_slt.pth"
UNISIGN_CHECKPOINT_BYTES = 1_186_925_112
UNISIGN_CHECKPOINT_SHA256 = "73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c"

UNISIGN_SOURCE_REPO = "ZechengLi19/Uni-Sign"
UNISIGN_SOURCE_COMMIT = "eed438bcb49e30405cd6ccdfcccca330c134e830"
UNISIGN_SOURCE_ZIP_SHA256 = "8022b073bb1de962a8e3f424e9858ee50465e46228bc8854b966d54107d8cc97"
UNISIGN_SOURCE_ZIP_BYTES = 8_600_007

MT5_REPO = "google/mt5-base"
MT5_REVISION = "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f"
MT5_FILES: Mapping[str, tuple[int, str | None]] = {
    "config.json": (702, None),
    "generation_config.json": (147, None),
    "pytorch_model.bin": (
        2_329_735_129,
        "180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614",
    ),
    "special_tokens_map.json": (65, None),
    "spiece.model": (4_309_802, None),
    "tokenizer_config.json": (376, None),
}
MT5_TOTAL_BYTES = sum(size for size, _ in MT5_FILES.values())

MODEL_LICENSE = "CC-BY-NC-4.0"
MT5_LICENSE = "Apache-2.0"
SOURCE_LICENSE_STATUS = "no-license-file-at-pinned-commit"
SOURCE_USAGE_NOTE = (
    "The pinned Uni-Sign source tree contains no explicit LICENSE file. "
    "Do not redistribute the upstream source in this repository or use it "
    "commercially without written clarification from the authors."
)
MODEL_USAGE_NOTE = (
    "Uni-Sign model weights are CC BY-NC 4.0: attribution is required and "
    "commercial use is not permitted by that license."
)


@dataclass(frozen=True)
class FileCheck:
    path: Path
    exists: bool
    bytes: int | None
    sha256: str | None
    ok: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "ok": self.ok,
            "reason": self.reason,
        }


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | Path, *, expected_bytes: int, expected_sha256: str | None = None) -> FileCheck:
    """Verify size and, when supplied, the exact SHA256 digest."""
    target = Path(path)
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError("expected_bytes must be a nonnegative integer")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256.lower())
    ):
        raise ValueError("expected_sha256 must be 64 hexadecimal characters")
    if not target.is_file():
        return FileCheck(target, False, None, None, False, "missing")
    size = target.stat().st_size
    digest = sha256_file(target) if expected_sha256 is not None else None
    if size != expected_bytes:
        return FileCheck(target, True, size, digest, False, f"size_mismatch expected={expected_bytes}")
    if expected_sha256 is not None and digest != expected_sha256.lower():
        return FileCheck(target, True, size, digest, False, "sha256_mismatch")
    return FileCheck(target, True, size, digest, True, None)


def verify_checkpoint(path: str | Path) -> FileCheck:
    return verify_file(path, expected_bytes=UNISIGN_CHECKPOINT_BYTES,
                       expected_sha256=UNISIGN_CHECKPOINT_SHA256)


def verify_mt5_dir(path: str | Path) -> dict[str, FileCheck]:
    root = Path(path)
    return {
        name: verify_file(root / name, expected_bytes=size, expected_sha256=digest)
        for name, (size, digest) in MT5_FILES.items()
    }


def all_checks_ok(checks: Mapping[str, FileCheck] | list[FileCheck]) -> bool:
    values = checks.values() if isinstance(checks, Mapping) else checks
    return all(check.ok for check in values)


def require_disk_space(path: str | Path, required_bytes: int, *, reserve_bytes: int = 2 * 1024**3) -> dict[str, int]:
    """Fail before a large download when the destination volume is too small."""
    if type(required_bytes) is not int or required_bytes < 0:
        raise ValueError("required_bytes must be a nonnegative integer")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    probe = Path(path).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    needed = required_bytes + reserve_bytes
    if usage.free < needed:
        raise OSError(
            f"Insufficient disk space at {probe}: free={usage.free} bytes, "
            f"required={needed} bytes (includes {reserve_bytes} byte reserve)"
        )
    return {
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
        "reserve_bytes": reserve_bytes,
        "post_download_floor_bytes": usage.free - required_bytes,
    }


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve()).replace("\\", "/")


def build_bundle_manifest(root: str | Path) -> dict[str, Any]:
    """Create the manifest consumed by the optional Uni-Sign runtime.

    ``root`` is the asset root and the writable bundle is ``root/model_bundle``.
    Paths in the manifest are relative so the whole ignored asset directory can
    be moved without editing runtime code.
    """
    root = Path(root).resolve()
    bundle_root = root / "model_bundle"
    checkpoint = root / UNISIGN_CHECKPOINT
    mt5_dir = root / "mt5-base"
    source_dir = root.parent / "uni_sign_source"
    return {
        "schema_version": 1,
        "run_id": "uni-sign-csl-daily-pose-only-182913ca",
        "display_name": "Uni-Sign Pose-only（CSL-Daily，实验）",
        "backend": "uni_sign_pose_slt",
        "task_type": "text",
        "sign_language": "csl",
        "capabilities": ["clip"],
        "checkpoint": _relative(checkpoint, bundle_root),
        "checkpoint_bytes": UNISIGN_CHECKPOINT_BYTES,
        "checkpoint_sha256": UNISIGN_CHECKPOINT_SHA256,
        "model_repo": UNISIGN_REPO,
        "model_revision": UNISIGN_REVISION,
        "mt5_dir": _relative(mt5_dir, bundle_root),
        "mt5_repo": MT5_REPO,
        "mt5_revision": MT5_REVISION,
        "mt5_model_file_sha256": MT5_FILES["pytorch_model.bin"][1],
        "source_dir": _relative(source_dir, bundle_root),
        "source_repo": UNISIGN_SOURCE_REPO,
        "source_commit": UNISIGN_SOURCE_COMMIT,
        "source_zip_sha256": UNISIGN_SOURCE_ZIP_SHA256,
        "runtime": {
            "device": "auto",
            "dtype": "auto",
            "max_frames": 256,
            "num_beams": 4,
            "max_new_tokens": 100,
            "pose_mode": "lightweight",
            "pose_backend": "onnxruntime",
            "pose_device": "auto",
            "pose_normalization": "isotropic_short_side",
            "pose_orientation": "preserve",
            "stabilize_after": 2,
            "max_video_frames": 600,
        },
        "license": {
            "model": MODEL_LICENSE,
            "model_note": MODEL_USAGE_NOTE,
            "mt5": MT5_LICENSE,
            "source_status": SOURCE_LICENSE_STATUS,
            "source_note": SOURCE_USAGE_NOTE,
        },
        "provenance": {
            "paper": "https://arxiv.org/abs/2501.15187",
            "model_card": f"https://huggingface.co/{UNISIGN_REPO}",
            "source": f"https://github.com/{UNISIGN_SOURCE_REPO}/tree/{UNISIGN_SOURCE_COMMIT}",
            "task": "CSL-Daily pose-only sign-language translation (SLT)",
            "not": "general-purpose CSL recognition or Libras classification",
        },
    }


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

