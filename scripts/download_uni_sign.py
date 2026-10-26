"""Download and verify the pinned Uni-Sign assets on a chosen volume.

The script is intentionally independent of the PyTorch environment. It uses the
official Hugging Face downloader (resume-capable) and writes temporary files
under the same ignored asset root before atomically promoting verified files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slbridge.uni_sign_assets import (  # noqa: E402
    MT5_FILES,
    MT5_REPO,
    MT5_REVISION,
    MT5_TOTAL_BYTES,
    MODEL_LICENSE,
    MODEL_USAGE_NOTE,
    SOURCE_LICENSE_STATUS,
    SOURCE_USAGE_NOTE,
    UNISIGN_CHECKPOINT,
    UNISIGN_CHECKPOINT_BYTES,
    UNISIGN_CHECKPOINT_SHA256,
    UNISIGN_REPO,
    UNISIGN_REVISION,
    UNISIGN_SOURCE_COMMIT,
    UNISIGN_SOURCE_ZIP_BYTES,
    UNISIGN_SOURCE_ZIP_SHA256,
    all_checks_ok,
    build_bundle_manifest,
    require_disk_space,
    sha256_file,
    verify_checkpoint,
    verify_file,
    verify_mt5_dir,
    write_json_atomic,
)


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    target_resolved = target.resolve()
    members = []
    roots = set()
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if not parts:
            continue
        roots.add(parts[0])
        members.append((member, parts[1:]))
    if len(roots) != 1:
        raise ValueError("Source archive must contain exactly one top-level directory")
    for member, relative_parts in members:
        if not relative_parts:
            continue
        destination = target_resolved.joinpath(*relative_parts).resolve()
        if destination != target_resolved and target_resolved not in destination.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def _source_ready(target: Path) -> bool:
    marker = target / "UPSTREAM_COMMIT.txt"
    return (marker.is_file() and (target / "models.py").is_file()
            and UNISIGN_SOURCE_COMMIT in marker.read_text(encoding="utf-8"))


def _download_source(root: Path, *, force: bool) -> dict:
    target = root.parent / "uni_sign_source"
    if _source_ready(target) and not force:
        return {"status": "cached", "path": str(target), "commit": UNISIGN_SOURCE_COMMIT}
    if target.exists() and not _source_ready(target):
        raise RuntimeError(f"Existing source directory is not the pinned source; refusing to overwrite: {target}")
    downloads = root / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"Uni-Sign-{UNISIGN_SOURCE_COMMIT}.zip"
    url = f"https://codeload.github.com/{UNISIGN_REPO}/zip/{UNISIGN_SOURCE_COMMIT}"
    if force or not archive.is_file():
        partial = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading source {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(partial, archive)
    check = verify_file(archive, expected_bytes=UNISIGN_SOURCE_ZIP_BYTES,
                        expected_sha256=UNISIGN_SOURCE_ZIP_SHA256)
    if not check.ok:
        raise RuntimeError(f"Source archive failed verification: {check.reason}")
    if target.exists():
        raise RuntimeError(f"Refusing to overlay an existing source directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        _safe_extract(zipped, target)
    (target / "UPSTREAM_COMMIT.txt").write_text(
        f"https://github.com/{UNISIGN_REPO}\n{UNISIGN_SOURCE_COMMIT}\n"
        f"zip_sha256={UNISIGN_SOURCE_ZIP_SHA256}\n", encoding="utf-8")
    return {"status": "downloaded", "path": str(target), "commit": UNISIGN_SOURCE_COMMIT,
            "zip_sha256": UNISIGN_SOURCE_ZIP_SHA256}


def _hub_download(repo_id: str, filename: str, revision: str, *,
                  local_dir: Path, cache_dir: Path, force: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub (or transformers) before downloading") from error
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                           local_dir=str(local_dir), cache_dir=str(cache_dir),
                           force_download=force)
    return Path(path)


def _download_checkpoint(root: Path, cache_dir: Path, *, force: bool) -> dict:
    final = root / UNISIGN_CHECKPOINT
    current = verify_checkpoint(final)
    if current.ok and not force:
        return {"status": "cached", **current.to_dict()}
    staging_dir = root / ".downloads/checkpoint"
    staging = _hub_download(UNISIGN_REPO, UNISIGN_CHECKPOINT, UNISIGN_REVISION,
                            local_dir=staging_dir, cache_dir=cache_dir, force=force)
    check = verify_file(staging, expected_bytes=UNISIGN_CHECKPOINT_BYTES,
                        expected_sha256=UNISIGN_CHECKPOINT_SHA256)
    if not check.ok:
        raise RuntimeError(f"Downloaded checkpoint failed verification: {check.reason}")
    root.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    final_check = verify_checkpoint(final)
    if not final_check.ok:
        raise RuntimeError(f"Promoted checkpoint failed verification: {final_check.reason}")
    return {"status": "downloaded", **final_check.to_dict()}


def _download_mt5(root: Path, cache_dir: Path, *, force: bool) -> dict:
    final_dir = root / "mt5-base"
    current = verify_mt5_dir(final_dir)
    if all_checks_ok(current) and not force:
        return {"status": "cached", "path": str(final_dir),
                "files": {name: check.to_dict() for name, check in current.items()}}
    staging_dir = root / ".downloads/mt5-base"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in MT5_FILES.items():
        source = final_dir / name
        destination = staging_dir / name
        if not force and verify_file(source, expected_bytes=size, expected_sha256=digest).ok and not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        _hub_download(MT5_REPO, name, MT5_REVISION, local_dir=staging_dir,
                      cache_dir=cache_dir, force=force)
    staged = verify_mt5_dir(staging_dir)
    if not all_checks_ok(staged):
        failed = {name: check.reason for name, check in staged.items() if not check.ok}
        raise RuntimeError(f"Downloaded mT5 files failed verification: {failed}")
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in MT5_FILES:
        os.replace(staging_dir / name, final_dir / name)
    final = verify_mt5_dir(final_dir)
    if not all_checks_ok(final):
        failed = {name: check.reason for name, check in final.items() if not check.ok}
        raise RuntimeError(f"Promoted mT5 files failed verification: {failed}")
    return {"status": "downloaded", "path": str(final_dir),
            "files": {name: check.to_dict() for name, check in final.items()}}


def _required_download_bytes(root: Path, *, skip_source: bool, skip_mt5: bool, force: bool) -> int:
    required = 0
    if not verify_checkpoint(root / UNISIGN_CHECKPOINT).ok or force:
        required += UNISIGN_CHECKPOINT_BYTES
    if not skip_mt5:
        mt5_current = verify_mt5_dir(root / "mt5-base")
        if force:
            required += MT5_TOTAL_BYTES
        else:
            required += sum(size for name, (size, _) in MT5_FILES.items()
                            if not mt5_current[name].ok)
    if not skip_source and (not _source_ready(root.parent / "uni_sign_source") or force):
        # The archive and extracted source coexist briefly.
        required += UNISIGN_SOURCE_ZIP_BYTES * 2
    return required


def _summary(root: Path, *, weight: dict | None = None, mt5: dict | None = None,
             source: dict | None = None) -> dict:
    weight_check = verify_checkpoint(root / UNISIGN_CHECKPOINT)
    mt5_checks = verify_mt5_dir(root / "mt5-base")
    source_dir = root.parent / "uni_sign_source"
    return {
        "asset_root": str(root),
        "checkpoint": weight_check.to_dict(),
        "mt5": {name: check.to_dict() for name, check in mt5_checks.items()},
        "mt5_total_bytes": MT5_TOTAL_BYTES,
        "source": {"path": str(source_dir), "ready": _source_ready(source_dir),
                   "commit": UNISIGN_SOURCE_COMMIT},
        "downloads": {"checkpoint": weight, "mt5": mt5, "source": source},
        "license": {
            "model": MODEL_LICENSE,
            "model_note": MODEL_USAGE_NOTE,
            "source_status": SOURCE_LICENSE_STATUS,
            "source_note": SOURCE_USAGE_NOTE,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download pinned Uni-Sign weights and mT5-base to a chosen volume")
    parser.add_argument("--root", type=Path, default=ROOT / "models/external/uni_sign")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--proxy", help="Explicit HTTP(S) proxy, e.g. http://127.0.0.1:7897")
    parser.add_argument("--reserve-gb", type=float, default=2.0,
                        help="Free-space floor retained after the downloads")
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--skip-mt5", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download weights; valid source is reused")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cache_dir = (args.cache_dir or root / ".cache/huggingface").resolve()
    if not 0 <= args.reserve_gb <= 100:
        parser.error("--reserve-gb must be between 0 and 100")
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "0")

    required = _required_download_bytes(root, skip_source=args.skip_source,
                                        skip_mt5=args.skip_mt5, force=args.force)
    try:
        disk = require_disk_space(root, required, reserve_bytes=int(args.reserve_gb * 1024**3))
    except OSError as error:
        print(json.dumps({"error": str(error), "asset_root": str(root)}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps(_summary(root), ensure_ascii=False, indent=2))
        return 0 if verify_checkpoint(root / UNISIGN_CHECKPOINT).ok and (
            args.skip_mt5 or all_checks_ok(verify_mt5_dir(root / "mt5-base"))) else 1

    root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = None if args.skip_source else _download_source(root, force=args.force)
        weight = _download_checkpoint(root, cache_dir, force=args.force)
        mt5 = None if args.skip_mt5 else _download_mt5(root, cache_dir, force=args.force)
        bundle_manifest = build_bundle_manifest(root)
        write_json_atomic(root / "model_bundle/manifest.json", bundle_manifest)
        summary = _summary(root, weight=weight, mt5=mt5, source=source)
        summary.update({
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "disk": disk,
            "bundle_manifest": str(root / "model_bundle/manifest.json"),
            "model_revision": UNISIGN_REVISION,
            "mt5_revision": MT5_REVISION,
        })
        write_json_atomic(root / "download_manifest.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        print(json.dumps({"error": str(error), "asset_root": str(root)}, ensure_ascii=False),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


