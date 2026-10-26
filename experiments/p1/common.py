"""Shared helpers for the P-1 experiment package.

Everything here is dependency-light (stdlib + optional torch for fingerprints)
so the offline evaluator and the reviewers can run without the ML environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from experiments.p1.gt_guard import assert_gt_clean

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "experiments" / "p1"
DATASETS = P1 / "datasets"
REFERENCES = P1 / "references"
CONFIGS = P1 / "configs"
RUNS = P1 / "runs"
SCHEMAS = P1 / "schemas"
PLAN_DIR = ROOT / "docs" / "experiments" / "P-1"

# Default local asset locations (overridable through run manifests / CLI).
VIDEO_DIR = Path(r"D:\Zenodo\csl-daliy")
SHARED_ARTIFACTS = ROOT / "artifacts" / "csldaily-video-test-627-20261026"
COMBINED_CSV = SHARED_ARTIFACTS / "combined-1086-paperrefs" / "comparison-both.csv"
UPSTREAM_REFS = (
    ROOT
    / "models"
    / "external"
    / "uni_sign_source"
    / "out"
    / "eval_csl_daily_pose_only"
    / "test_tmp_refs.txt"
)
UNISIGN_POSE_DIR = ROOT / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
UNISIGN_BUNDLE = ROOT / "models" / "external" / "uni_sign" / "model_bundle"
UNISIGN_CHECKPOINT = ROOT / "models" / "external" / "uni_sign" / "csl_daily_pose_only_slt.pth"
SLRT_CONFIG = ROOT / "configs" / "slrt-csl.example.json"
SLRT_CHECKPOINT = (
    ROOT / "models" / "external" / "EMNLP24_Online" / "CSL-Daily" / "ckpts" / "cslr_best.ckpt"
)
CSL_DAILY_ANNOTATIONS = Path(r"F:\model_weigth\data\csl-daily")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, guard: bool = True) -> int:
    """Append GT-free records to a JSONL file; returns number of rows written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if guard:
                assert_gt_clean(row, context=f"JSONL record for {path.name}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    return count


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, guard: bool = True) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if guard:
                assert_gt_clean(row, context=f"JSONL record for {path.name}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
            count += 1
    os.replace(tmp, path)
    return count


def git_state() -> dict[str, Any]:
    """Return repository identity. Requires git; never raises."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance must not break the run
            return ""

    status = run("status", "--porcelain")
    return {
        "repository": "SignLanguage-Bridge",
        "git_sha": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty_tree": bool(status),
        "dirty_path_count": len([line for line in status.splitlines() if line.strip()]),
    }


def environment_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_vram_mb"] = round(props.total_memory / 1024 / 1024)
    except Exception as error:  # noqa: BLE001 - optional dependency on CPU-only envs
        info["torch_error"] = f"{type(error).__name__}: {error}"
    return info


def host_alias() -> str:
    return platform.node() or "unknown-host"


class RunDirectory:
    """Create and populate one immutable P-1 run directory.

    Layout (see docs/02 数据与指标规范.md):
        runs/<run_id>/manifest.json
        runs/<run_id>/model_outputs.jsonl
        runs/<run_id>/runtime_metrics.jsonl
        runs/<run_id>/metrics.json
        runs/<run_id>/report.md
    """

    def __init__(self, run_id: str, *, experiment_id: str, create: bool = True) -> None:
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError(f"invalid run_id: {run_id!r}")
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.path = RUNS / run_id
        if create:
            self.path.mkdir(parents=True, exist_ok=False)

    # -- manifest -----------------------------------------------------
    def write_manifest(self, **fields: Any) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "run_status": fields.pop("run_status", "RUNNING"),
            "created_at": utc_now(),
            "purpose": self.experiment_id,
            "operator": fields.pop("operator", "p1-agent"),
            "code": {"repository": "SignLanguage-Bridge", **git_state()},
            "host": {"host_alias": host_alias()},
            "privacy_and_licensing": {
                "rights_checked": False,
                "no_raw_media_export": True,
                "no_credentials_in_logs": True,
            },
        }
        manifest.update(fields)
        write_json(self.path / "manifest.json", manifest)
        return manifest

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        manifest = read_json(self.path / "manifest.json")
        manifest.update(fields)
        write_json(self.path / "manifest.json", manifest)
        return manifest

    def record(self, name: str) -> Path:
        return self.path / name

    def write_outputs(self, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.record("model_outputs.jsonl")
        write_jsonl(path, rows)
        return path

    def write_runtime(self, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.record("runtime_metrics.jsonl")
        write_jsonl(path, rows, guard=False)
        return path

    def write_metrics(self, payload: dict[str, Any]) -> Path:
        return write_json(self.record("metrics.json"), payload)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)
