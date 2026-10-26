"""Shared helpers for the P-1 E04 performance probe and E05 prefix probe.

Dependency-light on purpose: the E04/E05 scripts run in the isolated
``.conda-env`` interpreter (Torch/ORT/RTMLib), so provenance and bookkeeping
must not add another runtime dependency.  Nothing here starts CUDA; the GPU
touch happens only inside the model runners after the orchestrator confirms the
GPU is free.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

P1 = ROOT / "experiments" / "p1"
PILOT_CSV = P1 / "datasets" / "pilot_50.csv"
RUNS_DIR = P1 / "runs"

SHARED_ARTIFACTS = ROOT / "artifacts" / "csldaily-video-test-627-20261026"
TRIMMED_DIR = SHARED_ARTIFACTS / "trimmed"
SLRT_NPZ_DIR = SHARED_ARTIFACTS / "pose"
UNISIGN_POSE_DIR = ROOT / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
UNISIGN_BUNDLE = ROOT / "models" / "external" / "uni_sign" / "model_bundle"
UNISIGN_UPSTREAM_PRES = (
    ROOT
    / "models"
    / "external"
    / "uni_sign_source"
    / "out"
    / "eval_csl_daily_pose_only"
    / "test_tmp_pres.txt"
)
SLRT_CONFIG_HRNET_NPZ = ROOT / "configs" / "slrt-csl-hrnet-npz.example.json"
SLRT_CONFIG_RTMPOSE = ROOT / "configs" / "slrt-csl.example.json"

# Segment keys that every perf_trace.jsonl row must carry.  ``None`` means
# "not measured on this path" -- never 0.0 for a missing stage.
SEGMENT_KEYS = (
    "video_decode_ms",
    "pose_ms",
    "forward_ms",
    "text_decode_ms",
    "queue_ms",
    "e2e_ms",
)

MODEL_IDS = ("unisign_pose", "unisign_video", "slrt_hrnet_npz")

MODEL_META = {
    "unisign_pose": {
        "display": "Uni-Sign pose path (official released CSL-Daily pkl -> mT5)",
        "task": "CSLT (text)",
        "input": "official pose-format pkl (start/end cropped, deterministic linspace<=256)",
        "device_path": "TorchUniSignRuntime.infer (CUDA sync)",
    },
    "unisign_video": {
        "display": "Uni-Sign video path (trimmed mp4 -> RTMPose wholebody -> mT5)",
        "task": "CSLT (text)",
        "input": "trimmed/<sample>.mp4 (already cropped sentence clip)",
        "device_path": "UniSignBackend.recognize_file (CUDA sync inside runtime)",
    },
    "slrt_hrnet_npz": {
        "display": "SLRT Online CSL-Daily (trimmed mp4 + HRNet NPZ -> S3D windows -> vote)",
        "task": "CSLR (gloss)",
        "input": "trimmed/<sample>.mp4 + pose/<sample>.npz (official keypoints aligned)",
        "device_path": "SLRTBackend.recognize_video (CUDA sync in SLRTRuntime.predict)",
    },
}


# --------------------------------------------------------------------------
# provenance and small IO
# --------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id(experiment: str) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment}-{stamp}"


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    return count


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    ordered = sorted(clean)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"count": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "max": max(clean),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# GPU / device sampling (read-only; whole-device values include other processes)
# --------------------------------------------------------------------------
def nvidia_smi(timeout: float = 5.0) -> dict[str, Any] | None:
    """One whole-device snapshot.  ``None`` when nvidia-smi is unusable."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    devices = []
    for line in proc.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            devices.append({
                "index": int(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mb": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "memory_free_mb": float(parts[5]),
                "utilization_gpu_pct": float(parts[6]),
            })
        except ValueError:
            continue
    return {"sampled_at": utc_now(), "devices": devices} if devices else None


class DeviceSampler:
    """Background nvidia-smi sampler; records what other processes do to VRAM."""

    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = max(0.2, float(interval_seconds))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "DeviceSampler":
        self._thread = threading.Thread(target=self._loop, name="e04-device-sampler", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        started = time.perf_counter()
        while not self._stop.is_set():
            snapshot = nvidia_smi()
            if snapshot is not None:
                for device in snapshot["devices"]:
                    self.samples.append({
                        "elapsed_seconds": time.perf_counter() - started,
                        **device,
                    })
            self._stop.wait(self.interval_seconds)

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return self.samples

    def peak(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        used = [sample["memory_used_mb"] for sample in self.samples]
        util = [sample["utilization_gpu_pct"] for sample in self.samples]
        return {
            "device_samples": len(self.samples),
            "peak_memory_used_mb": max(used),
            "min_memory_used_mb": min(used),
            "mean_utilization_gpu_pct": statistics.fmean(util),
            "max_utilization_gpu_pct": max(util),
            "scope": "whole_device_including_other_processes_and_other_models",
        }


# --------------------------------------------------------------------------
# dataset helpers
# --------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_pilot_samples(count: int = 5, *, require_assets: bool = True) -> list[dict[str, Any]]:
    """Deterministically pick ``count`` pilot samples across distinct signers.

    The pilot table is sorted by signer, then sample id; the first round takes
    one sample per signer, later rounds take the next sample per signer.  This
    keeps the tiny benchmark set signer-diverse without random sampling.
    """
    rows = read_csv_rows(PILOT_CSV)
    by_signer: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_signer.setdefault(row["signer_id"], []).append(dict(row))
    for signer in by_signer:
        by_signer[signer].sort(key=lambda item: item["sample_id"])
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < count and any(depth < len(items) for items in by_signer.values()):
        for signer in sorted(by_signer):
            if len(picked) >= count:
                break
            items = by_signer[signer]
            if depth < len(items):
                row = items[depth]
                sample = build_sample(row)
                if require_assets:
                    missing = [key for key in ("trimmed", "npz", "pkl") if not Path(sample[key]).is_file()]
                    if missing:
                        raise FileNotFoundError(f"{sample['sample_id']} missing assets: {missing}")
                picked.append(sample)
        depth += 1
    return picked


def build_sample(row: dict[str, str]) -> dict[str, Any]:
    sample_id = row["sample_id"]
    return {
        "sample_id": sample_id,
        "signer_id": row["signer_id"],
        "frames": int(row["frames"]),
        "cohort": row.get("cohort") or None,
        "trimmed": str(TRIMMED_DIR / f"{sample_id}.mp4"),
        "npz": str(SLRT_NPZ_DIR / f"{sample_id}.npz"),
        "pkl": str(UNISIGN_POSE_DIR / f"{sample_id}.pkl"),
        "fps": None,
        "video_seconds": None,
    }


def attach_fps(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill ``fps``/``video_seconds`` from the trimmed mp4 (CPU decode only).

    The pilot table's ``fps_measured`` column is empty, so the container value
    is recorded explicitly.  RTF denominators come from here, never from a
    hard-coded FPS.
    """
    import cv2

    for sample in samples:
        if sample.get("fps"):
            continue
        capture = cv2.VideoCapture(str(sample["trimmed"]))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
            count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0.0
        finally:
            capture.release()
        if not math.isfinite(fps) or fps <= 0:
            fps = None
        sample["fps"] = fps
        sample["video_seconds"] = (
            sample["frames"] / fps if fps else None
        )
        sample["container_frame_count"] = int(count) if math.isfinite(count) and count > 0 else None
    return samples


def load_pose_pkl(path: Path | str) -> tuple[Any, Any, dict[str, Any]]:
    """Port of ``evaluate_unisign_csldaily_pose.load_pose`` (upstream contract).

    The released pkl stores ``start``/``end`` sentence-crop fields; ignoring them
    shifts the frame range (documented pitfall).  Returns the *cropped* arrays.
    """
    import numpy as np
    import pickle

    with Path(path).open("rb") as stream:
        pose = pickle.load(stream)
    keypoints = np.asarray(pose["keypoints"], dtype=np.float32)
    scores = np.asarray(pose["scores"], dtype=np.float32)
    if keypoints.ndim == 4:
        keypoints = keypoints[:, 0]
    if scores.ndim == 3:
        scores = scores[:, 0]
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError(f"Unsupported keypoint shape: {keypoints.shape}")
    info = {"frames_total": int(keypoints.shape[0])}
    if "start" in pose and "end" in pose:
        start, end = int(pose["start"]), int(pose["end"])
        if end <= start:
            raise ValueError(f"Invalid trim range in {path}: {start}..{end}")
        keypoints = keypoints[start:end]
        scores = scores[start:end]
        info.update({"start": start, "end": end})
    info["frames_used"] = int(keypoints.shape[0])
    return keypoints, scores, info


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text)).strip()


def char_edit_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio 0..1 (no GT is involved)."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def parse_upstream_predictions(path: Path = UNISIGN_UPSTREAM_PRES) -> dict[str, str]:
    """Read the authors' released ``test_tmp_pres.txt`` (character-separated)."""
    predictions: dict[str, str] = {}
    if not Path(path).is_file():
        return predictions
    marker = ", prediction: "
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or marker not in line:
            continue
        head, prediction = line.split(marker, 1)
        sample = head.split("sample: ", 1)[-1].strip()
        predictions[sample] = normalize_text(prediction)
    return predictions


# --------------------------------------------------------------------------
# run directory + manifest
# --------------------------------------------------------------------------
def repo_state() -> dict[str, Any]:
    from experiments.p1.common import git_state

    return git_state()


def environment_fingerprint() -> dict[str, Any]:
    """Torch/CUDA facts; call only inside a model process (loads Torch)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host_alias": platform.node() or "unknown-host",
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_vram_mb": round(props.total_memory / 1024**2, 1),
                "gpu_capability": f"{props.major}.{props.minor}",
            })
    except Exception as error:  # noqa: BLE001 - fingerprint must not break a run
        info["torch_error"] = f"{type(error).__name__}: {error}"
    return info


def base_manifest(run_id: str, experiment: str, *, purpose: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": experiment,
        "run_status": "RUNNING",
        "created_at": utc_now(),
        "purpose": purpose,
        "code": repo_state(),
        "host": {
            "host_alias": platform.node() or "unknown-host",
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_total_gb": round(_total_memory_gb(), 2),
        },
        "gpu_baseline": nvidia_smi(),
        "privacy": {
            "no_gt_in_perf_artifacts": True,
            "no_raw_media_export": True,
            "no_credentials_in_logs": True,
        },
    }


def _total_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:  # noqa: BLE001
        return 0.0


def memory_snapshot() -> dict[str, Any]:
    """System RAM snapshot; two-process phases are gated on this (32 GB host)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / 1024**3, 2),
            "available_gb": round(vm.available / 1024**3, 2),
            "used_percent": vm.percent,
        }
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def update_manifest(run_dir: Path, **fields: Any) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    manifest = read_json(path) if path.is_file() else {}
    manifest.update(fields)
    write_json(path, manifest)
    return manifest


def model_assets_manifest() -> dict[str, Any]:
    """Checkpoint / config identities needed by the report."""
    assets: dict[str, Any] = {}
    bundle_manifest = UNISIGN_BUNDLE / "manifest.json"
    if bundle_manifest.is_file():
        payload = read_json(bundle_manifest)
        checkpoint = (UNISIGN_BUNDLE / payload.get("checkpoint", "")).resolve()
        assets["unisign"] = {
            "bundle": str(UNISIGN_BUNDLE),
            "bundle_manifest_sha256": sha256_file(bundle_manifest),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": payload.get("checkpoint_bytes"),
            "checkpoint_sha256_declared": payload.get("checkpoint_sha256"),
            "source_commit": payload.get("source_commit"),
            "model_revision": payload.get("model_revision"),
            "max_frames": payload.get("runtime", {}).get("max_frames"),
            "num_beams": payload.get("runtime", {}).get("num_beams"),
            "license": payload.get("license", {}).get("model"),
        }
    if SLRT_CONFIG_HRNET_NPZ.is_file():
        assets["slrt"] = {
            "config": str(SLRT_CONFIG_HRNET_NPZ),
            "config_sha256": sha256_file(SLRT_CONFIG_HRNET_NPZ),
            "checkpoint": str(
                (ROOT / "models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt")
            ),
            "checkpoint_bytes": _file_size(ROOT / "models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt"),
            "checkpoint_sha256_declared": "e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d",
            "vocabulary": str(ROOT / "models/external/EMNLP24_Online/CSL-Daily/meta/csl_iso_with_blank.vocab"),
            "window_size": 16,
            "stride": 1,
            "vote_size": 7,
        }
    return assets


def _file_size(path: Path) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def run_name_from_trace(row: dict[str, Any]) -> str:
    return f"{row.get('phase')}/{row.get('model')}"


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]
