"""task-24 performance validation -- production SLRT frame cache before/after.

Runs the same pilot samples through the *production* ``SLRTBackend`` twice:

  * ``cached``   -- ``frame_cache_enabled=True`` (new default, production path);
  * ``batched``  -- ``frame_cache_enabled=False`` (previous batched prepare path);

and reports e2e wall P50/P95, prepare totals, cache hit/miss counters, Gloss
equality (must be 100%) and corpus WER (EVAL-ONLY GT join, must be unchanged).
No library code is patched here: this exercises the merged implementation only.

Usage::

    .conda-env\\python.exe experiments/p1/e04/validate_slrt_frame_cache.py run \
        --run-dir experiments/p1/e04/opt-runs/slrt-frame-cache-validate-<stamp> --samples 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1.e04 import perf_common as common  # noqa: E402
from experiments.p1.e04.opt_slrt_preprocess import corpus_wer, read_gt  # noqa: E402

RETRYABLE = ("out of memory", "cuda error", "memory allocation failure", "acceleratorerror")
RETRY_SLEEP_SECONDS = 20.0


def run_arm(*, arm: str, cache_enabled: bool, samples: list[dict[str, Any]],
            gt: dict[str, dict[str, Any]], run_dir: Path, run_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend

    config = SLRTConfig.from_file(common.SLRT_CONFIG_HRNET_NPZ)
    backend = None
    load_seconds = None
    for attempt in range(3):
        try:
            started = perf_counter()
            backend = SLRTBackend(config, frame_cache_enabled=cache_enabled)
            load_seconds = perf_counter() - started
            break
        except BaseException as error:  # noqa: BLE001
            if attempt < 2 and any(marker in str(error).lower() for marker in RETRYABLE):
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            raise
    rows: dict[str, dict[str, Any]] = {}
    failures = 0
    try:
        for sample in samples:
            attempt = 0
            while True:
                attempt += 1
                started = perf_counter()
                try:
                    result = backend.recognize_video(sample["trimmed"], pose_npz=sample["npz"])
                    wall = perf_counter() - started
                    timings = result.get("timings") or {}
                    def total(name: str) -> float | None:
                        block = timings.get(name) or {}
                        return block.get("total")
                    cache_info = backend.runtime.frame_cache_info
                    rows[sample["sample_id"]] = {
                        "schema_version": 1, "run_id": run_id, "arm": arm,
                        "sample_id": sample["sample_id"],
                        "signer_id": sample["signer_id"],
                        "status": result.get("status"),
                        "gloss_sequence": result.get("gloss_sequence") or [],
                        "reference_gloss": gt.get(sample["sample_id"], {}).get("gloss", []),
                        "e2e_wall_ms": round(wall * 1000, 3),
                        "video_decode_ms": round(float(result["video_decode_ms"]), 3)
                        if isinstance(result.get("video_decode_ms"), (int, float)) else None,
                        "windows_processed": result.get("windows_processed"),
                        "frames_received": result.get("frames_received"),
                        "sum_preprocess_ms": round(float(total("preprocess_ms") or 0.0), 3),
                        "sum_inference_ms": round(float(total("inference_ms") or 0.0), 3),
                        "sum_window_total_ms": round(float(total("window_total_ms") or 0.0), 3),
                        "cache": cache_info,
                        "attempts": attempt,
                    }
                    break
                except BaseException as error:  # noqa: BLE001
                    if attempt <= 2 and any(marker in str(error).lower() for marker in RETRYABLE):
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    failures += 1
                    rows[sample["sample_id"]] = {
                        "schema_version": 1, "run_id": run_id, "arm": arm,
                        "sample_id": sample["sample_id"], "signer_id": sample["signer_id"],
                        "status": "failed", "error": f"{type(error).__name__}: {error}",
                        "attempts": attempt,
                    }
                    break
    finally:
        if backend is not None:
            backend.close()
    ok = {key: row for key, row in rows.items() if row.get("status") == "final" and not row.get("error")}
    def stats(key: str) -> dict[str, Any]:
        values = [row.get(key) for row in ok.values() if isinstance(row.get(key), (int, float))]
        return {"p50": common.percentile(values, 50), "p95": common.percentile(values, 95),
                "mean": round(statistics.fmean(values), 2) if values else None, "count": len(values)}
    summary = {
        "arm": arm,
        "frame_cache_enabled": cache_enabled,
        "load_seconds": round(load_seconds or 0.0, 2),
        "attempted": len(samples), "successful": len(ok), "failed": failures,
        "wer": corpus_wer([(gt.get(key, {}).get("gloss", []), row["gloss_sequence"])
                           for key, row in ok.items()]),
        "e2e_wall_ms": stats("e2e_wall_ms"),
        "sum_preprocess_ms": stats("sum_preprocess_ms"),
        "sum_inference_ms": stats("sum_inference_ms"),
        "sum_window_total_ms": stats("sum_window_total_ms"),
        "windows_processed": stats("windows_processed"),
        "cache_final": (list(ok.values())[-1].get("cache") if ok else None),
    }
    common.write_jsonl(run_dir / f"predictions-{arm}.jsonl", rows.values())
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-foreign-vram-mb", type=float, default=4500.0)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.run != "run":
        parser.error("only `run` is supported")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"slrt-frame-cache-validate-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    baseline = common.nvidia_smi()
    if baseline is None:
        raise SystemExit("nvidia-smi unavailable")
    device = baseline["devices"][0]
    if device["memory_used_mb"] > args.max_foreign_vram_mb and not args.allow_shared_gpu:
        raise SystemExit(f"GPU busy: {device['memory_used_mb']:.0f} MB used")
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    gt = read_gt()
    manifest = common.base_manifest(run_id, "E04-validate-slrt-frame-cache", purpose=(
        "production SLRT frame-cache validation: cached vs batched prepare path, "
        "Gloss/WER must be identical, e2e improvement expected ~16-18%"))
    manifest.update({
        "model_assets": common.model_assets_manifest(),
        "environment": common.environment_fingerprint(),
        "gpu_baseline": baseline,
        "samples": [sample["sample_id"] for sample in samples],
        "signers": sorted({sample["signer_id"] for sample in samples}),
        "gt_usage": "EVAL-ONLY gt_1086 gt_gloss joined after inference for corpus WER",
        "commands": [common.command_line()],
    })
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "baseline", "gpu": device, "samples": len(samples)}, ensure_ascii=False), flush=True)

    cached_rows, cached_summary = run_arm(arm="cached", cache_enabled=True, samples=samples,
                                          gt=gt, run_dir=run_dir, run_id=run_id)
    print(json.dumps({"event": "arm_done", **{k: cached_summary[k] for k in
                                              ("arm", "successful", "failed", "wer", "e2e_wall_ms")},
                      "cache": cached_summary["cache_final"]}, ensure_ascii=False), flush=True)
    batched_rows, batched_summary = run_arm(arm="batched", cache_enabled=False, samples=samples,
                                            gt=gt, run_dir=run_dir, run_id=run_id)
    print(json.dumps({"event": "arm_done", **{k: batched_summary[k] for k in
                                              ("arm", "successful", "failed", "wer", "e2e_wall_ms")}},
                     ensure_ascii=False), flush=True)

    shared = sorted(set(cached_rows) & set(batched_rows))
    identical = sum(
        (cached_rows[key].get("gloss_sequence") or []) == (batched_rows[key].get("gloss_sequence") or [])
        for key in shared
        if cached_rows[key].get("status") == "final" and batched_rows[key].get("status") == "final")
    speedup = (batched_summary["e2e_wall_ms"]["p50"] or 0) / (cached_summary["e2e_wall_ms"]["p50"] or 1)
    metrics = {
        "schema_version": 1, "run_id": run_id, "created_at": common.utc_now(),
        "samples": len(samples),
        "cached": cached_summary,
        "batched": batched_summary,
        "comparison": {
            "shared_samples": len(shared),
            "identical_gloss_sequences": identical,
            "gloss_equality": f"{identical}/{len(shared)}",
            "wer_cached": cached_summary["wer"]["wer"],
            "wer_batched": batched_summary["wer"]["wer"],
            "wer_delta": round((cached_summary["wer"]["wer"] or 0)
                               - (batched_summary["wer"]["wer"] or 0), 5),
            "e2e_p50_cached_ms": cached_summary["e2e_wall_ms"]["p50"],
            "e2e_p50_batched_ms": batched_summary["e2e_wall_ms"]["p50"],
            "e2e_p50_speedup": round(speedup, 3),
            "e2e_p50_reduction_pct": round((1 - 1 / speedup) * 100, 2) if speedup else None,
            "e2e_p95_cached_ms": cached_summary["e2e_wall_ms"]["p95"],
            "e2e_p95_batched_ms": batched_summary["e2e_wall_ms"]["p95"],
            "sum_preprocess_ms_p50_cached": cached_summary["sum_preprocess_ms"]["p50"],
            "sum_preprocess_ms_p50_batched": batched_summary["sum_preprocess_ms"]["p50"],
        },
        "gpu_baseline": baseline,
    }
    common.write_json(run_dir / "metrics.json", metrics)
    manifest.update({"run_status": "PASS" if cached_summary["failed"] == 0
                     and batched_summary["failed"] == 0 else "PARTIAL",
                     "finished_at": common.utc_now(),
                     "output_files": sorted(path.name for path in run_dir.iterdir() if path.is_file())})
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "done", "run_dir": str(run_dir),
                      "comparison": metrics["comparison"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


