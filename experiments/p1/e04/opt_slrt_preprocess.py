"""task-12 part 2 -- SLRT per-window preprocessing optimisation (no semantic change).

Baseline (measured in E04): ``SLRTBackend.recognize_video`` spends several seconds
per sentence outside the S3D forward, in the per-window preprocessing path
(``slrt/preprocess.py::prepare_inputs`` builds 16x RGB crops + 63-point Gaussian
heatmaps for every window).  Windows have stride 1, so **every frame is prepared
~16 times** (once per window it belongs to).

This script implements an experimentally patched ``prepare_inputs`` that:

  * computes each frame's ``(rgb, heatmap)`` pair **once** by calling the
    *original unmodified* function on a single frame (T=1), then concatenates;
  * caches the pair keyed by exact frame/keypoint bytes (content hash), capped at
    128 frames (FIFO clear) so VRAM stays bounded (~0.5 GB at 128 entries);
  * therefore changes no arithmetic: the values come from the same original code
    path on the same inputs.  Equivalence is measured explicitly (max abs diff).

Evidence produced (all in the run directory):
  * ``stage-profile.json``  -- per-stage wall (decode / frontend.extract /
    canonical_frame / prepare_inputs / model inference) for baseline;
  * ``equivalence.json``    -- original vs cached tensors on real windows
    (max abs diff, torch.equal) on cuda:0;
  * ``predictions-<arm>.jsonl`` -- per-sample gloss + wall + stage sums;
  * ``metrics.json``        -- corpus WER (EVAL-ONLY GT gloss, joined *after*
    inference), P50/P95 e2e, stage totals, identical-prediction counts, speedup.

Production defaults are NOT changed: the patch only lives in this benchmark
process.  ``git status`` shows no modification under ``slbridge/``.

Usage::

    .conda-env\\python.exe experiments/p1/e04/opt_slrt_preprocess.py run \
        --run-dir experiments/p1/e04/opt-runs/slrt-preprocess-<stamp> --samples 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1.e04 import perf_common as common  # noqa: E402

GT_CSV = common.P1 / "references" / "gt_1086.csv"
CACHE_LIMIT = 128
RETRYABLE = ("out of memory", "cuda error", "memory allocation failure", "acceleratorerror")
RETRY_SLEEP_SECONDS = 20.0


# ---------------------------------------------------------------------------
# corpus WER (token Levenshtein, same definition as slbridge.slrt.cli._wer aggregate)
# ---------------------------------------------------------------------------
def token_edit(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    table = [[(j, 0, 0, 0) for j in range(len(hypothesis) + 1)]]
    for i, token in enumerate(reference, 1):
        row = [(i, 0, 0, i)]
        for j, hyp in enumerate(hypothesis, 1):
            if token == hyp:
                row.append(table[i - 1][j - 1])
            else:
                diagonal = table[i - 1][j - 1]
                insertion = row[j - 1]
                deletion = table[i - 1][j]
                row.append(min((diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                               (insertion[0] + 1, insertion[1], insertion[2] + 1, insertion[3]),
                               (deletion[0] + 1, deletion[1], deletion[2], deletion[3] + 1)))
        table.append(row)
    edits, substitutions, insertions, deletions = table[-1][-1]
    return {"edits": edits, "substitutions": substitutions,
            "insertions": insertions, "deletions": deletions,
            "reference_tokens": len(reference)}


def corpus_wer(pairs: list[tuple[list[str], list[str]]]) -> dict[str, Any]:
    substitutions = insertions = deletions = reference_tokens = 0
    for reference, hypothesis in pairs:
        counts = token_edit(reference, hypothesis)
        substitutions += counts["substitutions"]
        insertions += counts["insertions"]
        deletions += counts["deletions"]
        reference_tokens += counts["reference_tokens"]
    total = substitutions + insertions + deletions
    return {
        "n": len(pairs),
        "wer": round(total / reference_tokens, 5) if reference_tokens else None,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "reference_tokens": reference_tokens,
        "correct_tokens": reference_tokens - total,
    }


def read_gt() -> dict[str, dict[str, Any]]:
    rows = common.read_csv_rows(GT_CSV)
    return {row["sample_id"]: {"signer_id": row["signer_id"],
                               "gloss": row["gt_gloss"].split(),
                               "sentence": row.get("gt_sentence_official") or row.get("ref_sentence_upstream")}
            for row in rows}


# ---------------------------------------------------------------------------
# instrumentation + the optimised (cache-only) prepare_inputs
# ---------------------------------------------------------------------------
class Instrumentation:
    """Wrap existing objects/module attributes; never edits slbridge sources.

    ``slbridge.slrt.runtime.SLRTRuntime.predict`` imports
    ``prepare_inputs`` from the preprocess module *inside* the call, so replacing
    the module attribute is enough to redirect the runtime without touching it.
    """

    def __init__(self) -> None:
        self.reset_sample()
        self.totals: dict[str, float] = {}
        self._original_prepare = None
        self._original_canonical = None
        self._patched = False

    def reset_sample(self) -> None:
        self.sample: dict[str, float] = {
            "decode_ms": 0.0, "feed_ms": 0.0, "extract_ms": 0.0,
            "canonical_ms": 0.0, "prepare_ms": 0.0, "predict_ms": 0.0,
            "model_inference_ms": 0.0, "windows": 0,
        }

    def add(self, key: str, value: float) -> None:
        self.sample[key] = self.sample.get(key, 0.0) + float(value)

    def install(self, backend: Any, *, use_cache: bool) -> None:
        import slbridge.slrt.preprocess as slrt_preprocess
        import slbridge.slrt.stream as slrt_stream

        self._original_prepare = slrt_preprocess.prepare_inputs
        self._original_canonical = slrt_stream.canonical_frame
        original_prepare = self._original_prepare
        instrumentation = self
        cache: dict[tuple[int, int], Any] = {}

        def timed_original(bgr, keypoints, *, device="cpu"):
            started = perf_counter()
            try:
                return original_prepare(bgr, keypoints, device=device)
            finally:
                instrumentation.add("prepare_ms", (perf_counter() - started) * 1000)

        def cached_prepare(bgr, keypoints, *, device="cpu"):
            import torch

            instrumentation.add("windows", 1)
            started = perf_counter()
            try:
                x = np.asarray(bgr)
                raw = np.asarray(keypoints)
                if x.ndim != 4 or x.shape[1:] != (320, 320, 3) or x.dtype != np.uint8:
                    raise ValueError("Expected T x 320 x 320 x 3 uint8 BGR frames")
                if raw.ndim != 3 or raw.shape[1:] != (133, 3) or len(raw) != len(x):
                    raise ValueError("Expected T x 133 x 3 raw COCO-WholeBody keypoints")
                rgb_parts, hm_parts = [], []
                for index in range(len(x)):
                    key = (hash(x[index].tobytes()), hash(raw[index].tobytes()))
                    item = cache.get(key)
                    if item is None:
                        if len(cache) >= CACHE_LIMIT:
                            cache.clear()
                        item = original_prepare(x[index:index + 1], raw[index:index + 1],
                                                device=device)
                        cache[key] = item
                    rgb_parts.append(item[0])
                    hm_parts.append(item[1])
                rgb = torch.cat(rgb_parts, dim=2)
                heatmaps = torch.cat(hm_parts, dim=2)
                return rgb, heatmaps
            finally:
                elapsed = (perf_counter() - started) * 1000
                instrumentation.add("prepare_ms", elapsed)
                if use_cache:
                    instrumentation.add("prepare_python_overhead_ms", elapsed)

        slrt_preprocess.prepare_inputs = cached_prepare if use_cache else timed_original

        def timed_canonical(bgr):
            started = perf_counter()
            try:
                return self._original_canonical(bgr)
            finally:
                instrumentation.add("canonical_ms", (perf_counter() - started) * 1000)

        slrt_stream.canonical_frame = timed_canonical

        original_new_stream = backend.new_stream

        def new_stream(*args: Any, **kwargs: Any):
            stream = original_new_stream(*args, **kwargs)
            frontend = getattr(stream, "frontend", None)
            if frontend is not None and hasattr(frontend, "extract"):
                original_extract = frontend.extract

                def timed_extract(bgr, *, frame_index, timestamp):
                    started = perf_counter()
                    try:
                        return original_extract(bgr, frame_index=frame_index, timestamp=timestamp)
                    finally:
                        instrumentation.add("extract_ms", (perf_counter() - started) * 1000)

                frontend.extract = timed_extract
            original_feed = stream.feed

            def timed_feed(packet):
                started = perf_counter()
                try:
                    return original_feed(packet)
                finally:
                    instrumentation.add("feed_ms", (perf_counter() - started) * 1000)

            stream.feed = timed_feed
            runtime = stream.runtime
            original_predict = runtime.predict
            if not getattr(runtime, "_opt_instrumented", False):
                def timed_predict(bgr, keypoints):
                    started = perf_counter()
                    try:
                        probabilities, timings = original_predict(bgr, keypoints)
                    finally:
                        instrumentation.add("predict_ms", (perf_counter() - started) * 1000)
                    instrumentation.add("model_inference_ms",
                                        float(timings.get("inference_ms") or 0.0))
                    return probabilities, timings

                runtime.predict = timed_predict
                runtime._opt_instrumented = True
            return stream

        backend.new_stream = new_stream
        self._patched = True

    def uninstall(self) -> None:
        if not self._patched:
            return
        import slbridge.slrt.preprocess as slrt_preprocess
        import slbridge.slrt.stream as slrt_stream

        slrt_preprocess.prepare_inputs = self._original_prepare
        slrt_stream.canonical_frame = self._original_canonical
        self._patched = False


# ---------------------------------------------------------------------------
# pipeline execution
# ---------------------------------------------------------------------------
def run_arm(*, arm: str, samples: list[dict[str, Any]], gt: dict[str, dict[str, Any]],
            run_dir: Path, run_id: str, use_cache: bool) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend

    config = SLRTConfig.from_file(common.SLRT_CONFIG_HRNET_NPZ)
    instrumentation = Instrumentation()
    backend = None
    load_seconds = None
    for attempt in range(3):
        try:
            started = perf_counter()
            backend = SLRTBackend(config)
            load_seconds = perf_counter() - started
            break
        except BaseException as error:  # noqa: BLE001
            if attempt < 2 and any(marker in str(error).lower() for marker in RETRYABLE):
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            raise
    instrumentation.install(backend, use_cache=use_cache)
    rows: dict[str, dict[str, Any]] = {}
    failures = 0
    try:
        for index, sample in enumerate(samples):
            instrumentation.reset_sample()
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
                        "dropped_frames": result.get("dropped_frames"),
                        "sum_prepare_ms": round(instrumentation.sample["prepare_ms"], 3),
                        "sum_extract_ms": round(instrumentation.sample["extract_ms"], 3),
                        "sum_canonical_ms": round(instrumentation.sample["canonical_ms"], 3),
                        "sum_feed_ms": round(instrumentation.sample["feed_ms"], 3),
                        "sum_predict_ms": round(instrumentation.sample["predict_ms"], 3),
                        "sum_model_inference_ms": round(instrumentation.sample["model_inference_ms"], 3),
                        "attempts": attempt,
                    }
                    break
                except BaseException as error:  # noqa: BLE001 - recorded as evidence
                    if attempt <= 2 and any(marker in str(error).lower() for marker in RETRYABLE):
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    failures += 1
                    rows[sample["sample_id"]] = {
                        "schema_version": 1, "run_id": run_id, "arm": arm,
                        "sample_id": sample["sample_id"], "signer_id": sample["signer_id"],
                        "status": "failed", "error": f"{type(error).__name__}: {error}",
                        "reference_gloss": gt.get(sample["sample_id"], {}).get("gloss", []),
                        "attempts": attempt,
                    }
                    break
            if (index + 1) % 5 == 0:
                print(json.dumps({"event": "progress", "arm": arm, "done": index + 1,
                                  "total": len(samples)}), flush=True)
    finally:
        instrumentation.uninstall()
        if backend is not None:
            backend.close()
    ok = {key: row for key, row in rows.items() if row.get("status") == "final"
          and not row.get("error")}

    def stats(key: str) -> dict[str, Any]:
        values = [row.get(key) for row in ok.values() if isinstance(row.get(key), (int, float))]
        return {"p50": common.percentile(values, 50), "p95": common.percentile(values, 95),
                "mean": round(statistics.fmean(values), 2) if values else None, "count": len(values)}

    summary = {
        "arm": arm,
        "use_cache": use_cache,
        "attempted": len(samples),
        "successful": len(ok),
        "failed": failures,
        "load_seconds": round(load_seconds or 0.0, 2),
        "wer": corpus_wer([(gt.get(key, {}).get("gloss", []), row["gloss_sequence"])
                           for key, row in ok.items()]),
        "e2e_wall_ms": stats("e2e_wall_ms"),
        "video_decode_ms": stats("video_decode_ms"),
        "sum_prepare_ms": stats("sum_prepare_ms"),
        "sum_extract_ms": stats("sum_extract_ms"),
        "sum_canonical_ms": stats("sum_canonical_ms"),
        "sum_feed_ms": stats("sum_feed_ms"),
        "sum_predict_ms": stats("sum_predict_ms"),
        "sum_model_inference_ms": stats("sum_model_inference_ms"),
        "windows_processed": stats("windows_processed"),
    }
    common.write_jsonl(run_dir / f"predictions-{arm}.jsonl", rows.values())
    return rows, summary


# ---------------------------------------------------------------------------
# profile + equivalence
# ---------------------------------------------------------------------------
def profile_baseline(samples: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend

    config = SLRTConfig.from_file(common.SLRT_CONFIG_HRNET_NPZ)
    instrumentation = Instrumentation()
    backend = SLRTBackend(config)
    instrumentation.install(backend, use_cache=False)
    rows = []
    try:
        for sample in samples[:2]:
            instrumentation.reset_sample()
            started = perf_counter()
            result = backend.recognize_video(sample["trimmed"], pose_npz=sample["npz"])
            wall = (perf_counter() - started) * 1000
            rows.append({
                "sample_id": sample["sample_id"],
                "e2e_wall_ms": round(wall, 2),
                "wall_minus_decode_minus_feed_ms": round(
                    wall - float(result.get("video_decode_ms") or 0.0)
                    - instrumentation.sample["feed_ms"], 2),
                **{key: round(value, 2) for key, value in instrumentation.sample.items()},
            })
    finally:
        instrumentation.uninstall()
        backend.close()
    payload = {"samples": rows,
               "note": ("baseline stage wall per sample; feed_ms contains extract+canonical+predict; "
                        "predict_ms contains prepare_inputs+model inference")}
    common.write_json(run_dir / "stage-profile.json", payload)
    return payload


def equivalence_check(samples: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    import cv2
    import torch

    import slbridge.slrt.preprocess as slrt_preprocess

    original_prepare = slrt_preprocess.prepare_inputs
    canonical = slrt_preprocess.canonical_frame
    select = slrt_preprocess.select_keypoints

    records = []
    for sample in samples[:2]:
        with np.load(sample["npz"], allow_pickle=False) as data:
            keypoints = data["keypoints"].copy()
        capture = cv2.VideoCapture(str(sample["trimmed"]))
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(canonical(frame))
        finally:
            capture.release()
        count = min(len(frames), len(keypoints))
        cache: dict[tuple[int, int], Any] = {}
        for start in (0, 5, 20, 40):
            if start + 16 > count:
                continue
            bgr = np.stack(frames[start:start + 16])
            kp = keypoints[start:start + 16]
            original_rgb, original_hm = original_prepare(bgr, kp, device="cuda:0")
            rgb_parts, hm_parts = [], []
            for index in range(16):
                key = (hash(bgr[index].tobytes()), hash(np.asarray(kp)[index].tobytes()))
                item = cache.get(key)
                if item is None:
                    item = original_prepare(bgr[index:index + 1], np.asarray(kp)[index:index + 1],
                                            device="cuda:0")
                    cache[key] = item
                rgb_parts.append(item[0])
                hm_parts.append(item[1])
            cached_rgb = torch.cat(rgb_parts, dim=2)
            cached_hm = torch.cat(hm_parts, dim=2)
            records.append({
                "sample_id": sample["sample_id"],
                "window_start": start,
                "rgb_max_abs_diff": float((original_rgb - cached_rgb).abs().max().item()),
                "hm_max_abs_diff": float((original_hm - cached_hm).abs().max().item()),
                "rgb_exact_equal": bool(torch.equal(original_rgb, cached_rgb)),
                "hm_exact_equal": bool(torch.equal(original_hm, cached_hm)),
                "rgb_shape": list(original_rgb.shape),
                "hm_shape": list(original_hm.shape),
            })
    payload = {
        "device": "cuda:0",
        "max_rgb_abs_diff": max(record["rgb_max_abs_diff"] for record in records),
        "max_hm_abs_diff": max(record["hm_max_abs_diff"] for record in records),
        "all_exact_equal": all(record["rgb_exact_equal"] and record["hm_exact_equal"]
                               for record in records),
        "windows_checked": len(records),
        "records": records,
        "note": ("cached path calls the original unmodified prepare_inputs on one frame at "
                 "a time and concatenates; this measures the exact numerical difference"),
    }
    common.write_json(run_dir / "equivalence.json", payload)
    return payload


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-foreign-vram-mb", type=float, default=4500.0)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.run != "run":
        parser.error("only `run` is supported")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"slrt-preprocess-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    baseline_gpu = common.nvidia_smi()
    if baseline_gpu is None:
        raise SystemExit("nvidia-smi unavailable")
    device = baseline_gpu["devices"][0]
    if device["memory_used_mb"] > args.max_foreign_vram_mb and not args.allow_shared_gpu:
        raise SystemExit(f"GPU busy: {device['memory_used_mb']:.0f} MB used")
    samples = common.attach_fps(common.select_pilot_samples(args.samples))
    gt = read_gt()
    manifest = common.base_manifest(run_id, "E04-opt-slrt-preprocess", purpose=(
        "SLRT per-window preprocessing optimisation: per-frame RGB/heatmap cache with "
        "numerical equivalence, before/after latency and corpus WER"))
    manifest.update({
        "model_assets": common.model_assets_manifest(),
        "environment": common.environment_fingerprint(),
        "gpu_baseline": baseline_gpu,
        "ram": common.memory_snapshot(),
        "samples": [sample["sample_id"] for sample in samples],
        "signers": sorted({sample["signer_id"] for sample in samples}),
        "protocol": "hrnet-npz, window 16, stride 1, vote 7; same videos/keypoints both arms",
        "gt_usage": ("EVAL-ONLY gt_1086 gt_gloss joined after inference for corpus WER; "
                     "never passed to the model"),
        "production_defaults_changed": False,
        "commands": [common.command_line()],
    })
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "baseline", "gpu": device, "samples": len(samples)},
                     ensure_ascii=False), flush=True)

    print("[opt] stage profile (baseline, 2 samples) ...", flush=True)
    profile = profile_baseline(samples, run_dir)
    print(json.dumps({"event": "profile", "rows": profile["samples"]}, ensure_ascii=False)[:1200], flush=True)

    print("[opt] numerical equivalence check (original vs cached) ...", flush=True)
    equivalence = equivalence_check(samples, run_dir)
    print(json.dumps({"event": "equivalence", "max_rgb_abs_diff": equivalence["max_rgb_abs_diff"],
                      "max_hm_abs_diff": equivalence["max_hm_abs_diff"],
                      "all_exact_equal": equivalence["all_exact_equal"],
                      "windows": equivalence["windows_checked"]}, ensure_ascii=False), flush=True)

    print("[opt] baseline arm ...", flush=True)
    baseline_rows, baseline_summary = run_arm(
        arm="baseline", samples=samples, gt=gt, run_dir=run_dir, run_id=run_id, use_cache=False)
    print(json.dumps({"event": "arm_done", **{k: baseline_summary[k] for k in
                                              ("arm", "successful", "failed", "wer",
                                               "e2e_wall_ms", "sum_prepare_ms")}},
                     ensure_ascii=False), flush=True)

    print("[opt] optimised arm ...", flush=True)
    optimized_rows, optimized_summary = run_arm(
        arm="optimized", samples=samples, gt=gt, run_dir=run_dir, run_id=run_id, use_cache=True)
    print(json.dumps({"event": "arm_done", **{k: optimized_summary[k] for k in
                                              ("arm", "successful", "failed", "wer",
                                               "e2e_wall_ms", "sum_prepare_ms")}},
                     ensure_ascii=False), flush=True)

    shared = sorted(set(baseline_rows) & set(optimized_rows))
    identical = sum(
        (baseline_rows[key].get("gloss_sequence") or []) == (optimized_rows[key].get("gloss_sequence") or [])
        for key in shared
        if baseline_rows[key].get("status") == "final" and optimized_rows[key].get("status") == "final")
    base_wer = baseline_summary["wer"]["wer"]
    opt_wer = optimized_summary["wer"]["wer"]
    speedup = (baseline_summary["e2e_wall_ms"]["p50"] or 0) / (optimized_summary["e2e_wall_ms"]["p50"] or 1)
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "samples": len(samples),
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "equivalence": {key: equivalence[key] for key in
                        ("max_rgb_abs_diff", "max_hm_abs_diff", "all_exact_equal", "windows_checked")},
        "comparison": {
            "shared_samples": len(shared),
            "identical_gloss_sequences": identical,
            "wer_baseline": base_wer,
            "wer_optimized": opt_wer,
            "wer_delta": round((opt_wer or 0) - (base_wer or 0), 5) if base_wer is not None else None,
            "e2e_p50_speedup": round(speedup, 3),
            "e2e_p50_baseline_ms": baseline_summary["e2e_wall_ms"]["p50"],
            "e2e_p50_optimized_ms": optimized_summary["e2e_wall_ms"]["p50"],
            "e2e_p95_baseline_ms": baseline_summary["e2e_wall_ms"]["p95"],
            "e2e_p95_optimized_ms": optimized_summary["e2e_wall_ms"]["p95"],
            "prepare_ms_baseline_p50": baseline_summary["sum_prepare_ms"]["p50"],
            "prepare_ms_optimized_p50": optimized_summary["sum_prepare_ms"]["p50"],
            "model_inference_ms_baseline_p50": baseline_summary["sum_model_inference_ms"]["p50"],
            "model_inference_ms_optimized_p50": optimized_summary["sum_model_inference_ms"]["p50"],
            "claim": ("numerically identical preprocessing (see equivalence); WER must not degrade; "
                      "production defaults unchanged"),
        },
        "gpu_baseline": baseline_gpu,
    }
    common.write_json(run_dir / "metrics.json", metrics)
    manifest.update({"run_status": "PASS" if baseline_summary["failed"] == 0
                     and optimized_summary["failed"] == 0 else "PARTIAL",
                     "finished_at": common.utc_now(),
                     "output_files": sorted(path.name for path in run_dir.iterdir() if path.is_file())})
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "done", "run_dir": str(run_dir),
                      "comparison": metrics["comparison"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

