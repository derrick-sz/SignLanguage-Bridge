"""Run the production RGB+pose adapter for the task-33 regression (Pilot-50 first, then 1086).

The adapter is loaded through the normal registry path
(``slbridge.model_adapters.load_model_adapter(bundle)``) and called through the
typed ``ModelAdapter.infer(..., input_kind="file")`` protocol.  Regression mode
uses the frozen E01 protocol:

    pose_source="official_archive"
    pose_dir=artifacts/cslnews-eval/csl-daily-pose/pose-format
    alignment_dir=artifacts/csldaily-video-test-627-20261026/pose   (npz metadata)
    enable_rgb=True
    fallback_to_pose_only=False
    device="cuda", dtype="bfloat16", num_beams=4, max_new_tokens=100

The production rtmpose mode is deliberately NOT used here; it is task-34's
performance path and is expected to differ.

Outputs (GT-free predictions, manifest, timings) and then the comparison via
``regression_rgb_pose_adapter`` are written into ``--run-dir``.

Usage:
    # after the Lead/p0a confirm the adapter is runnable
    python -m experiments.p1.e01.run_adapter_regression \
        --bundle <model_bundle_dir> --mode pilot \
        --run-dir experiments/p1/e01/runs/adapter-regression-pilot50-20261027

    # full 1086 (only after Pilot-50 PASS and the Lead schedules the GPU)
    python -m experiments.p1.e01.run_adapter_regression \
        --bundle <model_bundle_dir> --mode full \
        --run-dir experiments/p1/e01/runs/adapter-regression-1086-20261027
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1 import common
from experiments.p1.gt_guard import assert_gt_clean, find_gt_keys
from slbridge.model_adapters import load_model_adapter
from slbridge.uni_sign_rgb_pose import build_sample_inputs as build_rgb_pose_inputs
from slbridge.runtime_contracts import InferenceRequest
from slbridge.runtime_options import RuntimeOptions
from slbridge.contracts import RecognitionResult

from experiments.p1.e01 import regression_rgb_pose_adapter as comparator

POSE_DIR = ROOT / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
ALIGNMENT_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "pose"
VIDEO_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "trimmed"
FROZEN_PILOT = comparator.FROZEN_PILOT
FROZEN_1086 = comparator.FROZEN_1086
FROZEN_1086_METRICS = comparator.FROZEN_1086_METRICS
MODE_BASELINES = {"pilot": FROZEN_PILOT, "full": FROZEN_1086}


def nvidia_smi() -> str:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return probe.stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"unavailable: {type(error).__name__}"


def load_frozen_rows(path: Path) -> list[dict]:
    return [row for row in common.read_jsonl(path)]


def regression_bundle(base_bundle: Path, run_dir: Path, hand_conf_layout: str = "legacy") -> Path:
    """Create a run-local copy of the bundle manifest with frozen regression config.

    The production bundle may default to rtmpose; the regression must override the
    configuration, not silently fall back to a different pose source.
    """

    base_manifest = base_bundle / "manifest.json"
    if not base_manifest.is_file():
        raise SystemExit(f"bundle manifest not found: {base_manifest}")
    raw = json.loads(base_manifest.read_text(encoding="utf-8"))
    if raw.get("backend") != "unisign_rgb_pose_slt":
        raise SystemExit(f"bundle backend must be unisign_rgb_pose_slt, got {raw.get('backend')!r}")
    target = run_dir / "regression_bundle"
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(raw)
    manifest.update({
        "pose_source": "official_archive",
        "pose_dir": str(POSE_DIR),
        "alignment_dir": str(ALIGNMENT_DIR),
        "video_dir": str(VIDEO_DIR),
        "enable_rgb": True,
        "fallback_to_pose_only": False,
        "device": "cuda",
        "dtype": "bfloat16",
        "num_beams": 4,
        "max_new_tokens": 100,
    })
    if hand_conf_layout not in (None, "legacy"):
        # corrected mode (p0a adds hand_conf_layout); legacy keeps the frozen behaviour
        manifest["hand_conf_layout"] = hand_conf_layout
    # Resolve optional relative fields against the source bundle so the copy is self-contained.
    for key in ("checkpoint", "pose_only_checkpoint", "source_dir", "mt5_dir"):
        value = manifest.get(key)
        if value is not None and not Path(str(value)).is_absolute():
            manifest[key] = str((base_bundle / str(value)).resolve())
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"event": "regression_bundle", "path": str(target), "manifest": manifest},
                     ensure_ascii=False, indent=2))
    return target


def validate_adapter_config(adapter: Any, expected_layout: str | None = None) -> dict:
    config = getattr(adapter, "config", None)
    if config is None:
        raise SystemExit("adapter has no config; cannot verify regression configuration")
    expected = {
        "pose_source": "official_archive",
        "enable_rgb": True,
        "fallback_to_pose_only": False,
        "device": "cuda",
        "dtype": "bfloat16",
        "num_beams": 4,
        "max_new_tokens": 100,
        "hand_conf_layout": expected_layout or "legacy",
        "pose_dir": POSE_DIR.resolve(),
        "alignment_dir": ALIGNMENT_DIR.resolve(),
        "video_dir": VIDEO_DIR.resolve(),
    }
    problems = []
    for key, value in expected.items():
        actual = getattr(config, key, None)
        if isinstance(value, Path):
            if actual is None or Path(actual).resolve() != value:
                problems.append({"field": key, "expected": str(value), "actual": str(actual)})
        elif actual != value:
            problems.append({"field": key, "expected": value, "actual": actual})
    if problems:
        raise SystemExit(json.dumps({"event": "config_mismatch", "problems": problems},
                                    ensure_ascii=False, indent=2))
    return {
        "pose_source": config.pose_source,
        "pose_dir": str(config.pose_dir),
        "alignment_dir": str(config.alignment_dir),
        "video_dir": str(config.video_dir),
        "enable_rgb": config.enable_rgb,
        "fallback_to_pose_only": config.fallback_to_pose_only,
        "device": config.device,
        "dtype": config.dtype,
        "num_beams": config.num_beams,
        "max_new_tokens": config.max_new_tokens,
    }


def run(mode: str, bundle: Path, run_dir: Path, limit: int | None, hand_conf_layout: str = "legacy") -> dict:
    frozen_rows = load_frozen_rows(MODE_BASELINES[mode])
    expected_samples = [row["sample_id"] for row in frozen_rows]
    if limit is not None:
        expected_samples = expected_samples[:limit]
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest_sha = common.sha256_file(bundle / "manifest.json")
    gpu_before = nvidia_smi()

    # Explicit device: load_model_adapter applies options.device to the bundle config;
    # RuntimeOptions() defaults to "auto", which the frozen config validator rejects.
    adapter = load_model_adapter(bundle, options=RuntimeOptions(device="cuda"))
    config_actual = validate_adapter_config(adapter, expected_layout=hand_conf_layout)
    adapter.warmup()
    runtime_info = dict(getattr(adapter, "runtime_info", {}) or {})
    capabilities = getattr(adapter, "capabilities", None)
    gpu_after_load = nvidia_smi()

    predictions: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    for index, sample in enumerate(expected_samples, 1):
        video = VIDEO_DIR / f"{sample}.mp4"
        if not video.is_file():
            raise SystemExit(f"missing regression video: {video}")
        request = InferenceRequest(request_id=sample, sign_language="csl",
                                   expected_task_type="text")
        sample_started = time.perf_counter()
        try:
            result, timing = adapter.infer(video, request, input_kind="file")
        except Exception as error:  # noqa: BLE001 - record failure, continue, report
            failures.append({"sample_id": sample, "error": f"{type(error).__name__}: {str(error)[:300]}"})
            predictions.append({"sample_id": sample, "status": "error",
                                "error": f"{type(error).__name__}: {str(error)[:300]}", "text": ""})
            print(json.dumps({"event": "sample", "mode": mode, "index": index,
                              "n": len(expected_samples), "sample_id": sample,
                              "status": "error", "error": f"{type(error).__name__}"}, ensure_ascii=False))
            continue
        if not isinstance(result, RecognitionResult):
            raise SystemExit(f"adapter returned {type(result).__name__}, expected RecognitionResult")
        quality = dict(result.quality or {})
        if quality.get("fallback_reason") is not None:
            raise SystemExit(
                f"regression path fell back for {sample}: fallback_reason={quality.get('fallback_reason')!r}; "
                "this means the adapter is not running the frozen RGB+pose path")
        hand_frames = None
        if hand_conf_layout != "legacy":
            effective = quality.get("hand_conf_layout_effective")
            left_frames = quality.get("hand_frames_left")
            right_frames = quality.get("hand_frames_right")
            if effective is None or left_frames is None or right_frames is None:
                # adapter metadata may lag; recompute the same frozen preprocessing for the audit
                audit_inputs = build_rgb_pose_inputs(adapter.config, sample)
                effective = audit_inputs.get("hand_conf_layout_effective")
                left_frames = audit_inputs.get("hand_frames_left")
                right_frames = audit_inputs.get("hand_frames_right")
                quality["hand_conf_layout_effective"] = effective
                quality["hand_frames_left"] = left_frames
                quality["hand_frames_right"] = right_frames
            if effective != "squeeze":
                raise SystemExit(f"corrected run has hand_conf_layout_effective={effective!r} for {sample}, expected 'squeeze'")
            if not left_frames or not right_frames:
                raise SystemExit(f"corrected run has zero hand frames for {sample}: left={left_frames}, right={right_frames}")
            hand_frames = {"left": int(left_frames), "right": int(right_frames)}
        text = result.prediction.text if result.prediction is not None else ""
        row = {
            "run_id": run_dir.name,
            "mode": mode,
            "sample_id": sample,
            "status": result.status,
            "text": text,
            "model_id": result.model_id,
            "task_type": result.task_type,
            "quality": quality,
            "hand_conf_layout_effective": quality.get("hand_conf_layout_effective"),
            "hand_frames": hand_frames,
            "timing_adapter": {key: round(float(value), 3) for key, value in timing.items()},
            "sample_total_ms": round((time.perf_counter() - sample_started) * 1000.0, 3),
        }
        leaks = find_gt_keys(row)
        if leaks:
            raise SystemExit(f"GT-free guard violated: {leaks}")
        assert_gt_clean(row, context=f"adapter/{mode}/{sample}")
        predictions.append(row)
        print(json.dumps({"event": "sample", "mode": mode, "index": index,
                          "n": len(expected_samples), "sample_id": sample,
                          "status": result.status,
                          "text": text,
                          "model_total_ms": quality.get("model_total_ms"),
                          "fallback_reason": quality.get("fallback_reason")}, ensure_ascii=False))
    elapsed = time.perf_counter() - started
    try:
        adapter.close()
    except Exception:  # noqa: BLE001
        pass
    common.write_jsonl(run_dir / "predictions_rgb_pose_adapter.jsonl", predictions, guard=True)
    hand_summary = None
    if hand_conf_layout != "legacy":
        left_values = [row["hand_frames"]["left"] for row in predictions if row.get("hand_frames")]
        right_values = [row["hand_frames"]["right"] for row in predictions if row.get("hand_frames")]
        hand_summary = {
            "layout": hand_conf_layout,
            "n": len(left_values),
            "left_frames_mean": round(sum(left_values) / len(left_values), 2) if left_values else None,
            "left_frames_min": min(left_values) if left_values else None,
            "left_frames_max": max(left_values) if left_values else None,
            "right_frames_mean": round(sum(right_values) / len(right_values), 2) if right_values else None,
            "right_frames_min": min(right_values) if right_values else None,
            "right_frames_max": max(right_values) if right_values else None,
            "samples_with_zero_left": sum(1 for value in left_values if value == 0),
            "samples_with_zero_right": sum(1 for value in right_values if value == 0),
        }
    manifest = {
        "schema_version": 1,
        "kind": "adapter-regression-run",
        "run_id": run_dir.name,
        "mode": mode,
        "hand_conf_layout": hand_conf_layout,
        "created_at": common.utc_now(),
        "bundle": str(bundle),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "bundle_manifest": json.loads((bundle / "manifest.json").read_text(encoding="utf-8")),
        "adapter_config": config_actual,
        "runtime_info": runtime_info,
        "capabilities": (dataclasses.asdict(capabilities)
                         if capabilities is not None and dataclasses.is_dataclass(capabilities)
                         else str(capabilities) if capabilities is not None else None),
        "frozen_baseline": str(MODE_BASELINES[mode]),
        "frozen_baseline_sha256": common.sha256_file(MODE_BASELINES[mode]),
        "n_expected": len(expected_samples),
        "n_predictions": len(predictions),
        "n_failures": len(failures),
        "failures": failures,
        "hand_frame_summary": hand_summary,
        "gpu_before": gpu_before,
        "gpu_after_load": gpu_after_load,
        "gpu_after_inference": nvidia_smi(),
        "elapsed_seconds": round(elapsed, 1),
        "predictions": "predictions_rgb_pose_adapter.jsonl",
        "gt_free": True,
    }
    common.write_json(run_dir / "manifest.json", manifest)

    if hand_conf_layout != "legacy":
        # Corrected mode is expected to differ from the frozen (zero-RGB) baseline;
        # the corrected-vs-legacy/A2/control analysis is produced by
        # analyze_corrected_rgb_pose.py instead of a frozen exact-match check.
        print(json.dumps({"event": "skip_frozen_comparison", "reason": "corrected mode",
                          "hand_conf_layout": hand_conf_layout}, ensure_ascii=False))
        return {"status": "SKIPPED_FROZEN_COMPARISON", "hand_conf_layout": hand_conf_layout,
                "run_dir": str(run_dir), "n_predictions": len(predictions), "n_failures": len(failures)}

    # Comparison against the frozen baseline (Pilot -> exact; full -> metric tolerance)
    label = "pilot50" if mode == "pilot" else "full1086"
    comparison = comparator.compare_pair(
        MODE_BASELINES[mode],
        run_dir / "predictions_rgb_pose_adapter.jsonl",
        frozen_metrics_path=FROZEN_1086_METRICS if mode == "full" else None,
        label=label,
        candidate_run_dir=run_dir,
    )
    comparison["adapter_manifest"] = str(run_dir / "manifest.json")
    comparison["adapter_manifest_sha256"] = common.sha256_file(run_dir / "manifest.json")
    comparison["adapter_failures"] = failures
    common.write_json(run_dir / f"regression_{mode}.json", comparison)
    comparator.write_markdown(comparison, run_dir / f"regression_{mode}.md")

    print(json.dumps({
        "event": "done",
        "mode": mode,
        "run_dir": str(run_dir),
        "status": comparison["status"],
        "exact_match": comparison["exact_match"],
        "n_common": comparison["n_common"],
        "metric_comparison": comparison["metric_comparison"],
        "n_failures": len(failures),
        "elapsed_seconds": round(elapsed, 1),
    }, ensure_ascii=False, indent=2))
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RGB+pose production adapter regression (task-33)")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="model bundle dir with manifest.json backend=unisign_rgb_pose_slt")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="optional smoke limit (pilot mode)")
    parser.add_argument("--hand-conf-layout", default="legacy", choices=("legacy", "squeeze"),
                        help="hand confidence layout; legacy = frozen zero-RGB behaviour, squeeze = corrected real hand crops")
    parser.add_argument("--no-regression-bundle", action="store_true",
                        help="use --bundle as-is instead of creating a run-local frozen-config copy")
    args = parser.parse_args(argv)

    if not Path(args.bundle, "manifest.json").is_file():
        raise SystemExit(f"bundle manifest missing: {args.bundle / 'manifest.json'}")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = args.bundle.resolve()
    if not args.no_regression_bundle:
        bundle = regression_bundle(bundle, run_dir, hand_conf_layout=args.hand_conf_layout)
    run(mode=args.mode, bundle=bundle, run_dir=run_dir, limit=args.limit,
        hand_conf_layout=args.hand_conf_layout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
