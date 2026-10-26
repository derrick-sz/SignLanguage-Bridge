"""Run an alternative pinned checkpoint on CSL-News clips with cached poses.

Purpose: the release ships more checkpoints than the one this repository pins
(``csl_daily_pose_only_slt.pth``). This script lets an experiment point at one
of them without weakening the production contract:

* the production path (``UniSignBackend.load_model``) verifies the pinned
  CSL-Daily checkpoint bytes/SHA256 and must keep doing so;
* this experiment constructs ``TorchUniSignRuntime`` directly, records the
  actual checkpoint SHA256 in its own protocol file, and refuses to reuse an
  output directory whose protocol differs.

Only clips whose canonical pose is already cached are evaluated, so the run
costs model time alone (no RTMPose). An alternative checkpoint is an
**experiment**, never a production model: numbers are not comparable to the
pinned run unless the pose cache, frame sampling and decoding are identical,
which this script enforces by reading the main run's protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig  # noqa: E402
from slbridge.uni_sign_preprocess import RTMPOSE_KEYPOINT_COUNT  # noqa: E402
from slbridge.uni_sign_preprocess import orient_pose_window, prepare_pose_window  # noqa: E402

from evaluate_unisign_cslnews import compute_metrics, latency_stats  # noqa: E402
from evaluate_unisign_cecsl import (  # noqa: E402
    edit_similarity,
    load_pose,
    normalize_text,
    write_json,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Alternative-checkpoint experiment on cached CSL-News poses")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/run-cslnews-20261025",
                        help="Pinned run whose protocol, pose cache and predictions are reused")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--control-run", type=Path,
                        help="Run directory whose predictions form the paired control. Must use the "
                             "same pose source, otherwise the delta mixes pose-frontend differences.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Run on whatever clips are cached instead of requiring all selected")
    parser.add_argument("--pose-source", choices=("cache", "upstream-pkl"), default="cache",
                        help="cache = this run's rtmlib poses; upstream-pkl = released CSL-News pkl")
    parser.add_argument("--pose-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/frontend-ablation/pose-format")
    parser.add_argument("--pose-rescale", choices=("none", "isotropic"), default="none",
                        help="none = released pkl normalization (x/W, y/H); isotropic = convert to x/min(W,H)")
    args = parser.parse_args(argv)

    main_protocol = json.loads((args.run_dir / "run-protocol.json").read_text(encoding="utf-8"))
    pose_protocol = main_protocol["pose_protocol"]
    pose_normalization = pose_protocol.get("pose_normalization", "isotropic_short_side")
    pose_orientation = main_protocol.get("pose_orientation", "preserve")
    cache_roots = list((args.run_dir / "pose-cache").glob("*"))
    if not cache_roots:
        raise SystemExit("No pose cache found for the pinned run")
    cache_dir = cache_roots[0]

    selected = main_protocol["selected"]
    usable = []
    for item in selected:
        video = Path(item["video"])
        if args.pose_source == "cache":
            if (cache_dir / f"{item['number']}.npz").is_file():
                usable.append(item)
        elif (args.pose_dir / f"{video.stem}.pkl").is_file():
            usable.append(item)
    if args.limit is not None:
        usable = usable[:args.limit]
    if not usable:
        raise SystemExit("No poses available for the selected clips")
    if args.pose_source == "cache" and not args.allow_partial and len(usable) != len(selected):
        raise SystemExit(f"Only {len(usable)}/{len(selected)} clips are cached; "
                         "pass --allow-partial to run on the cached subset")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = file_sha256(checkpoint)

    config = UniSignConfig.from_bundle(args.bundle)
    config = replace(config, checkpoint=checkpoint, checkpoint_bytes=checkpoint.stat().st_size,
                     checkpoint_sha256=checkpoint_sha256,
                     run_id=f"uni-sign-experiment-{checkpoint.stem}")
    runtime = TorchUniSignRuntime(config, device=args.device,
                                  dtype="bfloat16" if args.device == "cuda" else "float32")

    protocol = {
        "schema_version": 1,
        "experiment": "alternative-checkpoint on cached CSL-News poses",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "production_pinned_checkpoint": main_protocol["checkpoint_sha256"],
        "pose_protocol": pose_protocol, "pose_normalization": pose_normalization,
        "pose_orientation": pose_orientation,
        "pose_source": args.pose_source,
        "pose_dir": str(args.pose_dir) if args.pose_source == "upstream-pkl" else None,
        "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "main_run_protocol_sha256": hashlib.sha256(
            (args.run_dir / "run-protocol.json").read_bytes()).hexdigest(),
        "selected": [{"number": item["number"], "video": item["video"],
                      "video_sha256": item["video_sha256"], "reference": item["reference"]}
                     for item in usable],
    }
    protocol_path = output_dir / "experiment-protocol.json"
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                            ensure_ascii=False).encode("utf-8")).hexdigest()
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise SystemExit("Experiment protocol changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)

    existing = {}
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("number"):
                    if row.get("protocol_id") != protocol_id:
                        raise SystemExit("Prediction record protocol does not match this experiment")
                    existing[row["number"]] = row
    done = {number for number, row in existing.items()
            if row.get("status") == "ok" and row.get("prediction", "").strip()}
    pending = [item for item in usable if item["number"] not in done]
    print(json.dumps({"event": "start", "usable": len(usable), "pending": len(pending),
                      "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha256,
                      "protocol_id": protocol_id}, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    new_count = 0
    warmup_seconds = None
    try:
        runtime.load()
        # TorchUniSignRuntime.load() loads weights only; the first generate pays
        # lazy CUDA/generation initialisation. Warm up so the first sample is not
        # charged with it (UniSignBackend.load_model does the same).
        warmup_points = np.zeros((48, 1, RTMPOSE_KEYPOINT_COUNT, 2), dtype=np.float32)
        warmup_scores = np.ones((48, 1, RTMPOSE_KEYPOINT_COUNT), dtype=np.float32)
        for index in range(48):
            warmup_points[index, 0, 0] = [0.30 + 0.05 * index / 47.0, 0.30]
            warmup_points[index, 0, 3:11] = [[0.25 + 0.02 * j, 0.35 + 0.01 * j] for j in range(8)]
            warmup_points[index, 0, 91:112] = [0.40, 0.45]
            warmup_points[index, 0, 112:133] = [0.55, 0.45]
            warmup_points[index, 0, 23:40:2] = [0.45, 0.25]
            warmup_points[index, 0, 83:91] = [0.45, 0.30]
            warmup_points[index, 0, 53] = [0.45, 0.25]
        warmup_started = time.perf_counter()
        runtime.infer(prepare_pose_window(warmup_points, warmup_scores,
                                          max_frames=int(config.runtime.get("max_frames", 256))),
                      num_beams=1, max_new_tokens=8)
        warmup_seconds = time.perf_counter() - warmup_started
        print(json.dumps({"event": "model_ready", "warmup_seconds": round(warmup_seconds, 3),
                          "runtime": runtime.runtime_info}, ensure_ascii=False), flush=True)
        for item in pending:
            video = Path(item["video"])
            row = {
                "number": item["number"], "video": str(video),
                "reference": item["reference"], "prediction": "",
                "status": "error", "error": None, "protocol_id": protocol_id,
                "checkpoint_sha256": checkpoint_sha256,
                "pose_source": ("disk_cache" if args.pose_source == "cache" else "upstream-pkl"), "pose_orientation": pose_orientation,
                "model_forward_ms": None, "generation_ms": None, "decode_ms": None,
                "model_total_ms": None, "e2e_seconds": None,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                if args.pose_source == "cache":
                    pose = load_pose(cache_dir / f"{item['number']}.npz", video,
                                     pose_normalization, pose_protocol=pose_protocol)
                else:
                    from evaluate_unisign_cslnews_upstream_pose import load_upstream_pose
                    keypoints, scores, _info = load_upstream_pose(
                        args.pose_dir / f"{video.stem}.pkl")
                    if args.pose_rescale == "isotropic":
                        import cv2
                        _cap = cv2.VideoCapture(str(video))
                        _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        _cap.release()
                        # released pkl is (x/W, y/H); isotropic short-side is (x/min, y/min)
                        _s = float(min(_w, _h))
                        keypoints = keypoints * np.asarray([_w / _s, _h / _s], np.float32)
                    # Released pkl stores isotropically normalized coordinates; the
                    # body-bbox step cancels any uniform pre-scale, so use as-is.
                    pose = prepare_pose_window(
                        keypoints, scores,
                        max_frames=int(pose_protocol.get("max_frames", 256)))
                text, timings = runtime.infer(
                    orient_pose_window(pose, pose_orientation),
                    num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
                row.update(timings)
                row["prediction"] = text.strip()
                row["status"] = "ok" if text.strip() else "empty"
                if not text.strip():
                    row["error"] = "empty_text_output"
            except KeyboardInterrupt:
                raise
            except Exception as error:  # noqa: BLE001 - recorded per sample
                row["error"] = f"{type(error).__name__}: {error}"
            row["e2e_seconds"] = time.perf_counter() - sample_started
            with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            existing[row["number"]] = row
            new_count += 1
            finished = sum(1 for entry in usable
                           if existing.get(entry["number"], {}).get("status") == "ok")
            print(json.dumps({"event": "sample", "number": row["number"],
                              "status": row["status"], "prediction": row["prediction"],
                              "error": row["error"], "completed": finished,
                              "total": len(usable),
                              "model_total_ms": (round(row["model_total_ms"], 1)
                                                 if row["model_total_ms"] else None),
                              "eta_minutes": round((time.perf_counter() - started)
                                                   / max(1, new_count)
                                                   * (len(usable) - finished) / 60.0, 1)},
                             ensure_ascii=False), flush=True)
            if new_count % args.save_every == 0:
                rows = [existing[entry["number"]] for entry in usable
                        if entry["number"] in existing]
                ok_rows = [entry for entry in rows if entry.get("status") == "ok"]
                write_json(output_dir / "summary.partial.json", {
                    "partial": True, "completed": len(rows), "successful": len(ok_rows),
                    "metrics_on_successful_subset": (
                        compute_metrics(ok_rows, scope="experiment partial") if ok_rows else None)})

        rows = [existing[entry["number"]] for entry in usable if entry["number"] in existing]
        rows.sort(key=lambda entry: entry["number"])
        successful = [entry for entry in rows if entry.get("status") == "ok"]
        failed = [entry for entry in rows if entry.get("status") != "ok"]
        metrics = compute_metrics(successful, scope="experiment successful subset") if successful else None

        pinned = {}
        control_dir = (args.control_run or args.run_dir).resolve()
        pinned_path = control_dir / "predictions.jsonl"
        if pinned_path.is_file():
            for line in pinned_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("number"):
                        pinned[row["number"]] = row
        pairs = [(entry, pinned[entry["number"]]) for entry in successful
                 if entry["number"] in pinned and pinned[entry["number"]].get("status") == "ok"]
        comparison = None
        if pairs:
            def edit(row):
                return edit_similarity(normalize_text(row["reference"]),
                                       normalize_text(row.get("prediction")))
            this_scores = [edit(pair[0]) for pair in pairs]
            pinned_scores = [edit(pair[1]) for pair in pairs]
            deltas = [a - b for a, b in zip(this_scores, pinned_scores)]
            comparison = {
                "paired_n": len(pairs),
                "this_checkpoint_mean_edit_similarity": statistics.mean(this_scores),
                "pinned_checkpoint_mean_edit_similarity": statistics.mean(pinned_scores),
                "mean_delta_this_minus_pinned": statistics.mean(deltas),
                "delta_p50": float(np.percentile(deltas, 50)),
                "identical_prediction_count": sum(
                    pair[0]["prediction"] == pair[1].get("prediction") for pair in pairs),
                "empty_this": sum(not entry["prediction"].strip() for entry in successful),
                "control_run": str(control_dir),
                "protocol_note": ("Paired on the same clips and decoding. The control side reuses "
                                  "another run's recorded predictions; pass --control-run so the "
                                  "control uses the same pose source, otherwise the delta mixes "
                                  "pose-frontend differences with the checkpoint change."),
            }

        summary = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment": "alternative-checkpoint on cached CSL-News poses",
            "not_a_production_model": True,
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
            "protocol_id": protocol_id,
            "usable_clips": len(usable), "successful": len(successful),
            "failed_or_empty": len(failed), "failed_samples": failed,
            "metrics_on_successful_subset": metrics,
            "paired_comparison_vs_pinned_checkpoint": comparison,
            "warmup_seconds_before_first_sample": warmup_seconds,
            "latency_model_only": {
                "model_total_ms": latency_stats([entry.get("model_total_ms") for entry in successful]),
                "e2e_seconds": latency_stats([entry.get("e2e_seconds") for entry in successful]),
                "protocol_note": "Poses come from disk cache: model-only latency, no RTMPose.",
            },
            "runtime": runtime.runtime_info,
            "examples": successful[:10],
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"event": "done", "summary": str(output_dir / "summary.json"),
                          "successful": len(successful), "failed": len(failed),
                          "metrics": metrics, "comparison": comparison},
                         ensure_ascii=False, indent=2), flush=True)
        return 0 if len(successful) == len(usable) else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())






