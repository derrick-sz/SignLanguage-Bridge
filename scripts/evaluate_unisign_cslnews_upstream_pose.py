"""Frontend ablation: upstream CSL-News pose files vs the local rtmlib frontend.

The released CSL-News pose archives were produced by the authors with MMPose
RTMPose, while this repository's runtime extracts poses with rtmlib RTMW-L at
256x192. The pitfalls memory flags that frontend difference as unquantified.
This script isolates it: the same clips, the same weights, the same decoding and
the same deterministic 256-frame sampling, but poses read from the upstream
``.pkl`` files instead of being re-extracted from video.

Geometry note: the released CSL-News ``.pkl`` files already store *normalized*
coordinates, not pixels. Evidence: for
``Common-Concerns_20201113_0-287_14330`` the body bounding-box scale implied by
the file is 0.6655, i.e. the same order as the 0.7258 that the local pipeline
records after ``isotropic_short_side`` normalization, whereas treating the
values as pixels would imply 0.0039. Feeding the values through the local
``isotropic_short_side`` normalization (multiplying by ``min(W, H)``) is a
uniform scale that ``_crop_scale`` cancels exactly, so the ablation keeps the
file values unchanged and therefore reproduces what the upstream loader builds
from these files.

This is an ablation, not the production input path. Its numbers must never be
reported as the integration's accuracy or latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.uni_sign import UniSignBackend, UniSignConfig  # noqa: E402
from slbridge.uni_sign_preprocess import (  # noqa: E402
    normalize_rtmpose_pixels,
    prepare_pose_window,
)

from evaluate_unisign_cslnews import (  # noqa: E402
    compute_metrics,
    edit_similarity,
    latency_stats,
    normalize_text,
    write_json,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_pose(path):
    """Read one upstream pose pkl and return raw pixel keypoints/scores."""
    with Path(path).open("rb") as stream:
        pose = pickle.load(stream)
    if not isinstance(pose, dict) or "keypoints" not in pose or "scores" not in pose:
        raise ValueError(f"Unexpected upstream pose structure: {sorted(pose)[:8]}")
    keypoints = np.asarray(pose["keypoints"], dtype=np.float32)
    scores = np.asarray(pose["scores"], dtype=np.float32)
    info = {
        "keypoints_shape": list(keypoints.shape),
        "scores_shape": list(scores.shape),
        "has_start_end": "start" in pose and "end" in pose,
    }
    if info["has_start_end"]:
        info["start"] = int(pose["start"])
        info["end"] = int(pose["end"])
    if keypoints.ndim == 4:
        keypoints = keypoints[:, 0]
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError(f"Unsupported keypoint shape: {keypoints.shape}")
    if scores.ndim == 3:
        scores = scores[:, 0]
    if scores.ndim != 2:
        raise ValueError(f"Unsupported score shape: {scores.shape}")
    info["frames"] = int(keypoints.shape[0])
    info["x_range"] = [float(keypoints[..., 0].min()), float(keypoints[..., 0].max())]
    info["y_range"] = [float(keypoints[..., 1].min()), float(keypoints[..., 1].max())]
    return keypoints, scores, info


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Frontend ablation on CSL-News: upstream pose pkl vs local rtmlib")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/run-cslnews-20261025",
                        help="Main evaluation directory holding run-protocol.json")
    parser.add_argument("--pose-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/frontend-ablation/pose-format")
    parser.add_argument("--pose-zip", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/frontend-ablation/archive_001.zip")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/ablation-upstream-pose")
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--inspect-only", action="store_true",
                        help="Report the first pose file's structure and exit")
    args = parser.parse_args(argv)

    main_protocol = json.loads((args.run_dir / "run-protocol.json").read_text(encoding="utf-8"))
    selected = main_protocol["selected"]
    if args.limit is not None:
        selected = selected[:args.limit]
    print(json.dumps({"event": "selected", "count": len(selected),
                      "main_protocol": main_protocol.get("dataset")}, ensure_ascii=False),
          flush=True)

    if args.inspect_only:
        first = selected[0]
        pkl = args.pose_dir / (Path(first["video"]).stem + ".pkl")
        if not pkl.is_file():
            raise FileNotFoundError(f"Pose file missing: {pkl}")
        keypoints, scores, info = load_upstream_pose(pkl)
        print(json.dumps({"event": "inspect", "pkl": str(pkl), **info}, ensure_ascii=False,
                         indent=2), flush=True)
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    options = RuntimeOptions(device=args.device, device_id=args.device_id)
    config = UniSignConfig.from_bundle(args.bundle)
    config.runtime.update(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
    backend = UniSignBackend(config, options)
    max_frames = int(config.runtime.get("max_frames", 256))

    pose_zip_sha256 = file_sha256(args.pose_zip) if args.pose_zip.is_file() else None
    protocol = {
        "schema_version": 1,
        "ablation": "upstream-cslnews-pose-pkl-vs-local-rtmlib",
        "main_run_protocol_id": hashlib.sha256(
            (args.run_dir / "run-protocol.json").read_bytes()).hexdigest(),
        "checkpoint_sha256": config.checkpoint_sha256,
        "source_commit": config.source_commit,
        "pose_source": "upstream-pkl",
        "pose_zip": str(args.pose_zip), "pose_zip_sha256": pose_zip_sha256,
        "pose_pipeline": ("normalize_rtmpose_pixels(isotropic_short_side) -> "
                          f"prepare_pose_window(max_frames={max_frames})"),
        "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "device": backend.device, "dtype": backend.dtype,
        "selected": [{"number": item["number"], "video": item["video"],
                      "video_sha256": item["video_sha256"], "reference": item["reference"]}
                     for item in selected],
    }
    protocol_path = output_dir / "ablation-protocol.json"
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                            ensure_ascii=False).encode("utf-8")).hexdigest()
    if protocol_path.is_file():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol != protocol:
            raise ValueError("Ablation protocol changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    print(json.dumps({"event": "protocol", "protocol_id": protocol_id,
                      "device": backend.device, "dtype": backend.dtype,
                      "pose_zip_sha256": pose_zip_sha256}, ensure_ascii=False), flush=True)

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
                        raise ValueError("Prediction record protocol does not match this ablation")
                    existing[row["number"]] = row
    completed = {number for number, row in existing.items()
                 if row.get("status") == "ok" and row.get("prediction", "").strip()}
    pending = [item for item in selected if item["number"] not in completed]
    print(json.dumps({"event": "start", "total": len(selected),
                      "pending": len(pending)}, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    new_count = 0
    try:
        if pending:
            backend.load_model()
        runtime = backend._runtime
        for item in pending:
            pkl = args.pose_dir / (Path(item["video"]).stem + ".pkl")
            row = {
                "number": item["number"], "video": item["video"],
                "video_sha256": item["video_sha256"], "reference": item["reference"],
                "prediction": "", "status": "error", "error": None,
                "protocol_id": protocol_id, "pose_source": "upstream-pkl",
                "pkl": str(pkl), "video_frames": item.get("video_frames"),
                "video_duration_s": item.get("video_duration_s"),
                "pose_load_ms": None, "pose_prepare_ms": None,
                "source_frames": None, "sampled_frames": None,
                "valid_body_fraction": None, "body_scale": None,
                "model_forward_ms": None, "generation_ms": None, "decode_ms": None,
                "model_total_ms": None, "e2e_seconds": None,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                if not pkl.is_file():
                    raise FileNotFoundError(f"missing pose: {pkl}")
                load_started = time.perf_counter()
                keypoints, scores, info = load_upstream_pose(pkl)
                row["pose_load_ms"] = (time.perf_counter() - load_started) * 1000.0
                prepare_started = time.perf_counter()
                points = normalize_rtmpose_pixels(
                    keypoints, int(item.get("video_width") or 0) or 1,
                    int(item.get("video_height") or 0) or 1, mode="isotropic_short_side")
                pose = prepare_pose_window(points, scores, max_frames=max_frames)
                row["pose_prepare_ms"] = (time.perf_counter() - prepare_started) * 1000.0
                row["source_frames"] = pose.quality.get("source_frames")
                row["sampled_frames"] = pose.quality.get("sampled_frames")
                row["valid_body_fraction"] = pose.quality.get("valid_body_fraction")
                row["body_scale"] = pose.quality.get("body_scale")
                if float(pose.quality.get("valid_body_fraction", 0.0)) < 0.1:
                    raise ValueError("insufficient_body_frames")
                text, timings = runtime.infer(pose, num_beams=args.num_beams,
                                              max_new_tokens=args.max_new_tokens)
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
            done = sum(1 for entry in selected
                       if existing.get(entry["number"], {}).get("status") == "ok")
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "sample", "number": row["number"], "status": row["status"],
                "prediction": row["prediction"], "error": row["error"],
                "completed": done, "total": len(selected),
                "e2e_seconds": round(row["e2e_seconds"], 3),
                "model_total_ms": (round(row["model_total_ms"], 1)
                                   if row["model_total_ms"] else None),
                "eta_minutes": round(elapsed / max(1, new_count)
                                     * (len(selected) - done) / 60.0, 1),
            }, ensure_ascii=False), flush=True)
            if new_count % args.save_every == 0:
                rows = [existing[entry["number"]] for entry in selected
                        if entry["number"] in existing]
                ok_rows = [entry for entry in rows if entry.get("status") == "ok"]
                write_json(output_dir / "summary.partial.json", {
                    "partial": True, "completed": len(rows), "successful": len(ok_rows),
                    "metrics_on_successful_subset": (
                        compute_metrics(ok_rows, scope="ablation partial") if ok_rows else None)})

        rows = [existing[entry["number"]] for entry in selected if entry["number"] in existing]
        rows.sort(key=lambda entry: entry["number"])
        successful = [entry for entry in rows if entry.get("status") == "ok"]
        failed = [entry for entry in rows if entry.get("status") != "ok"]
        metrics = compute_metrics(successful, scope="ablation successful subset") if successful else None

        comparison = None
        main_predictions = {}
        main_predictions_path = args.run_dir / "predictions.jsonl"
        if main_predictions_path.is_file():
            for line in main_predictions_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("number"):
                        main_predictions[row["number"]] = row
        pairs = [(entry, main_predictions[entry["number"]]) for entry in successful
                 if entry["number"] in main_predictions
                 and main_predictions[entry["number"]].get("status") == "ok"]
        if pairs:
            def edit(row):
                return edit_similarity(normalize_text(row["reference"]),
                                       normalize_text(row.get("prediction")))
            upstream_scores = [edit(pair[0]) for pair in pairs]
            local_scores = [edit(pair[1]) for pair in pairs]
            deltas = [a - b for a, b in zip(upstream_scores, local_scores)]
            comparison = {
                "paired_n": len(pairs),
                "upstream_pose_mean_edit_similarity": statistics.mean(upstream_scores),
                "local_rtmlib_mean_edit_similarity": statistics.mean(local_scores),
                "mean_delta_upstream_minus_local": statistics.mean(deltas),
                "delta_p50": float(np.percentile(deltas, 50)),
                "identical_prediction_count": sum(
                    pair[0]["prediction"] == pair[1].get("prediction") for pair in pairs),
                "protocol_note": ("Paired on the same clips, weights and decoding. The local "
                                  "side reuses the main run's recorded predictions, which were "
                                  "produced with the rtmlib frontend."),
            }

        summary = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "evaluation": "Uni-Sign csl_daily_pose_only_slt on CSL-News with upstream pose pkl",
            "ablation_not_production": True,
            "cross_domain": True,
            "domain_warning": ("Same cross-domain caveat as the main run: CSL-Daily weights on "
                              "CSL-News clips. This run only swaps the pose frontend."),
            "protocol_id": protocol_id,
            "pose_zip_sha256": pose_zip_sha256,
            "selected": len(selected), "attempted": len(rows),
            "successful": len(successful), "failed_or_empty": len(failed),
            "complete": len(successful) == len(selected),
            "metrics_on_successful_subset": metrics,
            "failed_samples": failed,
            "paired_comparison_vs_local_frontend": comparison,
            "latency": {
                "pose_load_ms": latency_stats([entry.get("pose_load_ms") for entry in successful]),
                "pose_prepare_ms": latency_stats(
                    [entry.get("pose_prepare_ms") for entry in successful]),
                "model_total_ms": latency_stats(
                    [entry.get("model_total_ms") for entry in successful]),
                "e2e_seconds": latency_stats([entry.get("e2e_seconds") for entry in successful]),
                "protocol_note": ("Model-only latency: poses come from disk, so no video decode "
                                  "or RTMPose cost is included."),
            },
            "runtime": backend.runtime_info,
            "examples": successful[:10],
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"event": "done", "summary": str(output_dir / "summary.json"),
                          "successful": len(successful), "failed": len(failed),
                          "metrics": metrics, "comparison": comparison},
                         ensure_ascii=False, indent=2), flush=True)
        return 0 if len(successful) == len(selected) else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
