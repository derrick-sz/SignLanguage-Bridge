"""Evaluate the pinned Uni-Sign pose-only weights on the local CSL-News subset.

Protocol boundaries (read before interpreting any number):

* The only released checkpoint on disk is ``csl_daily_pose_only_slt.pth``, a
  **CSL-Daily** pose-only SLT model. CSL-News is a different corpus, so every
  text metric produced here is **cross-domain** and is not CSL-News accuracy.
* The 1,694 local clips are exactly rows 1..1694 of ``CSL_News_Labels.csv``.
  Upstream ``datasets.py::S2T_Dataset_news`` treats the first 99% of that
  annotation order as its training portion, so this subset is *not* the
  upstream held-out split either. No CSL-News fine-tuned checkpoint has been
  released (HF ``ZechengLi19/Uni-Sign`` ships CSL-Daily/WLASL/How2Sign/OpenASL
  downstream weights plus stage-1/2 pretraining weights only), so this run
  cannot report in-domain CSL-News accuracy.
* Latency numbers are domain-independent: they describe the current
  RTMPose + Uni-Sign file-inference path on this hardware.

The run is resumable, binds every prediction to a protocol digest, and caches
canonical pose tensors per video content hash (same rules as the CE-CSL
evaluator in ``evaluate_unisign_cecsl.py``).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.uni_sign import UniSignBackend, UniSignConfig  # noqa: E402
from slbridge.uni_sign_preprocess import orient_pose_window  # noqa: E402

# Reuse the CE-CSL evaluator's protocol/metric helpers so both evaluations share
# exactly one definition of text normalization, cache identity and BLEU.
from evaluate_unisign_cecsl import (  # noqa: E402
    build_pose_protocol,
    edit_similarity,
    ensure_run_protocol,
    file_sha256,
    load_pose,
    normalize_text,
    protocol_digest,
    rouge_l_f1_char,
    save_pose,
    write_json,
)

DEFAULT_LABELS = ROOT / "data/CSL-NEWS/CSL_News_Labels.csv"
DEFAULT_VIDEO_ROOT = ROOT / "data/CSL-NEWS/archive_001"
FRAME_SPAN = re.compile(r"_(\d+)-(\d+)_")


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def latency_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def video_metadata(path):
    """Cheap container probe: no frame decode beyond the header."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video has no valid FPS: {path}")
    match = FRAME_SPAN.search(Path(path).name)
    name_span = int(match.group(2)) - int(match.group(1)) if match else None
    if frames <= 0 and name_span:
        frames = name_span
    return {
        "video_fps": fps,
        "video_frames": frames,
        "video_width": width,
        "video_height": height,
        "video_duration_s": frames / fps if frames > 0 else None,
        "name_frame_span": name_span,
    }


def load_records(labels_path, video_root, *, limit=None, offset=0, stride=1):
    """Local clips in CSV order; the row index is the stable sample id.

    The label file has 722k rows while only the local clips matter, so the
    directory is indexed once instead of stat-ing every label row.
    """
    video_root = Path(video_root)
    local = {entry.name: entry.resolve() for entry in video_root.iterdir() if entry.is_file()}
    if not local:
        raise ValueError(f"No clips found under {video_root}")
    records = []
    with Path(labels_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["video", "pose", "text"]:
            raise ValueError(f"Unexpected CSL-News label columns: {reader.fieldnames}")
        for row_index, row in enumerate(reader, 1):
            video_name = (row["video"] or "").strip()
            if not video_name:
                continue
            video = local.get(video_name)
            if video is None:
                continue
            records.append({
                "number": f"cslnews-{row_index:06d}",
                "row_index": row_index,
                "video_name": video_name,
                "pose_name": (row["pose"] or "").strip(),
                "reference": (row["text"] or "").strip(),
                "video": str(video),
            })
    if not records:
        raise ValueError("No local CSL-News clips matched the label file")
    selected = records[offset::stride]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Selection is empty")
    return records, selected


def compute_metrics(rows, *, scope):
    ordered = sorted(rows, key=lambda item: item["number"])
    refs = [normalize_text(row["reference"]) for row in ordered]
    hyps = [normalize_text(row.get("prediction")) for row in ordered]
    exact = sum(a == b for a, b in zip(refs, hyps))
    edits = [edit_similarity(a, b) for a, b in zip(refs, hyps)]
    lengths = [len(b) / max(1, len(a)) for a, b in zip(refs, hyps)]
    chrf_plus_plus = None
    try:
        import sacrebleu

        bleu_zh = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score
        bleu_char = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        chrf_plus_plus = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    except Exception:  # pragma: no cover - only if sacrebleu is absent
        bleu_zh = bleu_char = chrf = None
    return {
        "scope": scope,
        "n": len(ordered),
        "exact_match_count": exact,
        "exact_match_percent": 100.0 * exact / max(1, len(ordered)),
        "mean_char_edit_similarity": statistics.mean(edits) if edits else 0.0,
        "bleu_zh": bleu_zh,
        "bleu_char": bleu_char,
        "chrf": chrf,
        "chrf_plus_plus": chrf_plus_plus,
        "rouge_l_f1_char": statistics.mean(
            rouge_l_f1_char(ref, hyp) for ref, hyp in zip(refs, hyps)) if refs else 0.0,
        "mean_length_ratio_prediction_to_reference": statistics.mean(lengths) if lengths else 0.0,
        "empty_prediction_count": sum(not text for text in hyps),
        "reference_chars_total": sum(len(text) for text in refs),
        "prediction_chars_total": sum(len(text) for text in hyps),
        "edit_similarity_p50": percentile(edits, 50),
        "edit_similarity_p95": percentile(edits, 95),
        "edit_similarity_max": max(edits) if edits else None,
        "samples_at_or_above_0_5_edit_similarity": sum(1 for value in edits if value >= 0.5),
    }


def decode_benchmark(records, limit):
    """Decode-only cost, so RTMPose can be separated from container decode."""
    import cv2

    rows = []
    for record in records[:limit]:
        capture = cv2.VideoCapture(record["video"])
        if not capture.isOpened():
            rows.append({"number": record["number"], "error": "open_failed"})
            continue
        started = time.perf_counter()
        frames = 0
        try:
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                frames += 1
        finally:
            capture.release()
        elapsed = time.perf_counter() - started
        rows.append({
            "number": record["number"], "video_frames": frames,
            "decode_ms": elapsed * 1000.0,
            "decode_fps": frames / elapsed if elapsed > 0 else None,
        })
    ok_rows = [row for row in rows if "decode_ms" in row]
    return {
        "n": len(rows),
        "failed": len(rows) - len(ok_rows),
        "decode_ms": latency_stats([row["decode_ms"] for row in ok_rows]),
        "decode_fps": latency_stats([row["decode_fps"] for row in ok_rows]),
        "per_video": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross-domain Uni-Sign evaluation on the local CSL-News subset")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/run-cslnews-20261025")
    parser.add_argument("--pose-cache-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--pose-normalization", choices=("isotropic_short_side", "wh"),
                        help="Override runtime pose normalization for an explicit ablation")
    parser.add_argument("--pose-mode", choices=("lightweight", "balanced", "performance"))
    parser.add_argument("--pose-detect-stride", type=int,
                        help="Reuse the detected person box for N frames (1 = detect every frame)")
    parser.add_argument("--pose-sample-only", action="store_true",
                        help="Extract poses only on the frames the model window keeps")
    parser.add_argument("--pose-orientation", choices=("preserve", "mirror"), default="preserve",
                        help="Fixed a priori; never selected from this subset's answers")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--decode-benchmark", type=int, default=0,
                        help="Time decode-only reads for the first N selected clips")
    parser.add_argument("--mirror-probe", type=int, default=0,
                        help="Re-run the first N selected clips in both orientations "
                             "(direction sensitivity, not direction selection)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Build labels/metadata/protocol and exit without loading the model")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.offset < 0:
        parser.error("--stride must be >= 1 and --offset >= 0")
    if not 1 <= args.save_every <= 500:
        parser.error("--save-every must be in [1, 500]")

    output_dir = args.output_dir.resolve()
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    partial_path = output_dir / "summary.partial.json"
    csv_path = output_dir / "predictions.csv"
    cold_start_path = output_dir / "cold-start.json"

    all_records, selected = load_records(
        args.labels, args.video_root, limit=args.limit, offset=args.offset, stride=args.stride)
    print(json.dumps({"event": "labels", "local_records": len(all_records),
                      "selected": len(selected),
                      "first": selected[0]["number"], "last": selected[-1]["number"]},
                     ensure_ascii=False), flush=True)

    for record in selected:
        video = Path(record["video"])
        stat = video.stat()
        record["video_size"] = stat.st_size
        record["video_mtime_ns"] = stat.st_mtime_ns
        record["video_sha256"] = file_sha256(video)
        record.update(video_metadata(video))

    options = RuntimeOptions(device=args.device, device_id=args.device_id)
    config = UniSignConfig.from_bundle(args.bundle)
    if args.pose_normalization is not None:
        config.runtime["pose_normalization"] = args.pose_normalization
    if args.pose_mode is not None:
        config.runtime["pose_mode"] = args.pose_mode
    if args.pose_detect_stride is not None:
        config.runtime["pose_detect_stride"] = args.pose_detect_stride
    if args.pose_sample_only:
        config.runtime["pose_sample_only"] = True
    config.runtime.update(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
    pose_normalization = str(config.runtime.get("pose_normalization", "isotropic_short_side"))
    backend = UniSignBackend(config, options)
    pose_protocol = build_pose_protocol(config, pose_device=backend.pose_device)
    pose_protocol["pose_orientation"] = args.pose_orientation
    # Pose-affecting knobs must be part of the cache identity, otherwise a
    # stride-5 or sample-only run would share cached poses with a stride-1 run.
    # Only non-default values are added so existing protocols stay byte-identical.
    if config.runtime.get("pose_sample_only"):
        pose_protocol["pose_sample_only"] = True
    if int(config.runtime.get("pose_detect_stride", 1)) > 1:
        pose_protocol["pose_detect_stride"] = int(config.runtime["pose_detect_stride"])

    run_protocol = {
        "schema_version": 2,
        "dataset": "CSL-News",
        "subset": f"archive_001 local clips ({len(all_records)} of the CSV rows)",
        "subset_selection": {"offset": args.offset, "stride": args.stride, "limit": args.limit},
        "domain_note": ("CSL-News clips evaluated by CSL-Daily-trained weights: cross-domain. "
                        "Local rows 1..1694 fall inside the upstream training portion "
                        "(datasets.py S2T_Dataset_news uses [0, 99%) for train), so this is not "
                        "a held-out split of CSL-News either."),
        "checkpoint_sha256": config.checkpoint_sha256,
        "source_commit": config.source_commit,
        "mt5_revision": config.mt5_revision,
        "pose_protocol": pose_protocol,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "device": backend.device,
        "dtype": backend.dtype,
        "device_id": args.device_id,
        "pose_orientation": args.pose_orientation,
        "labels_sha256": hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
        "video_root": str(Path(args.video_root).resolve()),
        "selected": [
            {key: record[key] for key in ("number", "video", "video_sha256", "reference",
                                          "video_frames", "video_duration_s")}
            for record in selected
        ],
    }
    run_protocol_id = ensure_run_protocol(output_dir, run_protocol)
    cache_dir = ((args.pose_cache_dir or output_dir / "pose-cache").resolve()
                 / protocol_digest(pose_protocol)[:16])
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "protocol", "run_protocol_id": run_protocol_id,
                      "output_dir": str(output_dir), "pose_cache": str(cache_dir),
                      "device": backend.device, "dtype": backend.dtype,
                      "pose_normalization": pose_normalization,
                      "pose_orientation": args.pose_orientation,
                      "pose_protocol": pose_protocol}, ensure_ascii=False), flush=True)

    if args.decode_benchmark > 0:
        benchmark = decode_benchmark(selected, args.decode_benchmark)
        write_json(output_dir / "decode-benchmark.json", benchmark)
        print(json.dumps({"event": "decode_benchmark", "n": benchmark["n"],
                          "decode_ms": benchmark["decode_ms"],
                          "decode_fps": benchmark["decode_fps"]}, ensure_ascii=False), flush=True)

    if args.manifest_only:
        write_json(output_dir / "selection.json", {
            "schema_version": 1, "run_protocol_id": run_protocol_id,
            "count": len(selected),
            "records": [{key: record[key] for key in (
                "number", "row_index", "video_name", "video", "reference", "video_sha256",
                "video_frames", "video_fps", "video_width", "video_height",
                "video_duration_s", "name_frame_span", "video_size")} for record in selected],
        })
        write_json(output_dir / "selection-stats.json", {
            "count": len(selected),
            "frames": latency_stats([record["video_frames"] for record in selected]),
            "duration_s": latency_stats([record["video_duration_s"] for record in selected]),
            "reference_chars": latency_stats([len(record["reference"]) for record in selected]),
            "resolution": sorted({f"{record['video_width']}x{record['video_height']}"
                                  for record in selected}),
            "fps": sorted({record["video_fps"] for record in selected}),
            "total_frames": sum(record["video_frames"] for record in selected),
            "total_duration_s": sum(record["video_duration_s"] for record in selected),
            "over_max_frames_256": sum(1 for record in selected if record["video_frames"] > 256),
            "over_max_video_frames_600": sum(
                1 for record in selected if record["video_frames"] > 600),
        })
        print(json.dumps({"event": "manifest_only_done", "count": len(selected)},
                         ensure_ascii=False), flush=True)
        return 0

    existing = {}
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("number"):
                if row.get("run_protocol_id") != run_protocol_id:
                    raise ValueError("Prediction record protocol does not match this evaluation")
                existing[row["number"]] = row
    completed = {
        number for number, row in existing.items()
        if row.get("status") == "ok" and isinstance(row.get("prediction"), str)
        and row["prediction"].strip() and row.get("run_protocol_id") == run_protocol_id
    }
    pending = [record for record in selected if record["number"] not in completed]
    print(json.dumps({"event": "start", "total": len(selected),
                      "already_complete": len(selected) - len(pending),
                      "pending": len(pending), "output_dir": str(output_dir)},
                     ensure_ascii=False), flush=True)

    started = time.perf_counter()
    new_count = 0
    try:
        if pending:
            load_started = time.perf_counter()
            backend.load_model()
            load_seconds = time.perf_counter() - load_started
            runtime_info = backend.runtime_info
            write_json(cold_start_path, {
                "schema_version": 1,
                "measured_at": datetime.now().astimezone().isoformat(),
                "load_model_seconds": load_seconds,
                "excluded_from_per_video_latency": True,
                "note": ("load_model() also runs the RTMPose extractor warmup and one "
                         "synthetic Uni-Sign window; it is a cold start, not a per-video cost."),
                "runtime": runtime_info,
            })
            print(json.dumps({"event": "model_ready", "load_model_seconds": load_seconds,
                              "runtime": runtime_info}, ensure_ascii=False), flush=True)
        runtime = backend._runtime
        for index, record in enumerate(pending, 1):
            video = Path(record["video"])
            pose_path = cache_dir / f"{record['number']}.npz"
            row = {
                "number": record["number"],
                "row_index": record["row_index"],
                "video": str(video),
                "video_name": record["video_name"],
                "video_sha256": record["video_sha256"],
                "reference": record["reference"],
                "prediction": "",
                "status": "error",
                "error": None,
                "pose_source": "disk_cache",
                "pose_cache_hit": True,
                "pose_normalization": pose_normalization,
                "pose_orientation": args.pose_orientation,
                "run_protocol_id": run_protocol_id,
                "video_frames": record["video_frames"],
                "video_fps": record["video_fps"],
                "video_width": record["video_width"],
                "video_height": record["video_height"],
                "video_duration_s": record["video_duration_s"],
                "name_frame_span": record["name_frame_span"],
                "source_frames": None,
                "sampled_frames": None,
                "valid_body_fraction": None,
                "pose_extract_ms": None,
                "pose_preprocess_ms": None,
                "model_forward_ms": None,
                "generation_ms": None,
                "decode_ms": None,
                "model_total_ms": None,
                "e2e_seconds": None,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                if not video.is_file():
                    raise FileNotFoundError(f"missing video: {video}")
                pose = None
                if pose_path.is_file():
                    try:
                        pose = load_pose(pose_path, video, pose_normalization,
                                         pose_protocol=pose_protocol)
                    except Exception:
                        pose = None
                if pose is None:
                    pose, pose_metrics, _ = backend._video_pose_window(video)
                    save_pose(pose_path, pose, video, pose_normalization,
                              pose_protocol=pose_protocol)
                    row.update({
                        "pose_source": "extracted",
                        "pose_cache_hit": False,
                        "pose_extract_ms": pose_metrics.get("pose_extract_ms"),
                        "pose_preprocess_ms": pose_metrics.get("pose_preprocess_ms"),
                        # Persist the pose-sampling evidence so a reader can verify that
                        # pose_sample_only / pose_detect_stride actually took effect.
                        "window_frames": pose_metrics.get("window_frames"),
                        "pose_sample_only": pose_metrics.get("pose_sample_only"),
                        "pose_skipped_frames": pose_metrics.get("pose_skipped_frames"),
                    })
                row["source_frames"] = pose.quality.get("source_frames")
                row["sampled_frames"] = pose.quality.get("sampled_frames")
                row["valid_body_fraction"] = pose.quality.get("valid_body_fraction")
                if float(pose.quality.get("valid_body_fraction", 0.0)) < 0.1:
                    raise ValueError("insufficient_body_frames")
                text, timings = runtime.infer(
                    orient_pose_window(pose, args.pose_orientation),
                    num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
                row.update(timings)
                row["prediction"] = text.strip()
                row["status"] = "ok" if text.strip() else "empty"
                if not text.strip():
                    row["error"] = "empty_text_output"
            except KeyboardInterrupt:
                raise
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["e2e_seconds"] = time.perf_counter() - sample_started
            with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            existing[row["number"]] = row
            new_count += 1
            done = sum(1 for item in selected
                       if existing.get(item["number"], {}).get("status") == "ok")
            elapsed = time.perf_counter() - started
            rate = elapsed / max(1, new_count)
            print(json.dumps({
                "event": "sample", "number": row["number"], "status": row["status"],
                "pose_source": row["pose_source"], "prediction": row["prediction"],
                "error": row["error"], "completed": done, "total": len(selected),
                "e2e_seconds": round(row["e2e_seconds"], 3),
                "model_total_ms": round(row["model_total_ms"], 1) if row["model_total_ms"] else None,
                "pose_extract_ms": (round(row["pose_extract_ms"], 1)
                                    if row["pose_extract_ms"] else None),
                "eta_minutes": round(rate * (len(selected) - done) / 60.0, 1),
            }, ensure_ascii=False), flush=True)
            if new_count % args.save_every == 0:
                rows = [existing[item["number"]] for item in selected
                        if item["number"] in existing]
                successful = [item for item in rows if item.get("status") == "ok"]
                write_json(partial_path, {
                    "schema_version": 1, "partial": True, "selected": len(selected),
                    "completed": len(rows), "successful": len(successful),
                    "metrics_on_successful_subset": (
                        compute_metrics(successful, scope="partial successful subset")
                        if successful else None),
                })

        rows = [existing[item["number"]] for item in selected if item["number"] in existing]
        rows.sort(key=lambda item: item["number"])
        successful = [item for item in rows if item.get("status") == "ok"]
        failed = [item for item in rows if item.get("status") != "ok"]
        extracted = [item for item in successful if item.get("pose_source") == "extracted"]
        cached = [item for item in successful if item.get("pose_source") == "disk_cache"]

        mirror_probe = None
        if args.mirror_probe > 0 and successful:
            mirror_rows = []
            for item in successful[:args.mirror_probe]:
                video = Path(item["video"])
                pose_path = cache_dir / f"{item['number']}.npz"
                try:
                    pose = load_pose(pose_path, video, pose_normalization,
                                     pose_protocol=pose_protocol)
                except Exception as error:
                    mirror_rows.append({"number": item["number"],
                                        "error": f"{type(error).__name__}: {error}"})
                    continue
                entry = {"number": item["number"], "reference": item["reference"]}
                for orientation in ("preserve", "mirror"):
                    probe_started = time.perf_counter()
                    text, timings = runtime.infer(
                        orient_pose_window(pose, orientation),
                        num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
                    entry[orientation] = {
                        "prediction": text.strip(),
                        "model_total_ms": timings.get("model_total_ms"),
                        "wall_seconds": time.perf_counter() - probe_started,
                        "edit_similarity": edit_similarity(
                            normalize_text(item["reference"]), normalize_text(text)),
                    }
                mirror_rows.append(entry)
            usable = [entry for entry in mirror_rows if "preserve" in entry and "mirror" in entry]
            mirror_probe = {
                "n": len(mirror_rows),
                "usable": len(usable),
                "protocol_note": ("Sensitivity only: the main run keeps preserve. This probe "
                                  "must not be used to pick a direction for these clips."),
                "preserve_mean_edit_similarity": statistics.mean(
                    entry["preserve"]["edit_similarity"] for entry in usable) if usable else None,
                "mirror_mean_edit_similarity": statistics.mean(
                    entry["mirror"]["edit_similarity"] for entry in usable) if usable else None,
                "identical_text_count": sum(
                    entry["preserve"]["prediction"] == entry["mirror"]["prediction"]
                    for entry in usable),
                "rows": mirror_rows,
            }
            write_json(output_dir / "mirror-probe.json", mirror_probe)

        runtime_info = backend.runtime_info if backend._runtime is not None else {}
        total_video_seconds = sum(float(item["video_duration_s"] or 0.0) for item in successful)
        total_e2e_seconds = sum(float(item["e2e_seconds"] or 0.0) for item in successful)
        wall_seconds = time.perf_counter() - started
        summary = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "evaluation": "Uni-Sign csl_daily_pose_only_slt on the local CSL-News subset",
            "cross_domain": True,
            "domain_warning": ("CSL-News clips are not the CSL-Daily training domain. Every text "
                              "metric below is cross-domain and must not be reported as CSL-News "
                              "or product accuracy."),
            "split_warning": ("Local clips are rows 1..1694 of CSL_News_Labels.csv; upstream "
                              "datasets.py::S2T_Dataset_news reserves [0, 99%) of that order for "
                              "training, so this subset is inside the upstream training portion "
                              "and is not a held-out split."),
            "labels": str(Path(args.labels).resolve()),
            "video_root": str(Path(args.video_root).resolve()),
            "bundle": str(args.bundle.resolve()),
            "device": backend.device, "dtype": backend.dtype,
            "pose_normalization": pose_normalization,
            "pose_orientation": args.pose_orientation,
            "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
            "run_protocol_id": run_protocol_id, "pose_protocol": pose_protocol,
            "selected": len(selected), "attempted": len(rows),
            "successful": len(successful), "failed_or_empty": len(failed),
            "complete": len(successful) == len(selected),
            "metrics_on_successful_subset": (
                compute_metrics(successful, scope="successful subset of the local CSL-News clips")
                if successful else None),
            "failed_samples": failed,
            "latency": {
                "protocol_note": ("Per-video timings exclude the cold start in cold-start.json. "
                                  "'pose_source=extracted' rows include container decode plus "
                                  "RTMPose; 'disk_cache' rows are model-only reruns."),
                "wall_seconds": wall_seconds,
                "cold_start_seconds": (json.loads(cold_start_path.read_text(encoding="utf-8"))
                                       .get("load_model_seconds")
                                       if cold_start_path.is_file() else None),
                "e2e_seconds_extracted": latency_stats(
                    [item["e2e_seconds"] for item in extracted]),
                "e2e_seconds_disk_cache": latency_stats(
                    [item["e2e_seconds"] for item in cached]),
                "pose_extract_ms": latency_stats(
                    [item.get("pose_extract_ms") for item in extracted]),
                "pose_preprocess_ms": latency_stats(
                    [item.get("pose_preprocess_ms") for item in extracted]),
                "model_forward_ms": latency_stats(
                    [item.get("model_forward_ms") for item in successful]),
                "generation_ms": latency_stats(
                    [item.get("generation_ms") for item in successful]),
                "model_total_ms": latency_stats(
                    [item.get("model_total_ms") for item in successful]),
                "realtime_factor_extracted": latency_stats(
                    [float(item["e2e_seconds"]) / float(item["video_duration_s"])
                     for item in extracted if item.get("video_duration_s")]),
                "realtime_factor_disk_cache": latency_stats(
                    [float(item["e2e_seconds"]) / float(item["video_duration_s"])
                     for item in cached if item.get("video_duration_s")]),
                "videos_per_minute_extracted": (
                    60.0 * len(extracted) / sum(item["e2e_seconds"] for item in extracted)
                    if extracted else None),
                "frames_per_second_extracted": (
                    sum(item["video_frames"] for item in extracted)
                    / sum(item["e2e_seconds"] for item in extracted) if extracted else None),
                "video_seconds_processed": total_video_seconds,
                "wall_over_video_seconds": (total_e2e_seconds / total_video_seconds
                                            if total_video_seconds else None),
            },
            "runtime": runtime_info,
            "mirror_probe": mirror_probe,
            "manual_semantic_review": "NOT_RUN",
            "metric_definitions": {
                "exact_match": "normalized character-for-character equality",
                "bleu_zh": "SacreBLEU tokenize=zh on normalized predictions/references",
                "bleu_char": "SacreBLEU character-token BLEU",
                "chrf": "SacreBLEU chrF",
                "chrf_plus_plus": "SacreBLEU chrF++ (word_order=2)",
                "rouge_l_f1_char": "ROUGE-L F1 with each Chinese character as a token",
                "mean_char_edit_similarity": "mean normalized Levenshtein similarity",
                "realtime_factor": "end-to-end seconds / video duration seconds",
            },
            "examples": successful[:10],
        }
        write_json(summary_path, summary)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["number", "row_index", "video_name", "reference", "prediction",
                      "status", "error", "pose_source", "pose_cache_hit",
                      "video_frames", "video_duration_s", "video_fps",
                      "video_width", "video_height", "source_frames", "sampled_frames",
                      "valid_body_fraction", "pose_extract_ms", "pose_preprocess_ms",
                      "model_forward_ms", "generation_ms", "decode_ms", "model_total_ms",
                      "e2e_seconds", "pose_orientation", "run_protocol_id"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in rows:
                writer.writerow(item)
        print(json.dumps({"event": "done", "summary": str(summary_path), "csv": str(csv_path),
                          "selected": len(selected), "successful": len(successful),
                          "failed": len(failed),
                          "metrics": summary["metrics_on_successful_subset"],
                          "latency": {key: value for key, value in summary["latency"].items()
                                      if key != "protocol_note"}},
                         ensure_ascii=False, indent=2), flush=True)
        return 0 if len(successful) == len(selected) else 1
    except KeyboardInterrupt:
        rows = [existing[item["number"]] for item in selected if item["number"] in existing]
        successful = [item for item in rows if item.get("status") == "ok"]
        if successful:
            write_json(partial_path, {
                "schema_version": 1, "interrupted": True, "selected": len(selected),
                "successful": len(successful),
                "metrics_on_successful_subset": compute_metrics(
                    successful, scope="interrupted successful subset")})
        print(json.dumps({"event": "interrupted", "successful": len(successful)},
                         ensure_ascii=False), flush=True)
        return 130
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())



