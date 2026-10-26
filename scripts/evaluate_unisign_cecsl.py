"""Evaluate the pinned Uni-Sign model on the local CE-CSL test split.

This is a cross-domain evaluation: CSL-Daily-trained Uni-Sign weights are run
on CE-CSL videos. Results must not be described as CSL-Daily or general CSL
accuracy. The script is resumable and caches prepared pose tensors per video.
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

from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.uni_sign import UniSignBackend, UniSignConfig  # noqa: E402
from slbridge.uni_sign_preprocess import PoseWindow, orient_pose_window  # noqa: E402


def normalize_text(text):
    if text is None:
        return ""
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for j, char_b in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if char_a == char_b else max(current[-1], previous[j]))
        previous = current
    return previous[-1]


def rouge_l_f1_char(reference, hypothesis):
    if not reference and not hypothesis:
        return 1.0
    lcs = lcs_length(reference, hypothesis)
    precision = lcs / max(1, len(hypothesis))
    recall = lcs / max(1, len(reference))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def edit_similarity(a, b):
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / max(1, len(a), len(b))


def load_records(path, *, expected_count=500, split="test"):
    records = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            records.append({
                "number": row["Number"].strip(),
                "translator": row["Translator"].strip(),
                "reference": row["Chinese Sentences"].strip(),
                "gloss": row.get("Gloss", "").strip(),
            })
    if len(records) != expected_count:
        raise ValueError(f"Expected CE-CSL {split} to contain {expected_count} rows, got {len(records)}")
    if len({r["number"] for r in records}) != len(records):
        raise ValueError("Duplicate CE-CSL sample identifiers")
    if any(not re.fullmatch(re.escape(split) + r"-\d{5}", r["number"])
           or r["translator"] not in "ABCDEFGHIJKL" or len(r["translator"]) != 1 for r in records):
        raise ValueError(f"Expected {split} sample identifiers")
    return records


def pose_cache_path(cache_dir, number):
    return cache_dir / f"{number}.npz"


def protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pose_protocol(config, *, pose_device=None):
    """Identity of canonical poses before the explicitly selected direction."""
    runtime = config.runtime
    if pose_device is None:
        pose_device = runtime.get("pose_device", "auto")
        if pose_device == "auto":
            pose_device = runtime.get("device", "auto")
    if pose_device not in {"cpu", "cuda"}:
        raise ValueError("A resolved pose_device is required for cache identity")
    return {
        "schema_version": 3,
        "preprocessing": "rtmpose133-body-root-isotropic-v2",
        "source_commit": config.source_commit,
        "pose_mode": str(runtime.get("pose_mode", "lightweight")),
        "pose_backend": str(runtime.get("pose_backend", "onnxruntime")),
        "pose_device": pose_device,
        "pose_normalization": str(runtime.get("pose_normalization", "isotropic_short_side")),
        "max_frames": int(runtime.get("max_frames", 256)),
        "max_video_frames": int(runtime.get("max_video_frames", 600)),
        "pose_orientation": "preserve",
    }


def ensure_run_protocol(output_dir, protocol):
    """Refuse to mix outputs from different inputs, directions or decoding."""
    path = output_dir / "run-protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Evaluation protocol changed; use a new output directory")
    elif (output_dir / "predictions.jsonl").exists():
        raise ValueError("Existing predictions lack a run protocol; use a new output directory")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(path, protocol)
    return protocol_digest(protocol)


def load_direction_map(path, config, pose_protocol, signers):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1 or data.get("dataset") != "CE-CSL":
        raise ValueError("Unsupported CE-CSL direction map")
    split = data.get("calibration_split")
    if split not in {"train", "dev"}:
        raise ValueError("Direction calibration must use train or dev")
    ids = data.get("calibration_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(n, str) or not n.startswith(split + "-") for n in ids):
        raise ValueError("Direction map calibration sample identifiers do not match its split")
    if data.get("checkpoint_sha256") != config.checkpoint_sha256:
        raise ValueError("Direction map does not match the model checkpoint")
    # The map may predate a *format* bump of the pose protocol (schema_version 2 -> 3).
    # Accept that drift only when every semantic field still matches; any change that
    # actually alters the poses (normalization, mode, device, frames, orientation,
    # preprocessing, detection stride, sampling) must stay a hard failure.
    recorded = data.get("pose_protocol")
    if not isinstance(recorded, dict):
        raise ValueError("Direction map lacks a pose protocol")
    semantic_keys = ("preprocessing", "source_commit", "pose_mode", "pose_backend",
                     "pose_device", "pose_normalization", "max_frames", "max_video_frames",
                     "pose_orientation", "pose_sample_only", "pose_detect_stride")
    drift = {key: (recorded.get(key), pose_protocol.get(key))
             for key in semantic_keys if recorded.get(key) != pose_protocol.get(key)}
    if drift:
        raise ValueError(f"Direction map does not match the model/pose protocol: {drift}")
    directions = data.get("directions")
    if not isinstance(directions, dict) or any(not isinstance(v, str) or v not in {"preserve", "mirror"} for v in directions.values()):
        raise ValueError("Invalid directions in direction map")
    if set(signers) - directions.keys():
        raise ValueError("Direction map does not cover every selected signer")
    return {signer: directions[signer] for signer in signers}


def save_pose(path, pose, video, pose_normalization, *, pose_protocol=None):
    if pose_protocol is not None and pose.quality.get("pose_orientation", "preserve") != pose_protocol.get("pose_orientation", "preserve"):
        raise ValueError("pose orientation does not match cache protocol")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        body=pose.body,
        left=pose.left,
        right=pose.right,
        face_all=pose.face_all,
        quality=np.asarray(json.dumps(pose.quality, ensure_ascii=False)),
        pose_normalization=np.asarray([pose_normalization]),
        video_size=np.asarray([video.stat().st_size], dtype=np.int64),
        video_mtime_ns=np.asarray([video.stat().st_mtime_ns], dtype=np.int64),
        video_sha256=np.asarray(file_sha256(video)),
        pose_protocol=np.asarray(json.dumps(pose_protocol, sort_keys=True)),
    )


def load_pose(path, video, pose_normalization, *, pose_protocol=None):
    with np.load(path, allow_pickle=False) as data:
        if "pose_normalization" not in data or str(data["pose_normalization"][0]) != pose_normalization:
            raise ValueError("cached pose was produced with a different pose normalization")
        if pose_protocol is not None and ("pose_protocol" not in data or
                json.loads(str(data["pose_protocol"].item())) != pose_protocol):
            raise ValueError("cached pose was produced with a different pose protocol")
        if int(data["video_size"][0]) != video.stat().st_size or int(data["video_mtime_ns"][0]) != video.stat().st_mtime_ns:
            raise ValueError("cached pose does not match the current video file metadata")
        if "video_sha256" in data:
            if str(data["video_sha256"].item()) != file_sha256(video):
                raise ValueError("cached pose does not match current video content")
        elif pose_protocol is not None:
            raise ValueError("cached pose lacks video content identity")
        quality = json.loads(str(data["quality"].item()))
        if pose_protocol is not None and quality.get("pose_orientation", "preserve") != pose_protocol.get("pose_orientation", "preserve"):
            raise ValueError("cached pose orientation does not match protocol")
        parts = {name: data[name] for name in ("body", "left", "right", "face_all")}
        for part in parts.values():
            part.setflags(write=False)
        return PoseWindow(
            **parts,
            sampled_indices=tuple(int(value) for value in quality["sampled_indices"]), quality=quality)


def compute_metrics(records, *, split="test"):
    predictions = {row["number"]: row for row in records}
    ordered = [predictions[key] for key in sorted(predictions)]
    refs = [normalize_text(row["reference"]) for row in ordered]
    hyps = [normalize_text(row.get("prediction")) for row in ordered]
    exact = sum(a == b for a, b in zip(refs, hyps))
    edits = [edit_similarity(a, b) for a, b in zip(refs, hyps)]
    lengths = [len(b) / max(1, len(a)) for a, b in zip(refs, hyps)]

    import sacrebleu

    bleu_zh = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score
    bleu_char = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    rouge_l = statistics.mean(
        rouge_l_f1_char(ref, hyp) for ref, hyp in zip(refs, hyps)
    ) if refs else 0.0

    def group_metrics(key):
        groups = {}
        for translator in sorted({row["translator"] for row in ordered}):
            subset = [row for row in ordered if row["translator"] == translator]
            groups[translator] = {
                "n": len(subset),
                "exact_match": sum(normalize_text(row["reference"]) == normalize_text(row.get("prediction"))
                                   for row in subset),
                "mean_char_edit_similarity": statistics.mean(
                    edit_similarity(normalize_text(row["reference"]), normalize_text(row.get("prediction")))
                    for row in subset),
                "bleu_zh": sacrebleu.corpus_bleu(
                    [normalize_text(row.get("prediction")) for row in subset],
                    [[normalize_text(row["reference"]) for row in subset]], tokenize="zh").score,
                "chrf": sacrebleu.corpus_chrf(
                    [normalize_text(row.get("prediction")) for row in subset],
                    [[normalize_text(row["reference"]) for row in subset]]).score,
            }
        return groups

    return {
        "n": len(ordered),
        "domain": f"CE-CSL {split} split evaluated by CSL-Daily-trained Uni-Sign weights (cross-domain)",
        "exact_match_count": exact,
        "exact_match_percent": 100.0 * exact / max(1, len(ordered)),
        "mean_char_edit_similarity": statistics.mean(edits) if edits else 0.0,
        "bleu_zh": bleu_zh,
        "bleu_char": bleu_char,
        "chrf": chrf,
        "rouge_l_f1_char": rouge_l,
        "mean_length_ratio_prediction_to_reference": statistics.mean(lengths) if lengths else 0.0,
        "empty_prediction_count": sum(not text for text in hyps),
        "reference_chars_total": sum(len(text) for text in refs),
        "prediction_chars_total": sum(len(text) for text in hyps),
        "per_translator": group_metrics("translator"),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, records):
    fields = ["number", "translator", "reference", "prediction", "gloss", "status", "error",
              "video", "model_forward_ms", "generation_ms", "decode_ms", "model_total_ms",
              "pose_source", "source_frames", "sampled_frames", "valid_body_fraction",
              "pose_orientation", "run_protocol_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-domain Uni-Sign evaluation on CE-CSL")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/unisign-cecsl-eval/run-20261024")
    parser.add_argument("--pose-cache-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--pose-normalization", choices=("isotropic_short_side", "wh"),
                        help="Override runtime pose normalization for an explicit ablation")
    parser.add_argument("--pose-mode", choices=("lightweight", "balanced", "performance"))
    parser.add_argument("--pose-detect-stride", type=int,
                        help="Reuse the detected person box for N frames (1 = detect every frame)")
    parser.add_argument("--pose-sample-only", action="store_true",
                        help="Extract poses only on the frames the model window keeps")
    orientation_group = parser.add_mutually_exclusive_group()
    orientation_group.add_argument("--pose-orientation", choices=("preserve", "mirror"))
    orientation_group.add_argument("--direction-map", type=Path,
                                   help="Explicit map calibrated on train/dev; never selected using test answers")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args(argv)
    expected_count = {"train": 4973, "dev": 515, "test": 500}[args.split]
    if args.limit is not None and not 1 <= args.limit <= expected_count:
        parser.error(f"--limit must be in [1, {expected_count}]")
    if not 1 <= args.save_every <= 100:
        parser.error("--save-every must be in [1, 100]")
    dataset_root = Path(r"D:\Zenodo\CE-CSL\CE-CSL")
    labels_path = (args.labels or dataset_root / "label" / (args.split + ".csv")).resolve()
    video_root = (args.video_root or dataset_root / "video" / args.split).resolve()
    output_dir = args.output_dir.resolve()
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    partial_path = output_dir / "summary.partial.json"
    csv_path = output_dir / "predictions.csv"
    records = load_records(labels_path, expected_count=expected_count, split=args.split)
    selected = records[:args.limit] if args.limit is not None else records
    for row in selected:
        row["video"] = str((video_root / row["translator"] / f"{row['number']}.mp4").resolve())
        stat = Path(row["video"]).stat()
        row["video_size"] = stat.st_size
        row["video_mtime_ns"] = stat.st_mtime_ns
        row["video_sha256"] = file_sha256(row["video"])

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
    # Pose-affecting knobs belong to the cache identity; only non-default values are
    # added so existing run protocols stay byte-identical and resumable.
    if config.runtime.get("pose_sample_only"):
        pose_protocol["pose_sample_only"] = True
    if int(config.runtime.get("pose_detect_stride", 1)) > 1:
        pose_protocol["pose_detect_stride"] = int(config.runtime["pose_detect_stride"])
    signers = sorted({r["translator"] for r in selected})
    if args.direction_map is not None:
        directions = load_direction_map(args.direction_map, config, pose_protocol, signers)
    else:
        directions = {signer: args.pose_orientation or config.runtime.get("pose_orientation", "preserve") for signer in signers}
    run_protocol = {
        "schema_version": 2, "split": args.split,
        "checkpoint_sha256": config.checkpoint_sha256, "source_commit": config.source_commit,
        "mt5_revision": config.mt5_revision, "pose_protocol": pose_protocol,
        "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "device": backend.device, "dtype": backend.dtype, "device_id": args.device_id,
        "directions": directions,
        "direction_map_sha256": hashlib.sha256(args.direction_map.read_bytes()).hexdigest() if args.direction_map else None,
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "video_root": str(video_root), "selected": selected,
    }
    run_protocol_id = ensure_run_protocol(output_dir, run_protocol)
    cache_dir = (args.pose_cache_dir or output_dir / "pose-cache").resolve() / protocol_digest(pose_protocol)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)

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
    completed_numbers = {
        number for number, row in existing.items()
        if row.get("status") == "ok"
        and isinstance(row.get("prediction"), str)
        and row["prediction"].strip()
        and row.get("run_protocol_id") == run_protocol_id
    }
    pending = [row for row in selected if row["number"] not in completed_numbers]
    print(json.dumps({
        "event": "start", "total": len(selected), "already_complete": len(selected) - len(pending),
        "pending": len(pending), "output_dir": str(output_dir), "pose_cache": str(cache_dir),
        "domain_warning": "CE-CSL videos evaluated by CSL-Daily weights; cross-domain only",
        "pose_normalization": pose_normalization,
        "run_protocol_id": run_protocol_id, "directions": directions,
    }, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    new_count = 0
    try:
        if pending:
            backend.load_model()
        runtime = backend._runtime
        print(json.dumps({"event": "model_ready", "runtime": backend.runtime_info},
                         ensure_ascii=False), flush=True)
        for index, row in enumerate(pending, 1):
            number = row["number"]
            video = Path(row["video"])
            pose_path = pose_cache_path(cache_dir, number)
            record = {
                "number": number,
                "translator": row["translator"],
                "reference": row["reference"],
                "gloss": row["gloss"],
                "video": str(video),
                "prediction": "",
                "status": "error",
                "error": None,
                "pose_source": "disk_cache",
                "pose_normalization": pose_normalization,
                "pose_orientation": directions[row["translator"]],
                "run_protocol_id": run_protocol_id,
                "source_frames": None,
                "sampled_frames": None,
                "valid_body_fraction": None,
                "model_forward_ms": None,
                "generation_ms": None,
                "decode_ms": None,
                "model_total_ms": None,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                if not video.is_file():
                    raise FileNotFoundError(f"missing video: {video}")
                if pose_path.is_file():
                    try:
                        pose = load_pose(pose_path, video, pose_normalization, pose_protocol=pose_protocol)
                    except Exception:
                        pose = None
                else:
                    pose = None
                if pose is None:
                    pose, metrics, _ = backend._video_pose_window(video)
                    save_pose(pose_path, pose, video, pose_normalization, pose_protocol=pose_protocol)
                    record.update(metrics)
                    record["pose_source"] = "extracted"
                record["source_frames"] = pose.quality.get("source_frames")
                record["sampled_frames"] = pose.quality.get("sampled_frames")
                record["valid_body_fraction"] = pose.quality.get("valid_body_fraction")
                if float(pose.quality.get("valid_body_fraction", 0.0)) < 0.1:
                    raise ValueError("insufficient_body_frames")
                text, timings = runtime.infer(
                    orient_pose_window(pose, record["pose_orientation"]),
                    num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
                record.update(timings)
                record["prediction"] = text.strip()
                record["status"] = "ok" if text.strip() else "empty"
                if not text.strip():
                    record["error"] = "empty_text_output"
            except KeyboardInterrupt:
                raise
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            record["elapsed_seconds"] = time.perf_counter() - sample_started
            with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            existing[number] = record
            new_count += 1
            completed = sum(
                1 for row2 in selected
                if existing.get(row2["number"], {}).get("status") == "ok"
                and isinstance(existing.get(row2["number"], {}).get("prediction"), str)
            )
            elapsed = time.perf_counter() - started
            rate = elapsed / max(1, new_count)
            remaining = len(selected) - completed
            print(json.dumps({
                "event": "sample", "number": number, "translator": row["translator"],
                "status": record["status"], "prediction": record["prediction"],
                "error": record["error"], "completed": completed, "total": len(selected),
                "sample_seconds": round(record["elapsed_seconds"], 3),
                "eta_minutes": round(rate * remaining / 60.0, 1),
            }, ensure_ascii=False), flush=True)
            if new_count % args.save_every == 0:
                rows = [existing[key] for key in existing if key in {r["number"] for r in selected}]
                successful = [r for r in rows if r.get("status") == "ok"]
                partial = {
                    "schema_version": 1, "partial": True, "selected": len(selected),
                    "completed": completed, "successful": len(successful),
                    "failed_or_empty": len(selected) - len(successful),
                    "metrics_on_successful_subset": compute_metrics(successful, split=args.split) if successful else None,
                }
                write_json(partial_path, partial)

        all_rows = [existing[key] for key in existing if key in {r["number"] for r in selected}]
        all_rows.sort(key=lambda item: item["number"])
        successful = [r for r in all_rows if r.get("status") == "ok"]
        failed = [r for r in all_rows if r.get("status") != "ok"]
        metrics = compute_metrics(successful, split=args.split) if successful else None
        summary = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "evaluation": f"Uni-Sign csl_daily_pose_only_slt on CE-CSL {args.split} split",
            "cross_domain": True,
            "domain_warning": "CE-CSL is not the CSL-Daily training domain; metrics are cross-domain and not product accuracy.",
            "labels": str(labels_path), "video_root": str(video_root),
            "bundle": str(args.bundle.resolve()), "device": args.device,
            "pose_normalization": pose_normalization,
            "run_protocol_id": run_protocol_id, "pose_protocol": pose_protocol,
            "directions": directions, "split": args.split,
            "selected": len(selected), "full_test_set": args.split == "test" and len(selected) == 500,
            "full_split": len(selected) == expected_count,
            "successful": len(successful), "failed_or_empty": len(failed),
            "complete": len(successful) == len(selected),
            "metrics_on_successful_subset": metrics,
            "failed_samples": failed,
            "manual_semantic_review": "NOT_RUN",
            "metric_definitions": {
                "exact_match": "normalized character-for-character equality",
                "bleu_zh": "SacreBLEU tokenize=zh on normalized predictions/references",
                "bleu_char": "SacreBLEU character-token BLEU",
                "chrf": "SacreBLEU chrF",
                "rouge_l_f1_char": "ROUGE-L F1 with each Chinese character treated as a token",
                "mean_char_edit_similarity": "mean normalized Levenshtein similarity",
            },
            "examples": successful[:10],
        }
        write_json(summary_path, summary)
        write_csv(csv_path, all_rows)
        print(json.dumps({"event": "done", "summary": str(summary_path), "csv": str(csv_path),
                          "selected": len(selected), "successful": len(successful),
                          "failed": len(failed), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
        return 0 if not failed and len(successful) == len(selected) else 1
    except KeyboardInterrupt:
        all_rows = [existing[key] for key in existing if key in {r["number"] for r in selected}]
        successful = [r for r in all_rows if r.get("status") == "ok"]
        if successful:
            write_json(partial_path, {"schema_version": 1, "interrupted": True,
                                      "selected": len(selected), "successful": len(successful),
                                      "metrics_on_successful_subset": compute_metrics(successful, split=args.split)})
        print(json.dumps({"event": "interrupted", "successful": len(successful)}, ensure_ascii=False), flush=True)
        return 130
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())



