"""Evaluate the pinned Uni-Sign SLT model on the local CSL-Daily *videos*.

This is the video end-to-end path: each sentence clip (already cropped from the
raw recording by the SLRT alignment pass, ``trimmed/<sample>.mp4``) is decoded,
posed with the pinned RTMPose wholebody frontend and translated by
``csl_daily_pose_only_slt.pth``.  It is deliberately *not* the released-pose
protocol — the released ``csl_daily_pose_format`` numbers stay the reference for
"official pose in, text out"; this script answers "video in, text out" on the
same samples so the per-sample table can sit next to SLRT.

References come from the selection file built by
``scripts/build_csldaily_video_selection.py`` (text + gloss straight from
``csl-daily.<split>``).  BLEU is computed exactly like the released-pose
evaluation: character-separated reference/hypothesis and the upstream
``' '.join(list(text.replace(' ', '')))`` rewrite (``fine_tuning.py:282-284``)
before SacreBLEU 13a.  chrF, char-ROUGE-L and edit similarity use the raw texts.

Resumable per sample; use a new output directory when any protocol input changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from slbridge.runtime_contracts import InferenceRequest  # noqa: E402
from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.uni_sign import UniSignBackend, UniSignConfig  # noqa: E402

from evaluate_unisign_cecsl import edit_similarity, normalize_text, rouge_l_f1_char  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def clean_text(text: str) -> str:
    return "".join(str(text).replace("\n", "").split())


def character_separated(text: str) -> str:
    return " ".join(list(clean_text(text)))


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * (q / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def read_selection(path: Path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = selection.get("samples") or []
    if not samples:
        raise SystemExit(f"{path}: no samples")
    names = [item["sample"] for item in samples]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate samples")
    return selection, samples


def build_protocol(config, selection_path, selection, trimmed_dir, shard):
    return {
        "schema_version": 1,
        "evaluation": "Uni-Sign pinned CSL-Daily SLT on locally cropped CSL-Daily sentence videos",
        "input": "trimmed mp4 -> pinned RTMPose wholebody 133 -> pose window -> mT5",
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": config.checkpoint_sha256,
        "source_commit": config.source_commit,
        "pose_backend": "onnxruntime", "pose_mode": "lightweight",
        "pose_normalization": config.runtime.get("pose_normalization", "isotropic_short_side"),
        "pose_orientation": "preserve",
        "max_frames": int(config.runtime.get("max_frames", 256)),
        "num_beams": int(config.runtime.get("num_beams", 4)),
        "max_new_tokens": int(config.runtime.get("max_new_tokens", 100)),
        "trimmed_dir": str(trimmed_dir),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "selection_count": int(selection.get("count", len(selection["samples"]))),
        "samples": [item["sample"] for item in selection["samples"]],
        "shard": shard,
        "created_by": "scripts/evaluate_unisign_csldaily_video.py",
    }


def write_comparison(rows, summary, output_dir: Path):
    columns = ["sample", "frames", "ground_truth", "prediction", "char_edit_similarity",
               "rouge_l_f1_char", "exact_match", "status", "e2e_seconds", "pose_ms",
               "model_total_ms", "error"]
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    metrics = summary["metrics"]
    lines = [
        "# Uni-Sign CSL-Daily video end-to-end: prediction vs ground truth",
        "",
        f"Generated: {summary['created_at']}",
        "",
        f"Scope: {summary['scope']}",
        "",
        "| metric | value |", "|---|---|",
        f"| attempted / succeeded / failed or empty | {summary['attempted']} / {summary['successful']} / {summary['failed_or_empty']} |",
        f"| BLEU-4 (SacreBLEU 13a, upstream mt5 rewrite) | {metrics['bleu4_13a']:.4f} |",
        f"| chrF | {metrics['chrf']:.4f} |",
        f"| char ROUGE-L F1 | {metrics['rouge_l_f1_char']:.4f} |",
        f"| mean char edit similarity | {metrics['mean_char_edit_similarity']:.4f} |",
        f"| exact matches | {metrics['exact_match_count']} / {metrics['n']} |",
        f"| e2e seconds per clip (mean / P50 / P95) | {summary['latency']['e2e_seconds_mean']:.3f} / "
        f"{summary['latency']['e2e_seconds_p50']:.3f} / {summary['latency']['e2e_seconds_p95']:.3f} |",
        "",
        "| # | sample | frames | ground truth | model translation | edit sim | exact | status |",
        "|---:|---|---:|---|---|---:|---|---|",
    ]
    for index, row in enumerate(rows, 1):
        similarity = "" if row["char_edit_similarity"] is None else f"{row['char_edit_similarity']:.4f}"
        lines.append(
            f"| {index} | {row['sample']} | {row['frames']} | {row['ground_truth']} | "
            f"{row['prediction']} | {similarity} | {'Y' if row['exact_match'] else 'N'} | {row['status']} |")
    lines += ["", "Failures and empty outputs:", ""]
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        lines += [f"- `{row['sample']}`: {row['status']} — {row['error']}" for row in failures]
    else:
        lines.append("- none")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="artifact tree that holds trimmed/<sample>.mp4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--only-file", type=Path, default=None,
                        help="newline-separated sample names to run (merged with --only)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not 0 <= args.shard_index < max(1, args.shard_count):
        raise SystemExit("shard-index must be in [0, shard-count)")

    selection, samples = read_selection(args.selection)
    wanted = set(args.only or [])
    if args.only_file is not None:
        wanted.update(line.strip() for line in args.only_file.read_text(encoding="utf-8").splitlines()
                      if line.strip())
    if wanted:
        unknown = wanted - {item["sample"] for item in samples}
        if unknown:
            raise SystemExit(f"unknown samples: {sorted(unknown)[:5]}")
        samples = [item for item in samples if item["sample"] in wanted]
    if args.limit:
        samples = samples[:args.limit]
    if args.shard_count > 1:
        samples = [item for position, item in enumerate(samples)
                   if position % args.shard_count == args.shard_index]
    if not samples:
        raise SystemExit("no samples selected")

    trimmed_dir = (args.artifact_dir / "trimmed").resolve()
    missing = [item["sample"] for item in samples
               if not (trimmed_dir / f"{item['sample']}.mp4").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} trimmed clips missing, e.g. {missing[:5]}")

    config = UniSignConfig.from_bundle(args.bundle)
    backend = UniSignBackend(config, RuntimeOptions(device=args.device))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = {"index": args.shard_index, "count": args.shard_count}
    run_selection = dict(selection, count=len(samples),
                         samples=[{"sample": item["sample"]} for item in samples])
    protocol = build_protocol(config, args.selection.resolve(), run_selection, trimmed_dir, shard)
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False)
                                .encode("utf-8")).hexdigest()
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        stored = json.loads(protocol_path.read_text(encoding="utf-8"))
        if stored.get("protocol_id") != protocol_id:
            raise SystemExit("Protocol changed; use a new output directory")
    else:
        write_json(protocol_path, dict(protocol, protocol_id=protocol_id))

    suffix = "" if args.shard_count == 1 else f"-shard{args.shard_index}"
    predictions_path = output_dir / f"predictions{suffix}.jsonl"
    existing = {}
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("sample"):
                if row.get("protocol_id") != protocol_id:
                    raise SystemExit("Prediction record protocol does not match")
                existing[row["sample"]] = row
    done = {name for name, row in existing.items()
            if row.get("status") == "ok" and str(row.get("prediction") or "").strip()}
    pending = [item for item in samples if item["sample"] not in done or args.force]
    print(json.dumps({"event": "start", "samples": len(samples), "pending": len(pending),
                      "protocol_id": protocol_id, "checkpoint_sha256": config.checkpoint_sha256[:16]},
                     ensure_ascii=False), flush=True)

    started = time.perf_counter()
    try:
        if pending:
            backend.load_model()
            print(json.dumps({"event": "model_loaded", "runtime": backend.runtime_info},
                             ensure_ascii=False, default=str), flush=True)
        for position, item in enumerate(pending, 1):
            name = item["sample"]
            clip = trimmed_dir / f"{name}.mp4"
            row = {
                "sample": name, "frames": int(item["num_frames"]),
                "reference": item["reference"], "gloss": item["gloss"],
                "prediction": "", "status": "error", "error": None,
                "protocol_id": protocol_id, "clip": str(clip), "clip_sha256": None,
                "window_frames": None, "sampled_frames": None, "valid_body_fraction": None,
                "body_scale": None, "pose_extract_ms": None, "pose_ms": None,
                "model_total_ms": None, "adapter_total_ms": None, "e2e_seconds": None,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                row["clip_sha256"] = sha256(clip)
                request = InferenceRequest(request_id=f"csldaily-video-{name}",
                                           sign_language="csl", expected_task_type="text")
                result = backend.recognize_file(clip, request)
                quality = result.quality or {}
                row["prediction"] = (result.prediction.text or "") if result.prediction else ""
                if result.status == "recognized" and row["prediction"].strip():
                    row["status"] = "ok"
                else:
                    row["status"] = "empty" if result.status == "recognized" else "rejected"
                    row["error"] = result.reason or "empty_text_output"
                    row["prediction"] = ""
                for key in ("window_frames", "sampled_frames", "valid_body_fraction", "body_scale",
                            "pose_extract_ms", "pose_ms", "model_total_ms", "adapter_total_ms"):
                    row[key] = quality.get(key)
            except KeyboardInterrupt:
                raise
            except Exception as error:  # noqa: BLE001 - recorded per sample
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
            row["e2e_seconds"] = time.perf_counter() - sample_started
            with predictions_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            existing[name] = row
            if position % 20 == 0 or position == len(pending):
                print(json.dumps({"event": "progress", "done": position, "pending": len(pending),
                                  "last": name, "last_status": row["status"],
                                  "eta_minutes": round((time.perf_counter() - started) / position
                                                       * (len(pending) - position) / 60.0, 1)},
                                 ensure_ascii=False), flush=True)
    finally:
        backend.close()

    rows = [existing[item["sample"]] for item in samples if item["sample"] in existing]
    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    import sacrebleu

    refs_raw = [character_separated(row["reference"]) for row in successful]
    hyps_raw = [character_separated(row["prediction"]) for row in successful]
    refs_norm = [normalize_text(row["reference"]) for row in successful]
    hyps_norm = [normalize_text(row["prediction"]) for row in successful]
    bleu = sacrebleu.corpus_bleu(hyps_raw, [refs_raw], tokenize="13a")
    bleu_raw = sacrebleu.corpus_bleu([row["prediction"] for row in successful],
                                     [[row["reference"] for row in successful]], tokenize="13a")
    chrf = sacrebleu.corpus_chrf([row["prediction"] for row in successful],
                                 [[row["reference"] for row in successful]]).score
    edits = [edit_similarity(a, b) for a, b in zip(refs_norm, hyps_norm)]
    rouges = [rouge_l_f1_char(a, b) for a, b in zip(refs_norm, hyps_norm)]
    exact = sum(a == b for a, b in zip(refs_norm, hyps_norm))
    e2e = [row["e2e_seconds"] for row in successful]
    pose_ms = [row["pose_ms"] for row in successful if row.get("pose_ms")]
    model_ms = [row["model_total_ms"] for row in successful if row.get("model_total_ms")]
    for row in rows:
        reference = normalize_text(row["reference"])
        prediction = normalize_text(row["prediction"])
        if row["status"] == "ok":
            row["char_edit_similarity"] = edit_similarity(reference, prediction)
            row["rouge_l_f1_char"] = rouge_l_f1_char(reference, prediction)
            row["exact_match"] = reference == prediction
        else:
            row["char_edit_similarity"] = None
            row["rouge_l_f1_char"] = None
            row["exact_match"] = False
        row["ground_truth"] = row.pop("reference")
    csv_path = write_comparison(rows, {"scope": "local CSL-Daily test videos", "created_at": "",
                                       "attempted": len(rows), "successful": len(successful),
                                       "failed_or_empty": len(failed),
                                       "metrics": {"bleu4_13a": bleu.score, "chrf": chrf,
                                                   "rouge_l_f1_char": statistics.fmean(rouges) if rouges else 0.0,
                                                   "mean_char_edit_similarity": statistics.fmean(edits) if edits else 0.0,
                                                   "exact_match_count": exact, "n": len(successful)},
                                       "latency": {"e2e_seconds_mean": statistics.fmean(e2e) if e2e else 0.0,
                                                   "e2e_seconds_p50": percentile(e2e, 50) or 0.0,
                                                   "e2e_seconds_p95": percentile(e2e, 95) or 0.0}},
                                      output_dir)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "Uni-Sign pinned CSL-Daily pose-only SLT run on local video sentence clips "
                 f"({len(rows)} samples); video end-to-end, not the released-pose protocol",
        "protocol_id": protocol_id, "protocol": protocol,
        "attempted": len(rows), "successful": len(successful), "failed_or_empty": len(failed),
        "failed_samples": [{"sample": row["sample"], "status": row["status"], "error": row["error"]}
                           for row in failed],
        "metrics": {
            "n": len(successful),
            "bleu4_13a": bleu.score,
            "bleu4_13a_definition": "SacreBLEU 13a on character-separated text after the upstream "
                                    "'fix mt5 tokenizer bug' rewrite",
            "bleu4_13a_raw_no_mt5_fix": bleu_raw.score,
            "bleu4_precisions": [round(value, 2) for value in bleu.precisions],
            "bp": round(bleu.bp, 4),
            "chrf": chrf,
            "rouge_l_f1_char": statistics.fmean(rouges) if rouges else 0.0,
            "mean_char_edit_similarity": statistics.fmean(edits) if edits else 0.0,
            "edit_similarity_p50": percentile(edits, 50),
            "exact_match_count": exact,
            "prediction_chars_total": sum(len(text) for text in hyps_norm),
            "reference_chars_total": sum(len(text) for text in refs_norm),
        },
        "latency": {
            "e2e_seconds_mean": statistics.fmean(e2e) if e2e else None,
            "e2e_seconds_p50": percentile(e2e, 50),
            "e2e_seconds_p95": percentile(e2e, 95),
            "pose_ms_mean": statistics.fmean(pose_ms) if pose_ms else None,
            "model_total_ms_mean": statistics.fmean(model_ms) if model_ms else None,
            "note": "Single process, exclusive GPU during measurement? See run log; alignment ran "
                    "concurrently for part of the sweep.",
        },
        "runtime": backend.runtime_info,
        "comparison_csv": str(csv_path),
        "comparison_markdown": str(output_dir / "comparison.md"),
        "per_sample": rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"event": "done", "successful": len(successful), "metrics": summary["metrics"],
                      "latency": summary["latency"]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(successful) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
