"""Evaluate a Uni-Sign checkpoint on the released CSL-Daily pose archive.

Why this exists
---------------
The repository has no CSL-Daily videos, but the authors ship
``csl_daily_pose_format.zip`` and their own test predictions/references
(``out/eval_csl_daily_pose_only/test_tmp_{pres,refs}.txt``, 1,176 lines each).
Running the pinned checkpoint on those pose files gives the first *in-domain*
check on this machine, and it settles an open question: which pose
normalization the CSL-Daily weights expect.

Normalization probe
-------------------
The released CSL-Daily pkl stores normalized coordinates but not the source
frame size, so ``x/W, y/H`` cannot be reconstructed exactly. ``--rescale-x``
multiplies x before the body-bbox step, which is exactly the knob that
distinguishes the two conventions:

* ``1.0``            -> treat the file as isotropic (short-side) normalized;
* ``H/W`` (e.g. 0.56) -> emulate ``x/W, y/H`` for a 16:9 source.

The correct value is the one that reproduces the authors' own BLEU
(25.70 recomputed / 25.61 in the paper). This is a probe, not a claim that the
released archive uses one convention: the frame size is unknown.

Frame sampling differs from upstream (deterministic ``linspace`` vs
``random.sample``), so agreement is expected within about ±0.5 BLEU.
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
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig  # noqa: E402
from slbridge.uni_sign_preprocess import prepare_pose_window  # noqa: E402

from evaluate_unisign_cecsl import (  # noqa: E402
    edit_similarity,
    normalize_text,
    rouge_l_f1_char,
    write_json,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_reference_file(path):
    """Lines look like ``sample: S000020_P0004_T00, ground-truth: 他 今 年 四 岁 。``"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        marker = ", ground-truth: "
        if marker not in line:
            raise ValueError(f"Unexpected reference line: {line[:80]}")
        head, reference = line.split(marker, 1)
        sample = head.split("sample: ", 1)[-1].strip()
        rows.append({"sample": sample, "reference": reference.strip()})
    return rows


def load_pose(path):
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="In-domain Uni-Sign evaluation on the released CSL-Daily poses")
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True,
                        help="Upstream test_tmp_refs.txt (1,176 lines)")
    parser.add_argument("--upstream-pres", type=Path,
                        help="Upstream test_tmp_pres.txt for a paired comparison")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models/external/uni_sign/csl_daily_pose_only_slt.pth")
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rescale-x", type=float, default=1.0,
                        help="Multiply x before the body-bbox step (H/W emulates x/W, y/H)")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    records = parse_reference_file(args.refs)
    if args.limit is not None:
        records = records[:args.limit]
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = file_sha256(checkpoint)

    config = UniSignConfig.from_bundle(args.bundle)
    config = replace(config, checkpoint=checkpoint, checkpoint_bytes=checkpoint.stat().st_size,
                     checkpoint_sha256=checkpoint_sha256,
                     run_id=f"uni-sign-csldaily-{checkpoint.stem}")
    max_frames = int(config.runtime.get("max_frames", 256))
    runtime = TorchUniSignRuntime(config, device=args.device,
                                  dtype="bfloat16" if args.device == "cuda" else "float32")

    protocol = {
        "schema_version": 1,
        "evaluation": "CSL-Daily test poses, released archive",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "pose_dir": str(args.pose_dir.resolve()),
        "refs": str(args.refs.resolve()), "refs_sha256": file_sha256(args.refs),
        "rescale_x": args.rescale_x, "max_frames": max_frames,
        "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "frame_sampling": "deterministic linspace (upstream uses random.sample)",
        "device": args.device,
        "samples": [record["sample"] for record in records],
    }
    protocol_path = output_dir / "protocol.json"
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                            ensure_ascii=False).encode("utf-8")).hexdigest()
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise SystemExit("Protocol changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    print(json.dumps({"event": "protocol", "protocol_id": protocol_id,
                      "samples": len(records), "rescale_x": args.rescale_x,
                      "checkpoint_sha256": checkpoint_sha256}, ensure_ascii=False), flush=True)

    existing = {}
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("sample"):
                    if row.get("protocol_id") != protocol_id:
                        raise SystemExit("Prediction record protocol does not match")
                    existing[row["sample"]] = row
    done = {key for key, row in existing.items()
            if row.get("status") == "ok" and row.get("prediction", "").strip()}
    pending = [record for record in records if record["sample"] not in done]
    print(json.dumps({"event": "start", "pending": len(pending)}, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    new_count = 0
    try:
        if pending:
            runtime.load()
        for record in pending:
            pkl = args.pose_dir / f"{record['sample']}.pkl"
            row = {
                "sample": record["sample"], "reference": record["reference"],
                "prediction": "", "status": "error", "error": None,
                "protocol_id": protocol_id, "rescale_x": args.rescale_x,
                "frames_total": None, "frames_used": None, "sampled_frames": None,
                "valid_body_fraction": None, "body_scale": None,
                "model_forward_ms": None, "generation_ms": None, "decode_ms": None,
                "model_total_ms": None, "e2e_seconds": None,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                if not pkl.is_file():
                    raise FileNotFoundError(f"missing pose: {pkl}")
                keypoints, scores, info = load_pose(pkl)
                row.update({"frames_total": info["frames_total"],
                            "frames_used": info["frames_used"]})
                points = keypoints.copy()
                if args.rescale_x != 1.0:
                    points[..., 0] *= args.rescale_x
                pose = prepare_pose_window(points, scores, max_frames=max_frames)
                row["sampled_frames"] = pose.quality.get("sampled_frames")
                row["valid_body_fraction"] = pose.quality.get("valid_body_fraction")
                row["body_scale"] = pose.quality.get("body_scale")
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
            existing[record["sample"]] = row
            new_count += 1
            if new_count % 50 == 0:
                print(json.dumps({"event": "progress", "done": new_count,
                                  "pending": len(pending),
                                  "eta_minutes": round((time.perf_counter() - started)
                                                       / new_count
                                                       * (len(pending) - new_count) / 60.0, 1)},
                                 ensure_ascii=False), flush=True)

        rows = [existing[record["sample"]] for record in records if record["sample"] in existing]
        successful = [row for row in rows if row.get("status") == "ok"]
        failed = [row for row in rows if row.get("status") != "ok"]

        import sacrebleu

        refs_raw = [row["reference"] for row in successful]
        hyps_raw = [row["prediction"] for row in successful]
        refs_norm = [normalize_text(text) for text in refs_raw]
        hyps_norm = [normalize_text(text) for text in hyps_raw]
        # Upstream fine_tuning.py:282-284 rewrites every prediction as
        # ' '.join(list(text.replace(" ", ""))) ("fix mt5 tokenizer bug") before
        # scoring, and the references are character-separated. Without the same
        # rewrite BLEU collapses to ~0 while chrF/char-ROUGE stay normal, so the
        # summary records both forms and marks the protocol-correct one as primary.
        hyps_fixed = [" ".join(list(text.replace(" ", "").replace("\n", ""))) for text in hyps_raw]
        bleu = sacrebleu.corpus_bleu(hyps_fixed, [refs_raw], tokenize="13a")
        bleu_raw = sacrebleu.corpus_bleu(hyps_raw, [refs_raw], tokenize="13a")
        chrf = sacrebleu.corpus_chrf(hyps_raw, [refs_raw]).score
        edits = [edit_similarity(a, b) for a, b in zip(refs_norm, hyps_norm)]
        exact = sum(a == b for a, b in zip(refs_norm, hyps_norm))

        comparison = None
        if args.upstream_pres and args.upstream_pres.is_file():
            upstream = {}
            for line in args.upstream_pres.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                marker = ", prediction: "
                if marker not in line:
                    continue
                head, prediction = line.split(marker, 1)
                upstream[head.split("sample: ", 1)[-1].strip()] = prediction.strip()
            paired = [(row, upstream[row["sample"]]) for row in successful
                      if row["sample"] in upstream]
            if paired:
                upstream_bleu = sacrebleu.corpus_bleu(
                    [pair[1] for pair in paired], [[pair[0]["reference"] for pair in paired]],
                    tokenize="13a")
                identical = sum(pair[0]["prediction"] == pair[1] for pair in paired)
                identical_after_fix = sum(
                    " ".join(list(pair[0]["prediction"].replace(" ", "").replace("\n", "")))
                    == pair[1] for pair in paired)
                deltas = [edit_similarity(normalize_text(pair[0]["reference"]),
                                          normalize_text(pair[0]["prediction"]))
                          - edit_similarity(normalize_text(pair[0]["reference"]),
                                            normalize_text(pair[1]))
                          for pair in paired]
                comparison = {
                    "paired_n": len(paired),
                    "this_bleu4": bleu.score,
                    "upstream_bleu4": upstream_bleu.score,
                    "identical_prediction_count_raw_format": identical,
                    "identical_prediction_count_after_mt5_fix": identical_after_fix,
                    "mean_edit_delta_this_minus_upstream": statistics.mean(deltas),
                    "note": "Upstream uses random frame sampling; agreement is not bit-exact.",
                }

        summary = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "evaluation": "Uni-Sign on released CSL-Daily test poses (in-domain)",
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
            "rescale_x": args.rescale_x, "protocol_id": protocol_id,
            "attempted": len(rows), "successful": len(successful),
            "failed_or_empty": len(failed), "failed_samples": failed,
            "metrics": {
                "n": len(successful),
                "bleu4_13a": bleu.score,
                "bleu4_13a_definition": "sacrebleu tokenize=13a after the upstream 'fix mt5 tokenizer bug' rewrite",
                "bleu4_13a_raw_no_mt5_fix": bleu_raw.score,
                "bleu4_13a_raw_brevity_penalty": round(bleu_raw.bp, 6),
                "protocol_trap_note": ("bleu4_13a_raw_no_mt5_fix is only for demonstrating the "
                                       "format trap; never quote it as accuracy"),
                "bleu4_precisions": [round(value, 2) for value in bleu.precisions],
                "bp": round(bleu.bp, 4),
                "chrf": chrf,
                "rouge_l_f1_char": statistics.mean(
                    rouge_l_f1_char(a, b) for a, b in zip(refs_norm, hyps_norm))
                if refs_norm else 0.0,
                "mean_char_edit_similarity": statistics.mean(edits) if edits else 0.0,
                "edit_similarity_p50": percentile(edits, 50),
                "exact_match_count": exact,
                "prediction_chars_total": sum(len(text) for text in hyps_norm),
                "reference_chars_total": sum(len(text) for text in refs_norm),
            },
            "upstream_reference": {"recomputed_bleu4_13a": 25.70,
                                   "paper_bleu4": 25.61, "paper_rouge": 54.92},
            "paired_comparison_vs_upstream": comparison,
            "latency_model_only": {
                "model_total_ms_mean": statistics.mean(
                    [row["model_total_ms"] for row in successful if row["model_total_ms"]]),
                "e2e_seconds_mean": statistics.mean(
                    [row["e2e_seconds"] for row in successful]),
                "note": "Poses come from disk: model-only latency, no RTMPose.",
            },
            "runtime": runtime.runtime_info,
            "examples": successful[:8],
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"event": "done", "successful": len(successful),
                          "metrics": summary["metrics"],
                          "comparison": comparison}, ensure_ascii=False, indent=2), flush=True)
        return 0 if len(successful) == len(records) else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())


