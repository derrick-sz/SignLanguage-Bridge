"""task-12 part 1 -- Uni-Sign frontend switch comparison on CSL-Daily test.

Compares four frontend configurations of the **pinned CSL-Daily video path**
(``UniSignBackend.recognize_file``: trimmed mp4 -> RTMPose wholebody133 -> mT5):

    baseline             pose_sample_only=False, pose_detect_stride=1
    sample_only          pose_sample_only=True,  pose_detect_stride=1
    detect_stride5       pose_sample_only=False, pose_detect_stride=5
    sample_only_stride5  pose_sample_only=True,  pose_detect_stride=5

Everything else is held fixed: same checkpoint, dtype (bf16), beams (4),
max_frames (256), pose normalization (isotropic short side), orientation
(preserve), the same 50 pilot samples (10 signers x 5, signer-stratified) and the
same upstream reference file.  Only the two runtime switches change -- the two
switches were previously validated on CSL-News / CE-CSL only, never on
CSL-Daily test.

Reports per arm: BLEU-4 (upstream reference, upstream mt5-fix protocol), chrF,
character edit similarity, exact match; paired signer bootstrap (1000x) of
Delta vs baseline with 95% CI; latency (pose_ms / model_total_ms / e2e P50/P95);
pose-skip statistics (extractor calls, skipped frames, sampled frames).

Quality gate used for the recommendation:
    Delta edit similarity >= -0.005  AND  95% CI lower bound > -0.01
for switching a switch on by default.  Production defaults in the model bundle
are NOT modified; this script only produces evidence.

Usage::

    .conda-env\\python.exe experiments/p1/e04/opt_unisign_frontend.py run \
        --run-dir experiments/p1/e04/opt-runs/unisign-frontend-<stamp> \
        --samples 50 --arms baseline,sample_only,detect_stride5,sample_only_stride5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1.e04 import perf_common as common  # noqa: E402

UPSTREAM_REFS = common.ROOT / "models/external/uni_sign_source/out/eval_csl_daily_pose_only/test_tmp_refs.txt"

ARMS: dict[str, dict[str, Any]] = {
    "baseline": {"pose_sample_only": False, "pose_detect_stride": 1},
    "sample_only": {"pose_sample_only": True, "pose_detect_stride": 1},
    "detect_stride5": {"pose_sample_only": False, "pose_detect_stride": 5},
    "sample_only_stride5": {"pose_sample_only": True, "pose_detect_stride": 5},
}
# The two switches previously validated off-domain; CSL-Daily test is the new domain.
RETRYABLE = ("out of memory", "cuda error", "memory allocation failure", "acceleratorerror")
RETRY_SLEEP_SECONDS = 20.0
LOAD_RETRIES = 2


# ---------------------------------------------------------------------------
# text metrics (identical definitions to scripts/evaluate_unisign_*)
# ---------------------------------------------------------------------------
def normalize_text(text: Any) -> str:
    import unicodedata

    if text is None:
        return ""
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def character_separated(text: str) -> str:
    return " ".join(list(normalize_text(text)))


def edit_similarity(reference: str, hypothesis: str) -> float:
    a, b = normalize_text(reference), normalize_text(hypothesis)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def corpus_scores(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """pairs = [(reference, hypothesis)]; corpus BLEU uses the upstream mt5-fix protocol."""
    import sacrebleu

    refs_raw = [character_separated(reference) for reference, _ in pairs]
    hyps_raw = [character_separated(hypothesis) for _, hypothesis in pairs]
    hyps_fixed = [" ".join(list(normalize_text(hypothesis))) for _, hypothesis in pairs]
    bleu = sacrebleu.corpus_bleu(hyps_fixed, [refs_raw], tokenize="13a")
    chrf = sacrebleu.corpus_chrf(hyps_raw, [refs_raw]).score
    edits = [edit_similarity(reference, hypothesis) for reference, hypothesis in pairs]
    exact = sum(normalize_text(reference) == normalize_text(hypothesis) for reference, hypothesis in pairs)
    return {
        "n": len(pairs),
        "bleu4_13a": round(bleu.score, 4),
        "bleu4_precisions": [round(value, 2) for value in bleu.precisions],
        "bp": round(bleu.bp, 4),
        "chrf": round(chrf, 4),
        "mean_char_edit_similarity": round(statistics.fmean(edits), 4) if edits else None,
        "exact_match_count": exact,
        "empty_prediction_count": sum(1 for _, hypothesis in pairs if not normalize_text(hypothesis)),
    }


def read_upstream_references(path: Path = UPSTREAM_REFS) -> dict[str, str]:
    references: dict[str, str] = {}
    marker = ", ground-truth: "
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or marker not in line:
            continue
        head, reference = line.split(marker, 1)
        references[head.split("sample: ")[-1].strip()] = reference.strip()
    return references


# ---------------------------------------------------------------------------
# paired signer bootstrap
# ---------------------------------------------------------------------------
def paired_signer_bootstrap(
    baseline_rows: dict[str, dict[str, Any]],
    arm_rows: dict[str, dict[str, Any]],
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Resample signers with replacement; Delta = arm - baseline on the same samples."""
    import random

    rng = random.Random(seed)
    signers: dict[str, list[str]] = {}
    for sample_id, row in baseline_rows.items():
        if sample_id in arm_rows and row.get("status") == "ok" and arm_rows[sample_id].get("status") == "ok":
            signers.setdefault(row["signer_id"], []).append(sample_id)
    signer_names = sorted(signers)
    if not signer_names:
        return {"available": False}
    deltas_edit: list[float] = []
    deltas_bleu: list[float] = []
    deltas_exact: list[float] = []
    for _ in range(iterations):
        chosen = [rng.choice(signer_names) for _ in signer_names]
        sample_ids = [sample_id for signer in chosen for sample_id in signers[signer]]
        base_pairs = [(baseline_rows[s]["reference"], baseline_rows[s]["prediction"]) for s in sample_ids]
        arm_pairs = [(arm_rows[s]["reference"], arm_rows[s]["prediction"]) for s in sample_ids]
        base = corpus_scores(base_pairs)
        arm = corpus_scores(arm_pairs)
        deltas_edit.append(arm["mean_char_edit_similarity"] - base["mean_char_edit_similarity"])
        deltas_bleu.append(arm["bleu4_13a"] - base["bleu4_13a"])
        deltas_exact.append(arm["exact_match_count"] - base["exact_match_count"])
    def ci(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {
            "mean": round(statistics.fmean(values), 5),
            "ci_low": round(ordered[int(0.025 * (len(ordered) - 1))], 5),
            "ci_high": round(ordered[int(0.975 * (len(ordered) - 1))], 5),
        }
    return {
        "available": True,
        "iterations": iterations,
        "signers": signer_names,
        "delta_edit_similarity": ci(deltas_edit),
        "delta_bleu4": ci(deltas_bleu),
        "delta_exact_match_count": ci(deltas_exact),
    }


def select_samples(*, sample_set: str, count: int, min_frames: int) -> list[dict[str, Any]]:
    """pilot = signer-stratified pilot-50; long = manifest_1086 clips > min_frames (signer-stratified)."""
    if sample_set == "pilot":
        return common.attach_fps(common.select_pilot_samples(count))
    if sample_set != "long":
        raise SystemExit(f"unknown --sample-set {sample_set}")
    rows = common.read_csv_rows(common.P1 / "datasets" / "manifest_1086.csv")
    candidates = [
        row for row in rows
        if row.get("split") == "test" and row.get("trimmed_video_present") == "True"
        and row.get("frames") and int(row["frames"]) > min_frames
    ]
    by_signer: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        by_signer.setdefault(row["signer_id"], []).append(row)
    for signer in by_signer:
        by_signer[signer].sort(key=lambda item: int(item["frames"]))
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < count and any(depth < len(items) for items in by_signer.values()):
        for signer in sorted(by_signer):
            if len(picked) >= count:
                break
            items = by_signer[signer]
            if depth < len(items):
                row = items[depth]
                sample = common.build_sample(row)
                missing = [key for key in ("trimmed", "npz") if not Path(sample[key]).is_file()]
                if missing:
                    continue
                picked.append(sample)
        depth += 1
    return common.attach_fps(picked)


# ---------------------------------------------------------------------------
# arm runner
# ---------------------------------------------------------------------------
def build_backend(arm: str, switches: dict[str, Any]) -> tuple[Any, float]:
    from dataclasses import replace

    from slbridge.runtime_options import RuntimeOptions
    from slbridge.uni_sign import UniSignBackend, UniSignConfig

    base_config = UniSignConfig.from_bundle(common.UNISIGN_BUNDLE)
    # UniSignConfig is frozen: build a new config with an independent runtime dict
    # so the bundle manifest / production defaults are never mutated.
    config = replace(base_config, runtime={
        **base_config.runtime,
        "pose_sample_only": bool(switches["pose_sample_only"]),
        "pose_detect_stride": int(switches["pose_detect_stride"]),
    })
    last_error: BaseException | None = None
    for attempt in range(LOAD_RETRIES + 1):
        backend = UniSignBackend(config, RuntimeOptions(device="cuda"))
        started = perf_counter()
        try:
            backend.load_model()
            return backend, perf_counter() - started
        except BaseException as error:  # noqa: BLE001 - allocation retry rule (pitfall §7)
            last_error = error
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                pass
            if attempt < LOAD_RETRIES and any(marker in str(error).lower() for marker in RETRYABLE):
                print(f"[retry] {arm} load failed: {type(error).__name__}; retry "
                      f"{attempt + 1}/{LOAD_RETRIES} after {RETRY_SLEEP_SECONDS:.0f}s", flush=True)
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            raise
    raise last_error if last_error is not None else RuntimeError("backend load failed")


def run_arm(*, arm: str, backend: Any, samples: list[dict[str, Any]],
            references: dict[str, str], run_dir: Path, run_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from slbridge.runtime_contracts import InferenceRequest

    rows: dict[str, dict[str, Any]] = {}
    failures = 0
    for index, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        attempt = 0
        while True:
            attempt += 1
            request = InferenceRequest(
                request_id=f"opt-frontend-{arm}-{sample_id}-{attempt}",
                deadline_monotonic=time.monotonic() + 300.0,
            )
            started = perf_counter()
            try:
                result = backend.recognize_file(sample["trimmed"], request)
                wall_ms = (perf_counter() - started) * 1000
                quality = result.quality or {}
                text = (result.prediction.text
                        if result.prediction is not None and result.task_type == "text" else "")
                row = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "arm": arm,
                    "sample_id": sample_id,
                    "signer_id": sample["signer_id"],
                    "status": "ok" if result.status == "recognized" else result.status,
                    "prediction": text,
                    "reference": references.get(sample_id, ""),
                    "e2e_wall_ms": round(wall_ms, 3),
                    "pose_ms": round(float(quality["pose_ms"]), 3)
                    if isinstance(quality.get("pose_ms"), (int, float)) else None,
                    "pose_extract_ms": quality.get("pose_extract_ms"),
                    "pose_preprocess_ms": quality.get("pose_preprocess_ms"),
                    "model_forward_ms": quality.get("model_forward_ms"),
                    "generation_ms": quality.get("generation_ms"),
                    "decode_ms": quality.get("decode_ms"),
                    "model_total_ms": quality.get("model_total_ms"),
                    "adapter_total_ms": quality.get("adapter_total_ms"),
                    "window_frames": quality.get("window_frames"),
                    "sampled_frames": quality.get("sampled_frames"),
                    "source_frames": quality.get("source_frames"),
                    "skipped_frames": quality.get("skipped_frames"),
                    "pose_sample_only": quality.get("pose_sample_only"),
                    "pose_new_frames": quality.get("pose_new_frames"),
                    "pose_cache_hit_frames": quality.get("pose_cache_hit_frames"),
                    "valid_body_fraction": quality.get("valid_body_fraction"),
                    "attempts": attempt,
                    "reason": result.reason,
                }
                rows[sample_id] = row
                break
            except BaseException as error:  # noqa: BLE001 - recorded as evidence
                if attempt <= 2 and any(marker in str(error).lower() for marker in RETRYABLE):
                    print(f"[retry] {arm}/{sample_id} inference failed: {type(error).__name__}; "
                          f"retry {attempt}/2 after {RETRY_SLEEP_SECONDS:.0f}s", flush=True)
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue
                failures += 1
                rows[sample_id] = {
                    "schema_version": 1, "run_id": run_id, "arm": arm,
                    "sample_id": sample_id, "signer_id": sample["signer_id"],
                    "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "reference": references.get(sample_id, ""), "attempts": attempt,
                }
                break
        if (index + 1) % 10 == 0:
            print(json.dumps({"event": "progress", "arm": arm, "done": index + 1,
                              "total": len(samples)}), flush=True)
    ok_rows = {key: row for key, row in rows.items() if row.get("status") == "ok"}
    pairs = [(row["reference"], row["prediction"]) for row in ok_rows.values()]
    scores = corpus_scores(pairs) if pairs else {}
    def stats(key: str) -> dict[str, Any]:
        values = [row.get(key) for row in ok_rows.values() if isinstance(row.get(key), (int, float))]
        return {"p50": common.percentile(values, 50), "p95": common.percentile(values, 95),
                "mean": round(statistics.fmean(values), 2) if values else None, "count": len(values)}
    sampled = [row.get("sampled_frames") for row in ok_rows.values()]
    source = [row.get("source_frames") or row.get("window_frames") for row in ok_rows.values()]
    skipped = [
        (src - smp) if isinstance(src, (int, float)) and isinstance(smp, (int, float)) else None
        for src, smp in zip(source, sampled)
    ]
    summary = {
        "arm": arm,
        "attempted": len(samples),
        "successful": len(ok_rows),
        "failed": failures,
        "quality": scores,
        "latency": {
            "pose_ms": stats("pose_ms"),
            "pose_extract_ms": stats("pose_extract_ms"),
            "model_total_ms": stats("model_total_ms"),
            "adapter_total_ms": stats("adapter_total_ms"),
            "e2e_wall_ms": stats("e2e_wall_ms"),
        },
        "pose_skips": {
            "source_frames_mean": round(statistics.fmean([v for v in source if v is not None]), 2)
            if any(v is not None for v in source) else None,
            "sampled_frames_mean": round(statistics.fmean([v for v in sampled if v is not None]), 2)
            if any(v is not None for v in sampled) else None,
            "skipped_frames_mean": round(statistics.fmean([v for v in skipped if v is not None]), 2)
            if any(v is not None for v in skipped) else None,
            "skipped_frames_total": int(sum(v for v in skipped if v is not None)),
            "extractor_calls_mean": round(statistics.fmean(
                [row["pose_new_frames"] + row["pose_cache_hit_frames"]
                 for row in ok_rows.values()
                 if isinstance(row.get("pose_new_frames"), (int, float))
                 and isinstance(row.get("pose_cache_hit_frames"), (int, float))]), 2)
            if any(isinstance(row.get("pose_new_frames"), (int, float)) for row in ok_rows.values()) else None,
        },
    }
    common.write_jsonl(run_dir / f"predictions-{arm}.jsonl", rows.values())
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--sample-set", choices=("pilot", "long"), default="pilot",
                        help="pilot: pilot_50 signer-stratified; long: manifest_1086 clips with frames > --min-frames")
    parser.add_argument("--min-frames", type=int, default=256)
    parser.add_argument("--arms", default="baseline,sample_only,detect_stride5,sample_only_stride5")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--max-foreign-vram-mb", type=float, default=4500.0)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.run != "run":
        parser.error("only `run` is supported")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"unisign-frontend-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arm_names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    baseline_gpu = common.nvidia_smi()
    if baseline_gpu is None:
        raise SystemExit("nvidia-smi unavailable")
    device = baseline_gpu["devices"][0]
    if device["memory_used_mb"] > args.max_foreign_vram_mb and not args.allow_shared_gpu:
        raise SystemExit(f"GPU busy: {device['memory_used_mb']:.0f} MB used; pass --allow-shared-gpu")
    samples = select_samples(sample_set=args.sample_set, count=args.samples,
                             min_frames=args.min_frames)
    references = read_upstream_references()
    missing_refs = [sample["sample_id"] for sample in samples if sample["sample_id"] not in references]
    if missing_refs:
        raise SystemExit(f"missing upstream references for: {missing_refs}")

    manifest = common.base_manifest(run_id, "E04-opt-frontend", purpose=(
        "CSL-Daily test paired comparison of Uni-Sign pose_sample_only / "
        "pose_detect_stride frontend switches (quality + latency + skips)"))
    manifest.update({
        "model_assets": common.model_assets_manifest(),
        "environment": common.environment_fingerprint(),
        "gpu_baseline": baseline_gpu,
        "ram": common.memory_snapshot(),
        "samples": [sample["sample_id"] for sample in samples],
        "sample_set": args.sample_set,
        "min_frames": args.min_frames,
        "signers": sorted({sample["signer_id"] for sample in samples}),
        "arms": {name: ARMS[name] for name in arm_names},
        "references": {"path": str(UPSTREAM_REFS), "sha256": common.sha256_file(UPSTREAM_REFS),
                       "protocol": "upstream test_tmp_refs.txt; character-separated; BLEU via mt5-fix rewrite + 13a"},
        "held_fixed": {"checkpoint": "csl_daily_pose_only_slt.pth", "dtype": "bfloat16",
                       "num_beams": 4, "max_new_tokens": 100, "max_frames": 256,
                       "pose_normalization": "isotropic_short_side", "pose_orientation": "preserve"},
        "quality_gate": {"delta_edit_min": -0.005, "ci_low_min": -0.01,
                         "note": "switch may be recommended off-by-default only if the gate passes"},
        "commands": [common.command_line()],
    })
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "baseline", "gpu": device, "samples": len(samples),
                      "arms": arm_names}, ensure_ascii=False), flush=True)

    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        print(json.dumps({"event": "arm_start", "arm": arm, "switches": ARMS[arm]}), flush=True)
        backend = None
        try:
            backend, load_seconds = build_backend(arm, ARMS[arm])
            rows, summary = run_arm(arm=arm, backend=backend, samples=samples,
                                    references=references, run_dir=run_dir, run_id=run_id)
            summary["load_seconds"] = round(load_seconds, 2)
            summary["runtime_info"] = backend.runtime_info
            arm_rows[arm] = rows
            arm_summaries[arm] = summary
            print(json.dumps({"event": "arm_done", "arm": arm,
                              "successful": summary["successful"], "failed": summary["failed"],
                              "quality": summary["quality"],
                              "e2e_p50_ms": summary["latency"]["e2e_wall_ms"]["p50"],
                              "pose_p50_ms": summary["latency"]["pose_ms"]["p50"],
                              "skipped_frames_mean": summary["pose_skips"]["skipped_frames_mean"]},
                             ensure_ascii=False), flush=True)
        finally:
            if backend is not None:
                backend.close()

    baseline_rows = arm_rows.get("baseline")
    comparisons: dict[str, Any] = {}
    if baseline_rows:
        for arm in arm_names:
            if arm == "baseline" or arm not in arm_rows:
                continue
            ok_base = {key: row for key, row in baseline_rows.items() if row.get("status") == "ok"}
            ok_arm = {key: row for key, row in arm_rows[arm].items() if row.get("status") == "ok"}
            shared = sorted(set(ok_base) & set(ok_arm))
            pairs_base = [(ok_base[key]["reference"], ok_base[key]["prediction"]) for key in shared]
            pairs_arm = [(ok_arm[key]["reference"], ok_arm[key]["prediction"]) for key in shared]
            base_scores = corpus_scores(pairs_base)
            arm_scores = corpus_scores(pairs_arm)
            point = {
                "delta_edit_similarity": round(arm_scores["mean_char_edit_similarity"]
                                               - base_scores["mean_char_edit_similarity"], 5),
                "delta_bleu4": round(arm_scores["bleu4_13a"] - base_scores["bleu4_13a"], 4),
                "delta_chrf": round(arm_scores["chrf"] - base_scores["chrf"], 4),
                "delta_exact_match_count": arm_scores["exact_match_count"] - base_scores["exact_match_count"],
                "identical_predictions": sum(
                    normalize_text(ok_base[key]["prediction"]) == normalize_text(ok_arm[key]["prediction"])
                    for key in shared),
                "shared_samples": len(shared),
                "baseline_scores": base_scores,
                "arm_scores": arm_scores,
            }
            bootstrap = paired_signer_bootstrap(ok_base, ok_arm,
                                                iterations=args.bootstrap_iterations)
            gate_edit = (point["delta_edit_similarity"] >= -0.005)
            ci_low = bootstrap.get("delta_edit_similarity", {}).get("ci_low")
            gate_ci = (ci_low is not None) and (ci_low > -0.01)
            comparisons[arm] = {"point": point, "bootstrap": bootstrap,
                                "quality_gate_pass": bool(gate_edit and gate_ci),
                                "latency_delta_p50_ms": round(
                                    (arm_summaries[arm]["latency"]["e2e_wall_ms"]["p50"] or 0)
                                    - (arm_summaries["baseline"]["latency"]["e2e_wall_ms"]["p50"] or 0), 1)
                                if arm_summaries.get("baseline") else None}
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": common.utc_now(),
        "samples": len(samples),
        "signers": sorted({sample["signer_id"] for sample in samples}),
        "arms": arm_summaries,
        "comparisons_vs_baseline": comparisons,
        "gpu_baseline": baseline_gpu,
    }
    common.write_json(run_dir / "metrics.json", metrics)
    manifest.update({"run_status": "PASS" if all(s["failed"] == 0 for s in arm_summaries.values()) else "PARTIAL",
                     "finished_at": common.utc_now(),
                     "output_files": sorted(path.name for path in run_dir.iterdir() if path.is_file())})
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"event": "done", "run_dir": str(run_dir),
                      "comparisons": {arm: {
                          "delta_edit": cmp["point"]["delta_edit_similarity"],
                          "ci_low": cmp["bootstrap"].get("delta_edit_similarity", {}).get("ci_low"),
                          "delta_bleu4": cmp["point"]["delta_bleu4"],
                          "identical_predictions": cmp["point"]["identical_predictions"],
                          "gate_pass": cmp["quality_gate_pass"]}
                          for arm, cmp in comparisons.items()}},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
