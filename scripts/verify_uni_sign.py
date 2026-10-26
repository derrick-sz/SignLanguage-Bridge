"""Run the real Uni-Sign adapter on a video and write reproducible evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slbridge.runtime_contracts import InferenceRequest  # noqa: E402
from slbridge.runtime_options import RuntimeOptions  # noqa: E402
from slbridge.uni_sign import UniSignAdapter, UniSignConfig  # noqa: E402


def edit_similarity(reference: str, hypothesis: str) -> float:
    """Character-level normalized edit similarity for Chinese smoke evidence."""
    a, b = reference, hypothesis
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(1, len(a), len(b))


def char_bleu(reference: str, hypothesis: str) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    try:
        score = sacrebleu.corpus_bleu([hypothesis], [[reference]], tokenize="char")
        return float(score.score)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Uni-Sign checkpoint loading and real-video inference")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/external/uni_sign/model_bundle")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--reference-domain", default="unspecified")
    parser.add_argument("--pose-orientation", choices=("preserve", "mirror"),
                        help="Explicit correction for known mirrored input; default comes from the bundle")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.runs <= 10:
        parser.error("--runs must be in [1, 10]")
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    options = RuntimeOptions(device=args.device, device_id=args.device_id)
    config = UniSignConfig.from_bundle(args.bundle)
    if args.pose_orientation is not None:
        config.runtime["pose_orientation"] = args.pose_orientation
    started = time.perf_counter()
    adapter = UniSignAdapter(config, options)
    runs = []
    try:
        adapter.warmup()
        load_seconds = time.perf_counter() - started
        for index in range(1, args.runs + 1):
            request = InferenceRequest(f"verify-uni-sign-{index}")
            run_started = time.perf_counter()
            result, timing = adapter.infer(args.video, request, input_kind="file")
            wall_seconds = time.perf_counter() - run_started
            runs.append({
                "run": index,
                "wall_seconds": wall_seconds,
                "timing_ms": timing,
                "result": result.to_dict(),
            })
        runtime = adapter.runtime_info
        model_info = adapter.recognizer.get_model_info()
    finally:
        adapter.close()
    texts = [run["result"].get("prediction", {}).get("text") if run["result"].get("prediction") else None
             for run in runs]
    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "video": str(args.video.resolve()),
        "bundle": str(args.bundle.resolve()),
        "checkpoint": {
            "path": str(config.checkpoint),
            "bytes": config.checkpoint_bytes,
            "sha256": config.checkpoint_sha256,
            "revision": config.model_revision,
        },
        "source_commit": config.source_commit,
        "mt5_revision": config.mt5_revision,
        "load_seconds": load_seconds,
        "runtime": runtime,
        "model_info": model_info,
        "runs": runs,
        "reproducibility": {
            "texts": texts,
            "identical": len(set(texts)) == 1 and texts[0] is not None,
        },
        "quality_boundary": {
            "task": "CSL-Daily pose-only sign-language translation",
            "reference": args.reference,
            "reference_domain": args.reference_domain,
            "metric": None,
            "metric_reason": "Single-video smoke evidence; not a corpus-level BLEU/ROUGE result.",
        },
    }
    if args.reference is not None:
        hypothesis = texts[-1] or ""
        output["quality_boundary"]["metric"] = {
            "char_bleu": char_bleu(args.reference, hypothesis),
            "char_edit_similarity": edit_similarity(args.reference, hypothesis),
            "exact_match": hypothesis == args.reference,
            "samples": 1,
        }
    output_path = (args.output or ROOT / "artifacts/unisign-verification" /
                   f"verify-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "load_seconds": load_seconds,
                      "texts": texts, "identical": len(set(texts)) == 1,
                      "runtime": workspace_summary(runtime)}, ensure_ascii=False, indent=2))
    return 0


def workspace_summary(runtime: dict) -> dict:
    keys = ("device", "dtype", "pose_device", "pose_detection_providers",
            "pose_estimation_providers", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb")
    return {key: runtime.get(key) for key in keys}


if __name__ == "__main__":
    raise SystemExit(main())
