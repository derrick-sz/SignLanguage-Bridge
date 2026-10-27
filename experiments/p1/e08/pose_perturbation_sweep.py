"""E08：姿态扰动鲁棒性扫描（训练侧探索的第一步）。

背景：P-1.5 指出模型对"字面/表面新颖度"敏感（train 99.5% 完全匹配 / test 1.7%），
而训练侧真正的瓶颈是**数据**：本地只有 657 条 train pose（占 18,401 的 3.6%），
其余训练视频与 pose 都不存在，无法做正常规模的微调。

因此在训练前先回答一个更便宜、更关键的问题：**已发布 checkpoint 对哪些输入扰动最脆弱？**
这决定"若将来有训练预算，应优先用哪种数据增广"。

条件（同一批样本、同 seed）：
- baseline
- speed_x1.25 / speed_x0.80（抽帧/插帧，模拟语速变化）
- dropout_10 / dropout_25（随机丢帧，模拟遮挡/跟踪丢失）
- jitter_0.01 / jitter_0.03（关键点高斯噪声，模拟检测抖动）
- hand_conf_dropout_30（手部关键点置信度置零，模拟手部特写缺失）

产物：run 目录（manifest/predictions/metrics）+ 报告 md。
用法：
    python -m experiments.p1.e08.pose_perturbation_sweep --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE, *HERE.parents) if (p / "experiments" / "p1").is_dir())

POSE_DIR = REPO / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
DEFAULT_BUNDLE = REPO / "configs" / "bundles" / "unisign_stage2_pose"
REFS = REPO / "experiments" / "p1" / "references" / "gt_1086.csv"
SEED = 20261027

CONDITIONS = ("baseline", "speed_x1.25", "speed_x0.80", "dropout_10", "dropout_25",
              "jitter_0.002", "jitter_0.005", "jitter_0.01", "jitter_0.03", "hand_conf_dropout_30")


def norm(text: str | None) -> str:
    return "".join(unicodedata.normalize("NFKC", str(text or "")).split())


def char_sep(text: str) -> str:
    return " ".join(list(norm(text)))


def edit_similarity(reference: str, hypothesis: str) -> float:
    x, y = norm(reference), norm(hypothesis)
    if not x and not y:
        return 1.0
    prev = list(range(len(y) + 1))
    for i, ca in enumerate(x, 1):
        cur = [i]
        for j, cb in enumerate(y, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(1, len(x), len(y))


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with path.open("rb") as stream:
        pose = pickle.load(stream)
    keypoints = np.asarray(pose["keypoints"], dtype=np.float32)
    scores = np.asarray(pose["scores"], dtype=np.float32)
    if keypoints.ndim == 4:
        keypoints = keypoints[:, 0]
    if scores.ndim == 3:
        scores = scores[:, 0]
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError(f"unsupported keypoint shape {keypoints.shape} for {path.name}")
    if "start" in pose and "end" in pose:
        start, end = int(pose["start"]), int(pose["end"])
        if end > start:
            keypoints, scores = keypoints[start:end], scores[start:end]
    return keypoints, scores, {"frames_used": int(keypoints.shape[0])}


def perturb(points: np.ndarray, scores: np.ndarray, condition: str,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return a perturbed copy; baseline is returned unchanged."""
    pts, sc = points.copy(), scores.copy()
    n = len(pts)
    if condition == "baseline" or n < 8:
        return pts, sc

    if condition.startswith("speed_"):
        factor = 1.25 if "1.25" in condition else 0.80
        target = max(8, int(round(n / factor))) if factor > 1 else int(round(n / factor))
        idx = np.linspace(0, n - 1, target).round().astype(int)
        return pts[idx], sc[idx]

    if condition.startswith("dropout_"):
        rate = 0.10 if "10" in condition else 0.25
        keep = rng.random(n) >= rate
        keep[:4] = True                      # keep the first frames for body-scale estimation
        keep[-2:] = True
        return pts[keep], sc[keep]

    if condition.startswith("jitter_"):
        sigma = float(condition.split("_")[1])
        return pts + rng.normal(0.0, sigma, size=pts.shape).astype(np.float32), sc

    if condition == "hand_conf_dropout_30":
        hand = slice(91, 133)               # 133 COCO-wholebody keypoints: hands at 91..132
        mask = rng.random(n) < 0.30
        sc[np.ix_(mask, np.arange(hand.start, hand.stop))] = 0.0
        return pts, sc

    return pts, sc


def select_samples(limit: int | None) -> list[str]:
    with open(REFS, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    by_signer: dict[str, list[str]] = {}
    for row in rows:
        sid = row["sample_id"]
        if (POSE_DIR / f"{sid}.pkl").is_file():
            by_signer.setdefault(sid.split("_")[1], []).append(sid)
    rng = random.Random(SEED)
    chosen: list[str] = []
    per_signer = 10
    for signer in sorted(by_signer):
        pool = sorted(by_signer[signer])
        chosen.extend(rng.sample(pool, min(per_signer, len(pool))))
    if limit:
        chosen = chosen[:limit]
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uni-Sign pose perturbation robustness sweep")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args(argv)

    from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig
    from slbridge.uni_sign_preprocess import prepare_pose_window

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    run_id = args.run_id or f"pose-perturb-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    refs = {r["sample_id"]: r["ref_sentence_upstream"] for r in csv.DictReader(open(REFS, encoding="utf-8-sig"))}
    samples = select_samples(args.limit)
    config = UniSignConfig.from_bundle(args.bundle)
    max_frames = int(config.runtime.get("max_frames", 256))

    dtype = config.runtime.get("dtype") or "bfloat16"
    dtype = "bfloat16" if dtype == "auto" else dtype      # bundle manifests use "auto"
    device = args.device or ("cuda" if config.runtime.get("device") in (None, "auto") else config.runtime["device"])
    runtime = TorchUniSignRuntime(config, device=device, dtype=dtype)
    load_started = time.perf_counter()
    runtime.load()
    load_seconds = time.perf_counter() - load_started

    manifest = {
        "schema_version": 1, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E08-pose-perturbation-robustness",
        "bundle": str(args.bundle), "checkpoint_sha256": config.checkpoint_sha256,
        "source_commit": config.source_commit,
        "samples": len(samples), "conditions": conditions, "seed": SEED,
        "max_frames": max_frames, "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "device": args.device, "dtype": config.runtime.get("dtype"), "load_seconds": round(load_seconds, 3),
        "notes": ["pose perturbation only (no new training data available); "
                  "reference text used for offline scoring only (EVAL ONLY)"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    predictions: list[dict] = []
    started = time.perf_counter()
    for condition in conditions:
        rng = np.random.default_rng(SEED + abs(hash(condition)) % 1000)
        for index, sid in enumerate(samples, 1):
            points, scores, info = load_pose(POSE_DIR / f"{sid}.pkl")
            points, scores = perturb(points, scores, condition, rng)
            pose = prepare_pose_window(points, scores, max_frames=max_frames)
            try:
                text, timings = runtime.infer(pose, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
                status = "ok" if str(text).strip() else "empty"
            except Exception as error:  # noqa: BLE001 - recorded per sample
                text, timings, status = "", {}, f"error:{type(error).__name__}"
            predictions.append({"condition": condition, "sample_id": sid, "signer_id": sid.split("_")[1],
                                "status": status, "text": str(text).strip(),
                                "frames_used": info["frames_used"], **timings})
            if index % 20 == 0:
                print(json.dumps({"condition": condition, "done": index, "of": len(samples)}), flush=True)

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- metrics per condition (relative to baseline) ----
    import sacrebleu

    def aggregate(rows: list[dict]) -> dict:
        ok = [r for r in rows if r["status"] == "ok"]
        hyps = [char_sep(r["text"]) for r in ok]
        rf = [char_sep(refs[r["sample_id"]]) for r in ok]
        bleu = sacrebleu.corpus_bleu(hyps, [rf], tokenize="13a").score if ok else None
        chrf = sacrebleu.corpus_chrf(hyps, [rf]).score if ok else None
        edits = [edit_similarity(refs[r["sample_id"]], r["text"]) for r in ok]
        return {"n": len(rows), "ok": len(ok), "empty_or_error": len(rows) - len(ok),
                "bleu4_13a": bleu, "chrf": chrf,
                "mean_char_edit_similarity": statistics.fmean(edits) if edits else None,
                "exact_match": sum(1 for r in ok if norm(r["text"]) == norm(refs[r["sample_id"]]))}

    by_condition: dict[str, dict] = {}
    for condition in conditions:
        by_condition[condition] = aggregate([r for r in predictions if r["condition"] == condition])
    base = by_condition.get("baseline") or {}
    for condition, metrics in by_condition.items():
        if base.get("bleu4_13a") is not None and metrics.get("bleu4_13a") is not None:
            metrics["delta_bleu_vs_baseline"] = metrics["bleu4_13a"] - base["bleu4_13a"]
            metrics["delta_edit_vs_baseline"] = metrics["mean_char_edit_similarity"] - base["mean_char_edit_similarity"]
            metrics["prediction_identical_rate"] = None  # filled below
    # 逐样本一致率（与 baseline 输出逐字相同）
    base_text = {(r["sample_id"]): r["text"] for r in predictions if r["condition"] == "baseline"}
    for condition in conditions:
        rows = [r for r in predictions if r["condition"] == condition]
        same = sum(1 for r in rows if r["sample_id"] in base_text and r["text"] == base_text[r["sample_id"]])
        by_condition[condition]["prediction_identical_rate"] = same / len(rows) if rows else None

    elapsed = time.perf_counter() - started
    metrics = {"run_id": run_id, "elapsed_seconds": round(elapsed, 1), "conditions": by_condition,
               "per_signer": {}}
    for signer in sorted({r["signer_id"] for r in predictions}):
        metrics["per_signer"][signer] = {
            condition: aggregate([r for r in predictions
                                  if r["condition"] == condition and r["signer_id"] == signer])
            for condition in conditions}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runtime.close()

    print(json.dumps({c: {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in by_condition[c].items()
                          if k in ("n", "ok", "bleu4_13a", "mean_char_edit_similarity", "delta_bleu_vs_baseline",
                                   "delta_edit_vs_baseline", "prediction_identical_rate")}
                      for c in conditions}, ensure_ascii=False, indent=2))
    print(json.dumps({"run_dir": str(run_dir), "elapsed_seconds": round(elapsed, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())