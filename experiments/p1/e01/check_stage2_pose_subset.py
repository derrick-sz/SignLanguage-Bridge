"""Stage-2 pose-subset check (task-33, third segment).

Hypothesis (Lead, 2026-10-27): the released RGB+pose checkpoint's fusion gate is
all-zero, so the RGB branch is mathematically inert. Therefore a pose-only model
built from the 627 shared pose keys of `csl_daily_rgb_pose_slt.pth` must produce
**bit-identical** Pilot-50 predictions to the legacy RGB+pose run.

Steps:
  1. extract the 627 pose keys (same key set/shapes as csl_daily_pose_only_slt.pth)
     into `models/external/uni_sign/csl_daily_stage2_pose_subset.pth` with {"model": ...};
  2. build a run-local pose-only bundle (backend `uni_sign_pose_slt`) pointing at
     that checkpoint; validate it through the normal `load_model_adapter` registry;
  3. run the existing pose-only adapter runtime (`TorchUniSignRuntime`, same class
     used by `uni_sign_pose_slt`) on the frozen official-archive pose inputs;
  4. compare verbatim with the legacy RGB+pose frozen predictions (expect 100%),
     compute canonical metrics, and compare with A2/pose-control.

Outputs (GT-free predictions + manifest + report) go to
`experiments/p1/e01/runs/stage2-pose-subset-pilot50-20261027/`.

Usage:
    python -m experiments.p1.e01.check_stage2_pose_subset
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1 import common
from experiments.p1.gt_guard import assert_gt_clean, find_gt_keys
from experiments.p1.eval import metrics as M
from slbridge.model_adapters import load_model_adapter
from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig
from slbridge.uni_sign_preprocess import PoseWindow
from slbridge.uni_sign_rgb_pose import load_official_pose, load_part_kp

PILOT = common.P1 / "datasets" / "pilot_50.csv"
GT_PATH = common.REFERENCES / "gt_1086.csv"
CANDIDATES = common.P1 / "e02" / "candidate_outputs.jsonl"
RGB_POSE_CKPT = ROOT / "models" / "external" / "uni_sign" / "csl_daily_rgb_pose_slt.pth"
POSE_CKPT = ROOT / "models" / "external" / "uni_sign" / "csl_daily_pose_only_slt.pth"
SUBSET_CKPT = ROOT / "models" / "external" / "uni_sign" / "csl_daily_stage2_pose_subset.pth"
BASE_POSE_BUNDLE = ROOT / "models" / "external" / "uni_sign" / "model_bundle"
FROZEN_PILOT = common.P1 / "e01" / "runs" / "unisign-rgb-pose-pilot50-20261027"
FROZEN_PILOT_PRED = FROZEN_PILOT / "predictions_rgb_pose.jsonl"
FROZEN_CONTROL_PRED = FROZEN_PILOT / "predictions_pose_control.jsonl"
RUN_DIR = common.P1 / "e01" / "runs" / "stage2-pose-subset-pilot50-20261027"
REPORT = common.P1 / "e01" / "stage2_pose_subset_report.md"
POSE_DIR = ROOT / "artifacts" / "cslnews-eval" / "csl-daily-pose" / "pose-format"
MAX_LENGTH = 256
NUM_BEAMS = 4
MAX_NEW_TOKENS = 100


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_smi() -> str:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return probe.stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"unavailable: {type(error).__name__}"


def extract_pose_subset() -> dict:
    import torch

    rgb = torch.load(RGB_POSE_CKPT, map_location="cpu", weights_only=True, mmap=True)["model"]
    pose = torch.load(POSE_CKPT, map_location="cpu", weights_only=True, mmap=True)["model"]
    pose_keys = list(pose)
    missing = [key for key in pose_keys if key not in rgb]
    shape_mismatch = [key for key in pose_keys if key in rgb and tuple(pose[key].shape) != tuple(rgb[key].shape)]
    if missing or shape_mismatch:
        raise SystemExit(f"pose key set mismatch: missing={missing[:5]} shape={shape_mismatch[:5]}")
    subset = {key: rgb[key].clone() for key in pose_keys}
    SUBSET_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": subset}, SUBSET_CKPT)
    return {
        "path": str(SUBSET_CKPT.relative_to(ROOT)),
        "n_keys": len(subset),
        "bytes": SUBSET_CKPT.stat().st_size,
        "sha256": sha256(SUBSET_CKPT),
        "pose_only_keys": len(pose),
        "rgb_pose_keys": len(rgb),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "extra_rgb_keys": sorted(set(rgb) - set(pose)),
    }


def build_pose_bundle(subset_info: dict) -> Path:
    bundle = RUN_DIR / "pose_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE_POSE_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "checkpoint": str(SUBSET_CKPT),
        "checkpoint_bytes": subset_info["bytes"],
        "checkpoint_sha256": subset_info["sha256"],
        "run_id": "uni-sign-stage2-pose-subset-20261027",
        "display_name": "Uni-Sign stage2 pose subset (RGB+pose checkpoint minus RGB branch, experiment)",
    })
    manifest["mt5_dir"] = str((BASE_POSE_BUNDLE / manifest["mt5_dir"]).resolve())
    manifest["source_dir"] = str((BASE_POSE_BUNDLE / manifest["source_dir"]).resolve())
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return bundle


def pose_window_for_sample(sample: str) -> PoseWindow:
    pose = load_official_pose(POSE_DIR, sample)
    start, end = pose["start"], pose["end"]
    duration = end - start
    frames = list(range(duration))
    if duration > MAX_LENGTH:
        frames = sorted({int(round(value)) for value in __import__("numpy").linspace(0, duration - 1, MAX_LENGTH)})
    skeletons = [__import__("numpy").asarray(pose["keypoints"][start + index], dtype=__import__("numpy").float64)
                 for index in frames]
    confs = [__import__("numpy").asarray(pose["scores"][start + index], dtype=__import__("numpy").float32)
             for index in frames]
    parts = load_part_kp(skeletons, confs)
    return PoseWindow(body=parts["body"].numpy(), left=parts["left"].numpy(),
                      right=parts["right"].numpy(), face_all=parts["face_all"].numpy(),
                      sampled_indices=tuple(frames), quality={})


def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def load_candidates() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in common.read_jsonl(CANDIDATES):
        result.setdefault(row["sample_id"], {})[row["candidate_id"]] = str(
            (row.get("output") or {}).get("text") or "")
    return result


def load_texts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in common.read_jsonl(path):
        if row.get("status", "ok") not in {"ok", "recognized"}:
            continue
        text = str(row.get("text") or "")
        if text.strip():
            out[row["sample_id"]] = text
    return out


def corpus_metrics(texts: dict[str, str], gt: dict[str, dict]) -> dict:
    pairs = [(gt[sample]["ref_sentence_upstream"], text)
             for sample, text in sorted(texts.items()) if sample in gt and text.strip()]
    return M.corpus_metrics_cslt(pairs) if pairs else {}


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage2 pose-subset equivalence check")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args(argv)

    pilot = [row["sample_id"] for row in common.read_csv_rows(PILOT)]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    gpu_before = nvidia_smi()
    subset_info = extract_pose_subset()
    bundle = build_pose_bundle(subset_info)
    bundle_manifest_sha = sha256(bundle / "manifest.json")

    # normal registry validation (pose-only architecture)
    registry_adapter = load_model_adapter(bundle)
    registry_caps = registry_adapter.capabilities
    registry_runtime_info = dict(registry_adapter.runtime_info)
    registry_adapter.close()

    config = UniSignConfig.from_bundle(bundle)
    runtime = TorchUniSignRuntime(config, device="cuda", dtype="bfloat16")
    started = time.perf_counter()
    runtime.load()
    load_ms = (time.perf_counter() - started) * 1000.0
    gpu_after_load = nvidia_smi()
    predictions = []
    for index, sample in enumerate(pilot, 1):
        pose_window = pose_window_for_sample(sample)
        text, timings = runtime.infer(pose_window, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS)
        row = {
            "run_id": RUN_DIR.name,
            "system": "stage2_pose_subset",
            "sample_id": sample,
            "status": "ok" if text.strip() else "error",
            "text": text,
            "frames": len(pose_window.sampled_indices),
            "model_total_ms": round(timings["model_total_ms"], 3),
        }
        leaks = find_gt_keys(row)
        if leaks:
            raise SystemExit(f"GT-free guard violated: {leaks}")
        assert_gt_clean(row, context=f"stage2_pose_subset/{sample}")
        predictions.append(row)
        print(json.dumps({"event": "sample", "index": index, "n": len(pilot),
                          "sample_id": sample, "text": text,
                          "model_total_ms": round(timings["model_total_ms"], 1)}, ensure_ascii=False))
    gpu_after = nvidia_smi()
    if hasattr(runtime, "runtime_info"):
        runtime_info = dict(runtime.runtime_info)
    else:  # pragma: no cover
        runtime_info = {}
    runtime.close()
    common.write_jsonl(RUN_DIR / "predictions_stage2_pose_subset.jsonl", predictions, guard=True)

    frozen = load_texts(FROZEN_PILOT_PRED)
    control = load_texts(FROZEN_CONTROL_PRED)
    candidates = load_candidates()
    gt = load_gt()
    generated = {row["sample_id"]: row["text"] for row in predictions}
    diffs = [{"sample_id": sample, "frozen": frozen.get(sample), "stage2": generated.get(sample)}
             for sample in pilot if sample in frozen and frozen[sample] != generated.get(sample)]
    exact = len(pilot) - len(diffs)
    metrics = {
        "stage2_pose_subset": corpus_metrics(generated, gt),
        "legacy_rgb_pose": corpus_metrics({s: frozen[s] for s in pilot if s in frozen}, gt),
        "A2": corpus_metrics({s: candidates[s].get("A2", "") for s in pilot if s in candidates}, gt),
        "pose_control": corpus_metrics({s: control[s] for s in pilot if s in control}, gt),
    }
    payload = {
        "schema_version": 1,
        "kind": "stage2-pose-subset-equivalence",
        "created_at": common.utc_now(),
        "run_dir": str(RUN_DIR),
        "subset_checkpoint": subset_info,
        "pose_bundle": str(bundle),
        "pose_bundle_manifest_sha256": bundle_manifest_sha,
        "registry": {"adapter_type": type(registry_adapter).__name__,
                     "capabilities": str(registry_caps),
                     "runtime_info": registry_runtime_info},
        "runtime_info": runtime_info,
        "gpu_before": gpu_before,
        "gpu_after_load": gpu_after_load,
        "gpu_after": gpu_after,
        "load_ms": round(load_ms, 1),
        "n_samples": len(pilot),
        "n_exact": exact,
        "exact_rate": round(exact / len(pilot), 4),
        "n_differences": len(diffs),
        "differences": diffs[:20],
        "metrics": metrics,
        "gt_eval_only": {"path": str(GT_PATH.relative_to(ROOT)), "eval_only": True},
    }
    common.write_json(RUN_DIR / "stage2_pose_subset.json", payload)

    lines = ["# Stage2 pose-subset 等价性验证（task-33 第三段）", ""]
    lines.append("- 目的：验证 gate 恒零的 RGB+pose checkpoint 与其 627 个 pose 键的纯 pose 架构**逐字等价**，"
                 "从而可抛弃 RGB 分支、保留 stage2 收益且降低延迟。")
    lines.append(f"- 输入：pilot 50 条，官方 pose archive（与冻结基线同源）；pose-only adapter runtime（`TorchUniSignRuntime`，同一 `uni_sign_pose_slt` 架构）。")
    lines.append(f"- 子集 checkpoint：`{subset_info['path']}`（{subset_info['n_keys']} 键，{subset_info['bytes']} B，sha256 `{subset_info['sha256'][:16]}…`；"
                 f"pose-only 键集 {subset_info['pose_only_keys']}，rgb_pose 键集 {subset_info['rgb_pose_keys']}，缺失 0、形状不匹配 0）。")
    lines.append(f"- run：`{RUN_DIR}`（GT-free predictions、stage2_pose_subset.json）；GPU 基线 `{gpu_before}`；load {round(load_ms,1)} ms。")
    lines.append("")
    lines.append("## 1. 逐条一致率")
    lines.append("")
    lines.append(f"- 与冻结 legacy RGB+pose Pilot-50 预测逐字一致：**{exact}/{len(pilot)}（{payload['exact_rate']*100:.1f}%）**；差异 {len(diffs)} 条。")
    if diffs:
        lines.append("- 前 5 条差异：")
        for item in diffs[:5]:
            lines.append(f"  - `{item['sample_id']}`：frozen={item['frozen']!r} / stage2={item['stage2']!r}")
    else:
        lines.append("- 差异清单为空。")
    lines.append("")
    lines.append("## 2. 指标（EVAL ONLY，pilot-50）")
    lines.append("")
    lines.append("| 系统 | BLEU-4 | chrF | char edit |")
    lines.append("|---|---:|---:|---:|")
    for key, label in (("stage2_pose_subset", "stage2 pose subset（纯 pose 架构）"),
                       ("legacy_rgb_pose", "legacy RGB+pose（冻结，零 RGB）"),
                       ("A2", "A2（官方 pose）"), ("pose_control", "pose 控制")):
        entry = metrics.get(key) or {}
        lines.append(f"| {label} | {_fmt(entry.get('bleu4_13a'))} | {_fmt(entry.get('chrf'))} | "
                     f"{_fmt(entry.get('mean_char_edit_similarity'))} |")
    lines.append("")
    lines.append("## 3. 结论")
    lines.append("")
    if exact == len(pilot):
        lines.append("- **逐条 100% 一致**：stage2 pose 子集 + 纯 pose 架构与 legacy RGB+pose 完全等价。")
        lines.append("- 因此 **RGB 分支可以完全去掉**：保留 stage2 的 pose/mT5 权重（+1.39 pilot / +0.66 1086 相对 A2/pose-only 的提升来自这里），"
                     "延迟按 pose-only 路径（~278 ms 量级）而非 RGB+pose（~370 ms）。")
        lines.append("- 但需注意：这不是论文 RGB+pose 的复现；真实 RGB+pose 仍需作者提供非零 fusion gate 权重或重新训练 gate。")
    else:
        lines.append("- **未完全一致**：需逐条归因（pose-only 架构缺失键/形状、预处理数值差、tokenizer/精度）；"
                     "在归因前不得声称“可去掉 RGB 分支”。")
        lines.append(f"- 差异 {len(diffs)} 条（见 payload `stage2_pose_subset.json` 与上方前 5 条）。")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "run_dir": str(RUN_DIR),
        "report": str(REPORT),
        "subset_checkpoint": subset_info["path"],
        "subset_sha256": subset_info["sha256"],
        "n_exact": exact,
        "n_samples": len(pilot),
        "exact_rate": payload["exact_rate"],
        "n_differences": len(diffs),
        "metrics": {key: {k: (entry or {}).get(k) for k in ("bleu4_13a", "chrf", "mean_char_edit_similarity")}
                    for key, entry in metrics.items()},
        "gpu_before": gpu_before,
        "load_ms": round(load_ms, 1),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
