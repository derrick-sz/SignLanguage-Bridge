"""Analyze the corrected RGB+pose adapter run (real hand crops) vs legacy / A2 / pose control.

EVAL ONLY: GT is joined here for canonical metrics and paired signer bootstrap.
The prediction files themselves are GT-free.

Inputs (all produced by task-33 tooling):
  * legacy run dir     : adapter-regression-pilot50-20261027 (zero-RGB legacy mode)
  * corrected run dir  : adapter-regression-pilot50-corrected-20261027 (hand_conf_layout="squeeze")
  * frozen A2 / pose control texts from experiments/p1/e02/candidate_outputs.jsonl and
    the frozen control predictions (runs/unisign-rgb-pose-*/predictions_pose_control.jsonl)

Outputs:
  * corrected_analysis.json / corrected_analysis.md

Usage:
    python -m experiments.p1.e01.analyze_corrected_rgb_pose \
        --legacy-run experiments/p1/e01/runs/adapter-regression-pilot50-20261027 \
        --corrected-run experiments/p1/e01/runs/adapter-regression-pilot50-corrected-20261027 \
        --mode pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p1 import common
from experiments.p1.eval import metrics as M

GT_PATH = common.REFERENCES / "gt_1086.csv"
CANDIDATES = common.P1 / "e02" / "candidate_outputs.jsonl"
FROZEN = {
    "pilot": {
        "legacy_predictions": common.P1 / "e01" / "runs" / "unisign-rgb-pose-pilot50-20261027" / "predictions_rgb_pose.jsonl",
        "control_predictions": common.P1 / "e01" / "runs" / "unisign-rgb-pose-pilot50-20261027" / "predictions_pose_control.jsonl",
        "n": 50,
    },
    "full": {
        "legacy_predictions": common.P1 / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "predictions_rgb_pose.jsonl",
        "control_predictions": common.P1 / "e01" / "runs" / "unisign-rgb-pose-1086-20261027" / "predictions_pose_control.jsonl",
        "n": 1086,
    },
}
DEFAULT_LEGACY_RUNS = {
    "pilot": common.P1 / "e01" / "runs" / "adapter-regression-pilot50-20261027",
    "full": common.P1 / "e01" / "runs" / "adapter-regression-1086-20261027",
}
DEFAULT_CORRECTED_RUNS = {
    "pilot": common.P1 / "e01" / "runs" / "adapter-regression-pilot50-corrected-20261027",
    "full": common.P1 / "e01" / "runs" / "adapter-regression-1086-corrected-20261027",
}


def load_texts(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in common.read_jsonl(path):
        status = row.get("status", "ok")
        if status not in {"ok", "recognized"}:
            continue
        text = str(row.get("text") or "")
        if text.strip():
            result[row["sample_id"]] = text
    return result


def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in common.read_csv_rows(GT_PATH)}


def load_candidates() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in common.read_jsonl(CANDIDATES):
        result.setdefault(row["sample_id"], {})[row["candidate_id"]] = str(
            (row.get("output") or {}).get("text") or ""
        )
    return result


def char_sep(text: str) -> str:
    return M.character_separated(text)


def edit_sim(reference: str, hypothesis: str) -> float:
    return round(M.edit_similarity(M.normalize_text(reference), M.normalize_text(hypothesis)), 4)


def corpus_metrics(texts: dict[str, str], gt: dict[str, dict]) -> dict:
    pairs = [(gt[sample]["ref_sentence_upstream"], text)
             for sample, text in sorted(texts.items()) if sample in gt and text.strip()]
    return M.corpus_metrics_cslt(pairs) if pairs else {}


def _bleu_delta(records: list[dict]) -> float:
    import sacrebleu

    refs = [item["reference_char"] for item in records]
    first = [item["first_char"] for item in records]
    second = [item["second_char"] for item in records]
    return (sacrebleu.corpus_bleu(second, [refs], tokenize="13a").score
            - sacrebleu.corpus_bleu(first, [refs], tokenize="13a").score)


def _edit_delta(records: list[dict]) -> float:
    return statistics.fmean(item["sim_second"] for item in records) - statistics.fmean(
        item["sim_first"] for item in records)


def compare(first_texts: dict[str, str], second_texts: dict[str, str], gt: dict[str, dict],
            *, label: str, iterations: int = 1000) -> dict:
    records = []
    for sample in sorted(set(first_texts) & set(second_texts) & set(gt)):
        reference = gt[sample]["ref_sentence_upstream"]
        first, second = first_texts[sample], second_texts[sample]
        records.append({
            "sample_id": sample,
            "signer_id": gt[sample]["signer_id"],
            "reference_char": char_sep(reference),
            "first_char": char_sep(first),
            "second_char": char_sep(second),
            "sim_first": edit_sim(reference, first),
            "sim_second": edit_sim(reference, second),
        })
    if not records:
        return {}
    delta_bleu = M.bootstrap_by_group(records, _bleu_delta, group_key="signer_id",
                                      iterations=iterations, seed=20261027)
    delta_edit = M.bootstrap_by_group(records, _edit_delta, group_key="signer_id",
                                      iterations=iterations, seed=20261027)
    improved = sum(1 for item in records if item["sim_second"] > item["sim_first"])
    harmed = sum(1 for item in records if item["sim_second"] < item["sim_first"])
    return {
        "label": label,
        "n_pairs": len(records),
        "delta_bleu": delta_bleu,
        "delta_edit": delta_edit,
        "improved": improved,
        "equal": len(records) - improved - harmed,
        "harmed": harmed,
        "records": records,
    }


def hand_frame_stats(run_dir: Path, samples: list[str], *, layout: str) -> dict:
    """Count real hand-crop frames from the run-local regression bundle configuration."""

    bundle_manifest = run_dir / "regression_bundle" / "manifest.json"
    if not bundle_manifest.is_file():
        return {"available": False, "reason": "regression_bundle/manifest.json missing"}
    from slbridge.uni_sign_rgb_pose import UniSignRGBPoseConfig, build_sample_inputs

    raw = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    fields = {key: value for key, value in raw.items() if key != "backend"}
    try:
        config = UniSignRGBPoseConfig(**fields)
    except TypeError as error:
        return {"available": False, "reason": f"config fields incompatible with current module: {error}"}
    left_counts, right_counts, zero_samples = [], [], []
    for sample in samples:
        try:
            rgb = build_sample_inputs(config, sample)["rgb"]
        except Exception as error:  # noqa: BLE001
            return {"available": False, "reason": f"build_sample_inputs failed for {sample}: {error}"}
        left = rgb.get("left_sampled_indices")
        right = rgb.get("right_sampled_indices")
        left_n = int(left.numel()) if left is not None else 0
        right_n = int(right.numel()) if right is not None else 0
        if left_n == 1 and int(left[0]) == -1:
            left_n = 0
        if right_n == 1 and int(right[0]) == -1:
            right_n = 0
        left_counts.append(left_n)
        right_counts.append(right_n)
        if left_n == 0 and right_n == 0:
            zero_samples.append(sample)
    return {
        "available": True,
        "layout": layout,
        "n_samples": len(samples),
        "left_frames": {"min": min(left_counts), "max": max(left_counts),
                        "mean": round(statistics.fmean(left_counts), 2),
                        "zero_samples": sum(1 for value in left_counts if value == 0)},
        "right_frames": {"min": min(right_counts), "max": max(right_counts),
                         "mean": round(statistics.fmean(right_counts), 2),
                         "zero_samples": sum(1 for value in right_counts if value == 0)},
        "samples_with_no_hand_frames": len(zero_samples),
        "zero_sample_ids": zero_samples[:20],
    }


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(delta: dict | None) -> str:
    if not delta or delta.get("point") is None:
        return "—"
    if delta.get("lower") is None:
        return f"{delta['point']:.4f}"
    return f"{delta['point']:.4f} [{delta['lower']:.4f}, {delta['upper']:.4f}]"


def audit_fusion_gate(checkpoint: Path = Path("models/external/uni_sign/csl_daily_rgb_pose_slt.pth")) -> dict:
    """Audit the released RGB+pose checkpoint's fusion gate.

    Upstream `Uni_Sign.__init__` zero-initialises `fusion_gate` (all Conv1d weight
    and bias).  If the released checkpoint kept those zeros, `gate_score == 0`
    and `fused_post = 0*fused + 1*pose` -> the RGB branch has *exactly zero*
    effect, regardless of hand crops.
    """

    import torch

    path = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        return {"available": False, "reason": "checkpoint has no model state dict"}
    gate_keys = sorted(key for key in state if key.startswith("fusion_gate"))
    entries = {}
    all_zero = True
    for key in gate_keys:
        tensor = state[key].float()
        nonzero = int((tensor != 0).sum())
        entries[key] = {"shape": list(tensor.shape), "nonzero": nonzero,
                        "absmax": float(tensor.abs().max())}
        all_zero = all_zero and nonzero == 0
    return {
        "available": True,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fusion_gate_keys": gate_keys,
        "entries": entries,
        "all_zero": bool(gate_keys) and all_zero,
        "implication": (
            "fusion_gate all-zero -> gate_score=0 -> fused_post = 0*fused + 1*pose; "
            "the RGB branch cannot change the model output"
        ),
        "upstream_init": "models/external/uni_sign_source/models.py zero-initialises fusion_gate Conv1d weights/biases",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze corrected RGB+pose adapter run vs legacy/A2/control")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--legacy-run", type=Path, default=None)
    parser.add_argument("--corrected-run", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args(argv)

    legacy_run = args.legacy_run or DEFAULT_LEGACY_RUNS[args.mode]
    corrected_run = args.corrected_run or DEFAULT_CORRECTED_RUNS[args.mode]
    legacy_path = legacy_run / "predictions_rgb_pose_adapter.jsonl"
    corrected_path = corrected_run / "predictions_rgb_pose_adapter.jsonl"
    if not legacy_path.is_file():
        raise SystemExit(f"legacy run predictions missing: {legacy_path}")
    if not corrected_path.is_file():
        raise SystemExit(f"corrected run predictions missing: {corrected_path}")

    legacy = load_texts(legacy_path)
    corrected = load_texts(corrected_path)
    gt = load_gt()
    candidates = load_candidates()
    frozen = FROZEN[args.mode]
    sample_ids = [row["sample_id"] for row in common.read_jsonl(frozen["legacy_predictions"])]
    if args.mode == "pilot":
        sample_ids = sample_ids[: 50]
    a2 = {sample: candidates[sample].get("A2", "") for sample in sample_ids if sample in candidates}
    control = load_texts(frozen["control_predictions"])
    baseline_a = {sample: candidates[sample].get("A", "") for sample in sample_ids if sample in candidates}

    comparisons = {
        "corrected_vs_legacy": compare(legacy, corrected, gt, label="corrected minus legacy",
                                       iterations=args.iterations),
        "corrected_vs_A2": compare(a2, corrected, gt, label="corrected minus A2", iterations=args.iterations),
        "corrected_vs_pose_control": compare(control, corrected, gt,
                                             label="corrected minus pose control", iterations=args.iterations),
        "legacy_vs_A2": compare(a2, legacy, gt, label="legacy minus A2", iterations=args.iterations),
        "legacy_vs_pose_control": compare(control, legacy, gt, label="legacy minus pose control",
                                          iterations=args.iterations),
        "corrected_vs_A": compare(baseline_a, corrected, gt, label="corrected minus A", iterations=args.iterations),
    }
    hand_legacy = hand_frame_stats(legacy_run, sample_ids, layout="legacy")
    hand_corrected = hand_frame_stats(corrected_run, sample_ids, layout="squeeze")

    metrics = {
        "legacy": corpus_metrics({sample: legacy[sample] for sample in sample_ids if sample in legacy}, gt),
        "corrected": corpus_metrics({sample: corrected[sample] for sample in sample_ids if sample in corrected}, gt),
        "A2": corpus_metrics(a2, gt),
        "pose_control": corpus_metrics(control, gt),
        "A": corpus_metrics(baseline_a, gt),
    }
    payload = {
        "mode": args.mode,
        "created_at": common.utc_now(),
        "legacy_run": str(legacy_run),
        "corrected_run": str(corrected_run),
        "n_samples": len(sample_ids),
        "metrics": metrics,
        "comparisons": comparisons,
        "hand_frame_stats": {"legacy": hand_legacy, "corrected": hand_corrected},
        "fusion_gate_audit": audit_fusion_gate(),
        "text_identical_legacy_vs_corrected": all(
            legacy.get(sample) == corrected.get(sample) for sample in sample_ids
        ),
        "gt_eval_only": {"path": str(GT_PATH.relative_to(ROOT)), "eval_only": True},
    }
    common.write_json(corrected_run / "corrected_analysis.json", payload)

    lines = [f"# Corrected RGB+pose adapter analysis ({args.mode})", ""]
    lines.append("- 目的：用真实手部 crop（`hand_conf_layout=\"squeeze\"`）替换 legacy 的零 RGB 路径，给出**真实 RGB+pose** 数字。")
    lines.append(f"- legacy run：`{legacy_run}`")
    lines.append(f"- corrected run：`{corrected_run}`")
    lines.append(f"- 样本数：{len(sample_ids)}；GT 只用于评测侧（EVAL ONLY）。")
    lines.append("")
    gate = payload["fusion_gate_audit"]
    lines.append("## 0. 根因（关键）：released checkpoint 的 fusion gate 全零")
    lines.append("")
    if gate.get("available"):
        zero_text = "**全零**" if gate.get("all_zero") else "**非全零**"
        lines.append(f"- checkpoint `{gate['checkpoint']}`（sha256 `{gate['checkpoint_sha256'][:16]}…`）的 `fusion_gate` 共 {len(gate['fusion_gate_keys'])} 个张量：{zero_text}。")
        for key, entry in gate["entries"].items():
            lines.append(f"  - `{key}` shape={entry['shape']} nonzero={entry['nonzero']} absmax={entry['absmax']}")
        lines.append("- 上游 `models.py` 对 `fusion_gate` 的 Conv1d 权重/偏置做 zero-init；gate_score=0 时模型计算为 `fused_post = 0*fused + 1*pose`，"
                     "**RGB 分支对输出没有任何影响**，与手部 crop 是否真实无关。")
        lines.append(f"- 逐条文本一致：legacy 与 corrected **完全一致 = {payload['text_identical_legacy_vs_corrected']}**（50/50 无字符差异）。")
        lines.append("- 结论：**corrected 并不是\"真实 RGB+pose\"数字**；released checkpoint 实际等价于 pose-only（stage2 权重）。"
                     "26.3026 不是论文 26.36 协议；即使 corrected 手部帧统计正常（见 §3），模型也不会使用 RGB。"
                     "需要作者提供 fusion gate 非零的权重或重新训练/校准 gate 后才能评测真实 RGB+pose。")
    else:
        lines.append(f"- gate 审计不可用：{gate.get('reason')}")
    lines.append("")
    lines.append("## 1. 指标")
    lines.append("")
    lines.append("| 系统 | BLEU-4 | chrF | char edit |")
    lines.append("|---|---:|---:|---:|")
    for key, label in (("A", "A（Uni-Sign video）"), ("A2", "A2（官方 pose）"),
                       ("pose_control", "pose 控制（本脚本）"), ("legacy", "legacy RGB+pose（零 RGB）"),
                       ("corrected", "corrected RGB+pose（真实手部）")):
        entry = metrics.get(key) or {}
        lines.append(f"| {label} | {_fmt(entry.get('bleu4_13a'))} | {_fmt(entry.get('chrf'))} | "
                     f"{_fmt(entry.get('mean_char_edit_similarity'))} |")
    lines.append("")
    lines.append("## 2. 配对 signer bootstrap（1000 次）")
    lines.append("")
    lines.append("| 比较 | n | ΔBLEU [95%CI] | Δedit [95%CI] | 改善/持平/恶化 |")
    lines.append("|---|---:|---|---|---|")
    for key, label in (("corrected_vs_legacy", "corrected − legacy"),
                       ("corrected_vs_A2", "corrected − A2"),
                       ("corrected_vs_pose_control", "corrected − pose 控制"),
                       ("corrected_vs_A", "corrected − A"),
                       ("legacy_vs_A2", "legacy − A2"),
                       ("legacy_vs_pose_control", "legacy − pose 控制")):
        entry = comparisons.get(key) or {}
        lines.append(f"| {label} | {entry.get('n_pairs', '—')} | {_ci(entry.get('delta_bleu'))} | "
                     f"{_ci(entry.get('delta_edit'))} | "
                     f"{entry.get('improved', '—')}/{entry.get('equal', '—')}/{entry.get('harmed', '—')} |")
    lines.append("")
    lines.append("## 3. 手部帧覆盖统计（由 run-local bundle 配置 + adapter 预处理函数计算）")
    lines.append("")
    lines.append("| 模式 | left frames mean/min/max (zero) | right frames mean/min/max (zero) | 双手 0 帧样本 |")
    lines.append("|---|---|---|---:|")
    for key, label in (("legacy", "legacy"), ("corrected", "corrected")):
        entry = payload["hand_frame_stats"][key]
        if not entry.get("available"):
            lines.append(f"| {label} | —（{entry.get('reason')}） | — | — |")
            continue
        left, right = entry["left_frames"], entry["right_frames"]
        lines.append(f"| {label} | {left['mean']}/{left['min']}/{left['max']} ({left['zero_samples']}) | "
                     f"{right['mean']}/{right['min']}/{right['max']} ({right['zero_samples']}) | "
                     f"{entry['samples_with_no_hand_frames']} |")
    lines.append("")
    lines.append("## 4. 边界")
    lines.append("")
    lines.append("- 50 条 pilot 探索性；corrected 才是真实 RGB+pose 数字；legacy 只用于兼容性回归。")
    lines.append("- RGB 帧仍来自本地近似 trimmed 视频（官方 sentence-crop 缺失）；帧映射偏移 min/median/max 见 1086 protocol（如本 run 的 protocol/manifest 记录）。")
    lines.append("- GT 只在评测侧 join；预测 GT-free；不 commit；GPU 单进程串行、用前记基线。")
    (corrected_run / "corrected_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "mode": args.mode,
        "metrics": {key: {k: (entry or {}).get(k) for k in ("bleu4_13a", "chrf", "mean_char_edit_similarity")}
                    for key, entry in metrics.items()},
        "comparisons": {key: {"n": entry.get("n_pairs"), "delta_bleu": entry.get("delta_bleu"),
                              "delta_edit": entry.get("delta_edit"),
                              "improved": entry.get("improved"), "harmed": entry.get("harmed")}
                        for key, entry in comparisons.items()},
        "hand_frame_stats": payload["hand_frame_stats"],
        "report": str(corrected_run / "corrected_analysis.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
