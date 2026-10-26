"""Collect every P-1 experiment into one auditable summary.

Usage::

    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -m experiments.p1.collect_summary

It reads only existing artifacts (no model execution), re-verifies the
ground-truth isolation of every prediction payload, and writes
``experiments/p1/summary.json`` plus ``experiments/p1/summary.md``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from experiments.p1 import common
from experiments.p1.gt_guard import find_gt_keys

EXPERIMENTS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06")


def _load_json(path: Path) -> Any:
    if not Path(path).is_file():
        return None
    return common.read_json(path)


def _load_yaml(path: Path) -> Any:
    if not Path(path).is_file():
        return None
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - audit tool must not crash on a broken asset
        return {"_error": f"{type(error).__name__}: {error}"}


def _load_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return common.read_jsonl(path)


def scan_run_directories() -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(common.RUNS.glob("*")):
        if not run_dir.is_dir():
            continue
        manifest = _load_json(run_dir / "manifest.json") or {}
        outputs = run_dir / "model_outputs.jsonl"
        rows = 0
        gt_leaks: list[str] = []
        if outputs.is_file():
            for record in common.iter_jsonl(outputs):
                rows += 1
                gt_leaks.extend(find_gt_keys(record))
        runs.append(
            {
                "run_id": run_dir.name,
                "experiment_id": manifest.get("purpose") or manifest.get("experiment_id"),
                "run_status": manifest.get("run_status"),
                "created_at": manifest.get("created_at"),
                "model_outputs_rows": rows,
                "model_outputs_gt_free": not gt_leaks,
                "gt_leaks": gt_leaks[:10],
                "has_metrics": (run_dir / "metrics.json").is_file(),
                "has_eval_records": (run_dir / "eval_records.jsonl").is_file(),
                "manifest_path": str(run_dir / "manifest.json"),
            }
        )
    return runs


def scan_payload_gt_free() -> list[dict[str, Any]]:
    paths = [
        common.P1 / "e02" / "candidate_outputs.jsonl",
        common.P1 / "e06" / "candidate_inputs.jsonl",
    ]
    results = []
    for path in paths:
        if not path.is_file():
            continue
        rows = 0
        leaks: list[str] = []
        for record in common.iter_jsonl(path):
            rows += 1
            leaks.extend(find_gt_keys(record))
        results.append({"path": str(path), "rows": rows, "gt_free": not leaks, "gt_leaks": leaks[:10]})
    return results


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def experiment_summaries() -> dict[str, Any]:
    p1 = common.P1
    summary: dict[str, Any] = {}

    # E00 ------------------------------------------------------------------
    registry = _load_yaml(p1 / "e00" / "model_registry.yaml") or {}
    models = registry.get("models") or []
    summary["E00"] = {
        "status": "PASS (audit-side); G0 smoke evidence from E01",
        "model_registry": str(p1 / "e00" / "model_registry.yaml"),
        "models": [
            {
                "model_id": model.get("model_id"),
                "status": model.get("status"),
                "checkpoint_sha256": model.get("checkpoint_sha256"),
                "bytes": model.get("bytes"),
                "license": model.get("license"),
            }
            for model in models
        ],
        "environment_manifest": str(p1 / "e00" / "environment_manifest.json"),
        "data_snapshot": str(p1 / "e00" / "data_snapshot.json"),
        "inventory_report": str(p1 / "e00" / "inventory_report.md"),
    }

    # E01 ------------------------------------------------------------------
    e01 = _load_json(p1 / "e01" / "summary.json") or {}
    summary["E01"] = {
        "status": (e01.get("status") or {}).get("overall"),
        "report": str(p1 / "e01" / "reproduction_report.md"),
        "summary": str(p1 / "e01" / "summary.json"),
        "video_probe": str(p1 / "e01" / "video_probe.csv"),
        "preprocess_audit": str(p1 / "e01" / "preprocess_audit.csv"),
        "comparisons": {
            name: str(p1 / "e01" / f"compare_{name}.csv")
            for name in ("unisign_video", "unisign_pose", "slrt")
            if (p1 / "e01" / f"compare_{name}.csv").is_file()
        },
        "paths": (e01.get("paths") or {}).get("U1_unisign_video") and {
            key: {
                "status": value.get("status"),
                "history_compare": value.get("history_compare"),
                "canonical_metrics_rerun": value.get("canonical_metrics_rerun"),
                "latency": value.get("latency"),
            }
            for key, value in (e01.get("paths") or {}).items()
        },
        "video_vs_pose_pilot50": (e01.get("comparisons") or {}).get("unisign_video_vs_pose_on_pilot50"),
        "reference_variants": e01.get("reference_variants"),
        "limitations": e01.get("limitations"),
    }

    # E02 ------------------------------------------------------------------
    e02 = _load_json(p1 / "e02" / "translation_metrics.json") or {}
    summary["E02"] = {
        "status": "PASS (offline replay); B1/C/D/E/F/G BLOCKED",
        "run_id": e02.get("run_id"),
        "report": str(p1 / "e02" / "report.md"),
        "translation_metrics": str(p1 / "e02" / "translation_metrics.json"),
        "fusion_ablation": str(p1 / "e02" / "fusion_ablation.csv"),
        "comparison_table": str(p1 / "e02" / "comparison_table.md"),
        "candidate_outputs": str(p1 / "e02" / "candidate_outputs.jsonl"),
        "results": {
            name: payload.get("metrics")
            for name, payload in (e02.get("paths") or {}).items()
        },
        "paired_a_vs_a2": e02.get("paired_a_vs_a2"),
        "error_overlap": e02.get("error_overlap"),
        "oracle": e02.get("oracle"),
        "legacy_vs_canonical": e02.get("legacy_vs_canonical"),
        "human_review": e02.get("human_review"),
    }

    # E03 ------------------------------------------------------------------
    e03 = _load_json(p1 / "e03" / "calibration_results.json") or {}
    summary["E03"] = {
        "status": "PASS (SLRT score calibration); Uni-Sign confidence BLOCKED",
        "report": str(p1 / "e03" / "report.md"),
        "score_definition": str(p1 / "e03" / "score_definition.md"),
        "calibration_results": str(p1 / "e03" / "calibration_results.json"),
        "risk_coverage": str(p1 / "e03" / "risk_coverage.csv"),
        "per_sample_scores": str(p1 / "e03" / "per_sample_scores.csv"),
        "headline": e03.get("headline") or e03.get("summary") or {
            key: e03.get(key) for key in list(e03)[:8] if key not in ("reliability", "folds")
        },
        "blocked": "Uni-Sign TextPrediction(score=None); no logprob/score in historical JSONL",
    }

    # E04 / E05 ------------------------------------------------------------
    perf_trace = _first_existing(p1 / "e04" / "perf_trace.jsonl", p1 / "e04" / "perf_trace.json")
    summary["E04"] = {
        "status": "PASS" if perf_trace else "MISSING",
        "hardware_report": str(p1 / "e04" / "hardware_report.md"),
        "perf_trace": str(perf_trace) if perf_trace else None,
        "perf_trace_rows": len(_load_jsonl(perf_trace)) if perf_trace else 0,
        "parallel_vs_serial": str(p1 / "e04" / "parallel_vs_serial.csv"),
        "metrics_summary": str(p1 / "e04" / "metrics_summary.md"),
        "metrics": _load_json(p1 / "e04" / "metrics.json"),
    }
    prefix_probe = p1 / "e05" / "prefix_probe.jsonl"
    summary["E05"] = {
        "status": "PASS" if prefix_probe.is_file() else "MISSING",
        "streaming_ability_matrix": str(p1 / "e05" / "streaming_ability_matrix.md"),
        "future_frame_audit": str(p1 / "e05" / "future_frame_audit.md"),
        "prefix_probe": str(prefix_probe),
        "prefix_rows": len(_load_jsonl(prefix_probe)),
        "prefix_probe_summary": str(p1 / "e05" / "prefix_probe_summary.md"),
        "summary": _load_json(p1 / "e05" / "prefix_summary.json"),
    }

    # E06 ------------------------------------------------------------------
    router_rows = _load_jsonl(p1 / "e06" / "router_ablation.jsonl")
    summary["E06"] = {
        "status": "PASS (Rule-only); LLM/Jev/D-G BLOCKED",
        "report": str(p1 / "e06" / "report.md"),
        "router_ablation": str(p1 / "e06" / "router_ablation.jsonl"),
        "policy_candidate": str(p1 / "e06" / "policy_candidate.yaml"),
        "judge_error_analysis": str(p1 / "e06" / "judge_error_analysis.md"),
        "candidate_inputs": str(p1 / "e06" / "candidate_inputs.jsonl"),
        "strategies": router_rows,
    }
    return summary


def main() -> int:
    payload = {
        "schema_version": 1,
        "generated_at": common.utc_now(),
        "generated_by": "experiments/p1/collect_summary.py",
        "git": common.git_state(),
        "environment": common.environment_fingerprint(),
        "plan_package": {
            "path": "SignLanguageBridge_P-1_实验计划包",
            "plan_sha256": common.sha256_file(common.ROOT / "SignLanguageBridge_P-1_实验计划包" / "PLAN.md")
            if (common.ROOT / "SignLanguageBridge_P-1_实验计划包" / "PLAN.md").is_file()
            else None,
            "copied_to": str(common.PLAN_DIR),
        },
        "experiments": experiment_summaries(),
        "runs": scan_run_directories(),
        "payload_gt_checks": scan_payload_gt_free(),
        "datasets": {
            "manifest_1086": _load_json(common.DATASETS / "source_meta.json"),
            "references": _load_json(common.REFERENCES / "source_meta.json"),
        },
    }
    common.write_json(common.P1 / "summary.json", payload)

    lines = [
        "# P-1 实验执行汇总（自动生成）",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- Git：`{payload['git']['git_sha']}`（branch {payload['git']['branch']}，dirty={payload['git']['dirty_tree']}，"
        f"{payload['git']['dirty_path_count']} 项）",
        f"- 环境：Python {payload['environment'].get('python')} / torch {payload['environment'].get('torch')} / "
        f"{payload['environment'].get('gpu_name')}",
        "",
        "## 实验状态",
        "",
        "| 实验 | 状态 | 主产物 |",
        "|---|---|---|",
    ]
    for experiment in EXPERIMENTS:
        entry = payload["experiments"].get(experiment, {})
        primary = (
            entry.get("report")
            or entry.get("hardware_report")
            or entry.get("streaming_ability_matrix")
            or entry.get("model_registry")
            or ""
        )
        lines.append(f"| {experiment} | {entry.get('status', 'MISSING')} | `{primary}` |")

    lines += ["", "## 运行目录", "", "| run_id | 实验 | 状态 | 预测行数 | GT-free |", "|---|---|---|---:|---|"]
    for run in payload["runs"]:
        lines.append(
            f"| `{run['run_id']}` | {run['experiment_id']} | {run['run_status']} | {run['model_outputs_rows']} | "
            f"{'yes' if run['model_outputs_gt_free'] else 'NO'} |"
        )

    lines += ["", "## GT 隔离扫描（推理/融合载荷）", ""]
    for check in payload["payload_gt_checks"]:
        lines.append(f"- `{check['path']}`：{check['rows']} 行，GT-free={check['gt_free']}")
    lines.append("")
    lines.append("> 完整机器可读结果：`experiments/p1/summary.json`；每个 run 的 manifest/指标在其目录内。")
    (common.P1 / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(common.P1 / "summary.json"), "runs": len(payload["runs"]),
                      "experiments": list(payload["experiments"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
