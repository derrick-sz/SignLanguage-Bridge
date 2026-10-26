"""Mechanical audit of the P-1 / G2 evidence tree.

This is the deterministic counterpart to a human review: it re-checks the claims
that the reports make against the actual files and the git index.  It writes
``docs/experiments/P-1/P1_INDEPENDENT_AUDIT.md``.

Run::

    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -m experiments.p1.verify_all
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments.p1 import common
from experiments.p1.gt_guard import find_gt_keys

P1 = common.P1


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(common.ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout


def scan_gt_free(path: Path) -> tuple[int, int]:
    rows = 0
    leaks = 0
    if not path.is_file():
        return 0, 0
    for record in common.iter_jsonl(path):
        rows += 1
        leaks += len(find_gt_keys(record))
    return rows, leaks


def main() -> int:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, evidence: str = "") -> None:
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail, "evidence": evidence})

    # 1. run manifests -----------------------------------------------------
    run_dirs = sorted(p.parent for p in (P1 / "runs").glob("*/manifest.json"))
    statuses = {}
    for run_dir in run_dirs:
        manifest = common.read_json(run_dir / "manifest.json")
        statuses[run_dir.name] = manifest.get("run_status")
    bad = {k: v for k, v in statuses.items() if v != "PASS"}
    add(
        "run 目录 manifest 状态",
        not bad,
        f"{len(run_dirs)} 个 run，PASS={len(run_dirs) - len(bad)}，非 PASS={bad or '无'}",
        "experiments/p1/runs/*/manifest.json",
    )

    # 2. GT-free payloads --------------------------------------------------
    payload_paths = [
        P1 / "e02" / "candidate_outputs.jsonl",
        P1 / "e06" / "candidate_inputs.jsonl",
    ] + sorted((P1 / "runs").glob("*/model_outputs.jsonl")) + sorted(
        (P1 / "e02" / "llm_fusion" / "runs").glob("*/fusion_outputs.jsonl")
    )
    total_rows = total_leaks = 0
    for path in payload_paths:
        rows, leaks = scan_gt_free(path)
        total_rows += rows
        total_leaks += leaks
    add(
        "GT 隔离（预测/候选/融合载荷）",
        total_leaks == 0,
        f"扫描 {len(payload_paths)} 个 JSONL、{total_rows} 行，GT 键命中 {total_leaks}",
        ", ".join(str(p.relative_to(common.ROOT)) for p in payload_paths[:3]) + " …",
    )

    # 3. freezen boundary in git ------------------------------------------
    tracked = git("ls-files").splitlines()
    forbidden = [
        p
        for p in tracked
        if "gt_1086.csv" in p
        or "eval_records.jsonl" in p
        or "predictions-raw/" in p
        or "/artifact/" in p
        or p.startswith("Paper-algorithm/")
        or p.endswith(".mp4")
        or p.endswith(".npz")
    ]
    add(
        "git 冻结边界（原始 GT/媒体/vendored 不入库）",
        not forbidden,
        f"tracked={len(tracked)}，违规={forbidden[:5] or '无'}",
        "git ls-files",
    )
    ignore_ok = []
    for candidate in (
        "experiments/p1/references/gt_1086.csv",
        "experiments/p1/runs/e01-pilot50-u1-unisign-video-20261026/eval_records.jsonl",
        "experiments/p1/e06/human_review/README.md",
        "experiments/p1/e05/replay-probe-20261027T0002Z/negatives/black.mp4",
    ):
        out = git("check-ignore", candidate).strip()
        ignore_ok.append(bool(out))
    add("被排除文件仍受 .gitignore 保护", all(ignore_ok), f"check-ignore {ignore_ok}", ".gitignore")

    # 4. headline numbers --------------------------------------------------
    e02 = common.read_json(P1 / "e02" / "translation_metrics.json")
    metrics = {name: payload.get("metrics", {}) for name, payload in e02["paths"].items()}
    expected = {
        "A_unisign_video": 24.8157,
        "A2_unisign_pose": 25.5846,
        "B0_gloss_join": 12.2183,
    }
    ok = all(abs(metrics[name]["bleu4_13a"] - value) < 0.01 for name, value in expected.items())
    add(
        "E02 canonical BLEU",
        ok,
        json.dumps({name: metrics[name]["bleu4_13a"] for name in expected}, ensure_ascii=False),
        "experiments/p1/e02/translation_metrics.json",
    )

    e03 = common.read_json(P1 / "e03" / "calibration_results.json")
    e03_text = json.dumps(e03, ensure_ascii=False)
    add(
        "E03 校准关键值存在",
        ("0.3041" in e03_text or 0.3041 in e03_text) and ("0.7449" in e03_text or 0.7449 in e03_text),
        "raw ECE≈0.304 / OOF AUC≈0.745 在 calibration_results.json 中可检索",
        "experiments/p1/e03/calibration_results.json",
    )

    e04 = (P1 / "e04" / "hardware_report.md").read_text(encoding="utf-8")
    add(
        "E04 热启动 P50 出现在报告",
        all(token in e04 for token in ("276 / 339", "2679 / 3271", "9775 / 11175")),
        "pose 276/339 ms、video 2,679/3,271 ms、SLRT 9,775/11,175 ms",
        "experiments/p1/e04/hardware_report.md",
    )

    e05 = (P1 / "e05" / "prefix_probe_summary.md").read_text(encoding="utf-8")
    add(
        "E05 首稳前缀与改写率",
        ("90.0%" in e05 and "100.0%" in e05 and "0.6714" in e05),
        "mean 90% / median 100% / 改写率 0.6714",
        "experiments/p1/e05/prefix_probe_summary.md",
    )

    e06_rows = [json.loads(line) for line in (P1 / "e06" / "router_ablation.jsonl").open(encoding="utf-8")]
    p2 = next((row for row in e06_rows if row.get("policy") == "P2_rule_cascade_A2"), None)
    add(
        "E06 关键策略行",
        p2 is not None and abs(float(p2.get("second_path_call_rate", 0)) - 0.3812) < 0.001,
        f"P2 A2 调用率={p2.get('second_path_call_rate') if p2 else 'MISSING'}",
        "experiments/p1/e06/router_ablation.jsonl",
    )

    # 5. G2 real run consistency ------------------------------------------
    g2_dir = P1 / "e02" / "llm_fusion" / "runs" / "g2a-wbproxy-50-v2-20261026"
    g2_manifest = common.read_json(g2_dir / "manifest.json")
    g2_metrics = common.read_json(g2_dir / "metrics.json")
    fusion_rows = list(common.iter_jsonl(g2_dir / "fusion_outputs.jsonl"))
    ok_rows = [row for row in fusion_rows if str(row.get("status")) in ("ok", "OK", "success")]
    ok_by_group = {}
    for row in ok_rows:
        ok_by_group[row.get("group")] = ok_by_group.get(row.get("group"), 0) + 1
    metric_ok_by_group = {group: payload.get("n_ok") for group, payload in g2_metrics["groups"].items()}
    consistent = {
        group: (ok_by_group.get(group, 0) == count) for group, count in metric_ok_by_group.items()
    }
    add(
        "G2 真实批：成功子集与 metrics 一致",
        all(consistent.values()),
        f"fusion_outputs ok={ok_by_group}；metrics n_ok={metric_ok_by_group}",
        "experiments/p1/e02/llm_fusion/runs/g2a-wbproxy-50-v2-20261026/",
    )
    endpoint = g2_metrics.get("llm_endpoint", {})
    add(
        "G2 真实批：端点/模型/参数可追溯",
        endpoint.get("provider") == "wb-proxy"
        and endpoint.get("model") == "deepseek-v4.1-flash"
        and g2_manifest.get("decoding", {}).get("max_tokens") in (2048, None)
        and endpoint.get("credentials_ref", "").startswith("WB_PROXY_API_KEY"),
        f"provider={endpoint.get('provider')} model={endpoint.get('model')} "
        f"credentials_ref={endpoint.get('credentials_ref')}",
        "metrics.json llm_endpoint",
    )
    add(
        "G2 真实批：失败数披露",
        g2_metrics["groups"]["D"]["failed_calls"] > 0
        and "n_ok" in g2_metrics["groups"]["D"]
        and "REAL_EXPLORATORY_50" == g2_metrics.get("data_status"),
        f"D failed={g2_metrics['groups']['D']['failed_calls']}、E failed={g2_metrics['groups']['E']['failed_calls']}、"
        f"data_status={g2_metrics.get('data_status')}",
        "metrics.json",
    )
    key_scan = subprocess.run(
        ["git", "grep", "-l", "-I", "-E", "sk-[A-Za-z0-9]{20,}", "--", "experiments/"],
        cwd=str(common.ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    add("密钥扫描（sk- 形式）", not key_scan, f"命中文件: {key_scan or '无'}", "git grep")

    rows = "\n".join(
        f"| {item['check']} | {item['result']} | {item['detail']} | `{item['evidence']}` |" for item in checks
    )
    passed = sum(1 for item in checks if item["result"] == "PASS")
    report = f"""# P-1 / G2 独立机械审计（自动生成）

- 生成时间：{common.utc_now()}
- 工具：`experiments/p1/verify_all.py`（确定性脚本；不依赖 LLM 判断）
- Git：`{common.git_state()['git_sha']}`（dirty={common.git_state()['dirty_tree']}）
- 结论：**{passed}/{len(checks)} 项 PASS**

| 检查项 | 结果 | 细节 | 证据 |
|---|---|---|---|
{rows}

## 说明

- 本审计只验证"报告声称 ↔ 实际文件"的一致性，不重复评价模型质量。
- GT-free 扫描覆盖预测/候选/融合载荷；`eval_records.jsonl`、`gt_1086.csv`、`human_review/` 属 EVAL ONLY，按 FREEZE 边界不入 git，**不视为违规**。
- G2 数字为 50 条探索批（`REAL_EXPLORATORY_50`），不是最终 G2 结论。
"""
    out = common.PLAN_DIR / "P1_INDEPENDENT_AUDIT.md"
    out.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(out), "passed": passed, "total": len(checks), "failed": [c for c in checks if c["result"] == "FAIL"]}, ensure_ascii=False, indent=2))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
