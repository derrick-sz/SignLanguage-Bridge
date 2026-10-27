"""Round-2 人工语义评审包生成器（A / A2 / S2 盲评）。

目的：为「stage2 pose 子集是否可替换 stage1 pose-only（A2）作为生产默认」提供人工语义证据。
- A   = Uni-Sign 视频端到端（candidate_outputs, A）
- A2  = Uni-Sign stage1 官方 pose（candidate_outputs, A2；当前 pose 默认）
- S2  = Uni-Sign stage2 pose 子集（纯 pose 架构，2026-10-27 新候选）

抽样：signer 分层、每签者 3 条、**无偏随机**（不做难度优先——Round-2 的目的是估计 A2 与 S2 的真实差异，
难例优先会放大差异并让"纠正/改错"率失真）。seed=20261030；盲序 seed=20261031，同一样本候选不相邻。

产物（本目录）：round2_review_template.csv / .xlsx、blind_map_EVAL_ONLY.csv、sampling_manifest.json、README.md

用法：
    python make_round2_package.py build     # 生成包 + xlsx + 结构校验
    python make_round2_package.py check     # 只做结构校验
    python make_round2_package.py stats --review-file <filled.xlsx|.csv>   # 回填后统计（EVAL ONLY）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]

SEED_SAMPLING = 20261030
SEED_BLIND = 20261031
N_PER_SIGNER = 3
CANDIDATES = ("A", "A2", "S2")

COLS = [
    "行号", "review_id", "参考译文_EVAL_ONLY_仅评审用", "候选译文",
    "评分0-4", "人物_是否一致", "否定_是否一致", "数字_是否一致", "时间_是否一致", "动作_是否一致", "对象_是否一致",
    "严重语义错误", "流畅度1-5", "可用于交流", "评审人", "评审日期", "备注",
    "复核人", "复核评分0-4", "复核意见",
]
READONLY = COLS[:4]
REQUIRED_FILLED = ["评分0-4", "人物_是否一致", "否定_是否一致", "数字_是否一致", "时间_是否一致",
                   "动作_是否一致", "对象_是否一致", "严重语义错误", "流畅度1-5", "可用于交流",
                   "评审人", "评审日期"]

MODEL_INFO = {
    "A": {"model_id": "unisign_rgb_video", "model_revision": "unisign-video-1086-paperrefs",
          "source_run_id": "e02-offline-1086-20261026T134508Z"},
    "A2": {"model_id": "uni_sign_pose_slt_stage1", "model_revision": "csl_daily_pose_only_slt.pth",
           "source_run_id": "e02-offline-1086-20261026T134508Z"},
    "S2": {"model_id": "uni_sign_pose_slt_stage2_subset", "model_revision": "csl_daily_stage2_pose_subset.pth",
           "source_run_id": "stage2-pose-subset-1086-20261027"},
}

CANDIDATE_SOURCES = {
    "A": ("experiments/p1/e02/candidate_outputs.jsonl", "A"),
    "A2": ("experiments/p1/e02/candidate_outputs.jsonl", "A2"),
    "S2": ("experiments/p1/e01/runs/stage2-pose-subset-1086-20261027/predictions_stage2_pose_subset.jsonl", None),
}


def norm(text: str | None) -> str:
    return "".join(unicodedata.normalize("NFKC", str(text or "")).split())


def edit_sim(a: str | None, b: str | None) -> float:
    x, y = norm(a), norm(b)
    if not x and not y:
        return 1.0
    prev = list(range(len(y) + 1))
    for i, ca in enumerate(x, 1):
        cur = [i]
        for j, cb in enumerate(y, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(1, len(x), len(y))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for cand, (rel, cand_id) in CANDIDATE_SOURCES.items():
        path = REPO / rel
        if not path.is_file():
            raise SystemExit(f"missing prediction source for {cand}: {path}")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if cand_id is not None:
                    if row.get("status") != "OK" or row.get("candidate_id") != cand_id:
                        continue
                    text = row["output"]["text"]
                    sid = row["sample_id"]
                else:
                    if row.get("status") != "ok":
                        continue
                    text = row.get("text") or ""
                    sid = row["sample_id"]
                if not text:
                    continue
                out.setdefault(sid, {})[cand] = text
    return out


def load_references() -> dict[str, str]:
    path = REPO / "experiments/p1/references/gt_1086.csv"
    with open(path, encoding="utf-8-sig") as fh:
        return {r["sample_id"]: r["ref_sentence_upstream"] for r in csv.DictReader(fh)}


def select_samples(preds: dict[str, dict[str, str]]) -> tuple[list[str], dict[str, list[str]]]:
    usable = [s for s, d in preds.items() if all(c in d for c in CANDIDATES)]
    by_signer: dict[str, list[str]] = {}
    for sid in sorted(usable):
        by_signer.setdefault(sid.split("_")[1], []).append(sid)
    rng = random.Random(SEED_SAMPLING)
    chosen: dict[str, list[str]] = {}
    for signer in sorted(by_signer):
        pool = by_signer[signer]
        chosen[signer] = rng.sample(pool, min(N_PER_SIGNER, len(pool)))
    samples = [sid for signer in sorted(chosen) for sid in chosen[signer]]
    return samples, chosen


def build_package() -> int:
    preds = load_predictions()
    refs = load_references()
    samples, by_signer = select_samples(preds)
    missing = [s for s in samples if s not in refs]
    if missing:
        raise SystemExit(f"missing references for {len(missing)} samples: {missing[:3]}")

    rows: list[tuple[str, str, str]] = [(s, c, preds[s][c]) for s in samples for c in CANDIDATES]
    rng = random.Random(SEED_BLIND)
    for _ in range(500):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        if all(not (shuffled[i][0] == shuffled[i + 1][0]) for i in range(len(shuffled) - 1)):
            break
    else:
        raise SystemExit("could not blind-shuffle without adjacent same-sample rows")

    csv_path = HERE / "round2_review_template.csv"
    blind_path = HERE / "blind_map_EVAL_ONLY.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLS)
        for index, (sid, cand, text) in enumerate(shuffled, 1):
            row = {c: "" for c in COLS}
            row["行号"] = str(index)
            row["review_id"] = f"R2-{index:04d}"
            row["参考译文_EVAL_ONLY_仅评审用"] = refs[sid].replace(" ", "")
            row["候选译文"] = text
            writer.writerow([row[c] for c in COLS])

    blind_cols = ["review_id", "行号", "sample_id", "signer_id", "candidate_id", "model_id", "model_revision", "source_run_id"]
    with open(blind_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(blind_cols)
        for index, (sid, cand, _text) in enumerate(shuffled, 1):
            info = MODEL_INFO[cand]
            writer.writerow([f"R2-{index:04d}", index, sid, sid.split("_")[1], cand,
                             info["model_id"], info["model_revision"], info["source_run_id"]])

    candidate_counts = {c: sum(1 for _s, cc, _t in shuffled if cc == c) for c in CANDIDATES}
    manifest = {
        "package": "G2-round2-human-review",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "判断 stage2 pose 子集（S2）是否可在人工语义维度替代 stage1 pose-only（A2）；A 作为视频路径参照",
        "parent_package": "../quick_round1_90/",
        "reviewer": "user (Round-2 single reviewer; 复核列预留第二评审)",
        "data_scope": "CSL-Daily test retrospective_seen_1086 (NOT a fresh blind test)",
        "sampling": {
            "rule": "signer-stratified, 3 samples per signer, UNBIASED uniform random within each signer "
                    "(no difficulty prioritisation: Round-2 estimates the real A2-vs-S2 difference, "
                    "hard-case prioritisation would inflate it)",
            "seed": SEED_SAMPLING,
            "n_samples": len(samples),
            "n_rows": len(shuffled),
            "candidates": list(CANDIDATES),
            "signer_counts": {s: len(v) for s, v in sorted(by_signer.items())},
            "candidate_counts": candidate_counts,
            "sample_ids": samples,
            "sample_ids_sha256": hashlib.sha256("\n".join(samples).encode()).hexdigest(),
        },
        "blind_order": {
            "method": "global deterministic shuffle of all rows (random.Random(SEED_BLIND))",
            "seed": SEED_BLIND,
            "per_sample_candidates_are_not_adjacent": True,
            "identity_hidden_in_review_sheet": True,
            "mapping_file": "blind_map_EVAL_ONLY.csv",
            "mapping_file_is_eval_only": True,
        },
        "files": {},
        "inputs": {
            "A": CANDIDATE_SOURCES["A"][0], "A2": CANDIDATE_SOURCES["A2"][0], "S2": CANDIDATE_SOURCES["S2"][0],
            "references_EVAL_ONLY": "experiments/p1/references/gt_1086.csv",
        },
        "gt_discipline": "review sheet carries the reference text for human judgement only; it must never be fed "
                         "back into models/prompts; blind map is EVAL ONLY",
    }

    try:
        from xlsx_writer import write_workbook  # local helper
        xlsx_path = HERE / "round2_review_template.xlsx"
        write_workbook(csv_path, xlsx_path, COLS)
        manifest["files"]["round2_review_template.xlsx"] = {
            "path": xlsx_path.name, "sha256": sha256(xlsx_path), "rows": len(shuffled)}
        manifest["xlsx_build"] = {"tool": "openpyxl", "n_rows": len(shuffled)}
    except Exception as error:  # pragma: no cover - xlsx optional
        manifest["xlsx_build"] = {"error": f"{type(error).__name__}: {error}"}

    manifest["files"]["round2_review_template.csv"] = {
        "path": csv_path.name, "sha256": sha256(csv_path), "rows": len(shuffled)}
    manifest["files"]["blind_map_EVAL_ONLY.csv"] = {
        "path": blind_path.name, "sha256": sha256(blind_path), "rows": len(shuffled)}
    (HERE / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_readme(manifest, samples)
    return check_package()


def write_readme(manifest: dict[str, Any], samples: list[str]) -> None:
    text = f"""# Round-2 人工语义评审包（A / A2 / S2，{len(samples)} 样本 × 3 候选 = {manifest['sampling']['n_rows']} 行）

**目的**：判断 stage2 pose 子集（S2）能否在人工语义维度替代 stage1 pose-only（A2）成为 pose 默认；A（视频）作参照。

| 候选 | 含义 |
|---|---|
| A | Uni-Sign 视频端到端（现有视频路径） |
| A2 | Uni-Sign stage1 官方 pose（**当前 pose 默认**） |
| S2 | Uni-Sign stage2 pose 子集（纯 pose 架构，BLEU 26.3026 vs A2 25.5846，零延迟代价） |

## 三步走

1. 打开 `round2_review_template.xlsx`（或 CSV），只填黄色输入列：
   `评分0-4`、六个槽位（是/否/无法判断）、`严重语义错误`、`流畅度1-5`、`可用于交流`、`评审人`、`评审日期`。
2. 另存为 `round2_review_filled.xlsx`（不要覆盖模板）。
3. 校验/统计：
   ```powershell
   C:\\Python314\\python.exe make_round2_package.py stats --review-file round2_review_filled.xlsx
   ```

## 抽样与盲序

- signer 分层，每签者 {N_PER_SIGNER} 条 = {len(samples)} 样本；**无偏随机**（seed={SEED_SAMPLING}）
- 全局盲序（seed={SEED_BLIND}），同一样本三个候选不相邻；模型身份不在评审表
- `blind_map_EVAL_ONLY.csv` 是 review_id↔候选 映射，**评审时不要看**，只用于事后 join 统计

## 边界

- 数据是 `retrospective_seen_1086` 子集，不是新盲测；参考译文仅供判定语义，**不得回灌模型/提示词**
- 本包是 AI/人工评审材料，结论只作质量维度证据，不替代部署泛化验收
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


def check_package() -> int:
    checks: list[tuple[str, bool, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append((name, bool(ok), detail))

    csv_path = HERE / "round2_review_template.csv"
    blind_path = HERE / "blind_map_EVAL_ONLY.csv"
    manifest_path = HERE / "sampling_manifest.json"
    for path in (csv_path, blind_path, manifest_path):
        check(f"exists: {path.name}", path.is_file(), str(path))
    if not all(ok for _n, ok, _d in checks):
        return report(checks)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    with open(blind_path, encoding="utf-8") as fh:
        blind = list(csv.DictReader(fh))

    n_samples = manifest["sampling"]["n_samples"]
    check("review rows == 3 x samples", len(rows) == 3 * n_samples, len(rows))
    check("blind rows == review rows", len(blind) == len(rows), len(blind))
    check("header order matches schema", list(rows[0].keys()) == COLS, list(rows[0].keys())[:4])
    check("review_id unique", len({r["review_id"] for r in rows}) == len(rows), len(rows))
    check("review_id format R2-NNNN", all(r["review_id"].startswith("R2-") and r["review_id"][3:].isdigit() for r in rows))
    check("行号 sequential", [r["行号"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)])
    check("blind ids match review ids", [b["review_id"] for b in blind] == [r["review_id"] for r in rows])
    check("every sample has all 3 candidates",
          all(sorted(b["candidate_id"] for b in blind if b["sample_id"] == s) == sorted(CANDIDATES)
              for s in {b["sample_id"] for b in blind}))
    check("candidate counts", {c: sum(1 for b in blind if b["candidate_id"] == c) for c in CANDIDATES}
          == {c: n_samples for c in CANDIDATES})
    check("manifest candidate counts", manifest["sampling"]["candidate_counts"] == {c: n_samples for c in CANDIDATES},
          manifest["sampling"]["candidate_counts"])
    check("no same-sample adjacency",
          all(blind[i]["sample_id"] != blind[i + 1]["sample_id"] for i in range(len(blind) - 1)))
    check("candidate identity hidden in review sheet",
          not any(c["model_id"] in r["候选译文"] for r in rows for c in [{"model_id": "unisign"}]))
    check("fillable columns empty in template",
          all(not r[c] for r in rows for c in REQUIRED_FILLED))
    check("reference column present and non-empty",
          all(r["参考译文_EVAL_ONLY_仅评审用"] for r in rows))
    check("manifest sample_ids match blind map",
          sorted(manifest["sampling"]["sample_ids"]) == sorted({b["sample_id"] for b in blind}))
    check("manifest csv hash", manifest["files"]["round2_review_template.csv"]["sha256"] == sha256(csv_path))
    check("manifest blind hash", manifest["files"]["blind_map_EVAL_ONLY.csv"]["sha256"] == sha256(blind_path))
    if (HERE / "round2_review_template.xlsx").is_file():
        check("manifest xlsx hash", manifest["files"]["round2_review_template.xlsx"]["sha256"]
              == sha256(HERE / "round2_review_template.xlsx"))
    return report(checks)


def report(checks: list[tuple[str, bool, Any]]) -> int:
    failed = [c for c in checks if not c[1]]
    print(json.dumps({"checks_total": len(checks), "checks_failed": len(failed),
                      "failed": [{"name": n, "detail": d} for n, _ok, d in failed],
                      "status": "PASS" if not failed else "FAIL"}, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


def read_review_file(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb["评审表"] if "评审表" in wb.sheetnames else wb[wb.sheetnames[0]]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(v) if v is not None else "" for v in rows[0]]
        return [dict(zip(header, ["" if v is None else str(v) for v in row])) for row in rows[1:] if any(row)]
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def stats(review_file: Path) -> int:
    rows = read_review_file(review_file)
    with open(HERE / "blind_map_EVAL_ONLY.csv", encoding="utf-8") as fh:
        blind = {b["review_id"]: b for b in csv.DictReader(fh)}
    complete = [r for r in rows if all(str(r.get(c) or "").strip() for c in REQUIRED_FILLED)]
    missing = [r["review_id"] for r in rows if r not in complete]
    # Partial fills are expected while reviewing: statistics run on complete rows only.
    per_candidate: dict[str, list[dict[str, str]]] = {c: [] for c in CANDIDATES}
    for row in complete:
        cand = blind[row["review_id"]]["candidate_id"]
        per_candidate[cand].append(row)

    def score(r: dict[str, str]) -> float:
        return float(r["评分0-4"])

    summary: dict[str, Any] = {"n_rows": len(rows), "incomplete_rows": len(missing), "per_candidate": {}}
    for cand, items in per_candidate.items():
        if not items:
            continue
        slots = {}
        for slot in ["人物_是否一致", "否定_是否一致", "数字_是否一致", "时间_是否一致", "动作_是否一致", "对象_是否一致"]:
            answered = [r[slot] for r in items if r[slot] in {"是", "否"}]
            wrong = sum(1 for v in answered if v == "否")
            slots[slot] = {"n": len(answered), "error_rate": (wrong / len(answered)) if answered else None}
        summary["per_candidate"][cand] = {
            "n": len(items),
            "mean_score": statistics.fmean(score(r) for r in items),
            "severe_error_rate": sum(1 for r in items if r["严重语义错误"] == "是") / len(items),
            "mean_fluency": statistics.fmean(float(r["流畅度1-5"]) for r in items),
            "usable_yes_rate": sum(1 for r in items if r["可用于交流"] == "是") / len(items),
            "slot_error_rates": slots,
        }
    by_sample: dict[str, dict[str, float]] = {}
    for row in complete:
        info = blind[row["review_id"]]
        by_sample.setdefault(info["sample_id"], {})[info["candidate_id"]] = score(row)
    def delta(a: str, b: str) -> dict[str, Any] | None:
        pairs = [(v[a], v[b]) for v in by_sample.values() if a in v and b in v]
        if not pairs:
            return None
        diffs = [x - y for x, y in pairs]
        rng = random.Random(SEED_BLIND)
        boots = []
        for _ in range(1000):
            sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
            boots.append(statistics.fmean(sample))
        boots.sort()
        return {"n_pairs": len(pairs), "delta": statistics.fmean(diffs),
                "ci95": [boots[24], boots[974]],
                "improved": sum(1 for d in diffs if d > 0), "worsened": sum(1 for d in diffs if d < 0)}
    summary["paired_vs_a2"] = {c: delta(c, "A2") for c in CANDIDATES if c != "A2"}
    out = HERE / "round2_stats.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "report": str(out)}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round-2 review package (A/A2/S2)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    stats_parser = sub.add_parser("stats")
    stats_parser.add_argument("--review-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        return build_package()
    if args.command == "check":
        return check_package()
    return stats(args.review_file)


if __name__ == "__main__":
    raise SystemExit(main())
