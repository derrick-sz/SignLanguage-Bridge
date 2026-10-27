"""Round-3 人工语义评审包生成器（Rubric v2，双人独立盲评）。

目的：用修订后的量表，对 A / A2 / S2 做**双人独立盲评**，回答两件事：
1. S2 相对 A2 是否降低「关键事实错误」风险（正交定义的严重语义错误）；
2. S2 相对 A2 是否提升「可支持可靠沟通」率（v2 新增、与产品就绪直接相关）。

与 Round-1/2 的差异：
- 删除单一「可用于交流」，拆为「可理解部分信息」与「可支持可靠沟通」；
- 「严重语义错误」给正交定义（六槽位任一被改变且导致误解），不再等同于低分；
- **两份独立表**（reviewer A / reviewer B），各自盲填，事后计算一致性与仲裁清单；
- 复用 Round-2 的同一 30 个样本（seed 20261030，signer 分层无偏），但**新的盲序 seed**（20261032），
  以便与 Round-2 结果做同题对照，同时不暴露 Round-2 的行序。

用法：
    python make_round3_package.py build
    python make_round3_package.py check
    python make_round3_package.py stats --reviewer-a <A.xlsx|.csv> --reviewer-b <B.xlsx|.csv>
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


def _find_repo(start: Path) -> Path:
    """Ascend until the directory that contains experiments/p1 (works from any depth)."""
    for candidate in (start, *start.parents):
        if (candidate / "experiments" / "p1").is_dir():
            return candidate
    raise SystemExit(f"cannot locate repo root from {start}")


REPO = _find_repo(HERE)

SEED_SAMPLING = 20261030   # 与 Round-2 相同 -> 同一批 30 个样本（同题对照）
SEED_BLIND = 20261032      # Round-3 新盲序
N_PER_SIGNER = 3
CANDIDATES = ("A", "A2", "S2")
REVIEWERS = ("A", "B")

COLS = [
    "行号", "review_id", "参考译文_EVAL_ONLY_仅评审用", "候选译文",
    "评分0-4",
    "人物_是否一致", "否定_是否一致", "数字_是否一致", "时间_是否一致", "动作_是否一致", "对象_是否一致",
    "严重语义错误", "可理解部分信息", "可支持可靠沟通", "流畅度1-5",
    "备注", "评审人", "评审日期",
]
SLOTS = ["人物_是否一致", "否定_是否一致", "数字_是否一致", "时间_是否一致", "动作_是否一致", "对象_是否一致"]
REQUIRED_FILLED = ["评分0-4", *SLOTS, "严重语义错误", "可理解部分信息", "可支持可靠沟通",
                   "流畅度1-5", "评审人", "评审日期"]

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
                    text, sid = row["output"]["text"], row["sample_id"]
                else:
                    if row.get("status") != "ok":
                        continue
                    text, sid = row.get("text") or "", row["sample_id"]
                if text:
                    out.setdefault(sid, {})[cand] = text
    return out


def load_references() -> dict[str, str]:
    with open(REPO / "experiments/p1/references/gt_1086.csv", encoding="utf-8-sig") as fh:
        return {r["sample_id"]: r["ref_sentence_upstream"] for r in csv.DictReader(fh)}


def select_samples(preds: dict[str, dict[str, str]]) -> tuple[list[str], dict[str, list[str]]]:
    usable = [s for s, d in preds.items() if all(c in d for c in CANDIDATES)]
    by_signer: dict[str, list[str]] = {}
    for sid in sorted(usable):
        by_signer.setdefault(sid.split("_")[1], []).append(sid)
    rng = random.Random(SEED_SAMPLING)
    chosen = {sg: rng.sample(by_signer[sg], min(N_PER_SIGNER, len(by_signer[sg]))) for sg in sorted(by_signer)}
    return [sid for sg in sorted(chosen) for sid in chosen[sg]], chosen


def build_package() -> int:
    preds = load_predictions()
    refs = load_references()
    samples, by_signer = select_samples(preds)
    rows = [(s, c, preds[s][c]) for s in samples for c in CANDIDATES]

    rng = random.Random(SEED_BLIND)
    for _ in range(500):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        if all(shuffled[i][0] != shuffled[i + 1][0] for i in range(len(shuffled) - 1)):
            break
    else:
        raise SystemExit("could not blind-shuffle without adjacent same-sample rows")

    csv_path = HERE / "round3_review_template.csv"
    blind_path = HERE / "blind_map_EVAL_ONLY.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLS)
        for index, (sid, cand, text) in enumerate(shuffled, 1):
            row = {c: "" for c in COLS}
            row["行号"] = str(index)
            row["review_id"] = f"R3-{index:04d}"
            row["参考译文_EVAL_ONLY_仅评审用"] = refs[sid].replace(" ", "")
            row["候选译文"] = text
            writer.writerow([row[c] for c in COLS])

    blind_cols = ["review_id", "行号", "sample_id", "signer_id", "candidate_id", "model_id", "model_revision", "source_run_id"]
    with open(blind_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(blind_cols)
        for index, (sid, cand, _t) in enumerate(shuffled, 1):
            info = MODEL_INFO[cand]
            writer.writerow([f"R3-{index:04d}", index, sid, sid.split("_")[1], cand,
                             info["model_id"], info["model_revision"], info["source_run_id"]])

    files: dict[str, Any] = {}
    manifest = {
        "package": "G2-round3-human-review-rubric-v2",
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "双人独立盲评，用 Rubric v2 确认 S2 相对 A2 的关键事实错误风险与可靠沟通率",
        "rubric": {
            "version": "v2",
            "changes_from_v1": [
                "删除单一『可用于交流』，拆为『可理解部分信息』与『可支持可靠沟通』",
                "『严重语义错误』给出正交定义：六槽位任一项被改变且导致误解（不再等同于低分）",
                "『可支持可靠沟通=是』建议仅在 评分≥3 且 严重语义错误=否 时填写",
            ],
            "reviewers": list(REVIEWERS),
            "review_type": "two independent human reviewers (blind), arbitration by a third party/user",
        },
        "data_scope": "CSL-Daily test retrospective_seen_1086 (NOT a fresh blind test)",
        "sampling": {
            "rule": "signer-stratified unbiased random, 3 per signer; SAME 30 samples as Round-2 for comparability",
            "seed": SEED_SAMPLING, "n_samples": len(samples), "n_rows": len(shuffled),
            "candidates": list(CANDIDATES),
            "signer_counts": {sg: len(v) for sg, v in sorted(by_signer.items())},
            "candidate_counts": {c: sum(1 for _s, cc, _t in shuffled if cc == c) for c in CANDIDATES},
            "sample_ids": samples,
            "sample_ids_sha256": hashlib.sha256("\n".join(samples).encode()).hexdigest(),
        },
        "blind_order": {"method": "global deterministic shuffle (random.Random(SEED_BLIND))",
                        "seed": SEED_BLIND, "per_sample_candidates_are_not_adjacent": True,
                        "mapping_file": "blind_map_EVAL_ONLY.csv", "mapping_file_is_eval_only": True},
        "inputs": {c: CANDIDATE_SOURCES[c][0] for c in CANDIDATES} | {
            "references_EVAL_ONLY": "experiments/p1/references/gt_1086.csv"},
        "gt_discipline": "review sheet carries the reference for human judgement only; blind map is EVAL ONLY; "
                         "frozen scores must contain no reference/candidate text",
        "files": files,
    }

    try:
        from xlsx_writer_v2 import write_workbook
        for reviewer in REVIEWERS:
            xlsx = HERE / f"round3_reviewer_{reviewer}_template.xlsx"
            info = write_workbook(csv_path, xlsx, COLS, reviewer=reviewer)
            files[xlsx.name] = {"path": xlsx.name, "sha256": sha256(xlsx), "rows": info["rows"],
                                "validations": info["validations"], "reviewer": reviewer}
    except Exception as error:  # pragma: no cover
        files["xlsx_error"] = {"error": f"{type(error).__name__}: {error}"}

    files[csv_path.name] = {"path": csv_path.name, "sha256": sha256(csv_path), "rows": len(shuffled)}
    files[blind_path.name] = {"path": blind_path.name, "sha256": sha256(blind_path), "rows": len(shuffled)}
    (HERE / "sampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_readme(manifest, samples)
    return check_package()


def write_readme(manifest: dict[str, Any], samples: list[str]) -> None:
    text = f"""# Round-3 人工语义评审包（Rubric v2 · 双人独立盲评）

**目的**：用修订量表确认 ①S2 相对 A2 是否降低关键事实错误风险；②是否提升「可支持可靠沟通」率。
**规模**：{len(samples)} 样本 × 3 候选（A/A2/S2）= {manifest['sampling']['n_rows']} 行；与 Round-2 **同一批样本**，新盲序。

## 文件

| 文件 | 用途 |
|---|---|
| `round3_reviewer_A_template.xlsx` | 评审人 A 的填写表 |
| `round3_reviewer_B_template.xlsx` | 评审人 B 的填写表（独立，不要互相参考） |
| `round3_review_template.csv` | 同内容 CSV（便于脚本/替代填写） |
| `blind_map_EVAL_ONLY.csv` | review_id ↔ 候选 映射（**评审时不要看**） |
| `sampling_manifest.json` | 抽样/盲序/哈希/规则 |

## 量表变化（v2）

| 列 | 说明 |
|---|---|
| 评分0-4 | 同 v1（整体语义忠实度） |
| 六槽位 | 同 v1（人物/否定/数字/时间/动作/对象 是否与参考一致） |
| **严重语义错误** | **新定义**：六槽位任一被改变且会导致误解 → 是（与评分正交，评分 3 也可能是） |
| **可理解部分信息** | 能大致猜到大意 → 是/否/无法判断 |
| **可支持可靠沟通** | **关键**：含义准确到可直接用于交流（建议 评分≥3 且 严重错误=否 才填是） |
| 流畅度1-5 | 同 v1 |

## 流程

1. A、B 各自打开自己的 xlsx，**独立**填写黄色列，另存为 `round3_reviewer_A_filled.xlsx` / `round3_reviewer_B_filled.xlsx`。
2. 汇总（含一致性与仲裁清单）：
   ```powershell
   C:\\Python314\\python.exe make_round3_package.py stats --reviewer-a round3_reviewer_A_filled.xlsx --reviewer-b round3_reviewer_B_filled.xlsx
   ```
3. 对 评分 或 严重语义错误 不一致的行，按仲裁清单由第三人或用户裁定。

## 判定标准（写死在 stats 里）

- **关键事实错误风险**：S2 的 `严重语义错误` 率 **不高于** A2，且「消除 > 新增」（A2=是→S2=否 计数 > A2=否→S2=是）；
- **可靠沟通**：S2 的 `可支持可靠沟通=是` 率 **不低于** A2，且无「可用性丢失」（A2=是→S2=否 计数为 0 或极少）；
- **一致性**：两人在 `评分`（±1 内）与 `严重语义错误` 上的 agreement 与 kappa 需报告。

## 边界

- 数据是 `retrospective_seen_1086` 子集，**不是新盲测**；结论只能作质量维度证据，不替代新 signer 泛化验证。
- 参考译文仅供判定语义，不得回灌模型/提示词。
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


def check_package() -> int:
    checks: list[tuple[str, bool, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append((name, bool(ok), detail))

    csv_path = HERE / "round3_review_template.csv"
    blind_path = HERE / "blind_map_EVAL_ONLY.csv"
    manifest_path = HERE / "sampling_manifest.json"
    for path in (csv_path, blind_path, manifest_path):
        check(f"exists: {path.name}", path.is_file())
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
    check("header order matches", list(rows[0].keys()) == COLS)
    check("v2 columns present", "可支持可靠沟通" in rows[0] and "可理解部分信息" in rows[0]
          and "可用于交流" not in rows[0])
    check("review_id unique", len({r["review_id"] for r in rows}) == len(rows))
    check("review_id format R3-NNNN", all(r["review_id"].startswith("R3-") and r["review_id"][3:].isdigit() for r in rows))
    check("行号 sequential", [r["行号"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)])
    check("blind ids match review ids", [b["review_id"] for b in blind] == [r["review_id"] for r in rows])
    check("every sample has all 3 candidates",
          all(sorted(b["candidate_id"] for b in blind if b["sample_id"] == s) == sorted(CANDIDATES)
              for s in {b["sample_id"] for b in blind}))
    check("candidate counts", {c: sum(1 for b in blind if b["candidate_id"] == c) for c in CANDIDATES}
          == {c: n_samples for c in CANDIDATES})
    check("no same-sample adjacency",
          all(blind[i]["sample_id"] != blind[i + 1]["sample_id"] for i in range(len(blind) - 1)))
    check("fillable columns empty in template", all(not r[c] for r in rows for c in REQUIRED_FILLED))
    check("reference column non-empty", all(r["参考译文_EVAL_ONLY_仅评审用"] for r in rows))
    check("reference has no char-spacing", not any(" " in r["参考译文_EVAL_ONLY_仅评审用"] for r in rows))
    check("manifest sample_ids match blind map",
          sorted(manifest["sampling"]["sample_ids"]) == sorted({b["sample_id"] for b in blind}))
    check("manifest csv hash", manifest["files"][csv_path.name]["sha256"] == sha256(csv_path))
    check("manifest blind hash", manifest["files"][blind_path.name]["sha256"] == sha256(blind_path))
    for reviewer in REVIEWERS:
        xlsx = HERE / f"round3_reviewer_{reviewer}_template.xlsx"
        check(f"reviewer {reviewer} xlsx exists", xlsx.is_file())
        if xlsx.is_file():
            check(f"reviewer {reviewer} xlsx hash",
                  manifest["files"].get(xlsx.name, {}).get("sha256") == sha256(xlsx))
    check("manual-review flag: same samples as Round-2 seed", manifest["sampling"]["seed"] == SEED_SAMPLING)
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


def _kappa(pairs: list[tuple[str, str]], weights: str | None = None) -> float | None:
    """Cohen's kappa（weights=None 无序；'quadratic' 二次加权，用于 0-4 有序量表）。"""
    if not pairs:
        return None
    labels = sorted({a for p in pairs for a in p})
    try:
        values = sorted(float(x) for x in labels)
        ordinal = len(values) >= 2
    except ValueError:
        values, ordinal = [], False
    index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    table = [[0] * k for _ in range(k)]
    for a, b in pairs:
        table[index[a]][index[b]] += 1
    n = len(pairs)
    row = [sum(table[i]) for i in range(k)]
    col = [sum(table[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        if weights != "quadratic" or not ordinal:
            return 0.0 if i == j else 1.0
        span = (k - 1) ** 2
        return ((i - j) ** 2) / span if span else 0.0

    observed = sum(weight(i, j) * table[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return None
    return 1.0 - observed / expected


def stats(reviewer_a: Path, reviewer_b: Path | None) -> int:
    blind = {b["review_id"]: b for b in csv.DictReader(open(HERE / "blind_map_EVAL_ONLY.csv", encoding="utf-8"))}
    texts = {"A": read_review_file(reviewer_a)}
    if reviewer_b is not None:
        texts["B"] = read_review_file(reviewer_b)

    summary: dict[str, Any] = {"reviewers": {}, "per_candidate": {}, "paired_vs_a2": {}, "agreement": {}}
    complete_by_reviewer: dict[str, list[dict[str, str]]] = {}
    for who, rows in texts.items():
        complete = [r for r in rows if r.get("review_id") and all(str(r.get(c) or "").strip() for c in REQUIRED_FILLED)]
        complete_by_reviewer[who] = complete
        summary["reviewers"][who] = {"rows": len(rows), "complete": len(complete),
                                     "incomplete": len(rows) - len(complete)}

    def aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
        per: dict[str, list[dict[str, str]]] = {c: [] for c in CANDIDATES}
        for r in rows:
            per[blind[r["review_id"]]["candidate_id"]].append(r)
        out: dict[str, Any] = {}
        for cand, items in per.items():
            if not items:
                continue
            slots = {}
            for slot in SLOTS:
                answered = [r[slot] for r in items if r[slot] in {"是", "否"}]
                wrong = sum(1 for v in answered if v == "否")
                slots[slot] = {"n": len(answered), "error_rate": (wrong / len(answered)) if answered else None}
            out[cand] = {
                "n": len(items),
                "mean_score": statistics.fmean(float(r["评分0-4"]) for r in items),
                "severe_fact_error_rate": sum(1 for r in items if r["严重语义错误"] == "是") / len(items),
                "understandable_rate": sum(1 for r in items if r["可理解部分信息"] == "是") / len(items),
                "reliable_communication_rate": sum(1 for r in items if r["可支持可靠沟通"] == "是") / len(items),
                "mean_fluency": statistics.fmean(float(r["流畅度1-5"]) for r in items),
                "slot_error_rates": slots,
            }
        return out

    for who, rows in complete_by_reviewer.items():
        summary["per_candidate"][who] = aggregate(rows)

    # 合并双评审：按 review_id 取平均分；严重错误/可理解/可靠 取「任一人判是即算是」的保守口径 + 严格口径
    merged: dict[str, dict[str, float]] = {}
    by_rid: dict[str, list[dict[str, str]]] = {}
    for who, rows in complete_by_reviewer.items():
        for r in rows:
            by_rid.setdefault(r["review_id"], []).append(r)
    for rid, items in by_rid.items():
        cand = blind[rid]["candidate_id"]
        scores = [float(r["评分0-4"]) for r in items]
        merged.setdefault(cand, {})
        merged[cand][rid] = statistics.fmean(scores)

    def _by_sample(rid_meta: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, list[dict[str, str]]]]:
        by_sample: dict[str, dict[str, list[dict[str, str]]]] = {}
        for rid, items in rid_meta.items():
            info = blind[rid]
            by_sample.setdefault(info["sample_id"], {})[info["candidate_id"]] = items
        return by_sample

    def paired_score(rid_scores: dict[str, float],
                     rid_meta: dict[str, list[dict[str, str]]]) -> dict[str, Any] | None:
        """S2 − A2 的平均评分差（每样本先对双评审取均值）。"""
        deltas = []
        for sid, d in _by_sample(rid_meta).items():
            if "A2" not in d or "S2" not in d:
                continue
            s2 = statistics.fmean([rid_scores[r["review_id"]] for r in d["S2"]])
            a2 = statistics.fmean([rid_scores[r["review_id"]] for r in d["A2"]])
            deltas.append(s2 - a2)
        if not deltas:
            return None
        rng = random.Random(SEED_BLIND)
        boots = sorted(statistics.fmean([deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))])
                       for _ in range(1000))
        return {"n_pairs": len(deltas), "delta": statistics.fmean(deltas),
                "ci95": [boots[24], boots[974]],
                "improved": sum(1 for d in deltas if d > 0),
                "worsened": sum(1 for d in deltas if d < 0)}

    def paired_flag(rid_meta: dict[str, list[dict[str, str]]], col: str) -> dict[str, Any] | None:
        """某一列从「A2 未达标记」→「S2 达标」记为 gained；反向记为 lost。

        双评审口径：任一人判「是」即认为该候选在这一列达标（保守，避免漏掉风险）。
        """
        gained = lost = 0
        n = 0
        pos_a2 = pos_s2 = 0
        for sid, d in _by_sample(rid_meta).items():
            if "A2" not in d or "S2" not in d:
                continue
            n += 1
            a2 = any(r[col] == "是" for r in d["A2"])
            s2 = any(r[col] == "是" for r in d["S2"])
            pos_a2 += int(a2)
            pos_s2 += int(s2)
            if s2 and not a2:
                gained += 1
            if a2 and not s2:
                lost += 1
        if not n:
            return None
        return {"n_pairs": n, "rate_a2": pos_a2 / n, "rate_s2": pos_s2 / n,
                "gained": gained, "lost": lost}

    if by_rid:
        summary["paired_vs_a2"]["score"] = paired_score(
            {rid: v for cand in merged.values() for rid, v in cand.items()}, by_rid)
        summary["paired_vs_a2"]["severe_fact_error"] = paired_flag(by_rid, "严重语义错误")
        summary["paired_vs_a2"]["reliable_communication"] = paired_flag(by_rid, "可支持可靠沟通")
        summary["paired_vs_a2"]["understandable"] = paired_flag(by_rid, "可理解部分信息")

    # 一致性（双评审）
    if len(complete_by_reviewer) == 2:
        a = {r["review_id"]: r for r in complete_by_reviewer["A"]}
        b = {r["review_id"]: r for r in complete_by_reviewer["B"]}
        common = sorted(set(a) & set(b))
        if common:
            score_pairs = [(a[k]["评分0-4"], b[k]["评分0-4"]) for k in common]
            severe_pairs = [(a[k]["严重语义错误"], b[k]["严重语义错误"]) for k in common]
            reliable_pairs = [(a[k]["可支持可靠沟通"], b[k]["可支持可靠沟通"]) for k in common]
            within1 = sum(1 for x, y in score_pairs if abs(float(x) - float(y)) <= 1) / len(common)
            summary["agreement"] = {
                "n_common": len(common),
                "score_exact": sum(1 for x, y in score_pairs if x == y) / len(common),
                "score_within_1": within1,
                "score_kappa_quadratic": _kappa(score_pairs, weights="quadratic"),
                "severe_agreement": sum(1 for x, y in severe_pairs if x == y) / len(common),
                "severe_kappa": _kappa(severe_pairs),
                "reliable_agreement": sum(1 for x, y in reliable_pairs if x == y) / len(common),
                "reliable_kappa": _kappa(reliable_pairs),
                "arbitration_rows": [k for k in common
                                     if abs(float(a[k]["评分0-4"]) - float(b[k]["评分0-4"])) >= 2
                                     or a[k]["严重语义错误"] != b[k]["严重语义错误"]],
            }
            summary["agreement"]["arbitration_share"] = len(summary["agreement"]["arbitration_rows"]) / len(common)

    summary["blinded_map_used"] = "blind_map_EVAL_ONLY.csv (offline join; no GT text read)"
    out_path = HERE / "round3_stats.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "report": str(out_path)}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round-3 review package (rubric v2, two reviewers)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    stats_parser = sub.add_parser("stats")
    stats_parser.add_argument("--reviewer-a", type=Path, required=True)
    stats_parser.add_argument("--reviewer-b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        return build_package()
    if args.command == "check":
        return check_package()
    return stats(args.reviewer_a, args.reviewer_b)


if __name__ == "__main__":
    raise SystemExit(main())