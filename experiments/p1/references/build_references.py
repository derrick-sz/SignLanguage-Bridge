"""Build the evaluation-only GT reference table.

The output lives under ``experiments/p1/references/`` which is *evaluation
only*: inference workers, fusion controllers and LLM prompts must never read
this directory.  Two reference calibers are preserved explicitly because the
project already hit a ~2.8 BLEU trap from mixing them (see
docs/agents/model-evaluation-pitfalls.md):

* ``ref_sentence_upstream``: the paper-caliber reference file
  ``test_tmp_refs.txt`` (half-width punctuation, characters separated by spaces).
* ``gt_sentence_official``: the official annotation text (full-width punctuation,
  no separators).  Auditing only; do not compare across calibers.

The reference gloss comes from the official annotation and is the only label
used for corpus WER.
"""

from __future__ import annotations

import argparse

from experiments.p1 import common
from experiments.p1.datasets.build_manifest import parse_upstream_refs


def build() -> dict[str, object]:
    rows = common.read_csv_rows(common.COMBINED_CSV)
    upstream = parse_upstream_refs(common.UPSTREAM_REFS)
    if len(rows) != 1086:
        raise SystemExit(f"expected 1086 rows, found {len(rows)}")

    out: list[dict[str, str]] = []
    missing_upstream: list[str] = []
    for row in rows:
        sample = row["sample"].strip()
        ref_upstream = upstream.get(sample)
        if ref_upstream is None:
            missing_upstream.append(sample)
        out.append(
            {
                "sample_id": sample,
                "signer_id": row["sample"].split("_")[1],
                "ref_sentence_upstream": ref_upstream or "",
                "gt_sentence_official": row.get("gt_sentence", ""),
                "gt_gloss": row.get("gt_gloss", ""),
            }
        )
    if missing_upstream:
        raise SystemExit(f"{len(missing_upstream)} samples missing upstream reference")

    common.write_csv_rows(
        common.REFERENCES / "gt_1086.csv",
        ["sample_id", "signer_id", "ref_sentence_upstream", "gt_sentence_official", "gt_gloss"],
        out,
    )

    # The pilot diagnostic stratification is GT-derived -> evaluation-only copy.
    audit_source = common.PLAN_DIR / "data" / "pilot_selection_audit_EVAL_ONLY.csv"
    audit_rows = common.read_csv_rows(audit_source)
    common.write_csv_rows(
        common.REFERENCES / "pilot_50_diagnostic_EVAL_ONLY.csv",
        list(audit_rows[0].keys()),
        audit_rows,
    )

    (common.REFERENCES / "README.md").write_text(
        "# EVALUATION ONLY — do not feed to inference / fusion / LLM\n\n"
        "| 文件 | 用途 | 读取方 |\n|---|---|---|\n"
        "| `gt_1086.csv` | 1086 条 test 参考句（上游口径 + 官方标注口径）与参考 Gloss | 离线 EvaluationWorker |\n"
        "| `pilot_50_diagnostic_EVAL_ONLY.csv` | Pilot-50 的 GT 派生分层（cohort class、WER、字符相似度） | 仅人工分析/报告，禁止进入推理载荷 |\n\n"
        "规则：推理 Worker、Jev/LLM、融合控制器都不得读取本目录；一旦 GT 出现在\n"
        "预测/融合载荷中，`experiments/p1/gt_guard.py` 会抛 `GroundTruthLeakError`。\n\n"
        "两个参考口径（全角/半角标点）不能混用：同一份预测在两套参考上的 BLEU 可差约 2.8。\n",
        encoding="utf-8",
    )

    meta = {
        "created_at": common.utc_now(),
        "created_by": "experiments/p1/references/build_references.py",
        "eval_only": True,
        "sources": {
            "combined_csv_sha256": common.sha256_file(common.COMBINED_CSV),
            "upstream_refs_sha256": common.sha256_file(common.UPSTREAM_REFS),
            "pilot_audit_sha256": common.sha256_file(audit_source),
        },
        "counts": {
            "gt_1086": len(out),
            "unique_samples": len({row["sample_id"] for row in out}),
            "signers": len({row["signer_id"] for row in out}),
            "upstream_refs_total": len(upstream),
            "pilot_diagnostic_rows": len(audit_rows),
        },
        "reference_calibers": {
            "bleu_canonical": "sacreBLEU 13a on character-separated text, reference = ref_sentence_upstream",
            "chrf_canonical": "sacreBLEU chrF on raw strings with ref_sentence_upstream as stored",
        },
    }
    common.write_json(common.REFERENCES / "source_meta.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    meta = build()
    print(common.json.dumps(meta["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
