"""Offline EvaluationWorker for P-1 (GT join happens only here).

Usage examples::

    python -m experiments.p1.eval.evaluate \
        --predictions <run>/model_outputs.jsonl \
        --references experiments/p1/references/gt_1086.csv \
        --task CSLT --output <run>/metrics.json --run-dir <run>

The evaluator is the only component allowed to read ``references/``.  Its
per-sample output (``eval_records.jsonl``) contains the ground truth and is kept
next to the metrics, not next to the prediction files.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from experiments.p1 import common
from experiments.p1.eval import metrics as metric_lib


def load_references(path: Path) -> dict[str, dict[str, str]]:
    rows = common.read_csv_rows(path)
    return {row["sample_id"].strip(): row for row in rows}


def prediction_sample(record: dict[str, Any]) -> str | None:
    for key in ("sample_id", "sample", "name", "id"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return None


def prediction_success(record: dict[str, Any]) -> bool:
    status = str(record.get("status", "ok")).lower()
    if status in {"ok", "final", "success", "pass", "partial"}:
        return True
    if record.get("error"):
        return False
    return bool(record.get("prediction") or record.get("text") or record.get("gloss_sequence"))


def extract_text(record: dict[str, Any], task: str) -> str | None:
    if task == "CSLT":
        for key in ("prediction", "text", "hypothesis"):
            value = record.get(key)
            if isinstance(value, str):
                return value
        return None
    gloss = record.get("gloss_sequence")
    if isinstance(gloss, list):
        return " ".join(str(token) for token in gloss)
    for key in ("gloss", "prediction", "text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _bleu_statistic(pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    return metric_lib.corpus_metrics_cslt([(item["reference"], item["hypothesis"]) for item in pairs])["bleu4_13a"]


def _chrf_statistic(pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    return metric_lib.corpus_metrics_cslt([(item["reference"], item["hypothesis"]) for item in pairs])["chrf"]


def _edit_statistic(pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    return metric_lib.corpus_metrics_cslt([(item["reference"], item["hypothesis"]) for item in pairs])[
        "mean_char_edit_similarity"
    ]


def _wer_statistic(pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    return metric_lib.corpus_wer([(item["reference_gloss"], item["hypothesis"]) for item in pairs])["corpus_wer"]


def evaluate(
    predictions_path: Path,
    references_path: Path,
    *,
    task: str,
    bootstrap_iterations: int = 0,
    bootstrap_metrics: tuple[str, ...] = ("bleu4", "mean_char_edit_similarity"),
    count_failures_as_empty: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    references = load_references(references_path)
    records = common.read_jsonl(predictions_path)

    pairs: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing_reference: list[str] = []
    seen: set[str] = set()

    for record in records:
        sample = prediction_sample(record)
        if not sample:
            failures.append({"sample_id": None, "reason": "missing sample id", "raw": record})
            continue
        if sample in seen:
            continue
        seen.add(sample)
        reference_row = references.get(sample)
        if reference_row is None:
            missing_reference.append(sample)
            continue
        ok = prediction_success(record)
        hypothesis = extract_text(record, task = task)
        if ok and hypothesis is None:
            ok = False
        if not ok:
            if count_failures_as_empty:
                hypothesis = ""
            else:
                failures.append(
                    {
                        "sample_id": sample,
                        "status": record.get("status"),
                        "error": str(record.get("error"))[:500],
                    }
                )
                continue
        reference_text = (
            reference_row["ref_sentence_upstream"]
            if task == "CSLT"
            else reference_row["gt_gloss"]
        )
        if task == "CSLT" and not reference_text:
            reference_text = reference_row["gt_sentence_official"]
        pairs.append(
            {
                "sample_id": sample,
                "signer_id": reference_row["signer_id"],
                "reference": reference_text,
                "reference_gloss": reference_row["gt_gloss"],
                "hypothesis": hypothesis or "",
                "reference_official": reference_row["gt_sentence_official"],
            }
        )
        if task == "CSLT":
            reference_norm = metric_lib.normalize_text(reference_text)
            hypothesis_norm = metric_lib.normalize_text(hypothesis or "")
            per_sample = {
                "char_edit_similarity": round(
                    metric_lib.edit_similarity(reference_norm, hypothesis_norm), 4
                ),
            }
        else:
            per_sample = metric_lib.wer_breakdown(reference_text, hypothesis or "")
        eval_records.append(
            {
                "sample_id": sample,
                "signer_id": reference_row["signer_id"],
                **per_sample,
                "reference": reference_text,
                "reference_official": reference_row["gt_sentence_official"],
                "hypothesis": hypothesis,
                "status": record.get("status"),
            }
        )

    if missing_reference:
        raise SystemExit(
            f"{len(missing_reference)} prediction samples are not in the reference table, "
            f"e.g. {missing_reference[:5]}"
        )

    payload: dict[str, Any] = {
        "metric_definition_version": metric_lib.METRIC_DEFINITION_VERSION,
        "task": task,
        "created_at": common.utc_now(),
        "predictions_path": str(predictions_path),
        "predictions_sha256": common.sha256_file(predictions_path),
        "references_path": str(references_path),
        "references_sha256": common.sha256_file(references_path),
        "attempted": len(records),
        "unique_samples": len(seen),
        "successful": len(pairs),
        "failed": len(failures),
        "failure_rate": round(len(failures) / max(1, len(pairs) + len(failures)), 4),
        "missing_reference": 0,
        "failures": failures,
        "metric_scope": "successful_subset_all_succeeded_corpus_metrics"
        if not failures
        else "successful_subset_only; failures excluded and reported",
    }
    if task == "CSLT":
        corpus = metric_lib.corpus_metrics_cslt(
            [(item["reference"], item["hypothesis"]) for item in pairs],
            alternate_references=[item["reference_official"] for item in pairs],
        )
    else:
        corpus = metric_lib.corpus_wer([(item["reference"], item["hypothesis"]) for item in pairs])
    payload["metrics"] = corpus

    if bootstrap_iterations and pairs:
        payload["bootstrap"] = {}
        statistic_map = {
            "bleu4": _bleu_statistic,
            "chrf": _chrf_statistic,
            "mean_char_edit_similarity": _edit_statistic,
            "corpus_wer": _wer_statistic,
        }
        for metric_name in bootstrap_metrics:
            statistic = statistic_map.get(metric_name)
            if statistic is None:
                continue
            records_for_pairs = [
                {
                    "sample_id": item["sample_id"],
                    "signer_id": item["signer_id"],
                    "reference": item["reference"],
                    "reference_gloss": item["reference_gloss"],
                    "hypothesis": item["hypothesis"],
                }
                for item in pairs
            ]
            payload["bootstrap"][metric_name] = metric_lib.bootstrap_by_group(
                records_for_pairs,
                statistic,
                iterations=bootstrap_iterations,
            )

    if pairs and task == "CSLT":
        by_signer: dict[str, Any] = {}
        for signer in sorted({item["signer_id"] for item in pairs}):
            subset = [item for item in pairs if item["signer_id"] == signer]
            by_signer[signer] = {
                "n": len(subset),
                "bleu4_13a": metric_lib.corpus_metrics_cslt(
                    [(item["reference"], item["hypothesis"]) for item in subset]
                )["bleu4_13a"],
            }
        payload["per_signer"] = by_signer
    return payload, eval_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=common.REFERENCES / "gt_1086.csv")
    parser.add_argument("--task", choices=("CSLT", "CSLR"), default="CSLT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap iterations (0 = off)")
    parser.add_argument(
        "--bootstrap-metrics",
        default="bleu4,mean_char_edit_similarity",
        help="comma separated subset of bleu4,chrf,mean_char_edit_similarity,corpus_wer",
    )
    parser.add_argument(
        "--count-failures-as-empty",
        action="store_true",
        help="count failed samples as empty hypotheses (pure deletion for WER)",
    )
    args = parser.parse_args(argv)

    payload, eval_records = evaluate(
        args.predictions,
        args.references,
        task=args.task,
        bootstrap_iterations=args.bootstrap,
        bootstrap_metrics=tuple(item.strip() for item in args.bootstrap_metrics.split(",") if item.strip()),
        count_failures_as_empty=args.count_failures_as_empty,
    )
    common.write_json(args.output, payload)
    if args.run_dir is not None:
        # Evaluation records intentionally contain GT; they live with the evaluator.
        common.write_jsonl(Path(args.run_dir) / "eval_records.jsonl", eval_records, guard=False)
    summary_keys = ("attempted", "successful", "failed", "failure_rate", "metric_scope")
    print(common.json.dumps({key: payload[key] for key in summary_keys}, ensure_ascii=False))
    print(common.json.dumps(payload["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
