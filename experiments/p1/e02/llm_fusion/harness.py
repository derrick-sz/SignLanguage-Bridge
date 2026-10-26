"""G2 phase-2A: LLM fusion ablation harness (A/C/D/E groups).

Pipeline (CPU only, GT-free):

    candidate_outputs.jsonl (A/A2/B0, model outputs)
        + experiments/p1/e03/per_sample_scores.csv (whitelisted raw SLRT score columns only)
        -> prompt construction (prompts/v1_zh.json, versioned + hashed)
        -> backend.generate()            [stub now; codex_cli / openai_compat gated]
        -> runs/<run_id>/fusion_outputs.jsonl   (per-row gt_guard.assert_gt_clean)
        -> runs/<run_id>/manifest.json          (hashes, params, sample count, status)

GT join + canonical metrics live in `evaluate.py`, never here.

Usage:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -m experiments.p1.e02.llm_fusion.harness run \
        --backend stub --limit 15 --run-id g2a-stub-selftest-demo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.p1 import common
from experiments.p1.gt_guard import assert_gt_clean, find_gt_keys

from experiments.p1.e02.llm_fusion.backends import (
    BackendDisabled,
    BackendNotImplemented,
    get_backend,
    list_backends,
)

ROOT = common.ROOT
FUSION_DIR = common.P1 / "e02" / "llm_fusion"
PROMPTS_DIR = FUSION_DIR / "prompts"
RUNS_DIR = FUSION_DIR / "runs"
E02_CANDIDATES = common.P1 / "e02" / "candidate_outputs.jsonl"
E03_SCORES = common.P1 / "e03" / "per_sample_scores.csv"
MANIFEST_1086 = common.P1 / "datasets" / "manifest_1086.csv"

DEFAULT_PROMPT_VERSION = "v1_zh"
DEFAULT_GROUPS = ("A", "C", "D", "E")
SAMPLING_SEED = 20261026
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20261026

# Only raw (uncalibrated) SLRT confidence columns may be read from the E03 CSV.
# The same CSV also contains GT-label columns (wer/bad/error_any/...) and
# GT-trained OOF probabilities; those are explicitly excluded by this whitelist.
SCORE_FEATURE_WHITELIST = (
    "token_score_mean",
    "token_score_min",
    "token_score_median",
    "token_score_p10",
    "token_score_max",
    "window_top_prob_mean",
    "window_top_prob_min",
    "window_top_prob_p10",
    "window_top_prob_max",
    "low_prob_window_frac_05",
    "low_prob_window_frac_03",
    "blank_window_frac",
    "n_tokens",
    "n_windows",
)
# Documented for tests/reviewers; these must never be loaded into prompts.
FORBIDDEN_SCORE_FEATURES = {
    "wer", "wer_s", "wer_d", "wer_i", "error_any", "bad", "bad_03",
    "uni_video_edit_sim_EVAL_ONLY", "should_upgrade_proxy_EVAL_ONLY",
    "oof_p_bad_logistic_token", "oof_p_bad_logistic_window", "oof_p_bad_logistic_combo",
}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_prompt_spec(version: str = DEFAULT_PROMPT_VERSION) -> dict[str, Any]:
    json_path = PROMPTS_DIR / f"{version}.json"
    md_path = PROMPTS_DIR / f"{version}.md"
    if not json_path.exists():
        raise SystemExit(f"prompt spec not found: {json_path}")
    spec = json.loads(json_path.read_text(encoding="utf-8"))
    spec["_files"] = {
        "json": str(json_path.relative_to(ROOT)),
        "md": str(md_path.relative_to(ROOT)) if md_path.exists() else None,
        "json_sha256": common.sha256_file(json_path),
        "md_sha256": common.sha256_file(md_path) if md_path.exists() else None,
    }
    return spec


def load_candidates(path: Path = E02_CANDIDATES) -> dict[str, dict[str, str]]:
    """Load A/A2/B0 texts per sample from the GT-free E02 candidate file."""

    rows = common.read_jsonl(path)
    by_sample: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        text = (row.get("output") or {}).get("text")
        if text is None:
            text = ""
        by_sample[row["sample_id"]][row["candidate_id"]] = str(text)
    candidates: dict[str, dict[str, str]] = {}
    for sample, mapping in sorted(by_sample.items()):
        if not {"A", "B0"} <= set(mapping):
            continue
        candidates[sample] = {
            "A": mapping.get("A", ""),
            "A2": mapping.get("A2", ""),
            "B0": mapping.get("B0", ""),
        }
    return candidates


def load_score_features(path: Path = E03_SCORES) -> dict[str, dict[str, float | None]]:
    """Read ONLY whitelisted raw SLRT score columns; GT-label columns are dropped."""

    rows = common.read_csv_rows(path)
    features: dict[str, dict[str, float | None]] = {}
    for row in rows:
        sample = row.get("sample_id")
        if not sample:
            continue
        record: dict[str, float | None] = {}
        for column in SCORE_FEATURE_WHITELIST:
            raw = row.get(column)
            if raw is None or raw == "":
                record[column] = None
                continue
            try:
                record[column] = float(raw)
            except ValueError:
                record[column] = None
        features[sample] = record
    return features


def load_signer_map(path: Path = MANIFEST_1086) -> dict[str, str]:
    """Signer ids from the GT-free dataset manifest (used for stratified sampling only)."""

    return {row["sample_id"]: row["signer_id"] for row in common.read_csv_rows(path)}


def select_samples(
    candidates: dict[str, dict[str, str]],
    limit: int,
    *,
    seed: int = SAMPLING_SEED,
    signer_map: dict[str, str] | None = None,
) -> list[str]:
    """Deterministic signer-stratified selection (documented, no randomness needed for audit).

    Samples are grouped by signer; within each signer the order is sorted sample_id.
    Selection is round-robin across sorted signers until ``limit`` is reached.
    ``seed`` is recorded in the manifest for provenance even though selection is
    deterministic (no shuffle), which keeps the self-test reproducible.
    """

    signer_map = signer_map or load_signer_map()
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample in sorted(candidates):
        grouped[signer_map.get(sample, "UNKNOWN")].append(sample)
    signers = sorted(grouped)
    selected: list[str] = []
    round_index = 0
    while len(selected) < limit:
        advanced = False
        for signer in signers:
            pool = grouped[signer]
            if round_index < len(pool):
                selected.append(pool[round_index])
                advanced = True
                if len(selected) >= limit:
                    break
        if not advanced:
            break
        round_index += 1
    _ = random.Random(seed)  # recorded provenance only; selection is deterministic
    return selected


# --------------------------------------------------------------------------
# prompt construction (GT-free)
# --------------------------------------------------------------------------
def slot_evidence_text(unisign_text: str, gloss_text: str) -> str:
    """Deterministic slot hints for group E, reusing the E06 RuleJudge slot extraction."""

    from experiments.p1.e06.build_e06 import extract_slots

    a = extract_slots(unisign_text)
    g = extract_slots(gloss_text)

    def fmt(values) -> str:
        if not values:
            return "无"
        return "/".join(str(value) for value in sorted(values, key=lambda item: (str(type(item)), item)))

    numeric_status = "一致" if a["number_values"] == g["number_values"] else "冲突"
    negation_status = "一致" if a["negation"] == g["negation"] else "冲突"
    person_status = "一致" if a["persons"] == g["persons"] else "冲突"
    question_status = "一致" if a["question"] == g["question"] else "冲突"
    return "\n".join(
        [
            f"- 数字/数量：Uni-Sign={fmt(a['number_values'])}；Gloss={fmt(g['number_values'])}；{numeric_status}",
            f"- 否定极性：Uni-Sign={'有' if a['negation'] else '无'}；Gloss={'有' if g['negation'] else '无'}；{negation_status}",
            f"- 人物指代：Uni-Sign={fmt(a['persons'])}；Gloss={fmt(g['persons'])}；{person_status}",
            f"- 疑问语气：Uni-Sign={'有' if a['question'] else '无'}；Gloss={'有' if g['question'] else '无'}；{question_status}",
        ]
    )


def score_note(features: dict[str, float | None] | None) -> str:
    if not features:
        return "（无可用分数特征）"
    parts = []
    for key in ("token_score_mean", "window_top_prob_mean", "low_prob_window_frac_05"):
        value = features.get(key)
        if value is not None:
            parts.append(f"{key}={value:.4f}")
    return "；".join(parts) if parts else "（无可用分数特征）"


def build_messages(
    group: str,
    *,
    unisign_text: str,
    gloss_text: str,
    features: dict[str, float | None] | None,
    spec: dict[str, Any],
) -> list[dict[str, str]]:
    """Build chat messages for one group; group A returns no messages (no LLM)."""

    if group == "A":
        return []
    templates = spec["group_prompts"]
    if group not in templates or templates[group].get("template") is None:
        raise ValueError(f"no prompt template for group {group!r}")
    user_prompt = templates[group]["template"].format(
        unisign_text=unisign_text,
        gloss_text=gloss_text,
        slot_evidence=slot_evidence_text(unisign_text, gloss_text) if group == "E" else "",
        score_note=score_note(features),
    )
    return [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------------
# output parsing (version-specific output formats, e.g. v2 group E reason line)
# --------------------------------------------------------------------------
def parse_group_output(group: str, text: str, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (text_for_evaluation, extra_fields)."""

    raw = str(text or "")
    formats = spec.get("output_format") or {}
    fmt = formats.get(group)
    if not fmt:
        return raw.strip(), {"output_text_parse": "identity"}
    if fmt.get("mode") == "sentence_plus_reason":
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return "", {"output_text_parse": "empty", "output_raw_text": raw}
        translation = lines[0]
        reason = None
        prefix = fmt.get("reason_prefix", "理由：")
        for line in lines[1:]:
            if line.startswith(prefix):
                reason = line[len(prefix):].strip()
                break
        if reason is None and len(lines) > 1:
            reason = lines[1]
        return translation, {
            "output_text_parse": "sentence_plus_reason",
            "output_raw_text": raw,
            "output_reason": reason,
        }
    return raw.strip(), {"output_text_parse": "identity"}


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def _now() -> str:
    return common.utc_now()


def _make_run_id(backend_name: str, limit: int | None) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    scale = f"n{limit}" if limit is not None else "nall"
    return f"g2a-{backend_name}-{scale}-{stamp}"


def run(
    *,
    backend_name: str = "stub",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    limit: int | None = 15,
    run_id: str | None = None,
    sample_ids: list[str] | None = None,
    allow_real: bool = False,
    overwrite: bool = False,
    data_status: str = "STUB_ONLY_NOT_LLM_RESULTS",
    backend_kwargs: dict[str, Any] | None = None,
    concurrency: int = 1,
    max_tokens_override: int | None = None,
) -> dict[str, Any]:
    if not set(groups) <= set(DEFAULT_GROUPS):
        raise SystemExit(f"unknown groups: {groups}; allowed {DEFAULT_GROUPS}")
    if concurrency != 1:
        raise SystemExit("this harness runs strictly serial (concurrency=1) to protect the shared proxy")
    spec = load_prompt_spec(prompt_version)
    candidates = load_candidates()
    features = load_score_features()
    signers = load_signer_map()
    backend = get_backend(backend_name, **(backend_kwargs or {}))
    backend_description = backend.describe()
    if backend.is_real and not allow_real:
        raise SystemExit(
            f"backend {backend_name!r} is a real LLM backend; pass --allow-real only after the "
            "Lead/user approves the endpoint and model id. This round must stay on stub."
        )
    if backend.is_real and not backend_description.get("available"):
        raise SystemExit(f"backend {backend_name!r} is not available/enabled: {backend_description}")

    if sample_ids is None:
        sample_ids = select_samples(candidates, limit if limit is not None else len(candidates),
                                    signer_map=signers)
    sample_ids = sorted(set(sample_ids))
    run_id = run_id or _make_run_id(backend_name, limit)
    run_dir = RUNS_DIR / run_id
    if run_dir.exists() and not overwrite:
        raise SystemExit(f"run dir already exists: {run_dir} (pass --overwrite to regenerate)")
    run_dir.mkdir(parents=True, exist_ok=True)

    decoding = dict(spec["decoding"])
    if max_tokens_override is not None:
        decoding["max_tokens"] = int(max_tokens_override)
        decoding["override"] = {
            "max_tokens": int(max_tokens_override),
            "reason": "run-level override; prompt text (v1_zh) unchanged",
        }
    rows: list[dict[str, Any]] = []
    leak_errors: list[str] = []
    error_count = 0
    started = time.perf_counter()
    for sample in sample_ids:
        unisign_text = candidates[sample]["A"]
        gloss_text = candidates[sample]["B0"]
        sample_features = features.get(sample, {})
        for group in groups:
            prompt_messages = build_messages(
                group,
                unisign_text=unisign_text,
                gloss_text=gloss_text,
                features=sample_features,
                spec=spec,
            )
            base_record: dict[str, Any] = {
                "run_id": run_id,
                "sample_id": sample,
                "group": group,
                "prompt_version": prompt_version,
                "prompt_spec_sha256": spec["_files"]["json_sha256"],
                "backend": backend.name,
                "backend_version": backend.version,
                "decoding": {
                    "temperature": decoding["temperature"],
                    "max_tokens": decoding["max_tokens"],
                    "top_p": decoding["top_p"],
                },
                "evidence": {
                    "unisign_text": unisign_text,
                    "slrt_gloss_text": gloss_text,
                    "slot_evidence": slot_evidence_text(unisign_text, gloss_text) if group == "E" else None,
                    "score_note": score_note(sample_features) if group in ("D", "E") else None,
                },
                "score_features": sample_features,
                "system_prompt_sha256": sha256_text(spec["system_prompt"]) if prompt_messages else None,
                "user_prompt_sha256": sha256_text(prompt_messages[1]["content"]) if prompt_messages else None,
                "prompt_sha256": sha256_json(prompt_messages) if prompt_messages else None,
                "created_at": _now(),
            }
            if group == "A":
                base_record.update(
                    {
                        "model_version": "none (direct Uni-Sign baseline)",
                        "output_text": unisign_text,
                        "status": "ok" if unisign_text.strip() else "error",
                        "error": None if unisign_text.strip() else "empty baseline text",
                        "latency_ms": 0.0,
                        "tokens": {"in": None, "out": None, "total": None},
                        "provider": None,
                        "http_status": None,
                        "retries": None,
                    }
                )
            else:
                try:
                    result = backend.generate(
                        prompt_messages,
                        group=group,
                        evidence={
                            "unisign_text": unisign_text,
                            "gloss_text": gloss_text,
                            "score_features": sample_features,
                        },
                        temperature=decoding["temperature"],
                        max_tokens=decoding["max_tokens"],
                        top_p=decoding["top_p"],
                    )
                    result_extra = dict(result.extra or {})
                    evaluation_text, parse_extra = parse_group_output(group, result.text, spec)
                    base_record.update(
                        {
                            "model_version": result.model_version,
                            "output_text": evaluation_text,
                            "output_raw_text": result.text,
                            "status": result.status,
                            "error": result.error,
                            "latency_ms": round(result.latency_ms, 3),
                            "latency_s": round(result.latency_ms / 1000.0, 4),
                            "tokens": {
                                "in": result.tokens_in,
                                "out": result.tokens_out,
                                "total": result_extra.get("tokens_total"),
                            },
                            "provider": result_extra.get("provider"),
                            "http_status": result_extra.get("http_status"),
                            "retries": result_extra.get("retries"),
                            "attempts": result_extra.get("attempts"),
                            "backend_extra": result_extra,
                            **parse_extra,
                        }
                    )
                except (BackendDisabled, BackendNotImplemented) as exc:
                    base_record.update(
                        {
                            "model_version": None,
                            "output_text": "",
                            "status": "blocked",
                            "error": f"{type(exc).__name__}: {exc}",
                            "latency_ms": None,
                            "latency_s": None,
                            "tokens": {"in": None, "out": None, "total": None},
                            "provider": None,
                            "http_status": None,
                            "retries": None,
                        }
                    )
                    error_count += 1
                except Exception as exc:  # noqa: BLE001 - one failed call must not abort the batch
                    base_record.update(
                        {
                            "model_version": None,
                            "output_text": "",
                            "status": "error",
                            "error": f"{type(exc).__name__}"[:160],
                            "latency_ms": None,
                            "latency_s": None,
                            "tokens": {"in": None, "out": None, "total": None},
                            "provider": None,
                            "http_status": None,
                            "retries": None,
                        }
                    )
                    error_count += 1
            leaks = find_gt_keys(base_record)
            if leaks:
                leak_errors.append(f"{sample}/{group}: {leaks}")
                continue
            try:
                assert_gt_clean(base_record, context=f"{sample}/{group}")
            except Exception as exc:  # noqa: BLE001 - defensive: never write a leaky row
                leak_errors.append(f"{sample}/{group}: {exc}")
                continue
            if base_record.get("status") != "ok" and group != "A":
                error_count += 1
            rows.append(base_record)

    outputs_path = run_dir / "fusion_outputs.jsonl"
    common.write_jsonl(outputs_path, rows, guard=True)
    elapsed = time.perf_counter() - started

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment": "G2-2A-LLM-FUSION",
        "status": "REAL_RUN" if (backend.is_real and backend_description.get("available")) else "IMPLEMENTED_UNTESTED",
        "data_status": data_status,
        "created_at": _now(),
        "git": common.git_state(),
        "backend": backend_description,
        "backend_config_sha256": sha256_json(backend_description),
        "params_sha256": sha256_json(decoding),
        "llm_endpoint": {
            "provider": backend_description.get("provider"),
            "base_url": backend_description.get("base_url"),
            "endpoint": backend_description.get("endpoint"),
            "model": backend_description.get("model"),
            "credentials_ref": backend_description.get("credentials_ref"),
            "credentials_present": backend_description.get("credentials_present"),
            "credentials_source": backend_description.get("credentials_source"),
            "timeout_seconds": backend_description.get("timeout_seconds"),
            "max_retries": backend_description.get("max_retries"),
            "enable_env": backend_description.get("enable_env"),
        },
        "security": {
            "api_key_persisted": False,
            "api_key_in_manifest": False,
            "api_key_in_outputs": False,
            "credentials_reference_only": True,
            "note": "key is resolved at runtime in memory only; this manifest stores a reference string, not the value",
        },
        "model": {
            "model_version": backend.version if backend.is_stub else (backend_description.get("model") or None),
            "decoding": decoding,
        },
        "prompt": {
            "version": prompt_version,
            "files": spec["_files"],
            "system_prompt_sha256": sha256_text(spec["system_prompt"]),
        },
        "groups": list(groups),
        "sampling": {
            "strategy": "signer-stratified round-robin over sorted sample_id (deterministic)",
            "seed_recorded": SAMPLING_SEED,
            "limit": limit,
            "n_samples": len(sample_ids),
            "signer_counts": _count_signers(sample_ids, signers),
            "sample_ids_sha256": sha256_json(sample_ids),
        },
        "inputs": {
            "candidate_outputs": str(E02_CANDIDATES.relative_to(ROOT)),
            "candidate_outputs_sha256": common.sha256_file(E02_CANDIDATES),
            "e03_scores": str(E03_SCORES.relative_to(ROOT)),
            "e03_scores_sha256": common.sha256_file(E03_SCORES),
            "manifest_1086_sha256": common.sha256_file(MANIFEST_1086),
            "score_feature_whitelist": list(SCORE_FEATURE_WHITELIST),
            "forbidden_score_features_excluded": sorted(FORBIDDEN_SCORE_FEATURES),
        },
        "outputs": {
            "fusion_outputs": str((run_dir / "fusion_outputs.jsonl").relative_to(ROOT)),
            "fusion_outputs_sha256": common.sha256_file(outputs_path),
            "n_rows": len(rows),
            "n_expected_rows": len(sample_ids) * len(groups),
            "n_errors": error_count,
        },
        "gt_isolation": {
            "harness_reads_gt": False,
            "gt_reading_module": "llm_fusion/evaluate.py only (harness/backends/prompts never open a GT file)",
            "gt_reference_opened_here": False,
            "gt_guard_rows_checked": len(rows) + len(leak_errors),
            "gt_guard_leaks": len(leak_errors),
            "leak_details": leak_errors,
            "label_columns_dropped_from_e03_csv": sorted(FORBIDDEN_SCORE_FEATURES),
        },
        "cost": {
            "latency_ms_total": round(elapsed * 1000.0, 1),
            "concurrency": concurrency,
            "n_llm_calls": sum(1 for row in rows if row["group"] != "A"),
            "note": "stub latency is rule-execution time, not LLM latency"
            if backend.is_stub else "LLM latency recorded per call (latency_s); total is batch wall time",
        },
        "notes": [
            "real LLM backends are gated by P1_LLM_ENABLE_REAL=1 + --allow-real"
            if backend.is_stub else "REAL LLM run: provider/model/params recorded; stub numbers are separate",
            "stub outputs are deterministic rules, not model results; metrics are pipeline validation only"
            if backend.is_stub else "outputs are real model generations from the declared endpoint",
            "GT join and canonical metrics are done by llm_fusion/evaluate.py (EVAL ONLY)",
        ],
    }
    common.write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({
        "run_id": run_id,
        "n_samples": len(sample_ids),
        "n_rows": len(rows),
        "n_errors": error_count,
        "leaks": len(leak_errors),
        "backend": backend.name,
        "outputs": str(outputs_path),
        "manifest": str(run_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2))
    return {"run_id": run_id, "run_dir": run_dir, "manifest": manifest, "rows": rows}


def _count_signers(sample_ids: list[str], signers: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sample in sample_ids:
        counts[signers.get(sample, "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G2 phase-2A LLM fusion ablation harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="build prompts, call backend, write GT-free outputs")
    run_parser.add_argument("--backend", default="stub", choices=("stub", "codex_cli", "openai_compat"))
    run_parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    run_parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    run_parser.add_argument("--limit", type=int, default=15, help="number of samples; 0 = all")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--sample-ids", type=Path, default=None, help="optional file with one sample_id per line")
    run_parser.add_argument("--allow-real", action="store_true", help="allow a real LLM backend (requires endpoint approval)")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--data-status", default="STUB_ONLY_NOT_LLM_RESULTS")
    run_parser.add_argument("--base-url", default=None, help="OpenAI-compatible base url (real backend)")
    run_parser.add_argument("--model", default=None, help="fixed model id for a real backend")
    run_parser.add_argument("--credentials-file", default=None, help="yaml file with refs.<credentials-ref>")
    run_parser.add_argument("--credentials-ref", default="WB_PROXY_API_KEY")
    run_parser.add_argument("--timeout-seconds", type=int, default=90)
    run_parser.add_argument("--max-concurrency", type=int, default=1, help="must stay 1 (serial)")
    run_parser.add_argument("--max-tokens", type=int, default=None,
                            help="run-level max_tokens override (prompt text unchanged; recorded in manifest)")
    args = parser.parse_args(argv)

    if args.command == "run":
        sample_ids = None
        if args.sample_ids is not None:
            sample_ids = [line.strip() for line in args.sample_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        limit = None if args.limit == 0 else args.limit
        groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
        backend_kwargs: dict[str, Any] = {}
        if args.backend == "openai_compat":
            backend_kwargs = {
                "base_url": args.base_url,
                "model": args.model,
                "credentials_file": args.credentials_file,
                "credentials_ref": args.credentials_ref,
                "timeout_seconds": args.timeout_seconds,
                "enabled": bool(args.allow_real),
            }
        run(
            backend_name=args.backend,
            prompt_version=args.prompt_version,
            groups=groups,
            limit=limit,
            run_id=args.run_id,
            sample_ids=sample_ids,
            allow_real=args.allow_real,
            overwrite=args.overwrite,
            data_status=args.data_status,
            backend_kwargs=backend_kwargs,
            concurrency=args.max_concurrency,
            max_tokens_override=args.max_tokens,
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
