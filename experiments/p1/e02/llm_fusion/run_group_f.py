"""Run the REAL 50-sample group-F batch: SLRT gloss -> LLM sentence (v3_f).

This is the "LLM as language organizer" direction from
``docs/experiments/P-1/G2_IMPROVEMENT_ANALYSIS.md`` (direction 1).  It is a NEW
experiment path, not a rerun of the C/D/E fact-arbitration fusion:

    input to the LLM = SLRT gloss word sequence (candidate B0) ONLY
    no Uni-Sign translation, no slot evidence, no score note

Endpoint (same as v1/v2 real runs):
    provider  wb-proxy
    base_url  http://127.0.0.1:8788/v1
    model     deepseek-v4.1-flash
    auth      Bearer key resolved at runtime in memory from
              C:\\Users\\ADMIN\\.dsh\\.credentials.yaml refs.WB_PROXY_API_KEY
              (or the WB_PROXY_API_KEY / P1_OPENAI_API_KEY env var)

Security: the key is never printed and never written to any output file, JSONL,
manifest, log or report. Manifests only store ``credentials_ref``.

Run (CPU except the HTTP calls; strictly serial, 50 calls):
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    $env:P1_LLM_ENABLE_REAL='1'
    .\\.conda-env\\python.exe -m experiments.p1.e02.llm_fusion.run_group_f
"""
from __future__ import annotations

import argparse
import json
import sys

from experiments.p1 import common
from experiments.p1.e02.llm_fusion import evaluate as evaluate_module
from experiments.p1.e02.llm_fusion import harness
from experiments.p1.e02.llm_fusion.backends import get_backend

ROOT = common.ROOT
FUSION_DIR = harness.FUSION_DIR
SPACED_RUN_ID = "g2f-wbproxy-50-v3f-spaced-20261027"
JOINED_RUN_ID = "g2f-wbproxy-50-v3f-20261027"
RUN_ID = SPACED_RUN_ID
PROMPT_VERSION = "v3_f"
DEFAULT_GLOSS_INPUT_MODE = "spaced_tokens"
PILOT_50 = common.P1 / "datasets" / "pilot_50.csv"
BASE_URL = "http://127.0.0.1:8788/v1"
MODEL = "deepseek-v4.1-flash"
DATA_STATUS = "REAL_EXPLORATORY_50_GROUP_F"


def _pilot_sample_ids() -> list[str]:
    rows = common.read_csv_rows(PILOT_50)
    return [row["sample_id"] for row in rows]


def _backend_kwargs() -> dict:
    return {
        "base_url": BASE_URL,
        "model": MODEL,
        "credentials_ref": "WB_PROXY_API_KEY",
        "timeout_seconds": 90,
        "enabled": True,
    }


def run_batch(*, run_id: str = RUN_ID, overwrite: bool = True,
              gloss_input_mode: str = DEFAULT_GLOSS_INPUT_MODE) -> dict:
    sample_ids = _pilot_sample_ids()
    if len(sample_ids) != 50:
        raise SystemExit(f"expected 50 pilot samples, got {len(sample_ids)}")
    backend = get_backend("openai_compat", **_backend_kwargs())
    description = backend.describe()
    # security: only booleans / reference names leave this function
    print(json.dumps({
        "endpoint_probe": description.get("endpoint_probe"),
        "model": description.get("model"),
        "provider": description.get("provider"),
        "credentials_present": description.get("credentials_present"),
        "credentials_ref": description.get("credentials_ref"),
        "status": description.get("status"),
        "n_samples": len(sample_ids),
        "group": "F",
        "prompt_version": PROMPT_VERSION,
        "gloss_input_mode": gloss_input_mode,
    }, ensure_ascii=False, indent=2))
    if not description.get("available"):
        raise SystemExit(f"real backend not available: {description}")
    result = harness.run(
        backend_name="openai_compat",
        prompt_version=PROMPT_VERSION,
        groups=("F",),
        sample_ids=sample_ids,
        run_id=run_id,
        allow_real=True,
        overwrite=overwrite,
        data_status=DATA_STATUS,
        backend_kwargs=_backend_kwargs(),
        concurrency=1,
        max_tokens_override=None,
        gloss_input_mode=gloss_input_mode,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group F real batch: SLRT gloss -> LLM sentence")
    parser.add_argument("--run-id", default=None,
                        help=f"run id (default: {SPACED_RUN_ID} for spaced_tokens, {JOINED_RUN_ID} for candidate_text)")
    parser.add_argument("--gloss-input-mode", default=DEFAULT_GLOSS_INPUT_MODE,
                        choices=harness.GLOSS_INPUT_MODES,
                        help="group F gloss rendering: spaced_tokens (default, per experiment spec) or "
                             "candidate_text (E02 B0 verbatim, no spaces)")
    parser.add_argument("--joined-baseline", action="store_true",
                        help="shortcut for --gloss-input-mode candidate_text (B0 verbatim baseline)")
    parser.add_argument("--no-overwrite", action="store_true", help="refuse to overwrite an existing run dir")
    parser.add_argument("--no-evaluate", action="store_true", help="only run the batch; skip evaluate.py")
    args = parser.parse_args(argv)

    gloss_input_mode = "candidate_text" if args.joined_baseline else args.gloss_input_mode
    run_id = args.run_id or (SPACED_RUN_ID if gloss_input_mode == "spaced_tokens" else JOINED_RUN_ID)
    result = run_batch(run_id=run_id, overwrite=not args.no_overwrite, gloss_input_mode=gloss_input_mode)
    print(json.dumps({
        "run_id": result["run_id"],
        "run_dir": str(result["run_dir"]),
        "n_rows": len(result["rows"]),
        "n_errors": (result["manifest"].get("outputs") or {}).get("n_errors"),
        "leaks": (result["manifest"].get("gt_isolation") or {}).get("gt_guard_leaks"),
        "prompt_version": (result["manifest"].get("prompt") or {}).get("version"),
        "prompt_json_sha256": ((result["manifest"].get("prompt") or {}).get("files") or {}).get("json_sha256"),
        "prompt_md_sha256": ((result["manifest"].get("prompt") or {}).get("files") or {}).get("md_sha256"),
    }, ensure_ascii=False, indent=2))

    if not args.no_evaluate:
        evaluated = evaluate_module.evaluate_run(run_id)
        summary = {
            "run_id": run_id,
            "data_status": evaluated["metrics"].get("data_status"),
            "groups": {
                group: {
                    "n": entry.get("n_ok"),
                    "bleu4_13a": (entry.get("metrics") or {}).get("bleu4_13a"),
                    "chrf": (entry.get("metrics") or {}).get("chrf"),
                    "mean_char_edit_similarity": (entry.get("metrics") or {}).get("mean_char_edit_similarity"),
                    "delta_bleu_vs_A": (entry.get("delta_bleu_vs_A") or {}).get("point"),
                }
                for group, entry in evaluated["metrics"]["groups"].items()
            },
            "paired_group_comparisons": evaluated["metrics"].get("paired_group_comparisons"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
