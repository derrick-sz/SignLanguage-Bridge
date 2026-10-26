"""G2 phase-2A: LLM fusion ablation harness (C/D/E groups) with stub self-test.

The harness is deliberately split in two protections:

* `harness.py` builds prompts and calls a backend.  It never reads the GT
  reference table; prompts come only from `experiments/p1/e02/candidate_outputs.jsonl`
  (models' own predictions) plus whitelisted raw SLRT score columns.
* `evaluate.py` is the only module that joins `references/gt_1086.csv` (EVAL ONLY)
  and computes canonical metrics (BLEU-4 / chrF / edit similarity / exact match)
  plus signer-clustered paired bootstrap vs the A baseline.

Real LLM backends (codex_cli / openai_compat) are implemented as interfaces and
stay disabled until the Lead/user approves an endpoint and fixes the model id.
This round is `IMPLEMENTED_UNTESTED` for real endpoints; the stub run validates
the full pipeline on CPU only.
"""

__version__ = "0.1.0"
