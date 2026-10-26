# E06 run

- run_id: 20261026_e06_rule_judge_cascade
- script: experiments/p1/e06/build_e06.py
- command: `$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'; .\.conda-env\python.exe experiments/p1/e06/build_e06.py`
- candidate texts = measured imported historical predictions (E02 / SLRT artifacts); routing+selection = simulated offline
- no LLM/Jev calls (BLOCKED: no endpoint/credentials); no model inference performed
- analysis outputs: experiments/p1/e06/ (candidate_inputs.jsonl, router_ablation.jsonl, judge_error_analysis.md, policy_candidate.yaml, report.md)
- eval_router_decisions.jsonl is an EVAL-SIDE file (contains GT-derived edit similarities), guard=False by design.
