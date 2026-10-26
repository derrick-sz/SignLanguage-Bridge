# E03 run

- run_id: 20261026_e03_slrt_confidence
- script: experiments/p1/e03/build_e03.py
- command: `$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'; .\.conda-env\python.exe experiments/p1/e03/build_e03.py`
- analysis outputs: experiments/p1/e03/ (per_sample_scores.csv, calibration_results.json, risk_coverage.csv, score_definition.md, report.md)
- no inference was run; SLRT/Uni-Sign historical outputs were read only, GT only on the evaluation side.
