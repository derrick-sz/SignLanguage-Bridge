# Uni-Sign direction correction and evaluation integrity

Goal: implement the direction correction authorized by the user's request to verify and repair the preceding diagnosis, and validate it without selecting directions from test answers.

Architecture: preserve input orientation by default; expose explicit preserve/mirror correction at the pose boundary for file and captured clips. Keep pose caches canonical and bind evaluation/resume records to complete preprocessing and decoding identities. Calibrate a scoped CE-CSL direction map on dev only, then freeze it for test regression.

Constraints: preserve all existing local changes and historical evidence; snapshots are in artifacts/unisign-cecsl-eval/repair-20261024/before. No weight training, publication, implicit global flipping, per-test-row best-of-two selection or replacement of TFNet. Report pose-cache regression separately from real-video validation.

Public verification boundaries: pose preprocessing; Uni-Sign adapter inference for file/captured input; evaluation cache serialization and CLI resume; direction-map validation and dev calibration. Tests use independent COCO anatomical examples and negative controls.

- [x] Add a failing anatomical reflection test, then implement immutable direction correction with confidence and quality preserved.
- [x] Add backend configuration and validation; verify both input kinds and default behavior.
- [x] Add evaluator split/direction support and protocol-bound pose/resume identity, with stale-cache and changed-parameter negative controls.
- [x] Fix the historical trim key mismatch, make the repaired probe reproducible, and prove requested trims actually run. Mechanical verification passed; a full corpus trim BLEU rerun remains NOT_RUN and is outside this acceptance claim.
- [x] Calibrate direction on dev samples, lock the map, validate on disjoint dev samples and real video extraction.
- [x] Re-decode all 500 test pose caches with the locked dev map; label this as regression on an already inspected test set.
- [x] Run relevant regression tests and request an independent code/evidence review; fix important findings and write a repair report.

Result: 36 dev calibration samples and 24 disjoint dev validation samples; test BLEU-ZH 7.9934 -> 12.3019; 11 real-video public inference calls completed; targeted tests 34 passed; full suite 786 passed, 10 skipped. See docs/reports/unisign-cecsl-direction-repair-2026-10-24.md for exact boundaries and artifacts.
