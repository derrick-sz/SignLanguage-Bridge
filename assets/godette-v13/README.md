# Godette v13 reference-guided checkpoint

The editable frozen v13d Blender candidate and faithful GLB export are saved independently of v11/v12. This is a research candidate, not final aesthetic, linguistic or deployment acceptance.

Restore the `.blend.gz` and `.glb.gz` with `gzip -dc`, then verify decoded SHA256 against `BACKUP.json`. The two compressed files are lossless; both compressed and decoded identities are recorded.

V13 changes seven clothing meshes, trouser weights and 13 lower-body default-pose nodes. It retains original rest skeleton and inverse-bind matrices, face/hands, neutral v12 lip nodes, material definitions and UVs. See `README-modeling.md` and bounded export/roundtrip evidence in `validation/`. Keep the CC BY 4.0 attribution in `ATTRIBUTION.md`.

The exporter report is a historical pre-integration checkpoint: its statement that production v13 mouth support was not yet present refers to that moment. Subsequent runtime code and its independent regression evidence are committed separately on this branch. Do not copy a test-only runtime gate into deployment or pretend v13 has the v12 digest.

At the original export/integration checkpoint, the six real source-motion JSON files were absent and were not rerun. They were subsequently recovered and checked against the original handoff manifest. The bounded CPU rerun and minimal equal-rest interpolation fix are documented in [the six-motion report](../../docs/reports/godette-v13-six-motion-audit-2026-11-07.md), with [reproducible evidence](validation/six-motion-20261107/README.md). These results do not convert the earlier checkpoint into browser or linguistic acceptance. Production archive/catalog, browser/WebGL, semantic mouth shapes, arbitrary collision safety and mobile performance remain unverified. This model backup changes no v11/v12 file, merges no branch and deploys nothing.
