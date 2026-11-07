# Godette v13 reference-guided runtime integration

This is a versioned research checkpoint, not final appearance approval or a deployment. V11 and v12 remain available unchanged.

Base: v12 integration `218c1a64a8180918994e7cb17cfbf31170ec127d`. V13 model backup: `e66657fc633bec53c750cb6a03c0ee52c84502cc` on `codex/godette-v13-reference-20261107`.

## Frozen inputs

- Runtime GLB SHA256: `2d89e2e33271d2e06c5a834a56675663aa5551814e2c98ef86fc084ca0c151a8`.
- Editable v13d Blender SHA256: `f03f7ebecb892ac2b01a0768d3c140802cf06e207c4bad33bd0aa4f0b92822b4`.
- Lossless compressed backup and original decoded SHA256 values are in [assets/godette-v13/BACKUP.json](../../assets/godette-v13/BACKUP.json). Original Godette attribution and CC BY 4.0 derivative changes are retained.

The GLB export preserves the original 227 rest bones, hierarchy and inverse-bind matrices, plus the v12 neutral lip nodes. It intentionally changes 13 lower-body default pose nodes, seven clothing meshes and trouser weights. See the separately scoped [export/roundtrip report](../../assets/godette-v13/validation/README-export-validation.md).

## Runtime changes

The exact v13 identity joins v12 in the coordinated manual-mouth gate. Valid profile and mouth hierarchy are still required. Manual opening remains bounded to 0.04 rad; recognized v13 with invalid controls reports zero available angle and stays closed. Legacy automatic source-jaw calibration remains unavailable for v13. Old v11/v12 behavior and original source calibration are preserved.

Unknown identities retain the old generic jaw-only fallback. `manualMouthAvailable=false` must not be described as a universal guarantee that every unknown avatar stays still. The historical exporter `production-gate.json` documents the pre-integration v13 fallback, which these committed changes resolve for the exact frozen v13 bytes.

A separate v13 manifest/builder/CLI validates the frozen asset and all six original source-pilot bindings. It refuses an existing destination, cleans up failed staging, preserves input data, and publishes only the avatar, manifest and attribution. All review flags remain unverified/research-only, with no automatic nonmanual or export approval. Neither the original catalog/archive nor either old version's builder is changed.

## Verification

Fresh full Python suite: **1342 passed, 42 skipped, 0 failed, 17 warnings, 9 passed subtests**.

Full Node suite on both Three.js **0.169.0 and 0.170.0: 64 passed, 1 skipped each**. Actual v11, v12 and v13 files were enabled. The old calibrated-avatar GLB was absent, causing the single Node skip. Python skips cover Torch, external/pinned assets or explicit real-asset opt-ins, local ignored config and Windows-specific behavior. Skips are not passes; exact inventory and warnings are preserved in the logs.

The v13 actual GLTFLoader test independently runs the production candidate code, not the exporter's substituted test-only module. Across 101 intensities and 8,308 retained skin triangles: no relative normal reversals, minimum area ratio 0.870863, maximum area ratio 2.819055, maximum edge ratio 1.655685. Lower teeth drop 3.10653 mm, upper teeth and other meshes have zero motion, and repeat/backward/neutral resets are component-exact with zero vertex error.

New regression tests first failed against v12-only code, then passed after the exact v13 gate was added. Ten v13 Python tests cover real frozen bytes, v11/v12 rejection by the v13 builder, source-binding/digest checks, output ownership/races, immutable input, validation cleanup and the real CLI/registry/API chain using labeled synthetic motion fixtures. Wheel build contains all three versioned builders, manifests and attributions.

[Evidence and reproducible logs](evidence/godette-v13-integration/).

## Reproduce

Use the isolated QA requirements/constraints from the v11 recovery report. Restore GLBs by decoding their versioned `.glb.gz` backups and verify the decoded digests. Set all three `ANIMATION_V11_AVATAR_GLB`, `ANIMATION_V12_AVATAR_GLB` and `ANIMATION_V13_AVATAR_GLB` paths, and run:

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q -rs
ANIMATION_THREE_MODULE=/path/to/node_modules/three/build/three.module.js \
ANIMATION_GLTF_LOADER=/path/to/node_modules/three/examples/jsm/loaders/GLTFLoader.js \
node --test tests/animation*.test.mjs

python scripts/animation/build_godette_v13_presentation.py \
  --archive /path/to/verified-motion.zip --avatar /path/to/godette-reference-v13.glb \
  --output /path/to/new-v13-presentation
```

The final builder command requires the real source archive, which is unavailable in this workspace. No production presentation directory was created or deployed. Restore that archive and six actual motion JSON files before claiming real six-clip acceptance; current CLI/API tests use explicit synthetic fixtures. V13's 13 intentional lower-body default pose changes mean old full-world-matrix equality claims cannot be copied forward without interpreting the new baseline.

## Remaining limits and rollback

Six real clips, production corpus/catalog integration, browser/WebGL visuals, semantic mouth shapes, sign-language quality, arbitrary-pose collision safety, GPU generation and mobile performance are not certified. Texture decode in CPU loader tests is stubbed. Modeling/roundtrip tests are bounded engineering evidence, not those missing outcomes.

Select the unchanged v12 or v11 presentation directory to roll back. Do not replace hashes or relabel bytes. Branch HEAD, all committed text and Git content hashes/sizes are read back after publication. Compressed bytes were locally decompressed and verified; the connector cannot return large binary bytes, so no remote re-download/decompression is claimed. No branch was merged and nothing was deployed.
