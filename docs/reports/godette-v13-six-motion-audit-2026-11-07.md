# Godette v13 六段真实源动作复测（2026-11-07）

审计包与机器可读证据：[validation/six-motion-20261107](../../assets/godette-v13/validation/six-motion-20261107/README.md)。此报告只证明列出的工程检查，不代替语言、真人或浏览器验收。

## Result

The frozen v13 GLB and Blender file were recovered from the user's uploaded gzip files. Both decoded SHA256 values match the frozen checkpoint. The original six source JSON files were recovered from the original handoff ZIP, and all 26 listed files match the handoff manifest.

The original pinned runtime failed an exact-default assertion: identical imported, slightly non-unit quaternions still entered Three.js slerp, perturbing chest/Spine_2_198 and neck3/Neck_3_130. The largest component drift measured in six-motion playback was 1.3416793898102242e-8. Hip_218 and the other protected lower-body defaults did not drift. This is a numerical exactness fix, not a diagnosis or cure of the source's hand-shape jitter.

A minimal equal-key copy branch fixes that defect without changing unequal-key interpolation, authored endpoints, source tracks, avatar bytes or weights. The local candidate passes 618 focused regression checks and the full bounded six-motion run, with **zero default-node drift**. The candidate is identified by SHA256, separately from the original pinned commit.

## Selected frontend regression

The final selected frontend suite has **46 tests: 45 pass, 0 fail, 1 skip**. The sole skip is the v12 exact-GLB test because that asset was not supplied; v12 synthetic rig/stage tests pass. Thirteen existing test files match the pinned GitHub blobs, and the new permanent regression has a verified 2-failure negative control and 3/3 fixed pass. The actual v13 mouth/skin test also covers 101 manual levels and 8,308 triangles. This is a selected frontend suite, not the complete repository/Python suite. See `equal_rest_candidate/FRONTEND-REGRESSION.md`.

## Evidence

- `uploaded-model-verification.json`: decoded and compressed input identities.
- `handoff-verification.json`: original ZIP/manifest, six wrappers, source metadata and pinned runtime identities.
- `six-motion-results-run1.json` / `run1.log`: initial strict failure. The JSON remains NOT_COMPLETED, not PASS.
- `six-motion-diagnostic-run3.json` / `run3.log`: unmodified original code, explicitly diagnostic only. It measures the tiny drift rather than hiding the strict failure.
- `six-motion-strict-candidate-run4.json` / `run4.log`: strict candidate pass with diagnostic mode disabled.
- `equal_rest_candidate/equal-rest.patch`: minimal production change.
- `equal_rest_candidate/regression.mjs`, red/green logs and diagnosis: real-GLB regression and exact affected nodes.
- `dependencies.json`: exact Three.js 0.170.0 dependency hashes.

## Tested scope

Six clips, 1,138 source frames at 25 fps. Raw, temporal and temporal-plus-IK each test every integer frame and fractions 0.1/0.25/0.5/0.75/0.9 between adjacent frames. Current MotionStage is separately checked at those times against independently compiled tracks after clearing the rig. Including reverse/repeat samples: **27,255 pose checks and 6,186,885 bone-world-matrix checks**.

All 227 bones remain finite. The expected 40 signing bones actually vary. All 13 protected v13 lower-body local/world defaults and all unanimated local defaults remain unchanged in the fixed run. Source wrappers, runtime geometry, weights, inverse-bind matrices, bind matrices, joint order and hierarchy remain unchanged.

All **300,447 vertices** are CPU-skinned at **63 selected poses**: each clip's start/middle/end in all three modes, plus cap-0067 indices 102/103/104 (4.08/4.12/4.16 seconds). The latter retains `source-quaternion-temporal-v1`. Raw GLB and loader-normalized weights are reported separately: all 224 tiny positive trouser influences remain present; maximum weight-sum errors are approximately 1.51e-7 and 4.56e-8 respectively.

Seek/repeat/end hold, pause/play/speed flags, deterministic 0.5x clock, stop/default reset, exact-v13 manual mouth and neutral restoration of all four coordinated mouth controls pass. The legacy automatic jaw channel remains unavailable. No automatic mouth semantics were added.

## Reproduce

Inputs are intentionally not duplicated in this small audit-source bundle:

1. Place the original `dot-blender-handoff-20261106.zip` at `inputs/` and safely extract into `handoff/`. ZIP SHA256: `fb5cd09cdc5896baa7ffaba998a1a6f7227ce9439a63cff0068dec6bbff623f7`.
2. Restore the frozen GLB from the existing v13 backup. Decoded SHA256: `2d89e2e33271d2e06c5a834a56675663aa5551814e2c98ef86fc084ca0c151a8`.
3. Use Python 3, Node.js, and Three.js 0.170.0. Set `THREE_ROOT` to the installed package directory. Dependency bytes are verified.

From the directory containing this audit folder (adjust its name if moved):

```sh
export THREE_ROOT=/absolute/path/to/node_modules/three
python v13_six_motion_audit/verify-inputs.py
node --loader ./v13_six_motion_audit/node-loader.mjs \
  ./v13_six_motion_audit/equal_rest_candidate/regression.mjs
RIG_CANDIDATE=./v13_six_motion_audit/equal_rest_candidate/web/rig.js \
node --loader ./v13_six_motion_audit/node-loader.mjs \
  ./v13_six_motion_audit/run-six-motion-audit.mjs \
  /absolute/path/to/godette-reference-v13.glb \
  ./v13_six_motion_audit/fresh-result.json
```

A result path must not already exist. Each run begins with a NOT_COMPLETED marker and only writes PASS after completing every assertion. To reproduce the original diagnostic, omit `RIG_CANDIDATE` and set `ALLOW_FLOAT_REST_DIAGNOSTIC=1`; that result is not strict acceptance. The original unchanged runtime without either option reproduces the failure.

## Limits

This is actual Three.js GLTFLoader + production rig/retarget/stage CPU execution with stubbed texture decoding and rendering. It is **not** browser/WebGL, screenshot, mobile-performance or deployed-vendor equivalence evidence. Full original motion archive, MotionRegistry → CLI → package → server, deployment, video alignment, human signing quality, semantic correctness, semantic mouth shapes and general collision safety were not tested.

The retargeter maps upper-body signing joints, not lower-body source motion. The trousers stay static in these clips; their zero displacement is not proof of arbitrary hip/knee or clothing-contact safety. Selected-pose finite skinning does not prove no self-intersections. Historical v11/v12 whole-skeleton zero-difference results are not reused.

Wrapper SHA256, declared registry source-bundle SHA256, and declared source-video SHA256 are different identities. Only wrapper bytes were independently verified here; the declared registry hashes agree with both the old handoff and current preset. No new language/quality approval was granted, and the frozen models were not changed.
