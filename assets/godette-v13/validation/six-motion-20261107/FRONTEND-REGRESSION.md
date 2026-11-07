# Candidate frontend regression

Final run: **46 tests: 45 pass, 0 fail, 1 skip**, Node v24.22.1, Three r170 (0.170.0). Reproduce with `bash run-frontend.sh` from this directory (script resolves absolute paths). Full output: `frontend-final.log`.

Thirteen existing test files were fetched directly from betterway614/SignLanguage-Bridge at commit cc11b4d6ed0f51e0d727a45a2f2aba71ea1e289f, not copied from historical fusion_review tests. `verify-frontend-sources.py` independently recomputes Git blob hashes against the GitHub-returned hashes; all 13 match exactly. Copied pinned runtime dependencies also match recorded Git blobs. Provenance: `frontend-source-verification.json`.

Existing files in isolated `repo/tests/`:
- animation_interpolation.test.mjs
- animation_expression.test.mjs
- animation_stage.test.mjs
- animation_geometry.test.mjs
- animation_nonmanual.test.mjs
- animation_nonmanual_stage.test.mjs
- animation_nonmanual_rig.test.mjs
- animation_godette_v11.test.mjs
- animation_godette_v11_glb.test.mjs
- animation_godette_v12.test.mjs
- animation_godette_v12_glb.test.mjs
- animation_godette_v13.test.mjs
- animation_godette_v13_glb.test.mjs

The only skip is the v12 exact-GLB 101-level geometry test because that GLB is unavailable. The v12 synthetic rig/stage tests run and pass. Recovered v11 actual-asset test runs and passes. Old calibrated jaw uses handoff/presentation/avatar-clean.glb, SHA checked by its existing test. V13 uses inputs/uploaded-v13/godette-reference-v13.glb, SHA checked by its existing test.

The v13 real GLTFLoader/CPU skinning test passes 101 manual mouth levels, 227 bones and 8,308 retained triangles. Zero relative normal reversal samples, zero upper-teeth/nonface motion, zero neutral reset vertex error. Minimum area ratio 0.8708633395870086; maximum area ratio 2.819055094949218; maximum edge ratio 1.6556852353601865. Full report: `mouth-report/mouth-skin-three-170.json`. These are relative CPU geometry checks, not browser rendering or collision/semantic acceptance.

## Permanent regression deliverable

Add `repo/tests/animation_equal_rest_interpolation.test.mjs` as `tests/animation_equal_rest_interpolation.test.mjs` in the target repository. It uses the existing ANIMATION_THREE_MODULE convention, needs no model asset, and includes the actual frozen v13 chest quaternion as a minimal fixture. Three named tests cover missing-target defaults, separate equal under/over-unit keys, exact endpoints, changing/antipodal interpolation, source immutability and repeated backward fractions.

Verified negative control using the unmodified pinned rig: **1 pass, 2 fail, 0 skip**, exit 1 (`permanent-red.log`). With candidate rig: **3 pass, 0 fail, 0 skip**, exit 0 (`permanent-green.log`). Original rig was swapped only inside this isolated repo for the negative control and restored to candidate afterward. Neither pinned audit runtime nor parent harness was modified.

The production patch remains `equal-rest.patch`; candidate rig SHA256 c886ff0441af0873e113ab4934b75b77dc8449af2a112ac2b2ea55c7681c1505. This is a selected frontend regression suite, not all repository tests. Python whole-suite intentionally excluded for this JS-only work. No remote writes were performed.
