# Godette v11 fusion checkpoint

Base: PR #1 commit `ad3806f8e0ba2227231492860cfed33eb70347f9`. The upstream PR head was rechecked and remains unchanged. Work is on `codex/godette-v11-fusion-20261106`; no merge or deployment has been performed.

## Implementation

- Independent v11 presentation builder preserves all six pilot source hashes and the cap-0067 temporal profile, rejects mismatched/missing sources, and never copies old-target baked exports.
- v11 manual mouth uses an exact avatar SHA and separate 0.04 rad limit. Legacy template/target automatic-jaw gates and the SignSparK manifest preflight are unchanged.
- Quaternion interpolation normalizes detached changing intermediate samples only. Authored endpoints, unchanged keys and source arrays are preserved. Nominal integer times with floating-point residues can change, as documented.
- Display attribution, package resources and regression tests are included. Existing corpus/generation sources and the frozen model bytes remain unchanged.

## Backup

The exact editable `.blend` and losslessly compressed `.glb.gz` are committed under `assets/godette-v11/`, with sizes, SHA256 and Git blob identities in `BACKUP.json`. Decompression was checked byte-for-byte. The raw GLB request exceeded the 16 MiB transport limit; the gzip contains the exact original bytes and must be decompressed before use. The separately delivered complete model ZIP remains an additional archive.

## Recovery verification, 2026-11-07

This supersedes the incomplete-environment full-suite status below. The earlier run remains historical evidence rather than being relabeled a pass.

- Restored another 89 tracked files from the exact base commit, checking each Git blob identity. This is a 381-file source/test/config snapshot, not a claim that all repository assets were downloaded. Restored files were already upstream and were not republished as modifications.
- Installed official PyPI QA dependencies into an isolated cloud-workspace directory. OpenCV, MediaPipe, PySide6/Qt offscreen, scikit-learn and ONNX imports were verified. System packages and user-local data were not changed.
- Classified all 156 prior failure/error identities: missing OpenCV 103, PySide6 12, MediaPipe 11, missing tracked repository files 26, and a test-local configuration dependency 4. Background-event probes traced apparent timeouts to missing OpenCV, rather than assuming a timing defect.
- Fixed only `tests/test_model_defaults.py`: four desktop-default tests now copy the checked-in example into a temporary fixture and use the real configuration loader. No production model-selection behavior or ignored local configuration was changed. Red: 4 failed / 23 passed / 1 skipped. Green: 27 passed / 1 skipped, independently reviewed and rerun.
- Full Python suite completed twice: **1322 passed, 42 skipped, 0 failed, 0 errors, 3 subtests passed**, with 17 deprecation warnings, including two pytest parameterization warnings in project tests. Durations were 81.42 s and 72.50 s. The second run generated JUnit evidence and explicit skip reasons.
- The 42 skips remain unverified: 22 require Torch, 18 require external/pinned assets or explicit real-asset opt-in, one requires an ignored developer configuration, and one is Windows-specific. Collection-level skips can hide more than one eventual test.
- Node: **50 passed, 1 skipped, 0 failed**, using actual Three.js 0.170.0 and the exact v11 GLB. Only the old calibrated GLB test remained skipped. This is not a browser/WebGL or GPU acceptance run.
- Verified code commit: `491704e1ff0adf02d58cd965823ed92f94b22d76`. PR #1 remained at `ad3806f8e0ba2227231492860cfed33eb70347f9`. The corpus, generation and automatic-jaw identity gates remain unchanged.

See [recovery evidence](evidence/godette-v11-fusion/recovery-verification.json), [restored file identities](evidence/godette-v11-fusion/restored-files.json) and [reproduction steps](evidence/godette-v11-fusion/RECOVERY.md). This is engineering regression evidence, not sign-language quality approval.

## Historical partial-snapshot verification before the code backup

- Restored 292 source/test/config files directly from the base commit and verified every Git blob identity.
- `git apply --check` and apply succeeded; all 18 changed files match the independently reviewed delivery byte-for-byte.
- Python focused suite: 84 passed, 4 explicit skips, 3 subtests passed; one Starlette/AnyIO deprecation warning.
- All 51 Node tests ran with actual Three.js r170 and the frozen v11 GLB: 50 passed, one skipped because the old calibrated GLB was absent in the recovered environment. The v11 real-asset and interpolation tests passed. The earlier independent delivery included the old GLB and recorded 51/51; that is separate evidence, not a claim of a fresh old-asset rerun.
- Full pytest now completed for both the untouched base and final candidate in the same recovered environment. Base: 903 passed, 111 failed, 45 errors, 83 skipped. Candidate: 912 passed, 111 failed, 45 errors, 83 skipped, 3 subtests passed. Both recorded 3 warnings. All 156 failure/error identities match exactly; no new failures or errors were introduced by this patch. These are dependency/partial-snapshot limits and are NOT marked passed. The complete failure/error identity list is committed in [fresh verification](evidence/godette-v11-fusion/fresh-verification.json).
- No commit statuses or PR-triggered GitHub Actions runs were reported for the code commit at verification time. No CI success is claimed.

## Still unverified

Production motion archive/catalog, actual 19,877-record corpus, GPU generation, deployed vendor, browser/WebGL/texture rendering, end-to-end service deployment, contact/collision, sign-language quality and mobile performance. This is a research candidate, not language approval or deployment readiness.

See [integration instructions](../godette-v11-presentation.md) and [asset backup](../../assets/godette-v11/README.md). Godette (Rigged) by zahlenmaler, CC BY 4.0; original source and license links and modification descriptions accompany the files.

