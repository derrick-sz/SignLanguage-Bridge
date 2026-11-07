# Godette v11 fusion checkpoint

Base: PR #1 commit `ad3806f8e0ba2227231492860cfed33eb70347f9`. The upstream PR head was rechecked and remains unchanged. Work is on `codex/godette-v11-fusion-20261106`; no merge or deployment has been performed.

## Implementation

- Independent v11 presentation builder preserves all six pilot source hashes and the cap-0067 temporal profile, rejects mismatched/missing sources, and never copies old-target baked exports.
- v11 manual mouth uses an exact avatar SHA and separate 0.04 rad limit. Legacy template/target automatic-jaw gates and the SignSparK manifest preflight are unchanged.
- Quaternion interpolation normalizes detached changing intermediate samples only. Authored endpoints, unchanged keys and source arrays are preserved. Nominal integer times with floating-point residues can change, as documented.
- Display attribution, package resources and regression tests are included. Existing corpus/generation sources and the frozen model bytes remain unchanged.

## Backup

The exact editable `.blend` and losslessly compressed `.glb.gz` are committed under `assets/godette-v11/`, with sizes, SHA256 and Git blob identities in `BACKUP.json`. Decompression was checked byte-for-byte. The raw GLB request exceeded the 16 MiB transport limit; the gzip contains the exact original bytes and must be decompressed before use. The separately delivered complete model ZIP remains an additional archive.

## Fresh verification before this code backup

- Restored 292 source/test/config files directly from the base commit and verified every Git blob identity.
- `git apply --check` and apply succeeded; all 18 changed files match the independently reviewed delivery byte-for-byte.
- Python focused suite: 84 passed, 4 explicit skips, 3 subtests passed; one Starlette/AnyIO deprecation warning.
- All 51 Node tests ran with actual Three.js r170 and the frozen v11 GLB: 50 passed, one skipped because the old calibrated GLB was absent in the recovered environment. The v11 real-asset and interpolation tests passed. The earlier independent delivery included the old GLB and recorded 51/51; that is separate evidence, not a claim of a fresh old-asset rerun.
- Full pytest was started in the recovered environment and remains in progress at this checkpoint. It is not claimed green. The earlier comparison in the delivery reported identical 111 failures and 45 errors for baseline/candidate due to unavailable dependencies/resources; complete failure lists are retained in that delivery.

## Still unverified

Production motion archive/catalog, actual 19,877-record corpus, GPU generation, deployed vendor, browser/WebGL/texture rendering, end-to-end service deployment, contact/collision, sign-language quality and mobile performance. This is a research candidate, not language approval or deployment readiness.

See [integration instructions](../godette-v11-presentation.md) and [asset backup](../../assets/godette-v11/README.md). Godette (Rigged) by zahlenmaler, CC BY 4.0; original source and license links and modification descriptions accompany the files.
