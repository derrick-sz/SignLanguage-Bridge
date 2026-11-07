# Godette v12 runtime stage checkpoint

V12 is an intermediate modeling checkpoint. Reference-guided lower-body refinement continues in v13; freezing v12 bytes is not final appearance acceptance.

Base: `308aa8df30faf54b0e4511d640b544c2481d3b0d`. Model backup commit: `d7c9f5d33e686ecf939878e648d6d2fef3173118` on `codex/godette-v12-neutral-20261107`. No merge or deployment.

## Identity and behavior

- Runtime GLB SHA256: `a7942f7a69e8fb078d50767e85944c0e59b47dc20b40479ab86d8c16513717c4`.
- Editable Blend SHA256: `38666f464943ce2e5e9498ce6d461130e689aafce1c92304ea91a568de0420e4`.
- V12 alone gets the existing coordinated manual-mouth driver at 0.04 rad only when the profile and mouth hierarchy also match. Recognized v12 with invalid controls stays closed and reports zero available angle rather than falling back to jaw-only.
- The legacy automatic source-jaw calibration remains disabled for v12. V11 and the old calibrated avatar retain their existing gates. Unknown identities still have the pre-existing generic jaw-only fallback; this change is not a blanket guarantee that unknown models cannot move.
- Neutral lip pose is authored into the GLB before rig initialization, so expression reset and rig reset return to that neutral baseline. It is not applied as rest pose.
- A separate v12 builder/manifest/CLI publishes the avatar without overwriting an existing output. All six source pilot bindings, the rig profile, and research/unverified flags are retained. The motion archive/catalog is not altered.

## Verification

Fresh full Python run: **1332 passed, 42 skipped, 0 failed, 17 warnings, 6 passed subtests**. Full Node suite on Three.js **0.169.0 and 0.170.0: 57 passed, 1 skipped each**, with actual v11 and v12 GLBs enabled. Wheel build and installed-resource contents were verified.

The v12 real GLTFLoader/CPU-skinning test checks 227 bones, five retained face/hand/skin primitives, 8,308 nondegenerate triangles and 101 opening intensities. It reports zero relative normal reversals, minimum area ratio 0.870863, maximum area ratio 2.819055 and maximum edge ratio 1.655685. Lower teeth move downward 3.10653 mm; upper teeth and other meshes do not move. Repeated/backward samples and both resets are component-exact, with zero reset vertex error.

Red tests first reproduced the disabled v12 manual driver and unsafe v12 fallback, followed by passing tests using the exact new bytes. Builder tests cover digest/source rejection, immutable input, exclusive output creation, validation cleanup, real registry/API fixture flow and the real frozen v12 avatar. Synthetic source fixtures are not evidence of production-corpus acceptance.

Evidence: [logs and metrics](evidence/godette-v12-integration/). Python skip reasons are preserved in `python-full.log`; the Node skip is the absent old calibrated-avatar GLB. Python skips include Torch, external/pinned real assets, ignored local config and Windows-specific behavior. They are not passes. Existing deprecation warnings are retained.

## Reproduce

Restore both versioned GLBs from `assets/godette-v11` and `assets/godette-v12`, checking decoded SHA256 against their backup manifests. Use the isolated QA dependencies in `evidence/godette-v11-fusion/qa-requirements.txt` and `qa-constraints.txt` with Qt offscreen. Then run:

```sh
ANIMATION_V11_AVATAR_GLB=/path/to/godette-reference-v11.glb \
ANIMATION_V12_AVATAR_GLB=/path/to/godette-reference-v12.glb \
QT_QPA_PLATFORM=offscreen python -m pytest -q -rs

ANIMATION_THREE_MODULE=/path/to/node_modules/three/build/three.module.js \
ANIMATION_GLTF_LOADER=/path/to/node_modules/three/examples/jsm/loaders/GLTFLoader.js \
ANIMATION_V11_AVATAR_GLB=/path/to/godette-reference-v11.glb \
ANIMATION_V12_AVATAR_GLB=/path/to/godette-reference-v12.glb \
node --test tests/animation*.test.mjs

python scripts/animation/build_godette_v12_presentation.py \
  --archive /path/to/verified-motion.zip --avatar /path/to/godette-reference-v12.glb \
  --output /path/to/new-v12-presentation
```

The actual production archive and six motion JSON files were not available here, so the last command and six full motion clips were not run against production data. The CLI is tested with a real registry built from explicit synthetic fixtures and never silently fabricates the missing archive.

## Limits and rollback

CPU geometry uses a texture-decoding stub. It does not establish WebGL/browser rendering, arbitrary-pose collision safety, sign-language quality, semantic mouth shapes, lip sync, GPU generation or mobile performance. The modeling checkpoint contains its own bounded shape/roundtrip validation; these are distinct from runtime and production validation.

No existing v11 assets or builder are replaced. Roll back by using the unchanged v11 presentation directory/builder and its exact GLB, or by reverting this code checkpoint. Never relabel v12 bytes as v11 or as the legacy calibrated avatar.

Remote model readback verifies branch HEAD, tree paths, Git content hashes and byte sizes against locally computed values. Local gzip decompression was checked against the frozen original SHA256. The connector does not expose binary blob bytes for a second download, so no remote re-download/decompression is claimed.
