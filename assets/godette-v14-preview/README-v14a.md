# Godette v14A reference-fit candidate — 2026-11-07

**Selected candidate: A.** This is a separate, reversible clothing refinement based on the user's frozen v13. It is not a replacement for the tested v13 runtime asset, and is not an exact one-to-one reconstruction of the reference drawing.

## Concrete changes

- The jacket's horizontal hem is gently curved: lateral corners rise by up to 12 mm. The jacket, jade binding, lower placket, inner panel, princess seams and lower cloud ornaments follow the same deformation, avoiding detached trim.
- The trouser cuff rises 18 mm, with a smooth falloff ending at mid-shin. This increases ankle exposure toward the reference without changing shoes, ankles, upper trousers or the crotch shape.
- Twelve clothing mesh coordinate arrays change. Bone rests/hierarchy, all native default pose channels, transforms, topology, UVs, skinning weights, modifiers, materials and original face/hands are preserved.
- An optional sub-millimetre pressed crease (candidate B) was tested and rejected: independent image comparison found no meaningful visible improvement. B is not the selected delivery.

This pass does not reconstruct the pelvis or change skeletal binding. The v13 pelvis/leg refinement is retained. The newly exposed ankles retain the original simplified cylindrical appearance.

## Validation

- `validation/v13-v14a-delta.json`: original Object_7 and all 227 bone rests/defaults preserved; only the twelve clothing meshes' coordinates change.
- `validation/v14a-deformation-66-states.json`: native default, rest, neutral standing plus three 21-point hip/knee sweeps, 66 states total. All pants self-intersection BVH candidate counts are zero; all checked coordinates are finite. Hip/knee endpoints are unilateral 35°/80° and bilateral 25°/50°, with foot rotation -15°.
- `validation/native-coat-contact-comparison.json`: native-default jacket has 1,686 nonadjacent BVH candidates already present in v13. Candidate A has exactly the same triangle-pair set and bounding region (approximately z=1.204–1.332 m); zero added pairs. These are broad-phase candidates, not a proof of confirmed visible crossings. This inherited upper-jacket issue is not solved here.
- `review_independent/REVIEW.md` and `preservation-results.json`: independent actual-pixel review and data-preservation checks. No newly degenerate or inverted trouser triangles found. No new visible standing seam gap, cuff defect or disconnected ornament found.
- Actual native-default, neutral standing and eight stress-view renders were inspected. The neutral standing views are explicitly presentation poses, not the saved native default.

Zero pants candidates in sampled poses does not establish arbitrary-motion or cross-object collision safety. GLB export, inverse-bind round-trip, browser/WebGL, the six real motion clips, semantic signing, mouth semantics, runtime version recognition and deployment were not rerun on v14A. Do not reuse v13's runtime pass as v14 acceptance. No production code or v13 branch was changed by this model pass.

## Files and recovery

The main checkpoint is `godette-v14a-hem-cuff.blend`. Its gzip copy is lossless. The evidence ZIP contains selected actual renders, validation, review, scripts and this documentation; it intentionally excludes the model, dependencies and the user's reference image. Obtain both the model gzip and evidence ZIP for a complete checkpoint.

To recover the Blender file, decompress the `.blend.gz` using gzip or an archive utility, then verify SHA256 against `BACKUP-MANIFEST.json`. Do not apply the neutral presentation pose as a new bone rest. Existing v13 source remains the rollback point.

`ATTRIBUTION.md` must accompany redistribution. Godette (Rigged) by zahlenmaler is CC BY 4.0, not CC0. This pass adds no new downloaded third-party assets. The user-provided reference image and image-containing comparison sheet are excluded from the repository backup.

## Reproduction

Blender 4.3.2. Scripts are authored for this task and use Blender's bundled Python API. They write only under the isolated exploration directory. The build script records both A and the rejected B; use the A checkpoint for this delivery. Baseline input SHA256: `f03f7ebecb892ac2b01a0768d3c140802cf06e207c4bad33bd0aa4f0b92822b4`.

The workflow retains the public Blender retopology/rigging principles used in the preceding pass: inspect silhouette and joint flow, preserve protected data, and evaluate actual posed geometry and rendered results. No external Blender 5.2 scripts were installed or executed.
