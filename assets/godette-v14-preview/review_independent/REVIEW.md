# Independent v14 reference-fit review

## Decision

Retain **A, godette-v14a-hem-cuff.blend**, for the final hem/cuff refinement. Keep B only as an optional exploration checkpoint.

The curved jacket hem is visibly closer to the reference's gently curved lower outline in front and rear views. Shortening the cuffs improves the amount of ankle showing. B's sub-millimetre crease adds no meaningful visible reference fidelity in the provided renders, while A gives the same useful hem/cuff changes with less geometry alteration.

## Evidence examined

- Actual pixels of reference/image(20261107-042154).png.
- All v13/v14b standing front, side, rear and lower-front previews.
- v14a standing front and lower-front previews.
- Build, render and delta-validation scripts, manifests and recorded hashes.
- An independent read-only Blender inspection of baseline, A and B, with script auto-execution disabled. No candidate, baseline or motion-audit script was modified.

## Visual findings

1. **Jacket hem:** The lateral lift produces a gentler center-low outline than v13's straight horizontal cut. Lower binding, front piping and gold decoration remain attached in the visible views. No new obvious gap, disconnected ornament, hard crease or jagged hem appears.
2. **Trouser cuffs:** The lift exposes more ankle, consistent with the reference. The cuffs remain even and the lower legs remain smooth. There is no new visible hole or abrupt bend in the supplied views.
3. **B versus A:** At lower-front resolution 850 × 1000, the trouser region x315–535, y260–875 differs by mean absolute RGB 0.068/255, 99th percentile 1/255 and maximum 2/255. This supports the visual finding that B's added ridge is effectively imperceptible at the supplied display scale. It does not reproduce the clearer, thin drawn crease in the reference. These pixel statistics measure A/B difference, not resemblance to the reference.
4. **Remaining ankle issue, not a blocker introduced by this pass:** The visible ankles still read as straight, nearly cylindrical skin columns, especially just above the shoe openings in the front close-up and at the heel in side view. The cuff lift exposes more of this existing construction. Their meshes are unchanged; no adjustment to protected skin is recommended within this pass. Any future ankle work should be explicitly scoped separately.
5. **Remaining pelvis issue:** The reference has more garment seam/fold articulation around the seat and a different hip-to-leg silhouette. The candidate retains the existing soft crotch dimple. This pass should be described as hem/cuff reference-fit refinement, not another pelvis reconstruction.

## Independent preservation and geometry checks

See preservation-results.json and inspect_preservation.py.

- Baseline hash matches the supplied frozen SHA-256.
- A hash: 46398198b758afbb012ea2e9cad5b0e9236eec2ec2de36a9eb62124daedac6c9.
- B hash: b92681a0a1dcb54de0e91d338bea4ef93a409f861b7e7665da8214dad7acf6b9.
- All three have matching inspected rig data, bone rest matrices/hierarchy, pose matrix bases, rig constraints, object transforms, object modifiers and object constraints.
- Material scalar settings and node/socket/link signatures match. The independent check expands on the original delta script, which compared material slot names only.
- For the inspected meshes, only coordinate arrays differ. Mesh topology, edge lists, polygon flags/material assignment, UVs, groups, weights and shape-key signatures are unchanged. Object_7 and both ankle/instep meshes are unchanged.
- A alters 4,006 trouser vertices above 1e-8 tolerance; maximum displacement is 0.018 Blender units with scene scale_length 1.0. No trouser vertex with source z >= 0.29 changes in A.
- A and B both leave 1,924 sampled central-crotch vertices (|x| < 0.025, z > 0.70) exactly unchanged. Both leave the region above z=0.87 unchanged.
- Corresponding trouser triangle checks find zero newly degenerate triangles and zero inverted normals versus baseline for A or B. A's minimum corresponding normal dot is 0.9998485 and minimum triangle area ratio 0.866387. These are local shape checks, not a proof against self-intersection.

## Reporting limits and release hygiene

- Use the exact A file/hash for final deformation validation. B's motion audit must not be presented as the exact A audit. The parent has stated that it is rerunning A.
- The saved `godette-v14b-reference-standing.blend` is a posed presentation derivative. Its pose/camera/render state differs deliberately from the untouched native defaults in the A/B edit checkpoints. Do not conflate it with the preserved-default candidate.
- “Only mesh coordinates changed” should be scoped to inspected asset data. The build also changes a scene revision-description property, and these checks do not exhaustively serialize every possible datablock/custom property/animation driver.
- This review checks static supplied renders and independent local geometry preservation. It does not independently run or certify the 66-state audit, runtime exports, all self-collision cases or every animated frame.
- No 1:1 reproduction claim is supported. Head, hands, hair, overall proportions, folds and shading still differ from the illustrated reference, mostly outside this pass's allowed scope.
- Parent reports identical pre-existing upper-coat BVH self-intersection candidate sets for v13/A/B. Treat those as an inherited limitation pending diagnosis, not as new v14 damage or as proof the entire character is collision-free.

No newly introduced visible release blocker found in the inspected standing views. Retain A after its exact-file checks finish.
