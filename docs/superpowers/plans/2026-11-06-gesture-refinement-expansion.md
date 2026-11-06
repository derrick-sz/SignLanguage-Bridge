# Gesture Refinement and Expansion Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; fresh-context final review. User authorized options1+2.

**Goal:** Refine source gesture candidates and expand20 retained sentences to100 unique texts with source-bound animated exports.

**Architecture:** Offline, reversible quaternion repair creates an additional pose stream. The runtime selects raw/refined data explicitly and feeds the same selected poses into rotation retargeting and FK/IK. Source corpus and prior outputs remain unchanged.

**Tech Stack:** existing Python/numpy, Node/Three.js, Blender4.5.9.

**Spec:** ../specs/2026-11-06-gesture-refinement-expansion-design.md

## Global Constraints

- Only mapped source joints12..21 and25..54; preserve endpoints, raw parameters, identities, native30FPS basis and unverified semantic/alignment states.
- Trace isolated repairs; regular non-spike corrections <=3° fingers/5° body.
- Original mode remains raw; refined rotation/FK/IK/export use the same versioned stream.
- New100-asset presentation has128-file/256MiB aggregate budget and32MiB per-file cap.
- Keep existing uncommitted changes and prior corpus; no commit/push, no language approval fabrication.

## Review Focus

- Real rapid rotations/long holds/wrap-around encodings must survive; frame endpoints and channels not driven by avatar remain untouched.
- Missing/malformed refined poses or unknown profile fail validation instead of silently falling back.
- Expansion retains all20 source IDs and unique texts, rejects incomplete sequences, resets approvals.
- Runtime and exported target trajectories must agree for every frame and exact source SHA.
- Resource/count budgets must reject oversized bundles before bulk reads/baking.

### Task1: Refinement and numerical audit

Files: slbridge/animation/gesture.py; tests/test_gesture_refinement.py.

- [x] Write RED tests for isolated1/2-frame impulses, retained steady fast rotations/long holds, angle wrap, endpoints/immutability, channels excluded, correction bounds.
- [x] Implement refine_gestures(poses,fps) -> (refined,audit,mask), retain physical/quaternion metrics.
- [x] Run GREEN and audit20 actual source sequences before expanding.

### Task2: Source selection in registry/runtime and bounded100-asset presentation

Files: registry.py, presentation.py, web/retarget.js, scripts/animation/export_tracks.mjs; Python/Node tests.

- [x] RED tests for raw/refined selection, malformed profile/pose rejection,100-pilot acceptance and aggregate memory budget.
- [x] Implement explicit profile selection shared by retarget and kinematics, compatible source/export binding.
- [x] Run focused regression checks; old profiles and original20 raw data unchanged.

### Task3: Expand, bake and verify100 sentences

Files: scripts/animation/build_soke_corpus.py; tests/test_soke_corpus.py; report/pitfalls.

- [x] Add seed-corpus/refine flags and aligned100 count limits, retain all20 IDs then add80 distinct complete sentences.
- [x] Build new corpus onG, bake100 assets, verify arrays/identities, all-frame exported positions and old20 refinement metrics.
- [x] Fresh final review; fix substantive findings with regression coverage.
- [x] Verify real browser playback/comparison/download, replace only own8876 service and report residual/NOT_RUN boundaries.
