# Godette v13 six-source-motion checkpoint

The source-input blocker is resolved. Frozen GLB and Blender hashes match the v13 backup. Six clips / 1,138 source frames pass the bounded CPU audit after the equal-rest interpolation fix: 27,255 pose samples, 6,186,885 bone-world-matrix checks, zero unanimated/default drift. The selected frontend suite has 46 tests: 45 pass, 0 fail, 1 skip (v12 exact GLB unavailable).

- [Detailed scope, findings and limits](../../../../docs/reports/godette-v13-six-motion-audit-2026-11-07.md)
- [Strict six-motion machine-readable result](six-motion-strict-candidate-run4.json)
- [Source handoff identity checks](handoff-verification.json)
- [Model upload/decompression identities](uploaded-model-verification.json)
- [Selected frontend regression](FRONTEND-REGRESSION.md)
- [Complete audit source and evidence ZIP](godette-v13-six-motion-evidence-20261107.zip)

ZIP SHA256: 2621709b9057cbfdba9721389bcaa73b1c163929ef9c9b0550284ca4781d2658 (221,226 bytes). The ZIP excludes model files, motion sources and dependencies. It includes original pinned runtime, exact candidate, reproduction scripts, manifest, strict and diagnostic evidence, and regression logs. Unzip to recover `v13_six_motion_audit/`; its README explains input placement and commands.

The result JSON calls the tested rig a local candidate because that was its state at execution. Its SHA256 c886ff0441af0873e113ab4934b75b77dc8449af2a112ac2b2ea55c7681c1505 matches the rig.js promoted unchanged by this checkpoint. Pinned commit cc11b4d6 refers to the original code, not the fixed commit.

No frozen model, original motion or weight was changed. No automatic jaw calibration, language/quality approval, branch merge or deployment was added. Browser/WebGL, source video, semantic/human signing, arbitrary clothing collisions, full source archive and MotionRegistry→CLI→server remain outside this test. Lower-body defaults are intentionally static in this upper-body retargeter. Earlier `six-clip-availability.json` and export reports remain historical pre-recovery evidence.
