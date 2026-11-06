# CSL animation workspace implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Integrate existing motion assets into a local animation workspace with improved presentation, controlled playback and explicit confirmation of recognition inputs.

**Architecture:** A validated, server-owned ZIP registry supplies catalog metadata and immutable motion/renderer resources. A deterministic full-phrase planner feeds a Three.js stage through a same-origin FastAPI extension. Existing recognition APIs and evidence stay unchanged; a desktop entry hands off explicitly selected text without automatically playing it.

**Tech Stack:** Python standard library, existing FastAPI, Three.js ES modules, vanilla JavaScript/CSS, existing PySide6.

**Spec:** `docs/research/csl-animation-integration-and-visual-upgrade-design-2026-10-28.md` (four-item first phase approved by user on 2026-10-28).

## Global constraints

- Reuse local assets; no SOKE retraining, model-default changes, G-drive moves, frozen experiment edits, or changes to the separate SignBridge repository.
- Keep language approval separate from motion review; existing assets are research previews. No silent partial translation or invented sign IDs.
- Same-origin, loopback service; clients never choose filesystem paths.
- Preserve source poses; any rendering/retarget correction is explicit, reversible and versioned.
- Stop/edit/source changes invalidate older asynchronous playback operations.

## Review focus

- Negation or numbers outside a known phrase must block planning, not disappear.
- A modified ZIP or member must fail validation before serving motion.
- Pause/scrub/speed and repeated identical play requests must retain correct time and identity.
- A stopped/edited request must not restart after fetch/compilation completes.
- Recognition results and desktop handoffs remain unconfirmed until an explicit user action.

## Task 1: Validated registry and planner

**Files:** `slbridge/animation/registry.py`, `planner.py`, `__init__.py`; `tests/test_animation_registry.py`.

**Interfaces:** `MotionRegistry(archive, expected_sha256=None)`, `.catalog()`, `.motion(asset_id)`, `.resource(name)`; `plan_motion(registry, text, *, asset_id=None, confirmed=False, origin=None)`.

- [x] Write tests for full-phrase matching, ambiguity, unknown suffix/negation, explicit preview, read-only source data, member tampering and unsafe paths.
- [x] Run tests and observe missing implementation (RED).
- [x] Implement strict archive/member validation and small deterministic planner.
- [x] Run the complete task tests (GREEN) and commit.

## Task 2: Same-origin API and launch command

**Files:** `slbridge/animation/api.py`, `cli.py`, `slbridge/api.py`, `pyproject.toml`; `tests/test_animation_api.py`.

**Interfaces:** optional `animation_registry` on `create_app`; `/animation/`, `/v1/animation/assets`, `/v1/animation/plans`, `/v1/animation/motions/{id}`, fixed renderer resources. CLI `python -m slbridge.animation.cli --archive PATH [--catalog PATH]`.

- [x] Write transport tests for confirmation, blocked content, resource/path boundaries, absent optional registry and unchanged original APIs.
- [x] Observe RED, implement routes and CLI, observe GREEN.
- [x] Verify actual archive through the new API without loading a recognition model.
- [x] Commit.

## Task 3: Player, visual workspace and quality comparison

**Files:** `slbridge/animation/web/*`, MIT-attributed minimal retarget helpers; `tests/animation_player.test.mjs`.

**Interfaces:** pure `PlaybackClock` and `PlaybackGeneration`; renderer `loadMotion/play/pause/stop/seek/setSpeed/setView`.

- [x] Write Node tests for clock, seek/speed, replay and stale generation cancellation; observe RED.
- [x] Implement workspace, complete controls, catalog, input confirmation, original/improved view and immutable motion pipeline.
- [x] Keep full signing space visible, improve hand contrast/lighting, provide close-up and source quality labels.
- [x] Verify actual old/new screenshots and browser interactions; no semantic quality claim.
- [x] Commit.

## Task 4: Recognition/desktop handoff and acceptance

**Files:** `slbridge/animation/handoff.py`, `slbridge/desktop.py`, web recognition UI, `tests/test_animation_handoff.py`, user guide and execution report.

**Interfaces:** safe localhost handoff URL fragment carrying raw text/source metadata, never autoplay; browser uses existing model switching and `/v3/recognitions`.

- [x] Write tests for unsupported/rejected sources, explicit desktop action and metadata preservation; observe RED.
- [x] Implement desktop entry and recognition confirmation flow.
- [x] Run existing API/desktop regressions plus complete feature tests.
- [x] Verify a real local recognition request when runtime dependencies permit; report unmatched output as blocked rather than successful animation translation.
- [x] Obtain independent final review, fix material findings with regression tests, document remaining language/Blender limits and commit.

## Execution rulings

- Native execution in this chat; the user already approved the concrete design and requested development, so there is no second permission round for implementation.
- Use the existing absolute Python environments with the isolated checkout as cwd; verify imported source root.
- Shared model/ZIP files remain in their original locations and are not copied into Git.

## Completion evidence

Implementation at c8aac293c60e09b118c586d34f30d31e1e2c3b3c (following review fixes at 9d44f19). All four implementation tasks complete within the approved engineering scope. See [acceptance report](../../reports/csl-animation-workspace-acceptance-2026-10-28.md) for 1067 Python passes, 25 skips, 10 Node passes, real Uni-Sign/video UI proof and remaining language/Blender boundaries. Kept local branch and managed worktree; no remote push or merge.
