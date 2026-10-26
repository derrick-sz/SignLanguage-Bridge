# Continuous Video Recognition Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. A fresh final reviewer checks the full change.

**Goal:** Update recognition during motion from a bounded live video window instead of waiting for a pause.

**Architecture:** Add a bounded rolling RGB window to RecognitionSession and schedule a single background clip inference from the freshest data. Reuse the activity observer for hand/quality gating and retain the saved model preprocessing. Present revisable window results separately from confirmed caption history.

**Tech Stack:** Existing Python, PySide6, MediaPipe, RF / ONNX CTC; pinned desktop dependencies in the existing GPU virtual environment.

**Spec:** ../specs/2026-10-22-continuous-video-recognition-design.md

## Global Constraints

- No pause required before window inference. Still report model mode as clip.
- Only one inference and no pending-window queue; preserve capture and preview.
- Window 1.2–2.4 s, update interval 0.5 s, stale-result limit 3.0 s; max 180 frames / 96 MiB / width 640.
- Actual quality gating, source/model/mode identities and bounded resource ownership remain mandatory.
- Window outputs are revisable and never appended as confirmed caption events.
- Preserve previous language-switch changes and unrelated dirty primary-checkout files.
- No weight changes, retraining, remote publication, device reset or forced user-process termination.

## Review Focus

- Busy/slow inference: freshness and bounded buffering without overlapping model work.
- No hands, partial observations and source gaps: no stale or guessed labels remain visible.
- Cancel, stop, EOF, pause, clear, model switch: no late candidate survives invalidation.
- Old automatic/manual mode, caption history and language selector: compatibility and honest UI state.
- GPU dependency/provider selection: preserve pinned extractor identity and report real CPU fallback.

## Task 1: Rolling window and session scheduling

**Files:** Create `slbridge/continuous.py`, `tests/test_continuous.py`, `tests/test_continuous_session.py`; modify `slbridge/session.py`.

**Interfaces:** `ContinuousConfig` holds the exact defaults above. `RollingWindow.append(packet)` owns bounded copies; `snapshot()` returns a complete explicit window with source times and capture receipt time. Session exposes `continuous`, `enable_continuous(config=None)`, `disable_continuous()`, `continuous_status()`; emits `continuous_result`, `continuous_clear`, `continuous_status` events. Old pause segmentation and continuous recognition are mutually exclusive.

- [x] Write/run RED tests for motion-only updates, bounded latest window, slow-model freshness, quality loss, invalidation, no caption appends and mode exclusivity.
- [x] Implement the window/scheduler through the existing model worker, request validation and captured_clip adapter. Run new and existing session tests until green.

## Task 2: Desktop mode, GPU launch and validation

**Files:** Modify `slbridge/desktop.py`, `desktop_cli.py`, `README.md`, `docs/auto-captions-usage.md`; create `tests/test_continuous_desktop.py`, `scripts/start_demo.ps1` and a verification report.

**Interfaces:** Main UI `continuous_check` toggles continuous mode, remains stoppable while inference runs, labels current results as revisable and clears them on invalidation. CLI `--continuous` requests startup when model/source are ready. `start_demo.ps1` chooses the usable bundled GPU environment before the CPU environment and passes parameters without string-built commands.

- [x] Write/run RED Qt tests for live updates without pause, clear/stop behavior and unchanged committed history. Implement the controls and launch integration.
- [x] Install pinned desktop requirements into `.venv-gpu`, record dependency check and active providers; do not modify `.venv-libras`.
- [x] Run full CPU regression and targeted GPU checks. Measure real-video continuous replay with both actual models and bounded latest-window behavior. Report camera and accuracy boundaries explicitly.
- [x] Commit, obtain a fresh read-only review, repair material findings with regression checks, then transfer only verified changes to the primary workspace after checking its existing contents against the prior delivery baseline.

## Execution ruling

The user's follow-up explicitly directs video-stream recognition after the diagnosis. Execute this concrete flow within that request; do not repeat a general permission question. Preserve formal caption semantics and disclose revisable rolling inference rather than claiming an untrained native stream model.
