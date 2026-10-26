# Desktop Language Switch Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this approved plan inline. Use a fresh reviewer after implementation.

**Goal:** Let a desktop user switch between CSL and Libras on the main screen while camera preview continues.

**Architecture:** Add a small default-model-location helper and a language selector to the existing Qt source toolbar. Keep RecognitionSession and ModelManager as the single loading and activation path; synchronize the selector from the active model, including custom settings and rollback.

**Tech Stack:** Existing Python 3.12, PySide6, pytest, RF / ONNX CTC model packages.

**Spec:** ../specs/2026-10-22-desktop-language-switch-design.md

## Global Constraints

- Preserve camera instance and preview during model loading and on load failure.
- Never show the target language as active until the model transaction commits.
- Block switching while recording, recognizing, loading, or waiting for cancelled native work.
- Keep caption history; clear the current result and stop automatic segmentation on a switch.
- No automatic language detection, weight changes, retraining, new dependencies, push or publication.

## Review Focus

- Missing or damaged models: readable unavailable status and a usable original model after failure.
- Pending and cancelled loading: actual selection stays authoritative; no late candidate activation.
- Settings and rollback: path/language synchronization, including custom and unknown-language models.
- Camera, automatic mode and caption history: no source restart, no cross-model pending action or stale caption.
- Minimum window size: selector, input actions, preview and captions remain visible and nonoverlapping.

## Task 1: Main-screen language switching

**Files:**
- Create `slbridge/desktop_models.py`: default language-to-model locations, shared with the CLI default.
- Modify `slbridge/desktop.py`: selector, loading feedback, active-model synchronization and model-specific error handling.
- Modify `slbridge/desktop_cli.py`: reuse the model-location helper.
- Create `tests/test_desktop_language_switch.py`: user interaction and lifecycle checks.
- Modify `docs/auto-captions-usage.md` and `README.md`: explain switching and its limits.

**Interfaces:**
- `default_language_models(root: Path | None = None) -> dict[str, Path]` returns CSL/Libras candidate paths without loading weights.
- `SignLanguageApp(..., language_models: dict[str, Path] | None = None)` accepts optional candidate paths; existing callers remain compatible.
- `language_combo` is a visible QComboBox with stable data values `csl`, `libras` and a placeholder for unknown / unloaded models.
- Existing `load_model`, `rollback_model`, session events and active runtime metadata remain authoritative.

- [x] Step 1: Write tests for the missing main-screen selector, successful async switching with preview/history preservation, failure recovery, disabled missing options, settings/rollback sync, cancellation, blocked busy/recording states and minimum-size layout. Run the new tests and observe assertion failures for the missing feature.
- [x] Step 2: Implement the helper and selector, reusing the existing model transaction. Show target loading feedback separately from active selection; clear pending state on model success/error/cancel. Do not reset preview for a model-loading error. Run the new tests until green.
- [x] Step 3: Run the full pytest suite using the existing CPU/desktop environment. Verify current source import path and distinguish dependency skips.
- [x] Step 4: Exercise real RF → CSL → RF with Qt controls, real labelled files and camera preview where available; retain a compact screenshot and JSON evidence under ignored artifacts. Update usage docs and record results in the ledger.
- [x] Step 5: Commit implementation, request fresh review against baseline `cf291c2a81558faf6de0dc5a7243e730531e4367`, repair material findings with regression checks and verify the final state. Leave local work reviewable without publishing.

## Execution ruling

The user explicitly chose the recommended implementation scope in this chat. Proceed inline without another approval question. The managed worktree isolates implementation from the user's running Demo; the existing media virtual environment is reused without reinstalling packages.
