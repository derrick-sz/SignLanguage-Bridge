# SOKE CSL-Daily Corpus Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline; one fresh review at the end. Prior proposal accepted by user's “尝试下载并进行构建”.

**Goal:** Download the source release, inventory real 3D coverage and produce 20 traceable sentence candidates, numerical arrays and playable assets where template compatibility is confirmed.

**Architecture:** Keep the ZIP, stream its directory and selectively read members. Parse pickle as inert opcodes, validate numeric payloads, retain full parameters and export a separate upper-body playback bundle through the existing rig pipeline.

**Tech Stack:** Python 3.12/numpy/stdlib, existing Three.js runtime and Blender 4.5.9.

**Spec:** ../specs/2026-11-06-soke-corpus-build-design.md

## Global Constraints

- Existing codex/csl-animation-workspace checkout; preserve previous changes and services, no commit/push/publish.
- No full small-file extraction, model training, recognition configuration changes, external pickle execution or invented approvals.
- Authoritative source: public SOKE CSL-Daily-Fittings links, release indices and producer parameter interpretation.
- Raw source parameters and frame identity remain available; any canonical playback conversion is separately labeled.
- Human language/motion review stays unverified; source/license/template identities stay distinct.

## Review Focus

- Missing/duplicate/out-of-order frames must not be silently concatenated or filled.
- Numeric pickle decoding must not invoke GLOBAL/REDUCE or accept object arrays.
- Hand mean must be applied exactly once; raw and effective poses must remain distinguishable.
- Index counts must not be treated as confirmed complete 3D sequences or semantic approvals.
- Preview/export source SHA, sample ID, fps and frame count must agree.

### Task 1: Download source release

- [x] Download with bounded/resumable size verification to G, preserve partials on failure.
- [x] Record local SHA and expected bytes; retain upstream checksum status explicitly.

### Task 2: Safe parsing and archive inventory

**Files:** Create slbridge/animation/soke_pickle.py, slbridge/animation/soke.py; tests/test_soke_corpus.py.

**Interfaces:** parse_pickle(raw) -> inert containers / numeric arrays; scan_archive(path, selected_ids) -> sequence inventory and selected member metadata; read_member(path, entry) -> CRC-checked bytes.

- [x] Write failing tests: numpy numeric roundtrip, inert malicious GLOBAL/REDUCE, object-array rejection, unordered frame metadata/duplicates, CRC mismatch, hand mean mapping.
- [x] Implement static decoder, streaming ZIP64 directory reader, strict sequence and numeric validation.
- [x] Run targeted tests; scan the actual source archive and compare all IDs/frame ranges to released indices.

### Task 3: Build candidate sentence corpus

**Files:** Create scripts/animation/build_soke_corpus.py; extend tests/test_soke_corpus.py only for integration boundaries.

**Interfaces:** CLI consumes raw archive, index dir, local video root, base motion archive, model hand mean asset and a new output directory; emits inventory.json, corpus.jsonl, per-sequence NPZ, candidate-motion.zip and source-bound metadata.

- [x] Select 20 unique texts with local filename matches and 10 signers; reject incomplete parameter sequences. Actual originals are uncropped: retain explicit unverified video binding instead of claiming frame alignment (ledger ruling).
- [x] Confirm authoritative neutral hand mean; save its SHA and apply once for effective 55-joint poses.
- [x] Build independent MotionRegistry-compatible ZIP with unverified statuses and all per-file hashes; validate using the real registry.

### Task 4: Play/export and verify

**Files:** Reuse scripts/animation/export_tracks.mjs and build_presentation.py helpers; docs/reports/soke-corpus-build-2026-11-06.md and project pitfalls.

- [x] Bake candidates using the existing conservative retargeter and real avatar; bind exports to exact source IDs/hashes.
- [x] Verify exported frame/position agreement; start an isolated local workspace and inspect actual playback/downloads.
- [x] Run appropriate Python/Node regression checks, fresh final review and documentation checks.
- [x] Report downloaded/inventoried/converted/playable counts separately from NOT_RUN language acceptance.
