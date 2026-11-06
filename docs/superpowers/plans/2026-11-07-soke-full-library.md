# SOKE Full Library Implementation Plan

> **For agentic workers:** Use executing-plans with bounded delegated consumer work; TDD and fresh final review. User authorized continuous progress; retain uncommitted work and evidence, no commit/push.

**Goal:** Fully ingest the downloaded SOKE parameter corpus and expose bounded searchable per-sentence avatar playback, then continue quality/dictionary preparation.

**Architecture:** Stream ZIP directory member offsets into SQLite, convert one complete sentence at a time to atomic NPZ with bound source digests, and expose a read-only paginated library beside the existing100 assets.

**Tech Stack:** Python/numpy/sqlite3/FastAPI; existing Three.js avatar.

**Spec:** ../specs/2026-11-07-soke-full-library-design.md

## Global Constraints

- Source/template/index SHA and processing version bound on resume; immutable source/old100 retained.
- No implicit missing-frame repair; numeric candidates remain unverified, no model evaluation/training claim.
- Single writer; atomic NPZ; bounded members/poses/search; raw fields retained; no bulk ZIP extraction.
- Existing100 GLBs remain available; full library streams parameters without mandatory GLB baking.

## Review Focus

- Interrupted builds, missing/corrupt outputs and mismatched source/protocol must not silently skip.
- Unsupported/CRC-invalid members must fail only their sequence and remain traceable, not appear playable.
- Search/plan/motion routes must stay bounded and retain identity/approval separation across old and new assets.
- Concurrent writer/reader requests must not load the full corpus or trust disk paths from requests.
- Pagination/search races must not change the user's selected motion or play a stale request.

### Task1: Bounded archive index and resumable full conversion

Files: soke.py; new full_corpus.py; scripts/animation/build_soke_library.py; tests/test_soke_full_corpus.py.

- [x] RED: real ZIP fixtures cover complete/gapped/missing sequences, source mismatch, corrupt resume, CRC failure and immutable raw fields.
- [x] GREEN: callback archive scan, shared stream reader, SQLite build and single-writer atomic per-sequence conversion with progress.
- [x] Verify focused tests and start a full-source run onG; inspect live progress and resource use.

### Task2: Searchable library and lazy playback

Files: new corpus_store.py; api.py; cli.py; web/workspace.js/index.html; tests/test_animation_corpus.py and Node tests as needed.

Interfaces consume spec SQLite/NPZ schema; preserve MotionRegistry old routes/planner semantics. Exact new route names settled by consumer implementer and documented.

- [x] RED: real SQLite-backed paging/text/Gloss/filters, bounded motion load, hash/path/ID rejection and unreviewed plans.
- [x] GREEN: optional --corpus, query API and single-item dispatch, paginated UI keeping100 production pilots.
- [x] Verify APIs and real browser outside the original100, cancellation and same-frame raw/refined behavior.

### Task3: Full conversion acceptance and ongoing quality work

- [x] Audit all indexed states, converted arrays/hashes, residual quality distribution and text/Gloss coverage; no false whole-corpus pass.
- [x] Fresh final review, resolve substantive findings with regression evidence and complete relevant Python/Node checks.
- [x] Record current service/run handles, report, pitfalls and next priorities: alignment, non-manual expression, source-guided repairs, word-level review.
