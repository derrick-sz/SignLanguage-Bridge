# Model runtime API and ONNX Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement each scoped task and review the integrated result.

**Goal:** Provide versioned model invocation, safe clip-boundary replacement and an actual ONNX CPU/CUDA runtime without breaking existing Libras RF clients.

**Architecture:** One transport-independent ModelManager owns active/candidate/previous instances and single-request capacity. Desktop and HTTP use this manager. Concrete backend adapters own their tensor bindings and persistent runtime sessions.

**Tech Stack:** Python 3.12, FastAPI 0.118.0, ORT 1.23.2, PySide6 6.8.3, existing sklearn 1.4.0.

**Spec:** ../../模型运行API与ONNX_设计与执行_2026-10-09.md

## Global Constraints

- Keep old RF manifests readable; never overwrite original model bundles or frozen splits.
- Classification/clip only; ONNX adapter is not evidence of a trained temporal backbone.
- Strict CUDA never silently switches the session to CPU; report actual providers and automatic fallback reasons.
- Limit concurrent inference to one; switching pauses new requests and preserves the active instance on failure.
- Multipart uploads have receive-time and per-file limits; no client-selected server paths/URLs.
- Optional API/ONNX dependencies must not be imported by unrelated core clients or CLI help.

## Review Focus

- Cancel/close while a candidate is warming up must release it without activation.
- Failure after loading must preserve active identity and generation.
- GPU provider availability alone cannot prove GPU computation; inspect node execution profile.
- Compressed NPZ/NPY headers must be bounded before array allocation.
- HTTP errors and schemas must agree with actual runtime responses, including rejected inputs.

## Task 1: Registry and ONNX adapter

Files: contracts.py, bundle.py, models.py, inference.py, backends.py, onnx_backend.py, export_onnx.py; test_onnx_backend.py.

Interfaces: Recognizer.load(path, options=RuntimeOptions), warmup(), close(), runtime_info; recognize_profiled/recognize_clip_profiled return (RecognitionResult, timing dict). Bind NC features or NTC sequence plus optional mask. Export new bundles with checksum and genuine held-out numerical parity gate.

- [x] Run existing RF tests, implement backward compatible dispatch and real ONNX tests.
- [x] Verify invalid shapes/probabilities, logits, masks, CPU and strict/automatic device policies.
- [x] Verify export on real frozen Libras held-out inputs and preserve threshold/labels.

## Task 2: Lifecycle and desktop boundary

Files: runtime_options.py, runtime_contracts.py, runtime.py, session.py, desktop.py, desktop_presenter.py; test_runtime.py/test_session.py/test_desktop.py.

Interfaces: InferenceRequest(request_id, representation_id, sign_language, schema_version=1); ModelManager.prepare/activate/discard, switch, rollback, infer_file/infer_clip, run, status, close.

- [x] Establish failing tests for absent manager, implement versioned leases and candidates, pass lifecycle tests.
- [x] Connect desktop to manager, prevent recording-time switching, expose previous-model recovery.
- [x] Verify pending-candidate invalidation, native task shutdown and UI metadata consistency.

## Task 3: HTTP transport and bounded input

Files: api.py, api_schemas.py, api_cli.py, input_limits.py, features.py; test_api.py/test_input_limits.py; dependency extras and configs/runtime-models.example.json.

Interfaces: create_app(catalog, initial_model, options, manager, max_upload_bytes); v1 health/models/switch/rollback/recognitions endpoints over manager. Sync native work runs outside event loop.

- [x] Validate OpenAPI, body and expanded-array limits, cleanup, busy/error mapping, server-side alias restrictions.
- [x] Verify real manager descriptors and errors, core-only CLI help and bearer token configuration.

## Task 4: Acceptance and release

Files: experiments/validate_model_runtime.py, usage guide and reports/model-runtime-api-2026-10-09.md.

- [x] Measure P50/P95 preprocess/inference/total separately using frozen cached test clips; compare exported probabilities and decisions.
- [x] Perform actual HTTP video inference/switch/rollback and CUDA node profiling with explicit process-local DLL discovery.
- [x] Run full pytest suite, pip check, package and installed CLI validation, independent code review, then sync Git.
