# Communication Uni-Sign Switch Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in the current session. The user has authorized the switch.

**Goal:** Persistently route completed Chinese video clips to the existing Uni-Sign base model and verify the live product gateway.

**Architecture:** Keep the two-process gateway architecture. Read model aliases, device and startup timeout from the local profile; stage the new recognition process at 8766, then replace only the owned 8765 listener.

**Tech Stack:** PowerShell launcher; existing Python v3 API, Uni-Sign adapter, CUDA/RTMPose and HTTP gateway.

**Spec:** ../specs/2026-11-09-communication-unisign-switch-design.md

## Global Constraints

- Preserve the existing prod base checkpoint and independent Libras route; do not enable S2 or LLM fusion by default.
- Preserve concurrent meeting/animation changes and the running 8879 process.
- Keep protocol negotiation, original text and manual confirmation; never inject expected text into inference.
- Back up local configuration; verify process identity before stopping it; keep NOT_RUN boundaries explicit.

## Review Focus

- A stale TFNet process must not be reused under a Uni-Sign profile.
- Heavy initialization must fit a configurable startup wait.
- Chinese readiness cannot be inferred from Libras readiness.
- Auto device selection must be checked against actual model and pose providers.
- Invalid-route profiles must leave unrelated/running services intact.

## Task 1: Configurable launcher and model switch

- [x] Back up launcher/local profile/catalog and capture running service identities.
- [x] Modify scripts/communication_product.ps1 to use recognition.csl_model, libras_model, device and startup_timeout_seconds; preserve legacy profile defaults; require Chinese readiness.
- [x] Update configs/communication-product.example.json and configs/communication-recognition.example.json with the base Uni-Sign route and compatible Python environment.
- [x] Add the base bundle to the local catalog; choose it explicitly in the local profile.
- [x] Attempt shadow 8766; on confirmed Windows commit exhaustion, retry sequentially after releasing only the owned idle TFNet process (resource ruling in the ledger); verify actual CUDA and real HTTP inference.
- [x] Replace only the idle, identity-matched 8765 service and verify gateway text, WebM, Libras, launcher reuse and wrong-route protection.
- [x] Run relevant existing adapter/default/gateway regressions; inspect the diff and record evidence and runtime limits.
- [x] Update product instructions and model-evaluation pitfalls; report the realtime SLRT recommendation separately from the completed-clip switch.
