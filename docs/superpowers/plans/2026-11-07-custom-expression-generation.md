# Custom Expression Generation Implementation Plan

> **For agentic workers:** Inline execution with delegated independent asset/source audit, TDD for runtime/conversion code and fresh final review. Continuous work authorized; no training. The user's later explicit GitHub-sync request authorizes commits/push and related documentation for the current changes.

**Goal:** Run pretrained CSL-capable generation on genuinely new Chinese text and bind resulting motion to the existing avatar preview.

**Architecture:** Verified official assets, isolated GPU worker, batch1 sequential flow streams, explicit source-free conditioning, reviewed pose adapter and candidate output. Existing retrieval remains separate.

**Tech Stack:** Existing Python3.10/Torch cu128 GPU foundation; SignSparK pretrained flows; numpy/Three.js bridge.

**Spec:** ../specs/2026-11-07-custom-expression-generation-design.md

## Global Constraints

- No training, no reverse use of recognition weights, no GT target poses as novel generation.
- Preserve prior libraries/services and local recognition environment; model download manifests bind revision/size/SHA.
- Explicit candidate provenance, bounded sequence/batch/VRAM, single worker, no fabricated language approval.

## Review Focus

- Missing/wrong weights must fail rather than run a random initialized model under strict=False.
- Masked conditioning must not leak target motion/length into custom expression.
- Source rotations, means, handedness and normalisation must be proved, not guessed.
- Unknown vocabulary, excessive duration and cancelled/failed generation must retain clear status.
- Native Windows inference must not require unsupported NCCL distributed setup.

### Task1: Acquire reproducible assets and isolated runtime

- [x] Pin official source/model revisions; fetch pretrained hand/body/face with size+LFS SHA verification and resumability.

后续状态：三个流均已全文件SHA校验通过，总16,742,608,923B；下载进程正常退出。实际两条文本推理已启动，完成前仍不勾选Task2真实生成和Task3播放验收。
- [x] Isolated runtime dependency/import/CUDA checks; preserve existing GPU and CPU environments.

### Task2: New-text generation and pose adapter

- [x] RED/GREEN tests for zero-target conditioning, bounded new text/frames, strict checkpoints and proven6D/hand conventions.
- [x] Single-process sequential-stream generation for two novel Chinese sentences, raw outputs and timings/VRAM recorded.
- [x] Convert proven output schema to55-joint candidate with original/provenance preserved; no static replay substitution.

### Task3: Preview, review and next-stage retrieval enhancement

- [x] Real avatar playback and source/raw comparison for a generated candidate.
- [x] Fresh independent review and appropriate regression checks; report exact successful/NOT_RUN boundaries. Final pre-sync checks: Python1291/38skip; Node38/0skip; isolated-runtime66/0skip, plus actual GPU pilot and browser evidence in R36.
- [ ] Inspect official trained Text2Gloss/train dictionary enhancement; block unknown gloss mapping/drop before reuse. Source/asset audit completed in `artifacts/animation-upgrade/custom-expression/retrieval-readiness-audit.md`; strict runtime OOV gate remains pending.

2026-11-07实际结果：三模型总16.74GB完成认证；run01身体/手完成、face搬运OOM，保留failed。参数meta/CPU缓冲区修订后run02三流各859严格张量、两句共256帧完成，身体遮罩负控差0，六份可比较原始流逐位相同。仅修正跳变提示的preview-v2原始NPZ不变，产物审计PASS、两句实际人物/嘴部/同帧对照及review导出完成；新嘴部GLB未支持。发布审查修复可迁移认证清单与损坏dtype隔离；完整同步回归结果补记于[R36](../../reports/custom-expression-generation-2026-11-07.md)。CLI工程链路已跑通，浏览器任意文本生成、检索锚点/OOV gate及语言质量继续待完成，持续目标不在此收窄。
