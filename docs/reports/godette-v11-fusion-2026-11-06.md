# Godette v11 fusion checkpoint

Base: PR #1 commit `ad3806f8e0ba2227231492860cfed33eb70347f9`.

Status: integration in progress, not merged or language-approved. A verified 292-file source/test/config snapshot is available in the engineering workspace. The existing corpus, SignSparK generation pipeline and SHA-bound nonmanual gates will be preserved.

Current model: `godette-reference-v11.glb`, SHA256 `e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476`. Editable `.blend` and original delivery archive are retained. Binary backup is being checked separately; this note alone does not mean the model bytes are on GitHub.

Plan: create a separate source-bound presentation for v11, retain the six pilot bindings and temporal profiles, omit old-target baked exports, leave automatic source-jaw unavailable for the uncalibrated v11 target, and regression-test the narrow quaternion interpolation fix. No dataset, SMPL-X template or generation checkpoint will be committed.

Godette (Rigged) by zahlenmaler, CC BY 4.0. Source: https://sketchfab.com/3d-models/godette-rigged-dd05b69799a2438e97c90d166f6e416a . License: https://creativecommons.org/licenses/by/4.0/ . Modified hair, clothing, shoes, materials, clothing weights and trouser surface; retained original face, hands and 227-bone skeleton. Engineering checks do not establish sign-language accuracy, complete collision freedom or visual acceptance.
