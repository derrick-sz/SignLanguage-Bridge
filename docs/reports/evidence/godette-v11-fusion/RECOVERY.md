# Cloud QA recovery, 2026-11-07

The initial recovered workspace lacked tracked repository files and optional test dependencies. The unchanged base and fusion candidate both reported 111 failures and 45 errors. Their identical first-blocker classification was:

| First blocker | Failures | Errors |
| --- | ---: | ---: |
| OpenCV import, including asynchronous callers | 101 | 2 |
| PySide6 import | 4 | 8 |
| MediaPipe import | 0 | 11 |
| API contract YAML | 1 | 20 |
| Experiment package | 0 | 2 |
| Frozen protocol JSON | 0 | 2 |
| Adapter example document | 1 | 0 |
| Test dependency on ignored developer config | 4 | 0 |
| Total | 111 | 45 |

The first 152 blockers were resolved by restoring upstream files and installing supported test dependencies, without weakening assertions or production gates. The remaining four were an actual test-isolation issue: the desktop model-default tests expected the developer's ignored `configs/model_defaults.json`. A temporary copy of `configs/model_defaults.example.json` now exercises the real loader. Missing-config fallback behavior and the explicit skip for a developer's real local configuration remain intact.

## Reproduction

Use Python 3.12 and a clean checkout of the documentation checkpoint that contains this file on `codex/godette-v11-fusion-20261106`. The tested code commit is `491704e1ff0adf02d58cd965823ed92f94b22d76`; that earlier commit alone does not contain these new QA requirements and evidence files. The documentation checkpoint adds no runtime or test changes. These are optional Linux cloud QA dependencies, not new production requirements. The run used Python 3.12.14, Node v24.22.1 and Qt's offscreen platform.

```sh
QA="$(mktemp -d)"
EVIDENCE=docs/reports/evidence/godette-v11-fusion
python -m pip install --target "$QA/python" \
  -r "$EVIDENCE/qa-requirements.txt" -c "$EVIDENCE/qa-constraints.txt"
PYTHONPATH="$QA/python" QT_QPA_PLATFORM=offscreen \
  XDG_CACHE_HOME="$QA/cache" MPLCONFIGDIR="$QA/matplotlib" \
  python -m pytest -q -rs --junitxml="$QA/pytest.xml"

npm install --prefix "$QA/node" --ignore-scripts --no-audit --no-fund three@0.170.0
gzip -dc assets/godette-v11/godette-reference-v11.glb.gz > "$QA/godette-v11.glb"
sha256sum "$QA/godette-v11.glb"
ANIMATION_THREE_MODULE="$QA/node/node_modules/three/build/three.module.js" \
  ANIMATION_V11_AVATAR_GLB="$QA/godette-v11.glb" \
  node --test tests/animation*.test.mjs
```

The decoded GLB must have SHA256 `e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476`. The old calibrated avatar is a different asset and is not replaced by v11. To run its optional test, supply its exact verified bytes through `ANIMATION_AVATAR_GLB` separately; never relabel the v11 hash or disable the identity checks.

The historical initial workspace was a partial tracked-file snapshot. The recovery restored another 89 upstream files, all listed with their original Git blob hashes in `restored-files.json`. These files were already in GitHub and were not code changes. No production corpus, training data, pretrained weights or developer configuration was synthesized or uploaded.

## Results and limits

The final Python suite was run twice, yielding 1322 passed / 42 skipped / 0 failed / 0 errors / 3 passed subtests both times. The 17 deprecation warnings include two project-test parameterization warnings (passing `enumerate` into pytest marks), plus Starlette/AnyIO and Matplotlib/Pyparsing warnings. These are not runtime failures, but the project-test warnings will require adaptation before pytest 10. JUnit counts the three subtests separately, reporting 1367 tests; there are 1364 top-level testcase elements (1322 passed + 42 skipped). The skip inventory and log hashes are recorded in `recovery-verification.json`.

Node yielded 50 passed / 1 skipped / 0 failed. The v11 real GLB and actual Three.js ran, but the old calibrated avatar was absent. The skipped Python entries comprise Torch (22, including two collection-level entries), external/pinned assets or explicit real-asset opt-in (18), ignored local configuration (1), and Windows-only behavior (1). These are not passes.

GPU generation, the actual 19,877-record corpus and production motion archive/catalog, real saved-model acceptance, browser/WebGL rendering, contact/collision, sign-language quality and mobile performance remain unverified. The branch has not been merged or deployed. A passing cloud regression suite does not establish those outcomes.

The production `requirements-libras.txt` resolver dry-run succeeded; it was not modified. QA pins here reproduce this isolated environment and must not be substituted for a production dependency audit.
