# Godette v12 model checkpoint

Frozen neutral-mouth character with revised trouser/ankle contours and solid shoe soles. The editable Blender project and runtime GLB are saved separately from v11. These files are a research candidate, not aesthetic or sign-language acceptance.

- `godette-reference-v12.blend.gz`: losslessly compressed editable Blender project.
- `godette-reference-v12.glb.gz`: losslessly compressed runtime GLB.
- `BACKUP.json`: compressed and decoded sizes/SHA256 values.
- `neutral_lip_preset.json`: reversible 16-lip-bone neutral/default-pose sidecar. Do not apply it again to this already-neutral model, or apply it as rest pose.
- `ATTRIBUTION.md`: Godette by zahlenmaler, CC BY 4.0, including derivative-work changes.

Restore with `gzip -dc godette-reference-v12.glb.gz > godette-reference-v12.glb` and the equivalent command for `.blend.gz`. Verify each decoded SHA256 against `BACKUP.json` before use.

The v12 SHA must never be labeled as v11 or the old automatically calibrated avatar. Runtime integration follows in a separate checkpoint on this branch. This model-only backup does not by itself activate v12 in the application. No existing v11 files, production archive/catalog, deployment or branch merge is changed.

This is the v12 stage checkpoint. The neutral expression has been retained for subsequent work; lower-body appearance continues in v13 based on the new reference. Frozen bytes mean reproducible version identity, not final visual acceptance.
