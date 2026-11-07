# Frozen Godette v11 model backup

- `godette-reference-v11.blend` is the editable Blender source.
- `godette-reference-v11.glb.gz` is the exact frozen runtime GLB, losslessly gzip-compressed. Decompress it before using the presentation builder. Its SHA256 must be `e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476`.
- `BACKUP.json` records sizes and hashes. Compression/decompression was checked byte-for-byte. Compression changes storage only, not model data.
- Direct GLB upload exceeded the 16 MiB transport-request limit; this gzip backup and the separately delivered full model ZIP preserve the original bytes.

For example, `gzip -dk godette-reference-v11.glb.gz` produces the GLB while retaining the compressed backup. Keep ATTRIBUTION.md with any derivative or redistribution. The model is CC BY 4.0, not CC0.

This backup is not a claim of language accuracy, collision freedom, browser performance or completed deployment. Runtime integration instructions are in docs/godette-v11-presentation.md.
