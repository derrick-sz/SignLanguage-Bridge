Attempt 2 evidence: cold subprocess crashed before writing payload (missing rows), and 2/3 uni-sign video cold spawns died with
`CUDA error: out of memory` inside the model warm-up although >13 GB VRAM was free (host commit pressure: committed 55.7/65.2 GB).
Rows were removed from perf_trace.jsonl and the phases rerun with explicit retry-after-allocation-failure logic (spawn_attempts recorded).
