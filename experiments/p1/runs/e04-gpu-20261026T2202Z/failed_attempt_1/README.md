# Failed attempt 1 (2026-10-26 21:56-21:58) -- evidence only, no numbers reused
Cause: (1) benchmark_p1.py ll passed phase-specific CLI flags to every subcommand (argparse exit 2 for all cold/warm/load).
(2) During the accidental serial phase, host RAM was exhausted (free ~4.5 GB; ORT logged 'fatal: Memory allocation failure') and the child died with CUDA error: unknown error in the model warm-up conv2d; the pwsh host itself exited with 'Out of memory'.
Action: killed 3 orphan python processes, fixed the argv construction, reran each phase individually in a fresh process.
