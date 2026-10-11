"""Device-neutral entry point; the original CPU benchmark command stays valid."""
from __future__ import annotations

from .benchmark_ctc_cpu import main


if __name__ == "__main__":
    main()
