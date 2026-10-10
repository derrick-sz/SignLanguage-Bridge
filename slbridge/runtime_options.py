"""Deployment choices, kept separate from the trained model's input contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeOptions:
    device: str = "auto"
    device_id: int = 0
    intra_op_threads: int = 4
    gpu_mem_limit_mb: int = 2048
    cuda_dll_dirs: tuple[str, ...] = ()
    cpu_conv_subnormals: str = "flush_to_zero"

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if not isinstance(self.cpu_conv_subnormals, str) or self.cpu_conv_subnormals not in {"flush_to_zero", "preserve"}:
            raise ValueError("cpu_conv_subnormals must be flush_to_zero or preserve")
        for name, minimum, maximum in (("device_id", 0, 63), ("intra_op_threads", 1, 64),
                                       ("gpu_mem_limit_mb", 64, 1048576)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if (not isinstance(self.cuda_dll_dirs, tuple) or len(self.cuda_dll_dirs) > 4
                or any(not isinstance(path, str) or not Path(path).is_absolute()
                       for path in self.cuda_dll_dirs)):
            raise ValueError("cuda_dll_dirs must contain at most four absolute directory paths")

    def to_dict(self) -> dict:
        return asdict(self)
