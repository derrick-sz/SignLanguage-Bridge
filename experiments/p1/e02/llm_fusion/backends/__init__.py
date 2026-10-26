"""Backend registry for the G2 phase-2A LLM fusion ablation."""

from __future__ import annotations

from experiments.p1.e02.llm_fusion.backends.base import (
    BackendDisabled,
    BackendNotImplemented,
    BackendResult,
    LLMBackend,
)
from experiments.p1.e02.llm_fusion.backends.stub import StubBackend

__all__ = [
    "BackendDisabled",
    "BackendNotImplemented",
    "BackendResult",
    "LLMBackend",
    "StubBackend",
    "get_backend",
    "list_backends",
]


def get_backend(name: str = "stub", **kwargs) -> LLMBackend:
    if name == "stub":
        return StubBackend()
    if name == "codex_cli":
        from experiments.p1.e02.llm_fusion.backends.codex_cli import CodexCLIBackend

        return CodexCLIBackend(**kwargs)
    if name == "openai_compat":
        from experiments.p1.e02.llm_fusion.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r} (expected one of stub, codex_cli, openai_compat)")


def list_backends() -> list[dict]:
    backends = [StubBackend()]
    from experiments.p1.e02.llm_fusion.backends.codex_cli import CodexCLIBackend
    from experiments.p1.e02.llm_fusion.backends.openai_compat import OpenAICompatBackend

    backends.extend([CodexCLIBackend(), OpenAICompatBackend()])
    return [backend.describe() for backend in backends]
