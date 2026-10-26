"""Backend abstraction for the G2 phase-2A LLM fusion ablation.

A backend only receives the already-built prompt messages (plus the evidence
dict used by deterministic stub backends).  Backends never see ground truth:
`harness.py` builds prompts exclusively from `experiments/p1/e02/candidate_outputs.jsonl`
and whitelisted SLRT score columns.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendResult:
    """One backend call result (GT-free by construction)."""

    text: str
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    model_version: str | None = None
    status: str = "ok"
    error: str | None = None
    raw_event_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "latency_ms": round(self.latency_ms, 3),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model_version": self.model_version,
            "status": self.status,
            "error": self.error,
            "raw_event_count": self.raw_event_count,
            "extra": self.extra,
        }


class BackendDisabled(RuntimeError):
    """Raised when a real backend is selected without explicit enablement."""


class BackendNotImplemented(NotImplementedError):
    """Raised by interface-only backends (e.g. openai_compat in this round)."""


class LLMBackend(abc.ABC):
    """Minimal LLM backend interface."""

    name: str = "base"
    version: str = "0"
    is_stub: bool = False
    is_real: bool = False

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        group: str,
        evidence: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
    ) -> BackendResult:
        """Return one generated sentence; must raise on failure rather than return junk."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "is_stub": self.is_stub,
            "is_real": self.is_real,
            "available": False,
        }
