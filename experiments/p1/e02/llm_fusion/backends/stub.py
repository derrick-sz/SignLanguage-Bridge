"""Deterministic stub backend for full-pipeline self-tests.

The stub is **not** a language model and its outputs must never be reported as
LLM results.  It exists so that prompt construction -> backend call -> JSONL ->
canonical evaluation can be exercised end-to-end on CPU with a deterministic,
auditable rule set:

* C : echo the Uni-Sign text with light normalization (sentence-final punctuation).
* D : C + deterministic numeric alignment (when Uni-Sign and the SLRT gloss each
      contain exactly one numeric value and they differ, replace the Uni-Sign
      number with the gloss number).
* E : D + deterministic question-marker alignment (if the gloss carries a
      question marker and the Uni-Sign text does not, end the sentence with "？").

A does not use any backend (direct baseline).
"""
from __future__ import annotations

import time
from typing import Any

from experiments.p1.eval import metrics as M

from experiments.p1.e02.llm_fusion.backends.base import BackendResult, LLMBackend

SENTENCE_ENDINGS = ("。", "？", "！", "?", "!")


def _normalized_sentence(text: str) -> str:
    normalized = M.normalize_text(text or "")
    if not normalized:
        return ""
    if not normalized.endswith(SENTENCE_ENDINGS):
        normalized += "。"
    return normalized


def _rule_slots(unisign_text: str, gloss_text: str):
    """Reuse E06 RuleJudge slot extraction (deterministic, pure Python)."""

    from experiments.p1.e06.build_e06 import extract_slots, numeric_patch, slot_conflicts

    return extract_slots(unisign_text), extract_slots(gloss_text), slot_conflicts, numeric_patch


class StubBackend(LLMBackend):
    name = "stub"
    version = "stub-v1-deterministic"
    is_stub = True
    is_real = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "is_stub": True,
            "is_real": False,
            "available": True,
            "rule_set": {
                "C": "echo normalized Uni-Sign text",
                "D": "C + single-number alignment to SLRT gloss when values conflict",
                "E": "D + question-marker alignment to SLRT gloss",
            },
            "warning": "STUB ONLY: outputs are deterministic rules, not LLM results",
        }

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
        started = time.perf_counter()
        evidence = evidence or {}
        unisign_text = str(evidence.get("unisign_text") or "")
        gloss_text = str(evidence.get("gloss_text") or "")
        if group == "A":
            raise ValueError("group A is a no-LLM baseline; the harness must not call a backend for A")
        text = _normalized_sentence(unisign_text)
        if group in ("D", "E") and text and gloss_text:
            a_slots, g_slots, _conflicts, numeric_patch = _rule_slots(unisign_text, gloss_text)
            a_numbers = sorted(a_slots["number_values"])
            g_numbers = sorted(g_slots["number_values"])
            if len(a_numbers) == 1 and len(g_numbers) == 1 and a_numbers != g_numbers:
                patched, applied = numeric_patch(text, g_numbers[0])
                if applied:
                    text = _normalized_sentence(patched)
            if group == "E" and g_slots["question"] and not a_slots["question"]:
                if text and not text.endswith(("？", "?")):
                    text = text[:-1] + "？" if text[-1] in "。！" else text + "？"
        latency_ms = (time.perf_counter() - started) * 1000.0
        return BackendResult(
            text=text,
            latency_ms=latency_ms,
            tokens_in=None,
            tokens_out=None,
            model_version=self.version,
            status="ok" if text else "error",
            error=None if text else "empty stub output",
            raw_event_count=None,
            extra={"group": group, "stub_rules": True},
        )
