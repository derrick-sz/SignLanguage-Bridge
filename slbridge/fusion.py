"""FusionController interface placeholder (P0-A phase A5, task T-A5-1).

**G2 verdict: FAIL -- fusion does not enter the main path.**  The decision
record is ``docs/experiments/P-1/G2_DECISION_MEMO.md`` (2026-10-27): on the
50-sample exploratory batch every LLM arm (C/D/E, prompt v1 and v2) scored
below the single-model baseline A (C-A: -2.28 BLEU with CI excluding zero on
v2); the selective cascade cut calls to 28% but with a point estimate below A
and no significant difference.  This module therefore defines the interface
**shape only** -- no implementation, and (enforced by a repository-grep test)
**zero wiring**: no production module imports or calls it.

Restart conditions (G2 memo sec. 5) before any implementation may land:
1. a **new independent test set** (unseen signers/environments, GT-free inputs);
2. prompt tuning on a **development set** (never on the evaluation batch);
3. **human review** of the six key-fact slots showing fixes > regressions;
4. an explicit **cost ceiling** (tokens / latency budget);
5. demonstrated quality gain whose **confidence interval excludes zero**.

Interface contract (frozen here, mirrors the API contract ``DecisionIntent``):

* input: the evidence candidates of **one utterance** plus conflict features
  derived from them -- never ground truth, never GT-derived scores (WER/BLEU/
  edit similarity live only on the evaluation side, ``slbridge.evidence``'s
  GT guard applies to every payload that crosses this boundary);
* output: a :class:`FusionDecision` (action ACCEPT/FUSE/RETRY/UNCERTAIN,
  origin rules/jev/timeout_fallback, policy version, reason codes).

The placeholder exists so a future G2-restart implementation can slot in
without reshaping call sites; ADR-P1-005 keeps Jev/LLM out of the main path
in the meantime and the single-model route (Uni-Sign) remains the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .evidence import EvidenceCandidate

__all__ = ["DecisionIntent", "ConflictFeatures", "FusionDecision",
           "FusionController", "CONFLICT_FEATURE_FIELDS", "RESTART_CONDITIONS"]

#: Restart conditions copied verbatim-in-spirit from G2_DECISION_MEMO sec. 5;
#: ``tests/test_fusion_contract.py`` asserts the key phrases stay present.
RESTART_CONDITIONS: tuple[str, ...] = (
    "a new independent test set (unseen signers, GT-free inputs)",
    "prompt tuning on a development set, never the evaluation batch",
    "human review of six key-fact slots with fixes > regressions",
    "an explicit cost ceiling for tokens and latency",
    "a demonstrated quality gain whose confidence interval excludes zero",
)

#: The conflict-feature vocabulary an implementation may consume.  Deliberately
#: narrow: observable structural signals only, no GT-derived values.
CONFLICT_FEATURE_FIELDS: tuple[str, ...] = (
    "same_utterance", "time_overlap", "slot_conflicts", "missing_key_slots",
    "single_candidate", "deadline_elapsed",
)


class DecisionIntent:
    """The contract DecisionIntent enums (V1 API contract, frozen values)."""

    ACTIONS: tuple[str, ...] = ("ACCEPT", "FUSE", "RETRY", "UNCERTAIN")
    ORIGINS: tuple[str, ...] = ("rules", "jev", "timeout_fallback")


@dataclass(frozen=True)
class ConflictFeatures:
    """Structural conflict signals for one utterance (never GT-derived).

    ``slot_conflicts`` entries are ``(slot, value_a, value_b)`` triples such as
    ``("negation", "不", "没")``; ``missing_key_slots`` names safety-sensitive
    slots (person/number/time/negation/action/question) absent from one side.
    """

    same_utterance: bool
    time_overlap: bool
    slot_conflicts: tuple[tuple[str, str, str], ...] = ()
    missing_key_slots: tuple[str, ...] = ()
    single_candidate: bool = False
    deadline_elapsed: bool = False

    def __post_init__(self) -> None:
        for name in ("same_utterance", "time_overlap", "single_candidate", "deadline_elapsed"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("slot_conflicts", "missing_key_slots"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        for conflict in self.slot_conflicts:
            if (not isinstance(conflict, (tuple, list)) or len(conflict) != 3
                    or any(not isinstance(part, str) or not part for part in conflict)):
                raise ValueError("slot_conflicts entries must be (slot, value_a, value_b) strings")
        for slot in self.missing_key_slots:
            if not isinstance(slot, str) or not slot.strip():
                raise ValueError("missing_key_slots entries must be nonempty strings")

    def to_dict(self) -> dict:
        return {
            "same_utterance": self.same_utterance,
            "time_overlap": self.time_overlap,
            "slot_conflicts": [list(conflict) for conflict in self.slot_conflicts],
            "missing_key_slots": list(self.missing_key_slots),
            "single_candidate": self.single_candidate,
            "deadline_elapsed": self.deadline_elapsed,
        }


@dataclass(frozen=True)
class FusionDecision:
    """The judge's verdict for one utterance (contract DecisionIntent shape)."""

    action: str
    origin: str
    policy_version: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in DecisionIntent.ACTIONS:
            raise ValueError(f"action must be one of {DecisionIntent.ACTIONS}")
        if self.origin not in DecisionIntent.ORIGINS:
            raise ValueError(f"origin must be one of {DecisionIntent.ORIGINS}")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a nonempty string")
        if not isinstance(self.reason_codes, (tuple, list)) or any(
                not isinstance(code, str) for code in self.reason_codes):
            raise ValueError("reason_codes must be strings")
        if isinstance(self.reason_codes, list):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict:
        return {"action": self.action, "origin": self.origin,
                "policy_version": self.policy_version,
                "reason_codes": list(self.reason_codes)}


@runtime_checkable
class FusionController(Protocol):
    """Placeholder protocol -- shape only, no implementation before G2 restart.

    Implementations (when a restart clears the conditions in
    ``RESTART_CONDITIONS``) receive the evidence candidates of one utterance
    plus :class:`ConflictFeatures` and return a :class:`FusionDecision`.
    Ground truth must never appear in either side of this boundary; payloads
    crossing it are subject to ``slbridge.evidence.assert_gt_clean``.
    """

    policy_version: str

    def evaluate(self, candidates: Sequence[EvidenceCandidate],
                 conflicts: ConflictFeatures) -> FusionDecision: ...
