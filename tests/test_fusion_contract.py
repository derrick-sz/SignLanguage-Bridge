"""T-A5-1 tests: FusionController interface placeholder (zero wiring).

P0-A's final card.  G2 decided fusion does NOT enter the main path
(``docs/experiments/P-1/G2_DECISION_MEMO.md``: FAIL -- C/D/E all under the
single-model baseline, cascade saves 72% budget with no quality gain).  This
module therefore pins the interface **shape only**:

* enum values match the contract ``DecisionIntent`` (ACCEPT/FUSE/RETRY/
  UNCERTAIN, origins rules/jev/timeout_fallback);
* the Protocol signature is locked (inputs: evidence candidates + conflict
  features; output: intent + reasons; **never GT**);
* no implementation exists, none is constructible, and -- via a repository
  grep -- no production module imports or calls it (only this test file and
  the definition itself reference it).

Restart conditions (from the G2 memo §5) are documented in ``slbridge/fusion.py``
and mirrored here so the docstring cannot silently drift from the decision.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_type_hints

import pytest

from slbridge.evidence import EvidenceCandidate, EvidenceOutput, EvidenceWindow
from slbridge.fusion import CONFLICT_FEATURE_FIELDS, RESTART_CONDITIONS, ConflictFeatures, \
    DecisionIntent, FusionController, FusionDecision

REPO = Path(__file__).resolve().parents[1]


def candidate(sample_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        run_id="run-1", sample_id=sample_id, model_id="unisign_rgb",
        task="CSLT", sign_system="CSL", modality="RGB_video",
        model_revision="rev-1", preprocess_version="pose-v1",
        window=EvidenceWindow(start_frame=0, end_frame_exclusive=None,
                              start_pts_ms=0, end_pts_ms=2500),
        output=EvidenceOutput(kind="TEXT", gloss_tokens=None, text=text,
                              spoken_language="zh"),
        status="OK", window_id=f"{sample_id}-w1")


# ---------------------------------------------------------------------------
# Enum alignment (contract DecisionIntent)
# ---------------------------------------------------------------------------

def test_decision_intent_actions_match_contract():
    assert DecisionIntent.ACTIONS == ("ACCEPT", "FUSE", "RETRY", "UNCERTAIN")


def test_decision_intent_origins_match_contract():
    assert DecisionIntent.ORIGINS == ("rules", "jev", "timeout_fallback")


def test_fusion_decision_constructs_with_reasons():
    decision = FusionDecision(action="ACCEPT", origin="rules",
                              policy_version="p0a-placeholder-v1",
                              reason_codes=("single_candidate",))
    assert decision.action == "ACCEPT"
    assert decision.to_dict()["reason_codes"] == ["single_candidate"]
    for bad_action in ("REJECT", "accept", None):
        with pytest.raises(ValueError):
            FusionDecision(action=bad_action, origin="rules",
                           policy_version="v", reason_codes=())
    for bad_origin in ("gut", "llm"):
        with pytest.raises(ValueError):
            FusionDecision(action="ACCEPT", origin=bad_origin,
                           policy_version="v", reason_codes=())


# ---------------------------------------------------------------------------
# ConflictFeatures: the input contract (never GT)
# ---------------------------------------------------------------------------

def test_conflict_features_fields_frozen():
    assert CONFLICT_FEATURE_FIELDS == ("same_utterance", "time_overlap",
                                       "slot_conflicts", "missing_key_slots",
                                       "single_candidate", "deadline_elapsed")
    features = ConflictFeatures(same_utterance=True, time_overlap=True,
                                 slot_conflicts=(("negation", "不", "没"),),
                                 missing_key_slots=("number",), single_candidate=False,
                                 deadline_elapsed=False)
    payload = features.to_dict()
    assert payload["slot_conflicts"] == [["negation", "不", "没"]]
    with pytest.raises((ValueError, TypeError)):
        ConflictFeatures(same_utterance=True, time_overlap=True, slot_conflicts=(),
                         missing_key_slots=(), single_candidate=False,
                         deadline_elapsed=False, gt_wer=0.31)      # GT has no place here


def test_evidence_payloads_carry_no_gt_keys():
    from slbridge.evidence import assert_gt_clean
    features = ConflictFeatures(same_utterance=True, time_overlap=True,
                                 slot_conflicts=(), missing_key_slots=(), single_candidate=True,
                                 deadline_elapsed=False)
    assert_gt_clean(features.to_dict(), context="conflict features")
    assert_gt_clean(candidate("S1", "她今年50岁。").to_dict(), context="evidence")


# ---------------------------------------------------------------------------
# Protocol shape lock (signature exists, is typed, has no implementation)
# ---------------------------------------------------------------------------

def test_fusion_controller_protocol_signature():
    hints = get_type_hints(FusionController.evaluate)
    assert set(hints) >= {"candidates", "conflicts", "return"}
    # The decision output type is the contract-shaped FusionDecision.
    assert hints["return"] is FusionDecision


def test_fusion_controller_is_a_protocol_not_an_implementation():
    from typing import Protocol
    assert issubclass(FusionController, Protocol)
    # No default implementation: the placeholder cannot be instantiated.
    with pytest.raises(TypeError):
        FusionController()          # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Zero wiring: grep proof that production code never imports slbridge.fusion
# ---------------------------------------------------------------------------

def test_no_production_module_imports_fusion():
    allowed = {  # the definition itself and this test file
        REPO / "slbridge" / "fusion.py",
        Path(__file__).resolve(),
    }
    pattern = re.compile(r"\b(?:from\s+\.fusion\s+import|from\s+slbridge\.fusion\s+import|"
                         r"import\s+slbridge\.fusion\b)")
    offenders = []
    for path in (REPO / "slbridge").rglob("*.py"):
        if path.resolve() in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text):
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], f"production modules must not import slbridge.fusion: {offenders}"


def test_restart_conditions_documented():
    """The G2 restart conditions stay visible next to the interface."""
    joined = " ".join(RESTART_CONDITIONS)
    for key in ("independent", "development set", "human review", "cost",
                "confidence interval"):
        assert key.lower() in joined.lower()
