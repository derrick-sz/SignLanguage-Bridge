"""P-1 experiment package (model adaptation & fusion feasibility).

Execution program for ``SignLanguageBridge_P-1_实验计划包`` (PLAN.md).
See ``docs/experiments/P-1/`` for the plan, protocols and data contract.

Design rules enforced here:

* inference / fusion payloads never carry ground truth (see ``gt_guard``);
* every run has a run directory with a manifest and immutable JSONL outputs;
* metrics are computed by a separate offline evaluator only (``eval``);
* missing assets or weights are recorded as ``BLOCKED``, never fabricated.
"""

from __future__ import annotations

__all__ = ["common", "gt_guard", "eval", "datasets", "references"]
