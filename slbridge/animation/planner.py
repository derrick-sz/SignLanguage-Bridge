"""Full-phrase lookup. Never deletes unsupported words or invents CSL grammar."""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from uuid import uuid4


def normalize_text(text: str) -> str:
    # Keep decimal points, signs and digits: 36.2 and 362 are not the same input.
    return re.sub(r"[\s，。！？!?、；;]", "", unicodedata.normalize("NFKC", text))


def plan_motion(registry, text: str, *, asset_id: str | None = None, confirmed: bool = False,
                research_preview: bool = False, origin: dict | None = None) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        raise ValueError("Input text must contain 1..500 characters")
    if type(confirmed) is not bool or type(research_preview) is not bool:
        raise ValueError("Confirmation flags must be booleans")
    if origin is not None and not isinstance(origin, dict):
        raise ValueError("Origin must be an object")
    normalized = normalize_text(text)
    candidates = [a for a in registry.catalog()["assets"]
                  if normalized and normalized in {normalize_text(s) for s in [a["text"], *a["aliases"]]}]
    if asset_id is not None:
        candidates = [a for a in candidates if a["id"] == asset_id]
    plan = {"schema_version": 1, "plan_id": uuid4().hex, "input_text": text,
            "origin": deepcopy(origin or {"kind": "typed"}), "sign_system": "CSL",
            "archive_sha256": registry.sha256, "language_review": "unverified",
            "can_play": False, "items": [], "candidates": candidates, "unmatched": []}
    if not candidates:
        plan.update(status="blocked", unmatched=[text], reason="没有完整表达资源；未省略或猜测输入内容。")
    elif len(candidates) > 1:
        plan.update(status="ambiguous", reason="存在多个动作版本，请明确选择。")
    elif not confirmed:
        plan.update(status="needs_confirmation", reason="请确认输入内容后再预览。")
    elif not research_preview:
        plan.update(status="needs_language_review", reason="语言尚未认证，只可明确选择研究预览。")
    else:
        chosen = candidates[0]
        plan.update(status="research_preview", can_play=True,
                    items=[{"asset_id": chosen["id"], "sha256": chosen["sha256"],
                            "duration": chosen["duration"]}],
                    reason="研究预览：动作含义与 CSL 表达尚未认证。")
    return plan
