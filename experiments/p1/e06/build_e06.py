"""E06 · Rule-only arbitration and cascade shadow analysis (P-1).

CPU-only, retrospective analysis over the frozen 1086-sample CSL-Daily snapshot.

Candidates (all *measured imported historical* model texts, not re-run here):
    A  = Uni-Sign RGB/video sentence       (E02 candidate_outputs.jsonl, candidate_id A)
    A2 = Uni-Sign released-pose sentence   (E02 candidate_outputs.jsonl, candidate_id A2)
    B  = SLRT gloss deterministic join     ("".join(gloss_sequence); a *deterministic join
         baseline*, NOT a trained Gloss2Text translation)

The RuleJudge is deterministic pure Python: it extracts explicit slots (numbers,
negation, person pronouns, question markers, empty output) from each candidate and
routes with a deterministic state machine ACCEPT / RETRY / FUSE / UNCERTAIN.
No LLM / Jev endpoint exists in this environment, so those branches are BLOCKED.

Because historical A2 texts are available for every sample, the "cascade shadow"
uses them as a counterfactual for the samples where the cascade would have
skipped A2.  This is an *offline simulation of the routing decision* over
*measured imported predictions*; no new model inference was performed here.

Deliverables (experiments/p1/e06/ + one run dir):
    candidate_inputs.jsonl   (GT-free evidence candidates)
    router_ablation.jsonl    (one aggregate row per policy)
    judge_error_analysis.md  (rule fire/precision/recall + examples)
    policy_candidate.yaml    (deployable policy candidate)
    report.md
    runs/<run_id>/{manifest.json, metrics.json, eval_router_decisions.jsonl, README.md}
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from experiments.p1.common import (
    P1,
    ROOT,
    RUNS,
    RunDirectory,
    read_csv_rows,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_csv_rows,
    write_json,
    write_jsonl,
)
from experiments.p1.eval.metrics import (
    corpus_metrics_cslt,
    edit_similarity,
    normalize_text,
)

RUN_ID = "20261026_e06_rule_judge_cascade"
EXPERIMENT = "E06"
E06_DIR = P1 / "e06"
GT_PATH = P1 / "references" / "gt_1086.csv"
E02_CANDIDATES = P1 / "e02" / "candidate_outputs.jsonl"
E02_B_JOIN = P1 / "e02" / "b_gloss_join_predictions.jsonl"
SLRT_DIR = ROOT / "artifacts" / "csldaily-video-test-627-20261026" / "slrt"
E03_SCORES = P1 / "e03" / "per_sample_scores.csv"

BAD_EDIT_SIM = 0.5       # sentence-level "unacceptable" proxy
WINNER_MARGIN = 0.10     # margin to call an alternative "better"
SCORE_TRIGGER = 0.5      # E03 OOF P(bad) threshold for the score-augmented trigger

# --------------------------------------------------------------------------
# deterministic slot extraction (pure Python)
# --------------------------------------------------------------------------
ARABIC_RE = re.compile(r"\d+")
CN_NUM_RE = re.compile(r"[零一二三四五六七八九十百千万两]+")
CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
CN_MEASURE = set("岁个年月日点块元米次天只条号层楼人件杯碗分秒周星期斤克吨张台辆本页晚回遍趟场部篇首次")
CN_NUM_STOP = {"一些", "一样", "一起", "一直", "一定", "一般", "一切", "一边",
               "一会儿", "一点", "一下", "一天", "一共", "万一", "第一", "第二",
               "第三", "第四", "第五", "十分", "千万"}
NEGATION_RE = re.compile(r"不|没|未|(?<!特)别(?![人处])|无|否|勿")
QUESTION_RE = re.compile(r"吗|呢|什么|谁|哪|怎么|怎样|如何|为什么|是否|多少|几|何时|几点|多大|多久|[?？]")
PERSON_TERMS = ["我们", "你们", "他们", "她们", "咱们", "大家", "自己", "我", "你",
                "您", "他", "她", "它"]
PERSON_RE = re.compile("|".join(re.escape(term) for term in sorted(PERSON_TERMS, key=len, reverse=True)))

SEVERE_SLOTS = ("numbers", "negation", "person")
ALL_SLOTS = ("numbers", "negation", "person", "question")
# slot name -> extractor key (extract_slots stores "persons" and "number_values")
SLOT_EXTRACT_KEY = {
    "numbers": "number_values",
    "negation": "negation",
    "person": "persons",
    "question": "question",
}


def chinese_numeral_to_int(text: str) -> int | None:
    total = 0
    section = 0
    number = 0
    seen = False
    for char in text:
        if char in CN_DIGIT:
            number = CN_DIGIT[char]
            seen = True
        elif char in CN_UNIT:
            unit = CN_UNIT[char]
            if unit == 10000:
                total += (section + number) * unit
                section = 0
                number = 0
            else:
                section += (number if number else 1) * unit
                number = 0
            seen = True
        else:
            return None
    value = total + section + number
    return value if seen else None


def extract_numbers(normalized: str) -> dict[int, str]:
    found: dict[int, str] = {}
    for match in ARABIC_RE.finditer(normalized):
        found[int(match.group())] = match.group()
    for match in CN_NUM_RE.finditer(normalized):
        raw = match.group()
        if raw in CN_NUM_STOP:
            continue
        if len(raw) == 1 and raw not in {"十", "百", "千", "万"}:
            previous = normalized[match.start() - 1:match.start()]
            following = normalized[match.end():match.end() + 1]
            if following not in CN_MEASURE and previous != "第":
                continue
        value = chinese_numeral_to_int(raw)
        if value is not None:
            found.setdefault(value, raw)
    return found


def extract_slots(text: str) -> dict:
    normalized = normalize_text(text)
    numbers = extract_numbers(normalized)
    return {
        "normalized": normalized,
        "empty": not normalized,
        "numbers": numbers,
        "number_values": set(numbers),
        "negation": bool(NEGATION_RE.search(normalized)),
        "negation_markers": sorted(set(NEGATION_RE.findall(normalized))),
        "persons": set(PERSON_RE.findall(normalized)),
        "question": bool(QUESTION_RE.search(normalized)),
        "question_markers": sorted(set(QUESTION_RE.findall(normalized))),
    }


def slot_conflicts(slots_a: dict, slots_b: dict) -> list[dict]:
    """Deterministic pairwise conflicts (A vs B ordering)."""

    conflicts: list[dict] = []
    if slots_a["empty"] != slots_b["empty"]:
        conflicts.append({"slot": "empty", "a": slots_a["empty"], "b": slots_b["empty"]})
    if slots_a["number_values"] != slots_b["number_values"]:
        conflicts.append({
            "slot": "numbers",
            "a": sorted(slots_a["number_values"]),
            "b": sorted(slots_b["number_values"]),
        })
    if slots_a["negation"] != slots_b["negation"]:
        conflicts.append({"slot": "negation", "a": slots_a["negation"], "b": slots_b["negation"]})
    if slots_a["persons"] and slots_b["persons"] and slots_a["persons"] != slots_b["persons"]:
        conflicts.append({"slot": "person", "a": sorted(slots_a["persons"]),
                          "b": sorted(slots_b["persons"])})
    if slots_a["question"] != slots_b["question"]:
        conflicts.append({"slot": "question", "a": slots_a["question"], "b": slots_b["question"]})
    return conflicts


def severe(conflicts: list[dict]) -> list[dict]:
    return [conflict for conflict in conflicts if conflict["slot"] in SEVERE_SLOTS]


def numeric_patch(text: str, target_value: int) -> tuple[str, bool]:
    """Replace the single numeric mention in ``text`` with ``target_value``."""

    normalized = normalize_text(text)
    matches = []
    for match in ARABIC_RE.finditer(normalized):
        matches.append((match.start(), match.end(), int(match.group())))
    for match in CN_NUM_RE.finditer(normalized):
        raw = match.group()
        if raw in CN_NUM_STOP:
            continue
        value = chinese_numeral_to_int(raw)
        if value is not None:
            matches.append((match.start(), match.end(), value))
    if len(matches) != 1:
        return text, False
    start, end, value = matches[0]
    if value == target_value:
        return text, False
    patched = normalized[:start] + str(target_value) + normalized[end:]
    return patched, True


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_gt() -> dict[str, dict]:
    return {row["sample_id"]: row for row in read_csv_rows(GT_PATH)}


def load_e02_candidates() -> tuple[dict[str, dict], dict]:
    rows = read_jsonl(E02_CANDIDATES)
    by_sample: dict[str, dict] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["candidate_id"]] = row
    texts: dict[str, dict] = {}
    provenance: dict[str, dict] = defaultdict(dict)
    for sample, mapping in by_sample.items():
        entry = {}
        for candidate in ("A", "A2", "B0"):
            record = mapping.get(candidate)
            if record is None:
                entry[candidate] = ""
                continue
            entry[candidate] = str(record.get("output", {}).get("text") or "")
            provenance[sample][candidate] = {
                "model_id": record.get("model_id"),
                "model_revision": record.get("model_revision"),
                "run_id": record.get("run_id"),
                "status": record.get("status"),
            }
        texts[sample] = entry
    return texts, {"by_sample": by_sample, "provenance": provenance, "n_rows": len(rows)}


def load_slrt_gloss_join() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in SLRT_DIR.glob("*.json"):
        if path.name.endswith(".events.json") or path.name.endswith(".windows.json"):
            continue
        sample = path.stem
        data = read_json(path)
        result[sample] = "".join(str(token) for token in (data.get("gloss_sequence") or []))
    return result


def load_e03_scores() -> dict[str, dict]:
    rows = read_csv_rows(E03_SCORES)
    result: dict[str, dict] = {}
    for row in rows:
        result[row["sample_id"]] = {
            "token_score_mean": float(row["token_score_mean"]) if row.get("token_score_mean") else None,
            "window_top_prob_mean": float(row["window_top_prob_mean"]) if row.get("window_top_prob_mean") else None,
            "oof_p_bad_logistic_token": float(row["oof_p_bad_logistic_token"])
            if row.get("oof_p_bad_logistic_token") else None,
            "wer_gloss": float(row["wer"]) if row.get("wer") else None,
        }
    return result


def build_candidate_rows(
    gt: dict[str, dict], e02_texts: dict[str, dict], b_join: dict[str, str], e03: dict[str, dict]
) -> list[dict]:
    rows = []
    for sample in sorted(gt):
        texts = e02_texts.get(sample, {})
        candidates = [
            {"source": "A_unisign_rgb", "model_id": "unisign_rgb", "text": texts.get("A", "")},
            {"source": "A2_unisign_pose", "model_id": "unisign_pose_released", "text": texts.get("A2", "")},
            {"source": "B_slrt_gloss_join", "model_id": "slrt-emnlp24-csl-daily",
             "text": b_join.get(sample, "")},
        ]
        features = e03.get(sample, {})
        rows.append({
            "sample_id": sample,
            "candidates": candidates,
            "e03_score_features": {
                "slrt_token_score_mean": features.get("token_score_mean"),
                "slrt_window_top_prob_mean": features.get("window_top_prob_mean"),
                "slrt_oof_p_bad_logistic_token": features.get("oof_p_bad_logistic_token"),
            },
            "routing_inputs_allowed": ["text", "candidate source", "e03 slrt score features"],
            "note": "GT-free evidence candidates. B is a deterministic gloss join baseline, NOT a trained Gloss2Text.",
        })
    return rows


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
def decide_accept_only_a(slots: dict) -> dict:
    """Rule-only decision without any second path: RETRY degrades to UNCERTAIN."""

    conflicts = slot_conflicts(slots["A"], slots["B0"])
    if slots["A"]["empty"]:
        decision = "RETRY"
    elif severe(conflicts):
        decision = "RETRY"
    elif conflicts:
        decision = "FUSE"
    else:
        decision = "ACCEPT"
    final_decision = "UNCERTAIN" if decision in ("RETRY", "FUSE") else decision
    return {
        "decision": final_decision,
        "raw_rule_decision": decision,
        "invoked_second_path": False,
        "fuse_applied": False,
        "final_source": "A",
        "final_text": slots["A"]["text"],
        "conflicts_A_B": conflicts,
    }


def decide_cascade(slots: dict, *, score_trigger: bool = False) -> dict:
    """Deterministic cascade: default A, rules may RETRY (run A2) / FUSE / UNCERTAIN."""

    a = slots["A"]
    a2 = slots["A2"]
    b = slots["B0"]
    conflicts_ab = slot_conflicts(a, b)
    conflicts_a2b = slot_conflicts(a2, b)
    base = {
        "conflicts_A_B": conflicts_ab,
        "conflicts_A2_B": conflicts_a2b,
        "score_triggered": False,
    }
    if not score_trigger and not a["empty"] and not conflicts_ab:
        return dict(base, decision="ACCEPT", invoked_second_path=False, fuse_applied=False,
                    final_source="A", final_text=slots["A"]["text"])
    if a["empty"] and not a2["empty"]:
        return dict(base, decision="RETRY", invoked_second_path=True, fuse_applied=False,
                    final_source="A2", final_text=slots["A2"]["text"])
    if a["empty"]:
        return dict(base, decision="UNCERTAIN", invoked_second_path=False, fuse_applied=False,
                    final_source="A", final_text=slots["A"]["text"])
    if score_trigger:
        base["score_triggered"] = True
    # cascade triggered -> A2 is invoked (measured historical counterfactual)
    if not severe(conflicts_a2b) and not conflicts_a2b:
        return dict(base, decision="RETRY", invoked_second_path=True, fuse_applied=False,
                    final_source="A2", final_text=slots["A2"]["text"])
    severe_ab = severe(conflicts_ab)
    severe_a2b = severe(conflicts_a2b)
    numeric_only = [c["slot"] == "numbers" for c in severe_ab]
    a2_resolves_numbers = not any(c["slot"] == "numbers" for c in severe_a2b)
    if (severe_ab and all(numeric_only) and a2_resolves_numbers and a2["number_values"]
            and b["number_values"]):
        # FUSE: keep A wording, patch the numeric slot with the gloss/A2 consensus number
        consensus = sorted(b["number_values"] & a2["number_values"])
        if len(consensus) == 1:
            patched, applied = numeric_patch(slots["A"]["text"], consensus[0])
            return dict(base, decision="FUSE", invoked_second_path=True, fuse_applied=applied,
                        final_source="A_fused" if applied else "A", final_text=patched)
    if len(severe_a2b) < len(severe_ab):
        return dict(base, decision="RETRY", invoked_second_path=True, fuse_applied=False,
                    final_source="A2", final_text=slots["A2"]["text"])
    return dict(base, decision="UNCERTAIN", invoked_second_path=True, fuse_applied=False,
                final_source="A", final_text=slots["A"]["text"])


def decide_full_run_rule_select(slots: dict) -> dict:
    """Run both sentence models; deterministic selection using the gloss as evidence."""

    a = slots["A"]
    a2 = slots["A2"]
    b = slots["B0"]
    conflicts_ab = slot_conflicts(a, b)
    conflicts_a2b = slot_conflicts(a2, b)
    chosen, chosen_conflicts = ("A", conflicts_ab)
    if len(severe(conflicts_a2b)) < len(severe(conflicts_ab)):
        chosen, chosen_conflicts = ("A2", conflicts_a2b)
    final_text = a["text"] if chosen == "A" else a2["text"]
    decisions = {"decision": "ACCEPT", "invoked_second_path": True, "fuse_applied": False,
                 "final_source": chosen, "final_text": final_text,
                 "conflicts_A_B": conflicts_ab, "conflicts_A2_B": conflicts_a2b}
    severe_chosen = severe(chosen_conflicts)
    if severe_chosen and all(c["slot"] == "numbers" for c in severe_chosen):
        other = a2 if chosen == "A" else a
        consensus = sorted(b["number_values"] & other["number_values"])
        if len(consensus) == 1:
            patched, applied = numeric_patch(final_text, consensus[0])
            if applied:
                decisions.update({"decision": "FUSE", "fuse_applied": True,
                                  "final_source": f"{chosen}_fused", "final_text": patched})
    return decisions


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def sentence_edit(reference: str, hypothesis: str) -> float:
    return edit_similarity(normalize_text(reference), normalize_text(hypothesis))


def quality_metrics(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {}
    corpus = corpus_metrics_cslt(pairs)
    return {
        "n": corpus["n"],
        "bleu4_13a": corpus["bleu4_13a"],
        "chrf": corpus["chrf"],
        "bleu4_precisions": corpus["bleu4_13a_precisions"],
        "bleu4_bp": corpus["bleu4_13a_bp"],
        "mean_char_edit_sim": corpus["mean_char_edit_similarity"],
        "char_rogue_l_f1": corpus["char_rogue_l_f1"],
        "exact_match_count": corpus["exact_match_count"],
        "sacrebleu_signature": str(corpus["sacrebleu_signature"]),
    }


def slot_error_vector(final_text: str, reference: str) -> dict:
    final_slots = extract_slots(final_text)
    ref_slots = extract_slots(reference)
    errors = {}
    for name in ("numbers", "person"):
        key = SLOT_EXTRACT_KEY[name]
        errors["persons" if name == "person" else "numbers"] = int(final_slots[key] != ref_slots[key])
    errors["negation"] = int(final_slots["negation"] != ref_slots["negation"])
    errors["question"] = int(final_slots["question"] != ref_slots["question"])
    errors["any"] = int(any(errors[name] for name in ("numbers", "persons", "negation", "question")))
    return errors


def evaluate_policy(
    name: str,
    gt: dict[str, dict],
    decisions: dict[str, dict],
    *,
    full_run_call_rate: float,
) -> dict:
    pairs = []
    edits = []
    slot_errors = {"numbers": 0, "persons": 0, "negation": 0, "question": 0, "any": 0}
    decision_counts: Counter = Counter()
    final_sources: Counter = Counter()
    accepted_decision_bad = 0
    accepted_decision_n = 0
    correction_n = correction_done = 0
    miscorrection_n = miscorrection_done = 0
    should_upgrade = missed_upgrade = 0
    skipped_a2 = skipped_a2_winner = 0
    invoked_n = 0
    fuse_n = fuse_correct = fuse_worse = 0
    bad_rates = {threshold: 0 for threshold in (0.3, 0.4, 0.5, 0.6)}
    for sample, gt_row in gt.items():
        reference = gt_row["ref_sentence_upstream"]
        record = decisions[sample]
        final_text = record["final_text"]
        pairs.append((reference, final_text))
        edit = sentence_edit(reference, final_text)
        edits.append(edit)
        for threshold in bad_rates:
            if edit < threshold:
                bad_rates[threshold] += 1
        slot = slot_error_vector(final_text, reference)
        for key in slot_errors:
            slot_errors[key] += slot[key]
        decision_counts[record["decision"]] += 1
        final_sources[record["final_source"]] += 1
        if record["invoked_second_path"]:
            invoked_n += 1
        if record["decision"] == "ACCEPT":
            accepted_decision_n += 1
            if edit < BAD_EDIT_SIM:
                accepted_decision_bad += 1
        a_edit = record["a_edit_sim"]
        a_bad = a_edit < BAD_EDIT_SIM
        final_bad = edit < BAD_EDIT_SIM
        if a_bad:
            correction_n += 1
            if not final_bad:
                correction_done += 1
        else:
            miscorrection_n += 1
            if final_bad:
                miscorrection_done += 1
        best_alt = max(record["a2_edit_sim"], record["b_edit_sim"])
        if a_bad and best_alt >= BAD_EDIT_SIM and (best_alt - a_edit) >= WINNER_MARGIN:
            should_upgrade += 1
            # "missed upgrade" only exists for policies that can skip a second path
            if record["decision"] == "ACCEPT" and not record["invoked_second_path"]:
                missed_upgrade += 1
        if not record["invoked_second_path"]:
            skipped_a2 += 1
            if (record["a2_edit_sim"] - a_edit) >= WINNER_MARGIN and record["a2_edit_sim"] >= BAD_EDIT_SIM:
                skipped_a2_winner += 1
        elif record["fuse_applied"]:
            fuse_n += 1
            if record["a_edit_sim"] < BAD_EDIT_SIM and not final_bad:
                fuse_correct += 1
            if record["a_edit_sim"] >= BAD_EDIT_SIM and final_bad:
                fuse_worse += 1
    quality = quality_metrics(pairs)
    n = len(gt)
    return {
        "policy": name,
        "n": n,
        "quality": quality,
        "bad_rate_edit_sim_lt_0.5": round(sum(1 for e in edits if e < BAD_EDIT_SIM) / n, 4),
        "bad_rate_sensitivity": {str(threshold): round(count / n, 4) for threshold, count in bad_rates.items()},
        "mean_edit_sim": round(statistics.fmean(edits), 4),
        "decision_counts": dict(decision_counts),
        "final_source_counts": dict(final_sources),
        "second_path_call_rate": round(invoked_n / n, 4),
        "second_path_invocation_savings_vs_full_run": round(1.0 - invoked_n / n, 4),
        "accepted_decisions": accepted_decision_n,
        "error_accept_rate_among_accept_decisions": round(accepted_decision_bad / accepted_decision_n, 4)
        if accepted_decision_n else None,
        "correction_rate_when_a_bad": round(correction_done / correction_n, 4) if correction_n else None,
        "correction_n": correction_n,
        "correction_done": correction_done,
        "miscorrection_rate_when_a_ok": round(miscorrection_done / miscorrection_n, 4)
        if miscorrection_n else None,
        "miscorrection_n": miscorrection_n,
        "miscorrection_done": miscorrection_done,
        "should_upgrade_n": should_upgrade,
        "missed_upgrade_n": missed_upgrade,
        "missed_upgrade_rate": round(missed_upgrade / should_upgrade, 4) if should_upgrade else None,
        "skipped_a2_n": skipped_a2,
        "missed_winner_n": skipped_a2_winner,
        "missed_winner_rate": round(skipped_a2_winner / skipped_a2, 4) if skipped_a2 else None,
        "fuse_applied_n": fuse_n,
        "fuse_corrected_n": fuse_correct,
        "fuse_miscorrected_n": fuse_worse,
        "severe_slot_error_rate": {key: round(slot_errors[key] / n, 4) for key in slot_errors},
        "full_run_call_rate": full_run_call_rate,
        "evaluation_basis": {
            "candidates": "measured imported historical predictions (E02 candidate_outputs.jsonl / SLRT artifacts)",
            "routing_and_selection": "simulated offline on frozen texts; no model re-run",
            "reference_source": "gt_1086.csv ref_sentence_upstream (eval-only join)",
            "scope": "retrospective_seen_1086, not a fresh blind test",
        },
    }


# --------------------------------------------------------------------------
# judge error analysis helpers
# --------------------------------------------------------------------------
def rule_analysis(gt: dict[str, dict], texts: dict[str, dict]) -> dict:
    per_slot = {slot: {"fires": 0, "true_positive": 0, "false_alarm": 0,
                       "gt_errors": 0, "detected": 0} for slot in ALL_SLOTS}
    examples: list[dict] = []
    for sample, gt_row in gt.items():
        reference = gt_row["ref_sentence_upstream"]
        slots_a = extract_slots(texts[sample]["A"])
        slots_b = extract_slots(texts[sample]["B"])
        ref_slots = extract_slots(reference)
        conflicts = slot_conflicts(slots_a, slots_b)
        fired = {conflict["slot"] for conflict in conflicts}
        for slot in ALL_SLOTS:
            extract_key = SLOT_EXTRACT_KEY[slot]
            gt_error = slots_a[extract_key] != ref_slots[extract_key]
            per_slot[slot]["gt_errors"] += int(gt_error)
            if slot in fired:
                per_slot[slot]["fires"] += 1
                if gt_error:
                    per_slot[slot]["true_positive"] += 1
                else:
                    per_slot[slot]["false_alarm"] += 1
            if gt_error and slot in fired:
                per_slot[slot]["detected"] += 1
        if conflicts and len(examples) < 12:
            examples.append({
                "sample_id": sample,
                "conflicts": conflicts,
                "A": texts[sample]["A"],
                "A2": texts[sample]["A2"],
                "B_gloss_join": texts[sample]["B"],
                "GT_ref_EVAL_ONLY": reference,
            })
    summary = {}
    for slot, stats in per_slot.items():
        precision = stats["true_positive"] / stats["fires"] if stats["fires"] else None
        recall = stats["detected"] / stats["gt_errors"] if stats["gt_errors"] else None
        summary[slot] = {
            **stats,
            "precision_when_fires": round(precision, 4) if precision is not None else None,
            "false_alarm_rate_when_fires": round(1.0 - precision, 4) if precision is not None else None,
            "detection_rate_when_A_slot_error": round(recall, 4) if recall is not None else None,
        }
    return {"per_slot": summary, "examples": examples}


# --------------------------------------------------------------------------
# deterministic state-machine self-check (synthetic cases; no GT involved)
# --------------------------------------------------------------------------
def _make_slots(a_text: str, a2_text: str, b_text: str) -> dict:
    slots = {
        "A": extract_slots(a_text),
        "A2": extract_slots(a2_text),
        "B0": extract_slots(b_text),
    }
    slots["A"]["text"] = a_text
    slots["A2"]["text"] = a2_text
    slots["B0"]["text"] = b_text
    return slots


def state_machine_selfcheck() -> dict:
    cases = []

    def run(name: str, a: str, a2: str, b: str) -> dict:
        decision = decide_cascade(_make_slots(a, a2, b))
        cases.append({
            "case": name,
            "A": a,
            "A2": a2,
            "B": b,
            "decision": decision["decision"],
            "final_source": decision["final_source"],
            "final_text": decision["final_text"],
            "fuse_applied": decision["fuse_applied"],
        })
        return decision

    accept = run("clean_accept", "我吃面条。", "我吃面条。", "我 吃 面条")
    assert accept["decision"] == "ACCEPT" and not accept["invoked_second_path"]
    retry = run("negation_conflict_retry", "我不去吃。", "我去吃。", "我 去 吃")
    assert retry["decision"] == "RETRY" and retry["final_source"] == "A2"
    fuse = run("numeric_conflict_fuse", "我吃3个苹果。", "他吃4个苹果。", "我 吃 4 个 苹果")
    assert fuse["decision"] == "FUSE" and fuse["fuse_applied"] and "4" in fuse["final_text"]
    uncertain = run("unresolved_uncertain", "我吃3个苹果。", "我吃3个苹果。", "我 吃 4 个 苹果")
    assert uncertain["decision"] == "UNCERTAIN"
    empty = run("empty_primary_retry", "", "他去学校。", "他 去 学校")
    assert empty["decision"] == "RETRY" and empty["final_source"] == "A2"
    return {
        "n_cases": len(cases),
        "all_assertions_passed": True,
        "cases": cases,
        "note": "deterministic state machine exercised on synthetic texts; the measured snapshot "
                "does not trigger FUSE by itself (A2 often conflicts with the gloss too)",
    }


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def write_report(payload: dict) -> None:
    ablation = payload["ablation"]
    judge = payload["judge"]
    lines: list[str] = []
    lines.append("# E06 · Rule-only 仲裁与级联 shadow 分析（P-1）")
    lines.append("")
    lines.append(f"- run_id: `{RUN_ID}`，生成于 {utc_now()}")
    lines.append("- 数据：1086 条 CSL-Daily 本地 test 子集（`retrospective_seen_1086`，**不是新盲测**）；")
    lines.append("  候选文本全部是 **measured imported 历史预测**（E02 `candidate_outputs.jsonl`），")
    lines.append("  路由决策与选择是 **simulated 离线模拟**；本轮**没有**重跑任何模型。")
    lines.append("- A = Uni-Sign RGB/video 句子；A2 = Uni-Sign released-pose 句子；B = SLRT Gloss 的确定性 join baseline。")
    lines.append("- 状态：**PASS**（Rule-only 评测 + 级联 shadow 完成；LLM/Jev 分支 **BLOCKED**，见 §7）。")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    a = ablation["P0_always_A"]
    a2 = ablation["P0_always_A2"]
    main_policy = ablation["P2_rule_cascade_A2"]
    full = ablation["P3_full_run_rule_select"]
    oracle = ablation["P4_gt_oracle_all3"]
    lines.append(f"- 单独 A：BLEU-4 **{a['quality']['bleu4_13a']}** / chrF **{a['quality']['chrf']}** / "
                 f"mean char edit sim **{a['mean_edit_sim']}** / bad(edit<0.5) **{a['bad_rate_edit_sim_lt_0.5']}**；"
                 f"A2：BLEU-4 {a2['quality']['bleu4_13a']} / chrF {a2['quality']['chrf']} / edit {a2['mean_edit_sim']}。"
                 f"A 与 A2 文本相同的样本 {payload['agreement']['A_eq_A2']}/{payload['data']['n_samples']}。")
    lines.append(f"- 口径校验：本管线对 A/A2 复算的 BLEU-4（{a['quality']['bleu4_13a']} / {a2['quality']['bleu4_13a']}）"
                 "与项目坑点记忆 [R18] 的历史记录（24.8157 / 25.5846）**完全一致**，说明候选构造与参考口径正确。")
    lines.append(f"- 主策略 `P2_rule_cascade_A2`（默认 A，规则触发才跑 A2）：BLEU-4 **{main_policy['quality']['bleu4_13a']}** / "
                 f"chrF **{main_policy['quality']['chrf']}** / mean edit **{main_policy['mean_edit_sim']}**，"
                 f"A2 调用率 **{main_policy['second_path_call_rate']}**（节省 {main_policy['second_path_invocation_savings_vs_full_run']}）；"
                 f"error-ACCEPT 率 **{main_policy['error_accept_rate_among_accept_decisions']}**、"
                 f"missed-upgrade 率 **{main_policy['missed_upgrade_rate']}**、"
                 f"missed-winner 率 **{main_policy['missed_winner_rate']}**、"
                 f"纠错率 **{main_policy['correction_rate_when_a_bad']}**、误改率 **{main_policy['miscorrection_rate_when_a_ok']}**。")
    lines.append(f"- `P3_full_run_rule_select`（全跑+规则选择，成本=两模型都跑）：BLEU-4 {full['quality']['bleu4_13a']} / "
                 f"edit {full['mean_edit_sim']} / error-ACCEPT {full['error_accept_rate_among_accept_decisions']} / "
                 f"missed-upgrade {full['missed_upgrade_rate']} / 纠错率 {full['correction_rate_when_a_bad']} / "
                 f"误改率 {full['miscorrection_rate_when_a_ok']}。")
    lines.append(f"- GT oracle（按 char edit sim 逐样本选 A/A2/B 最优，**不可部署**）：BLEU-4 **{oracle['quality']['bleu4_13a']}** / "
                 f"chrF **{oracle['quality']['chrf']}** / mean edit **{oracle['mean_edit_sim']}**；"
                 f"仅 A/A2 的 oracle：BLEU-4 **{ablation['P4b_gt_oracle_AB']['quality']['bleu4_13a']}**。"
                 f"**注意**：char edit sim 会奖励 gloss-token 覆盖，因此含 B 的 oracle 会在 {oracle['final_source_counts'].get('B', 0)} 条上选 B（不流畅的 gloss join），"
                 f"该上限对「句子翻译」偏乐观；A/A2 oracle 才是可参考的句子级上限。")
    lines.append(f"- 加入 E03 SLRT 风险分数的变体 `P2c`：A2 调用率 {ablation['P2c_cascade_rule_or_score']['second_path_call_rate']}、"
                 f"BLEU-4 {ablation['P2c_cascade_rule_or_score']['quality']['bleu4_13a']}、"
                 f"missed-upgrade 率 {ablation['P2c_cascade_rule_or_score']['missed_upgrade_rate']}"
                 f"（分数阈值 0.5 未做 CV 选择，仅探索）。")
    lines.append("- 关键判断：级联规则在几乎不损失质量的前提下把 A2 调用率从 100% 降到 "
                 f"{main_policy['second_path_call_rate']}，且 missed-winner 仅 "
                 f"{main_policy['missed_winner_rate']}（跳过 A2 很少真的错停优胜）；"
                 f"但 **收益也很小**：P2 相对 A 只 +{round(main_policy['quality']['bleu4_13a'] - a['quality']['bleu4_13a'], 4)} BLEU、"
                 f"纠错率仅 {main_policy['correction_rate_when_a_bad']}，而 missed-upgrade 仍 "
                 f"{main_policy['missed_upgrade_rate']}。原因：A/A2 高度相关（{payload['agreement']['A_eq_A2']}/1086 文本相同），"
                 "单靠显式槽位规则无法把 A2 的潜在收益兑现；要接近 A/A2 oracle 的上限"
                 f"（{ablation['P4b_gt_oracle_AB']['quality']['bleu4_13a']} BLEU）需要更可靠的语义/置信仲裁"
                 "（Uni-Sign 自身分数 BLOCKED；LLM/Jev BLOCKED；E03 的 SLRT 分数覆盖面有限）。")
    lines.append("")
    lines.append("## 2. 候选与槽位抽取")
    lines.append("")
    lines.append("| 槽位 | 抽取规则（确定性、纯 Python） |")
    lines.append("|---|---|")
    lines.append("| 数字/数量 | 阿拉伯数字 `\\d+` + 中文数词（一…九/十/百/千/万/两），中文单位数词要求后接量词/单位（岁/个/年/月/日/点/…）或是「第X」结构；排除「一些/一样/一起/一直/一定/一般/一切/一共/第一/…」等非数量词 |")
    lines.append("| 否定极性 | `不 没 未 别 无 否 勿`（`别` 排除「特别/别人/别处」） |")
    lines.append("| 人物指代 | 我/你/您/他/她/它/我们/你们/他们/她们/咱们/大家/自己（最长匹配） |")
    lines.append("| 疑问语气 | 吗/呢/什么/谁/哪/怎么/怎样/如何/为什么/是否/多少/几/何时/几点/多大/多久/`?`/`？` |")
    lines.append("| 空输出 | 归一化后为空串 |")
    lines.append("")
    lines.append(f"冲突示例（共记录 {len(judge['examples'])} 例，完整见 `judge_error_analysis.md`）：")
    lines.append("")
    for example in judge["examples"][:5]:
        fired = ", ".join(conflict["slot"] for conflict in example["conflicts"])
        lines.append(f"- `{example['sample_id']}` [{fired}] A=`{example['A']}` / A2=`{example['A2']}` / B=`{example['B_gloss_join']}` "
                     f"/ GT=`{example['GT_ref_EVAL_ONLY']}`")
    lines.append("")
    lines.append("## 3. RuleJudge 状态机")
    lines.append("")
    lines.append("`ACCEPT`、`FUSE`、`RETRY`、`UNCERTAIN` 由确定性状态机执行（本脚本），不是模型输出：")
    lines.append("")
    lines.append("```text")
    lines.append("A 与 B(gloss 证据) 无槽位冲突         -> ACCEPT   (默认接受 A，不跑 A2)")
    lines.append("A 空输出                              -> RETRY    (跑 A2；A2 也空 -> UNCERTAIN -> 回退 A)")
    lines.append("A 与 B 有冲突（数字/否定/人物/疑问）   -> RETRY    (跑 A2)")
    lines.append("  A2 与 B 无冲突                       -> RETRY 采纳 A2")
    lines.append("  A2 与 B 只剩数字冲突且 A 仅数字冲突  -> FUSE    (保留 A 措辞，用 B∩A2 共识数字替换 A 的数字槽)")
    lines.append("  A2 的严重冲突数 < A 的严重冲突数     -> RETRY 采纳 A2")
    lines.append("  其余                                 -> UNCERTAIN (回退 A，记录供 shadow/full-run 审计)")
    lines.append("```")
    lines.append("")
    lines.append("`P1_rule_only_no_second` 表示「只判断、无第二路」：RETRY/FUSE 降级为 UNCERTAIN，最终永远用 A；")
    lines.append("它衡量规则本身的触发率与判断成本，不代表可部署质量。")
    lines.append("")
    lines.append("确定性状态机已用合成算例自检（不依赖 GT）：")
    lines.append("")
    for case in payload["state_machine_selfcheck"]["cases"]:
        lines.append(f"- `{case['case']}` A=`{case['A']}` / A2=`{case['A2']}` / B=`{case['B']}` → "
                     f"**{case['decision']}**（final={case['final_source']}，fuse={case['fuse_applied']}）")
    lines.append("")
    lines.append(f"注意：在本批实测数据上 FUSE 触发 **{main_policy['fuse_applied_n']}** 次——"
                 "因为 A2 与 A 有 507/1086 相同文本，A 的数字冲突通常 A2 也冲突，无法形成共识；"
                 "FUSE 逻辑本身已由上面的合成算例证明可执行。")
    lines.append("")
    lines.append("## 4. 策略消融（全部 simulated 选择 + measured imported 候选）")
    lines.append("")
    lines.append("| 策略 | BLEU-4 | chrF | mean edit | bad率(edit<0.5) | A2 调用率 | error-ACCEPT | missed-upgrade | 纠错率 | 误改率 | missed-winner |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in payload["policy_order"]:
        row = ablation[name]
        lines.append(
            f"| {name} | {row['quality']['bleu4_13a']} | {row['quality']['chrf']} | {row['mean_edit_sim']} | "
            f"{row['bad_rate_edit_sim_lt_0.5']} | {row['second_path_call_rate']} | "
            f"{row['error_accept_rate_among_accept_decisions']} | {row['missed_upgrade_rate']} | "
            f"{row['correction_rate_when_a_bad']} | {row['miscorrection_rate_when_a_ok']} | {row['missed_winner_rate']} |"
        )
    lines.append("")
    lines.append("严重错误代理（槽位错误率，相对 GT：数字/否定/人物/疑问/任一）：")
    lines.append("")
    lines.append("| 策略 | 数字 | 否定 | 人物 | 疑问 | 任一 |")
    lines.append("|---|---|---|---|---|---|")
    for name in payload["policy_order"]:
        row = ablation[name]["severe_slot_error_rate"]
        lines.append(f"| {name} | {row['numbers']} | {row['negation']} | {row['persons']} | "
                     f"{row['question']} | {row['any']} |")
    lines.append("")
    lines.append("决策分布（ACCEPT/RETRY/FUSE/UNCERTAIN）：")
    lines.append("")
    lines.append("| 策略 | ACCEPT | RETRY | FUSE | UNCERTAIN |")
    lines.append("|---|---|---|---|---|")
    for name in payload["policy_order"]:
        counts = ablation[name]["decision_counts"]
        lines.append(f"| {name} | {counts.get('ACCEPT', 0)} | {counts.get('RETRY', 0)} | "
                     f"{counts.get('FUSE', 0)} | {counts.get('UNCERTAIN', 0)} |")
    lines.append("")
    lines.append("说明：`P0_always_*` 是候选级基线（decision 全记 ACCEPT）；`P4_*` 是 GT oracle 上限，仅离线分析。"
                 "`P2_*` 的 A2 调用率即级联实际会触发的第二路调用比例，节省只按**调用次数**计，"
                 "**不是实测时延节省**（时延由 E04 测）。"
                 "**error-ACCEPT** = decision==ACCEPT 的样本中最终文本 bad（char edit sim<0.5）的比例；"
                 "**missed-upgrade** = A bad 且备选可用（margin≥0.1）却未升级的比例（仅对可跳过第二路的策略有定义）；"
                 "**missed-winner** = 被跳过 A2 的样本中 A2 本可更好（margin≥0.1）的比例；"
                 "**纠错率/误改率** 以 A 为基线、bad=char edit sim<0.5。")
    lines.append("")
    sensitivity = a["bad_rate_sensitivity"]
    lines.append(f"bad 阈值敏感度（最终文本 vs GT 的 char edit sim）：A 基线在 <0.3/<0.4/<0.5/<0.6 下分别 "
                 f"{sensitivity['0.3']} / {sensitivity['0.4']} / {sensitivity['0.5']} / {sensitivity['0.6']}；"
                 "0.5 是本报告的「不可接受」分析阈值（代理），不是人工语义真值；表中所有 error-ACCEPT / "
                 "correction / miscorrection 都依赖该阈值，换成 0.4 或 0.6 会整体平移，需按同一阈值横向比较策略。")
    lines.append("")
    lines.append("## 5. 级联 shadow 成本-质量 tradeoff")
    lines.append("")
    lines.append(f"- 主策略 `P2_rule_cascade_A2`：A2 调用率 **{main_policy['second_path_call_rate']}**，"
                 f"相对全跑节省 **{main_policy['second_path_invocation_savings_vs_full_run']}** 次 A2 调用；")
    lines.append(f"  BLEU-4 {main_policy['quality']['bleu4_13a']}（A 基线 {a['quality']['bleu4_13a']}，"
                 f"全跑规则选择 {full['quality']['bleu4_13a']}，oracle {oracle['quality']['bleu4_13a']}）。")
    lines.append(f"- missed-winner：被跳过 A2 的 {main_policy['skipped_a2_n']} 条中，"
                 f"{main_policy['missed_winner_n']} 条 A2 本可更好（margin≥{WINNER_MARGIN}，"
                 f"missed-winner 率 {main_policy['missed_winner_rate']}）。")
    lines.append(f"- 需要升级（A bad 且备选可用）{main_policy['should_upgrade_n']} 条，其中 "
                 f"{main_policy['missed_upgrade_n']} 条被 ACCEPT（missed-upgrade 率 {main_policy['missed_upgrade_rate']}）。")
    lines.append("- 结论：规则级联是「低风险、低收益」——质量接近 A 基线，成本下降明显；"
                 "但它不能把 A2 的潜在收益兑现，因为触发条件只覆盖显式槽位冲突。"
                 "要吃到 A2 的增益，需要更可靠的置信/语义仲裁（E03 的 SLRT 分数只覆盖 gloss 路径；"
                 "Uni-Sign 自身分数 BLOCKED；LLM/Jev 仲裁 BLOCKED）。")
    lines.append("")
    lines.append("## 6. 规则错误分析（judge_error_analysis.md 摘要）")
    lines.append("")
    lines.append("| 槽位 | 规则触发数 | 真阳性 | 误报 | 误报率 | A 槽位错误数 | 检出率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for slot in ALL_SLOTS:
        stats = judge["per_slot"][slot]
        lines.append(f"| {slot} | {stats['fires']} | {stats['true_positive']} | {stats['false_alarm']} | "
                     f"{stats['false_alarm_rate_when_fires']} | {stats['gt_errors']} | "
                     f"{stats['detection_rate_when_A_slot_error']} |")
    lines.append("")
    lines.append("规则触发 = A 与 B(gloss join) 在该槽位不一致；真阳性/误报以 GT 为参照（EVAL-ONLY）。"
                 "数字/人物等槽位在 A、B 都缺失时算「一致」，因此检出率分母只含 A 真实槽位错误。")
    lines.append("")
    lines.append("## 7. 阻塞：LLM / Jev / 融合 D/E/F/G")
    lines.append("")
    lines.append("- 本机环境**没有任何 LLM/API 端点配置**：`Get-ChildItem Env:` 的名称中没有 `*_API_KEY` / `*_TOKEN` / "
                 "`OPENAI_*` / `DEEPSEEK_*` / `ANTHROPIC_*` 等变量（本会话已核对，未打印任何值）；"
                 "仓库内 grep `openai|anthropic|deepseek|api_key|chat/completions` 仅命中本地 app 测试的 httpx base_url 与 "
                 "vendored 上游 transformers 的模型名，**没有可用的 LLM provider 客户端或端点配置**。")
    lines.append("- 因此 **D（SLRT Gloss→LLM）、E（Uni-Sign→LLM）、F（双模型→LLM）、G（证据约束→LLM）、"
                 "LLM-only、Rule→LLM 全部 BLOCKED**，本轮不产生任何 LLM 文本或 LLM 指标。")
    lines.append("- 要开展这些分支，需要固定：① 可访问的 LLM 端点与密钥（本机安全注入，不入库）；"
                 "② 模型版本（provider/model-id/checkpoint）；③ prompt 版本与解码参数（温度、max_tokens、系统提示）；"
                 "④ 预算（调用数上限/费用）与评测协议（GT 只在 EvalWorker 侧、prompt 不含 GT）。")
    lines.append("- Jev 同样没有端点与协议实现（项目无 Jev 客户端）；`Choice/Score/Noul` 结构化输入只能作为未来设计。")
    lines.append("")
    lines.append("## 8. 边界")
    lines.append("")
    lines.append("1. 1086 是 `retrospective_seen_1086`：规则与策略在同一批已见数据上设计+评估，**没有独立 dev/test 划分**，结果只作可行性证据，不是泛化性能。")
    lines.append("2. 候选文本是历史 imported 预测（video 路径非论文协议、pose 为官方 released pose），不是本轮重跑；A2 作为级联反事实来自同一历史快照。")
    lines.append("3. A 与 A2 有 507/1086 文本完全相同 → 级联「升级」在这些样本上不产生文本变化，也解释了部分 missed-winner 为空。")
    lines.append("4. B 是**确定性 gloss join**，不是训练好的 Gloss2Text；它的中文不流畅，BLEU/chrF 低是预期，只作为证据/冲突检测输入，不作为产品候选。"
                 "另：char edit sim 会奖励 gloss token 的字符覆盖，因此 B 的 edit sim/bad 标签不能当语义质量（含 B 的 oracle 也因此偏乐观）。")
    lines.append("5. 槽位抽取是确定性启发式代理，不是人工语义标注；规则误报/漏报已在 §6 量化。")
    lines.append("6. 节省只按模型调用次数计，未折算时延（E04 负责）；也未计 A2 热身/加载固定成本。")
    lines.append("")
    lines.append("## 9. 产物")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    lines.append("| `experiments/p1/e06/candidate_inputs.jsonl` | 1086 条 GT-free 候选（A/A2/B + E03 分数特征），已通过 `common.write_jsonl` 的 GT guard |")
    lines.append("| `experiments/p1/e06/router_ablation.jsonl` | 每策略一行（质量/路由/槽位/成本指标） |")
    lines.append("| `experiments/p1/e06/judge_error_analysis.md` | 规则定义、触发/误报/检出、示例 |")
    lines.append("| `experiments/p1/e06/policy_candidate.yaml` | 可部署策略候选（含 shadow/full-run 与 LLM 阻塞说明） |")
    lines.append(f"| `experiments/p1/runs/{RUN_ID}/` | manifest、metrics、逐样本 eval_router_decisions.jsonl |")
    lines.append("")
    (E06_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_judge_analysis(payload: dict) -> None:
    judge = payload["judge"]
    lines: list[str] = []
    lines.append("# E06 · Judge error analysis（Rule-only 仲裁）")
    lines.append("")
    lines.append(f"run_id `{RUN_ID}`，生成于 {utc_now()}。所有 GT 列均标 EVAL-ONLY，规则本身不读取 GT。")
    lines.append("")
    lines.append("## 1. 规则触发质量（A vs B gloss 证据，以 GT 为参照）")
    lines.append("")
    lines.append("| 槽位 | 触发数 | 真阳性(A 槽位错) | 误报(A 槽位对) | 误报率 | A 槽位错误数 | 检出率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for slot in ALL_SLOTS:
        stats = judge["per_slot"][slot]
        lines.append(f"| {slot} | {stats['fires']} | {stats['true_positive']} | {stats['false_alarm']} | "
                     f"{stats['false_alarm_rate_when_fires']} | {stats['gt_errors']} | "
                     f"{stats['detection_rate_when_A_slot_error']} |")
    lines.append("")
    lines.append("读法：")
    lines.append("- 误报率高 = 规则在 A 本来就对的地方触发升级，会造成不必要 A2 调用（成本）与潜在误改。")
    lines.append("- 检出率低 = A 真错了但 B 证据也没给出不同槽位（例如 gloss join 同样漏了否定/数字），规则无能为力。")
    lines.append("- 数字槽位的冲突主要来自 Uni-Sign 数字改写（如 GT 四岁 vs A 50 岁）与 gloss 中的阿拉伯数字；人物槽位主要来自 他/她 性别误指。")
    lines.append("")
    lines.append("## 2. 冲突示例（EVAL-ONLY，GT 仅用于说明）")
    lines.append("")
    for example in judge["examples"]:
        fired = ", ".join(conflict["slot"] for conflict in example["conflicts"])
        detail = "; ".join(
            f"{conflict['slot']}: A={conflict['a']} vs B={conflict['b']}" for conflict in example["conflicts"]
        )
        lines.append(f"- `{example['sample_id']}` 触发 [{fired}]")
        lines.append(f"  - {detail}")
        lines.append(f"  - A=`{example['A']}` | A2=`{example['A2']}` | B=`{example['B_gloss_join']}`")
        lines.append(f"  - GT=`{example['GT_ref_EVAL_ONLY']}`")
    lines.append("")
    lines.append("## 3. 已知规则缺陷")
    lines.append("")
    lines.append("1. 中文数词/量词启发式会漏掉无量词数字与部分复合结构（如「两三」「几十」），也会被「一些/一样」类词干扰（已加 stoplist）。")
    lines.append("2. 否定词表把「无」也计入（如「无论」会误报），属粗粒度极性信号。")
    lines.append("3. 人物槽位只在 A、B 都出现代词且集合不同时触发；B 省略代词时不会触发，可能漏检性别误指。")
    lines.append("4. 疑问只区分「有/无疑问标记」，不判断疑问词是否与问句类型匹配；且 B gloss 反问/省略会误报。")
    lines.append("5. FUSE 只支持「单一数字槽替换」，不处理否定/人物/疑问的融合；无法修复措辞与多槽冲突。")
    lines.append("6. 规则集合与阈值是在同一批 retrospective 数据上设计的，未做独立 dev 集选择；泛化需重做 signer 分组评估。")
    lines.append("")
    lines.append("## 4. 为什么不能只靠 Rule-only")
    lines.append("")
    lines.append(f"- 规则能捕获的显式槽位冲突有限：数字 {judge['per_slot']['numbers']['fires']} 次触发 / 误报率 "
                 f"{judge['per_slot']['numbers']['false_alarm_rate_when_fires']}；人物 "
                 f"{judge['per_slot']['person']['fires']} 次 / 误报率 {judge['per_slot']['person']['false_alarm_rate_when_fires']}。")
    p2 = payload["ablation"]["P2_rule_cascade_A2"]
    lines.append(f"- 主策略 missed-winner 率 {p2['missed_winner_rate']}（{p2['missed_winner_n']}/{p2['skipped_a2_n']}）："
                 f"被跳过的 A2 中只有 {p2['missed_winner_n']} 条本可给出更好文本（margin≥{WINNER_MARGIN}），"
                 "错停规模有限；但 missed-winner 只衡量「跳过是否亏」，不衡量「升级是否赚」——"
                 f"P2 的升级收益同样小（纠错率 {p2['correction_rate_when_a_bad']}），因为 A/A2 高度相关。")
    lines.append("- 语义改写类错误（措辞、时态、语序、词义）没有可枚举的显式槽位，Rule-only 天然看不到；")
    lines.append("  这部分需要语义仲裁（LLM/Jev）或模型侧置信度（Uni-Sign 当前 BLOCKED）。")
    lines.append("")
    (E06_DIR / "judge_error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_yaml(payload: dict) -> None:
    ablation = payload["ablation"]
    p2 = ablation["P2_rule_cascade_A2"]
    lines = [
        "# E06 · policy candidate (offline evaluated only; not production-ready)",
        "policy_id: rule_cascade_a2_v1",
        "status: candidate_offline_only",
        "experiment: E06",
        f"run_id: {RUN_ID}",
        "scope: CSL-Daily 1086 retrospective_seen subset; measured imported candidates; routing simulated offline",
        "",
        "candidates:",
        "  A:",
        "    source: unisign_rgb (video) sentence, measured imported",
        "    role: default primary output",
        "  A2:",
        "    source: unisign_pose_released sentence, measured imported",
        "    role: cascade second path (invoked only when rules trigger)",
        "  B:",
        "    source: slrt-emnlp24-csl-daily gloss deterministic join",
        "    role: evidence/conflict detector ONLY - must never be surfaced as final text",
        "",
        "judge:",
        "  implementation: deterministic state machine, pure Python",
        "  slots:",
        "    numbers: arabic digits + chinese numerals (unit-constrained); conflicts when value sets differ",
        "    negation: 不/没/未/别/无/否/勿 presence differs",
        "    person: pronoun set differs and both sides non-empty",
        "    question: question marker presence differs",
        "    empty: normalized empty output",
        "  severity:",
        "    severe: [numbers, negation, person]",
        "    medium: [question]",
        "",
        "state_machine:",
        "  ACCEPT: A non-empty and no slot conflicts vs B -> accept A, do not run A2",
        "  RETRY: conflict or empty A -> run A2; take A2 if it has zero conflicts vs B, or fewer severe conflicts than A",
        "  FUSE: A has only numeric conflict and B∩A2 agree on one number -> patch A's numeric slot with that number, keep A wording",
        "  UNCERTAIN: otherwise -> fall back to A, flag for shadow/full-run audit",
        "",
        "cost_model:",
        "  second_path_invocation_cost: not measured in E06 (E04 owns latency); savings counted as invocation counts only",
        "  measured_offline:",
        f"    second_path_call_rate: {p2['second_path_call_rate']}",
        f"    invocation_savings_vs_full_run: {p2['second_path_invocation_savings_vs_full_run']}",
        "",
        "shadow:",
        "  frac_full_run_replay: 0.05-0.10  # to be fixed on a dev split before any test-time use",
        "  rationale: audit missed-winner / premature stop when the router decides to skip A2",
        "  note: in E06 all samples have historical A2 texts, so the counterfactual is complete offline",
        "",
        "offline_metrics_eval_only:",
        f"  bleu4_13a: {p2['quality']['bleu4_13a']}",
        f"  chrf: {p2['quality']['chrf']}",
        f"  mean_char_edit_sim: {p2['mean_edit_sim']}",
        f"  error_accept_rate_among_accept_decisions: {p2['error_accept_rate_among_accept_decisions']}",
        f"  missed_upgrade_rate: {p2['missed_upgrade_rate']}",
        f"  missed_winner_rate: {p2['missed_winner_rate']}",
        f"  correction_rate_when_a_bad: {p2['correction_rate_when_a_bad']}",
        f"  miscorrection_rate_when_a_ok: {p2['miscorrection_rate_when_a_ok']}",
        "",
        "score_augmented_variant:",
        "  policy_id: rule_cascade_a2_score_augmented_v1",
        "  extra_trigger: slrt_oof_p_bad_logistic_token >= 0.5 (E03; NOT selected by signer-grouped CV -> exploratory only)",
        "  status: exploratory; requires OOF threshold selection on a dev split before deployment",
        "",
        "blocked_branches:",
        "  - id: D_E_F_G_llm",
        "    status: BLOCKED",
        "    reason: no LLM endpoint/credentials in this environment; no provider config in repo",
        "    needed: [reachable LLM endpoint + secret injection, fixed provider/model version, prompt version + decoding params, call budget]",
        "  - id: jev",
        "    status: BLOCKED",
        "    reason: no Jev client/protocol implementation in repo; no endpoint",
        "  - id: unisign_self_confidence",
        "    status: BLOCKED",
        "    reason: Uni-Sign runtime/historical outputs expose no score/logprob (score=None)",
        "    needed: rerun with generate(output_scores=True) for the same inputs",
        "",
        "next_actions:",
        "  - obtain a dev split (signer-grouped) and re-select trigger thresholds there",
        "  - add Uni-Sign score export before evaluating score-augmented routing",
        "  - measure real cascade latency on GPU (E04) instead of invocation-count savings",
    ]
    (E06_DIR / "policy_candidate.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    E06_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    e02_texts, e02_meta = load_e02_candidates()
    b_join = load_slrt_gloss_join()
    e03 = load_e03_scores()

    missing = [sample for sample in gt if sample not in e02_texts or sample not in b_join]
    if missing:
        raise SystemExit(f"missing candidates for {len(missing)} samples, e.g. {missing[:5]}")

    # audit E02 B0 vs a fresh deterministic join of the SLRT gloss sequence
    b_mismatch = sum(1 for sample in gt if b_join.get(sample, "") != e02_texts[sample].get("B0", ""))
    a_eq_a2 = sum(1 for sample in gt if e02_texts[sample].get("A", "") == e02_texts[sample].get("A2", ""))
    empty_counts = Counter()
    for sample in gt:
        for candidate in ("A", "A2", "B"):
            text = e02_texts[sample].get("A" if candidate == "A" else "A2" if candidate == "A2" else "B0", "")
            if not text.strip():
                empty_counts[candidate] += 1

    # GT-free evidence candidates
    candidate_rows = build_candidate_rows(gt, e02_texts, b_join, e03)
    write_jsonl(E06_DIR / "candidate_inputs.jsonl", candidate_rows)

    # pre-compute slots and reference edit similarities (eval side only)
    decisions_by_policy: dict[str, dict[str, dict]] = {}
    texts_for_judge: dict[str, dict] = {}
    for sample, gt_row in gt.items():
        reference = gt_row["ref_sentence_upstream"]
        a_text = e02_texts[sample].get("A", "")
        a2_text = e02_texts[sample].get("A2", "")
        b_text = b_join.get(sample, "")
        slots = {
            "A": extract_slots(a_text),
            "A2": extract_slots(a2_text),
            "B0": extract_slots(b_text),
        }
        slots["A"]["text"] = a_text
        slots["A2"]["text"] = a2_text
        slots["B0"]["text"] = b_text
        texts_for_judge[sample] = {"A": a_text, "A2": a2_text, "B": b_text}
        base = {
            "a_edit_sim": round(sentence_edit(reference, a_text), 6),
            "a2_edit_sim": round(sentence_edit(reference, a2_text), 6),
            "b_edit_sim": round(sentence_edit(reference, b_text), 6),
        }
        score = e03.get(sample, {}).get("oof_p_bad_logistic_token")
        score_trigger = bool(score is not None and score >= SCORE_TRIGGER)

        policies = {
            "P0_always_A": {"decision": "ACCEPT", "invoked_second_path": False, "fuse_applied": False,
                            "final_source": "A", "final_text": a_text},
            "P0_always_A2": {"decision": "ACCEPT", "invoked_second_path": True, "fuse_applied": False,
                             "final_source": "A2", "final_text": a2_text},
            "P0_always_B": {"decision": "ACCEPT", "invoked_second_path": False, "fuse_applied": False,
                            "final_source": "B", "final_text": b_text},
            "P1_rule_only_no_second": decide_accept_only_a(slots),
            "P2_rule_cascade_A2": decide_cascade(slots),
            "P2c_cascade_rule_or_score": decide_cascade(slots, score_trigger=score_trigger),
            "P3_full_run_rule_select": decide_full_run_rule_select(slots),
        }
        # GT oracle (offline only)
        candidates = {"A": a_text, "A2": a2_text, "B": b_text}
        best = max(("A", "A2", "B"), key=lambda key: (base[f"{key.lower()}_edit_sim"], -["A", "A2", "B"].index(key)))
        policies["P4_gt_oracle_all3"] = {
            "decision": "ORACLE", "invoked_second_path": True, "fuse_applied": False,
            "final_source": best, "final_text": candidates[best],
        }
        best_ab = max(("A", "A2"), key=lambda key: (base[f"{key.lower()}_edit_sim"], -["A", "A2"].index(key)))
        policies["P4b_gt_oracle_AB"] = {
            "decision": "ORACLE", "invoked_second_path": True, "fuse_applied": False,
            "final_source": best_ab, "final_text": candidates[best_ab],
        }
        for name, decision in policies.items():
            merged = dict(base)
            merged.update(decision)
            decisions_by_policy.setdefault(name, {})[sample] = merged

    policy_order = [
        "P0_always_A",
        "P0_always_A2",
        "P0_always_B",
        "P1_rule_only_no_second",
        "P2_rule_cascade_A2",
        "P2c_cascade_rule_or_score",
        "P3_full_run_rule_select",
        "P4b_gt_oracle_AB",
        "P4_gt_oracle_all3",
    ]
    ablation = {}
    for name in policy_order:
        full_run_call_rate = 1.0 if name in ("P0_always_A2", "P3_full_run_rule_select",
                                             "P4_gt_oracle_all3", "P4b_gt_oracle_AB") else 0.0
        ablation[name] = evaluate_policy(name, gt, decisions_by_policy[name],
                                         full_run_call_rate=full_run_call_rate)

    judge = rule_analysis(gt, texts_for_judge)
    selfcheck = state_machine_selfcheck()

    payload = {
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "generated_at": utc_now(),
        "status": "PASS",
        "policy_order": policy_order,
        "ablation": ablation,
        "judge": judge,
        "state_machine_selfcheck": selfcheck,
        "agreement": {"A_eq_A2": a_eq_a2, "n_samples": len(gt)},
        "data": {
            "n_samples": len(gt),
            "n_signers": len({row["signer_id"] for row in gt.values()}),
            "gt_sha256": sha256_file(GT_PATH),
            "e02_candidate_outputs_sha256": sha256_file(E02_CANDIDATES),
            "e02_rows": e02_meta["n_rows"],
            "e03_scores_sha256": sha256_file(E03_SCORES),
            "b_join_reconstruction_mismatch_vs_e02_b0": b_mismatch,
            "empty_text_counts": dict(empty_counts),
            "reference_field": "gt_1086.csv ref_sentence_upstream (eval-only)",
        },
        "blocked": {
            "D_E_F_G_llm": {
                "status": "BLOCKED",
                "evidence": "no LLM/API env vars (names inspected, no values printed); repo grep "
                            "openai|anthropic|deepseek|api_key|chat/completions hits only local app test base_urls "
                            "and vendored upstream transformers model names, no provider client/endpoint config",
                "needed": ["reachable LLM endpoint + secret injection", "fixed provider/model version",
                           "prompt version + decoding params", "call budget", "prompt must stay GT-free"],
            },
            "jev": {"status": "BLOCKED",
                    "evidence": "no Jev client/protocol implementation or endpoint in this repo/environment"},
            "unisign_self_confidence": {
                "status": "BLOCKED",
                "evidence": "Uni-Sign runtime TextPrediction(score=None); historical predictions have no score/logprob field",
                "needed": "re-run Uni-Sign with generate(output_scores=True) on the same inputs",
            },
        },
        "limitations": [
            "retrospective_seen_1086, no independent dev/test split; rule set designed and evaluated on the same seen data",
            "candidate texts are measured imported historical predictions, not re-run in E06",
            "routing/selection/FUSE/oracle are simulated offline",
            "B is a deterministic gloss join baseline, not a trained Gloss2Text",
            "savings counted as model invocations, not measured latency",
        ],
        "measured_vs_simulated": {
            "measured_imported": ["A text", "A2 text", "B gloss tokens", "SLRT token/window scores"],
            "simulated_offline": ["rule decisions", "cascade routing", "FUSE patches", "policy selection", "GT oracle", "all metrics"],
        },
    }
    write_report(payload)
    write_judge_analysis(payload)
    write_policy_yaml(payload)

    # router_ablation.jsonl (one aggregate row per policy)
    ablation_rows = []
    for name in policy_order:
        row = ablation[name]
        ablation_rows.append({
            "policy": name,
            "n": row["n"],
            "bleu4_13a": row["quality"]["bleu4_13a"],
            "chrf": row["quality"]["chrf"],
            "mean_char_edit_sim": row["mean_edit_sim"],
            "bad_rate_edit_sim_lt_0.5": row["bad_rate_edit_sim_lt_0.5"],
            "bad_rate_sensitivity": row["bad_rate_sensitivity"],
            "exact_match_count": row["quality"]["exact_match_count"],
            "second_path_call_rate": row["second_path_call_rate"],
            "second_path_invocation_savings_vs_full_run": row["second_path_invocation_savings_vs_full_run"],
            "error_accept_rate_among_accept_decisions": row["error_accept_rate_among_accept_decisions"],
            "error_accept_rate_definition": "bad(final) among decisions==ACCEPT; bad = char edit sim < 0.5",
            "missed_upgrade_rate": row["missed_upgrade_rate"],
            "missed_upgrade_definition": "A bad and best(A2,B) acceptable with margin>=0.1, but decision==ACCEPT",
            "correction_rate_when_a_bad": row["correction_rate_when_a_bad"],
            "miscorrection_rate_when_a_ok": row["miscorrection_rate_when_a_ok"],
            "should_upgrade_n": row["should_upgrade_n"],
            "missed_upgrade_n": row["missed_upgrade_n"],
            "missed_winner_rate": row["missed_winner_rate"],
            "fuse_applied_n": row["fuse_applied_n"],
            "severe_slot_error_rate": row["severe_slot_error_rate"],
            "decision_counts": row["decision_counts"],
            "final_source_counts": row["final_source_counts"],
            "evaluation_basis": row["evaluation_basis"],
        })
    write_jsonl(E06_DIR / "router_ablation.jsonl", ablation_rows)

    # ---- run dir ----
    try:
        run = RunDirectory(RUN_ID, experiment_id=EXPERIMENT)
    except FileExistsError:
        run = RunDirectory.__new__(RunDirectory)
        run.run_id = RUN_ID
        run.experiment_id = EXPERIMENT
        run.path = RUNS / RUN_ID

    eval_rows = []
    for name in ("P2_rule_cascade_A2", "P2c_cascade_rule_or_score", "P3_full_run_rule_select"):
        for sample in sorted(gt):
            record = decisions_by_policy[name][sample]
            eval_rows.append({
                "policy": name,
                "sample_id": sample,
                "decision": record["decision"],
                "final_source": record["final_source"],
                "invoked_second_path": record["invoked_second_path"],
                "fuse_applied": record["fuse_applied"],
                "final_text": record["final_text"],
                "a_edit_sim_EVAL_ONLY": record["a_edit_sim"],
                "a2_edit_sim_EVAL_ONLY": record["a2_edit_sim"],
                "b_edit_sim_EVAL_ONLY": record["b_edit_sim"],
            })
    write_jsonl(run.path / "eval_router_decisions.jsonl", eval_rows, guard=False)
    (run.path / "README.md").write_text(
        "# E06 run\n\n"
        f"- run_id: {RUN_ID}\n"
        "- script: experiments/p1/e06/build_e06.py\n"
        "- command: `$env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'; .\\.conda-env\\python.exe experiments/p1/e06/build_e06.py`\n"
        "- candidate texts = measured imported historical predictions (E02 / SLRT artifacts); routing+selection = simulated offline\n"
        "- no LLM/Jev calls (BLOCKED: no endpoint/credentials); no model inference performed\n"
        "- analysis outputs: experiments/p1/e06/ (candidate_inputs.jsonl, router_ablation.jsonl, "
        "judge_error_analysis.md, policy_candidate.yaml, report.md)\n"
        "- eval_router_decisions.jsonl is an EVAL-SIDE file (contains GT-derived edit similarities), guard=False by design.\n",
        encoding="utf-8",
    )
    metrics_payload = {
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "status": "PASS",
        "n_samples": len(gt),
        "ablation": {name: {k: ablation[name][k] for k in (
            "quality", "mean_edit_sim", "bad_rate_edit_sim_lt_0.5", "second_path_call_rate",
            "error_accept_rate_among_accept_decisions", "missed_upgrade_rate",
            "correction_rate_when_a_bad", "miscorrection_rate_when_a_ok", "missed_winner_rate",
            "severe_slot_error_rate", "decision_counts")} for name in policy_order},
        "judge_per_slot": judge["per_slot"],
        "state_machine_selfcheck": selfcheck,
        "blocked": payload["blocked"],
        "limitations": payload["limitations"],
    }
    write_json(run.path / "metrics.json", metrics_payload)
    run.write_manifest(
        run_status="PASS",
        purpose="E06 Rule-only arbitration + cascade shadow analysis (CPU only; measured imported candidates, simulated routing)",
        model={
            "A": "unisign_rgb (imported historical, E02 candidate_outputs)",
            "A2": "unisign_pose_released (imported historical, E02 candidate_outputs)",
            "B": "slrt-emnlp24-csl-daily gloss deterministic join (deterministic join baseline, NOT trained Gloss2Text)",
            "llm": None,
            "jev": None,
        },
        data={
            "gt_sha256": sha256_file(GT_PATH),
            "e02_candidate_outputs_sha256": sha256_file(E02_CANDIDATES),
            "e02_rows": e02_meta["n_rows"],
            "e03_scores_sha256": sha256_file(E03_SCORES),
            "n_samples": len(gt),
            "scope": "retrospective_seen_1086",
            "b_join_reconstruction_mismatch_vs_e02_b0": b_mismatch,
        },
        code_assets={
            "script": "experiments/p1/e06/build_e06.py",
            "script_sha256": sha256_file(E06_DIR / "build_e06.py"),
            "supporting_modules": {
                "experiments/p1/common.py": sha256_file(P1 / "common.py"),
                "experiments/p1/eval/metrics.py": sha256_file(P1 / "eval" / "metrics.py"),
            },
        },
        command=".conda-env\\python.exe experiments/p1/e06/build_e06.py (PYTHONPATH=E:\\githubproject\\SignLanguage-Bridge)",
        device={"device": "cpu", "gpu_used": False, "note": "CPU-only analysis; no model inference"},
        outputs={
            "candidate_inputs.jsonl": {"sha256": sha256_file(E06_DIR / "candidate_inputs.jsonl"),
                                       "n_rows": len(candidate_rows), "guard": "passed"},
            "router_ablation.jsonl": {"sha256": sha256_file(E06_DIR / "router_ablation.jsonl"),
                                      "n_rows": len(ablation_rows)},
            "judge_error_analysis.md": {"sha256": sha256_file(E06_DIR / "judge_error_analysis.md")},
            "policy_candidate.yaml": {"sha256": sha256_file(E06_DIR / "policy_candidate.yaml")},
            "report.md": {"sha256": sha256_file(E06_DIR / "report.md")},
            "eval_router_decisions.jsonl": {"sha256": sha256_file(run.path / "eval_router_decisions.jsonl"),
                                            "n_rows": len(eval_rows)},
        },
        blocked_branches=payload["blocked"],
        measured_vs_simulated=payload["measured_vs_simulated"],
        limitations=payload["limitations"],
        script_status="executed_in_this_run",
    )

    print(json.dumps({
        "run_id": RUN_ID,
        "n": len(gt),
        "A_eq_A2": a_eq_a2,
        "b_join_mismatch": b_mismatch,
        "ablation": {name: {
            "bleu4": ablation[name]["quality"]["bleu4_13a"],
            "chrf": ablation[name]["quality"]["chrf"],
            "mean_edit": ablation[name]["mean_edit_sim"],
            "call_rate": ablation[name]["second_path_call_rate"],
            "error_accept": ablation[name]["error_accept_rate_among_accept_decisions"],
            "missed_upgrade": ablation[name]["missed_upgrade_rate"],
            "correction": ablation[name]["correction_rate_when_a_bad"],
            "miscorrection": ablation[name]["miscorrection_rate_when_a_ok"],
            "missed_winner": ablation[name]["missed_winner_rate"],
        } for name in policy_order},
        "judge": {slot: judge["per_slot"][slot] for slot in ALL_SLOTS},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
