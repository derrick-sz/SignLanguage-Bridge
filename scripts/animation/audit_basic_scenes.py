"""Audit scene coverage and real full-expression motions without running ML."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slbridge.animation.corpus_store import CorpusStore
from slbridge.animation.scenes import SceneLibrary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--definitions", type=Path,
                        default=Path(__file__).resolve().parents[2] / "slbridge/animation/basic_scenes_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    store = CorpusStore(args.corpus)
    library = SceneLibrary(store, args.definitions)
    coverage = library.coverage()
    catalog = library.catalog(limit=1)
    checked, failures = [], []
    for scene in catalog["scenes"]:
        candidates = library.catalog(scene=scene["id"], limit=8)["assets"]
        for asset in candidates:
            try:
                plan = library.plan(asset["text"], confirmed=True, research_preview=True)
                assert plan["can_play"], plan["reason"]
                motion = library.motion(plan)
                clip = motion["clip"]
                poses = np.asarray(clip["poses"])
                assert poses.shape == (clip["frames"], 55, 3)
                assert np.isfinite(poses).all()
                assert len(motion["parents"]) == 55
                assert len(motion["restJoints"]) == 55
                checked.append({"scene": scene["id"], "sample_id": plan["items"][0]["sample_id"],
                                "text": asset["text"], "frames": clip["frames"],
                                "motion_sha256": motion["sha256"], "numeric_check": "PASS",
                                "language_review": "NOT_RUN"})
            except Exception as exc:
                failures.append({"scene": scene["id"], "sample_id": asset["sample_id"],
                                 "error": f"{type(exc).__name__}: {exc}"})

    sequence_texts = [
        "你好！请坐。你想喝什么？",
        "你好！我想喝茶。谢谢！",
        "对不起！我不懂你在讲什么。请帮助我！",
        "你好！我是手语翻译。很高兴认识你。",
        "请帮助我！我丢了钥匙。谢谢！",
        "我身体不舒服，想去医院。我不得不请假。",
    ]
    sequences = []
    for text in sequence_texts:
        try:
            plan = library.plan(text, confirmed=True, research_preview=True)
            assert plan["can_play"], plan["reason"]
            motion = library.motion(plan)
            clip = motion["clip"]
            poses = np.asarray(clip["poses"])
            expected_frames = sum(item["frames"] for item in plan["items"]) + 9 * (len(plan["items"]) - 1)
            assert poses.shape == (expected_frames, 55, 3)
            cursor = 0
            for item in plan["items"]:
                source = store._parameters(store._row(item["sample_id"]))
                assert np.array_equal(poses[cursor:cursor + len(source)], source)
                cursor += len(source) + 9
            sequences.append({"text": text, "mode": plan["mode"], "frames": clip["frames"],
                              "items": plan["items"], "source_frames_unchanged": True,
                              "motion_sha256": motion["sha256"], "language_review": "NOT_RUN",
                              "naturalness_review": "NOT_RUN"})
        except Exception as exc:
            failures.append({"sequence": text, "error": f"{type(exc).__name__}: {exc}"})

    negatives = []
    for text in ("你好！请帮助我去火星。", "我叫纳米火箭。", "我明天下午三点去火星。", "你好？"):
        plan = library.plan(text, confirmed=True, research_preview=True)
        blocked = not plan["can_play"] and plan["status"] == "blocked"
        negatives.append({"text": text, "blocked": blocked, "unmatched": plan["unmatched"]})
        if not blocked:
            failures.append({"negative": text, "error": "Unexpected playable plan"})

    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "read-only corpus index and selected numeric motions; no ML or language acceptance",
              "definition_sha256": hashlib.sha256(args.definitions.read_bytes()).hexdigest(),
              "catalog": catalog["scenes"], "checked_assets": checked, "sequences": sequences,
              "negative_controls": negatives, "failures": failures,
              "numeric_status": "PASS" if not failures else "FAIL", "language_review": "NOT_RUN"}
    for name, data in (("coverage.json", coverage), ("numeric-audit.json", report)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"numeric_status": report["numeric_status"], "checked_assets": len(checked),
                      "sequences": len(sequences), "negative_controls": len(negatives),
                      "failures": len(failures), "output": str(args.output)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
