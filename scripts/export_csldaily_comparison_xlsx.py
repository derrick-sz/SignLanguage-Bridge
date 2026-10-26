"""Export the combined CSL-Daily comparison to a formatted Excel workbook.

Input is the ``summary.json`` produced by ``scripts/build_csldaily_video_comparison.py``.
Two sheets are written:

* ``Summary`` — coverage, primary metrics per system and the protocol notes;
* ``PerSample`` — the GT sentence/Gloss and every prediction with its per-sample
  metric, frozen header row, autofilter and wrapped text columns.

The workbook is a convenience copy of the JSON/CSV artifacts; it is not a
measurement source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")

COLUMNS = [
    ("index", "#", 5, None),
    ("sample", "sample", 20, None),
    ("frames", "frames", 7, "0"),
    ("gt_sentence", "GT sentence", 42, None),
    ("unisign_video_prediction", "Uni-Sign (video) prediction", 42, None),
    ("unisign_video_edit", "Uni-Sign video edit sim", 10, "0.0000"),
    ("unisign_video_status", "Uni-Sign video status", 12, None),
    ("unisign_pose_prediction", "Uni-Sign (released pose) prediction", 42, None),
    ("unisign_pose_edit", "Uni-Sign pose edit sim", 10, "0.0000"),
    ("gt_gloss", "GT Gloss", 34, None),
    ("slrt_prediction", "SLRT prediction (Gloss)", 34, None),
    ("slrt_wer", "SLRT WER", 8, "0.000"),
    ("slrt_s", "S", 5, "0"),
    ("slrt_d", "D", 5, "0"),
    ("slrt_i", "I", 5, "0"),
    ("alignment_start", "align start", 9, "0"),
    ("alignment_px", "align px", 8, "0.00"),
    ("slrt_status", "SLRT status", 10, None),
]


def metric_row(label, value):
    return [label, value]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    uvv = summary.get("unisign_video", {})
    uv = uvv.get("metrics", {}) or {}
    pose = summary.get("unisign_pose") or {}
    up = (pose.get("metrics") or {}) if pose else {}
    slrt = summary.get("slrt", {}) or {}

    rows = [
        ["CSL-Daily test videos: GT vs Uni-Sign vs SLRT", ""],
        ["scope", summary.get("scope")],
        ["samples", summary.get("count")],
        ["Uni-Sign successful (video)", uvv.get("successful")],
        ["Uni-Sign BLEU-4 (video)", uv.get("bleu4_13a")],
        ["Uni-Sign chrF (video)", uv.get("chrf")],
        ["Uni-Sign char ROUGE-L (video)", uv.get("rouge_l_f1_char")],
        ["Uni-Sign mean char edit similarity (video)", uv.get("mean_char_edit_similarity")],
        ["Uni-Sign exact matches (video)", uv.get("exact_match_count")],
        ["Uni-Sign successful (released pose)", pose.get("successful")],
        ["Uni-Sign BLEU-4 (released pose)", up.get("bleu4_13a")],
        ["Uni-Sign mean char edit similarity (released pose)", up.get("mean_char_edit_similarity")],
        ["paired pose minus video edit similarity", summary.get("unisign_pose_minus_video_edit_delta")],
        ["SLRT successful", slrt.get("successful")],
        ["SLRT corpus WER", slrt.get("corpus_wer")],
        ["SLRT successful-subset WER", slrt.get("successful_subset_wer")],
        ["SLRT S / D / I", f"{slrt.get('total_substitutions')} / {slrt.get('total_deletions')} / {slrt.get('total_insertions')}"],
        ["SLRT per-sample WER mean", slrt.get("per_sample_wer_mean")],
        ["SLRT per-sample WER median", slrt.get("per_sample_wer_median")],
        ["alignment distance mean (px)", slrt.get("alignment_px_mean")],
        ["alignment vs WER Pearson", slrt.get("alignment_wer_pearson")],
        ["note", summary.get("note")],
    ]
    for row in rows:
        sheet.append(row)
    sheet["A1"].font = Font(bold=True, size=14)
    for column in ("A", "B"):
        for cell in sheet[column]:
            cell.alignment = WRAP
    sheet.column_dimensions["A"].width = 46
    sheet.column_dimensions["B"].width = 110

    detail = workbook.create_sheet("PerSample")
    for index, (key, title, width, number_format) in enumerate(COLUMNS, 1):
        cell = detail.cell(row=1, column=index, value=title)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        detail.column_dimensions[get_column_letter(index)].width = width
    for row_index, row in enumerate(summary.get("per_sample", []), 2):
        for column_index, (key, title, width, number_format) in enumerate(COLUMNS, 1):
            value = row.get(key)
            cell = detail.cell(row=row_index, column=column_index, value=value)
            if number_format and isinstance(value, (int, float)):
                cell.number_format = number_format
            if key in ("gt_sentence", "unisign_video_prediction", "unisign_pose_prediction",
                       "gt_gloss", "slrt_prediction"):
                cell.alignment = WRAP
    detail.freeze_panes = "B2"
    detail.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{detail.max_row}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "per_sample_rows": len(summary.get("per_sample", []))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
