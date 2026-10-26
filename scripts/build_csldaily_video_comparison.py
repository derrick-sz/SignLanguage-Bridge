"""Combine GT (sentence + Gloss) with Uni-Sign and SLRT predictions per sample.

Three systems run on the same local CSL-Daily test videos:

1. **Uni-Sign, video end-to-end** — cropped sentence clip -> pinned RTMPose ->
   ``csl_daily_pose_only_slt.pth`` -> Chinese sentence (SLT).
2. **SLRT, video + official HRNet pose** — same cropped clip -> pinned SLRT
   online Gloss model -> Gloss sequence (SLR).
3. **(optional) Uni-Sign, released pose archive** — official test pose -> the
   same checkpoint; this is the paper-protocol reference on the same samples.

The tasks are different (translation vs Gloss recognition), so the table is a
side-by-side per-sample view, *not* a leaderboard: Uni-Sign is scored with
BLEU/chrF/char-ROUGE/edit similarity, SLRT with token-level WER.

Nothing is recomputed here; every number comes from the per-sample records of the
three evaluation runs, and the paths/SHAs are recorded in the summary.

Usage:

    .\\.conda-env\\python.exe scripts\\build_csldaily_video_comparison.py `
        --selection artifacts\\csldaily-video-test-627-20261026\\selection.json `
        --unisign-video-summary artifacts\\csldaily-video-test-627-20261026\\unisign-video\\summary.json `
        --slrt-summary artifacts\\csldaily-video-test-627-20261026\\summary.json `
        --unisign-pose-summary artifacts\\cslnews-eval\\csldaily-test-627-pose-20261026\\summary.json `
        --output-dir artifacts\\csldaily-video-test-627-20261026\\combined
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--unisign-video-summary", type=Path, required=True)
    parser.add_argument("--slrt-summary", type=Path, required=True)
    parser.add_argument("--unisign-pose-predictions", type=Path, default=None,
                        help="predictions.jsonl from evaluate_unisign_csldaily_pose.py")
    parser.add_argument("--unisign-pose-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    selection = read_json(args.selection)
    samples = selection["samples"]
    by_name = {item["sample"]: item for item in samples}

    unisign_video = read_json(args.unisign_video_summary)
    uv_rows = {row["sample"]: row for row in unisign_video.get("per_sample", [])}
    slrt = read_json(args.slrt_summary)
    slrt_rows = {row["sample"]: row for row in slrt.get("per_sample", [])}
    unisign_pose = read_json(args.unisign_pose_summary) if args.unisign_pose_summary else None
    up_rows = {}
    if args.unisign_pose_predictions and Path(args.unisign_pose_predictions).is_file():
        for line in Path(args.unisign_pose_predictions).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                up_rows[row["sample"]] = row

    rows = []
    for index, item in enumerate(samples, 1):
        name = item["sample"]
        uv = uv_rows.get(name, {})
        up = up_rows.get(name, {})
        sl = slrt_rows.get(name, {})
        alignment_px = sl.get("best_mean_px")
        rows.append({
            "index": index,
            "sample": name,
            "frames": item["num_frames"],
            "gt_sentence": item["reference"],
            "gt_gloss": item["gloss"],
            "unisign_video_prediction": uv.get("prediction", ""),
            "unisign_video_edit": uv.get("char_edit_similarity"),
            "unisign_video_rouge": uv.get("rouge_l_f1_char"),
            "unisign_video_exact": uv.get("exact_match"),
            "unisign_video_status": uv.get("status", "missing"),
            "unisign_pose_prediction": up.get("prediction", ""),
            "unisign_pose_edit": (_edit(up) if up.get("status") == "ok" else None),
            "unisign_pose_exact": (_exact(up) if up.get("status") == "ok" else None),
            "unisign_pose_status": up.get("status", "missing"),
            "slrt_prediction": " ".join(sl.get("predicted") or []),
            "slrt_wer": sl.get("wer"),
            "slrt_s": sl.get("substitutions"),
            "slrt_d": sl.get("deletions"),
            "slrt_i": sl.get("insertions"),
            "slrt_status": sl.get("status", "missing"),
            "alignment_start": sl.get("best_start"),
            "alignment_px": alignment_px,
            "alignment_low_confidence": sl.get("low_confidence"),
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = list(rows[0])
    csv_path = output_dir / "comparison-both.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    uv_metrics = unisign_video.get("metrics", {})
    up_metrics = unisign_pose.get("metrics", {}) if unisign_pose else {}
    if up_rows and not up_metrics:
        pose_edits = [row["unisign_pose_edit"] for row in rows
                      if row["unisign_pose_status"] == "ok" and row["unisign_pose_edit"] is not None]
        up_metrics = {
            "mean_char_edit_similarity": statistics.fmean(pose_edits) if pose_edits else None,
            "exact_match_count": sum(1 for row in rows if row["unisign_pose_exact"]),
            "note": "computed from predictions.jsonl; BLEU/chrF were not recomputed here",
        }
    slrt_success = [row for row in rows if row["slrt_status"] == "final"]
    wer_values = [row["slrt_wer"] for row in slrt_success if row["slrt_wer"] is not None]
    uv_ok = [row for row in rows if row["unisign_video_status"] == "ok"]
    uv_edits = [row["unisign_video_edit"] for row in uv_ok if row["unisign_video_edit"] is not None]
    up_edits = [row["unisign_pose_edit"] for row in rows
                if row["unisign_pose_status"] == "ok" and row["unisign_pose_edit"] is not None]
    paired = [(_edit(up), uv.get("char_edit_similarity"))
              for name in by_name
              if (up := up_rows.get(name)) and (uv := uv_rows.get(name))
              and up.get("status") == "ok" and uv.get("status") == "ok"]
    alignment_distances = [row["alignment_px"] for row in rows if row["alignment_px"] is not None]
    correlation = None
    if len(alignment_distances) > 2 and len(wer_values) > 2:
        pairs = [(row["alignment_px"], row["slrt_wer"]) for row in rows
                 if row["alignment_px"] is not None and row["slrt_wer"] is not None]
        if len(pairs) > 2:
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            denom = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
            correlation = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
                           if denom else None)

    summary = {
        "scope": "Local CSL-Daily test videos: Uni-Sign (video end-to-end) vs SLRT (video + official "
                 "HRNet keypoints) on identical cropped sentence clips, with GT sentence and GT Gloss",
        "count": len(rows),
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "inputs": {
            "unisign_video_summary": str(args.unisign_video_summary.resolve()),
            "unisign_video_summary_sha256": sha256(args.unisign_video_summary),
            "slrt_summary": str(args.slrt_summary.resolve()),
            "slrt_summary_sha256": sha256(args.slrt_summary),
            "unisign_pose_summary": (str(args.unisign_pose_summary.resolve())
                                     if args.unisign_pose_summary else None),
        },
        "unisign_video": {
            "successful": len(uv_ok),
            "metrics": uv_metrics,
            "mean_char_edit_similarity_recomputed": statistics.fmean(uv_edits) if uv_edits else None,
        },
        "unisign_pose": ({"successful": sum(1 for row in rows if row["unisign_pose_status"] == "ok"),
                          "metrics": up_metrics} if (unisign_pose or up_rows) else None),
        "unisign_pose_minus_video_edit_delta": (
            statistics.fmean(a - b for a, b in paired) if paired else None),
        "unisign_pose_vs_video_paired_n": len(paired),
        "slrt": {
            "successful": len(slrt_success),
            "corpus_wer": slrt.get("corpus_wer"),
            "successful_subset_wer": slrt.get("successful_subset_wer"),
            "total_substitutions": slrt.get("total_substitutions"),
            "total_insertions": slrt.get("total_insertions"),
            "total_deletions": slrt.get("total_deletions"),
            "per_sample_wer_mean": statistics.fmean(wer_values) if wer_values else None,
            "per_sample_wer_median": statistics.median(wer_values) if wer_values else None,
            "alignment_px_mean": statistics.fmean(alignment_distances) if alignment_distances else None,
            "alignment_wer_pearson": correlation,
        },
        "note": "Uni-Sign is translation (BLEU/chrF/ROUGE/edit similarity); SLRT is Gloss "
                "recognition (token WER). Metrics are not comparable across the two tasks; "
                "the per-sample table is the comparison. SLRT alignment is a local approximation "
                "because the official sentence_frames-512x512.zip is not available.",
        "per_sample": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CSL-Daily test: GT vs Uni-Sign vs SLRT",
        "",
        f"Scope: {summary['scope']}",
        "",
        "## Summary",
        "",
        "| item | Uni-Sign video end-to-end | Uni-Sign released pose | SLRT |",
        "|---|---:|---:|---:|",
        f"| samples succeeded | {summary['unisign_video']['successful']} / {summary['count']} | "
        f"{(summary['unisign_pose'] or {}).get('successful', '-')} | {summary['slrt']['successful']} / {summary['count']} |",
        f"| primary metric | BLEU-4 {fmt(uv_metrics.get('bleu4_13a'))} | BLEU-4 {fmt(up_metrics.get('bleu4_13a'))} | "
        f"WER {fmt(summary['slrt']['corpus_wer'], 4)} |",
        f"| chrF | {fmt(uv_metrics.get('chrf'))} | {fmt(up_metrics.get('chrf'))} | - |",
        f"| char ROUGE-L F1 | {fmt(uv_metrics.get('rouge_l_f1_char'))} | {fmt(up_metrics.get('rouge_l_f1_char'))} | - |",
        f"| mean char edit similarity | {fmt(uv_metrics.get('mean_char_edit_similarity'))} | "
        f"{fmt(up_metrics.get('mean_char_edit_similarity'))} | - |",
        f"| exact matches | {uv_metrics.get('exact_match_count', '-')} | {up_metrics.get('exact_match_count', '-')} | - |",
        f"| S / D / I | - | - | {summary['slrt']['total_substitutions']} / "
        f"{summary['slrt']['total_deletions']} / {summary['slrt']['total_insertions']} |",
        "",
        "Notes:",
        f"- {summary['note']}",
        f"- Uni-Sign released pose minus video end-to-end edit similarity (paired n={summary['unisign_pose_vs_video_paired_n']}): "
        f"{fmt(summary['unisign_pose_minus_video_edit_delta'])}",
        f"- SLRT alignment distance mean: {fmt(summary['slrt']['alignment_px_mean'], 2)} px; "
        f"alignment–WER Pearson: {fmt(summary['slrt']['alignment_wer_pearson'], 3)}",
        "",
        "## Per-sample table",
        "",
        "| # | sample | frames | GT sentence | Uni-Sign (video) | edit | GT Gloss | SLRT Gloss | WER | align px |",
        "|---:|---|---:|---|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['sample']} | {row['frames']} | {row['gt_sentence']} | "
            f"{row['unisign_video_prediction'] or '*(empty)*'} | {fmt(row['unisign_video_edit'])} | "
            f"{row['gt_gloss']} | {row['slrt_prediction'] or '*(empty)*'} | {fmt(row['slrt_wer'])} | "
            f"{fmt(row['alignment_px'], 2)} |")
    failure_rows = [row for row in rows
                    if row["unisign_video_status"] != "ok" or row["slrt_status"] != "final"]
    lines += ["", "## Failures / missing", ""]
    if failure_rows:
        lines += [f"- `{row['sample']}`: unisign-video={row['unisign_video_status']} "
                  f"slrt={row['slrt_status']}" for row in failure_rows]
    else:
        lines.append("- none")
    (output_dir / "comparison-both.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": len(rows),
        "unisign_video_ok": len(uv_ok),
        "slrt_ok": len(slrt_success),
        "unisign_video_bleu4": uv_metrics.get("bleu4_13a"),
        "slrt_corpus_wer": slrt.get("corpus_wer"),
    }, ensure_ascii=False, indent=2))
    return 0


def _norm(text) -> str:
    import unicodedata
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def _edit(row):
    """Character edit similarity between a row's reference and prediction."""
    a, b = _norm(row.get("reference", "")), _norm(row.get("prediction", ""))
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / max(1, len(a), len(b))


def _exact(row) -> bool:
    return _norm(row.get("reference", "")) == _norm(row.get("prediction", ""))


if __name__ == "__main__":
    raise SystemExit(main())
