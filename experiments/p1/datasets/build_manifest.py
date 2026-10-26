"""Build the P-1 dataset manifests (GT-free) from the frozen 1086 snapshot.

Outputs (under experiments/p1/datasets/):

* ``manifest_1086.csv``      sample index only: ids, signer, frames, paths, hashes.
                            NEVER contains reference sentence / gloss / WER.
* ``pilot_50.csv``           the plan's 50 diagnostic samples with resolved paths.
* ``source_meta.json``       hashes and counts of every source file used.

The GT (reference sentence, reference gloss) is built separately into
``experiments/p1/references/`` by ``build_references.py`` and is documented as
evaluation-only.  Do not merge the two directories.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from experiments.p1 import common

SIGNER_RE = re.compile(r"_P(\d{4})_")


def signer_of(sample_id: str) -> str:
    match = SIGNER_RE.search(sample_id)
    if not match:
        raise ValueError(f"cannot parse signer from sample id {sample_id!r}")
    return f"P{match.group(1)}"


def parse_upstream_refs(path: Path) -> dict[str, str]:
    """Parse ``sample: X, ground-truth: 他 今 年 四 岁 。`` lines."""

    refs: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("sample:") or ", ground-truth:" not in line:
            continue
        sample_part, ref = line.split(", ground-truth:", 1)
        sample = sample_part[len("sample:") :].strip()
        refs[sample] = ref.strip()
    return refs


def build(force: bool = False, hash_videos: int = 0) -> dict[str, object]:
    rows = common.read_csv_rows(common.COMBINED_CSV)
    if len(rows) != 1086:
        raise SystemExit(f"expected 1086 rows in {common.COMBINED_CSV}, found {len(rows)}")

    delta_file = common.SHARED_ARTIFACTS / "delta-samples-459.txt"
    delta = {
        line.strip()
        for line in delta_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if delta_file.is_file() else set()

    video_missing = 0
    rows_out: list[dict[str, object]] = []
    for row in rows:
        sample = row["sample"].strip()
        raw_video = common.VIDEO_DIR / f"{sample}.mp4"
        trimmed = common.SHARED_ARTIFACTS / "trimmed" / f"{sample}.mp4"
        npz = common.SHARED_ARTIFACTS / "pose" / f"{sample}.npz"
        present = raw_video.is_file()
        if not present:
            video_missing += 1
        trimmed_sha = None
        if npz.is_file():
            try:
                import numpy as np

                with np.load(npz, allow_pickle=False) as data:
                    if "metadata_json" in data:
                        import json as _json

                        meta = _json.loads(str(data["metadata_json"]))
                        trimmed_sha = meta.get("video_sha256")
            except Exception:  # noqa: BLE001 - hash is optional provenance
                trimmed_sha = None
        rows_out.append(
            {
                "sample_id": sample,
                "signer_id": signer_of(sample),
                "split": "test",
                "source_dataset": "CSL-Daily",
                "analysis_scope": "retrospective_seen_1086",
                "cohort": "added_459" if sample in delta else "existing_627",
                "frames": int(float(row["frames"])) if row.get("frames") else None,
                "raw_video_path": str(raw_video),
                "raw_video_present": present,
                "trimmed_video_path": str(trimmed),
                "trimmed_video_present": trimmed.is_file(),
                "trimmed_video_sha256": trimmed_sha,
                "pose_npz_path": str(npz),
                "pose_npz_present": npz.is_file(),
            }
        )

    fieldnames = list(rows_out[0].keys())
    common.write_csv_rows(common.DATASETS / "manifest_1086.csv", fieldnames, rows_out)

    # -- video hashes (optional; the plan asks for raw_video_sha256) ----------
    if hash_videos:
        hash_rows: list[dict[str, object]] = []
        for row in rows_out:
            if not row["raw_video_present"]:
                continue
            if hash_videos > 0 and len(hash_rows) >= hash_videos:
                break
            row_hash = {
                "sample_id": row["sample_id"],
                "raw_video_path": row["raw_video_path"],
                "raw_video_sha256": common.sha256_file(Path(str(row["raw_video_path"]))),
                "raw_video_bytes": Path(str(row["raw_video_path"])).stat().st_size,
            }
            hash_rows.append(row_hash)
        common.write_csv_rows(
            common.DATASETS / "video_sha256.csv",
            ["sample_id", "raw_video_path", "raw_video_sha256", "raw_video_bytes"],
            hash_rows,
        )

    # -- pilot 50 ------------------------------------------------------------
    pilot_source = common.PLAN_DIR / "data" / "pilot_50_exploratory.csv"
    pilot_rows = common.read_csv_rows(pilot_source)
    if len(pilot_rows) != 50:
        raise SystemExit(f"expected 50 pilot rows, found {len(pilot_rows)}")
    by_sample = {row["sample_id"]: row for row in rows_out}
    pilot_out: list[dict[str, object]] = []
    for pilot in pilot_rows:
        sample = pilot["sample_id"].strip()
        if sample not in by_sample:
            raise SystemExit(f"pilot sample {sample} missing from the 1086 manifest")
        manifest_row = by_sample[sample]
        pilot_out.append(
            {
                "sample_id": sample,
                "signer_id": pilot["signer_id"].strip(),
                "frames": int(float(pilot["frames"])) if pilot.get("frames") else None,
                "cohort": pilot["cohort"].strip(),
                "usage": pilot.get("usage", "retrospective_diagnostic_only"),
                "raw_video_path": manifest_row["raw_video_path"],
                "raw_video_present": manifest_row["raw_video_present"],
                "trimmed_video_path": manifest_row["trimmed_video_path"],
                "pose_npz_path": manifest_row["pose_npz_path"],
                "fps_measured": None,  # filled by E01 video probe
            }
        )
    common.write_csv_rows(
        common.DATASETS / "pilot_50.csv",
        [
            "sample_id",
            "signer_id",
            "frames",
            "cohort",
            "usage",
            "raw_video_path",
            "raw_video_present",
            "trimmed_video_path",
            "pose_npz_path",
            "fps_measured",
        ],
        pilot_out,
    )

    meta = {
        "created_at": common.utc_now(),
        "created_by": "experiments/p1/datasets/build_manifest.py",
        "gt_free": True,
        "sources": {
            "comparison_csv": str(common.COMBINED_CSV),
            "comparison_sha256": common.sha256_file(common.COMBINED_CSV),
            "upstream_refs": str(common.UPSTREAM_REFS),
            "upstream_refs_sha256": common.sha256_file(common.UPSTREAM_REFS),
            "delta_samples_459": str(delta_file),
            "delta_sha256": common.sha256_file(delta_file) if delta_file.is_file() else None,
            "pilot_csv": str(pilot_source),
            "pilot_sha256": common.sha256_file(pilot_source),
        },
        "counts": {
            "manifest_1086": len(rows_out),
            "unique_samples": len({row["sample_id"] for row in rows_out}),
            "raw_videos_present": sum(1 for row in rows_out if row["raw_video_present"]),
            "raw_videos_missing": video_missing,
            "trimmed_clips_present": sum(1 for row in rows_out if row["trimmed_video_present"]),
            "pose_npz_present": sum(1 for row in rows_out if row["pose_npz_present"]),
            "pilot_50": len(pilot_out),
            "pilot_50_videos_present": sum(1 for row in pilot_out if row["raw_video_present"]),
        },
    }
    common.write_json(common.DATASETS / "source_meta.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="kept for symmetry; always rewrites")
    parser.add_argument(
        "--hash-videos",
        type=int,
        default=0,
        help="hash N raw videos into video_sha256.csv (0 = skip, -1 = all)",
    )
    args = parser.parse_args(argv)
    meta = build(force=args.force, hash_videos=args.hash_videos)
    print(common.json.dumps(meta["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
