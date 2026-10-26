"""E01 step 6 input prep: bootstrap a pilot-50 SLRT artifact tree.

``scripts/align_slrt_csl_daily_subset.py --stage infer`` reads
``<artifact-dir>/trimmed/<sample>.mp4`` and ``<artifact-dir>/pose/<sample>.npz``
and writes ``<artifact-dir>/slrt/<sample>.json``.  Re-running the align stage
would re-extract RTMPose and re-cut every clip, so this copies the reviewed
alignment outputs (trimmed / pose / alignment) for the 50 pilot samples into a
fresh dir; only the inference stage runs there.  The copies are verified against
the NPZ metadata ``video_sha256`` before inference is allowed to start.

Also writes the filtered selection file the script's ``load_samples`` requires.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\prepare_s1_inputs.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402

RUN_ID = "e01-pilot50-s1-slrt-gloss-20261026"
SOURCE_ARTIFACT = common.SHARED_ARTIFACTS


def main() -> int:
    ids_path = common.P1 / "e01" / "inputs" / "pilot_50_ids.txt"
    wanted_order = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(wanted_order)
    run_dir = common.RUNS / RUN_ID
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    # ---- filtered selection ------------------------------------------------
    selection_path = SOURCE_ARTIFACT / "selection-1086.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_by_sample = {item["sample"]: item for item in selection["samples"]}
    missing = wanted - set(source_by_sample)
    if missing:
        raise SystemExit(f"pilot samples missing from selection-1086: {sorted(missing)[:5]}")
    filtered = {
        "count": len(wanted_order),
        "samples": [source_by_sample[sample] for sample in wanted_order],
        "source": str(selection_path),
        "source_sha256": common.sha256_file(selection_path),
        "filter": "pilot_50_ids.txt order",
        "note": "same sample entries as selection-1086.json (num_frames/gloss cross-checked by the script)",
    }
    selection_out = inputs / "selection-pilot50.json"
    common.write_json(selection_out, filtered)

    # ---- artifact tree -----------------------------------------------------
    artifact = run_dir / "artifact"
    copies = []
    for sub, extension in (("trimmed", ".mp4"), ("pose", ".npz"), ("alignment", ".json")):
        destination = artifact / sub
        destination.mkdir(parents=True, exist_ok=True)
        for sample in wanted_order:
            source = SOURCE_ARTIFACT / sub / f"{sample}{extension}"
            if not source.is_file():
                raise SystemExit(f"missing source artifact: {source}")
            target = destination / source.name
            if not target.is_file():
                shutil.copy2(source, target)
            copies.append({"kind": sub, "sample": sample, "source": str(source),
                           "target": str(target), "sha256": common.sha256_file(target)})

    # ---- verify the copied clip still matches the NPZ metadata hash --------
    import numpy as np

    mismatches = []
    for sample in wanted_order:
        bundle = np.load(artifact / "pose" / f"{sample}.npz", allow_pickle=True)
        metadata = json.loads(str(bundle["metadata_json"]))
        digest = common.sha256_file(artifact / "trimmed" / f"{sample}.mp4")
        if digest != metadata.get("video_sha256"):
            mismatches.append(sample)
    if mismatches:
        raise SystemExit(f"copied clips do not match NPZ metadata sha256: {mismatches[:5]}")

    manifest = {
        "run_id": RUN_ID,
        "source_artifact": str(SOURCE_ARTIFACT),
        "source_selection": str(selection_path),
        "selection_sha256": common.sha256_file(selection_out),
        "filtered_selection": str(selection_out),
        "artifact_dir": str(artifact),
        "copies": len(copies),
        "copied_kinds": {"trimmed": len(wanted_order), "pose": len(wanted_order),
                         "alignment": len(wanted_order)},
        "trimmed_sha256_matches_npz_metadata": True,
        "note": "align stage NOT re-run: trimmed clips + HRNet NPZ + alignment records are copied "
                "from the reviewed csldaily-video-test-627-20261026 artifacts, so the inference "
                "input is byte-identical to the historical SLRT run",
    }
    common.write_json(inputs / "s1-bootstrap.json", manifest)
    print(json.dumps({"event": "s1-bootstrap", **{k: manifest[k] for k in
                      ("run_id", "artifact_dir", "copies", "trimmed_sha256_matches_npz_metadata")}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
