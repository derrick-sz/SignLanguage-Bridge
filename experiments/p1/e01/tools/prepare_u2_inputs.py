"""E01 step 5 input prep: filter the upstream Uni-Sign pose refs/pres to pilot 50.

The frozen script ``scripts/evaluate_unisign_csldaily_pose.py`` has no ``--only``;
it consumes a refs file and takes exactly those samples.  This writes the pilot-50
subset of the upstream released ``test_tmp_refs.txt`` / ``test_tmp_pres.txt`` so
the pose rerun covers the same 50 samples as the video and SLRT paths.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\prepare_u2_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402

UPSTREAM_DIR = ROOT / "models" / "external" / "uni_sign_source" / "out" / "eval_csl_daily_pose_only"
RUN_ID = "e01-pilot50-u2-unisign-pose-20261026"
MARKERS = (", ground-truth: ", ", prediction: ")


def parse_line(line: str) -> tuple[str, str] | None:
    for marker in MARKERS:
        if marker in line:
            head, value = line.split(marker, 1)
            sample = head.split("sample: ", 1)[-1].strip()
            return sample, value
    return None


def filter_file(source: Path, destination: Path, wanted: set[str], wanted_order: list[str]) -> int:
    parsed: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = parse_line(line)
        if item is None:
            raise SystemExit(f"unparsed line in {source}: {line[:80]}")
        sample, value = item
        if sample in wanted:
            parsed[sample] = line
    missing = wanted - set(parsed)
    if missing:
        raise SystemExit(f"{source.name}: {len(missing)} pilot samples missing, e.g. {sorted(missing)[:5]}")
    # keep the pilot order so per-line bookkeeping matches the audit tables
    lines = [parsed[sample] for sample in wanted_order if sample in parsed]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


def main() -> int:
    ids_path = common.P1 / "e01" / "inputs" / "pilot_50_ids.txt"
    wanted_order = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(wanted_order)
    run_dir = common.RUNS / RUN_ID
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    refs = filter_file(UPSTREAM_DIR / "test_tmp_refs.txt", inputs / "refs-pilot50.txt", wanted, wanted_order)
    pres = filter_file(UPSTREAM_DIR / "test_tmp_pres.txt", inputs / "pres-pilot50.txt", wanted, wanted_order)
    metadata = {
        "run_id": RUN_ID,
        "source_refs": str(UPSTREAM_DIR / "test_tmp_refs.txt"),
        "source_refs_sha256": common.sha256_file(UPSTREAM_DIR / "test_tmp_refs.txt"),
        "source_pres": str(UPSTREAM_DIR / "test_tmp_pres.txt"),
        "source_pres_sha256": common.sha256_file(UPSTREAM_DIR / "test_tmp_pres.txt"),
        "filtered_refs": str(inputs / "refs-pilot50.txt"),
        "filtered_refs_sha256": common.sha256_file(inputs / "refs-pilot50.txt"),
        "filtered_pres": str(inputs / "pres-pilot50.txt"),
        "filtered_pres_sha256": common.sha256_file(inputs / "pres-pilot50.txt"),
        "pilot_ids": str(ids_path),
        "pilot_ids_sha256": common.sha256_file(ids_path),
        "count": refs,
        "pres_count": pres,
        "note": "upstream released reference/prediction snapshot subset; not re-tokenized",
    }
    common.write_json(inputs / "u2-inputs.json", metadata)
    print(json.dumps({"event": "u2-inputs", **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
