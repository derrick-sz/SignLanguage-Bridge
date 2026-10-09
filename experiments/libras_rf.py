"""Run the predeclared 12-item Libras RGB baseline; resume extracted clips."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

from slbridge._io import canonical_hash, configure_utf8_streams, read_json, resolve_resource, write_json_atomic
from slbridge.contracts import SplitManifest
from slbridge.data import create_split, load_manifest, project_split, save_manifest, select_labels
from slbridge.training import TrainConfig, train


PILOT_LABELS = tuple(f"item:{value}" for value in (2, 7, 92, 93, 94, 100, 105, 106, 107, 110, 111, 113))


def frozen_json(path: Path, value: dict) -> None:
    if path.exists() and canonical_hash(read_json(path)) != canonical_hash(value):
        raise FileExistsError(f"Frozen experiment definition differs: {path}; use a new work directory")
    write_json_atomic(path, value)


def run(source_manifest: Path, work_dir: Path, labels: tuple[str, ...] = PILOT_LABELS, seed: int = 42) -> Path:
    from slbridge.media import ExtractionConfig, extractor_identity
    from slbridge.extraction import extract_dataset

    work_dir = work_dir.resolve()
    full = load_manifest(source_manifest)
    if work_dir.is_relative_to(Path(full.root).resolve()) or Path(full.root).resolve().is_relative_to(work_dir):
        raise ValueError("Experiment output and source data must have disjoint roots")
    subset = select_labels(full, labels)
    config = ExtractionConfig()
    definition = {"source_dataset_hash": full.fingerprint, "source_subset_hash": subset.fingerprint,
                  "labels": list(labels), "seed": seed, "extractor": config.to_dict(),
                  "protocol": "freeze local_informant split before extraction; no global signer identity",
                  "rf": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 1, "n_jobs": 1},
                  "augmentation_repeats": 0, "threshold_candidates": [0.0, 0.4, 0.6], "minimum_validation_coverage": 0.8}
    work_dir.mkdir(parents=True, exist_ok=True)
    frozen_json(work_dir / "definition.json", definition)
    save_manifest(subset, work_dir / "source_manifest.json")
    source_split = create_split(subset, seed=seed, group_fields=("local_informant",))
    frozen_json(work_dir / "source_split.json", source_split.to_dict())
    start = time.monotonic()
    def progress(value: dict) -> None:
        state = {"stage": "extracting", **value}
        write_json_atomic(work_dir / "state.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    print(json.dumps({"stage": "starting", "samples": len(subset.samples), "labels": len(labels),
                      "purpose": "Real RGB -> landmark -> frozen-group RF -> validation/test -> independent video inference"}), flush=True)
    derived = extract_dataset(subset, work_dir / "features", config, progress)
    save_manifest(derived, work_dir / "feature_manifest.json")
    # Derived files inherit the pre-extraction source assignment. No test-aware
    # selection, regrouping or feature failures can silently move between folds.
    split = project_split(subset, source_split, derived)
    frozen_json(work_dir / "feature_split.json", split.to_dict())
    extractor = {"config": config.to_dict(), "identity": extractor_identity(config)}
    training_config = TrainConfig("mediapipe225_rgb_v1", target_frames=30, seed=seed, rf=definition["rf"],
                                  augmentation_repeats=0, threshold_candidates=tuple(definition["threshold_candidates"]),
                                  minimum_validation_coverage=0.8, extractor=extractor)
    frozen_json(work_dir / "training_config.json", training_config.to_dict())
    write_json_atomic(work_dir / "state.json", {"stage": "training", "samples": len(derived.samples), "train": len(split.train),
                                                "validation": len(split.validation), "test": len(split.test), "excluded": len(split.excluded)})
    run_id = "libras-rf-pilot-20261009"
    model_dir = work_dir / "models" / run_id
    if model_dir.exists():
        from slbridge.bundle import load_bundle
        bundle = load_bundle(model_dir)
        if bundle.manifest.dataset_hash != derived.fingerprint or bundle.manifest.split_hash != split.fingerprint or canonical_hash(bundle.manifest.parameters) != canonical_hash(training_config.to_dict()):
            raise ValueError("Existing model does not match this frozen experiment")
    else:
        model_dir = train(derived, split, training_config, work_dir / "models", run_id)
    metrics = read_json(model_dir / "metrics.json")
    by_id = {record.sample_id: record for record in subset.samples}
    example = by_id[split.test[0]]
    summary = {"stage": "trained", "model": str(model_dir), "labels": len(labels), "source_samples": len(subset.samples),
               "usable_samples": sum(record.quality_status == "ready" for record in derived.samples),
               "split": {key: len(getattr(split, key)) for key in ("train", "validation", "test", "excluded")},
               "validation": {key: metrics["validation"][key] for key in ("accuracy", "macro_f1", "coverage", "threshold")},
               "test": {key: metrics["test"][key] for key in ("accuracy", "macro_f1", "coverage", "threshold")},
               "test_video": str(resolve_resource(subset.root, example.resource)), "test_label": example.label_id,
               "test_sample_id": example.sample_id, "elapsed_seconds": time.monotonic() - start,
               "evaluation_scope": "finite 12-source-item classification; local informant groups, global signer unknown"}
    write_json_atomic(work_dir / "result.json", summary)
    write_json_atomic(work_dir / "state.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return model_dir


def main() -> None:
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("artifacts/datasets/libras-rgb/manifest.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/experiments/libras-rf-pilot-20261009"))
    parser.add_argument("--label-id", action="append")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.source_manifest, args.work_dir, tuple(args.label_id) if args.label_id else PILOT_LABELS, args.seed)


if __name__ == "__main__":
    main()
