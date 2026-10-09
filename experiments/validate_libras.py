"""Independent-process validation of held-out raw video and archive parity."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from slbridge._io import configure_utf8_streams, read_json, resolve_resource, write_json_atomic
from slbridge.contracts import SplitManifest
from slbridge.data import load_manifest
from slbridge.features import feature_vector, prepare_input
from slbridge.inference import Recognizer


def run(work_dir: Path) -> dict:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work_dir = work_dir.resolve()
    result = read_json(work_dir / "result.json")
    source = load_manifest(work_dir / "source_manifest.json")
    derived = load_manifest(work_dir / "feature_manifest.json")
    split = SplitManifest.from_dict(read_json(work_dir / "feature_split.json"))
    raw_records = {record.sample_id: record for record in source.samples}
    features = {record.sample_id: record for record in derived.samples}
    recognizer = Recognizer.load(result["model"])
    examples = []
    # One already frozen test example per class, selected by sample ID only.
    labels_seen = set()
    for identifier in split.test:
        record = raw_records[identifier]
        if record.label_id in labels_seen:
            continue
        labels_seen.add(record.label_id)
        path = resolve_resource(source.root, record.resource)
        archive = resolve_resource(derived.root, features[identifier].resource)
        started = time.monotonic()
        raw_prepared = prepare_input(path, recognizer.bundle.manifest.input_spec)
        archive_prepared = prepare_input(archive, recognizer.bundle.manifest.input_spec)
        raw_vector, archive_vector = feature_vector(raw_prepared), feature_vector(archive_prepared)
        max_error = float(np.max(np.abs(raw_vector - archive_vector)))
        mask_equal = bool(np.array_equal(raw_prepared.mask, archive_prepared.mask))
        archive_probabilities = recognizer.bundle.backend.predict_proba([archive_prepared])[0]
        raw_probabilities = recognizer.bundle.backend.predict_proba([raw_prepared])[0]
        prediction = recognizer.recognize(path, sample_id=identifier).to_dict()
        payload = prediction["prediction"]
        examples.append({"sample_id": identifier, "source_resource": record.resource, "expected_label": record.label_id,
                         "expected_text": record.display_text, "prediction": prediction,
                         "correct": payload is not None and payload["label_id"] == record.label_id,
                         "max_coordinate_error": max_error, "mask_equal": mask_equal,
                         "probability_max_error": float(np.max(np.abs(archive_probabilities - raw_probabilities))),
                         "elapsed_seconds": time.monotonic() - started})
        write_json_atomic(work_dir / "independent_video_validation.json", {"stage": "validating", "examples": examples})
        print(f"Independent raw-video check {len(examples)}/{len(recognizer.bundle.labels)}: {record.label_id}; coordinate error {max_error:.3g}", flush=True)
    summary = {"stage": "complete", "model_id": recognizer.bundle.manifest.run_id, "examples": examples,
               "tested_source": "one pre-frozen test video per class; full test metrics use all held-out extracted clips",
               "tested_classes": len(examples), "correct": sum(item["correct"] for item in examples),
               "processing_max_error": max(item["max_coordinate_error"] for item in examples),
               "all_masks_equal": all(item["mask_equal"] for item in examples),
               "probability_max_error": max(item["probability_max_error"] for item in examples)}
    if not summary["all_masks_equal"] or summary["processing_max_error"] > 1e-5:
        raise ValueError("Raw video and archived feature preparation differ; inspect extraction state before trusting model")
    write_json_atomic(work_dir / "independent_video_validation.json", summary)
    metrics = read_json(Path(result["model"]) / "metrics.json")
    matrix = np.asarray(metrics["test"]["confusion_matrix"])
    label_order = metrics["test"]["class_order"]
    texts = {record.label_id: record.display_text.get("pt-BR", record.label_id) for record in source.samples}
    fig, axis = plt.subplots(figsize=(11, 9), layout="constrained")
    view = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(label_order)), yticks=range(len(label_order)),
             xticklabels=[texts[label] for label in label_order], yticklabels=[texts[label] for label in label_order],
             xlabel="Predicted source item", ylabel="True source item", title="Libras RGB RF baseline — frozen test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(view, ax=axis, label="Test videos")
    fig.savefig(work_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)
    write_json_atomic(work_dir / "state.json", {**result, "stage": "complete", "independent_video_validation": str(work_dir / "independent_video_validation.json")})
    return summary


def main() -> None:
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/experiments/libras-rf-pilot-20261009"))
    args = parser.parse_args()
    print(run(args.work_dir))


if __name__ == "__main__":
    main()
