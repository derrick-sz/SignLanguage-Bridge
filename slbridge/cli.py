"""Thin CLI for indexing, frozen splits, training and independent recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ._io import configure_utf8_streams, read_json, write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slbridge", description="SignLanguage Bridge core (landmark training and RGB clip recognition)")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Build an immutable-source dataset manifest")
    index.add_argument("--source", choices=("legacy", "libras"), required=True)
    index.add_argument("--root", type=Path, required=True)
    index.add_argument("--dataset-id", required=True)
    index.add_argument("--representation", choices=("hand63_v1", "pose_hands225_legacy"))
    index.add_argument("--sign-language", help="Omit when legacy sign language is unverified")
    index.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Verify source integrity against a manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    select = commands.add_parser("select", help="Create an explicit subset by stable label ID")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--label-id", action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract", help="Extract timestamped RGB landmarks with resumable cache")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--output-manifest", type=Path, required=True)
    extract.add_argument("--config", type=Path)
    split = commands.add_parser("split", help="Freeze train/validation/test with leakage constraints")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.add_argument("--group-field", action="append", default=[])
    split.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Train and publish a new model bundle")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--split", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    train.add_argument("--run-id")
    predict = commands.add_parser("predict", help="Recognize a compatible landmark clip or raw video")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--representation", choices=("hand63_v1", "pose_hands225_legacy", "mediapipe225_rgb_v1", "tfnet_rgb_v1"))
    predict.add_argument("--sample-id")
    return parser


def _metadata_output(path: Path, source_root: str) -> None:
    if path.suffix.lower() != ".json" or path.resolve().is_relative_to(Path(source_root).resolve()):
        raise ValueError("Metadata output must be a .json file outside the read-only dataset")


def main(argv: list[str] | None = None) -> int:
    configure_utf8_streams()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            from .data import import_legacy, import_libras, save_manifest
            _metadata_output(args.output, str(args.root))
            if args.source == "legacy":
                if args.representation is None:
                    raise ValueError("Legacy import requires --representation; no implicit width inference")
                manifest = import_legacy(args.root, args.dataset_id, args.representation, args.sign_language)
            else:
                if args.representation is not None or args.sign_language not in {None, "libras"}:
                    raise ValueError("Libras RGB adapter declares its own representation and sign language")
                manifest = import_libras(args.root, args.dataset_id)
            save_manifest(manifest, args.output)
            result = {"manifest": str(args.output.resolve()), "dataset_hash": manifest.fingerprint,
                      "samples": len(manifest.samples), "ready": sum(s.quality_status == "ready" for s in manifest.samples),
                      "quarantined": sum(s.quality_status != "ready" for s in manifest.samples)}
        elif args.command == "validate":
            from .data import load_manifest, verify_manifest
            manifest = load_manifest(args.manifest)
            verify_manifest(manifest)
            result = {"valid": True, "dataset_hash": manifest.fingerprint, "samples": len(manifest.samples)}
        elif args.command == "select":
            from .data import load_manifest, save_manifest, select_labels
            original = load_manifest(args.manifest)
            _metadata_output(args.output, original.root)
            if args.output.resolve() == args.manifest.resolve():
                raise ValueError("Subset must not overwrite its source manifest")
            subset = select_labels(original, tuple(args.label_id))
            save_manifest(subset, args.output)
            result = {"manifest": str(args.output.resolve()), "samples": len(subset.samples), "labels": sorted(set(args.label_id))}
        elif args.command == "extract":
            from .data import load_manifest, save_manifest
            from .media import ExtractionConfig
            from .extraction import extract_dataset
            original = load_manifest(args.manifest)
            _metadata_output(args.output_manifest, original.root)
            _metadata_output(args.output_manifest, str(args.output_dir))
            if args.output_manifest.resolve() == args.manifest.resolve():
                raise ValueError("Derived manifest must not overwrite its source manifest")
            config = ExtractionConfig.from_dict(read_json(args.config)) if args.config else ExtractionConfig()
            def progress(value: dict) -> None:
                print(json.dumps({"extraction_progress": value}, sort_keys=True), file=sys.stderr, flush=True)
            derived = extract_dataset(original, args.output_dir, config=config, progress=progress)
            save_manifest(derived, args.output_manifest)
            result = {"manifest": str(args.output_manifest.resolve()), "samples": len(derived.samples),
                      "ready": sum(record.quality_status == "ready" for record in derived.samples)}
        elif args.command == "split":
            from .data import create_split, load_manifest, verify_manifest
            manifest = load_manifest(args.manifest)
            _metadata_output(args.output, manifest.root)
            if args.output.resolve() == args.manifest.resolve():
                raise ValueError("Split output must not overwrite the dataset manifest")
            verify_manifest(manifest)
            split = create_split(manifest, seed=args.seed, validation_fraction=args.validation_fraction,
                                 test_fraction=args.test_fraction, group_fields=tuple(args.group_field))
            if args.output.exists() and read_json(args.output) != json.loads(json.dumps(split.to_dict())):
                raise FileExistsError("Frozen split already exists with a different definition; use a new output path")
            write_json_atomic(args.output, split.to_dict())
            result = {"split": str(args.output.resolve()), "split_hash": split.fingerprint,
                      "train": len(split.train), "validation": len(split.validation), "test": len(split.test),
                      "excluded": len(split.excluded), "protocol": split.protocol}
        elif args.command == "train":
            from .contracts import SplitManifest
            from .data import load_manifest
            from .training import TrainConfig, train
            bundle_path = train(load_manifest(args.manifest), SplitManifest.from_dict(read_json(args.split)),
                                TrainConfig.from_dict(read_json(args.config)), args.output_dir, args.run_id)
            result = {"model": str(bundle_path), "metrics": read_json(bundle_path / "metrics.json")}
        else:
            from .inference import Recognizer
            result = Recognizer.load(args.model).recognize(args.input, sample_id=args.sample_id,
                                                          representation_id=args.representation).to_dict()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, NotImplementedError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
