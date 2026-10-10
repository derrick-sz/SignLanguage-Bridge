"""Explicit CE-CSL freeze, native/ONNX evaluation and sequence packaging tools."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
from functools import wraps
import json
import math
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from typing import Any, Callable
import uuid

from ._io import (canonical_hash, configure_utf8_streams, read_json, resolve_resource,
                  sha256_file, write_json_atomic)
from .cecsl import OFFICIAL_SPLITS, RGBInputSpec, freeze_cecsl_dataset, prepare_rgb_file

PINNED_CECSL_CSV_SHA256 = {
    "train": "0b17fbf12805430a43fa6c6a6972b8d5f0cee23232ea4b458cdc660d4d9fbbb7",
    "dev": "c4802e5c2f1c46ffc0c12b4c74d52d0f42dfd450dbacef4ccc1f9f889c680776",
    "test": "4a517b05b276caca58742f3c5249eedeb2e2537c8bfb5dc654e7230cf3e7cf82",
}
PINNED_CECSL_TOKEN_HASH = "824a3d119df15149891594b9ed55b4a7f75baf135abb98c057898cacc14a6a94"


def _require_checkpoint_vocabulary(frozen: dict) -> None:
    """The audited checkpoint requires exact class semantics, not just 3516."""
    if canonical_hash(list(frozen["tokens"])) != PINNED_CECSL_TOKEN_HASH:
        raise ValueError("CE-CSL token order differs from the audited TFNet checkpoint vocabulary")
    for origin in (frozen["vocabulary"]["provenance"], frozen["index"]["provenance"]):
        actual = {split: origin.get("csv", {}).get(split, {}).get("sha256") for split in OFFICIAL_SPLITS}
        if actual != PINNED_CECSL_CSV_SHA256:
            raise ValueError("CE-CSL CSV provenance differs from the audited TFNet checkpoint corpus")


@contextmanager
def _evaluation_lock(output: str | Path):
    """OS-owned nonblocking lock; the persistent file is never unlinked.

    Lock ownership vanishes on process exit, so a terminated evaluator cannot
    leave a stale lock preventing recovery. Holding the inode avoids delete races.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(output) + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    stream = os.fdopen(descriptor, "r+b")
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise FileExistsError(f"Another evaluator owns this output: {output}") from error
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _locked_evaluation(function):
    @wraps(function)
    def call(frozen, selected, predictor, output, **kwargs):
        with _evaluation_lock(output):
            return function(frozen, selected, predictor, output, **kwargs)
    return call


def _resume_row(row: dict, expected: dict, frozen: dict, spec: RGBInputSpec) -> dict:
    for name in ("sample_id", "signer", "source_split", "resource"):
        if row.get(name) != expected[name]:
            raise ValueError(f"Evaluation journal must match the exact frozen selection prefix: {name}")
    if row.get("reference") != expected["glosses"]:
        raise ValueError("Evaluation journal reference differs from frozen annotations")
    hypothesis_ids, hypothesis = row.get("hypothesis_ids"), row.get("hypothesis")
    if (not isinstance(hypothesis_ids, list)
            or any(type(identifier) is not int or not 1 <= identifier < len(frozen["tokens"])
                   for identifier in hypothesis_ids)
            or hypothesis != [frozen["tokens"][identifier] for identifier in hypothesis_ids]):
        raise ValueError("Evaluation journal hypothesis IDs/glosses disagree with ordered vocabulary")
    count = row.get("source_frames")
    if type(count) is not int or not 2 <= count <= spec.max_frames:
        raise ValueError("Evaluation journal source_frames violates the bounded input specification")
    steps = (count + 3) // 4
    if (type(row.get("output_steps")) is not int or row["output_steps"] != steps
            or row.get("input_shape") != [1, 4 * steps + 12, 3, 224, 224]):
        raise ValueError("Evaluation journal input/output lengths disagree with TFNet preprocessing")
    edits, reference_count = edit_distance(expected["glosses"], hypothesis), len(expected["glosses"])
    if (type(row.get("reference_tokens")) is not int or row["reference_tokens"] != reference_count
            or type(row.get("edit_distance")) is not int or row["edit_distance"] != edits):
        raise ValueError("Evaluation journal WER fields disagree with recomputed frozen-reference edit distance")
    for name in ("preprocess_seconds", "inference_seconds", "decode_seconds", "total_seconds"):
        value = row.get(name)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError(f"Evaluation journal has invalid timing: {name}")
    path, digest = _video_identity(frozen, expected)
    if digest != row.get("video_sha256"):
        raise ValueError("Previously evaluated video bytes changed since the resume journal")
    return {**row, "reference": list(expected["glosses"]), "reference_tokens": reference_count,
            "edit_distance": edits}


def load_frozen_cecsl(frozen_dir: str | Path) -> dict[str, Any]:
    root = Path(frozen_dir).resolve()
    summary = read_json(root / "freeze_summary.json")
    expected = {"vocabulary.json", "preprocessing.json", "dataset_index.json", "official_splits.json"}
    if summary.get("schema_version") != 1 or set(summary.get("artifacts", {})) != expected:
        raise ValueError("Invalid frozen CE-CSL summary")
    values = {}
    for name in sorted(expected):
        path = resolve_resource(root, name)
        identity = summary["artifacts"][name]
        value = read_json(path)
        if sha256_file(path) != identity["sha256"] or canonical_hash(value) != identity["canonical_hash"]:
            raise ValueError(f"Frozen CE-CSL artifact checksum mismatch: {name}")
        values[name] = value
    vocabulary, index, splits = values["vocabulary.json"], values["dataset_index.json"], values["official_splits.json"]
    from .sequence import _vocabulary
    tokens = _vocabulary(vocabulary)
    if tokens[0] != " " or len(tokens) != summary["vocabulary_size"]:
        raise ValueError("Frozen CE-CSL blank or vocabulary size mismatch")
    if splits.get("dataset_index_hash") != canonical_hash(index):
        raise ValueError("Frozen official splits do not reference the dataset index")
    samples = index["samples"]
    identifiers = [sample["sample_id"] for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Frozen CE-CSL sample IDs must be unique")
    by_id = dict(zip(identifiers, samples))
    split_ids = []
    for split in OFFICIAL_SPLITS:
        if not isinstance(splits.get(split), list):
            raise ValueError("Frozen CE-CSL official split membership is missing")
        for identifier in splits[split]:
            if identifier not in by_id or by_id[identifier]["source_split"] != split:
                raise ValueError("Frozen CE-CSL split membership disagrees with sample provenance")
        split_ids.extend(splits[split])
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(identifiers):
        raise ValueError("Frozen CE-CSL split IDs duplicate or omit samples")
    for record in samples:
        if [tokens[index] for index in record["gloss_ids"]] != record["glosses"]:
            raise ValueError("Frozen CE-CSL gloss IDs disagree with ordered vocabulary")
    return {"root": root, "summary": summary, "vocabulary": vocabulary, "tokens": tokens,
            "index": index, "splits": splits, "by_id": by_id,
            "spec": RGBInputSpec.from_dict(values["preprocessing.json"]["input_spec"]),
            "preprocessing": values["preprocessing.json"],
            "identity": {name: summary["artifacts"][name]["sha256"] for name in sorted(expected)}}


def select_official_samples(frozen: dict, split: str, *, limit: int | None = None,
                            offset: int = 0) -> list[dict]:
    if split not in OFFICIAL_SPLITS or type(offset) is not int or offset < 0:
        raise ValueError("Select an official train/dev/test split and a nonnegative offset")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive or omitted for the whole official split")
    identifiers = frozen["splits"][split]
    selected = identifiers[offset:] if limit is None else identifiers[offset:offset + limit]
    if not selected:
        raise ValueError("Official split selection is empty")
    return [frozen["by_id"][identifier] for identifier in selected]


def edit_distance(reference: list | tuple, hypothesis: list | tuple) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, target in enumerate(reference, start=1):
        current = [row]
        for column, candidate in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (target != candidate)))
        previous = current
    return previous[-1]


def wer_metrics(records: list[dict]) -> dict[str, Any]:
    if not records or any(record["reference_tokens"] < 1 for record in records):
        raise ValueError("WER requires nonempty reference sequences and evaluated records")
    edits = sum(record["edit_distance"] for record in records)
    tokens = sum(record["reference_tokens"] for record in records)
    return {"samples": len(records), "edit_distance": edits, "reference_tokens": tokens,
            "utterance_mean_wer_percent": 100 * sum(record["edit_distance"] / record["reference_tokens"]
                                                    for record in records) / len(records),
            "corpus_wer_percent": 100 * edits / tokens}


def _video_identity(frozen: dict, record: dict) -> tuple[Path, str]:
    path = resolve_resource(frozen["index"]["provenance"]["dataset_root"], record["resource"])
    if not record["video_exists"] or not path.is_file() or path.stat().st_size != record["video_bytes"]:
        raise ValueError(f"Frozen video is missing or its byte size changed: {record['sample_id']}")
    return path, sha256_file(path)


@_locked_evaluation
def evaluate_selected(frozen: dict, selected: list[dict], predictor: Callable,
                      output: str | Path, *, split: str, backend_identity: dict,
                      decoder: str = "greedy", beam_width: int = 10, token_top_k: int = 20,
                      spec: RGBInputSpec | None = None, resume: bool = False,
                      progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Evaluate every selected official member; errors stop rather than skip."""
    from .sequence import CTCDecoderConfig, greedy_decode, prefix_beam_decode
    config = CTCDecoderConfig(decoder, beam_width, 0, token_top_k)
    spec = spec or frozen["spec"]
    output = Path(output).resolve()
    journal = Path(str(output) + ".samples.jsonl")
    state_path = Path(str(output) + ".progress.json")
    if output.exists():
        raise FileExistsError(f"Evaluation destination already exists: {output}")
    identity = {"frozen_artifacts": frozen["identity"], "backend": backend_identity,
                "split": split, "selected_ids": [record["sample_id"] for record in selected],
                "decoder": config.to_dict(), "input_spec": spec.to_dict(),
                "punctuation_policy": "retain_all_reference_and_hypothesis_tokens",
                "decoder_protocol": "corrected_greedy_or_token_pruned_prefix_beam_not_upstream_ctcdecode"}
    records = []
    if journal.exists() or state_path.exists():
        if not resume or not journal.is_file() or not state_path.is_file():
            raise FileExistsError("Evaluation progress exists; explicitly use --resume for a matching run")
        if read_json(state_path).get("identity") != identity:
            raise ValueError("Evaluation resume identity differs from frozen selection/backend/decoder/spec")
        original_journal = journal.read_bytes()
        complete_bytes = (original_journal if not original_journal or original_journal.endswith(b"\n")
                          else original_journal[:original_journal.rfind(b"\n") + 1])
        for line in complete_bytes.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Evaluation journal samples must be JSON objects")
            records.append(row)
        if len(records) > len(selected):
            raise ValueError("Evaluation journal is longer than its selected official split")
        records = [_resume_row(row, selected_record, frozen, spec) for row, selected_record in zip(records, selected)]
        # A crash during append may leave only the final, non-newline-terminated
        # record incomplete. Validate every complete row before truncating that
        # exact suffix while holding the OS output lock.
        if complete_bytes != original_journal:
            with journal.open("r+b") as stream:
                stream.truncate(len(complete_bytes))
                stream.flush()
                os.fsync(stream.fileno())
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    state = {"schema_version": 1, "identity": identity, "status": "running", "completed": len(records),
             "selected": len(selected), "last_sample_id": records[-1]["sample_id"] if records else None}
    write_json_atomic(state_path, state)
    try:
        with journal.open("a" if resume else "x", encoding="utf-8", newline="\n") as stream:
            for record in selected[len(records):]:
                sample_started = perf_counter()
                path, digest = _video_identity(frozen, record)
                preprocess_started = perf_counter()
                prepared = prepare_rgb_file(path, spec)
                preprocess_seconds = perf_counter() - preprocess_started
                logits, inference_seconds = predictor(prepared)
                decode_started = perf_counter()
                prediction = (greedy_decode(logits, frozen["tokens"]) if decoder == "greedy" else
                              prefix_beam_decode(logits, frozen["tokens"], beam_width=beam_width,
                                                 token_top_k=token_top_k))
                hypothesis = list(prediction.glosses)
                row = {"sample_id": record["sample_id"], "signer": record["signer"], "source_split": split,
                       "resource": record["resource"], "video_sha256": digest,
                       "reference": record["glosses"], "hypothesis": hypothesis,
                       "hypothesis_ids": [token.token_id for token in prediction.tokens],
                       "reference_tokens": len(record["glosses"]),
                       "edit_distance": edit_distance(record["glosses"], hypothesis),
                       "source_frames": prepared.source_frames, "input_shape": list(prepared.values.shape),
                       "output_steps": logits.shape[0], "preprocess_seconds": preprocess_seconds,
                       "inference_seconds": inference_seconds, "decode_seconds": perf_counter() - decode_started,
                       "total_seconds": perf_counter() - sample_started}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(row)
                state.update({"completed": len(records), "last_sample_id": row["sample_id"],
                              "metrics": wer_metrics(records), "elapsed_this_invocation_seconds": perf_counter() - started})
                write_json_atomic(state_path, state)
                if progress:
                    progress({"completed": len(records), "selected": len(selected), "sample_id": row["sample_id"],
                              "source_frames": prepared.source_frames, "inference_seconds": inference_seconds,
                              "utterance_mean_wer_percent": state["metrics"]["utterance_mean_wer_percent"]})
        full_split = identity["selected_ids"] == frozen["splits"][split]
        report = {"schema_version": 1, "status": "complete", "identity": identity,
                  "scope": "entire_official_split" if full_split else "selected_official_split_subset",
                  "official_split_samples": len(frozen["splits"][split]), "metrics": wer_metrics(records),
                  "sample_journal": str(journal), "sample_journal_sha256": sha256_file(journal),
                  "records": records, "signer_overlap": frozen["summary"]["signer_overlap"],
                  "upstream_advertised_wer_verified": False,
                  "metric_note": "Utterance mean follows source aggregation; decoder differs, so this is a separately named baseline.",
                  "created_at": datetime.now(timezone.utc).isoformat()}
        # Re-check immediately before final publication; completed reports never
        # replace an unrelated run, including when two invocations race.
        if output.exists():
            raise FileExistsError(output)
        write_json_atomic(output, report)
        state["status"] = "complete"
        write_json_atomic(state_path, state)
        return report
    except BaseException as error:
        state.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      "error_type": type(error).__name__, "error": str(error)})
        write_json_atomic(state_path, state)
        raise


def evaluate_native(frozen_dir: str | Path, checkpoint: str | Path, source_dir: str | Path,
                    output: str | Path, *, device: str = "cuda:0", split: str = "test",
                    limit: int | None = None, offset: int = 0, **kwargs) -> dict:
    from .tfnet_native import TFNetNativeRunner
    frozen = load_frozen_cecsl(frozen_dir)
    _require_checkpoint_vocabulary(frozen)
    selected = select_official_samples(frozen, split, limit=limit, offset=offset)
    runner = TFNetNativeRunner(checkpoint, source_dir, device=device)

    def predict(prepared):
        prediction = runner.predict(prepared.values)
        return prediction.scores[:int(prediction.lengths[0]), 0], prediction.elapsed_seconds

    try:
        return evaluate_selected(frozen, selected, predict, output, split=split,
                                 backend_identity={"backend": "tfnet_native", **runner.provenance}, **kwargs)
    finally:
        runner.close()


def evaluate_onnx(frozen_dir: str | Path, model: str | Path, output: str | Path,
                  *, split: str = "test", limit: int | None = None, offset: int = 0,
                  options: Any = None, **kwargs) -> dict:
    from .sequence import CTCOnnxConfig, OnnxCTCBackend, load_sequence_bundle
    frozen = load_frozen_cecsl(frozen_dir)
    _require_checkpoint_vocabulary(frozen)
    selected = select_official_samples(frozen, split, limit=limit, offset=offset)
    model = Path(model).resolve()
    if model.is_dir():
        bundle = load_sequence_bundle(model, options=options)
        if bundle.manifest.vocabulary_hash != canonical_hash(frozen["vocabulary"]):
            bundle.backend.close()
            raise ValueError("ONNX sequence model vocabulary differs from frozen CE-CSL vocabulary")
        if kwargs.get("spec") is not None and kwargs["spec"] != bundle.manifest.input_spec:
            bundle.backend.close()
            raise ValueError("Explicit RGB limits disagree with the bundle's frozen input limits")
        backend, model_hash = bundle.backend, bundle.manifest.artifacts["model.onnx"]
        kwargs["spec"] = bundle.manifest.input_spec
    else:
        backend = OnnxCTCBackend.load(model, input_spec=kwargs.get("spec") or frozen["spec"],
                                      vocabulary=frozen["tokens"], config=CTCOnnxConfig(), options=options)
        model_hash = sha256_file(model)

    def predict(prepared):
        started = perf_counter()
        logits = backend.predict_logits(prepared)
        return logits, perf_counter() - started

    try:
        return evaluate_selected(frozen, selected, predict, output, split=split,
                                 backend_identity={"backend": "onnx_ctc", "onnx_sha256": model_hash,
                                                   "runtime": backend.runtime_info}, **kwargs)
    finally:
        backend.close()


def package_sequence(onnx: str | Path, frozen_dir: str | Path, output: str | Path,
                     *, run_id: str, decoder: str = "greedy", beam_width: int = 10,
                     token_top_k: int = 20) -> dict:
    """Package only a parity-verified export and frozen identities into a v2 bundle."""
    from .sequence import CTCDecoderConfig, CTCOnnxConfig, SequenceManifest
    frozen = load_frozen_cecsl(frozen_dir)
    _require_checkpoint_vocabulary(frozen)
    onnx, output = Path(onnx).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Sequence destination already exists: {output}")
    metadata = read_json(Path(str(onnx) + ".metadata.json"))
    if metadata.get("parity_passed") is not True or metadata.get("onnx_sha256") != sha256_file(onnx):
        raise ValueError("ONNX export has no matching successful numerical parity metadata")
    # Export inputs must bind the same preprocessing, vocabulary and frozen
    # corpus; export metadata emitted by the CLI records these exact identities.
    freeze_identity = metadata.get("cecsl_frozen_artifacts")
    if freeze_identity != frozen["identity"]:
        raise ValueError("ONNX export is not bound to this frozen CE-CSL identity")
    provenance = {"export": metadata, "frozen_preprocessing": frozen["preprocessing"],
                  "frozen_artifact_hashes": frozen["identity"],
                  "task": "continuous_gloss_ctc", "natural_language_translation": False}
    spec = RGBInputSpec.from_dict(metadata.get("cecsl_input_spec", frozen["spec"].to_dict()))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        shutil.copyfile(onnx, staging / "model.onnx")
        for name in ("vocabulary.json", "dataset_index.json", "official_splits.json"):
            shutil.copyfile(frozen["root"] / name, staging / name)
        write_json_atomic(staging / "preprocessing.json", spec.to_dict())
        write_json_atomic(staging / "provenance.json", provenance)
        artifacts = {path.name: sha256_file(path) for path in staging.iterdir() if path.is_file()}
        try:
            runtime_version = importlib.metadata.version("onnxruntime")
        except importlib.metadata.PackageNotFoundError:
            runtime_version = importlib.metadata.version("onnxruntime-gpu")
        manifest = SequenceManifest(run_id=run_id, input_spec=spec,
                                    vocabulary_hash=canonical_hash(frozen["vocabulary"]), artifacts=artifacts,
                                    parameters=CTCDecoderConfig(decoder, beam_width, 0, token_top_k).to_dict(),
                                    dependencies={"onnxruntime": runtime_version}, provenance=provenance,
                                    created_at=datetime.now(timezone.utc).isoformat(), sign_language="csl",
                                    runtime_config=CTCOnnxConfig().to_dict(),
                                    dataset_hash=artifacts["dataset_index.json"], split_hash=artifacts["official_splits.json"],
                                    frozen_evidence={"dataset": "dataset_index.json", "split": "official_splits.json"})
        write_json_atomic(staging / "manifest.json", manifest.to_dict())
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return {"bundle": str(output), "manifest": manifest.to_dict()}
    finally:
        # staging is a fresh UUID path within the known output parent.
        if staging.exists() and staging.resolve().parent == output.parent.resolve():
            shutil.rmtree(staging)


def _progress(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--dataset-root", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--source-dir")
    for name in ("native", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--frozen-dir", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--split", choices=OFFICIAL_SPLITS, default="test")
        command.add_argument("--limit", type=int)
        command.add_argument("--offset", type=int, default=0)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--decoder", choices=("greedy", "prefix_beam"), default="greedy")
        command.add_argument("--beam-width", type=int, default=10)
        command.add_argument("--token-top-k", type=int, default=20)
        if name == "native":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--source-dir", required=True)
            command.add_argument("--device", default="cuda:0")
        else:
            command.add_argument("--model", required=True)
            command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
            command.add_argument("--device-id", type=int, default=0)
            command.add_argument("--intra-op-threads", type=int, default=4)
            command.add_argument("--cpu-conv-subnormals", choices=("flush_to_zero", "preserve"), default="flush_to_zero")
            command.add_argument("--gpu-mem-limit-mb", type=int, default=8192)
            command.add_argument("--cuda-dll-dir", action="append", default=[])
        command.add_argument("--max-frames", type=int)
        command.add_argument("--max-seconds", type=float)
        command.add_argument("--max-resource-bytes", type=int)
    export = sub.add_parser("export")
    export.add_argument("--frozen-dir", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--source-dir", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--verification-video", action="append", required=True)
    export.add_argument("--device", default="cuda:0")
    export.add_argument("--provider", action="append")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--rtol", type=float, default=1e-4)
    export.add_argument("--atol", type=float, default=1e-4)
    export.add_argument("--max-frames", type=int)
    export.add_argument("--max-seconds", type=float)
    export.add_argument("--max-resource-bytes", type=int)
    package = sub.add_parser("package")
    package.add_argument("--onnx", required=True)
    package.add_argument("--frozen-dir", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--run-id", required=True)
    package.add_argument("--decoder", choices=("greedy", "prefix_beam"), default="greedy")
    package.add_argument("--beam-width", type=int, default=10)
    package.add_argument("--token-top-k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_cecsl_dataset(args.dataset_root, args.output, upstream_root=args.source_dir)
    elif args.command == "package":
        result = package_sequence(args.onnx, args.frozen_dir, args.output, run_id=args.run_id,
                                  decoder=args.decoder, beam_width=args.beam_width, token_top_k=args.token_top_k)
    else:
        frozen = load_frozen_cecsl(args.frozen_dir)
        changes = {name: getattr(args, name) for name in ("max_frames", "max_seconds", "max_resource_bytes")
                   if getattr(args, name) is not None}
        spec = replace(frozen["spec"], **changes)
        if args.command == "export":
            from .tfnet_export import export_tfnet
            _require_checkpoint_vocabulary(frozen)
            inputs = [prepare_rgb_file(path, spec) for path in args.verification_video]
            result = export_tfnet(args.checkpoint, args.source_dir, args.output, inputs, device=args.device,
                                  opset_version=args.opset, rtol=args.rtol, atol=args.atol,
                                  providers=args.provider or ("CPUExecutionProvider",))
            result.update({"cecsl_frozen_artifacts": frozen["identity"], "cecsl_input_spec": spec.to_dict(),
                           "verification_video_sha256": {str(Path(path).resolve()): sha256_file(path)
                                                        for path in args.verification_video}})
            write_json_atomic(Path(str(Path(args.output).resolve()) + ".metadata.json"), result)
        else:
            common = {"split": args.split, "limit": args.limit, "offset": args.offset,
                      "decoder": args.decoder, "beam_width": args.beam_width, "token_top_k": args.token_top_k,
                      "spec": spec, "resume": args.resume, "progress": _progress}
            if args.command == "evaluate" and not changes:
                # An unmodified CLI default defers to a bundle's frozen limits.
                # A raw graph still defaults to the frozen corpus input spec.
                common.pop("spec")
            if args.command == "native":
                result = evaluate_native(args.frozen_dir, args.checkpoint, args.source_dir, args.output,
                                          device=args.device, **common)
            else:
                from .runtime_options import RuntimeOptions
                options = RuntimeOptions(device=args.device, device_id=args.device_id,
                                          intra_op_threads=args.intra_op_threads,
                                          gpu_mem_limit_mb=args.gpu_mem_limit_mb,
                                          cuda_dll_dirs=tuple(args.cuda_dll_dir),
                                          cpu_conv_subnormals=args.cpu_conv_subnormals)
                result = evaluate_onnx(args.frozen_dir, args.model, args.output, options=options, **common)
    if args.command in {"native", "evaluate"}:
        printed = {"output": str(Path(args.output).resolve()), "status": result["status"],
                   "scope": result["scope"], "metrics": result["metrics"],
                   "sample_journal": result["sample_journal"]}
    elif args.command == "export":
        printed = {key: result[key] for key in ("onnx_path", "onnx_sha256", "parity_passed")}
    elif args.command == "package":
        printed = {"bundle": result["bundle"], "run_id": result["manifest"]["run_id"]}
    else:
        printed = {key: result[key] for key in ("vocabulary_size", "rows", "videos_present", "videos_missing", "artifacts")}
    print(json.dumps(printed, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
