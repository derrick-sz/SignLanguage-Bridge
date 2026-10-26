"""Assemble a full-vocabulary CSL Gloss2Text model directory from base mBART.

This is a *fallback* for the authors' pruned ``mBART_zh`` assets: without the
CSL-Daily sentence annotations (``csl.train/dev/test``) the notebook's text-side
pruning (``old2new_vocab.pkl``) cannot be reproduced. Keeping the full mBART
vocabulary makes the model runnable, at the cost of a larger output vocabulary.
It must not be reported as the authors' pruned checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-mbart", type=Path, required=True)
    p.add_argument("--gloss-assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base = args.base_mbart
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "sentencepiece.bpe.model", "sentence.bpe.model", "pytorch_model.bin"):
        source = base / name
        if not source.is_file():
            raise SystemExit(f"base mBART file missing: {source}")
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    actions = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json", "sentencepiece.bpe.model", "sentence.bpe.model", "pytorch_model.bin"):
        actions[name] = link_or_copy(base / name, output / name)
    identity = {index: index for index in range(vocab_size)}
    with (output / "old2new_vocab.pkl").open("wb") as handle:
        pickle.dump(identity, handle, protocol=4)
    with (output / "map_ids.pkl").open("wb") as handle:
        pickle.dump(identity, handle, protocol=4)
    for name in ("gloss_embeddings.bin", "gloss2ids.pkl"):
        actions[name] = link_or_copy(args.gloss_assets / name, output / name)
    readme = (
        "# Full-vocabulary mBART_zh fallback\n\n"
        "This directory was assembled from `facebook/mbart-large-cc25` plus the\n"
        "CSL-Daily 2001-class vocabulary. `old2new_vocab.pkl` and `map_ids.pkl` are\n"
        "identity maps, so the text vocabulary is the full mBART vocabulary rather\n"
        "than the authors' pruned vocabulary. It is runnable but is not the exact\n"
        "released/derived `mBART_zh` model; reproduce the pruning notebook once\n"
        "`csl.train/dev/test` are available.\n"
    )
    (output / "README_FALLBACK.md").write_text(readme, encoding="utf-8")
    summary = {
        "base_mbart": str(base.resolve()),
        "gloss_assets": str(args.gloss_assets.resolve()),
        "output": str(output.resolve()),
        "vocab_size": vocab_size,
        "fallback": "full-vocab identity pruning; not the authors' pruned model",
        "actions": actions,
    }
    (output / "ASSEMBLY_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
