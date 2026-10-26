"""Build data-independent SLRT Gloss2Text assets from a base mBART model.

This reproduces the gloss-embedding half of the upstream
``Online/SLT/utils/prune_embedding.ipynb`` workflow:

- ``gloss_embeddings.bin``: special-token aliases plus the mean subword
  embedding of every CSL gloss.
- ``gloss2ids.pkl``: insertion-order mapping used by ``GlossEmbedding``.

The text-side pruning (``old2new_vocab.pkl`` / ``map_ids.pkl``) depends on the
CSL-Daily sentence annotations (``csl.train/dev/test``) and is intentionally not
attempted here. Run the notebook/derivation again once those files exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-mbart", type=Path, required=True)
    p.add_argument("--csl-vocab", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from transformers import MBartForConditionalGeneration, MBartTokenizer

    base = args.base_mbart
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(args.csl_vocab.read_text(encoding="utf-8"))
    if not isinstance(vocab, list) or vocab[0] != "<blank>" or len(vocab) != 2001:
        raise SystemExit("CSL vocabulary must be an ordered 2001-class JSON list with <blank> at index 0")

    tokenizer = MBartTokenizer.from_pretrained(str(base), src_lang="zh_CN")
    model = MBartForConditionalGeneration.from_pretrained(str(base))
    model.to(args.device)
    model.eval()
    weight = model.model.shared.weight.detach().cpu()

    def emb_of_token(token: str) -> torch.Tensor:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise SystemExit(f"token not in mBART vocabulary: {token}")
        return weight[token_id].clone()

    glosses = {}
    for token in ("<s>", "<pad>", "</s>", "<unk>", "<mask>", "zh_CN", "de_DE"):
        glosses[token] = emb_of_token(token)
    glosses["zh_CSL"] = glosses["zh_CN"].clone()
    glosses["de_DGS"] = glosses["de_DE"].clone()
    for gloss in vocab[1:]:
        gloss = gloss.lower()
        ids = tokenizer(gloss)["input_ids"][:-2]  # drop </s> and the language token
        if not ids:
            raise SystemExit(f"gloss has no subword ids: {gloss!r}")
        glosses[gloss] = torch.stack([weight[i] for i in ids]).mean(dim=0)

    torch.save(glosses, output / "gloss_embeddings.bin")
    gloss2ids = {gloss: index for index, gloss in enumerate(glosses)}
    with (output / "gloss2ids.pkl").open("wb") as handle:
        pickle.dump(gloss2ids, handle, protocol=4)
    summary = {
        "base_mbart": str(base.resolve()),
        "base_mbart_model_sha256": sha256(base / "pytorch_model.bin"),
        "csl_vocab": str(args.csl_vocab.resolve()),
        "csl_vocab_sha256": sha256(args.csl_vocab),
        "gloss_embeddings": str((output / "gloss_embeddings.bin").resolve()),
        "gloss2ids": str((output / "gloss2ids.pkl").resolve()),
        "gloss_count": len(vocab) - 1,
        "embedding_rows": len(glosses),
        "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "zh_CN", "de_DE", "zh_CSL", "de_DGS"],
        "note": "Text-side pruning requires CSL-Daily csl.train/dev/test; not generated here.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
