"""Attach a trained LongCat n-gram input module to the QK baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ngram", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    from transformers import AutoModelForCausalLM
    load = dict(dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    baseline = AutoModelForCausalLM.from_pretrained(args.baseline, **load)
    donor = AutoModelForCausalLM.from_pretrained(args.ngram, **load)
    for field in ("hidden_size", "vocab_size"):
        if getattr(baseline.config, field) != getattr(donor.config, field):
            raise ValueError(f"checkpoint mismatch for {field}")
    if not getattr(baseline.config, "qk_norm", False):
        raise ValueError("the baseline must be QK-normalized")

    values = baseline.config.to_dict()
    for key in ("_name_or_path", "auto_map", "architectures", "model_type", "transformers_version"):
        values.pop(key, None)
    for key in (
        "ngram_max_n", "ngram_num_heads", "ngram_table_vocab_sizes",
        "ngram_embedding_amplification", "ngram_hash_multipliers",
        "ngram_min_pairwise_size_gap",
    ):
        if hasattr(donor.config, key):
            values[key] = getattr(donor.config, key)
    values["architectures"] = ["LlamaLongCatNgram"]
    # Use the donor checkpoint's packaged classes so its exact historical hash
    # function is preserved (newer local implementations use salted hashes).
    config = type(donor.config)(**values)
    graft = type(donor)(config)
    incompatible = graft.load_state_dict(baseline.state_dict(), strict=False)
    graft.ngram_embedder = copy.deepcopy(donor.ngram_embedder)
    graft.to(dtype=next(baseline.parameters()).dtype)
    graft.load_state_dict(graft.state_dict(), strict=True)

    args.output.mkdir(parents=True, exist_ok=True)
    graft.save_pretrained(args.output, safe_serialization=True)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        source = args.baseline / filename
        if source.is_file():
            (args.output / filename).write_bytes(source.read_bytes())
    manifest = {
        "baseline_checkpoint": str(args.baseline.resolve()),
        "ngram_checkpoint": str(args.ngram.resolve()),
        "intervention": "trained donor ngram_embedder attached to unchanged QK baseline backbone",
        "baseline_layers": baseline.config.num_hidden_layers,
        "donor_layers": donor.config.num_hidden_layers,
        "missing_before_transplant": sorted(incompatible.missing_keys),
        "unexpected_before_transplant": sorted(incompatible.unexpected_keys),
    }
    (args.output / "graft_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
