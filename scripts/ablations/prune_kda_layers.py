"""Remove complete KDA decoder blocks from a hybrid KDA checkpoint.

The retained full-attention blocks keep their original weights and order.  The
saved checkpoint is explicitly configured as a shorter, all-GQA LlamaKDA model
so save/reload cannot silently reconstruct the removed KDA blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "config.json").is_file() or not (source / "model.safetensors").is_file():
        parser.error(f"not a complete HF checkpoint: {source}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    layer_types = list(getattr(model.config, "kda_layer_types", []))
    if len(layer_types) != len(model.model.layers):
        raise ValueError(
            "checkpoint must expose one kda_layer_types entry per decoder block"
        )

    removed = [index for index, kind in enumerate(layer_types) if kind == "kda"]
    kept = [index for index, kind in enumerate(layer_types) if kind == "full"]
    unknown = sorted(set(layer_types) - {"kda", "full"})
    if unknown or not removed or not kept:
        raise ValueError(
            f"expected a mixed KDA/full checkpoint; types={unknown}, "
            f"removed={removed}, kept={kept}"
        )

    model.model.layers = torch.nn.ModuleList([model.model.layers[index] for index in kept])
    for new_index, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index

    depth = len(kept)
    model.config.num_hidden_layers = depth
    model.config.kda_full_attn_layers = list(range(depth))
    model.config.kda_full_attn_range = None
    model.config.kda_full_attn_every = None
    model.config.kda_every = None
    model.config.kda_offset = 0
    model.config.kda_layer_types = ["full"] * depth
    model.config.use_cache = False

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        candidate = source / filename
        if candidate.is_file():
            (output / filename).write_bytes(candidate.read_bytes())

    manifest = {
        "source_checkpoint": str(source),
        "intervention": "removed complete decoder blocks whose attention type was KDA",
        "original_depth": len(layer_types),
        "pruned_depth": depth,
        "removed_original_layers": removed,
        "retained_original_layers": kept,
        "new_to_original_layer": {str(new): old for new, old in enumerate(kept)},
        "final_layer_types": ["full"] * depth,
        "recovery_training": False,
    }
    (output / "pruning_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved pruned checkpoint to {output}")


if __name__ == "__main__":
    main()
