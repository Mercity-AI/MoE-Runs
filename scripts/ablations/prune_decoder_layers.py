"""Remove selected complete decoder blocks from a reloadable HF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_layers(value: str) -> list[int]:
    try:
        layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
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
        source, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    original_depth = len(model.model.layers)
    invalid = [index for index in args.layers if not 0 <= index < original_depth]
    if invalid:
        raise ValueError(f"layer indices outside [0, {original_depth}): {invalid}")
    if len(args.layers) == original_depth:
        raise ValueError("cannot remove every decoder layer")

    original_types = list(
        getattr(model.config, "kda_layer_types", ["full"] * original_depth)
    )
    if len(original_types) != original_depth:
        raise ValueError("kda_layer_types length does not match checkpoint depth")

    removed_set = set(args.layers)
    kept = [index for index in range(original_depth) if index not in removed_set]
    final_types = [original_types[index] for index in kept]
    model.model.layers = torch.nn.ModuleList([model.model.layers[index] for index in kept])
    for new_index, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index

    depth = len(kept)
    model.config.num_hidden_layers = depth
    if hasattr(model.config, "kda_full_attn_layers"):
        model.config.kda_full_attn_layers = [
            index for index, kind in enumerate(final_types) if kind == "full"
        ]
        model.config.kda_full_attn_range = None
        model.config.kda_full_attn_every = None
        model.config.kda_every = None
        model.config.kda_offset = 0
        model.config.kda_layer_types = final_types
        model.config.use_cache = False

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        candidate = source / filename
        if candidate.is_file():
            (output / filename).write_bytes(candidate.read_bytes())

    manifest = {
        "source_checkpoint": str(source),
        "intervention": "removed selected complete decoder blocks",
        "original_depth": original_depth,
        "pruned_depth": depth,
        "removed_original_layers": args.layers,
        "removed_original_layer_types": {
            str(index): original_types[index] for index in args.layers
        },
        "retained_original_layers": kept,
        "new_to_original_layer": {str(new): old for new, old in enumerate(kept)},
        "final_layer_types": final_types,
        "recovery_training": False,
    }
    (output / "pruning_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
