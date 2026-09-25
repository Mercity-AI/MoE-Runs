"""Create persistent KDA/GQA attention-transplant checkpoints.

The destination is always rebuilt as ``LlamaKDA`` with an explicit per-layer
layout.  This matters for KDA -> baseline grafts: merely assigning a KDA module
to a dense LLaMA object works in memory but silently becomes dense again after
``save_pretrained`` / ``from_pretrained``.

Examples
--------
GQA donors from layers 0 and 16 into the KDA model::

    python graft_kda_gqa.py \
      --direction gqa-into-kda --layers 0,16 \
      --baseline /checkpoints/baseline/step_003053_hf \
      --kda /checkpoints/kda/step_003053_hf \
      --output /checkpoints/grafts/gqa_into_kda_l00_l16

KDA donors from layers 0 and 16 into the dense baseline::

    python graft_kda_gqa.py \
      --direction kda-into-gqa --layers 0,16 \
      --baseline /checkpoints/baseline/step_003053_hf \
      --kda /checkpoints/kda/step_003053_hf \
      --output /checkpoints/grafts/kda_into_gqa_l00_l16
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import torch


def parse_layers(value: str) -> list[int]:
    """Parse a comma-separated, unique, ascending layer list."""
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layer indices must be unique")
    return sorted(layers)


def resolved_kda_layers(config) -> set[int]:
    # Imported lazily so selection/unit tests do not require the model stack.
    from model import resolve_kda_layer_types

    return {
        index
        for index, layer_type in enumerate(resolve_kda_layer_types(config))
        if layer_type == "kda"
    }


def validate_compatible(baseline, kda) -> None:
    """Refuse a graft unless all shared residual-space dimensions match."""
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
    )
    mismatches = [
        f"{name}: baseline={getattr(baseline.config, name, None)!r}, "
        f"kda={getattr(kda.config, name, None)!r}"
        for name in fields
        if getattr(baseline.config, name, None) != getattr(kda.config, name, None)
    ]
    if not getattr(baseline.config, "qk_norm", False):
        mismatches.append("baseline qk_norm is false (use the QK-normalized baseline)")
    if not getattr(kda.config, "qk_norm", False):
        mismatches.append("KDA softmax-layer qk_norm is false")
    if mismatches:
        raise ValueError("Incompatible checkpoints:\n  " + "\n  ".join(mismatches))


def check_layers(layers: Iterable[int], depth: int, available_kda: set[int], direction: str) -> None:
    invalid = [index for index in layers if not 0 <= index < depth]
    if invalid:
        raise ValueError(f"Layer indices outside [0, {depth}): {invalid}")
    if direction == "kda-into-gqa":
        unavailable = sorted(set(layers) - available_kda)
        if unavailable:
            raise ValueError(
                "The KDA donor has no KDA module at layers "
                f"{unavailable}; available KDA layers are {sorted(available_kda)}"
            )


def make_recipient_config(kda_config, kda_layers: set[int]):
    """Clone the KDA config and encode the final hybrid layout explicitly."""
    from model import LlamaKDAConfig

    values = kda_config.to_dict()
    for derived_key in (
        "_name_or_path",
        "auto_map",
        "architectures",
        "model_type",
        "transformers_version",
    ):
        values.pop(derived_key, None)
    values.update(
        architectures=["LlamaKDA"],
        kda_full_attn_layers=[
            index for index in range(kda_config.num_hidden_layers) if index not in kda_layers
        ],
        kda_full_attn_range=None,
        kda_full_attn_every=None,
        kda_every=None,
        kda_offset=0,
        use_cache=False,
    )
    return LlamaKDAConfig(**values)


def transplant(*, baseline, kda, direction: str, layers: list[int]):
    """Build and return a reloadable hybrid model plus an audit manifest."""
    from model import LlamaKDA

    validate_compatible(baseline, kda)
    original_kda = resolved_kda_layers(kda.config)
    depth = kda.config.num_hidden_layers
    check_layers(layers, depth, original_kda, direction)

    if direction == "gqa-into-kda":
        final_kda = original_kda - set(layers)
        base_state = kda.state_dict()
        donor = baseline
    elif direction == "kda-into-gqa":
        final_kda = set(layers)
        base_state = baseline.state_dict()
        donor = kda
    else:  # protected by argparse; useful for library callers
        raise ValueError(f"Unknown direction: {direction!r}")

    recipient = LlamaKDA(make_recipient_config(kda.config, final_kda))
    incompatible = recipient.load_state_dict(base_state, strict=False)
    for index in layers:
        recipient.model.layers[index].self_attn = copy.deepcopy(
            donor.model.layers[index].self_attn
        )

    # New modules are constructed in the process default (usually FP32), while
    # checkpoint weights are BF16. Keep the graft at the source checkpoint's
    # dtype so saving does not unnecessarily double its size.
    recipient.to(dtype=next(baseline.parameters()).dtype)

    # A strict self-load catches missing parameters after all donor modules land.
    recipient.load_state_dict(recipient.state_dict(), strict=True)
    manifest = {
        "direction": direction,
        "transplanted_layers": layers,
        "original_kda_layers": sorted(original_kda),
        "final_kda_layers": sorted(final_kda),
        "final_gqa_layers": sorted(set(range(depth)) - final_kda),
        "base_load_missing_keys": sorted(incompatible.missing_keys),
        "base_load_unexpected_keys": sorted(incompatible.unexpected_keys),
    }
    return recipient, manifest


def load_models(baseline_path: Path, kda_path: Path, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": dtype, "trust_remote_code": True, "device_map": "cpu"}
    baseline = AutoModelForCausalLM.from_pretrained(baseline_path, **kwargs)
    kda = AutoModelForCausalLM.from_pretrained(kda_path, **kwargs)
    return baseline, kda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("gqa-into-kda", "kda-into-gqa"), required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--kda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    for label, path in (("baseline", args.baseline), ("KDA", args.kda)):
        if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
            parser.error(f"{label} is not a complete HF checkpoint: {path}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    baseline, kda = load_models(args.baseline, args.kda, dtype)
    graft, manifest = transplant(
        baseline=baseline, kda=kda, direction=args.direction, layers=args.layers
    )
    args.output.mkdir(parents=True, exist_ok=True)
    graft.save_pretrained(args.output, safe_serialization=True)
    # Both source checkpoints use the same tokenizer; retain the baseline copy.
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        source = args.baseline / filename
        if source.is_file():
            (args.output / filename).write_bytes(source.read_bytes())
    manifest.update(
        baseline_checkpoint=str(args.baseline.resolve()),
        kda_checkpoint=str(args.kda.resolve()),
    )
    (args.output / "graft_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved graft checkpoint to {args.output.resolve()}")


if __name__ == "__main__":
    main()
