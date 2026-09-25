"""KDA checkpoint-level interpretability: gate/decay distributions and
before/after activation norms.

Extends the gate-decay formula already used for live training diagnostics
(see ``_kda_gate_decay_stats`` in ``baseline_llama_KDA.py``) into a
standalone, checkpoint-level analysis, per future-experiments.md section 4:

    Record learned gate and decay distributions by layer, head, and channel.
    Convert decay values to effective memory half-lives.
    Compare half-life distributions between early, middle, and late KDA layers.
    Measure activation norms immediately before and after KDA layers.

Not covered here (left as follow-up, each needs more machinery than a single
forward pass over plain text): retrieval-success-vs-context-length
correlation, short-convolution ablation, and local-mixer-vs-long-memory
layer classification.

Usage:
    python kda_interpretability.py --checkpoint /workspace/moe/checkpoints/kda/step_003053_hf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def load_probe_text(num_docs: int, seq_len: int, tokenizer) -> torch.Tensor:
    """Stream a few real FineWeb documents (same source/domain as pretraining)
    and pack them into fixed-length sequences. Gate/decay values are
    data-dependent (computed from f_proj(hidden_states)), so this needs real
    text, not random tokens.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ids: list[int] = []
    for i, example in enumerate(ds):
        if i >= num_docs:
            break
        ids.extend(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id or 0)
    if len(ids) < seq_len:
        raise ValueError(
            f"probe stream produced only {len(ids)} tokens (< seq_len={seq_len}); "
            "increase --num-docs or lower --seq-len."
        )
    n_seqs = len(ids) // seq_len
    ids = ids[: n_seqs * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n_seqs, seq_len)


def half_life_from_log_decay(log_decay: torch.Tensor) -> torch.Tensor:
    """Steps until a value decays to half strength: decay**n = 0.5, i.e.
    n = ln(0.5) / ln(decay). Takes ln(decay) directly (computed upstream as
    ``-exp(A_log) * softplus(...)``) rather than re-deriving it via
    ``log(decay.exp())`` -- decay is often within float32 epsilon of 1.0 (very
    slow forgetting), at which point exp() then log() collapses back to
    exactly log(1.0)==0.0 and the division blows up to +/-inf.
    """
    log_decay = log_decay.double().clamp(max=-1e-12)  # strictly negative, avoid /0
    return (torch.log(torch.tensor(0.5, dtype=torch.float64)) / log_decay).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/workspace/moe/checkpoints/kda/step_003053_hf"),
    )
    parser.add_argument("--num-docs", type=int, default=32, help="FineWeb documents to stream for the probe pass.")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--output", type=Path,
        default=SCRIPT_DIR.parent / "kda_interpretability_results" / "kda_gate_decay_report.json",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[kda-interp] loading {args.checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    layer_types = list(getattr(model.config, "kda_layer_types", []))
    kda_layer_idxs = [i for i, kind in enumerate(layer_types) if kind == "kda"]
    if not kda_layer_idxs:
        raise ValueError(f"no KDA layers found in {args.checkpoint} (kda_layer_types={layer_types})")
    print(f"[kda-interp] KDA layers: {kda_layer_idxs}")

    gate_probe: dict[int, list[torch.Tensor]] = {i: [] for i in kda_layer_idxs}
    kda_modules: dict[int, torch.nn.Module] = {}
    layer_io_norms: dict[int, dict[str, list[float]]] = {i: {"input": [], "output": []} for i in range(len(layer_types))}
    handles = []

    for idx in kda_layer_idxs:
        kda_module = model.model.layers[idx].self_attn.kda
        kda_modules[idx] = kda_module

        def _capture_gate(_module, _inputs, output, i=idx):
            # The hook fires once per forward; accumulate every probe sequence so
            # the decay distribution spans the whole probe pass, not just the last
            # one. Kept on CPU in float32 so the full stack does not sit in GPU
            # memory (concatenated one layer at a time below).
            gate_probe[i].append(output.detach().to("cpu", torch.float32))

        handles.append(kda_module.f_proj.register_forward_hook(_capture_gate))

    for idx, layer in enumerate(model.model.layers):
        def _capture_io_norms(_module, inputs, output, i=idx):
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, tuple) else output
            layer_io_norms[i]["input"].append(hidden_in.float().norm(dim=-1).mean().item())
            layer_io_norms[i]["output"].append(hidden_out.float().norm(dim=-1).mean().item())

        handles.append(layer.register_forward_hook(_capture_io_norms))

    print(f"[kda-interp] streaming {args.num_docs} FineWeb docs for the probe pass")
    batches = load_probe_text(args.num_docs, args.seq_len, tokenizer)
    print(f"[kda-interp] probing with {batches.shape[0]} sequences of length {args.seq_len}")

    with torch.no_grad():
        for seq in batches:
            model(input_ids=seq.unsqueeze(0).to(device))

    for h in handles:
        h.remove()

    report: dict = {"checkpoint": str(args.checkpoint), "kda_layers": kda_layer_idxs, "layers": {}}
    third = max(1, len(kda_layer_idxs) // 3)
    early, middle, late = (
        kda_layer_idxs[:third],
        kda_layer_idxs[third: 2 * third],
        kda_layer_idxs[2 * third:],
    )

    group_half_lives: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    group_of = {i: "early" for i in early} | {i: "middle" for i in middle} | {i: "late" for i in late}

    for idx in kda_layer_idxs:
        kda_module = kda_modules[idx]
        # Concatenate all probe sequences along the batch dim so the reduction over
        # batch/token dims below spans the entire probe pass. Moved back to the
        # model device for the decay math against dt_bias/A_log.
        gate_input = torch.cat(gate_probe[idx], dim=0).to(device)
        num_v_heads, head_k_dim = kda_module.num_v_heads, kda_module.head_k_dim
        with torch.no_grad():
            g = gate_input.float().view(*gate_input.shape[:-1], num_v_heads, head_k_dim)
            dt_bias = kda_module.dt_bias.float().view(num_v_heads, head_k_dim)
            a_log = kda_module.A_log.float().view(num_v_heads, 1)
            # Keep the log-domain decay around directly (never round-trip
            # through exp() then log() -- see half_life_from_log_decay).
            log_decay = -torch.exp(a_log) * torch.nn.functional.softplus(g + dt_bias)
            decay = log_decay.exp()
            # decay/log_decay: [..., num_v_heads, head_k_dim] -> reduce over batch/token dims only
            reduce_dims = tuple(range(decay.dim() - 2))
            per_head_channel_mean = decay.mean(dim=reduce_dims)         # [num_v_heads, head_k_dim]
            per_head_mean = per_head_channel_mean.mean(dim=-1)          # [num_v_heads]
            per_head_channel_log_mean = log_decay.mean(dim=reduce_dims)  # [num_v_heads, head_k_dim]
            per_head_log_mean = per_head_channel_log_mean.mean(dim=-1)   # [num_v_heads]

        hl_overall = half_life_from_log_decay(log_decay).clamp(max=1e6)
        hl_per_head = half_life_from_log_decay(per_head_log_mean).clamp(max=1e6)

        layer_report = {
            "decay_mean": decay.mean().item(),
            "decay_min": decay.min().item(),
            "decay_max": decay.max().item(),
            "half_life_tokens_mean": hl_overall.mean().item(),
            "half_life_tokens_median": hl_overall.median().item(),
            "per_head_decay_mean": per_head_mean.tolist(),
            "per_head_half_life_tokens": hl_per_head.tolist(),
            "per_head_channel_decay_mean": per_head_channel_mean.tolist(),
            "activation_norm_input_mean": sum(layer_io_norms[idx]["input"]) / len(layer_io_norms[idx]["input"]),
            "activation_norm_output_mean": sum(layer_io_norms[idx]["output"]) / len(layer_io_norms[idx]["output"]),
            "group": group_of[idx],
        }
        layer_report["activation_norm_delta"] = (
            layer_report["activation_norm_output_mean"] - layer_report["activation_norm_input_mean"]
        )
        report["layers"][idx] = layer_report
        group_half_lives[group_of[idx]].append(layer_report["half_life_tokens_mean"])

    report["group_summary"] = {
        group: {
            "layers": [i for i in kda_layer_idxs if group_of[i] == group],
            "mean_half_life_tokens": (sum(vals) / len(vals)) if vals else None,
        }
        for group, vals in group_half_lives.items()
    }
    report["all_layers_activation_norms"] = {
        str(i): {
            "input_mean": sum(v["input"]) / len(v["input"]),
            "output_mean": sum(v["output"]) / len(v["output"]),
            "type": layer_types[i],
        }
        for i, v in layer_io_norms.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'Layer':<8}{'Group':<8}{'Decay mean':<14}{'Half-life (tok)':<18}{'Act Δ':<10}")
    print("=" * 70)
    for idx in kda_layer_idxs:
        r = report["layers"][idx]
        print(f"{idx:<8}{r['group']:<8}{r['decay_mean']:<14.4f}{r['half_life_tokens_mean']:<18.1f}{r['activation_norm_delta']:<10.3f}")
    print("=" * 70)
    for group, summary in report["group_summary"].items():
        print(f"{group:<8} layers={summary['layers']} mean half-life={summary['mean_half_life_tokens']}")
    print(f"\n[kda-interp] full report saved to {args.output}")


if __name__ == "__main__":
    main()
