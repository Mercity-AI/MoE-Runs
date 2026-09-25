"""Post-training quantization + Shared-9 quality check, per
future-experiments.md section 5.

Covers, as a "basic" first pass:
    1. BF16 reference (no-op baseline for comparison).
    2. INT8 weight-only (bitsandbytes LLM.int8(), quantizes nn.Linear layers).
    3. NF4 4-bit weight-only (bitsandbytes), a simpler stand-in for the
       calibrated W4A16/AWQ scheme the doc describes -- not AWQ itself.

Not covered here: activation-aware calibration (AWQ proper), selective
n-gram-table-vs-projection quantization, selective KDA quantization (leaving
A_log/dt_bias/norms/short-conv in BF16 while only quantizing q/k/v/o/f/g/b
projections), and packed-kernel throughput/memory benchmarking beyond the
static checkpoint footprint reported here.

Requires: pip install bitsandbytes (not in requirements.txt yet).

Usage:
    python quantize_eval.py --checkpoint /workspace/moe/checkpoints/baseline/step_003053_hf --method int8
    python quantize_eval.py --checkpoint /workspace/moe/checkpoints/baseline/step_003053_hf --method nf4 --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TASKS = [
    "hellaswag", "winogrande", "arc_challenge", "arc_easy", "piqa",
    "openbookqa", "commonsense_qa", "lambada_openai", "sciq",
]


def load_model(checkpoint: Path, method: str):
    from transformers import AutoModelForCausalLM

    common = dict(trust_remote_code=True)
    if method == "bf16":
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, **common)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    elif method in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig

        if method == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, quantization_config=quant_config, device_map="auto", **common
        )
    else:
        raise ValueError(f"unknown method: {method}")
    model.eval()
    return model


def run(
    checkpoint: Path,
    method: str,
    tokenizer_path: Path | None,
    limit: int | None,
    batch_size,
    output: Path | None,
) -> None:
    from lm_eval import evaluator, utils
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    print(f"[quantize-eval] loading {checkpoint} with method={method}")
    model = load_model(checkpoint, method)
    footprint_bytes = model.get_memory_footprint()
    print(f"[quantize-eval] checkpoint memory footprint: {footprint_bytes / 1e9:.3f} GB")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or checkpoint)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = evaluator.simple_evaluate(
        model=lm, tasks=TASKS, num_fewshot=0, batch_size=batch_size,
        log_samples=False, limit=limit,
    )

    print("\n" + "=" * 60)
    print(f"{'Task':<20} {'Metric':<25} {'Value':>8}")
    print("=" * 60)
    for task, metrics in results["results"].items():
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{task:<20} {metric:<25} {value:>8.4f}")
    print("=" * 60)

    out_path = output or (
        checkpoint.parent
        / f"{checkpoint.name}_{method}_eval{'_limit_' + str(limit) if limit is not None else ''}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "method": method,
        "memory_footprint_bytes": footprint_bytes,
        "results": results["results"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=utils.handle_non_serializable)
    print(f"\n[quantize-eval] saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("bf16", "int8", "nf4"), default="bf16")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.method,
        args.tokenizer,
        args.limit,
        int(args.batch_size) if args.batch_size.isdigit() else args.batch_size,
        args.output,
    )
