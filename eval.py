"""
Evaluation script - HellaSwag + Winogrande
==========================================
Runs likelihood-based multiple choice eval on a checkpoint.
No generation needed — scores each candidate by its log-likelihood
and picks the argmax.

Usage:
    pip install lm-eval
    python eval.py
    python eval.py --checkpoint ./checkpoints-2704/checkpoint-2500
"""

import argparse
import json

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

CHECKPOINT = "./checkpoints_llama_baseline_final/checkpoint-2500"
TOKENIZER = "meta-llama/Llama-3.2-1B"
TASKS = ["hellaswag", "winogrande"]
NUM_SHOTS = 0  # zero-shot; set to 5 for few-shot


def run(checkpoint: str, tokenizer: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] loading {checkpoint} on {device}")
    print(f"[eval] tasks: {TASKS} | shots: {NUM_SHOTS}\n")

    lm = HFLM(
        pretrained=checkpoint,
        tokenizer=tokenizer or checkpoint,
        dtype=torch.bfloat16,
        device=device,
        attn_implementation="eager",
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=NUM_SHOTS,
        batch_size="auto",  # auto-tunes batch size to fill VRAM
        log_samples=False,
    )

    print("\n" + "=" * 60)
    print(f"{'Task':<20} {'Metric':<25} {'Value':>8}")
    print("=" * 60)
    for task, metrics in results["results"].items():
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{task:<20} {metric:<25} {value:>8.4f}")
    print("=" * 60)

    # also dump full results to json for logging
    out_path = checkpoint.rstrip("/") + "_eval.json"
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\n[eval] full results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--shots", type=int, default=NUM_SHOTS)
    args = parser.parse_args()
    NUM_SHOTS = args.shots
    run(args.checkpoint, tokenizer=args.tokenizer)
