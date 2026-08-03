"""
Evaluation script - HellaSwag + Winogrande
==========================================
Runs likelihood-based multiple choice eval on a checkpoint.
No generation needed — scores each candidate by its log-likelihood
and picks the argmax.

Usage:
    pip install -r requirements.txt
    python eval.py
    python eval.py --checkpoint ./ckpts/step_002500_hf
"""

import argparse
from pathlib import Path
import json

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = SCRIPT_DIR / "ckpts" / "step_003053_hf"
TASKS = [
    "hellaswag",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "piqa",
    "openbookqa",
    "commonsense_qa",
    "lambada_openai",
    "sciq",
]
NUM_SHOTS = 0  # zero-shot; set to 5 for few-shot


def run(
    checkpoint: str | Path,
    tokenizer: str | Path | None = None,
    limit: int | None = None,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Expected a Hugging Face checkpoint with model.safetensors: {checkpoint}"
        )

    tokenizer = (
        str(Path(tokenizer).expanduser().resolve())
        if tokenizer is not None
        else str(checkpoint)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] loading {checkpoint} on {device}")
    print(f"[eval] tasks: {TASKS} | shots: {NUM_SHOTS}\n")

    lm = HFLM(
        pretrained=str(checkpoint),
        tokenizer=tokenizer,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=NUM_SHOTS,
        batch_size="auto",  # auto-tunes batch size to fill VRAM
        log_samples=False,
        limit=limit,
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
    out_path = checkpoint.parent / f"{checkpoint.name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\n[eval] full results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--shots", type=int, default=NUM_SHOTS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only this many examples per task (useful for a smoke test).",
    )
    args = parser.parse_args()
    NUM_SHOTS = args.shots
    run(args.checkpoint, tokenizer=args.tokenizer, limit=args.limit)
