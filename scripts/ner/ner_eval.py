"""Entity-level NER scoring for an SFT run (LoRA adapter or full model).

Generates ``type: entity`` lines for each validation sentence, parses them
with the same helpers used to build training targets, and reports micro /
macro / per-class precision, recall, and F1 via scikit-learn.

Usage::

    python ner_eval.py \\
      --run /workspace/moe/ner_runs/baseline_lora_smoke \\
      --base /workspace/moe/checkpoints/baseline/step_003053_hf \\
      --eval-limit 200
"""

from __future__ import annotations

# Standard library
import argparse
import json
from pathlib import Path

# Third party
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local
import ner_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    run: Path,
    base: Path | None,
    attn_implementation: str | None = None,
):
    dtype = torch.bfloat16
    model_kwargs: dict = dict(dtype=dtype, trust_remote_code=True)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if base is not None:
        model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
        model = PeftModel.from_pretrained(model, run)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(run, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(run, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(run, trust_remote_code=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run", type=Path, required=True,
        help="SFT output dir (adapter or full model).",
    )
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--output-format", default=None,
                        choices=["json", "inline"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--show", type=int, default=5,
        help="Print this many example generations.",
    )
    args = parser.parse_args()

    # --- Resolve config from run directory -----------------------------------
    run_cfg: dict = {}
    for cfg_name in ("ner_config.yaml", "ner_sft_config.json"):
        cfg_path = args.run / cfg_name
        if cfg_path.exists():
            if cfg_name.endswith(".yaml"):
                with open(cfg_path) as f:
                    run_cfg = yaml.safe_load(f)
            else:
                run_cfg = json.loads(cfg_path.read_text())
            break

    dataset_key = args.dataset or run_cfg.get("dataset", ner_data.DEFAULT_DATASET)
    eval_limit = args.eval_limit or run_cfg.get("eval_limit", 200)
    attn_impl = args.attn_implementation or run_cfg.get("attn_implementation")
    output_format = (args.output_format
                     or run_cfg.get("output_format", ner_data.DEFAULT_OUTPUT_FORMAT))

    is_adapter = (args.run / "adapter_config.json").is_file()
    base = args.base
    if is_adapter and base is None:
        base_str = run_cfg.get("base_checkpoint") or run_cfg.get("checkpoint")
        if base_str:
            base = Path(base_str)
        else:
            raise SystemExit(
                f"{args.run} is a LoRA adapter; pass --base <checkpoint>"
            )

    # --- Load model ----------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ner-eval] loading model (adapter={is_adapter})")
    model, tokenizer = load_model(
        args.run, base if is_adapter else None, attn_impl,
    )
    model.to(device).eval()

    # --- Load data -----------------------------------------------------------
    raw, schema, label_column = ner_data.load_split(
        dataset_key, args.split, eval_limit,
        output_format=output_format,
    )
    print(f"[ner-eval] dataset={dataset_key} | format={output_format} | {len(schema.types)} types")
    print(f"[ner-eval] evaluating {len(raw)} examples from '{args.split}'")

    bos = (
        [tokenizer.bos_token_id]
        if tokenizer.bos_token_id is not None
        else []
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    # --- Generate & score ----------------------------------------------------
    y_true: list[str] = []
    y_pred: list[str] = []
    shown: list[dict] = []
    all_samples: list[dict] = []

    all_prompt_ids: list[list[int]] = []
    all_examples: list[dict] = []
    for example in raw:
        prompt = schema.build_prompt(example["tokens"])
        prompt_ids = bos + tokenizer(
            prompt, add_special_tokens=False,
        )["input_ids"]
        all_prompt_ids.append(prompt_ids[-args.max_seq_len:])
        all_examples.append(example)

    n_batches = (len(all_prompt_ids) + args.batch_size - 1) // args.batch_size
    for batch_start in range(0, len(all_prompt_ids), args.batch_size):
        batch_idx = batch_start // args.batch_size + 1
        print(f"[ner-eval] batch {batch_idx}/{n_batches} ({batch_start}/{len(all_prompt_ids)} examples)", flush=True)
        batch_prompts = all_prompt_ids[batch_start:batch_start + args.batch_size]
        batch_examples = all_examples[batch_start:batch_start + args.batch_size]

        max_len = max(len(p) for p in batch_prompts)
        padded_ids = [
            [pad_id] * (max_len - len(p)) + p for p in batch_prompts
        ]
        attn_masks = [
            [0] * (max_len - len(p)) + [1] * len(p) for p in batch_prompts
        ]

        input_ids = torch.tensor(padded_ids, device=device)
        attention_mask = torch.tensor(attn_masks, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, example in enumerate(batch_examples):
            new_tokens = outputs[j, input_ids.shape[1]:]
            completion = tokenizer.decode(
                new_tokens, skip_special_tokens=True,
            )

            pred_set = schema.parse(completion)
            gold_set = schema.gold_set(
                example["tokens"], example[label_column],
            )
            raw_text = completion.strip()
            if output_format == "json":
                parse_ok = "{" in raw_text and "}" in raw_text
            else:
                parse_ok = ":" in raw_text or raw_text.lower() == "none" or raw_text == ""

            for t, _ in pred_set & gold_set:
                y_true.append(t)
                y_pred.append(t)
            for t, _ in gold_set - pred_set:
                y_true.append(t)
                y_pred.append("O")
            for t, _ in pred_set - gold_set:
                y_true.append("O")
                y_pred.append(t)

            sample_record = {
                "text": " ".join(example["tokens"]),
                "generated": raw_text,
                "pred": sorted(pred_set),
                "gold": sorted(gold_set),
                "parse_ok": parse_ok,
            }
            all_samples.append(sample_record)

            idx = batch_start + j
            if idx < args.show:
                shown.append(sample_record)

    # --- Report --------------------------------------------------------------
    entity_types = sorted(schema.types)
    report_dict = classification_report(
        y_true, y_pred, labels=entity_types,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred, labels=entity_types, zero_division=0,
    )

    print("\n=== Example Generations ===")
    for ex in shown:
        print(f"\nTEXT: {ex['text']}")
        print(f"GEN : {ex['generated']!r}")
        print(f"PRED: {ex['pred']}")
        print(f"GOLD: {ex['gold']}")

    parse_ok_count = sum(1 for s in all_samples if s["parse_ok"])
    parse_fail_count = len(all_samples) - parse_ok_count

    print(f"\n{'=' * 70}")
    print(f"Parse: {parse_ok_count}/{len(all_samples)} OK, {parse_fail_count} failed")
    print(f"{'=' * 70}")
    print("scikit-learn classification_report (entity-level exact match):")
    print(f"{'=' * 70}")
    print(report_text)
    print(f"{'=' * 70}")

    if parse_fail_count:
        print(f"\n=== Failed Parses (first 10) ===")
        fails = [s for s in all_samples if not s["parse_ok"]]
        for s in fails[:10]:
            print(f"\nTEXT: {s['text'][:120]}")
            print(f"GEN : {s['generated']!r}")

    # --- Save ----------------------------------------------------------------
    result = {
        "run": str(args.run),
        "base": str(base) if is_adapter else None,
        "dataset": dataset_key,
        "split": args.split,
        "examples": len(raw),
        "parse_ok": parse_ok_count,
        "parse_fail": parse_fail_count,
        "report": report_dict,
        "examples_shown": shown,
        "samples": all_samples,
    }

    out_path = args.output or (args.run / f"ner_eval_{args.split}.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[ner-eval] report saved to {out_path}")


if __name__ == "__main__":
    main()
