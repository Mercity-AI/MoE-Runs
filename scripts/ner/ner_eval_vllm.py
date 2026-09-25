"""Entity-level NER eval using vLLM for fast batched inference.

Merges a LoRA adapter into its base checkpoint (or loads a full model),
serves it through vLLM's offline LLM engine, and scores entity-level
precision / recall / F1 on the validation set.

Usage::

    python ner_eval_vllm.py \
      --run /workspace/moe/ner_runs/ngram_longcat_25_json/checkpoint-1250 \
      --eval-limit 500 \
      --max-new-tokens 128 \
      --show 10

    # Full model (no adapter):
    python ner_eval_vllm.py \
      --run /workspace/moe/checkpoints/baseline/step_003053_hf \
      --eval-limit 500

N-gram models (LlamaLongCatNgram) decode one sequence at a time
(``max_num_seqs=1``); other architectures batch 64 sequences by default.
Check a new checkpoint with ``analysis/vllm_hf_parity.py`` before trusting it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import vllm_env  # noqa: F401  — FlashInfer sm_120 patch + in-process engine; must precede vllm

import torch
import yaml
from sklearn.metrics import classification_report

import ner_data
import vllm_ngram_model  # noqa: F401  — registers custom architectures with vLLM


def merge_adapter(run: Path, base: Path, out_dir: Path) -> None:
    """Merge LoRA adapter into base model and save to out_dir."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(run))
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(run, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    for name in ("model.py", "configuration_*.py"):
        import glob
        for src in glob.glob(str(base / name)):
            dst = out_dir / Path(src).name
            if not dst.exists():
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", type=Path, required=True,
                        help="SFT output dir (adapter or full model).")
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Left-truncate prompts to this many tokens (matches ner_eval.py).")
    parser.add_argument("--output-format", default=None,
                        choices=["json", "inline"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", type=int, default=5,
                        help="Print this many example generations.")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree for vLLM.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Max concurrent sequences (default: 1 for n-gram models, 64 otherwise; "
                             "n-gram models must use 1).")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs.")
    args = parser.parse_args()

    # --- Resolve config from run directory ---
    run_cfg: dict = {}
    for cfg_name in ("ner_config.yaml", "ner_sft_config.json"):
        cfg_path = args.run / cfg_name
        if cfg_path.exists():
            if cfg_name.endswith(".yaml"):
                run_cfg = yaml.safe_load(cfg_path.read_text())
            else:
                run_cfg = json.loads(cfg_path.read_text())
            break

    dataset_key = args.dataset or run_cfg.get("dataset", ner_data.DEFAULT_DATASET)
    eval_limit = args.eval_limit or run_cfg.get("eval_limit", 200)
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

    # --- Merge adapter if needed, then load with vLLM ---
    tmp_dir = None
    if is_adapter:
        tmp_dir = tempfile.mkdtemp(prefix="ner_vllm_merged_")
        merged_path = Path(tmp_dir)
        print(f"[ner-eval-vllm] merging adapter into base → {merged_path}")
        merge_adapter(args.run, base, merged_path)
        model_path = str(merged_path)
    else:
        model_path = str(args.run)

    arch = json.loads((Path(model_path) / "config.json").read_text())["architectures"][0]
    is_ngram = arch == "LlamaLongCatNgram"
    if args.max_num_seqs is None:
        args.max_num_seqs = 1 if is_ngram else 64
    if is_ngram and args.max_num_seqs != 1:
        raise SystemExit("n-gram models need --max-num-seqs 1 (the n-gram context "
                         "buffer holds a single sequence)")
    print(f"[ner-eval-vllm] architecture={arch} max_num_seqs={args.max_num_seqs}")
    print(f"[ner-eval-vllm] loading model with vLLM from {model_path}")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        # The n-gram embedder rebuilds context from a per-position token buffer
        # that a prefix-cache hit would leave stale.
        enable_prefix_caching=False,
    )

    # --- Load data ---
    raw, schema, label_column = ner_data.load_split(
        dataset_key, args.split, eval_limit, output_format=output_format,
    )
    print(f"[ner-eval-vllm] dataset={dataset_key} | format={output_format} | {len(schema.types)} types")
    print(f"[ner-eval-vllm] evaluating {len(raw)} examples from '{args.split}'")

    # --- Build prompts ---
    # Tokenize exactly as training / ner_eval.py do: BOS + prompt, no special
    # tokens from the tokenizer (this tokenizer never adds BOS on its own, so
    # passing raw strings to vLLM would silently drop it).
    tokenizer = AutoTokenizer.from_pretrained(args.run, trust_remote_code=True)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prompts: list[TokensPrompt] = []
    examples: list[dict] = []
    for example in raw:
        prompt_ids = bos + tokenizer(
            schema.build_prompt(example["tokens"]), add_special_tokens=False,
        )["input_ids"]
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids[-args.max_seq_len:]))
        examples.append(example)

    # --- Generate with vLLM ---
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0,
    )

    print(f"[ner-eval-vllm] generating {len(prompts)} completions...")
    outputs = llm.generate(prompts, sampling_params)

    # --- Score ---
    y_true: list[str] = []
    y_pred: list[str] = []
    all_samples: list[dict] = []
    shown: list[dict] = []

    for i, (output, example) in enumerate(zip(outputs, examples)):
        completion = output.outputs[0].text

        pred_set = schema.parse(completion)
        gold_set = schema.gold_set(example["tokens"], example[label_column])

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
        if i < args.show:
            shown.append(sample_record)

    # --- Report ---
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

    # --- Save ---
    result = {
        "run": str(args.run),
        "base": str(base) if is_adapter else None,
        "dataset": dataset_key,
        "split": args.split,
        "examples": len(examples),
        "parse_ok": parse_ok_count,
        "parse_fail": parse_fail_count,
        "report": report_dict,
        "examples_shown": shown,
        "samples": all_samples,
    }

    out_path = args.output or (args.run / f"ner_eval_vllm_{args.split}.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[ner-eval-vllm] report saved to {out_path}")

    # --- Cleanup ---
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[ner-eval-vllm] cleaned up merged model at {tmp_dir}")


if __name__ == "__main__":
    main()
