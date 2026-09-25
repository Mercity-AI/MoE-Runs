"""NER supervised fine-tuning (LoRA or full) with WandB logging.

Reads all parameters from a YAML config file.  Logs training loss, eval
accuracy, per-layer weight/grad norms, and generation-based NER precision /
recall / F1 (micro, macro, per-class) to Weights & Biases.

Usage::

    python ner_sft.py configs/smoke_test.yaml
    python ner_sft.py configs/qknorm_baseline.yaml
"""

from __future__ import annotations

# Standard library
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

# Third party
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Local
import ner_data


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_FLOAT_KEYS = {"lr", "warmup_ratio", "lora_dropout", "epochs"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in _FLOAT_KEYS:
        if key in cfg and cfg[key] is not None:
            cfg[key] = float(cfg[key])
    if cfg.get("lr") is None:
        cfg["lr"] = 2e-4 if cfg.get("method") == "lora" else 1e-5
    return cfg


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def find_lora_target_modules(model: nn.Module) -> list[str]:
    """Discover ``nn.Linear`` leaf names for LoRA, excluding lm_head.

    Also excludes leaf names shared with ``nn.Embedding`` modules (e.g. the
    n-gram hash tables in LongCat models) — wrapping those breaks attribute
    access the model's forward pass relies on.
    """
    linear_names: set[str] = set()
    embedding_names: set[str] = set()
    for module_name, module in model.named_modules():
        leaf = module_name.split(".")[-1]
        if isinstance(module, nn.Linear):
            if leaf in {"lm_head"} or leaf.isdigit():
                continue
            linear_names.add(leaf)
        elif isinstance(module, nn.Embedding):
            embedding_names.add(leaf)
    return sorted(linear_names - embedding_names)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def preprocess_logits_for_metrics(
    logits: torch.Tensor | tuple, labels: torch.Tensor,
) -> torch.Tensor:
    """Keep only argmax predictions to avoid OOM on large vocabs."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Next-token prediction accuracy on non-masked target positions."""
    preds, labels = eval_pred
    # CausalLM: logits[j] predicts token[j+1], so shift to align.
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100
    if mask.sum() == 0:
        return {"eval_accuracy": 0.0}
    return {"eval_accuracy": float((preds[mask] == labels[mask]).mean())}


def collect_layer_norms(model: nn.Module) -> dict[str, float]:
    """Per-transformer-layer L2 weight and grad norms (trainable params only)."""
    weight_sq: dict[int, float] = defaultdict(float)
    grad_sq: dict[int, float] = defaultdict(float)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None:
            continue
        weight_sq[layer_idx] += param.data.float().norm(2).item() ** 2
        if param.grad is not None:
            grad_sq[layer_idx] += param.grad.float().norm(2).item() ** 2

    norms: dict[str, float] = {}
    for idx in sorted(weight_sq):
        norms[f"weight_norm/layer_{idx}"] = weight_sq[idx] ** 0.5
    for idx in sorted(grad_sq):
        norms[f"grad_norm/layer_{idx}"] = grad_sq[idx] ** 0.5
    return norms


def entity_metrics_from_sets(
    y_true: list[str],
    y_pred: list[str],
    entity_types: list[str],
) -> dict[str, float]:
    """Compute per-class, micro, and macro P/R/F1 with scikit-learn."""
    if not y_true:
        return {}
    report = classification_report(
        y_true, y_pred, labels=entity_types,
        output_dict=True, zero_division=0,
    )
    metrics: dict[str, float] = {}
    for type_name in entity_types:
        if type_name in report:
            metrics[f"ner_{type_name}_precision"] = report[type_name]["precision"]
            metrics[f"ner_{type_name}_recall"] = report[type_name]["recall"]
            metrics[f"ner_{type_name}_f1"] = report[type_name]["f1-score"]
    if "micro avg" in report:
        metrics["ner_micro_precision"] = report["micro avg"]["precision"]
        metrics["ner_micro_recall"] = report["micro avg"]["recall"]
        metrics["ner_micro_f1"] = report["micro avg"]["f1-score"]
    if "macro avg" in report:
        metrics["ner_macro_precision"] = report["macro avg"]["precision"]
        metrics["ner_macro_recall"] = report["macro avg"]["recall"]
        metrics["ner_macro_f1"] = report["macro avg"]["f1-score"]
    return metrics


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CheckpointConfigCallback(TrainerCallback):
    """Copy the training config into each saved checkpoint directory."""

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.exists():
            with open(ckpt_dir / "ner_config.yaml", "w") as f:
                yaml.dump(
                    self.config_dict, f,
                    default_flow_style=False, sort_keys=False,
                )


# ---------------------------------------------------------------------------
# Trainer subclass with generation-based NER eval
# ---------------------------------------------------------------------------

class NERTrainer(Trainer):
    """Adds generation-based NER P/R/F1 evaluation at each eval step."""

    def __init__(
        self,
        *args,
        ner_schema: ner_data.NERSchema | None = None,
        eval_raw=None,
        label_column: str = "ner_tags",
        gen_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ner_schema = ner_schema
        self.eval_raw = eval_raw
        self.label_column = label_column
        self.gen_config = gen_config or {}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        norms = collect_layer_norms(self.model)
        if norms and wandb.run is not None:
            wandb.log(norms, step=self.state.global_step, commit=False)

        return metrics

    def _generation_eval(self) -> dict[str, float]:
        """Generate completions, parse entities, score with scikit-learn."""
        model = self.model
        model.eval()

        tokenizer = self.processing_class
        schema = self.ner_schema
        raw = self.eval_raw
        label_column = self.label_column
        max_new_tokens = self.gen_config.get("max_new_tokens", 128)
        limit = min(
            self.gen_config.get("generation_eval_limit", 200), len(raw),
        )
        batch_size = self.gen_config.get("eval_batch_size", 8)
        max_seq_len = self.gen_config.get("max_seq_len", 512)

        device = next(model.parameters()).device
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

        all_prompt_ids: list[list[int]] = []
        all_examples: list[dict] = []
        for i, example in enumerate(raw):
            if i >= limit:
                break
            prompt = schema.build_prompt(example["tokens"])
            prompt_ids = bos + tokenizer(
                prompt, add_special_tokens=False,
            )["input_ids"]
            all_prompt_ids.append(prompt_ids[-max_seq_len:])
            all_examples.append(example)

        y_true: list[str] = []
        y_pred: list[str] = []

        model.config.use_cache = True
        try:
            for batch_start in range(0, len(all_prompt_ids), batch_size):
                batch_prompts = all_prompt_ids[batch_start:batch_start + batch_size]
                batch_examples = all_examples[batch_start:batch_start + batch_size]

                max_len = max(len(p) for p in batch_prompts)
                padded_ids = [[pad_id] * (max_len - len(p)) + p for p in batch_prompts]
                attn_masks = [[0] * (max_len - len(p)) + [1] * len(p) for p in batch_prompts]

                input_ids = torch.tensor(padded_ids, device=device)
                attention_mask = torch.tensor(attn_masks, device=device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
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

                    for t, _ in pred_set & gold_set:
                        y_true.append(t)
                        y_pred.append(t)
                    for t, _ in gold_set - pred_set:
                        y_true.append(t)
                        y_pred.append("O")
                    for t, _ in pred_set - gold_set:
                        y_true.append("O")
                        y_pred.append(t)
        finally:
            model.config.use_cache = False

        return entity_metrics_from_sets(y_true, y_pred, sorted(schema.types))


# ---------------------------------------------------------------------------
# Data verification
# ---------------------------------------------------------------------------

def print_data_examples(
    tokenizer, schema: ner_data.NERSchema, raw, label_column: str, n: int = 3,
) -> None:
    """Print a few examples showing prompt / target for verification."""
    print("\n" + "=" * 70)
    print("DATA VERIFICATION — prompt → target for first examples")
    print("=" * 70)
    for i, example in enumerate(raw):
        if i >= n:
            break
        prompt = schema.build_prompt(example["tokens"])
        target = schema.target_text(example["tokens"], example[label_column])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        gold = schema.gold_set(example["tokens"], example[label_column])
        print(f"\n--- Example {i} ---")
        print(f"TOKENS : {example['tokens']}")
        print(f"TAGS   : {example[label_column]}")
        print(f"PROMPT ({len(prompt_ids)} tok):\n{prompt}")
        print(f"TARGET ({len(target_ids)} tok): {target!r}")
        print(f"GOLD   : {sorted(gold)}")
        print(f"TOTAL  : {len(prompt_ids) + len(target_ids) + 2} tokens (BOS+prompt+target+EOS)")
        print(f"LABELS : [-100]*{len(prompt_ids)+1} then target_ids+[EOS]")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NER SFT — all params from YAML config",
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"[ner-sft] ERROR: output directory not empty: {output_dir}"
        )

    # --- WandB ---------------------------------------------------------------
    wandb.init(
        project=cfg.get("wandb_project", "ner-sft"),
        name=cfg.get("wandb_run_name"),
        config=cfg,
    )

    # --- Model ---------------------------------------------------------------
    checkpoint = Path(cfg["checkpoint"])
    print(f"[ner-sft] loading model from {checkpoint}")

    model_kwargs: dict = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    attn_impl = cfg.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ner-sft] model type  : {model.config.model_type}")
    print(f"[ner-sft] total params: {total_params:,}")
    print(f"[ner-sft] attn impl   : {getattr(model.config, '_attn_implementation', 'default')}")

    # --- LoRA ----------------------------------------------------------------
    if cfg["method"] == "lora":
        targets = find_lora_target_modules(model)
        print(f"[ner-sft] LoRA targets: {targets}")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=targets,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if cfg.get("grad_checkpointing"):
            model.enable_input_require_grads()

    if cfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()

    # --- Data ----------------------------------------------------------------
    dataset_key = cfg.get("dataset", ner_data.DEFAULT_DATASET)
    output_format = cfg.get("output_format", ner_data.DEFAULT_OUTPUT_FORMAT)

    train_ds, schema = ner_data.build_examples(
        tokenizer,
        dataset_key=dataset_key,
        split="train",
        max_seq_len=cfg["max_seq_len"],
        limit=cfg.get("train_limit"),
        output_format=output_format,
    )
    eval_ds, _ = ner_data.build_examples(
        tokenizer,
        dataset_key=dataset_key,
        split="validation",
        max_seq_len=cfg["max_seq_len"],
        limit=cfg.get("eval_limit"),
        output_format=output_format,
    )
    eval_raw, eval_schema, label_column = ner_data.load_split(
        dataset_key, "validation", cfg.get("eval_limit"),
        output_format=output_format,
    )

    print(f"[ner-sft] dataset : {dataset_key}")
    print(f"[ner-sft] format  : {output_format}")
    print(f"[ner-sft] types   : {schema.types}")
    print(f"[ner-sft] train   : {len(train_ds)} examples")
    print(f"[ner-sft] eval    : {len(eval_ds)} examples")

    print_data_examples(tokenizer, schema, eval_raw, label_column)

    # --- Trainer -------------------------------------------------------------
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        padding="longest",
        return_tensors="pt",
    )

    import math
    _max_steps = cfg.get("max_steps", -1)
    if _max_steps > 0:
        _total_steps = _max_steps
    else:
        _steps_per_epoch = math.ceil(len(train_ds) / (cfg["batch_size"] * cfg["grad_accum"]))
        _total_steps = _steps_per_epoch * cfg["epochs"]
    _warmup_steps = int(cfg["warmup_ratio"] * _total_steps)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_steps=_max_steps,
        learning_rate=cfg["lr"],
        warmup_steps=_warmup_steps,
        lr_scheduler_type="cosine",
        optim=cfg.get("optim", "adamw_torch_fused"),
        bf16=torch.cuda.is_available(),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        report_to=["wandb"],
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        label_names=["labels"],
        seed=cfg["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=cfg.get("grad_checkpointing", False),
    )

    gen_config = {
        "max_new_tokens": cfg.get("max_new_tokens", 128),
        "generation_eval_limit": cfg.get("generation_eval_limit", 200),
        "eval_batch_size": cfg.get("eval_batch_size", 8),
        "max_seq_len": cfg["max_seq_len"],
    }

    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[CheckpointConfigCallback(cfg)],
        ner_schema=eval_schema,
        eval_raw=eval_raw,
        label_column=label_column,
        gen_config=gen_config,
    )

    # --- Train ---------------------------------------------------------------
    print("[ner-sft] starting training")
    train_result = trainer.train()
    metrics = train_result.metrics

    print("[ner-sft] running final evaluation")
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    # --- Save ----------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    shutil.copy2(args.config, output_dir / "ner_config.yaml")

    run_config = {
        "base_checkpoint": str(checkpoint.resolve()),
        "dataset": dataset_key,
        "method": cfg["method"],
        "lr": cfg["lr"],
        "epochs": cfg["epochs"],
        "max_steps": cfg.get("max_steps", -1),
        "batch_size": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"],
        "max_seq_len": cfg["max_seq_len"],
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "lora": (
            {
                "r": cfg["lora_r"],
                "alpha": cfg["lora_alpha"],
                "dropout": cfg["lora_dropout"],
            }
            if cfg["method"] == "lora"
            else None
        ),
        "train_metrics": {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        },
    }
    (output_dir / "ner_sft_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )

    wandb.finish()

    print(f"\n[ner-sft] saved to {output_dir.resolve()}")
    print(
        f"[ner-sft] evaluate with:\n"
        f"  python ner_eval.py --run {output_dir} "
        f"{'--base ' + str(checkpoint) + ' ' if cfg['method'] == 'lora' else ''}"
        f"--dataset {dataset_key}"
    )


if __name__ == "__main__":
    main()
