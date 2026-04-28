"""
Baseline Pretraining - LLaMA Architecture via HuggingFace Trainer
=================================================================
Trains a dense LLaMA-style model from scratch on FineWeb using:
  - Real LlamaForCausalLM from transformers
  - HuggingFace Trainer for grad accum, mixed precision, and checkpointing
  - Flash Attention 2 for fast attention at long sequence lengths
  - Llama 3 tokenizer (128256 vocab)
  - Document packing for efficient next-token pretraining

Run:
    python baseline_llama.py
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # --- Model (LLaMA architecture, ~1B-ish params with this shape) ---
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_2",
    # --- Tokenizer ---
    "tokenizer_name": "meta-llama/Llama-3.2-1B",
    # --- Training ---
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 125,  # ~5% of 2500 steps
    "max_steps": 2500,
    "per_device_batch_size": 12,
    "grad_accum_steps": 8,
    "max_seq_len": 4096,
    # --- Data ---
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 256,
    # --- Eval & logging ---
    "eval_every_steps": 500,
    "save_every_steps": 500,
    "log_every_steps": 1,
    # --- Misc ---
    "seed": 42,
    "output_dir": "./checkpoints_llama_baseline_final",
    "use_wandb": True,
    "wandb_project": "lec-llama-baseline-pretrain-final",
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,
    "torch_compile_mode": "default",
}

# =============================================================================
# IMPORTS
# =============================================================================

import math
from typing import Optional

import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)


def build_model(config: dict) -> LlamaForCausalLM:
    """Instantiate a LLaMA model from scratch."""
    llama_cfg = LlamaConfig(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"],
        initializer_range=config["initializer_range"],
        tie_word_embeddings=config["tie_word_embeddings"],
        attention_bias=config["attention_bias"],
        hidden_act=config["hidden_act"],
    )

    model = LlamaForCausalLM._from_config(
        llama_cfg,
        attn_implementation=config["attn_implementation"],
    )

    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] LLaMA baseline: {n:,} params ({n / 1e9:.2f}B)")
    print(f"[Model] Attention: {config['attn_implementation']}")
    return model


# =============================================================================
# TOKENIZER
# =============================================================================


def build_tokenizer(config: dict):
    """Load the tokenizer used for pretraining."""
    tok = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tok.model_max_length = config["max_seq_len"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(
        f"[Tokenizer] vocab_size={tok.vocab_size} | total_tokens={len(tok)} | eos={tok.eos_token_id}"
    )
    return tok


# =============================================================================
# DATASET - streaming + packing
# =============================================================================


class PackedFineWebDataset(IterableDataset):
    """
    Streaming FineWeb with document packing.

    Documents are concatenated end-to-end with EOS between them and sliced
    into exactly max_seq_len tokens. No padding, no wasted compute.

    Labels are identical to input_ids because HF causal LM heads shift labels
    internally when computing next-token loss.
    """

    def __init__(
        self,
        config: dict,
        tokenizer,
        seed: int = 42,
        max_examples: Optional[int] = None,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_seq_len = config["max_seq_len"]
        self.eos_id = tokenizer.eos_token_id
        self.seed = seed
        self.epoch = 0
        self.buffer_size = config["streaming_buffer_size"]
        self.max_examples = max_examples

        self.dataset = load_dataset(
            config["dataset_name"],
            name=config["dataset_config"],
            split="train",
            streaming=True,
        )
        print(f"[Data] {config['dataset_name']} / {config['dataset_config']} streaming")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        dataset = self.dataset.shuffle(
            seed=self.seed + self.epoch,
            buffer_size=self.buffer_size,
        )
        dataset_iter = iter(dataset)
        buffer = []
        examples_yielded = 0

        while True:
            if self.max_examples is not None and examples_yielded >= self.max_examples:
                return

            while len(buffer) < self.max_seq_len:
                try:
                    doc = next(dataset_iter)  # fixed typo: was dataset_ite
                except StopIteration:
                    return

                text = doc.get("text", "")
                if not text or not text.strip():
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if tokens:
                    buffer.extend(tokens)
                    buffer.append(self.eos_id)

            chunk = buffer[: self.max_seq_len]
            buffer = buffer[self.max_seq_len :]

            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "labels": torch.tensor(chunk, dtype=torch.long),
            }
            examples_yielded += 1


# =============================================================================
# TRAINER - perplexity logging
# =============================================================================


class PretrainTrainer(Trainer):
    """Trainer with perplexity logging. LR schedule handled by TrainingArguments."""

    def log(self, logs, start_time=None):
        if "loss" in logs:
            try:
                logs["ppl"] = round(math.exp(min(logs["loss"], 20)), 2)
            except Exception:
                pass
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


# =============================================================================
# ENTRY
# =============================================================================


def train(config: dict):
    torch.manual_seed(config["seed"])

    model = build_model(config)
    tokenizer = build_tokenizer(config)

    tokenizer_size = len(tokenizer)
    if tokenizer_size != model.config.vocab_size:
        print(
            f"[Warning] Resizing embeddings: {model.config.vocab_size} -> {tokenizer_size}"
        )
        model.resize_token_embeddings(tokenizer_size)

    train_ds = PackedFineWebDataset(config, tokenizer, seed=config["seed"])
    eval_ds = PackedFineWebDataset(
        config,
        tokenizer,
        seed=config["seed"] + 9999,
        max_examples=config["eval_max_examples"],
    )

    tokens_per_step = (
        config["per_device_batch_size"]
        * config["grad_accum_steps"]
        * config["max_seq_len"]
    )
    print(f"\n[Train] Tokens/step: {tokens_per_step:,}")
    print(
        f"[Train] Total: {tokens_per_step * config['max_steps'] / 1e9:.1f}B tokens "
        f"over {config['max_steps']:,} steps\n"
    )

    args = TrainingArguments(
        output_dir=config["output_dir"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["beta1"],
        adam_beta2=config["beta2"],
        max_grad_norm=config["grad_clip"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type="cosine",  # HF native cosine; decays to 0
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=config["log_every_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_every_steps"],
        save_strategy="steps",
        save_steps=config["save_every_steps"],
        save_only_model=True,
        save_total_limit=10,
        dataloader_num_workers=config["dataloader_workers"],
        dataloader_prefetch_factor=config["dataloader_prefetch_factor"],
        seed=config["seed"],
        report_to="wandb" if config["use_wandb"] else "none",
        run_name="llama_baseline",
        remove_unused_columns=False,
        label_names=["labels"],
        torch_compile=config["torch_compile"],
        torch_compile_mode=config["torch_compile_mode"],
    )

    trainer = PretrainTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    print("[Train] LLaMA baseline pretraining complete.")


if __name__ == "__main__":
    train(CONFIG)
