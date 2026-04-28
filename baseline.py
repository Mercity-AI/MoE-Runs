"""
Baseline Pretraining — OLMo-2 Architecture via HuggingFace Trainer
====================================================================
Trains a dense OLMo-2 model from scratch on FineWeb using:
  - Real Olmo2ForCausalLM from transformers (no hand-rolled architecture)
  - HuggingFace Trainer (handles grad accum, mixed precision, checkpointing)
  - Flash Attention 2 for fast attention at long sequence lengths
  - cl100k tokenizer (same as OLMo-2, vocab=100352)
  - Document packing (no padding waste)

This is the B1 baseline in the LEC ablation plan.
Every result from the LEC run is a delta relative to this.

Install:
    pip install torch transformers datasets accelerate wandb
    pip install flash-attn --no-build-isolation

Run:
    python pretrain_baseline.py
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # --- Model (OLMo-2 architecture, ~1B params) ---
    # Field names match Olmo2Config exactly — no translation needed.
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,  # GQA: 8 KV, 16 Q heads
    "intermediate_size": 8192,  # SwiGLU hidden dim (~4x hidden_size)
    "vocab_size": 100352,  # cl100k vocab (OLMo-2 native)
    "max_position_embeddings": 4096,
    "rope_theta": 500000.0,  # OLMo-2 value; needed for 4096+ context
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,  # OLMo-2 does NOT tie embed/unembed
    "attention_bias": False,
    "hidden_act": "silu",
    # Flash Attention 2 for ~2-3x attention speedup.
    # Fall back to "sdpa" if flash-attn is not installed.
    "attn_implementation": "flash_attention_2",
    # --- Tokenizer ---
    # Load OLMo-2's cl100k tokenizer directly from the model card.
    "tokenizer_name": "allenai/OLMo-2-0425-1B",
    # --- Training ---
    "learning_rate": 3e-5,
    "min_lr_ratio": 0.1,  # cosine decays to 10% of peak LR
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 1000,
    "max_steps": 5000,
    "per_device_batch_size": 4,
    "grad_accum_steps": 12,
    "max_seq_len": 2048,
    # --- Data ---
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    # --- Eval & logging ---
    "eval_every_steps": 500,
    "save_every_steps": 2000,
    "log_every_steps": 10,
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 2_048,
    # --- Misc ---
    "seed": 42,
    "output_dir": "./checkpoints_baseline",
    "use_wandb": False,
    "wandb_project": "lec-baseline-pretrain",
    "dataloader_workers": 0,
}

# =============================================================================
# IMPORTS
# =============================================================================

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    Olmo2Config,
    Olmo2ForCausalLM,
    Trainer,
    TrainingArguments,
)

def build_model(config: dict) -> Olmo2ForCausalLM:
    """Instantiate OLMo-2 from scratch (random init, no pretrained weights)."""
    olmo_cfg = Olmo2Config(
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

    model = Olmo2ForCausalLM._from_config(
        olmo_cfg,
        attn_implementation=config["attn_implementation"],
    )

    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] OLMo-2 baseline: {n:,} params ({n / 1e9:.2f}B)")
    print(f"[Model] Attention: {config['attn_implementation']}")
    return model


# =============================================================================
# TOKENIZER
# =============================================================================


def build_tokenizer(config: dict):
    """
    Load OLMo-2's cl100k tokenizer.
    Set model_max_length to suppress the spurious "> 1024 tokens" warning
    that comes from the tokenizer's default max_length setting.
    """
    tok = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tok.model_max_length = config["max_seq_len"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[Tokenizer] vocab_size={tok.vocab_size} | eos={tok.eos_token_id}")
    return tok


# =============================================================================
# DATASET — streaming + packing
# =============================================================================


class PackedFineWebDataset(IterableDataset):
    """
    Streaming FineWeb with document packing.

    Documents are concatenated end-to-end with EOS between them and sliced
    into exactly max_seq_len tokens. No padding, no wasted compute.

    Yields: {"input_ids": [L], "labels": [L]}
    Labels are identical to input_ids because HF causal LM heads shift the
    labels internally when computing the next-token loss.

    Cross-document attention is NOT masked — standard practice for pretraining.
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
        print(
            f"[Data] {config['dataset_name']} / {config['dataset_config']} streaming ✓"
        )

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
                    doc = next(dataset_iter)
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
# TRAINER — cosine LR with configurable floor + perplexity logging
# =============================================================================


class PretrainTrainer(Trainer):
    """
    HuggingFace Trainer extended with:
      - Cosine LR decay to a configurable minimum (default Trainer decays to 0)
      - Perplexity in logged metrics
    """

    def __init__(self, *args, min_lr: float = 3e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_lr = min_lr

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        max_lr = self.args.learning_rate
        min_lr = self.min_lr
        warmup = self.args.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(num_training_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + (max_lr - min_lr) * cosine) / max_lr

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

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

    # Resize embeddings if tokenizer vocab differs from config (safety net)
    if tokenizer.vocab_size != model.config.vocab_size:
        print(
            f"[Warning] Resizing embeddings: {model.config.vocab_size} → {tokenizer.vocab_size}"
        )
        model.resize_token_embeddings(tokenizer.vocab_size)

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
        lr_scheduler_type="cosine",  # overridden by PretrainTrainer
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=config["log_every_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_every_steps"],
        save_strategy="steps",
        save_steps=config["save_every_steps"],
        save_total_limit=3,
        dataloader_num_workers=config["dataloader_workers"],
        seed=config["seed"],
        report_to="wandb" if config["use_wandb"] else "none",
        run_name="baseline",
        remove_unused_columns=False,
        label_names=["labels"],
        torch_compile=False,
    )

    trainer = PretrainTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        min_lr=config["learning_rate"] * config["min_lr_ratio"],
    )

    trainer.train()
    print("[Train] Baseline pretraining complete.")


if __name__ == "__main__":
    train(CONFIG)
