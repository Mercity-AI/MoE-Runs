"""
LEC Pretraining - LLaMA + PLE + Router via HuggingFace Trainer
==============================================================
Trains a LLaMA-style model with Lightweight Expert Conditioning from scratch.
All settings mirror initial_lec.py except the base architecture is LLaMA.

Install:
    pip install torch transformers datasets accelerate wandb
    pip install flash-attn --no-build-isolation

Run:
    python initial_lec_llama.py
"""

CONFIG = {
    # Model
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 4096,
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_2",
    "tokenizer_name": "NousResearch/Meta-Llama-3.1-8B",
    # LEC
    "num_experts": 8,
    "top_k": 1,
    "ple_dim": 128,
    "lec_lr_multiplier": 3.0,
    "lec_warmdown_steps": 10000,
    "lb_ema_alpha": 0.1,
    "lb_bias_lr": 1e-3,
    "entropy_collapse_threshold": 0.5,
    # Training
    "learning_rate": 3e-4,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 2000,
    "max_steps": 50000,
    "per_device_batch_size": 4,
    "grad_accum_steps": 24,
    "max_seq_len": 4096,
    # Data
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 2_048,
    # Eval
    "eval_every_steps": 500,
    "save_every_steps": 2000,
    "log_every_steps": 10,
    # Misc
    "seed": 42,
    "output_dir": "./checkpoints_llama_lec",
    "use_wandb": False,
    "wandb_project": "llama-lec-pretrain",
    "dataloader_workers": 0,
    "torch_compile": True,
    "torch_compile_mode": "default",
}

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)


class PLETable(nn.Module):
    """Per-layer expert embedding table [K, L, d_ple]."""

    def __init__(self, num_experts, num_layers, ple_dim, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.zeros(num_experts, num_layers, ple_dim))
        self.proj = nn.Linear(ple_dim, hidden_size, bias=False)
        self._init_orthogonal()

    def _init_orthogonal(self):
        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                if self.num_experts <= self.ple_dim:
                    q, _ = torch.linalg.qr(torch.randn(self.ple_dim, self.ple_dim))
                    self.embeddings[:, layer_idx, :] = q[: self.num_experts] * 0.02
                else:
                    nn.init.normal_(self.embeddings[:, layer_idx, :], std=0.01)

    def get(self, layer_idx: int, expert_ids: torch.Tensor) -> torch.Tensor:
        vecs = self.embeddings[:, layer_idx, :]
        flat = vecs[expert_ids.reshape(-1)]
        return self.proj(flat).reshape(*expert_ids.shape, self.hidden_size)

    def pairwise_cosine(self, layer_idx: int = 0) -> float:
        with torch.no_grad():
            vecs = self.proj(self.embeddings[:, layer_idx, :]).float()
            norms = F.normalize(vecs, dim=-1)
            sim = norms @ norms.T
            mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=sim.device)
            return sim[mask].mean().item()


class LECRouter(nn.Module):
    """Linear router with gradient-free EMA load-balancing bias."""

    def __init__(
        self, hidden_size, num_experts, top_k, lb_ema_alpha=0.1, lb_bias_lr=1e-3
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lb_ema_alpha = lb_ema_alpha
        self.lb_bias_lr = lb_bias_lr
        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)
        self.register_buffer("lb_bias", torch.zeros(num_experts))
        self.register_buffer("expert_ema_load", torch.ones(num_experts) / num_experts)

    def forward(self, h):
        logits = self.linear(h)
        biased = logits + self.lb_bias.to(logits.dtype)
        probs = F.softmax(biased, dim=-1)
        top_w, top_ids = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return top_ids, top_w, logits

    @torch.no_grad()
    def update_load_balance(self, expert_ids):
        flat = expert_ids.reshape(-1).cpu()
        counts = (
            torch.bincount(flat, minlength=self.num_experts)
            .float()
            .to(self.lb_bias.device)
        )
        load = counts / counts.sum().clamp_min(1.0)
        self.expert_ema_load = (
            1 - self.lb_ema_alpha
        ) * self.expert_ema_load + self.lb_ema_alpha * load
        self.lb_bias -= self.lb_bias_lr * (
            self.expert_ema_load - 1.0 / self.num_experts
        )

    def entropy_norm(self, logits):
        with torch.no_grad():
            p = F.softmax(logits.float(), dim=-1)
            h = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean().item()
            return h / math.log(self.num_experts)

    @staticmethod
    def batch_load(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        with torch.no_grad():
            flat = expert_ids.reshape(-1)
            counts = torch.bincount(flat, minlength=num_experts).float()
            return counts / counts.sum().clamp_min(1.0)


class LlamaLEC(LlamaForCausalLM):
    """
    LLaMA with PLE conditioning injected via forward hooks.
    All weights trained jointly from random init.

    Routing: once from embeddings, expert assignment reused at every layer.
    Each layer uses its own PLE vector E[k, layer_idx, :].
    """

    def __init__(self, config, lec_config: dict):
        super().__init__(config)
        self.lec_config = lec_config

        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_experts = lec_config["num_experts"]

        self.router = LECRouter(
            hidden_size,
            num_experts,
            lec_config["top_k"],
            lec_config["lb_ema_alpha"],
            lec_config["lb_bias_lr"],
        )
        self.ple = PLETable(
            num_experts,
            num_layers,
            lec_config["ple_dim"],
            hidden_size,
        )

        self._current_expert_ids = None
        self._hook_handles = []
        self._register_ple_hooks()

        base = sum(
            p.numel()
            for n, p in self.named_parameters()
            if "router" not in n and "ple" not in n
        )
        lec = sum(p.numel() for p in self.router.parameters()) + sum(
            p.numel() for p in self.ple.parameters()
        )
        print(
            f"[LEC] Base: {base:,} | LEC overhead: {lec:,} ({100 * lec / (base + lec):.3f}%)"
        )

    def _register_ple_hooks(self):
        for layer_idx, layer in enumerate(self.model.layers):
            handle = layer.register_forward_pre_hook(self._make_hook(layer_idx))
            self._hook_handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, args):
            if not isinstance(args, tuple) or not isinstance(args[0], torch.Tensor):
                return args
            if self._current_expert_ids is None:
                return args
            hidden = args[0]
            ple_shift = self.ple.get(layer_idx, self._current_expert_ids)
            ple_shift = ple_shift.to(dtype=hidden.dtype, device=hidden.device)
            return (hidden + ple_shift,) + args[1:]

        return hook

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        forced_expert_id=None,
        **kwargs,
    ):
        embeds = self.model.embed_tokens(input_ids)
        expert_ids, _, router_logits = self.router(embeds)
        expert_ids = expert_ids[..., 0]

        if forced_expert_id is not None:
            expert_ids = torch.full_like(expert_ids, forced_expert_id)

        self._current_expert_ids = expert_ids

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        self._current_expert_ids = None

        if self.training and forced_expert_id is None:
            self.router.update_load_balance(expert_ids)

        metrics = {
            "router_entropy": self.router.entropy_norm(router_logits),
            "ple_pairwise_cosine": self.ple.pairwise_cosine(),
        }

        batch_load = self.router.batch_load(expert_ids, self.router.num_experts)
        metrics["expert_load_balance_std"] = batch_load.std(unbiased=False).item()
        metrics["expert_load_balance_min"] = batch_load.min().item()
        metrics["expert_load_balance_max"] = batch_load.max().item()
        metrics["expert_utilization"] = (batch_load > 0).float().mean().item()
        for idx, load in enumerate(self.router.expert_ema_load.tolist()):
            metrics[f"expert_load_{idx}"] = float(load)

        outputs.expert_ids = expert_ids.detach()
        outputs.router_logits = (
            router_logits.detach() if router_logits is not None else None
        )
        outputs.lec_metrics = metrics
        return outputs


def build_model(config: dict) -> LlamaLEC:
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

    lec_config = {
        key: config[key]
        for key in ["num_experts", "top_k", "ple_dim", "lb_ema_alpha", "lb_bias_lr"]
    }

    model = LlamaLEC._from_config(
        llama_cfg,
        lec_config=lec_config,
        attn_implementation=config["attn_implementation"],
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] LLaMA + LEC: {n:,} params ({n / 1e9:.2f}B)")
    return model


def build_tokenizer(config: dict):
    tok = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tok.model_max_length = config["max_seq_len"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(
        f"[Tokenizer] vocab_size={tok.vocab_size} | total_tokens={len(tok)} | eos={tok.eos_token_id}"
    )
    return tok


class PackedFineWebDataset(IterableDataset):
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
        print(f"[Data] {config['dataset_name']} streaming")

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


class LECTrainer(Trainer):
    """Trainer with separate LR for LEC params and cosine floor."""

    def __init__(self, *args, min_lr: float = 3e-5, lec_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_lr = min_lr
        self.lec_config = lec_config or {}
        self._metric_sums = {"train": {}, "eval": {}}
        self._metric_counts = {"train": 0, "eval": 0}

    def _accumulate_metrics(self, split: str, metrics: dict):
        if not metrics:
            return

        bucket = self._metric_sums[split]
        for key, value in metrics.items():
            bucket[key] = bucket.get(key, 0.0) + float(value)
        self._metric_counts[split] += 1

    def _flush_metrics(self, split: str, prefix: str = "") -> dict:
        count = self._metric_counts[split]
        if count == 0:
            return {}

        averaged = {
            f"{prefix}{key}": value / count
            for key, value in self._metric_sums[split].items()
        }
        self._metric_sums[split] = {}
        self._metric_counts[split] = 0
        return averaged

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        split = "train" if model.training else "eval"
        self._accumulate_metrics(split, getattr(outputs, "lec_metrics", None))

        if return_outputs:
            return loss, outputs
        return loss

    def create_optimizer(self):
        model = self.model
        decay_base = [
            p
            for n, p in model.named_parameters()
            if p.dim() >= 2 and "router" not in n and "ple" not in n
        ]
        no_decay_base = [
            p
            for n, p in model.named_parameters()
            if p.dim() < 2 and "router" not in n and "ple" not in n
        ]
        lec_params = list(model.router.parameters()) + list(model.ple.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_base,
                    "weight_decay": self.args.weight_decay,
                    "tag": "base",
                },
                {"params": no_decay_base, "weight_decay": 0.0, "tag": "base_nd"},
                {
                    "params": lec_params,
                    "weight_decay": self.args.weight_decay,
                    "tag": "lec",
                },
            ],
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            fused=torch.cuda.is_available(),
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        max_lr = self.args.learning_rate
        min_lr = self.min_lr
        warmup = self.args.warmup_steps
        mult = self.lec_config.get("lec_lr_multiplier", 3.0)
        warmdown = self.lec_config.get("lec_warmdown_steps", 10000)

        def base_lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(num_training_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + (max_lr - min_lr) * cosine) / max_lr

        def lec_lr_lambda(step):
            base = base_lr_lambda(step)
            effective_mult = mult - (mult - 1.0) * min(step / max(warmdown, 1), 1.0)
            return base * effective_mult

        self.lr_scheduler = LambdaLR(
            self.optimizer, [base_lr_lambda, base_lr_lambda, lec_lr_lambda]
        )

    def log(self, logs, start_time=None):
        if "loss" in logs:
            try:
                logs["ppl"] = round(math.exp(min(logs["loss"], 20)), 2)
            except Exception:
                pass
            logs.update(self._flush_metrics("train"))
        if "eval_loss" in logs:
            try:
                logs["eval_ppl"] = round(math.exp(min(logs["eval_loss"], 20)), 2)
            except Exception:
                pass
            logs.update(self._flush_metrics("eval", prefix="eval_"))
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


def train(config: dict):
    torch.manual_seed(config["seed"])

    model = build_model(config)
    tokenizer = build_tokenizer(config)

    tokenizer_size = len(tokenizer)
    if tokenizer_size != model.config.vocab_size:
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
        f"[Train] Total: {tokens_per_step * config['max_steps'] / 1e9:.1f}B tokens\n"
    )

    lec_config = {
        k: config[k]
        for k in [
            "lec_lr_multiplier",
            "lec_warmdown_steps",
            "num_experts",
            "top_k",
            "ple_dim",
            "lb_ema_alpha",
            "lb_bias_lr",
        ]
    }

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
        lr_scheduler_type="cosine",
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
        run_name="llama_lec",
        remove_unused_columns=False,
        label_names=["labels"],
        torch_compile=config["torch_compile"],
        torch_compile_mode=config["torch_compile_mode"],
    )

    trainer = LECTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        min_lr=config["learning_rate"] * config["min_lr_ratio"],
        lec_config=lec_config,
    )

    trainer.train()
    model.remove_hooks()
    print("[Train] LLaMA LEC pretraining complete.")


if __name__ == "__main__":
    train(CONFIG)
