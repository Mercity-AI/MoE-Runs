"""
LEC Pretraining - LLaMA + PLE + Router via HuggingFace Trainer
==============================================================

Hook-based implementation: preserves the official HuggingFace LlamaForCausalLM
forward path, RoPE handling, causal mask construction, attention implementation,
MLP implementation, and built-in causal-LM loss.

LEC injection is done with forward pre-hooks on each decoder layer:
    hidden_states <- hidden_states + PLE(router_l(hidden_states), layer=l)

No custom layer loop. No manual attention implementation. No custom LR scheduler.
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Model
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_2",
    "tokenizer_name": "meta-llama/Llama-3.2-1B",

    # LEC (Phase 1: PLE + Router only)
    "num_experts": 8,
    "ple_dim": 512,
    "lb_ema_alpha": 0.1,
    "lb_bias_lr": 1e-3,
    "entropy_collapse_threshold": 0.5,
    "lec_metric_log_every_steps": 10,

    # Training
    "learning_rate": 3e-4,
    "min_lr_ratio": 0.1,  # retained in config for bookkeeping; HF cosine ignores this
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 250,
    "max_steps": 2500,
    "per_device_batch_size": 12,
    "grad_accum_steps": 8,
    "max_seq_len": 2048,

    # Data
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 256,

    # Eval
    "eval_every_steps": 500,
    "save_every_steps": 250,
    "log_every_steps": 1,

    # Misc
    "seed": 42,
    "output_dir": "./checkpoints_llama_lec",
    "use_wandb": False,
    "wandb_project": "llama-lec-pretrain",
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


# =============================================================================
# PLE TABLE
# =============================================================================

class PLETable(nn.Module):
    """
    Per-Layer Expert Embedding table: [K, L, d_ple].

    Each expert k has one d_ple-dimensional vector per layer l.
    A single shared projection maps d_ple -> hidden_size.
    Vectors are initialized as full-dimensional, mutually orthogonal expert
    directions for symmetry breaking.
    """

    def __init__(self, num_experts: int, num_layers: int, ple_dim: int, hidden_size: int):
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
                    q, _ = torch.linalg.qr(
                        torch.randn(
                            self.ple_dim,
                            self.ple_dim,
                            device=self.embeddings.device,
                            dtype=self.embeddings.dtype,
                        )
                    )
                    self.embeddings[:, layer_idx, :] = q[:, :self.num_experts].T * 0.02
                else:
                    nn.init.normal_(self.embeddings[:, layer_idx, :], std=0.02)

    def get_all_projected(self, layer_idx: int) -> torch.Tensor:
        return self.proj(self.embeddings[:, layer_idx, :])  # [K, H]

    @torch.no_grad()
    def pairwise_cosine(self) -> float:
        total = 0.0
        for layer_idx in range(self.num_layers):
            vecs = self.proj(self.embeddings[:, layer_idx, :]).detach().float()
            norms = F.normalize(vecs, dim=-1)
            sim = norms @ norms.T
            mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=sim.device)
            total += sim[mask].mean().item()
        return total / self.num_layers


# =============================================================================
# ROUTER
# =============================================================================

class LECRouter(nn.Module):
    """One decoder layer's linear router with EMA load-balancing bias."""

    def __init__(self, hidden_size: int, num_experts: int, lb_ema_alpha: float = 0.1, lb_bias_lr: float = 1e-3):
        super().__init__()
        self.num_experts = num_experts
        self.lb_ema_alpha = lb_ema_alpha
        self.lb_bias_lr = lb_bias_lr

        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)

        self.register_buffer("lb_bias", torch.zeros(num_experts))
        self.register_buffer("expert_ema_load", torch.ones(num_experts) / num_experts)
        self._load_accum: Optional[torch.Tensor] = None
        self._accum_count: int = 0

    def forward(self, h: torch.Tensor):
        """
        Straight-through top-1 routing.
        Forward is hard top-1; backward uses softmax gradient.
        """
        logits = self.linear(h)                                      # [B, T, K]
        biased_logits = logits + self.lb_bias.to(logits.dtype)       # [B, T, K]
        probs = F.softmax(biased_logits.float(), dim=-1).to(h.dtype) # [B, T, K]

        expert_ids = probs.argmax(dim=-1)                            # [B, T]
        hard = F.one_hot(expert_ids, num_classes=self.num_experts).to(h.dtype)
        gate = hard + probs - probs.detach()
        return expert_ids, gate, logits, biased_logits, probs

    def accumulate_load(self, expert_ids: torch.Tensor):
        """Accumulate hard expert loads across micro-steps."""
        with torch.no_grad():
            flat = expert_ids.detach().reshape(-1)
            counts = torch.bincount(flat, minlength=self.num_experts).float()
            load = counts / counts.sum().clamp_min(1.0)
            load = load.to(self.lb_bias.device)

            if self._load_accum is None:
                self._load_accum = load
            else:
                self._load_accum += load
            self._accum_count += 1

    @torch.no_grad()
    def flush_load_balance(self):
        """Apply one EMA/bias load-balancing update and clear accumulators."""
        if self._load_accum is None or self._accum_count == 0:
            return
        avg_load = (self._load_accum / self._accum_count).to(self.lb_bias.device)
        self.expert_ema_load = (
            (1 - self.lb_ema_alpha) * self.expert_ema_load
            + self.lb_ema_alpha * avg_load
        )
        self.lb_bias -= self.lb_bias_lr * (self.expert_ema_load - 1.0 / self.num_experts)
        self._load_accum = None
        self._accum_count = 0

    @torch.no_grad()
    def soft_entropy_norm(self, biased_logits: torch.Tensor) -> float:
        p = F.softmax(biased_logits.detach().float(), dim=-1)
        entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean().item()
        return entropy / math.log(self.num_experts)

    @staticmethod
    @torch.no_grad()
    def hard_load_entropy_norm(expert_ids: torch.Tensor, num_experts: int) -> float:
        flat = expert_ids.detach().reshape(-1)
        counts = torch.bincount(flat, minlength=num_experts).float()
        load = counts / counts.sum().clamp_min(1.0)
        entropy = -(load * torch.log(load + 1e-10)).sum().item()
        return entropy / math.log(num_experts)

    @staticmethod
    @torch.no_grad()
    def batch_load(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        flat = expert_ids.detach().reshape(-1)
        counts = torch.bincount(flat, minlength=num_experts).float()
        return counts / counts.sum().clamp_min(1.0)


# =============================================================================
# HOOKED LLAMA + LEC
# =============================================================================

class LlamaLEC(LlamaForCausalLM):
    """
    LLaMA + LEC using forward pre-hooks on the official HF decoder layers.

    We do not reimplement LLaMA forward, RoPE, attention, masks, or MLP.
    HuggingFace handles all internals. Each decoder layer receives a PLE shift
    immediately before its official forward is called.
    """

    def __init__(self, config: LlamaConfig, lec_config: dict):
        super().__init__(config)
        self.lec_config = lec_config
        self.num_experts = lec_config["num_experts"]
        self.num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        self.routers = nn.ModuleList([
            LECRouter(
                hidden_size=hidden_size,
                num_experts=self.num_experts,
                lb_ema_alpha=lec_config["lb_ema_alpha"],
                lb_bias_lr=lec_config["lb_bias_lr"],
            )
            for _ in range(self.num_layers)
        ])
        self.ple = PLETable(self.num_experts, self.num_layers, lec_config["ple_dim"], hidden_size)

        self.log_lec_metrics = False
        self._metric_layer_records = []
        self._hook_handles = []
        self._install_lec_hooks()

        base = sum(p.numel() for n, p in self.named_parameters() if "routers" not in n and "ple" not in n)
        lec = sum(p.numel() for p in self.routers.parameters()) + sum(p.numel() for p in self.ple.parameters())
        print(f"[LEC] Base: {base:,} | LEC overhead: {lec:,} ({100 * lec / (base + lec):.3f}%)")
        print(f"[LEC] Using {self.num_layers} per-layer routers via decoder-layer pre-hooks.")

    def _install_lec_hooks(self):
        for layer_idx, layer in enumerate(self.model.layers):
            handle = layer.register_forward_pre_hook(self._make_layer_pre_hook(layer_idx), with_kwargs=True)
            self._hook_handles.append(handle)

    def _make_layer_pre_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if len(args) > 0:
                hidden_states = args[0]
                rest_args = args[1:]
            else:
                hidden_states = kwargs["hidden_states"]
                rest_args = ()

            router = self.routers[layer_idx]
            expert_ids, gate, _logits, biased_logits, _probs = router(hidden_states)

            ple_all = self.ple.get_all_projected(layer_idx).to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )  # [K, H]
            ple_shift = torch.matmul(gate, ple_all)  # [B, T, H]
            hidden_states = hidden_states + ple_shift

            if self.training:
                router.accumulate_load(expert_ids)

            if self.log_lec_metrics:
                with torch.no_grad():
                    ei = expert_ids.detach()
                    bl = biased_logits.detach()
                    load = LECRouter.batch_load(ei, self.num_experts)
                    self._metric_layer_records.append({
                        "soft_entropy": router.soft_entropy_norm(bl),
                        "hard_entropy": LECRouter.hard_load_entropy_norm(ei, self.num_experts),
                        "load": load.detach(),
                    })

            if len(args) > 0:
                return (hidden_states, *rest_args), kwargs
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden_states
            return args, new_kwargs

        return hook

    def _reset_lec_metric_records(self):
        self._metric_layer_records = []

    @torch.no_grad()
    def _finalize_lec_metrics(self) -> dict:
        if not self._metric_layer_records:
            return {}

        soft = [r["soft_entropy"] for r in self._metric_layer_records]
        hard = [r["hard_entropy"] for r in self._metric_layer_records]
        loads = torch.stack([r["load"] for r in self._metric_layer_records], dim=0)  # [L, K]
        avg_load = loads.mean(dim=0)

        router_soft_entropy = float(sum(soft) / len(soft))
        router_hard_entropy = float(sum(hard) / len(hard))

        metrics = {
            "router_soft_entropy": router_soft_entropy,
            "router_hard_entropy": router_hard_entropy,
            "router_entropy": router_hard_entropy,
            "router_hard_entropy_min": float(min(hard)),
            "ple_pairwise_cosine": self.ple.pairwise_cosine(),
            "expert_load_balance_std": avg_load.std(unbiased=False).item(),
            "expert_load_balance_min": avg_load.min().item(),
            "expert_load_balance_max": avg_load.max().item(),
            "expert_load_max_any_layer": loads.max().item(),
            "expert_utilization": (avg_load > 0).float().mean().item(),
        }

        avg_ema_load = torch.stack([r.expert_ema_load.detach() for r in self.routers], dim=0).mean(dim=0)
        for idx, load in enumerate(avg_ema_load.tolist()):
            metrics[f"expert_ema_load_{idx}"] = float(load)
            metrics[f"expert_load_{idx}"] = float(load)

        collapse_threshold = self.lec_config.get("entropy_collapse_threshold", 0.5)
        metrics["router_collapsed_layers"] = float(sum(1 for x in hard if x < collapse_threshold))
        return metrics

    def forward(self, *args, **kwargs):
        # Clear previous hook-collected metrics, call the official HF forward,
        # then attach no-grad LEC metrics if Trainer requested them this step.
        self._reset_lec_metric_records()
        outputs = super().forward(*args, **kwargs)

        lec_metrics = self._finalize_lec_metrics() if self.log_lec_metrics else {}

        # During Trainer loss computation we only need loss. Returning full logits
        # makes Accelerate convert [B,T,V] to fp32, which can add ~12GB for this run.
        labels_present = kwargs.get("labels", None) is not None
        if labels_present:
            out = CausalLMOutputWithPast(loss=outputs.loss, logits=None)
        else:
            out = outputs
        out.lec_metrics = lec_metrics
        return out


# =============================================================================
# BUILD HELPERS
# =============================================================================

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
    lec_config = {k: config[k] for k in [
        "num_experts", "ple_dim", "lb_ema_alpha", "lb_bias_lr", "entropy_collapse_threshold"
    ]}

    model = LlamaLEC._from_config(
        llama_cfg,
        lec_config=lec_config,
        attn_implementation=config["attn_implementation"],
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] LLaMA + LEC hooks: {n:,} params ({n / 1e9:.2f}B)")
    print(f"[Attention] impl={config['attn_implementation']} | class={model.model.layers[0].self_attn.__class__.__name__}")
    return model


def build_tokenizer(config: dict):
    tok = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tok.model_max_length = config["max_seq_len"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[Tokenizer] vocab={tok.vocab_size} | total={len(tok)} | eos={tok.eos_token_id}")
    return tok


# =============================================================================
# DATASET
# =============================================================================

class PackedFineWebDataset(IterableDataset):
    """Streaming FineWeb with document packing to exact max_seq_len chunks."""

    def __init__(self, config: dict, tokenizer, seed: int = 42, max_examples: Optional[int] = None):
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
        dataset = self.dataset.shuffle(seed=self.seed + self.epoch, buffer_size=self.buffer_size)
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
# TRAINER
# =============================================================================

class LECTrainer(Trainer):
    """
    Minimal Trainer extension:
      - accumulates LEC metrics when enabled every N optimizer steps
      - flushes router load-balancing bias once per optimizer step
      - logs perplexity

    Optimizer and LR scheduler are standard HF Trainer/TrainingArguments.
    """

    def __init__(self, *args, lec_metric_log_every_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.lec_metric_log_every_steps = lec_metric_log_every_steps
        self._metric_sums = {}
        self._metric_count = 0

    def _accumulate_metrics(self, metrics: Optional[dict]):
        if not metrics:
            return
        for k, v in metrics.items():
            self._metric_sums[k] = self._metric_sums.get(k, 0.0) + float(v)
        self._metric_count += 1

    def _flush_metrics(self) -> dict:
        if self._metric_count == 0:
            return {}
        out = {k: v / self._metric_count for k, v in self._metric_sums.items()}
        self._metric_sums = {}
        self._metric_count = 0
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        unwrapped = self.accelerator.unwrap_model(model)
        should_log_lec = (
            self.lec_metric_log_every_steps > 0
            and self.state.global_step > 0
            and self.state.global_step % self.lec_metric_log_every_steps == 0
        )
        if hasattr(unwrapped, "log_lec_metrics"):
            unwrapped.log_lec_metrics = should_log_lec

        outputs = model(**inputs)
        loss = outputs.loss

        if should_log_lec:
            self._accumulate_metrics(getattr(outputs, "lec_metrics", None))

        if hasattr(unwrapped, "log_lec_metrics"):
            unwrapped.log_lec_metrics = False

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.accelerator.sync_gradients:
            unwrapped = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped, "routers"):
                for router in unwrapped.routers:
                    router.flush_load_balance()
        return loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            try:
                logs["ppl"] = round(math.exp(min(float(logs["loss"]), 20)), 2)
            except Exception:
                pass
            logs.update(self._flush_metrics())

        if "eval_loss" in logs:
            try:
                logs["eval_ppl"] = round(math.exp(min(float(logs["eval_loss"]), 20)), 2)
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

    if len(tokenizer) != model.config.vocab_size:
        print(f"[Warning] Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    train_ds = PackedFineWebDataset(config, tokenizer, seed=config["seed"])
    eval_ds = PackedFineWebDataset(config, tokenizer, seed=config["seed"] + 9999, max_examples=config["eval_max_examples"])

    tokens_per_step = config["per_device_batch_size"] * config["grad_accum_steps"] * config["max_seq_len"]
    print(f"\n[Train] tokens/step : {tokens_per_step:,}")
    print(f"[Train] total tokens : {tokens_per_step * config['max_steps'] / 1e9:.1f}B\n")

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
        save_only_model=True,
        save_total_limit=10,
        dataloader_num_workers=config["dataloader_workers"],
        dataloader_prefetch_factor=config["dataloader_prefetch_factor"],
        seed=config["seed"],
        report_to="wandb" if config["use_wandb"] else "none",
        run_name="llama_lec_hooks",
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
        lec_metric_log_every_steps=config["lec_metric_log_every_steps"],
    )

    trainer.train()
    print("[Train] LLaMA LEC hook-based pretraining complete.")


if __name__ == "__main__":
    train(CONFIG)
