"""
LEC Pretraining - LLaMA + PLE + Router via HuggingFace Trainer
==============================================================
Trains a LLaMA-style model with Lightweight Expert Conditioning from scratch.

Architecture (Phase 1 — PLE + Router only):
  - Per-layer routing: at each transformer layer, that layer's router runs on
    the current hidden state and selects top-1 expert for that layer independently.
  - PLE table: [K, L, d_ple] — per-expert, per-layer embedding vectors.
  - Shared projection W_ple: d_ple → hidden_size.
  - All weights (base model + router + PLE) train jointly from scratch.
  - Auxiliary-loss-free load balancing via per-layer EMA bias (DeepSeek-V3 style).
  - Orthogonal full-dimensional PLE vector init for expert symmetry breaking.

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
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    # Runtime note: use "sdpa" for this custom layer-by-layer forward.
    # Some recent transformers+flash-attn combinations expose query tensors in
    # a layout that does not match the default RoPE broadcast path when decoder
    # layers are called manually. Switch back to "flash_attention_2" only after
    # a smoke test passes in your exact environment.
    "attn_implementation": "eager",
    "tokenizer_name": "meta-llama/Llama-3.2-1B",
    # LEC (Phase 1: PLE + Router only)
    "num_experts": 8,
    "ple_dim": 512,
    "lb_ema_alpha": 0.1,
    "lb_bias_lr": 1e-3,
    "lec_lr_multiplier": 3.0,
    "lec_warmdown_steps": 10000,
    "entropy_collapse_threshold": 0.5,
    # Training
    "learning_rate": 3e-4,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 250,
    "max_steps": 2500,
    "per_device_batch_size": 4,
    "grad_accum_steps": 4,
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
    "lec_metric_log_every_steps": 10,
}
import math
import os
from typing import Optional

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    import torch._dynamo as _torch_dynamo

    _torch_dynamo.config.suppress_errors = True
    _dynamo_disable = _torch_dynamo.disable
except Exception:
    _dynamo_disable = lambda fn: fn


# =============================================================================
# PLE TABLE
# =============================================================================


class PLETable(nn.Module):
    """
    Per-Layer Expert Embedding table: [K, L, d_ple].

    Each expert k has one d_ple-dimensional vector per layer l.
    A single shared projection maps d_ple → hidden_size.
    Vectors are initialized as full-dimensional, mutually orthogonal
    expert directions for symmetry breaking — this preserves full-space
    headroom while making expert choices distinct from step 1.
    """

    def __init__(
        self, num_experts: int, num_layers: int, ple_dim: int, hidden_size: int
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        # Trainer toggles this true only on LEC-metric logging steps.
        # Loss is still computed every step; expensive/no-grad diagnostics are not.
        self.log_lec_metrics = False
        self.ple_dim = ple_dim
        self.hidden_size = hidden_size

        self.embeddings = nn.Parameter(torch.zeros(num_experts, num_layers, ple_dim))
        self.proj = nn.Linear(ple_dim, hidden_size, bias=False)
        self._init_orthogonal()

    def _init_orthogonal(self):
        """
        Initialize each layer's expert PLE vectors as full-dimensional,
        mutually orthogonal directions.

        This is symmetry breaking, not a subspace constraint: every expert is
        free to learn anywhere in the full d_ple space after initialization.
        """
        with torch.no_grad():
            for l in range(self.num_layers):
                if self.num_experts <= self.ple_dim:
                    q, _ = torch.linalg.qr(
                        torch.randn(
                            self.ple_dim,
                            self.ple_dim,
                            device=self.embeddings.device,
                            dtype=self.embeddings.dtype,
                        )
                    )
                    # q has orthonormal columns; transpose the first K columns
                    # to obtain K full-dimensional expert vectors.
                    self.embeddings[:, l, :] = q[:, : self.num_experts].T * 0.02
                else:
                    nn.init.normal_(self.embeddings[:, l, :], std=0.02)

    def get_all_projected(self, layer_idx: int) -> torch.Tensor:
        """
        Project every expert's PLE vector for one layer.

        Returns:
            Tensor [num_experts, hidden_size]. Used by straight-through top-1
            routing so the forward pass is hard top-1 but gradients flow to
            router probabilities.
        """
        return self.proj(self.embeddings[:, layer_idx, :])

    def get(self, layer_idx: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve and project PLE vectors for a batch of expert assignments.

        Args:
            layer_idx:  which transformer layer we're at
            expert_ids: LongTensor [batch, seq_len] of expert indices (0..K-1)

        Returns:
            Tensor [batch, seq_len, hidden_size] — the PLE shift to add to h
        """
        vecs = self.embeddings[:, layer_idx, :]  # [K, d_ple]
        flat = vecs[expert_ids.reshape(-1)]  # [B*T, d_ple]
        return self.proj(flat).reshape(*expert_ids.shape, self.hidden_size)

    def pairwise_cosine(self) -> float:
        """
        Mean pairwise cosine similarity across all experts and all layers.
        Decreasing = experts are diverging. Checked across all layers, not just 0.
        """
        with torch.no_grad():
            # average over layers for a more representative signal
            total = 0.0
            for l in range(self.num_layers):
                vecs = self.proj(self.embeddings[:, l, :]).float()
                norms = F.normalize(vecs, dim=-1)
                sim = norms @ norms.T
                mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=sim.device)
                total += sim[mask].mean().item()
            return total / self.num_layers


# =============================================================================
# ROUTER
# =============================================================================


class LECRouter(nn.Module):
    """
    One transformer layer's linear router with gradient-free EMA load-balancing bias.

    Input:  hidden state h at the current layer [batch, seq_len, hidden_size]
    Output: top-1 expert ids [batch, seq_len], router logits for entropy logging

    LlamaLEC owns one LECRouter per transformer layer. Load balancing follows
    DeepSeek-V3: a per-expert bias is adjusted
    gradient-free based on EMA of expert utilization, avoiding auxiliary losses
    that can dominate the weak specialization gradient signal.

    IMPORTANT: update_load_balance() must be called once per optimizer step,
    not per micro-step, to avoid effective lb_bias_lr being multiplied by
    grad_accum_steps.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        lb_ema_alpha: float = 0.1,
        lb_bias_lr: float = 1e-3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.lb_ema_alpha = lb_ema_alpha
        self.lb_bias_lr = lb_bias_lr

        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)

        self.register_buffer("lb_bias", torch.zeros(num_experts))
        self.register_buffer("expert_ema_load", torch.ones(num_experts) / num_experts)

        # accumulator for load-balance updates across micro-steps
        # flushed once per optimizer step in LECTrainer
        self._load_accum: Optional[torch.Tensor] = None
        self._accum_count: int = 0

    def forward(self, h: torch.Tensor):
        """
        Straight-through top-1 routing.

        Forward semantics are hard top-1: each token selects exactly one expert.
        Backward semantics use softmax probabilities, so the router receives
        task-loss gradient even though dispatch is discrete in the forward pass.

        Args:
            h: [batch, seq_len, hidden_size]

        Returns:
            expert_ids:    LongTensor [batch, seq_len]
            gate:          Tensor [batch, seq_len, K], hard in forward / soft in backward
            logits:        Tensor [batch, seq_len, K], pre-bias router logits
            biased_logits: Tensor [batch, seq_len, K], logits used for routing
            probs:         Tensor [batch, seq_len, K], softmax over biased_logits
        """
        logits = self.linear(h)  # [B, T, K]
        biased_logits = logits + self.lb_bias.to(logits.dtype)  # [B, T, K]
        probs = F.softmax(biased_logits.float(), dim=-1).to(h.dtype)  # [B, T, K]

        expert_ids = probs.argmax(dim=-1)  # [B, T]
        hard = F.one_hot(expert_ids, num_classes=self.num_experts).to(h.dtype)

        # Straight-through estimator: numerically hard in forward, soft in backward.
        gate = hard + probs - probs.detach()
        return expert_ids, gate, logits, biased_logits, probs

    def accumulate_load(self, expert_ids: torch.Tensor):
        """
        Accumulate per-expert token counts across micro-steps.
        Call this on every forward pass during training; flush once per
        optimizer step in LECTrainer.
        """
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
        """
        Update EMA and lb_bias using the accumulated load across micro-steps.
        Call exactly ONCE per optimizer step (not per micro-step).
        """
        if self._load_accum is None or self._accum_count == 0:
            return
        avg_load = (self._load_accum / self._accum_count).to(self.lb_bias.device)
        self.expert_ema_load = (
            1 - self.lb_ema_alpha
        ) * self.expert_ema_load + self.lb_ema_alpha * avg_load
        self.lb_bias -= self.lb_bias_lr * (
            self.expert_ema_load - 1.0 / self.num_experts
        )
        self._load_accum = None
        self._accum_count = 0

    def soft_entropy_norm(self, biased_logits: torch.Tensor) -> float:
        """
        Normalized entropy of the actual biased router distribution.
        1.0 = uniform soft distribution; 0.0 = fully confident.
        """
        with torch.no_grad():
            p = F.softmax(biased_logits.float(), dim=-1)
            entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean().item()
            return entropy / math.log(self.num_experts)

    @staticmethod
    def hard_load_entropy_norm(expert_ids: torch.Tensor, num_experts: int) -> float:
        """
        Normalized entropy of the realized hard expert assignment histogram.
        1.0 = evenly used experts; 0.0 = all tokens routed to one expert.
        """
        with torch.no_grad():
            flat = expert_ids.reshape(-1)
            counts = torch.bincount(flat, minlength=num_experts).float()
            load = counts / counts.sum().clamp_min(1.0)
            entropy = -(load * torch.log(load + 1e-10)).sum().item()
            return entropy / math.log(num_experts)

    @staticmethod
    def batch_load(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        with torch.no_grad():
            flat = expert_ids.reshape(-1)
            counts = torch.bincount(flat, minlength=num_experts).float()
            return counts / counts.sum().clamp_min(1.0)


# =============================================================================
# LLAMA + LEC  (per-layer routing via custom forward loop)
# =============================================================================


class LlamaLEC(LlamaForCausalLM):
    """
    LLaMA with per-layer PLE conditioning.

    At each transformer layer l:
        expert_ids_l = router_l(h_l)        # top-1, from current hidden state
        ple_shift    = PLE.get(l, expert_ids_l)
        h_l          = h_l + ple_shift
        h_l+1        = transformer_layer_l(h_l)

    Routing is independent at every layer — the expert chosen at layer 3
    need not match the expert chosen at layer 7. Each layer has its own router
    parameters and its own load-balancing EMA/bias state, matching the usual
    Transformer-MoE pattern more closely than a single shared router.

    All weights (base + per-layer routers + PLE) train jointly from scratch.
    """

    def __init__(self, config: LlamaConfig, lec_config: dict):
        super().__init__(config)
        self.lec_config = lec_config
        num_experts = lec_config["num_experts"]
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        self.num_experts = num_experts
        self.num_layers = num_layers
        # Trainer toggles this true only on LEC-metric logging steps.
        # Loss is still computed every step; expensive/no-grad diagnostics are not.
        self.log_lec_metrics = False

        self.routers = nn.ModuleList(
            [
                LECRouter(
                    hidden_size,
                    num_experts,
                    lec_config["lb_ema_alpha"],
                    lec_config["lb_bias_lr"],
                )
                for _ in range(num_layers)
            ]
        )
        self.ple = PLETable(num_experts, num_layers, lec_config["ple_dim"], hidden_size)

        base = sum(
            p.numel()
            for n, p in self.named_parameters()
            if "router" not in n and "ple" not in n
        )
        lec = sum(p.numel() for p in self.routers.parameters()) + sum(
            p.numel() for p in self.ple.parameters()
        )
        print(
            f"[LEC] Base: {base:,} | LEC overhead: {lec:,} ({100 * lec / (base + lec):.3f}%)"
        )
        print(f"[LEC] Using {num_layers} per-layer routers, one per transformer layer.")

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """RoPE helper: rotate last-dimension halves."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped-query attention.

        Input:  [B, H_kv, T, D]
        Output: [B, H_kv * n_rep, T, D]
        """
        if n_rep == 1:
            return x
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)

    def _manual_decoder_layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Manual LLaMA decoder layer forward.

        This avoids calling HF LlamaAttention.forward directly. In the user's
        installed transformers build, manual decoder-layer calls expose a RoPE
        broadcast mismatch inside apply_rotary_pos_emb. Here we explicitly keep
        Q/K/V in [B, heads, T, head_dim] layout and apply RoPE ourselves.
        """
        bsz, seq_len, _ = hidden_states.shape
        attn = layer.self_attn

        # Attention block
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)

        num_heads = getattr(attn, "num_heads", self.config.num_attention_heads)
        num_kv_heads = getattr(
            attn, "num_key_value_heads", self.config.num_key_value_heads
        )
        head_dim = getattr(attn, "head_dim", self.config.hidden_size // num_heads)
        n_rep = num_heads // num_kv_heads

        q = attn.q_proj(x).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings  # expected [B, T, D]
        cos = cos[:, :seq_len, :].unsqueeze(1).to(dtype=q.dtype)  # [B,1,T,D]
        sin = sin[:, :seq_len, :].unsqueeze(1).to(dtype=q.dtype)  # [B,1,T,D]
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        k = self._repeat_kv(k, n_rep)
        v = self._repeat_kv(v, n_rep)

        # Packed dataset has no padding. If a mask is ever supplied, combine it
        # with an explicit causal mask; otherwise use SDPA's efficient causal path.
        sdpa_mask = None
        is_causal = True
        if attention_mask is not None:
            if attention_mask.dim() == 2 and not bool(attention_mask.all()):
                pad = attention_mask[:, None, None, :].to(torch.bool)
                causal = torch.ones(
                    seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
                ).tril()
                sdpa_mask = pad & causal[None, None, :, :]
                is_causal = False
            elif attention_mask.dim() == 4:
                sdpa_mask = attention_mask
                is_causal = False

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.config.hidden_size)
        )
        attn_out = attn.o_proj(attn_out)
        hidden_states = residual + attn_out

        # MLP block
        residual = hidden_states
        x = layer.post_attention_layernorm(hidden_states)
        x = layer.mlp(x)
        hidden_states = residual + x
        return hidden_states

    @_dynamo_disable
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Custom forward loop that applies per-layer PLE conditioning.
        Replaces the hook-based approach with an explicit layer-by-layer pass.
        """
        # --- embedding ---
        h = self.model.embed_tokens(input_ids)

        # Build cache_position / position_ids the same way modern HF LLaMA expects.
        # Newer transformers LLaMA decoder layers require precomputed RoPE
        # position_embeddings=(cos, sin), passed into every layer.
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(0, seq_len, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)

        # Shared RoPE embeddings for all decoder layers. Without this, recent
        # transformers versions fail inside LlamaAttention with:
        #   TypeError: cannot unpack non-iterable NoneType object
        position_embeddings = self.model.rotary_emb(h, position_ids)

        # --- per-layer forward with PLE injection ---
        collect_lec_metrics = bool(getattr(self, "log_lec_metrics", False))
        all_biased_logits = [] if collect_lec_metrics else None
        all_expert_ids = [] if collect_lec_metrics else None

        causal_mask = (
            self.model._update_causal_mask(
                attention_mask, h, cache_position, None, False
            )
            if hasattr(self.model, "_update_causal_mask")
            else attention_mask
        )

        for layer_idx, layer in enumerate(self.model.layers):
            router = self.routers[layer_idx]
            expert_ids, gate, router_logits, biased_logits, router_probs = router(h)

            ple_all = self.ple.get_all_projected(layer_idx).to(dtype=h.dtype)  # [K, H]
            ple_shift = torch.matmul(gate, ple_all)  # [B, T, H]
            h = h + ple_shift

            h = self._manual_decoder_layer_forward(
                layer,
                h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

            if collect_lec_metrics:
                # Detach so metric collection never keeps the training graph alive.
                all_expert_ids.append(expert_ids.detach())
                all_biased_logits.append(biased_logits.detach())

            # Accumulate hard realized load for gradient-free lb_bias update.
            if self.training:
                router.accumulate_load(expert_ids)

        # --- final norm + lm_head ---
        h = self.model.norm(h)
        logits = self.lm_head(h)

        # --- loss ---
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # --- optional no-grad diagnostics ---
        # Toggled by LECTrainer every N optimizer steps. They are never part of
        # the training graph and are absent on normal steps.
        lec_metrics = None
        if collect_lec_metrics:
            with torch.no_grad():
                router_soft_entropy = sum(
                    router.soft_entropy_norm(bl.detach())
                    for router, bl in zip(self.routers, all_biased_logits)
                ) / len(all_biased_logits)

                router_hard_entropy = sum(
                    LECRouter.hard_load_entropy_norm(ei.detach(), self.num_experts)
                    for ei in all_expert_ids
                ) / len(all_expert_ids)

                collapse_threshold = self.lec_config.get(
                    "entropy_collapse_threshold", 0.5
                )
                if router_hard_entropy < collapse_threshold:
                    print(
                        f"[LEC][WARN] hard routing collapsed: "
                        f"{router_hard_entropy:.4f} < {collapse_threshold}"
                    )

                loads = [
                    LECRouter.batch_load(ei.detach(), self.num_experts)
                    for ei in all_expert_ids
                ]
                load_stack = torch.stack(loads, dim=0)  # [L, K]
                avg_load = load_stack.mean(dim=0)  # [K]

                lec_metrics = {
                    "router_soft_entropy": float(router_soft_entropy),
                    "router_hard_entropy": float(router_hard_entropy),
                    "router_entropy": float(router_hard_entropy),
                    "ple_pairwise_cosine": float(self.ple.pairwise_cosine()),
                    "expert_load_balance_std": float(
                        avg_load.std(unbiased=False).item()
                    ),
                    "expert_load_balance_min": float(avg_load.min().item()),
                    "expert_load_balance_max": float(avg_load.max().item()),
                    "expert_load_max_any_layer": float(load_stack.max().item()),
                    "expert_utilization": float((avg_load > 0).float().mean().item()),
                }

                avg_ema_load = torch.stack(
                    [r.expert_ema_load.detach() for r in self.routers], dim=0
                ).mean(dim=0)
                for idx, load in enumerate(avg_ema_load.tolist()):
                    lec_metrics[f"expert_ema_load_{idx}"] = float(load)
                    lec_metrics[f"expert_load_{idx}"] = float(load)

        # Do not return full logits. Accelerate may fp32-convert returned bf16
        # tensors; [B, T, vocab] logits are ~12GB at this config.
        out = CausalLMOutputWithPast(loss=loss, logits=None)
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
    # Be explicit: custom manual decoder-layer loop is safest with eager attention.
    # Passing through _from_config alone is not always enough across transformers versions.
    llama_cfg._attn_implementation = config["attn_implementation"]
    lec_config = {
        k: config[k]
        for k in [
            "num_experts",
            "ple_dim",
            "lb_ema_alpha",
            "lb_bias_lr",
            "entropy_collapse_threshold",
        ]
    }
    model = LlamaLEC._from_config(
        llama_cfg,
        lec_config=lec_config,
        attn_implementation=config["attn_implementation"],
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] LLaMA + LEC: {n:,} params ({n / 1e9:.2f}B)")
    print(
        f"[Attention] impl={getattr(model.config, '_attn_implementation', None)} | class={model.model.layers[0].self_attn.__class__.__name__}"
    )
    return model


def build_tokenizer(config: dict):
    tok = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tok.model_max_length = int(
        1e9
    )  # packing enforces max_seq_len; avoid long-document tokenizer warnings
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(
        f"[Tokenizer] vocab={tok.vocab_size} | total={len(tok)} | eos={tok.eos_token_id}"
    )
    return tok


# =============================================================================
# DATASET
# =============================================================================


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
        print(f"[Data] {config['dataset_name']} / {config['dataset_config']} streaming")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Shard streaming data across DataLoader workers and distributed ranks.
        # Without this, IterableDataset workers can duplicate the same stream.
        ds = self.dataset
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        num_shards = max(1, world_size * num_workers)
        shard_index = rank * num_workers + worker_id
        if num_shards > 1:
            ds = ds.shard(num_shards=num_shards, index=shard_index)

        ds = ds.shuffle(
            seed=self.seed + self.epoch + shard_index, buffer_size=self.buffer_size
        )
        it = iter(ds)
        buf = []
        n = 0
        while True:
            if self.max_examples is not None and n >= self.max_examples:
                return
            while len(buf) < self.max_seq_len:
                try:
                    doc = next(it)
                except StopIteration:
                    return
                text = doc.get("text", "")
                if not text or not text.strip():
                    continue
                tokens = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    verbose=False,
                )["input_ids"]
                if tokens:
                    buf.extend(tokens)
                    buf.append(self.eos_id)
            chunk = buf[: self.max_seq_len]
            buf = buf[self.max_seq_len :]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "labels": torch.tensor(chunk, dtype=torch.long),
            }
            n += 1


# =============================================================================
# TRAINER
# =============================================================================


class LECTrainer(Trainer):
    """
    Minimal HF Trainer extension:
      - Uses the standard HF optimizer and cosine scheduler from TrainingArguments.
      - Accumulates LEC metrics only every `lec_metric_log_every_steps` optimizer steps.
      - Flushes per-layer router load-balancing state once per optimizer step.
      - Adds perplexity logging.
    """

    def __init__(self, *args, lec_metric_log_every_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.lec_metric_log_every_steps = max(1, int(lec_metric_log_every_steps))
        self._metric_sums = {"train": {}, "eval": {}}
        self._metric_counts = {"train": 0, "eval": 0}

    def _accumulate(self, split: str, metrics: dict):
        if not metrics:
            return
        bucket = self._metric_sums[split]
        for k, v in metrics.items():
            bucket[k] = bucket.get(k, 0.0) + float(v)
        self._metric_counts[split] += 1

    def _flush(self, split: str, prefix: str = "") -> dict:
        count = self._metric_counts[split]
        if count == 0:
            return {}
        out = {f"{prefix}{k}": v / count for k, v in self._metric_sums[split].items()}
        self._metric_sums[split] = {}
        self._metric_counts[split] = 0
        return out

    def _should_collect_lec_metrics(self, model) -> bool:
        # While training, global_step is completed optimizer steps. For the
        # upcoming optimizer step, use global_step + 1 so metrics at N=10 land
        # in the step-10 log and cover all micro-steps for that step.
        target_step = (
            self.state.global_step + 1 if model.training else self.state.global_step
        )
        return target_step > 0 and (target_step % self.lec_metric_log_every_steps == 0)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        unwrapped = self.accelerator.unwrap_model(model)
        should_collect = self._should_collect_lec_metrics(model)
        if hasattr(unwrapped, "log_lec_metrics"):
            unwrapped.log_lec_metrics = should_collect
        try:
            outputs = model(**inputs)
        finally:
            if hasattr(unwrapped, "log_lec_metrics"):
                unwrapped.log_lec_metrics = False

        loss = outputs.loss
        split = "train" if model.training else "eval"
        self._accumulate(split, getattr(outputs, "lec_metrics", None))
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            unwrapped = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped, "routers"):
                for router in unwrapped.routers:
                    router.flush_load_balance()
            elif hasattr(unwrapped, "router"):
                unwrapped.router.flush_load_balance()
        return loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            try:
                logs["ppl"] = round(math.exp(min(logs["loss"], 20)), 2)
            except Exception:
                pass
            logs.update(self._flush("train"))
        if "eval_loss" in logs:
            try:
                logs["eval_ppl"] = round(math.exp(min(logs["eval_loss"], 20)), 2)
            except Exception:
                pass
            logs.update(self._flush("eval", prefix="eval_"))
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
        model.resize_token_embeddings(len(tokenizer))

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
    print(f"\n[Train] tokens/step : {tokens_per_step:,}")
    print(
        f"[Train] total tokens : {tokens_per_step * config['max_steps'] / 1e9:.1f}B\n"
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
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=config["log_every_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_every_steps"],
        save_strategy="steps",
        save_steps=config["save_every_steps"],
        save_only_model=True,
        save_total_limit=3,
        dataloader_num_workers=config["dataloader_workers"],
        dataloader_prefetch_factor=config["dataloader_prefetch_factor"],
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
        lec_metric_log_every_steps=config.get("lec_metric_log_every_steps", 10),
    )

    trainer.train()
    print("[Train] LLaMA LEC pretraining complete.")


if __name__ == "__main__":
    train(CONFIG)
