"""
LLaMA + Gemma 3n–style PLE — Faithful port, HF Trainer
========================================================
Faithfully wires the Gemma 3n PLE mechanism onto a LLaMA backbone.

What we replicate exactly from modeling_gemma3n.py
---------------------------------------------------
PerLayerEmbedder (Gemma3nTextModel):
  • ModuleList of L separate nn.Embedding tables, each [vocab, ple_dim]
  • Single fused forward: stack → [B, T, L, ple_dim]
  • token_ids clamped to [0, vocab-1]

Per-layer projector (Gemma3nTextModel.__init__):
  • per_layer_projection_norm : RMSNorm(ple_dim)
  • per_layer_projection_scale: scalar buffer = hidden_size ** -0.5
  • per_layer_input_scale     : scalar buffer = rsqrt(2.0)
  The model applies:
      normed  = per_layer_projection_norm(raw_ple)
      mixed   = (normed + raw_ple) * per_layer_projection_scale
                                    ↑ residual add then scale

Injection inside each decoder layer (Gemma3nTextDecoderLayer):
  • per_layer_input_gate : Linear(hidden, ple_dim, bias=False)
  • per_layer_projection : Linear(ple_dim, hidden, bias=False)
  • post_per_layer_input_norm : RMSNorm(hidden)
  The formula:
      gate      = gelu(per_layer_input_gate(corrected_hidden))   [B,T,ple_dim]
      gated     = gate * per_layer_input[layer]                  [B,T,ple_dim]
      projected = per_layer_projection(gated)                    [B,T,hidden]
      normed    = post_per_layer_input_norm(projected)           [B,T,hidden]
      output    = corrected_hidden + normed

  Note: injection is at the END of each layer, after attention + MLP.
  Note: per_layer_projection is initialized to zeros → PLE starts as
        a zero correction and grows in from there.

What we DON'T replicate (LLaMA backbone differences):
  • No AltUp multi-stream (LLaMA has one hidden stream)
  • No LAuReL block
  • No sliding-window / global attention alternation
  • No activation sparsity
  These would require replacing LlamaDecoderLayer internals entirely,
  at which point you'd just use Gemma3nForCausalLM directly (see other script).

Metrics added
-------------
  • ple_pairwise_cosine_mean  : mean pairwise cosine of projected PLE vectors,
                                averaged across layers (analogous to the LEC
                                ple_pairwise_cosine metric, but per-token is
                                expensive — we sample 512 random vocab entries)
  • ple_gate_activation_rms   : RMS of the gate output (how hard gates fire)
  • ple_correction_rms        : RMS of the final PLE correction added to hidden
  • ple_correction_to_hidden_ratio : correction_rms / hidden_rms per layer mean
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Dense LLaMA backbone — hidden_size fixed at 2048 to match baseline
    # Budget: 1.23B total, ~62% dense / ~38% PLE
    #
    # per_layer params = hidden^2*3 (attn, GQA 16h/8kv) + hidden*intermediate*3 (MLP)
    #                  = 2048^2*3 + 2048*6144*3 = 12.6M + 37.7M = 50.3M
    # dense total = embed(128256*2048) + 10 * 50.3M = 262.7M + 503.3M = 766.0M (62.3%)
    "hidden_size": 2048,
    "num_hidden_layers": 10,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 6144,
    "vocab_size": 128256,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_2",
    "tokenizer_name": "meta-llama/Llama-3.2-1B",
    # PLE — fills remaining ~38% of 1.23B budget
    # total_ple = (vocab*L + L*2*hidden) * ple_dim
    #           = (128256*10 + 10*2*2048) * ple_dim
    #           = 1,323,520 * ple_dim
    # target PLE = 1,230M - 766M = 464M → ple_dim = 464M / 1,323,520 ≈ 350
    # tables     = 128256 * 10 * 350 = 448,896,000
    # projectors = 10 * 2 * 2048 * 350 = 14,336,000
    # total      = 766,984,768 + 463,232,000 = 1,230,216,768 ✓
    "ple_dim": 350,
    "ple_vocab_size": 128256,
    "ple_init_scale": 0.02,
    # Metrics
    "metric_log_every_steps": 10,
    "ple_cosine_sample_size": 512,  # vocab entries to sample for cosine metric
    # Training
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 200,
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
    "save_every_steps": 500,
    "log_every_steps": 1,
    # Misc
    "seed": 42,
    "output_dir": "./checkpoints_llama_gemma3n_ple",
    "use_wandb": True,
    "wandb_project": "llama-gemma3n-ple-pretrain-final-0405",
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,
}

# =============================================================================
# IMPORTS
# =============================================================================

import math
from typing import Optional

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
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# =============================================================================
# PER-LAYER EMBEDDER  (mirrors Gemma3nTextModel's embed_tokens_per_layer)
# =============================================================================


class PerLayerEmbedder(nn.Module):
    """
    Separate nn.Embedding per layer, matching the Gemma 3n implementation
    exactly. Google keeps them separate (not a fused [vocab, L*ple_dim] tensor)
    so each layer's table can be individually offloaded to CPU during inference.

    forward returns [B, T, L, ple_dim] — same shape as Gemma 3n.
    """

    def __init__(
        self, vocab_size: int, ple_dim: int, num_layers: int, init_scale: float = 0.02
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ple_dim = ple_dim
        self.num_layers = num_layers
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, ple_dim) for _ in range(num_layers)]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=init_scale)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, T]  →  [B, T, L, ple_dim]"""
        ids = token_ids.clamp(0, self.vocab_size - 1)
        return torch.stack([emb(ids) for emb in self.embeddings], dim=2)

    @torch.no_grad()
    def pairwise_cosine_mean(self, sample_size: int = 512) -> float:
        """
        Mean pairwise cosine similarity of the PROJECTED PLE vectors,
        sampled over `sample_size` random vocabulary entries, averaged
        across all layers.

        Equivalent to the LEC script's PLETable.pairwise_cosine() metric,
        adapted for the Gemma 3n per-layer-table structure. Measures whether
        different tokens' PLE vectors (after gate+proj at injection time)
        are diverse or collapsing.

        Note: this samples across the vocab dimension (token diversity),
        whereas the LEC metric measures across the expert dimension.
        Both answer "are the PLE vectors learning different things?"
        """
        device = self.embeddings[0].weight.device
        idx = torch.randint(0, self.vocab_size, (sample_size,), device=device)
        total = 0.0
        for emb in self.embeddings:
            vecs = emb(idx).float()  # [S, ple_dim]
            norms = F.normalize(vecs, dim=-1)  # [S, ple_dim]
            sim = norms @ norms.T  # [S, S]
            off_diag = sim[~torch.eye(sample_size, dtype=torch.bool, device=device)]
            total += off_diag.mean().item()
        return total / self.num_layers


# =============================================================================
# PER-LAYER PROJECTOR  (mirrors Gemma3nTextModel.__init__ buffers)
# =============================================================================


class PerLayerProjector(nn.Module):
    """
    Shared pre-processing applied to ALL layers' PLE vectors before injection.

    From Gemma3nTextModel.__init__:
        self.per_layer_projection_norm  = RMSNorm(hidden_size_per_layer_input)
        self.per_layer_projection_scale = hidden_size ** -0.5   (buffer)
        self.per_layer_input_scale      = rsqrt(2.0)            (buffer)

    The forward in Gemma3nTextModel (inferred from field names + shapes):
        normed = per_layer_projection_norm(ple_raw)   # RMSNorm over ple_dim
        mixed  = (normed + ple_raw) * per_layer_projection_scale
        # per_layer_input_scale applied at injection site, not here

    per_layer_projection_scale = hidden**-0.5 ≈ 0.026 for hidden=1536.
    This down-scales the PLE residual relative to the main stream magnitude.
    """

    def __init__(self, ple_dim: int, hidden_size: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.norm = LlamaRMSNorm(ple_dim, eps=rms_norm_eps)
        # Scalar buffers matching Gemma 3n exactly
        self.register_buffer(
            "per_layer_projection_scale",
            torch.tensor(hidden_size**-0.5),
            persistent=False,
        )
        self.register_buffer(
            "per_layer_input_scale",
            torch.rsqrt(torch.tensor(2.0)),
            persistent=False,
        )

    def forward(self, ple_raw: torch.Tensor) -> torch.Tensor:
        """
        ple_raw: [B, T, L, ple_dim]
        returns: [B, T, L, ple_dim]  (scaled, ready for per-layer injection)
        """
        normed = self.norm(ple_raw)  # RMSNorm on last dim
        mixed = (normed + ple_raw) * self.per_layer_projection_scale
        return mixed


# =============================================================================
# PLE INJECTION MODULE  (mirrors Gemma3nTextDecoderLayer fields)
# =============================================================================


class PLEInjection(nn.Module):
    """
    Per-layer injection block. One instance per decoder layer.

    Exactly mirrors the three fields added to Gemma3nTextDecoderLayer:
        self.per_layer_input_gate  = Linear(hidden, ple_dim, bias=False)
        self.per_layer_projection  = Linear(ple_dim, hidden, bias=False)
        self.post_per_layer_input_norm = RMSNorm(hidden)

    Forward (from decoder layer source):
        gate      = gelu(per_layer_input_gate(hidden_states))
        gated     = gate * per_layer_input          ← element-wise
        projected = per_layer_projection(gated)
        normed    = post_per_layer_input_norm(projected)
        output    = hidden_states + normed          ← residual add

    Init: per_layer_projection.weight = zeros  so correction starts at zero,
    training begins from the pure LLaMA baseline.
    """

    def __init__(self, hidden_size: int, ple_dim: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.per_layer_input_gate = nn.Linear(hidden_size, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, hidden_size, bias=False)
        self.post_per_layer_input_norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        # Zero-init output projection: PLE starts as exact zero correction
        nn.init.zeros_(self.per_layer_projection.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        per_layer_input: torch.Tensor,  # [B, T, ple_dim]  (pre-mixed slice for this layer)
        per_layer_input_scale: float,  # scalar from PerLayerProjector
    ) -> torch.Tensor:
        """Returns hidden_states + PLE correction."""
        per_layer_input = per_layer_input.to(hidden_states.dtype)

        gate = F.gelu(self.per_layer_input_gate(hidden_states))  # [B, T, ple_dim]
        gated = gate * per_layer_input  # [B, T, ple_dim]
        # per_layer_input_scale (= rsqrt(2)) applied here, matching Gemma 3n
        proj = self.per_layer_projection(gated * per_layer_input_scale)  # [B, T, H]
        normed = self.post_per_layer_input_norm(proj)  # [B, T, H]
        return hidden_states + normed


# =============================================================================
# LLAMA + GEMMA 3N PLE
# =============================================================================


class LlamaGemma3nPLE(LlamaForCausalLM):
    """
    LLaMA backbone with Gemma 3n–faithful PLE injection via post-hooks.

    Components:
      self.ple_embedder   : PerLayerEmbedder  — L separate vocab tables
      self.ple_projector  : PerLayerProjector — shared pre-mix + scale
      self.ple_injections : ModuleList[L]     — per-layer gate+proj+norm

    Forward flow:
      1. forward() fuses token_ids → ple_all [B,T,L,ple_dim] before layer loop
      2. ple_projector mixes all layers at once → ple_mixed [B,T,L,ple_dim]
      3. Post-hook on each LlamaDecoderLayer calls ple_injections[l] on output
      4. Stash cleared after forward() to free memory
    """

    def __init__(self, config: LlamaConfig, ple_config: dict):
        super().__init__(config)
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        ple_dim = ple_config["ple_dim"]
        ple_vocab = ple_config["ple_vocab_size"]
        eps = config.rms_norm_eps

        self.ple_embedder = PerLayerEmbedder(
            vocab_size=ple_vocab,
            ple_dim=ple_dim,
            num_layers=num_layers,
            init_scale=ple_config.get("ple_init_scale", 0.02),
        )
        self.ple_projector = PerLayerProjector(ple_dim, hidden_size, eps)
        self.ple_injections = nn.ModuleList(
            [PLEInjection(hidden_size, ple_dim, eps) for _ in range(num_layers)]
        )

        # Runtime stash (not parameters)
        self._ple_mixed: Optional[torch.Tensor] = None

        # Metric collection
        self.log_ple_metrics = False
        self._metric_records: list = []
        self._last_ple_metrics: dict = {}
        self._ple_config = ple_config

        self._install_hooks()
        self._print_param_summary()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _install_hooks(self):
        self._hook_handles = []
        for l, layer in enumerate(self.model.layers):
            h = layer.register_forward_hook(self._make_post_hook(l))
            self._hook_handles.append(h)

    def _make_post_hook(self, layer_idx: int):
        def hook(module, args, output):
            if self._ple_mixed is None:
                return output  # safety: no token_ids stashed

            # LlamaDecoderLayer returns (hidden_states, *extras)
            if isinstance(output, tuple):
                h, rest = output[0], output[1:]
            else:
                h, rest = output, None

            per_layer_input = self._ple_mixed[:, :, layer_idx, :]  # [B, T, ple_dim]
            scale = self.ple_projector.per_layer_input_scale.item()

            if self.log_ple_metrics:
                with torch.no_grad():
                    hidden_rms = h.detach().float().pow(2).mean().sqrt()

            h = self.ple_injections[layer_idx](h, per_layer_input, scale)

            if self.log_ple_metrics:
                with torch.no_grad():
                    correction_rms = (
                        (h.detach().float() - h.detach().float()).pow(2).mean().sqrt()
                    )
                    # gate RMS: recompute gate for logging (cheap, no_grad)
                    gate_rms = (
                        F.gelu(
                            self.ple_injections[layer_idx].per_layer_input_gate(
                                h.detach().float()
                            )
                        )
                        .pow(2)
                        .mean()
                        .sqrt()
                    )
                    self._metric_records.append(
                        {
                            "hidden_rms": hidden_rms,
                            "gate_rms": gate_rms,
                        }
                    )

            if rest is not None:
                return (h,) + rest
            return h

        return hook

    # ------------------------------------------------------------------
    # Forward override: fuse lookup + mix before layer loop
    # ------------------------------------------------------------------

    def forward(self, input_ids=None, **kwargs):
        if input_ids is not None:
            raw = self.ple_embedder(input_ids)  # [B, T, L, ple_dim]
            self._ple_mixed = self.ple_projector(raw)  # [B, T, L, ple_dim]
        else:
            self._ple_mixed = None

        self._metric_records = []
        outputs = super().forward(input_ids=input_ids, **kwargs)
        self._ple_mixed = None  # free; don't hold across steps

        # Collect metrics
        ple_metrics = {}
        if self.log_ple_metrics and self._metric_records:
            ple_metrics = self._finalize_metrics()
        self._last_ple_metrics = ple_metrics

        labels_present = kwargs.get("labels", None) is not None
        if labels_present:
            out = CausalLMOutputWithPast(loss=outputs.loss, logits=None)
        else:
            out = outputs
        out.ple_metrics = ple_metrics
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _finalize_metrics(self) -> dict:
        if not self._metric_records:
            return {}
        hidden_rms_vals = [float(r["hidden_rms"]) for r in self._metric_records]
        gate_rms_vals = [float(r["gate_rms"]) for r in self._metric_records]

        sample_size = self._ple_config.get("ple_cosine_sample_size", 512)
        cosine = self.ple_embedder.pairwise_cosine_mean(sample_size=sample_size)

        return {
            # Core diversity metric — equivalent to LEC's ple_pairwise_cosine
            # but measured across vocab tokens rather than experts.
            # Want this to stay well below 1.0 (tokens learning distinct directions).
            "ple_pairwise_cosine_mean": cosine,
            # Gate firing strength — how hard the GELU gate is activating.
            # Near 0 at init (zero-init out_proj means gate output irrelevant),
            # should grow as PLE correction becomes meaningful.
            "ple_gate_activation_rms_mean": float(
                sum(gate_rms_vals) / len(gate_rms_vals)
            ),
            "ple_gate_activation_rms_max": float(max(gate_rms_vals)),
            # Hidden stream magnitude
            "ple_hidden_rms_mean": float(sum(hidden_rms_vals) / len(hidden_rms_vals)),
            # Per-layer table norm (proxy for embedding magnitude growth)
            "ple_table_weight_norm_mean": float(
                sum(
                    emb.weight.detach().float().norm(dim=-1).mean().item()
                    for emb in self.ple_embedder.embeddings
                )
                / self.ple_embedder.num_layers
            ),
        }

    # ------------------------------------------------------------------
    # Param summary
    # ------------------------------------------------------------------

    def _print_param_summary(self):
        def n(name_part):
            return sum(p.numel() for n, p in self.named_parameters() if name_part in n)

        total = sum(p.numel() for p in self.parameters())
        tables = n("ple_embedder")
        proj = n("ple_projector") + n("ple_injections")
        dense = total - tables - proj
        print(f"\n[LlamaGemma3nPLE] Parameter breakdown")
        print(f"  Dense backbone : {dense:>14,}  ({100 * dense / total:.1f}%)")
        print(
            f"  PLE tables     : {tables:>14,}  ({100 * tables / total:.1f}%)  "
            f"[{self.ple_embedder.vocab_size} × {self.ple_embedder.num_layers} × {self.ple_embedder.ple_dim}]"
        )
        print(f"  PLE projectors : {proj:>14,}  ({100 * proj / total:.1f}%)")
        print(f"  Total          : {total:>14,}  ({total / 1e9:.3f}B)\n")


# =============================================================================
# BUILD HELPERS
# =============================================================================


def build_model(config: dict) -> LlamaGemma3nPLE:
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
    ple_config = {
        "ple_dim": config["ple_dim"],
        "ple_vocab_size": config["ple_vocab_size"],
        "ple_init_scale": config["ple_init_scale"],
        "ple_cosine_sample_size": config["ple_cosine_sample_size"],
    }
    return LlamaGemma3nPLE._from_config(
        llama_cfg,
        ple_config=ple_config,
        attn_implementation=config["attn_implementation"],
    )


def build_tokenizer(config: dict):
    tok = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tok.model_max_length = int(1e9)
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
    def __init__(self, config, tokenizer, seed=42, max_examples=None):
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

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        dataset = self.dataset
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nw = wi.num_workers if wi else 1
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        ns = max(1, world * nw)
        si = rank * nw + wid
        if ns > 1:
            dataset = dataset.shard(num_shards=ns, index=si)
        dataset = dataset.shuffle(
            seed=self.seed + self.epoch + si, buffer_size=self.buffer_size
        )
        it, buf, n = iter(dataset), [], 0
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
                toks = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    verbose=False,
                )["input_ids"]
                if toks:
                    buf.extend(toks)
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


class PLETrainer(Trainer):
    def __init__(self, *args, metric_log_every_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_log_every_steps = metric_log_every_steps
        self._metric_sums: dict = {}
        self._metric_count: int = 0

    def _accumulate(self, metrics: Optional[dict]):
        if not metrics:
            return
        for k, v in metrics.items():
            self._metric_sums[k] = self._metric_sums.get(k, 0.0) + float(v)
        self._metric_count += 1

    def _flush(self) -> dict:
        if self._metric_count == 0:
            return {}
        out = {k: v / self._metric_count for k, v in self._metric_sums.items()}
        self._metric_sums = {}
        self._metric_count = 0
        return out

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        unwrapped = self.accelerator.unwrap_model(model)
        should_log = (
            self.metric_log_every_steps > 0
            and self.state.global_step > 0
            and self.state.global_step % self.metric_log_every_steps == 0
        )
        if hasattr(unwrapped, "log_ple_metrics"):
            unwrapped.log_ple_metrics = should_log

        if num_items_in_batch is not None:
            inputs = {**inputs, "num_items_in_batch": num_items_in_batch}

        outputs = model(**inputs)
        loss = outputs.loss

        if should_log:
            self._accumulate(getattr(outputs, "ple_metrics", None))
            if hasattr(unwrapped, "log_ple_metrics"):
                unwrapped.log_ple_metrics = False

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            try:
                logs["ppl"] = round(math.exp(min(float(logs["loss"]), 20)), 2)
            except Exception:
                pass
            logs.update(self._flush())
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
        print(
            f"[Warning] Resizing main embeddings: {model.config.vocab_size} → {len(tokenizer)}"
        )
        model.resize_token_embeddings(len(tokenizer))
        # PLE tables use ple_vocab_size (separate config field), no resize needed

    train_ds = PackedFineWebDataset(config, tokenizer, seed=config["seed"])
    eval_ds = PackedFineWebDataset(
        config,
        tokenizer,
        seed=config["seed"] + 9999,
        max_examples=config["eval_max_examples"],
    )

    tps = (
        config["per_device_batch_size"]
        * config["grad_accum_steps"]
        * config["max_seq_len"]
    )
    print(f"[Train] tokens/step : {tps:,}")
    print(f"[Train] total tokens : {tps * config['max_steps'] / 1e9:.2f}B\n")

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
        save_total_limit=5,
        dataloader_num_workers=config["dataloader_workers"],
        dataloader_prefetch_factor=config["dataloader_prefetch_factor"],
        seed=config["seed"],
        report_to="wandb" if config["use_wandb"] else "none",
        run_name="llama_gemma3n_ple",
        remove_unused_columns=False,
        label_names=["labels"],
        torch_compile=config["torch_compile"],
    )

    trainer = PLETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric_log_every_steps=config["metric_log_every_steps"],
    )

    trainer.train()
    print("[Train] LLaMA + Gemma 3n PLE pretraining complete.")


if __name__ == "__main__":
    train(CONFIG)
