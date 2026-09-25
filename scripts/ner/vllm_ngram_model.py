"""vLLM-compatible model class for LlamaLongCatNgram and LlamaQKNorm.

Registers two model architectures with vLLM:
  - LlamaQKNorm: Llama + QK normalization
  - LlamaLongCatNgram: Llama + QK norm + n-gram hash embedding

Usage::

    import vllm_ngram_model  # registers on import
    from vllm import LLM
    llm = LLM(model="path/to/checkpoint", trust_remote_code=True,
              max_num_seqs=1, enable_prefix_caching=False, ...)
    # n-gram context buffer holds a single sequence; see LongCatNgramEmbedder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vllm import ModelRegistry
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)


# ---------------------------------------------------------------------------
# QK-Norm Attention
# ---------------------------------------------------------------------------

class LlamaQKNormAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = args[0] if args else kwargs.get("config")
        eps = getattr(config, "qk_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaQKNormDecoderLayer(LlamaDecoderLayer):

    def __init__(self, vllm_config, prefix="", **kwargs):
        super().__init__(
            vllm_config,
            prefix=prefix,
            attn_layer_type=LlamaQKNormAttention,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# N-gram Hash Embedding (position-indexed GPU token buffer)
# ---------------------------------------------------------------------------

class LongCatNgramEmbedder(nn.Module):
    """N-gram hash embedding matching the HF ``LongCatNgramEmbedder``.

    Operates on 1D ``input_ids`` / ``positions`` (vLLM format).  n-gram context
    comes from a GPU token buffer indexed by absolute position: every forward
    writes ``buf[positions] = input_ids`` and then gathers ``buf[positions - j]``.
    This is exact for prefill, chunked prefill and decode, needs no host sync,
    and is CUDA-graph safe.  It holds ONE sequence's tokens, so it requires
    ``max_num_seqs=1`` and ``enable_prefix_caching=False`` (a prefix-cache hit
    skips the forward that would have written the cached positions).
    """

    def __init__(self, config, max_positions: int):
        super().__init__()
        self.max_n = config.ngram_max_n
        self.num_heads = config.ngram_num_heads
        self.orders = list(range(2, self.max_n + 1))
        self.eos_token_id = config.eos_token_id

        num_tables = len(self.orders) * self.num_heads
        table_vocab_sizes = config.ngram_table_vocab_sizes
        multipliers = getattr(config, "ngram_hash_multipliers", None)
        if multipliers is None or len(multipliers) != num_tables:
            # Older checkpoints (e.g. longcat_ngram_50) hash with base = vocab.
            multipliers = [config.vocab_size] * num_tables
        sub_dim = config.hidden_size // num_tables

        self.tables = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.multipliers: dict[str, int] = {}

        idx = 0
        for n in self.orders:
            for k in range(self.num_heads):
                key = f"n{n}_k{k}"
                self.tables[key] = nn.Embedding(table_vocab_sizes[idx], sub_dim)
                self.projections[key] = nn.Linear(
                    sub_dim, config.hidden_size, bias=False,
                )
                self.multipliers[key] = int(multipliers[idx])
                idx += 1

        amplification = config.ngram_embedding_amplification.strip().lower()
        if amplification == "layer_norm":
            self.amplification = nn.LayerNorm(config.hidden_size)
            self.amplification_scale = 1.0
        elif amplification == "sqrt_d":
            self.amplification = nn.Identity()
            self.amplification_scale = math.sqrt(config.hidden_size)
        else:
            self.amplification = nn.Identity()
            self.amplification_scale = 1.0

        self.register_buffer(
            "token_buf", torch.zeros(max_positions, dtype=torch.long),
            persistent=False,
        )

    def _shifted_tokens(self, positions: torch.Tensor) -> dict[int, torch.Tensor]:
        """Token at ``pos - shift`` (0 before BOS), zeroed if the span crosses EOS.

        Mirrors HF ``_shift_right``: for shift s the context is dropped when any
        of the tokens at offsets 1..s is EOS.
        """
        shifted: dict[int, torch.Tensor] = {}
        crosses_eos = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(1, self.max_n):
            prev_pos = positions - shift
            prev = torch.where(
                prev_pos >= 0,
                self.token_buf[prev_pos.clamp(min=0)],
                torch.zeros_like(prev_pos),
            )
            crosses_eos = crosses_eos | prev.eq(self.eos_token_id)
            shifted[shift] = prev.masked_fill(crosses_eos, 0)
        return shifted

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        base_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        positions = positions.long()
        input_ids = input_ids.long()
        self.token_buf[positions] = input_ids
        shifted = self._shifted_tokens(positions)

        combined = base_embeddings
        for n in self.orders:
            for k in range(self.num_heads):
                key = f"n{n}_k{k}"
                table_size = self.tables[key].num_embeddings
                mult = self.multipliers[key]
                h = torch.zeros_like(input_ids)
                for j in range(n - 1, -1, -1):
                    tok = input_ids if j == 0 else shifted[j]
                    h = (h * mult + tok) % table_size
                combined = combined + self.projections[key](self.tables[key](h))

        combined = combined / (len(self.orders) * self.num_heads + 1)
        return self.amplification(combined) * self.amplification_scale


# ---------------------------------------------------------------------------
# vLLM model classes
# ---------------------------------------------------------------------------

class VllmLlamaQKNorm(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=LlamaQKNormDecoderLayer,
        )


class VllmLlamaLongCatNgram(VllmLlamaQKNorm):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.ngram_embedder = LongCatNgramEmbedder(
            vllm_config.model_config.hf_config,
            max_positions=vllm_config.model_config.max_model_len,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if input_ids is not None and inputs_embeds is None:
            base_embeddings = self.model.embed_input_ids(input_ids)
            inputs_embeds = self.ngram_embedder(
                input_ids, positions, base_embeddings,
            )
            input_ids = None
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds,
        )


# ---------------------------------------------------------------------------
# Register on import
# ---------------------------------------------------------------------------
ModelRegistry.register_model("LlamaQKNorm", VllmLlamaQKNorm)
ModelRegistry.register_model("LlamaLongCatNgram", VllmLlamaLongCatNgram)
