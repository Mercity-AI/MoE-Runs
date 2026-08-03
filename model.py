"""Model architectures shared by the baseline and LongCat training scripts."""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama as llama_modeling


class LlamaLongCatNgramConfig(LlamaConfig):
    """Serializable configuration for :class:`LlamaLongCatNgram`."""

    model_type = "llama_longcat_ngram"

    def __init__(
        self,
        ngram_max_n: int = 4,
        ngram_num_heads: int = 2,
        ngram_table_vocab_sizes: Optional[list[int]] = None,
        ngram_embedding_amplification: str = "layer_norm",
        qk_norm: bool = False,
        qk_norm_eps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ngram_max_n = ngram_max_n
        self.ngram_num_heads = ngram_num_heads
        self.ngram_table_vocab_sizes = ngram_table_vocab_sizes
        self.ngram_embedding_amplification = ngram_embedding_amplification
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps


class LongCatNgramEmbedder(nn.Module):
    """LongCat N-gram Embedding from Eq. 2 and Eq. 3 of arXiv:2601.21204."""

    def __init__(self, config: LlamaLongCatNgramConfig):
        super().__init__()
        self.max_n = config.ngram_max_n
        self.num_heads = config.ngram_num_heads
        self.base_vocab_size = config.vocab_size
        self.eos_token_id = config.eos_token_id
        self.orders = list(range(2, self.max_n + 1))

        num_tables = len(self.orders) * self.num_heads
        table_vocab_sizes = config.ngram_table_vocab_sizes
        if config.hidden_size % num_tables != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"(ngram_max_n-1)*ngram_num_heads ({num_tables})."
            )
        if table_vocab_sizes is None or len(table_vocab_sizes) != num_tables:
            actual = None if table_vocab_sizes is None else len(table_vocab_sizes)
            raise ValueError(
                f"Expected {num_tables} ngram_table_vocab_sizes "
                f"((ngram_max_n-1)*ngram_num_heads), got {actual}."
            )
        self.sub_dim = config.hidden_size // num_tables

        self.tables = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        idx = 0
        for n in self.orders:
            for k in range(self.num_heads):
                key = f"n{n}_k{k}"
                self.tables[key] = nn.Embedding(
                    table_vocab_sizes[idx], self.sub_dim
                )
                self.projections[key] = nn.Linear(
                    self.sub_dim, config.hidden_size, bias=False
                )
                idx += 1

        amplification = config.ngram_embedding_amplification.strip().lower()
        if amplification == "layer_norm":
            self.amplification = nn.LayerNorm(config.hidden_size)
            self.amplification_scale = 1.0
        elif amplification == "sqrt_d":
            self.amplification = nn.Identity()
            self.amplification_scale = math.sqrt(config.hidden_size)
        elif amplification == "none":
            self.amplification = nn.Identity()
            self.amplification_scale = 1.0
        else:
            raise ValueError(
                "ngram_embedding_amplification must be one of "
                "{'layer_norm', 'sqrt_d', 'none'}, got "
                f"{amplification!r}."
            )

    def _shift_right(self, x: torch.Tensor, shift: int) -> torch.Tensor:
        """Causal shift, zeroing context that crosses an EOS boundary."""
        if shift == 0:
            return x
        pad = x.new_zeros(x.shape[0], shift)
        shifted = torch.cat([pad, x[:, :-shift]], dim=1)

        crosses_eos = torch.zeros_like(x, dtype=torch.bool)
        for offset in range(1, shift + 1):
            previous = torch.cat(
                [x.new_zeros(x.shape[0], offset), x[:, :-offset]], dim=1
            )
            crosses_eos |= previous.eq(self.eos_token_id)
        return shifted.masked_fill(crosses_eos, 0)

    def _hash_ngram(
        self,
        input_ids: torch.Tensor,
        n: int,
        table_size: int,
        shifted_tokens: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Eq. 2: sum_j t[i-j] * V0**j mod table_size."""
        h = torch.zeros_like(input_ids)
        for j in range(n - 1, -1, -1):
            tok = (
                input_ids
                if j == 0
                else shifted_tokens[j]
                if shifted_tokens is not None
                else self._shift_right(input_ids, j)
            )
            h = (h * self.base_vocab_size + tok) % table_size
        return h

    def forward(
        self, input_ids: torch.Tensor, base_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Return amplified Eq. 3 embeddings with shape [B, T, H]."""
        combined = base_embeddings
        shifted_tokens = {
            shift: self._shift_right(input_ids, shift)
            for shift in range(1, self.max_n)
        }
        for n in self.orders:
            for k in range(self.num_heads):
                key = f"n{n}_k{k}"
                table_size = self.tables[key].num_embeddings
                hash_ids = self._hash_ngram(
                    input_ids, n, table_size, shifted_tokens
                )
                combined = combined + self.projections[key](
                    self.tables[key](hash_ids)
                )

        combined = combined / (len(self.orders) * self.num_heads + 1)
        return self.amplification(combined) * self.amplification_scale


class LlamaQKNormAttention(llama_modeling.LlamaAttention):
    """LLaMA attention with per-head RMSNorm on Q and K before RoPE."""

    def __init__(self, config: LlamaLongCatNgramConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        eps = config.qk_norm_eps if config.qk_norm_eps is not None else config.rms_norm_eps
        self.q_norm = llama_modeling.LlamaRMSNorm(self.head_dim, eps=eps)
        self.k_norm = llama_modeling.LlamaRMSNorm(self.head_dim, eps=eps)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings=None,
        attention_mask=None, past_key_values=None, **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = llama_modeling.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = llama_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, llama_modeling.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )
        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
        return attn_output, attn_weights


class LlamaLongCatNgram(LlamaForCausalLM):
    """LLaMA using LongCat's standard input N-gram Embedding (NE)."""

    config_class = LlamaLongCatNgramConfig

    def __init__(self, config: LlamaLongCatNgramConfig):
        super().__init__(config)
        if config.qk_norm:
            for layer_idx, layer in enumerate(self.model.layers):
                layer.self_attn = LlamaQKNormAttention(config, layer_idx)
        self.ngram_embedder = LongCatNgramEmbedder(config)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if input_ids is not None:
            base_embeddings = self.model.embed_tokens(input_ids)
            inputs_embeds = self.ngram_embedder(input_ids, base_embeddings)
            input_ids = None
        return super().forward(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Preserve causal n-gram context when HF slices cached decode inputs."""
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        prepared_ids = model_inputs.get("input_ids")
        if prepared_ids is None:
            return model_inputs

        base_embeddings = self.model.embed_tokens(input_ids)
        full_embeddings = self.ngram_embedder(input_ids, base_embeddings)
        model_inputs["inputs_embeds"] = full_embeddings[:, -prepared_ids.shape[1] :]
        model_inputs["input_ids"] = None
        return model_inputs


# Instruct save_pretrained() to package this source file and write AutoClass
# metadata. Loading the resulting checkpoint requires trust_remote_code=True.
LlamaLongCatNgramConfig.register_for_auto_class()
LlamaLongCatNgram.register_for_auto_class("AutoModelForCausalLM")


__all__ = [
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaLongCatNgramConfig",
    "LongCatNgramEmbedder",
    "LlamaQKNormAttention",
    "LlamaLongCatNgram",
]
