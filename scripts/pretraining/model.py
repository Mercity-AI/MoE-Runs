"""Model architectures shared by the baseline and LongCat training scripts."""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama as llama_modeling

try:
    # Kimi Delta Attention layer (linear attention) from flash-linear-attention.
    # Optional: only the KDA architecture needs it, so the baseline/longcat scripts
    # keep importing model.py even when FLA is absent.
    from fla.layers.kda import KimiDeltaAttention

    FLA_KDA_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without FLA
    KimiDeltaAttention = None
    FLA_KDA_IMPORT_ERROR = exc


# Distinct prime hash multipliers ("salts"), one per (order, head) table. Giving
# every head a different polynomial base makes the K hash functions genuinely
# independent *regardless of table size*, so the historical K->1 collapse — two
# heads that shared a table size hashed every n-gram to the identical slot —
# cannot recur. Primes larger than the base vocab keep each per-head polynomial
# injective over the token range, and being coprime to the table sizes avoids the
# base-multiple collision spike the LongCat paper reports (Fig. 3b): the effective
# base (multiplier mod table_size) is then a scrambled value rather than the raw
# vocab size. All of this is enforced at build time by _validate_ngram_hashing.
_DEFAULT_HASH_MULTIPLIERS = (
    40009, 100003, 262147, 524287, 1000003, 2000003,
    3000017, 4000037, 5000011, 6000101, 7000127, 8000009,
)

# Default minimum pairwise separation between table sizes (fraction of the
# smaller size). Near-equal sizes are the condition that silently disabled
# multi-head hashing before, so the build refuses to start below this.
_DEFAULT_MIN_PAIRWISE_SIZE_GAP = 0.005


def _validate_ngram_hashing(
    table_sizes: list[int],
    multipliers: list[int],
    base_vocab: int,
    min_pairwise_size_gap: float,
) -> None:
    """Refuse to build a degenerate n-gram hashing setup. Raises ValueError.

    Guards, in order of how badly they corrupt the experiment:

    1. No two tables may be the *same hash function*. Two heads are identical iff
       they share both a table size and an effective base (multiplier mod size);
       that is the exact K->1 collapse. This is the load-bearing invariant.
    2. Each multiplier must be >= base vocab, or distinct n-grams alias before the
       modulus (the base-`m` polynomial stops being injective over token digits).
    3. Each multiplier must be coprime to its table size, or one n-gram coordinate
       collapses into gcd-many classes (the mechanism behind the paper's spike).
    4. Table sizes must not be near-duplicates — the config that hid the clone bug.

    A soft warning also fires when a size sits within 5% of base vocab of an
    integer multiple of it (paper Fig. 3b), which prime multipliers mitigate but
    do not fully erase.
    """
    n = len(table_sizes)
    if len(multipliers) != n:
        raise ValueError(
            f"Expected {n} ngram hash multipliers (one per table), got {len(multipliers)}."
        )

    # (1) precise clone check: identical (size, effective base) => identical indices.
    seen: dict[tuple[int, int], int] = {}
    for idx, (m, s) in enumerate(zip(multipliers, table_sizes)):
        key = (s, m % s)
        if key in seen:
            raise ValueError(
                f"N-gram hash tables {seen[key]} and {idx} are the SAME hash function "
                f"(table size {s}, effective base {m % s}). Two heads that hash "
                "identically collapse K sub-tables to K=1 — exactly the bug this guard "
                "exists to prevent. Give them distinct multipliers or distinct sizes."
            )
        seen[key] = idx

    for m, s in zip(multipliers, table_sizes):
        # (2) injectivity of the base-`m` polynomial over token digits [0, base_vocab).
        if m < base_vocab:
            raise ValueError(
                f"N-gram hash multiplier {m} must be >= base vocab {base_vocab}; "
                "a smaller base aliases distinct n-grams before the modulus is applied."
            )
        # (3) coprimality: gcd > 1 collapses a coordinate into gcd-many residues.
        g = math.gcd(m, s)
        if g != 1:
            raise ValueError(
                f"N-gram hash multiplier {m} shares factor {g} with table size {s}. "
                "Pick a multiplier coprime to the table size (a prime larger than every "
                "table size is always safe) so no n-gram coordinate collapses."
            )

    # (4) near-duplicate sizes: the historical trigger for the clone collapse.
    order = sorted(range(n), key=lambda i: table_sizes[i])
    for a, b in zip(order, order[1:]):
        sa, sb = table_sizes[a], table_sizes[b]
        rel = abs(sa - sb) / min(sa, sb)
        if rel < min_pairwise_size_gap:
            raise ValueError(
                f"N-gram table sizes {sa} and {sb} differ by only {rel * 100:.3f}% "
                f"(guard requires >= {min_pairwise_size_gap * 100:.3f}%). Near-equal "
                "sizes are the condition that silently disabled multi-head hashing "
                "before; spread the table sizes apart."
            )

    # (5) soft: sizes near an integer multiple of base vocab (paper Fig. 3b).
    for s in table_sizes:
        dist = min(s % base_vocab, base_vocab - s % base_vocab)
        if dist / base_vocab < 0.05:
            warnings.warn(
                f"N-gram table size {s} is within {dist} of an integer multiple of base "
                f"vocab {base_vocab}; the LongCat paper reports collision spikes there. "
                "The prime multipliers mitigate this, but consider nudging the size.",
                stacklevel=2,
            )


class LlamaLongCatNgramConfig(LlamaConfig):
    """Serializable configuration for :class:`LlamaLongCatNgram`."""

    model_type = "llama_longcat_ngram"

    def __init__(
        self,
        ngram_max_n: int = 4,
        ngram_num_heads: int = 2,
        ngram_table_vocab_sizes: Optional[list[int]] = None,
        ngram_embedding_amplification: str = "layer_norm",
        ngram_hash_multipliers: Optional[list[int]] = None,
        ngram_min_pairwise_size_gap: float = _DEFAULT_MIN_PAIRWISE_SIZE_GAP,
        qk_norm: bool = False,
        qk_norm_eps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ngram_max_n = ngram_max_n
        self.ngram_num_heads = ngram_num_heads
        self.ngram_table_vocab_sizes = ngram_table_vocab_sizes
        self.ngram_embedding_amplification = ngram_embedding_amplification
        self.ngram_hash_multipliers = ngram_hash_multipliers
        self.ngram_min_pairwise_size_gap = ngram_min_pairwise_size_gap
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

        # Resolve the per-head hash multipliers (built-in defaults unless the
        # config overrides them) and refuse to build a degenerate setup.
        configured = config.ngram_hash_multipliers
        if configured is None:
            if num_tables > len(_DEFAULT_HASH_MULTIPLIERS):
                raise ValueError(
                    f"Need {num_tables} hash multipliers but only "
                    f"{len(_DEFAULT_HASH_MULTIPLIERS)} defaults are defined; pass "
                    "ngram_hash_multipliers explicitly."
                )
            configured = _DEFAULT_HASH_MULTIPLIERS[:num_tables]
        multipliers: list[int] = [int(m) for m in configured]
        _validate_ngram_hashing(
            list(table_vocab_sizes),
            multipliers,
            self.base_vocab_size,
            config.ngram_min_pairwise_size_gap,
        )
        # Persist the resolved list so it is serialized in config.json.
        config.ngram_hash_multipliers = multipliers

        self.tables = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.multipliers: dict[str, int] = {}
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
                self.multipliers[key] = multipliers[idx]
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
        multiplier: int,
        shifted_tokens: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Eq. 2 with a per-head base: sum_j t[i-j] * multiplier**j mod table_size.

        `multiplier` is this head's hash salt (a distinct prime >= base vocab), so
        two heads never compute the same indices even at equal table sizes. The
        modulus is applied every Horner step, so the result matches the full
        polynomial mod `table_size` while staying far inside int64.
        """
        h = torch.zeros_like(input_ids)
        for j in range(n - 1, -1, -1):
            tok = (
                input_ids
                if j == 0
                else shifted_tokens[j]
                if shifted_tokens is not None
                else self._shift_right(input_ids, j)
            )
            h = (h * multiplier + tok) % table_size
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
                    input_ids, n, table_size, self.multipliers[key], shifted_tokens
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


class LlamaKDAConfig(LlamaConfig):
    """Config for :class:`LlamaKDA` — a hybrid Kimi-Delta / softmax LLaMA.

    A subset of decoder layers use Kimi Delta Attention (KDA, a gated-delta
    linear attention); the rest keep standard softmax self-attention (with the
    same FA backend and optional QK-norm as the baseline). Which layers are which
    is resolved by :func:`resolve_kda_layer_types`, honoring (in priority order)
    ``kda_full_attn_layers`` > ``kda_full_attn_range`` > ``kda_full_attn_every``.
    With none set, every layer is KDA (pure linear attention).
    """

    model_type = "llama_kda"

    def __init__(
        self,
        # --- Hybrid layout: which layers keep softmax (full) attention ---
        kda_full_attn_layers: Optional[list[int]] = None,
        kda_full_attn_every: Optional[int] = None,
        kda_full_attn_range: Optional[list[int]] = None,
        # Sparse-KDA interleave (the inverse of kda_full_attn_every): one KDA layer
        # every `kda_every` layers, at indices where i % kda_every == kda_offset;
        # every other layer is full (GQA) attention.
        kda_every: Optional[int] = None,
        kda_offset: int = 0,
        # --- KDA layer hyperparameters (forwarded to fla KimiDeltaAttention) ---
        kda_head_dim: int = 128,
        kda_num_heads: Optional[int] = None,
        kda_num_v_heads: Optional[int] = None,
        kda_expand_v: float = 1.0,
        kda_use_short_conv: bool = True,
        kda_conv_size: int = 4,
        kda_conv_bias: bool = False,
        kda_allow_neg_eigval: bool = False,
        kda_lower_bound: Optional[float] = None,
        kda_safe_gate: bool = False,
        # --- QK-norm applies to the softmax (full-attention) layers only ---
        qk_norm: bool = False,
        qk_norm_eps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kda_full_attn_layers = kda_full_attn_layers
        self.kda_full_attn_every = kda_full_attn_every
        self.kda_full_attn_range = kda_full_attn_range
        self.kda_every = kda_every
        self.kda_offset = kda_offset
        self.kda_head_dim = kda_head_dim
        self.kda_num_heads = kda_num_heads
        self.kda_num_v_heads = kda_num_v_heads
        self.kda_expand_v = kda_expand_v
        self.kda_use_short_conv = kda_use_short_conv
        self.kda_conv_size = kda_conv_size
        self.kda_conv_bias = kda_conv_bias
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.kda_lower_bound = kda_lower_bound
        self.kda_safe_gate = kda_safe_gate
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps


def resolve_kda_layer_types(config: LlamaKDAConfig) -> list[str]:
    """Return a per-layer list of ``"kda"`` / ``"full"`` (softmax) attention.

    Priority: explicit ``kda_full_attn_layers`` > contiguous ``kda_full_attn_range``
    ``[start, end)`` > interleaved ``kda_full_attn_every`` (the last layer of every
    block of ``n`` is full attention, e.g. ``4`` -> Kimi/Qwen-style 3:1) > ``kda_every``
    (the INVERSE — KDA is the sparse type: one KDA layer every ``kda_every`` layers at
    ``i % kda_every == kda_offset``, all others full). If none is set, all layers are KDA.
    """
    n = config.num_hidden_layers
    if config.kda_full_attn_layers is not None:
        full = set(int(i) for i in config.kda_full_attn_layers)
    elif config.kda_full_attn_range is not None:
        start, end = config.kda_full_attn_range
        full = set(range(int(start), int(end)))
    elif config.kda_full_attn_every:
        every = int(config.kda_full_attn_every)
        if every < 1:
            raise ValueError(f"kda_full_attn_every must be >= 1, got {every}.")
        full = {i for i in range(n) if (i + 1) % every == 0}
    elif getattr(config, "kda_every", None):
        # Inverse of kda_full_attn_every: KDA is the SPARSE type. One KDA layer every
        # `kda_every` layers at i % kda_every == kda_offset; every other layer is full.
        every = int(config.kda_every)
        if every < 1:
            raise ValueError(f"kda_every must be >= 1, got {every}.")
        offset = int(getattr(config, "kda_offset", 0) or 0) % every
        kda = {i for i in range(n) if i % every == offset}
        full = set(range(n)) - kda
    else:
        full = set()
    for i in full:
        if not 0 <= i < n:
            raise ValueError(
                f"Full-attention layer index {i} is out of range for "
                f"num_hidden_layers={n}."
            )
    return ["full" if i in full else "kda" for i in range(n)]


# fla ships two KDA kernels: ``chunk_kda`` (chunked-parallel, the ONLY one with a
# real Triton autograd backward) and ``fused_recurrent_kda`` (step-wise, for
# short/decode inference). Training must always run ``chunk``. fla self-guards
# (`assert mode == "chunk"` when a module is in training mode, and it only
# auto-switches to fused_recurrent for q_len <= 64 in eval), but we pin and assert
# the training kernel at our own boundary so a config/library drift can never
# silently train on the inference kernel.
KDA_TRAINING_MODE = "chunk"


class LlamaKDAAttention(nn.Module):
    """Adapter wrapping fla's :class:`KimiDeltaAttention` for a LLaMA decoder layer.

    KDA is linear attention: it carries no RoPE and normalizes q/k internally
    (L2-norm), so ``position_embeddings`` are ignored here. The decoder layer
    expects a ``(hidden_states, attn_weights)`` pair back; KDA returns a triple, so
    we drop the cache/weights. A 4-D causal mask (built by ``LlamaModel`` for the
    softmax layers) is meaningless to KDA — only a 2-D ``[B, T]`` padding mask is
    forwarded; anything else becomes ``None`` (packed training carries no padding).

    The module is run **stateless**: it never reads or writes ``past_key_values``.
    HF's ``LlamaModel`` hands every layer an HF ``DynamicCache`` (incompatible with
    fla's recurrent-state cache), which is fine for full-sequence LM-loss training
    and eval but means this wrapper does not support HF incremental ``generate``.
    """

    def __init__(self, config: LlamaKDAConfig, layer_idx: int):
        super().__init__()
        if KimiDeltaAttention is None:
            raise ImportError(
                "LlamaKDA requires flash-linear-attention (fla) for KimiDeltaAttention."
            ) from FLA_KDA_IMPORT_ERROR
        head_dim = config.kda_head_dim
        num_heads = config.kda_num_heads or (config.hidden_size // head_dim)
        if num_heads * head_dim != config.hidden_size:
            # fla supports q/k dim != hidden; Kimi-Linear over-provisions ~1.8x.
            warnings.warn(
                f"KDA q/k dim {num_heads * head_dim} != hidden_size "
                f"{config.hidden_size}; layer params/state will differ from a "
                "same-width softmax layer.",
                stacklevel=2,
            )
        self.layer_idx = layer_idx
        self.kda = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            expand_v=config.kda_expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=config.kda_num_v_heads,
            mode=KDA_TRAINING_MODE,
            use_short_conv=config.kda_use_short_conv,
            conv_size=config.kda_conv_size,
            conv_bias=config.kda_conv_bias,
            allow_neg_eigval=config.kda_allow_neg_eigval,
            safe_gate=config.kda_safe_gate,
            lower_bound=config.kda_lower_bound,
            layer_idx=layer_idx,
            norm_eps=config.rms_norm_eps,
        )
        # Construction guard: fail loud if fla ever stops honouring the requested
        # kernel (so training can't silently fall onto a non-backward path).
        built_mode = getattr(self.kda, "mode", None)
        if built_mode != KDA_TRAINING_MODE:
            raise ValueError(
                f"KDA layer {layer_idx} built with mode={built_mode!r}, but training "
                f"requires the {KDA_TRAINING_MODE!r} kernel (only chunk_kda has a backward)."
            )

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings=None,
        attention_mask=None, past_key_values=None, use_cache=False, **kwargs,
    ):
        # Runtime guard: any forward that will build a graph (module in training
        # mode) MUST use the chunk kernel. fla routes to fused_recurrent only when
        # ``q_len <= 64 and not self.training``; asserting training-mode==chunk here
        # closes that path at our boundary and catches a layer left in eval mode
        # during a training step (which, with a short q_len, would lose gradients).
        if self.training and getattr(self.kda, "mode", None) != KDA_TRAINING_MODE:
            raise RuntimeError(
                f"KDA layer {self.layer_idx} is in training mode but its kernel is "
                f"{getattr(self.kda, 'mode', None)!r}, not {KDA_TRAINING_MODE!r}; refusing "
                "to train on the inference (fused_recurrent) path."
            )
        mask = attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None
        forward_kwargs = {k: v for k, v in kwargs.items() if k == "cu_seqlens"}
        # Always stateless: never read/write a recurrent cache (past_key_values=None,
        # use_cache=False), so the recurrent-state branch is unreachable regardless.
        attn_output, _, _ = self.kda(
            hidden_states=hidden_states,
            attention_mask=mask,
            past_key_values=None,
            use_cache=False,
            **forward_kwargs,
        )
        return attn_output, None


class LlamaKDA(LlamaForCausalLM):
    """LLaMA whose attention is a config-driven hybrid of KDA and softmax layers."""

    config_class = LlamaKDAConfig

    def __init__(self, config: LlamaKDAConfig):
        super().__init__(config)
        layer_types = resolve_kda_layer_types(config)
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_types[layer_idx] == "kda":
                layer.self_attn = LlamaKDAAttention(config, layer_idx)
            elif config.qk_norm:
                layer.self_attn = LlamaQKNormAttention(config, layer_idx)
            # otherwise keep the default softmax LlamaAttention from super().__init__.
        # Persist the resolved layout so it lands in config.json and can be logged.
        config.kda_layer_types = layer_types


def assert_kda_training_kernels(model) -> int:
    """Fail fast if any KDA layer would train on the wrong kernel or in eval mode.

    Call this right after ``model.train()`` (and any DDP/compile wrap) but before
    the training loop, so a misconfiguration is caught before burning compute
    rather than mid-step. Returns the number of KDA layers verified. Non-KDA
    (softmax) layers are ignored, so it is safe to call on any LlamaKDA model.
    """
    base = getattr(model, "module", model)          # unwrap DDP
    base = getattr(base, "_orig_mod", base)          # unwrap torch.compile
    checked = 0
    for idx, layer in enumerate(base.model.layers):
        kda = getattr(layer.self_attn, "kda", None)
        if kda is None:
            continue
        checked += 1
        mode = getattr(kda, "mode", None)
        if mode != KDA_TRAINING_MODE:
            raise RuntimeError(
                f"KDA layer {idx} kernel is {mode!r}, not {KDA_TRAINING_MODE!r}; "
                "refusing to start training on a non-backward kernel."
            )
        if not kda.training:
            raise RuntimeError(
                f"KDA layer {idx} is in eval mode at training start; call model.train() "
                "first (eval + short q_len would route to the inference kernel)."
            )
    return checked


LlamaKDAConfig.register_for_auto_class()
LlamaKDA.register_for_auto_class("AutoModelForCausalLM")


class LlamaQKNormConfig(LlamaConfig):
    """Plain dense LLaMA config with optional QK-RMSNorm (Qwen3/Gemma-style).

    Same backbone as ``LlamaConfig``; adds ``qk_norm`` so the baseline can carry
    the per-head Q/K RMSNorm that the KDA and LongCat models already use. With
    ``qk_norm=False`` this is behaviourally identical to a vanilla LLaMA.
    """

    model_type = "llama_qknorm"

    def __init__(
        self,
        qk_norm: bool = False,
        qk_norm_eps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps


class LlamaQKNorm(LlamaForCausalLM):
    """Dense LLaMA whose softmax attention gets per-head Q/K RMSNorm when enabled."""

    config_class = LlamaQKNormConfig

    def __init__(self, config: LlamaQKNormConfig):
        super().__init__(config)
        if getattr(config, "qk_norm", False):
            for layer_idx, layer in enumerate(self.model.layers):
                layer.self_attn = LlamaQKNormAttention(config, layer_idx)
        # otherwise keep the default softmax LlamaAttention from super().__init__.


LlamaQKNormConfig.register_for_auto_class()
LlamaQKNorm.register_for_auto_class("AutoModelForCausalLM")


__all__ = [
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaLongCatNgramConfig",
    "LongCatNgramEmbedder",
    "LlamaQKNormAttention",
    "LlamaQKNormConfig",
    "LlamaQKNorm",
    "LlamaLongCatNgram",
    "LlamaKDAConfig",
    "LlamaKDAAttention",
    "LlamaKDA",
    "assert_kda_training_kernels",
    "KDA_TRAINING_MODE",
    "resolve_kda_layer_types",
    "_validate_ngram_hashing",
]
