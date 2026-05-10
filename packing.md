# Real-packing on the FA4 + TorchTitan + HF Llama stack

This doc covers **document-boundary varlen packing** for pretraining: each
training sample is a fixed-length sequence built by concatenating multiple
FineWeb documents, and the attention kernel is told (via `cu_seqlens`) to
*not* let tokens from doc A attend to tokens from doc B inside the same
packed sequence. This is what people usually mean when they say "real
packing" or "padding-free training".

The *naive* version of packing — what `PackedFineWebDataset` already does —
is **concat-pack with no masking**: tokens from different docs in the same
chunk happily attend to each other. That's the cheap version. "Real
packing" adds the document-boundary mask on top.

---

## The kernel-level picture

Varlen attention is just one kernel call:

```python
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=True,
)
```

- `q`, `k`, `v` are flattened to `(total_tokens, n_heads, head_dim)` —
  the batch and sequence dimensions are merged.
- `cu_seqlens_*` is a 1-D int32 tensor `[0, len_doc1, len_doc1+len_doc2, ...]`
  marking where each document starts/ends in the flattened sequence.
- `max_seqlen_*` is a scalar — the longest document in the batch.

The kernel then computes attention **only inside each document's slice**
(intra-doc causal triangle), never across slices. Total cross-doc
attention work = 0.

The hard part isn't the kernel — that's installed and working
(`flash_attn.cute.flash_attn_varlen_func`). The hard part is plumbing
those `cu_seqlens` tensors from the dataset all the way down into the
attention layer of every transformer block.

There are five different ways to do that plumbing. **Option 2 was the
chosen path.**

---

## All options (1 through 5)

### Option 1 — HF padding-free via `position_ids` restart

Stay on `LlamaForCausalLM` + `attn_implementation="flash_attention_4"`.
HF transformers already supports padding-free training: if it sees a
`position_ids` tensor that restarts at every doc boundary
(e.g. `[0,1,2,3,0,1,2,0,1,2,3,4]`), `_flash_attention_forward` auto-derives
`cu_seqlens` and dispatches `flash_attn.cute.flash_attn_varlen_func`.

**What you change**
1. Dataset emits `position_ids` alongside `input_ids` (counter that resets
   after every EOS).
2. Training loop passes `position_ids=...` into
   `model(input_ids, position_ids=...)`.

**Caveat** HF's auto-detector `_is_packed_sequence` requires
`batch_size == 1`. To batch-pack you have to flatten `(B, S)` → `(1, B*S)`
before forward. That works, but it means your batch dim and seq dim are
fused together, which complicates anything that wants per-sample stats.

**Pros** Smallest change. Stays on FA4 + Liger. No model surgery, no
TorchTitan native rewrite.
**Cons** B=1 flatten trick. Brittle if HF changes the position_ids
detection heuristic.

### Option 2 — HF model with explicit `cu_seq_lens_q/k` kwargs (CHOSEN)

Same as Option 1 but you build `cu_seq_lens_q`, `cu_seq_lens_k`,
`max_length_q`, `max_length_k` yourself and pass them as kwargs into
`model(input_ids, cu_seq_lens_q=..., cu_seq_lens_k=..., max_length_q=..., max_length_k=...)`.
HF threads them through to `_flash_attention_forward` via the
`FlashAttentionKwargs` mechanism — no B=1 restriction, no position-ids
inference.

The four "things" — `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`,
`max_length_k` — are **tensors/scalars you compute per batch**, not
functions. The actual kernel that consumes them is shipped in the
`flash-attn-4` wheel. You don't write CUDA, you don't compile anything.
You construct four small tensors per training step and pass them in.

**Pros** Cleanest for batched packing. No flatten trick. No B=1
restriction. Reuses an existing helper to build the cu_seqlens.
**Cons** Slightly fiddly to make sure the kwargs reach the attention
layer on every transformers version (the kwarg-routing has churned).

Sub-options A/B/C below describe **how** you produce those four values.

### Option 3 — TorchTitan native Llama3 with `attn_type="varlen"`

Drop HF Llama entirely. Use `torchtitan.models.llama3.Transformer`
(TorchTitan's own implementation) and set `attn_type="varlen"` on the
model args. TorchTitan internally calls
`create_varlen_metadata_for_document(input_batch, eos_id)` per forward
and dispatches to `torch.nn.attention.varlen.varlen_attn` (PyTorch's
native varlen, compiled with
`torch.compile(mode="max-autotune-no-cudagraphs")`).

**What you change** Swap `LlamaForCausalLM` for
`torchtitan.models.llama3.Transformer`, mirror config fields, remove
`_patch_initialize_weights_compat` (TorchTitan native doesn't need it).
Liger no-ops because Liger only patches the HF Llama modules — you'd
lose the Liger kernels and would need TorchTitan's own fused norms /
RoPE (it has them).

**Pros** Native TorchTitan path = no HF/torchtitan skew. Real document
masking out of the box. Compile-friendly. Plays well with FSDP2 / CP / PP.
**Cons** Uses PyTorch `varlen_attn`, **not** FA4 — at seq=2048 the gap
is small (single-digit %), at seq=8K+ FA4 wins more. Lose Liger; gain
TorchTitan-native fused kernels (similar territory).

### Option 4 — TorchTitan native Llama3 with `attn_type="flex"`

Same as Option 3 but FlexAttention + a `block_causal_doc` mask (causal
AND same-document). Easiest to extend (sliding window, prefix-LM, etc.
— just edit the mask_mod function).

**Pros** Most flexible. Same correctness as varlen.
**Cons** Slowest of the four kernel paths at our shapes — mask
construction + indirection cost. ~20–30% slower than dedicated varlen
kernels at long seq.

### Option 5 — Monkey-patch `LlamaAttention.forward`

Stay on HF Llama, but at startup overwrite
`transformers.models.llama.modeling_llama.LlamaAttention.forward` with
a custom version that flattens q/k/v and calls
`flash_attn.cute.flash_attn_varlen_func` directly. Keep Liger. Bring
your own cu_seqlens via a thread-local set per batch.

**Pros** Total control, FA4 + Liger together, no HF kwarg-routing
surprises.
**Cons** You own the patch — have to update it whenever transformers
changes the `LlamaAttention` signature (which they do, frequently).

---

## Kernel-level comparison (single H100)

Rough numbers; based on FA4 paper + TorchTitan microbench numbers,
not re-measured on this exact stack.

| kernel                                            | varlen? | rel speed at seq 2K | rel speed at seq 8K | code surface |
|---------------------------------------------------|---------|---------------------|---------------------|--------------|
| `flash_attn.cute.flash_attn_varlen_func` (FA4)    | yes     | **1.00× (fastest)** | **1.00×**           | small        |
| `torch.nn.attention.varlen.varlen_attn`           | yes     | ~0.92–0.97×         | ~0.85×              | small        |
| FlexAttention + block-causal-doc mask             | yes     | ~0.80–0.90×         | ~0.70×              | tiny         |
| HF SDPA mem-eff (current path, no doc masking)    | no      | ~0.95×              | ~0.88×              | none         |

Options 1, 2, 5 all use FA4 varlen → top row.
Option 3 → second row.
Option 4 → third row.

---

## Throughput projections (does packing actually save time?)

For a transformer step, FLOPs split roughly into:

- **Attention QKV/out projections + MLP**: scale as `S × D²`. Linear in S.
- **Attention itself (softmax(QK^T)V)**: scales as `S² × D`. Quadratic in S.
- **LM head**: `S × D × V` (V=128256 — huge).

Packing only reduces the **`S² × D`** term, by replacing one big causal
triangle with many small ones (sum of doc-lengths squared instead of S²).
The other terms are untouched.

For the 500M Llama (D=1536, V=128256, 12 layers, FineWeb avg doc ≈ 500–1000
tokens):

| seq_len | attn-S²-term as % of step | max packing speedup |
|--------:|---------------------------:|--------------------:|
| 2048    | ~2–4%                     | ~1–3%               |
| 4096    | ~5–8%                     | ~3–7%               |
| 8192    | ~12–18%                   | ~8–15%              |
| 16384   | ~25–35%                   | ~15–25%             |

Concrete wall-clock projections (assuming 1×H100, FA4 throughput from the
benchmarks in `FA4.md`):

- **500M × 500M tokens, seq=2048:** 83 min → packing saves 1–3 min. Not material.
- **1.5B × 5B tokens, seq=2048:** ≈46 hours → packing saves 25–80 min.
- **1.5B × 5B tokens, seq=4096:** ≈50 hours → packing saves 1.5–3.5 hours.
- **1.5B × 5B tokens, seq=8192:** ≈60 hours → packing saves **5–9 hours**.

There's also a **quality** angle that doesn't show up in the table:
proper doc-masking eliminates cross-doc attention noise → empirically
1–5% perplexity improvement at fixed token budget (T5, Pythia, Granite
ablations). That's "you converge with 1–5% fewer tokens" — equivalent to
saving 1–5% of training time as a sample-efficiency win. For 1.5B × 5B
tokens that's another 30–150 min on top of the kernel savings.

**Bottom line for the 1.5B / 5B-token plan:** packing is a clear win at
seq=4096+, marginal at seq=2048.

---

## DECISION: Option 2, sub-option A

We are using **Option 2 (HF model with explicit `cu_seq_lens_q/k` kwargs)**.
Within Option 2, three sub-options exist for *how* to produce those four
values; we are using **sub-option A (TorchTitan's helper)**.

### Why Option 2

- Keeps FA4 (fastest kernel at our shapes).
- Keeps Liger (rope + swiglu + rms_norm patches stay).
- Keeps the HF Llama model — no native-TorchTitan rewrite, no losing
  the existing `_patch_initialize_weights_compat` workaround.
- No B=1 flatten gymnastics (which is the only annoying bit about Option 1).
- Dataset change is small and isolated; no model surgery.

### Why sub-option A (TorchTitan helper)

It's already written, tested, and in the same package we depend on. Zero
new code for the hard part (cu_seqlens construction).

### How sub-options A / B / C produce the four tensors

These three are **interchangeable** — they all yield the same four
tensors and the same FA4 varlen call. They differ only in *who computes
the cu_seqlens*.

#### A. Use TorchTitan's helper — CHOSEN

```python
from torchtitan.models.attention import create_varlen_metadata_for_document

meta = create_varlen_metadata_for_document(input_ids, eos_id=tokenizer.eos_id)
# meta is a VarlenMetadata dataclass:
#   meta.cu_seq_q  -> int32 tensor, doc offsets
#   meta.cu_seq_k  -> same as cu_seq_q for self-attention
#   meta.max_q     -> int, longest doc in batch
#   meta.max_k     -> int, longest doc in batch
```

Source: `/usr/local/lib/python3.12/dist-packages/torchtitan/models/attention.py`
(function `create_varlen_metadata_for_document`, around line 307).

You take `meta.cu_seq_q`, `meta.cu_seq_k`, `meta.max_q`, `meta.max_k`
and pass them as kwargs to `model.forward(...)`. **No cu_seqlens code
written by us.**

#### B. Use HF transformers' helper

```python
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

(cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = \
    prepare_fa_kwargs_from_position_ids(position_ids)
```

Same outputs as A, but takes a `position_ids` tensor (the doc-restart
counter) instead of `input_ids` + `eos_id`. Also already shipped — just
import and call.

Source: `/usr/local/lib/python3.12/dist-packages/transformers/modeling_flash_attention_utils.py`
line 437.

Trade-off vs A: requires you to build `position_ids` first
(see Option 1's dataset change), which is a small extra step.

#### C. Don't even call a helper — let HF auto-derive

If you give HF a `position_ids` tensor that restarts at doc boundaries
*and* `batch_size=1` (i.e. flatten `(B, S)` → `(1, B*S)`), HF
transformers detects the packed layout inside `_flash_attention_forward`
and calls `prepare_fa_kwargs_from_position_ids` itself. Your training
loop becomes:

```python
outputs = model(input_ids=flat_ids, position_ids=flat_pos)
```

No cu_seqlens kwargs in your code at all.

Trade-off: this is exactly Option 1 — same B=1 flatten constraint.
We're not using it because we picked Option 2 (kwarg path) specifically
to avoid that flatten.

### Summary table — what gets written

| What                          | Who writes it        | Lines |
|-------------------------------|----------------------|-------|
| FA4 varlen kernel             | already shipped      | 0     |
| `cu_seqlens` builder          | TorchTitan (sub-A)   | 0     |
| FA4 dispatch routing          | transformers         | 0     |
| Dataset emits doc-aware batch | **us**               | ~10   |
| Training loop passes kwargs   | **us**               | ~5    |

Total work on our side: ~15 lines, no kernel code, no attention math,
no cu_seqlens logic.

---

## Implementation sketch (Option 2 + sub-option A)

This is the diff to land. Not yet committed — just the plan.

### 1. Dataset (`PackedFineWebDataset.__iter__`)

Currently yields `{"input_ids": ...}`. Keep that. The varlen metadata
will be computed in the training step, not in the dataset, because
`create_varlen_metadata_for_document` operates on the batched tensor
(it expects shape `(B, S)`), not per-sample.

So the dataset is **unchanged**. The `eos_id` is already inserted between
documents during packing.

### 2. Training step (`forward_loss` in `baseline_llama_torchtitan.py`)

Add the metadata build + pass cu_seqlens kwargs into the model:

```python
from torchtitan.models.attention import create_varlen_metadata_for_document

def forward_loss(model, loss_fn, input_ids, device, eos_id):
    input_ids = input_ids.to(device, non_blocking=True)

    meta = create_varlen_metadata_for_document(input_ids, eos_id=eos_id)

    outputs = model(
        input_ids,
        cu_seq_lens_q=meta.cu_seq_q,
        cu_seq_lens_k=meta.cu_seq_k,
        max_length_q=meta.max_q,
        max_length_k=meta.max_k,
    )
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    local_tokens = shift_labels.numel()
    local_tokens_tensor = torch.tensor(local_tokens, device=device, dtype=torch.float32)
    loss = loss_fn(shift_logits, shift_labels, local_tokens_tensor)
    return loss, local_tokens
```

The eos_id needs to be threaded through from the tokenizer. In the
caller, pass `eos_id=tokenizer.eos_id` (TorchTitan tokenizer) or
`tokenizer.eos_token_id` (HF tokenizer), depending on which one is in
scope.

### 3. Verify FA4 varlen actually fires

Set `attn_implementation` is already `flash_attention_4`. Sanity check
in the first training step:

- Forward should still run end-to-end.
- Loss curve should match (or slightly improve over) the no-packing
  curve — same starting loss (~12.08 from random init), descent should
  be at least as fast.
- If it errors with "received unexpected kwarg `cu_seq_lens_q`", that
  means the HF kwarg-routing on our transformers version doesn't accept
  these names — fall back to passing them via the
  `FlashAttentionKwargs`-style dict or use sub-option B
  (`prepare_fa_kwargs_from_position_ids` + `position_ids`).

---

## What this does NOT change

- **PackedFineWebDataset's concat-pack semantics**: still concatenating
  documents end-to-end with EOS between them, slicing into fixed
  `seq_len` chunks. The packing layout is identical; only the attention
  mask is now doc-aware.
- **Liger kernels**: still on (`rope`, `swiglu`, `rms_norm`). Liger
  patches RMSNorm/RoPE/SwiGLU modules, not attention dispatch — orthogonal.
- **Loss computation**: still TorchTitan's `cross_entropy_loss` outside
  the model, with `local_tokens_tensor` normalization.
- **Optimizer / scheduler / checkpoint**: untouched.

---

## What to watch for

- **Token efficiency check**: with proper masking, train loss at the
  same step count should be slightly lower (cleaner gradient signal).
  Run a short A/B at step 200 on the same seed to confirm.
- **Throughput delta at seq=2048**: expect ~1–3% faster step time.
  If it's slower, something's wrong (likely cu_seqlens being rebuilt on
  CPU; should be device-resident).
- **Throughput delta at seq=4096+**: expect 3–7% (seq=4096) up to
  8–15% (seq=8192). This is where the engineering pays for itself.
- **Eval path**: same `cu_seqlens` plumbing needs to be added in
  `evaluate()` — currently it just calls `forward_loss` so threading
  the eos_id through is the only required change.
