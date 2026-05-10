# FlashAttention-4 + TorchTitan + Liger on H100

End-to-end record of getting `baseline_llama_torchtitan.py` running on a single
H100 with the latest stack: PyTorch 2.11, FlashAttention-4 (CuTeDSL), Liger
kernels, and TorchTitan's HF backend bits. Includes throughput numbers and a
projection for 500M params on 500M tokens.

## TL;DR

- **FlashAttention-4 installs as a pip wheel now, no `nvcc` build needed.** The
  old "wait 30 minutes for `pip install flash-attn` to compile" pain is gone:
  FA4 is shipped via CuTeDSL JIT, install is ~5 seconds.
- The `lesj0610/flash-attn-2.8.3-torch2.10` wheel from the Slack thread is no
  longer relevant. We don't use it; we use FA4.
- Three small edits to the repo are enough — model still 508M params, train +
  eval + checkpoint loop runs cleanly.
- **~83 min for 500M params × 500M tokens on one H100** (FA4 + Liger, seq 2048).

## Final working stack

```
torch              2.11.0+cu130     # default PyPI wheel, ships its own CUDA 13 libs
transformers       5.8.0            # native flash_attention_4 dispatch
torchtitan         0.2.2
liger-kernel       0.8.0
flash-attn-4       4.0.0b12
nvidia-cutlass-dsl 4.5.0            # pulled in by flash-attn-4[cu13]
quack-kernels      0.4.1            # FA4 dependency
Python 3.12, NVIDIA H100 80GB HBM3, driver 580.126.09 / CUDA 13.0
```

## Install (clean machine, exact commands)

This is the only install procedure we found that works end-to-end without
compile errors, version skew, or OOM during build. The classic
`pip install flash-attn` from the Slack thread is **not** what we use.

### Step 0: Prerequisites

```bash
# Confirm GPU is Hopper or Blackwell — FA4 won't run on Ampere (A100, A10) or older
nvidia-smi --query-gpu=name --format=csv,noheader
# expect: NVIDIA H100 80GB HBM3, H200, B100, B200, RTX PRO 6000, etc.

# Confirm CUDA driver version. FA4 needs CUDA >= 12.3 driver
nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader
# CUDA 12.x driver → use the cu12 install path (Step 3 alt)
# CUDA 13.x driver → use the cu13 install path (Step 3 main)
# Compute capability must be >= 9.0 (Hopper) or >= 10.0 (Blackwell)

# Python 3.10/3.11/3.12 — flash-attn-4 ships wheels for these
python --version
```

If nvidia-smi shows an A100 (compute 8.0) or older, **stop** — FA4 will
not work. FA2/FA3 still work on Ampere if you need flash attention there.

### Step 1: Reset PyTorch

RunPod's default image ships with torch 2.8.0+cu128. That version has a
known TorchTitan varlen shape error (the original Slack-thread issue),
and 2.8's wheel doesn't pair with FA4's CuTeDSL JIT cleanly. Wipe it:

```bash
pip uninstall -y torch torchvision torchaudio
```

### Step 2: Install torch 2.11

```bash
# Default PyPI — gives torch-2.11.0+cu130 (CUDA 13 wheels, the version
# we tested). Works if `nvidia-smi` shows CUDA 13.x driver.
pip install torch==2.11.0 torchvision torchaudio

# Alt for CUDA 12.x driver (pin to cu128 wheels, still torch 2.11):
# pip install torch==2.11.0 torchvision torchaudio \
#   --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.11.0+cu130 13.0 True   (or +cu128 12.8 True)
```

The wheel bundles cuDNN, cuBLAS, NCCL, etc. — you do **not** need a system
CUDA toolkit installed. `nvcc --version` doesn't need to match.

### Step 3: Training stack

```bash
pip install torchtitan transformers datasets wandb liger-kernel accelerate \
            trackio tiktoken sentencepiece protobuf tqdm ninja
```

Single command, no version pins — the latest of each works together as
of this writing (transformers 5.8.0, torchtitan 0.2.2, liger-kernel 0.8.0,
datasets 4.8.5).

### Step 4: FlashAttention-4

```bash
# CUDA 13 driver path (matches Step 2 main):
pip install --pre "flash-attn-4[cu13]"

# CUDA 12 driver path (matches Step 2 alt):
# pip install --pre flash-attn-4
```

This is a **pure-Python wheel, ~330 KB**. No `nvcc` invocation. No
`MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST="9.0"`. No 30-minute build.
Total install time: ~5 seconds plus dep download (CUTLASS DSL libs are
~80 MB, biggest component).

The `[cu13]` extra pulls `nvidia-cutlass-dsl-libs-cu13`. Without it, FA4
won't find CUTLASS at runtime and you'll see import errors when calling
the kernel. The `--pre` flag is required because flash-attn-4 is still
on beta releases (4.0.0bN).

### Step 5: Sanity check

```python
import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

B, S, H, D = 2, 1024, 8, 64
q = k = v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
out, _ = flash_attn_func(q, k, v, causal=True)   # returns (output, lse_or_None)
assert out.shape == (B, S, H, D)
print("FA4 OK")
```

First call triggers the CuTeDSL JIT compile (~3–5 seconds). Subsequent
calls reuse the compiled kernel.

### `flash-attn-4` vs `flash-attn` — important

These are two different PyPI packages that both install into the
`flash_attn` Python namespace.

- `flash-attn` (PyPI) → FA2: exposes `flash_attn.flash_attn_func`,
  `flash_attn.flash_attn_varlen_func`, `flash_attn.bert_padding`, etc.
- `flash-attn-4` (PyPI) → FA4: exposes `flash_attn.cute.flash_attn_func`,
  `flash_attn.cute.flash_attn_varlen_func`. **Nothing else.**

If you only install `flash-attn-4`, code that does
`from flash_attn import flash_attn_func` will fail with ImportError —
that's the FA2 entry point, which doesn't exist in the FA4-only package.
For our pretraining run we only need `flash_attn.cute.*`, which transformers
5.x dispatches to via `attn_implementation="flash_attention_4"`. We do not
install `flash-attn`.

If for some reason you need both FA2 and FA4 simultaneously (rare —
inference codebases sometimes do): install `flash-attn-4` first, then
`pip install flash-attn --no-build-isolation` second. The classic FA2
compile from source still takes 20–30 minutes and needs `nvcc` matching
your torch CUDA version.

### Common install errors

| symptom | cause | fix |
|---|---|---|
| `ImportError: cannot import name 'flash_attn_func' from 'flash_attn'` | importing FA2 path with only FA4 installed | use `from flash_attn.cute import flash_attn_func` instead |
| `ModuleNotFoundError: nvidia.cutlass_dsl` at first kernel call | missed the `[cu13]`/`[cu12]` extra | `pip install --pre "flash-attn-4[cu13]"` (or `[cu12]`) |
| `AssertionError: inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2` | passing fp32 tensors to FA4 | wrap call in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` |
| `RuntimeError: cuDNN Frontend error: No valid execution plans built` | torch 2.11 + this image's cuDNN doesn't have an SDPA plan for the shape (only hits the SDPA path, not FA4) | `torch.backends.cuda.enable_cudnn_sdp(False)` falls back to the math/mem-eff backend |
| `pip install flash-attn` hangs for 20+ minutes compiling | you installed `flash-attn` (FA2 source build) instead of `flash-attn-4` | uninstall, install `flash-attn-4[cu13]` |
| `OSError: cannot access gated repo for ... meta-llama/Llama-3.2-1B` | gated HF model without `HF_TOKEN` | swap tokenizer to `unsloth/Llama-3.2-1B` (identical vocab 128256) |
| First training step takes ~10 seconds | normal — FA4 CuTeDSL JIT + Triton autotune for Liger kernels + dataloader prefetch | ignore; steady state hits at step 2 |
| `torch.OutOfMemoryError ... cross_entropy` at bs=15 seq=2048 | (B, S, 128256) fp32 logits = ~15 GB | enable Liger `fused_linear_cross_entropy` — see `flce_compile_results.md` |

### Why not the lesj0610 wheel from the Slack thread

The wheel `lesj0610/flash-attention/.../flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
is FA2 prebuilt for torch 2.10 / CUDA 12. It was a workaround for two
problems we no longer have:

1. *FA2's source build is slow.* Our path uses FA4, which has no source
   build at all (CuTeDSL JIT, pure-Python wheel).
2. *FA2 had no torch 2.10 wheel on PyPI.* Our path uses torch 2.11 +
   FA4, both available as official wheels.

If you're stuck on torch 2.10 and an Ampere GPU, that wheel is still the
fastest path to FA2. For Hopper/Blackwell on torch 2.11+, ignore it —
the FA4 install above is faster and correctly targets the GPU.

## Patches applied to `baseline_llama_torchtitan.py`

Three changes, total diff:

```diff
-    "attn_implementation": "sdpa",
+    "attn_implementation": "flash_attention_4",
-    "tokenizer_name": "meta-llama/Llama-3.2-1B",
+    "tokenizer_name": "unsloth/Llama-3.2-1B",
@@ evaluate()
-    for batch in eval_pbar:
-        loss, token_count = forward_loss(model, loss_fn, batch["input_ids"], device)
+    for batch in eval_pbar:
+        with get_autocast_context(device):
+            loss, token_count = forward_loss(model, loss_fn, batch["input_ids"], device)
```

Why each one:

1. **`flash_attention_4`** — `transformers >= 5.x` exposes
   `attn_implementation="flash_attention_4"` and dispatches to
   `flash_attn.cute.flash_attn_func` automatically (see
   `transformers/modeling_flash_attention_utils.py`, FA4 entry in
   `FLASH_ATTENTION_COMPATIBILITY_MATRIX`). No glue code needed.
2. **`unsloth/Llama-3.2-1B`** — `meta-llama/Llama-3.2-1B` is gated and the
   environment had no HF token. Unsloth's mirror has identical vocab
   (128256, eos=128001), so the LlamaConfig is unchanged.
3. **autocast around eval** — FA4 only accepts `bf16/fp16/fp8`. The training
   loop wraps the forward in `get_autocast_context(...)` (autocast → bf16),
   so it works. `evaluate()` did not, so the first eval call after training
   threw `AssertionError: inputs must be float16, bfloat16, fp8 e4m3fn, or
   fp8 e5m2`. The model itself is held in fp32 (master weights); only the
   compute is bf16, same as during training.

## Things investigated but not needed

- The `lesj0610` FA2 wheel hack — irrelevant once we're on `flash-attn-4`.
- Building from source with `MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=9.0` —
  not needed; FA4 is JIT'd at runtime.
- Torchtitan varlen path with `attn_type="varlen"` — only available on
  TorchTitan's *native* Llama3 model, not on the HF backend the script uses.
  See "Packing" section below.
- The torch 2.8 / TorchTitan varlen shape error from the Slack thread —
  doesn't apply at torch 2.11.

## Verified end-to-end

- `_patch_initialize_weights_compat` in the script handles the
  TorchTitan ↔ transformers `smart_apply` arity mismatch — still needed,
  unchanged.
- Liger patches `rope` + `swiglu` + `rms_norm` for Llama on transformers 5.8.
  `cross_entropy` is off; `fused_linear_cross_entropy` (FLCE) is **on** —
  see `flce_compile_results.md` for that work; FA4.md numbers below were
  taken before FLCE was wired in.
- 4-step run completes: train → eval at step 2 → train → eval at step 4 →
  checkpoint saved.
- FA4 loss numbers match SDPA loss numbers bit-for-bit at every step
  (12.054 → 11.158 → 10.463 → ...), confirming numerical correctness.

## Throughput on a single H100, 508M-param Llama, Liger on, bf16

| seq  | BS × GA | tokens/optim_step | FA4 tok/s    | SDPA tok/s | FA4 speedup |
|-----:|--------:|------------------:|-------------:|-----------:|------------:|
| 2048 | 8 × 4   | 65,504            | **100,501**  | 95,713     | **+5.0%**   |
| 4096 | 4 × 4   | 65,520            | **96,572**   | 88,622     | **+9.0%**   |

Step times (warmup excluded): FA4 ≈ 0.65–0.68 s/optim_step, SDPA ≈ 0.68–0.74 s.
Steady-state reached after step 1 (which includes Triton/CuTe JIT compile and
the first dataloader prefetch — ~10s spike, ignored in the average).

The "SDPA" numbers are with `torch.backends.cuda.enable_cudnn_sdp(False)` —
cuDNN's SDPA frontend errored with `[cudnn_frontend] No valid execution plans
built` on this image's cuDNN/torch combo, so cuDNN is disabled and SDPA falls
back to PyTorch's mem-eff (Flash-style) backend. The math backend would be
much slower; the mem-eff backend is itself a flash-attention-style kernel,
which is why the FA4 speedup over it is only 5–9% rather than 1.5–2×.

The reason the FA4 win is moderate at these shapes:

- head_dim 128, seq 2048–4096, no varlen, dense causal mask — this is
  matmul-bound on H100, not attention-bound. Attention is ~10–15% of the
  step at seq=2048.
- FA4's biggest wins come at long context (8K+), with FP8, or with varlen
  packing. None of those are turned on here.

## How long does 500M params on 500M tokens take?

From the measured 100,501 tok/s steady state:

```
500e6 tokens / 100,501 tok/s ≈ 4,975 s ≈ 82.9 min ≈ 1 h 23 m
```

Single H100, FA4 + Liger, seq=2048, bs=8 × ga=4. Wall-clock excluding the
~10 s first-step JIT compile and any eval/checkpoint stops you turn on.

Other cells in the same matrix:

| config                                    | tok/s   | 500M tokens take |
|-------------------------------------------|--------:|-----------------:|
| FA4 + Liger,  seq 2048                    | 100,501 | **1 h 23 m**     |
| SDPA + Liger, seq 2048                    | 95,713  | 1 h 27 m         |
| FA4 + Liger,  seq 4096                    | 96,572  | 1 h 26 m         |
| SDPA + Liger, seq 4096                    | 88,622  | 1 h 34 m         |

These are single-GPU. On 8×H100 with linear scaling: ~10–11 minutes.

## Time-cost breakdown of the FA4 saving

500M tokens, FA4 vs SDPA at seq=2048: 5,225 s − 4,975 s = ~250 s saved (~4 min,
~5%). The headline FA4 numbers from the paper require either (a) seq 8K+,
(b) FP8, or (c) varlen with dense packing. With varlen + correct
document-boundary masking, FA4 should pull ahead more.

## On packing with TorchTitan

What the script already does: greedy concat-and-slice. Tokenize each FineWeb
doc, append EOS, append into a buffer, slice off `seq_len`-token chunks.
This is what almost every pretraining setup calls "packing." Tokens from
doc A and doc B in the same chunk *do* attend to each other, but in
practice that's fine for pretraining because the EOS provides a strong
"ignore previous" signal.

What TorchTitan provides on top:
- `torchtitan.hf_datasets.text_datasets.HuggingFaceTextDataset` — same
  greedy concat packing, just productized. Drop-in replacement for the
  custom `PackedFineWebDataset` in this script if you want.
- `torchtitan.models.attention.create_varlen_metadata_for_document(input_batch, eos_id)`
  — builds `cu_seq_q`/`cu_seq_k`/`max_q`/`max_k` so each document inside a
  packed sequence is a separate causal block (no cross-doc attention).
- `attn_type="varlen"` on the **native** TorchTitan Llama3 model uses
  `torch.nn.attention.varlen.varlen_attn` with that metadata. Not reachable
  from the HF-backend path the script currently uses.
- `attn_type="flex"` does the same thing via FlexAttention + a block-causal
  block mask.

To get FA4 + true document-boundary packing, the path is:
1. Build cu_seqlens for the batch using EOS positions
   (`create_varlen_metadata_for_document` is the reference impl).
2. Swap the HF attention call for `flash_attn.cute.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)`.
3. Either patch `LlamaAttention.forward` directly, or move off the HF
   modeling code entirely onto TorchTitan's native Llama3 with
   `attn_type="varlen"` (in which case you'd be on PyTorch native varlen,
   not FA4 — the speed difference is small at our shapes).

For now the script's concat-pack is fine; document-boundary masking is a
quality micro-optimization that won't move the throughput needle on a
1-H100 500M-param run.

## Files in this repo touched

- `baseline_llama_torchtitan.py` — 3 small edits described above (FA4
  attn impl, tokenizer swap, autocast around eval) plus the FLCE refactor
  documented in `flce_compile_results.md`.
- `bench_attn.py` — new throughput harness (FA4 vs SDPA at configurable
  seq/BS, FLCE-aware).
- `FA4.md` — this document.
- `flce_compile_results.md` — FLCE + torch.compile experiment results,
  with the chosen config (FLCE on, compile off, bs=15).
- `packing.md` — document-boundary varlen packing options 1–5, decision
  to use Option 2 sub-A (TorchTitan's `create_varlen_metadata_for_document`
  helper).

## Reproducing the numbers

```bash
# Throughput, seq=2048, bs=8 ga=4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python bench_attn.py --attn flash_attention_4 --warmup 3 --steps 6 --bs 8 --ga 4

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python bench_attn.py --attn sdpa --warmup 3 --steps 6 --bs 8 --ga 4

# Throughput, seq=4096, bs=4 ga=4
BENCH_SEQ_LEN=4096 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python bench_attn.py --attn flash_attention_4 --warmup 3 --steps 6 --bs 4 --ga 4

# Full pipeline (4 steps + eval + checkpoint)
WANDB_MODE=disabled TRACKIO_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_llama_torchtitan.py
```

## Known caveats / gotchas hit

- `meta-llama/Llama-3.2-1B` is gated; without `HF_TOKEN` set, swap in
  `unsloth/Llama-3.2-1B` (identical tokenizer, vocab 128256).
- `attn_implementation="flash_attention_4"` requires bf16/fp16/fp8. Anywhere
  the model runs in fp32 (eval, debug paths) will assert. Wrap in
  `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.
- `cudnn` SDPA backend can fail to find a plan; disable with
  `torch.backends.cuda.enable_cudnn_sdp(False)` and the math/mem-eff
  backend takes over.
- `bs=15 × seq=2048` from the original CONFIG OOMs on a single H100 80GB
  during cross-entropy (the 15 × 2047 × 128256 fp32 logits buffer is
  ~15 GB). The fix is Liger `fused_linear_cross_entropy` — now enabled in
  CONFIG. See `flce_compile_results.md` for that work.
- TorchTitan's HF backend (`HFTransformerModel`) is not used directly —
  the script's `build_direct_hf_model` builds a plain `LlamaForCausalLM`
  to dodge two TorchTitan ↔ transformers skews (`update_from_config`
  ignoring the saved config; `smart_apply` arity mismatch). Both worked
  around in-place; don't "fix" by switching to `HFTransformerModel`.
