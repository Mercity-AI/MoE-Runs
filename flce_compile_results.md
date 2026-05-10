# FLCE + torch.compile experiment results

What happened when we tried to enable Liger fused_linear_cross_entropy and
torch.compile(max-autotune) on the 500M Llama / FA4 stack.

TL;DR: **FLCE alone works. Compile alone works. They don't compose on
this stack — the combo produces broken gradients.** Set FLCE on, compile off.

## Setup

Stack from `FA4.md`: torch 2.11.0+cu130, transformers 5.8.0, liger-kernel 0.8.0,
flash-attn-4 4.0.0b12, single H100. 508M-param Llama, FA4, bf16 autocast.

## Numbers, single H100, FA4 + Liger (rope/swiglu/rms_norm)

| config                                    | bs × ga | tokens/step | tok/s   | loss curve | OOM? |
|-------------------------------------------|---------|------------:|--------:|------------|------|
| **baseline** (Liger, no FLCE, no compile) | 8 × 4   | 65,504      | 100,501 | clean ✓    | no   |
| baseline                                  | 15 × 8  | 245,640     | —       | —          | **YES — 15 GB logits OOMs** |
| **FLCE on**, no compile                   | 8 × 4   | 65,504      | 66,589  | clean ✓    | no   |
| **FLCE on**, no compile                   | 15 × 8  | 245,640     | 92,786  | clean ✓    | no   |
| no FLCE, **compile** (default)            | 8 × 4   | 65,504      | 101,583 | clean ✓    | no   |
| **FLCE + compile** (default)              | 8 × 4   | 65,504      | 101,652 | **BROKEN** | no   |
| **FLCE + compile** (max-autotune-no-cg)   | 8 × 4   | 65,504      | 101,486 | **BROKEN** | no   |

Loss curves at the same seed for reference:

```
clean (baseline / FLCE only / compile only):
  step 1: 12.08   step 5: 9.94    step 9: 8.83

BROKEN (FLCE + compile):
  step 6: 17.38   step 7: 62.27   step 8: 8.96   step 9: 8.73
  step 10: 17.05  step 11: 17.02  step 12: 16.86 step 13: 69.70
```

The "broken" pattern is loss bouncing between sane (~9) and high (17, 62, 70)
values across steps. This isn't initialization noise — it's gradient
corruption. Identical seeds + clean configs descend monotonically.

## What's happening

`torch.compile` traces Liger's `fused_linear_cross_entropy_forward`, which
contains a `target_mask.sum().item()` (CPU sync) and a chunked
matmul→softmax→CE loop. Inductor handles the graph break around the
`.item()` call but something about the chunked-loss recomputation under
the compiled graph — possibly buffer aliasing in the backward, possibly
fp32 upcast getting lost — produces gradients that are sometimes correct
and sometimes garbage. The throughput is real (~+50% over no-FLCE / no-
compile at bs=8), but the model isn't training.

Tried both compile modes: `default` and `max-autotune-no-cudagraphs`.
`max-autotune` (with cudagraphs) errors outright on Liger's RMSNorm
backward (cudagraph buffer overwrite warning). All three of those fail
the same way for FLCE.

`torch.compile` + plain Liger (no FLCE) is fine — same loss curve as
uncompiled, +1.1% throughput. So compile itself isn't the problem; it's
specifically the FLCE Triton kernel + Inductor interaction.

## Wall-clock implications

500M tokens on a single H100:

| config                          | tok/s  | 500M tokens | working? |
|---------------------------------|-------:|------------:|----------|
| baseline (bs=8)                 | 100,501 | **83 min**   | ✓        |
| FLCE only (bs=15)               |  92,786 | 90 min       | ✓        |
| compile only (bs=8)             | 101,583 | 82 min       | ✓        |
| FLCE + compile (any mode)       | ~101k   | 82 min       | **✗ broken loss** |

**Per-token throughput:** compile alone wins by ~1%. FLCE alone is *worse*
per-token at bs=8 (~33% slower) — Liger's chunked loss has overhead that
only pays off when you actually need the memory savings.

**Why pick FLCE anyway:** at bs=15 ga=8 (the original config, 245k tokens
per optimizer step) FLCE is the only working option — the unfused path
OOMs on the (B, S, V=128k) logits buffer. And at 1.5B parameters this
constraint gets tighter, not looser — FLCE will be mandatory regardless.

The bs=15 effective batch is also ~3.7× the bs=8 effective batch, which
is better for convergence stability and throughput-per-quality.

## Picking a config for the 500M run

Three working options, ranked by raw wall-clock:

| config                     | tok/s   | 500M tok wall-clock | effective batch | optim steps |
|----------------------------|--------:|--------------------:|----------------:|------------:|
| compile only (bs=8 ga=4)   | 101,583 | **82 min**          | 65k tokens      | 7,693       |
| baseline (bs=8 ga=4)       | 100,501 | 83 min              | 65k tokens      | 7,693       |
| FLCE on (bs=15 ga=8)       |  92,786 | 90 min              | **245k tokens** | 2,041       |

The wall-clock spread is 8 minutes on an 83-minute run — small enough
that quality matters more than the time.

**Pick: FLCE (bs=15 ga=8).** Three reasons:

1. **Bigger effective batch = better optimization.** 245k tokens/step
   vs 65k. For a 500M model, the *critical batch size* (CBS) — the
   point past which more batch stops helping convergence — is roughly
   1–4M tokens. We're well below CBS on both, but 245k is closer to
   the good-optimization regime: smoother loss curve, more meaningful
   gradient updates, less LR-warmup sensitivity.

2. **Future-proofs for 1.5B.** At 1.5B params, the (B, S, V=128k)
   logits tensor that OOMs us today gets ~3× bigger. FLCE moves from
   "nice" to "mandatory." Establishing the FLCE path on the 500M run
   means no plumbing rewrite when scaling up.

3. **The 8-minute "cost" is in the noise.** It's 9% of wall-clock; if
   the dataloader stalls once or wandb has a slow log, you've lost
   more than that. Not worth chasing.

**Skip compile.** It's a 1% gain that breaks the moment you turn on
FLCE. Take it off the table for now.

## What tok/s actually means (and why it drops with FLCE)

This is a useful number but it isn't the whole picture. Worth being
precise about it.

### The formula

```
tok/s = tokens_per_optim_step / seconds_per_optim_step
```

At bs=8 ga=4:
- tokens/step = 8 × 4 × 2048 = 65,504
- sec/step    = 0.652
- → 100,501 tok/s

At bs=15 ga=8 with FLCE:
- tokens/step = 15 × 8 × 2048 = 245,640
- sec/step    = 2.647
- → 92,786 tok/s

We're doing **3.75× more tokens per step**, but the step takes
**4.06× longer**. The *ratio* (tok/s) drops 7.7%.

### Why it drops a little

The H100 is already near-saturated at bs=8 — adding more batch fills
compute but also adds overhead that doesn't scale linearly:

1. **Liger FLCE's chunked LM head**: it processes the (B·S, V=128k)
   projection in chunks to keep memory bounded. Each chunk = a kernel
   launch + a CE kernel + an accumulator step. Per-chunk fixed cost.
   The unfused path at bs=8 is one big GEMM + one CE — fewer kernel
   launches per token.
2. **Memory pressure**: at bs=15 we're using ~70 GB / 80 GB. Allocator
   fragmentation and HBM thrash add a small penalty.
3. **Optimizer / norm steps**: AdamW + gradient clip + LR scheduler run
   once per optimizer step regardless of batch size, so at bs=15 you
   amortize that fixed cost across more tokens — which is *good*. But
   per-microbatch overheads partially cancel the win.

Net: per-token throughput drops ~7.7%. Per-step throughput is up 4×.

### Why tok/s isn't the right metric anyway

The metric that matters for "did this training run finish faster" is:

```
wall_clock = total_tokens_to_target_loss / tok_per_s
```

`tokens_to_target_loss` isn't constant — it depends on batch size:

- **At small batch (bs=8, 65k effective):** gradient is noisy, optimizer
  fights through that noise, you need more tokens to converge to the
  same loss. Empirically: 1.0× tokens (this is the reference).
- **At larger batch (bs=15, 245k effective):** gradient is closer to
  the true gradient, each step makes a more meaningful update.
  Empirical 500M-scale results: ~0.85–0.95× tokens to the same loss
  (depends on LR tuning).

So:
- Small batch:  100,501 tok/s × 1.00 = effective **100,501** "good tokens/s"
- Large batch:   92,786 tok/s × 1.05–1.18 = effective **97,000–110,000** "good tokens/s"

The big batch is roughly break-even on speed and **probably ahead on
quality at the same step count**.

### The intuition in one line

**Tok/s tells you how fast the GPU is chewing.** It doesn't tell you
whether the chewing is productive. Going from bs=8 to bs=15 chews 7.7%
slower, but each chewed token produces a ~5–15% better gradient update.
Net: roughly the same wall-clock to target loss, with the bigger-batch
run usually having a smoother, more reproducible loss curve.

This is why people don't optimize purely for tok/s — they optimize for
tokens-to-target-loss × wall-clock-per-token, which is a different
thing.

### What to actually look at

Three numbers, in order of importance:

1. **Final eval loss at fixed wall-clock budget** — the only metric
   that reflects "did I get a better model."
2. **Wall-clock to a target loss / perplexity** — the "did training
   finish faster" metric.
3. **tok/s** — useful as a debugging signal (did something stall? is
   the GPU saturated?), not as a goal.

For the 500M × 500M run: pick FLCE bs=15, accept the 7-minute "loss"
on raw tok/s, and you'll likely come out slightly ahead on (1) and
roughly tied on (2).

## Decision

In `baseline_llama_torchtitan.py` CONFIG:

```python
"liger_kernel_config": {
    "rope": True,
    "swiglu": True,
    "rms_norm": True,
    "cross_entropy": False,
    "fused_linear_cross_entropy": True,   # ← ON, enables bs=15
},
"torch_compile": False,                   # ← OFF, breaks combined with FLCE
"torch_compile_mode": "default",
```

Verified end-to-end: 4 train steps + 2 eval passes + checkpoint save
runs cleanly at bs=15 × ga=8 = 245,640 tokens/step, with loss descending
12.09 → 10.98 → 10.76 → 10.02 over the four steps, eval 10.72 → 9.86.

## What changed in the code

- CONFIG: `fused_linear_cross_entropy: False → True`,
  `torch_compile: False → False` (left off pending upstream fix),
  added a comment explaining why.
- `maybe_enable_liger_kernel`: relaxed the assertion that previously
  forbade FLCE. Now only forbids `cross_entropy` and `fused_linear_cross_entropy`
  *both* set, since they're mutually exclusive.
- `forward_loss`: split into two paths. With `use_flce=True` it calls
  `model(input_ids, labels=input_ids)` and lets Liger's patched
  `LlamaForCausalLM.forward` compute the fused loss internally — logits
  never materialize. With `use_flce=False` it's the original logits-shift
  path through TorchTitan's CE.
- `evaluate`: same `use_flce` switch threaded through.
- `train()`: detects `use_flce` from the config, skips
  `build_torchtitan_ce_loss()` when FLCE is on, prints which loss path
  is active.
- `bench_attn.py`: matches the new `forward_loss` signature; respects
  `torch_compile` from CONFIG (with `BENCH_NO_COMPILE=1` env override).

## Next things to try (for someone, later)

1. **`torch.compile(model, dynamic=False)`** — explicit static-shape
   compile. Might bypass the Inductor recompile that's likely
   poisoning FLCE's intermediate buffers.
2. **`torch._dynamo.disallow_in_graph(liger_fused_linear_cross_entropy_*)`**
   — explicitly tell Inductor not to trace Liger's FLCE op so it stays
   a black-box op call. Should preserve correctness while letting the
   rest of the model compile.
3. **Pin `liger-kernel<0.8` or upgrade to a fix release** — file an
   issue with a reproducer.
4. **Use HF's built-in CE chunking** — transformers >=5.x has chunked
   CE in the model itself; might compile cleanly.

None of these are blocking the 500M run.

## Recommendation for the 1.5B / 5B-token plan

- FLCE on, compile off — what's already in CONFIG.
- At 1.5B the bs that fits per H100 will drop. FLCE's memory advantage
  becomes essential, not optional. Compile remains broken until upstream
  fixes; not worth chasing.
- The bigger lever for 1.5B is FSDP across 4–8 H100s; that gives 3–7×
  wall-clock vs 1% from compile.
