# Correction Report — LongCat N-gram 1B Run

**Date:** 2026-08-04
**Branch:** `claude/fa4-work`
**Scope:** Restored the LongCat n-gram training stack to the configuration that
produced the documented July 27 results, and patched a weight-decay bug present
in all versions.

---

## 1. Background — why this correction was needed

The training scripts on `claude/fa4-work` had **silently diverged** from the code
that produced the results documented in the Notion report *"Scaling NGram
embeddings down to 1B."* The divergence was not visible in any metric — the
diverged scripts run fine and train to convergence — but they encode a different
(and partly broken) architecture than the one the report describes.

The authoritative July code was recovered from `scripts (12).zip` (dated
2026-07-27). Comparing it against the branch revealed that:

- `model.py` was **byte-for-byte identical** in both — the model code was never
  the problem.
- The **n-gram training script** on the branch carried a drifted config: 15
  layers (vs 16), 4 KV heads (vs 6), and uniformized table sizes
  `[322336]*3 + [322335]*3` (vs six distinct sizes). Two of the three "bugs"
  below were consequences of that drifted config alone.
- The **baseline training script** carried an independent config drift: 4 KV
  heads instead of 6.
- `utils.py` contained a **genuine bug** (weight decay on the embedding tables)
  that was present in *both* the branch and the July zip.

The source paper referenced throughout is **arXiv:2601.21204v1, "Scaling
Embeddings Outperforms Scaling Experts in Language Models"** (Meituan LongCat
Team). Section numbers below refer to that paper.

---

## 2. The three bugs

### Bug 1 — The two hash heads were identical clones (2 of 3 orders)

**Category:** silent semantic bug — the model trains normally but is
representationally a *different, weaker* architecture than intended. No metric
flags it.

**What it was.** The n-gram embedder builds `K=2` hash tables per order and sums
their looked-up embeddings, the paper's collision-mitigation mechanism (§3.2.3:
"the number of sub-tables K ... governs the number of distinct hash functions
applied to each n-gram, thereby substantially mitigating the probability of hash
collisions"). The hash function (`model.py:_hash_ngram`) distinguishes one table
from another **only by its modulus (`table_size`)** — the head index `k` never
enters the hash:

```python
h = (h * self.base_vocab_size + tok) % table_size   # only n and table_size matter
```

The tables are assigned in loop order (`model.py:65`) from the flat size list.
The drifted config was `[322336]*3 + [322335]*3`, which assigns:

- `n2 → (322336, 322336)`  — identical size
- `n3 → (322336, 322335)`  — the only decorrelated pair
- `n4 → (322335, 322335)`  — identical size

Same order + same `table_size` + no per-head term ⇒ **head 0 and head 1 compute
identical indices, bit-for-bit**, for orders 2 and 4. Algebraically this collapses
`K=2` to `K=1` with a wider table for two of three orders — and the paper found
`K=1` "notably inferior" (§3.2.3). Measured on synthetic Zipf data, ~39% of
distinct 2-grams lost unique identity under the clone config vs ~0% under distinct
sizes.

**Why it was easy to introduce.** Three individually-reasonable choices combined
into it: (1) implementing the hash with `table_size` as the only per-table knob —
a faithful reading of the paper's Eq. 2, which shows no per-head term; (2)
flattening sizes into one list consumed by loop order — ordinary config style;
(3) uniformizing the sizes to keep the parameter budget tidy — the moment
head-diversity silently evaporated, because size was the *only* thing separating
the heads. Nobody wrote "make the heads identical."

**Fix.** Resolved entirely by restoring the July config's six **pairwise-distinct**
sizes `[267003, 367007, 305011, 341013, 345017, 308962]`. Each `K=2` pair now has
a different modulus, so every pair decorrelates. No code change was required —
`model.py` was already correct.

**Residual (not applied).** The hash still has no per-head salt, so head diversity
depends on distinct sizes. This is fine for this run. If tables are ever equalized
again, a per-head salt (distinct multiplier/seed per `k`) plus a build-time assert
(`no two heads may share (hash_fn, table_size)`) would convert this class of bug
from silent-wrong into a startup crash. Deferred deliberately — salting would
change table semantics and break comparability with the July run.

---

### Bug 2 — Table vocab sizes sat on a near-integer multiple of base vocab

**Category:** design-principle violation (config), tied to Bug 1's root cause.

**What it was.** The paper's §3.2.2 boxed design principle:

> "The vocabulary size of N-gram Embedding should significantly deviate from
> integer multiples of the base vocabulary size to prevent Hash collisions."

Collision counts spike near integer multiples because the modulo indexing
degenerates (with `V = m·V0`, the hash discards all but `t mod m` of the context).
The drifted size `322336 = 10.073 × 32000` sat only **0.073** off the `10×`
multiple — right beside the cliff. The giveaway that this was a refactor accident:
`3×322336 + 3×322335 = 1,934,013`, the *exact* sum of the six distinct sizes in
the July config. Someone preserved the total table budget but uniformized the
sizes, and the whole budget landed on a near-multiple.

**Fix.** Resolved by the same config restore. The July sizes correspond to
8.34× / 11.47× / 9.53× / 10.66× / 10.78× / 9.65× the 32k base vocab — all ≥0.22
off the nearest integer multiple, compliant with §3.2.2.

**Caveat.** The paper measured collision magnitudes at a **128k** base vocab with
~30× table sizes. This run is at **32k** base with ~8–11× sizes, so the *principle*
(avoid integer multiples) transfers but the exact magnitudes were not measured in
this regime.

---

### Bug 3 — Weight decay was eroding the n-gram tables

**Category:** genuine bug, present in **both** the branch and the July zip. A mild,
consistent drag (not run-invalidating), but pointed against the paper's own design.

**What it was.** The optimizer's no-decay exemption (`utils.py:_is_no_decay`)
exempts parameters by **shape and name** — 1-D tensors, `.bias`, and names
containing `"norm"`:

```python
return param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
```

An `nn.Embedding` weight is a **2-D matrix** (`[vocab_size, sub_dim]`), named
`ngram_embedder.tables.n2_k0.weight` — it matches none of those conditions, so it
fell through into the `weight_decay = 0.1` group. **Every n-gram table was being
weight-decayed at 0.1.** The exemption list was written for the dense model
(norms, biases); nobody added a clause for the embedding tables, and their 2-D
shape means shape-based filters can't catch them.

**Why decay on these tables specifically is harmful.** AdamW's decay is
*decoupled* — `θ ← θ·(1 − lr·wd)` is applied to every parameter **every step**,
regardless of whether it received a gradient. A dense weight gets a gradient every
step, so decay is a balanced regularizer. An embedding-table row only gets a
gradient on the steps where a token's n-gram hashes to it; on every other step its
gradient is exactly zero, but decay still fires. So **decay is dense in time while
the table's learning signal is sparse in time**, and the imbalance is worst for
the **rare high-order (3-/4-gram) rows** — which are exactly the rows the paper
identifies as both the hardest to learn (§3.2.3: high-order n-grams "appear
infrequently ... this sparsity significantly exacerbates the challenge of learning
effective embeddings") and the reason the tables exist. It also runs directly
against §3.2.4 Embedding Amplification, whose entire purpose is to *protect and
boost* the embedding signal over training.

**Magnitude.** Peak `lr 3e-4 × wd 0.1 = 3e-5`/step; with cosine averaging over
3,053 steps a never-hit row shrinks ≈5%. Modest here, not run-invalidating, and it
affected both arms equally so the July comparison stays fair. It compounds badly at
larger token budgets (~16% at 20B tokens).

**Note on paper support.** The word "decay" appears **zero times** in the paper —
it neither prescribes nor forbids exempting the tables. The fix is therefore
**engineering judgment consistent with the paper's sparsity/amplification
reasoning** and with universal LLM practice (embeddings/norms/biases are
conventionally excluded from weight decay — which is literally what `_is_no_decay`
already does for every 1-D parameter). It is *not* backed by a direct on/off
ablation. Unlike Bugs 1–2, this one does not carry a paper citation.

**Fix (applied).** Added a name-based clause to `_is_no_decay` (`utils.py:445`):

```python
def _is_no_decay(name, param):
    return (
        param.ndim == 1
        or name.endswith(".bias")
        or "norm" in name.lower()
        or ".tables." in name        # n-gram hash tables — sparse-gradient embeddings
    )
```

Keyed on the **name** (`.tables.`), not shape, because embeddings are 2-D and
structurally indistinguishable from a dense linear. Verified classification:

| Parameter | Group |
|---|---|
| `ngram_embedder.tables.*.weight` | **no-decay (0.0)** ← fixed |
| `ngram_embedder.projections.*.weight` | decay (0.1) — intended, real dense linear |
| `model.embed_tokens.weight` | decay (0.1) — left deliberately (tied to LM head) |
| RMSNorm / QK-norm / biases | no-decay (0.0) |

`projections` stays decayed (ordinary dense linear, full gradients). `embed_tokens`
is left decayed on purpose — it is tied to the LM head, and decaying the
unembedding is common practice.

---

## 3. Config drift corrected (provenance)

Restored to match the Notion report (the July 27 run):

| Field | Drifted (branch) | Restored (July / correct) |
|---|---|---|
| `num_hidden_layers` (ngram) | 15 | **16** |
| `num_key_value_heads` (ngram) | 4 | **6** |
| `num_key_value_heads` (baseline) | 4 | **6** |
| `max_steps` (ngram) | 3052 | **3053** |
| `ngram_table_vocab_sizes` | `[322336]*3 + [322335]*3` | `[267003, 367007, 305011, 341013, 345017, 308962]` |
| `warmup_steps` | 150 | **150** (Notion table's "100" is a stale cell; 150 confirmed correct) |

**Baseline KV=6 note.** The baseline script carried `num_key_value_heads: 4` in
*both* the branch and the July zip — a stale value that never reflected the real
run (confirmed by the run owner to be 6). Correcting it to 6 tightens parameter
parity between arms and removes the KV-head confound.

---

## 4. Verified clean (unchanged, confirmed correct)

LayerNorm embedding amplification (§3.2.4); Horner-method hash with per-step
modulus, overflow-safe (§3.2.3 footnote); blake2b 0.5% document holdout applied
consistently to train/eval; Muon-vs-AdamW optimizer partition (only
`model.layers.*` matrices go to Muon; tables/embeddings/projections/head go to
AdamW); `_init_weights` re-applied over the whole tree after construction;
EOS-boundary n-gram masking (n-grams never cross packed-document boundaries);
FLCE loss path shared by train and eval; `pack_gqa=False` for FA4+GQA;
checkpoint resume guard. The standalone `eval.py` carries the full 9-task
benchmark suite matching the Notion table.

---