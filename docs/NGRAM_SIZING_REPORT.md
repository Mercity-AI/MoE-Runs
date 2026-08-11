# N-gram Table Sizing for the Dense LLaMA + LongCat NE Ablation

**Date:** 2026-08-07
**Scope:** choosing n-gram hash-table sizes and layer counts so a LongCat-style
input N-gram Embedding (NE) takes a target fraction (20–25%, plus exploratory
75/90%) of a ~1B-param dense LLaMA-3-style model, trained on 6B FineWeb tokens
on a single B200.
**Reference:** *Scaling Embeddings Outperforms Scaling Experts in Language Models*
(arXiv:2601.21204).

---

## 1. Mechanism recap

For each position and each order n ∈ {2, 3}, the n-gram is the sliding window of
the last n tokens ending at that position — `(t_{i-1}, t_i)` for bigrams,
`(t_{i-2}, t_{i-1}, t_i)` for trigrams. There is no segmentation algorithm; one
n-gram per position per order. Context that crosses an EOS boundary is zeroed, so
n-grams never span packed documents.

Each (order, head) pair owns one hash table. The n-gram is hashed with a rolling
polynomial hash under that table's private prime multiplier m:
`h = (t_{i-1}·m + t_i) mod size` for bigrams (one more Horner step for trigrams).
Because every table has a distinct m, two tables send the same n-gram to
unrelated rows — the heads are independent hash functions.

Each table lookup returns a `sub_dim`-vector; a per-table learned linear map
projects it to hidden size. The base token embedding plus all T projected
lookups are **summed**, divided by T+1 (a mean), then LayerNorm'd (the paper's
"embedding amplification"). That output replaces the input embedding; the
transformer proper is unchanged.

**Why sub_dim = hidden / T** (paper Eq. 3): each table owns an equal slice of the
hidden dimension, so the concatenation of all T lookups is exactly D-dimensional
and the T projections are blocks of one D×D matrix. Consequence: **each table row
costs D/T params — doubling the table count halves the per-row price**, which is
why an 8-table variant needs twice the rows of a 4-table variant at the same
parameter budget.

## 2. Shared architecture (all configs below)

| Hyperparameter | Value |
|---|---|
| Architecture | LLaMA-3-style dense, GQA + per-head QK-RMSNorm |
| Hidden size | 1,536 |
| Intermediate size | 5,120 (SiLU/SwiGLU) |
| Attention heads / KV heads | 12 / 6 (head_dim 128) |
| Max position embeddings / seq len | 8,192 |
| Vocab | 32,000 (Llama-2 tokenizer), tied embeddings |
| RoPE theta | 1,000,000 |
| N-gram orders | 2–3 (bigram + trigram only) |
| NE amplification | LayerNorm |
| NE hash multipliers (first 8, in table order) | 40009, 100003, 262147, 524287, 1000003, 2000003, 3000017, 4000037 |
| Optimizer | Muon (lr 0.02, momentum 0.95, NS 5, Nesterov, wd 0.1) + aux AdamW (3e-4, β 0.9/0.95, wd 0.1) for non-matrix params incl. tables |
| Schedule | cosine, 150 warmup, min-lr factor 0.1, grad clip 1.0 |
| Batch | 12/device × 20 grad-accum × 8,192 seq; 3,053 steps ≈ 6B tokens |
| Data | FineWeb sample-10BT (streaming, packed, cross-doc attention) |

Parameter accounting used throughout: per transformer layer = QKVO
(2·1536² + 2·1536·768 = 7.078M) + MLP (3·1536·5120 = 23.593M) + norms ≈
**30.674M**; tied embedding = 32000·1536 = **49.15M**. NE params =
Σ(table sizes)·sub_dim + T·sub_dim·1536 (projections) + 2·1536 (LayerNorm).

## 3. Audit of the pre-existing configs

| Config | Layers | Tables | NE params | Total | NE fraction |
|---|---|---|---|---|---|
| Completed 50% run (orders 2–4, K=2, sizes 267003/367007/305011/341013/345017/308962) | 16 | 6 | 497.5M | 1.037B | **48.0%** |
| Drafted 4-table config (K=2, sizes 53003/109009/165013/198078) | 25 | 4 | 204.0M | 1.020B | **20.0%** |
| Drafted 8-table config (K=4, sizes 41011/53003/70001/88003/101009/117013/134017/151027) | 27 | 8 | 147.3M | 1.025B | **14.4%** ⚠ |

**Finding:** the drafted 8-table config misses the 20–25% target — its sizes are
too small for the 27 layers it carries. Running it as-is would confound
table-count with parameter fraction. Fixed below (configs B/C).

## 4. What the paper actually establishes

- **Avoid integer multiples of the base vocab** for table sizes (their Fig. 3b —
  collision spikes there even for primes).
- **≤50% of total params** to NE; beyond that it loses to the baseline. The
  advantage grows with model width (dies early at 280M activated, holds to 50%
  at 1.3B activated). Our dense 1B activates all params — near their favorable
  regime, though their backbones are sparse MoE (see §8).
- **N≥3, K≥2 is a flat region**: "setting N in the range of 3 to 5 consistently
  yields near-optimal performance"; only N=2, K=1 is clearly bad. The trigram cap
  (orders 2–3) loses nothing measurable vs the earlier 4-gram setup.
- **Hit rate**: order-2 tables fill gradually; order-3+ saturate to 1.0 fast —
  the direction of "give trigrams more rows."
- Their corpus is **internal (Meituan), 300B tokens** for scaling studies (11T for
  Flash-Lite). No public data, and **no guidance linking table size or order to
  data budget** — that question is open (§8).

## 5. Collision model (why near-equal sizes, why no tiny tables)

There is no such thing as a cross-table collision: each table is an independent
hash. The only cross-table failure is two heads computing the *identical* hash
function, which requires the same size AND the same effective base
(multiplier mod size) — impossible with distinct prime multipliers; the size-gap
guard (§7) is insurance on top.

Within a table, two n-grams collide at rate ≈ 1/size. The model only fully
confuses two n-grams when they collide in **all K heads** of an order (the summed
per-head signatures otherwise differ), so full-confusion probability ≈ 1/∏ sizes.
The parameter budget fixes the **sum** of sizes, and a product with fixed sum is
maximized by equal terms →

> **At fixed budget, near-equal per-head sizes minimize full collisions.**

Worked example (order-2 budget of 324k rows): two tables of 162k → full-collision
≈ 1/2.6×10¹⁰; one 16k + one 308k → ≈ 1/4.9×10⁹, i.e. **~5× more confusions for
the same params**. A sub-32k table is also a partition *coarser than the unigram
vocab* — a mushy cluster feature, not an n-gram memory. (Multi-resolution schemes
exist in the hash-embeddings literature but are unablated in the paper and
dominated by this metric; treat as a separate experiment if ever.)

This argument (standard multi-hash sketching: Bloom filters / count-sketch /
Svenstrup et al. 2017 "Hash Embeddings") is **not from the LongCat paper** — see
§10 for the provenance split.

## 6. The sizing recipe, in full

The goal: given a target NE fraction `f`, produce (a) a layer count and (b) a
list of table sizes such that NE params / total params = `f` exactly, while
satisfying every hashing constraint in §7. Everything is closed-form because the
architecture is fixed; only two knobs exist — layers on the dense side, rows on
the NE side.

**Step 1 — pick the structural knobs.**
Choose `f` (e.g. 0.25) and heads-per-order `K`. With orders {2, 3} the table
count is T = 2K, and sub_dim = 1536/T (so T must divide 1536: K ∈ {2, 3, 4, 6}).
Choose the layer count L for the dense side. L is what sets the *absolute* scale:
`dense(L) = 49.15M (tied embedding) + 30.674M·L + tiny norms`. For a ~1B total at
f = 25%, L = 23 gives dense = 754.7M; at f = 20%, L = 25 gives dense = 816.0M.
(Pick L first by total-size target, since layers are integer and rows are not.)

**Step 2 — convert the fraction into an NE budget.**
If NE must be fraction `f` of the total, then NE / (NE + dense) = f, which
rearranges to:

```
NE budget = dense · f / (1 − f)
```

e.g. 754.7M × 0.25/0.75 = 251.6M. This is the step that makes the fraction exact
rather than approximate — the budget is derived *from* the dense side actually
built, not from a round total.

**Step 3 — convert the budget into a row budget.**
The NE block has fixed costs that don't scale with table size: T projection
matrices of sub_dim×1536 each, plus a LayerNorm (2×1536). Subtract those, then
divide by the per-row cost (sub_dim):

```
rows = (NE budget − T·sub_dim·1536 − 2·1536) / sub_dim
```

e.g. (251.6M − 4·384·1536 − 3072) / 384 = **649k rows** for f=25%, K=2. Note the
same budget with K=4 (sub_dim 192) yields 1.30M rows — same params, twice the
rows, because each row is half as wide.

**Step 4 — split rows across orders.**
Divide the row budget between order 2 and order 3. We use **45/55**
(bigrams/trigrams). This exact ratio is a heuristic, not a paper result: the
paper's hit-rate finding says trigram tables saturate much faster than bigram
tables (there are far more distinct trigrams than bigrams in any corpus), which
argues for giving trigrams *more* rows, but gives no number. The defensible
version is to measure distinct bi/trigram counts on a FineWeb sample and
load-balance (§10); the paper's flat N/K region suggests the split is
second-order, so 45/55 is a reasonable placeholder pointed in the right
direction. e.g. 649k → 292k (order 2) / 357k (order 3).

**Step 5 — split each order's rows across its K heads, near-equally.**
Per §5, equal sizes minimize full collisions at fixed budget. Stagger them ~2%
apart (e.g. for K=2: base×0.99 and base×1.01) purely to clear the anti-clone
size-gap guard with headroom — with distinct multipliers, even identical sizes
would be safe, so the stagger is insurance, not load-bearing.
e.g. order 2: 292k/2 → targets ≈ 144.6k and 147.5k.

**Step 6 — snap each target to an admissible prime.**
Walk upward from each target to the first prime that (a) is at least 10% of the
vocab (≥3,200) away from every integer multiple of 32,000, and (b) is ≥0.5% away
from every size already chosen. Primes are not strictly required — coprimality
with the multiplier is the real constraint — but a prime different from the
multipliers satisfies it automatically and is easy to audit.
e.g. 144.6k → 144563 (prime, 4.518×32000, 48.2% from the nearest multiple).

**Step 7 — validate.**
Check the full constraint set (§7) over the final list, and recompute the exact
fraction with the final (snapped) sizes — snapping moves the total by <0.1%.

## 7. Hashing constraint set (what "valid" means)

1. **No two tables are the same hash function**: no two may share both a size and
   an effective base (multiplier mod size). This is the invariant that prevents
   K heads collapsing to 1.
2. **Every multiplier ≥ 32,000** (base vocab), or distinct n-grams alias before
   the modulus.
3. **gcd(multiplier, size) = 1**, or one n-gram coordinate collapses into
   gcd-many classes.
4. **Pairwise size gap ≥ 0.5%** (we use ~2%) between any two tables.
5. **Distance ≥ 10% of vocab (3,200) from any integer multiple of 32,000**
   (paper Fig. 3b; the codebase's guard requires 5% — we hold ourselves to 10%).

## 8. Recommended configs

All sizes prime; distance to the nearest multiple of 32,000 is ≥19.7% of vocab
(worst case, config C's 121697; most are 25–48% away); all pass the §7
constraints with the default multipliers listed in §2. Within each list, the
first half are the bigram tables, the second half trigram (45/55 row split);
table order pairs positionally with the multiplier list.

**A — 25%, 4 tables (K=2, sub_dim 384), 23 layers** *(recommended primary)*
```
num_hidden_layers      = 23
ngram_num_heads        = 2       # per order; orders {2,3} → 4 tables
ngram_table_vocab_sizes = [144563, 147481, 176677, 180247]
```
→ NE 251.6M / total 1.006B = 25.00%

**B — 25%, 8 tables (K=4, sub_dim 192), 23 layers** *(table-count ablation partner)*
```
num_hidden_layers      = 23
ngram_num_heads        = 4       # per order; orders {2,3} → 8 tables
ngram_table_vocab_sizes = [141637, 144563, 147481, 150401,
                           173137, 176677, 180247, 183823]
```
→ NE 251.6M / total 1.006B = 25.00%

**C — 20%, 8 tables (K=4, sub_dim 192), 25 layers** *(fixes the drafted 8-table
config; apples-to-apples with the existing 20% 4-table draft)*
```
num_hidden_layers      = 25
ngram_num_heads        = 4
ngram_table_vocab_sizes = [114613, 116969, 119359, 121697,
                           140111, 142963, 145861, 148747]
```
→ NE 204.0M / total 1.020B = 20.00%

**D — 75% far-endpoint (K=2, sub_dim 384), 7 layers** *(exploratory; preferred
over 90%, which forces a 2-layer backbone — an n-gram model with a tiny mixer)*
dense 264M + NE 792M ≈ 1.055B total, ~2.06M rows. Sizes to be generated with the
same recipe if the run is approved. A sweep of {20/25 (planned), 48 (done), 75}
brackets the knee of the scaling curve with one extra run.

Layer counts are load-bearing: config A/B sizes at 25 layers give ~23.5% of a
1.07B model, not 25%.

## 9. Compute and data-budget analysis

**FLOPs are computable offline** from the pinned architecture (6×matmul params +
attention terms; NE adds only 0.2–0.7%):

| Config | GFLOPs/token | 6B-token total | Est. B200 hrs @ ~40% MFU |
|---|---|---|---|
| Baseline (25L dense, no NE) | 6.78 | 4.07×10¹⁹ | ~12–14 |
| 25% NE (23L) | 6.28 | 3.77×10¹⁹ (−7%) | ~11–13 |
| 48% run (16L) | 4.46 | 2.68×10¹⁹ (−34%) | ~7–9 |
| 75% (7L) | 2.13 | 1.28×10¹⁹ (−69%) | ~3.5–4.5 |

Key framing for the writeup: the paper compares **iso-params only** (their MoE
swaps sparse capacity for sparse capacity, so FLOPs stay matched by
construction). In a *dense* backbone, NE swaps FLOPs-bearing layers for near-free
lookups, so iso-param comparisons give NE models a FLOPs advantage — **the
completed 48% run matched the baseline loss curve at ~2/3 the FLOPs**, a claim
the paper's own setting cannot make. Report both loss-vs-tokens and
loss-vs-FLOPs.

**Data budget:** no published tokens-per-embedding-param law exists. The paper
fixes 300B tokens and never varies data; Kaplan et al. found scaling laws fit
better *excluding* embedding params; the sparse-memory literature (product-key
memories, Memory Layers at Scale) treats tables as cheap memorization capacity
that still needs data flow per row. Occupancy check at 6B tokens: ~41k n-gram
instances per row for 146k-row tables, ~10k even for the 75%-config's 585k-row
tables — averages healthy, undertraining risk concentrated in the frequency tail
and in the 75/90% configs. The 20–25% runs are comfortably fed at 6B tokens.

## 10. Provenance and open items

| Claim | Source |
|---|---|
| Parameter accounting, row-budget formula, FLOPs table | arithmetic from the pinned architecture (§2) |
| Avoid vocab multiples; ≤50%; N≥3/K≥2 flat; hit-rate direction | LongCat paper |
| Near-equal per-head sizes (product-of-sizes argument) | standard sketching theory, not LongCat |
| **45/55 order split** | **heuristic** — direction from hit-rate, ratio unverified |

**Open item — measure the order split:** no published law gives the
bigram:trigram *type* ratio at the BPE level (Heaps'-law growth and word-level
counts like Google Web-1T suggest trigram types ≈ 3× bigram types, but words ≠
Llama-2 BPE on FineWeb). Plan: tokenize a ~100M-token FineWeb sample, count
distinct bi/trigrams, fit the Heaps exponent, extrapolate to 6B, and set
per-order rows to equalize (frequency-weighted) load — capping skew near ~40/60
since rare-rare collisions are cheap and the paper's flat region says the split
is second-order.

**Other open items:** apply config C (or B) to the drafted 8-table training
config; compute achieved MFU from logged run durations; generate config-D sizes
if the 75% run is approved.
