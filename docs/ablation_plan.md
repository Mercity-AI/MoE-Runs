# PLE Ablation Plan — Embedding-Based Conditioning Mechanisms

## Context

The proposal ([`Lightweight Expert Conditioning...`](docs/Lightweight%20Expert%20Conditioning%20on%20Shared%20Weights%20%20319bc705ab73805ca28fe38514269d61.md)) targets **Mechanism 1: Per-Layer Embeddings (PLE)** — a per-expert, per-layer additive vector `e_l^(k)` projected into the hidden state (`§3.2`), positioned as a near-zero-HBM way to differentiate experts without duplicating FFN weights. PLE is already implemented (`initial_lec_llama.py`, `initial_lec_llama_torchtitan.py`, `initial_lec_llama_no_router.py`) and will run as the primary experiment — it is **excluded** from the recommendations below by design.

**Baseline** (`baseline_llama_torchtitan.py`): dense 1B LLaMA-style backbone (32 layers, hidden=1536, GQA 12/4 heads, vocab=32000, tied embeddings ≈1.0B total params), Muon+AuxAdam, FA4 + Liger FLCE + TorchTitan, trained on 6B FineWeb tokens on a B200, final loss ≈1.68 (well short of Chinchilla-optimal ~20B tokens for 1B params, but scalable later).

The open question this plan addresses: **is PLE-style embedding conditioning actually a good bet, relative to other cheap embedding-augmentation mechanisms that don't require the expert/router machinery at all?** The four ablations below let PLE be judged against the current state of the art in *n-gram-based* embedding augmentation, which is a competing (and in some cases empirically stronger) way to spend the same parameter budget.

---

## Recommendation 1 — DeepSeek Engram (hashed n-gram conditional memory)

**Paper:** [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/html/2601.07372v1) (DeepSeek-AI, 2026) · [code](https://github.com/deepseek-ai/Engram)

**Algorithm.** Compress the vocab via NFKC-normalization/case-folding (~23% reduction), form suffix 2-/3-grams of the compressed token stream, hash each n-gram order through *K* independent multiplicative-XOR hash heads into prime-sized embedding tables, concatenate the retrieved vectors into a memory vector `e_t`. A context-gate `α_t = σ(RMSNorm(h_t)ᵀRMSNorm(k_t)/√d)` (current hidden state as query, retrieved memory as key/value) modulates how much of `e_t` is trusted, followed by a causal depthwise conv (kernel 4) + SiLU + residual, injected additively into a handful of layers (not every layer — placement is a tunable). Conv weights zero-init so the module starts as identity.

**Why it's a good ablation against PLE.** Both are additive, near-zero-inference-cost, offloadable conditioning signals — but Engram conditions on *n-gram context* (deterministic, data-driven) rather than a *learned per-expert code* (PLE). It directly tests whether the useful information PLE is trying to inject at each layer is actually just "which local n-gram pattern is this," which would be cheaper to get from a lookup table than from a learned router+embedding. DeepSeek's own ablations (Fig. 5) show layer-placement sensitivity (best at layers 2+6 of a 12-layer net) — directly comparable to how PLE's per-layer injection point matters.

**Cost.** DeepSeek's "Sparsity Allocation Law" recommends ~20–25% of the *sparse* (inactive) parameter budget go to Engram vs. MoE experts — but our baseline is dense, so there's no free inactive budget to reallocate. For a first pass: a modest table (~500K–1M hashed slots × 256–512 dim, single branch, 2–3 grams) adds tens of millions of params, offloadable to host memory (measured ~2–3% inference throughput overhead for a 100B-param table in the paper, so a table 1000x smaller is effectively free). Engineering cost is **moderate-high**: multi-head hashing, prime-bucket tables, gating, and depthwise conv all need implementing; no off-the-shelf kernel in our current stack (Liger/FA4/TorchTitan).

---

## Recommendation 2 — LongCat n-gram embedding scaling

**Paper:** [Scaling Embeddings Outperforms Scaling Experts in Language Models](https://arxiv.org/html/2601.21204v1) (Meituan LongCat, 2026) — production-validated in [LongCat-Flash-Lite](https://www.longcatai.org/models/flash-lite) (68.5B total, 31.4B in n-gram embeddings).

**Algorithm.** For token *i*, hash n-grams of order 2..N via polynomial rolling hash `H_n = (Σ tᵢ₋ⱼ·V₀ʲ) mod Vₙ`. To fight collisions, each order is split into *K* independent sub-tables `E_{n,k} ∈ ℝ^{V_{n,k} × D/((N-1)K)}` with linear projections `W_{n,k}` back to full width — this keeps total params invariant as N/K change, which makes it a clean knob for ablation. In the paper's standard N-gram Embedding (NE), the projected branches are averaged together with the ordinary token embedding and injected once as the transformer's augmented **input embedding**. LongCat found this signal was initially drowned out by attention (10x larger norm) until they added scale/LayerNorm correction (−0.02 loss). The paper separately studies Per-Layer N-gram Embedding (PLNE), which substitutes a layer-specific n-gram embedding for the SwiGLU up-projection inside selected FFNs; that is not the standard NE mechanism used for its main scaling results or LongCat-Flash-Lite.

**Why it's a good ablation against PLE.** This is the most directly comparable mechanism in the literature: LongCat explicitly frames it as *scaling embeddings instead of scaling experts*, i.e. the exact question this ablation plan exists to answer, and reports it beating a parameter-matched MoE baseline on their production model. It also isolates a different design axis than Engram: amplified input-embedding augmentation vs. additive gated memory injection, and un-gated deterministic lookup vs. Engram's learned context-gate. Their own ablation shows N≥3, K≥2 is the robust regime, and diminishing returns allocating >50% of params to n-grams — useful priors for sizing our version.

**Cost.** The mechanism augments the input embedding and leaves the transformer blocks unchanged — **moderate** engineering effort: hash + sub-table decomposition + projection, no gating network or conv needed. Same collision/table-size tuning knobs as Engram. Parameter footprint is fully controllable via the sub-table budget (LongCat's own production model allocates ~46% of total parameters to N-gram embeddings — we'd want to start much smaller, e.g. matching PLE's parameter budget for an apples-to-apples comparison).

---

## Recommendation 3 — N-Grammer (PQ-based latent n-grams)

**Paper:** [N-Grammer: Augmenting Transformers with Latent N-grams](https://arxiv.org/abs/2207.06366) (Google, 2022)

**Algorithm.** Cluster unigram embeddings into a discrete latent code via product quantization (PQ), then compute n-gram IDs over the *discrete latent codes* (not raw token IDs), and look up a bigram embedding table indexed by those latent n-gram IDs. The result is summed into the token representation early in the stack. Sparse lookup only — no hashing collisions to manage (PQ code space is small and controlled).

**Why it's a good ablation against PLE.** This is the oldest and cheapest of the n-gram-embedding family, and it isolates a narrower question than Engram/LongCat: is *any* n-gram signal useful at all on our stack, before paying for hashed tables, gating networks, or conv layers? It's a strong "cheap first ablation" — if N-Grammer-style bigram conditioning shows no lift over the dense baseline, that's a signal not to invest further engineering in Engram/LongCat before checking other explanations (data, seq length, model scale). It also uses a fundamentally different mechanism (PQ discretization) than the hash-based methods, giving a useful second data point on whether *discretization* vs. *hashing* matters for n-gram embedding quality.

**Cost. Lowest** of the four: PQ codebook (small, e.g. 512–1024 codes) + one bigram embedding table sized by the codebook, no gating/conv machinery. This is the fastest ablation to stand up and should probably run first to sanity-check the whole n-gram direction before investing in Engram or LongCat's heavier machinery.

---

## Recommendation 4 — SCONE (parametric f-gram embeddings, Google)

**Paper:** [Scaling Embedding Layers in Language Models](https://arxiv.org/html/2502.01637v3) (Yu, Cohen, Ghazi et al., Google, 2025)

**Algorithm.** Mine frequent n-grams ("f-grams", n up to ~5) from the training corpus via K−1 linear corpus scans with a frequency floor. Rather than a directly-trained lookup table (which suffers sparse-gradient problems at huge vocab sizes — only 7.3% of a 2M-entry vocab gets >100 updates), train a small separate transformer (`𝒜_f-gram`) that *generates* contextualized f-gram embeddings; cache its outputs for every f-gram after training. At inference, greedy longest-f-gram-match lookup against the cached table (in host RAM or NVMe) replaces the token embedding, falling back to standard token embedding on no match.

**Why it's a good ablation against PLE.** SCONE decouples embedding-table size from vocabulary size entirely (up to 100M–1B f-gram entries), and reports that a 1B accelerator-resident model + 1B offloaded f-gram embeddings **beats a 1.9B dense baseline** at ~48% less inference FLOPs/memory — a genuinely different and stronger result than PLE's expressivity argument (§4 of the proposal), since it's an empirically validated scaling result rather than a first-order Jacobian argument. It's a good "ceiling" ablation: if PLE's real value is "cheap offloadable per-token conditioning," SCONE is evidence of how far that idea can be pushed when done well.

**Cost. Highest** engineering lift of the four: requires offline corpus scanning to mine f-grams, training a second model (`𝒜_f-gram`), a caching/offload pipeline (LMDB or similar), and longest-match lookup logic at both train and inference time. Recommend running this **last**, only if Recommendations 1–3 show that n-gram-family embeddings are competitive with or better than PLE — at which point it's worth knowing how much further the approach scales.

---

## Excluded: Kimi Delta Attention (KDA) — not a valid embedding ablation

Per clarification, "K-means delta" in the original notes referred to **Kimi Delta Attention** ([Kimi Linear paper](https://huggingface.co/papers/2510.26692), Moonshot AI, 2025) — the delta-rule, channel-wise-gated linear attention mechanism used in Kimi Linear (3:1 KDA:full-attention hybrid, [code](https://github.com/MoonshotAI/Kimi-Linear)).

**This does not belong in this ablation set.** KDA is a full replacement for the *attention/token-mixing* mechanism (an alternative to Gated DeltaNet), not an embedding-conditioning or expert-differentiation mechanism — it doesn't touch token or per-layer embeddings at all. Swapping it in would confound the "is embedding-based conditioning worth it" question this plan is answering with an unrelated architecture change to the attention backbone, and it doesn't map onto the near-zero-HBM per-expert-conditioning framing that PLE/Engram/LongCat/N-Grammer/SCONE all share. If there's interest in KDA specifically, it warrants its own separate ablation track (attention-backbone efficiency), not a slot in this embedding-mechanism comparison.

---

## Suggested execution order

| Order | Ablation | Engineering cost | Rationale |
|---|---|---|---|
| 1 | N-Grammer | Low | Cheapest sanity check: is n-gram signal useful at all on this stack |
| 2 | DeepSeek Engram | Moderate–High | Most directly comparable framing to PLE (additive, gated, offloadable) |
| 3 | LongCat n-gram scaling | Moderate | Explicitly validated against MoE-style scaling; different injection point (multiplicative FFN gate) |
| 4 | SCONE | High | Run only if 1–3 show promise; tests the ceiling of the approach |

All four should be run at a parameter budget roughly matched to PLE's footprint for a first pass, then (if promising) swept up following each paper's own scaling ablations, using the same 6B-FineWeb / Muon / FA4+Liger training recipe as the baseline for a fair loss comparison.
