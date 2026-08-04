# Lightweight Expert Conditioning on Shared Weights: Toward HBM-Efficient Mixture-of-Experts via Per-Layer Embeddings, Expert Tokens, and Modular Conditioning

## A Research Proposal

---

**Abstract.** Current Mixture-of-Experts (MoE) architectures achieve strong performance by routing inputs to distinct expert subnetworks, but do so at the cost of massive parameter counts and high-bandwidth memory (HBM) consumption—each expert is a full copy or partition of the model’s feed-forward layers. We propose *Lightweight Expert Conditioning* (LEC), a framework that maintains a single shared set of dense base weights while differentiating expert behavior through a modular stack of ultra-lightweight, per-expert conditioning mechanisms: Per-Layer Embeddings (PLE), learned expert prefix tokens, per-expert value residual scalars and bias vectors, per-expert LayerNorm affine parameters (FiLM), per-expert Canon convolution kernels, and sparse LoRA adapters. Mathematical analysis using tools from linear algebra, information theory, and function space geometry shows that the nonlinear compounding of small additive perturbations through depth yields exponentially richer expressivity than naive parameter counting suggests. We complement this expressivity analysis with an honest treatment of the Shannon capacity gap—the information-theoretic bound separating *behavioral specialization* from *knowledge capacity expansion*—and identify per-expert parameter budget as the key scaling vector, ranging from ~2 MB/expert (LEC-Lite, for routing specialization at near-zero HBM cost) to ~200 MB/expert (LEC-Full, for capacity expansion competitive with 2–4× larger dense models). We further propose an *asymmetric attention* design in which LoRA adapters are applied only to $W_Q$ and $W_O$ projections while $W_K$ and $W_V$ remain strictly shared, preserving a universal metric space and enabling unfragmented KV cache sharing during inference. Expert symmetry breaking is addressed through orthogonal subspace initialization, ensuring maximal gradient signal to the router from the first training step. We present a detailed research plan including architecture design, theoretical expressivity bounds, a staged training recipe with orthogonal initialization, comprehensive ablation studies, hardware cost analysis, kernel fusion considerations, and risk mitigation strategies. The goal is to achieve 40–80% of full MoE quality gains at a fraction of the per-expert parameter overhead, with near-zero additional inference cost.

---

## Subpages

[PLE + Router abalations](https://www.notion.so/PLE-Router-abalations-34abc705ab7380cba226de1d2812be40?pvs=21)

[Framework comparisons for pretraining ](https://www.notion.so/Framework-comparisons-for-pretraining-35abc705ab73809e88fff7e8eef09a46?pvs=21)

[Experiments page](https://www.notion.so/Experiments-page-360bc705ab738005ba66d51e6a148df2?pvs=21)

## 1. Introduction and Motivation

### 1.1 The Cost Problem of Mixture-of-Experts

The Mixture-of-Experts paradigm has become central to frontier language models. DeepSeek-V3, Mixtral 8×7B, and (reportedly) GPT-4 all employ MoE to decouple model capacity from per-token compute cost. However, this efficiency comes with severe memory requirements: a Mixtral 8×7B model, despite activating only 2 experts per token (~12B effective FLOPs), must store all 8 expert FFN copies in memory, yielding ~47B total parameters. For on-device deployment, edge inference, or memory-constrained settings, this is prohibitive.

The fundamental question driving this proposal is: **Can we achieve the specialization benefits of MoE while keeping memory costs close to those of a single dense model?**

### 1.2 Key Insight: Expert Differentiation Through Conditioning, Not Duplication

Our core observation is that MoE experts, particularly those created via “upcycling” from a dense checkpoint, share enormous redundancy. Research on upcycled MoE models has shown that expert weights can be decomposed into one shared base weight plus small per-expert delta weights, and even extreme compression of these deltas (e.g., 1-bit quantization or 99% sparsification) produces minimal performance degradation (DeRS; Li et al., 2025). This suggests that the “knowledge” largely resides in the shared structure, and expert specialization is a relatively low-dimensional phenomenon.

We propose to formalize this insight: rather than creating experts by duplicating or partitioning weights, we condition a single shared model through lightweight, per-expert signals injected at every layer. Drawing on recent innovations in Per-Layer Embeddings (Gemma 3n; Google, 2025), conditional memory via N-gram lookup (Engram; DeepSeek, 2025), value residual learning (ResFormer; Zhou et al., 2024), Canon layers (Allen-Zhu, 2025), and Mixture-of-LoRA approaches (X-LoRA, MoLE, MoLoRA, HydraLoRA), we construct a modular “expert conditioning stack” that can be mixed, matched, and ablated.

### 1.3 Contributions

This proposal makes the following contributions:

1. **A unified framework (LEC)** that combines six distinct lightweight conditioning mechanisms into a coherent MoE architecture, all operating on shared base weights, with two concrete operating points: LEC-Lite (~2 MB/expert for behavioral specialization) and LEC-Full (~200 MB/expert for capacity expansion).
2. **Theoretical expressivity analysis** using linear algebra (Jacobian rank analysis), information theory (channel capacity and information leverage), and function space geometry (covering number bounds on the reachable function family) showing that depth-distributed lightweight conditioning achieves exponentially richer expressivity than parameter counting suggests.
3. **An honest capacity analysis** distinguishing expressivity (function space coverage) from Shannon knowledge capacity, identifying the per-expert parameter budget as the key scaling vector, and clarifying the relationship to Gemma 3n’s PLE and DeepSeek’s Engram.
4. **An asymmetric attention design** that applies per-expert LoRA only to $W_Q$ and $W_O$ while keeping $W_K$ and $W_V$ strictly shared, preserving KV cache integrity and metric space consistency for production inference.
5. **A detailed hardware cost analysis** demonstrating that LEC-Lite expert switching incurs <0.1% overhead relative to dense inference, including kernel fusion analysis for deployment.
6. **A staged training recipe** with orthogonal subspace initialization for expert symmetry breaking, optional router warmup, per-group learning rates, and auxiliary-loss-free load balancing.
7. **Comprehensive ablation study designs** isolating the contribution of each conditioning mechanism and their interactions.
8. **Identification of key risks and mitigation strategies**, including router collapse, KV cache fracture from expert prefix tokens, gradient interference, and expressivity ceilings.

---

## 2. Background and Related Work

### 2.1 Sparse Upcycling: Dense to MoE Conversion

**Sparse Upcycling** (Komatsuzaki et al., 2022) introduced the idea of initializing an MoE model from a dense checkpoint by duplicating FFN layers and adding a router, achieving strong performance using only ~50% of the original dense pretraining compute. The approach was validated on T5 (Base, Large, XL) and Vision Transformer (Base, Large) models on SuperGLUE and ImageNet respectively.

**Upcycling LLMs into MoE** (He et al., 2024) from NVIDIA conducted an extensive study at billion-parameter scale, proposing a “virtual group” initialization scheme and demonstrating that softmax-then-topK expert routing outperforms topK-then-softmax, and higher granularity MoEs improve accuracy. Their upcycled Nemotron-4 15B achieved 67.6% MMLU versus 65.3% for continued dense training on the same 1T token budget.

**UpIT** (Hui et al., 2024) proposed using intermediate checkpoints from instruction tuning as naturally specialized experts, expanding the expert count via genetic algorithms and parameter merging for diversity. This work emphasizes the importance of expert diversity in upcycling.

**BAM (Branch-Attend-Mix)** (Zhang et al., 2024) extended upcycling beyond FFN layers by also leveraging attention parameters, initializsoping a soft Mixture-of-Attention (MoA) layer from dense attention weights.

### 2.2 Dense-to-MoE via Partitioning

**MoEfication** (Zhang et al., 2021) pioneered the idea of converting a dense model into an MoE by splitting FFN neurons into expert groups based on weight clustering.

**D2DMoE (Dense to Dense MoE)** (NeurIPS 2024) built on MoEfication, using balanced k-means to cluster neurons by weight similarity and split linear layers into experts. They demonstrated that enforcing activation sparsity before conversion improves results, and that static top-k gating is suboptimal for converted models due to high per-token and per-layer variance.

**DeRS (Decompose, Replace, Synthesize)** (Li et al., 2025) decomposed upcycled expert weights into a shared base weight plus expert-specific delta weights, then compressed the deltas via sparsification or quantization. Remarkably, even under extreme compression settings (e.g., 0.99 drop rate sparsification or 1-bit quantization), performance degradation was minimal—providing direct empirical support for our hypothesis that expert differentiation is low-dimensional.

**Parameter-Efficient Sparsity Crafting** (Wu et al., 2024; EMNLP) specifically addressed converting LLaMA-architecture dense models into MoE with parameter efficiency.

### 2.3 Per-Layer Embeddings and Memory-Efficient Architectures

**Gemma 3n** (Google, 2025) introduced Per-Layer Embeddings (PLE) as a core architectural innovation for on-device deployment. PLE parameters are generated separately, cached to fast storage, and streamed to the model during inference layer-by-layer. The E2B model has 5B total parameters but operates with an effective memory load of ~2B parameters by offloading PLE to CPU. The PLE approach stores a per-token embedding vector for each layer, allowing different facets of token identity to be accessed at the layers where they are relevant rather than being crammed into a single initial embedding.

**Engram** (DeepSeek, 2025) introduced conditional memory as a complementary sparsity axis alongside MoE’s conditional computation. Engram modernizes classic N-gram embeddings for constant-time O(1) lookups, separating static pattern recall from dynamic reasoning. A key finding was that reallocating roughly 20–25% of the sparse parameter budget from MoE to Engram yields optimal performance, following a U-shaped scaling law. The 100B-parameter embedding table can be offloaded entirely to host memory with less than 3% inference overhead, since indices are deterministic and can be asynchronously prefetched via PCIe.

### 2.4 Value Residual Learning

**ResFormer** (Zhou et al., 2024) addresses the over-smoothing problem in deep transformers by adding a residual connection from the first layer’s value vectors to all subsequent layers. The formulation is:

$$
U_n = \text{Attn}(Q_n, K_n, V_n) + \lambda_n(V_1 - V_n)
$$

where $\lambda_n$ is a learnable scalar per layer. ResFormer achieves equivalent validation loss with 16.11% fewer model parameters and 20.3% less training data compared to vanilla Transformers. The variant SVFormer shares the first layer’s value embedding across all layers, reducing KV cache by nearly 50%.

### 2.5 Canon Layers

**Canon Layers** (Allen-Zhu, 2025; Physics of Language Models Part 4.1) are lightweight trainable 1-D causal convolutions (kernel size 4) that promote horizontal information flow across neighboring tokens. With only a 0.5% increase in trainable parameters, Canon layers enhance reasoning depth by 2–4×, reasoning breadth by 30%, knowledge capacity by 10–15%, and knowledge manipulation length by 30%. They can be placed at four positions within a transformer block (Canon-A/B/C/D) and integrate seamlessly with any sequence architecture.

### 2.6 Mixture-of-LoRA Approaches

**X-LoRA** (Buehler & Buehler, 2024) uses learned, layer-wise, token-level scaling values to mix pre-trained LoRA adapters, creating deep layer-wise combinations to solve tasks. All LoRA adapters and the base model are frozen during X-LoRA training.

**MoLE (Mixture of LoRA Experts)** (ICLR 2024 submission) harnesses hierarchical control and branch selection for LoRA fusion, outperforming direct arithmetic merging.

**MoLORA** (Zadouri et al., 2024; ICLR 2024) introduces Mixture of Vectors (MoV) and Mixture of LORA as parameter-efficient MoE adaptations, achieving full fine-tuning parity by updating less than 1% of parameters.

**HydraLoRA** (NeurIPS 2024) shares the A matrix across subdomains while deploying multiple B matrices as experts, with an MoE router for automatic segregation—directly relevant to our sparse LoRA component.

**MoLA** (NAACL 2025) applies LoRA-MoE with layer-wise expert allocation, demonstrating that different layers benefit from different numbers of experts.

---

## 3. Proposed Architecture: Lightweight Expert Conditioning (LEC)

### 3.1 Overview

Given a pretrained dense transformer with $L$ layers, hidden dimension $d$, and $n_h$ attention heads, we construct an MoE with $K$ experts where:

- **Shared components** (loaded once in HBM): All FFN weights ($W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}$), attention key and value projections ($W_K, W_V$), shared LayerNorm parameters, and the token embedding table. In the LEC-Lite configuration, all attention weights are shared; in LEC-Full, FFN weights receive per-expert LoRA adapters.
- **Asymmetric attention constraint:** $W_K$ and $W_V$ are *always* strictly shared and never receive per-expert modifications, preserving a universal metric space and unfragmented KV cache (see Section 3.7.1 for detailed rationale).
- **Per-expert components** (lightweight, offloadable): A modular stack of conditioning mechanisms, described below.

A learned router $R(x) \in \mathbb{R}^K$ takes the hidden state as input and selects the top-$k$ experts per token (typically $k=1$ or $k=2$). The selected expert’s conditioning parameters are activated for that token’s forward pass.

### 3.2 Conditioning Mechanism 1: Per-Layer Embeddings (PLE)

Inspired by Gemma 3n, each expert $k$ maintains a PLE table of dimension $d_{\text{ple}}$ for each layer $l$:

$$
e_l^{(k)} \in \mathbb{R}^{d_{\text{ple}}}
$$

At layer $l$, before the attention operation, the PLE is projected to hidden dimension (if $d_{\text{ple}} < d$) and added to the hidden state:

$$
\tilde{h}_l = h_l + W_{\text{ple}} \cdot e_l^{(k)}
$$

where $W_{\text{ple}} \in \mathbb{R}^{d \times d_{\text{ple}}}$ is a shared (across experts) projection matrix.

**Parameter cost per expert:** $L \times d_{\text{ple}} \times p$ bits. For $L=32$, $d_{\text{ple}}=256$, $p=16$: **~16KB**.

**HBM cost:** Zero during inference—PLE vectors can be offloaded to CPU/flash and streamed layer-by-layer, following the Gemma 3n approach. Prefetch latency is masked by the compute time of the preceding layer.

**Expressivity character:** Shifts each layer’s operating point, creating multiplicative effects through the bilinear structure of attention and compounding nonlinearly through depth.

### 3.3 Conditioning Mechanism 2: Learned Expert Prefix Tokens

Each expert $k$ maintains $m$ learned token embeddings:

$$
T^{(k)} = \{t_1^{(k)}, \ldots, t_m^{(k)}\} \quad \text{where } t_i^{(k)} \in \mathbb{R}^d
$$

These are prepended to the input sequence. A modified attention mask ensures:
- Content tokens attend to expert tokens (they are in “the past” under causal masking).
- Expert tokens attend to each other bidirectionally.
- Expert tokens do NOT attend to content tokens (they are static conditioning signals).

This is identical to the prefix tuning setup and requires only a mask modification.

**Parameter cost per expert:** $m \times d \times p$. For $m=10$, $d=3072$, $p=16$: **~60KB**.

**HBM cost:** ~60KB (trivial).

**Expressivity character:** Provides content-adaptive modulation through the attention mechanism. Each content token attends to expert tokens differently based on its query vector, creating per-head, per-token, input-dependent expert influence. This is fundamentally different from PLE (which is position-independent) and more powerful per parameter than LoRA because the modification weights (attention scores) are themselves adaptive.

> **⚠ Critical Deployment Concern: KV Cache Fracture.** Expert prefix tokens introduce a structural problem for autoregressive inference. When token $t$ routes to Expert A and attends to Expert A’s prefix tokens, the resulting KV state for token $t$ becomes semantically bound to Expert A’s context. If the subsequent token $t+1$ routes to Expert B, it must attend to token $t$’s cached key/value vectors, which were computed under Expert A’s prefix context. This creates cross-expert context bleeding within the KV cache. The consequences are twofold: (1) the KV cache can no longer be cleanly shared across experts, breaking the assumptions of PagedAttention and continuous batching systems (vLLM, TGI) that are standard in production serving; and (2) in principle, the model may learn to mitigate this inconsistency by collapsing to single-expert-per-sequence routing, sacrificing per-token routing granularity. **We retain expert prefix tokens in the full research ablation** (Section 7) to measure their isolated expressivity contribution, but **recommend their removal from deployment configurations.** The expressivity they provide can be largely recovered through the combination of PLE (additive input conditioning) and asymmetric LoRA on $W_Q$/$W_O$ (Section 3.7), without compromising the KV cache.
> 

### 3.4 Conditioning Mechanism 3: Per-Expert Value Residual Parameters

Building on ResFormer’s value residual connections, we make the residual weight expert-specific and add a per-expert learnable bias vector:

$$
U_n^{(k)} = \text{Attn}(Q_n, K_n, V_n) + \lambda_n^{(k)} \cdot (V_1 - V_n + \delta_n^{(k)})
$$

where $\lambda_n^{(k)} \in \mathbb{R}$ is a per-expert, per-layer scalar and $\delta_n^{(k)} \in \mathbb{R}^d$ is a per-expert, per-layer bias vector.

**Parameter cost per expert:** $(1 + d) \times L \times p$. For $d=3072$, $L=32$, $p=16$: **~192KB**.

**HBM cost:** ~192KB (trivial).

**Expressivity character:** Controls how much raw token identity (from layer 1) vs. deep semantic representation flows through the network. The $\lambda$ scalar gives global layer-level control; the $\delta$ vector is distributed across positions by the attention matrix, providing position-adaptive modulation. One expert might favor high $\lambda$ (heavy reliance on token features—good for factual recall), while another might favor low $\lambda$ (deep reasoning).

### 3.5 Conditioning Mechanism 4: Per-Expert LayerNorm Affine Parameters

Replace the standard shared LayerNorm with per-expert affine parameters:

$$
\text{LN}^{(k)}(x) = \gamma^{(k)} \odot \frac{x - \mu}{\sigma} + \beta^{(k)}
$$

where $\gamma^{(k)}, \beta^{(k)} \in \mathbb{R}^d$ are per-expert per-layer.

**Parameter cost per expert:** $2d \times L_{\text{norm}} \times p$. For $d=3072$, $L_{\text{norm}}=64$ (2 norms per layer × 32 layers), $p=16$: **~768KB**.

**HBM cost:** ~768KB.

**Expressivity character:** Multiplicative rescaling of activations. This directly controls which neurons in the subsequent FFN are active or suppressed, effectively gating different “sub-circuits” within the shared model. LayerNorm scale/shift is one of the most influential operations per parameter because it acts multiplicatively on the entire hidden state.

### 3.6 Conditioning Mechanism 5: Per-Expert Canon Convolution Kernels

Each expert $k$ has its own 1-D causal convolution kernels at Canon positions within each transformer block:

$$
h_t^{(k)} = h_t + \text{Conv1D}^{(k)}(h_{t-3:t})
$$

with kernel size 4, applied at Canon-A and Canon-C positions (before attention and before FFN).

**Parameter cost per expert:** $4d \times 2 \times L \times p$. For $d=3072$, $L=32$, $p=16$: **~16MB**.

This is the most expensive conditioning mechanism. A lighter alternative is shared Canon kernels with per-expert scaling/bias on the Canon output: cost drops to $(2d) \times 2 \times L \times p \approx 768\text{KB}$.

**HBM cost:** 16MB (full) or 768KB (scaled variant). Even the full version is feasible.

**Expressivity character:** Controls local horizontal mixing patterns—how much each position borrows from its neighbors. Code experts might want very different local mixing than natural language experts. Canon layers have been shown to recover knowledge capacity lost in MoE architectures.

### 3.7 Conditioning Mechanism 6: Sparse LoRA Adapters

For direct modification of the model’s learned transformations, we add rank-$r$ LoRA adapters at strategic points:

$$
y = Wx + A^{(k)} B^{(k)} x
$$

### 3.7.1 Asymmetric Attention LoRA (Recommended)

A critical design decision concerns *which* attention projections receive per-expert LoRA. We propose an **asymmetric** design:

- **Expert-conditioned (LoRA applied):** $W_Q$ and $W_O$ only.
- **Strictly shared (no LoRA):** $W_K$ and $W_V$.

The rationale is both theoretical and systems-level:

**KV Cache Preservation.** In autoregressive serving, the KV cache stores key and value vectors computed at previous positions. If $W_K$ and $W_V$ are expert-dependent, each cached entry becomes bound to the expert that produced it. When a subsequent token routes to a different expert, it must attend to keys and values computed under a foreign projection—creating metric space inconsistency. By keeping $W_K$ and $W_V$ shared, the KV cache represents a single, universal “library” of context, and different experts simply learn different “queries” ($W_Q$) and “projections” ($W_O$) to read from that shared library differently.

**Metric Space Consistency.** Attention computes geometric distances in a learned metric space via $Q_t K_{t'}^T$. If both $Q$ and $K$ are expert-specific, and consecutive tokens route to different experts, the cross-expert dot products $Q^{(A)}_t \cdot K^{(B)T}_{t'}$ evaluate distances using misaligned projections. While the base $W_Q$ and $W_K$ dominate (LoRA is a small perturbation), maintaining strict $W_K$/$W_V$ sharing eliminates this concern entirely.

**Parameter Efficiency.** Applying LoRA to 2 projections instead of 4 halves the attention LoRA budget, which can be reallocated to FFN LoRA (Section 3.7.2) for greater capacity impact.

In the default sparse configuration, LoRA is applied to $W_Q$ and $W_O$ on $n_s$ heads across $L_s$ layers.

**Parameter cost per expert (4 heads, 4 layers, rank-8, Q+O only):** $2 \times 4 \times 4 \times 2 \times d_h \times r \times p$. For $d_h = 96$, $r=8$: **~96KB**.

### 3.7.2 FFN LoRA (For High-Capacity Configurations)

When the per-expert budget permits, LoRA on the SwiGLU FFN projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) provides the primary vector for knowledge capacity expansion. FFN layers are unary operations (processing one token at a time), making per-expert routing mathematically clean—there is no cross-token interaction as in attention.

For a high-capacity configuration with rank $r$ across all $L$ layers on all three FFN projections:

$$
\text{Params/expert} = L \times 3 \times r \times (d + d_{ff}) \times p
$$

For $L=32$, $d=3072$, $d_{ff}=8192$, $r=96$: **~108M params ≈ 200MB per expert at BF16**. This is the budget used in the LEC-Full configuration (Section 12.3).

**LoRA switching cost analysis:** At H100 HBM bandwidth of 3.35 TB/s, reading one expert’s LoRA weights (96KB–200MB depending on configuration) takes $\approx 0.03\mu s$ to $60\mu s$. A single token generation step takes 1–10 ms. Therefore LoRA switching overhead is $< 0.01\%$ (sparse) to $< 6\%$ (full FFN rank-96) of step time.

### 3.8 Complete Architecture Summary

| Component | Params/Expert | HBM Cost | Offloadable? | Nature of Modification |
| --- | --- | --- | --- | --- |
| Per-Layer Embeddings | ~16 KB | 0 | Yes (CPU/flash) | Additive → multiplicative through attention |
| Expert Prefix Tokens (m=10) | ~60 KB | ~60 KB | No (needed in KV cache) | Content-adaptive attention bias **[⚠ remove for deployment; see §3.3]** |
| Value Residual λ + δ | ~192 KB | ~192 KB | Partially | Controls token identity propagation |
| Per-Expert LayerNorm/FiLM γ, β | ~768 KB | ~768 KB | No | Multiplicative gating of sub-circuits |
| Canon Kernels (AC, scaled) | ~768 KB | ~768 KB | Partially | Per-expert local horizontal mixing |
| Asymmetric LoRA ($W_Q$, $W_O$ only; 4h, 4L, r=8) | ~96 KB | ~96 KB | Yes | Expert-specific attention queries/projections |
| **Total per expert (LEC-Lite)** | **~1.9 MB** | **~1.9 MB** |  |  |
| **Total for 10 experts** | **~19 MB** | **~19 MB** |  |  |
| **As % of 3B base model (~6 GB)** | **0.32%** | **0.32%** |  |  |

For comparison, a traditional MoE with 10 full FFN experts on a 3B model would add ~20B parameters (~40 GB)—**a 2000× overhead ratio** compared to our LEC-Lite approach.

---

> **Note on LEC-Full:** When targeting capacity expansion (Section 12.3), the per-expert budget increases to ~200MB by adding rank-96 LoRA on all FFN projections across all layers. In this configuration, 6 experts on a 4B base model fit within ~11.2 GB HBM (including KV cache), targeting consumer GPUs with 24 GB VRAM.
> 

## 4. Theoretical Expressivity Analysis

### 4.1 Framework 1: Linear Algebra — Jacobian Rank Analysis

We analyze the first-order effect of expert conditioning through the network. Define $g_l(\cdot)$ as the layer $l$ function with Jacobian $J_l = \frac{\partial g_l}{\partial h}\big|_h$.

After PLE injection $e_l$ at layer $l$, the first-order perturbation to layer $l$’s output is:

$$
\Delta h_l \approx J_l \cdot e_l
$$

The total perturbation to the final hidden state, with PLE injected at every layer, is:

$$
\Delta h_L \approx \sum_{l=1}^{L} \left(\prod_{i=L}^{l+1} J_i\right) J_l e_l
$$

Each summand is a $d$-dimensional vector. With $L$ layers, the first-order perturbation space has dimension:

$$
\dim(\text{span}) = \min(L \times d_{\text{ple}}, d)
$$

For $L = 32$, $d_{\text{ple}} = 256$: $32 \times 256 = 8192 > d = 3072$. The perturbation can span the **entire** output space even at first order, with massive redundancy.

Beyond first order, the bilinear structure of attention introduces cross terms. After PLE injection, the attention logits become:

$$
QK^T = (hW_Q + eW_Q)(hW_K + eW_K)^T
$$

The expansion yields the original term $hW_QW_K^Th^T$ plus cross terms $hW_QW_K^Te^T + eW_QW_K^Th^T$ (multiplicative interaction between PLE and content) plus a PLE self-interaction term. These cross terms are followed by softmax (highly nonlinear) and value weighting. The result is that PLE’s effect is **not** simply additive at the output—it is a genuinely different computation path.

After $L$ layers of such nonlinear interactions, the effective expressivity grows **exponentially** with depth, not linearly.

**LoRA comparison:** Rank-$r$ LoRA on one weight matrix provides a rank-$r$ perturbation to that layer’s output. Across 4 projections and $L$ layers, compositions can achieve rank $d$. PLE and LoRA have comparable first-order expressivity at the output. The difference is that LoRA modifies the transformation (input-dependent effect ab initio), while PLE modifies the input to the transformation (gaining input-dependence through subsequent nonlinearities).

**Expert tokens add an adaptive rank-$m$ contribution** per head per layer. With $m = 10$ expert tokens and 32 heads, this gives up to rank 320 per layer, but crucially the weights (attention scores) are input-dependent—making this more expressive per parameter than LoRA.

### 4.2 Framework 2: Information Theory — Channel Capacity and Leverage

We quantify how much “information” each conditioning mechanism injects.

**PLE channel capacity:**

$$
C_{\text{PLE}} = L \times d_{\text{ple}} \times p = 32 \times 256 \times 16 = 131,072 \text{ bits} \approx 16 \text{ KB/expert}
$$

**Expert token capacity:**

$$
C_{\text{tokens}} = m \times d \times p = 10 \times 3072 \times 16 = 491,520 \text{ bits} \approx 60 \text{ KB/expert}
$$

**LoRA capacity (sparse setup):**

$$
C_{\text{LoRA}} = 4 \times 4 \times 4 \times 2 \times d_h \times r \times p = 1,572,864 \text{ bits} \approx 192 \text{ KB/expert}
$$

**Combined capacity:** ~268 KB of raw expert-specific information.

However, raw bits do not equal effective capacity. The key concept is **information leverage**: PLE bits injected at every layer get amplified by nonlinear processing. Each bit of early-layer PLE can influence exponentially many bits of the final representation.

If each layer amplifies the PLE effect by factor $\gamma > 1$ (from attention’s multiplicative structure), PLE at layer $l$ has effective influence $\gamma^{L-l}$. For modest $\gamma = 1.1$ and $L = 32$:

$$
\gamma^{32} \approx 21\times \text{ amplification for layer-1 PLEs}
$$

The total effective information is therefore significantly larger than the raw parameter count, especially for early-layer conditioning signals. This is consistent with empirical findings from Engram and Gemma 3n where relatively small embedding additions produce large quality improvements.

### 4.3 Framework 3: Function Space Geometry — Covering Numbers

Consider the set of functions reachable by varying expert conditioning:

$$
\mathcal{F}_\theta = \{f(\cdot; \theta, c) : c \in \mathcal{C}\}
$$

where $\theta$ are shared base weights and $\mathcal{C}$ is the space of all valid conditionings.

The dimensionality of the reachable manifold in function space equals the number of free conditioning parameters ($d_c \approx 330K$ per our design). For full MoE, this is $\sim 50M$ per layer for FFN weights alone—roughly 150× larger per expert.

However, the **curvature** of our manifold compensates for its lower dimensionality. The softmax nonlinearity in attention creates high curvature near “critical” inputs where heads are close to switching their dominant key. At these points, small PLE perturbations create large changes in information flow.

The covering number of the reachable set scales as:

$$
\log \mathcal{N}(\mathcal{F}_\theta, \epsilon) = \Omega(d_c \cdot L \cdot \log(1/\epsilon))
$$

The factor of $L$ arises from depth-distributed injection—PLE at each layer adds new “directions” of functional variation. For full MoE, the corresponding bound is $\Omega(d_{\text{MoE}} \cdot \log(1/\epsilon))$ without the $L$ factor. This depth-amplification partially compensates for fewer parameters per layer.

### 4.4 Expressivity of the Attention Interaction (Detailed Derivation)

We provide a more detailed derivation of why PLE is not purely additive through attention. At layer $l$, let the input be $h$ and PLE be $e$ (dropping subscripts for clarity). After injection, $\tilde{h} = h + e$.

**Queries and Keys:**

$$
Q = \tilde{h}W_Q = hW_Q + eW_Q
$$

$$
K = \tilde{h}W_K = hW_K + eW_K
$$

**Attention logits (single head):**

$$
A = \frac{QK^T}{\sqrt{d_h}} = \frac{1}{\sqrt{d_h}}\left[hW_QW_K^Th^T + hW_QW_K^Te^T + eW_QW_K^Th^T + eW_QW_K^Te^T\right]
$$

**Attention weights**

after softmax:

$$
\alpha = \text{softmax}(A)
$$

The softmax is a highly nonlinear function. Even if $\|eW_Q\|$ is small relative to $\|hW_Q\|$, the attention weights $\alpha$ can change dramatically near “decision boundaries” where two keys have similar dot products with the query. This is the mechanism by which small PLE perturbations create large behavioral changes.

**Values:**

$$
V = \tilde{h}W_V = hW_V + eW_V
$$

**Attention output:**

$$
\text{out} = \alpha \cdot V = \alpha \cdot (hW_V + eW_V)
$$

Since $\alpha$ is a nonlinear function of $e$ (through softmax), and $V$ is a linear function of $e$, the output is a product of a nonlinear and a linear function of $e$—this is **not** representable as $\text{out}_{\text{original}} + f(e)$ for any function $f$. It is genuinely a different computation.

**After LayerNorm and FFN,** the nonlinearity compounds further. The GeLU/SiLU activation in the FFN means that even small shifts in input (from PLE) can cross activation thresholds, turning neurons on or off. After $L$ layers of such compounding, two experts with different PLE vectors will produce representations in **qualitatively different** regions of the hidden state space.

### 4.5 Capacity vs. Expressivity: The Shannon Bound and the Knowledge Gap

The analysis in Sections 4.1–4.4 establishes that LEC’s conditioning stack provides rich *expressivity*—the ability to reach diverse regions of function space. However, expressivity and *knowledge capacity* are distinct properties, and we must address this gap honestly.

### The Information-Theoretic Constraint

The Shannon entropy $H(W)$ of a neural network’s weights provides an upper bound on its factual knowledge retention. A model with $N$ parameters at $p$-bit precision can store at most $N \times p$ bits of information. The ~2 MB of per-expert conditioning in LEC-Lite encodes at most $\sim 16 \times 10^6$ bits of expert-specific information. This is sufficient to specify *how* the shared model should behave differently (routing specialization), but it cannot encode substantial *new factual knowledge* beyond what the base model already contains.

For context, the distinction maps to two qualitatively different goals:

**Behavioral specialization** (what LEC-Lite targets): Different experts process inputs using different computational strategies—one expert might favor syntactic patterns, another might favor entity-centric attention. The base model’s knowledge is the same; the *access pattern* differs. Shannon requirement: low (routing decisions and transformation perturbations occupy a small information subspace).

**Knowledge capacity expansion** (what standard MoE achieves): Each expert stores distinct factual associations in its FFN weights—Expert A memorizes entity embeddings for biology, Expert B for legal text. Total factual knowledge scales with $K \times$ FFN parameters. Shannon requirement: high (proportional to the number of independent facts stored).

### Quantifying the Gap

A 4B parameter dense model at BF16 stores ~8 GB = $6.4 \times 10^{10}$ bits of information. A standard MoE with 6 full FFN experts adds ~$4 \times$ this capacity. LEC-Lite’s 2 MB/expert adds ~$10^7$ bits—a factor of $\sim 6000\times$ less than full MoE per expert.

The question is not whether this gap exists (it does), but whether it matters for the target performance range. Our goal is not to match $K \times$ base capacity (e.g., 24B-equivalent for 6 experts on a 4B base). We target the 8–16B effective capacity range: meaningful improvement over the dense baseline through better utilization of existing knowledge plus modest capacity expansion.

### The Scaling Vector: Per-Expert Parameter Budget

The per-expert parameter budget is the primary lever for trading off between LEC’s two operating points:

At the **LEC-Lite** end (~2 MB/expert), nearly all capacity comes from the shared base model. Expert conditioning provides behavioral specialization only. This is the regime where our expressivity analysis (Sections 4.1–4.4) is most relevant: small perturbations, amplified nonlinearly through depth, create meaningfully different processing pipelines over the same knowledge base.

At the **LEC-Full** end (~200 MB/expert), high-rank LoRA on FFN projections provides genuine per-expert capacity. For a 4B model with $d=3072$, $d_{ff}=8192$, rank-96 LoRA on all three SwiGLU projections across 32 layers yields ~108M parameters per expert. Six experts contribute ~648M additional parameters, bringing the effective parameter count to ~4.65B—a meaningful capacity boost that targets the 8–12B performance range through a combination of added parameters and expert specialization.

### Relationship to Gemma 3n and Engram

It is important to clarify the relationship between our PLE mechanism and Gemma 3n’s Per-Layer Embeddings. Gemma 3n’s PLE is dimensioned as $V \times d \times L$ (Vocabulary $\times$ Dimension $\times$ Layers), yielding ~2.35B parameters (~4.7 GB at BF16) that are streamed from CPU. This is fundamentally a per-token lookup table that *replaces* the standard embedding, providing massive knowledge capacity via offloaded storage.

Our PLE is architecturally different: it is a per-*expert* (not per-token) conditioning signal, with $K \times L \times d_{\text{ple}}$ total parameters. Our PLE vectors condition the *computation* rather than encoding token-level knowledge. The information-theoretic requirements are therefore orders of magnitude smaller, and the expressivity analysis of Section 4.1 (showing that depth-distributed injection spans the full output perturbation space) applies to our usage but not to Gemma 3n’s.

Similarly, DeepSeek’s Engram achieves knowledge capacity expansion through a 100B-parameter N-gram embedding table offloaded to host memory—an approach complementary to LEC (see Section 8.2) but operating in a fundamentally different capacity regime.

---

## 5. Hardware Cost Analysis

### 5.1 Memory Budget

For a 3B parameter dense model at BF16 precision:

| Component | Size | Location |
| --- | --- | --- |
| Base model weights | ~6.0 GB | HBM |
| KV cache (seqlen=2048, batch=1) | ~256 MB | HBM |
| Expert conditioning (10 experts, all mechanisms) | ~20 MB | HBM (or partially CPU) |
| Activations during inference | ~100 MB | HBM |
| **Total** | **~6.4 GB** |  |
| Dense model baseline | ~6.35 GB | HBM |
| **Overhead** | **~0.8%** |  |

For comparison, a standard 10-expert MoE (full FFN duplication) on the same base model would require ~26 GB—a 4× increase.

### 5.2 Bandwidth Analysis During Token Generation

At each decoding step on H100 (3.35 TB/s HBM bandwidth):

| Operation | Data Movement | Time | % of Step |
| --- | --- | --- | --- |
| Read base weights (3B params) | 6 GB | 1.79 ms | ~95% |
| Read KV cache | ~256 MB | 76 μs | ~4% |
| Read active expert conditioning | ~2 MB | 0.6 μs | ~0.03% |
| Router computation | negligible | ~1 μs | ~0.05% |
| **Total expert overhead** |  | **~1.6 μs** | **~0.08%** |

The expert conditioning read is completely hidden behind the base weight read. **Expert switching is free.**

### 5.3 Comparison with Full MoE Inference

In standard top-2 MoE with full FFN experts, each decoding step reads 2 full expert FFNs instead of 1 shared FFN. For a 3B model where FFN constitutes ~67% of parameters:

| Architecture | Effective Weight Reads | Time | Speedup |
| --- | --- | --- | --- |
| Dense 3B | 6 GB | 1.79 ms | — |
| MoE 3B (10 experts, top-2) | ~10 GB (shared attn + 2 FFN experts) | 2.99 ms | 0.60× (slower) |
| LEC 3B (10 experts, top-1) | ~6.002 GB (shared + conditioning) | 1.79 ms | 1.00× (same as dense) |

LEC inference speed is **virtually identical to dense model inference**, because the base weights (the bandwidth bottleneck) are read exactly once regardless of expert selection.

### 5.4 Kernel Fusion and CUDA Scheduling Considerations

The bandwidth analysis above shows that expert conditioning data transfer is negligible. However, memory bandwidth is not the only constraint on inference speed—**kernel launch overhead** is a separate bottleneck when multiple small operations must be dispatched sequentially.

In standard PyTorch eager execution, each conditioning mechanism triggers a separate CUDA kernel launch. A naive implementation applying PLE injection, per-expert LayerNorm, value residual scaling, Canon convolution, and LoRA multiplication would launch 5–6 additional small kernels per layer. On an H100, each kernel launch incurs ~5–10 μs of CPU-side scheduling overhead. Across 32 layers, this accumulates to $32 \times 5 \times 7.5\mu s \approx 1.2\text{ ms}$—a significant fraction of the ~1.8 ms token generation step.

**This is a deployment engineering concern, not a research validity concern.** All MoE research papers, including Mixtral and DeepSeek-V3, initially validate with research-grade code before optimized serving kernels are developed. Nevertheless, we identify the kernel fusion path for each mechanism:

| Mechanism | Fusion Strategy | Difficulty |
| --- | --- | --- |
| PLE (additive bias) | Trivially fused into existing LayerNorm or attention kernels as a vector add | Easy |
| Per-Expert LayerNorm/FiLM | Fused into existing LayerNorm kernel (just different γ, β) | Easy |
| LoRA (attention + FFN) | Grouped GEMM / Segmented MatMul (validated by S-LoRA, Punica) | Medium |
| Value Residual | Requires modification of attention kernel internals | Medium |
| Canon Convolutions | Separate 1-D causal conv; harder to fuse with MatMul-dominated graph | Hard |
| Expert Prefix Tokens | Requires per-token attention mask modification | Hard (+ KV cache concerns) |

The **recommended deployment configuration** (PLE + FiLM + Asymmetric LoRA) uses only mechanisms rated “Easy” or “Medium” for kernel fusion, and can be served via existing Grouped GEMM infrastructure (S-LoRA, LoRAX/Punica, vLLM). The full research configuration includes all mechanisms for ablation purposes, where kernel launch overhead is acceptable during training and evaluation.

---

## 6. Training Methodology

### 6.1 Overview

We propose a four-phase training pipeline. All expert-specific components are initialized near-identity/near-zero to ensure the model starts as a valid dense model.

### 6.2 Phase 0: Preparation

Starting from a pretrained dense checkpoint:

### 6.2.1 The Symmetry Breaking Problem

A critical challenge in training experts that share a common backbone is *expert collapse due to symmetric initialization*. If all expert conditioning parameters are initialized near-identity (zero LoRA $B$, unit LayerNorm γ, zero PLE), then on the first forward pass every expert produces the same output, the loss is identical across experts, and backpropagated gradients to each expert’s parameters are also identical. Even with small random noise, the gradient signal distinguishing experts is proportional to the noise magnitude—far weaker than the shared gradient, which can wash out differentiation for many training steps.

This is a known failure mode in MoE training. Standard MoE upcycling (Komatsuzaki et al., 2022) mitigates it by starting from full FFN copies that are already large and diverse enough for random noise to produce measurable output differences. In our setting, with ~2 MB of conditioning per expert, near-zero initialization produces near-zero differentiation signal, making the router effectively blind during early training.

### 6.2.2 Orthogonal Subspace Initialization (Recommended)

To guarantee maximal gradient signal from the first training step, we initialize expert LoRA matrices to occupy **strictly orthogonal subspaces** in weight space. Specifically, for each LoRA adapter location, we construct the set of $K$ expert matrices $\{A_k B_k\}_{k=1}^K$ such that:

$$
\langle A_i B_i, A_j B_j \rangle_F \approx 0 \quad \text{for all } i \neq j
$$

where $\langle \cdot, \cdot \rangle_F$ is the Frobenius inner product. This is achieved by:

1. Sample a random matrix $M \in \mathbb{R}^{d \times d}$ and compute its SVD: $M = U \Sigma V^T$.
2. Partition the top $K \times r$ singular vectors into $K$ groups of $r$ consecutive vectors.
3. For expert $k$, set $A_k = U[:, kr:(k+1)r]$ (scaled by $\sqrt{\sigma}$) and $B_k = V[:, kr:(k+1)r]^T$ (scaled by $\sqrt{\sigma}$), where $\sigma$ are the corresponding singular values.

This ensures that on the very first forward pass, routing a token to Expert $i$ vs. Expert $j$ computes **mathematically disjoint** perturbations to the hidden state. The router receives high-signal loss gradients immediately, and expert divergence begins at step 1 rather than after hundreds or thousands of steps of waiting for random noise to amplify.

For non-LoRA conditioning parameters, the orthogonality principle extends naturally:
- **PLE vectors:** Initialize from orthogonal random bases (e.g., columns of a random orthogonal matrix), ensuring each expert shifts the hidden state in a maximally different direction.
- **LayerNorm γ:** Initialize as $1 + \epsilon_k$ where $\epsilon_k$ vectors are drawn from orthogonal subspaces.
- **Value residual λ:** Space uniformly across the range [0.1, 0.5] to enforce different information flow profiles from step 1.

This approach is related to orthogonal initialization (Saxe et al., 2014) and the “virtual group” initialization of He et al. (2024), but applied specifically to per-expert conditioning in a shared-backbone MoE context.

### 6.2.3 Near-Identity Initialization (Alternative)

An alternative, simpler initialization preserves the property that the model starts as a valid dense model (all experts produce identical outputs at $T=0$):

1. Add all per-expert parameters with initialization:
    - PLE vectors: $\mathcal{N}(0, 0.01)$
    - Expert tokens: Initialized from random token embeddings in the vocabulary with small noise
    - Value residual $\lambda$: Initialized to a shared default (e.g., 0.3) with per-expert noise $\epsilon \sim \mathcal{N}(0, 0.01)$
    - Value residual $\delta$: $\mathcal{N}(0, 0.001)$
    - LayerNorm $\gamma$: $1 + \mathcal{N}(0, 0.001)$; $\beta$: $\mathcal{N}(0, 0.001)$
    - Canon kernels: Initialized to approximate identity (weight on $t$ ≈ 1, weights on $t-1, t-2, t-3$ ≈ 0) with per-expert noise
    - LoRA $A$: Kaiming normal; $B$: zero (standard LoRA init)
2. Initialize router randomly or from domain classifiers.

**Tradeoff:** Near-identity initialization guarantees zero loss increase at $T=0$ (the model is exactly the pretrained dense model), but requires Phase 1 (router warmup) to compensate for the weak initial gradient signal. Orthogonal initialization introduces a small loss perturbation at $T=0$ (since experts now produce different outputs), but eliminates the need for supervised router warmup and produces faster expert divergence.

**Recommendation:** Use orthogonal subspace initialization as the primary approach, with near-identity as a fallback if the initial loss perturbation is unacceptably large (e.g., at very large model scales where training stability is more fragile). The ablation study (Section 7) should compare both initialization strategies.

### 6.3 Phase 1: Router Warmup (5–10% of Total Training, Optional with Orthogonal Init)

**Objective:** Bootstrap the router with meaningful expert assignments before expert specialization begins.

**When required:** With near-identity initialization (Section 6.2.3), expert outputs are nearly identical, so the router receives negligible gradient signal from the output loss. Supervised warmup is essential in this case.

**When optional:** With orthogonal subspace initialization (Section 6.2.2), expert outputs differ meaningfully from step 1, and the router can learn from output loss gradients directly. Router warmup may still accelerate convergence but is no longer critical for avoiding collapse.

**Procedure (when used):**
- Freeze all model parameters including expert-specific params.
- Train only the router.
- Use a diverse calibration dataset with domain labels (e.g., code, math, creative writing, factual QA, multilingual, etc.).
- For each batch, hard-assign domains to experts using a predefined mapping.
- Train the router via cross-entropy against these hard assignments.

**Rationale:** With near-identical experts, the router receives almost zero gradient signal from output loss. Supervised bootstrapping gives the router a meaningful initialization, which subsequent phases can refine.

### 6.4 Phase 2: Expert Specialization (75–85% of Total Training)

**Objective:** Train expert-specific parameters to develop genuine specialization.

**Procedure:**
- Freeze base model weights.
- Train router + all expert-specific parameters jointly.
- Use auxiliary-loss-free load balancing (per-expert bias terms on router logits, adjusted based on recent expert load, following DeepSeek-V3’s approach).
- Per-parameter-group learning rates:

| Parameter Group | Relative LR | Rationale |
| --- | --- | --- |
| Router | 1.0× | Needs fastest adaptation |
| Expert tokens | 0.5–1.0× | Standard embedding LR |
| PLE vectors | 0.5× | Similar to embeddings, many params |
| Value residual λ (scalars) | 0.1× | Very sensitive; small changes have large effect |
| Value residual δ (vectors) | 0.3× | Modifies attention output directly |
| LayerNorm γ, β | 0.3× | Multiplicative; sensitive |
| Canon kernels | 0.3× | Small convolutions; sensitive |
| LoRA A, B matrices | 0.3× | Standard LoRA guidance |
- Optimizer: AdamW with weight decay 0.01, applied to all expert params except biases and LayerNorm.
- Data: Standard pretraining distribution (e.g., SlimPajama, FineWeb) without explicit domain labels.

### 6.5 Phase 3: Joint Fine-Tuning (5–10% of Total Training, Optional)

**Objective:** Allow the shared backbone to adapt to the MoE structure.

**Procedure:**
- Unfreeze base model weights with very small LR (0.01–0.05× base).
- Continue training all parameters.
- Monitor for catastrophic forgetting by tracking perplexity on held-out validation data.

### 6.6 Phase 4: Instruction Tuning / RLHF (Standard)

After MoE conversion, apply standard post-training as for any language model. The expert conditioning should be maintained during post-training.

### 6.7 Training Data Budget

Based on findings from sparse upcycling and LoRA literature:
- Sparse Upcycling found that ~50% of original pretraining compute suffices for strong results.
- LoRA-based methods typically need much less data.
- We recommend: **10–50B tokens for a 3B model**, adjustable based on convergence monitoring.

### 6.8 Load Balancing Strategy

We specifically recommend against auxiliary loss-based load balancing for this architecture, because:
1. Experts are so similar that auxiliary gradients can dominate the (weak) specialization gradient signal.
2. Load imbalance in LEC is less harmful than in standard MoE—all experts share the same base compute, so imbalance doesn’t create idle hardware.

Instead, use DeepSeek-V3’s approach: per-expert bias terms on router logits, adjusted based on exponential moving average of expert utilization. This is gradient-free and does not interfere with the learning signal.

---

## 7. Experimental Design and Ablation Studies

### 7.1 Baselines

| Model | Description | Params in HBM |
| --- | --- | --- |
| Dense-3B | Original pretrained 3B model, continued training | 6 GB |
| Dense-3B-Extended | Same model, trained on same total tokens as LEC | 6 GB |
| MoE-3B-Upcycled | Standard sparse upcycling (Komatsuzaki et al., 2022), 8 experts top-2 | ~18 GB |
| MoE-3B-D2DMoE | Partitioning-based conversion (D2DMoE), 8 experts | ~18 GB |
| LoRA-MoE-3B | Full LoRA-MoE (MoLoRA-style), 8 experts, rank-64 on all layers | ~7.5 GB |
| LEC-3B | Our full method, 10 experts top-1 | ~6.02 GB |

### 7.2 Evaluation Benchmarks

**Language understanding:** MMLU, MMLU-Pro, ARC-Challenge, HellaSwag, WinoGrande.

**Reasoning:** GSM8K, MATH, BBH (Big Bench Hard).

**Code:** HumanEval, MBPP.

**Knowledge:** TriviaQA, Natural Questions.

**Long context:** RULER benchmark, Needle-in-a-Haystack.

**Expert diversity metrics:** Expert utilization entropy, pairwise expert output cosine similarity, domain-conditioned expert preference (measure whether certain input domains consistently activate certain experts).

### 7.3 Ablation Study Design

We design a systematic ablation across 7 configurations to isolate each mechanism’s contribution:

| Config | PLE | Expert Tokens | Value Residual | LayerNorm/FiLM | Canon | Asymmetric LoRA | Expected Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | — | — | — | — | — | — | Dense baseline (no experts) |
| A1 | ✓ | — | — | — | — | — | Isolate PLE contribution |
| A2 | — | ✓ | — | — | — | — | Isolate expert token contribution **[research only; see §3.3]** |
| A3 | — | — | ✓ | — | — | — | Isolate value residual contribution |
| A4 | — | — | — | ✓ | — | — | Isolate LayerNorm/FiLM contribution |
| A5 | — | — | — | — | ✓ | — | Isolate Canon contribution |
| A6 | — | — | — | — | — | ✓ | Isolate asymmetric LoRA ($W_Q$/$W_O$) contribution |
| A7 | ✓ | ✓ | — | — | — | — | PLE + tokens synergy |
| A8 | ✓ | — | ✓ | ✓ | — | — | Deployment-friendly stack (no tokens, no Canon) |
| A9 | ✓ | — | ✓ | ✓ | ✓ | — | Full stack minus LoRA |
| A10 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full LEC (all mechanisms) |
| A11 | ✓ | — | ✓ | ✓ | — | ✓ | **Recommended deployment config** (PLE + FiLM + ValRes + LoRA) |

> **Initialization ablation:** Configurations A0–A11 should each be run with both orthogonal subspace initialization (Section 6.2.2) and near-identity initialization (Section 6.2.3) to measure the impact of symmetry breaking strategy. At minimum, configs A0, A6, A10, and A11 should be compared under both initialization schemes.
> 

Each ablation should report validation perplexity, downstream benchmark scores, expert utilization statistics, and wall-clock training/inference time.

### 7.4 Scaling Experiments

To validate scaling behavior, test LEC across multiple model sizes:

| Base Model | # Experts | Expert Params/Expert | Total Overhead |
| --- | --- | --- | --- |
| 125M | 4, 8, 16 | ~100 KB | 0.1–0.4% |
| 350M | 4, 8, 16 | ~300 KB | 0.1–0.3% |
| 1.3B | 8, 16 | ~1 MB | 0.05–0.1% |
| 3B | 8, 10, 16 | ~2 MB | 0.05–0.1% |
| 7B+ | 8, 16 | ~4 MB | 0.03–0.06% |

Key questions: Does LEC’s relative benefit grow or shrink with model scale? Is there a critical model size below which the shared weights aren’t rich enough to support meaningful expert differentiation?

### 7.5 Expert Count and Routing Experiments

Test the interaction between number of experts and top-k routing:

| Configuration | Description |
| --- | --- |
| K=4, top-1 | Few experts, hard routing |
| K=8, top-1 | Moderate experts, hard routing |
| K=16, top-1 | Many experts, hard routing |
| K=8, top-2 | Moderate experts, soft routing (combine 2 experts) |
| K=16, top-2 | Many experts, soft routing |
| K=8, soft-merge | All experts weighted by router, no hard selection |

Since LEC experts are so lightweight, soft merging (activating all experts with different weights) is computationally feasible and may outperform hard routing.

### 7.6 Domain Specialization Analysis

To assess whether experts genuinely specialize:

1. Run diverse evaluation data through the model and record router decisions.
2. Compute the mutual information $I(\text{expert}; \text{domain})$ — should be significantly above zero.
3. Visualize expert activation heatmaps across domains (code, math, creative writing, factual QA, multilingual).
4. Compute pairwise expert output divergence: for the same input, swap experts and measure KL divergence of output distributions. Higher divergence = more specialized experts.
5. Measure attention pattern differences: for the same input with different experts, compare attention entropy and attention to expert tokens.

---

## 8. Additional Ideas and Extensions

### 8.1 Hierarchical Expert Conditioning

Instead of flat routing to $K$ experts, organize experts hierarchically:
- Level 1: Route to 4 “meta-experts” (e.g., code, math, language, multimodal).
- Level 2: Within each meta-expert, route to 4 sub-experts (e.g., within language: formal, casual, creative, technical).

Each level adds its own conditioning stack. This allows coarse-grained specialization (from level 1 PLE/tokens) combined with fine-grained specialization (from level 2).

### 8.2 Dynamic Expert Conditioning via Engram-Style Memory

Integrate Engram-style N-gram conditional memory as an additional per-expert signal:
- Each expert has its own N-gram embedding table (or shared table with per-expert gating).
- Static knowledge patterns are routed to different experts based on the N-gram memory.
- The Engram module’s deterministic addressing allows asynchronous prefetch from CPU, perfectly complementing our offloading strategy.

This is particularly compelling because Engram and MoE experts address different types of specialization: Engram separates static knowledge from dynamic reasoning, while our expert conditioning separates different “modes” of processing.

### 8.3 Conditional Expert Activation

Since our experts are so cheap, we can activate more than 2 per token without significant cost:
- For simple tokens (e.g., common function words), activate 1 expert.
- For ambiguous or complex tokens, activate 2–3 experts and soft-merge.
- Use the router’s confidence (entropy of the routing distribution) to dynamically decide how many experts to activate.

This is infeasible with standard MoE (each additional expert activation doubles FFN compute) but nearly free with LEC.

### 8.4 Expert Composition via Arithmetic

Because our expert conditioning is additive/modular, we can potentially compose experts at inference time:
- “Expert 3 + 0.5 × Expert 7” = apply a weighted sum of their PLE vectors, expert tokens, etc.
- This creates a continuous expert space rather than a discrete set.
- Could be useful for fine-grained control (e.g., “70% formal + 30% creative” style blending).

### 8.5 Per-Token Expert Switching vs. Per-Sequence

Standard MoE routes per-token. But with LEC, we have additional design choices:
- **Per-token routing:** Maximum flexibility, standard MoE behavior. Expert conditioning switches every token.
- **Per-segment routing:** Group tokens into segments (sentences, paragraphs) and route segments to experts. More stable behavior, easier training.
- **Per-sequence routing:** One expert per entire sequence. Simplest, but potentially wasteful.

Experiment with all three and measure the quality-stability tradeoff.

### 8.6 Learnable Expert Mixing for Residual Streams

For each residual connection in the transformer, add a per-expert scalar that controls the residual-to-transformation ratio:

$$
h_{l+1} = \alpha_l^{(k)} \cdot h_l + (1 - \alpha_l^{(k)}) \cdot f_l(h_l)
$$

This gives experts control over the effective depth of the network—some experts might “skip” certain layers (high $\alpha$) while others use them fully (low $\alpha$). Cost: one scalar per layer per expert = 640 bytes total for 10 experts over 32 layers.

### 8.7 Attention Head Pruning as Expert Differentiation

Rather than adding parameters, experts could *remove* parameters by pruning attention heads differently:
- Each expert has a binary mask over attention heads per layer.
- Different experts attend with different subsets of heads.
- Cost: zero additional parameters; the “expert identity” is encoded in the mask.
- Combines naturally with sparse LoRA on the remaining active heads.

---

## 9. Potential Pitfalls and Risk Mitigation

### 9.1 Risk: Router Collapse

**Description:** The router fails to differentiate near-identical experts and converges to always selecting the same expert(s).

**Probability:** HIGH — this is the most likely failure mode.

**Detection:** Monitor expert utilization entropy. If it drops below $\log(K) \times 0.5$, router collapse is occurring.

**Mitigation:**
1. **Orthogonal subspace initialization** (Section 6.2.2) — ensures experts produce distinct outputs from step 1, providing the router with high-signal gradients immediately.
2. Router warmup phase with supervised domain-expert assignments (Phase 1) — serves as either the primary mitigation (with near-identity init) or an accelerant (with orthogonal init).
3. Auxiliary-loss-free load balancing via DeepSeek-V3’s bias adjustment.
4. If collapse persists: temporarily increase noise in router logits (dropout on router), or enforce uniform routing for a fraction of training steps.

### 9.2 Risk: Expressivity Ceiling

**Description:** The lightweight conditioning mechanisms are insufficient for meaningful expert specialization, and LEC converges to essentially the same function regardless of expert.

**Probability:** MEDIUM — mathematical analysis suggests sufficient expressivity exists, but empirical validation is needed.

**Detection:** Measure pairwise expert output KL divergence on validation data. If KL < 0.01 nats for all expert pairs, the expressivity ceiling has been hit.

**Mitigation:**
1. Increase PLE dimension.
2. Add more expert tokens (research config only; see KV cache concerns in Section 3.3).
3. Increase LoRA rank or apply LoRA to more heads/layers.
4. Add per-expert Canon kernels.
5. **Scale to LEC-Full** (Section 12.3): add rank-96 FFN LoRA, providing ~108M additional parameters per expert. As established in Section 4.5, the expressivity ceiling for LEC-Lite is fundamentally a *capacity* ceiling—when behavioral specialization is insufficient, genuine per-expert parameter capacity (via high-rank FFN LoRA) is the correct response.

### 9.3 Risk: Gradient Interference on Shared Weights

**Description:** During Phase 3 (joint fine-tuning), different experts push shared weights in conflicting directions, causing slow convergence or oscillation.

**Probability:** LOW–MEDIUM — the expert conditioning params should absorb most expert-specific gradients.

**Detection:** Monitor gradient variance across experts for shared weight parameters. If $\text{Var}[\nabla_\theta \mathcal{L}_k]$ across experts $k$ is large relative to $\|\mathbb{E}[\nabla_\theta \mathcal{L}_k]\|$, interference is occurring.

**Mitigation:**
1. Simply skip Phase 3 (freeze base weights entirely). Our framework is designed to work without modifying base weights.
2. If Phase 3 is desired: use gradient scaling that reduces the contribution of high-variance expert gradients.
3. GradDrop: randomly drop expert gradient contributions with probability proportional to their disagreement with the mean gradient.

### 9.4 Risk: Expert Token Attention Domination

**Description:** Content tokens attend too strongly to expert tokens, overwhelming the actual content-based attention patterns and reducing model quality.

**Probability:** LOW — prefix tuning literature suggests this doesn’t happen with reasonable numbers of prefix tokens.

**Detection:** Monitor the fraction of total attention weight allocated to expert tokens vs. content tokens.

**Mitigation:**
1. Reduce number of expert tokens ($m$).
2. Add a learnable temperature on expert token attention logits.
3. Initialize expert tokens to produce small attention scores (near-zero initialization).

### 9.5 Risk: Training Instability from Multiple Conditioning Signals

**Description:** The six conditioning mechanisms interact in unpredictable ways, causing training instability (loss spikes, NaN gradients).

**Probability:** LOW — each mechanism is initialized near-identity, so the model starts as a valid dense model.

**Detection:** Monitor training loss, gradient norms per parameter group, and LayerNorm statistics.

**Mitigation:**
1. Ablate: if instability occurs, disable mechanisms one by one to identify the culprit.
2. Use gradient clipping per parameter group (not global clipping).
3. Reduce learning rates for the offending mechanism.
4. Introduce mechanisms gradually (e.g., start with only PLE, add other mechanisms after 10% of training).

### 9.6 Risk: Undertraining of Expert Parameters

**Description:** The expert-specific components are so lightweight that they need disproportionately more data/steps to develop meaningful specialization.

**Probability:** MEDIUM — this is a real concern given the small parameter count.

**Detection:** Track expert specialization metrics (pairwise divergence, domain mutual information) over training. If these plateau early at low values, undertraining is occurring.

**Mitigation:**
1. Increase training budget for Phase 2.
2. Use higher learning rates for expert-specific parameters.
3. Use curriculum learning: start with data that has clear domain boundaries, gradually introduce mixed data.

### 9.7 Risk: KV Cache Fracture from Expert Prefix Tokens

**Description:** Expert prefix tokens introduce cross-expert context bleeding into the autoregressive KV cache (see detailed analysis in Section 3.3). When consecutive tokens route to different experts, each token’s cached KV state is semantically bound to its routing expert’s prefix context, but subsequent tokens from other experts must attend to this foreign-context cache. This breaks KV cache sharing assumptions in production serving systems (PagedAttention, continuous batching) and may induce router sequence collapse (the model learns to assign all tokens in a sequence to the same expert to avoid cross-expert cache inconsistency).

**Probability:** HIGH for deployment; LOW for research evaluation (single-sequence inference is unaffected).

**Detection:** Monitor per-sequence expert entropy (if it drops to near-zero while per-corpus entropy remains high, sequence collapse is occurring). Compare model quality with vs. without expert prefix tokens in evaluation.

**Mitigation:**
1. **Primary recommendation:** Remove expert prefix tokens from deployment configurations entirely. Their expressivity contribution can be recovered through PLE + asymmetric LoRA on $W_Q$/$W_O$ (Section 3.7.1).
2. If retained for research: reduce $m$, use expert tokens only in a subset of layers, or restrict to per-sequence (not per-token) routing when prefix tokens are active.
3. For long-context inference at 32K+ tokens, the KV cache overhead from $m=10$ prefix tokens is 0.03%—negligible in size, but the semantic fracture concern remains regardless of cache size.

---

## 10. Comparison with Existing Approaches

| Approach | Expert Differentiation | Params/Expert | HBM Overhead | Inference Overhead | Requires Retraining? |
| --- | --- | --- | --- | --- | --- |
| Standard MoE (Mixtral) | Full FFN copies | 100% of FFN | 100% × K | Significant (multi-expert read) | From scratch or upcycle |
| Sparse Upcycling | Full FFN copies | 100% of FFN | 100% × K | Significant | Yes (continued pretraining) |
| D2DMoE (Partitioning) | Neuron clusters | 100% of FFN (redistributed) | ~100% of FFN | Moderate | Yes (router + sparse training) |
| LoRA-MoE (MoLoRA) | LoRA adapters as experts | 1–5% of model | 1–5% × K | Small | Yes (LoRA + router training) |
| X-LoRA | Pre-trained LoRA scaling | <1% of model | <1% × K | Small | Yes (scaling network only) |
| **LEC-Lite (Ours)** | PLE + FiLM + value res. + asymmetric LoRA ($W_Q$/$W_O$) | **<0.1% of model** | **<0.5% total** | **~Zero** | Yes (conditioning + router) |
| **LEC-Full (Ours)** | LEC-Lite + rank-96 FFN LoRA | **~5% of model** | **~15% total** | **Small (5–10%)** | Yes (conditioning + router + FFN LoRA) |

The key advantage of LEC-Lite is the combination of **extreme parameter efficiency** with **near-zero inference overhead**, enabled by the small size of conditioning params and their offloadability. LEC-Full trades a modest increase in memory and inference cost for genuine per-expert capacity expansion, while preserving the asymmetric attention design that keeps the KV cache universal.

---

## 11. Implementation Notes

### 11.1 Framework Requirements

LEC can be implemented on top of any standard transformer framework (PyTorch, JAX, Hugging Face Transformers) with the following modifications:

1. **Modified forward pass:** At each layer, inject PLE before attention, apply per-expert LayerNorm/FiLM, apply per-expert value residual, apply per-expert Canon convolutions (research config), and apply asymmetric LoRA on $W_Q$/$W_O$ projections.
2. **Asymmetric attention constraint:** $W_K$ and $W_V$ are never modified by expert conditioning. This is enforced at the implementation level by excluding these projections from the expert parameter registry.
3. **Router module:** A simple linear layer + softmax over expert logits, applied to the input hidden state at each MoE decision point (every layer, every N layers, or once per sequence).
4. **Expert parameter storage:** A dictionary mapping expert ID to conditioning parameters, stored as separate tensors that can be independently offloaded/loaded.
5. **Attention mask modification (research config only):** Extend the causal mask to include expert prefix tokens with appropriate bidirectional/causal structure. **Note:** This is excluded from the deployment configuration due to KV cache fracture concerns (Section 3.3, Section 9.7).

### 11.2 Distributed Training Considerations

- Expert conditioning params are small enough to be replicated across all data-parallel workers (no expert parallelism needed).
- Router training follows standard MoE router training (no special distributed infrastructure).
- If Phase 3 is used (unfreezing base weights), standard data parallelism applies.
- The approach is inherently **much easier to train** than standard MoE because there is no expert parallelism, no all-to-all communication, and no load-imbalance-induced idle time.

### 11.3 Serving and Deployment

For inference serving:
- Load base model weights into HBM once.
- **LEC-Lite:** Store all expert conditioning params in CPU memory (~20 MB total) or a small HBM cache. At each token, the router selects an expert and the corresponding conditioning params are fetched to HBM. Since all experts share the same base compute, there is no need for expert parallelism or token dropping—the serving infrastructure is identical to dense model serving.
- **LEC-Full:** All expert conditioning (including FFN LoRA) is resident in HBM (~1.2 GB for 6 experts). Use Grouped GEMM (S-LoRA / Punica) to batch tokens by expert assignment: the kernel groups tokens, fetches the shared dense weights once, loads the small active LoRA matrices into SRAM in parallel, and computes projections simultaneously. The asymmetric attention design ensures the KV cache remains universal across all experts, maintaining full compatibility with PagedAttention and continuous batching.
- **Expert prefix tokens are excluded from serving configurations** due to KV cache fracture concerns (Sections 3.3, 9.7).

---

## 12. Expected Outcomes and Success Criteria

### 12.1 Primary Success Criteria

1. **LEC with 10 experts achieves at least 40% of the perplexity improvement** that standard sparse upcycling (8 full experts) achieves over continued dense training, on the same token budget.
2. **LEC inference speed is within 2% of dense model speed** (as opposed to 40%+ slowdown for standard MoE).
3. **LEC HBM usage is within 1% of dense model usage.**
4. **Expert utilization entropy exceeds $0.7 \times \log(K)$** (experts are meaningfully used, not collapsed).

### 12.2 Stretch Goals

1. LEC achieves 60–80% of full MoE gains.
2. Expert composition (arithmetic blending of experts) produces meaningful intermediate behavior.
3. LEC scales to 7B+ models with maintained or improved relative gains.
4. Per-token expert switching shows clear domain-dependent routing patterns.

### 12.3 Operating Points: LEC-Lite and LEC-Full

The LEC framework naturally spans a continuum of per-expert parameter budgets. We identify two concrete operating points that target distinct use cases.

### LEC-Lite (~2 MB/expert, 10 experts)

**Target:** Maximum memory efficiency—MoE-quality routing specialization at near-zero HBM overhead.

**Configuration:** PLE ($d_{\text{ple}}=256$) + FiLM (per-expert LayerNorm) + value residual ($\lambda$, $\delta$) + asymmetric sparse LoRA (rank-8, $W_Q$/$W_O$ only, 4 heads × 4 layers). Expert prefix tokens and Canon kernels retained in ablations but excluded from the deployment configuration.

**Memory profile (3B base model, 10 experts):** ~19 MB total expert overhead (0.32% of base). Full system fits in ~6.4 GB including KV cache.

**Performance target:** 40–60% of full MoE gains over dense baseline. The mechanism is primarily *behavioral specialization*—different experts learn different processing strategies over the same shared knowledge. The expressivity analysis of Sections 4.1–4.4 establishes that depth-distributed lightweight conditioning provides sufficient function-space coverage for meaningful specialization, even though per-expert Shannon capacity is limited (Section 4.5).

**Use cases:** On-device deployment, edge inference, memory-constrained serving, scenarios where adding any expert parallelism infrastructure is infeasible. LEC-Lite can be served using identical infrastructure to the dense baseline.

### LEC-Full (~200 MB/expert, 6 experts)

**Target:** Capacity expansion—a 4B base model targeting 8–12B effective performance within consumer GPU memory (24 GB).

**Configuration:** Everything in LEC-Lite, plus rank-96 LoRA on all SwiGLU FFN projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) across all layers. Per-expert FFN LoRA parameters: $32 \times 3 \times 96 \times (3072 + 8192) \times 2 \approx 200\text{ MB}$ at BF16.

**Memory profile (4B base model, 6 experts):** Base weights (~8 GB) + expert conditioning (~1.2 GB) + KV cache (~2 GB) = ~11.2 GB, fitting entirely in HBM on an RTX 4090/5090.

**Performance target:** 60–80% of full MoE gains, competitive with 8–12B dense models on downstream benchmarks. The FFN LoRA provides genuine per-expert capacity (~108M parameters/expert × 6 = 648M additional parameters), while the asymmetric attention LoRA on $W_Q$/$W_O$ provides expert-specific query strategies without fragmenting the KV cache.

**Serving:** Uses Grouped GEMM (S-LoRA / Punica) to batch tokens by expert assignment and compute shared base + per-expert LoRA simultaneously. The asymmetric attention design ensures the KV cache remains universal and PagedAttention-compatible. Inference overhead is moderate (~5–10% over dense baseline from FFN LoRA reads) but far below standard MoE (40%+ overhead from full FFN expert reads).

**Use cases:** Consumer GPU deployment, high-quality inference on commodity hardware, scenarios where cloud MoE serving infrastructure is unavailable or too expensive.

### Relationship Between Operating Points

LEC-Lite and LEC-Full are not separate architectures—they are points on the same continuum, sharing the same routing mechanism, the same training recipe, and the same lightweight conditioning stack. The difference is solely the rank and coverage of the LoRA component. This means:

- Ablation results from LEC-Lite directly inform LEC-Full design decisions.
- A model can be trained as LEC-Lite and *upgraded* to LEC-Full by adding FFN LoRA and continuing training—no architectural changes required.
- Intermediate operating points (e.g., rank-32 FFN LoRA at ~70 MB/expert) can be explored along the same axis.

### 12.4 Negative Results of Value

Even if LEC underperforms, the ablation study will provide:
1. **Per-mechanism expressivity benchmarks:** How much does each conditioning mechanism contribute in isolation?
2. **Scaling laws for lightweight conditioning:** How does expressivity scale with conditioning parameter count?
3. **Practical limits on weight sharing:** At what point does expert differentiation require modifying the base weights directly?

---

## 13. Timeline and Resource Requirements

### 13.1 Estimated Timeline

| Phase | Duration | Description |
| --- | --- | --- |
| Month 1 | 4 weeks | Implementation of LEC framework and all conditioning mechanisms |
| Month 2 | 4 weeks | Small-scale experiments (125M, 350M) with full ablation suite |
| Month 3 | 4 weeks | Medium-scale experiments (1.3B, 3B) with best configurations from ablations |
| Month 4 | 2 weeks | Large-scale validation (7B if resources permit) |
| Month 4–5 | 4 weeks | Analysis, expert behavior visualization, paper writing |
| **Total** | **~5 months** |  |

### 13.2 Compute Requirements

| Experiment | GPUs | Duration | Total GPU-Hours |
| --- | --- | --- | --- |
| 125M ablations (11 configs × 10B tokens) | 4× H100 | ~3 days | ~288 |
| 350M ablations (6 configs × 15B tokens) | 8× H100 | ~5 days | ~960 |
| 1.3B experiments (4 configs × 30B tokens) | 16× H100 | ~7 days | ~2,688 |
| 3B experiments (3 configs × 50B tokens) | 32× H100 | ~10 days | ~7,680 |
| 7B validation (1 config × 50B tokens) | 64× H100 | ~7 days | ~10,752 |
| **Total** |  |  | **~22,368 GPU-hours** |

This is a fraction of what full MoE training or sparse upcycling experiments require, since we only train lightweight parameters for most of the pipeline.

---

## 14. References

1. Komatsuzaki, A., Puigcerver, J., Lee-Thorp, J., et al. (2022). Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints. *arXiv:2212.05055*.
2. He, E., et al. (2024). Upcycling Large Language Models into Mixture of Experts. *arXiv:2410.07524*.
3. Hui, T., et al. (2024). Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging. *arXiv:2410.01610*.
4. Zhang, Q., et al. (2024). BAM! Just Like That: Simple and Efficient Parameter Upcycling for Mixture of Experts. *arXiv:2408.08274*.
5. Zhang, Z., et al. (2021). MoEfication: Transformer Feed-forward Layers are Mixtures of Experts. *arXiv:2110.01786*.
6. D2DMoE. (2024). Exploiting Activation Sparsity with Dense to Dense MoE. *NeurIPS 2024*.
7. Li, X., et al. (2025). DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models. *arXiv:2503.01359*.
8. Wu, H., et al. (2024). Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks. *EMNLP 2024*.
9. Google. (2025). Gemma 3n Technical Report. *ai.google.dev/gemma/docs/gemma-3n*.
10. DeepSeek AI. (2025). Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram). *arXiv:2601.07372*.
11. Zhou, Z., et al. (2024). Value Residual Learning. *arXiv:2410.17897*. ACL 2025.
12. Allen-Zhu, Z. (2025). Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers. *arXiv:2512.17351*. NeurIPS 2025.
13. Buehler, E.L. & Buehler, M.J. (2024). X-LoRA: Mixture of Low-Rank Adapter Experts. *APL Machine Learning*, 2(2), 026119.
14. Zadouri, T., et al. (2024). Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning. *ICLR 2024*.
15. MoLE. (2024). Mixture of LoRA Experts. *OpenReview, ICLR 2024 submission*.
16. HydraLoRA. (2024). An Asymmetric LoRA Architecture for Efficient Fine-Tuning. *NeurIPS 2024*.
17. MoLA. (2025). MoE LoRA with Layer-wise Expert Allocation. *NAACL 2025 Findings*.
18. Hu, E.J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
19. Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*.
20. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23(120), 1–39.
21. Liu, A., et al. (2024). DeepSeek-V3 Technical Report. *arXiv:2412.19437*.
22. Pagliardini, M., et al. (2024). DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Average. *arXiv:2402.02622*.
23. Panda, A., et al. (2025). Dense Backpropagation Improves Training for Sparse Mixture-of-Experts. *arXiv:2504.12463*.
24. Saxe, A.M., McClelland, J.L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear networks. *ICLR 2014*.

---

## Appendix A: Detailed LoRA Switching Cost Calculation

### Full Asymmetric LoRA on $W_Q$ and $W_O$ projections, all 32 layers, rank-8

For a 3B model with hidden dimension $d = 3072$:

- Per projection per layer: $(d \times r + r \times d) \times 2\text{ bytes} = (3072 \times 8 + 8 \times 3072) \times 2 = 98,304\text{ bytes} \approx 96\text{ KB}$
- 2 projections ($W_Q$, $W_O$) × 32 layers: $2 \times 32 \times 96\text{ KB} = 6,144\text{ KB} \approx 6\text{ MB per expert}$
- All 10 experts pre-loaded: $60\text{ MB}$ — on a 3B model (~6 GB), this is ~1% overhead.
- Reading one expert’s LoRA from HBM at 3.35 TB/s: $6\text{ MB} / 3.35\text{ TB/s} \approx 1.8\mu s$.
- Token generation step time: 1–10 ms. LoRA switching is ~0.02–0.18% of step time.

### Sparse Asymmetric LoRA (4 heads, 4 layers, rank-8, $W_Q$ + $W_O$ only)

- Head dimension: $d_h = 3072/32 = 96$
- Per head per projection per layer: $(96 \times 8 + 8 \times 96) \times 2 = 3,072\text{ bytes} \approx 3\text{ KB}$
- 4 heads × 2 projections × 4 layers: $4 \times 2 \times 4 \times 3\text{ KB} = 96\text{ KB per expert}$
- All 10 experts: $960\text{ KB}$ total.
- Reading one expert: $96\text{ KB} / 3.35\text{ TB/s} \approx 0.029\mu s$.
- **Conclusion: Sparse asymmetric LoRA switching is ~0.003% of step time. Completely invisible.**

### LEC-Full: Rank-96 FFN LoRA (all 3 SwiGLU projections, all 32 layers)

- Per projection per layer: $(3072 \times 96 + 96 \times 8192) \times 2\text{ bytes} \approx 2.16\text{ MB}$ (for gate/up, $d_{ff}=8192$)
- 3 projections × 32 layers: $3 \times 32 \times 2.16\text{ MB} \approx 207\text{ MB per expert}$
- Reading one expert from HBM at 3.35 TB/s: $207\text{ MB} / 3.35\text{ TB/s} \approx 62\mu s$.
- Token generation step time: 1–10 ms. FFN LoRA switching is ~0.6–6.2% of step time.
- **Note:** In practice, this is amortized via Grouped GEMM which reads base weights + active LoRA in a single fused kernel pass.

---

## Appendix B: Complete Per-Expert Parameter Budget (3B Model, d=3072, L=32, 10 Experts, LEC-Lite)

| Mechanism | Formula | Per Expert | All 10 Experts |
| --- | --- | --- | --- |
| PLE (dim=256) | $L \times d_{\text{ple}} \times 2$ | 16 KB | 160 KB |
| Expert Tokens (m=10) | $m \times d \times 2$ | 60 KB | 600 KB |
| Value Residual λ | $L \times 2$ | 64 B | 640 B |
| Value Residual δ | $L \times d \times 2$ | 192 KB | 1.9 MB |
| LayerNorm/FiLM γ, β | $2 \times 2L \times d \times 2$ | 768 KB | 7.5 MB |
| Canon Kernels (AC, per-expert scaling) | $2 \times L \times 2d \times 2$ | 768 KB | 7.5 MB |
| Asymmetric LoRA ($W_Q$, $W_O$; 4h, 4L, r=8) | $2 \times 4 \times 4 \times 2 \times d_h \times r \times 2$ | 96 KB | 960 KB |
| Router | $d \times K \times 2$ (shared, not per-expert) | — | 60 KB |
| **Grand Total (LEC-Lite)** |  | **~1.9 MB** | **~18.7 MB** |
| **As % of 3B model (6 GB)** |  |  | **0.31%** |

> **Note:** Expert prefix tokens (60 KB/expert) are included in the ablation budget but excluded from the deployment configuration. Canon kernels (768 KB/expert) are included in the full research configuration; the deployment configuration may use PLE + FiLM + asymmetric LoRA only (~880 KB/expert, 8.8 MB total).
> 

> **LEC-Full addendum (4B base, 6 experts):** Adding rank-96 LoRA on all SwiGLU FFN projections adds $32 \times 3 \times 96 \times (3072 + 8192) \times 2 \approx 200\text{ MB/expert}$, for a total of ~1.2 GB expert overhead across 6 experts.
> 

---

## Appendix C: Notation Summary

| Symbol | Definition |
| --- | --- |
| $L$ | Number of transformer layers |
| $d$ | Hidden dimension |
| $d_h$ | Attention head dimension ($d / n_h$) |
| $d_{ff}$ | FFN intermediate dimension |
| $n_h$ | Number of attention heads |
| $K$ | Number of experts |
| $k$ | Index of a specific expert |
| $m$ | Number of expert prefix tokens |
| $d_{\text{ple}}$ | Dimension of per-layer embedding vectors |
| $r$ | LoRA rank |
| $\lambda_n^{(k)}$ | Per-expert, per-layer value residual scalar |
| $\delta_n^{(k)}$ | Per-expert, per-layer value residual bias vector |
| $\gamma^{(k)}, \beta^{(k)}$ | Per-expert LayerNorm/FiLM scale and shift |
| $e_l^{(k)}$ | Expert $k$’s PLE vector at layer $l$ |
| $T^{(k)}$ | Expert $k$’s prefix token embeddings |
| $R(x)$ | Router output (probability distribution over experts) |
| $J_l$ | Jacobian of layer $l$’s forward function |
| $A_k, B_k$ | LoRA low-rank matrices for expert $k$ |
| $\langle \cdot, \cdot \rangle_F$ | Frobenius inner product |
| $H(W)$ | Shannon entropy of weight matrix $W$ |