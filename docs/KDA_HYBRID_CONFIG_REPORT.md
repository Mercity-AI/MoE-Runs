# Choosing a Hybrid of Kimi Delta Attention and Full Attention for a 1B Dense Model

**Date:** 2026-08-11
**Scope:** deciding how to mix Kimi Delta Attention layers with standard grouped-query softmax attention layers in a roughly one-billion-parameter dense LLaMA-style model trained on six billion FineWeb tokens, as the next ablation in the series. The three questions this report answers: how many attention layers should stay softmax, where those layers should sit in the stack, and how large the linear-attention layers should be so the comparison against the existing dense baseline stays fair.

---

## 1. Background: what Kimi Delta Attention is

Kimi Delta Attention (KDA) is a linear attention mechanism introduced by Moonshot AI in the Kimi Linear paper (Zhang et al., 2025). Instead of comparing every token against every previous token the way softmax attention does, a KDA layer maintains a fixed-size memory: a small matrix per head that is updated once per token using a gated delta rule. The gate is fine-grained — every channel of the memory has its own learned forgetting rate — which is the main refinement KDA makes over its predecessor, Gated DeltaNet. Because the memory does not grow with sequence length, the cost per token is constant, and at inference time there is no key-value cache to store for these layers.

The price of that efficiency is that the memory is lossy. A fixed-size state cannot store an exact record of everything that came before; older information fades or gets overwritten. That single fact drives essentially every design decision in this report.

## 2. What the production hybrid models do

No frontier lab ships a purely linear model. Every recent system mixes linear layers with a minority of full softmax attention layers, and the recipes are strikingly consistent:

- **Kimi Linear** (48B mixture-of-experts, 3B active) interleaves KDA with full attention in a uniform three-to-one ratio: every block of four layers is three KDA layers followed by one full-attention layer, and the final layer of the network is full attention. Their full-attention layers use multi-head latent attention and, notably, carry **no positional encoding at all** — the KDA layers supply all position information. They ablated ratios of one-to-one, three-to-one, seven-to-one, and fifteen-to-one, and found three-to-one the best trade of quality against cost (Zhang et al., 2025).
- **Qwen3-Next** uses the same three-to-one pattern with Gated DeltaNet as the linear layer.
- **Jamba** (AI21's attention–Mamba hybrid) found no substantial quality difference between one attention layer per four and one per eight (Lieber et al., 2024).
- **MiniMax-01** runs one softmax layer per eight lightning-attention layers.

A second, less advertised commonality: the linear layers in these models are generously sized. Kimi Linear gives its KDA layers thirty-two heads of dimension 128 against a hidden width of only 2,304 — the query, key and value projections are nearly 1.8 times wider than the residual stream. Qwen3-Next similarly doubles the value width of its linear layers. These labs can afford that because in a mixture-of-experts model the attention stack is a small fraction of total parameters. In a dense model it is not, which is where our sizing problem comes from (section 5).

## 3. What controlled studies say about the ratio

The most directly useful evidence comes from an academic study that trained seventy-two hybrid models specifically to answer this question — thirty-six at 340M parameters and thirty-six at 1.3B parameters, which brackets our scale — covering six linear-attention variants (including Gated DeltaNet, KDA's direct ancestor) at five hybrid ratios (Wang et al., 2025). Their findings:

1. **Ordinary language-modeling quality is almost insensitive to the ratio.** Moving from one full-attention layer per twenty-four layers to one per three changed language-modeling scores by less than one percent.
2. **In-context recall is very sensitive to it.** Over that same range, recall performance nearly doubled. Recall means retrieving specific information from earlier in the context — the kind of ability tested by needle-in-a-haystack and key-value lookup tasks.
3. Their recommendation is a linear-to-full ratio between three-to-one and six-to-one, and they emphasize that the quality of the linear layer's gating and forgetting mechanism matters more than the exact ratio — good news for KDA, whose per-channel gating is precisely its selling point.

Independent studies agree. NVIDIA's empirical study of Mamba-based models found validation loss was minimized when roughly eight percent of layers were self-attention (Waleffe et al., 2024). A recent systematic analysis of hybrid architectures found a one-to-one ratio gives the best raw quality, but around one-to-five gives the best balance of quality and efficiency (Bick et al., 2025). Jamba's own ablations, as noted, found one-in-four and one-in-eight indistinguishable.

The consistent picture: **perplexity-style metrics will barely move across a wide range of ratios; retrieval-style abilities are what the softmax layers actually buy.** Since our evaluation suite (language-model loss, HellaSwag, WinoGrande) contains nothing recall-heavy, we should expect small measured differences between ratios — and should consider adding one retrieval-flavored evaluation if we want the axis on which these configurations genuinely differ to be visible.

## 4. What controlled studies say about placement, and why hybrids work at all

The same systematic analysis ablated where the full-attention layers sit. Placing them **early in the network is consistently harmful** — in some settings worse than having none at all — while placing them in the middle, or interleaving them evenly, works best (Bick et al., 2025).

The mechanistic literature explains why. In-context retrieval in transformers runs on *induction heads*: a two-part circuit in which one attention layer writes "what token preceded me" into each position, and a **later** attention layer uses that to look up "where have I seen this pattern before" (Olsson et al., 2022). Fixed-state recurrent layers struggle to implement this at long range because the lookup target has to survive inside a bounded, decaying memory (Bhattamishra et al., 2025; Arora et al., 2024). In a hybrid, the division of labor is therefore: linear layers handle local mixing and progressive consolidation of context cheaply, while the sparse full-attention layers periodically give the whole network exact, random-access retrieval over the sequence. For the circuit to form, attention layers need refined representations beneath them — which is exactly why front-loading them fails and why an even interleave with the first softmax layer a few blocks up, and one softmax layer at the very top, matches both the evidence and Kimi Linear's shipped layout.

One recipe element we deliberately do not copy: Kimi Linear's positionless full-attention layers. That choice was validated at forty-eight-billion-parameter scale with million-token context ambitions. At our one-billion scale with eight-thousand-token context and a small run budget, keeping ordinary rotary position embeddings on the softmax layers is the safer, well-understood default.

## 5. Sizing: keeping the comparison fair in a dense model

Our baseline is a dense thirty-two-layer model, hidden width 1,536, totalling 1.031 billion parameters. Each of its attention blocks (grouped-query, twelve query heads, six key-value heads) costs about 7.1 million parameters. A KDA layer is not free to "just replace" that block at equal size, because everything that gives KDA more memory — more heads, wider values — costs parameters, whereas softmax attention's memory (the key-value cache) costs none.

With the Kimi-standard head size of 128, the choices at our width are:

| KDA heads per layer | KDA layer size | Whole model (three-to-one mix) | Versus baseline |
|---|---|---|---|
| 12 (matches hidden width) | 10.3M | 1.107B | +7.4% |
| 10 | 8.6M | 1.068B | +3.6% |
| 8 (parameter-matched to the softmax block) | 7.0M | 1.028B | −0.2% |

Copying Kimi Linear's 1.8-times over-provisioning would mean twenty or more heads and roughly a quarter more total parameters — a confound, not an ablation. Our working constraint is that all models in this series stay within five percent of the baseline's size. That rules out twelve heads and selects **ten heads** as the default: the model lands 3.6 percent over the baseline, and each KDA layer is still about twenty percent larger than the softmax block it replaces, so the linear layers are not being starved to make the accounting look good. The eight-head variant is kept in reserve as an exactly parameter-matched control.

## 6. Recommended configurations

Five candidate runs, all inside the five-percent size window; the intent is to pick two or three.

| # | Configuration | Model size | Question it answers |
|---|---|---|---|
| 1 | Three-to-one interleave, ten KDA heads *(flagship)* | 1.068B (+3.6%) | Does the industry-standard recipe beat the softmax baseline at matched tokens? |
| 2 | One-to-one interleave, ten heads | 1.055B (+2.4%) | Does more softmax attention help at this scale, or does three-to-one already saturate? |
| 3 | Fully linear — no softmax layers, ten heads | 1.080B (+4.8%) | What do the softmax layers actually buy? Boldest contrast, cheapest inference. |
| 4 | Seven-to-one interleave, ten heads | 1.074B (+4.2%) | Does quality really stay flat down to only four softmax layers, as the literature predicts? |
| 5 | Three-to-one, eight heads (parameter-matched) | 1.028B (−0.2%) | Removes the "the hybrid only wins because it is bigger" objection; isolates the value of extra state. |

**Preferred trio: runs 1, 2 and 3.** Together with the already-trained baseline (all thirty-two layers softmax), they form a clean sweep of the number of full-attention layers — zero, eight, sixteen, thirty-two — with every model within five percent of the same size. That directly tests the literature's central claim (quality flat in the ratio, retrieval not) at exactly our scale and token budget, and any monotonic trend across the sweep is immediately interpretable. Run 5 is the best substitute if the parameter-fairness question feels more pressing than the ratio curve.

In every hybrid run the layout follows the evidence of sections 2 and 4: full-attention layers spread uniformly through the stack, the first appearing at the fourth layer, the last layer of the network always full attention. All other training settings — optimizer, schedule, batch, data, sequence length — stay identical to the baseline run.

The subsections below give each run's rationale and the exact settings to change in the training script (`baseline_llama_KDA.py`). The script's current configuration *is* run 1, so the architecture knobs only need touching for runs 2–5. One housekeeping rule applies to **every** run, including run 1: give each run its own output directory, checkpoint-bucket folder, experiment-tracker run name, and assets directory (the assets directory stores the model configuration the run was built from — reusing it across runs would silently carry one run's architecture into the next). The tables below spell these out with suggested values.

### 6.1 Run 1 — three-to-one interleave, ten heads (flagship)

**Why:** this is the consensus recipe. Kimi Linear ablated one-to-one through fifteen-to-one and shipped three-to-one; Qwen3-Next independently converged on the same ratio; the seventy-two-model academic sweep recommends three-to-one to six-to-one as the region where recall is already saturating while cost keeps falling (Wang et al., 2025). Eight softmax layers sit evenly through the stack (every fourth layer, ending with the topmost), which matches both the placement evidence and Kimi Linear's shipped layout. Ten heads keeps the model at 1.068B, inside the five-percent window, while leaving each linear layer about twenty percent larger than the softmax block it replaces — so the linear side is not handicapped. If the series only gets one hybrid run, it is this one; it is the direct "does the industry recipe beat our baseline at matched tokens" test.

| Setting | Current value | Change to |
|---|---|---|
| `kda_full_attn_every` | `4` | `4` (no change) |
| `kda_num_heads` | `10` | `10` (no change) |
| `wandb_run_name` | `"llama-6b-1b-kda"` | `"llama-1b-kda-3to1-h10"` (optional, for a consistent naming scheme across the sweep) |
| `output_dir` | `"./checkpoints_llama_1b_kda_6b"` | `"./checkpoints_llama_1b_kda_3to1_6b"` (optional, same reason) |
| `checkpoint_bucket_folder` | `"checkpoints_llama_1b_kda_6b"` | `"checkpoints_llama_1b_kda_3to1_6b"` (optional) |
| `hf_assets_dir` | `"./hf_assets_llama_1b_kda_titan"` | `"./hf_assets_llama_1b_kda_3to1"` (optional) |

### 6.2 Run 2 — one-to-one interleave, ten heads

**Why:** this brackets the ratio from above. The systematic-analysis study found one-to-one gives the best raw quality of any mix (Bick et al., 2025), while Kimi Linear found it no better than three-to-one at their scale. Those two findings disagree, and which one holds at one billion parameters and six billion tokens is exactly the open question. Sixteen softmax layers alternate with sixteen linear layers (every second layer softmax, still ending with the topmost), so if run 1 underperforms the baseline, this run tells us whether the cure is simply more attention — and if run 1 matches the baseline, this run tells us whether three-to-one had already saturated. It is also the second point on the clean zero/eight/sixteen/thirty-two sweep of softmax-layer counts.

| Setting | Current value | Change to |
|---|---|---|
| `kda_full_attn_every` | `4` | `2` |
| `kda_num_heads` | `10` | `10` (no change) |
| `wandb_run_name` | `"llama-6b-1b-kda"` | `"llama-1b-kda-1to1-h10"` |
| `output_dir` | `"./checkpoints_llama_1b_kda_6b"` | `"./checkpoints_llama_1b_kda_1to1_6b"` |
| `checkpoint_bucket_folder` | `"checkpoints_llama_1b_kda_6b"` | `"checkpoints_llama_1b_kda_1to1_6b"` |
| `hf_assets_dir` | `"./hf_assets_llama_1b_kda_titan"` | `"./hf_assets_llama_1b_kda_1to1"` |

### 6.3 Run 3 — fully linear, no softmax layers, ten heads

**Why:** the maximal contrast, and the anchor of the sweep at zero softmax layers. Every controlled study predicts a specific signature for this model: ordinary language-modeling loss close to the hybrids, with the damage concentrated in retrieval-style abilities that our current benchmark suite mostly does not measure. If this run lands close to the baseline on our evaluations, that is a genuinely interesting result about what a one-billion-parameter model at eight-thousand-token context actually needs softmax attention for — and it is the cheapest configuration to serve, with no key-value cache at all. If it lands clearly below, the gap between it and run 1 is a direct measurement of what eight softmax layers buy. Setting the interleave interval to none makes every layer linear (the explicit layer list stays unset).

| Setting | Current value | Change to |
|---|---|---|
| `kda_full_attn_every` | `4` | `None` |
| `kda_num_heads` | `10` | `10` (no change) |
| `wandb_run_name` | `"llama-6b-1b-kda"` | `"llama-1b-kda-pure-h10"` |
| `output_dir` | `"./checkpoints_llama_1b_kda_6b"` | `"./checkpoints_llama_1b_kda_pure_6b"` |
| `checkpoint_bucket_folder` | `"checkpoints_llama_1b_kda_6b"` | `"checkpoints_llama_1b_kda_pure_6b"` |
| `hf_assets_dir` | `"./hf_assets_llama_1b_kda_titan"` | `"./hf_assets_llama_1b_kda_pure"` |

### 6.4 Run 4 — seven-to-one interleave, ten heads

**Why:** the aggressive end of the published range. MiniMax-01 ships one softmax layer per eight; Jamba found one-in-eight indistinguishable from one-in-four; NVIDIA's study put the loss-optimal attention share near eight percent of layers, which at thirty-two layers is almost exactly the four softmax layers this configuration keeps (at the eighth, sixteenth, twenty-fourth and thirty-second positions). Its value is testing whether the "quality is flat far below three-to-one" claim survives at our scale — if it does, future runs in this series get most of the inference savings of the fully linear model with a safety net of retained attention. It overlaps in purpose with run 3, which is why it is fourth in priority: run 3 asks the sharper question, and this one mainly interpolates between runs 1 and 3.

| Setting | Current value | Change to |
|---|---|---|
| `kda_full_attn_every` | `4` | `8` |
| `kda_num_heads` | `10` | `10` (no change) |
| `wandb_run_name` | `"llama-6b-1b-kda"` | `"llama-1b-kda-7to1-h10"` |
| `output_dir` | `"./checkpoints_llama_1b_kda_6b"` | `"./checkpoints_llama_1b_kda_7to1_6b"` |
| `checkpoint_bucket_folder` | `"checkpoints_llama_1b_kda_6b"` | `"checkpoints_llama_1b_kda_7to1_6b"` |
| `hf_assets_dir` | `"./hf_assets_llama_1b_kda_titan"` | `"./hf_assets_llama_1b_kda_7to1"` |

### 6.5 Run 5 — three-to-one interleave, eight heads (parameter-matched)

**Why:** the fairness control. With eight heads of dimension 128, each linear layer costs almost exactly what the grouped-query softmax block it replaces costs, and the whole model lands at 1.028B — two tenths of a percent *under* the baseline. Any win this configuration scores cannot be attributed to extra parameters, which closes the most obvious objection to run 1's plus-3.6-percent size. The cost is a third less recurrent memory per layer than the ten-head version, so pairing this run with run 1 also isolates a second question the ratio sweep cannot: how much the size of the linear layers' state matters at fixed ratio. Prefer this over run 4 if reviewer-proofing the comparison matters more than mapping the ratio curve.

| Setting | Current value | Change to |
|---|---|---|
| `kda_full_attn_every` | `4` | `4` (no change) |
| `kda_num_heads` | `10` | `8` |
| `wandb_run_name` | `"llama-6b-1b-kda"` | `"llama-1b-kda-3to1-h8"` |
| `output_dir` | `"./checkpoints_llama_1b_kda_6b"` | `"./checkpoints_llama_1b_kda_3to1_h8_6b"` |
| `checkpoint_bucket_folder` | `"checkpoints_llama_1b_kda_6b"` | `"checkpoints_llama_1b_kda_3to1_h8_6b"` |
| `hf_assets_dir` | `"./hf_assets_llama_1b_kda_titan"` | `"./hf_assets_llama_1b_kda_3to1_h8"` |

Everything not listed in these tables stays at the baseline's values in every run: head dimension 128, single value head per query head with no value expansion, short convolution of width four enabled, no negative eigenvalues, no gate clamping, rotary-embedding QK-norm settings, and the entire optimizer, schedule, batch, sequence-length and data configuration. Holding all of that fixed is what makes the runs a controlled sweep rather than five loosely related experiments.

## References

- Zhang et al., *Kimi Linear: An Expressive, Efficient Attention Architecture*, 2025. arXiv:2510.26692.
- Wang et al., *A Systematic Analysis of Hybrid Linear Attention*, 2025. arXiv:2507.06457.
- Bick et al., *Hybrid Architectures for Language Models: Systematic Analysis and Design Insights*, 2025. arXiv:2510.04800.
- Waleffe et al., *An Empirical Study of Mamba-based Language Models*, 2024. arXiv:2406.07887.
- Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model*, 2024. arXiv:2403.19887.
- Olsson et al., *In-context Learning and Induction Heads*, Transformer Circuits Thread, 2022.
- Bhattamishra et al., *Mechanistic Evaluation of Transformers and State Space Models*, 2025. arXiv:2505.15105.
- Arora et al., *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff*, 2024. arXiv:2402.18668.
- Kimi-Linear 48B model configuration, Hugging Face: moonshotai/Kimi-Linear-48B-A3B-Instruct.
