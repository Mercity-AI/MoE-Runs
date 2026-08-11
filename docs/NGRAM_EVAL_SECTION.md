# Evaluations — N-gram Table Sizing (Baseline vs 50% vs 25%)

Three models, identical training recipe (six billion FineWeb tokens, 3,053 steps, same optimizer and schedule), differing only in how the parameter budget is split between transformer layers and n-gram embedding tables:

| Model | Layers | Tables | Share of parameters in tables | Total parameters |
| --- | --- | --- | --- | --- |
| Baseline | 32 | none | 0% | 1.031B |
| N-gram 50% | 16 | 6 | 48% | 1.037B |
| N-gram 25% | 23 | 4 | 25% | 1.006B |

All numbers are zero-shot accuracies from LM Evaluation Harness at the final checkpoint (~6B tokens). The shared-9 average is the plain mean over the nine benchmarks.

| Benchmark | Baseline | N-gram 50% | N-gram 25% | 25% − Baseline |
| --- | --- | --- | --- | --- |
| HellaSwag | 39.0 | 35.2 | 37.4 | −1.6 |
| WinoGrande | 50.4 | 49.2 | 52.6 | +2.2 |
| ARC-Easy | 39.9 | 38.5 | 38.9 | −1.0 |
| ARC-Challenge | 24.9 | 22.9 | 24.7 | −0.2 |
| PIQA | 68.0 | 65.1 | 65.9 | −2.1 |
| OpenBookQA | 30.4 | 28.2 | 28.2 | −2.2 |
| CommonsenseQA | 19.8 | 19.9 | 19.8 | 0.0 |
| SciQ | 66.3 | 63.3 | 62.5 | −3.8 |
| LAMBADA | 26.7 | 33.6 | 35.1 | +8.4 |
| **Shared-9 average** | **40.6** | **39.5** | **40.6** | **0.0** |

The rounded averages hide a very small difference: baseline is **40.61%** and
N-gram 25% is **40.56%**, a gap of just **0.05 percentage points**.

![Final-checkpoint comparison across baseline, N-gram 50%, and N-gram 25%](comparison_plots/final_three_way_bar_chart.png)

## N-gram 25% checkpoint trajectory

Unlike the original report, the 25% run was evaluated at every saved checkpoint
with no sample limit. The values below use normalized accuracy where LM Evaluation
Harness provides it, and raw accuracy otherwise—the same convention used in the
final comparison above.

| Step | HellaSwag | WinoGrande | ARC-E | ARC-C | PIQA | OBQA | CSQA | SciQ | LAMBADA | Shared-9 avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 28.16 | 51.07 | 32.53 | 20.56 | 59.74 | 26.60 | 19.82 | 58.90 | 22.74 | 35.57 |
| 1,000 | 31.49 | 51.70 | 35.52 | 21.25 | 63.17 | 27.20 | 19.82 | 60.50 | 30.29 | 37.88 |
| 1,500 | 34.02 | 51.62 | 37.04 | 23.29 | 64.91 | 27.20 | 19.82 | 60.00 | 29.48 | 38.60 |
| 2,000 | 36.01 | 51.46 | 38.13 | 22.95 | 66.16 | 29.20 | 19.82 | 61.70 | 32.85 | 39.81 |
| 2,500 | 36.83 | 53.04 | 38.22 | 24.40 | 66.32 | 29.20 | 19.82 | 63.00 | 34.80 | **40.62** |
| 3,000 | 37.31 | 51.62 | 38.43 | **25.00** | 66.27 | 28.80 | 19.82 | 62.60 | 34.97 | 40.53 |
| 3,053 | **37.38** | 52.64 | **38.89** | 24.66 | 65.89 | 28.20 | 19.82 | 62.50 | **35.07** | 40.56 |

### 25% trajectory interpretation

- **Most of the aggregate gain arrives by step 2,500.** The shared-nine average
  rises from 35.57% at step 500 to a peak of 40.62% at step 2,500, then remains
  effectively flat through step 3,053 (40.56%). The final 553 steps redistribute
  performance across tasks rather than improving the overall score.
- **Language modeling continues to improve after the aggregate plateaus.** HellaSwag
  and LAMBADA reach their best values at the final checkpoint. LAMBADA climbs by
  12.33 points from step 500 to 3,053, again providing the clearest evidence that
  the n-gram tables add useful next-token signal.
- **The reasoning and QA metrics peak at different times.** PIQA peaks at step 2,500
  (66.32%), ARC-Challenge at step 3,000 (25.00%), and OpenBookQA at steps 2,000–2,500
  (29.20%). Consequently, step 3,053 is not uniformly the best checkpoint even
  though it is best for HellaSwag, ARC-Easy, and LAMBADA.
- **WinoGrande remains noisy rather than monotonic.** It ranges from 51.07% to
  53.04% and finishes at 52.64%. Since it is binary and close to chance, the final
  lead over baseline should still be treated cautiously.
- **CommonsenseQA is completely flat at 19.82%.** This confirms the final comparison
  is not hiding learning on that task; the model stays at the five-way random floor
  throughout training.

### Interpretation

Quick reads off these results — first impressions, not final conclusions:

- **The headline: 25% is the right neighborhood, and the sizing-report recommendation held up.** Halving the table share (48% → 25%) recovered almost the entire 1.07-point average deficit the 50% run had against the baseline: the 25% model finishes only 0.05 points behind baseline (40.56 vs 40.61). It beats the 50% model on six of nine benchmarks, ties it on OpenBookQA, and trails it only on CommonsenseQA (by a negligible 0.08 points) and SciQ.
- **The compute framing makes the tie a win.** The 25% model runs 23 transformer layers instead of 32, so it spends roughly a quarter fewer FLOPs per token than the baseline (the 50% model, at 16 layers, roughly half). Matching the baseline's benchmark average at that discount is the result to lead with; loss-versus-FLOPs is the chart that will show it.
- **LAMBADA is where the tables clearly earn their keep: +8.4 points over baseline, and the single largest effect in the entire comparison.** Both n-gram models show it (+6.9 for the 50% run), it appears from the earliest checkpoint, and it grows through training. LAMBADA is final-word prediction over a broad context, and the mechanism writes itself: the tables behave like a learned n-gram predictor fused into the input embedding, sharpening exactly the local, surface-statistics component of next-word prediction. This is the most robust, most mechanistically sensible finding here.
- **The knowledge-flavored benchmarks pay for it.** SciQ (−3.8), OpenBookQA (−2.2), and PIQA (−2.1) are down for the 25% model — and by almost identical amounts for the 50% model, even though the two differ hugely in table share. That pattern points at the shallower backbone rather than the tables themselves: factual and physical-commonsense knowledge seems to live in transformer depth and feed-forward capacity, which is what both n-gram configs traded away. Worth checking deliberately later — a depth-matched control would separate "tables displace knowledge" from "fewer layers store less knowledge."
- **HellaSwag scales cleanly with backbone size** (39.0 at 32 layers, 37.4 at 23, 35.2 at 16), consistent with the same depth story.
- **WinoGrande's +2.2 for the 25% model should be treated as noise until it repeats.** It is a binary task where all three models sit near chance, and the baseline's own trajectory wobbled between 49 and 52 across checkpoints — the spread between checkpoints of one model is as large as the spread between models.
- **CommonsenseQA is dead weight at this scale**: all three models sit at 19.8–19.9 against a 20% random floor. It contributes nothing but a constant to the average; consider dropping it or replacing it with something recall- or retrieval-flavored, which would also serve the upcoming attention ablations better.
- **Trajectory note:** the 50% n-gram model led baseline early, while the 25% run reaches 35.57% at step 500 and rises to 40.62% by step 2,500. Direct crossover claims between all three runs should be made carefully because the baseline and 50% figures came from their report tables while the 25% checkpoints came from the new harness run, but the available evidence supports the same broad story: tables accelerate surface-statistics learning, and retaining more transformer depth lets the 25% configuration preserve that benefit without the 50% model's final average deficit.

Caveats for whatever gets written on top of this: single seed per configuration, most gaps outside LAMBADA are within one to two points (ordinary multiple-choice noise at this scale), and the two n-gram runs differ in both table share and layer count at once, so per-benchmark deltas between them mix two effects.
