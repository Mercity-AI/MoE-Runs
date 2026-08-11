#!/usr/bin/env python3
"""Plot the shared benchmark results transcribed from the two exported reports."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "comparison_plots"
BASELINE_STEPS = np.array([500, 1000, 1500, 2000, 2500, 3000, 3052])
NGRAM_STEPS = np.array([500, 1000, 1500, 2000, 2500, 3000, 3053])

# Values are copied from the evaluation tables in the two exported Markdown
# reports. All values are fractions (0-1), not percentages.
BASELINE = {
    "HellaSwag": [0.2793, 0.3194, 0.3446, 0.3721, 0.3845, 0.3885, 0.3901],
    "WinoGrande": [0.5162, 0.4980, 0.5091, 0.5193, 0.5209, 0.5075, 0.5043],
    "ARC-Easy": [0.3169, 0.3519, 0.3691, 0.3956, 0.3977, 0.3969, 0.3994],
    "ARC-Challenge": [0.2108, 0.2227, 0.2287, 0.2398, 0.2483, 0.2457, 0.2491],
    "PIQA": [0.5892, 0.6094, 0.6425, 0.6616, 0.6806, 0.6795, 0.6801],
    "OpenBookQA": [0.2520, 0.2780, 0.2800, 0.3100, 0.2960, 0.3020, 0.3040],
    "CommonsenseQA": [0.1949, 0.1982, 0.1982, 0.1982, 0.2007, 0.1982, 0.1982],
    "SciQ": [0.5330, 0.5730, 0.6020, 0.6400, 0.6530, 0.6600, 0.6630],
    "LAMBADA": [0.1261, 0.1683, 0.2133, 0.2544, 0.2606, 0.2659, 0.2666],
}

NGRAM = {
    "HellaSwag": [0.2773, 0.3059, 0.3265, 0.3430, 0.3497, 0.3537, 0.3522],
    "WinoGrande": [0.4996, 0.5170, 0.5225, 0.4949, 0.4870, 0.4901, 0.4925],
    "ARC-Easy": [0.3291, 0.3628, 0.3758, 0.3809, 0.3838, 0.3830, 0.3847],
    "ARC-Challenge": [0.2167, 0.2287, 0.2193, 0.2227, 0.2176, 0.2287, 0.2287],
    "PIQA": [0.5827, 0.6219, 0.6360, 0.6594, 0.6561, 0.6469, 0.6507],
    "OpenBookQA": [0.2520, 0.2800, 0.2700, 0.2760, 0.2840, 0.2860, 0.2820],
    "CommonsenseQA": [0.1974, 0.1982, 0.1974, 0.1982, 0.1998, 0.2007, 0.1990],
    "SciQ": [0.5890, 0.5980, 0.6130, 0.6250, 0.6230, 0.6260, 0.6330],
    "LAMBADA": [0.1972, 0.2655, 0.2763, 0.3138, 0.3284, 0.3330, 0.3357],
}


def style_axis(ax, title):
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, max(max(BASELINE[title]), max(NGRAM[title])) * 1.18)


def draw_benchmark(ax, benchmark, legend=False):
    ax.plot(BASELINE_STEPS, BASELINE[benchmark], "o-", label="Baseline", linewidth=2)
    ax.plot(NGRAM_STEPS, NGRAM[benchmark], "o-", label="N-gram", linewidth=2)
    style_axis(ax, benchmark)
    if legend:
        ax.legend()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    individual_dir = OUT_DIR / "individual"
    individual_dir.mkdir(exist_ok=True)

    benchmarks = list(BASELINE)
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for ax, benchmark in zip(axes.flat, benchmarks):
        draw_benchmark(ax, benchmark)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.suptitle("1B Baseline vs N-gram: Benchmark Accuracy", y=0.995, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "all_benchmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for benchmark in benchmarks:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        draw_benchmark(ax, benchmark, legend=True)
        fig.savefig(individual_dir / f"{benchmark.lower().replace('-', '_')}.png", dpi=200)
        plt.close(fig)

    baseline_avg = np.mean(np.array(list(BASELINE.values())), axis=0)
    ngram_avg = np.mean(np.array(list(NGRAM.values())), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(BASELINE_STEPS, baseline_avg, "o-", label="Baseline", linewidth=2)
    ax.plot(NGRAM_STEPS, ngram_avg, "o-", label="N-gram", linewidth=2)
    ax.set(title="Mean Accuracy Across the 9 Shared Benchmarks", xlabel="Training step", ylabel="Mean accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUT_DIR / "shared_9_average.png", dpi=200)
    plt.close(fig)

    final_delta = np.array([NGRAM[b][-1] - BASELINE[b][-1] for b in benchmarks])
    colors = np.where(final_delta >= 0, "#2ca02c", "#d62728")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    order = np.argsort(final_delta)
    ax.barh(np.array(benchmarks)[order], final_delta[order] * 100, color=colors[order])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Final Checkpoint: N-gram Minus Baseline", xlabel="Accuracy difference (percentage points)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(OUT_DIR / "final_checkpoint_delta.png", dpi=200)
    plt.close(fig)

    with (OUT_DIR / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", *benchmarks, "shared_9_average"])
        for model, steps, values, avg in (
            ("baseline", BASELINE_STEPS, BASELINE, baseline_avg),
            ("ngram", NGRAM_STEPS, NGRAM, ngram_avg),
        ):
            for i, step in enumerate(steps):
                writer.writerow([model, step, *(values[b][i] for b in benchmarks), avg[i]])

    print(f"Wrote plots and data to {OUT_DIR}")
    print(f"Final shared-9 average: baseline={baseline_avg[-1]:.4f}, ngram={ngram_avg[-1]:.4f}")
    for benchmark, delta in zip(benchmarks, final_delta):
        print(f"{benchmark:16s} baseline={BASELINE[benchmark][-1]:.4f} ngram={NGRAM[benchmark][-1]:.4f} delta={delta:+.4f}")


if __name__ == "__main__":
    main()
