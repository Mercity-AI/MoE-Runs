#!/usr/bin/env python3
"""Plot final-checkpoint accuracy for baseline, n-gram 50%, and n-gram 25%."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


benchmarks = [
    "HellaSwag",
    "WinoGrande",
    "ARC-Easy",
    "ARC-Challenge",
    "PIQA",
    "OpenBookQA",
    "CommonsenseQA",
    "SciQ",
    "LAMBADA",
]

baseline = np.array([39.01, 50.43, 39.94, 24.91, 68.01, 30.40, 19.82, 66.30, 26.66])
ngram_50 = np.array([35.22, 49.25, 38.47, 22.87, 65.07, 28.20, 19.90, 63.30, 33.57])
ngram_25 = np.array([37.38, 52.64, 38.89, 24.66, 65.89, 28.20, 19.82, 62.50, 35.07])

# Compute the averages from the plotted values rather than hard-coding them.
labels = benchmarks + ["Shared-9 avg"]
series = {
    "Baseline": np.append(baseline, baseline.mean()),
    "N-gram 50%": np.append(ngram_50, ngram_50.mean()),
    "N-gram 25%": np.append(ngram_25, ngram_25.mean()),
}

x = np.arange(len(labels))
width = 0.25
colors = ["#3274A1", "#E1812C", "#3A923A"]

fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
for offset, ((name, values), color) in enumerate(zip(series.items(), colors)):
    bars = ax.bar(x + (offset - 1) * width, values, width, label=name, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, rotation=90)

ax.axvline(len(benchmarks) - 0.5, color="#777777", linewidth=1, linestyle="--")
ax.set_title("Final Checkpoint Benchmark Comparison", fontsize=17, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x, labels, rotation=30, ha="right")
ax.set_ylim(0, 77)
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=3, loc="upper center", frameon=False)

out = Path(__file__).resolve().parent / "comparison_plots" / "final_three_way_bar_chart.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220)
print(out)

