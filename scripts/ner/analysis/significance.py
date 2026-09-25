"""Paired significance tests between two NER eval outputs on the same sentences.

    python analysis/significance.py <reference_eval.json> <other_eval.json>

Inputs are the JSON files written by ner_eval_vllm.py / ner_eval.py. Differences
are reported as other - reference (e.g. ngram - baseline).

Unit of resampling = sentence (entities within a sentence are correlated).
  * paired bootstrap  -> 95% CI of F1(ngram) - F1(baseline)
  * approximate randomization (swap the two systems' outputs per sentence)
    -> two-sided p-value, Holm-corrected across the per-type tests
  * McNemar on "sentence fully correct"
"""
import json
import sys

import numpy as np
from scipy.stats import binomtest

TYPES = ["art", "building", "event", "location", "organization", "other", "person", "product"]
R = 10000
rng = np.random.default_rng(0)

A = json.load(open(sys.argv[1]))["samples"]  # reference (e.g. baseline)
B = json.load(open(sys.argv[2]))["samples"]  # other (e.g. ngram)
assert [a["text"] for a in A] == [b["text"] for b in B], "eval files must cover the same sentences"
n = len(A)


def counts(samples):
    """(n, 8 types, 3) array of per-sentence tp/fp/fn."""
    out = np.zeros((len(samples), len(TYPES), 3))
    for i, s in enumerate(samples):
        g = {tuple(e) for e in s["gold"]}
        p = {tuple(e) for e in s["pred"]}
        for t, _ in p & g: out[i, TYPES.index(t), 0] += 1
        for t, _ in p - g: out[i, TYPES.index(t), 1] += 1
        for t, _ in g - p: out[i, TYPES.index(t), 2] += 1
    return out


def f1s(tot):
    """tot: (..., 8, 3) -> (..., 10) = per-type F1, micro F1, macro F1."""
    tp, fp, fn = tot[..., 0], tot[..., 1], tot[..., 2]
    per = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), 0)
    mtp, mfp, mfn = tp.sum(-1), fp.sum(-1), fn.sum(-1)
    micro = 2 * mtp / (2 * mtp + mfp + mfn)
    return np.concatenate([per, micro[..., None], per.mean(-1, keepdims=True)], -1)


ca, cb = counts(A), counts(B)
obs = f1s(cb.sum(0)) - f1s(ca.sum(0))
fa, fb = f1s(ca.sum(0)), f1s(cb.sum(0))
flat_a, flat_b = ca.reshape(n, -1), cb.reshape(n, -1)
diff = flat_b - flat_a

boot, perm = [], []
for chunk in range(R // 500):
    idx = rng.integers(0, n, (500, n))
    W = np.zeros((500, n))
    np.add.at(W, (np.arange(500)[:, None], idx), 1)
    ta = (W @ flat_a).reshape(500, len(TYPES), 3)
    tb = (W @ flat_b).reshape(500, len(TYPES), 3)
    boot.append(f1s(tb) - f1s(ta))
    S = rng.integers(0, 2, (500, n)).astype(float)  # 1 = swap this sentence
    pa = (flat_a.sum(0) + S @ diff).reshape(500, len(TYPES), 3)
    pb = (flat_b.sum(0) - S @ diff).reshape(500, len(TYPES), 3)
    perm.append(f1s(pb) - f1s(pa))
boot, perm = np.concatenate(boot), np.concatenate(perm)
p = ((np.abs(perm) >= np.abs(obs) - 1e-12).sum(0) + 1) / (R + 1)

# Holm across the 8 per-type tests
order = np.argsort(p[:8])
holm = np.empty(8)
running = 0
for rank, j in enumerate(order):
    running = max(running, min(1, (8 - rank) * p[j]))
    holm[j] = running

names = TYPES + ["MICRO", "MACRO"]
print(f"{'metric':<13}{'baseline':>9}{'ngram':>8}{'diff':>8}   {'95% bootstrap CI':<20}{'p (AR)':>9}{'Holm p':>9}")
for j, nm in enumerate(names):
    lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
    hp = f"{holm[j]:.4f}" if j < 8 else "   -"
    print(f"{nm:<13}{fa[j]:>9.3f}{fb[j]:>8.3f}{obs[j]:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]    {p[j]:>9.4f}{hp:>9}")

ok_a = np.array([{tuple(e) for e in s["pred"]} == {tuple(e) for e in s["gold"]} for s in A])
ok_b = np.array([{tuple(e) for e in s["pred"]} == {tuple(e) for e in s["gold"]} for s in B])
b_only, a_only = int((ok_b & ~ok_a).sum()), int((ok_a & ~ok_b).sum())
print(f"\nMcNemar (sentence fully correct): baseline-only={a_only} ngram-only={b_only} "
      f"p={binomtest(b_only, a_only + b_only).pvalue:.2e}")
print(f"(smallest attainable AR p-value with R={R}: {1 / (R + 1):.1e})")
