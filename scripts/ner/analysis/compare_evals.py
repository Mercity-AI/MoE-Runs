"""Per-sample comparison of two NER eval outputs on the same sentences.

    python analysis/compare_evals.py <A.json> <B.json> [nameA nameB [n_examples]]

Typical uses: HF vs vLLM eval of the same checkpoint (expect ~95% identical
generations and near-equal F1; two HF runs agree about as much), or two models.
Prints F1, identical-generation rate, per-sentence win/loss, an error breakdown,
and the first n differing generations.
"""
import json
import sys
from collections import Counter

a, b = (json.load(open(p))["samples"] for p in sys.argv[1:3])
na, nb = sys.argv[3:5] if len(sys.argv) > 4 else ("A", "B")
assert [x["text"] for x in a] == [x["text"] for x in b]


def f1(samples):
    tp = fp = fn = 0
    for s in samples:
        p, g = {tuple(x) for x in s["pred"]}, {tuple(x) for x in s["gold"]}
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    return 2 * tp / (2 * tp + fp + fn)


def degenerate(s):
    return len(s["generated"]) > 400 or not s["generated"].rstrip().endswith("}")


same_gen = sum(x["generated"] == y["generated"] for x, y in zip(a, b))
same_pred = sum(x["pred"] == y["pred"] for x, y in zip(a, b))
print(f"{na}: F1={f1(a):.3f} degenerate={sum(map(degenerate, a))} | "
      f"{nb}: F1={f1(b):.3f} degenerate={sum(map(degenerate, b))}")
print(f"identical generations: {same_gen}/{len(a)}  identical pred sets: {same_pred}/{len(a)}")

# per-sample win/loss on entity correctness
wins = Counter()
for x, y in zip(a, b):
    g = {tuple(e) for e in x["gold"]}
    ca, cb = len({tuple(e) for e in x["pred"]} & g), len({tuple(e) for e in y["pred"]} & g)
    wins["A>B" if ca > cb else "B>A" if cb > ca else "tie"] += 1
print("per-sample correct-entity comparison:", dict(wins))

# what kinds of errors does B make that A doesn't
kinds = Counter()
for x, y in zip(a, b):
    g = {tuple(e) for e in x["gold"]}
    pa, pb = {tuple(e) for e in x["pred"]}, {tuple(e) for e in y["pred"]}
    for e in (pb - g) - (pa - g):
        # wrong type for a real span vs. invented/mis-spanned span
        kinds["B-only FP: right span, wrong type" if any(e[1] == s for _, s in g)
              else "B-only FP: span not in gold"] += 1
    for e in (g - pb) - (g - pa):
        kinds["B-only miss"] += 1
    for e in (pa - g) - (pb - g):
        kinds["A-only FP"] += 1
    for e in (g - pa) - (g - pb):
        kinds["A-only miss"] += 1
print("error breakdown:", dict(kinds))

n_show = int(sys.argv[5]) if len(sys.argv) > 5 else 0
shown = 0
for x, y in zip(a, b):
    if x["generated"] != y["generated"] and shown < n_show:
        shown += 1
        print(f"\nTEXT: {x['text'][:160]}\nGOLD: {x['gold']}\n{na}: {x['generated'][:200]}\n{nb}: {y['generated'][:200]}")
