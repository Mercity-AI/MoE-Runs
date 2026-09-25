"""Per-type metrics and failure-mode breakdown for one or more NER eval outputs.

    python analysis/failure_modes.py baseline=base.json ngram25=ng25.json [--examples N]

Error modes, for each wrong prediction:
  type error      right text, wrong type
  boundary error  overlaps a gold entity but the text differs (+/- wrong type)
  hallucinated    text does not appear in the sentence at all
  spurious        real text from the sentence that is not a gold entity
and, for each gold entity with no overlapping prediction: missed entirely.
Also reports F1 by entities-per-sentence, sentence-level flags, the top type
confusions, and F1 if duplicate JSON keys were merged (the scorer keeps only
the last occurrence of a repeated key).
"""
import json
import re
import sys
from collections import Counter, defaultdict

args = [a for a in sys.argv[1:] if "=" in a]
n_ex = int(sys.argv[sys.argv.index("--examples") + 1]) if "--examples" in sys.argv else 0
runs = {k: json.load(open(v))["samples"] for k, v in (a.split("=", 1) for a in args)}
TYPES = ["art", "building", "event", "location", "organization", "other", "person", "product"]


def overlap(a, b):
    ta, tb = a.split(), b.split()
    return bool(set(ta) & set(tb)) and (a in b or b in a or len(set(ta) & set(tb)) >= 1)


def analyze(samples):
    tp, fp, fn = Counter(), Counter(), Counter()
    modes = Counter()           # counted per pred error / per gold miss
    confusion = Counter()       # (gold_type, pred_type) for same-span type errors
    examples = defaultdict(list)
    sent = Counter()
    buckets = defaultdict(lambda: [0, 0, 0])  # n_gold bucket -> tp, fp, fn
    for s in samples:
        text = s["text"].lower()
        gold = {tuple(e) for e in s["gold"]}
        pred = {tuple(e) for e in s["pred"]}
        gen = s["generated"]
        for t, _ in pred & gold: tp[t] += 1
        for t, _ in pred - gold: fp[t] += 1
        for t, _ in gold - pred: fn[t] += 1
        ng = len(gold)
        b = "0" if ng == 0 else "1-2" if ng <= 2 else "3-5" if ng <= 5 else "6+"
        buckets[b][0] += len(pred & gold); buckets[b][1] += len(pred - gold); buckets[b][2] += len(gold - pred)

        # sentence-level flags
        if not s["parse_ok"] or not gen.rstrip().endswith("}"):
            sent["unterminated / parse fail"] += 1
        if len(gen) > 400:
            sent["runaway generation (>400 chars)"] += 1
        keys = re.findall(r'"([a-z]+)":\s*\[', gen)
        if len(keys) != len(set(keys)):
            sent["duplicate JSON keys (later key overwrites earlier)"] += 1
        if not gold and not pred:
            sent["empty gold, correctly empty"] += 1
        if not gold and pred:
            sent["empty gold, predicted entities"] += 1
        if pred == gold:
            sent["sentence fully correct"] += 1

        gold_spans = {sp: t for t, sp in gold}
        pred_spans = {sp: t for t, sp in pred}
        for t, sp in pred - gold:
            if sp in gold_spans:
                m = "type error (right span, wrong type)"
                confusion[(gold_spans[sp], t)] += 1
            elif sp not in text:
                m = "hallucinated (span not in sentence)"
            elif any(overlap(sp, g) for g in gold_spans):
                gt = next(gold_spans[g] for g in gold_spans if overlap(sp, g))
                m = "boundary error" + ("" if gt == t else " + wrong type")
            else:
                m = "spurious (real text, not an entity in gold)"
            modes["FP: " + m] += 1
            examples[m].append((s["text"], sp, t, sorted(gold)))
        for t, sp in gold - pred:
            if sp in pred_spans:
                continue  # already counted as type error
            if any(overlap(sp, p) for p in pred_spans):
                continue  # already counted as boundary error
            modes["FN: missed entirely"] += 1
            examples["missed entirely"].append((s["text"], sp, t, sorted(pred)))
    return tp, fp, fn, modes, confusion, sent, buckets, examples


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


res = {k: analyze(v) for k, v in runs.items()}
names = list(res)
print(f"n sentences: " + ", ".join(f"{k}={len(runs[k])}" for k in names))
print("\n## Per-type P / R / F1 (support = gold count)")
print(f"{'type':<13}{'support':>8}  " + "  ".join(f"{k:>22}" for k in names))
for t in TYPES + ["MICRO"]:
    cells = []
    for k in names:
        tp, fp, fn = res[k][:3]
        if t == "MICRO":
            a, b, c = sum(tp.values()), sum(fp.values()), sum(fn.values())
        else:
            a, b, c = tp[t], fp[t], fn[t]
        p, r, f = prf(a, b, c)
        cells.append(f"{p:.3f}/{r:.3f}/{f:.3f}")
    tp, fp, fn = res[names[0]][:3]
    sup = sum(tp.values()) + sum(fn.values()) if t == "MICRO" else tp[t] + fn[t]
    print(f"{t:<13}{sup:>8}  " + "  ".join(f"{c:>22}" for c in cells))
macro = {k: sum(prf(res[k][0][t], res[k][1][t], res[k][2][t])[2] for t in TYPES) / len(TYPES) for k in names}
print(f"{'MACRO F1':<21}  " + "  ".join(f"{macro[k]:>22.3f}" for k in names))

print("\n## F1 by number of gold entities in sentence")
for b in ["0", "1-2", "3-5", "6+"]:
    print(f"  {b:<5}" + "  ".join(
        f"{k}: F1={prf(*res[k][6][b])[2]:.3f} (fp={res[k][6][b][1]})" for k in names))

print("\n## Error modes (counts)")
allm = sorted({m for k in names for m in res[k][3]})
for m in allm:
    print(f"  {m:<55}" + "  ".join(f"{k}={res[k][3][m]:>6}" for k in names))

print("\n## Sentence-level")
alls = sorted({m for k in names for m in res[k][5]})
for m in alls:
    print(f"  {m:<55}" + "  ".join(f"{k}={res[k][5][m]:>6}" for k in names))

print("\n## Top type confusions (gold -> predicted), same span")
for k in names:
    tot = sum(res[k][4].values())
    print(f"  {k}: " + ", ".join(f"{g}->{p} {c} ({c / tot:.0%})" for (g, p), c in res[k][4].most_common(8)))

def parse_merged(text):
    """Parse like ner_data but merge repeated JSON keys instead of last-one-wins."""
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return set()

    def hook(pairs):
        d = {}
        for k, v in pairs:
            d[k] = d[k] + v if k in d and isinstance(d[k], list) and isinstance(v, list) else v
        return d
    try:
        obj = json.loads(text[s:e + 1], object_pairs_hook=hook)
    except ValueError:
        return set()
    out = set()
    for k, v in obj.items():
        k = k.strip().lower()
        if k in TYPES:
            for it in v if isinstance(v, list) else [v]:
                if str(it).strip():
                    out.add((k, str(it).strip().lower()))
    return out


print("\n## Micro F1 if duplicate JSON keys were merged (scoring-artifact check)")
for k in names:
    tp = fp = fn = 0
    for s in runs[k]:
        g, p = {tuple(e) for e in s["gold"]}, parse_merged(s["generated"])
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    print(f"  {k}: {2 * tp / (2 * tp + fp + fn):.4f}")

if n_ex:
    k = names[-1]
    for m, exs in res[k][7].items():
        print(f"\n### {k}: {m} (e.g.)")
        for text, sp, t, other in exs[:n_ex]:
            print(f"  [{t}] {sp!r}  <- {text[:140]}")
