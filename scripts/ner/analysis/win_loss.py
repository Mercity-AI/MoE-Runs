"""Where does each model win? Entity-level paired comparison by feature bucket.

    python analysis/win_loss.py <reference_eval.json> <other_eval.json> [--tokenizer DIR]

For every gold entity: did the reference (A, e.g. baseline) / other (B, e.g.
ngram) model predict it exactly? Entities only one model gets right are what
distinguish the models; each table shows B's share of those by bucket
(50% = tie) with a sign-test p-value. Buckets: entity type, span length,
how often the entity text appears in the Few-NERD training set, training-label
ambiguity, sub-word pieces per word, digits, non-ASCII characters.
Training-set entity counts are cached next to this script.
"""
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ner_data  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

A = json.load(open(sys.argv[1]))["samples"]
B = json.load(open(sys.argv[2]))["samples"]
TOKENIZER = (sys.argv[sys.argv.index("--tokenizer") + 1] if "--tokenizer" in sys.argv
             else str(Path(__file__).resolve().parents[3] / "checkpoints/baseline/step_003053_hf"))
N_EX = 6

cache = Path(__file__).resolve().parent / "train_entity_counts.pkl"
if cache.exists():
    surf_cnt, surf_types = pickle.load(open(cache, "rb"))
else:
    raw, schema, col = ner_data.load_split("few-nerd", "train")
    surf_cnt, surf_types = Counter(), defaultdict(Counter)
    for ex in raw:
        for t, s in schema.gold_set(ex["tokens"], ex[col]):
            surf_cnt[s] += 1
            surf_types[s][t] += 1
    pickle.dump((surf_cnt, dict(surf_types)), open(cache, "wb"))
tok = AutoTokenizer.from_pretrained(TOKENIZER)


def feats(t, s, text):
    c = surf_cnt.get(s, 0)
    types = surf_types.get(s, {})
    words = s.split()
    pieces = len(tok(s, add_special_tokens=False)["input_ids"]) / max(len(words), 1)
    return {
        "type": t,
        "span words": str(min(len(words), 4)) + ("+" if len(words) >= 4 else ""),
        "train freq (surface)": "0 (unseen)" if c == 0 else "1-4" if c < 5 else "5-49" if c < 50 else "50+",
        "train label ambiguity": ("unseen" if not types else
                                  "always this type" if set(types) == {t} else
                                  "seen, never this type" if t not in types else
                                  "seen with mixed types"),
        "subword pieces / word": "<=1.5" if pieces <= 1.5 else "1.5-3" if pieces <= 3 else ">3",
        "has digit": str(bool(re.search(r"\d", s))),
        "non-ASCII chars": str(bool(re.search(r"[^\x00-\x7f]", s))),
        "inside `` quotes ''": str(f"`` {s} ''" in text.lower()),
    }


rows = []  # (feature dict, a_ok, b_ok, text, surface, type)
for a, b in zip(A, B):
    gold = {tuple(e) for e in a["gold"]}
    pa, pb = {tuple(e) for e in a["pred"]}, {tuple(e) for e in b["pred"]}
    for t, s in gold:
        rows.append((feats(t, s, a["text"]), (t, s) in pa, (t, s) in pb, a["text"], s, t,
                     a["generated"], b["generated"]))

print(f"gold entities: {len(rows)}")
a_only = sum(r[1] and not r[2] for r in rows)
b_only = sum(r[2] and not r[1] for r in rows)
print(f"discordant: baseline-only correct={a_only}  ngram-only correct={b_only}  "
      f"-> ngram wins {b_only / (a_only + b_only):.1%} of disagreements\n")

for fname in rows[0][0]:
    print(f"## {fname}")
    print(f"  {'bucket':<24}{'n':>7}{'recall base':>12}{'recall ngram':>13}{'diff':>8}"
          f"{'base-only':>10}{'ngram-only':>11}{'ngram win%':>11}{'sign p':>9}")
    groups = defaultdict(list)
    for r in rows:
        groups[r[0][fname]].append(r)
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[k]
        ra = np.mean([r[1] for r in g]); rb = np.mean([r[2] for r in g])
        ao = sum(r[1] and not r[2] for r in g); bo = sum(r[2] and not r[1] for r in g)
        p = binomtest(bo, ao + bo).pvalue if ao + bo else 1
        print(f"  {k:<24}{len(g):>7}{ra:>12.3f}{rb:>13.3f}{rb - ra:>+8.3f}"
              f"{ao:>10}{bo:>11}{bo / max(ao + bo, 1):>11.1%}{p:>9.3f}")
    print()

# Precision side: false positives by train frequency of predicted span
print("## False positives by train frequency of the PREDICTED span")
for name, S in (("baseline", A), ("ngram", B)):
    c = Counter()
    for s in S:
        g = {tuple(e) for e in s["gold"]}
        for t, sp in {tuple(e) for e in s["pred"]} - g:
            n = surf_cnt.get(sp, 0)
            c["0 (unseen)" if n == 0 else "1-4" if n < 5 else "5-49" if n < 50 else "50+"] += 1
    print(f"  {name:<9}" + "  ".join(f"{k}: {c[k]}" for k in ["0 (unseen)", "1-4", "5-49", "50+"]))

rng = np.random.default_rng(0)
wins = [r for r in rows if r[2] and not r[1]]
losses = [r for r in rows if r[1] and not r[2]]
for title, L in (("NGRAM-ONLY correct (ngram wins)", wins), ("BASELINE-ONLY correct (ngram loses)", losses)):
    print(f"\n### {title}: random examples")
    for i in rng.choice(len(L), N_EX, replace=False):
        f, _, _, text, s, t, ga, gb = L[i]
        print(f"  [{t}] {s!r}  (train freq {surf_cnt.get(s, 0)})\n     text : {text[:150]}\n"
              f"     base : {ga[:150]}\n     ngram: {gb[:150]}")
