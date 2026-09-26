#!/usr/bin/env python3
"""Do different architectures miss the same vulnerable samples?

Test-7b found that the deployed scanner model (UniXcoder text-only, @0.45) and
GraphCodeBERT+DFG (@0.50) share 750 false negatives, 5.6x what independence
predicts. That is only evidence that failures belong to the inputs if it is
compared with the right baseline: how heavily do two SEEDS OF ONE MODEL share
false negatives? If two different architectures overlap as much as two reseeds
of the same one, which samples fail is decided by the samples, not by the
architecture or the DFG.

Sixteen models, all scored on the same duplicate-filtered 18,541 rows in the
same order, every one at the same 0.50 threshold:
  * the six Table 2 checkpoints        results/predictions/test_probs_*.npy
  * five GraphCodeBERT text-only seeds results/test3/test3_seed*_probs.npy
  * five GraphCodeBERT + DFG seeds     results/test3b/test3b_dfgseed*_probs.npy

Overlap is reported as Jaccard |A n B| / |A u B| over false-negative sets, and
as enrichment |A n B| / (|A| |B| / positives), the ratio to what independent
errors would share. CPU only.

    python3 test_scripts/fn_overlap.py
"""
import itertools, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEEDS = (42, 123, 2025, 7, 2718)
OUT = P('results', 'fn_overlap_summary.txt')

y = np.load(P('results', 'predictions', 'test_labels.npy'))
POS = int(y.sum())


def p1(path):
    a = np.load(path)
    return a[:, 1] if a.ndim == 2 else a


def fn_set(p, thr=0.5):
    return frozenset(np.flatnonzero((y == 1) & (p <= thr)).tolist())


models = {}
for bb in ('codebert', 'graphcodebert', 'unixcoder'):
    for arm in ('text', 'dfg'):
        models[f'T2 {bb} {arm}'] = p1(P('results', 'predictions', f'test_probs_{bb}_{arm}.npy'))
for s in SEEDS:
    models[f'GCB text seed {s}'] = p1(P('results', 'test3', f'test3_seed{s}_probs.npy'))
    models[f'GCB dfg seed {s}'] = p1(P('results', 'test3b', f'test3b_dfgseed{s}_probs.npy'))

# Every array must reproduce its own reported accuracy against these labels,
# or rows are misaligned and every overlap below is meaningless.
EXPECT = {'T2 codebert text': 87.6814, 'T2 codebert dfg': 87.5196,
          'T2 graphcodebert text': 88.2692, 'T2 graphcodebert dfg': 87.8593,
          'T2 unixcoder text': 88.3447, 'T2 unixcoder dfg': 88.3124}
for k, v in EXPECT.items():
    acc = ((models[k] > 0.5).astype(int) == y).mean() * 100
    assert f'{acc:.4f}' == f'{v:.4f}', (k, acc, v)

FN = {k: fn_set(p) for k, p in models.items()}


def stats(a, b):
    A, B = FN[a], FN[b]
    inter = len(A & B)
    return dict(j=inter / len(A | B), enr=inter / (len(A) * len(B) / POS), inter=inter)


def group(pairs):
    s = [stats(a, b) for a, b in pairs]
    j = np.array([x['j'] for x in s]); e = np.array([x['enr'] for x in s])
    return len(s), j.mean(), j.min(), j.max(), e.mean()


text_seeds = [f'GCB text seed {s}' for s in SEEDS]
dfg_seeds = [f'GCB dfg seed {s}' for s in SEEDS]
T2 = [k for k in models if k.startswith('T2 ')]
bb = lambda k: k.split()[1]
arm = lambda k: k.split()[2]

GROUPS = [
    ('Same configuration, different seed', None),
    ('  GCB text-only, seed vs seed', list(itertools.combinations(text_seeds, 2))),
    ('  GCB + DFG, seed vs seed', list(itertools.combinations(dfg_seeds, 2))),
    ('Text-only vs DFG, same backbone', None),
    ('  GCB text seed vs GCB DFG seed (all 25)', list(itertools.product(text_seeds, dfg_seeds))),
    ('  Table 2, text vs DFG within backbone', [(a, b) for a, b in itertools.combinations(T2, 2)
                                                if bb(a) == bb(b)]),
    ('Different backbone', None),
    ('  Table 2, same arm', [(a, b) for a, b in itertools.combinations(T2, 2)
                             if bb(a) != bb(b) and arm(a) == arm(b)]),
    ('  Table 2, different arm', [(a, b) for a, b in itertools.combinations(T2, 2)
                                  if bb(a) != bb(b) and arm(a) != arm(b)]),
]

w = []
p = w.append
p('False-negative overlap across 16 models, all at threshold 0.50')
p('=' * 78)
p(f'Test rows 18,541; vulnerable {POS:,}. False negatives per model: '
  f'{min(len(v) for v in FN.values()):,} to {max(len(v) for v in FN.values()):,}.')
p('')
p(f'{"comparison":<44}{"pairs":>6}{"Jaccard mean":>14}{"[min, max]":>16}{"enrich.":>9}')
p('-' * 89)
summary = {}
for name, pairs in GROUPS:
    if pairs is None:
        p(name); continue
    n, jm, jlo, jhi, em = group(pairs)
    summary[name.strip()] = (jm, jlo, jhi, em)
    p(f'{name:<44}{n:>6}{jm:>14.3f}{f"[{jlo:.3f}, {jhi:.3f}]":>16}{em:>8.1f}x')
p('')

# The test-7b pair, at its original thresholds and at a matched one.
ux = p1(P('results', 'predictions', 'test7b_probs_unixcoder_text.npy'))
g = models['T2 graphcodebert dfg']
for label, a in (('UniXcoder text @0.45 vs GCB+DFG @0.50 (test-7b)', fn_set(ux, 0.45)),
                 ('UniXcoder text @0.50 vs GCB+DFG @0.50 (matched)', fn_set(ux, 0.50))):
    B = fn_set(g); i = len(a & B)
    p(f'{label}: shared {i:,}, Jaccard {i/len(a | B):.3f}, '
      f'enrichment {i/(len(a)*len(B)/POS):.1f}x')
p('')

# Samples that every model misses, and what independence would predict.
allfn = frozenset.intersection(*FN.values())
rates = [len(v) / POS for v in FN.values()]
p(f'Missed by all 16 models: {len(allfn):,} of {POS:,} vulnerable samples '
  f'({len(allfn)/POS:.1%}); independent errors would give {POS*np.prod(rates):.2e}.')
for k in ('T2 graphcodebert dfg', 'T2 unixcoder text'):
    p(f'  share of {k[3:]}\'s {len(FN[k]):,} false negatives: {len(allfn)/len(FN[k]):.1%}')
p('')

# Which of GCB+DFG's false negatives are in that core: by source, by confidence,
# and among the 20 most confident, which Section 7 of the paper reads by hand.
import json
rec = {int(r['subset_index']): r for r in
       json.load(open(P('results', 'test7_false_negatives.json')))['false_negatives']}
assert set(rec) == FN['T2 graphcodebert dfg'], 'test-7 records do not match the probabilities'
p('GraphCodeBERT + DFG false negatives that every model misses')
p('-' * 78)
for src in ('LVDAndro', 'Draper', 'Devign'):
    mine = [i for i, r in rec.items() if r['source'] == src]
    c = sum(i in allfn for i in mine)
    p(f'  {src:<9} {c:>4} of {len(mine):>4}  ({c/len(mine):.1%})')
conf = lambda idx: float(np.median([float(rec[i]['confidence_safe']) for i in idx]))
p(f'  median P(safe): core {conf(allfn):.3f}, all {len(rec):,} false negatives {conf(rec):.3f}')
top20 = sorted(rec, key=lambda i: -float(rec[i]['confidence_safe']))[:20]
p(f'  of the 20 most confident (read by hand in the paper): {sum(i in allfn for i in top20)} are in the core')
p('')
p('Provenance: test_scripts/fn_overlap.py')

txt = '\n'.join(w)
print(txt)
with open(OUT, 'w') as f:
    f.write(txt + '\n')
print(f'\nSaved -> {os.path.relpath(OUT, ROOT)}')
