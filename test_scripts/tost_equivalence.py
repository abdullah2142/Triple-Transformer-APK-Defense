#!/usr/bin/env python3
"""Two one-sided tests (TOST) for equivalence between the five cold-start
GraphCodeBERT text-only seeds and the fixed GraphCodeBERT+DFG checkpoint.

Answers a reviewer point that a non-significant difference is absence of
evidence rather than evidence of absence. Reports the tightest equivalence
bound the data support at alpha = 0.05.

Asymmetry to keep in view: the five runs vary the TEXT arm only. The DFG arm
is one checkpoint held fixed, so this bounds the text arm's distribution around
that point -- not the distance between two distributions. Seeding both arms
(~50 GPU-hours) is what would give the paired version.

    python3 test_scripts/tost_equivalence.py
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_test3 import t_sf_two_sided

SEEDS = [87.7569, 87.9510, 88.1614, 87.9834, 87.7838]   # results/test3/
DFG   = 87.8593                                          # Table 2, GCB+DFG
ALPHA = 0.05

def tost(seeds, ref, delta, alpha=ALPHA):
    n = len(seeds)
    m = sum(seeds) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in seeds) / (n - 1))
    se = sd / math.sqrt(n)
    d, df = m - ref, n - 1
    t_lo, t_hi = (d + delta) / se, (d - delta) / se
    p_lo, p_hi = t_sf_two_sided(t_lo, df) / 2, t_sf_two_sided(t_hi, df) / 2
    return d, se, t_lo, t_hi, p_lo, p_hi, (p_lo < alpha and p_hi < alpha)

if __name__ == '__main__':
    n = len(SEEDS); m = sum(SEEDS) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in SEEDS) / (n - 1))
    print(f'n = {n}   mean = {m:.4f}%   sd = {sd:.4f}pp   reference = {DFG:.4f}%')
    print(f'observed difference = {m - DFG:+.4f}pp\n')
    print(f'{"bound":>7}{"p_lower":>10}{"p_upper":>10}   equivalent at alpha=0.05?')
    print('-' * 52)
    tightest = None
    for delta in (0.60, 0.50, 0.40, 0.30, 0.25, 0.22, 0.20, 0.15):
        *_, p_lo, p_hi, ok = tost(SEEDS, DFG, delta)
        if ok:
            tightest = (delta, p_lo, p_hi)
        print(f'{delta:>6.2f} {p_lo:>10.4f}{p_hi:>10.4f}   {"YES" if ok else "no"}')
    if tightest:
        d, a, b = tightest
        print(f'\nTightest bound supported: +/-{d:.2f}pp  (p = {a:.4f}, {b:.4f})')
        print('This is the figure quoted in Section V-D of the manuscript.')
