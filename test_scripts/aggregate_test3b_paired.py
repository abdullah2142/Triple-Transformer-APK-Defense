#!/usr/bin/env python3
"""Paired analysis of the two multi-seed arms.

Test 3  (results/test3/)    GraphCodeBERT text-only, five cold-start seeds
Test 3b (results/test3b/)   GraphCodeBERT + DFG,     the same five seeds

Both arms score the identical duplicate-filtered 18,541 partition, so the runs
pair by seed and the analysis below is paired:

    delta_s = accuracy(text, s) - accuracy(DFG, s)

PRE-REGISTERED 2026-09-18, before any DFG seed was run (PAPER.md 3.3b):
  * n = 5, seeds 42/123/2025/7/2718
  * primary   : paired t-test on the five differences
  * secondary : TOST equivalence at +/-0.25 pp, the bound Section V-D reports
  * reported at n = 5 whatever the outcome; a reversal is reported as a reversal

    python3 test_scripts/aggregate_test3b_paired.py
"""
import argparse, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_test3 import t_sf_two_sided

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228}
T_CRIT_1 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
            8: 1.860, 9: 1.833, 10: 1.812}
EQUIV_BOUND = 0.25          # pre-registered


def load(pattern, key):
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        if not d.get('config', {}).get('cold_start'):
            print(f'  SKIP {p}: not a cold-start run')
            continue
        out[d['seed']] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text-dir', default='results/test3')
    ap.add_argument('--dfg-dir', default='results/test3b')
    ap.add_argument('--out', default='results/test3b_paired_summary.txt')
    a = ap.parse_args()

    text = load(os.path.join(a.text_dir, 'test3_seed*_results.json'), 'text')
    dfg = load(os.path.join(a.dfg_dir, 'test3b_dfgseed*_results.json'), 'dfg')
    if not text or not dfg:
        raise SystemExit(f'need both arms: found {len(text)} text, {len(dfg)} DFG')

    paired = sorted(set(text) & set(dfg))
    unpaired = sorted((set(text) | set(dfg)) - set(paired))

    w = []
    p = w.append
    p('Test 3b -- paired multi-seed comparison, text vs DFG')
    p('=' * 78)
    p(f'Seeds paired : {len(paired)}  ({", ".join(str(s) for s in paired)})')
    if unpaired:
        p(f'Unpaired     : {unpaired}  -- excluded from the paired analysis')
    p('')

    p(f'{"seed":>6} {"text":>11} {"DFG":>11} {"delta":>10} {"text FN":>9} {"DFG FN":>8}')
    p('-' * 78)
    deltas = []
    for s in paired:
        t_acc = text[s]['accuracy'] * 100
        d_acc = dfg[s]['accuracy'] * 100
        deltas.append(t_acc - d_acc)
        p(f'{s:>6} {t_acc:10.4f}% {d_acc:10.4f}% {t_acc-d_acc:+9.4f}pp '
          f'{text[s]["false_negatives"]:9,d} {dfg[s]["false_negatives"]:8,d}')
    p('')

    n = len(deltas)
    if n < 2:
        p('Fewer than two paired seeds; no test possible.')
        print('\n'.join(w)); return

    m = sum(deltas) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1

    p('Paired t-test on the per-seed differences')
    p('-' * 78)
    p(f'  mean difference   {m:+.4f}pp   (text minus DFG)')
    p(f'  sd of differences {sd:.4f}pp')
    p(f'  seeds favouring text: {sum(1 for d in deltas if d > 0)}/{n}')
    if se > 0:
        t = m / se
        pv = t_sf_two_sided(t, df)
        p(f'  t({df}) = {t:.3f}   critical {T_CRIT.get(df, float("nan")):.3f}   p = {pv:.4f}')
        p(f'  verdict: {"SIGNIFICANT" if pv < 0.05 else "not significant"} at 0.05')
    p('')

    p(f'TOST equivalence, pre-registered bound +/-{EQUIV_BOUND}pp')
    p('-' * 78)
    if se > 0:
        tc = T_CRIT_1.get(df, float('nan'))
        t_lo = (m + EQUIV_BOUND) / se
        t_hi = (m - EQUIV_BOUND) / se
        p_lo = t_sf_two_sided(t_lo, df) / 2
        p_hi = t_sf_two_sided(t_hi, df) / 2
        ok = p_lo < 0.05 and p_hi < 0.05
        p(f'  lower: t = {t_lo:.3f}  p = {p_lo:.4f}')
        p(f'  upper: t = {t_hi:.3f}  p = {p_hi:.4f}   (one-sided critical {tc:.3f})')
        p(f'  verdict: {"EQUIVALENT" if ok else "NOT equivalent"} within '
          f'+/-{EQUIV_BOUND}pp at alpha = 0.05')
        if ok:
            for b in (0.22, 0.20, 0.18, 0.15, 0.12, 0.10):
                l = t_sf_two_sided((m + b) / se, df) / 2
                h = t_sf_two_sided((m - b) / se, df) / 2
                if not (l < 0.05 and h < 0.05):
                    p(f'  tightest bound that still holds is between {b:.2f} and the next step up')
                    break
    p('')

    p('Each arm on its own')
    p('-' * 78)
    for name, arm in (('text', text), ('DFG', dfg)):
        v = [arm[s]['accuracy'] * 100 for s in paired]
        mm = sum(v) / len(v)
        ss = math.sqrt(sum((x - mm) ** 2 for x in v) / (len(v) - 1))
        p(f'  {name:<5} mean {mm:.4f}%   sd {ss:.4f}pp   range {max(v)-min(v):.4f}pp')
    p('')
    p('  A paired design removes whatever variance the two arms share at a given')
    p('  seed, so the sd of the differences may be smaller than either arm\'s own.')
    p('')
    p(f'Provenance: aggregate_test3b_paired.py over {a.text_dir} and {a.dfg_dir}')

    text_out = '\n'.join(w)
    print(text_out)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        f.write(text_out + '\n')
    print(f'\nSaved -> {a.out}')


if __name__ == '__main__':
    main()
