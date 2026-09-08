# Test 3 — five cold-start seeds (the real Table 3)

Five independent fine-tuning runs of GraphCodeBERT text-only, cold-started from
`microsoft/graphcodebert-base`, ~50 GPU-hours total, 2026-09-07/08. Seed count
fixed at five **before** any result was seen (PAPER.md §3.3a).

Inputs verified on arrival: every `probs.npy` reproduces its own reported
accuracy, FN and FP exactly; all five arrays are distinct; all five `config`
blocks identical; all five scored the same duplicate-filtered 18,541 partition
(split_seed 42 fixed, so seed variance is not confounded with partition
variance).

| seed | accuracy | ROC-AUC | F1 | FN / FP | epochs | best val |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 87.7569% | 0.9575 | 0.8757 | 1,159 / 1,111 | 6 | 88.9854 @4 |
| 123 | 87.9510% | 0.9574 | 0.8756 | 1,293 / 941 | 6 | 89.1417 @4 |
| 2025 | 88.1614% | 0.9588 | 0.8770 | 1,325 / 870 | 6 | 89.2730 @4 |
| 7 | 87.9834% | 0.9549 | 0.8749 | 1,363 / 865 | 8 | 88.9792 @6 |
| 2718 | 87.7838% | 0.9568 | 0.8744 | 1,270 / 995 | 6 | 88.8479 @4 |

**mean 87.9273%, sample sd 0.1644pp** (itself ±0.0581pp at n=5), range 0.4045pp.

Summary: `results/test3_multiseed_summary.txt`.
Notebooks: `test_scripts/test_3_multiseed/`.

## The finding

Against the same GCB+DFG checkpoint (87.8593%), none of the five reaches
significance — and two carry the wrong sign:

| run | delta | p (McNemar) |
|---|---:|---:|
| seed 2025 | +0.302pp | 0.143 |
| seed 7 | +0.124pp | 0.567 |
| seed 123 | +0.092pp | 0.668 |
| seed 2718 | −0.075pp | 0.714 |
| seed 42 | −0.102pp | 0.624 |
| **Table 1's checkpoint** | **+0.410pp** | **0.037 — the only one** |

One-sample t-test on the five: effect +0.0680pp, t(4) = 0.925, p = 0.4074.

Table 1's text checkpoint sits **2.08 sd above the mean of its own procedure**.
It also has the smallest validation→test gap of all six runs (0.9208pp vs
0.9958–1.2285pp): its advantage is specific to the test set, which is what a
lucky draw looks like rather than a better model.

More seeds cannot rescue this. At the observed effect and sd, significance would
need ~22 seeds (~225 GPU-hours), and 9–41 across the sd's own uncertainty. The
noise is 2.4× the signal and two of five runs point the other way — this is
consistent with zero, not with an underpowered real effect.

## The comparison that settles it

Seeds 42 and 2025 differ **only** in random seed — same model, same config, same
partition. McNemar between them: |b−c| = 75, p = 0.037.

Table 1's text model vs GCB+DFG — the paper's architecture claim: |b−c| = 76,
p = 0.037.

Two runs of one model produce the same McNemar signature as the architecture
comparison. Seed-to-seed disagreement runs 6.4–7.8% of samples; test-8 puts the
GCB-text/GCB+DFG architecture difference at 7.00%, inside that range. The
architecture is not moving more predictions than a reseed does.
