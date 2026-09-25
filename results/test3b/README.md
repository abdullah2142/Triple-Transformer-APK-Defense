# Test 3b — five cold-start seeds, DFG arm (the paired half of Table 3)

Five independent fine-tuning runs of GraphCodeBERT + DFG-aware attention,
cold-started from `microsoft/graphcodebert-base`, on Kaggle, completed by 2026-09-25.
The same five seeds as the text arm (`results/test3/`), so the runs pair by seed.
Design, seed count and both tests were fixed **before** any DFG seed ran
(PAPER.md §3.3b, pre-registered 2026-09-18). Notebooks:
`test_scripts/test_3b_dfg_multiseed/`.

The `_results.json` files are gitignored (`*.json`, repo-wide), as in `results/test3/`.
Each ships inside its seed's zip; unzip both arms in place before rerunning
`test_scripts/aggregate_test3b_paired.py`.

## Verified on arrival

- Every `probs.npy` (18,541 float64, all in [0, 1]) reproduces its own reported
  accuracy, FN, FP and F1 exactly against `results/predictions/test_labels.npy`;
  the same labels reproduce all five text-arm runs, so both arms are scored on
  the same rows in the same order.
- ROC-AUC agrees to within 1e-6, not bit-for-bit: FP16 softmax leaves 8,185–11,606
  tied probabilities per run, and the notebook's argsort ranking breaks ties in
  arbitrary order. Tie-averaged AUC agrees to within 5e-7. Invisible at the
  four decimals reported.
- Every JSON matches its Kaggle log: accuracy, FN/FP, ROC-AUC, F1, best validation
  accuracy, stop reason, and the per-epoch validation history.
- All five `config` blocks are identical: cold start, 384 code + 128 DFG,
  16 × 2 = 32 effective, 10 epochs / patience 2, lr 2e-5, 10% warmup, split seed 42,
  `cudnn_deterministic: false` (as in the text arm).
- All five arrays are distinct from one another, from every text-arm run, and from
  the Table 2 GCB+DFG checkpoint. Seeds 7 and 123 share an accuracy (both 2,268
  errors) but disagree on 1,388 test rows: different models.
- Each run's pre-flight printed identical DFG statistics (median/mean 65/80.2 nodes,
  11.3% at the 128-node cap, mask 0.564 open, 126 code–node links).
- Seed 2718 has one probability exactly 0.5, scored negative by the `>` threshold,
  the same rule the text arm used.

| seed | accuracy | ROC-AUC | F1 | FN / FP | epochs | best val | ended by | h/epoch |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 42 | 87.9348% | 0.9556 | 0.8764 | 1,219 / 1,018 | 7 | 88.3291 @5 | early stopping | 1.19 |
| 123 | 87.7677% | 0.9523 | 0.8751 | 1,207 / 1,061 | 9 | 88.5416 @8 | **time budget** | 1.19 |
| 2025 | 87.9564% | 0.9583 | 0.8746 | 1,363 / 870 | 7 | 88.5979 @5 | early stopping | 1.29 |
| 7 | 87.7677% | 0.9560 | 0.8781 | 983 / 1,285 | 7 | 88.5291 @5 | early stopping | 1.34 |
| 2718 | 87.7515% | 0.9578 | 0.8717 | 1,438 / 833 | 5 | 88.2540 @3 | early stopping | 1.21 |

**mean 87.8356%, sample sd 0.1009pp**, range 0.2050pp (text arm: 87.9273%, 0.1644pp).

**Seed 123 was cut by the clock, not by early stopping.** After epoch 9 (patience
1/2, best at epoch 8) the predictive budget projected 11.90h against the 11.0h limit
and stopped. The truncation could only have cost it a better checkpoint, so it can
only have moved that seed's difference toward the text arm. Reported as run; not
re-run, not dropped. The text arm never met its own limit (all five early-stopped).

## The pre-registered result

Summary: `results/test3b_paired_summary.txt` (`test_scripts/aggregate_test3b_paired.py`).

| seed | text | DFG | Δ (text − DFG) |
|---|---:|---:|---:|
| 42 | 87.7569% | 87.9348% | −0.1780pp |
| 123 | 87.9510% | 87.7677% | +0.1834pp |
| 2025 | 88.1614% | 87.9564% | +0.2050pp |
| 7 | 87.9834% | 87.7677% | +0.2157pp |
| 2718 | 87.7838% | 87.7515% | +0.0324pp |

- **Primary, paired t-test:** mean Δ +0.0917pp, sd 0.1680pp, t(4) = 1.221,
  **p = 0.2893. Not significant.** 4/5 seeds favour text.
- **Secondary, TOST at ±0.25pp:** lower p = 0.0052, upper p = 0.0514.
  **Not equivalent at α = 0.05.** Rejects "DFG better by ≥ 0.25pp"; narrowly fails
  to reject "text better by ≥ 0.25pp".

Neither pre-registered row fits: the difference is not significant, and equivalence
at the fixed bound is not established. It is reported that way. The bound is not
moved and no seeds are added. Doing either after seeing p = 0.0514 is what the
pre-registration forbids.

Descriptive, from the same t(4) distribution and not a separate test: the 95% CI on
Δ is [−0.117, +0.300]pp, and the one-sided 95% bound on any DFG *gain* is +0.068pp.
The unresolved side is the one where DFG is worse.

Pairing bought little. The sd of the differences (0.168pp) sits near what two
independent arms would give (√(0.1644² + 0.1009²) = 0.193pp): a shared seed
number does not produce shared randomness across two different input pipelines.
