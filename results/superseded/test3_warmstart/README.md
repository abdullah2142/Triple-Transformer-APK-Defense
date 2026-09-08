# Test 3 — three WARM-START seeds (superseded)

Committed in `d43babd` ("Added results"). **These are the runs behind Table 3's
original ±0.10% claim.** Kept so that number can be traced; do not cite them.

| seed | accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| 42 | 88.8278% | 0.9588 | 0.9596 |
| 123 | 88.8928% | 0.9609 | 0.9625 |
| 2025 | 89.0728% | 0.9596 | 0.9617 |

mean **88.9311%**, sample sd **0.1270pp**.

## Why superseded

The notebook that produced them loaded our own already-fine-tuned
`best_model_text_only.bin`, stripped the classifier, and retrained the head.
Every "seed" therefore started from the same converged encoder that had already
seen the whole training set. What varied between runs was head initialisation
and data order — not fine-tuning. That measures the wrong thing for Table 3,
whose entire purpose is to bound the noise on Table 2's DFG deltas, and it
understates the spread in the direction that makes those deltas look more
trustworthy than the evidence supports.

Two further mismatches against Table 1's procedure: a 5-epoch ceiling (vs 10,
patience 2) and 384-token training windows (vs 512).

These accuracies sit ~1pp above the cold-start runs, but the cause cannot be
pinned down from what was saved: the files record only accuracy, ROC-AUC and
PR-AUC — no test-set size, no config block. Warm-start memorisation is the
obvious candidate (a duplicate filter cannot un-see rows the encoder was
fine-tuned on), but these outputs may also predate the duplicate filter itself,
in which case part of the gap is a larger, dirtier test set. Both would push the
same direction and the evidence does not separate them. Treat the ~1pp as
unexplained, not as a measured warm-start effect.

Superseded by the five cold-start runs in `results/test3/`, notebooks in
`test_scripts/test_3_multiseed/`. The retired notebooks are in
`test_scripts/superseded/test_3_multiseed_warmstart/`.
