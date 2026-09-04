# P0 — the PBO gate cannot reliably reject a noise grid at a 0.50 threshold

Status: **open, blocking M4 and M6.** Found while building M4. Feature work on the
statistical adversary is stopped per CLAUDE.md ("If a golden fixture's verdict
flips, that is a P0. Stop feature work.").

Nothing here is a decision. All three options at the bottom are Sheshakanth's.

## The finding

`overfit_grid`'s PBO verdict depends on the seed. Seven seeds, same construction
(2y, 1,000 MA-crossover variants over one GBM path), gate is `PBO < 0.5` passes:

| seed | observed SR | DSR | DSR verdict | PBO | PBO verdict |
|---|---|---|---|---|---|
| 20260905 | 0.821 | 0.3688 | REJECT | 0.7719 | REJECT |
| 1 | 1.742 | 0.7524 | REJECT | 0.5871 | REJECT |
| 2 | 0.343 | 0.0904 | REJECT | 0.6019 | REJECT |
| 7 | 1.673 | 0.7926 | REJECT | 0.5118 | REJECT |
| 42 | 0.091 | 0.0459 | REJECT | 0.7704 | REJECT |
| 99 | 1.322 | 0.3921 | REJECT | 0.2421 | **PASS** |
| 12345 | 0.927 | 0.1225 | REJECT | 0.2138 | **PASS** |

**Deflated Sharpe rejects on 7 of 7. PBO rejects on 5 of 7.**

## Why — this is not an implementation bug

For a candidate set with no true edge, every variant has a true Sharpe of zero,
so the in-sample winner's expected out-of-sample rank is exactly the median.
**PBO's expected value under the null is therefore 0.50 — precisely where
BUILD.md §6.2 puts the threshold.** The gate is a coin flip on noise by
construction, not by accident.

Measured across four constructions, six seeds each, PBO on pure noise:

| construction | PBO across seeds | min |
|---|---|---|
| independent, 300 cols, 504 obs | 0.59 0.76 0.44 0.44 0.56 0.43 | 0.43 |
| independent, 1000 cols, 504 obs | 0.61 0.50 0.55 0.55 0.58 0.61 | 0.50 |
| independent, 1000 cols, 252 obs | 0.60 0.32 0.50 0.44 0.49 0.72 | 0.32 |
| independent, 2000 cols, 252 obs | 0.53 0.43 0.41 0.46 0.44 0.52 | 0.41 |

No construction is robustly above 0.50.

## The worse half

PBO also failed to separate a genuine edge from noise in the same harness:

| case | PBO across seeds |
|---|---|
| pure noise | 0.59 0.76 0.44 0.44 0.56 0.43 |
| one strong edge (Sharpe ~1.7 annual) | 0.45 0.77 0.39 0.47 0.56 0.40 |
| one modest edge (Sharpe ~0.7 annual) | 0.59 0.76 0.44 0.44 0.56 0.43 |

The modest-edge row is byte-identical to noise. The reason is diagnostic rather
than damning: with 300 candidates over 252 training observations, the maximum of
299 noise Sharpes (~0.18 per period) exceeds the true edge's own Sharpe (~0.11
per period), so **the genuine strategy is almost never the one selected.** PBO is
measuring the selection process faithfully; the selection process simply cannot
find a real edge in a candidate set that large relative to the sample.

That is itself a finding worth keeping: it says something true about grid search,
and it means a low PBO must never be read as evidence that an edge is real.

## Corrections to earlier claims

Two things previously written down were wrong and have been removed:

1. The `overfit_grid` docstring claimed variance heterogeneity across variants
   drives PBO. Isolating the variable did not support it — holding variance
   constant and varying correlation, and vice versa, both produced non-monotonic
   results. The mechanism was asserted, not measured.
2. `test_a_genuine_persistent_winner_produces_low_pbo` passed on its seed, but
   for the wrong reason: the injected edge was never strong enough to be selected,
   so the test was measuring noise either way.

## Options

**A. Lower the PBO threshold.** Move the gate to, say, `PBO < 0.35`, below the
null's expectation of 0.50, so a noise grid reliably fails. Cost: needs evidence
about where genuinely-selected strategies land, and the table above suggests real
edges may not score meaningfully lower — so this risks rejecting everything.

**B. Demote PBO from a hard gate to an evidence panel.** Deflated Sharpe rejects
`overfit_grid` on 7 of 7 and is the robust catch. PBO still gets reported, with
its instability stated. Cost: departs from BUILD.md §7's gate list, and leaves
overfit_grid caught by one gate rather than two.

**C. Replace the point estimate with an interval.** Reject only when a bootstrap
lower bound on PBO exceeds the threshold, which makes the gate robust to the
sampling noise documented above. Cost: more machinery, and it needs its own
calibration against both fixtures.

## What is not in question

The deflated Sharpe gate is behaving well and separates the fixtures cleanly:

```
overfit_grid           observed 0.82 -> deflated 0.3688   (n_trials=5,000)
true_edge_synthetic    observed 0.60 -> deflated 0.9711   (n_trials=1)
separation 0.6023
```

`true_edge_synthetic` passes on 7 of 7 seeds at 0.9709–0.9712. The higher observed
Sharpe deflating to far less is the entire thesis working as intended.
