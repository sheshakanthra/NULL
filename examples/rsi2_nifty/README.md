# RSI(2) on NIFTY 50 — M7 prep

**Status: prepped, not yet audited.** Everything except the benchmark comparison
is built and run against the real, committed, corporate-action-adjusted OHLCV
cache. There is no committed NIFTY 50 TRI series yet (see `docs/data_sources.md`),
so `null audit` cannot run to completion until one lands. One statistic needs no
benchmark at all — deflated Sharpe — and it is computed below.

## What is here

| file | what it is |
|---|---|
| `strategy.py` | RSI(2) signal generation and the 108-variant grid runner. The strategy being audited, not part of NULL's own engine — see the module docstring for why it lives under `examples/` rather than `null/`. |
| `build_run.py` | Orchestration: loads the real cache, runs the grid, prints the raw distribution and the deflated-Sharpe diagnostic, writes the artifacts below. |
| `run.json` | The audit input. `n_trials=108` declared honestly, one `TrialRecord` per variant carrying its Sharpe, weights for the best (highest net Sharpe) variant. **Per-trial return series are NOT embedded** — see the split, below. |
| `run.trials.parquet` | Sibling to `run.json` by naming convention (`<stem>.trials.parquet`): the 108 variants' real net-return series, wide format (`date` column plus one column per `param_hash`). Same data as would otherwise sit inside `run.json`. |
| `grid_report.csv` | All 108 variants: parameters, gross and net Sharpe, position-change count. Auditable by eye. |
| `sensitivity.json` | The `SensitivityResult` built directly from the same grid — see "On the sensitivity surface" below. |

## The run.json / run.trials.parquet split

Embedding all 108 return series inline made `run.json` 18.7 MB — unwieldy for a
repository people clone to see a demo, and GitHub warns above 50 MB per file. The
contract already permits this: `StrategyRun.trials` may be "empty or a subset ...
never a substitute for `n_trials`," so a lean `run.json` is contract-legitimate on
its own.

But shipping it alone would silently degrade PBO to `NOT_COMPUTABLE` the day
someone actually runs `null audit` on it — a real loss of evidence, not a
formatting change. `null/cli.py` gained a `--trials-parquet PATH` flag for exactly
this: given the sibling file, it rehydrates full per-trial returns onto the parsed
`StrategyRun` before the gates run (`enrich_trials_from_parquet`, using
`model_copy` since both `StrategyRun` and `TrialRecord` are frozen), so the split
costs nothing but disk layout. It refuses to proceed on a partial match — every
trial's `param_hash` must have a column, or it raises, rather than silently
enriching some trials and leaving others empty. Result: `run.json` shrank from
18.7 MB to 1.4 MB; the sibling parquet is 3.1 MB. Determinism is asserted on both
files independently (`tests/examples/test_rsi2_nifty.py`), and the round-trip
back to the original per-variant returns is asserted byte-for-byte, not just
"some value present."

To audit with full PBO evidence once TRI lands:

```
null audit examples/rsi2_nifty/run.json \
  --trials-parquet examples/rsi2_nifty/run.trials.parquet \
  --benchmark <tri.parquet> ...
```

## A stated assumption: the fourth holding-cap value

BUILD.md's M7 section lists the grid as `holding cap {3,5,10}` — three values —
while also stating the grid has 108 variants. `3 x 3 x 3 x 3 = 81`, not 108;
`3 x 3 x 3 x 4 = 108` exactly. `strategy.py` adds a fourth holding-cap value, 15,
as the natural continuation of the stated sequence, so that `n_trials=108` is
literally true rather than quietly reporting 81 while claiming 108. **This is an
assumption, not a correction on Sheshakanth's authority, and needs confirmation.**

## The raw distribution, before any NULL gate has run

This is what the user of a normal backtester sees. It is not what `null audit`
will eventually say — that gap is the whole point of BUILD.md's M7 section.

```
               gross       net
min           -0.029      -0.352
p25            0.245      -0.066
median         0.363       0.010
mean           0.417       0.024
p75            0.591       0.122
max            0.967       0.422
```

**Best by net Sharpe:** period=2, entry=15, exit=70, holding_cap=15 →
gross 0.952, net 0.422, 20,888 position changes across the 50-name universe.

**Best by gross Sharpe:** period=2, entry=15, exit=70, holding_cap=3 →
gross 0.967, **net −0.027**. Selecting on the number a naive backtest would show
you picks a strategy that loses money net of cost. This is not a contrived
example — it is the actual best-by-gross variant in this actual grid.

**Cost erosion at the best point is 0.531 Sharpe** — more than the entire net
Sharpe of the variant that survives it. RSI(2) mean-reversion is inherently
high-turnover, and BUILD.md's cost model — the DP charge in particular — is doing
real, visible work before NULL's statistical gates ever run.

## Deflated Sharpe — needs no benchmark, computed now

This is the number BUILD.md §6.1 exists for, and it does not need TRI: only the
grid's own trial Sharpes and the best variant's own realised return series.

```
Best variant's own net Sharpe:                    0.422
Expected max Sharpe from 108 trials,
  3,700 observations, by chance alone:             0.411
Realised skew:                                    -3.084
Realised kurtosis:                                69.046
Deflated Sharpe  P[true Sharpe > 0]:                0.516
```

**The best variant's Sharpe (0.422) is barely above what 108 trials of pure noise
would be expected to produce (0.411) — a margin of 0.011.** Deflated Sharpe of
0.516 is a coin flip: after adjusting for 108 trials and the return series' own
shape, the evidence that this strategy has a real edge is indistinguishable from
no evidence at all. Against BUILD.md's `deflated_sharpe > 0.95` gate this is not
close.

The skew (−3.08) and kurtosis (69, versus 3 for a normal distribution) are large
but genuine, not a data artifact — checked directly: the worst day in the best
variant's 3,700-day series is −10.1%, against a typical day of roughly ±0.5%. One
sharp loss day among thousands of calm ones is exactly the signature a
mean-reversion strategy produces when it buys an oversold name that keeps
falling, and it is what drags deflated Sharpe down further than the raw Sharpe
numbers alone suggest.

**A correction, on the record.** The first version of this calculation reported
an expected-max-Sharpe of 6.5 — apparently swamping the observed 0.422 outright.
That number was wrong: `deflated_sharpe_ratio` expects trial Sharpes in
*per-period* units and annualises internally, and the first pass fed it
*already-annualised* Sharpes, double-annualising the variance across trials and
inflating the result by a further factor of `sqrt(252) ≈ 15.9` — confirmed in
isolation on a synthetic check (12.2 vs. the correct 0.77) before it was trusted
on the real grid. `per_period_trial_sharpes()` in `build_run.py` computes the
correct per-period figure directly from each variant's raw returns, and a test
(`test_per_period_trial_sharpes_are_not_annualised`) pins the distinction so it
cannot regress silently. The corrected finding — deflated Sharpe at coin-flip
level, not "swamped by noise outright" — is the one that stands.

## On the sensitivity surface

`build_sensitivity` does not reuse `null/sensitivity/neighborhood.py`'s generic
+/-1/+/-2 stepper. That module was built for parameters with an even, arbitrary
step size; this grid has four parameters with **unequal level counts** (3, 3, 3,
4), and forcing an uneven grid through a fixed-step scan risks index wraparound
at a boundary — the same family of silent bug this project has already found
twice in weight-alignment code this session. Instead, the neighbourhood is
defined directly as index Hamming-distance 1 from the best point on the real
108-point grid: every variant differing from the best in exactly one parameter,
by exactly one index step. A parameter sitting at the edge of its own list
(e.g. the best point already at `holding_cap=15`, the top of the range) simply
has one fewer neighbour counted, rather than wrapping to the other end of the
list or being clamped.

Measured `neighborhood_ratio: 0.770` against the gate's 0.60 threshold — the
result, on this dimension, looks like a plateau rather than a spike. That
finding is unaudited by everything else and should not be read on its own,
particularly next to a deflated Sharpe sitting at 0.516.

## What is deliberately not done here

- **No benchmark comparison, no PBO, no reality check, no verdict.** Building
  those now would mean either fabricating a benchmark series or running the audit
  on an obviously incomplete `Evidence` object and calling it a result. Neither is
  honest. This step is separate and waits on the TRI cache.
