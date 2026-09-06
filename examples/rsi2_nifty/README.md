# RSI(2) on NIFTY 50 — M7 prep

**Status: prepped, not yet audited.** Everything except the benchmark comparison
is built and run against the real, committed, corporate-action-adjusted OHLCV
cache. There is no committed NIFTY 50 TRI series yet (see `docs/data_sources.md`),
so `null audit` cannot run to completion on `run.json` until one lands.

## What is here

| file | what it is |
|---|---|
| `strategy.py` | RSI(2) signal generation and the 108-variant grid runner. The strategy being audited, not part of NULL's own engine — see the module docstring for why it lives under `examples/` rather than `null/`. |
| `build_run.py` | Orchestration: loads the real cache, runs the grid, prints the raw Sharpe distribution, writes the artifacts below. |
| `run.json` | A `StrategyRun`, `n_trials=108` declared honestly, `trials` populated with all 108 variants' real net-of-cost return series (not an assumed variance). Selected variant is the one with the highest **net** Sharpe. |
| `grid_report.csv` | All 108 variants: parameters, gross and net Sharpe, position-change count. Auditable by eye. |
| `sensitivity.json` | The `SensitivityResult` built directly from the same grid — see "On the sensitivity surface" below. |

## A stated assumption: the fourth holding-cap value

BUILD.md's M7 section lists the grid as `holding cap {3,5,10}` — three values —
while also stating the grid has 108 variants. `3 x 3 x 3 x 3 = 81`, not 108;
`3 x 3 x 3 x 4 = 108` exactly. `strategy.py` adds a fourth holding-cap value, 15,
as the natural continuation of the stated sequence, so that `n_trials=108` is
literally true rather than quietly reporting 81 while claiming 108. **This is an
assumption, not a correction on Sheshakanth's authority, and needs confirmation.**

## The raw distribution, before any NULL gate has run

This is what the user of a normal backtester sees. It is not what `null audit`
will eventually say — that gap is the whole point of BUILD.md's M7 section, and
the numbers below only mean something once compared against a verdict later.

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
high-turnover (the best variant alone makes 20,888 position changes over 15
years across 50 names), and BUILD.md's cost model — the DP charge in particular —
is doing real, visible work before NULL's statistical gates ever run.

None of this is NULL's judgement yet. It is the honest pre-audit picture: what a
retail trader running this exact grid search would see, net of real transaction
costs, before anyone asks whether 108 trials and one selected winner constitute
evidence of anything.

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
finding is unaudited by everything else and should not be read on its own.

## What is deliberately not done here

- **No benchmark comparison, no DSR, no PBO, no reality check, no verdict.**
  Building those now would mean either fabricating a benchmark series or running
  the audit on an obviously incomplete Evidence object and calling it a result.
  Neither is honest. This step is separate and waits on the TRI cache.
- **`run.json` is 18.7 MB.** This is because `trials` carries real net-return
  series for all 108 variants rather than a subset or an assumed variance, per
  the explicit instruction that the trial matrix be populated from the grid.
  This is a genuine trade-off — deferred rather than decided here — between
  artifact completeness (real PBO evidence, not a guessed variance) and
  repository size.
