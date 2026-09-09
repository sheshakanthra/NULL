# RSI(2) on NIFTY 50 — M7 prep

**Status: prepped, not yet audited.** Everything except the benchmark comparison
is built and run against the real, committed, corporate-action-adjusted OHLCV
cache. There is no committed NIFTY 50 TRI series yet (see `docs/data_sources.md`),
so `null audit` cannot run to completion until one lands.

Two results need no benchmark at all and are computed below: the **deflated
Sharpe**, and a **±25% sweep of every charge component** establishing that the
cost finding here does not depend on charge rates that were never reconciled
against a broker contract note.

## What is here

| file | what it is |
|---|---|
| `strategy.py` | RSI(2) signal generation and the 108-variant grid runner. The strategy being audited, not part of NULL's own engine — see the module docstring for why it lives under `examples/` rather than `null/`. |
| `build_run.py` | Orchestration: loads the real cache, runs the grid, prints the raw distribution and the deflated-Sharpe diagnostic, writes the artifacts below. |
| `run.json` | The audit input. `n_trials=108` declared honestly, one `TrialRecord` per variant carrying its Sharpe, weights for the best (highest net Sharpe) variant. **Per-trial return series are NOT embedded** — see the split, below. |
| `run.trials.parquet` | Sibling to `run.json` by naming convention (`<stem>.trials.parquet`): the 108 variants' real net-return series, wide format (`date` column plus one column per `param_hash`). Same data as would otherwise sit inside `run.json`. |
| `grid_report.csv` | All 108 variants: parameters, gross and net Sharpe, position-change count. Auditable by eye. |
| `cost_robustness.py` | Scales every charge component by +/-25% and re-runs the whole grid against each scaled model. 25 full runs. |
| `cost_robustness.csv` | The result of that sweep, one row per case. What the limitations band cites. |
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
  --cost-robustness examples/rsi2_nifty/cost_robustness.csv \
  --benchmark <tri.parquet> ...
```

## The fourth holding-cap value — settled

BUILD.md's M7 section originally listed the grid as `holding cap {3,5,10}` — three
values — while also stating the grid has 108 variants. `3 x 3 x 3 x 3 = 81`, not
108; `3 x 3 x 3 x 4 = 108` exactly. The list now carries a fourth value, 15, so
that `n_trials=108` is literally true rather than quietly reporting 81 while
claiming 108.

This was carried here as an open assumption while it was Sheshakanth's call to
make. It is now **ratified and recorded in the spec** — BUILD.md section 9, *Spec
correction — the fourth holding-cap value*, which also records why the alternative
(keep three caps, correct the count to 81) was rejected: nothing depends on which
caps are used, and a great deal depends on `n_trials` being true.

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
gross 0.967, net −0.027. Selecting on the number a naive backtest puts in front of
you picks a different variant, and one that ranks **69th of 108** once costs are
charged — giving up **0.45 Sharpe** against selecting on net.

The rank is the claim. It is measured across a ±25% error band on every charge
component below, and it holds throughout.

**Cost erosion at the best point is 0.531 Sharpe** — more than the entire net
Sharpe of the variant that survives it. RSI(2) mean-reversion is inherently
high-turnover, and BUILD.md's cost model — the DP charge in particular — is doing
real, visible work before NULL's statistical gates ever run.

## Are the charge rates right? No — and here is exactly what that voids

The rates in `configs/costs_india_equity.yaml` were written from the published
Indian equity charge stack and have **never been reconciled against a live broker
contract note**. That is not going to change; sourcing one would mean opening a
position purely to generate a receipt.

The lazy response is to stamp "indicative only" on every cost figure. That warns a
reader without telling them what it voids, so they either discard the whole report
or ignore the whole warning. Neither is a disclosure.

The alternative is to measure. Every charge component was scaled independently by
±25% and the full 108-variant grid re-run against each scaled model — 25 full runs
(`cost_robustness.py` → `cost_robustness.csv`), including a joint case with every
component moved at once, and both brokerage fields as a negative control (they are
provably inert at a zero configured brokerage, and came back with a delta of
exactly zero, which is how the sweep demonstrates it is perturbing the config
rather than something of its own).

### The result

| measure | across the whole ±25% range, all 25 cases |
|---|---|
| Was the best-by-gross variant ever also the best-by-net variant? | **Never — 0 of 25** |
| Rank of the gross pick on the net-ranked grid | **24th to 93rd of 108** |
| Net Sharpe given up by selecting on gross rather than net | **0.332 to 0.615** |

Even at the corner — every component 25% cheaper *simultaneously*, a larger error
than any single published rate is plausibly wrong by — selecting on gross still
lands **24th of 108** and gives up **0.332 Sharpe**. In the other direction it
reaches 93rd of 108 and 0.615.

So the demo opens with this, and only this:

> Selecting on the gross Sharpe a normal backtester puts in front of you picks a
> variant that ranks **69th of 108** once costs are charged, giving up **0.45
> Sharpe** against selecting on net. Every charge component was varied by ±25%
> and the result holds across all of it.

The sweep also killed an earlier, weaker version of the same claim — that the
gross pick loses money outright. That one is recorded in
[`docs/findings.md`](../../docs/findings.md) as item 5, a claim that failed its own
test, along with why it failed and what replaced it. It is **deliberately not
restated here**: it is a strictly weaker form of the finding above, and stating
both would only invite the more dramatic number to be the one that gets quoted.

### What the limitations band now says

`null/verdict/limitations.py` no longer says the cost numbers are "indicative
only". It separates two things the unverified rates affect differently — cost
**levels**, which carry the rate error directly, and cost-driven **rankings**,
which may or may not — and then reports whether the sensitivity was actually
measured for this run. The sentence is derived from `cost_robustness.csv` by
`null/costs/robustness.py`, not hand-written, so a report cannot claim a
robustness the sweep does not support:

```
null audit examples/rsi2_nifty/run.json \
  --trials-parquet examples/rsi2_nifty/run.trials.parquet \
  --cost-robustness examples/rsi2_nifty/cost_robustness.csv \
  --benchmark <tri.parquet> ...
```

Without `--cost-robustness` the band states, conservatively, that rate sensitivity
was **not** measured and that cost-dependent conclusions are unestablished rather
than merely imprecise. An absent sweep is not a passing one.

**On the sweep's determinism.** All 25 cases were independently re-run and
compared field-by-field against the committed CSV. **Every one reproduced
exactly** — all 16 fields, to the CSV's full 6-decimal resolution.

That check is not a one-off; it is a mode of the script, so anyone who clones this
can repeat it:

```
python examples/rsi2_nifty/cost_robustness.py --verify
```

It re-runs the sweep, compares against the committed CSV rather than overwriting
it, and exits non-zero on any mismatch. The verifier is itself tested against
planted discrepancies — a 1e-6 drift in a float, a changed integer rank, a missing
case, an extra case, and an absent CSV — because a verifier that cannot fail would
make the whole check theatre
(`tests/examples/test_cost_robustness_sweep.py`).

Why this and not a spot check: the claim above is *derived from that file*. A
sweep whose numbers move between runs would not be weak evidence for it, it would
be no evidence at all.

The reader refuses a malformed sweep rather than degrading to a reassuring
default, and it **recomputes** each case's verdict from the raw columns instead of
trusting the file's own `inversion_holds` column — if the two disagree it raises,
because a sweep that mislabels itself would otherwise launder that straight onto a
report.

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
- **The charge rates are not verified and will not be.** They are read from the
  published charge stack, never reconciled against a contract note. This does not
  make every number here unreliable in general, and the report no longer says it
  does: every cost **level** carries that error directly, while the ranking result
  above was measured across a ±25% error band on every component and survives it
  in all 25 cases.
