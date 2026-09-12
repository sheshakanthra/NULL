# NULL — Master Build Spec

**A deterministic strategy audit engine. Default verdict: REJECT.**

Version 0.1 · Owner: Sheshakanth · Drives: Claude Code

---

## 0. Read this first

NULL is not a trading bot. NULL is not a backtester. NULL is a **prosecutor**.

You hand it a strategy's realised positions. It tries to prove the strategy has no edge.
It reports PASS only when every attempt to kill it fails.

There are hundreds of open-source backtesters. There is almost nothing that assumes the
strategy is garbage and makes it earn its way out. That asymmetry is the entire product.

### Anti-goals — do not build these into NULL

| Not in scope | Why |
|---|---|
| Order placement, broker write access | NULL never touches execution. Separate system, separate repo. |
| Signal generation, strategy discovery | NULL judges. It does not propose. |
| **Any LLM call inside the audit path** | The judge must be bit-for-bit reproducible. LLMs write strategies; NULL scores them. If an LLM can influence a verdict, the verdict is worthless. |
| A UI | CLI + JSON artifact. A report renderer is M5, and it is dumb HTML. |
| Live/streaming data | Historical, point-in-time, batch. Full stop. |

### Prime directive

> Same inputs → byte-identical `verdict.json`. Always. Enforced by a test.

Seed everything. No wall-clock in the audit path. No dict iteration order dependence.
No network calls at audit time (data fetch is a separate, cached, offline-first stage).

---

## 1. Repository layout

```
null/
├── pyproject.toml
├── README.md
├── null/
│   ├── __init__.py
│   ├── contracts.py          # Pydantic models — the ONLY interface anyone codes against
│   ├── costs/
│   │   ├── __init__.py
│   │   ├── model.py          # CostModel protocol
│   │   ├── india_equity.py   # config-driven charge stack
│   │   └── slippage.py       # spread + square-root impact
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── buyhold.py        # the thing that kills 40/40
│   │   └── risk_match.py     # de-lever / cash-adjust before comparing
│   ├── leakage/
│   │   ├── __init__.py
│   │   ├── timestamps.py     # signal→fill alignment
│   │   ├── survivorship.py   # point-in-time universe
│   │   └── corporate_actions.py
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── deflated_sharpe.py
│   │   ├── pbo.py            # CSCV
│   │   ├── reality_check.py  # White's RC / Hansen SPA, stationary bootstrap
│   │   ├── bootstrap.py      # Politis–Romano stationary bootstrap
│   │   └── mtrl.py           # minimum track record length
│   ├── partition/
│   │   ├── __init__.py
│   │   ├── walkforward.py    # purge + embargo
│   │   └── regimes.py        # vol / trend / era splits
│   ├── sensitivity/
│   │   ├── __init__.py
│   │   └── neighborhood.py   # plateau vs spike
│   ├── verdict/
│   │   ├── __init__.py
│   │   ├── gates.py          # each gate = pure fn(Evidence) -> GateResult
│   │   └── engine.py         # AND of all gates, default REJECT
│   ├── report/
│   │   └── render.py         # verdict.json -> static HTML
│   └── cli.py
├── configs/
│   ├── costs_india_equity.yaml
│   ├── costs_india_fno.yaml
│   └── gates_default.yaml
├── tests/
│   ├── golden/               # synthetic strategies with KNOWN verdicts
│   └── unit/
└── data/                     # gitignored; cached parquet
```

---

## 2. Contracts (`null/contracts.py`) — build this first, freeze it

Everything else codes against these. Pydantic v2, frozen models.

```python
class Bar(BaseModel):
    ts: datetime          # tz-aware, IST, bar CLOSE time
    symbol: str
    open: float; high: float; low: float; close: float
    volume: float
    adv_20: float | None  # 20d avg daily value traded, for impact model

class TargetWeight(BaseModel):
    """Canonical strategy output. NOT a trade list."""
    ts: datetime          # decision time
    symbol: str
    weight: float         # fraction of equity, signed. -0.5 = 50% short.

class StrategyRun(BaseModel):
    strategy_id: str
    param_hash: str       # hash of the exact param set used
    n_trials: int         # HOW MANY VARIANTS WERE TRIED TO GET HERE. Required.
    universe: list[str]
    weights: list[TargetWeight]
    decision_lag_bars: int = 1   # >=1 enforced
    initial_capital: float

class Evidence(BaseModel):
    """Everything the gates consume. Produced by the audit pipeline."""
    equity_curve: Series
    benchmark_curve: Series
    net_returns: Series
    gross_returns: Series
    cost_breakdown: dict[str, float]
    turnover_annual: float
    time_in_market: float
    metrics: PerfMetrics
    benchmark_metrics: PerfMetrics
    alpha: RegressionResult
    deflated_sharpe: float
    pbo: float
    reality_check_p: float
    mtrl_years: float
    walkforward: list[FoldResult]
    regimes: dict[str, PerfMetrics]
    sensitivity: SensitivityResult
    leakage_flags: list[LeakageFlag]

class GateResult(BaseModel):
    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    rationale: str        # plain English, goes straight into the report

class Verdict(BaseModel):
    result: Literal["REJECT", "PASS"]
    gates: list[GateResult]
    evidence_hash: str
    spec_version: str
    generated_from: StrategyRun
```

**`n_trials` is mandatory and non-defaulted.** A strategy that will not declare how many
variants were tried cannot be audited. If a caller passes `n_trials=1` after a 5,000-run
grid search, that is fraud, and the deflated-Sharpe gate is the only thing standing between
them and a lie. Document this loudly.

---

## 3. M1 — Cost model

The single most common reason retail backtests are fiction. Config-driven, never hardcoded.

`configs/costs_india_equity.yaml` — populate with the **current** charge stack and stamp
the source and date in the file. Verify against your broker's live charge list before
trusting any number; these rates change and a stale config silently inflates every result.

Charge components to model for Indian equities:

- Brokerage (delivery vs intraday; per-order cap)
- STT (asymmetric: buy+sell on delivery, sell-only on intraday and futures, premium-based on options)
- Exchange transaction charges (differs equity / futures / options)
- SEBI turnover fee
- Stamp duty (buy side only, differs by segment)
- GST on (brokerage + exchange + SEBI)
- DP charges — per scrip, per day, on delivery sells. Flat fee, so it **murders small-notional strategies**. Model it explicitly.

```yaml
# configs/costs_india_equity.yaml
_source: "<broker charge list URL>"
_verified_on: "YYYY-MM-DD"
_warning: "Stale rates silently inflate every backtest. Re-verify quarterly."

segments:
  equity_delivery:
    brokerage: {pct: 0.0, per_order_cap: 0.0}
    stt: {buy_pct: ..., sell_pct: ...}
    exchange_txn_pct: ...
    sebi_turnover_pct: ...
    stamp_duty_buy_pct: ...
    gst_pct: 18.0
    dp_charge_per_scrip_per_sell: ...
```

### Slippage (`costs/slippage.py`)

```
cost_bps = half_spread_bps + k * sigma_daily * sqrt(order_value / adv_20) * 1e4
```

Square-root impact law. Default `k = 0.5`, configurable. Half-spread from a per-liquidity-
tier table (large cap / mid / small). **Requirement:** slippage must scale with order size.
A model that charges a flat 5bps regardless of notional is how strategies fake capacity.

### Acceptance test M1

Round-trip a ₹10,000 delivery position and a ₹10,00,000 delivery position in the same
symbol. Cost as a fraction of notional must be **strictly higher** for the small one
(DP charge dominates) and impact-bps must be **strictly higher** for the large one.
If both aren't true, the model is wrong.

---

## 4. M2 — Benchmark harness *(this is the one that kills 40/40)*

```python
def benchmark_check(run: StrategyRun, bars, costs) -> BenchmarkEvidence
```

Rules, all of them non-negotiable:

1. **Benchmark is NIFTY 50 TRI**, not price index. Price index quietly hides ~1.2–1.5%/yr
   of dividends. Every strategy that "beats NIFTY by 1%" is beating a strawman.
2. Benchmark pays entry cost once, using the same `CostModel`. Not zero-cost.
3. **Same capital, same period, same currency.**
4. **Risk-match before comparing.** If the strategy sits 45% in cash, it is not comparable
   to a fully-invested index. Two adjustments, report both:
   - de-lever the benchmark to the strategy's realised vol, and
   - regress strategy excess returns on benchmark excess returns; report α, β, and the
     t-stat of α. **α with a t-stat below 2 is not alpha.**
5. Report `net_of_everything` CAGR side by side with a single-line verdict sentence.

### `PerfMetrics`

CAGR, annualised vol, Sharpe, Sortino, max drawdown, Calmar, longest underwater period,
hit rate, avg win/avg loss, turnover (annualised, 2-sided), time-in-market, tail ratio,
worst 5 days.

### Acceptance test M2

Run `benchmark_clone` (a strategy that just holds the index) through NULL.
Expected: α ≈ 0, β ≈ 1, and verdict REJECT with the rationale naming *"no alpha over
buy-and-hold after costs."* If it PASSes, the harness is broken.

---

## 5. M3 — Leakage audit (runs before any statistics)

If this fails, short-circuit to REJECT immediately. Do not compute Sharpe on a strategy
that can see the future — you'll be tempted to believe the number.

| Check | Rule |
|---|---|
| **Decision lag** | Signal computed on bar `t` close may not fill before bar `t+1` open. `decision_lag_bars >= 1` enforced at contract level. Reject `decision_lag_bars == 0`. |
| **Timestamp monotonicity** | No weight timestamp may precede the bar timestamp it depends on. Walk the dependency graph. |
| **Survivorship** | Universe must be point-in-time. Any symbol in `run.universe` that was not an index constituent / not listed on that date → flag. Delisted names must be present in history with a terminal value, not silently dropped. |
| **Corporate actions** | Split/bonus/dividend adjustment must be applied consistently and *backwards only*. Detect single-bar returns beyond ±25% that coincide with a known action date and are unadjusted. |
| **NaN forward-fill** | Any indicator using `.fillna(method='ffill')` across a data gap longer than N bars → flag. Common silent look-ahead. |
| **Universe rebalance timing** | Index reconstitution must be applied on the effective date, not the announcement date, and not retroactively. |

### Known data risk — flag it in the README

Free point-in-time NIFTY constituent history is genuinely hard to source. NSE bhavcopy
gives you prices, not membership history. `yfinance` adjusted closes have documented
adjustment errors on Indian tickers. **Budget real time for this.** Options: scrape and
archive NSE index-maintenance PDFs monthly going forward, or accept a documented
survivorship bias and print it on every report. Do not pretend the problem doesn't exist.

---

## 6. M4 — The statistical adversary

This is the part the internet skips, and it's the reason NULL exists.

### 6.1 Deflated Sharpe Ratio — the headline gate

Bailey & López de Prado. Adjusts the observed Sharpe for (a) how many variants were tried,
(b) the variance across those trials, (c) non-normality of returns (skew, excess kurtosis),
(d) sample length.

Inputs: `sharpe_observed`, `n_trials`, `var_of_trial_sharpes`, `skew`, `kurtosis`, `T`.
Output: probability the true Sharpe exceeds zero.

**Gate: `DSR > 0.95`.** A Sharpe of 1.8 from 2,000 grid-search trials on 4 years of data
deflates to nothing. Show the user the deflation explicitly: "observed 1.80 → deflated 0.31,
because you tried 2,000 variants."

### 6.2 PBO via CSCV

Combinatorially Symmetric Cross-Validation. Split the return matrix into `S` submatrices
(S=16 default), form all `C(S, S/2)` train/test partitions, and measure how often the
in-sample-best configuration lands below median out-of-sample.

**Gate: `PBO < 0.5`.** Above 0.5 the selection process is worse than a coin flip.

### 6.3 White's Reality Check / Hansen's SPA

Bootstrap the null that the best of `N` strategies does not beat the benchmark, using a
**stationary bootstrap** (Politis–Romano, geometric block length, mean block ≈ sqrt(T)).
IID resampling of financial returns destroys autocorrelation and produces p-values that
are too optimistic. Use blocks.

**Gate: `p < 0.05`.**

### 6.4 Minimum Track Record Length

How many observations would be needed for this Sharpe to be significant at 95%, given the
observed skew and kurtosis. Report in years. If MTRL > the backtest length, the result is
under-powered by construction — say that in plain English in the report.

### 6.5 Walk-forward with purge + embargo (`partition/walkforward.py`)

Standard k-fold leaks when labels overlap in time. Implement:

- **Purge:** drop training samples whose label window overlaps the test window.
- **Embargo:** drop a further `e` bars after the test window (default `e = 0.01 * T`).

Report per-fold metrics. **Gate: the strategy must be net-positive after costs in ≥ 60%
of folds.** A strategy carried by one fold is one lucky regime, not an edge.

### 6.6 Regime decomposition (`partition/regimes.py`)

Split on: India VIX terciles, 200-DMA trend state (above/below), and calendar eras
(pre-2016 / 2016–2020 / 2020–2022 / 2022+). Report metrics per bucket.
Not a hard gate — an evidence panel. But a strategy whose entire return comes from
March–May 2020 must be visible at a glance.

### 6.7 Parameter sensitivity (`sensitivity/neighborhood.py`)

Perturb each parameter ±1 and ±2 steps, recompute Sharpe, build the surface.

**Gate: the mean Sharpe of the immediate neighbourhood must be ≥ 60% of the peak.**
A spike is curve-fitting. A plateau is (weak) evidence of structure.

---

## 7. M5 — Verdict engine

`verdict/engine.py` is a pure function. Default `REJECT`. Every gate must pass.

```yaml
# configs/gates_default.yaml
gates:
  leakage_clean:        {required: true}                 # hard short-circuit
  beats_benchmark_net:  {metric: alpha_tstat, min: 2.0}
  deflated_sharpe:      {min: 0.95}
  pbo:                  {max: 0.5}
  reality_check:        {max_p: 0.05}
  walkforward_consistency: {min_fold_win_rate: 0.60}
  sensitivity_plateau:  {min_neighborhood_ratio: 0.60}
  capacity:             {max_adv_participation: 0.05}
  # drawdown_tolerance and pbo are NOT gates — see configs/gates_default.yaml
```

Every `GateResult.rationale` is written for a human, not a log file:

> ❌ **deflated_sharpe** — observed Sharpe 1.80, but you declared 2,000 trials. Adjusted
> for selection, skew (-0.9) and kurtosis (7.2) over 1,006 observations, the probability
> the true Sharpe is above zero is 0.31. Threshold is 0.95. This is a grid search finding
> noise.

That paragraph is the product. Write the rationale strings with care.

---

## 8. M6 — Golden test suite (`tests/golden/`)

**Build these before you trust a single real result.** Each is a synthetic strategy with a
known correct verdict. CI fails if any verdict flips.

| Fixture | Construction | Required verdict | Which gate must catch it |
|---|---|---|---|
| `oracle_lookahead` | Buys at `t` using close of `t+1` | REJECT | leakage_clean |
| `pure_noise` | Random entries, seeded | REJECT | deflated_sharpe / reality_check |
| `benchmark_clone` | Holds NIFTY | REJECT | beats_benchmark_net (α≈0) |
| `costed_scalper` | +0.4%/trade gross, 8 trades/day | REJECT | beats_benchmark_net after costs |
| `overfit_grid` | Best of 5,000 params fit on GBM noise | REJECT | deflated_sharpe, pbo |
| `one_regime_wonder` | All PnL from Mar–May 2020 | REJECT | walkforward_consistency |
| `capacity_bomb` | Real edge, 40% of ADV per order | REJECT | capacity |
| **`true_edge_synthetic`** | Injected persistent 0.6 Sharpe edge, low turnover, 1 trial | **PASS** | — must survive all gates |

That last row matters as much as the other seven. **A harness that rejects everything is
exactly as useless as one that accepts everything.** If `true_edge_synthetic` cannot pass,
your thresholds are miscalibrated and you will throw away real edges.

### What "all eight green" does NOT mean

The golden suite runs the real gates against real statistics, but four fields of the
`Evidence` it judges are **supplied by the test harness rather than computed by the
pipeline**: walk-forward fold returns, the sensitivity surface, `max_adv_participation`,
and the leakage flags. `tests/golden/harness.SYNTHESISED` names them, and the limitations
band on every report lists them.

So a green suite proves the *gates* behave correctly on known inputs. It does not prove
the pipeline produces those inputs correctly from a strategy and its bars.

**Settled at M7.** All four are now computed from real inputs on the real audit path:
`partition/walkforward.py`, `sensitivity/neighborhood.py` (surface supplied via
`--sensitivity`), ADV participation and `leakage/audit.py` are wired into
`null/cli.build_evidence`, and `examples/rsi2_nifty` exercises every one of them on real
NIFTY 50 bars. The four entries remain in `SYNTHESISED` because they describe the
**fixtures**, not the pipeline: a golden fixture is a synthetic return series with no
bars, no grid and no weights, so there is nothing for those stages to read. That list
does not shrink by wiring anything — only by a fixture gaining the input it lacks.

Read the two claims separately. "All eight green" is a statement about the judge.
"`examples/rsi2_nifty` synthesises nothing" is the statement about the whole machine,
and as of M7 it holds.

### Open decision — the benchmark series (must be settled before M6 is called green)

M2 built the benchmark harness with `benchmark_bars` as an **injected parameter**.
`benchmark_check` never fetches or defaults a series, and that design stays. The source
is still unchosen, and it is recorded here so it cannot quietly default to a price index
later. **The NIFTY 50 price index is not a candidate** — it excludes roughly 1.2–1.5%/yr
of dividends and hands every strategy that much free alpha, which is the exact bias M2
exists to remove.

Two candidates, neither picked:

- **(a) NSE Indices daily index files**, archived forward from now and backfilled as far
  as retention allows. Authoritative TRI. Cost: history is limited by whatever retention
  reaches back to, and the archive only deepens with time.
- **(b) NIFTYBEES-style ETF NAV as a TRI proxy.** Embeds dividends, but also embeds an
  expense ratio and tracking error, so its total return sits *systematically below* true
  index TRI — understating the benchmark, the same direction of error as the price index,
  just smaller. If chosen it **must be labelled a proxy in every limitations block**, with
  the direction of the bias named.

See `docs/data_sources.md` for what was tested against the official NSE endpoint and why
a plain HTTP client cannot reach it.

---

## 9. M7 — The demo that proves it works

### Prerequisite — multi-symbol `benchmark_check`

M2 shipped `benchmark_check` single-symbol; it raises `NotImplementedError` on a run whose
universe holds more than one symbol. The RSI(2) demo runs across NIFTY 50 constituents, so
**M7 is blocked until multi-symbol portfolio accounting lands**: per-symbol weight tracking,
per-symbol turnover and cost attribution, and portfolio aggregation into a single net return
series. Not a refactor of the existing path — the single-symbol case is a special case of it.

Reproduce the reel's claim on your own harness.

1. Implement RSI(2) mean-reversion on NIFTY 50 constituents, daily, long-only.
2. Grid search it: RSI period {2,3,4}, entry {5,10,15}, exit {50,60,70}, holding cap
   {3,5,10,15} → 108 variants. **Record `n_trials=108` honestly.**
   (Holding cap corrected from {3,5,10} — see *Spec correction* below.)
3. Run the best variant through NULL.
4. Expected output: REJECT, with the deflation number and the after-cost benchmark
   comparison stated in one sentence each.
5. Commit the `verdict.json` and the rendered HTML to `examples/rsi2_nifty/`.

### M7 — DONE

Verdict **REJECT** on `rsi2_nifty50`, all seven gates judged, none `NOT_COMPUTABLE`:

| gate | state |
|---|---|
| `leakage_clean` | PASS |
| `beats_benchmark_net` | FAIL — 3.16% CAGR net against 13.25% benchmark; alpha t-stat 1.58 (Newey-West, 8 lags) against a threshold of 2.0 |
| `deflated_sharpe` | FAIL — 0.00 against 0.95 |
| `reality_check` | FAIL — p = 0.977 against 0.050 |
| `walkforward_consistency` | PASS — net-positive in 4 of 5 out-of-sample folds |
| `sensitivity_plateau` | PASS — neighbourhood mean Sharpe 77% of peak (0.32 against 0.42), threshold 60% |
| `capacity` | FAIL — largest order 5.1% of 20-day ADV against a 5.0% tolerance |

Benchmark is the real NIFTY 50 **TRI**, 6,768 sessions from 1999-06-30, validated
against the price index before it was committed (24 of 24 three-year windows dominated,
annualised gap 1.5170% against the published 1.35%).

Artifacts in `examples/rsi2_nifty/audit_out/`. `verdict.json` splits its per-trial return
series into the sibling `verdict.trials.parquet`, the same split `run.json` uses, which
takes the JSON from 18.7MB to 1.4MB. Both files plus `report.html` are byte-identical
across runs; `evidence_hash 3f40aa2efda9a477…`.

**Note the shape of this REJECT.** The strategy passes on plateau and walk-forward and
still dies — it is not curve-fit to a spike and its result is not carried by one fold. It
fails because after real costs against a real total-return benchmark it does not clear
the market, and what outperformance there is (p = 0.977) is what chance produces on this
history. That is the intended kill: not a harness artifact, and not a parameter accident.

### Spec correction — the fourth holding-cap value *(settled; not an open question)*

This section originally specified holding cap **{3,5,10}** while also asserting the grid
has **108 variants**. Those two statements are inconsistent: 3 × 3 × 3 × 3 = **81**, not
108. 3 × 3 × 3 × 4 = 108 exactly.

**Decision: the holding-cap list gains a fourth value, 15**, as the natural continuation of
the stated sequence. The variant count is what is load-bearing here, not the specific cap
values. `n_trials` is the denominator of the deflated-Sharpe correction — the one gate
BUILD.md §6.1 calls the headline — and CLAUDE.md invariant 7 makes it required and
non-defaulted precisely because understating it is the cheapest available way to launder an
overfit result.

Declaring 108 while running 81 would at least err in the *safe* direction — a larger
`n_trials` deflates harder, so it cannot manufacture a PASS. That is exactly why it is
worth being explicit about rejecting it anyway: `n_trials` is not a conservatism dial, it
is a count of what was searched. Once it is allowed to be approximately true in the
harmless direction, there is no principled place to stop, and the gate's denominator stops
being a fact about the run.

The alternative — keep {3,5,10} and correct the count to 81 — was available and rejected.
It would leave the demo's headline count differing from the spec's, for no gain: nothing
about the strategy or the audit depends on which four caps are used, while a great deal
depends on `n_trials` being literally true.

Ratified by Sheshakanth. It was carried as a stated assumption in
`examples/rsi2_nifty/README.md` and `strategy.py` while it remained his call; those now
point here instead. Recorded rather than silently applied because a spec that quietly
edits itself to match its implementation is not a spec.

---

**This artifact is the portfolio piece.** Not the code — the verdict. A one-page report
that says "here is a strategy that looks like it makes 22% CAGR, here is why it doesn't,
here are the six independent tests that say so" is more impressive than any dashboard.

---

## 10. Integration seams (build the hooks, not the integrations)

Keep NULL standalone. Expose seams only.

- **← Consilium:** an adapter that turns a Consilium thesis (with its explicit
  machine-checkable claims and invalidation level) into a `StrategyRun`. Consilium's
  fact-checker verdicts ride along as metadata. NULL does not call Consilium.
- **← PLUMB:** NULL emits a structured run trace at every stage boundary. PLUMB records it.
  Change a prompt upstream in Consilium, replay, diff the `verdict.json`. Any change in a
  NULL verdict from an upstream prompt edit is, by definition, drift — because NULL itself
  is deterministic. **This is the single strongest claim the whole system makes and nobody
  copying that reel can make it.**
- **→ Execution:** NULL writes `verdict.json`. Something else reads it. NULL has no broker
  credentials, ever, in any environment.

---

## 11. Compliance boundary — do not drift across this line

Under SEBI's retail algo framework (fully mandatory 1 April 2026): algorithmic orders carry
an exchange-assigned Algo ID, brokers are principals and algo providers are agents who must
route through a registered broker, sessions require daily 2FA with no persistent refresh
tokens, orders come only from a whitelisted static IP, and third-party platforms must be
empanelled and hosted inside broker infrastructure. Under 10 orders/second you're treated
as an ordinary API user rather than an algo provider.

NULL is a research tool. It reads. It never places an order. Keeping execution
human-approved and read-only keeps the whole system on the right side of that line, and
the design is better for it anyway.

*Not legal or financial advice, and I'm not a licensed advisor. Confirm current
requirements with your broker before wiring anything to a live account.*

---

## 12. Build order — do not skip ahead

```
M0  Repo scaffold, contracts.py frozen, CI, determinism test        [1 day]
M1  Cost model + slippage + acceptance test                         [2 days]
M2  Benchmark harness + PerfMetrics + risk matching                 [2 days]
M3  Leakage audit + point-in-time universe plumbing                 [3 days]  ← data pain lives here
M4  Statistical adversary (DSR → PBO → bootstrap RC → walkforward)  [4 days]
M5  Verdict engine + gate configs + rationale strings               [1 day]
M6  Golden suite, all 8 fixtures green                              [2 days]
M7  RSI(2) kill demo + rendered report                              [1 day]  ← DONE
```

**Gate on M6.** Do not point NULL at a strategy you actually care about until all eight
golden fixtures produce their expected verdicts. Until then you have no idea whether a
REJECT means the strategy is bad or the harness is.

---

## 13. Definition of done for v0.1

- [ ] `null audit run.json --config configs/gates_default.yaml` → `verdict.json` + `report.html`
- [ ] Same input produces byte-identical output across two machines (test enforces it)
- [ ] Zero LLM calls anywhere in `null/` (test greps for it and fails the build)
- [ ] All 8 golden fixtures return their expected verdicts in CI
- [ ] `examples/rsi2_nifty/` committed with a REJECT verdict and readable rationale
- [ ] README opens with the sentence: *"NULL assumes your strategy is worthless and makes
      it prove otherwise. Most don't."*