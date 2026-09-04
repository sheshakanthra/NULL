NULL assumes your strategy is worthless and makes it prove otherwise. Most don't.

---

> ## ⚠️ Do not trust any number this repository currently produces
>
> NULL is under construction and is **not yet a working auditor**. Three specific
> things are wrong right now, and each of them makes output misleading rather
> than merely incomplete:
>
> **1. The cost rates have never been checked against a real broker.**
> `configs/costs_india_equity.yaml` carries `_verified_on: "UNVERIFIED"`. Those
> rates were written from general knowledge and have not been reconciled against
> any live charge list. `CostConfig.rates_are_verified` returns `False`. **Every
> cost number M1 produces is indicative only.** Rates wrong in the optimistic
> direction silently inflate every backtest — the exact failure NULL exists to
> catch.
>
> **2. There is no benchmark series yet.** NIFTY 50 TRI is unresolved. The
> official NSE Indices endpoint is real and free but refuses plain HTTP clients
> (three documented attempts, all returning the HTML page instead of JSON — see
> [docs/data_sources.md](docs/data_sources.md)). Until a source is chosen, NULL
> cannot compare anything to anything. Note that the obvious fallback, the NIFTY
> price index, is **not acceptable**: it excludes ~1.2–1.5%/yr of dividends and
> would hand every strategy that much free alpha.
>
> **3. The golden fixtures are not green.** None of the eight fixtures in
> BUILD.md §8 exist yet. Until they do, a REJECT from NULL might be telling you
> the strategy is bad, or might be telling you the harness is. You cannot tell
> which, and neither can I.
>
> An unaudited auditor is worth less than no auditor, because it invites belief.
> Treat this repository as scaffolding until the table below shows M6 green.

---

NULL is a deterministic strategy audit engine. It is not a backtester. You hand it
a strategy's realised target weights and it tries to prove the strategy has no
edge. It reports PASS only when every attempt to kill it fails.

The default verdict is REJECT.

## Status

| Milestone | What | State |
|---|---|---|
| M0 | Repo scaffold, `contracts.py` frozen, CI, determinism test | **done** |
| M1 | Cost model + slippage | **done** — rates unverified, see warning 1 |
| M2 | Benchmark harness + `PerfMetrics` + risk matching | **blocked** — no TRI source, see warning 2 |
| M3 | Leakage audit + point-in-time universe | not started |
| M4 | Statistical adversary (DSR, PBO, bootstrap RC, walk-forward) | not started |
| M5 | Verdict engine + gate configs | not started |
| M6 | Golden suite, 8 fixtures | not started — see warning 3 |
| M7 | RSI(2) kill demo | not started |

Contracts are at spec **0.2.0**. They froze at M0; the one revision since then
(`RegressionResult.se_method` / `hac_lags`, `PerfMetrics.basis`) was raised and
approved at review rather than slipped in — see CLAUDE.md, "Frozen contract
decisions".

## What already holds

These are enforced by tests, not by intent:

- **Deterministic.** Same input, byte-identical output. Seeded throughout, no
  wall-clock in the audit path, no dependence on dict or set iteration order.
  Floats are quantised to 12 significant digits at every contract boundary so a
  differing BLAS build cannot move a verdict hash.
  (`tests/unit/test_determinism.py`)
- **No LLM anywhere in `null/`.** Not for explanations, not for report prose, not
  behind a flag. Enforced by a source grep with negative controls that prove the
  grep actually fires. (`tests/unit/test_no_llm.py`)
- **Offline.** No network imports in `null/`. Data fetch is a separate, cached,
  offline-first stage. (`tests/unit/test_source_invariants.py`)
- **Read-only.** No broker credentials, no order placement, no write access to any
  account, in any file or environment. Also grep-enforced.
- **Default REJECT is structural.** A `Verdict` claiming PASS while carrying a
  failing gate — or carrying no gates at all — cannot be constructed. Likewise an
  `Evidence` holding a gross-basis strategy against a net-basis benchmark, which
  would report its own cost drag as alpha.

## Known data limitations

Beyond the TRI question above:

**Point-in-time index membership.** Free point-in-time NIFTY constituent history
is not readily available. NSE bhavcopy gives prices, not membership history, and
`yfinance` adjusted closes have documented adjustment errors on Indian tickers.
This gets decided at M3. Whichever option is chosen — archiving NSE
index-maintenance PDFs going forward, or accepting a documented survivorship bias
— the choice will be printed as a stated limitation on every report. A
survivorship-biased universe inflates returns. A NULL report that does not name
its universe source is incomplete.

## Compliance boundary

NULL is a research tool. It reads. It never places an order. Keeping execution
human-approved and read-only keeps the system clear of SEBI's retail algo
framework (fully mandatory 1 April 2026), which governs order placement, not
analysis.

*Not legal or financial advice. Confirm current requirements with your broker
before wiring anything to a live account.*

## Development

```bash
pip install -e ".[dev]"

pytest                              # full suite
mypy --strict null/                 # type check
pytest tests/unit/test_no_llm.py    # the invariant that matters most
```

`BUILD.md` is the master spec. `CLAUDE.md` holds the invariants that apply to
every session. Read the relevant section before touching a milestone.
