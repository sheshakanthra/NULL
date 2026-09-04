NULL assumes your strategy is worthless and makes it prove otherwise. Most don't.

---

NULL is a deterministic strategy audit engine. It is not a backtester. You hand it a
strategy's realised target weights and it tries to prove the strategy has no edge. It
reports PASS only when every attempt to kill it fails.

The default verdict is REJECT.

## Status

**M0 complete.** Contracts are frozen. Nothing downstream of them is built yet.

| Milestone | What | State |
|---|---|---|
| M0 | Repo scaffold, `contracts.py` frozen, CI, determinism test | done |
| M1 | Cost model + slippage | not started |
| M2 | Benchmark harness + `PerfMetrics` + risk matching | not started |
| M3 | Leakage audit + point-in-time universe | not started |
| M4 | Statistical adversary (DSR, PBO, bootstrap RC, walk-forward) | not started |
| M5 | Verdict engine + gate configs | not started |
| M6 | Golden suite, 8 fixtures | not started |
| M7 | RSI(2) kill demo | not started |

Do not point NULL at real market data before M6 is green. Until all eight golden
fixtures return their expected verdicts, a REJECT tells you nothing about the strategy —
it might just be telling you about the harness.

## Guarantees

- **Deterministic.** Same input, byte-identical `verdict.json`. Seeded everywhere, no
  wall-clock in the audit path, no dependence on dict or set iteration order. Enforced
  by `tests/unit/test_determinism.py`.
- **No LLM anywhere in `null/`.** Not for explanations, not for report prose, not behind
  a flag. If a language model could influence a verdict, the verdict would be worthless.
  Enforced by `tests/unit/test_no_llm.py` in CI.
- **Offline.** `null audit` runs with the network off. Data fetch is a separate, cached,
  offline-first stage.
- **Read-only.** No broker credentials, no order placement, no write access to any
  account, in any file or environment. NULL reads.

## Known data limitation — read this before believing a verdict

Free point-in-time NIFTY constituent history is not readily available. NSE bhavcopy gives
prices, not membership history, and `yfinance` adjusted closes have documented adjustment
errors on Indian tickers.

This is unresolved as of M0. It gets decided at M3, and whichever option is chosen —
archiving NSE index-maintenance PDFs going forward, or accepting a documented
survivorship bias — **the choice will be printed as a stated limitation on every report.**
A survivorship-biased universe inflates returns. If you are reading a NULL report that
does not name its universe source, the report is incomplete.

## Compliance boundary

NULL is a research tool. It reads. It never places an order. Keeping execution
human-approved and read-only keeps the system clear of SEBI's retail algo framework
(fully mandatory 1 April 2026), which governs order placement, not analysis.

*Not legal or financial advice. Confirm current requirements with your broker before
wiring anything to a live account.*

## Development

```bash
pip install -e ".[dev]"

pytest                    # full suite
mypy --strict null/       # type check
pytest tests/unit/test_no_llm.py    # the invariant that matters most
```

`BUILD.md` is the master spec. `CLAUDE.md` holds the invariants that apply to every
session. Read the relevant section before touching a milestone.
