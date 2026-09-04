# NULL — standing instructions

`BUILD.md` is the master spec. Read the relevant section before touching a milestone.
This file holds the invariants that apply to **every** session. If something here conflicts
with what seems convenient right now, this file wins.

## What NULL is

A deterministic strategy audit engine. It assumes the strategy is worthless and tries to
prove it. Default verdict is REJECT. It is a prosecutor, not a backtester.

## Hard invariants — never violate these

1. **No LLM calls anywhere in `null/`.** Not for explanations, not for report prose, not
   "just this once" behind a flag. There is a CI test that greps for it. Do not add an
   exception path.
2. **No network calls in the audit path.** Data fetch is a separate, cached, offline-first
   stage. `null audit` must run with the network off.
3. **No broker credentials, no order placement, no write access to any account.** In any
   file, any environment, any test fixture. NULL reads.
4. **Determinism.** Seed everything. No `datetime.now()` in the audit path. No reliance on
   dict/set iteration order. Same input → byte-identical `verdict.json`.
5. **`contracts.py` is frozen after M0.** If a milestone seems to need a contract change,
   stop and ask. Do not widen a model to make a downstream module easier.
6. **Default REJECT.** A new gate defaults to failing. A gate that errors fails. Never add
   a fallback that lets a strategy through on missing evidence.
7. **`n_trials` is required and non-defaulted.** Never give it a default value, never infer
   it, never let it be optional "for now."

## Frozen contract decisions (settled at M0 — do not re-litigate)

These shape `contracts.py` and were decided by Sheshakanth before it was written.
BUILD.md §2 does not answer any of them, so they are recorded here instead.

1. **`Series` is a Pydantic model, not a `pandas.Series`.** Parallel frozen tuples of
   tz-aware timestamps and floats, strictly increasing in time. A pandas object is
   mutable and its byte representation depends on pandas internals, so neither can back
   `evidence_hash`. Compute converts at the boundary via `to_numpy()` / `to_pandas()` /
   `from_pandas()`. Do not push a DataFrame through a contract.
2. **Floats are quantised to 12 significant digits on validation, and non-finite floats
   are rejected.** Different BLAS builds disagree in the last bits of a float, and one
   such bit would flip the verdict hash. NaN and infinity are rejected for the same
   reason as invariant 6: missing evidence must fail loudly, never serialise into an
   artifact as a silent hole.

## Working rules

- **One milestone per session.** Do not start M2 because M1 finished early. Stop and report.
- **Acceptance test before implementation.** Each milestone in BUILD.md names its acceptance
  test. Write that test first, watch it fail, then implement.
- **Do not point NULL at real market data before M6 is green.** Until all eight golden
  fixtures return their expected verdicts, a REJECT tells you nothing about the strategy.
- **If a golden fixture's verdict flips, that is a P0.** Stop feature work. A harness whose
  verdicts move is worse than no harness.
- **Rates and thresholds live in `configs/`.** Never hardcode a charge, a fee, or a gate
  threshold in Python. If you find one hardcoded, move it.
- **Say when results are bad.** If a test fails, an approach isn't working, or a number
  looks wrong, say so plainly in the session summary. Do not paper over it or report
  partial success as success.

## Code standards

- Python 3.11+, Pydantic v2 (frozen models), **`pandas`** — chosen at M0, stay. `polars` is
  not a dependency and must not become one. Rationale: everything in `stats/` is
  numpy/scipy-shaped (statsmodels for the α regression, scipy for DSR and the stationary
  bootstrap), the datasets are small, and polars' advantages are relational rather than
  statistical.
- Type hints everywhere. `mypy --strict` on `null/`.
- Every gate is a pure function `fn(Evidence) -> GateResult`. No I/O, no state, no globals.
- `GateResult.rationale` is written for a human reader and names the observed number, the
  threshold, and *why* it failed. These strings are the product. Write them with care.
- Tests use fixed seeds. No `random` without a seed. No flaky test gets merged.

## Commits

One commit per logical unit, message names the milestone: `M1: cost model — DP charge on
delivery sells`. Do not batch a milestone into one commit.

## Session summary format

End every session with:

```
Milestone:      M<n>
Done:           <what now works>
Acceptance:     <test name> — PASS / FAIL
Not done:       <what was deferred and why>
Blockers:       <anything needing a decision from Sheshakanth>
Next:           <the single next step>
```

## Known hard problem — do not hand-wave

Point-in-time NIFTY constituent history is not freely available, and `yfinance` adjusted
closes have documented errors on Indian tickers. When M3 hits this, **stop and surface the
options** rather than silently picking one. Whatever gets chosen must be printed on every
report as a stated limitation.