"""M7 prep: run the RSI(2) grid on real NIFTY 50 bars and report the raw
distribution BEFORE any audit runs.

    python examples/rsi2_nifty/build_run.py

Offline: reads the committed adjusted OHLCV cache
(data/reference/nifty50_ohlcv.parquet). No network. This does NOT run the
benchmark comparison, alpha regression, reality check, or the verdict engine --
there is still no committed TRI series for the benchmark step, and that is a
separate task. It DOES compute the deflated Sharpe / expected-max-Sharpe
diagnostic, because that needs no benchmark at all -- only the grid's own trial
Sharpes and the best variant's own return series.

Two artifacts, not one:

  run.json               the audit input: strategy_id, n_trials=108, universe,
                          weights, and one TrialRecord per variant carrying its
                          sharpe but NOT its full return series.
  run.trials.parquet     the 108 variants' real net-return series, wide format
                          (one "date" column, one column per param_hash),
                          sibling to run.json by naming convention (same stem,
                          ".trials.parquet" suffix).

This split exists because embedding all 108 return series inline made run.json
18.7 MB -- unwieldy for a repository people clone to see a demo, and GitHub warns
above 50MB per file. The contract already treats per-trial returns as optional
("trials may be empty or a subset ... never a substitute for n_trials"), so a
lean run.json is contract-legitimate on its own. But shipping it alone would
silently degrade PBO to NOT_COMPUTABLE the day someone actually runs `null audit`
on it -- a real loss of evidence, not just a formatting change. null/cli.py's
``--trials-parquet`` flag closes that gap: given this sibling file, it rehydrates
full per-trial returns before the gates run, so the split costs nothing but disk
layout.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from examples.rsi2_nifty.strategy import ALL_VARIANTS, build_sensitivity, run_grid
from null.contracts import StrategyRun, TrialRecord
from null.costs.india_equity import IndiaEquityCostModel
from null.data.ohlcv import DEFAULT_CACHE as OHLCV_CACHE
from null.data.ohlcv import load_bars
from null.stats.deflated_sharpe import deflated_sharpe_ratio

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COSTS_CONFIG = REPO / "configs" / "costs_india_equity.yaml"
CONSTITUENTS = REPO / "configs" / "nifty50_constituents.txt"

RUN_OUT = HERE / "run.json"
TRIALS_PARQUET_OUT = HERE / "run.trials.parquet"  # <run stem>.trials.parquet
GRID_REPORT_OUT = HERE / "grid_report.csv"
SENSITIVITY_OUT = HERE / "sensitivity.json"


def _load_universe() -> tuple[str, ...]:
    lines = CONSTITUENTS.read_text(encoding="utf-8").splitlines()
    return tuple(sorted(l.strip() for l in lines if l.strip() and not l.strip().startswith("#")))


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": statistics.quantiles(ordered, n=4)[0],
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p75": statistics.quantiles(ordered, n=4)[2],
        "max": ordered[-1],
    }


def per_period_trial_sharpes(results: tuple) -> np.ndarray:
    """Per-period (not annualised) Sharpe for each variant's net returns.

    ``deflated_sharpe_ratio`` annualises internally; feeding it already-annualised
    trial Sharpes double-annualises the variance across trials. See the comment at
    the call site in ``main()`` for how that was caught.
    """
    return np.asarray(
        [
            float(np.mean(vals) / np.std(vals, ddof=1)) if np.std(vals, ddof=1) > 0.0 else 0.0
            for vals in (r.net_returns.to_numpy() for r in results)
        ],
        dtype=np.float64,
    )


def write_run_artifacts(
    results: tuple,
    *,
    universe: tuple[str, ...],
    initial_capital: float,
    run_out: Path,
    trials_parquet_out: Path,
    n_trials: int | None = None,
) -> tuple[Path, Path]:
    """Write the split artifacts: a lean run.json plus a sibling trial-returns
    parquet. Shared by the real 108-variant run and by tests using a small slice
    of the grid, so both exercise exactly the same writing logic.

    Every variant must share one date index -- checked, not assumed, since
    run_grid backtests every variant over the same bars and a divergence here
    would mean something upstream broke that invariant.
    """
    best = max(results, key=lambda r: r.net_sharpe)

    reference_ts = results[0].net_returns.ts
    for r in results:
        assert r.net_returns.ts == reference_ts, (
            f"variant {r.variant.param_hash} has a different date index than the "
            "rest of the grid; the wide-format trial parquet assumes a shared "
            "timeline and this assumption just broke"
        )
    trials_frame = pd.DataFrame(
        {"date": [pd.Timestamp(ts) for ts in reference_ts]}
        | {r.variant.param_hash: r.net_returns.values for r in results}
    )
    trials_parquet_out.parent.mkdir(parents=True, exist_ok=True)
    trials_frame.to_parquet(trials_parquet_out, index=False)

    trials = tuple(
        TrialRecord(param_hash=r.variant.param_hash, sharpe=r.net_sharpe, returns=None)
        for r in results
    )
    run = StrategyRun(
        strategy_id="rsi2_nifty50",
        param_hash=best.variant.param_hash,
        n_trials=n_trials if n_trials is not None else len(results),
        universe=universe,
        weights=best.weights,
        decision_lag_bars=1,
        initial_capital=initial_capital,
        trials=trials,
    )
    run_out.parent.mkdir(parents=True, exist_ok=True)
    run_out.write_bytes(run.canonical_json())
    return run_out, trials_parquet_out


def main() -> int:
    if not OHLCV_CACHE.exists():
        print(f"No OHLCV cache at {OHLCV_CACHE}. Nothing to run.")
        return 2

    universe = _load_universe()
    print(f"Loading bars for {len(universe)} NIFTY 50 constituents from {OHLCV_CACHE.name}")
    bars = load_bars(OHLCV_CACHE, symbols=universe)
    print(f"  {len(bars):,} bars loaded")

    costs = IndiaEquityCostModel.from_yaml(COSTS_CONFIG)

    print(f"Running the {len(ALL_VARIANTS)}-variant grid...")
    started = time.monotonic()
    results = run_grid(
        bars=bars,
        universe=universe,
        costs=costs,
        initial_capital=10_000_000.0,
    )
    elapsed = time.monotonic() - started
    print(f"  done in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # THE RAW DISTRIBUTION, BEFORE ANYTHING ELSE. Sheshakanth asked to see this
    # before NULL judges the strategy at all.
    # ------------------------------------------------------------------
    gross = [r.gross_sharpe for r in results]
    net = [r.net_sharpe for r in results]
    best = max(results, key=lambda r: r.net_sharpe)
    best_gross = max(results, key=lambda r: r.gross_sharpe)

    print()
    print("=" * 72)
    print("RAW GRID DISTRIBUTION -- before any NULL gate has run")
    print("=" * 72)
    print(f"{'':10}{'gross':>10}{'net':>10}")
    dg, dn = _distribution(gross), _distribution(net)
    for key in ("min", "p25", "median", "mean", "p75", "max"):
        print(f"{key:10}{dg[key]:10.3f}{dn[key]:10.3f}")
    print()
    print(f"Best by NET Sharpe:   {best.variant} -> gross {best.gross_sharpe:.3f}, "
          f"net {best.net_sharpe:.3f}, {best.n_position_changes} position changes")
    print(f"Best by GROSS Sharpe: {best_gross.variant} -> gross "
          f"{best_gross.gross_sharpe:.3f}, net {best_gross.net_sharpe:.3f}")
    print(f"Cost erosion at the best point: {best.gross_sharpe - best.net_sharpe:.3f} "
          f"Sharpe ({best.n_position_changes} position changes across the universe)")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # THE ONE SENTENCE THE DEMO TURNS ON. No benchmark needed: DSR only requires
    # the grid's own trial Sharpes and the best variant's own realised return
    # series (its skew and kurtosis are computed FROM that series, not assumed).
    #
    # deflated_sharpe_ratio expects trial_sharpes in PER-PERIOD units -- it does
    # its own annualisation internally, the same way it annualises `returns`'
    # own Sharpe. `r.net_sharpe` is already annualised (run_variant multiplies by
    # sqrt(252)), so passing that list directly would double-annualise: the
    # variance across trials would be inflated by ~252x and expected_max_sharpe
    # by ~sqrt(252) on top of that. Caught by testing this in isolation before
    # trusting the first number it produced (12.2 vs the correct 0.77 on a
    # synthetic check) -- recomputed here as plain per-period mean/std.
    # ------------------------------------------------------------------
    dsr = deflated_sharpe_ratio(
        returns=best.net_returns.to_numpy(),
        n_trials=len(ALL_VARIANTS),
        trial_sharpes=per_period_trial_sharpes(results),
    )
    print("=" * 72)
    print("DEFLATED SHARPE -- needs no benchmark, computed now")
    print("=" * 72)
    print(f"Best variant's own net Sharpe:        {dsr.observed_sharpe_annual:.3f}")
    print(f"Expected max Sharpe from {len(ALL_VARIANTS)} trials,")
    print(f"  {dsr.n_obs:,} observations, by chance alone: {dsr.expected_max_sharpe_annual:.3f}")
    print(f"Realised skew:     {dsr.skew:+.3f}")
    print(f"Realised kurtosis: {dsr.kurtosis:.3f}")
    print(f"Deflated Sharpe (P[true Sharpe > 0]): {dsr.deflated_sharpe:.3f}")
    print()
    if dsr.expected_max_sharpe_annual > dsr.observed_sharpe_annual:
        print(
            f"EXPECTED-MAX EXCEEDS THE BEST VARIANT'S OWN SHARPE: "
            f"{dsr.expected_max_sharpe_annual:.3f} > {dsr.observed_sharpe_annual:.3f}. "
            "Noise alone, searched across 108 variants on this many observations, is "
            "EXPECTED to produce a better Sharpe than what this grid actually found. "
            "The best-of-108 selected number is statistically indistinguishable from "
            "what a random search would hand you by chance."
        )
    else:
        print(
            f"Best variant's Sharpe ({dsr.observed_sharpe_annual:.3f}) exceeds what "
            f"noise alone would be expected to produce ({dsr.expected_max_sharpe_annual:.3f})."
        )
    print()
    print(dsr.selection_diagnostic)
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Grid report CSV, every variant, auditable.
    # ------------------------------------------------------------------
    with GRID_REPORT_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["param_hash", "period", "entry", "exit", "holding_cap",
             "gross_sharpe", "net_sharpe", "n_position_changes"]
        )
        for r in results:
            writer.writerow(
                [r.variant.param_hash, r.variant.period, r.variant.entry,
                 r.variant.exit, r.variant.holding_cap,
                 f"{r.gross_sharpe:.6f}", f"{r.net_sharpe:.6f}", r.n_position_changes]
            )
    print(f"Wrote {GRID_REPORT_OUT}")

    # ------------------------------------------------------------------
    # Sensitivity surface, from the grid.
    # ------------------------------------------------------------------
    sensitivity = build_sensitivity(results, metric="net_sharpe")
    SENSITIVITY_OUT.write_bytes(sensitivity.canonical_json())
    print(f"Wrote {SENSITIVITY_OUT}  "
          f"(neighborhood_ratio={sensitivity.neighborhood_ratio:.3f})")

    # ------------------------------------------------------------------
    # The split artifacts: lean run.json + sibling trial-returns parquet.
    # ------------------------------------------------------------------
    write_run_artifacts(
        results,
        universe=universe,
        initial_capital=10_000_000.0,
        run_out=RUN_OUT,
        trials_parquet_out=TRIALS_PARQUET_OUT,
        n_trials=len(ALL_VARIANTS),
    )
    print(f"Wrote {TRIALS_PARQUET_OUT} "
          f"({TRIALS_PARQUET_OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(results[0].net_returns)} rows x {len(results)} trial columns)")
    print(f"Wrote {RUN_OUT} ({RUN_OUT.stat().st_size / 1024:.0f} KB, "
          f"n_trials={len(ALL_VARIANTS)}, {len(best.weights)} weight change points, "
          f"{len(results)} trial records, returns NOT embedded)")

    print()
    print("To audit with full PBO evidence once TRI is available:")
    print(f"  null audit {RUN_OUT.relative_to(REPO)} "
          f"--trials-parquet {TRIALS_PARQUET_OUT.relative_to(REPO)} ...")
    print()
    print("NOT run: benchmark comparison, reality check, verdict engine.")
    print("No committed TRI series exists yet; `null audit` needs one to proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
