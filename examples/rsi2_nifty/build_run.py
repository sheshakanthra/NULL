"""M7 prep: run the RSI(2) grid on real NIFTY 50 bars and report the raw
distribution BEFORE any audit runs.

    python examples/rsi2_nifty/build_run.py

Offline: reads the committed adjusted OHLCV cache
(data/reference/nifty50_ohlcv.parquet). No network. This does NOT run the
benchmark comparison, DSR, PBO, reality check, or the verdict engine -- there is
still no committed TRI series for the benchmark step, and that is a separate task.
What this produces is everything a `null audit` run needs except the benchmark:
a fully honest StrategyRun (n_trials=108, trials populated from the real grid)
and the sensitivity surface, both derived from the grid rather than stubbed.

The point of printing the raw Sharpe distribution here, before anything is
written, is stated in the session record: the gap between the best variant's own
number and what NULL eventually says about it is the entire story BUILD.md's M7
section is built to tell. Seeing the "looks great" number first is the only way to
recognise that gap later.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from examples.rsi2_nifty.strategy import ALL_VARIANTS, build_sensitivity, run_grid
from null.contracts import StrategyRun, TargetWeight, TrialRecord
from null.costs.india_equity import IndiaEquityCostModel
from null.data.ohlcv import DEFAULT_CACHE as OHLCV_CACHE
from null.data.ohlcv import load_bars

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COSTS_CONFIG = REPO / "configs" / "costs_india_equity.yaml"
CONSTITUENTS = REPO / "configs" / "nifty50_constituents.txt"

RUN_OUT = HERE / "run.json"
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
    # StrategyRun: n_trials=108, honestly, trials populated from the real grid.
    # ------------------------------------------------------------------
    trials = tuple(
        TrialRecord(
            param_hash=r.variant.param_hash,
            sharpe=r.net_sharpe,
            returns=r.net_returns,
        )
        for r in results
    )
    run = StrategyRun(
        strategy_id="rsi2_nifty50",
        param_hash=best.variant.param_hash,
        n_trials=len(ALL_VARIANTS),
        universe=universe,
        weights=best.weights,
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
        trials=trials,
    )
    RUN_OUT.write_bytes(run.canonical_json())
    print(f"Wrote {RUN_OUT}  (n_trials={run.n_trials}, {len(run.weights)} weight "
          f"change points, {len(run.trials)} trial records)")

    print()
    print("NOT run: benchmark comparison, DSR, PBO, reality check, verdict engine.")
    print("No committed TRI series exists yet; `null audit` needs one to proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
