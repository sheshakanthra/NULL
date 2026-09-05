"""Multi-symbol portfolio accounting. The hard M7 blocker.

RSI(2) runs across NIFTY 50 constituents, so ``benchmark_check`` has to handle a
real portfolio rather than raising NotImplementedError.

Two acceptance criteria, both analytic rather than regression-style, because the
whole point is that the aggregation is *provably* right:

  1. A two-symbol portfolio whose combined series is known by hand must match.
  2. A 50-name equal-weight portfolio must be charged 50 DP fees on a rebalance
     day, not one. That is the failure most likely to go quietly wrong -- the DP
     charge is flat per scrip per day, so costing the portfolio aggregate instead
     of each symbol would undercount it fiftyfold and make a 50-name strategy look
     cheap to run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from null.benchmark.buyhold import benchmark_check
from null.contracts import Bar, StrategyRun, TargetWeight
from null.costs.india_equity import IndiaEquityCostModel

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "costs_india_equity.yaml"
IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2022, 1, 3, 15, 30, tzinfo=IST)


@pytest.fixture(scope="module")
def costs() -> IndiaEquityCostModel:
    return IndiaEquityCostModel.from_yaml(CONFIG)


def _bars(symbol: str, returns: np.ndarray, start_price: float = 1000.0, adv=5e9):
    """Bars for one symbol from a known return series."""
    levels = start_price * np.cumprod(1.0 + returns)
    out = []
    for i, close in enumerate(levels):
        prev = start_price if i == 0 else float(levels[i - 1])
        out.append(
            Bar(
                ts=START + timedelta(days=i),
                symbol=symbol,
                open=prev,
                high=max(prev, float(close)) * 1.001,
                low=min(prev, float(close)) * 0.999,
                close=float(close),
                volume=1e6,
                adv_20=adv,
            )
        )
    return out


# ---------------------------------------------------------------------------
# acceptance 1: analytic two-symbol aggregation
# ---------------------------------------------------------------------------


def test_two_symbol_portfolio_matches_the_analytic_combination(costs) -> None:
    """w_A * r_A + w_B * r_B, computed by hand, must be what the engine produces."""
    # rtol 1e-7, not 1e-9: the expected values round-trip through cumprod when the
    # bars are built and back out through price ratios, which costs a few ULPs. An
    # off-by-one in the weight alignment would show up around 1e-2, five orders of
    # magnitude above this, so the tolerance still separates the thing being tested.
    n = 40
    rng = np.random.default_rng(7)
    ra = rng.normal(0.0006, 0.010, n)
    rb = rng.normal(0.0002, 0.014, n)
    wa, wb = 0.6, 0.3  # deliberately sums to 0.9: the remaining 10% is cash

    bars = tuple(_bars("AAA", ra) + _bars("BBB", rb))
    bench = tuple(_bars("BENCH", rng.normal(0.0004, 0.009, n)))

    weights = []
    for bar in bars:
        if bar.symbol == "AAA":
            weights.append(TargetWeight(ts=bar.ts, symbol="AAA", weight=wa))
        elif bar.symbol == "BBB":
            weights.append(TargetWeight(ts=bar.ts, symbol="BBB", weight=wb))

    run = StrategyRun(
        strategy_id="two_symbol",
        param_hash="p",
        n_trials=1,
        universe=("AAA", "BBB"),
        weights=tuple(weights),
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
    )

    evidence = benchmark_check(
        run=run, bars=bars, benchmark_bars=bench, costs=costs
    )

    # Returns start at bar 1; weights decided on bar t apply from t+1, so the first
    # return bar is still flat and the combination starts one bar later.
    expected = wa * ra[1:] + wb * rb[1:]
    actual = np.asarray(evidence.strategy_gross_returns.values)
    assert actual.size == expected.size
    # Bar 0 of the return series has no position yet.
    assert actual[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-7, atol=1e-12)


def test_cash_earns_nothing_and_is_not_an_error(costs) -> None:
    """Weights summing to 0.5 is a real state, not a validation failure.

    A long-only signal strategy is in cash whenever it has no signal, and that is
    how it actually behaves. The uninvested half must contribute exactly zero.
    """
    n = 30
    r = np.full(n, 0.01)  # every day +1%
    bars = tuple(_bars("AAA", r))
    bench = tuple(_bars("BENCH", np.zeros(n)))
    run = StrategyRun(
        strategy_id="half_cash",
        param_hash="p",
        n_trials=1,
        universe=("AAA",),
        weights=tuple(
            TargetWeight(ts=b.ts, symbol="AAA", weight=0.5) for b in bars
        ),
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
    )
    evidence = benchmark_check(run=run, bars=bars, benchmark_bars=bench, costs=costs)
    gross = np.asarray(evidence.strategy_gross_returns.values)
    # 50% invested in a +1%/day asset = +0.5%/day. The rest is cash at zero.
    np.testing.assert_allclose(gross[1:], 0.005, rtol=1e-7)


def test_a_symbol_absent_on_a_day_contributes_zero_and_drops_no_day(costs) -> None:
    """Universe entry and exit mid-period. Zero, never NaN, never a shortened series."""
    n = 40
    rng = np.random.default_rng(11)
    ra = rng.normal(0.0005, 0.01, n)
    rb = rng.normal(0.0005, 0.01, n)

    full = _bars("AAA", ra)
    partial = _bars("BBB", rb)[20:]  # BBB only exists for the second half
    bars = tuple(full + partial)
    bench = tuple(_bars("BENCH", rng.normal(0.0004, 0.009, n)))

    weights = [TargetWeight(ts=b.ts, symbol="AAA", weight=0.5) for b in full]
    weights += [TargetWeight(ts=b.ts, symbol="BBB", weight=0.5) for b in partial]

    run = StrategyRun(
        strategy_id="entering_universe",
        param_hash="p",
        n_trials=1,
        universe=("AAA", "BBB"),
        weights=tuple(weights),
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
    )
    evidence = benchmark_check(run=run, bars=bars, benchmark_bars=bench, costs=costs)
    gross = np.asarray(evidence.strategy_gross_returns.values)

    assert gross.size == n - 1, "a day was dropped when a symbol was absent"
    assert np.all(np.isfinite(gross)), "an absent symbol produced NaN instead of zero"
    # Before BBB exists the portfolio is 50% AAA and 50% cash.
    np.testing.assert_allclose(gross[1:19], 0.5 * ra[2:20], rtol=1e-7, atol=1e-12)


# ---------------------------------------------------------------------------
# acceptance 2: DP charge is per scrip, not per portfolio
# ---------------------------------------------------------------------------


def test_fifty_name_rebalance_charges_fifty_dp_fees_not_one(costs) -> None:
    """The failure most likely to go quietly wrong.

    DP is flat per scrip per day on delivery sells. Costing the portfolio aggregate
    instead of each symbol would charge one fee where fifty are due, understating
    the cost of a 50-name strategy by a factor of fifty.
    """
    n_symbols, n_days = 50, 12
    rng = np.random.default_rng(3)
    symbols = tuple(f"S{i:02d}" for i in range(n_symbols))

    bars: list[Bar] = []
    for sym in symbols:
        bars.extend(_bars(sym, rng.normal(0.0004, 0.01, n_days)))
    bench = tuple(_bars("BENCH", rng.normal(0.0004, 0.009, n_days)))

    # Hold all fifty at 2% each, then sell everything on one day.
    weights = []
    for bar in bars:
        held = 0.02 if bar.ts < START + timedelta(days=8) else 0.0
        weights.append(TargetWeight(ts=bar.ts, symbol=bar.symbol, weight=held))

    run = StrategyRun(
        strategy_id="fifty_names",
        param_hash="p",
        n_trials=1,
        universe=symbols,
        weights=tuple(weights),
        decision_lag_bars=1,
        initial_capital=50_000_000.0,
    )
    evidence = benchmark_check(
        run=run, bars=tuple(bars), benchmark_bars=bench, costs=costs
    )

    per_scrip = costs.config.segments[
        list(costs.config.segments)[0]
    ].dp_charge_per_scrip_per_sell
    dp_total = evidence.cost_breakdown["dp_charge"]

    assert dp_total == pytest.approx(n_symbols * per_scrip, rel=1e-9), (
        f"charged {dp_total:.2f} in DP fees; fifty scrips selling should cost "
        f"{n_symbols * per_scrip:.2f}. Charging {per_scrip:.2f} would mean the "
        "portfolio was costed in aggregate rather than per symbol."
    )
    assert dp_total > per_scrip * 40


# ---------------------------------------------------------------------------
# capacity: max across symbols and days, never the mean
# ---------------------------------------------------------------------------


def test_adv_participation_is_the_worst_symbol_not_the_average(costs) -> None:
    """One illiquid name at 40% of ADV invalidates the whole run."""
    n = 20
    rng = np.random.default_rng(5)
    liquid = _bars("LIQUID", rng.normal(0.0004, 0.01, n), adv=1e12)
    illiquid = _bars("TINY", rng.normal(0.0004, 0.01, n), adv=2e6)
    bars = tuple(liquid + illiquid)
    bench = tuple(_bars("BENCH", rng.normal(0.0004, 0.009, n)))

    run = StrategyRun(
        strategy_id="one_bad_name",
        param_hash="p",
        n_trials=1,
        universe=("LIQUID", "TINY"),
        weights=tuple(
            TargetWeight(ts=b.ts, symbol=b.symbol, weight=0.5) for b in bars
        ),
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
    )
    evidence = benchmark_check(run=run, bars=bars, benchmark_bars=bench, costs=costs)

    # 50% of Rs 1 crore into a name with Rs 20 lakh ADV is a huge participation.
    assert evidence.max_adv_participation > 1.0
    # The mean across both symbols would be roughly half that and would hide it.
    assert evidence.max_adv_participation > 2 * 0.5 * 1e7 / 1e12


# ---------------------------------------------------------------------------
# the regression path is unchanged from M2
# ---------------------------------------------------------------------------


def test_regression_still_runs_on_the_portfolio_series(costs) -> None:
    n = 300
    rng = np.random.default_rng(13)
    bars = tuple(
        _bars("AAA", rng.normal(0.0005, 0.01, n))
        + _bars("BBB", rng.normal(0.0005, 0.01, n))
    )
    bench = tuple(_bars("BENCH", rng.normal(0.0004, 0.009, n)))
    run = StrategyRun(
        strategy_id="regression",
        param_hash="p",
        n_trials=1,
        universe=("AAA", "BBB"),
        weights=tuple(
            TargetWeight(ts=b.ts, symbol=b.symbol, weight=0.5) for b in bars
        ),
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
    )
    evidence = benchmark_check(run=run, bars=bars, benchmark_bars=bench, costs=costs)
    assert evidence.alpha.se_method == "newey_west"
    assert evidence.alpha.n_obs == n - 1
    assert evidence.metrics.basis == "net"


def test_single_symbol_still_works_unchanged(costs) -> None:
    """The single-symbol case is a special case of the general one, not a branch."""
    n = 50
    rng = np.random.default_rng(17)
    r = rng.normal(0.0005, 0.01, n)
    bars = tuple(_bars("AAA", r))
    bench = tuple(_bars("BENCH", rng.normal(0.0004, 0.009, n)))
    run = StrategyRun(
        strategy_id="single",
        param_hash="p",
        n_trials=1,
        universe=("AAA",),
        weights=tuple(TargetWeight(ts=b.ts, symbol="AAA", weight=1.0) for b in bars),
        decision_lag_bars=1,
        initial_capital=10_000_000.0,
    )
    evidence = benchmark_check(run=run, bars=bars, benchmark_bars=bench, costs=costs)
    gross = np.asarray(evidence.strategy_gross_returns.values)
    np.testing.assert_allclose(gross[1:], r[2:], rtol=1e-7, atol=1e-12)
