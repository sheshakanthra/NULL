"""RSI(2) strategy and grid, M7 prep.

Everything here runs on tiny synthetic bars, not the real 15-year cache -- these
are unit tests for the strategy logic, not a replay of the real demo. The real
grid is run and reported separately (see build_run.py), because it needs the
committed OHLCV cache and takes materially longer than a test suite should.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from examples.rsi2_nifty.strategy import (
    ALL_VARIANTS,
    ENTRY_THRESHOLDS,
    EXIT_THRESHOLDS,
    HOLDING_CAPS,
    RSI_PERIODS,
    GridVariant,
    build_sensitivity,
    compute_rsi,
    generate_weights_for_symbol,
    run_grid,
)
from null.contracts import Bar, StrategyRun, TargetWeight
from null.costs.india_equity import IndiaEquityCostModel

IST = timezone(timedelta(hours=5, minutes=30))
COSTS_CONFIG = Path("configs/costs_india_equity.yaml")


def _bars(symbol: str, closes: list[float], start: datetime | None = None) -> list[Bar]:
    base = start or datetime(2020, 1, 1, 15, 30, tzinfo=IST)
    out = []
    for i, close in enumerate(closes):
        out.append(
            Bar(
                ts=base + timedelta(days=i),
                symbol=symbol,
                open=close,
                high=close * 1.001,
                low=close * 0.999,
                close=close,
                volume=1_000_000.0,
                adv_20=1e9,
            )
        )
    return out


# ---------------------------------------------------------------------------
# RSI correctness
# ---------------------------------------------------------------------------


def test_rsi_is_100_for_a_strictly_increasing_series() -> None:
    closes = np.asarray([100.0 + i for i in range(20)])
    rsi = compute_rsi(closes, period=2)
    valid = rsi[~np.isnan(rsi)]
    assert valid.size > 0
    assert np.all(valid == pytest.approx(100.0))


def test_rsi_is_0_for_a_strictly_decreasing_series() -> None:
    closes = np.asarray([100.0 - i for i in range(20)])
    rsi = compute_rsi(closes, period=2)
    valid = rsi[~np.isnan(rsi)]
    assert valid.size > 0
    assert np.all(valid == pytest.approx(0.0))


def test_rsi_is_neutral_for_a_flat_series() -> None:
    closes = np.full(20, 100.0)
    rsi = compute_rsi(closes, period=2)
    valid = rsi[~np.isnan(rsi)]
    assert np.all(valid == pytest.approx(50.0))


def test_rsi_is_undefined_before_the_window_fills() -> None:
    closes = np.asarray([100.0, 101.0, 102.0])
    rsi = compute_rsi(closes, period=5)
    assert np.all(np.isnan(rsi))


def test_rsi_is_bounded_zero_to_hundred_on_a_realistic_walk() -> None:
    rng = np.random.default_rng(11)
    closes = 1000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, 500))
    for period in RSI_PERIODS:
        rsi = compute_rsi(closes, period)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0.0) and np.all(valid <= 100.0)


# ---------------------------------------------------------------------------
# entry / exit / holding-cap logic, hand-verified
# ---------------------------------------------------------------------------


def test_enters_on_rsi_below_entry_and_exits_on_rsi_above_exit() -> None:
    bars = _bars("AAA", [100.0] * 10)
    # Hand-crafted RSI: dips under 5 at bar 3, climbs over 70 at bar 7.
    rsi = np.array([np.nan, np.nan, 50.0, 3.0, 10.0, 40.0, 65.0, 75.0, 80.0, 60.0])
    weights = generate_weights_for_symbol(
        bars, rsi, entry=5, exit=70, holding_cap=100, weight_when_long=0.5
    )
    assert len(weights) == 2
    assert weights[0].symbol == "AAA" and weights[0].weight == pytest.approx(0.5)
    assert weights[0].ts == bars[3].ts
    assert weights[1].weight == pytest.approx(0.0)
    assert weights[1].ts == bars[7].ts


def test_holding_cap_forces_an_exit_even_if_rsi_never_recovers() -> None:
    bars = _bars("AAA", [100.0] * 10)
    # RSI stays low (never crosses the exit threshold) through the holding window,
    # then rises above entry so the forced exit is not immediately re-entered --
    # isolating the holding-cap behaviour from re-entry behaviour, which is a
    # separate, also-correct thing the strategy does (see the no-pyramiding test).
    rsi = np.array([np.nan, np.nan, 50.0, 3.0, 4.0, 4.0, 4.0, 40.0, 40.0, 40.0])
    weights = generate_weights_for_symbol(
        bars, rsi, entry=5, exit=70, holding_cap=3, weight_when_long=1.0
    )
    assert len(weights) == 2
    entry, exit_ = weights
    assert entry.ts == bars[3].ts
    # bars_held reaches holding_cap=3 three bars after entry.
    assert exit_.ts == bars[6].ts


def test_no_pyramiding_while_already_long() -> None:
    bars = _bars("AAA", [100.0] * 8)
    rsi = np.array([np.nan, np.nan, 3.0, 2.0, 1.0, 40.0, 75.0, 60.0])
    weights = generate_weights_for_symbol(
        bars, rsi, entry=5, exit=70, holding_cap=100, weight_when_long=1.0
    )
    # Only ONE entry despite RSI staying under 5 for three consecutive bars.
    entries = [w for w in weights if w.weight > 0.0]
    assert len(entries) == 1
    assert entries[0].ts == bars[2].ts


def test_no_signal_produces_no_weight_changes() -> None:
    bars = _bars("AAA", [100.0] * 6)
    rsi = np.array([np.nan, np.nan, 50.0, 50.0, 50.0, 50.0])
    weights = generate_weights_for_symbol(
        bars, rsi, entry=5, exit=70, holding_cap=100, weight_when_long=1.0
    )
    assert weights == []


# ---------------------------------------------------------------------------
# the grid itself
# ---------------------------------------------------------------------------


def test_the_grid_has_exactly_108_variants() -> None:
    assert len(ALL_VARIANTS) == 108
    assert len(RSI_PERIODS) * len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS) * len(
        HOLDING_CAPS
    ) == 108


def test_every_variant_has_a_unique_param_hash() -> None:
    hashes = {v.param_hash for v in ALL_VARIANTS}
    assert len(hashes) == 108


def test_param_hash_is_deterministic() -> None:
    a = GridVariant(period=2, entry=5, exit=50, holding_cap=3)
    b = GridVariant(period=2, entry=5, exit=50, holding_cap=3)
    assert a.param_hash == b.param_hash


def test_offsets_from_index_into_each_parameters_own_list() -> None:
    v = GridVariant(period=4, entry=15, exit=70, holding_cap=15)
    assert GridVariant(period=4, entry=15, exit=70, holding_cap=15).offsets_from == {
        "period": 2,
        "entry": 2,
        "exit": 2,
        "holding_cap": 3,
    }


# ---------------------------------------------------------------------------
# run_grid end to end, tiny synthetic universe
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def costs() -> IndiaEquityCostModel:
    return IndiaEquityCostModel.from_yaml(COSTS_CONFIG)


@pytest.fixture(scope="module")
def tiny_universe_bars() -> tuple[Bar, ...]:
    rng = np.random.default_rng(42)
    n = 300
    bars: list[Bar] = []
    for symbol, seed in (("AAA", 1), ("BBB", 2), ("CCC", 3)):
        r = np.random.default_rng(seed).normal(0.0003, 0.014, n)
        closes = 500.0 * np.cumprod(1.0 + r)
        bars.extend(_bars(symbol, list(closes)))
    return tuple(bars)


def test_run_grid_produces_one_result_per_variant(tiny_universe_bars, costs) -> None:
    variants = ALL_VARIANTS[:6]  # a slice: the real grid is exercised in build_run
    results = run_grid(
        bars=tiny_universe_bars,
        universe=("AAA", "BBB", "CCC"),
        costs=costs,
        initial_capital=1_000_000.0,
        variants=variants,
    )
    assert len(results) == len(variants)
    for r in results:
        assert np.isfinite(r.gross_sharpe)
        assert np.isfinite(r.net_sharpe)
        assert len(r.gross_returns) == len(r.net_returns)


def test_run_grid_is_deterministic(tiny_universe_bars, costs) -> None:
    variants = ALL_VARIANTS[:4]
    kwargs = dict(
        bars=tiny_universe_bars,
        universe=("AAA", "BBB", "CCC"),
        costs=costs,
        initial_capital=1_000_000.0,
        variants=variants,
    )
    a = run_grid(**kwargs)
    b = run_grid(**kwargs)
    for ra, rb in zip(a, b):
        assert ra.gross_returns.values == rb.gross_returns.values
        assert ra.net_returns.values == rb.net_returns.values


def test_net_sharpe_never_exceeds_gross_sharpe_by_much_and_costs_are_nonnegative(
    tiny_universe_bars, costs
) -> None:
    """Costs should drag, not inflate, performance."""
    results = run_grid(
        bars=tiny_universe_bars,
        universe=("AAA", "BBB", "CCC"),
        costs=costs,
        initial_capital=1_000_000.0,
        variants=ALL_VARIANTS[:10],
    )
    for r in results:
        gross_mean = float(np.mean(r.gross_returns.values))
        net_mean = float(np.mean(r.net_returns.values))
        assert net_mean <= gross_mean + 1e-12


def test_no_network_or_wall_clock_in_the_strategy_module() -> None:
    import ast
    from pathlib import Path

    source = Path("examples/rsi2_nifty/strategy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_modules = {"socket", "urllib", "requests", "httpx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_modules
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_modules
    assert "datetime.now(" not in source
    assert "time.time(" not in source


# ---------------------------------------------------------------------------
# sensitivity surface from the grid
# ---------------------------------------------------------------------------


def test_build_sensitivity_identifies_the_correct_peak(tiny_universe_bars, costs) -> None:
    results = run_grid(
        bars=tiny_universe_bars,
        universe=("AAA", "BBB", "CCC"),
        costs=costs,
        initial_capital=1_000_000.0,
        variants=ALL_VARIANTS[:20],
    )
    sensitivity = build_sensitivity(results, metric="net_sharpe")
    best = max(results, key=lambda r: r.net_sharpe)
    assert sensitivity.peak_sharpe == pytest.approx(best.net_sharpe)
    peak_points = [p for p in sensitivity.points if all(v == 0 for v in p.offsets.values())]
    assert len(peak_points) == 1
    assert peak_points[0].sharpe == pytest.approx(best.net_sharpe)


def test_build_sensitivity_neighbourhood_ratio_is_bounded() -> None:
    """A synthetic surface with a known exact answer, not the real grid.

    Peak sharpe 2.0 at the origin; every Hamming-distance-1 neighbour is 1.0;
    everything else is irrelevant. Ratio must be exactly 0.5.
    """

    class _Stub:
        def __init__(self, variant, net_sharpe):
            self.variant = variant
            self.net_sharpe = net_sharpe

    results = []
    for v in ALL_VARIANTS:
        idx = v.offsets_from
        best_idx = {"period": 0, "entry": 0, "exit": 0, "holding_cap": 0}
        offsets = {k: idx[k] - best_idx[k] for k in idx}
        nonzero = [k for k, o in offsets.items() if o != 0]
        if not nonzero:
            sharpe = 2.0
        elif len(nonzero) == 1 and abs(offsets[nonzero[0]]) == 1:
            sharpe = 1.0
        else:
            sharpe = -5.0  # must not be counted
        results.append(_Stub(v, sharpe))

    sensitivity = build_sensitivity(tuple(results), metric="net_sharpe")
    assert sensitivity.peak_sharpe == pytest.approx(2.0)
    assert sensitivity.neighborhood_mean_sharpe == pytest.approx(1.0)
    assert sensitivity.neighborhood_ratio == pytest.approx(0.5)


def test_sensitivity_ratio_is_zero_not_undefined_for_a_nonpositive_peak() -> None:
    class _Stub:
        def __init__(self, variant, net_sharpe):
            self.variant = variant
            self.net_sharpe = net_sharpe

    results = tuple(_Stub(v, -1.0) for v in ALL_VARIANTS)
    sensitivity = build_sensitivity(results, metric="net_sharpe")
    assert sensitivity.neighborhood_ratio == 0.0


# ---------------------------------------------------------------------------
# the resulting StrategyRun validates, n_trials honest
# ---------------------------------------------------------------------------


def test_best_variant_builds_a_valid_strategy_run_with_honest_n_trials(
    tiny_universe_bars, costs
) -> None:
    variants = ALL_VARIANTS[:12]
    results = run_grid(
        bars=tiny_universe_bars,
        universe=("AAA", "BBB", "CCC"),
        costs=costs,
        initial_capital=1_000_000.0,
        variants=variants,
    )
    best = max(results, key=lambda r: r.net_sharpe)

    from null.contracts import TrialRecord

    run = StrategyRun(
        strategy_id="rsi2_nifty50_test",
        param_hash=best.variant.param_hash,
        n_trials=len(variants),
        universe=("AAA", "BBB", "CCC"),
        weights=best.weights,
        decision_lag_bars=1,
        initial_capital=1_000_000.0,
        trials=tuple(
            TrialRecord(param_hash=r.variant.param_hash, sharpe=r.net_sharpe, returns=r.net_returns)
            for r in results
        ),
    )
    assert run.n_trials == len(variants)
    assert len(run.trials) == len(variants)
    assert run.param_hash == best.variant.param_hash
