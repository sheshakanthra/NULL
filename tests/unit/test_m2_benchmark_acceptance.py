"""M2 acceptance test -- BUILD.md section 4.

    Run `benchmark_clone` (a strategy that just holds the index) through NULL.
    Expected: alpha ~= 0, beta ~= 1, and verdict REJECT with the rationale naming
    "no alpha over buy-and-hold after costs." If it PASSes, the harness is broken.

This is the test that kills 40 out of 40. A strategy that holds the index has, by
construction, no edge over the index. If NULL cannot say so, no other verdict it
produces means anything.

Data note: this test uses a seeded synthetic index series, deliberately. The
acceptance criterion is a property of the harness, not of any particular market
history, and a test that needs the network is a test that cannot run in CI. Real
NIFTY 50 TRI sourcing is a separate question and is NOT settled -- see
docs/data_sources.md. Nothing here should be read as evidence that the TRI
problem is solved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from null.contracts import Bar, StrategyRun, TargetWeight
from null.costs.india_equity import IndiaEquityCostModel

# M2's comparison logic is deliberately unwritten. Two decisions block it and both
# belong to Sheshakanth, not to this file:
#
#   1. Which NIFTY 50 TRI source NULL uses. The official NSE endpoint is real but
#      refuses plain HTTP clients; see docs/data_sources.md for what was tested
#      and the five options. Falling back to the price index is NOT one of them --
#      it hands every strategy ~1.3%/yr of free alpha, the exact bias M2 removes.
#   2. Whether RegressionResult gains se_method and PerfMetrics gains basis. Both
#      are frozen-contract changes (invariant 5) and both change what this test
#      can assert -- see the last two tests in this module.
#
# The test is written and was watched to fail with ModuleNotFoundError on
# null.benchmark.buyhold, per the acceptance-test-first rule. It is skipped rather
# than left red so CI stays meaningful. Remove this skip when M2 lands.
pytestmark = pytest.mark.skip(
    reason="M2 unimplemented: blocked on TRI source and two contract decisions "
    "(see docs/data_sources.md)"
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "costs_india_equity.yaml"
IST = timezone(timedelta(hours=5, minutes=30))

SEED = 20260905
N_DAYS = 1_000
START_LEVEL = 20_000.0
SYMBOL = "NIFTY50_TRI"


def _index_bars() -> tuple[Bar, ...]:
    """A seeded synthetic total-return index. Fixed seed, no wall-clock."""
    rng = np.random.default_rng(SEED)
    daily = rng.normal(loc=0.0004, scale=0.010, size=N_DAYS)
    level = START_LEVEL * np.cumprod(1.0 + daily)
    start = datetime(2020, 1, 1, 15, 30, tzinfo=IST)
    bars: list[Bar] = []
    for i, close in enumerate(level):
        prev = START_LEVEL if i == 0 else float(level[i - 1])
        bars.append(
            Bar(
                ts=start + timedelta(days=i),
                symbol=SYMBOL,
                open=prev,
                high=max(prev, float(close)) * 1.001,
                low=min(prev, float(close)) * 0.999,
                close=float(close),
                volume=1e7,
                adv_20=8e9,
            )
        )
    return tuple(bars)


def _benchmark_clone(bars: tuple[Bar, ...]) -> StrategyRun:
    """Holds the index at 100% weight, every bar, forever. One trial, honestly."""
    return StrategyRun(
        strategy_id="benchmark_clone",
        param_hash="none",
        n_trials=1,
        universe=(SYMBOL,),
        weights=tuple(
            TargetWeight(ts=b.ts, symbol=SYMBOL, weight=1.0) for b in bars
        ),
        decision_lag_bars=1,
        initial_capital=1_000_000.0,
    )


@pytest.fixture(scope="module")
def evidence():
    bars = _index_bars()
    return benchmark_check(
        run=_benchmark_clone(bars),
        bars=bars,
        benchmark_bars=bars,
        costs=IndiaEquityCostModel.from_yaml(CONFIG_PATH),
    )


# ---------------------------------------------------------------------------
# the acceptance criteria
# ---------------------------------------------------------------------------


def test_alpha_is_approximately_zero(evidence) -> None:
    """Holding the index cannot generate alpha over the index."""
    assert abs(evidence.alpha.alpha_annual) < 0.01, (
        f"benchmark_clone showed {evidence.alpha.alpha_annual:.4%} annual alpha "
        "against the index it holds. That is a harness bug, not an edge."
    )


def test_beta_is_approximately_one(evidence) -> None:
    assert evidence.alpha.beta == pytest.approx(1.0, abs=0.05), (
        f"beta was {evidence.alpha.beta:.4f}; a clone of the index must track it."
    )


def test_alpha_tstat_is_below_the_gate_threshold(evidence) -> None:
    """Section 4 rule 4: alpha with a t-stat below 2 is not alpha."""
    assert evidence.alpha.alpha_tstat < 2.0


def test_verdict_is_reject(evidence) -> None:
    assert evidence.verdict_result == "REJECT", (
        "benchmark_clone PASSed. The harness is broken -- every other verdict it "
        "produces is now worthless."
    )


def test_rationale_names_no_alpha_over_buy_and_hold_after_costs(evidence) -> None:
    """BUILD.md names this phrase specifically. The rationale is the product."""
    gate = evidence.gate
    assert not gate.passed
    assert "no alpha over buy-and-hold after costs" in gate.rationale.lower(), (
        f"rationale did not name the required phrase:\n  {gate.rationale}"
    )


# ---------------------------------------------------------------------------
# non-negotiable rules from section 4 that the acceptance rests on
# ---------------------------------------------------------------------------


def test_benchmark_pays_entry_cost_through_the_same_cost_model(evidence) -> None:
    """Rule 2: the benchmark is not zero-cost."""
    assert evidence.benchmark_entry_cost > 0.0, (
        "a zero-cost benchmark is a strawman; it hands the strategy free alpha "
        "equal to the benchmark's own entry cost"
    )


def test_report_surfaces_that_cost_rates_are_unverified(evidence) -> None:
    """M1 carried this forward: rates_are_verified is still False."""
    assert evidence.rates_are_verified is False
    assert "unverified" in evidence.limitations_text.lower(), (
        "a report built on unverified charge rates must say so on its face"
    )


def test_alpha_tstat_standard_error_method_is_recorded(evidence) -> None:
    """OLS errors on autocorrelated daily returns inflate the t-stat materially.

    The gate is alpha_tstat >= 2.0, so the SE method changes verdicts. Whichever
    is used, the artifact must say which.
    """
    assert evidence.alpha_se_method in {"ols", "newey_west"}


def test_strategy_and_benchmark_metrics_share_a_basis(evidence) -> None:
    """A gross-basis strategy compared to a net-basis benchmark manufactures
    alpha exactly equal to the cost drag."""
    assert evidence.metrics_basis == evidence.benchmark_metrics_basis == "net"
