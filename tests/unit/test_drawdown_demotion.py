"""Why drawdown_tolerance is not a gate. BUILD.md §7 lists it; we do not.

The argument is short: a 35% drawdown limit rejects NIFTY itself. A harness that
rejects its own benchmark on risk grounds is incoherent, because every strategy it
judges is being compared against a series the harness would refuse.

Drawdown answers "would you hold this", which is a preference about risk appetite.
The other gates answer "is there an edge", which is a claim about evidence. Only
the second belongs in a verdict. A deep drawdown is a real fact about a strategy
and it is still reported on every report -- it simply does not vote.

Capacity, by contrast, STAYS a gate and the distinction is worth stating: a
strategy that cannot execute at size never had the returns it claims. That is
evidence the backtest is wrong, not a preference about whether to hold it.

This module builds the NIFTY-like series and demonstrates the failure, so the
reasoning survives in the suite rather than only in a commit message.
"""

from __future__ import annotations

import numpy as np
import pytest

from null.metrics import compute_metrics
from null.verdict.engine import load_gate_config
from null.verdict.gates import drawdown_tolerance
from tests.golden.harness import _series, build_report
from tests.golden.fixtures import benchmark_clone

BUILD_MD_DRAWDOWN_THRESHOLD = 0.35


def nifty_like_buy_and_hold(seed: int = 2008) -> np.ndarray:
    """A long-run index with a 2008-style collapse.

    NIFTY fell roughly 60% peak-to-trough through 2008. Any realistic long-run
    Indian equity series contains that event, and so does every benchmark NULL
    would ever be pointed at.
    """
    rng = np.random.default_rng(seed)
    n = 15 * 252
    daily = rng.normal(0.00048, 0.0102, n)
    # The crash: roughly a year of steady decline totalling about -60%.
    crash = slice(int(n * 0.20), int(n * 0.28))
    daily[crash] -= 0.0025  # calibrated to ~60%, the real 2008 NIFTY figure
    return daily


def test_a_nifty_like_index_really_does_draw_down_about_sixty_percent() -> None:
    """Establish the premise before relying on it."""
    metrics = compute_metrics(
        _series(nifty_like_buy_and_hold()),
        basis="net",
        turnover_annual=0.0,
        time_in_market=1.0,
    )
    assert 0.50 < metrics.max_drawdown < 0.75, metrics.max_drawdown


def test_the_benchmark_itself_would_fail_the_drawdown_gate() -> None:
    """The incoherence, demonstrated.

    Run buy-and-hold NIFTY through the drawdown gate at BUILD.md's threshold. It
    fails. A harness cannot reject its own benchmark on risk grounds and still
    claim to be judging strategies against it.
    """
    metrics = compute_metrics(
        _series(nifty_like_buy_and_hold()),
        basis="net",
        turnover_annual=0.0,
        time_in_market=1.0,
    )

    class _StubEvidence:
        pass

    stub = _StubEvidence()
    stub.metrics = metrics  # type: ignore[attr-defined]

    result = drawdown_tolerance(
        stub,  # type: ignore[arg-type]
        {"max_dd": BUILD_MD_DRAWDOWN_THRESHOLD},
    )
    assert result.state == "FAIL", (
        "buy-and-hold NIFTY passed a 35% drawdown limit; the premise of this "
        "demotion no longer holds and it should be revisited"
    )
    assert result.observed > BUILD_MD_DRAWDOWN_THRESHOLD


def test_drawdown_tolerance_does_not_vote() -> None:
    """The demotion itself."""
    assert "drawdown_tolerance" not in load_gate_config()


def test_drawdown_is_still_reported() -> None:
    """Demoted, not deleted. A deep drawdown is a real fact about a strategy."""
    report = build_report(benchmark_clone())
    assert "drawdown" in report.panels
    assert "does not vote" in report.panels["drawdown"].lower()


def test_capacity_remains_a_gate() -> None:
    """The distinction the demotion turns on.

    Capacity is evidence that the backtest is wrong, not a preference about risk.
    """
    assert "capacity" in load_gate_config()
