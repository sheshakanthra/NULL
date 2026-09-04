"""Regression estimator behaviour -- BUILD.md section 4 rule 4.

The degenerate-fit test below exists because of a real bug found at M2. An exact
clone of the benchmark leaves residuals at the floating-point floor, so alpha and
its standard error both collapse and the t-stat becomes tiny/tiny. It came out at
-4.59 on the first run, which happened to REJECT and made the acceptance test pass.
The sign was luck. A +4.59 would have PASSED benchmark_clone, which BUILD.md says
means the harness is broken.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from null.benchmark.risk_match import (
    newey_west_lags,
    regress_excess_returns,
    risk_match_benchmark,
)
from null.contracts import Series

IST = timezone(timedelta(hours=5, minutes=30))
SEED = 20260905


def _series(values: np.ndarray) -> Series:
    start = datetime(2020, 1, 1, 15, 30, tzinfo=IST)
    return Series(
        ts=tuple(start + timedelta(days=i) for i in range(values.size)),
        values=tuple(float(v) for v in values),
    )


# ---------------------------------------------------------------------------
# the degenerate case
# ---------------------------------------------------------------------------


def test_exact_clone_reports_no_evidence_of_alpha() -> None:
    """tiny/tiny is not a t-stat. No evidence must mean no alpha, not random alpha."""
    rng = np.random.default_rng(SEED)
    b = rng.normal(0.0004, 0.01, 1000)
    result = regress_excess_returns(_series(b), _series(b))
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.alpha_tstat == 0.0, (
        f"an exact clone produced t-stat {result.alpha_tstat}; its sign is "
        "floating-point noise and a positive value would PASS the benchmark gate"
    )
    assert result.beta == pytest.approx(1.0, abs=1e-9)


def test_exact_clone_never_passes_the_alpha_threshold_across_many_seeds() -> None:
    """The original bug was sign-dependent, so sweep seeds rather than trust one."""
    for seed in range(40):
        rng = np.random.default_rng(seed)
        b = rng.normal(0.0004, 0.01, 500)
        result = regress_excess_returns(_series(b), _series(b))
        assert result.alpha_tstat < 2.0, f"clone passed the gate on seed {seed}"


# ---------------------------------------------------------------------------
# the non-degenerate case: the regression must still work
# ---------------------------------------------------------------------------


def test_recovers_a_known_injected_alpha() -> None:
    """A guard that zeroes everything would also pass the degenerate test."""
    rng = np.random.default_rng(SEED)
    b = rng.normal(0.0004, 0.01, 2000)
    injected_daily = 0.0006
    s = 1.0 * b + injected_daily + rng.normal(0.0, 0.002, 2000)
    result = regress_excess_returns(_series(s), _series(b), periods_per_year=252)
    assert result.alpha_annual == pytest.approx(injected_daily * 252, rel=0.15)
    assert result.beta == pytest.approx(1.0, abs=0.05)
    assert result.alpha_tstat > 2.0, "a genuine injected alpha must clear the gate"


def test_recovers_a_known_beta() -> None:
    rng = np.random.default_rng(SEED)
    b = rng.normal(0.0004, 0.01, 2000)
    s = 0.5 * b + rng.normal(0.0, 0.001, 2000)
    result = regress_excess_returns(_series(s), _series(b))
    assert result.beta == pytest.approx(0.5, abs=0.02)


def test_no_alpha_when_none_was_injected() -> None:
    rng = np.random.default_rng(SEED)
    b = rng.normal(0.0004, 0.01, 2000)
    s = 1.0 * b + rng.normal(0.0, 0.002, 2000)
    result = regress_excess_returns(_series(s), _series(b))
    assert abs(result.alpha_tstat) < 2.0


# ---------------------------------------------------------------------------
# standard errors
# ---------------------------------------------------------------------------


def test_newey_west_is_the_default_and_is_recorded() -> None:
    rng = np.random.default_rng(SEED)
    b = rng.normal(0.0004, 0.01, 500)
    s = b + rng.normal(0.0, 0.002, 500)
    r = regress_excess_returns(_series(s), _series(b))
    assert r.se_method == "newey_west"
    assert r.hac_lags == newey_west_lags(500)


def test_ols_reports_a_larger_tstat_than_newey_west_on_autocorrelated_residuals() -> None:
    """This is the whole reason se_method is on the contract.

    With positively autocorrelated residuals OLS understates the standard error,
    inflating the t-stat toward the gate.
    """
    rng = np.random.default_rng(SEED)
    n = 2000
    b = rng.normal(0.0004, 0.01, n)
    noise = np.zeros(n)
    eps = rng.normal(0.0, 0.002, n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + eps[i]  # AR(1), strongly autocorrelated
    s = b + 0.0004 + noise

    ols = regress_excess_returns(_series(s), _series(b), se_method="ols")
    hac = regress_excess_returns(_series(s), _series(b), se_method="newey_west")

    assert ols.hac_lags is None
    assert hac.hac_lags is not None
    assert abs(ols.alpha_tstat) > abs(hac.alpha_tstat), (
        f"OLS t-stat {ols.alpha_tstat:.3f} was not larger than Newey-West "
        f"{hac.alpha_tstat:.3f}; if OLS is not the optimistic estimator here, the "
        "HAC correction is not doing anything"
    )


def test_newey_west_lag_rule_matches_the_published_formula() -> None:
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1000) == 6
    assert newey_west_lags(1) == 0


# ---------------------------------------------------------------------------
# risk matching
# ---------------------------------------------------------------------------


def test_risk_matching_scales_benchmark_to_strategy_vol() -> None:
    """A strategy half in cash must not be compared to a fully-invested index."""
    rng = np.random.default_rng(SEED)
    b = rng.normal(0.0004, 0.01, 1000)
    s = 0.5 * b
    matched, scale = risk_match_benchmark(_series(s), _series(b))
    assert scale == pytest.approx(0.5, abs=0.02)
    assert float(np.std(matched.to_numpy(), ddof=1)) == pytest.approx(
        float(np.std(s, ddof=1)), rel=1e-9
    )


def test_regression_refuses_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        regress_excess_returns(_series(np.array([0.01, 0.02])), _series(np.array([0.01, 0.02])))
