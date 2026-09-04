"""Deflated Sharpe behaviour and its negative controls. BUILD.md section 6.1."""

from __future__ import annotations

import numpy as np
import pytest

from null.stats.deflated_sharpe import (
    DEFAULT_ASSUMED_TRIAL_SHARPE_VARIANCE,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

SEED = 20260905


def _returns(n=2520, sharpe_annual=0.6, vol=0.011, seed=SEED):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    return raw * vol + sharpe_annual * vol / np.sqrt(252)


# ---------------------------------------------------------------------------
# negative controls -- prove the deflation is load-bearing
# ---------------------------------------------------------------------------


def test_without_the_trial_adjustment_an_overfit_sharpe_would_pass() -> None:
    """The control for the whole gate.

    Same returns, same everything, only n_trials differs. If declaring 5,000
    trials does not move the verdict, the deflation is decorative.
    """
    r = _returns(n=2520, sharpe_annual=0.85)
    trial_sharpes = np.random.default_rng(SEED).normal(0.0, 0.03, 5000)

    honest = deflated_sharpe_ratio(returns=r, n_trials=1)
    searched = deflated_sharpe_ratio(
        returns=r, n_trials=5000, trial_sharpes=trial_sharpes
    )

    assert honest.deflated_sharpe > searched.deflated_sharpe, (
        "declaring 5,000 trials did not reduce the deflated Sharpe at all"
    )
    assert honest.deflated_sharpe > 0.95 > searched.deflated_sharpe, (
        f"the same return series should pass at 1 trial ({honest.deflated_sharpe:.4f}) "
        f"and fail at 5,000 ({searched.deflated_sharpe:.4f}); it did not, so the gate "
        "is not actually responding to selection"
    )


def test_more_trials_never_increases_the_deflated_sharpe() -> None:
    """Monotonicity. More searching cannot make a result more credible."""
    r = _returns(n=1008, sharpe_annual=1.2)
    ts = np.random.default_rng(SEED).normal(0.0, 0.04, 10000)
    values = [
        deflated_sharpe_ratio(returns=r, n_trials=n, trial_sharpes=ts[:n]).deflated_sharpe
        for n in (2, 10, 100, 1000, 10000)
    ]
    assert values == sorted(values, reverse=True), values


def test_a_wider_spread_across_trials_deflates_harder() -> None:
    """More dispersion among trials means a luckier maximum, so a higher bar."""
    r = _returns(n=1008, sharpe_annual=1.2)
    rng = np.random.default_rng(SEED)
    narrow = deflated_sharpe_ratio(
        returns=r, n_trials=1000, trial_sharpes=rng.normal(0, 0.01, 1000)
    )
    wide = deflated_sharpe_ratio(
        returns=r, n_trials=1000, trial_sharpes=rng.normal(0, 0.05, 1000)
    )
    assert wide.deflated_sharpe < narrow.deflated_sharpe


# ---------------------------------------------------------------------------
# the moment adjustments
# ---------------------------------------------------------------------------


def test_negative_skew_reduces_confidence() -> None:
    """A great Sharpe with a long left tail deserves less belief, not more."""
    rng = np.random.default_rng(SEED)
    n = 2520
    symmetric = rng.normal(0.0004, 0.011, n)
    skewed = -np.abs(rng.normal(0, 0.011, n)) ** 1.5 * 3 + 0.0004 + rng.normal(0, 0.005, n)
    a = deflated_sharpe_ratio(returns=symmetric, n_trials=1)
    b = deflated_sharpe_ratio(returns=skewed, n_trials=1)
    assert b.skew < a.skew


def test_longer_samples_increase_confidence_for_the_same_sharpe() -> None:
    short = deflated_sharpe_ratio(returns=_returns(n=504), n_trials=1)
    long = deflated_sharpe_ratio(returns=_returns(n=2520), n_trials=1)
    assert long.deflated_sharpe > short.deflated_sharpe


# ---------------------------------------------------------------------------
# single trial
# ---------------------------------------------------------------------------


def test_one_trial_means_no_selection_adjustment() -> None:
    assert expected_max_sharpe(n_trials=1, var_trial_sharpes=0.5) == 0.0


def test_expected_max_sharpe_grows_with_trial_count() -> None:
    values = [expected_max_sharpe(n_trials=n, var_trial_sharpes=0.04) for n in (2, 10, 1000)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# the assumed-variance path must announce itself
# ---------------------------------------------------------------------------


def test_missing_trial_sharpes_are_assumed_and_declared() -> None:
    r = _returns(n=1008, sharpe_annual=1.5)
    result = deflated_sharpe_ratio(returns=r, n_trials=2000, trial_sharpes=None)
    assert result.variance_was_assumed is True
    assert result.var_trial_sharpes == DEFAULT_ASSUMED_TRIAL_SHARPE_VARIANCE
    assert "not supplied" in result.rationale.lower()
    assert "assumed" in result.rationale.lower()


def test_supplied_trial_sharpes_are_not_flagged_as_assumed() -> None:
    r = _returns(n=1008)
    ts = np.random.default_rng(SEED).normal(0, 0.03, 500)
    result = deflated_sharpe_ratio(returns=r, n_trials=500, trial_sharpes=ts)
    assert result.variance_was_assumed is False


# ---------------------------------------------------------------------------
# rationale content -- section 6.1 wants the arithmetic shown
# ---------------------------------------------------------------------------


def test_rationale_shows_observed_and_deflated_and_the_trial_count() -> None:
    r = _returns(n=1008, sharpe_annual=1.8)
    ts = np.random.default_rng(SEED).normal(0, 0.04, 2000)
    text = deflated_sharpe_ratio(returns=r, n_trials=2000, trial_sharpes=ts).rationale
    assert "Observed Sharpe" in text
    assert "deflated" in text
    assert "2,000 trials" in text
    assert "->" in text


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def test_rejects_zero_trials() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        deflated_sharpe_ratio(returns=_returns(), n_trials=0)


def test_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        deflated_sharpe_ratio(returns=np.array([0.01]), n_trials=1)


def test_psr_is_bounded_to_a_probability() -> None:
    for sr in (-5.0, 0.0, 5.0):
        p = probabilistic_sharpe_ratio(
            observed_sharpe=sr, benchmark_sharpe=0.0, n_obs=1000, skew=0.0, kurtosis=3.0
        )
        assert 0.0 <= p <= 1.0
