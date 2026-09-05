"""Minimum Track Record Length. BUILD.md section 6.4."""

from __future__ import annotations

import numpy as np
import pytest

from null.stats.mtrl import minimum_track_record_length

SEED = 20260905


def _returns(n, sharpe_annual, vol=0.011, seed=SEED):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    return raw * vol + sharpe_annual * vol / np.sqrt(252)


def test_a_higher_sharpe_needs_a_shorter_record() -> None:
    weak = minimum_track_record_length(_returns(2520, 0.3))
    strong = minimum_track_record_length(_returns(2520, 1.5))
    assert strong.mtrl_years < weak.mtrl_years


def test_a_short_backtest_with_a_modest_sharpe_is_flagged_underpowered() -> None:
    """The case BUILD.md wants stated in plain English."""
    result = minimum_track_record_length(_returns(504, 0.4))
    assert result.is_underpowered
    assert "under-powered by construction" in result.rationale
    assert result.mtrl_years > result.n_obs_years


def test_a_long_backtest_with_a_strong_sharpe_is_adequately_powered() -> None:
    result = minimum_track_record_length(_returns(2520, 1.5))
    assert not result.is_underpowered
    assert "clears that requirement" in result.rationale


def test_a_sharpe_below_the_target_can_never_become_significant() -> None:
    result = minimum_track_record_length(_returns(2520, -0.2))
    assert result.is_underpowered
    assert "nothing to establish" in result.rationale


def test_negative_skew_and_fat_tails_lengthen_the_requirement() -> None:
    """Same Sharpe, worse distribution shape, more data needed to believe it."""
    rng = np.random.default_rng(SEED)
    n = 2520
    clean = _returns(n, 0.8)
    ugly = clean.copy()
    ugly[rng.integers(0, n, 25)] -= 0.06  # a fat left tail
    ugly = (ugly - ugly.mean()) / ugly.std(ddof=1) * 0.011 + clean.mean()
    a = minimum_track_record_length(clean)
    b = minimum_track_record_length(ugly)
    assert b.mtrl_years > a.mtrl_years, (
        f"a fat left tail did not lengthen the requirement: {a.mtrl_years:.2f} -> "
        f"{b.mtrl_years:.2f} years"
    )


def test_rationale_names_the_years_and_the_confidence() -> None:
    text = minimum_track_record_length(_returns(1008, 0.9)).rationale
    assert "years of observations" in text
    assert "95%" in text


def test_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        minimum_track_record_length(np.array([0.01, 0.02, 0.03]))
