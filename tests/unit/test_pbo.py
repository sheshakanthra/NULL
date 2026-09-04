"""PBO / CSCV behaviour and its negative controls. BUILD.md section 6.2."""

from __future__ import annotations

import numpy as np
import pytest

from null.stats.pbo import compute_pbo

SEED = 20260905


def _independent_noise(t=1008, n=200, seed=SEED):
    """Independent strategies, identical variance. True Sharpe zero everywhere."""
    return np.random.default_rng(seed).normal(0.0, 0.011, (t, n))


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------


def test_pbo_on_pure_noise_sits_near_the_coin_flip() -> None:
    """The null's expectation, and the reason the 0.50 gate is a P0.

    Every variant has a true Sharpe of zero, so the in-sample winner's expected
    out-of-sample rank is the median and PBO converges on 0.50 -- exactly where
    BUILD.md puts the threshold. Measured range across seeds was 0.32 to 0.76.
    See docs/pbo_calibration.md.
    """
    values = [compute_pbo(_independent_noise(seed=s)).pbo for s in (1, 2, 7, 42)]
    assert all(0.2 < v < 0.85 for v in values), values
    assert 0.35 < float(np.mean(values)) < 0.70, (
        f"noise PBO centred at {np.mean(values):.3f}, not near the 0.5 null"
    )


def test_pbo_verdict_on_noise_is_not_stable_across_seeds() -> None:
    """Pins the P0 so it cannot be quietly forgotten.

    This asserts the CURRENT broken behaviour on purpose. When the gate is fixed
    (docs/pbo_calibration.md, options A-C) this test should start failing, and
    that failure is the signal to delete it.
    """
    values = [compute_pbo(_independent_noise(seed=s)).pbo for s in (1, 2, 7, 42, 99)]
    assert min(values) < 0.5 < max(values), (
        f"PBO on noise no longer straddles the 0.5 gate ({values}) -- if the gate "
        "was fixed, remove this test and docs/pbo_calibration.md"
    )


# ---------------------------------------------------------------------------
# the degenerate case
# ---------------------------------------------------------------------------


def test_no_trials_means_no_selection_means_zero_pbo() -> None:
    """One trial did no selecting, so there is nothing that could be overfit."""
    result = compute_pbo(None)
    assert result.pbo == 0.0
    assert result.degenerate_no_selection is True
    assert "no selection" in result.rationale.lower()


def test_a_single_strategy_column_is_also_degenerate() -> None:
    result = compute_pbo(np.random.default_rng(SEED).normal(0, 0.01, (500, 1)))
    assert result.degenerate_no_selection is True
    assert result.pbo == 0.0


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_combination_count_matches_the_binomial() -> None:
    from math import comb

    result = compute_pbo(_independent_noise(t=504, n=20), n_splits=16)
    assert result.n_combinations == comb(16, 8) == 12870


def test_odd_split_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        compute_pbo(_independent_noise(), n_splits=15)


def test_too_few_observations_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        compute_pbo(np.random.default_rng(SEED).normal(0, 0.01, (20, 10)), n_splits=16)


def test_pbo_is_a_probability() -> None:
    result = compute_pbo(_independent_noise())
    assert 0.0 <= result.pbo <= 1.0


def test_rationale_names_the_partition_count_and_the_chance_baseline() -> None:
    text = compute_pbo(_independent_noise()).rationale
    assert "12,870" in text
    assert "0.50" in text or "chance" in text.lower()
