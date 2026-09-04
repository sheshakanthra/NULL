"""Stationary bootstrap behaviour and negative controls. BUILD.md section 6.3."""

from __future__ import annotations

import numpy as np
import pytest

from null.stats.bootstrap import (
    default_mean_block_length,
    stationary_bootstrap_indices,
    stationary_bootstrap_samples,
)

SEED = 20260905


def _ar1(n=2000, rho=0.7, seed=SEED):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def _lag1_autocorr(x):
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    denom = float((xc**2).sum())
    return float((xc[1:] * xc[:-1]).sum() / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# the negative control that justifies the whole module
# ---------------------------------------------------------------------------


def test_iid_resampling_destroys_autocorrelation_but_blocks_preserve_it() -> None:
    """The control for choosing blocks over IID.

    If block resampling did not retain materially more dependence than IID
    resampling, there would be no reason to pay for this machinery -- and the
    p-values it feeds would be no better than the optimistic ones it exists to
    replace.
    """
    x = _ar1(rho=0.7)
    rng = np.random.default_rng(SEED)
    original = _lag1_autocorr(x)
    assert original > 0.6, original

    block_ac = [
        _lag1_autocorr(x[stationary_bootstrap_indices(x.size, rng=rng)])
        for _ in range(30)
    ]
    iid_ac = [
        _lag1_autocorr(x[rng.integers(0, x.size, x.size)]) for _ in range(30)
    ]

    assert float(np.mean(block_ac)) > 0.4, np.mean(block_ac)
    assert abs(float(np.mean(iid_ac))) < 0.1, np.mean(iid_ac)
    assert float(np.mean(block_ac)) > float(np.mean(iid_ac)) + 0.3


def test_longer_blocks_preserve_more_dependence() -> None:
    x = _ar1(rho=0.8)
    rng = np.random.default_rng(SEED)
    short = float(np.mean([
        _lag1_autocorr(x[stationary_bootstrap_indices(x.size, rng=rng, mean_block_length=2)])
        for _ in range(30)
    ]))
    long = float(np.mean([
        _lag1_autocorr(x[stationary_bootstrap_indices(x.size, rng=rng, mean_block_length=200)])
        for _ in range(30)
    ]))
    assert long > short


# ---------------------------------------------------------------------------
# block-length behaviour
# ---------------------------------------------------------------------------


def test_default_mean_block_length_is_sqrt_t() -> None:
    assert default_mean_block_length(10_000) == pytest.approx(100.0)
    assert default_mean_block_length(1) == 1.0


def test_realised_mean_block_length_matches_the_target() -> None:
    """Geometric block lengths should average to the requested mean."""
    rng = np.random.default_rng(SEED)
    n, target = 20_000, 50.0
    idx = stationary_bootstrap_indices(n, rng=rng, mean_block_length=target)
    # A new block starts wherever the index is not the previous index + 1 (mod n).
    continues = idx[1:] == (idx[:-1] + 1) % n
    n_blocks = 1 + int((~continues).sum())
    realised = n / n_blocks
    assert 0.7 * target < realised < 1.4 * target, realised


def test_block_length_one_reduces_to_iid() -> None:
    x = _ar1(rho=0.8)
    rng = np.random.default_rng(SEED)
    ac = float(np.mean([
        _lag1_autocorr(x[stationary_bootstrap_indices(x.size, rng=rng, mean_block_length=1)])
        for _ in range(30)
    ]))
    assert abs(ac) < 0.1


# ---------------------------------------------------------------------------
# shape, determinism, guards
# ---------------------------------------------------------------------------


def test_indices_are_in_range_and_correct_length() -> None:
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_indices(500, rng=rng)
    assert idx.shape == (500,)
    assert idx.min() >= 0 and idx.max() < 500


def test_same_seed_gives_identical_resamples() -> None:
    a = stationary_bootstrap_indices(300, rng=np.random.default_rng(7))
    b = stationary_bootstrap_indices(300, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_resamples() -> None:
    a = stationary_bootstrap_indices(300, rng=np.random.default_rng(7))
    b = stationary_bootstrap_indices(300, rng=np.random.default_rng(8))
    assert not np.array_equal(a, b)


def test_samples_resample_rows_together_preserving_cross_section() -> None:
    """Columns must move together, or CSCV and the Reality Check lose the
    correlation structure they are reasoning about."""
    rng = np.random.default_rng(SEED)
    data = np.column_stack([np.arange(100.0), np.arange(100.0) * 10.0])
    out = stationary_bootstrap_samples(data, n_replicates=5, rng=rng)
    assert out.shape == (5, 100, 2)
    for rep in out:
        assert np.allclose(rep[:, 1], rep[:, 0] * 10.0)


def test_rejects_non_positive_inputs() -> None:
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="n_obs"):
        stationary_bootstrap_indices(0, rng=rng)
    with pytest.raises(ValueError, match="mean_block_length"):
        stationary_bootstrap_indices(10, rng=rng, mean_block_length=-1.0)
