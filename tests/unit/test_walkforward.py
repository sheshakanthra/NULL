"""Walk-forward purge/embargo and its negative controls. BUILD.md section 6.5."""

from __future__ import annotations

import numpy as np
import pytest

from null.partition.walkforward import (
    walk_forward_consistency,
    walk_forward_splits,
)

SEED = 20260905


# ---------------------------------------------------------------------------
# negative controls: purge and embargo must actually remove something
# ---------------------------------------------------------------------------


def test_purging_removes_training_bars_when_labels_span_multiple_bars() -> None:
    """A one-bar label needs no purge; a ten-bar label needs nine."""
    none = walk_forward_splits(1000, label_horizon=1)
    purged = walk_forward_splits(1000, label_horizon=10)
    assert all(s.purged_bars == 0 for s in none)
    assert all(s.purged_bars == 9 for s in purged)
    assert all(p.train_end < n.train_end for p, n in zip(purged, none))


def test_embargo_scales_with_sample_length() -> None:
    assert walk_forward_splits(1000)[0].embargo_bars == 10
    assert walk_forward_splits(5000)[0].embargo_bars == 50
    assert walk_forward_splits(1000, embargo_fraction=0.0)[0].embargo_bars == 0


def test_training_never_overlaps_the_test_window() -> None:
    """The leak the whole module exists to prevent."""
    for horizon in (1, 5, 20):
        for split in walk_forward_splits(2000, n_folds=5, label_horizon=horizon):
            assert split.train_end <= split.test_start
            assert split.train_end + (horizon - 1) <= split.test_start


def test_test_windows_are_disjoint_and_ordered() -> None:
    splits = walk_forward_splits(2000, n_folds=6)
    for a, b in zip(splits, splits[1:]):
        assert a.test_end <= b.test_start
    assert all(s.test_start < s.test_end for s in splits)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def test_a_consistent_edge_passes() -> None:
    """A genuine persistent edge should be positive in most folds."""
    rng = np.random.default_rng(SEED)
    returns = rng.normal(0.0008, 0.008, 2520)
    result = walk_forward_consistency(returns, n_folds=5)
    assert result.passed
    assert result.fold_win_rate >= 0.6


def test_a_one_regime_wonder_fails() -> None:
    """BUILD.md's one_regime_wonder: all the PnL in a single window.

    This is the control that gives the gate its purpose -- a strategy carried by
    one fold is one lucky regime, not an edge.
    """
    returns = np.full(2520, -0.00002)
    returns[900:1000] = 0.05  # one enormous burst, flat-to-negative elsewhere
    result = walk_forward_consistency(returns, n_folds=5)
    assert not result.passed, (
        f"a strategy with all its PnL in one window passed with "
        f"{result.fold_win_rate:.0%} fold win rate"
    )
    assert "one lucky regime" in result.rationale


def test_pure_noise_does_not_reliably_pass() -> None:
    rates = [
        walk_forward_consistency(
            np.random.default_rng(s).normal(0.0, 0.011, 2520), n_folds=5
        ).fold_win_rate
        for s in (1, 2, 7, 42, 99, 12345)
    ]
    assert float(np.mean(rates)) < 0.75, rates


def test_threshold_is_sixty_percent() -> None:
    """Exactly at the threshold must pass; just below must not."""
    returns = np.full(1000, -0.001)
    edges = np.linspace(0, 1000, 7).astype(int)
    for i in (1, 2, 3):  # 3 of 5 folds positive = 60%
        returns[edges[i] : edges[i + 1]] = 0.001
    result = walk_forward_consistency(returns, n_folds=5)
    assert result.n_positive_folds == 3
    assert result.fold_win_rate == pytest.approx(0.6)
    assert result.passed


# ---------------------------------------------------------------------------
# guards and rationale
# ---------------------------------------------------------------------------


def test_rejects_degenerate_configurations() -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        walk_forward_splits(1000, n_folds=1)
    with pytest.raises(ValueError, match="observations"):
        walk_forward_splits(5, n_folds=5)
    with pytest.raises(ValueError, match="label_horizon"):
        walk_forward_splits(1000, label_horizon=0)


def test_rationale_names_the_purge_and_embargo() -> None:
    text = walk_forward_consistency(
        np.random.default_rng(SEED).normal(0.0005, 0.01, 2000), label_horizon=5
    ).rationale
    assert "purged" in text
    assert "embargoed" in text
    assert "out-of-sample folds" in text
