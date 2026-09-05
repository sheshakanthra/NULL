"""Walk-forward with purge and embargo. BUILD.md section 6.5.

Standard k-fold leaks when labels overlap in time. Two corrections:

  **Purge**  drop training samples whose label window overlaps the test window.
             A label computed over the following ``h`` bars at time ``t`` depends
             on data inside the test window whenever ``t + h`` reaches into it, so
             training on it is training on the answer.

  **Embargo** drop a further ``e`` bars after the test window. Serial correlation
             means a training sample taken immediately after the test period still
             carries information about it, even with no label overlap. Default
             ``e = 0.01 * T``.

**Gate: net-positive after costs in at least 60% of folds.** A strategy carried by
one fold is one lucky regime, not an edge.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from null.contracts import NonEmptyStr, NullFloat, NullModel, Probability

__all__ = [
    "DEFAULT_EMBARGO_FRACTION",
    "DEFAULT_N_FOLDS",
    "WalkForwardResult",
    "WalkForwardSplit",
    "walk_forward_splits",
    "walk_forward_consistency",
]

DEFAULT_N_FOLDS = 5
DEFAULT_EMBARGO_FRACTION = 0.01
MIN_FOLD_WIN_RATE = 0.60


class WalkForwardSplit(NullModel):
    fold_index: int
    train_start: int
    train_end: int
    """Exclusive."""
    test_start: int
    test_end: int
    """Exclusive."""
    purged_bars: int
    embargo_bars: int


class WalkForwardResult(NullModel):
    fold_win_rate: Probability
    n_folds: int
    n_positive_folds: int
    fold_returns: tuple[NullFloat, ...]
    purged_bars_total: int
    embargo_bars_total: int
    passed: bool
    rationale: NonEmptyStr


def walk_forward_splits(
    n_obs: int,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    label_horizon: int = 1,
    embargo_fraction: float = DEFAULT_EMBARGO_FRACTION,
) -> tuple[WalkForwardSplit, ...]:
    """Anchored walk-forward splits with purge and embargo applied.

    Anchored rather than rolling: training always starts at the beginning of the
    sample, which is what a practitioner actually does when re-fitting over time.
    """
    if n_folds < 2:
        raise ValueError(f"need at least 2 folds, got {n_folds}")
    if n_obs < n_folds * 2:
        raise ValueError(f"need at least {n_folds * 2} observations, got {n_obs}")
    if label_horizon < 1:
        raise ValueError(f"label_horizon must be at least 1, got {label_horizon}")

    embargo = int(np.floor(embargo_fraction * n_obs))
    edges = np.linspace(0, n_obs, n_folds + 2).astype(int)

    splits: list[WalkForwardSplit] = []
    for i in range(n_folds):
        test_start = int(edges[i + 1])
        test_end = int(edges[i + 2])
        # Purge: a training sample at t carries a label spanning [t, t+horizon),
        # so it must end before the test window opens by the full horizon.
        purge = label_horizon - 1
        train_end = max(0, test_start - purge)
        splits.append(
            WalkForwardSplit(
                fold_index=i,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purged_bars=test_start - train_end,
                embargo_bars=embargo,
            )
        )
    return tuple(splits)


def walk_forward_consistency(
    net_returns: npt.NDArray[np.float64],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    label_horizon: int = 1,
    embargo_fraction: float = DEFAULT_EMBARGO_FRACTION,
    min_win_rate: float = MIN_FOLD_WIN_RATE,
) -> WalkForwardResult:
    """Fraction of out-of-sample folds that were net-positive after costs."""
    r = np.asarray(net_returns, dtype=np.float64)
    splits = walk_forward_splits(
        r.size,
        n_folds=n_folds,
        label_horizon=label_horizon,
        embargo_fraction=embargo_fraction,
    )

    fold_returns: list[float] = []
    for split in splits:
        window = r[split.test_start : split.test_end]
        fold_returns.append(float(np.prod(1.0 + window) - 1.0) if window.size else 0.0)

    positive = sum(1 for x in fold_returns if x > 0.0)
    win_rate = positive / len(fold_returns)
    passed = win_rate >= min_win_rate

    verdict = (
        f"clears the {min_win_rate:.0%} threshold"
        if passed
        else (
            f"falls short of the {min_win_rate:.0%} threshold. A strategy carried by "
            "one fold is one lucky regime, not an edge"
        )
    )
    return WalkForwardResult(
        fold_win_rate=win_rate,
        n_folds=len(splits),
        n_positive_folds=positive,
        fold_returns=tuple(fold_returns),
        purged_bars_total=sum(s.purged_bars for s in splits),
        embargo_bars_total=sum(s.embargo_bars for s in splits),
        passed=passed,
        rationale=(
            f"Net-positive after costs in {positive} of {len(splits)} out-of-sample "
            f"folds ({win_rate:.0%}), which {verdict}. Fold returns: "
            + ", ".join(f"{x:+.2%}" for x in fold_returns)
            + f". Training windows are purged by {splits[0].purged_bars} bar(s) for a "
            f"{label_horizon}-bar label horizon and embargoed by "
            f"{splits[0].embargo_bars} bars ({embargo_fraction:.0%} of the sample), so "
            "no training sample overlaps or immediately abuts the window it is "
            "evaluated on."
        ),
    )
