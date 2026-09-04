"""Politis-Romano stationary bootstrap. BUILD.md section 6.3.

IID resampling of financial returns destroys autocorrelation and produces
p-values that are too optimistic. It is the same class of error as using OLS
standard errors on autocorrelated residuals: the estimator assumes independence
that is not there, understates the true variance, and hands back more confidence
than the data supports.

The stationary bootstrap resamples *blocks* of consecutive observations with
geometrically distributed lengths, so the local dependence structure survives
resampling. Geometric rather than fixed lengths is what makes the resampled
series stationary -- fixed blocks introduce artefacts at the block boundaries.

Mean block length defaults to sqrt(T), which is the conventional choice and is
what BUILD.md specifies.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "default_mean_block_length",
    "stationary_bootstrap_indices",
    "stationary_bootstrap_samples",
]


def default_mean_block_length(n_obs: int) -> float:
    """sqrt(T), floored at 1. Longer blocks preserve more dependence, at the cost
    of fewer effectively independent blocks in each replicate."""
    if n_obs <= 1:
        return 1.0
    return max(1.0, float(np.sqrt(n_obs)))


def stationary_bootstrap_indices(
    n_obs: int,
    *,
    rng: np.random.Generator,
    mean_block_length: float | None = None,
) -> npt.NDArray[np.int64]:
    """One resampled index vector of length ``n_obs``.

    Each step either continues the current block (wrapping at the end of the
    series, which is what makes the scheme circular and therefore stationary) or
    starts a new block at a uniformly random position, with probability
    ``1 / mean_block_length``.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    length = mean_block_length or default_mean_block_length(n_obs)
    if length <= 0.0:
        raise ValueError(f"mean_block_length must be positive, got {length}")

    p = min(1.0, 1.0 / length)
    starts = rng.random(n_obs) < p
    starts[0] = True

    anchors = rng.integers(0, n_obs, size=n_obs)
    positions = np.arange(n_obs)
    # Index of the most recent block start at or before each position.
    last_start = np.maximum.accumulate(np.where(starts, positions, -1))
    offsets = positions - last_start
    idx = (anchors[last_start] + offsets) % n_obs
    return np.asarray(idx, dtype=np.int64)


def stationary_bootstrap_samples(
    data: npt.NDArray[np.float64],
    *,
    n_replicates: int,
    rng: np.random.Generator,
    mean_block_length: float | None = None,
) -> npt.NDArray[np.float64]:
    """``n_replicates`` resamples of ``data`` along axis 0.

    Rows are resampled together, so cross-sectional relationships between columns
    are preserved. That matters for the Reality Check and for CSCV: resampling
    each strategy independently would destroy the correlation between candidates,
    which is precisely the structure those procedures are reasoning about.
    """
    arr = np.asarray(data, dtype=np.float64)
    n_obs = arr.shape[0]
    out = np.empty((n_replicates, *arr.shape), dtype=np.float64)
    for b in range(n_replicates):
        idx = stationary_bootstrap_indices(
            n_obs, rng=rng, mean_block_length=mean_block_length
        )
        out[b] = arr[idx]
    return out
