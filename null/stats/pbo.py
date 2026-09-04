"""Probability of Backtest Overfitting via CSCV. BUILD.md section 6.2.

Combinatorially Symmetric Cross-Validation. Split the return matrix into ``S``
submatrices, form all ``C(S, S/2)`` train/test partitions, and measure how often
the configuration that looked best in-sample lands below median out-of-sample.

**The gate is an interval, not a point estimate.** Under the null of no selection
skill every candidate is exchangeable, so the in-sample winner's out-of-sample
rank is uniform and PBO's expected value is exactly 0.50. BUILD.md's original
`PBO < 0.5` therefore sat precisely on the null's mean and had no power: on a
noise grid it rejected about half the time, decided by the seed. Measured across
seven seeds, ``overfit_grid`` rejected on five and passed on two. No threshold
tweak fixes that honestly -- moving the gate to 0.3 would just be choosing a
number that makes the current fixtures behave.

So: PBO = 0.50 is the null, and rejecting it requires evidence. The gate passes
only when the **upper** bound of the bootstrap interval falls below 0.50. A point
estimate of 0.42 is not evidence against 0.50; an interval of [0.05, 0.28] is.

The interval comes from resampling the underlying time series and recomputing
CSCV per replicate. It is emphatically *not* a bootstrap over the 12,870
partitions: those share data and are heavily dependent, so resampling them would
understate the interval badly.

It is also *not* a with-replacement bootstrap of the rows, which was tried first
and is wrong here. Resampling with replacement puts the same original observation
into both the train and test halves of a replicate, which is leakage inside the
resample: the in-sample winner looks artificially persistent out of sample and PBO
is driven down. Measured, that produced intervals that did not even contain their
own point estimate (point 0.506, interval [0.021, 0.417]) and passed pure noise on
six seeds out of six -- a false PASS, worse than the coin flip it replaced.

Instead this uses **moving-block subsampling**: contiguous windows of length m < T,
so no observation is duplicated within a window and train/test stay disjoint. The
interval is built from the recentred, rescaled subsample distribution in the
standard way (Politis-Romano-Wolf), with the sqrt(m/T) convergence-rate factor.

Three states, not two, because a single-trial strategy can never produce a
selection to measure:

  1. ``n_trials == 1``                      NOT APPLICABLE -- gate passes.
  2. ``n_trials > 1``, no trial matrix      NOT COMPUTABLE -- gate fails.
  3. ``n_trials > 1``, matrix supplied      compute the interval, apply the gate.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
import numpy.typing as npt

from null.contracts import NonEmptyStr, NullModel, Probability


__all__ = [
    "DEFAULT_N_SPLITS",
    "PBOResult",
    "PBOState",
    "compute_pbo",
]

DEFAULT_N_SPLITS = 16
DEFAULT_N_SUBSAMPLES = 60
DEFAULT_WINDOW_FRACTION = 0.5
DEFAULT_PARTITION_SUBSAMPLE = 512
PBO_NULL = 0.5

#: The finding that must not be forgotten. Embedded verbatim in every rationale.
LOW_PBO_CAVEAT = (
    "A low PBO is never evidence that an edge is real. With many candidates over a "
    "short sample, the maximum of N noise Sharpes routinely exceeds a genuine "
    "edge's own Sharpe, so the real strategy is not the one selected and PBO ends "
    "up describing a search that never found it."
)

PBOState = Literal["not_applicable", "not_computable", "computed"]

_COMBO_BATCH = 512


class PBOResult(NullModel):
    state: PBOState
    passed: bool
    pbo: Probability
    """Point estimate. Meaningful only when ``state`` is 'computed'."""
    pbo_upper: Probability
    """Upper bound of the bootstrap interval. This is what the gate reads."""
    pbo_lower: Probability
    confidence: Probability
    n_strategies: int
    n_splits: int
    n_combinations: int
    n_subsamples: int
    partitions_subsampled: bool
    median_oos_rank: Probability
    rationale: NonEmptyStr


def _block_moments(
    returns: npt.NDArray[np.float64], n_splits: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    t, n = returns.shape
    edges = np.linspace(0, t, n_splits + 1).astype(int)
    s = np.zeros((n_splits, n), dtype=np.float64)
    s2 = np.zeros((n_splits, n), dtype=np.float64)
    counts = np.zeros(n_splits, dtype=np.float64)
    for b in range(n_splits):
        chunk = returns[edges[b] : edges[b + 1]]
        s[b] = chunk.sum(axis=0)
        s2[b] = (chunk**2).sum(axis=0)
        counts[b] = chunk.shape[0]
    return s, s2, counts


def _sharpe_from_moments(
    s: npt.NDArray[np.float64], s2: npt.NDArray[np.float64], n: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    mean = s / n[:, None]
    var = np.maximum(s2 / n[:, None] - mean**2, 0.0)
    sd = np.sqrt(var)
    result = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 1e-300)
    return np.asarray(result, dtype=np.float64)


def _pbo_once(
    returns: npt.NDArray[np.float64],
    n_splits: int,
    combos: list[tuple[int, ...]],
) -> tuple[float, float]:
    """PBO and median OOS rank for one return matrix."""
    s, s2, counts = _block_moments(returns, n_splits)
    n_strategies = returns.shape[1]
    below = 0
    total = 0
    ranks: list[float] = []
    for start in range(0, len(combos), _COMBO_BATCH):
        batch = combos[start : start + _COMBO_BATCH]
        mask = np.zeros((len(batch), n_splits), dtype=np.float64)
        for i, c in enumerate(batch):
            mask[i, list(c)] = 1.0
        anti = 1.0 - mask

        tr = _sharpe_from_moments(mask @ s, mask @ s2, mask @ counts)
        te = _sharpe_from_moments(anti @ s, anti @ s2, anti @ counts)

        best = np.argmax(tr, axis=1)
        best_oos = te[np.arange(len(batch)), best]
        rank = (te < best_oos[:, None]).sum(axis=1) / (n_strategies - 1)
        ranks.extend(float(x) for x in rank)
        below += int((rank <= 0.5).sum())
        total += len(batch)
    return below / total, float(np.median(np.asarray(ranks)))


def compute_pbo(
    trial_returns: npt.NDArray[np.float64] | None,
    *,
    n_trials: int = 1,
    n_splits: int = DEFAULT_N_SPLITS,
    n_subsamples: int = DEFAULT_N_SUBSAMPLES,
    confidence: float = 0.95,
    partition_subsample: int = DEFAULT_PARTITION_SUBSAMPLE,
    seed: int = 0,
) -> PBOResult:
    """CSCV with a stationary-bootstrap interval. See the module docstring."""
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even to split in half, got {n_splits}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    # State 1: no selection was performed.
    if n_trials == 1:
        return PBOResult(
            state="not_applicable",
            passed=True,
            pbo=0.0,
            pbo_lower=0.0,
            pbo_upper=0.0,
            n_strategies=1,
            rationale=(
                "Not applicable: one trial was declared, so no selection was "
                "performed and there is nothing for PBO to measure. This gate judges "
                "the process that picked a configuration; a strategy that ran a single "
                f"variant did not pick. {LOW_PBO_CAVEAT}"
            ),
            n_splits=n_splits,
            n_subsamples=0,
            partitions_subsampled=False,
            confidence=confidence,
            median_oos_rank=0.5,
            n_combinations=0,
        )

    # State 2: selection happened but the evidence to judge it was not supplied.
    supplied = (
        trial_returns is not None
        and np.asarray(trial_returns).ndim == 2
        and np.asarray(trial_returns).shape[1] >= 2
    )
    if not supplied:
        n_cols = 0 if trial_returns is None else int(np.asarray(trial_returns).shape[-1])
        return PBOResult(
            state="not_computable",
            passed=False,
            pbo=1.0,
            pbo_lower=0.0,
            pbo_upper=1.0,
            n_strategies=n_cols,
            rationale=(
                f"Not computable: {n_trials:,} trials were declared but per-trial "
                f"return series were not supplied ({n_cols} usable columns). CSCV needs "
                "the return matrix, not the count. Missing evidence fails the gate "
                "rather than passing it -- a search that will not show its candidates "
                f"cannot be shown to have avoided overfitting them. {LOW_PBO_CAVEAT}"
            ),
            n_splits=n_splits,
            n_subsamples=0,
            partitions_subsampled=False,
            confidence=confidence,
            median_oos_rank=0.5,
            n_combinations=0,
        )

    r = np.asarray(trial_returns, dtype=np.float64)
    t, n_strategies = r.shape
    if t < n_splits * 2:
        raise ValueError(
            f"need at least {n_splits * 2} observations for {n_splits} splits, got {t}"
        )

    all_combos = list(combinations(range(n_splits), n_splits // 2))
    point, median_rank = _pbo_once(r, n_splits, all_combos)

    # Moving-block subsampling: contiguous windows, so no observation is
    # duplicated and the train/test halves of each replicate stay disjoint.
    rng = np.random.default_rng(seed)
    window = max(n_splits * 2, int(t * DEFAULT_WINDOW_FRACTION))
    subsampled = partition_subsample < len(all_combos)
    if window >= t:
        replicates = np.asarray([point], dtype=np.float64)
    else:
        starts = rng.integers(0, t - window, size=n_subsamples)
        vals = np.empty(n_subsamples, dtype=np.float64)
        for b, s0 in enumerate(starts):
            combos = all_combos
            if subsampled:
                pick = rng.choice(len(all_combos), size=partition_subsample, replace=False)
                combos = [all_combos[i] for i in pick]
            vals[b], _ = _pbo_once(r[s0 : s0 + window], n_splits, combos)
        replicates = vals

    # Politis-Romano-Wolf: centre on the full-sample estimate and rescale the
    # subsample spread by the convergence-rate factor sqrt(m/T).
    alpha = 1.0 - confidence
    scale = float(np.sqrt(window / t))
    centred = replicates - point
    lower = float(np.clip(point - scale * np.quantile(centred, 1.0 - alpha / 2.0), 0.0, 1.0))
    upper = float(np.clip(point - scale * np.quantile(centred, alpha / 2.0), 0.0, 1.0))
    passed = upper < PBO_NULL

    subsample_note = (
        f" Each replicate used a random {partition_subsample:,} of the "
        f"{len(all_combos):,} partitions to keep the cost tractable; the point "
        "estimate uses all of them."
        if subsampled
        else ""
    )

    verdict = (
        f"the interval lies entirely below {PBO_NULL}, which is evidence the "
        "selection process did better than chance"
        if passed
        else (
            f"the interval reaches {upper:.2f}, at or above the null of {PBO_NULL}, so "
            "there is no evidence the selection process beat chance"
        )
    )

    return PBOResult(
        state="computed",
        passed=passed,
        pbo=point,
        pbo_lower=lower,
        pbo_upper=upper,
        n_strategies=int(n_strategies),
        n_splits=n_splits,
        n_subsamples=n_subsamples,
        partitions_subsampled=subsampled,
        confidence=confidence,
        median_oos_rank=median_rank,
        n_combinations=len(all_combos),
        rationale=(
            f"Across {len(all_combos):,} symmetric train/test partitions of "
            f"{n_strategies:,} variants, the in-sample best landed below the "
            f"out-of-sample median {point:.0%} of the time (median relative rank "
            f"{median_rank:.2f}). Under no selection skill every candidate is "
            f"exchangeable and PBO's expected value is exactly {PBO_NULL}, so the "
            "point estimate alone proves nothing; a moving-block subsampling interval "
            f"over {n_subsamples} moving-block subsamples of the underlying series "
            f"gives "
            f"[{lower:.2f}, {upper:.2f}] at {confidence:.0%} confidence, and "
            f"{verdict}.{subsample_note} {LOW_PBO_CAVEAT}"
        ),
    )
