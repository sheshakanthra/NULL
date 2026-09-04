"""Probability of Backtest Overfitting via CSCV. BUILD.md section 6.2.

Combinatorially Symmetric Cross-Validation. Split the return matrix into ``S``
submatrices, form all ``C(S, S/2)`` train/test partitions, and measure how often
the configuration that looked best in-sample lands below median out-of-sample.

**Gate: PBO < 0.5.** Above 0.5 the selection process is worse than a coin flip --
picking the in-sample winner is actively worse than picking at random.

PBO measures the *selection process*, not the strategy. That distinction decides
the degenerate case: a run with one trial did no selection, so there is nothing
that could have been overfit and PBO is zero. That is the correct answer, not
missing evidence, and it is why an honest single-trial strategy is not punished
by this gate.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import numpy.typing as npt

from null.contracts import NonEmptyStr, NullModel, Probability

__all__ = ["DEFAULT_N_SPLITS", "PBOResult", "compute_pbo"]

DEFAULT_N_SPLITS = 16

#: Combinations are processed in batches so that a wide trial matrix cannot blow
#: memory: C(16,8) is 12,870 partitions, and materialising that against thousands
#: of strategies at once would be hundreds of megabytes.
_COMBO_BATCH = 512


class PBOResult(NullModel):
    pbo: Probability
    n_strategies: int
    n_splits: int
    n_combinations: int
    median_oos_rank: Probability
    """Median relative out-of-sample rank of the in-sample winner. 0.5 is chance."""
    degenerate_no_selection: bool
    rationale: NonEmptyStr


def _block_moments(
    returns: npt.NDArray[np.float64], n_splits: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-block sums, sums of squares and counts, so partitions combine cheaply."""
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
    """Sharpe per strategy from pooled sums. Zero-variance strategies score zero."""
    mean = s / n[:, None]
    var = s2 / n[:, None] - mean**2
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    result = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 1e-300)
    return np.asarray(result, dtype=np.float64)


def compute_pbo(
    trial_returns: npt.NDArray[np.float64] | None,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
) -> PBOResult:
    """CSCV over the trial return matrix.

    ``trial_returns`` is (T observations, N strategies). ``None`` or a single
    column means no selection took place.
    """
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even to split in half, got {n_splits}")

    if trial_returns is None or np.asarray(trial_returns).ndim < 2 or (
        np.asarray(trial_returns).shape[1] < 2
    ):
        n = 0 if trial_returns is None else int(np.asarray(trial_returns).shape[1])
        return PBOResult(
            pbo=0.0,
            n_strategies=n,
            n_splits=n_splits,
            n_combinations=0,
            median_oos_rank=0.5,
            degenerate_no_selection=True,
            rationale=(
                f"Only {n} trial return series were supplied, so no selection took "
                "place and there is nothing that could have been overfit. PBO is zero "
                "by construction here, not by evidence: this gate measures the "
                "selection process, and a strategy that ran one variant did not select."
            ),
        )

    r = np.asarray(trial_returns, dtype=np.float64)
    t, n_strategies = r.shape
    if t < n_splits * 2:
        raise ValueError(
            f"need at least {n_splits * 2} observations for {n_splits} splits, got {t}"
        )

    s, s2, counts = _block_moments(r, n_splits)
    half = n_splits // 2
    all_blocks = np.arange(n_splits)
    combos = list(combinations(range(n_splits), half))

    logits: list[float] = []
    ranks: list[float] = []

    for start in range(0, len(combos), _COMBO_BATCH):
        batch = combos[start : start + _COMBO_BATCH]
        mask = np.zeros((len(batch), n_splits), dtype=np.float64)
        for i, c in enumerate(batch):
            mask[i, list(c)] = 1.0
        anti = 1.0 - mask

        tr_s, tr_s2, tr_n = mask @ s, mask @ s2, mask @ counts
        te_s, te_s2, te_n = anti @ s, anti @ s2, anti @ counts

        tr_sharpe = _sharpe_from_moments(tr_s, tr_s2, tr_n)
        te_sharpe = _sharpe_from_moments(te_s, te_s2, te_n)

        best = np.argmax(tr_sharpe, axis=1)
        rows = np.arange(len(batch))
        best_oos = te_sharpe[rows, best]

        # Relative rank of the in-sample winner within the out-of-sample results.
        rank = (te_sharpe < best_oos[:, None]).sum(axis=1) / (n_strategies - 1)
        rank = np.clip(rank, 1e-9, 1.0 - 1e-9)
        ranks.extend(float(x) for x in rank)
        logits.extend(float(x) for x in np.log(rank / (1.0 - rank)))

    logit_arr = np.asarray(logits, dtype=np.float64)
    pbo = float(np.mean(logit_arr <= 0.0))
    median_rank = float(np.median(np.asarray(ranks, dtype=np.float64)))

    return PBOResult(
        pbo=pbo,
        n_strategies=int(n_strategies),
        n_splits=n_splits,
        n_combinations=len(combos),
        median_oos_rank=median_rank,
        degenerate_no_selection=False,
        rationale=(
            f"Across {len(combos):,} symmetric train/test partitions of "
            f"{n_strategies:,} variants, the configuration that ranked best in-sample "
            f"landed below the out-of-sample median {pbo:.0%} of the time (median "
            f"relative rank {median_rank:.2f}, where 0.50 is chance). A PBO at or "
            "above 0.50 means selecting the in-sample winner is no better than "
            "picking at random, and the reported backtest is a property of the search "
            "rather than of the market."
        ),
    )
