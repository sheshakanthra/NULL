"""White's Reality Check / Hansen's SPA. BUILD.md section 6.3.

Bootstrap the null that the best of N strategies does not beat the benchmark,
using the stationary bootstrap so that autocorrelation survives resampling. IID
resampling produces p-values that are too optimistic; see null/stats/bootstrap.py.

The test statistic is the best mean outperformance across candidates, scaled by
sqrt(T). Under the null every candidate has non-positive expected outperformance,
so the bootstrap distribution is built from **recentred** replicates -- each
candidate's bootstrap mean minus its full-sample mean -- which imposes the null
without assuming anything about which candidate is best.

Unlike PBO, this statistic does not need a distribution over candidate-set
realisations: it asks whether the observed maximum outperformance is large
relative to what the same candidate set produces under resampling of its own
history. That is answerable from one dataset, which is why the same demotion
argument does not apply here -- but the controls below are what establish it
rather than the argument.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from null.contracts import NonEmptyStr, NullFloat, NullModel, Probability
from null.stats.bootstrap import (
    default_mean_block_length,
    stationary_bootstrap_indices,
)

__all__ = ["DEFAULT_N_BOOTSTRAP", "RealityCheckResult", "reality_check"]

DEFAULT_N_BOOTSTRAP = 1000


class RealityCheckResult(NullModel):
    p_value: Probability
    """P(the best candidate's outperformance arises under the null)."""
    observed_statistic: NullFloat
    best_candidate: int
    n_candidates: int
    n_obs: int
    n_bootstrap: int
    mean_block_length: NullFloat
    rationale: NonEmptyStr


def reality_check(
    strategy_returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    mean_block_length: float | None = None,
    seed: int = 0,
) -> RealityCheckResult:
    """White's Reality Check over a candidate set.

    ``strategy_returns`` is (T, N) -- one column per candidate. A single candidate
    is allowed; the test then simply asks whether that one beats the benchmark.
    """
    s = np.asarray(strategy_returns, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    b = np.asarray(benchmark_returns, dtype=np.float64)
    if b.ndim != 1:
        raise ValueError(f"benchmark must be one-dimensional, got shape {b.shape}")

    n_obs = min(s.shape[0], b.size)
    if n_obs < 4:
        raise ValueError(f"need at least 4 aligned observations, got {n_obs}")
    s, b = s[:n_obs], b[:n_obs]
    n_candidates = int(s.shape[1])

    # Outperformance of each candidate over the benchmark, per period.
    f = s - b[:, None]
    means = f.mean(axis=0)
    root_t = np.sqrt(n_obs)
    observed = float(root_t * means.max())
    best = int(np.argmax(means))

    length = mean_block_length or default_mean_block_length(n_obs)
    rng = np.random.default_rng(seed)

    exceedances = 0
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n_obs, rng=rng, mean_block_length=length)
        # Recentre by the full-sample means: this imposes the null that no
        # candidate truly outperforms, without presuming which one is best.
        boot = root_t * (f[idx].mean(axis=0) - means)
        if float(boot.max()) >= observed:
            exceedances += 1

    p_value = exceedances / n_bootstrap

    verdict = (
        "no candidate's outperformance survives the null"
        if p_value >= 0.05
        else "the best candidate's outperformance is larger than resampling the "
        "same history under the null produces"
    )
    return RealityCheckResult(
        p_value=p_value,
        observed_statistic=observed,
        best_candidate=best,
        n_candidates=n_candidates,
        n_obs=n_obs,
        n_bootstrap=n_bootstrap,
        mean_block_length=length,
        rationale=(
            f"Best of {n_candidates:,} candidate(s) outperformed the benchmark by "
            f"{means.max() * 252:.2%}/yr, giving a test statistic of {observed:.3f}. "
            f"Against {n_bootstrap:,} stationary-bootstrap resamples (mean block "
            f"{length:.0f} bars, chosen as sqrt(T) so autocorrelation survives "
            f"resampling), that was matched or exceeded {p_value:.1%} of the time, so "
            f"{verdict}. IID resampling would have destroyed the autocorrelation in "
            "these returns and produced a p-value that is too optimistic."
        ),
    )
