"""Deflated Sharpe Ratio. Bailey & Lopez de Prado. BUILD.md section 6.1.

The headline gate. It adjusts an observed Sharpe for four things at once:

  (a) how many variants were tried,
  (b) the variance across those trials,
  (c) non-normality of returns (skew and excess kurtosis),
  (d) sample length.

A Sharpe of 1.8 from 2,000 grid-search trials on four years of data deflates to
approximately nothing, and the output here is built so the report can show that
arithmetic rather than just its conclusion.

Two pieces:

  PSR(SR*)  the probability the true Sharpe exceeds a benchmark SR*, given the
            observed Sharpe, sample length, skew and kurtosis.
  SR_0      the Sharpe you would expect the *best* of N independent trials to
            show purely by chance, given the spread across trials.

  DSR = PSR(SR_0)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import stats

from null.contracts import NonEmptyStr, NonNegativeFloat, NullFloat, NullModel, Probability

__all__ = [
    "DEFAULT_ASSUMED_TRIAL_SHARPE_VARIANCE",
    "DSRResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]

TRADING_DAYS = 252

#: Euler-Mascheroni constant, used in the expected-maximum order statistic.
EULER_MASCHERONI = 0.5772156649015329

#: Used only when a caller declares n_trials > 1 but supplies no per-trial Sharpes.
#: Deliberately generous to the strategy rather than punitive -- a small assumed
#: spread produces a small SR_0 and so deflates less. The rationale string says
#: out loud that it was assumed, because an assumed input to the headline gate
#: that goes unmentioned is exactly the kind of quiet fudge NULL exists to catch.
DEFAULT_ASSUMED_TRIAL_SHARPE_VARIANCE = 0.5


class DSRResult(NullModel):
    observed_sharpe: NullFloat
    """Per-period, as the formula uses it."""
    observed_sharpe_annual: NullFloat
    deflated_sharpe: Probability
    """Probability the true Sharpe exceeds the selection-adjusted benchmark."""
    expected_max_sharpe: NullFloat
    """SR_0: what the best of n_trials would show by luck alone, per period."""
    var_trial_sharpes: NonNegativeFloat
    variance_was_assumed: bool
    skew: NullFloat
    kurtosis: NullFloat
    """Full kurtosis, not excess. Normal is 3."""
    n_obs: int
    n_trials: int
    rationale: NonEmptyStr


def probabilistic_sharpe_ratio(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """P(true Sharpe > benchmark), adjusting for skew, kurtosis and sample length.

    All Sharpes per-period. Negative skew and fat tails both widen the estimator's
    standard error, which is why a strategy with a great Sharpe and a long left
    tail deserves less confidence than the ratio alone suggests.
    """
    if n_obs < 2:
        return 0.0
    denominator_sq = (
        1.0
        - skew * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    )
    if denominator_sq <= 0.0:
        # The variance estimate has gone non-positive, which means the moment
        # estimates are not usable. No evidence, not free confidence.
        return 0.0
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(
        denominator_sq
    )
    return float(stats.norm.cdf(z))


def expected_max_sharpe(*, n_trials: int, var_trial_sharpes: float) -> float:
    """SR_0 -- the expected maximum Sharpe across n_trials under a zero-Sharpe null.

    With a single trial there was no selection, so nothing needs deflating and the
    benchmark is zero. That is not a special case bolted on; it is what the order
    statistic means when there is one draw.
    """
    if n_trials <= 1:
        return 0.0
    if var_trial_sharpes <= 0.0:
        return 0.0
    sigma = math.sqrt(var_trial_sharpes)
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def _build_rationale(
    *,
    observed_annual: float,
    deflated: float,
    n_trials: int,
    sr0_annual: float,
    skew: float,
    kurtosis: float,
    n_obs: int,
    assumed: bool,
) -> str:
    """Show the deflation, not just the verdict. The arithmetic is the teaching."""
    lead = (
        f"Observed Sharpe {observed_annual:.2f} -> deflated {deflated:.2f}, "
        f"because you declared {n_trials:,} trial{'s' if n_trials != 1 else ''}."
    )
    if n_trials <= 1:
        body = (
            " With a single declared trial there is no selection to adjust for, so "
            "the deflation reflects only sample length and the shape of the return "
            f"distribution: skew {skew:.2f}, kurtosis {kurtosis:.2f}, {n_obs:,} "
            "observations."
        )
    else:
        body = (
            f" The best of {n_trials:,} trials would be expected to show a Sharpe of "
            f"{sr0_annual:.2f} by luck alone, so that is the bar the observed value "
            f"has to clear rather than zero. Adjusted for that, for skew {skew:.2f} "
            f"and kurtosis {kurtosis:.2f}, over {n_obs:,} observations, the "
            f"probability the true Sharpe is above zero is {deflated:.2f}."
        )
    caveat = (
        " The variance across trial Sharpes was NOT supplied and a default of "
        f"{DEFAULT_ASSUMED_TRIAL_SHARPE_VARIANCE} was assumed; supply per-trial "
        "Sharpes for a defensible number."
        if assumed
        else ""
    )
    return lead + body + caveat


def deflated_sharpe_ratio(
    *,
    returns: npt.NDArray[np.float64],
    n_trials: int,
    trial_sharpes: npt.NDArray[np.float64] | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> DSRResult:
    """The headline gate's number, with the arithmetic that produced it."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    r = np.asarray(returns, dtype=np.float64)
    n_obs = int(r.size)
    if n_obs < 2:
        raise ValueError(f"need at least 2 observations, got {n_obs}")

    sd = float(np.std(r, ddof=1))
    sr = float(np.mean(r) / sd) if sd > 0.0 else 0.0
    skew = float(stats.skew(r, bias=False)) if n_obs > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if n_obs > 3 else 3.0

    assumed = False
    if trial_sharpes is not None and np.asarray(trial_sharpes).size >= 2:
        var_trials = float(np.var(np.asarray(trial_sharpes, dtype=np.float64), ddof=1))
    elif n_trials > 1:
        var_trials = DEFAULT_ASSUMED_TRIAL_SHARPE_VARIANCE
        assumed = True
    else:
        var_trials = 0.0

    sr0 = expected_max_sharpe(n_trials=n_trials, var_trial_sharpes=var_trials)
    dsr = probabilistic_sharpe_ratio(
        observed_sharpe=sr,
        benchmark_sharpe=sr0,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt,
    )

    ann = math.sqrt(periods_per_year)
    return DSRResult(
        observed_sharpe=sr,
        observed_sharpe_annual=sr * ann,
        deflated_sharpe=dsr,
        expected_max_sharpe=sr0,
        var_trial_sharpes=var_trials,
        variance_was_assumed=assumed,
        skew=skew,
        kurtosis=kurt,
        n_obs=n_obs,
        n_trials=n_trials,
        rationale=_build_rationale(
            observed_annual=sr * ann,
            deflated=dsr,
            n_trials=n_trials,
            sr0_annual=sr0 * ann,
            skew=skew,
            kurtosis=kurt,
            n_obs=n_obs,
            assumed=assumed,
        ),
    )
