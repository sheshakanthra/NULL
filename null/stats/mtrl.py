"""Minimum Track Record Length. BUILD.md section 6.4.

How many observations would be needed for an observed Sharpe to be significant at
95%, given the sample's skew and kurtosis. Reported in years.

If MTRL exceeds the backtest length, the result is under-powered by construction:
the track record is too short to support the claim no matter how good the number
looks. That is not a marginal caveat -- it means the evidence cannot distinguish
the strategy from noise, and the report has to say so in plain English.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import stats

from null.contracts import NonEmptyStr, NonNegativeFloat, NullFloat, NullModel

__all__ = ["MTRLResult", "minimum_track_record_length"]

TRADING_DAYS = 252


class MTRLResult(NullModel):
    mtrl_obs: NonNegativeFloat
    mtrl_years: NonNegativeFloat
    observed_sharpe_annual: NullFloat
    n_obs: int
    n_obs_years: NonNegativeFloat
    is_underpowered: bool
    """True when the backtest is shorter than the track record its own Sharpe needs."""
    rationale: NonEmptyStr


def minimum_track_record_length(
    returns: npt.NDArray[np.float64],
    *,
    target_sharpe: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
) -> MTRLResult:
    """Observations required for the observed Sharpe to clear ``target_sharpe``."""
    r = np.asarray(returns, dtype=np.float64)
    n_obs = int(r.size)
    if n_obs < 4:
        raise ValueError(f"need at least 4 observations, got {n_obs}")

    sd = float(np.std(r, ddof=1))
    sr = float(np.mean(r) / sd) if sd > 0.0 else 0.0
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    ann = math.sqrt(periods_per_year)

    excess = sr - target_sharpe
    if excess <= 0.0:
        # No amount of data makes a Sharpe at or below the target significant
        # above it. Report the sample length as a floor and say so.
        return MTRLResult(
            mtrl_obs=float(n_obs),
            mtrl_years=n_obs / periods_per_year,
            observed_sharpe_annual=sr * ann,
            n_obs=n_obs,
            n_obs_years=n_obs / periods_per_year,
            is_underpowered=True,
            rationale=(
                f"Observed Sharpe {sr * ann:.2f} does not exceed the target of "
                f"{target_sharpe * ann:.2f}, so no track record length would make it "
                "significantly better. The question of how much data would be needed "
                "does not arise; there is nothing to establish."
            ),
        )

    z = float(stats.norm.ppf(confidence))
    variance_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    variance_term = max(variance_term, 1e-12)
    mtrl_obs = 1.0 + variance_term * (z / excess) ** 2
    mtrl_years = mtrl_obs / periods_per_year
    observed_years = n_obs / periods_per_year
    underpowered = mtrl_obs > n_obs

    if underpowered:
        verdict = (
            f"The backtest is {observed_years:.1f} years long, so it is under-powered "
            "by construction: this Sharpe would need a longer track record than exists "
            "before it could be called significant, and no analysis of the existing "
            "data changes that."
        )
    else:
        verdict = (
            f"The backtest is {observed_years:.1f} years long, which clears that "
            "requirement."
        )

    return MTRLResult(
        mtrl_obs=mtrl_obs,
        mtrl_years=mtrl_years,
        observed_sharpe_annual=sr * ann,
        n_obs=n_obs,
        n_obs_years=observed_years,
        is_underpowered=underpowered,
        rationale=(
            f"An annualised Sharpe of {sr * ann:.2f}, with skew {skew:.2f} and "
            f"kurtosis {kurt:.2f}, would need {mtrl_years:.1f} years of observations "
            f"before it could be called significant at {confidence:.0%} confidence. "
            f"{verdict}"
        ),
    )
