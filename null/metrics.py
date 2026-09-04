"""The performance metric panel from BUILD.md section 4.

A pure function of a return series. No I/O, no wall-clock, no globals.

Not in the section 1 layout -- section 1 has no home for metrics, but folds
(section 6.5), regimes (section 6.6) and the benchmark comparison (section 4) all
need the same panel, so it lives at package root rather than being duplicated
inside three subpackages.

Every division here is guarded. ``NullFloat`` rejects NaN and infinity at the
contract boundary, so an unguarded zero-vol or zero-drawdown case would not
produce a misleading number -- it would raise. The guards choose the conservative
value explicitly instead.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from null.contracts import MetricBasis, PerfMetrics, Series

__all__ = ["TRADING_DAYS", "compute_metrics", "max_drawdown", "longest_underwater"]

#: Annualisation factor for daily Indian equity data. Not a rate or a gate
#: threshold, so it stays here rather than in configs/ -- but it is a parameter on
#: every function below, because a weekly or monthly series needs a different one.
TRADING_DAYS = 252


def _as_array(returns: Series) -> npt.NDArray[np.float64]:
    return returns.to_numpy()


def max_drawdown(equity: npt.NDArray[np.float64]) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    # A non-positive peak means the account is wiped out; report a full drawdown
    # rather than dividing by zero and producing an infinity the contract rejects.
    if np.any(peak <= 0.0):
        return 1.0
    return float(np.max((peak - equity) / peak))


def longest_underwater(equity: npt.NDArray[np.float64]) -> int:
    """Longest run of bars spent below a previous peak."""
    if equity.size == 0:
        return 0
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    longest = 0
    current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _annualised_vol(r: npt.NDArray[np.float64], periods: int) -> float:
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def _cagr(equity: npt.NDArray[np.float64], n_obs: int, periods: int) -> float:
    if n_obs < 2 or equity.size == 0:
        return 0.0
    final = float(equity[-1])
    if final <= 0.0:
        return -1.0  # total loss; a negative base under a fractional power is not real
    years = n_obs / periods
    if years <= 0.0:
        return 0.0
    return float(final ** (1.0 / years) - 1.0)


def compute_metrics(
    returns: Series,
    *,
    basis: MetricBasis,
    turnover_annual: float,
    time_in_market: float,
    periods_per_year: int = TRADING_DAYS,
) -> PerfMetrics:
    """The full section 4 panel for one return series.

    ``basis`` is required and not inferred. Whether a series is gross or net is a
    fact about how it was produced, and guessing it is how a cost drag gets
    reported as alpha.
    """
    r = _as_array(returns)
    n_obs = int(r.size)
    equity: npt.NDArray[np.float64] = (
        np.cumprod(1.0 + r, dtype=np.float64)
        if n_obs
        else np.zeros(0, dtype=np.float64)
    )

    vol = _annualised_vol(r, periods_per_year)
    cagr = _cagr(equity, n_obs, periods_per_year)
    mean_daily = float(np.mean(r)) if n_obs else 0.0

    sharpe = float(mean_daily / np.std(r, ddof=1) * np.sqrt(periods_per_year)) if (
        n_obs >= 2 and np.std(r, ddof=1) > 0.0
    ) else 0.0

    downside = r[r < 0.0]
    downside_dev = float(np.std(downside, ddof=1)) if downside.size >= 2 else 0.0
    sortino = (
        float(mean_daily / downside_dev * np.sqrt(periods_per_year))
        if downside_dev > 0.0
        else 0.0
    )

    mdd = max_drawdown(equity)
    calmar = cagr / mdd if mdd > 0.0 else 0.0

    wins = r[r > 0.0]
    losses = r[r < 0.0]
    hit_rate = float(wins.size / n_obs) if n_obs else 0.0
    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0

    # Tail ratio: 95th percentile gain over the magnitude of the 5th percentile
    # loss. Above 1 means the right tail is fatter than the left.
    if n_obs >= 20:
        p95 = float(np.percentile(r, 95))
        p05 = abs(float(np.percentile(r, 5)))
        tail_ratio = p95 / p05 if p05 > 0.0 and p95 > 0.0 else 0.0
    else:
        tail_ratio = 0.0

    worst_5 = tuple(float(x) for x in np.sort(r)[:5]) if n_obs else ()

    return PerfMetrics(
        basis=basis,
        cagr=cagr,
        vol_annual=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        longest_underwater_days=longest_underwater(equity),
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        turnover_annual=turnover_annual,
        time_in_market=time_in_market,
        tail_ratio=tail_ratio,
        worst_5_days=worst_5,
        n_obs=n_obs,
    )
