"""De-lever the benchmark, then regress. BUILD.md section 4, rule 4.

    If the strategy sits 45% in cash, it is not comparable to a fully-invested
    index. Two adjustments, report both: de-lever the benchmark to the strategy's
    realised vol, and regress strategy excess returns on benchmark excess returns;
    report alpha, beta, and the t-stat of alpha. Alpha with a t-stat below 2 is
    not alpha.

The regression defaults to **Newey-West** standard errors, not OLS. Daily strategy
returns are autocorrelated -- momentum and mean-reversion both induce it -- and
under autocorrelation OLS standard errors understate the true SE, which inflates
the t-stat. The gate is ``alpha_tstat >= 2.0``, so that inflation converts
rejections into passes. ``RegressionResult.se_method`` records which estimator ran
so the artifact can be audited (contract spec 0.2.0).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from null.contracts import RegressionResult, SEMethod, Series
from null.metrics import TRADING_DAYS

__all__ = ["newey_west_lags", "regress_excess_returns", "risk_match_benchmark"]


def newey_west_lags(n_obs: int) -> int:
    """Newey-West (1994) automatic lag truncation: floor(4 * (T/100)^(2/9))."""
    if n_obs <= 1:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def risk_match_benchmark(
    strategy_returns: Series,
    benchmark_returns: Series,
    *,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[Series, float]:
    """Scale the benchmark to the strategy's realised vol.

    Returns the de-levered series and the scaling factor applied. A strategy
    sitting half in cash has half the vol of a fully-invested index, and comparing
    the two raw flatters whichever is less volatile. The remainder is treated as
    earning nothing, which is conservative in the direction that matters: it does
    not invent a cash return the strategy never earned.
    """
    s = strategy_returns.to_numpy()
    b = benchmark_returns.to_numpy()
    if s.size < 2 or b.size < 2:
        return benchmark_returns, 1.0

    s_vol = float(np.std(s, ddof=1))
    b_vol = float(np.std(b, ddof=1))
    if b_vol <= 0.0:
        return benchmark_returns, 1.0

    scale = s_vol / b_vol
    return (
        Series(ts=benchmark_returns.ts, values=tuple(float(x) for x in b * scale)),
        scale,
    )


def _hac_variance(
    x: npt.NDArray[np.float64], resid: npt.NDArray[np.float64], lags: int
) -> npt.NDArray[np.float64]:
    """Newey-West HAC meat matrix with Bartlett weights."""
    n = x.shape[0]
    u = x * resid[:, None]
    s: npt.NDArray[np.float64] = np.asarray(u.T @ u, dtype=np.float64)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = np.asarray(u[lag:].T @ u[:-lag], dtype=np.float64)
        s = s + w * (gamma + gamma.T)
    return np.asarray(s / n, dtype=np.float64)


def regress_excess_returns(
    strategy_returns: Series,
    benchmark_returns: Series,
    *,
    risk_free_per_period: float = 0.0,
    se_method: SEMethod = "newey_west",
    periods_per_year: int = TRADING_DAYS,
) -> RegressionResult:
    """Regress strategy excess returns on benchmark excess returns.

    ``risk_free_per_period`` defaults to zero and that is a **stated assumption,
    not a finding**. NULL has no risk-free series yet; sourcing one is an open
    data question alongside the TRI. With both sides of the regression reduced by
    the same constant, beta is unaffected and alpha shifts by (1 - beta) * rf,
    which is small when beta is near 1 -- but it is not zero, and any report built
    on this must say so.
    """
    s = strategy_returns.to_numpy() - risk_free_per_period
    b = benchmark_returns.to_numpy() - risk_free_per_period
    n = int(min(s.size, b.size))
    if n < 3:
        raise ValueError(
            f"need at least 3 aligned observations to regress, got {n}. A regression "
            "on fewer points reports an alpha that means nothing."
        )
    s, b = s[:n], b[:n]

    x = np.column_stack([np.ones(n), b])
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta_hat = xtx_inv @ x.T @ s
    resid = s - x @ beta_hat

    if se_method == "ols":
        dof = n - 2
        sigma2 = float(resid @ resid) / dof if dof > 0 else 0.0
        cov = sigma2 * xtx_inv
        lags: int | None = None
    else:
        lags = newey_west_lags(n)
        meat = _hac_variance(x, resid, lags)
        bread = xtx_inv * n
        cov = bread @ meat @ bread / n

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    alpha_se, beta_se = float(se[0]), float(se[1])

    alpha_per_period = float(beta_hat[0])
    beta = float(beta_hat[1])

    ss_res = float(resid @ resid)
    ss_tot = float(((s - s.mean()) ** 2).sum())

    # Degenerate fit. A strategy that tracks the benchmark exactly -- benchmark_clone
    # is precisely this -- leaves residuals at the floating-point floor, so both the
    # alpha estimate and its standard error collapse toward zero and the t-stat
    # becomes tiny/tiny: numerically meaningless, and its SIGN is noise. Left
    # unguarded, an exact clone produces |t| of several units at random, and half
    # the time that is +4.6 and PASSES the gate. There is no alpha to detect here,
    # so report no evidence and let default REJECT do its job.
    degenerate = ss_tot > 0.0 and ss_res <= 1e-18 * ss_tot
    if degenerate:
        alpha_tstat = 0.0
        beta_tstat = 0.0
    else:
        alpha_tstat = alpha_per_period / alpha_se if alpha_se > 0.0 else 0.0
        beta_tstat = beta / beta_se if beta_se > 0.0 else 0.0
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    r_squared = min(max(r_squared, 0.0), 1.0)

    return RegressionResult(
        alpha_annual=alpha_per_period * periods_per_year,
        alpha_stderr=alpha_se * periods_per_year,
        alpha_tstat=alpha_tstat,
        se_method=se_method,
        hac_lags=lags,
        beta=beta,
        beta_tstat=beta_tstat,
        r_squared=r_squared,
        n_obs=n,
    )
