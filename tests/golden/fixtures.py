"""Synthetic strategies with known correct verdicts. BUILD.md section 8.

Every fixture is seeded. No wall-clock, no network, no market data.

These exist to test the harness, not the market. The point of
``true_edge_synthetic`` in particular is that a harness which rejects everything
is exactly as useless as one that accepts everything -- so its parameters are
chosen from the spec (0.6 Sharpe, low turnover, one trial) and a defensible
sample length, and are **not** to be tuned until it passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

TRADING_DAYS = 252

# Ten years of daily data. Chosen before running anything, because it is what a
# "persistent edge" claim would realistically be backed by -- not because it makes
# a threshold work out.
TEN_YEARS = 10 * TRADING_DAYS


@dataclass(frozen=True)
class SyntheticStrategy:
    """A strategy's realised returns plus the trial history that produced it."""

    name: str
    returns: npt.NDArray[np.float64]
    n_trials: int
    trial_returns: npt.NDArray[np.float64] | None
    """(T, N) matrix of per-trial returns, or None when there was no search."""

    @property
    def trial_sharpes(self) -> npt.NDArray[np.float64] | None:
        if self.trial_returns is None:
            return None
        mu = self.trial_returns.mean(axis=0)
        sd = self.trial_returns.std(axis=0, ddof=1)
        return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)


def _gbm(n: int, rng: np.random.Generator, *, vol: float = 0.011) -> npt.NDArray[np.float64]:
    """Driftless geometric Brownian noise. No edge exists in this series."""
    return rng.normal(0.0, vol, n)


def _ma_crossover_returns(
    prices: npt.NDArray[np.float64],
    noise: npt.NDArray[np.float64],
    fast: int,
    slow: int,
) -> npt.NDArray[np.float64]:
    """Returns of a moving-average crossover, with the decision lag applied.

    Neighbouring parameter values produce correlated strategies, which is what
    makes a grid search over this genuinely overfittable -- unlike independent
    random strategies, whose in-sample winners are no more likely than chance to
    be out-of-sample losers.

    The shift matters and is the whole reason this helper is not two lines. A
    position computed from prices through bar ``t`` may only earn the return of
    bar ``t+1``. An earlier version of this fixture multiplied the position at
    ``t`` by the return that *produced* ``prices[t]``, which handed every variant
    perfect foresight: PBO came out at 0.0 with the in-sample winner ranking in
    the top 1% out-of-sample, because look-ahead is persistent rather than
    overfit. The fixture was wrong, not the gate.
    """
    n = prices.size
    pos = np.zeros(n, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(prices)])
    for t in range(slow, n):
        fast_ma = (cs[t + 1] - cs[t + 1 - fast]) / fast
        slow_ma = (cs[t + 1] - cs[t + 1 - slow]) / slow
        pos[t] = 1.0 if fast_ma > slow_ma else -1.0
    out = np.zeros(n, dtype=np.float64)
    out[1:] = pos[:-1] * noise[1:]  # decided on t, earns t+1
    return out


#: Two years. Short windows are what make a grid search overfit hard, and a
#: two-year backtest is entirely typical of the retail searches NULL exists to
#: judge. Ten years of noise leaves far less for a search to latch onto.
TWO_YEARS = 2 * TRADING_DAYS


def overfit_grid(seed: int = 20260905, n_obs: int = TWO_YEARS) -> SyntheticStrategy:
    """Best of a 5,000-variant grid search fit on GBM noise.

    Declares n_trials=5000 honestly. The underlying series has no edge whatsoever,
    so any Sharpe here is selection, and both the deflated Sharpe and PBO should
    say so independently.

    A subset of the trial return series is retained rather than all 5,000: the
    contract permits it (trials may be a subset, never exceeding n_trials), CSCV
    needs a matrix rather than a count, and 1,000 correlated variants is ample to
    measure how often the in-sample winner loses out of sample.

    The parameter range spans very fast to very slow deliberately, giving 1,000
    variants over a short window -- the shape of a real retail grid search.

    KNOWN P0 (see docs/pbo_calibration.md): this fixture's PBO verdict is
    SEED-DEPENDENT. Measured across seven seeds, PBO ran 0.21 to 0.77 and fell
    below the 0.50 gate on two of them, meaning overfit_grid would PASS the PBO
    gate on those draws. The deflated-Sharpe verdict is robust across all seven
    (0.046 to 0.79, always rejecting). Do not pick a seed that makes PBO pass --
    the instability is the finding.
    """
    rng = np.random.default_rng(seed)
    noise = _gbm(n_obs, rng)
    prices = 1000.0 * np.cumprod(1.0 + noise)

    grids = [(f, s) for f in range(2, 22) for s in range(22, 72)]  # 1,000 retained
    trials = np.column_stack(
        [_ma_crossover_returns(prices, noise, f, s) for f, s in grids]
    )
    sharpes = trials.mean(axis=0) / trials.std(axis=0, ddof=1)
    best = int(np.argmax(sharpes))

    return SyntheticStrategy(
        name="overfit_grid",
        returns=trials[:, best],
        n_trials=5_000,
        trial_returns=trials,
    )


def pure_noise(seed: int = 20260906, n_obs: int = TEN_YEARS) -> SyntheticStrategy:
    """Random seeded entries. No search, no edge, nothing to find."""
    rng = np.random.default_rng(seed)
    noise = _gbm(n_obs, rng)
    positions = rng.choice([-1.0, 1.0], size=n_obs)
    return SyntheticStrategy(
        name="pure_noise",
        returns=positions * noise,
        n_trials=1,
        trial_returns=None,
    )


def true_edge_synthetic(
    seed: int = 20260907, n_obs: int = TEN_YEARS, target_sharpe: float = 0.6
) -> SyntheticStrategy:
    """A persistent, genuine edge. Low turnover. One trial, declared honestly.

    THE CALIBRATION FIXTURE. If this cannot pass, the thresholds are wrong and
    real edges will be discarded. Do not tune it into passing -- if it fails,
    that is a finding about the gates, not about the fixture.

    The edge is a small constant drift added to an otherwise noiseless-mean
    series, which is the cleanest possible form of "persistent": it does not
    concentrate in one regime, and it does not depend on turnover.
    """
    rng = np.random.default_rng(seed)
    vol = 0.011
    # Standardise the draw, then impose the target exactly. Sampling from a
    # 0.6-Sharpe process is not the same as HAVING a 0.6 Sharpe: on this seed a
    # naive draw realised 0.2011 over ten years, about -1.25 standard errors, and
    # the fixture would then have been testing whether the gates reject a 0.2
    # Sharpe -- which they should. The fixture's contract is "a strategy with a
    # 0.6 Sharpe", so it is constructed to have one.
    raw = rng.normal(0.0, 1.0, n_obs)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    daily_edge = target_sharpe * vol / np.sqrt(TRADING_DAYS)
    returns = raw * vol + daily_edge
    return SyntheticStrategy(
        name="true_edge_synthetic",
        returns=returns,
        n_trials=1,
        trial_returns=None,
    )
