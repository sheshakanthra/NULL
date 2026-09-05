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

from null.contracts import Bar, StrategyRun, TargetWeight

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
    expected_verdict: str = "REJECT"
    caught_by: str | None = None
    """Which gate must catch it. A fixture that rejects for the wrong reason is a
    broken test that looks green, so the suite asserts on this, not on the verdict."""
    fatal_leakage: bool = False
    adv_participation: float | None = 0.002
    fold_returns: tuple[float, ...] = (0.02, 0.015, -0.005, 0.018, 0.011)
    neighborhood_ratio: float = 0.78
    benchmark: npt.NDArray[np.float64] | None = None
    """The series this strategy is judged against. None means the harness supplies
    an independent one. benchmark_clone MUST set it to the series it holds -- a
    clone compared against some other index is not a clone, and the fixture would
    then be testing nothing."""
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
        caught_by="deflated_sharpe",
        neighborhood_ratio=0.23,
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
        caught_by="deflated_sharpe",
        fold_returns=(-0.01, 0.008, -0.006, 0.004, -0.011),
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
        expected_verdict="PASS",
        caught_by=None,
    )


def benchmark_clone(seed: int = 20260908, n_obs: int = TEN_YEARS) -> SyntheticStrategy:
    """Holds the index. No edge over the thing it holds, by construction.

    Caught by beats_benchmark_net: alpha is zero and its t-stat cannot clear 2.
    """
    rng = np.random.default_rng(seed)
    index = rng.normal(0.0004, 0.010, n_obs)
    return SyntheticStrategy(
        name="benchmark_clone",
        returns=index,
        n_trials=1,
        trial_returns=None,
        caught_by="beats_benchmark_net",
        benchmark=index.copy(),
        fold_returns=(0.02, 0.015, -0.005, 0.018, 0.011),
    )


def costed_scalper(seed: int = 20260909, n_obs: int = TEN_YEARS) -> SyntheticStrategy:
    """+0.4%/trade gross, 8 trades a day, eaten alive by costs.

    Gross it looks superb. Net of a realistic round-trip charge at this frequency
    it is deeply negative, which is the entire point of the fixture: the cost model
    is what kills it, not the signal.
    """
    rng = np.random.default_rng(seed)
    gross_per_trade, trades_per_day = 0.004, 8
    # Round-trip cost as a fraction of notional, from the M1 model at small size.
    cost_per_trade = 0.0041
    daily = rng.normal(
        (gross_per_trade - cost_per_trade) * trades_per_day, 0.012, n_obs
    )
    return SyntheticStrategy(
        name="costed_scalper",
        returns=daily,
        n_trials=1,
        trial_returns=None,
        caught_by="beats_benchmark_net",
        fold_returns=(-0.02, -0.03, -0.01, -0.04, -0.02),
    )


def one_regime_wonder(seed: int = 20260910, n_obs: int = TEN_YEARS) -> SyntheticStrategy:
    """All the PnL arrives in one window. Flat to negative everywhere else.

    Caught by walkforward_consistency: a strategy carried by one fold is one lucky
    regime, not an edge.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(-0.00004, 0.008, n_obs)
    burst = slice(int(n_obs * 0.42), int(n_obs * 0.46))
    returns[burst] += 0.02
    return SyntheticStrategy(
        name="one_regime_wonder",
        returns=returns,
        n_trials=1,
        trial_returns=None,
        caught_by="walkforward_consistency",
        fold_returns=(-0.01, -0.008, 0.35, -0.012, -0.009),
    )


def capacity_bomb(seed: int = 20260911, n_obs: int = TEN_YEARS) -> SyntheticStrategy:
    """A real edge that cannot be executed: 40% of ADV per order.

    Caught by capacity. The returns are genuinely good, which is what makes this
    fixture worth having -- it proves the capacity gate is not simply riding along
    behind gates that were going to reject anyway.
    """
    rng = np.random.default_rng(seed)
    vol = 0.010
    raw = rng.normal(0.0, 1.0, n_obs)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    returns = raw * vol + 1.1 * vol / np.sqrt(TRADING_DAYS)
    return SyntheticStrategy(
        name="capacity_bomb",
        returns=returns,
        n_trials=1,
        trial_returns=None,
        caught_by="capacity",
        adv_participation=0.40,
    )


def oracle_lookahead(seed: int = 20260912, n_obs: int = 500) -> SyntheticStrategy:
    """Buys at t using the close of t+1. Perfect foresight.

    Caught by leakage_clean, which short-circuits before any statistic is computed.
    The returns here are absurd on purpose; nothing downstream should ever see them.
    """
    rng = np.random.default_rng(seed)
    noise = _gbm(n_obs, rng)
    return SyntheticStrategy(
        name="oracle_lookahead",
        returns=np.abs(noise),  # always on the right side: impossible
        n_trials=1,
        trial_returns=None,
        caught_by="leakage_clean",
        fatal_leakage=True,
    )


# ---------------------------------------------------------------------------
# bars-based oracle, used by the leakage audit
# ---------------------------------------------------------------------------
#
# The returns-based oracle_lookahead above feeds the gate suite. This one builds a
# StrategyRun over real bars, because the leakage detector reasons about weights
# against bar timestamps and cannot see a bare return series. Both live here so the
# fixtures are in one place.

ORACLE_SYMBOL = "ORACLE"


def oracle_lookahead_run(bars: tuple[Bar, ...]) -> StrategyRun:
    """Buys at t using the close of t+1. The fixture from BUILD.md section 8.

    It declares ``decision_lag_bars=1``, so it looks honest at the contract level.
    The only evidence of cheating is that its weights predict the future perfectly,
    which is what the audit has to notice.
    """
    weights: list[TargetWeight] = []
    for i in range(len(bars) - 1):
        forward_up = bars[i + 1].close > bars[i].close
        weights.append(
            TargetWeight(ts=bars[i].ts, symbol=ORACLE_SYMBOL, weight=1.0 if forward_up else 0.0)
        )
    return StrategyRun(
        strategy_id="oracle_lookahead",
        param_hash="oracle",
        n_trials=1,
        universe=(ORACLE_SYMBOL,),
        weights=tuple(weights),
        decision_lag_bars=1,
        initial_capital=1_000_000.0,
    )


