"""Buy-and-hold benchmark, and the comparison that kills most strategies.

BUILD.md section 4, the non-negotiable rules:

  1. The benchmark is NIFTY 50 **TRI**, not the price index. **NULL does not
     choose that series.** ``benchmark_check`` takes benchmark bars as a
     parameter and never fetches or selects them -- see null/benchmark/tri.py,
     which raises rather than falling back, because quietly defaulting to the
     price index would hand every strategy ~1.35%/yr of free alpha.
  2. The benchmark pays entry cost once, through the same ``CostModel``. Not zero.
  3. Same capital, same period, same currency.
  4. Risk-match before comparing (see risk_match.py).
  5. Report net_of_everything CAGR side by side with a one-line verdict sentence.

Portfolio accounting handles any number of symbols. The single-symbol case is this
same path with a one-column matrix, not a separate branch.

Three things about the aggregation that are easy to get quietly wrong:

  * **A symbol with no bar on a date contributes zero, never NaN**, and never
    shortens the series. Listings and delistings must not silently drop days for
    every other holding.
  * **Weights need not sum to one.** The remainder is cash and earns nothing. That
    is not an error state, it is how a long-only signal strategy behaves whenever
    it has no signal.
  * **Costs are charged per symbol, never on the portfolio aggregate.** The DP
    charge is flat per scrip per day on delivery sells, so fifty names rebalancing
    owe fifty fees. Costing the aggregate would undercount that fiftyfold and make
    a wide portfolio look cheap to run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt

from null.benchmark.risk_match import regress_excess_returns, risk_match_benchmark
from null.contracts import (
    Bar,
    GateResult,
    NonEmptyStr,
    NonNegativeFloat,
    NullFloat,
    NullModel,
    PerfMetrics,
    RegressionResult,
    Series,
    StrategyRun,
)
from null.costs.india_equity import IndiaEquityCostModel
from null.costs.model import Segment, Side
from null.metrics import TRADING_DAYS, compute_metrics

__all__ = ["BenchmarkEvidence", "benchmark_check"]

ALPHA_TSTAT_THRESHOLD = 2.0
GATE_NAME = "beats_benchmark_net"

#: The phrase BUILD.md section 4 requires the rationale to carry.
NO_ALPHA_PHRASE = "no alpha over buy-and-hold after costs"


class BenchmarkEvidence(NullModel):
    """Everything section 4 asks to be reported, plus the gate it drives."""

    strategy_returns: Series
    strategy_gross_returns: Series
    """Before costs. Kept so the aggregation can be checked analytically."""
    benchmark_returns: Series
    benchmark_returns_risk_matched: Series
    metrics: PerfMetrics
    benchmark_metrics: PerfMetrics
    alpha: RegressionResult
    risk_match_scale: NullFloat
    benchmark_entry_cost: NonNegativeFloat
    strategy_total_cost: NonNegativeFloat
    cost_breakdown: dict[str, NonNegativeFloat]
    """Per charge component, summed across symbols and days."""
    max_adv_participation: NonNegativeFloat
    """Worst single order across ALL symbols and days, never the mean. One illiquid
    name at 40% of ADV invalidates the whole run, and an average would hide it."""
    n_symbols: int
    rates_are_verified: bool
    limitations_text: NonEmptyStr
    gate: GateResult
    verdict_result: Literal["REJECT", "PASS"]

    @property
    def alpha_se_method(self) -> str:
        return self.alpha.se_method

    @property
    def metrics_basis(self) -> str:
        return self.metrics.basis

    @property
    def benchmark_metrics_basis(self) -> str:
        return self.benchmark_metrics.basis


def _bar_returns(bars: tuple[Bar, ...]) -> Series:
    """Close-to-close simple returns for a single-symbol series."""
    if len(bars) < 2:
        raise ValueError(f"need at least 2 bars to compute returns, got {len(bars)}")
    closes = np.asarray([b.close for b in bars], dtype=np.float64)
    if np.any(closes <= 0.0):
        raise ValueError("non-positive close price; returns would be undefined")
    rets = closes[1:] / closes[:-1] - 1.0
    return Series(
        ts=tuple(b.ts for b in bars[1:]), values=tuple(float(x) for x in rets)
    )


def _timeline(bars: tuple[Bar, ...]) -> tuple[datetime, ...]:
    """Every distinct bar timestamp, sorted. The portfolio's common calendar.

    The union rather than the intersection: a symbol that lists midway through, or
    delists, must not shorten the series for everything else.
    """
    return tuple(sorted({b.ts for b in bars}))


def _panel(
    bars: tuple[Bar, ...], timeline: tuple[datetime, ...], universe: tuple[str, ...]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Close-price and ADV matrices, shape (dates, symbols), NaN where absent."""
    row = {ts: i for i, ts in enumerate(timeline)}
    col = {sym: j for j, sym in enumerate(universe)}
    prices = np.full((len(timeline), len(universe)), np.nan, dtype=np.float64)
    adv = np.full((len(timeline), len(universe)), np.nan, dtype=np.float64)
    for bar in bars:
        j = col.get(bar.symbol)
        if j is None:
            continue
        prices[row[bar.ts], j] = bar.close
        if bar.adv_20 is not None:
            adv[row[bar.ts], j] = bar.adv_20
    return prices, adv


def _returns_matrix(prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Close-to-close returns, ZERO wherever either end is missing.

    Zero, not NaN. A symbol that is not trading contributes nothing; it must not
    poison the whole day's portfolio return.
    """
    prev, curr = prices[:-1], prices[1:]
    valid = np.isfinite(prev) & np.isfinite(curr) & (prev > 0.0)
    out = np.zeros_like(curr)
    np.divide(curr - prev, prev, out=out, where=valid)
    return out


def _weights_matrix(
    run: StrategyRun, timeline: tuple[datetime, ...], universe: tuple[str, ...]
) -> npt.NDArray[np.float64]:
    """Target weight per (return bar, symbol), after the decision lag.

    A weight decided on bar t applies from t + decision_lag_bars and persists until
    changed.
    """
    row = {ts: i for i, ts in enumerate(timeline)}
    col = {sym: j for j, sym in enumerate(universe)}
    held = np.zeros((len(timeline), len(universe)), dtype=np.float64)
    matched = 0
    for weight in run.weights:
        j = col.get(weight.symbol)
        i = row.get(weight.ts)
        if j is None or i is None:
            continue
        matched += 1
        start = i + run.decision_lag_bars
        if start < len(timeline):
            held[start:, j] = weight.weight

    if run.weights and matched == 0:
        # Every weight missed every bar. Left alone this produces an all-zero return
        # series, which the gates would then judge as "no edge" -- reporting a verdict
        # on a strategy that was never actually simulated. That is a broken input, and
        # it has to say so rather than be quietly audited.
        raise ValueError(
            f"none of the {len(run.weights):,} target weights line up with a bar "
            f"timestamp. First weight is at {run.weights[0].ts.isoformat()}; first bar "
            f"is at {timeline[0].isoformat()}. Weight timestamps must match bar CLOSE "
            "times exactly, including time of day and timezone."
        )

    # held[k] is the position in force AT bar k. The return from bar k to bar k+1
    # is earned by the position held at bar k, so the alignment is held[:-1].
    #
    # held[1:] would pair the weight in force at the END of a period with the
    # return earned DURING it -- a one-bar look-ahead. It survived from M2 to here
    # because every fixture that exercised this path held a constant weight, and a
    # constant weight is identical under both alignments. The analytic two-symbol
    # test is what distinguishes them.
    return held[:-1]


def benchmark_check(
    *,
    run: StrategyRun,
    bars: tuple[Bar, ...],
    benchmark_bars: tuple[Bar, ...],
    costs: IndiaEquityCostModel,
    segment: Segment = Segment.EQUITY_DELIVERY,
    sigma_daily: float = 0.018,
    risk_free_per_period: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> BenchmarkEvidence:
    """Compare a portfolio to buy-and-hold, net of everything, risk-matched."""
    universe = tuple(run.universe)
    timeline = _timeline(bars)
    if len(timeline) < 2:
        raise ValueError(f"need at least 2 bar dates, got {len(timeline)}")

    prices, adv = _panel(bars, timeline, universe)
    asset_returns = _returns_matrix(prices)
    weights = _weights_matrix(run, timeline, universe)

    # Portfolio gross return: the weighted sum across symbols. Whatever the weights
    # do not account for is cash, which earns zero and so is simply absent from
    # this sum rather than modelled as a position.
    gross = np.einsum("ij,ij->i", weights, asset_returns)

    # --- costs, charged PER SYMBOL -------------------------------------------
    equity = run.initial_capital
    cost_drag = np.zeros(gross.shape[0], dtype=np.float64)
    breakdown: dict[str, float] = {}
    total_cost = 0.0
    worst_participation = 0.0
    previous = np.zeros(len(universe), dtype=np.float64)

    for i in range(weights.shape[0]):
        row_price = prices[i + 1]
        row_adv = adv[i + 1]
        day_cost = 0.0
        for j, symbol in enumerate(universe):
            delta = abs(float(weights[i, j]) - float(previous[j]))
            if delta <= 0.0:
                continue
            price = float(row_price[j])
            if not np.isfinite(price) or price <= 0.0:
                continue
            traded = delta * equity
            symbol_adv = float(row_adv[j])
            has_adv = np.isfinite(symbol_adv) and symbol_adv > 0.0
            if has_adv:
                worst_participation = max(worst_participation, traded / symbol_adv)
            charge = costs.charge(
                symbol=symbol,
                side=Side.BUY if weights[i, j] > previous[j] else Side.SELL,
                quantity=traded / price,
                price=price,
                segment=segment,
                sigma_daily=sigma_daily,
                adv_20=symbol_adv if has_adv else 1.0,
            )
            day_cost += charge.total
            for component, value in charge.as_dict().items():
                breakdown[component] = breakdown.get(component, 0.0) + value
        cost_drag[i] = day_cost / equity if equity > 0.0 else 0.0
        total_cost += day_cost
        previous = weights[i].copy()

    net = gross - cost_drag
    return_ts = timeline[1:]
    strategy_returns = Series(ts=return_ts, values=tuple(float(x) for x in net))
    strategy_gross = Series(ts=return_ts, values=tuple(float(x) for x in gross))

    # --- benchmark pays its entry cost once (rule 2) --------------------------
    bench_gross = _bar_returns(benchmark_bars)
    bench_price = float(benchmark_bars[0].close)
    # An index level series has no traded volume. Rather than invent one -- which
    # would charge a vast square-root impact against a fictional ADV -- the entry is
    # sized against a notional far above the order so the impact term vanishes and
    # only the spread and statutory charges apply.
    bench_adv = benchmark_bars[0].adv_20 or (run.initial_capital * 1e6)
    bench_entry = costs.charge(
        symbol=benchmark_bars[0].symbol,
        side=Side.BUY,
        quantity=run.initial_capital / bench_price if bench_price > 0.0 else 0.0,
        price=bench_price,
        segment=segment,
        sigma_daily=sigma_daily,
        adv_20=float(bench_adv),
    )
    bench_net = bench_gross.to_numpy().copy()
    if bench_net.size and run.initial_capital > 0.0:
        bench_net[0] -= bench_entry.total / run.initial_capital
    benchmark_returns = Series(
        ts=bench_gross.ts, values=tuple(float(x) for x in bench_net)
    )

    # --- risk matching and regression, unchanged from M2 ----------------------
    matched, scale = risk_match_benchmark(
        strategy_returns, benchmark_returns, periods_per_year=periods_per_year
    )
    alpha = regress_excess_returns(
        strategy_returns,
        matched,
        risk_free_per_period=risk_free_per_period,
        periods_per_year=periods_per_year,
    )

    turnover = float(np.abs(np.diff(weights, axis=0, prepend=0.0)).sum())
    years = max(weights.shape[0] / periods_per_year, 1e-9)
    invested = np.abs(weights).sum(axis=1)
    metrics = compute_metrics(
        strategy_returns,
        basis="net",
        turnover_annual=turnover / years,
        time_in_market=float(np.mean(invested > 0.0)) if invested.size else 0.0,
        periods_per_year=periods_per_year,
    )
    benchmark_metrics = compute_metrics(
        benchmark_returns,
        basis="net",
        turnover_annual=0.0,
        time_in_market=1.0,
        periods_per_year=periods_per_year,
    )

    limitations = _limitations(costs)
    gate = _build_gate(metrics, benchmark_metrics, alpha, limitations)

    return BenchmarkEvidence(
        strategy_returns=strategy_returns,
        strategy_gross_returns=strategy_gross,
        benchmark_returns=benchmark_returns,
        benchmark_returns_risk_matched=matched,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        alpha=alpha,
        risk_match_scale=scale,
        benchmark_entry_cost=bench_entry.total,
        strategy_total_cost=total_cost,
        cost_breakdown=breakdown,
        max_adv_participation=worst_participation,
        n_symbols=len(universe),
        rates_are_verified=costs.config.rates_are_verified,
        limitations_text=limitations,
        gate=gate,
        verdict_result="PASS" if gate.passed else "REJECT",
    )


def _limitations(costs: IndiaEquityCostModel) -> str:
    """Every stated limitation, printed on the report. Never silently omitted."""
    lines = []
    if not costs.config.rates_are_verified:
        lines.append(
            f"Charge rates are UNVERIFIED (_verified_on: {costs.config.verified_on!r}); "
            "they have not been reconciled against a live broker charge list, so every "
            "cost figure here is indicative only."
        )
    lines.append(
        "The benchmark series is supplied by the caller. NULL does not select it. "
        "If it is a price index rather than a total-return index, roughly 1.35%/yr "
        "of dividends are missing and the alpha below is overstated by that much: "
        "NSE reports 11.09% annualised for the NIFTY 50 price index against 12.44% "
        "for total return over the 20 years to February 2026."
    )
    lines.append(
        "Risk-free rate assumed to be zero; NULL has no risk-free series. Beta is "
        "unaffected; alpha is shifted by (1 - beta) * rf."
    )
    return " ".join(lines)


def _build_gate(
    metrics: PerfMetrics,
    benchmark_metrics: PerfMetrics,
    alpha: RegressionResult,
    limitations: str,
) -> GateResult:
    """The rationale is the product. Name the number, the threshold, and why."""
    passed = alpha.alpha_tstat >= ALPHA_TSTAT_THRESHOLD
    se_note = (
        f"Newey-West standard errors, {alpha.hac_lags} lags"
        if alpha.se_method == "newey_west"
        else "OLS standard errors (not autocorrelation-robust)"
    )

    if passed:
        rationale = (
            f"Strategy returned {metrics.cagr:.2%} CAGR net of everything against "
            f"{benchmark_metrics.cagr:.2%} for risk-matched buy-and-hold. Alpha of "
            f"{alpha.alpha_annual:.2%}/yr carries a t-stat of {alpha.alpha_tstat:.2f} "
            f"({se_note}) over {alpha.n_obs} observations, at or above the threshold "
            f"of {ALPHA_TSTAT_THRESHOLD:.1f}. Beta {alpha.beta:.2f}. "
            f"LIMITATIONS: {limitations}"
        )
    else:
        rationale = (
            f"{NO_ALPHA_PHRASE.capitalize()}. Strategy returned {metrics.cagr:.2%} "
            f"CAGR net of everything against {benchmark_metrics.cagr:.2%} for "
            f"risk-matched buy-and-hold, with beta {alpha.beta:.2f}. Annualised alpha "
            f"is {alpha.alpha_annual:.2%} with a t-stat of {alpha.alpha_tstat:.2f} "
            f"({se_note}) over {alpha.n_obs} observations; the threshold is "
            f"{ALPHA_TSTAT_THRESHOLD:.1f}, and alpha with a t-stat below 2 is not "
            f"alpha. LIMITATIONS: {limitations}"
        )

    return GateResult(
        name=GATE_NAME,
        state="PASS" if passed else "FAIL",
        passed=passed,
        observed=alpha.alpha_tstat,
        threshold=ALPHA_TSTAT_THRESHOLD,
        rationale=rationale,
    )
