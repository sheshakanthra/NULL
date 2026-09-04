"""Buy-and-hold benchmark, and the comparison that kills most strategies.

BUILD.md section 4, the non-negotiable rules:

  1. The benchmark is NIFTY 50 **TRI**, not the price index. **NULL does not
     choose that series.** ``benchmark_check`` takes benchmark bars as a
     parameter and never fetches or selects them -- the source is an unresolved
     decision recorded in docs/data_sources.md, and quietly defaulting to the
     price index would hand every strategy ~1.3%/yr of free alpha, which is the
     bias this module exists to remove.
  2. The benchmark pays entry cost once, through the same ``CostModel``. Not zero.
  3. Same capital, same period, same currency.
  4. Risk-match before comparing (see risk_match.py).
  5. Report net_of_everything CAGR side by side with a one-line verdict sentence.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

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
    benchmark_returns: Series
    benchmark_returns_risk_matched: Series
    metrics: PerfMetrics
    benchmark_metrics: PerfMetrics
    alpha: RegressionResult
    risk_match_scale: NullFloat
    benchmark_entry_cost: NonNegativeFloat
    strategy_total_cost: NonNegativeFloat
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
    """Close-to-close simple returns. First bar has no prior close, so it is dropped."""
    if len(bars) < 2:
        raise ValueError(f"need at least 2 bars to compute returns, got {len(bars)}")
    closes = np.asarray([b.close for b in bars], dtype=np.float64)
    if np.any(closes <= 0.0):
        raise ValueError("non-positive close price; returns would be undefined")
    rets = closes[1:] / closes[:-1] - 1.0
    return Series(
        ts=tuple(b.ts for b in bars[1:]),
        values=tuple(float(x) for x in rets),
    )


def _weight_series(run: StrategyRun, bars: tuple[Bar, ...], symbol: str) -> np.ndarray:
    """Target weight in force for each return bar, after applying decision lag.

    A weight decided on bar t may not act on the return of bar t (that is the
    look-ahead M3 audits for). It applies from bar t + decision_lag_bars.
    """
    ts_index = {b.ts: i for i, b in enumerate(bars)}
    held = np.zeros(len(bars), dtype=np.float64)
    for w in run.weights:
        if w.symbol != symbol:
            continue
        i = ts_index.get(w.ts)
        if i is None:
            continue
        start = i + run.decision_lag_bars
        if start < len(bars):
            held[start:] = w.weight
    # Return series drops the first bar, so weights align to bars[1:].
    return held[1:]


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
    """Compare a strategy to buy-and-hold, net of everything, risk-matched.

    ``benchmark_bars`` is a parameter and is never sourced here. See the module
    docstring.
    """
    if len(run.universe) != 1:
        raise NotImplementedError(
            f"M2 compares single-symbol runs; got {len(run.universe)} symbols. "
            "Multi-symbol portfolio accounting arrives with the full audit pipeline."
        )
    symbol = run.universe[0]

    asset = _bar_returns(bars)
    bench_gross = _bar_returns(benchmark_bars)
    weights = _weight_series(run, bars, symbol)

    gross = asset.to_numpy() * weights

    # --- costs ---------------------------------------------------------------
    # Turnover charged whenever the target weight moves. Entry is the first move,
    # from flat, and it is charged like any other.
    adv = np.mean([b.adv_20 for b in bars if b.adv_20 is not None] or [0.0])
    price = float(bars[0].close)
    equity = run.initial_capital

    prev_w = 0.0
    cost_drag = np.zeros_like(gross)
    total_cost = 0.0
    for i, w in enumerate(weights):
        delta = abs(float(w) - prev_w)
        if delta > 0.0:
            traded = delta * equity
            qty = traded / price if price > 0.0 else 0.0
            charge = costs.charge(
                symbol=symbol,
                side=Side.BUY if float(w) > prev_w else Side.SELL,
                quantity=qty,
                price=price,
                segment=segment,
                sigma_daily=sigma_daily,
                adv_20=float(adv),
            )
            cost_drag[i] = charge.total / equity if equity > 0.0 else 0.0
            total_cost += charge.total
        prev_w = float(w)

    net = gross - cost_drag
    strategy_returns = Series(ts=asset.ts, values=tuple(float(x) for x in net))

    # --- benchmark pays its entry cost once (rule 2) --------------------------
    bench_entry = costs.charge(
        symbol=symbol,
        side=Side.BUY,
        quantity=run.initial_capital / price if price > 0.0 else 0.0,
        price=price,
        segment=segment,
        sigma_daily=sigma_daily,
        adv_20=float(adv),
    )
    bench_net = bench_gross.to_numpy().copy()
    if bench_net.size and run.initial_capital > 0.0:
        bench_net[0] -= bench_entry.total / run.initial_capital
    benchmark_returns = Series(
        ts=bench_gross.ts, values=tuple(float(x) for x in bench_net)
    )

    # --- risk matching and regression (rule 4) --------------------------------
    matched, scale = risk_match_benchmark(
        strategy_returns, benchmark_returns, periods_per_year=periods_per_year
    )
    alpha = regress_excess_returns(
        strategy_returns,
        matched,
        risk_free_per_period=risk_free_per_period,
        periods_per_year=periods_per_year,
    )

    turnover = float(np.abs(np.diff(np.concatenate([[0.0], weights]))).sum())
    years = max(len(weights) / periods_per_year, 1e-9)
    metrics = compute_metrics(
        strategy_returns,
        basis="net",
        turnover_annual=turnover / years,
        time_in_market=float(np.mean(weights != 0.0)) if weights.size else 0.0,
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
        benchmark_returns=benchmark_returns,
        benchmark_returns_risk_matched=matched,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        alpha=alpha,
        risk_match_scale=scale,
        benchmark_entry_cost=bench_entry.total,
        strategy_total_cost=total_cost,
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
        "If it is a price index rather than a total-return index, roughly 1.2-1.5%/yr "
        "of dividends are missing and the alpha below is overstated by that much."
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
        passed=passed,
        observed=alpha.alpha_tstat,
        threshold=ALPHA_TSTAT_THRESHOLD,
        rationale=rationale,
    )
