"""RSI(2) mean-reversion on NIFTY 50 constituents. BUILD.md M7.

This is the STRATEGY BEING AUDITED, not part of NULL's engine. It lives under
examples/, not null/, for the same reason BUILD.md's anti-goals list "signal
generation, strategy discovery" as out of scope for null/: NULL judges, it does
not propose. A real user's strategy code would look like this file and would
never live inside the package that judges it.

Long-only, daily, one RSI variant per grid point:

    period in {2, 3, 4}            RSI(period)
    entry  in {5, 10, 15}          enter when RSI(period) < entry
    exit   in {50, 60, 70}         exit when RSI(period) > exit, or holding_cap
    holding_cap in {3, 5, 10, 15}  forced exit after this many bars regardless

3 x 3 x 3 x 4 = 108 variants.

NOTE on holding_cap: BUILD.md's M7 section lists "holding cap {3,5,10}" -- three
values -- while also stating the grid has 108 variants. 3x3x3x3 = 81, not 108;
3x3x3x4 = 108 exactly. This module adopts a fourth holding-cap value, 15, as the
natural extension of the stated sequence, to make n_trials=108 literally true
rather than silently reporting 81 while claiming 108. This is a stated assumption,
not a silent correction, and it is flagged again in the session report.

Position sizing is likewise not specified beyond "long-only": each symbol gets an
equal weight of 1/N_universe while a position is open, 0 otherwise. No pyramiding
-- a symbol already long does not re-enter on a fresh signal.

RSI uses Wilder's smoothing (the classic definition, and what "RSI(2)" means in
the Connors mean-reversion literature this demo reproduces).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import numpy.typing as npt

from null.benchmark import buyhold as _bh
from null.contracts import (
    Bar,
    ParamPoint,
    SensitivityResult,
    Series,
    StrategyRun,
    TargetWeight,
)
from null.costs.india_equity import IndiaEquityCostModel
from null.costs.model import Segment, Side
from null.metrics import TRADING_DAYS

__all__ = [
    "RSI_PERIODS",
    "ENTRY_THRESHOLDS",
    "EXIT_THRESHOLDS",
    "HOLDING_CAPS",
    "GridVariant",
    "VariantResult",
    "ALL_VARIANTS",
    "compute_rsi",
    "generate_weights_for_symbol",
    "run_variant",
    "run_grid",
    "build_sensitivity",
]

RSI_PERIODS: tuple[int, ...] = (2, 3, 4)
ENTRY_THRESHOLDS: tuple[int, ...] = (5, 10, 15)
EXIT_THRESHOLDS: tuple[int, ...] = (50, 60, 70)
#: Fourth value (15) added beyond BUILD.md's literal {3,5,10} to make the grid
#: 3*3*3*4=108 as required. See module docstring.
HOLDING_CAPS: tuple[int, ...] = (3, 5, 10, 15)


def compute_rsi(closes: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """Wilder's RSI. NaN for bars before the smoothing window fills.

    ``closes`` is one symbol's close prices in date order. The returned array is
    the same length; ``result[i]`` is the RSI computed using ``closes[0..i]``, so
    it is knowable at bar ``i`` close -- consistent with the decision-lag contract
    that a signal computed at bar ``t`` close may not fill before bar ``t+1``.
    """
    n = closes.size
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return rsi

    delta = np.diff(closes)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)

    avg_gain = float(np.mean(gain[:period]))
    avg_loss = float(np.mean(loss[:period]))
    rsi[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period, gain.size):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rsi[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    return rsi


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0  # no movement at all; conventionally neutral
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@dataclass(frozen=True)
class GridVariant:
    period: int
    entry: int
    exit: int
    holding_cap: int

    @property
    def param_hash(self) -> str:
        raw = f"rsi2:{self.period}:{self.entry}:{self.exit}:{self.holding_cap}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def offsets_from(self) -> dict[str, int]:
        """Index of each parameter within its own grid list. Used for sensitivity."""
        return {
            "period": RSI_PERIODS.index(self.period),
            "entry": ENTRY_THRESHOLDS.index(self.entry),
            "exit": EXIT_THRESHOLDS.index(self.exit),
            "holding_cap": HOLDING_CAPS.index(self.holding_cap),
        }


def _build_grid() -> tuple[GridVariant, ...]:
    return tuple(
        GridVariant(period=p, entry=e, exit=x, holding_cap=h)
        for p, e, x, h in product(RSI_PERIODS, ENTRY_THRESHOLDS, EXIT_THRESHOLDS, HOLDING_CAPS)
    )


ALL_VARIANTS: tuple[GridVariant, ...] = _build_grid()


def generate_weights_for_symbol(
    bars: Sequence[Bar],
    rsi: npt.NDArray[np.float64],
    *,
    entry: int,
    exit: int,
    holding_cap: int,
    weight_when_long: float,
) -> list[TargetWeight]:
    """Change-point weight schedule for one symbol: a TargetWeight only when the
    position actually changes, not one row per day.

    Long-only, no pyramiding: while already in a position, a fresh entry signal is
    ignored. Exit fires on whichever comes first, RSI crossing the exit threshold
    or the holding cap being reached.
    """
    out: list[TargetWeight] = []
    in_position = False
    bars_held = 0

    for i, bar in enumerate(bars):
        value = rsi[i]
        if np.isnan(value):
            continue

        if not in_position:
            if value < entry:
                out.append(TargetWeight(ts=bar.ts, symbol=bar.symbol, weight=weight_when_long))
                in_position = True
                bars_held = 0
            continue

        bars_held += 1
        if value > exit or bars_held >= holding_cap:
            out.append(TargetWeight(ts=bar.ts, symbol=bar.symbol, weight=0.0))
            in_position = False

    return out


@dataclass(frozen=True)
class VariantResult:
    variant: GridVariant
    weights: tuple[TargetWeight, ...]
    gross_returns: Series
    net_returns: Series
    gross_sharpe: float
    net_sharpe: float
    n_position_changes: int


def _annualised_sharpe(returns: npt.NDArray[np.float64], periods: int = TRADING_DAYS) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(periods))


def run_variant(
    *,
    variant: GridVariant,
    bars: tuple[Bar, ...],
    universe: tuple[str, ...],
    rsi_by_symbol: dict[str, npt.NDArray[np.float64]],
    bars_by_symbol: dict[str, tuple[Bar, ...]],
    costs: IndiaEquityCostModel,
    initial_capital: float,
    segment: Segment = Segment.EQUITY_DELIVERY,
    sigma_daily: float = 0.018,
) -> VariantResult:
    """Backtest one grid point across the whole universe.

    Reuses null.benchmark.buyhold's private portfolio-aggregation helpers
    (_timeline, _panel, _returns_matrix, _weights_matrix) rather than re-deriving
    the same weight-to-return alignment a third time in this codebase. That
    alignment carried a real one-bar look-ahead bug earlier in this project's
    history, found only by an analytic multi-symbol test; re-implementing it here
    would risk reintroducing exactly that class of bug with no equivalent test
    covering this module.
    """
    weight_when_long = 1.0 / len(universe)

    weights: list[TargetWeight] = []
    for symbol in universe:
        weights.extend(
            generate_weights_for_symbol(
                bars_by_symbol[symbol],
                rsi_by_symbol[symbol],
                entry=variant.entry,
                exit=variant.exit,
                holding_cap=variant.holding_cap,
                weight_when_long=weight_when_long,
            )
        )
    weights.sort(key=lambda w: (w.ts, w.symbol))

    timeline = _bh._timeline(bars)
    prices, adv = _bh._panel(bars, timeline, universe)
    asset_returns = _bh._returns_matrix(prices)

    # A real StrategyRun, not a duck-typed stand-in: _weights_matrix's signature
    # expects one, and building a genuine (if internal-only) instance is both
    # type-clean and no more expensive than validating the weights would be
    # anyway. n_trials=1 here has no relation to the grid's own n_trials=108 --
    # this object exists only to drive one variant's return calculation.
    internal_run = StrategyRun(
        strategy_id="_rsi2_internal",
        param_hash=variant.param_hash,
        n_trials=1,
        universe=universe,
        weights=tuple(weights),
        decision_lag_bars=1,
        initial_capital=initial_capital,
    )
    weight_matrix = _bh._weights_matrix(internal_run, timeline, universe)

    gross = np.einsum("ij,ij->i", weight_matrix, asset_returns)

    equity = initial_capital
    cost_drag = np.zeros(gross.shape[0], dtype=np.float64)
    previous = np.zeros(len(universe), dtype=np.float64)
    for i in range(weight_matrix.shape[0]):
        row_price = prices[i + 1]
        row_adv = adv[i + 1]
        day_cost = 0.0
        for j, symbol in enumerate(universe):
            delta = abs(float(weight_matrix[i, j]) - float(previous[j]))
            if delta <= 0.0:
                continue
            price = float(row_price[j])
            if not np.isfinite(price) or price <= 0.0:
                continue
            traded = delta * equity
            symbol_adv = float(row_adv[j])
            has_adv = np.isfinite(symbol_adv) and symbol_adv > 0.0
            charge = costs.charge(
                symbol=symbol,
                side=Side.BUY if weight_matrix[i, j] > previous[j] else Side.SELL,
                quantity=traded / price,
                price=price,
                segment=segment,
                sigma_daily=sigma_daily,
                adv_20=symbol_adv if has_adv else 1.0,
            )
            day_cost += charge.total
        cost_drag[i] = day_cost / equity if equity > 0.0 else 0.0
        previous = weight_matrix[i].copy()

    net = gross - cost_drag
    return_ts = timeline[1:]

    gross_series = Series(ts=return_ts, values=tuple(float(x) for x in gross))
    net_series = Series(ts=return_ts, values=tuple(float(x) for x in net))

    return VariantResult(
        variant=variant,
        weights=tuple(weights),
        gross_returns=gross_series,
        net_returns=net_series,
        gross_sharpe=_annualised_sharpe(gross),
        net_sharpe=_annualised_sharpe(net),
        n_position_changes=len(weights),
    )


def run_grid(
    *,
    bars: tuple[Bar, ...],
    universe: tuple[str, ...],
    costs: IndiaEquityCostModel,
    initial_capital: float,
    variants: tuple[GridVariant, ...] = ALL_VARIANTS,
) -> tuple[VariantResult, ...]:
    """Run every grid variant. RSI is computed once per (symbol, period), not once
    per variant, since only 3 distinct periods exist across 108 variants."""
    bars_by_symbol_raw: dict[str, list[Bar]] = {}
    for bar in bars:
        bars_by_symbol_raw.setdefault(bar.symbol, []).append(bar)
    bars_by_symbol: dict[str, tuple[Bar, ...]] = {
        s: tuple(sorted(v, key=lambda b: b.ts)) for s, v in bars_by_symbol_raw.items()
    }

    closes_by_symbol = {
        s: np.asarray([b.close for b in v], dtype=np.float64)
        for s, v in bars_by_symbol.items()
    }

    rsi_cache: dict[tuple[str, int], npt.NDArray[np.float64]] = {}
    for symbol in universe:
        if symbol not in closes_by_symbol:
            continue
        for period in RSI_PERIODS:
            rsi_cache[(symbol, period)] = compute_rsi(closes_by_symbol[symbol], period)

    effective_universe = tuple(s for s in universe if s in bars_by_symbol)

    results = []
    for variant in variants:
        rsi_by_symbol = {
            symbol: rsi_cache[(symbol, variant.period)]
            for symbol in effective_universe
            if (symbol, variant.period) in rsi_cache
        }
        results.append(
            run_variant(
                variant=variant,
                bars=bars,
                universe=effective_universe,
                rsi_by_symbol=rsi_by_symbol,
                bars_by_symbol=bars_by_symbol,
                costs=costs,
                initial_capital=initial_capital,
            )
        )
    return tuple(results)


def build_sensitivity(
    results: tuple[VariantResult, ...], *, metric: str = "net_sharpe"
) -> SensitivityResult:
    """SensitivityResult built directly from the full grid, not via a stepped
    scan. The grid's four parameters have unequal level counts (3, 3, 3, 4), so a
    generic +/-1/+/-2 stepper risks index wraparound at a boundary -- the same
    family of silent bug this project has already found twice in weight
    alignment. Instead the immediate neighbourhood is defined directly as index
    Hamming-distance 1 from the best point: every grid point that differs from
    the best in exactly one parameter, by exactly one index step. Points outside
    the grid at a boundary simply do not exist and are not counted, rather than
    being clamped or wrapped.
    """
    best = max(results, key=lambda r: getattr(r, metric))
    best_idx = best.variant.offsets_from

    points: list[ParamPoint] = []
    neighbour_values: list[float] = []
    for result in results:
        idx = result.variant.offsets_from
        offsets = {name: idx[name] - best_idx[name] for name in best_idx}
        value = getattr(result, metric)
        points.append(
            ParamPoint(param_hash=result.variant.param_hash, offsets=offsets, sharpe=value)
        )
        nonzero = [name for name, off in offsets.items() if off != 0]
        if len(nonzero) == 1 and abs(offsets[nonzero[0]]) == 1:
            neighbour_values.append(value)

    peak = float(getattr(best, metric))
    mean_neighbour = float(np.mean(neighbour_values)) if neighbour_values else peak
    ratio = max(0.0, min(mean_neighbour / peak, 1.0)) if peak > 0.0 else 0.0

    return SensitivityResult(
        param_names=("period", "entry", "exit", "holding_cap"),
        peak_sharpe=peak,
        neighborhood_mean_sharpe=mean_neighbour,
        neighborhood_ratio=ratio,
        points=tuple(points),
    )
