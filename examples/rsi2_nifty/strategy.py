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

NOTE on holding_cap: BUILD.md originally listed "holding cap {3,5,10}" -- three
values -- while also stating the grid has 108 variants. 3x3x3x3 = 81, not 108;
3x3x3x4 = 108 exactly. The fourth value, 15, is now part of the spec: see BUILD.md
section 9, "Spec correction -- the fourth holding-cap value", ratified rather than
assumed. n_trials=108 is therefore literally true, which is what CLAUDE.md
invariant 7 and the deflated-Sharpe gate both require of it.

Position sizing is likewise not specified beyond "long-only": each symbol gets an
equal weight of 1/N_universe while a position is open, 0 otherwise. No pyramiding
-- a symbol already long does not re-enter on a fresh signal.

RSI uses Wilder's smoothing (the classic definition, and what "RSI(2)" means in
the Connors mean-reversion literature this demo reproduces).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
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
from null.contracts import _canonical_float
from null.costs.india_equity import IndiaEquityCostModel
from null.costs.model import Segment
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
#: Four values, per BUILD.md section 9's spec correction: {3,5,10} would make the
#: grid 81, not the 108 the spec requires. See module docstring.
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
    timestamps: Sequence[datetime],
    symbol: str,
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

    Takes ``timestamps`` (this symbol's own bar calendar, index-aligned with
    ``rsi``) rather than the ``Bar`` objects themselves -- the only two Bar
    fields this ever read were ``ts`` and ``symbol``, and ``symbol`` is
    constant across the whole call. See ``run_grid``'s docstring for why that
    matters for memory.
    """
    out: list[TargetWeight] = []
    in_position = False
    bars_held = 0

    for i, ts in enumerate(timestamps):
        value = rsi[i]
        if np.isnan(value):
            continue

        if not in_position:
            if value < entry:
                out.append(TargetWeight(ts=ts, symbol=symbol, weight=weight_when_long))
                in_position = True
                bars_held = 0
            continue

        bars_held += 1
        if value > exit or bars_held >= holding_cap:
            out.append(TargetWeight(ts=ts, symbol=symbol, weight=0.0))
            in_position = False

    return out


def _cost_drag_vectorized(
    *,
    weight_matrix: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    adv: npt.NDArray[np.float64],
    costs: IndiaEquityCostModel,
    initial_capital: float,
    segment: Segment,
    sigma_daily: float,
) -> npt.NDArray[np.float64]:
    """Per-day cost drag as a fraction of equity -- a numpy-vectorised
    restatement of the exact arithmetic ``IndiaEquityCostModel.charge()``
    does per (day, symbol) cell, not a re-derivation of the cost model.

    Replaced a ~20-million-iteration pure-Python loop (day x symbol x
    variant across the real 108-variant grid) that held the GIL for minutes
    on a free-tier CPU -- see docs/findings.md and service/jobs.py's module
    docstring for the production incident this caused. Every operation here
    mirrors the scalar version's sequence exactly (same order: traded ->
    quantity -> notional -> each charge component -> sum), because
    ``evidence_hash`` for the committed grid is pinned to this function's
    output and a numerically-equivalent-but-differently-ordered computation
    is exactly the kind of change that pinning exists to catch. Verified
    byte-identical against the committed artifact after this change; see
    docs/findings.md.

    ``equity`` is ``initial_capital`` for every cell -- the scalar version
    never updated it from realised P&L either (a pre-existing property of
    this strategy's accounting, not something this vectorisation changed),
    which is what makes each day's cost independent of every other day's
    and safe to compute as one array operation instead of a running loop.
    """
    equity = initial_capital
    n_days, n_symbols = weight_matrix.shape

    previous = np.vstack([np.zeros((1, n_symbols), dtype=np.float64), weight_matrix[:-1]])
    delta = np.abs(weight_matrix - previous)

    row_price = prices[1:]
    row_adv = adv[1:]

    valid_price = np.isfinite(row_price) & (row_price > 0.0)
    active = (delta > 0.0) & valid_price

    # Real price where valid; an arbitrary positive placeholder elsewhere,
    # to avoid a division by zero for cells `active` will zero out anyway.
    safe_price = np.where(valid_price, row_price, 1.0)
    traded = delta * equity
    quantity = traded / safe_price
    notional = quantity * safe_price

    is_buy = weight_matrix > previous

    has_adv = np.isfinite(row_adv) & (row_adv > 0.0)
    adv_used = np.where(has_adv, row_adv, 1.0)

    rates = costs.config.segments[segment]
    if rates.brokerage_pct > 0.0:
        brokerage = np.minimum(
            notional * rates.brokerage_pct / 100.0, rates.brokerage_per_order_cap
        )
    else:
        brokerage = np.zeros_like(notional)

    stt_pct = np.where(is_buy, rates.stt_buy_pct, rates.stt_sell_pct)
    stt = notional * stt_pct / 100.0
    exchange_txn = notional * rates.exchange_txn_pct / 100.0
    sebi_turnover = notional * rates.sebi_turnover_pct / 100.0
    stamp_duty = np.where(is_buy, notional * rates.stamp_duty_buy_pct / 100.0, 0.0)

    # GST on brokerage + exchange txn + SEBI turnover only -- see
    # null/costs/india_equity.py's charge(), which this mirrors exactly.
    gst = (brokerage + exchange_txn + sebi_turnover) * rates.gst_pct / 100.0
    dp_charge = np.where(is_buy, 0.0, rates.dp_charge_per_scrip_per_sell)

    slippage_cfg = costs.config.slippage
    tiers = slippage_cfg.liquidity_tiers  # ordered most to least liquid
    half_spread_bps = np.select(
        [adv_used >= tier.min_adv for tier in tiers],
        [np.full_like(notional, tier.half_spread_bps) for tier in tiers],
        default=tiers[-1].half_spread_bps,
    )
    # order_value=notional, adv_20=adv_used: adv_used is always > 0 (the
    # has_adv fallback above), so this never needs slippage.py's own
    # adv_20 <= 0 guard -- that branch existed for a caller that might pass
    # a non-positive adv_20 directly, which this call site never does.
    participation = notional / adv_used
    impact_bps = slippage_cfg.impact_k * sigma_daily * np.sqrt(participation) * 1e4
    slippage = notional * (half_spread_bps + impact_bps) / 1e4

    # The scalar version builds a ChargeBreakdown -- a frozen NullModel --
    # per cell, and every one of its float fields is quantised to 12
    # significant digits on construction (null/contracts.py's
    # _canonical_float). That quantisation happens to EACH component
    # BEFORE they're summed into .total, not to the sum afterward, and the
    # two do not always agree: skipping it here reproduced 106 of the
    # committed grid's 108 trial Sharpes exactly, but differed in the 13th
    # significant digit on 2 of them -- underneath the threshold "should
    # absorb summation-order differences" covers, not "quantise-then-sum
    # vs raw-then-sum is always the same operation" (it isn't; quantising
    # is a rounding step, and rounding does not commute with addition in
    # general). Reusing null.contracts's own quantisation function, not a
    # numeric reimplementation of %.12g rounding, is the only way to
    # guarantee this matches the scalar path bit-for-bit -- see
    # docs/findings.md for the full account of finding this the hard way.
    # Only active cells need it: an inactive cell's components are exactly
    # 0.0, and _canonical_float(0.0) is 0.0 by its own fast path, so
    # quantising those would be a costly no-op.
    components = (brokerage, stt, exchange_txn, sebi_turnover, stamp_duty, gst, dp_charge, slippage)
    active_idx = np.nonzero(active)
    total = np.zeros_like(notional)
    for component in components:
        quantized = component.copy()
        quantized[active_idx] = [_canonical_float(v) for v in component[active_idx]]
        total = total + quantized
    total = np.where(active, total, 0.0)

    # Sequential column-by-column accumulation, not total.sum(axis=1):
    # numpy's own reduction can pick a different internal summation order
    # (e.g. pairwise/SIMD-chunked) than left-to-right, and while that's
    # normally within the 12-significant-digit quantisation's tolerance, it
    # measurably wasn't for 2 of the committed grid's 108 variants (~1e-13,
    # right at the rounding boundary). This loop is over symbols (~50), not
    # days x symbols x variants -- a few dozen vectorised array additions,
    # not the O(20M) scalar loop this function replaced -- and it reproduces
    # the scalar version's exact accumulation order: day_cost += charge.total
    # for each symbol in turn.
    day_cost = np.zeros(n_days, dtype=np.float64)
    for col in range(n_symbols):
        day_cost = day_cost + total[:, col]
    if equity > 0.0:
        return np.asarray(day_cost / equity, dtype=np.float64)
    return np.zeros(n_days, dtype=np.float64)


@dataclass(frozen=True)
class VariantResult:
    variant: GridVariant
    #: Empty unless explicitly requested (``run_variant``'s
    #: ``include_weights``) -- a high-turnover variant's weight-change list
    #: can run into the thousands of ``TargetWeight`` objects, and
    #: ``run_grid`` only ever needs one variant's (the best's), not all of
    #: them held simultaneously. ``n_position_changes`` below is the cheap
    #: summary every variant keeps; this is the expensive detail only one
    #: needs to.
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
    timeline: tuple[datetime, ...],
    prices: npt.NDArray[np.float64],
    adv: npt.NDArray[np.float64],
    asset_returns: npt.NDArray[np.float64],
    universe: tuple[str, ...],
    rsi_by_symbol: dict[str, npt.NDArray[np.float64]],
    timestamps_by_symbol: dict[str, tuple[datetime, ...]],
    costs: IndiaEquityCostModel,
    initial_capital: float,
    segment: Segment = Segment.EQUITY_DELIVERY,
    sigma_daily: float = 0.018,
    include_weights: bool = True,
) -> VariantResult:
    """Backtest one grid point across the whole universe.

    ``timeline``/``prices``/``adv``/``asset_returns`` come from
    ``null.benchmark.buyhold``'s private portfolio-aggregation helpers
    (``_timeline``, ``_panel``, ``_returns_matrix``) -- computed ONCE by
    ``run_grid`` and passed in here, not recomputed per variant. They are
    identical for every variant in a grid (they depend only on ``bars`` and
    ``universe``, neither of which varies across variants); the original
    version called ``_timeline``/``_panel``/``_returns_matrix`` fresh inside
    this function on every one of a grid's calls, needlessly rebuilding the
    same (days, symbols) matrices 108 times over instead of once. Reusing
    those helpers at all (rather than re-deriving the same weight-to-return
    alignment a third time in this codebase) is what matters for
    correctness -- that alignment carried a real one-bar look-ahead bug
    earlier in this project's history, found only by an analytic
    multi-symbol test, and re-implementing it here would risk
    reintroducing exactly that class of bug with no equivalent test
    covering this module. *Where* they're called only matters for speed and
    memory, which is why this changed and the correctness argument didn't.

    ``include_weights=False`` (``run_grid``'s default for every variant
    except the one that turns out best) still generates and uses the full
    weight-change list internally -- it has to, to build ``weight_matrix``
    -- it just doesn't carry it into the returned ``VariantResult``, so it's
    eligible for garbage collection the moment this call returns rather than
    living in a 108-variant list for the rest of the grid's run. See
    ``docs/findings.md`` for the memory incident this exists to fix.
    """
    weight_when_long = 1.0 / len(universe)

    weights: list[TargetWeight] = []
    for symbol in universe:
        weights.extend(
            generate_weights_for_symbol(
                timestamps_by_symbol[symbol],
                symbol,
                rsi_by_symbol[symbol],
                entry=variant.entry,
                exit=variant.exit,
                holding_cap=variant.holding_cap,
                weight_when_long=weight_when_long,
            )
        )
    weights.sort(key=lambda w: (w.ts, w.symbol))

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

    cost_drag = _cost_drag_vectorized(
        weight_matrix=weight_matrix,
        prices=prices,
        adv=adv,
        costs=costs,
        initial_capital=initial_capital,
        segment=segment,
        sigma_daily=sigma_daily,
    )

    net = gross - cost_drag
    return_ts = timeline[1:]

    gross_series = Series(ts=return_ts, values=tuple(float(x) for x in gross))
    net_series = Series(ts=return_ts, values=tuple(float(x) for x in net))

    return VariantResult(
        variant=variant,
        weights=tuple(weights) if include_weights else (),
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
    per variant -- the distinct periods are read off ``variants`` itself (three
    of them across the real 108-variant grid), not assumed to be module-level
    ``RSI_PERIODS``. A caller passing variants outside the module's own grid
    (a service backtesting a caller-chosen period, say) must still get RSI
    computed for the periods it actually asked for, not silently get an empty
    ``rsi_by_symbol`` for anything outside {2, 3, 4}."""
    bars_by_symbol_raw: dict[str, list[Bar]] = {}
    for bar in bars:
        bars_by_symbol_raw.setdefault(bar.symbol, []).append(bar)
    sorted_bars_by_symbol: dict[str, list[Bar]] = {
        s: sorted(v, key=lambda b: b.ts) for s, v in bars_by_symbol_raw.items()
    }
    del bars_by_symbol_raw

    # Only ts and close/adv (the latter via _bh._panel below) are ever read off
    # a Bar in this function -- everything past this point works off these two
    # plain structures instead of the Bar objects themselves, so the full
    # per-symbol Bar lists can be dropped once they're built rather than kept
    # alive for the rest of the grid search. See the `del bars` note below.
    timestamps_by_symbol: dict[str, tuple[datetime, ...]] = {
        s: tuple(b.ts for b in v) for s, v in sorted_bars_by_symbol.items()
    }
    closes_by_symbol = {
        s: np.asarray([b.close for b in v], dtype=np.float64)
        for s, v in sorted_bars_by_symbol.items()
    }
    del sorted_bars_by_symbol

    periods_needed = sorted({variant.period for variant in variants})
    rsi_cache: dict[tuple[str, int], npt.NDArray[np.float64]] = {}
    for symbol in universe:
        if symbol not in closes_by_symbol:
            continue
        for period in periods_needed:
            rsi_cache[(symbol, period)] = compute_rsi(closes_by_symbol[symbol], period)

    effective_universe = tuple(s for s in universe if s in timestamps_by_symbol)

    # Computed once, not once per variant: identical for every variant in
    # this grid (function only of `bars` and `effective_universe`, neither
    # of which the variant loop below ever changes). The original version
    # had run_variant call _timeline/_panel/_returns_matrix fresh on every
    # one of a grid's calls -- 108 rebuilds of the same (days, symbols)
    # matrices instead of one. See run_variant's own docstring.
    timeline = _bh._timeline(bars)
    prices, adv = _bh._panel(bars, timeline, effective_universe)
    asset_returns = _bh._returns_matrix(prices)

    # Every remaining use in this function reads timeline/prices/adv/
    # asset_returns/timestamps_by_symbol/closes_by_symbol/rsi_cache, never
    # `bars` itself again -- dropping it here (the caller does the same right
    # after this call returns) frees the full-universe Bar tuple, ~300MB on
    # the real NIFTY 50 window, before the 108-variant loop's own working set
    # grows, instead of holding both peaks at once. This was the dominant
    # remaining cost in the OOM investigated in docs/findings.md #11.
    del bars

    # include_weights=False for every variant here: run_variant still builds
    # the full weight-change list internally (it has to, to construct
    # weight_matrix), but doesn't carry it into the returned VariantResult,
    # so it's freed once that call returns rather than living in `results`
    # for the rest of the grid. Only the best variant's weights are ever
    # read by any caller (write_run_artifacts's `best.weights`) -- holding
    # all of them for a 108-variant, high-turnover grid simultaneously was
    # the dominant contributor to a real out-of-memory production incident;
    # see docs/findings.md.
    results: list[VariantResult] = []
    best_idx: int | None = None
    for variant in variants:
        rsi_by_symbol = {
            symbol: rsi_cache[(symbol, variant.period)]
            for symbol in effective_universe
            if (symbol, variant.period) in rsi_cache
        }
        result = run_variant(
            variant=variant,
            timeline=timeline,
            prices=prices,
            adv=adv,
            asset_returns=asset_returns,
            universe=effective_universe,
            rsi_by_symbol=rsi_by_symbol,
            timestamps_by_symbol=timestamps_by_symbol,
            costs=costs,
            initial_capital=initial_capital,
            include_weights=False,
        )
        results.append(result)
        if best_idx is None or result.net_sharpe > results[best_idx].net_sharpe:
            best_idx = len(results) - 1

    # One more run, for the variant that actually needs its weights kept --
    # cheap relative to the 108-variant grid this follows (one extra variant
    # out of however many were just run), and the only way to have exactly
    # one VariantResult carrying weights without holding all of them at once
    # along the way. Deterministic like everything else here: the identical
    # inputs reproduce the identical VariantResult bit-for-bit.
    if best_idx is not None:
        best_variant = results[best_idx].variant
        rsi_by_symbol = {
            symbol: rsi_cache[(symbol, best_variant.period)]
            for symbol in effective_universe
            if (symbol, best_variant.period) in rsi_cache
        }
        results[best_idx] = run_variant(
            variant=best_variant,
            timeline=timeline,
            prices=prices,
            adv=adv,
            asset_returns=asset_returns,
            universe=effective_universe,
            rsi_by_symbol=rsi_by_symbol,
            timestamps_by_symbol=timestamps_by_symbol,
            costs=costs,
            initial_capital=initial_capital,
            include_weights=True,
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
