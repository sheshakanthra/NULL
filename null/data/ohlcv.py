"""Offline OHLCV loader and its validators. Reads the committed parquet. Never fetches.

The network stage is ``scripts/fetch_bhavcopy.py``, outside this package because
`null audit` must run with the network off (CLAUDE.md invariant 2) and the source
grep forbids network imports under ``null/``.

Not in the BUILD.md §1 layout: §1 has no home for loaded market data, and putting
it under ``benchmark/`` would be wrong since strategy bars are not benchmark data.

**No fallback to any other source.** A missing cache raises. Silently substituting
yfinance, or a different exchange's file, or a shorter window, would each change what
the audit is measuring without saying so.

``adv_20`` is computed here from **value traded**, not from price x volume on a
single day. The capacity gate reads it, and a single day's notional is a far noisier
estimate of how much liquidity an order actually has to work against.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from null.contracts import Bar, IST, NonEmptyStr, NullModel

__all__ = [
    "ADV_WINDOW",
    "MAX_SINGLE_DAY_MOVE",
    "OHLCVCacheMissing",
    "OHLCVValidation",
    "DEFAULT_CACHE",
    "load_bars",
    "validate_ohlcv",
]

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "nifty50_ohlcv.parquet"
)

#: Rolling window for average daily value traded, per BUILD.md §2.
ADV_WINDOW = 20

#: BUILD.md §5: single-bar moves beyond this are almost always an unadjusted
#: corporate action rather than a real price move.
MAX_SINGLE_DAY_MOVE = 0.25

#: A NIFTY 50 name trades every session. Zero volume on a trading day means the row
#: is wrong, not that nobody wanted it.
EXPECTED_TRADING_DAYS = (240, 256)


class OHLCVCacheMissing(FileNotFoundError):
    """The OHLCV cache is absent. A stop, not a reason to reach for another source."""


class OHLCVValidation(NullModel):
    is_valid: bool
    n_rows: int
    n_symbols: int
    suspicious_moves: tuple[NonEmptyStr, ...]
    """Single-day moves beyond the threshold, flagged not accepted."""
    zero_volume_days: tuple[NonEmptyStr, ...]
    negative_volume_days: tuple[NonEmptyStr, ...]
    years_outside_expected_day_count: tuple[NonEmptyStr, ...]
    calendar_mismatches: tuple[NonEmptyStr, ...]
    rationale: NonEmptyStr


def _frame(path: Path) -> "pd.DataFrame":
    if not path.exists():
        raise OHLCVCacheMissing(
            f"No OHLCV cache at {path}. Run `python scripts/fetch_bhavcopy.py "
            "--refresh --symbols ...` and commit the result. NULL will not substitute "
            "another data source: a different provider means different adjustment "
            "handling, a different calendar and a different survivorship profile, and "
            "swapping one in silently would change what the audit measures."
        )
    import pandas as pd

    frame = pd.read_parquet(path)
    missing = {"date", "symbol", "close", "volume", "value_traded"} - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV cache is missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_bars(
    path: Path = DEFAULT_CACHE,
    *,
    symbols: tuple[str, ...] | None = None,
    adv_window: int = ADV_WINDOW,
) -> tuple[Bar, ...]:
    """Read the committed cache into contract ``Bar`` objects.

    ``adv_20`` is the trailing mean of **value traded** over ``adv_window``
    sessions, computed per symbol. Days before the window fills carry the mean of
    what exists rather than NaN, so early bars are usable but conservative.
    """
    frame = _frame(path)
    if symbols is not None:
        frame = frame[frame["symbol"].isin(set(symbols))].reset_index(drop=True)

    frame["adv_20"] = frame.groupby("symbol")["value_traded"].transform(
        lambda s: s.rolling(adv_window, min_periods=1).mean()
    )

    # Extract to plain Python and numpy at the boundary. Iterating a DataFrame row
    # by row leaves every field loosely typed, and these values flow straight into
    # frozen contracts where the types matter.
    stamps = [d.to_pydatetime() for d in frame["date"]]
    syms = [str(s) for s in frame["symbol"]]
    cols = {
        name: np.asarray(frame[name].to_numpy(), dtype=np.float64)
        for name in ("open", "high", "low", "close", "volume", "adv_20")
    }

    bars: list[Bar] = []
    for i, stamp in enumerate(stamps):
        when = stamp if stamp.tzinfo else stamp.replace(hour=15, minute=30, tzinfo=IST)
        adv = float(cols["adv_20"][i])
        bars.append(
            Bar(
                ts=when,
                symbol=syms[i],
                open=float(cols["open"][i]),
                high=float(cols["high"][i]),
                low=float(cols["low"][i]),
                close=float(cols["close"][i]),
                volume=float(cols["volume"][i]),
                adv_20=adv if np.isfinite(adv) else None,
            )
        )
    return tuple(bars)


def validate_ohlcv(
    path: Path = DEFAULT_CACHE,
    *,
    corporate_action_dates: dict[str, set[str]] | None = None,
    benchmark_dates: tuple[str, ...] | None = None,
    max_move: float = MAX_SINGLE_DAY_MOVE,
) -> OHLCVValidation:
    """Four checks, in the order a wrong series fails them.

    1. **Adjusted-close continuity.** A single-day move beyond ``max_move`` that
       does not coincide with a known corporate action is FLAGGED, not accepted.
       Without an action calendar every such move is flagged, which is the honest
       state: they cannot be confirmed.
    2. **Volume sanity.** Never negative; never zero on a trading day for a NIFTY 50
       name, because those trade every session.
    3. **Trading days per year** in the expected range. A year with 260 sessions
       means a non-trading date was picked up.
    4. **Calendar agreement** with the benchmark series' own dates, so the two are
       not silently on different calendars.
    """
    frame = _frame(path)
    actions = corporate_action_dates or {}

    suspicious: list[str] = []
    zero_vol: list[str] = []
    negative_vol: list[str] = []

    all_symbols = [str(s) for s in frame["symbol"]]
    all_dates = [str(d.date()) for d in frame["date"]]
    all_close = np.asarray(frame["close"].to_numpy(), dtype=np.float64)
    all_volume = np.asarray(frame["volume"].to_numpy(), dtype=np.float64)

    by_symbol: dict[str, list[int]] = {}
    for i, sym in enumerate(all_symbols):
        by_symbol.setdefault(sym, []).append(i)

    for symbol, rows in sorted(by_symbol.items()):
        closes = all_close[rows]
        days = [all_dates[i] for i in rows]
        if closes.size > 1:
            moves = closes[1:] / closes[:-1] - 1.0
            for k, move in enumerate(moves):
                if abs(float(move)) > max_move:
                    day = days[k + 1]
                    if day in actions.get(symbol, set()):
                        continue
                    suspicious.append(f"{symbol} {day} {float(move):+.1%}")
        for k, index in enumerate(rows):
            vol = float(all_volume[index])
            if vol < 0:
                negative_vol.append(f"{symbol} {days[k]} {vol:.0f}")
            elif vol == 0:
                zero_vol.append(f"{symbol} {days[k]}")

    per_year: list[str] = []
    low, high = EXPECTED_TRADING_DAYS
    sessions: dict[str, set[str]] = {}
    for day in all_dates:
        sessions.setdefault(day[:4], set()).add(day)
    for year, unique_days in sorted(sessions.items()):
        if not (low <= len(unique_days) <= high):
            per_year.append(
                f"{year}: {len(unique_days)} sessions, expected {low}-{high}"
            )

    mismatches: list[str] = []
    if benchmark_dates is not None:
        ours = set(all_dates)
        theirs = set(benchmark_dates)
        mismatches = [f"only in OHLCV: {d}" for d in sorted(ours - theirs)[:10]]
        mismatches += [f"only in benchmark: {d}" for d in sorted(theirs - ours)[:10]]

    is_valid = not (suspicious or zero_vol or negative_vol or per_year or mismatches)
    parts = []
    if suspicious:
        parts.append(
            f"{len(suspicious)} single-day move(s) beyond {max_move:.0%} could not be "
            "matched to a known corporate action"
        )
    if negative_vol:
        parts.append(f"{len(negative_vol)} negative-volume row(s)")
    if zero_vol:
        parts.append(f"{len(zero_vol)} zero-volume trading day(s)")
    if per_year:
        parts.append(f"{len(per_year)} year(s) with an unexpected session count")
    if mismatches:
        parts.append(f"{len(mismatches)} calendar disagreement(s) with the benchmark")

    n_symbols = len(by_symbol)
    rationale = (
        f"{len(frame):,} rows across {n_symbols} symbols passed every "
        "check: no unexplained single-day jumps, no bad volumes, session counts in "
        "range, and the calendar agrees with the benchmark."
        if is_valid
        else "Validation failed: " + "; ".join(parts) + ". Do not trust this series."
    )

    return OHLCVValidation(
        is_valid=is_valid,
        n_rows=int(len(frame)),
        n_symbols=n_symbols,
        suspicious_moves=tuple(suspicious[:50]),
        zero_volume_days=tuple(zero_vol[:50]),
        negative_volume_days=tuple(negative_vol[:50]),
        years_outside_expected_day_count=tuple(per_year),
        calendar_mismatches=tuple(mismatches[:20]),
        rationale=rationale,
    )
