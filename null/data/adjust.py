"""Back-adjust prices using the exchange's own corporate action calendar.

Bhavcopy closes are raw traded prices. A 1:10 split appears as a -90% single-day
return, and a strategy measured across that is being measured across fiction.

**This module does not infer.** An earlier version tried to detect splits from the
price series alone, using a clean-ratio test corroborated by value-traded
continuity. That failed structurally, not by calibration:

  * The two conditions were not independent. Value traded staying roughly flat
    across a boundary is true of **99.02%** of ordinary trading days on this
    dataset, so it corroborated nothing and the price ratio was working alone.
  * No tolerance separates a 1:3 bonus (factor 1.333) from the COVID crash
    (implied 1.387). Tight enough to reject the crash also rejected seven genuine
    splits sitting at 4-5.6% ratio error, because a split day carries the split
    *and* that day's own price move.
  * It could not see stacked actions. BAJFINANCE 2025-06-16 was a 1:2 split AND a
    4:1 bonus, combining to exactly 10 -- readable only from the calendar.
  * It mistook the ADANIENT 2015 demerger for a 1:6 split and adjusted it, which
    inflates the parent's history rather than correcting it.

Now a move on a date matching an announced action is adjusted by the announced
ratio, full stop. A move matching nothing is a market event and stays unadjusted.

Demergers are detected and **never adjusted**: value genuinely leaves the parent,
so back-adjusting would be a fabrication. They go in the limitations band instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from null.contracts import NonEmptyStr, NonNegativeFloat, NullFloat, NullModel
from null.data.corporate_actions import CorporateAction, adjustment_factors

__all__ = [
    "MIN_MOVE",
    "AppliedAdjustment",
    "back_adjust",
    "reconcile_moves",
]

#: Moves past this are reconciled against the calendar. Matches the leakage audit.
MIN_MOVE = 0.25


class AppliedAdjustment(NullModel):
    """One large move, and what the calendar says about it."""

    symbol: NonEmptyStr
    date: NonEmptyStr
    raw_move: NullFloat
    implied_factor: NonNegativeFloat
    """previous_close / close, from the prices alone."""
    calendar_factor: NonNegativeFloat
    """From the announced action. 0.0 when the calendar names nothing."""
    calendar_subject: NonEmptyStr
    adjusted: bool
    residual_move: NullFloat
    """The move that remains after adjustment -- the day's genuine price change."""


def back_adjust(
    frame: "pd.DataFrame", actions: tuple[CorporateAction, ...]
) -> "pd.DataFrame":
    """Apply announced ratios backwards. The raw frame is left untouched.

    Prices strictly before an ex-date are divided by the cumulative factor and
    volumes multiplied by it, so the series is continuous and value traded is
    preserved. Backwards only: the most recent price is the traded price, which is
    what BUILD.md §5 requires.
    """
    import pandas as pd

    factors = adjustment_factors(actions)
    by_symbol: dict[str, list[tuple[str, float]]] = {}
    for (symbol, ex_date), factor in factors.items():
        by_symbol.setdefault(symbol, []).append((ex_date, factor))

    out = frame.copy()
    out["adjustment_factor"] = 1.0

    for symbol, events in by_symbol.items():
        mask = out["symbol"] == symbol
        rows = out.index[mask]
        if not len(rows):
            continue
        dates = pd.to_datetime(out.loc[rows, "date"])
        cumulative = np.ones(len(rows), dtype=np.float64)
        for ex_date, factor in sorted(events, reverse=True):
            before = (dates < pd.Timestamp(ex_date)).to_numpy()
            cumulative[before] *= factor
        out.loc[rows, "adjustment_factor"] = cumulative

    factor_column = np.asarray(out["adjustment_factor"].to_numpy(), dtype=np.float64)
    for column in ("open", "high", "low", "close"):
        out[column] = np.asarray(out[column].to_numpy(), dtype=np.float64) / factor_column
    out["volume"] = np.asarray(out["volume"].to_numpy(), dtype=np.float64) * factor_column
    # value_traded is price x volume, invariant under the adjustment, left alone.
    return out


def reconcile_moves(
    raw: "pd.DataFrame", actions: tuple[CorporateAction, ...]
) -> list[AppliedAdjustment]:
    """Every large raw move, matched against the calendar. The auditable artifact.

    Produced from the RAW frame so it shows what was found and what was done about
    it, including moves the calendar does not explain.
    """
    factors = adjustment_factors(actions)
    subjects: dict[tuple[str, str], str] = {}
    for action in actions:
        key = (action.symbol, action.ex_date)
        if action.is_adjustable:
            existing = subjects.get(key)
            subjects[key] = f"{existing} + {action.subject}" if existing else action.subject

    out: list[AppliedAdjustment] = []
    for symbol, group in raw.groupby("symbol", sort=True):
        group = group.sort_values("date")
        closes = np.asarray(group["close"].to_numpy(), dtype=np.float64)
        dates = [str(d.date()) for d in group["date"]]
        if closes.size < 2:
            continue
        moves = closes[1:] / closes[:-1] - 1.0
        for i, move in enumerate(moves):
            if abs(float(move)) <= MIN_MOVE:
                continue
            here = i + 1
            implied = float(closes[i] / closes[here]) if closes[here] > 0 else 0.0
            key = (str(symbol), dates[here])
            calendar = factors.get(key, 0.0)
            residual = (
                float(closes[here] * calendar / closes[i] - 1.0) if calendar else float(move)
            )
            out.append(
                AppliedAdjustment(
                    symbol=str(symbol),
                    date=dates[here],
                    raw_move=float(move),
                    implied_factor=implied,
                    calendar_factor=calendar,
                    calendar_subject=subjects.get(key, "no announced action"),
                    adjusted=bool(calendar),
                    residual_move=residual,
                )
            )
    return out
