"""Cost model protocol and the charge breakdown every implementation returns.

BUILD.md section 3: the cost model is the single most common reason retail
backtests are fiction. Every rate is config-driven and none of them appear in
this package as a literal.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from null.contracts import NonNegativeFloat, NullFloat, NullModel, Symbol

__all__ = [
    "ChargeBreakdown",
    "CostModel",
    "Segment",
    "Side",
]


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Segment(StrEnum):
    EQUITY_DELIVERY = "equity_delivery"
    EQUITY_INTRADAY = "equity_intraday"


class ChargeBreakdown(NullModel):
    """Every charge on one order, itemised.

    Itemised rather than a single number because the report has to be able to say
    *which* charge killed the strategy. "Costs ate your edge" is not actionable;
    "the flat DP charge is 1.3% of a Rs 10,000 position" is.
    """

    symbol: Symbol
    side: Side
    segment: Segment
    quantity: NonNegativeFloat
    price: NonNegativeFloat
    notional: NonNegativeFloat

    brokerage: NonNegativeFloat
    stt: NonNegativeFloat
    exchange_txn: NonNegativeFloat
    sebi_turnover: NonNegativeFloat
    stamp_duty: NonNegativeFloat
    gst: NonNegativeFloat
    dp_charge: NonNegativeFloat
    slippage: NonNegativeFloat

    half_spread_bps: NonNegativeFloat
    impact_bps: NonNegativeFloat

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_turnover
            + self.stamp_duty
            + self.gst
            + self.dp_charge
            + self.slippage
        )

    @property
    def cost_fraction(self) -> float:
        """Total charges as a fraction of notional. Zero notional costs nothing."""
        return self.total / self.notional if self.notional > 0.0 else 0.0

    def as_dict(self) -> dict[str, float]:
        """Shaped for ``Evidence.cost_breakdown``. Sorted keys, no surprises."""
        return {
            "brokerage": self.brokerage,
            "dp_charge": self.dp_charge,
            "exchange_txn": self.exchange_txn,
            "gst": self.gst,
            "sebi_turnover": self.sebi_turnover,
            "slippage": self.slippage,
            "stamp_duty": self.stamp_duty,
            "stt": self.stt,
        }


@runtime_checkable
class CostModel(Protocol):
    """What the audit pipeline codes against.

    An implementation must be a pure function of its inputs and its config. No
    I/O at charge time, no wall-clock, no hidden state -- the verdict has to be
    reproducible, and a cost model that reads the network is not.
    """

    def charge(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        segment: Segment,
        sigma_daily: float,
        adv_20: float,
    ) -> ChargeBreakdown:
        """All charges for one order, itemised."""
        ...
