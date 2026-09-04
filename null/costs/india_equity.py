"""Config-driven Indian equity charge stack.

Every rate comes from ``configs/costs_india_equity.yaml``. Nothing in this module
is a rate. If you find yourself typing a number that a broker could change, it
belongs in the config (CLAUDE.md, working rules).

The charge components, per BUILD.md section 3:

  brokerage          delivery vs intraday, with a per-order cap
  STT                asymmetric -- both legs on delivery, sell-only on intraday
  exchange txn       differs by segment
  SEBI turnover      flat rate on turnover
  stamp duty         buy side only
  GST                on brokerage + exchange + SEBI, and on nothing else
  DP charge          flat, per scrip, per day, on delivery sells
  slippage           half-spread + square-root impact (see slippage.py)

The DP charge is the one people forget. It is flat, so as notional falls it
becomes an ever-larger fraction of the trade, and it is what makes small-notional
high-frequency retail strategies arithmetically impossible.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from null.contracts import NonNegativeFloat, NonEmptyStr, NullModel
from null.costs.model import ChargeBreakdown, Segment, Side
from null.costs.slippage import SlippageConfig, SlippageModel

__all__ = ["CostConfig", "IndiaEquityCostModel", "RoundTripCost", "SegmentRates"]


class SegmentRates(NullModel):
    """The charge stack for one segment. Percentages, as written in the config."""

    brokerage_pct: NonNegativeFloat
    brokerage_per_order_cap: NonNegativeFloat
    stt_buy_pct: NonNegativeFloat
    stt_sell_pct: NonNegativeFloat
    exchange_txn_pct: NonNegativeFloat
    sebi_turnover_pct: NonNegativeFloat
    stamp_duty_buy_pct: NonNegativeFloat
    gst_pct: NonNegativeFloat
    dp_charge_per_scrip_per_sell: NonNegativeFloat


class CostConfig(NullModel):
    source: NonEmptyStr
    verified_on: NonEmptyStr
    warning: NonEmptyStr
    segments: dict[Segment, SegmentRates]
    slippage: SlippageConfig

    @property
    def rates_are_verified(self) -> bool:
        """True only when ``_verified_on`` is a real ISO date.

        Deliberately does not compare against today: there is no wall-clock in the
        audit path (CLAUDE.md invariant 4). Staleness is a human review question;
        this flag only answers "has anyone ever checked these against a broker?"
        """
        try:
            date.fromisoformat(self.verified_on)
        except ValueError:
            return False
        return True

    @classmethod
    def from_yaml(cls, path: Path) -> CostConfig:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(
            {
                "source": raw["_source"],
                "verified_on": raw["_verified_on"],
                "warning": raw["_warning"],
                "segments": raw["segments"],
                "slippage": raw["slippage"],
            }
        )


class RoundTripCost(NullModel):
    """A buy and the matching sell, costed together."""

    buy: ChargeBreakdown
    sell: ChargeBreakdown

    @property
    def total_charges(self) -> float:
        return self.buy.total + self.sell.total

    @property
    def notional(self) -> float:
        """Entry notional. The denominator a trader actually thinks in."""
        return self.buy.notional

    @property
    def cost_fraction(self) -> float:
        return self.total_charges / self.notional if self.notional > 0.0 else 0.0

    @property
    def impact_bps(self) -> float:
        """Impact paid across both legs."""
        return self.buy.impact_bps + self.sell.impact_bps


class IndiaEquityCostModel:
    """Pure function of its config. Satisfies the ``CostModel`` protocol."""

    def __init__(self, config: CostConfig) -> None:
        self.config = config
        self._slippage = SlippageModel(config.slippage)

    @classmethod
    def from_yaml(cls, path: Path) -> IndiaEquityCostModel:
        return cls(CostConfig.from_yaml(path))

    def with_scaled_rate(
        self, segment: Segment, field: str, factor: float
    ) -> IndiaEquityCostModel:
        """A copy with one rate scaled. Exists so tests can prove rates are config-driven."""
        rates = self.config.segments[segment]
        if field not in type(rates).model_fields:
            raise KeyError(f"{field!r} is not a rate on {segment.value}")
        updated = rates.model_copy(update={field: getattr(rates, field) * factor})
        segments = {**self.config.segments, segment: updated}
        return IndiaEquityCostModel(self.config.model_copy(update={"segments": segments}))

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
        if segment not in self.config.segments:
            raise KeyError(
                f"no rates configured for segment {segment.value!r}; add it to the "
                "cost config rather than defaulting to zero"
            )
        rates = self.config.segments[segment]
        notional = quantity * price
        is_buy = side is Side.BUY

        brokerage = min(
            notional * rates.brokerage_pct / 100.0, rates.brokerage_per_order_cap
        ) if rates.brokerage_pct > 0.0 else 0.0

        stt_pct = rates.stt_buy_pct if is_buy else rates.stt_sell_pct
        stt = notional * stt_pct / 100.0

        exchange_txn = notional * rates.exchange_txn_pct / 100.0
        sebi_turnover = notional * rates.sebi_turnover_pct / 100.0
        stamp_duty = notional * rates.stamp_duty_buy_pct / 100.0 if is_buy else 0.0

        # GST is levied on brokerage, exchange transaction charges and the SEBI
        # turnover fee. Not on STT, not on stamp duty, not on the DP charge (whose
        # configured figure is already GST-inclusive).
        gst = (brokerage + exchange_txn + sebi_turnover) * rates.gst_pct / 100.0

        # Flat, per scrip, per day, and only when stock actually leaves the demat
        # account. This is what makes small-notional delivery strategies fail.
        dp_charge = 0.0 if is_buy else rates.dp_charge_per_scrip_per_sell

        half_spread_bps = self._slippage.half_spread_bps(adv_20)
        impact_bps = self._slippage.impact_bps(
            order_value=notional, sigma_daily=sigma_daily, adv_20=adv_20
        )
        slippage = notional * (half_spread_bps + impact_bps) / 1e4

        return ChargeBreakdown(
            symbol=symbol,
            side=side,
            segment=segment,
            quantity=quantity,
            price=price,
            notional=notional,
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn,
            sebi_turnover=sebi_turnover,
            stamp_duty=stamp_duty,
            gst=gst,
            dp_charge=dp_charge,
            slippage=slippage,
            half_spread_bps=half_spread_bps,
            impact_bps=impact_bps,
        )

    def round_trip(
        self,
        *,
        symbol: str,
        notional: float,
        price: float,
        segment: Segment,
        sigma_daily: float,
        adv_20: float,
    ) -> RoundTripCost:
        """Buy then sell the same position.

        Quantity is floored to whole shares -- Indian equities do not trade in
        fractions, and rounding up would understate the cost fraction on exactly
        the small notionals this model exists to expose.
        """
        if price <= 0.0:
            raise ValueError(f"price must be positive, got {price}")
        quantity = float(int(notional // price))
        if quantity <= 0.0:
            raise ValueError(
                f"Rs {notional:,.2f} does not buy a single share at Rs {price:,.2f}"
            )
        def leg(side: Side) -> ChargeBreakdown:
            return self.charge(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                segment=segment,
                sigma_daily=sigma_daily,
                adv_20=adv_20,
            )

        return RoundTripCost(buy=leg(Side.BUY), sell=leg(Side.SELL))
