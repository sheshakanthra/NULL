"""Spread plus square-root market impact.

    cost_bps = half_spread_bps + k * sigma_daily * sqrt(order_value / adv_20) * 1e4

BUILD.md section 3 states the requirement plainly: slippage must scale with order
size. A model that charges a flat 5bps regardless of notional is how strategies
fake capacity, and a capacity claim that survives only because the cost model
ignores size is the exact fiction NULL exists to catch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from null.contracts import NonNegativeFloat, NonEmptyStr, NullModel, PositiveFloat

__all__ = ["LiquidityTier", "SlippageConfig", "SlippageModel"]


class LiquidityTier(NullModel):
    """Half-spread for names above a 20d average-daily-value threshold."""

    name: NonEmptyStr
    min_adv: NonNegativeFloat
    half_spread_bps: NonNegativeFloat


class SlippageConfig(NullModel):
    impact_k: NonNegativeFloat
    liquidity_tiers: tuple[LiquidityTier, ...]

    def tier_for(self, adv_20: float) -> LiquidityTier:
        """First tier whose threshold is met. Tiers are ordered most to least liquid."""
        for tier in self.liquidity_tiers:
            if adv_20 >= tier.min_adv:
                return tier
        # Unreachable with a well-formed config (the last tier has min_adv 0), but
        # falling back to the widest spread is the conservative direction.
        return self.liquidity_tiers[-1]


class SlippageModel:
    """Pure function of its config. No I/O, no state, no wall-clock."""

    def __init__(self, config: SlippageConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path) -> SlippageModel:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(SlippageConfig.model_validate(raw["slippage"]))

    def half_spread_bps(self, adv_20: float) -> float:
        return self.config.tier_for(adv_20).half_spread_bps

    def impact_bps(
        self, *, order_value: float, sigma_daily: float, adv_20: float
    ) -> float:
        """Square-root impact. Quadrupling order value doubles the impact.

        An order in a name with no reported ADV cannot be sized responsibly, so it
        is treated as fully illiquid rather than free.
        """
        if order_value <= 0.0:
            return 0.0
        if adv_20 <= 0.0:
            raise ValueError(
                f"adv_20 must be positive to size impact, got {adv_20}. A symbol with "
                "no traded value has no capacity, and charging zero impact would let "
                "an untradeable strategy through."
            )
        participation = order_value / adv_20
        return self.config.impact_k * sigma_daily * math.sqrt(participation) * 1e4

    def total_bps(
        self, *, order_value: float, sigma_daily: float, adv_20: float
    ) -> float:
        return self.half_spread_bps(adv_20) + self.impact_bps(
            order_value=order_value, sigma_daily=sigma_daily, adv_20=adv_20
        )

    def cost(
        self, *, order_value: float, sigma_daily: float, adv_20: float
    ) -> float:
        """Slippage in currency for one order."""
        bps = self.total_bps(
            order_value=order_value, sigma_daily=sigma_daily, adv_20=adv_20
        )
        return order_value * bps / 1e4
