"""M1 acceptance test -- BUILD.md section 3.

    Round-trip a Rs 10,000 delivery position and a Rs 10,00,000 delivery position
    in the same symbol. Cost as a fraction of notional must be strictly higher
    for the small one (DP charge dominates) and impact-bps must be strictly
    higher for the large one. If both aren't true, the model is wrong.

Both halves matter. The first catches a model that forgets the flat DP charge and
so pretends small-notional strategies are viable. The second catches a model that
charges a flat 5bps regardless of size, which is how strategies fake capacity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from null.costs.india_equity import IndiaEquityCostModel
from null.costs.model import Segment, Side
from null.costs.slippage import SlippageModel

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "costs_india_equity.yaml"

SMALL = 10_000.0
LARGE = 1_000_000.0

# One symbol, one liquidity profile, one volatility. Only notional differs.
SYMBOL = "RELIANCE"
PRICE = 2_500.0
SIGMA_DAILY = 0.018
ADV_20 = 8_000_000_000.0


@pytest.fixture(scope="module")
def model() -> IndiaEquityCostModel:
    return IndiaEquityCostModel.from_yaml(CONFIG_PATH)


def _round_trip(model: IndiaEquityCostModel, notional: float):
    return model.round_trip(
        symbol=SYMBOL,
        notional=notional,
        price=PRICE,
        segment=Segment.EQUITY_DELIVERY,
        sigma_daily=SIGMA_DAILY,
        adv_20=ADV_20,
    )


# ---------------------------------------------------------------------------
# the acceptance test proper
# ---------------------------------------------------------------------------


def test_small_notional_pays_a_strictly_higher_cost_fraction(
    model: IndiaEquityCostModel,
) -> None:
    """The DP charge is flat, so it murders small-notional strategies."""
    small = _round_trip(model, SMALL)
    large = _round_trip(model, LARGE)
    assert small.cost_fraction > large.cost_fraction, (
        f"Rs {SMALL:,.0f} paid {small.cost_fraction:.6%} of notional and "
        f"Rs {LARGE:,.0f} paid {large.cost_fraction:.6%}. A flat per-sell DP charge "
        "must dominate at small notional; if it does not, the charge stack is wrong."
    )


def test_large_notional_pays_strictly_higher_impact_bps(
    model: IndiaEquityCostModel,
) -> None:
    """Square-root impact law: slippage must scale with order size."""
    small = _round_trip(model, SMALL)
    large = _round_trip(model, LARGE)
    assert large.impact_bps > small.impact_bps, (
        f"Rs {SMALL:,.0f} was charged {small.impact_bps:.4f} impact bps and "
        f"Rs {LARGE:,.0f} was charged {large.impact_bps:.4f}. A model that charges "
        "flat bps regardless of notional is how strategies fake capacity."
    )


def test_acceptance_both_halves_hold_together(model: IndiaEquityCostModel) -> None:
    """BUILD.md: 'If both aren't true, the model is wrong.'"""
    small = _round_trip(model, SMALL)
    large = _round_trip(model, LARGE)
    assert small.cost_fraction > large.cost_fraction
    assert large.impact_bps > small.impact_bps


# ---------------------------------------------------------------------------
# supporting properties the acceptance test relies on
# ---------------------------------------------------------------------------


def test_dp_charge_is_levied_once_per_round_trip_on_the_sell_leg(
    model: IndiaEquityCostModel,
) -> None:
    """Per scrip, per day, on delivery sells -- not on the buy."""
    buy = model.charge(
        symbol=SYMBOL, side=Side.BUY, quantity=4, price=PRICE,
        segment=Segment.EQUITY_DELIVERY, sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    sell = model.charge(
        symbol=SYMBOL, side=Side.SELL, quantity=4, price=PRICE,
        segment=Segment.EQUITY_DELIVERY, sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    assert buy.dp_charge == 0.0
    assert sell.dp_charge > 0.0


def test_stt_is_asymmetric_on_delivery(model: IndiaEquityCostModel) -> None:
    """Delivery pays STT on both legs; the config, not Python, sets the rates."""
    kw = dict(
        symbol=SYMBOL, quantity=400, price=PRICE, segment=Segment.EQUITY_DELIVERY,
        sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    assert model.charge(side=Side.BUY, **kw).stt > 0.0
    assert model.charge(side=Side.SELL, **kw).stt > 0.0


def test_stamp_duty_is_buy_side_only(model: IndiaEquityCostModel) -> None:
    kw = dict(
        symbol=SYMBOL, quantity=400, price=PRICE, segment=Segment.EQUITY_DELIVERY,
        sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    assert model.charge(side=Side.BUY, **kw).stamp_duty > 0.0
    assert model.charge(side=Side.SELL, **kw).stamp_duty == 0.0


def test_impact_scales_as_the_square_root_of_participation(
    model: IndiaEquityCostModel,
) -> None:
    """Quadrupling order value must roughly double impact, not quadruple it."""
    slip = SlippageModel.from_yaml(CONFIG_PATH)
    one = slip.impact_bps(order_value=1_000_000.0, sigma_daily=SIGMA_DAILY, adv_20=ADV_20)
    four = slip.impact_bps(order_value=4_000_000.0, sigma_daily=SIGMA_DAILY, adv_20=ADV_20)
    assert four == pytest.approx(2.0 * one, rel=1e-9)


def test_total_is_the_sum_of_its_parts(model: IndiaEquityCostModel) -> None:
    """A breakdown that does not add up cannot be reported honestly."""
    c = model.charge(
        symbol=SYMBOL, side=Side.SELL, quantity=400, price=PRICE,
        segment=Segment.EQUITY_DELIVERY, sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    parts = (
        c.brokerage + c.stt + c.exchange_txn + c.sebi_turnover
        + c.stamp_duty + c.gst + c.dp_charge + c.slippage
    )
    assert c.total == pytest.approx(parts, rel=1e-12)


def test_gst_applies_only_to_brokerage_exchange_and_sebi(
    model: IndiaEquityCostModel,
) -> None:
    """Not on STT, not on stamp duty. Getting this wrong inflates every backtest."""
    c = model.charge(
        symbol=SYMBOL, side=Side.BUY, quantity=400, price=PRICE,
        segment=Segment.EQUITY_DELIVERY, sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    rate = model.config.segments[Segment.EQUITY_DELIVERY].gst_pct / 100.0
    expected = (c.brokerage + c.exchange_txn + c.sebi_turnover) * rate
    assert c.gst == pytest.approx(expected, rel=1e-9)


def test_rates_come_from_config_not_python(model: IndiaEquityCostModel) -> None:
    """CLAUDE.md: never hardcode a charge or a fee in Python."""
    doubled = model.with_scaled_rate(Segment.EQUITY_DELIVERY, "stt_sell_pct", 2.0)
    kw = dict(
        symbol=SYMBOL, side=Side.SELL, quantity=400, price=PRICE,
        segment=Segment.EQUITY_DELIVERY, sigma_daily=SIGMA_DAILY, adv_20=ADV_20,
    )
    assert doubled.charge(**kw).stt == pytest.approx(2.0 * model.charge(**kw).stt, rel=1e-9)


def test_config_states_its_provenance(model: IndiaEquityCostModel) -> None:
    """Stale rates silently inflate every backtest, so provenance is mandatory."""
    assert model.config.source, "config must name where its rates came from"
    assert model.config.verified_on, "config must state when they were last verified"
