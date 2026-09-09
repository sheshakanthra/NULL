"""The M7 cost-rate sweep harness.

A robustness sweep is a machine for producing reassurance, so the thing most worth
testing is that it can still say no. Two failure modes matter:

  * a case that does not actually scale the component it is named after, which
    would report coverage the run never had, and
  * a summary that calls a case a pass when its own numbers say otherwise.

Neither is visible from the sweep's output -- both produce a clean CSV full of
plausible numbers. These tests run against the scaling and summarising logic
directly, without the 108-variant grid behind it, so they are fast enough to run
every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from examples.rsi2_nifty.cost_robustness import (
    BASELINE,
    JOINT,
    RATE_COMPONENTS,
    SLIPPAGE_COMPONENTS,
    build_model,
    cases,
    summarise,
)
from examples.rsi2_nifty.strategy import ALL_VARIANTS, GridVariant
from null.costs.india_equity import IndiaEquityCostModel
from null.costs.model import Segment, Side

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "costs_india_equity.yaml"

#: Provably inert on this config: brokerage_pct is 0.0, and the model
#: short-circuits to a zero brokerage before the per-order cap's min() is ever
#: reached. Swept anyway, as a negative control on the harness.
INERT = ("brokerage_pct", "brokerage_per_order_cap")

_PROBE = dict(
    symbol="RELIANCE",
    quantity=100.0,
    price=1000.0,
    segment=Segment.EQUITY_DELIVERY,
    sigma_daily=0.018,
    adv_20=5e8,
)


@pytest.fixture(scope="module")
def base() -> IndiaEquityCostModel:
    return IndiaEquityCostModel.from_yaml(CONFIG)


def _legs(model: IndiaEquityCostModel) -> tuple[float, float]:
    return (
        model.charge(side=Side.BUY, **_PROBE).total,
        model.charge(side=Side.SELL, **_PROBE).total,
    )


def test_the_sweep_covers_every_configured_charge_component(
    base: IndiaEquityCostModel,
) -> None:
    """"Every charge component" has to mean every field, not every field someone
    remembered. Reading the model's own field list keeps the two in step."""
    declared = set(RATE_COMPONENTS)
    actual = set(type(base.config.segments[Segment.EQUITY_DELIVERY]).model_fields)
    assert declared == actual, (
        "the sweep's component list has drifted from SegmentRates; "
        f"missing {actual - declared}, stale {declared - actual}"
    )


@pytest.mark.parametrize(
    "component", [c for c in RATE_COMPONENTS + SLIPPAGE_COMPONENTS if c not in INERT]
)
def test_each_swept_component_actually_moves_the_charge(
    base: IndiaEquityCostModel, component: str
) -> None:
    """A case that changes nothing is a case that proves nothing.

    Without this, a typo'd field name or a scaler that silently no-ops would still
    produce a full CSV of cases all reporting 'inversion holds' -- and the sweep
    would be evidence of nothing at all.
    """
    base_legs = _legs(base)
    for factor in (0.75, 1.25):
        scaled = _legs(build_model(base, component, factor))
        moved = [abs(s - b) for s, b in zip(scaled, base_legs)]
        assert max(moved) > 0.0, (
            f"{component}@{factor} left the charge identical; this case measures "
            "nothing"
        )
        direction = 1.0 if factor > 1.0 else -1.0
        for scaled_leg, base_leg in zip(scaled, base_legs):
            if scaled_leg != base_leg:
                assert (scaled_leg - base_leg) * direction > 0.0, (
                    f"{component}@{factor} moved the charge the wrong way"
                )


@pytest.mark.parametrize("component", INERT)
def test_the_negative_control_components_move_nothing(
    base: IndiaEquityCostModel, component: str
) -> None:
    """The other half of the control.

    These two cannot move the charge on this config, and their appearing in the
    sweep with a delta of exactly zero is what shows the sweep is reading the
    config rather than perturbing something of its own. If one of them ever starts
    moving, either the config gained a non-zero brokerage -- in which case they are
    no longer inert and this test should be deleted -- or the scaler is reaching
    somewhere it should not.
    """
    for factor in (0.75, 1.25):
        assert _legs(build_model(base, component, factor)) == _legs(base)


def test_the_joint_case_moves_more_than_any_single_component(
    base: IndiaEquityCostModel,
) -> None:
    joint = _legs(build_model(base, JOINT, 1.25))[1] - _legs(base)[1]
    singles = [
        _legs(build_model(base, c, 1.25))[1] - _legs(base)[1]
        for c in RATE_COMPONENTS + SLIPPAGE_COMPONENTS
    ]
    assert joint > max(singles), "the joint corner must dominate every single move"
    assert joint == pytest.approx(sum(singles), rel=0.05), (
        "the components are near-additive at this notional; a joint case far from "
        "the sum of its parts means one component is being applied twice or not at all"
    )


def test_the_baseline_case_is_the_config_untouched(base: IndiaEquityCostModel) -> None:
    assert _legs(build_model(base, BASELINE, 1.0)) == _legs(base)


def test_every_case_is_listed_exactly_once() -> None:
    listed = cases()
    assert len(set(listed)) == len(listed)
    expected = 1 + 2 * (len(RATE_COMPONENTS) + len(SLIPPAGE_COMPONENTS)) + 2
    assert len(listed) == expected == 25


# ---------------------------------------------------------------------------
# summarise(): the step between a grid run and a row of the CSV.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Result:
    """Just enough of a VariantResult for ``summarise``.

    The variant is a real ``GridVariant``, not a stand-in: ``summarise`` formats its
    parameters and ranks on its ``param_hash``, so a fake would let a change to
    either pass unnoticed here.
    """

    variant: GridVariant
    gross_sharpe: float
    net_sharpe: float


#: Three real grid points, kept distinct so ranks are unambiguous.
A, B, C = ALL_VARIANTS[0], ALL_VARIANTS[1], ALL_VARIANTS[2]


def _grid(*rows: tuple[GridVariant, float, float]) -> tuple:
    return tuple(_Result(v, g, n) for v, g, n in rows)


def test_summarise_reports_the_inversion_when_the_gross_pick_loses_money() -> None:
    result = summarise(
        "baseline",
        1.0,
        _grid((A, 0.967, -0.027), (B, 0.952, 0.422), (C, 0.500, 0.100)),
    )
    assert result.best_gross_hash == A.param_hash
    assert result.best_net_hash == B.param_hash
    assert result.inversion_holds is True
    assert result.naive_pick_loses_money is True
    # Ranked by net: B (0.422), C (0.100), A (-0.027).
    assert result.gross_pick_rank_by_net == 3
    assert result.net_sharpe_given_up_by_selecting_on_gross == pytest.approx(0.449)


def test_summarise_calls_the_inversion_broken_when_the_gross_pick_makes_money() -> None:
    """The stt_buy_pct@0.75 case, in miniature. Still a different variant, no longer
    a losing one -- and only half the claim survives."""
    result = summarise(
        "stt_buy_pct",
        0.75,
        _grid((A, 0.967, 0.055), (B, 0.952, 0.465)),
    )
    assert result.inversion_holds is False
    assert result.naive_pick_loses_money is False
    assert result.best_gross_hash != result.best_net_hash, (
        "the different-variant half is unaffected by the sign of the net Sharpe"
    )
    assert result.gross_pick_rank_by_net == 2


def test_summarise_calls_it_broken_when_one_variant_wins_on_both_measures() -> None:
    result = summarise("x", 1.0, _grid((A, 0.9, -0.1), (B, 0.5, -0.4)))
    assert result.best_gross_hash == result.best_net_hash == A.param_hash
    assert result.naive_pick_loses_money is True
    assert result.inversion_holds is False, "nothing is inverted if one point wins both"
    assert result.gross_pick_rank_by_net == 1


def test_the_rank_breaks_ties_deterministically() -> None:
    """Two variants on identical net Sharpe must not rank by whichever order the
    grid happened to produce them in -- the CSV is a committed artifact."""
    forward = summarise("x", 1.0, _grid((A, 0.9, 0.5), (B, 0.8, 0.5)))
    backward = summarise("x", 1.0, _grid((B, 0.8, 0.5), (A, 0.9, 0.5)))
    assert forward.gross_pick_rank_by_net == backward.gross_pick_rank_by_net
    # A and B tie on net; the winner is whichever param_hash sorts first, not
    # whichever the grid emitted first.
    expected = 1 if A.param_hash < B.param_hash else 2
    assert forward.gross_pick_rank_by_net == expected
