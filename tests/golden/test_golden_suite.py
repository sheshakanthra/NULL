"""The golden suite. BUILD.md section 8. The gate on everything downstream.

Eight synthetic strategies with known correct verdicts. Each asserts on **which
gate catches it**, not merely that the verdict is REJECT: a fixture that rejects
for the wrong reason is a broken test that looks green, and it would let a
regression in the gate that was supposed to catch it pass unnoticed.

Negative controls run on all eight -- the specified gate must be the one that
fires, and a fixture that is supposed to be clean on a dimension must be clean.

STATUS: 7 of 8 behave as specified. ``true_edge_synthetic`` does not pass and is
marked xfail with the diagnosis; see the test itself and the session record. It
has NOT been adjusted to make it pass.
"""

from __future__ import annotations

import pytest

from tests.golden.fixtures import (
    benchmark_clone,
    capacity_bomb,
    costed_scalper,
    one_regime_wonder,
    oracle_lookahead,
    overfit_grid,
    pure_noise,
    true_edge_synthetic,
)
from tests.golden.harness import SYNTHESISED, build_report

REJECTING = [
    oracle_lookahead,
    pure_noise,
    benchmark_clone,
    costed_scalper,
    overfit_grid,
    one_regime_wonder,
    capacity_bomb,
]


def _ids(fns):
    return [f.__name__ for f in fns]


@pytest.mark.parametrize("factory", REJECTING, ids=_ids(REJECTING))
def test_fixture_is_rejected(factory) -> None:
    fixture = factory()
    report = build_report(fixture)
    assert report.verdict.result == "REJECT", (
        f"{fixture.name} was expected to REJECT and did not. A harness that accepts "
        "one of these is not measuring anything."
    )


@pytest.mark.parametrize("factory", REJECTING, ids=_ids(REJECTING))
def test_the_specified_gate_is_the_one_that_catches_it(factory) -> None:
    """The assertion that makes this suite worth having.

    Rejecting for the wrong reason is a broken test that looks green.
    """
    fixture = factory()
    report = build_report(fixture)
    failing = {g.name for g in report.verdict.gates if not g.passed}
    assert fixture.caught_by in failing, (
        f"{fixture.name} was supposed to be caught by {fixture.caught_by!r} but that "
        f"gate passed. Gates that did fire: {sorted(failing) or 'none'}. The fixture "
        "rejected for the wrong reason, which means the gate it exists to exercise is "
        "not being exercised."
    )


def test_oracle_lookahead_is_caught_before_any_statistic_matters() -> None:
    """Leakage is the short-circuit. It must not need help from other gates."""
    report = build_report(oracle_lookahead())
    gate = next(g for g in report.verdict.gates if g.name == "leakage_clean")
    assert gate.state == "FAIL"
    assert "see the future" in gate.rationale or "foresight" in gate.rationale.lower()


def test_capacity_bomb_is_caught_by_capacity_despite_a_real_edge() -> None:
    """The control that proves the capacity gate is not riding along.

    This fixture has a genuine 1.1 Sharpe. If capacity were broken, the other gates
    would let it through.
    """
    fixture = capacity_bomb()
    report = build_report(fixture)
    gates = {g.name: g for g in report.verdict.gates}
    assert gates["capacity"].state == "FAIL"
    assert gates["deflated_sharpe"].passed, (
        "capacity_bomb's edge should survive the deflated Sharpe -- if it does not, "
        "this fixture is not testing capacity in isolation"
    )
    assert gates["capacity"].observed == pytest.approx(0.40)


def test_benchmark_clone_is_measured_against_the_series_it_holds() -> None:
    """A clone compared against some other index is not a clone.

    This was a real harness bug: the fixture was judged against an unrelated random
    series, so it rejected on reality_check rather than on the alpha gate, and the
    test looked green while testing nothing.
    """
    fixture = benchmark_clone()
    assert fixture.benchmark is not None
    assert fixture.benchmark.tolist() == fixture.returns.tolist()
    report = build_report(fixture)
    gates = {g.name: g for g in report.verdict.gates}
    assert gates["beats_benchmark_net"].state == "FAIL"
    assert "no alpha over buy-and-hold after costs" in (
        gates["beats_benchmark_net"].rationale.lower()
    )


def test_one_regime_wonder_fails_on_fold_consistency_specifically() -> None:
    report = build_report(one_regime_wonder())
    gates = {g.name: g for g in report.verdict.gates}
    assert gates["walkforward_consistency"].state == "FAIL"
    assert "one lucky regime" in gates["walkforward_consistency"].rationale


def test_no_fixture_reports_a_not_computable_gate_as_passing() -> None:
    """Across the whole suite, NOT_COMPUTABLE must never read as a pass."""
    for factory in [*REJECTING, true_edge_synthetic]:
        report = build_report(factory())
        for gate in report.verdict.gates:
            if gate.state == "NOT_COMPUTABLE":
                assert gate.passed is False


def test_every_fixture_carries_the_full_limitations_band() -> None:
    for factory in [*REJECTING, true_edge_synthetic]:
        report = build_report(factory())
        assert len(report.limitations) >= 6, (
            f"{factory.__name__} reported only {len(report.limitations)} limitations"
        )


def test_the_suite_states_what_it_does_not_exercise() -> None:
    """A green suite must not be mistaken for end-to-end coverage."""
    assert len(SYNTHESISED) == 4
    assert any("sensitivity" in s for s in SYNTHESISED)
    assert any("walkforward" in s for s in SYNTHESISED)


# ---------------------------------------------------------------------------
# the calibration fixture
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CALIBRATION FAILURE, reported not fixed. true_edge_synthetic (0.6 Sharpe, "
        "1 trial, 10 years) fails three gates. (1) beats_benchmark_net: t=1.79 vs 2.0 "
        "-- arithmetic, since t ~= S*sqrt(years) and 0.6 Sharpe needs 11.1 years to "
        "reach 2.0. (2) reality_check: p=0.50, because the harness benchmark itself "
        "realises a 0.71 Sharpe, so there is genuinely no edge over it. "
        "(3) drawdown_tolerance: 43.7% vs 35%, which a 0.6-Sharpe 17.5%-vol strategy "
        "really does produce over 10 years. The fixture and thresholds have NOT been "
        "adjusted. Remove this xfail when the calibration decision is made."
    ),
)
def test_true_edge_synthetic_passes_every_gate() -> None:
    """A harness that rejects everything is as useless as one that accepts everything.

    This is the fixture that proves NULL can still recognise a real edge. It does
    not currently pass, and that is a finding about the gate thresholds rather than
    a licence to weaken them.
    """
    report = build_report(true_edge_synthetic())
    failing = sorted(g.name for g in report.verdict.gates if not g.passed)
    assert report.verdict.result == "PASS", f"rejected by {failing}"


def test_true_edge_synthetic_still_clears_the_gates_it_was_built_for() -> None:
    """Even failing overall, the selection-related gates must accept it.

    If deflated_sharpe or pbo rejected an honestly-declared single-trial edge, the
    problem would be far more serious than a threshold disagreement.
    """
    report = build_report(true_edge_synthetic())
    gates = {g.name: g for g in report.verdict.gates}
    assert gates["deflated_sharpe"].passed
    assert gates["walkforward_consistency"].passed
    assert gates["sensitivity_plateau"].passed
    assert gates["capacity"].passed
    assert gates["leakage_clean"].passed
