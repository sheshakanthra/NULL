"""Parameter sensitivity scan, with negative controls. BUILD.md section 6.7."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pytest

from null.sensitivity.neighborhood import (
    DEFAULT_STEPS,
    neighbourhood_offsets,
    scan_neighbourhood,
)
from null.verdict.gates import sensitivity_plateau

THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# the two shapes the gate exists to separate
# ---------------------------------------------------------------------------


def test_a_plateau_passes() -> None:
    """Neighbouring parameters work almost as well. Weak evidence of structure."""

    def flat(offsets: Mapping[str, int]) -> float:
        penalty = 0.02 * sum(abs(v) for v in offsets.values())
        return 1.0 - penalty

    result = scan_neighbourhood(("fast", "slow"), flat)
    assert result.peak_sharpe == pytest.approx(1.0)
    assert result.neighborhood_ratio > THRESHOLD
    assert sensitivity_plateau(
        _stub(result), {"min_neighborhood_ratio": THRESHOLD}
    ).passed


def test_a_spike_fails() -> None:
    """One exact parameter set works and nothing near it does. Curve-fitting."""

    def spike(offsets: Mapping[str, int]) -> float:
        return 1.8 if all(v == 0 for v in offsets.values()) else 0.05

    result = scan_neighbourhood(("fast", "slow"), spike)
    assert result.peak_sharpe == pytest.approx(1.8)
    assert result.neighborhood_ratio < THRESHOLD
    gate = sensitivity_plateau(_stub(result), {"min_neighborhood_ratio": THRESHOLD})
    assert not gate.passed
    assert "curve-fitting" in gate.rationale


def test_the_ratio_moves_monotonically_with_how_sharp_the_peak_is() -> None:
    """Negative control on the mechanism, not just the two extremes."""
    ratios = []
    for decay in (0.0, 0.1, 0.3, 0.6, 0.95):
        def surface(offsets: Mapping[str, int], d: float = decay) -> float:
            return 1.0 * (1.0 - d) ** sum(abs(v) for v in offsets.values())

        ratios.append(scan_neighbourhood(("a",), surface).neighborhood_ratio)
    assert ratios == sorted(ratios, reverse=True), ratios


# ---------------------------------------------------------------------------
# the scan itself
# ---------------------------------------------------------------------------


def test_offsets_include_the_peak_and_walk_each_parameter() -> None:
    offsets = neighbourhood_offsets(("a", "b"))
    assert {"a": 0, "b": 0} in offsets
    # One-at-a-time: 1 peak + 2 params x 4 non-zero steps.
    assert len(offsets) == 1 + 2 * (len(DEFAULT_STEPS) - 1)
    for combo in offsets:
        assert sum(1 for v in combo.values() if v != 0) <= 1


def test_the_full_product_is_available_but_larger() -> None:
    one_at_a_time = neighbourhood_offsets(("a", "b"), one_at_a_time=False)
    assert len(one_at_a_time) == len(DEFAULT_STEPS) ** 2


def test_only_the_immediate_ring_feeds_the_ratio() -> None:
    """The +/-2 ring is reported but the gate reads +/-1, per the spec."""
    seen: list[int] = []

    def record(offsets: Mapping[str, int]) -> float:
        step = sum(abs(v) for v in offsets.values())
        seen.append(step)
        return {0: 1.0, 1: 0.9, 2: 0.1}[step]

    result = scan_neighbourhood(("a",), record)
    assert 2 in seen, "the +/-2 ring should still be probed and reported"
    # Ratio uses only the +/-1 ring, so it is 0.9 rather than an average with 0.1.
    assert result.neighborhood_ratio == pytest.approx(0.9)
    assert len(result.points) == len(DEFAULT_STEPS)


def test_a_non_positive_peak_reports_no_plateau_rather_than_a_large_ratio() -> None:
    """Conservative direction: there is no plateau around a Sharpe of zero."""
    result = scan_neighbourhood(("a",), lambda o: -0.5)
    assert result.neighborhood_ratio == 0.0
    assert not sensitivity_plateau(
        _stub(result), {"min_neighborhood_ratio": THRESHOLD}
    ).passed


def test_a_non_finite_sharpe_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        scan_neighbourhood(("a",), lambda o: float("nan"))


def test_no_parameters_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        neighbourhood_offsets(())


def test_points_carry_their_offsets_for_the_report() -> None:
    result = scan_neighbourhood(("fast", "slow"), lambda o: 1.0)
    peak = next(p for p in result.points if all(v == 0 for v in p.offsets.values()))
    assert set(peak.offsets) == {"fast", "slow"}
    assert all("fast" in p.param_hash or "slow" in p.param_hash for p in result.points)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


def _stub(sensitivity):
    """Minimal stand-in so the gate can be exercised directly on a surface."""

    class _E:
        pass

    e = _E()
    e.sensitivity = sensitivity  # type: ignore[attr-defined]
    return e
