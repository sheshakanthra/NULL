"""M4 acceptance test -- BUILD.md sections 6 and 8.

Three fixtures, three known verdicts:

  overfit_grid          REJECT on deflated_sharpe (PBO was demoted to a panel --
                        see docs/pbo_calibration.md)
  pure_noise            REJECT
  true_edge_synthetic   PASS

The last one carries the most weight. A harness that rejects everything is
exactly as useless as one that accepts everything. If true_edge_synthetic cannot
pass, the thresholds are miscalibrated and real edges get discarded -- and the fix
is to report that, never to tune the fixture until it goes green.
"""

from __future__ import annotations

import numpy as np
import pytest

from null.stats.deflated_sharpe import deflated_sharpe_ratio
from null.stats.pbo import compute_pbo
from tests.golden.fixtures import overfit_grid, pure_noise, true_edge_synthetic

DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5


@pytest.fixture(scope="module")
def overfit():
    return overfit_grid()


@pytest.fixture(scope="module")
def noise():
    return pure_noise()


@pytest.fixture(scope="module")
def edge():
    return true_edge_synthetic()


def _dsr(fixture):
    return deflated_sharpe_ratio(
        returns=fixture.returns,
        n_trials=fixture.n_trials,
        trial_sharpes=fixture.trial_sharpes,
    )


# ---------------------------------------------------------------------------
# overfit_grid -- must fail on two independent grounds
# ---------------------------------------------------------------------------


def test_overfit_grid_fails_deflated_sharpe(overfit) -> None:
    result = _dsr(overfit)
    assert result.deflated_sharpe < DSR_THRESHOLD, (
        f"overfit_grid deflated to {result.deflated_sharpe:.4f} from an observed "
        f"Sharpe of {result.observed_sharpe_annual:.2f} over {result.n_trials} "
        "declared trials, and still cleared the threshold. Selection bias is not "
        "being penalised."
    )


def test_overfit_grid_pbo_is_reported_as_a_panel_not_a_gate(overfit) -> None:
    """PBO was demoted (docs/pbo_calibration.md). Assert the panel, not a verdict.

    The number is still produced and still printed. It simply does not vote, and
    the result object has no ``passed`` field to be misread as one.
    """
    from null.stats.pbo import LOW_PBO_CAVEAT, PBOResult

    result = compute_pbo(overfit.trial_returns, n_trials=overfit.n_trials)
    assert result.state == "computed"
    assert 0.0 <= result.pbo_lower <= result.pbo_upper <= 1.0
    assert LOW_PBO_CAVEAT in result.rationale
    assert "does not vote" in result.rationale
    assert "passed" not in PBOResult.model_fields, (
        "PBOResult grew a passed field; PBO is an evidence panel and must not vote"
    )


def test_deflated_sharpe_rejects_overfit_grid_on_every_seed_tested() -> None:
    """What PBO cannot currently do, DSR does robustly.

    The half of the overfit_grid acceptance that actually holds. Kept as a sweep
    rather than a single seed precisely because the PBO half did not survive one.
    """
    from tests.golden.fixtures import overfit_grid as build

    for seed in (20260905, 1, 2, 7, 42, 99, 12345):
        f = build(seed=seed)
        result = deflated_sharpe_ratio(
            returns=f.returns, n_trials=f.n_trials, trial_sharpes=f.trial_sharpes
        )
        assert result.deflated_sharpe < DSR_THRESHOLD, (
            f"seed {seed}: overfit_grid deflated to {result.deflated_sharpe:.4f}"
        )


def test_true_edge_passes_deflated_sharpe_on_every_seed_tested() -> None:
    """The calibration fixture must not be a one-seed accident either."""
    from tests.golden.fixtures import true_edge_synthetic as build

    for seed in (20260907, 1, 2, 7, 42, 99, 12345):
        result = deflated_sharpe_ratio(returns=build(seed=seed).returns, n_trials=1)
        assert result.deflated_sharpe > DSR_THRESHOLD, (
            f"seed {seed}: true_edge deflated to {result.deflated_sharpe:.4f}"
        )


# ---------------------------------------------------------------------------
# pure_noise
# ---------------------------------------------------------------------------


def test_pure_noise_fails_deflated_sharpe(noise) -> None:
    result = _dsr(noise)
    assert result.deflated_sharpe < DSR_THRESHOLD, (
        f"random entries produced DSR {result.deflated_sharpe:.4f}"
    )


# ---------------------------------------------------------------------------
# true_edge_synthetic -- the calibration fixture
# ---------------------------------------------------------------------------


def test_true_edge_passes_deflated_sharpe(edge) -> None:
    result = _dsr(edge)
    assert result.deflated_sharpe > DSR_THRESHOLD, (
        f"true_edge_synthetic deflated to {result.deflated_sharpe:.4f} against a "
        f"threshold of {DSR_THRESHOLD}, from an observed annual Sharpe of "
        f"{result.observed_sharpe_annual:.3f} over {result.n_obs} observations with "
        f"n_trials={result.n_trials}. A genuine, honestly-declared edge is being "
        "thrown away. Report this -- do not adjust the fixture."
    )


def test_true_edge_pbo_panel_reports_not_applicable(edge) -> None:
    """One trial means no selection, so there is nothing to have overfit."""
    result = compute_pbo(edge.trial_returns, n_trials=edge.n_trials)
    assert result.state == "not_applicable"


# ---------------------------------------------------------------------------
# separation -- the two must be far apart, not merely on opposite sides
# ---------------------------------------------------------------------------


def test_deflation_separates_the_two_fixtures_by_a_wide_margin(overfit, edge) -> None:
    """A threshold that only just separates them is a threshold that will misfire."""
    over = _dsr(overfit)
    good = _dsr(edge)
    assert good.deflated_sharpe - over.deflated_sharpe > 0.5, (
        f"separation is only {good.deflated_sharpe - over.deflated_sharpe:.4f}: "
        f"overfit_grid {over.deflated_sharpe:.4f} vs true_edge "
        f"{good.deflated_sharpe:.4f}"
    )


def test_rationale_shows_the_deflation_explicitly(overfit) -> None:
    """The number alone teaches nobody. BUILD.md §6.1 wants the arithmetic shown."""
    text = _dsr(overfit).rationale.lower()
    assert "observed" in text and "deflated" in text
    assert str(overfit.n_trials) in text or "5,000" in text
    assert "trial" in text
