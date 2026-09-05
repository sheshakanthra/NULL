"""White's Reality Check and its negative controls. BUILD.md section 6.3.

The controls here answer a specific question posed after PBO was demoted: does
the Reality Check suffer the same single-path limitation? It does not, and the
asymmetry is what matters.

PBO's failure was a false PASS -- it let noise through, producing a persuasive
row that carried no evidence. The Reality Check's worst behaviour is the
opposite: on a weak genuine edge it fails to reject the null, which under a
``p < 0.05`` gate rejects the strategy. Errs toward REJECT, which is the safe
direction here, and never toward false confidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from null.stats.reality_check import reality_check

SEED = 20260905


def _setup(seed, n_candidates, edge=0.0, n_obs=1008):
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0003, 0.011, n_obs)
    candidates = rng.normal(0.0, 0.011, (n_obs, n_candidates)) + benchmark[:, None]
    if edge:
        candidates[:, 0] += edge
    return candidates, benchmark


# ---------------------------------------------------------------------------
# negative controls: it must never pass noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_candidates", [1, 5, 20, 200])
def test_no_edge_is_never_declared_significant(n_candidates: int) -> None:
    """The control that matters. Zero false rejections is the requirement.

    Measured over 24 cases (4 candidate counts x 6 seeds) at build time: 0
    rejections. Unlike PBO, this statistic does not fail toward false confidence.
    """
    for seed in (1, 2, 7, 42, 99, 12345):
        cand, bench = _setup(seed, n_candidates)
        result = reality_check(cand, bench, n_bootstrap=400, seed=seed)
        assert result.p_value >= 0.05, (
            f"{n_candidates} candidates, seed {seed}: p={result.p_value:.4f} declared "
            "a non-existent edge significant"
        )


def test_a_large_edge_is_detected_regardless_of_candidate_count() -> None:
    """The other half: a gate that never fires is worth nothing."""
    for seed in (1, 2, 7, 42, 99, 12345):
        cand, bench = _setup(seed, 200, edge=0.0030)
        assert reality_check(cand, bench, n_bootstrap=400, seed=seed).p_value < 0.05


def test_power_degrades_with_a_larger_candidate_set() -> None:
    """More candidates means a higher bar, which is the point of the test."""
    edge = 0.0012
    few = [
        reality_check(*_setup(s, 5, edge=edge), n_bootstrap=400, seed=s).p_value
        for s in (1, 2, 7, 42)
    ]
    many = [
        reality_check(*_setup(s, 500, edge=edge), n_bootstrap=400, seed=s).p_value
        for s in (1, 2, 7, 42)
    ]
    assert float(np.mean(many)) > float(np.mean(few))


# ---------------------------------------------------------------------------
# block resampling must be doing something
# ---------------------------------------------------------------------------


def test_block_resampling_recovers_the_hac_variance_that_iid_understates() -> None:
    """The property that justifies the whole stationary-bootstrap dependency.

    Asserting a DIRECTION on the p-value would be meaningless: p moves toward 0.5
    as the null distribution widens, so its direction depends on the sign of the
    observed statistic, not on the resampling scheme. The variance of the
    bootstrap sample mean is the real claim, and it is checkable against theory.

    Measured on AR(1) residuals with rho = 0.8, T = 1008:

        L = 1   (IID)        sd 0.000402   vs IID formula      0.000397
        L = 32  (sqrt T)     sd 0.001181   vs HAC-adjusted     0.001192
        L = 300              sd 0.000756

    IID reproduces the naive standard error and so understates the truth by a
    factor of three. Blocks at sqrt(T) recover the HAC value to within 1%.
    Over-long blocks shrink it again, because the circular wrap starts
    reproducing the original series.
    """
    from null.stats.bootstrap import stationary_bootstrap_indices

    n, rho = 1008, 0.8
    rng = np.random.default_rng(SEED)
    eps = rng.normal(0, 0.008, n)
    f = np.zeros(n)
    for i in range(1, n):
        f[i] = rho * f[i - 1] + eps[i]

    def boot_sd(length: float) -> float:
        r = np.random.default_rng(5)
        means = [
            f[stationary_bootstrap_indices(n, rng=r, mean_block_length=length)].mean()
            for _ in range(2000)
        ]
        return float(np.std(means))

    iid_formula = float(f.std(ddof=1) / np.sqrt(n))
    hac_expected = iid_formula * np.sqrt((1 + rho) / (1 - rho))

    sd_iid = boot_sd(1.0)
    sd_blocks = boot_sd(float(np.sqrt(n)))

    assert sd_iid == pytest.approx(iid_formula, rel=0.15), (
        "IID resampling should reproduce the naive standard error"
    )
    assert sd_blocks == pytest.approx(hac_expected, rel=0.25), (
        f"block resampling gave sd {sd_blocks:.6f}, not the HAC-adjusted "
        f"{hac_expected:.6f}; the correction is not working"
    )
    assert sd_blocks > sd_iid * 2.0, (
        "blocks must recover materially more variance than IID, or the p-values "
        "they feed are no better than the optimistic ones they replace"
    )


def test_default_block_length_is_sqrt_t() -> None:
    cand, bench = _setup(SEED, 5)
    result = reality_check(cand, bench, n_bootstrap=200, seed=SEED)
    assert result.mean_block_length == pytest.approx(np.sqrt(1008), rel=1e-9)


# ---------------------------------------------------------------------------
# structure and guards
# ---------------------------------------------------------------------------


def test_p_value_is_a_probability_and_result_is_reproducible() -> None:
    cand, bench = _setup(SEED, 20, edge=0.001)
    a = reality_check(cand, bench, n_bootstrap=300, seed=3)
    b = reality_check(cand, bench, n_bootstrap=300, seed=3)
    assert 0.0 <= a.p_value <= 1.0
    assert a.p_value == b.p_value


def test_a_single_candidate_is_accepted() -> None:
    cand, bench = _setup(SEED, 1)
    assert reality_check(cand[:, 0], bench, n_bootstrap=200, seed=1).n_candidates == 1


def test_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        reality_check(np.zeros((3, 2)), np.zeros(3))


def test_rationale_names_the_block_length_and_the_iid_hazard() -> None:
    cand, bench = _setup(SEED, 10, edge=0.001)
    text = reality_check(cand, bench, n_bootstrap=200, seed=1).rationale
    assert "mean block" in text
    assert "sqrt(T)" in text
    assert "IID" in text
