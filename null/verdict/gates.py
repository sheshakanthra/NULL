"""Every gate is a pure function ``fn(Evidence, config) -> GateResult``.

No I/O, no state, no globals. Thresholds arrive from ``configs/gates_default.yaml``
and never appear here as literals.

Three outcomes, and the third is the one that needs care:

  PASS            the gate ran and the strategy cleared it
  FAIL            the gate ran and the strategy did not
  NOT_COMPUTABLE  the gate could not run, because the evidence it needs is absent

NOT_COMPUTABLE is not a pass. "We could not check" and "we checked and it failed"
are different claims, and a reader has to be able to tell them apart -- but neither
lets a strategy through. A gate that cannot run fails closed (CLAUDE.md invariant
6: never add a fallback that lets a strategy through on missing evidence).
"""

from __future__ import annotations

from typing import Any, Callable

from null.contracts import Evidence, GateResult

__all__ = ["GATES", "GateFn", "not_computable", "run_gate"]

GateFn = Callable[[Evidence, dict[str, Any]], GateResult]


def not_computable(name: str, reason: str) -> GateResult:
    """The evidence this gate needs is absent. Fails closed, distinctly."""
    return GateResult(
        name=name,
        state="NOT_COMPUTABLE",
        passed=False,
        observed="not computable",
        threshold="n/a",
        rationale=(
            f"Not computable: {reason} This gate did not judge the strategy either "
            "way. It does not count as a pass -- a strategy cannot clear a test that "
            "was never run -- but it is reported separately from a genuine failure so "
            "the difference stays visible."
        ),
    )


def _decide(
    name: str, passed: bool, observed: float | str, threshold: float | str, rationale: str
) -> GateResult:
    return GateResult(
        name=name,
        state="PASS" if passed else "FAIL",
        passed=passed,
        observed=observed,
        threshold=threshold,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------


def leakage_clean(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    fatal = [f for f in evidence.leakage_flags if f.is_fatal]
    warnings = [f for f in evidence.leakage_flags if not f.is_fatal]
    if fatal:
        first = fatal[0]
        return _decide(
            "leakage_clean",
            False,
            first.kind,
            "no fatal leakage flags",
            f"Fatal leakage: {len(fatal)} flag(s), the first of kind {first.kind!r}. "
            f"{first.detail} No performance statistic was computed downstream of this "
            "-- a Sharpe ratio for a strategy that can see the future is not a weak "
            "result, it is a meaningless one.",
        )
    return _decide(
        "leakage_clean",
        True,
        0.0,
        "no fatal leakage flags",
        f"No fatal leakage detected. {len(warnings)} non-fatal warning(s) raised. "
        "A clean result here is only as strong as the list of checks that could "
        "actually run; see the limitations band.",
    )


def beats_benchmark_net(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["min"])
    observed = evidence.alpha.alpha_tstat
    se = (
        f"Newey-West, {evidence.alpha.hac_lags} lags"
        if evidence.alpha.se_method == "newey_west"
        else "OLS (not autocorrelation-robust)"
    )
    passed = observed >= threshold
    lead = (
        "Alpha over risk-matched buy-and-hold survives"
        if passed
        else "No alpha over buy-and-hold after costs"
    )
    return _decide(
        "beats_benchmark_net",
        passed,
        observed,
        threshold,
        f"{lead}. Strategy returned {evidence.metrics.cagr:.2%} CAGR net of "
        f"everything against {evidence.benchmark_metrics.cagr:.2%} for the "
        f"benchmark, beta {evidence.alpha.beta:.2f}. Annualised alpha "
        f"{evidence.alpha.alpha_annual:.2%} with a t-stat of {observed:.2f} ({se}) "
        f"over {evidence.alpha.n_obs:,} observations; threshold is {threshold:.1f}, "
        "and alpha with a t-stat below 2 is not alpha.",
    )


def deflated_sharpe(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["min"])
    observed = evidence.deflated_sharpe
    passed = observed > threshold
    return _decide(
        "deflated_sharpe",
        passed,
        observed,
        threshold,
        f"The probability the true Sharpe exceeds zero, after adjusting for how many "
        f"variants were tried and for the shape of the return distribution, is "
        f"{observed:.2f} against a threshold of {threshold:.2f}. "
        + (
            "The observed result survives the deflation."
            if passed
            else "Adjusted for selection, this is a search finding noise."
        ),
    )


def reality_check(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["max_p"])
    observed = evidence.reality_check_p
    passed = observed < threshold
    return _decide(
        "reality_check",
        passed,
        observed,
        threshold,
        f"Bootstrapping the null that the best candidate does not beat the benchmark, "
        f"using a stationary bootstrap so autocorrelation survives resampling, gives "
        f"p = {observed:.3f} against a threshold of {threshold:.3f}. "
        + (
            "The outperformance is larger than resampling the same history produces."
            if passed
            else "The outperformance is within what chance produces on this history."
        ),
    )


def walkforward_consistency(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["min_fold_win_rate"])
    if not evidence.walkforward:
        return not_computable(
            "walkforward_consistency",
            "no walk-forward folds were supplied, so per-fold consistency cannot be "
            "measured.",
        )
    positive = sum(1 for f in evidence.walkforward if f.net_return > 0.0)
    observed = positive / len(evidence.walkforward)
    passed = observed >= threshold
    return _decide(
        "walkforward_consistency",
        passed,
        observed,
        threshold,
        f"Net-positive after costs in {positive} of {len(evidence.walkforward)} "
        f"out-of-sample folds ({observed:.0%}) against a threshold of "
        f"{threshold:.0%}. "
        + (
            "The result is spread across folds rather than carried by one."
            if passed
            else "A strategy carried by one fold is one lucky regime, not an edge."
        ),
    )


def sensitivity_plateau(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["min_neighborhood_ratio"])
    if len(evidence.sensitivity.points) <= 1:
        # Only the peak was supplied, so there is no neighbourhood to judge. That is
        # not a spike -- calling it curve-fitting would be an accusation the evidence
        # does not support -- and it is not a pass either.
        return not_computable(
            "sensitivity_plateau",
            "no parameter neighbourhood was supplied, only the submitted point, so "
            "whether the result is a plateau or a spike cannot be determined. Run the "
            "neighbourhood scan (null/sensitivity/neighborhood.py) and supply the "
            "surface.",
        )
    observed = evidence.sensitivity.neighborhood_ratio
    passed = observed >= threshold
    return _decide(
        "sensitivity_plateau",
        passed,
        observed,
        threshold,
        f"The mean Sharpe of the immediate parameter neighbourhood is {observed:.0%} "
        f"of the peak ({evidence.sensitivity.neighborhood_mean_sharpe:.2f} against "
        f"{evidence.sensitivity.peak_sharpe:.2f}), threshold {threshold:.0%}. "
        + (
            "A plateau is weak evidence of structure."
            if passed
            else "A spike is curve-fitting: the result depends on the exact parameters "
            "chosen, and neighbouring values do not work."
        ),
    )


def capacity(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["max_adv_participation"])
    observed = evidence.max_adv_participation
    if observed is None:
        return not_computable(
            "capacity",
            "max_adv_participation was not supplied, so the share of daily traded "
            "volume this strategy would consume cannot be checked. It is computable "
            "from weight changes and Bar.adv_20 during evidence build.",
        )
    passed = observed <= threshold
    return _decide(
        "capacity",
        passed,
        observed,
        threshold,
        f"The largest single order would consume {observed:.1%} of 20-day average "
        f"daily traded value, against a tolerance of {threshold:.1%}. "
        + (
            "The strategy fits in the liquidity available to it."
            if passed
            else "An order this size does not execute at the prices the backtest "
            "assumed; the returns shown are not attainable at this capital."
        ),
    )


def drawdown_tolerance(evidence: Evidence, config: dict[str, Any]) -> GateResult:
    threshold = float(config["max_dd"])
    observed = evidence.metrics.max_drawdown
    passed = observed <= threshold
    return _decide(
        "drawdown_tolerance",
        passed,
        observed,
        threshold,
        f"Maximum peak-to-trough decline was {observed:.1%} against a tolerance of "
        f"{threshold:.1%}, with the longest underwater period lasting "
        f"{evidence.metrics.longest_underwater_days:,} bars. "
        + (
            "Within tolerance."
            if passed
            else "A drawdown this deep is not survivable in practice regardless of "
            "the eventual recovery."
        ),
    )


#: The only gate names the engine accepts. An unknown name in config is an error,
#: not something to skip -- a typo must not silently remove a gate from the AND.
GATES: dict[str, GateFn] = {
    "leakage_clean": leakage_clean,
    "beats_benchmark_net": beats_benchmark_net,
    "deflated_sharpe": deflated_sharpe,
    "reality_check": reality_check,
    "walkforward_consistency": walkforward_consistency,
    "sensitivity_plateau": sensitivity_plateau,
    "capacity": capacity,
    "drawdown_tolerance": drawdown_tolerance,
}


def run_gate(name: str, evidence: Evidence, config: dict[str, Any]) -> GateResult:
    """Run one gate. Anything it raises becomes a failure, never an exception."""
    fn = GATES[name]
    try:
        return fn(evidence, config)
    except Exception as exc:  # noqa: BLE001 - a raising gate must fail, not propagate
        return GateResult(
            name=name,
            state="FAIL",
            passed=False,
            observed="error",
            threshold="n/a",
            rationale=(
                f"The {name} gate raised {type(exc).__name__}: {exc}. A gate that "
                "errors fails. It is reported as a failure rather than a missing "
                "result so that a broken gate can never be mistaken for a clean one."
            ),
        )
