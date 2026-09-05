"""The disclosure registry. Every stated limitation, collected in one place.

Limitations are *derived from evidence*, not hand-written per report. A new
limitation is added by registering a detector here, and it then appears on every
affected report automatically. Nobody has to remember to mention it, which is the
only way a disclosure survives contact with a deadline.

BUILD.md and CLAUDE.md both require the chosen data compromises to be printed on
every report. This is where that promise is kept.
"""

from __future__ import annotations

from typing import Callable

from null.contracts import Evidence, NonEmptyStr, NullModel

__all__ = ["Limitation", "collect_limitations", "register"]


class Limitation(NullModel):
    key: NonEmptyStr
    severity: NonEmptyStr
    """'blocking' means a number on this report is known to be wrong or unverified."""
    text: NonEmptyStr


Detector = Callable[[Evidence, dict[str, object]], Limitation | None]

_REGISTRY: list[Detector] = []


def register(detector: Detector) -> Detector:
    _REGISTRY.append(detector)
    return detector


@register
def _unverified_cost_rates(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    if context.get("rates_are_verified", False):
        return None
    return Limitation(
        key="unverified_cost_rates",
        severity="blocking",
        text=(
            "Charge rates have NOT been reconciled against a live broker charge list "
            "(configs/costs_india_equity.yaml carries _verified_on: UNVERIFIED). Every "
            "cost figure on this report is indicative only, and rates wrong in the "
            "optimistic direction inflate every result shown here."
        ),
    )


@register
def _benchmark_series_unconfirmed(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    if context.get("benchmark_is_total_return", False):
        return None
    return Limitation(
        key="benchmark_series",
        severity="blocking",
        text=(
            "The benchmark series is supplied by the caller and NULL does not select "
            "it. It has not been confirmed as a total-return index. If it is a price "
            "index, roughly 1.2-1.5%/yr of dividends are missing from the benchmark "
            "and every alpha figure here is overstated by that much."
        ),
    )


@register
def _survivorship(evidence: Evidence, context: dict[str, object]) -> Limitation | None:
    if context.get("universe_is_point_in_time", False):
        return None
    return Limitation(
        key="survivorship",
        severity="blocking",
        text=(
            "The universe is NOT point-in-time. No index-membership source is wired "
            "up, so symbols that were not constituents on a given date cannot be "
            "detected and delisted names may be silently absent. A survivorship-biased "
            "universe inflates returns."
        ),
    )


@register
def _unchecked_leakage(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    unchecked = context.get("leakage_checks_unchecked", ())
    count = len(unchecked) if isinstance(unchecked, (list, tuple)) else 0
    if count == 0:
        return None
    return Limitation(
        key="unchecked_leakage",
        severity="blocking",
        text=(
            f"{count} of the leakage checks in BUILD.md section 5 could not run and "
            "were not evaluated: point-in-time constituency, universe rebalance "
            "timing, delisting terminal values, corporate-action confirmation, and "
            "NaN forward-fill detection. A clean leakage result is only as strong as "
            "this list is short."
        ),
    )


@register
def _risk_free_assumed(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    if context.get("risk_free_supplied", False):
        return None
    return Limitation(
        key="risk_free",
        severity="stated",
        text=(
            "Risk-free rate assumed to be zero; NULL has no risk-free series. Beta is "
            "unaffected; alpha is shifted by (1 - beta) times the true rate."
        ),
    )


@register
def _not_computable_gates(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    names = context.get("not_computable_gates", ())
    if not isinstance(names, (list, tuple)) or not names:
        return None
    listed = ", ".join(sorted(str(n) for n in names))
    return Limitation(
        key="not_computable_gates",
        severity="blocking",
        text=(
            f"{len(names)} gate(s) could not be evaluated at all and did not judge "
            f"this strategy either way: {listed}. They are counted as failures, not "
            "passes, but the strategy has not actually been tested against them."
        ),
    )


@register
def _golden_suite_incomplete(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    if context.get("golden_suite_green", False):
        return None
    return Limitation(
        key="golden_suite",
        severity="blocking",
        text=(
            "The golden fixture suite is not complete. Until all eight fixtures in "
            "BUILD.md section 8 return their expected verdicts, a REJECT from NULL "
            "may be describing the harness rather than the strategy, and the two "
            "cannot be told apart from this report."
        ),
    )


def collect_limitations(
    evidence: Evidence, context: dict[str, object]
) -> tuple[Limitation, ...]:
    """Every limitation that currently applies, in registration order."""
    found = [d(evidence, context) for d in _REGISTRY]
    return tuple(x for x in found if x is not None)
