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
    """Unverified rates, stated precisely rather than as a blanket disclaimer.

    The old wording said every cost figure was "indicative only". That is true of
    cost LEVELS and useless as a disclosure: it warns the reader that something
    might be wrong without telling them which conclusions actually depend on it, so
    a reader either ignores the whole report or ignores the whole band. Neither is
    what a disclosure is for.

    What the rates being unverified does and does not touch is separable:

      * a rupee cost, a cost-drag percentage, a net Sharpe LEVEL -- carries the
        rate error directly, and nothing here fixes that;
      * a RANKING of two variants priced under the same rates -- survives a rate
        error unless the ranking is thin relative to it, which is a measurable
        question, not a rhetorical one.

    So the band states the provenance, separates those two, and then reports
    whether the sensitivity was actually measured for this run. Severity stays
    "blocking" in both branches: a measured sweep shows which conclusions survive
    the unverified rates, it does not make the rates verified, and the levels on
    the report are still unconfirmed either way.
    """
    if context.get("rates_are_verified", False):
        return None

    provenance = (
        "Charge rates are taken from the published Indian equity charge stack and "
        "have NOT been reconciled against a live broker contract note "
        "(configs/costs_india_equity.yaml carries _verified_on: UNVERIFIED). Cost "
        "LEVELS on this report -- every rupee figure, every cost-drag percentage, "
        "every net Sharpe -- carry that error directly. A RANKING between variants "
        "priced under the same rates does not automatically, but whether a "
        "particular ranking survives a plausible rate error is a measurable "
        "question rather than an assumption. "
    )

    measured = context.get("cost_rate_robustness")
    if isinstance(measured, str) and measured.strip():
        return Limitation(
            key="unverified_cost_rates",
            severity="blocking",
            text=provenance + measured.strip(),
        )

    return Limitation(
        key="unverified_cost_rates",
        severity="blocking",
        text=(
            provenance
            + "It was NOT measured for this run: no cost-rate sensitivity sweep was "
            "supplied, so it is not known which conclusions here would survive the "
            "rates being wrong and which would not. Treat every cost-dependent "
            "result as unestablished rather than merely imprecise."
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
            "The benchmark series has NOT been confirmed as a total-return index. If "
            "it is the NIFTY price index, the alpha above is overstated by roughly "
            "1.35%/yr: NSE reports 11.09% annualised for the price index against "
            "12.44% for total return over the 20 years to February 2026, and that gap "
            "is dividends the price index omits. NULL never substitutes the price "
            "index on its own -- a missing TRI raises rather than falling back -- so "
            "this warning means a caller supplied one."
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


@register
def _synthesised_evidence(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    fields = context.get("synthesised_evidence_fields", ())
    if not isinstance(fields, (list, tuple)) or not fields:
        return None
    listed = "; ".join(str(f) for f in fields)
    return Limitation(
        key="synthesised_evidence",
        severity="blocking",
        text=(
            f"{len(fields)} field(s) of the Evidence behind this verdict were SUPPLIED "
            f"rather than computed by the pipeline: {listed}. The gates ran on those "
            "values honestly, but a gate is only as good as its input, and these inputs "
            "did not come from measuring the strategy. A green result here is not "
            "end-to-end evidence."
        ),
    )


@register
def _reviewed_market_events(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    count = context.get("reviewed_market_events", 0)
    if not isinstance(count, int) or count <= 0:
        return None
    return Limitation(
        key="reviewed_market_events",
        severity="stated",
        text=(
            f"{count} large single-day price moves were accepted as genuine market "
            "events by human review rather than being matched to a corporate action. "
            "Each carries a checkable reason in configs/reviewed_market_events.csv "
            "and is visible in git history. A human judged these; they were not "
            "waived by configuration."
        ),
    )


@register
def _symbol_continuity(
    evidence: Evidence, context: dict[str, object]
) -> Limitation | None:
    affected = context.get("symbols_without_corporate_actions", ())
    if not isinstance(affected, (list, tuple)) or not affected:
        return None
    return Limitation(
        key="symbol_continuity",
        severity="blocking",
        text=(
            f"Symbol continuity across corporate restructuring is UNHANDLED, and "
            f"{len(affected)} of the universe's symbols are affected: "
            f"{', '.join(sorted(str(s) for s in affected))}. The corporate action "
            "source keys on current symbols, so filings made under a prior identity "
            "do not resolve. For an affected name the price series is NOT adjusted "
            "for its historical splits, and any strategy trading it is being measured "
            "across raw price discontinuities."
        ),
    )


def collect_limitations(
    evidence: Evidence, context: dict[str, object]
) -> tuple[Limitation, ...]:
    """Every limitation that currently applies, in registration order."""
    found = [d(evidence, context) for d in _REGISTRY]
    return tuple(x for x in found if x is not None)
