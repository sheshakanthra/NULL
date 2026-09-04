"""Audit orchestration. Default REJECT, and leakage short-circuits everything.

BUILD.md section 5 is unambiguous about ordering: the leakage audit runs **before
any statistics**, and a fatal flag ends the audit immediately. This module is
where that ordering is enforced rather than merely intended.

The short-circuit is not an optimisation. Computing a Sharpe for a strategy that
can see the future produces a number that is both meaningless and persuasive, and
having produced it, someone will look at it. Not computing it is the point.

At M3 this runs leakage, then the benchmark comparison. The full AND-of-all-gates
arrives at M5; ``stages_run`` already records what executed so the short-circuit
stays observable as more stages land.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from null.benchmark.buyhold import BenchmarkEvidence, benchmark_check
from null.contracts import (
    SPEC_VERSION,
    Bar,
    GateResult,
    LeakageFlag,
    NullModel,
    StrategyRun,
    Verdict,
)
from null.costs.india_equity import IndiaEquityCostModel
from null.leakage.audit import LeakageConfig, LeakageReport, audit_leakage

__all__ = ["AuditOutcome", "AuditStage", "run_audit"]

LEAKAGE_GATE = "leakage_clean"


class AuditStage(StrEnum):
    """Pipeline stages, in execution order."""

    LEAKAGE = "leakage"
    BENCHMARK = "benchmark"
    STATISTICS = "statistics"


class AuditOutcome(NullModel):
    verdict: Verdict
    leakage_flags: tuple[LeakageFlag, ...]
    leakage_report: LeakageReport
    stages_run: tuple[AuditStage, ...]
    short_circuited: bool
    evidence: BenchmarkEvidence | None
    """``None`` whenever the audit short-circuited.

    Deliberately absent rather than zero-filled: there must be no Sharpe on this
    object for a reader to be tempted by."""


def _leakage_gate(report: LeakageReport) -> GateResult:
    """The rationale is the product. Name the flag, the symbol, and the consequence."""
    if report.is_clean:
        warnings = [f for f in report.flags if not f.is_fatal]
        detail = (
            f" {len(warnings)} non-fatal warning(s) were raised and are reported "
            "below." if warnings else ""
        )
        return GateResult(
            name=LEAKAGE_GATE,
            passed=True,
            observed=0.0,
            threshold=0.0,
            rationale=(
                f"No fatal leakage detected across {len(report.checks_run)} checks."
                f"{detail} {len(report.unchecked)} check(s) from BUILD.md §5 could not "
                "run and are listed as stated limitations; a clean result here is only "
                "as strong as that list is short."
            ),
        )

    fatal = report.fatal
    first = fatal[0]
    return GateResult(
        name=LEAKAGE_GATE,
        passed=False,
        observed=first.kind,
        threshold="no fatal leakage flags",
        rationale=(
            f"Fatal leakage: {len(fatal)} flag(s), the first of kind {first.kind!r}. "
            f"{first.detail} The audit was stopped here and no performance statistic "
            "was computed — a Sharpe ratio for a strategy that can see the future is "
            "not a weak result, it is a meaningless one, and reporting it would invite "
            "belief in a number that describes nothing."
        ),
    )


def run_audit(
    *,
    run: StrategyRun,
    bars: tuple[Bar, ...],
    benchmark_bars: tuple[Bar, ...],
    costs: IndiaEquityCostModel,
    leakage_config: LeakageConfig | None = None,
) -> AuditOutcome:
    """Run the audit. Leakage first, and it short-circuits."""
    stages: list[AuditStage] = [AuditStage.LEAKAGE]
    report = audit_leakage(run, bars, config=leakage_config)
    gate = _leakage_gate(report)

    if not report.is_clean:
        return AuditOutcome(
            verdict=Verdict(
                result="REJECT",
                gates=(gate,),
                evidence_hash=report.content_hash(),
                spec_version=SPEC_VERSION,
                generated_from=run,
            ),
            leakage_flags=report.flags,
            leakage_report=report,
            stages_run=tuple(stages),
            short_circuited=True,
            evidence=None,
        )

    stages.append(AuditStage.BENCHMARK)
    evidence = benchmark_check(
        run=run, bars=bars, benchmark_bars=benchmark_bars, costs=costs
    )

    gates = (gate, evidence.gate)
    # Default REJECT: PASS only when every gate passed. M5 adds the remaining gates;
    # until they exist a PASS here means "survived what has been built", which is why
    # M6 gates the whole thing.
    result: Literal["REJECT", "PASS"] = (
        "PASS" if all(g.passed for g in gates) else "REJECT"
    )

    return AuditOutcome(
        verdict=Verdict(
            result=result,
            gates=gates,
            evidence_hash=evidence.content_hash(),
            spec_version=SPEC_VERSION,
            generated_from=run,
        ),
        leakage_flags=report.flags,
        leakage_report=report,
        stages_run=tuple(stages),
        short_circuited=False,
        evidence=evidence,
    )
