"""Verdict engine. Reads the gate config, ANDs the gates, defaults to REJECT.

BUILD.md section 7. Three rules that are not negotiable and are enforced here
rather than trusted:

  * **Unknown gate name in config is an error.** A typo must not silently remove a
    gate from the AND and leave a strategy passing a test nobody ran.
  * **A gate that raises fails.** The exception becomes a FAIL with the error in
    the rationale, never a propagated crash and never a skip.
  * **Missing evidence is NOT_COMPUTABLE, and NOT_COMPUTABLE is not a pass.**

Leakage short-circuits everything, per section 5: a fatal flag ends the audit
before any statistic is computed.

``pbo`` is deliberately not a gate. It is an evidence panel and appears on the
report without voting -- see docs/pbo_calibration.md.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml

from null.benchmark.buyhold import BenchmarkEvidence, benchmark_check
from null.contracts import (
    SPEC_VERSION,
    Bar,
    Evidence,
    GateResult,
    LeakageFlag,
    NullModel,
    StrategyRun,
    Verdict,
)
from null.costs.india_equity import IndiaEquityCostModel
from null.leakage.audit import LeakageConfig, LeakageReport, audit_leakage
from null.verdict.gates import GATES, run_gate
from null.verdict.limitations import Limitation, collect_limitations

LEAKAGE_GATE = "leakage_clean"

__all__ = [
    "DEFAULT_GATES_CONFIG",
    "AuditOutcome",
    "run_audit",
    "AuditStage",
    "GateConfigError",
    "VerdictReport",
    "load_gate_config",
    "evaluate",
]

DEFAULT_GATES_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "gates_default.yaml"
)

#: Reported alongside the gates but never counted in the AND.
PANEL_ONLY = frozenset({"pbo", "drawdown_tolerance"})


class GateConfigError(ValueError):
    """The gate config names something the engine cannot run."""


class AuditStage(StrEnum):
    LEAKAGE = "leakage"
    BENCHMARK = "benchmark"
    STATISTICS = "statistics"
    GATES = "gates"


class VerdictReport(NullModel):
    """Everything the renderer needs. The verdict plus what did not vote."""

    verdict: Verdict
    not_computable: tuple[str, ...]
    limitations: tuple[Limitation, ...]
    panels: dict[str, str]
    """Evidence panels: name -> rationale. Reported, never decisive."""


def load_gate_config(path: Path = DEFAULT_GATES_CONFIG) -> dict[str, dict[str, Any]]:
    """Load and validate the gate list. Unknown names are an error."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    gates = raw.get("gates") or {}
    if not gates:
        raise GateConfigError(
            f"{path} declares no gates. An empty gate list would let every strategy "
            "through, which is the opposite of default REJECT."
        )
    unknown = sorted(set(gates) - set(GATES))
    if unknown:
        raise GateConfigError(
            f"{path} names gates the engine cannot run: {unknown}. Known gates are "
            f"{sorted(GATES)}. A typo here would silently drop a gate from the AND, "
            "so it is an error rather than something to skip."
        )
    panel_in_gates = sorted(set(gates) & PANEL_ONLY)
    if panel_in_gates:
        raise GateConfigError(
            f"{path} lists {panel_in_gates} as gate(s), but they are evidence panels "
            "and must not vote. See docs/pbo_calibration.md."
        )
    return {name: dict(cfg or {}) for name, cfg in gates.items()}


def evaluate(
    *,
    run: StrategyRun,
    evidence: Evidence,
    context: dict[str, Any] | None = None,
    config_path: Path = DEFAULT_GATES_CONFIG,
) -> VerdictReport:
    """Run every configured gate and AND the results. Default REJECT."""
    gate_config = load_gate_config(config_path)
    ctx: dict[str, Any] = dict(context or {})

    results: list[GateResult] = [
        run_gate(name, evidence, cfg) for name, cfg in gate_config.items()
    ]

    not_computable = tuple(
        r.name for r in results if r.state == "NOT_COMPUTABLE"
    )
    ctx["not_computable_gates"] = not_computable

    result: Literal["REJECT", "PASS"] = (
        "PASS" if results and all(r.passed for r in results) else "REJECT"
    )

    panels = {
        "pbo": ctx.get("pbo_rationale", "")
        or "PBO was not computed for this run.",
        "drawdown": (
            f"Maximum peak-to-trough decline was {evidence.metrics.max_drawdown:.1%}, "
            f"with the longest underwater period lasting "
            f"{evidence.metrics.longest_underwater_days:,} bars. Reported for context; "
            "it does not vote. Drawdown answers whether you would hold a strategy, "
            "which is a preference about risk appetite, not whether it has an edge. A "
            "35% limit would reject a NIFTY-like buy-and-hold, which draws down around "
            "60% through 2008."
        ),
    }
    if ctx.get("expected_max_sharpe_sentence"):
        panels["expected_max_sharpe"] = str(ctx["expected_max_sharpe_sentence"])
    if ctx.get("mtrl_rationale"):
        panels["mtrl"] = str(ctx["mtrl_rationale"])

    return VerdictReport(
        verdict=Verdict(
            result=result,
            gates=tuple(results),
            evidence_hash=evidence.content_hash(),
            spec_version=SPEC_VERSION,
            generated_from=run,
        ),
        not_computable=not_computable,
        limitations=collect_limitations(evidence, ctx),
        panels=panels,
    )


# ---------------------------------------------------------------------------
# M3 pipeline entry point: leakage short-circuit, then benchmark
# ---------------------------------------------------------------------------
#
# Kept distinct from evaluate(). This path runs the audit from raw bars and stops
# the moment leakage is fatal; evaluate() judges an Evidence that has already been
# assembled. The short-circuit lives here because it is about ORDER -- not
# computing a statistic at all -- while evaluate() is about the AND.


class AuditOutcome(NullModel):
    verdict: Verdict
    leakage_flags: tuple[LeakageFlag, ...]
    leakage_report: LeakageReport
    stages_run: tuple[AuditStage, ...]
    short_circuited: bool
    evidence: BenchmarkEvidence | None
    """``None`` whenever the audit short-circuited. Deliberately absent rather than
    zero-filled: there must be no Sharpe on this object to be tempted by."""


def run_audit(
    *,
    run: StrategyRun,
    bars: tuple[Bar, ...],
    benchmark_bars: tuple[Bar, ...],
    costs: IndiaEquityCostModel,
    leakage_config: LeakageConfig | None = None,
) -> AuditOutcome:
    """Run the audit from bars. Leakage first, and it short-circuits."""
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


def _leakage_gate(report: LeakageReport) -> GateResult:
    """The leakage gate as built by the bars pipeline."""
    from null.verdict.gates import leakage_clean as _lc

    if report.is_clean:
        return GateResult(
            name=LEAKAGE_GATE,
            state="PASS",
            passed=True,
            observed=0.0,
            threshold="no fatal leakage flags",
            rationale=(
                f"No fatal leakage detected across {len(report.checks_run)} checks. "
                f"{len(report.unchecked)} check(s) from BUILD.md section 5 could not "
                "run and are listed as stated limitations; a clean result here is only "
                "as strong as that list is short."
            ),
        )
    fatal = report.fatal
    first = fatal[0]
    return GateResult(
        name=LEAKAGE_GATE,
        state="FAIL",
        passed=False,
        observed=first.kind,
        threshold="no fatal leakage flags",
        rationale=(
            f"Fatal leakage: {len(fatal)} flag(s), the first of kind {first.kind!r}. "
            f"{first.detail} The audit was stopped here and no performance statistic "
            "was computed -- a Sharpe ratio for a strategy that can see the future is "
            "not a weak result, it is a meaningless one, and reporting it would invite "
            "belief in a number that describes nothing."
        ),
    )
