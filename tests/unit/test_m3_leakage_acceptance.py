"""M3 acceptance test -- BUILD.md section 5.

    If this fails, short-circuit to REJECT immediately. Do not compute Sharpe on a
    strategy that can see the future -- you'll be tempted to believe the number.

The verdict is the easy half. This module asserts on the **short-circuit itself**:
a REJECT reached after computing a Sharpe is a different bug wearing the right
answer, and it would look identical from the outside.

Three independent ways of checking that, because one is not enough:

  1. ``stages_run`` records what actually executed, and must contain leakage alone.
  2. The outcome carries no evidence -- there is no Sharpe to be tempted by.
  3. The statistics entry points are monkeypatched to raise. If the pipeline
     touches them, the test explodes rather than quietly passing.

Check 3 is the one that cannot be faked by bookkeeping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from null.contracts import Bar, StrategyRun, TargetWeight
from null.costs.india_equity import IndiaEquityCostModel
from null.verdict.engine import AuditStage, run_audit
from tests.golden.fixtures import oracle_lookahead_run as oracle_lookahead

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "costs_india_equity.yaml"
IST = timezone(timedelta(hours=5, minutes=30))

SEED = 20260905
N_DAYS = 500
SYMBOL = "ORACLE"


def _bars() -> tuple[Bar, ...]:
    rng = np.random.default_rng(SEED)
    daily = rng.normal(0.0004, 0.011, N_DAYS)
    level = 1000.0 * np.cumprod(1.0 + daily)
    start = datetime(2021, 1, 1, 15, 30, tzinfo=IST)
    out: list[Bar] = []
    for i, close in enumerate(level):
        prev = 1000.0 if i == 0 else float(level[i - 1])
        out.append(
            Bar(
                ts=start + timedelta(days=i),
                symbol=SYMBOL,
                open=prev,
                high=max(prev, float(close)) * 1.002,
                low=min(prev, float(close)) * 0.998,
                close=float(close),
                volume=1e6,
                adv_20=5e8,
            )
        )
    return tuple(out)


def honest_strategy(bars: tuple[Bar, ...]) -> StrategyRun:
    """A control. Seeded random weights, no foresight, same shape."""
    rng = np.random.default_rng(SEED + 1)
    weights = tuple(
        TargetWeight(
            ts=b.ts, symbol=SYMBOL, weight=float(rng.integers(0, 2))
        )
        for b in bars[:-1]
    )
    return StrategyRun(
        strategy_id="honest_control",
        param_hash="honest",
        n_trials=1,
        universe=(SYMBOL,),
        weights=weights,
        decision_lag_bars=1,
        initial_capital=1_000_000.0,
    )


@pytest.fixture(scope="module")
def bars() -> tuple[Bar, ...]:
    return _bars()


@pytest.fixture(scope="module")
def costs() -> IndiaEquityCostModel:
    return IndiaEquityCostModel.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="module")
def outcome(bars, costs):
    return run_audit(
        run=oracle_lookahead(bars), bars=bars, benchmark_bars=bars, costs=costs
    )


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------


def test_oracle_lookahead_is_rejected(outcome) -> None:
    assert outcome.verdict.result == "REJECT"


def test_leakage_gate_failed_and_names_the_lookahead(outcome) -> None:
    gate = next(g for g in outcome.verdict.gates if g.name == "leakage_clean")
    assert not gate.passed
    assert "future" in gate.rationale.lower() or "look" in gate.rationale.lower()


def test_a_fatal_leakage_flag_was_raised(outcome) -> None:
    fatal = [f for f in outcome.leakage_flags if f.is_fatal]
    assert fatal, "oracle_lookahead produced no fatal leakage flag"
    assert any(f.kind == "decision_lag" for f in fatal)


# ---------------------------------------------------------------------------
# the short-circuit -- the part that matters
# ---------------------------------------------------------------------------


def test_only_the_leakage_stage_ran(outcome) -> None:
    """A REJECT reached after computing a Sharpe is a different bug."""
    assert outcome.stages_run == (AuditStage.LEAKAGE,), (
        f"expected the audit to stop after leakage, but it ran {outcome.stages_run}"
    )


def test_statistics_stage_did_not_run(outcome) -> None:
    assert AuditStage.STATISTICS not in outcome.stages_run
    assert AuditStage.BENCHMARK not in outcome.stages_run


def test_no_evidence_was_produced(outcome) -> None:
    """There must be no Sharpe on the outcome to be tempted by."""
    assert outcome.evidence is None
    assert outcome.short_circuited is True


def test_pipeline_never_touches_the_statistics_code(bars, costs, monkeypatch) -> None:
    """The strongest form: make the stats path explode and audit anyway.

    Bookkeeping in stages_run could lie. This cannot -- if the pipeline computes a
    single metric or runs the benchmark comparison, the test raises instead of
    passing.
    """

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "statistics were computed on a strategy with fatal leakage; the "
            "short-circuit did not happen"
        )

    monkeypatch.setattr("null.metrics.compute_metrics", explode)
    monkeypatch.setattr("null.verdict.engine.compute_metrics", explode, raising=False)
    monkeypatch.setattr("null.verdict.engine.benchmark_check", explode, raising=False)
    monkeypatch.setattr("null.benchmark.buyhold.benchmark_check", explode)
    monkeypatch.setattr(
        "null.benchmark.risk_match.regress_excess_returns", explode
    )

    result = run_audit(
        run=oracle_lookahead(bars), bars=bars, benchmark_bars=bars, costs=costs
    )
    assert result.verdict.result == "REJECT"
    assert result.short_circuited is True


# ---------------------------------------------------------------------------
# the control -- the detector must not fire on everything
# ---------------------------------------------------------------------------


def test_an_honest_strategy_does_not_short_circuit(bars, costs) -> None:
    """A detector that flags every strategy is worth exactly nothing."""
    result = run_audit(
        run=honest_strategy(bars), bars=bars, benchmark_bars=bars, costs=costs
    )
    assert result.short_circuited is False
    assert not [f for f in result.leakage_flags if f.is_fatal]
    assert AuditStage.BENCHMARK in result.stages_run


def test_honest_strategy_still_reaches_a_verdict(bars, costs) -> None:
    result = run_audit(
        run=honest_strategy(bars), bars=bars, benchmark_bars=bars, costs=costs
    )
    assert result.verdict.result in {"REJECT", "PASS"}
    assert result.evidence is not None
