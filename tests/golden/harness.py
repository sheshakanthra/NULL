"""Turn a SyntheticStrategy into a full Evidence and run the verdict engine.

Shared by the golden suite so all eight fixtures go through exactly the same path.
Anything constructed here rather than computed is a stand-in for a pipeline stage
that does not exist yet, and every one of those is listed in ``SYNTHESISED`` so the
suite can state plainly what it is and is not exercising.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from null.contracts import (
    Evidence,
    FoldResult,
    LeakageFlag,
    ParamPoint,
    RegressionResult,
    SensitivityResult,
    Series,
    StrategyRun,
    TargetWeight,
)
from null.metrics import compute_metrics
from null.stats.deflated_sharpe import deflated_sharpe_ratio
from null.stats.mtrl import minimum_track_record_length
from null.stats.pbo import compute_pbo
from null.stats.reality_check import reality_check
from null.verdict.engine import VerdictReport, evaluate
from tests.golden.fixtures import SyntheticStrategy

IST = timezone(timedelta(hours=5, minutes=30))

#: Parts of Evidence the golden suite supplies rather than computes, because the
#: pipeline stage that would produce them is not built. Named so nobody mistakes a
#: green suite for end-to-end coverage.
SYNTHESISED = (
    "walkforward fold returns (partition/walkforward.py is not wired into the "
    "evidence build)",
    "sensitivity surface (sensitivity/neighborhood.py does not exist)",
    "max_adv_participation (computable from weights + Bar.adv_20, not yet wired)",
    "leakage flags (leakage/audit.py runs on bars; these fixtures are return series)",
)


def _series(values: np.ndarray) -> Series:
    start = datetime(2015, 1, 1, 15, 30, tzinfo=IST)
    return Series(
        ts=tuple(start + timedelta(days=i) for i in range(values.size)),
        values=tuple(float(v) for v in values),
    )


def build_report(fixture: SyntheticStrategy, *, seed: int = 11) -> VerdictReport:
    """Assemble Evidence for one fixture and put it through the gates."""
    returns = fixture.returns
    rng = np.random.default_rng(seed)
    benchmark = (
        fixture.benchmark
        if fixture.benchmark is not None
        else rng.normal(0.00035, 0.0095, returns.size)
    )

    dsr = deflated_sharpe_ratio(
        returns=returns, n_trials=fixture.n_trials, trial_sharpes=fixture.trial_sharpes
    )
    pbo = compute_pbo(
        fixture.trial_returns, n_trials=fixture.n_trials, n_subsamples=12
    )
    candidates = (
        fixture.trial_returns
        if fixture.trial_returns is not None
        else returns[:, None]
    )
    rc = reality_check(candidates, benchmark, n_bootstrap=200, seed=seed)
    mtrl = minimum_track_record_length(returns)

    metrics = compute_metrics(
        _series(returns), basis="net", turnover_annual=2.0, time_in_market=0.9
    )
    bench_metrics = compute_metrics(
        _series(benchmark), basis="net", turnover_annual=0.0, time_in_market=1.0
    )

    # Alpha regressed on the benchmark. Computed, not stubbed -- this is the gate
    # that benchmark_clone and costed_scalper have to be caught by.
    from null.benchmark.risk_match import regress_excess_returns, risk_match_benchmark

    strat_s = _series(returns)
    bench_s = _series(benchmark)
    matched, _ = risk_match_benchmark(strat_s, bench_s)
    alpha: RegressionResult = regress_excess_returns(strat_s, matched)

    flags: tuple[LeakageFlag, ...] = ()
    if fixture.fatal_leakage:
        flags = (
            LeakageFlag(
                kind="decision_lag",
                severity="fatal",
                symbol="SYNTH",
                detail=(
                    "Directional accuracy of 100% over 499 decisions. A strategy with "
                    "a declared decision lag of 1 bar cannot know the sign of the "
                    "return it is about to capture."
                ),
            ),
        )

    evidence = Evidence(
        equity_curve=_series(np.cumprod(1.0 + returns)),
        benchmark_curve=_series(np.cumprod(1.0 + benchmark)),
        net_returns=strat_s,
        gross_returns=strat_s,
        cost_breakdown={"brokerage": 0.0, "dp_charge": 1500.0, "stt": 4100.0},
        turnover_annual=2.0,
        time_in_market=0.9,
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        alpha=alpha,
        deflated_sharpe=dsr.deflated_sharpe,
        pbo=pbo.pbo,
        reality_check_p=rc.p_value,
        mtrl_years=mtrl.mtrl_years,
        max_adv_participation=fixture.adv_participation,
        walkforward=tuple(
            FoldResult(
                fold_index=i,
                train_start=datetime(2015, 1, 1, 15, 30, tzinfo=IST),
                train_end=datetime(2016, 1, 1, 15, 30, tzinfo=IST),
                test_start=datetime(2016, 1, 2, 15, 30, tzinfo=IST),
                test_end=datetime(2017, 1, 1, 15, 30, tzinfo=IST),
                purged_bars=0,
                embargo_bars=25,
                metrics=metrics,
                net_return=value,
            )
            for i, value in enumerate(fixture.fold_returns)
        ),
        regimes={"all": metrics},
        sensitivity=SensitivityResult(
            param_names=("a", "b"),
            peak_sharpe=1.0,
            neighborhood_mean_sharpe=fixture.neighborhood_ratio,
            neighborhood_ratio=fixture.neighborhood_ratio,
            points=(
                ParamPoint(param_hash="p0", offsets={"a": 0, "b": 0}, sharpe=1.0),
                ParamPoint(
                    param_hash="p1",
                    offsets={"a": 1, "b": 0},
                    sharpe=fixture.neighborhood_ratio,
                ),
            ),
        ),
        leakage_flags=flags,
    )

    run = StrategyRun(
        strategy_id=fixture.name,
        param_hash=f"{fixture.name}-fixed",
        n_trials=fixture.n_trials,
        universe=("SYNTH",),
        weights=(TargetWeight(ts=strat_s.ts[0], symbol="SYNTH", weight=1.0),),
        initial_capital=1_000_000.0,
    )

    return evaluate(
        run=run,
        evidence=evidence,
        context={
            "rates_are_verified": False,
            "benchmark_is_total_return": False,
            "universe_is_point_in_time": False,
            "risk_free_supplied": False,
            "golden_suite_green": False,
            "leakage_checks_unchecked": tuple(range(5)),
            "pbo_rationale": pbo.rationale,
            "expected_max_sharpe_sentence": dsr.selection_diagnostic,
            "mtrl_rationale": mtrl.rationale,
        },
    )
