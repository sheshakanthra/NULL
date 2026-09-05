"""M5 acceptance test -- BUILD.md section 7.

Renders the overfit_grid verdict end to end: fixture -> Evidence -> gates ->
verdict.json -> HTML. Asserts the HTML is byte-identical across two runs and that
every limitation currently true of this repository appears in the band.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from null.contracts import (
    Evidence,
    FoldResult,
    ParamPoint,
    RegressionResult,
    SensitivityResult,
    Series,
    StrategyRun,
    TargetWeight,
)
from null.metrics import compute_metrics
from null.report.render import render_html, write_report
from null.stats.deflated_sharpe import deflated_sharpe_ratio
from null.stats.mtrl import minimum_track_record_length
from null.stats.pbo import compute_pbo
from null.stats.reality_check import reality_check
from null.verdict.engine import GateConfigError, evaluate, load_gate_config
from tests.golden.fixtures import overfit_grid

IST = timezone(timedelta(hours=5, minutes=30))
OUT = Path(__file__).resolve().parents[2] / "examples" / "overfit_grid"


def _series(values: np.ndarray) -> Series:
    start = datetime(2021, 1, 1, 15, 30, tzinfo=IST)
    return Series(
        ts=tuple(start + timedelta(days=i) for i in range(values.size)),
        values=tuple(float(v) for v in values),
    )


@pytest.fixture(scope="module")
def built():
    """Assemble a full Evidence for overfit_grid and run the engine on it."""
    fixture = overfit_grid()
    returns = fixture.returns
    rng = np.random.default_rng(7)
    benchmark = rng.normal(0.0003, 0.009, returns.size)

    dsr = deflated_sharpe_ratio(
        returns=returns, n_trials=fixture.n_trials, trial_sharpes=fixture.trial_sharpes
    )
    pbo = compute_pbo(
        fixture.trial_returns, n_trials=fixture.n_trials, n_subsamples=15
    )
    rc = reality_check(
        fixture.trial_returns, benchmark, n_bootstrap=200, seed=1
    )
    mtrl = minimum_track_record_length(returns)

    metrics = compute_metrics(
        _series(returns), basis="net", turnover_annual=12.0, time_in_market=0.95
    )
    bench_metrics = compute_metrics(
        _series(benchmark), basis="net", turnover_annual=0.0, time_in_market=1.0
    )

    evidence = Evidence(
        equity_curve=_series(np.cumprod(1.0 + returns)),
        benchmark_curve=_series(np.cumprod(1.0 + benchmark)),
        net_returns=_series(returns),
        gross_returns=_series(returns),
        cost_breakdown={"brokerage": 0.0, "dp_charge": 15200.0, "stt": 41000.0},
        turnover_annual=12.0,
        time_in_market=0.95,
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        alpha=RegressionResult(
            alpha_annual=0.004,
            alpha_stderr=0.02,
            alpha_tstat=0.31,
            se_method="newey_west",
            hac_lags=6,
            beta=0.12,
            beta_tstat=3.1,
            r_squared=0.02,
            n_obs=int(returns.size),
        ),
        deflated_sharpe=dsr.deflated_sharpe,
        pbo=pbo.pbo,
        reality_check_p=rc.p_value,
        mtrl_years=mtrl.mtrl_years,
        walkforward=tuple(
            FoldResult(
                fold_index=i,
                train_start=datetime(2021, 1, 1, 15, 30, tzinfo=IST),
                train_end=datetime(2021, 6, 1, 15, 30, tzinfo=IST),
                test_start=datetime(2021, 6, 2, 15, 30, tzinfo=IST),
                test_end=datetime(2021, 9, 1, 15, 30, tzinfo=IST),
                purged_bars=0,
                embargo_bars=5,
                metrics=metrics,
                net_return=(-0.01 if i % 2 else 0.02),
            )
            for i in range(5)
        ),
        regimes={"high_vol": metrics},
        sensitivity=SensitivityResult(
            param_names=("fast", "slow"),
            peak_sharpe=0.82,
            neighborhood_mean_sharpe=0.19,
            neighborhood_ratio=0.23,
            points=(
                ParamPoint(param_hash="p0", offsets={"fast": 0, "slow": 0}, sharpe=0.82),
                ParamPoint(param_hash="p1", offsets={"fast": 1, "slow": 0}, sharpe=0.19),
            ),
        ),
        leakage_flags=(),
    )

    run = StrategyRun(
        strategy_id="overfit_grid",
        param_hash="best-of-5000",
        n_trials=fixture.n_trials,
        universe=("SYNTH",),
        weights=(TargetWeight(ts=_series(returns).ts[0], symbol="SYNTH", weight=1.0),),
        initial_capital=1_000_000.0,
    )

    report = evaluate(
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
    return report, dsr, evidence


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


def test_config_drives_the_gate_list(built) -> None:
    report, _, _ = built
    configured = set(load_gate_config())
    assert {g.name for g in report.verdict.gates} == configured
    assert len(configured) == 7  # drawdown_tolerance demoted to a panel


def test_pbo_is_never_in_the_gate_list(built) -> None:
    """It is a panel. It appears on the report and never votes."""
    report, _, _ = built
    assert "pbo" not in {g.name for g in report.verdict.gates}
    assert "pbo" not in load_gate_config()
    assert "pbo" in report.panels


def test_unknown_gate_name_in_config_is_an_error(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("gates:\n  deflated_sharpe: {min: 0.95}\n  nonsense: {}\n")
    with pytest.raises(GateConfigError, match="nonsense"):
        load_gate_config(bad)


def test_listing_pbo_as_a_gate_is_an_error(tmp_path) -> None:
    bad = tmp_path / "panel.yaml"
    bad.write_text("gates:\n  pbo: {max: 0.5}\n")
    with pytest.raises(GateConfigError, match="pbo"):
        load_gate_config(bad)


def test_empty_gate_list_is_an_error(tmp_path) -> None:
    bad = tmp_path / "empty.yaml"
    bad.write_text("gates: {}\n")
    with pytest.raises(GateConfigError, match="no gates"):
        load_gate_config(bad)


def test_overfit_grid_is_rejected(built) -> None:
    report, _, _ = built
    assert report.verdict.result == "REJECT"


def test_not_computable_is_distinct_from_fail_in_the_verdict(built) -> None:
    """The distinction must survive into verdict.json, not just the HTML."""
    report, _, _ = built
    states = {g.name: g.state for g in report.verdict.gates}
    assert states["capacity"] == "NOT_COMPUTABLE"
    assert states["deflated_sharpe"] == "FAIL"
    assert report.not_computable == ("capacity",)
    for gate in report.verdict.gates:
        if gate.state == "NOT_COMPUTABLE":
            assert gate.passed is False


def test_a_raising_gate_fails_rather_than_crashing(built) -> None:
    from null.verdict.gates import run_gate

    _, _, evidence = built
    result = run_gate("deflated_sharpe", evidence, {})  # missing "min" key
    assert result.state == "FAIL"
    assert "raised" in result.rationale


# ---------------------------------------------------------------------------
# the rendered report
# ---------------------------------------------------------------------------


def _render(report, dsr, evidence) -> str:
    return render_html(
        report,
        observed_sharpe=dsr.observed_sharpe_annual,
        deflated_sharpe=dsr.deflated_sharpe,
        alpha_tstat=evidence.alpha.alpha_tstat,
        n_observations=evidence.metrics.n_obs,
    )


def test_html_is_byte_identical_across_two_renders(built) -> None:
    report, dsr, evidence = built
    first = _render(report, dsr, evidence).encode("utf-8")
    second = _render(report, dsr, evidence).encode("utf-8")
    assert first == second


def test_report_has_no_network_or_script_references(built) -> None:
    text = _render(*built)
    for forbidden in ("<script", "http://", "https://", "//cdn", "<iframe"):
        assert forbidden not in text, f"report references {forbidden!r}"


def test_not_computable_renders_distinctly_and_never_as_a_tick(built) -> None:
    text = _render(*built)
    assert "NOT COMPUTABLE" in text
    assert 'class="gate nc"' in text
    nc_row = text[text.index('class="gate nc"') : text.index('class="gate nc"') + 200]
    assert "✓" not in nc_row


def test_every_current_limitation_appears_in_the_band(built) -> None:
    report, _, _ = built
    text = _render(*built)
    band = text[text.index('class="band"') :]
    keys = {lim.key for lim in report.limitations}
    assert keys == {
        "unverified_cost_rates",
        "benchmark_series",
        "survivorship",
        "unchecked_leakage",
        "risk_free",
        "not_computable_gates",
        "golden_suite",
    }
    for lim in report.limitations:
        assert lim.text[:40] in band, f"{lim.key} missing from the band"


def test_limitations_band_is_after_the_verdict_not_a_footnote(built) -> None:
    text = _render(*built)
    assert text.index("Stated limitations") > text.index("Why it failed")


def test_rationales_appear_verbatim(built) -> None:
    report, _, _ = built
    text = _render(*built)
    import html as _h

    for gate in report.verdict.gates:
        if gate.state != "PASS":
            assert _h.escape(gate.rationale, quote=True) in text


def test_expected_max_sharpe_sentence_is_its_own_paragraph(built) -> None:
    text = _render(*built)
    assert "selection diagnostic" in text
    assert "noise alone is expected to produce a maximum Sharpe" in text


def test_panels_are_labelled_as_non_voting_with_the_caveat(built) -> None:
    text = _render(*built)
    assert "do not vote" in text
    assert "low PBO is never evidence" in text


def test_writes_the_report_and_the_verdict_json(built) -> None:
    report, dsr, evidence = built
    path = write_report(
        report,
        OUT / "report.html",
        observed_sharpe=dsr.observed_sharpe_annual,
        deflated_sharpe=dsr.deflated_sharpe,
        alpha_tstat=evidence.alpha.alpha_tstat,
        n_observations=evidence.metrics.n_obs,
    )
    (OUT / "verdict.json").write_bytes(report.verdict.canonical_json())
    assert path.exists() and path.stat().st_size > 2000
    assert (OUT / "verdict.json").exists()
