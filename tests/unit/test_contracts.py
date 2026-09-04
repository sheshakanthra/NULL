"""Behavioural spec for ``null/contracts.py`` (BUILD.md section 2).

These models are frozen after M0. Every assertion here is an invariant the rest
of the build codes against, so read this file before proposing a contract change.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from null.contracts import (
    SPEC_VERSION,
    Bar,
    Evidence,
    GateResult,
    LeakageFlag,
    ParamPoint,
    PerfMetrics,
    RegressionResult,
    SensitivityResult,
    Series,
    StrategyRun,
    TargetWeight,
    TrialRecord,
    Verdict,
)

IST = timezone(timedelta(hours=5, minutes=30))


def _ts(day: int) -> datetime:
    return datetime(2024, 1, day, 15, 30, tzinfo=IST)


def _run(**overrides: object) -> StrategyRun:
    kwargs: dict[str, object] = {
        "strategy_id": "s",
        "param_hash": "p",
        "n_trials": 4,
        "universe": ("A", "B"),
        "weights": (TargetWeight(ts=_ts(2), symbol="A", weight=1.0),),
        "initial_capital": 100_000.0,
    }
    kwargs.update(overrides)
    return StrategyRun(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# n_trials -- CLAUDE.md invariant 7
# ---------------------------------------------------------------------------


def test_n_trials_is_required() -> None:
    """Omitting n_trials must be a validation error, never an inferred default."""
    with pytest.raises(ValidationError) as exc:
        StrategyRun(  # type: ignore[call-arg]
            strategy_id="s",
            param_hash="p",
            universe=("A",),
            weights=(TargetWeight(ts=_ts(2), symbol="A", weight=1.0),),
            initial_capital=100.0,
        )
    assert any(e["loc"] == ("n_trials",) for e in exc.value.errors())


def test_n_trials_has_no_default_on_the_model_field() -> None:
    assert StrategyRun.model_fields["n_trials"].is_required()


def test_n_trials_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        _run(n_trials=0)


def test_trials_may_be_empty_or_a_subset_but_never_exceed_n_trials() -> None:
    assert _run(n_trials=3, trials=()).trials == ()
    assert len(_run(n_trials=3, trials=(TrialRecord(param_hash="x", sharpe=1.0),)).trials) == 1
    with pytest.raises(ValidationError):
        _run(
            n_trials=1,
            trials=(
                TrialRecord(param_hash="x", sharpe=1.0),
                TrialRecord(param_hash="y", sharpe=2.0),
            ),
        )


# ---------------------------------------------------------------------------
# decision lag -- BUILD.md section 5, enforced at contract level
# ---------------------------------------------------------------------------


def test_decision_lag_of_zero_is_rejected() -> None:
    """A signal on bar t close cannot fill before bar t+1 open."""
    with pytest.raises(ValidationError):
        _run(decision_lag_bars=0)


def test_decision_lag_defaults_to_one() -> None:
    assert _run().decision_lag_bars == 1


# ---------------------------------------------------------------------------
# frozen-ness and strictness
# ---------------------------------------------------------------------------


def test_strategy_run_is_frozen() -> None:
    with pytest.raises(ValidationError):
        _run().n_trials = 1  # type: ignore[misc]


def test_target_weight_is_frozen() -> None:
    with pytest.raises(ValidationError):
        TargetWeight(ts=_ts(2), symbol="A", weight=1.0).weight = 2.0  # type: ignore[misc]


def test_series_is_frozen() -> None:
    with pytest.raises(ValidationError):
        Series(ts=(_ts(2),), values=(1.0,)).values = (9.0,)  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _run(sharpe_i_wish_i_had=3.0)


# ---------------------------------------------------------------------------
# float canonicalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected(bad: float) -> None:
    """Missing evidence must fail loudly, not serialise as invalid JSON."""
    with pytest.raises(ValidationError):
        TargetWeight(ts=_ts(2), symbol="A", weight=bad)


def test_floats_are_quantised_to_twelve_significant_digits() -> None:
    assert TargetWeight(ts=_ts(2), symbol="A", weight=0.1 + 0.2).weight == 0.3


def test_quantisation_preserves_small_magnitudes() -> None:
    """Significant digits, not decimal places -- a 1e-9 weight must survive."""
    w = TargetWeight(ts=_ts(2), symbol="A", weight=1.234567890123e-9)
    assert w.weight == pytest.approx(1.234567890123e-9, rel=1e-12)


def test_negative_zero_is_normalised() -> None:
    assert math.copysign(1.0, TargetWeight(ts=_ts(2), symbol="A", weight=-0.0).weight) > 0


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TargetWeight(ts=datetime(2024, 1, 2, 15, 30), symbol="A", weight=1.0)


def test_timestamps_are_normalised_to_ist() -> None:
    w = TargetWeight(ts=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc), symbol="A", weight=1.0)
    assert w.ts.utcoffset() == timedelta(hours=5, minutes=30)
    assert w.ts.hour == 15 and w.ts.minute == 30


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def test_series_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        Series(ts=(_ts(2), _ts(3)), values=(1.0,))


def test_series_requires_strictly_increasing_timestamps() -> None:
    with pytest.raises(ValidationError):
        Series(ts=(_ts(3), _ts(2)), values=(1.0, 2.0))
    with pytest.raises(ValidationError):
        Series(ts=(_ts(2), _ts(2)), values=(1.0, 2.0))


def test_series_supports_an_empty_curve() -> None:
    assert len(Series(ts=(), values=())) == 0


def test_series_converts_to_numpy_at_the_compute_boundary() -> None:
    s = Series(ts=(_ts(2), _ts(3)), values=(1.5, -2.5))
    arr = s.to_numpy()
    assert arr.tolist() == [1.5, -2.5]
    assert str(arr.dtype) == "float64"


def test_series_round_trips_through_pandas() -> None:
    pd = pytest.importorskip("pandas")
    s = Series(ts=(_ts(2), _ts(3)), values=(1.5, -2.5))
    revived = Series.from_pandas(s.to_pandas())
    assert revived == s
    assert isinstance(s.to_pandas(), pd.Series)


# ---------------------------------------------------------------------------
# StrategyRun cross-field invariants
# ---------------------------------------------------------------------------


def test_universe_is_canonically_sorted_and_deduplicated() -> None:
    assert _run(universe=("B", "A")).universe == ("A", "B")
    with pytest.raises(ValidationError):
        _run(universe=("A", "A"))
    with pytest.raises(ValidationError):
        _run(universe=())


def test_weights_are_canonically_ordered() -> None:
    run = _run(
        weights=(
            TargetWeight(ts=_ts(3), symbol="B", weight=0.5),
            TargetWeight(ts=_ts(2), symbol="A", weight=0.5),
        )
    )
    assert [(w.ts.day, w.symbol) for w in run.weights] == [(2, "A"), (3, "B")]


def test_duplicate_weight_entries_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _run(
            weights=(
                TargetWeight(ts=_ts(2), symbol="A", weight=0.5),
                TargetWeight(ts=_ts(2), symbol="A", weight=0.9),
            )
        )


def test_weights_must_reference_symbols_in_the_universe() -> None:
    with pytest.raises(ValidationError):
        _run(weights=(TargetWeight(ts=_ts(2), symbol="ZZZ", weight=1.0),))


def test_initial_capital_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _run(initial_capital=0.0)


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------


def _bar(**overrides: object) -> Bar:
    kwargs: dict[str, object] = {
        "ts": _ts(2),
        "symbol": "A",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "volume": 1000.0,
    }
    kwargs.update(overrides)
    return Bar(**kwargs)  # type: ignore[arg-type]


def test_bar_accepts_a_sane_ohlc_and_optional_adv() -> None:
    assert _bar().adv_20 is None
    assert _bar(adv_20=5_000_000.0).adv_20 == 5_000_000.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"high": 98.0},  # high below low
        {"low": 101.0},  # low above open
        {"high": 103.0},  # high below close
        {"open": -1.0},  # negative price
        {"volume": -1.0},  # negative volume
    ],
)
def test_bar_rejects_impossible_ohlc(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _bar(**overrides)


# ---------------------------------------------------------------------------
# GateResult / Verdict -- CLAUDE.md invariant 6, default REJECT
# ---------------------------------------------------------------------------


def _gate(passed: bool, name: str = "deflated_sharpe") -> GateResult:
    return GateResult(
        name=name,
        passed=passed,
        observed=0.31,
        threshold=0.95,
        rationale="observed 0.31 against a threshold of 0.95",
    )


def test_gate_rationale_may_not_be_empty() -> None:
    """The rationale strings are the product (CLAUDE.md, code standards)."""
    with pytest.raises(ValidationError):
        GateResult(name="g", passed=False, observed=1.0, threshold=2.0, rationale="   ")


def test_gate_observed_and_threshold_accept_floats_or_strings() -> None:
    g = GateResult(
        name="leakage_clean",
        passed=False,
        observed="oracle_lookahead",
        threshold="no fatal flags",
        rationale="a fatal leakage flag was raised",
    )
    assert g.observed == "oracle_lookahead"


def test_a_pass_verdict_with_any_failing_gate_is_unconstructible() -> None:
    """Default REJECT is a contract invariant, not just engine behaviour."""
    with pytest.raises(ValidationError):
        Verdict(
            result="PASS",
            gates=(_gate(True), _gate(False, name="pbo")),
            evidence_hash="0" * 64,
            spec_version=SPEC_VERSION,
            generated_from=_run(),
        )


def test_a_pass_verdict_with_no_gates_is_unconstructible() -> None:
    """Zero gates run is missing evidence, and missing evidence never passes."""
    with pytest.raises(ValidationError):
        Verdict(
            result="PASS",
            gates=(),
            evidence_hash="0" * 64,
            spec_version=SPEC_VERSION,
            generated_from=_run(),
        )


def test_a_reject_verdict_is_always_constructible() -> None:
    for gates in ((), (_gate(True),), (_gate(False),)):
        v = Verdict(
            result="REJECT",
            gates=gates,
            evidence_hash="0" * 64,
            spec_version=SPEC_VERSION,
            generated_from=_run(),
        )
        assert v.result == "REJECT"


def test_a_pass_verdict_with_all_gates_passing_is_constructible() -> None:
    v = Verdict(
        result="PASS",
        gates=(_gate(True), _gate(True, name="pbo")),
        evidence_hash="a" * 64,
        spec_version=SPEC_VERSION,
        generated_from=_run(),
    )
    assert v.result == "PASS"


def test_evidence_hash_must_be_a_sha256_hex_digest() -> None:
    with pytest.raises(ValidationError):
        Verdict(
            result="REJECT",
            gates=(),
            evidence_hash="not-a-hash",
            spec_version=SPEC_VERSION,
            generated_from=_run(),
        )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _metrics() -> PerfMetrics:
    return PerfMetrics(
        cagr=0.12,
        vol_annual=0.18,
        sharpe=0.67,
        sortino=0.9,
        max_drawdown=0.22,
        calmar=0.55,
        longest_underwater_days=180,
        hit_rate=0.51,
        avg_win=0.011,
        avg_loss=-0.009,
        turnover_annual=3.2,
        time_in_market=0.78,
        tail_ratio=1.05,
        worst_5_days=(-0.06, -0.05, -0.04, -0.038, -0.031),
        n_obs=1006,
    )


def _evidence(**overrides: object) -> Evidence:
    curve = Series(ts=(_ts(2), _ts(3)), values=(1.0, 1.01))
    kwargs: dict[str, object] = {
        "equity_curve": curve,
        "benchmark_curve": curve,
        "net_returns": curve,
        "gross_returns": curve,
        "cost_breakdown": {"stt": 1200.0, "dp": 320.0},
        "turnover_annual": 3.2,
        "time_in_market": 0.78,
        "metrics": _metrics(),
        "benchmark_metrics": _metrics(),
        "alpha": RegressionResult(
            alpha_annual=0.004,
            alpha_stderr=0.01,
            alpha_tstat=0.4,
            beta=0.98,
            beta_tstat=31.0,
            r_squared=0.87,
            n_obs=1006,
        ),
        "deflated_sharpe": 0.31,
        "pbo": 0.62,
        "reality_check_p": 0.41,
        "mtrl_years": 7.4,
        "walkforward": (),
        "regimes": {"high_vol": _metrics()},
        "sensitivity": SensitivityResult(
            param_names=("rsi_period", "entry"),
            peak_sharpe=1.8,
            neighborhood_mean_sharpe=0.54,
            neighborhood_ratio=0.3,
            points=(
                ParamPoint(param_hash="p0", offsets={"rsi_period": 0, "entry": 0}, sharpe=1.8),
                ParamPoint(param_hash="p1", offsets={"rsi_period": 1, "entry": 0}, sharpe=0.4),
            ),
        ),
        "leakage_flags": (),
    }
    kwargs.update(overrides)
    return Evidence(**kwargs)  # type: ignore[arg-type]


def test_evidence_is_constructible_and_hashable() -> None:
    e = _evidence()
    assert len(e.content_hash()) == 64
    assert e.content_hash() == _evidence().content_hash()


@pytest.mark.parametrize(
    "field", ["deflated_sharpe", "pbo", "reality_check_p", "time_in_market"]
)
@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_probability_fields_are_bounded_to_the_unit_interval(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        _evidence(**{field: bad})


def test_leakage_flag_carries_a_severity_and_a_human_detail() -> None:
    flag = LeakageFlag(
        kind="decision_lag",
        severity="fatal",
        symbol="INFY",
        ts=_ts(2),
        detail="weight at 2024-01-02 depends on the close of the same bar",
    )
    assert flag.is_fatal is True
    assert LeakageFlag(kind="nan_ffill", severity="warning", detail="d").is_fatal is False


def test_spec_version_is_exported_and_non_empty() -> None:
    assert isinstance(SPEC_VERSION, str) and SPEC_VERSION
