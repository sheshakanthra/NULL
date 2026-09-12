"""``null`` CLI end to end. BUILD.md §13.

The exit code is the interface: a CI job gates on it without parsing output. So a
malformed input must never exit 1 and be mistaken for a considered rejection.

    0   PASS
    1   REJECT
    2   usage or input error
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from null.cli import EXIT_PASS, EXIT_REJECT, EXIT_USAGE, main
from null.contracts import StrategyRun, Verdict

IST = timezone(timedelta(hours=5, minutes=30))
REPO = Path(__file__).resolve().parents[2]
GATES = REPO / "configs" / "gates_default.yaml"
COSTS = REPO / "configs" / "costs_india_equity.yaml"


@pytest.fixture
def bars_parquet(tmp_path: Path) -> Path:
    """A small OHLCV cache for two symbols plus a benchmark."""
    rng = np.random.default_rng(19)
    days = pd.bdate_range("2022-01-03", periods=400)
    rows = []
    for symbol in ("AAA", "BBB"):
        price = 1000.0
        for day in days:
            price *= 1.0 + rng.normal(0.0004, 0.011)
            volume = float(rng.integers(1_000_000, 4_000_000))
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": price * 0.999,
                    "high": price * 1.004,
                    "low": price * 0.996,
                    "close": price,
                    "volume": volume,
                    "value_traded": volume * price,
                    "format_handler": "test",
                }
            )
    path = tmp_path / "bars.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def benchmark_parquet(tmp_path: Path) -> Path:
    """A single-symbol benchmark. Never the strategy's own bars."""
    rng = np.random.default_rng(23)
    days = pd.bdate_range("2022-01-03", periods=400)
    price = 20_000.0
    rows = []
    for day in days:
        price *= 1.0 + rng.normal(0.00045, 0.0095)
        rows.append(
            {
                "date": day,
                "symbol": "NIFTY50_TRI",
                "open": price * 0.999,
                "high": price * 1.003,
                "low": price * 0.997,
                "close": price,
                "volume": 1e7,
                "value_traded": 1e12,
                "format_handler": "test",
            }
        )
    path = tmp_path / "bench.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def run_json(tmp_path: Path, bars_parquet: Path) -> Path:
    frame = pd.read_parquet(bars_parquet)
    stamps = sorted({d for d in frame["date"]})
    weights = [
        {
            "ts": pd.Timestamp(d).replace(hour=15, minute=30).tz_localize(IST).isoformat(),
            "symbol": sym,
            "weight": 0.5,
        }
        for d in stamps
        for sym in ("AAA", "BBB")
    ]
    payload = {
        "strategy_id": "cli_smoke",
        "param_hash": "abc123",
        "n_trials": 1,
        "universe": ["AAA", "BBB"],
        "weights": weights,
        "decision_lag_bars": 1,
        "initial_capital": 10_000_000.0,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def lean_run_with_trials(
    run_json: Path, bars_parquet: Path, tmp_path: Path
) -> tuple[Path, Path]:
    """A run.json declaring three trials without series, plus the sibling parquet."""
    frame = pd.read_parquet(bars_parquet)
    stamps = sorted({d for d in frame["date"]})
    payload = json.loads(run_json.read_text())
    payload["n_trials"] = 3
    payload["trials"] = [
        {"param_hash": "h0", "sharpe": 0.9},
        {"param_hash": "h1", "sharpe": 0.3},
        {"param_hash": "h2", "sharpe": -0.1},
    ]
    lean = tmp_path / "lean_run_fixture.json"
    lean.write_text(json.dumps(payload), encoding="utf-8")

    localised = [
        pd.Timestamp(s).replace(hour=15, minute=30).tz_localize(IST) for s in stamps
    ]
    rng = np.random.default_rng(5)
    n = len(localised) - 1
    trials_path = tmp_path / "fixture.trials.parquet"
    pd.DataFrame(
        {
            "date": localised[1:],
            "h0": rng.normal(0.001, 0.01, n),
            "h1": rng.normal(0.0002, 0.01, n),
            "h2": rng.normal(-0.0005, 0.01, n),
        }
    ).to_parquet(trials_path, index=False)
    return lean, trials_path


def _argv(
    run_json: Path, bars: Path, out: Path, *extra: str, benchmark: Path | None = None
) -> list[str]:
    argv = [
        "audit", str(run_json),
        "--config", str(GATES),
        "--costs", str(COSTS),
        "--bars", str(bars),
        "--out", str(out),
    ]
    if benchmark is not None:
        argv += ["--benchmark", str(benchmark)]
    return argv + list(extra)


# ---------------------------------------------------------------------------
# exit codes -- the interface
# ---------------------------------------------------------------------------


def test_a_rejected_strategy_exits_one(run_json, bars_parquet, tmp_path, benchmark_parquet) -> None:
    out = tmp_path / "o"
    code = main(_argv(run_json, bars_parquet, out, benchmark=benchmark_parquet))
    assert code == EXIT_REJECT
    assert (out / "verdict.json").exists()
    assert (out / "report.html").exists()
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["result"] == "REJECT"


def test_a_missing_run_file_exits_two_not_one(tmp_path) -> None:
    """The distinction a CI job depends on. Usage error is not a rejection."""
    code = main(_argv(tmp_path / "nope.json", tmp_path / "nope.parquet", tmp_path))
    assert code == EXIT_USAGE


def test_malformed_json_exits_two_with_a_readable_message(
    tmp_path, bars_parquet, capsys
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json at all", encoding="utf-8")
    code = main(_argv(bad, bars_parquet, tmp_path))
    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "not valid JSON" in err
    assert "line" in err and "column" in err
    assert "Traceback" not in err


def test_a_run_missing_n_trials_gets_a_human_message(
    tmp_path, bars_parquet, capsys
) -> None:
    """A Pydantic traceback is not a usable error for someone auditing a strategy."""
    bad = tmp_path / "no_trials.json"
    bad.write_text(
        json.dumps(
            {
                "strategy_id": "x",
                "param_hash": "p",
                "universe": ["AAA"],
                "weights": [],
                "initial_capital": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    code = main(_argv(bad, bars_parquet, tmp_path))
    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "n_trials" in err
    assert "required" in err.lower()
    assert "Traceback" not in err
    assert "pydantic" not in err.lower()


def test_a_bad_gate_config_exits_two(tmp_path, run_json, bars_parquet, benchmark_parquet) -> None:
    bad = tmp_path / "gates.yaml"
    bad.write_text("gates:\n  nonsense: {}\n", encoding="utf-8")
    argv = _argv(run_json, bars_parquet, tmp_path / "o2", benchmark=benchmark_parquet)
    argv[argv.index("--config") + 1] = str(bad)
    assert main(argv) == EXIT_USAGE


# ---------------------------------------------------------------------------
# never overwrite a verdict silently
# ---------------------------------------------------------------------------


def test_an_existing_verdict_is_not_overwritten_without_force(
    run_json, bars_parquet, tmp_path, capsys
, benchmark_parquet) -> None:
    out = tmp_path / "o"
    assert main(_argv(run_json, bars_parquet, out, benchmark=benchmark_parquet)) == EXIT_REJECT
    original = (out / "verdict.json").read_bytes()

    code = main(_argv(run_json, bars_parquet, out, benchmark=benchmark_parquet))
    assert code == EXIT_USAGE
    assert "--force" in capsys.readouterr().err
    assert (out / "verdict.json").read_bytes() == original


def test_force_allows_the_overwrite(run_json, bars_parquet, tmp_path, benchmark_parquet) -> None:
    out = tmp_path / "o"
    main(_argv(run_json, bars_parquet, out, benchmark=benchmark_parquet))
    assert main(_argv(run_json, bars_parquet, out, "--force", benchmark=benchmark_parquet)) == EXIT_REJECT


# ---------------------------------------------------------------------------
# determinism -- both artifacts, not just the verdict
# ---------------------------------------------------------------------------


def test_two_invocations_produce_byte_identical_artifacts(
    run_json, bars_parquet, tmp_path
, benchmark_parquet) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert main(_argv(run_json, bars_parquet, first, benchmark=benchmark_parquet)) == EXIT_REJECT
    assert main(_argv(run_json, bars_parquet, second, benchmark=benchmark_parquet)) == EXIT_REJECT

    assert (first / "verdict.json").read_bytes() == (second / "verdict.json").read_bytes()
    assert (first / "report.html").read_bytes() == (second / "report.html").read_bytes()


# ---------------------------------------------------------------------------
# invariant 2: the audit path never touches the network
# ---------------------------------------------------------------------------


def test_the_cli_runs_with_every_network_path_rigged_to_explode(
    run_json, bars_parquet, tmp_path, monkeypatch
, benchmark_parquet) -> None:
    """Not a grep. Make the sockets themselves raise and audit anyway.

    The source-invariant grep proves null/ contains no network imports. This proves
    the running CLI never reaches one by any route -- a lazily imported module, a
    subprocess, a transitive dependency.
    """
    import socket
    import urllib.request

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("the audit path attempted a network call")

    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(socket, "getaddrinfo", explode)
    monkeypatch.setattr(urllib.request, "urlopen", explode)

    assert main(_argv(run_json, bars_parquet, tmp_path / "offline", benchmark=benchmark_parquet)) == EXIT_REJECT
    assert (tmp_path / "offline" / "verdict.json").exists()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_prints_package_and_spec_version(capsys) -> None:
    """A verdict is only interpretable against the spec that produced it."""
    from null.contracts import SPEC_VERSION

    assert main(["--version"]) == EXIT_PASS
    out = capsys.readouterr().out
    assert "null 0.1.0" in out
    assert SPEC_VERSION in out


def test_no_command_exits_two(capsys) -> None:
    assert main([]) == EXIT_USAGE
    assert "command is required" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the packaged entry point
# ---------------------------------------------------------------------------


def test_pyproject_ships_the_console_script() -> None:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'null = "null.cli:main"' in text


def test_missing_data_cache_is_a_usage_error_not_a_rejection(
    run_json, tmp_path, capsys
) -> None:
    """A missing cache must not read as 'we audited it and it failed'."""
    code = main(_argv(run_json, tmp_path / "absent.parquet", tmp_path / "o3"))
    assert code == EXIT_USAGE
    assert "will not substitute" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# two bugs found while building this, pinned so they stay found
# ---------------------------------------------------------------------------


def test_weights_that_match_no_bar_are_an_input_error_not_a_zero_series(
    tmp_path, bars_parquet, benchmark_parquet, capsys
) -> None:
    """Found while writing the CLI's own smoke test.

    Weight timestamps at midnight never match bars at 15:30, and the portfolio then
    returns a series of exact zeros. Left alone the gates judge that as "no edge" --
    a verdict on a strategy that was never simulated. It has to be an input error.
    """
    frame = pd.read_parquet(bars_parquet)
    stamps = sorted({d for d in frame["date"]})
    payload = {
        "strategy_id": "misaligned",
        "param_hash": "p",
        "n_trials": 1,
        "universe": ["AAA", "BBB"],
        "weights": [
            # Midnight, not the 15:30 bar close.
            {"ts": pd.Timestamp(d).tz_localize(IST).isoformat(), "symbol": s,
             "weight": 0.5}
            for d in stamps for s in ("AAA", "BBB")
        ],
        "initial_capital": 10_000_000.0,
    }
    bad = tmp_path / "misaligned.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "mis"
    code = main(_argv(bad, bars_parquet, out, benchmark=benchmark_parquet))

    # Exit 1, not 2, and that is correct: the leakage audit catches it first as a
    # fatal timestamp_monotonicity flag. A decision time that corresponds to no
    # observable bar is exactly what that check exists for, so it is a genuine
    # finding rather than a malformed file. What matters is that it is never
    # silently audited as a flat, edgeless strategy.
    assert code == EXIT_REJECT
    verdict = json.loads((out / "verdict.json").read_text())
    assert [g["name"] for g in verdict["gates"]] == ["leakage_clean"]
    assert "timestamp_monotonicity" in verdict["gates"][0]["observed"]


def test_the_alignment_guard_fires_for_callers_that_skip_the_leakage_audit(
    bars_parquet,
) -> None:
    """Defence in depth for benchmark_check called directly.

    The leakage audit is the first line and catches this in the CLI. The guard in
    the portfolio builder is the second, for anything that reaches it another way.
    """
    from datetime import datetime

    from null.benchmark.buyhold import benchmark_check
    from null.contracts import StrategyRun, TargetWeight
    from null.costs.india_equity import IndiaEquityCostModel
    from null.data.ohlcv import load_bars

    bars = load_bars(bars_parquet)
    bench = tuple(b for b in bars if b.symbol == "AAA")
    off_by_hours = datetime(2022, 1, 3, 9, 15, tzinfo=IST)
    run = StrategyRun(
        strategy_id="misaligned",
        param_hash="p",
        n_trials=1,
        universe=("AAA",),
        weights=(TargetWeight(ts=off_by_hours, symbol="AAA", weight=1.0),),
        initial_capital=1_000_000.0,
    )
    with pytest.raises(ValueError, match="line up with a bar timestamp"):
        benchmark_check(
            run=run,
            bars=bars,
            benchmark_bars=bench,
            costs=IndiaEquityCostModel.from_yaml(COSTS),
        )


def test_fatal_leakage_short_circuits_before_any_statistic(
    tmp_path, bars_parquet, benchmark_parquet, monkeypatch
) -> None:
    """BUILD.md §5, enforced in the CLI and not merely in the engine.

    The first CLI draft computed the full Evidence regardless and only then let the
    leakage gate fail. That reports a Sharpe for a strategy that can see the future,
    which is precisely what the short-circuit exists to prevent.
    """
    from null.contracts import LeakageFlag
    from null.leakage.audit import LeakageReport

    def leaky(run, bars, **kwargs):
        return LeakageReport(
            flags=(
                LeakageFlag(
                    kind="decision_lag",
                    severity="fatal",
                    detail="planted fatal flag for the short-circuit test",
                ),
            ),
            checks_run=("decision_lag",),
            unchecked=("everything else",),
        )

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("a statistic was computed despite fatal leakage")

    monkeypatch.setattr("null.cli.audit_leakage", leaky)
    monkeypatch.setattr("null.cli.deflated_sharpe_ratio", explode)
    monkeypatch.setattr("null.cli.reality_check", explode)
    monkeypatch.setattr("null.cli.compute_pbo", explode)
    monkeypatch.setattr("null.cli.benchmark_check", explode)

    out = tmp_path / "leak"
    assert main(_argv(run_json_for(tmp_path, bars_parquet), bars_parquet, out,
                      benchmark=benchmark_parquet)) == EXIT_REJECT
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["result"] == "REJECT"
    assert [g["name"] for g in verdict["gates"]] == ["leakage_clean"]
    assert "no performance statistic was computed" in verdict["gates"][0]["rationale"]
    # The real proof is that the monkeypatched statistics never fired -- any of them
    # would have raised. The report additionally says why there is no number on it.
    html = (out / "report.html").read_text()
    assert "no Sharpe ratio on this report" in html
    assert "Deflated Sharpe" not in html  # no metric cards: nothing was measured


def run_json_for(tmp_path: Path, bars_parquet: Path) -> Path:
    frame = pd.read_parquet(bars_parquet)
    stamps = sorted({d for d in frame["date"]})
    payload = {
        "strategy_id": "leaky",
        "param_hash": "p",
        "n_trials": 1,
        "universe": ["AAA", "BBB"],
        "weights": [
            {"ts": pd.Timestamp(d).replace(hour=15, minute=30).tz_localize(IST).isoformat(),
             "symbol": s, "weight": 0.5}
            for d in stamps for s in ("AAA", "BBB")
        ],
        "initial_capital": 10_000_000.0,
    }
    path = tmp_path / "leaky.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# --trials-parquet: a run.json may legitimately omit per-trial returns to stay
# small, and this restores real PBO evidence from a sibling file rather than
# silently degrading to NOT_COMPUTABLE.
# ---------------------------------------------------------------------------


def test_enrich_trials_from_parquet_restores_real_returns(tmp_path) -> None:
    from null.cli import enrich_trials_from_parquet
    from null.contracts import StrategyRun, TargetWeight, TrialRecord

    stamps = [datetime(2022, 1, 3 + i, 15, 30, tzinfo=IST) for i in range(5)]
    run = StrategyRun(
        strategy_id="grid",
        param_hash="best",
        n_trials=2,
        universe=("AAA",),
        weights=(TargetWeight(ts=stamps[0], symbol="AAA", weight=1.0),),
        initial_capital=1_000_000.0,
        trials=(
            TrialRecord(param_hash="hash_a", sharpe=1.5, returns=None),
            TrialRecord(param_hash="hash_b", sharpe=0.2, returns=None),
        ),
    )
    assert all(t.returns is None for t in run.trials)

    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(s) for s in stamps],
            "hash_a": [0.01, -0.02, 0.03, 0.0, 0.01],
            "hash_b": [0.00, 0.00, -0.01, 0.02, -0.01],
        }
    )
    path = tmp_path / "trials_test.parquet"
    frame.to_parquet(path, index=False)

    enriched = enrich_trials_from_parquet(run, path)
    assert all(t.returns is not None for t in enriched.trials)
    a = next(t for t in enriched.trials if t.param_hash == "hash_a")
    assert a.returns is not None
    assert list(a.returns.values) == pytest.approx([0.01, -0.02, 0.03, 0.0, 0.01])
    # The original run object is untouched -- frozen models, a new one returned.
    assert all(t.returns is None for t in run.trials)


def test_enrich_trials_from_parquet_raises_on_a_missing_trial_column(tmp_path) -> None:
    from null.cli import InputError, enrich_trials_from_parquet
    from null.contracts import StrategyRun, TargetWeight, TrialRecord

    stamps = [datetime(2022, 1, 3, 15, 30, tzinfo=IST)]
    run = StrategyRun(
        strategy_id="grid",
        param_hash="best",
        n_trials=2,
        universe=("AAA",),
        weights=(TargetWeight(ts=stamps[0], symbol="AAA", weight=1.0),),
        initial_capital=1_000_000.0,
        trials=(
            TrialRecord(param_hash="hash_a", sharpe=1.5, returns=None),
            TrialRecord(param_hash="hash_missing", sharpe=0.2, returns=None),
        ),
    )
    frame = pd.DataFrame({"date": [pd.Timestamp(stamps[0])], "hash_a": [0.01]})
    path = tmp_path / "trials_partial.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(InputError, match="hash_missing|missing columns"):
        enrich_trials_from_parquet(run, path)


def test_trials_parquet_gives_pbo_real_evidence_instead_of_not_computable(
    run_json, bars_parquet, benchmark_parquet, tmp_path
) -> None:
    """End to end: a lean run.json plus --trials-parquet must reach compute_pbo
    with a real matrix, not fall back to NOT_COMPUTABLE."""
    frame = pd.read_parquet(bars_parquet)
    stamps = sorted({d for d in frame["date"]})
    payload = json.loads(run_json.read_text())
    payload["n_trials"] = 3
    payload["trials"] = [
        {"param_hash": "h0", "sharpe": 0.9},
        {"param_hash": "h1", "sharpe": 0.3},
        {"param_hash": "h2", "sharpe": -0.1},
    ]
    lean = tmp_path / "lean_run.json"
    lean.write_text(json.dumps(payload), encoding="utf-8")

    # bars_parquet's own "date" column is naive midnight -- load_bars is what
    # normalises it to 15:30 IST bar-close time. The strategy's own return
    # series (bench.strategy_returns.ts) will carry THOSE normalised
    # timestamps, so the trial parquet's dates must match that shape.
    localised = [
        pd.Timestamp(s).replace(hour=15, minute=30).tz_localize(IST) for s in stamps
    ]
    rng = np.random.default_rng(5)
    n = len(localised) - 1
    trials_frame = pd.DataFrame(
        {
            "date": localised[1:],
            "h0": rng.normal(0.001, 0.01, n),
            "h1": rng.normal(0.0002, 0.01, n),
            "h2": rng.normal(-0.0005, 0.01, n),
        }
    )
    trials_path = tmp_path / "run.trials.parquet"
    trials_frame.to_parquet(trials_path, index=False)

    out = tmp_path / "enriched_out"
    argv = _argv(lean, bars_parquet, out, benchmark=benchmark_parquet)
    argv += ["--trials-parquet", str(trials_path)]
    main(argv)

    verdict = json.loads((out / "verdict.json").read_text())
    report_html = (out / "report.html").read_text()
    assert "pbo" not in {g["name"] for g in verdict["gates"]}  # panel, not a gate

    # Isolate the PBO panel specifically -- other things on the page (e.g. the
    # unrelated sensitivity_plateau gate) legitimately say "not computable" in
    # this smoke scenario, so the assertion must not be page-wide.
    pbo_start = report_html.lower().index('class="g">pbo</span>')
    pbo_panel = report_html.lower()[pbo_start : pbo_start + 900]
    assert "not applicable" not in pbo_panel
    assert "not computable" not in pbo_panel
    assert "symmetric train/test partitions" in pbo_panel  # real computed evidence


# ---------------------------------------------------------------------------
# --sensitivity: a parameter grid IS the neighbourhood surface. Without one the
# gate abstains, and NOT_COMPUTABLE is not a pass.
# ---------------------------------------------------------------------------


def _surface(peak_hash: str, *, peak: float = 1.0, ring: float = 0.8) -> dict:
    """A one-parameter surface: the peak plus its +/-1 and +/-2 rings."""
    points = [{"param_hash": peak_hash, "offsets": {"p": 0}, "sharpe": peak}]
    points += [
        {"param_hash": f"n{o}", "offsets": {"p": o}, "sharpe": ring}
        for o in (-2, -1, 1, 2)
    ]
    return {
        "param_names": ["p"],
        "peak_sharpe": peak,
        "neighborhood_mean_sharpe": ring,
        "neighborhood_ratio": ring / peak,
        "points": points,
    }


def test_without_a_surface_the_plateau_gate_abstains_rather_than_passing(
    run_json, bars_parquet, benchmark_parquet, tmp_path
) -> None:
    out = tmp_path / "no_surface"
    main(_argv(run_json, bars_parquet, out, benchmark=benchmark_parquet))
    verdict = json.loads((out / "verdict.json").read_text())
    gate = next(g for g in verdict["gates"] if g["name"] == "sensitivity_plateau")
    assert gate["state"] == "NOT_COMPUTABLE"
    assert gate["passed"] is False, "abstaining must never count as a pass"


def test_a_supplied_surface_makes_the_plateau_gate_judge(
    run_json, bars_parquet, benchmark_parquet, tmp_path
) -> None:
    payload = json.loads(run_json.read_text())
    surface = tmp_path / "surface.json"
    surface.write_text(json.dumps(_surface(payload["param_hash"])), encoding="utf-8")

    out = tmp_path / "with_surface"
    argv = _argv(run_json, bars_parquet, out, benchmark=benchmark_parquet)
    argv += ["--sensitivity", str(surface)]
    main(argv)

    verdict = json.loads((out / "verdict.json").read_text())
    gate = next(g for g in verdict["gates"] if g["name"] == "sensitivity_plateau")
    assert gate["state"] in ("PASS", "FAIL"), "a supplied surface must be judged"
    assert gate["state"] == "PASS"  # ratio 0.80 clears the 0.60 threshold


def test_a_spike_surface_fails_the_plateau_gate(
    run_json, bars_parquet, benchmark_parquet, tmp_path
) -> None:
    """The gate must still be capable of failing. A wiring that only ever passes
    would be worse than the abstention it replaced."""
    payload = json.loads(run_json.read_text())
    surface = tmp_path / "spike.json"
    surface.write_text(
        json.dumps(_surface(payload["param_hash"], peak=1.0, ring=0.1)),
        encoding="utf-8",
    )
    out = tmp_path / "spike_out"
    argv = _argv(run_json, bars_parquet, out, benchmark=benchmark_parquet)
    argv += ["--sensitivity", str(surface)]
    main(argv)

    verdict = json.loads((out / "verdict.json").read_text())
    gate = next(g for g in verdict["gates"] if g["name"] == "sensitivity_plateau")
    assert gate["state"] == "FAIL"


def test_a_surface_around_a_different_point_is_refused(
    run_json, bars_parquet, benchmark_parquet, tmp_path
) -> None:
    """A neighbourhood built around someone else's peak must not excuse this run."""
    surface = tmp_path / "foreign.json"
    surface.write_text(json.dumps(_surface("some_other_hash")), encoding="utf-8")

    out = tmp_path / "foreign_out"
    argv = _argv(run_json, bars_parquet, out, benchmark=benchmark_parquet)
    argv += ["--sensitivity", str(surface)]
    assert main(argv) == EXIT_USAGE


def test_a_surface_with_no_peak_is_refused(
    run_json, bars_parquet, benchmark_parquet, tmp_path
) -> None:
    payload = json.loads(run_json.read_text())
    surface = _surface(payload["param_hash"])
    surface["points"] = [p for p in surface["points"] if p["offsets"]["p"] != 0]
    path = tmp_path / "peakless.json"
    path.write_text(json.dumps(surface), encoding="utf-8")

    out = tmp_path / "peakless_out"
    argv = _argv(run_json, bars_parquet, out, benchmark=benchmark_parquet)
    argv += ["--sensitivity", str(path)]
    assert main(argv) == EXIT_USAGE
