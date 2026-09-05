"""``null`` command line. BUILD.md §13.

    null audit run.json --config configs/gates_default.yaml

Writes ``verdict.json`` and ``report.html``, and exits with the verdict:

    0   PASS
    1   REJECT
    2   usage or input error

The exit code is the interface. A CI job gates on it without parsing output, so a
malformed input must never exit 1 and be mistaken for a considered rejection.

**Fully offline.** Nothing here reaches the network: bars and the benchmark come
from committed parquet caches, and if they are absent the loaders raise rather than
fetching or substituting (CLAUDE.md invariant 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from pydantic import ValidationError

from null.benchmark.buyhold import benchmark_check
from null.contracts import (
    SPEC_VERSION,
    Bar,
    Evidence,
    FoldResult,
    ParamPoint,
    SensitivityResult,
    GateResult,
    Series,
    StrategyRun,
    Verdict,
)
from null.costs.india_equity import IndiaEquityCostModel
from null.data.ohlcv import DEFAULT_CACHE as OHLCV_CACHE
from null.benchmark.tri import load_nifty50_tri
from null.data.ohlcv import load_bars
from null.leakage.audit import LeakageReport, audit_leakage
from null.partition.walkforward import walk_forward_consistency, walk_forward_splits
from null.report.render import write_report
from null.stats.deflated_sharpe import deflated_sharpe_ratio
from null.stats.mtrl import minimum_track_record_length
from null.stats.pbo import compute_pbo
from null.stats.reality_check import reality_check
from null.verdict.engine import DEFAULT_GATES_CONFIG, GateConfigError, evaluate

__all__ = ["main"]

PACKAGE_VERSION = "0.1.0"

EXIT_PASS = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

DEFAULT_COSTS = (
    Path(__file__).resolve().parents[1] / "configs" / "costs_india_equity.yaml"
)


class InputError(Exception):
    """Something about the caller's input is wrong. Always exit 2, never exit 1."""


def _readable_validation_error(path: Path, error: ValidationError) -> str:
    """A Pydantic traceback is not a usable message for someone auditing a strategy."""
    lines = [f"{path} is not a valid strategy run:"]
    for item in error.errors():
        location = ".".join(str(p) for p in item["loc"]) or "(top level)"
        lines.append(f"  {location}: {item['msg']}")
    lines.append("")
    lines.append(
        "A strategy run needs: strategy_id, param_hash, n_trials (required, never "
        "defaulted), universe, weights, initial_capital. decision_lag_bars defaults "
        "to 1 and must be at least 1."
    )
    return "\n".join(lines)


def load_run(path: Path) -> StrategyRun:
    if not path.exists():
        raise InputError(f"{path} does not exist.")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"could not read {path}: {exc}") from exc
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}."
        ) from exc
    try:
        return StrategyRun.model_validate_json(raw)
    except ValidationError as exc:
        raise InputError(_readable_validation_error(path, exc)) from exc


def _series(values: np.ndarray, stamps: Sequence[object]) -> Series:
    return Series(
        ts=tuple(stamps),  # type: ignore[arg-type]
        values=tuple(float(v) for v in values),
    )


def build_evidence(
    run: StrategyRun,
    bars: tuple[Bar, ...],
    benchmark_bars: tuple[Bar, ...],
    costs: IndiaEquityCostModel,
) -> tuple[Evidence, dict[str, object]]:
    """Assemble the Evidence the gates consume, plus the report context.

    Only reached when leakage is clean; the caller short-circuits otherwise.
    """
    leakage = audit_leakage(run, bars)
    bench = benchmark_check(
        run=run, bars=bars, benchmark_bars=benchmark_bars, costs=costs
    )

    net = bench.strategy_returns.to_numpy()
    trial_returns = None
    trial_sharpes = None
    supplied = [t.returns for t in run.trials if t.returns is not None]
    if len(supplied) >= 2:
        width = min(len(s) for s in supplied)
        trial_returns = np.column_stack([s.to_numpy()[:width] for s in supplied])
        trial_sharpes = np.asarray(
            [t.sharpe for t in run.trials if t.returns is not None], dtype=np.float64
        )
    elif run.trials:
        trial_sharpes = np.asarray([t.sharpe for t in run.trials], dtype=np.float64)

    dsr = deflated_sharpe_ratio(
        returns=net, n_trials=run.n_trials, trial_sharpes=trial_sharpes
    )
    pbo = compute_pbo(trial_returns, n_trials=run.n_trials)
    candidates = trial_returns if trial_returns is not None else net[:, None]
    bench_net = bench.benchmark_returns.to_numpy()
    width = min(candidates.shape[0], bench_net.size)
    rc = reality_check(candidates[:width], bench_net[:width])
    mtrl = minimum_track_record_length(net)

    folds = walk_forward_consistency(net)
    splits = walk_forward_splits(net.size)
    stamps = bench.strategy_returns.ts
    walkforward = tuple(
        FoldResult(
            fold_index=split.fold_index,
            train_start=stamps[split.train_start],
            train_end=stamps[max(split.train_end - 1, split.train_start)],
            test_start=stamps[split.test_start],
            test_end=stamps[min(split.test_end - 1, len(stamps) - 1)],
            purged_bars=split.purged_bars,
            embargo_bars=split.embargo_bars,
            metrics=bench.metrics,
            net_return=value,
        )
        for split, value in zip(splits, folds.fold_returns)
    )

    # No parameter grid arrives with a single run, so the surface holds the peak
    # alone and the gate reports NOT_COMPUTABLE rather than crying curve-fitting.
    sensitivity = SensitivityResult(
        param_names=("submitted",),
        peak_sharpe=dsr.observed_sharpe_annual,
        neighborhood_mean_sharpe=dsr.observed_sharpe_annual,
        neighborhood_ratio=1.0,
        points=(
            ParamPoint(
                param_hash=run.param_hash,
                offsets={"submitted": 0},
                sharpe=dsr.observed_sharpe_annual,
            ),
        ),
    )

    evidence = Evidence(
        equity_curve=_series(np.cumprod(1.0 + net), stamps),
        benchmark_curve=_series(np.cumprod(1.0 + bench_net), bench.benchmark_returns.ts),
        net_returns=bench.strategy_returns,
        gross_returns=bench.strategy_gross_returns,
        cost_breakdown=dict(bench.cost_breakdown),
        turnover_annual=bench.metrics.turnover_annual,
        time_in_market=bench.metrics.time_in_market,
        metrics=bench.metrics,
        benchmark_metrics=bench.benchmark_metrics,
        alpha=bench.alpha,
        deflated_sharpe=dsr.deflated_sharpe,
        pbo=pbo.pbo,
        reality_check_p=rc.p_value,
        mtrl_years=mtrl.mtrl_years,
        max_adv_participation=bench.max_adv_participation,
        walkforward=walkforward,
        regimes={"full_sample": bench.metrics},
        sensitivity=sensitivity,
        leakage_flags=leakage.flags,
    )

    context: dict[str, object] = {
        "rates_are_verified": costs.config.rates_are_verified,
        "benchmark_is_total_return": False,
        "universe_is_point_in_time": False,
        "risk_free_supplied": False,
        "golden_suite_green": False,
        "leakage_checks_unchecked": leakage.unchecked,
        "pbo_rationale": pbo.rationale,
        "expected_max_sharpe_sentence": dsr.selection_diagnostic,
        "mtrl_rationale": mtrl.rationale,
        "_dsr": dsr,
    }
    return evidence, context


def _load_benchmark(path_arg: str | None) -> tuple[Bar, ...]:
    """The benchmark series. Never the strategy's own bars.

    Defaulting to the audited bars would compare a strategy against itself, and for
    a multi-symbol universe it is not even a coherent series -- the timestamps
    interleave across symbols. There is no fallback here for the same reason
    null/benchmark/tri.py has none: a missing benchmark is a stop.
    """
    if path_arg:
        candidate = load_bars(Path(path_arg))
        symbols = {b.symbol for b in candidate}
        if len(symbols) != 1:
            raise ValueError(
                f"{path_arg} holds {len(symbols)} symbols {sorted(symbols)}; the "
                "benchmark must be a single series."
            )
        return candidate

    series = load_nifty50_tri()
    values = series.to_numpy()
    # The index is a level series, not a traded instrument. adv_20 is left unset,
    # and benchmark_check charges the benchmark's entry spread without an impact
    # term rather than inventing a traded volume for an index.
    return tuple(
        Bar(
            ts=stamp,
            symbol="NIFTY50_TRI",
            open=float(values[max(i - 1, 0)]),
            high=float(max(values[max(i - 1, 0)], values[i])),
            low=float(min(values[max(i - 1, 0)], values[i])),
            close=float(values[i]),
            volume=0.0,
            adv_20=None,
        )
        for i, stamp in enumerate(series.ts)
    )


def _leakage_only_verdict(run: StrategyRun, leakage: LeakageReport) -> Verdict:
    """A verdict carrying the leakage gate alone. No statistics exist to report."""
    fatal = leakage.fatal
    gate = GateResult(
        name="leakage_clean",
        state="FAIL",
        passed=False,
        observed=fatal[0].kind,
        threshold="no fatal leakage flags",
        rationale=(
            f"Fatal leakage: {len(fatal)} flag(s), the first of kind "
            f"{fatal[0].kind!r}. {fatal[0].detail} The audit stopped here and no "
            "performance statistic was computed -- a Sharpe ratio for a strategy that "
            "can see the future is not a weak result, it is a meaningless one, and "
            "reporting it would invite belief in a number that describes nothing."
        ),
    )
    return Verdict(
        result="REJECT",
        gates=(gate,),
        evidence_hash=leakage.content_hash(),
        spec_version=SPEC_VERSION,
        generated_from=run,
    )


def _leakage_only_report(run: StrategyRun, leakage: LeakageReport) -> str:
    flags = "".join(
        f"<li><strong>{f.kind}</strong> &mdash; {f.detail}</li>" for f in leakage.fatal
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>NULL verdict &mdash; {run.strategy_id}</title></head><body>"
        f"<h1>{run.strategy_id}</h1><p><strong>REJECT</strong> &mdash; fatal leakage."
        "</p><h2>Why</h2><ul>" + flags + "</ul><p>The audit stopped before any "
        "statistic was computed. There is no Sharpe ratio on this report because "
        "computing one for a strategy that can see the future would produce a number "
        "that is both meaningless and persuasive.</p></body></html>\n"
    )


def run_audit_command(args: argparse.Namespace) -> int:
    out = Path(args.out)
    verdict_path = out / "verdict.json"
    report_path = out / "report.html"

    if verdict_path.exists() and not args.force:
        raise InputError(
            f"{verdict_path} already exists. Refusing to overwrite a verdict without "
            "--force: a verdict is an audit artifact, and silently replacing one "
            "loses the record of what was previously concluded."
        )

    run = load_run(Path(args.run))

    try:
        costs = IndiaEquityCostModel.from_yaml(Path(args.costs))
    except (OSError, KeyError, ValidationError) as exc:
        raise InputError(f"could not load cost config {args.costs}: {exc}") from exc

    bars_path = Path(args.bars) if args.bars else OHLCV_CACHE
    try:
        bars = load_bars(bars_path, symbols=tuple(run.universe))
        benchmark_bars = _load_benchmark(args.benchmark)
    except (FileNotFoundError, ValueError) as exc:
        raise InputError(str(exc)) from exc

    if not bars:
        raise InputError(
            f"no bars found in {bars_path} for symbols {list(run.universe)}. The "
            "universe and the cache do not overlap."
        )

    # BUILD.md §5: leakage runs BEFORE any statistic, and a fatal flag ends the
    # audit. Computing a Sharpe here and reporting it alongside the rejection would
    # be the exact failure the short-circuit exists to prevent.
    leakage = audit_leakage(run, bars)
    if not leakage.is_clean:
        out.mkdir(parents=True, exist_ok=True)
        verdict = _leakage_only_verdict(run, leakage)
        verdict_path.write_bytes(verdict.canonical_json())
        report_path.write_text(
            _leakage_only_report(run, leakage), encoding="utf-8", newline="\n"
        )
        print(f"REJECT: {run.strategy_id}")
        print("  failed gates: leakage_clean")
        print("  audit stopped before any statistic was computed")
        print(f"  {verdict_path}")
        print(f"  {report_path}")
        return EXIT_REJECT

    try:
        evidence, context = build_evidence(run, bars, benchmark_bars, costs)
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    dsr = context.pop("_dsr")

    try:
        report = evaluate(
            run=run,
            evidence=evidence,
            context=context,
            config_path=Path(args.config),
        )
    except GateConfigError as exc:
        raise InputError(str(exc)) from exc

    out.mkdir(parents=True, exist_ok=True)
    verdict_path.write_bytes(report.verdict.canonical_json())
    write_report(
        report,
        report_path,
        observed_sharpe=dsr.observed_sharpe_annual,  # type: ignore[attr-defined]
        deflated_sharpe=evidence.deflated_sharpe,
        alpha_tstat=evidence.alpha.alpha_tstat,
        n_observations=evidence.metrics.n_obs,
    )

    failed = [g.name for g in report.verdict.gates if not g.passed]
    print(f"{report.verdict.result}: {run.strategy_id}")
    if failed:
        print(f"  failed gates: {', '.join(failed)}")
    if report.not_computable:
        print(f"  not computable: {', '.join(report.not_computable)}")
    print(f"  {verdict_path}")
    print(f"  {report_path}")
    return EXIT_PASS if report.verdict.result == "PASS" else EXIT_REJECT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="null", description=__doc__)
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the package version and the contract spec version",
    )
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser("audit", help="audit a strategy run")
    audit.add_argument("run", help="path to run.json")
    audit.add_argument("--config", default=str(DEFAULT_GATES_CONFIG))
    audit.add_argument("--costs", default=str(DEFAULT_COSTS))
    audit.add_argument("--bars", default=None, help="OHLCV parquet (default: cache)")
    audit.add_argument("--benchmark", default=None, help="benchmark parquet")
    audit.add_argument("--out", default=".", help="output directory")
    audit.add_argument("--force", action="store_true", help="overwrite verdict.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        # A verdict is only interpretable against the spec that produced it.
        print(f"null {PACKAGE_VERSION}")
        print(f"contract spec_version {SPEC_VERSION}")
        return EXIT_PASS

    if args.command != "audit":
        parser.print_usage(sys.stderr)
        print("null: a command is required (try `null audit run.json`)", file=sys.stderr)
        return EXIT_USAGE

    try:
        return run_audit_command(args)
    except InputError as exc:
        print(f"null: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
