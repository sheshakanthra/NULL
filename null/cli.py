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
from null.costs.robustness import load_rate_robustness
from null.data.ohlcv import DEFAULT_CACHE as OHLCV_CACHE
from null.benchmark.tri import DEFAULT_CACHE as DEFAULT_TRI_CACHE
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


def enrich_trials_from_parquet(run: StrategyRun, path: Path) -> StrategyRun:
    """Rehydrate ``run.trials[i].returns`` from a sibling trial-returns parquet.

    A ``run.json`` may legitimately omit per-trial return series -- the contract
    says so explicitly: "trials may be empty or a subset ... never a substitute
    for n_trials." That is what lets a large grid's run.json stay small. But
    dropping the return series silently costs PBO its real evidence and leaves it
    reporting NOT_COMPUTABLE, which is a genuine loss of audit fidelity, not a
    convenience. This is the other half: given a sibling file, restore it.

    Schema: a wide-format parquet with a ``date`` column (bar-close timestamps,
    matching the strategy's own return series) plus one float column per trial,
    named by that trial's ``param_hash``. One row per date, shared across every
    trial -- true whenever every variant was backtested over the same bars, which
    a parameter grid search always is.

    Raises rather than silently proceeding with partial data if a trial's
    param_hash has no matching column: a partially-enriched trial set would look
    complete and quietly understate the evidence PBO sees.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        raise InputError(f"{path} has no 'date' column; not a trial-returns parquet.")
    stamps = tuple(ts.to_pydatetime() for ts in pd.to_datetime(frame["date"]))

    missing = [t.param_hash for t in run.trials if t.param_hash not in frame.columns]
    if missing:
        raise InputError(
            f"{path} is missing columns for {len(missing)} trial(s) declared in the "
            f"run, e.g. {missing[0]!r}. Every trial must be fully enriched or not "
            "enriched at all -- a partial join would understate PBO's evidence "
            "while looking complete."
        )

    enriched = tuple(
        trial.model_copy(
            update={"returns": _series(frame[trial.param_hash].to_numpy(), stamps)}
        )
        for trial in run.trials
    )
    return run.model_copy(update={"trials": enriched})


def _series(values: np.ndarray, stamps: Sequence[object]) -> Series:
    return Series(
        ts=tuple(stamps),  # type: ignore[arg-type]
        values=tuple(float(v) for v in values),
    )


def load_sensitivity(path: Path, run: StrategyRun) -> SensitivityResult:
    """Read a parameter-neighbourhood surface and check it belongs to this run.

    The surface is the evidence the sensitivity_plateau gate judges on, so it is
    read and then verified, never trusted. Two checks, both fatal:

    * the surface must contain the peak, the all-zero offset point. Without it
      there is nothing for the neighbourhood to be a neighbourhood *of*, and the
      ratio the gate reads would be measured against an arbitrary member.
    * that peak's ``param_hash`` must equal the run's. A surface built around a
      different point describes some other strategy's neighbourhood, and pairing
      it with this run would let a plateau found elsewhere excuse a spike here.

    Both are the failure this gate exists to catch, so neither degrades to a
    warning -- a surface that cannot be tied to the run is missing evidence.
    """
    try:
        surface = SensitivityResult.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise InputError(_readable_validation_error(path, exc)) from exc

    peaks = [p for p in surface.points if all(v == 0 for v in p.offsets.values())]
    if not peaks:
        raise InputError(
            f"{path} has no all-zero offset point, so it carries no peak. A "
            "neighbourhood without the point it surrounds cannot be scored."
        )
    if peaks[0].param_hash != run.param_hash:
        raise InputError(
            f"{path} is a neighbourhood around param_hash {peaks[0].param_hash!r}, "
            f"but the run submitted {run.param_hash!r}. This surface describes a "
            "different point; scoring the run against it would let a plateau found "
            "elsewhere stand in for this one."
        )
    return surface


def build_evidence(
    run: StrategyRun,
    bars: tuple[Bar, ...],
    benchmark_bars: tuple[Bar, ...],
    costs: IndiaEquityCostModel,
    cost_robustness_path: Path | None = None,
    benchmark_is_total_return: bool = False,
    sensitivity_surface: SensitivityResult | None = None,
) -> tuple[Evidence, dict[str, object]]:
    """Assemble the Evidence the gates consume, plus the report context.

    Only reached when leakage is clean; the caller short-circuits otherwise.

    ``cost_robustness_path`` is an optional cost-rate sensitivity sweep. When the
    charge rates are unverified, the limitations band otherwise has to say only
    that -- which warns the reader without telling them which conclusions the
    unverified rates actually put at risk. A sweep turns that into a measured
    statement. It is read, never trusted: the finding is recomputed from the
    file's raw columns and a malformed file raises rather than degrading to a
    reassuring default.
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

    # A supplied surface is the real neighbourhood and the gate judges on it. With
    # none, no parameter grid arrived with the run, so the surface holds the peak
    # alone and the gate reports NOT_COMPUTABLE rather than crying curve-fitting.
    sensitivity = sensitivity_surface or SensitivityResult(
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
        "benchmark_is_total_return": benchmark_is_total_return,
        "universe_is_point_in_time": False,
        "risk_free_supplied": False,
        "golden_suite_green": False,
        "leakage_checks_unchecked": leakage.unchecked,
        "pbo_rationale": pbo.rationale,
        "expected_max_sharpe_sentence": dsr.selection_diagnostic,
        "mtrl_rationale": mtrl.rationale,
        "_dsr": dsr,
    }
    if cost_robustness_path is not None:
        context["cost_rate_robustness"] = load_rate_robustness(
            cost_robustness_path, n_variants=run.n_trials
        ).sentence
    return evidence, context


def _cached_tri_is_validated() -> bool:
    """Whether the committed TRI cache carries a PASSING total-return validation.

    Fails closed. The claim "this benchmark is total return" has to be backed by
    evidence on disk -- the validation the fetcher recorded in the provenance
    sidecar -- not by the fact that a particular code path was taken. No sidecar,
    no validation block, or a failing one, all read as unconfirmed, and the
    limitations band then says so.
    """
    sidecar = DEFAULT_TRI_CACHE.with_name("nifty50_tri.provenance.json")
    if not sidecar.exists():
        return False
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    validation = recorded.get("tri_validation")
    return isinstance(validation, dict) and validation.get("is_valid") is True


def _load_benchmark(path_arg: str | None) -> tuple[tuple[Bar, ...], bool]:
    """The benchmark series, and whether it is a confirmed total-return index.

    Defaulting to the audited bars would compare a strategy against itself, and for
    a multi-symbol universe it is not even a coherent series -- the timestamps
    interleave across symbols. There is no fallback here for the same reason
    null/benchmark/tri.py has none: a missing benchmark is a stop.

    A caller-supplied series is never treated as total return: NULL has not seen it
    validated, and assuming otherwise would silence the one limitation that catches
    a price index being benchmarked against.
    """
    if path_arg:
        candidate = load_bars(Path(path_arg))
        symbols = {b.symbol for b in candidate}
        if len(symbols) != 1:
            raise ValueError(
                f"{path_arg} holds {len(symbols)} symbols {sorted(symbols)}; the "
                "benchmark must be a single series."
            )
        return candidate, False

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
    ), _cached_tri_is_validated()


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
    if args.trials_parquet:
        run = enrich_trials_from_parquet(run, Path(args.trials_parquet))

    try:
        costs = IndiaEquityCostModel.from_yaml(Path(args.costs))
    except (OSError, KeyError, ValidationError) as exc:
        raise InputError(f"could not load cost config {args.costs}: {exc}") from exc

    bars_path = Path(args.bars) if args.bars else OHLCV_CACHE
    try:
        bars = load_bars(bars_path, symbols=tuple(run.universe))
        benchmark_bars, benchmark_is_tri = _load_benchmark(args.benchmark)
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
        evidence, context = build_evidence(
            run,
            bars,
            benchmark_bars,
            costs,
            cost_robustness_path=(
                Path(args.cost_robustness) if args.cost_robustness else None
            ),
            benchmark_is_total_return=benchmark_is_tri,
            sensitivity_surface=(
                load_sensitivity(Path(args.sensitivity), run)
                if args.sensitivity
                else None
            ),
        )
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
    audit.add_argument(
        "--trials-parquet",
        default=None,
        help=(
            "sibling parquet of per-trial return series (wide format: a 'date' "
            "column plus one column per trial param_hash), for when run.json "
            "declares its trials without embedding full Series -- restores real "
            "PBO evidence instead of NOT_COMPUTABLE"
        ),
    )
    audit.add_argument(
        "--cost-robustness",
        default=None,
        help=(
            "cost-rate sensitivity sweep CSV (see null/costs/robustness.py for the "
            "required columns). Charge rates that have never been reconciled against "
            "a broker contract note make every cost LEVEL unverified; a sweep "
            "establishes which CONCLUSIONS survive that, and the limitations band "
            "reports the measured result instead of a blanket disclaimer"
        ),
    )
    audit.add_argument(
        "--sensitivity",
        default=None,
        help=(
            "parameter-neighbourhood surface JSON (a serialised SensitivityResult, "
            "as examples/rsi2_nifty/build_run.py writes). Without it the "
            "sensitivity_plateau gate has only the submitted point and reports "
            "NOT_COMPUTABLE -- it does not judge, and NOT_COMPUTABLE is not a pass. "
            "A parameter grid already IS this surface; supply it and the gate can "
            "tell a plateau from a spike"
        ),
    )
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
