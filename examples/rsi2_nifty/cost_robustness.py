"""M7: does the gross-vs-net ranking inversion survive plausible rate error?

    python examples/rsi2_nifty/cost_robustness.py [--workers N]

The charge rates in ``configs/costs_india_equity.yaml`` are UNVERIFIED -- written
from the published Indian equity charge stack, never reconciled against a live
contract note, and they are going to stay that way. The honest response to an
assumption you cannot verify is not to hedge every number that depends on it into
uselessness. It is to show what would have to be wrong for the conclusion to
break.

So: scale every charge component independently by +/-25%, re-run the full
108-variant grid against each scaled model, and ask the two questions the M7
finding as originally stated rests on.

  1. Does the best-by-GROSS variant stay net-negative?
  2. Does the best-by-NET variant stay a DIFFERENT variant from the best-by-gross?

Together those are the inversion: selecting on the number a naive backtest shows
you picks a strategy that loses money after costs.

Question 1 is a SIGN test, and at baseline the number it tests sits 0.027 from
zero. A sign test that close to its own boundary is fragile by construction: it
can flip on a rate error without anything about the strategy having changed, and
a finding that flips like that was never really about the strategy. So this sweep
also records two measures of the same phenomenon that do not balance on a knife
edge:

  3. Where does the naive gross pick RANK on the net-ranked grid, out of 108?
  4. How much net Sharpe does selecting on gross GIVE UP against selecting on net?

Those are reported whatever questions 1 and 2 do. If the sign test breaks and the
rank does not, that is not a rescue of the original claim -- it is a different and
better-founded claim, and the README states it as such rather than quietly
substituting one for the other.

Two design notes.

Slippage is swept alongside the statutory rates. It is a charge component like any
other and on a strategy trading 20,888 times it is the largest one; sweeping only
brokerage and STT would be sweeping the small half of the stack and calling it
coverage.

``brokerage_pct`` and ``brokerage_per_order_cap`` are swept even though the
configured brokerage is zero, which makes them provably inert (the model
short-circuits to 0.0 when the rate is zero, and the cap only enters through the
min() inside that branch). They are kept as a negative control on this harness:
two components that CANNOT move the number, which must therefore come back with a
delta of exactly zero. A sweep where every case moves is a sweep that has not
demonstrated it is measuring what it claims to.

Determinism: each case is an independent full grid run, results are collected and
written in a fixed case order, so the CSV is byte-comparable across runs.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from examples.rsi2_nifty.strategy import GridVariant, run_grid
from null.contracts import Bar
from null.costs.india_equity import IndiaEquityCostModel
from null.costs.model import Segment
from null.data.ohlcv import DEFAULT_CACHE as OHLCV_CACHE
from null.data.ohlcv import load_bars

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COSTS_CONFIG = REPO / "configs" / "costs_india_equity.yaml"
CONSTITUENTS = REPO / "configs" / "nifty50_constituents.txt"
REPORT_OUT = HERE / "cost_robustness.csv"

INITIAL_CAPITAL = 10_000_000.0

#: Every rate on the equity_delivery segment. The strategy is delivery-only, so
#: the intraday segment never gets consulted and sweeping it would be theatre.
RATE_COMPONENTS: tuple[str, ...] = (
    "brokerage_pct",
    "brokerage_per_order_cap",
    "stt_buy_pct",
    "stt_sell_pct",
    "exchange_txn_pct",
    "sebi_turnover_pct",
    "stamp_duty_buy_pct",
    "gst_pct",
    "dp_charge_per_scrip_per_sell",
)
SLIPPAGE_COMPONENTS: tuple[str, ...] = ("impact_k", "half_spread_bps")

#: +/-25%, per Sheshakanth's instruction. Not a confidence interval -- a plausible
#: error band for a published charge list read carefully but never reconciled.
FACTORS: tuple[float, ...] = (0.75, 1.25)

#: Beyond the literal ask. Every component moved in the same direction at once is
#: not "independent variation", it is the corner of the box, and it is the case
#: that would break first. Reported separately and labelled as such.
JOINT = "ALL_COMPONENTS_TOGETHER"

BASELINE = "baseline"


def cases() -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = [(BASELINE, 1.0)]
    for component in RATE_COMPONENTS + SLIPPAGE_COMPONENTS:
        for factor in FACTORS:
            out.append((component, factor))
    for factor in FACTORS:
        out.append((JOINT, factor))
    return tuple(out)


def build_model(
    base: IndiaEquityCostModel, component: str, factor: float
) -> IndiaEquityCostModel:
    """The scaled cost model for one case. Pure; the base model is never mutated."""
    if component == BASELINE:
        return base
    if component == JOINT:
        model = base
        for rate_field in RATE_COMPONENTS:
            model = model.with_scaled_rate(Segment.EQUITY_DELIVERY, rate_field, factor)
        for slip_field in SLIPPAGE_COMPONENTS:
            model = model.with_scaled_slippage(slip_field, factor)
        return model
    if component in RATE_COMPONENTS:
        return base.with_scaled_rate(Segment.EQUITY_DELIVERY, component, factor)
    if component in SLIPPAGE_COMPONENTS:
        return base.with_scaled_slippage(component, factor)
    raise KeyError(f"unknown charge component {component!r}")


def _params(variant: GridVariant) -> str:
    return (
        f"period={variant.period},entry={variant.entry},"
        f"exit={variant.exit},cap={variant.holding_cap}"
    )


@dataclass(frozen=True)
class CaseResult:
    component: str
    factor: float
    best_gross_params: str
    best_gross_hash: str
    best_gross_gross_sharpe: float
    #: The number question 1 turns on: the naive pick's Sharpe AFTER costs.
    best_gross_net_sharpe: float
    best_net_params: str
    best_net_hash: str
    best_net_gross_sharpe: float
    best_net_net_sharpe: float
    #: Question 2: is the net-best a different grid point from the gross-best?
    inversion_holds: bool
    naive_pick_loses_money: bool
    n_variants_net_negative: int
    cost_erosion_at_net_best: float
    #: Where the naive gross pick actually lands once the grid is ranked by net,
    #: 1-based out of 108. "It goes net-negative" is a SIGN test on a number sitting
    #: 0.027 from zero, and a sign test that close to the boundary is fragile by
    #: construction -- it can flip on a rate error without anything about the
    #: strategy changing. The rank says the same thing without balancing on a knife
    #: edge: how far down the net-ranked grid does selecting on gross drop you.
    gross_pick_rank_by_net: int
    #: Sharpe given up by selecting on gross instead of net. The magnitude behind
    #: the rank, and the number a reader can act on.
    net_sharpe_given_up_by_selecting_on_gross: float


def summarise(component: str, factor: float, results: tuple) -> CaseResult:
    best_net = max(results, key=lambda r: r.net_sharpe)
    best_gross = max(results, key=lambda r: r.gross_sharpe)
    naive_loses = best_gross.net_sharpe < 0.0
    different = best_gross.variant.param_hash != best_net.variant.param_hash

    # Rank the gross pick by net. Ties broken by param_hash so the rank is
    # deterministic rather than dependent on sort stability across runs.
    by_net = sorted(results, key=lambda r: (-r.net_sharpe, r.variant.param_hash))
    rank = 1 + next(
        i for i, r in enumerate(by_net)
        if r.variant.param_hash == best_gross.variant.param_hash
    )
    return CaseResult(
        component=component,
        factor=factor,
        best_gross_params=_params(best_gross.variant),
        best_gross_hash=best_gross.variant.param_hash,
        best_gross_gross_sharpe=best_gross.gross_sharpe,
        best_gross_net_sharpe=best_gross.net_sharpe,
        best_net_params=_params(best_net.variant),
        best_net_hash=best_net.variant.param_hash,
        best_net_gross_sharpe=best_net.gross_sharpe,
        best_net_net_sharpe=best_net.net_sharpe,
        inversion_holds=naive_loses and different,
        naive_pick_loses_money=naive_loses,
        n_variants_net_negative=sum(1 for r in results if r.net_sharpe < 0.0),
        cost_erosion_at_net_best=best_net.gross_sharpe - best_net.net_sharpe,
        gross_pick_rank_by_net=rank,
        net_sharpe_given_up_by_selecting_on_gross=(
            best_net.net_sharpe - best_gross.net_sharpe
        ),
    )


# Worker-process state. Each worker loads the bars once and reuses them across
# every case it is handed; on Windows there is no fork, so this cannot be
# inherited from the parent and is rebuilt per process instead.
_BARS: tuple[Bar, ...] | None = None
_UNIVERSE: tuple[str, ...] | None = None


def _universe() -> tuple[str, ...]:
    lines = CONSTITUENTS.read_text(encoding="utf-8").splitlines()
    return tuple(
        sorted(s for s in (l.strip() for l in lines) if s and not s.startswith("#"))
    )


def _worker_state() -> tuple[tuple[Bar, ...], tuple[str, ...]]:
    global _BARS, _UNIVERSE
    if _BARS is None or _UNIVERSE is None:
        _UNIVERSE = _universe()
        _BARS = load_bars(OHLCV_CACHE, symbols=_UNIVERSE)
    return _BARS, _UNIVERSE


def run_case(case: tuple[str, float]) -> CaseResult:
    component, factor = case
    bars, universe = _worker_state()
    base = IndiaEquityCostModel.from_yaml(COSTS_CONFIG)
    results = run_grid(
        bars=bars,
        universe=universe,
        costs=build_model(base, component, factor),
        initial_capital=INITIAL_CAPITAL,
    )
    return summarise(component, factor, results)


def main() -> int:
    parser = argparse.ArgumentParser(description="cost-rate robustness sweep")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="parallel grid runs; memory-bound, not CPU-bound (default 3)",
    )
    args = parser.parse_args()

    if not OHLCV_CACHE.exists():
        print(f"No OHLCV cache at {OHLCV_CACHE}. Nothing to run.")
        return 2

    todo = cases()
    print(f"{len(todo)} full 108-variant grid runs, {args.workers} at a time.")
    print(
        f"Components swept: {len(RATE_COMPONENTS)} statutory rates + "
        f"{len(SLIPPAGE_COMPONENTS)} slippage, at {FACTORS}, plus a joint case."
    )
    started = time.monotonic()

    collected: dict[tuple[str, float], CaseResult] = {}
    with mp.Pool(processes=args.workers) as pool:
        for done, result in enumerate(pool.imap_unordered(run_case, todo), start=1):
            collected[(result.component, result.factor)] = result
            print(
                f"  [{done:2d}/{len(todo)}] {result.component}@{result.factor:.2f}  "
                f"naive-pick net {result.best_gross_net_sharpe:+.3f} "
                f"(rank {result.gross_pick_rank_by_net}/108)  "
                f"best-net {result.best_net_net_sharpe:+.3f}  "
                f"gives up {result.net_sharpe_given_up_by_selecting_on_gross:.3f}  "
                f"sign-test={'HOLDS' if result.inversion_holds else 'BROKEN'}  "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )

    # Fixed case order, not completion order: the artifact must be byte-stable.
    ordered = [collected[case] for case in todo]

    fields = list(asdict(ordered[0]).keys())
    with REPORT_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            record: dict[str, object] = dict(asdict(row))
            for key, value in record.items():
                if isinstance(value, float):
                    record[key] = f"{value:.6f}"
            writer.writerow(record)
    print(f"\nWrote {REPORT_OUT}")

    broken = [r for r in ordered if not r.inversion_holds]
    ranks = [r.gross_pick_rank_by_net for r in ordered]
    given_up = [r.net_sharpe_given_up_by_selecting_on_gross for r in ordered]
    same_variant = [r for r in ordered if r.best_net_hash == r.best_gross_hash]

    print()
    print("=" * 72)
    print("THE SIGN TEST -- does the naive gross pick stay net-NEGATIVE?")
    print("=" * 72)
    if broken:
        print(f"  BREAKS in {len(broken)} of {len(ordered)} cases:")
        for r in broken:
            print(
                f"    {r.component}@{r.factor:.2f}: naive-pick net "
                f"{r.best_gross_net_sharpe:+.3f}"
            )
    else:
        print(f"  Holds in all {len(ordered)} cases.")
    print()
    print("=" * 72)
    print("THE RANK -- where does the naive gross pick land on the net-ranked grid?")
    print("=" * 72)
    print(f"  Rank of the gross pick by net, across all {len(ordered)} cases: "
          f"{min(ranks)} to {max(ranks)} out of 108")
    print(f"  Net Sharpe given up by selecting on gross: "
          f"{min(given_up):.3f} to {max(given_up):.3f}")
    print(f"  Cases where the gross pick WAS the net pick: {len(same_variant)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
