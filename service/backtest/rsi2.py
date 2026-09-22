"""Bounded RSI(2) backtester for the ``POST /audit/rsi2`` endpoint.

Turns a caller-chosen RSI(2) grid (period / entry / exit / holding-cap value
lists) into a ``run.json`` + trials parquet in the exact schema ``null audit``
consumes -- by driving the same strategy code the committed
``examples/rsi2_nifty`` demo uses (``examples/rsi2_nifty/strategy.py``,
``examples/rsi2_nifty/build_run.py``), not a reimplementation of it. The one
thing genuinely new here is *bounding*: the reference demo hardcodes one
108-variant grid; a live endpoint takes the grid from an HTTP caller, so the
grid itself is now untrusted input and must be validated before a single bar
of backtest runs.

The whole point of this module, per the phase brief: the backtester must not
feed the auditor garbage. A ``run.json`` built wrong means NULL audits noise
and the verdict that comes back is meaningless -- worse than no verdict,
because it looks like one. So the same discipline CLAUDE.md holds ``null/``
to applies here: validate rather than clamp (a silently clamped grid audits a
different strategy than the one the caller asked about), fail loudly on
missing data, and let ``n_trials`` be exactly the caller's own grid size --
never hardcoded, never inferred, never inflated past what was actually run.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from examples.rsi2_nifty.build_run import write_run_artifacts
from examples.rsi2_nifty.strategy import GridVariant, VariantResult, run_grid
from null.contracts import ParamPoint, SensitivityResult
from null.costs.india_equity import IndiaEquityCostModel
from null.data.ohlcv import DEFAULT_CACHE as OHLCV_CACHE
from null.data.ohlcv import load_bars

__all__ = [
    "GridSpec",
    "GridSpecError",
    "BacktestArtifacts",
    "MAX_GRID_VARIANTS",
    "MIN_PERIOD",
    "MAX_PERIOD",
    "MIN_THRESHOLD",
    "MAX_THRESHOLD",
    "MIN_HOLDING_CAP",
    "MAX_HOLDING_CAP",
    "build_grid_spec",
    "run_backtest",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
COSTS_CONFIG = REPO_ROOT / "configs" / "costs_india_equity.yaml"
CONSTITUENTS = REPO_ROOT / "configs" / "nifty50_constituents.txt"

#: Same capital the committed example uses -- a position-sizing choice, not a
#: contract value, so it has no configs/ entry of its own. Kept identical to
#: examples/rsi2_nifty/build_run.py so a caller's chosen grid is judged on the
#: same footing as the committed one, not a different capacity regime.
INITIAL_CAPITAL = 10_000_000.0

# Bounds are a safety surface, not a strategy opinion -- see the module
# docstring. Each is wide enough to cover the committed grid (period {2,3,4},
# entry {5,10,15}, exit {50,60,70}, cap {3,5,10,15}) with room either side.
MIN_PERIOD, MAX_PERIOD = 2, 50
MIN_THRESHOLD, MAX_THRESHOLD = 1, 99
MIN_HOLDING_CAP, MAX_HOLDING_CAP = 1, 120

#: A grid this size already takes real wall-clock time against the full
#: 50-name universe (the committed 108-variant grid takes tens of seconds).
#: Past this, a request stops being "try a parameter neighbourhood" and
#: starts being "run an unbounded grid search through a free HTTP endpoint" --
#: exactly the DoS surface the phase brief calls out to bound now rather than
#: defer to W2.
MAX_GRID_VARIANTS = 200


class GridSpecError(ValueError):
    """The caller's grid is outside what this backtester will run.

    Raised rather than clamped: silently shrinking an out-of-band value to
    the nearest allowed one would audit a different strategy than the one the
    caller asked about, under the same request. A caller needs to know their
    grid was rejected, not discover after the fact that "period 200" quietly
    became "period 50".
    """


@dataclass(frozen=True)
class GridSpec:
    """A validated RSI(2) grid: four axes, each a de-duplicated, sorted,
    in-band tuple of int values. Construct via :func:`build_grid_spec`, never
    directly -- the invariants above are exactly what that function checks.
    """

    periods: tuple[int, ...]
    entries: tuple[int, ...]
    exits: tuple[int, ...]
    holding_caps: tuple[int, ...]

    @property
    def n_variants(self) -> int:
        return (
            len(self.periods)
            * len(self.entries)
            * len(self.exits)
            * len(self.holding_caps)
        )

    def offsets_of(self, variant: GridVariant) -> dict[str, int]:
        """Index of a variant's parameters within this spec's own axes.

        The sensitivity surface's neighbourhood is defined over *this*
        request's grid, not the committed example's -- a caller running a
        6-value period axis has a different neighbourhood shape than the
        committed 3-value one, and hardcoding the committed axes here (as
        ``examples/rsi2_nifty/strategy.py``'s ``GridVariant.offsets_from``
        does) would silently mis-index any grid that doesn't match them.
        """
        return {
            "period": self.periods.index(variant.period),
            "entry": self.entries.index(variant.entry),
            "exit": self.exits.index(variant.exit),
            "holding_cap": self.holding_caps.index(variant.holding_cap),
        }


def _validate_axis(name: str, values: tuple[int, ...], low: int, high: int) -> tuple[int, ...]:
    if len(values) == 0:
        raise GridSpecError(f"{name} must have at least one value.")
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise GridSpecError(f"{name} has duplicate value(s): {duplicates}.")
    out_of_band = sorted(v for v in values if not (low <= v <= high))
    if out_of_band:
        raise GridSpecError(
            f"{name} value(s) {out_of_band} are outside the allowed band "
            f"[{low}, {high}]."
        )
    return tuple(sorted(int(v) for v in values))


def build_grid_spec(
    periods: tuple[int, ...],
    entries: tuple[int, ...],
    exits: tuple[int, ...],
    holding_caps: tuple[int, ...],
) -> GridSpec:
    """Validate caller-supplied axes into a :class:`GridSpec`, or raise
    :class:`GridSpecError` naming exactly what was rejected and why."""
    spec = GridSpec(
        periods=_validate_axis("periods", periods, MIN_PERIOD, MAX_PERIOD),
        entries=_validate_axis("entries", entries, MIN_THRESHOLD, MAX_THRESHOLD),
        exits=_validate_axis("exits", exits, MIN_THRESHOLD, MAX_THRESHOLD),
        holding_caps=_validate_axis(
            "holding_caps", holding_caps, MIN_HOLDING_CAP, MAX_HOLDING_CAP
        ),
    )
    if spec.n_variants > MAX_GRID_VARIANTS:
        raise GridSpecError(
            f"grid has {spec.n_variants} variants "
            f"({len(spec.periods)} periods x {len(spec.entries)} entries x "
            f"{len(spec.exits)} exits x {len(spec.holding_caps)} holding caps), "
            f"over the {MAX_GRID_VARIANTS}-variant cap for this endpoint."
        )
    return spec


def _load_universe() -> tuple[str, ...]:
    """The NIFTY 50 constituent list, same source and parsing as
    ``examples/rsi2_nifty/build_run.py``'s own ``_load_universe`` -- not
    imported from there since it's a private, six-line helper and the two
    call sites should not couple across a module boundary for that."""
    lines = CONSTITUENTS.read_text(encoding="utf-8").splitlines()
    return tuple(
        sorted(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))
    )


def _build_sensitivity(
    results: tuple[VariantResult, ...], spec: GridSpec, *, metric: str = "net_sharpe"
) -> SensitivityResult:
    """Mirrors ``examples/rsi2_nifty/strategy.py``'s ``build_sensitivity``
    exactly, except the neighbourhood is indexed against ``spec``'s own axes
    (see :meth:`GridSpec.offsets_of`) rather than that module's hardcoded
    ``RSI_PERIODS`` / ``ENTRY_THRESHOLDS`` / ``EXIT_THRESHOLDS`` /
    ``HOLDING_CAPS`` constants, which only describe the one committed grid.
    """
    best = max(results, key=lambda r: getattr(r, metric))
    best_idx = spec.offsets_of(best.variant)

    points: list[ParamPoint] = []
    neighbour_values: list[float] = []
    for result in results:
        idx = spec.offsets_of(result.variant)
        offsets = {name: idx[name] - best_idx[name] for name in best_idx}
        value = getattr(result, metric)
        points.append(
            ParamPoint(param_hash=result.variant.param_hash, offsets=offsets, sharpe=value)
        )
        nonzero = [name for name, off in offsets.items() if off != 0]
        if len(nonzero) == 1 and abs(offsets[nonzero[0]]) == 1:
            neighbour_values.append(value)

    peak = float(getattr(best, metric))
    mean_neighbour = float(np.mean(neighbour_values)) if neighbour_values else peak
    ratio = max(0.0, min(mean_neighbour / peak, 1.0)) if peak > 0.0 else 0.0

    return SensitivityResult(
        param_names=("period", "entry", "exit", "holding_cap"),
        peak_sharpe=peak,
        neighborhood_mean_sharpe=mean_neighbour,
        neighborhood_ratio=ratio,
        points=tuple(points),
    )


@dataclass(frozen=True)
class BacktestArtifacts:
    run_path: Path
    trials_parquet_path: Path
    sensitivity_path: Path
    n_trials: int


def run_backtest(spec: GridSpec, out_dir: Path) -> BacktestArtifacts:
    """Backtest every variant in ``spec`` against the committed NIFTY 50 cache
    and write ``run.json`` / ``run.trials.parquet`` / ``sensitivity.json``
    into ``out_dir``, in exactly the layout ``null audit --trials-parquet
    ... --sensitivity ...`` expects.

    Offline, like everything upstream of the audit path: reads only the
    committed OHLCV cache, never fetches. Raises rather than falling back if
    that cache is missing -- a silently-substituted bar series would audit a
    strategy against data it never actually ran on.
    """
    if not OHLCV_CACHE.exists():
        raise FileNotFoundError(f"no OHLCV cache at {OHLCV_CACHE}")

    universe = _load_universe()
    bars = load_bars(OHLCV_CACHE, symbols=universe)
    if not bars:
        raise ValueError(
            f"no bars in {OHLCV_CACHE} for the {len(universe)}-symbol NIFTY 50 "
            "universe."
        )

    costs = IndiaEquityCostModel.from_yaml(COSTS_CONFIG)

    variants = tuple(
        GridVariant(period=p, entry=e, exit=x, holding_cap=h)
        for p, e, x, h in product(spec.periods, spec.entries, spec.exits, spec.holding_caps)
    )
    if len(variants) != spec.n_variants:
        raise AssertionError(
            f"built {len(variants)} variants from a spec declaring "
            f"{spec.n_variants} -- the grid and n_trials must never disagree."
        )

    results = run_grid(
        bars=bars,
        universe=universe,
        costs=costs,
        initial_capital=INITIAL_CAPITAL,
        variants=variants,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = _build_sensitivity(results, spec, metric="net_sharpe")
    sensitivity_path = out_dir / "sensitivity.json"
    sensitivity_path.write_bytes(sensitivity.canonical_json())

    run_path = out_dir / "run.json"
    trials_parquet_path = out_dir / "run.trials.parquet"
    write_run_artifacts(
        results,
        universe=universe,
        initial_capital=INITIAL_CAPITAL,
        run_out=run_path,
        trials_parquet_out=trials_parquet_path,
        n_trials=len(variants),
    )

    return BacktestArtifacts(
        run_path=run_path,
        trials_parquet_path=trials_parquet_path,
        sensitivity_path=sensitivity_path,
        n_trials=len(variants),
    )
