"""Reading a cost-rate robustness sweep, so a report can cite one as evidence.

Some charge rates cannot be verified. Reconciling a charge list against a live
contract note needs an account that has actually traded, and a project that does
not have one is stuck with rates read carefully off a published schedule. The
temptation then is to hedge: stamp "indicative only" on every cost figure and let
the reader work out what that voids. That is not a disclosure, it is an
abdication -- it tells the reader the numbers might be wrong without telling them
which conclusions actually depend on it.

The alternative is to measure. Scale every charge component across a plausible
error band, re-run the whole thing against each scaled model, and see which
conclusions move. A conclusion that survives the sweep is not proven right, but it
is demonstrably not resting on the unverified rate. A conclusion that does not
survive is exactly the one the reader needed warning about.

This module reads the result of such a sweep and turns it into a sentence the
limitations band can state. It defines the CSV format so that ``null`` owns the
schema and the caller conforms to it, rather than a limitation in ``null`` being
shaped around whatever some example script happened to emit.

The reader is deliberately unforgiving. A missing column, an empty file or a
non-boolean verdict raises rather than defaulting -- a robustness claim that
silently degrades to "holds" on a malformed file is worse than no claim at all,
and CLAUDE.md invariant 6 says missing evidence fails rather than passes.
"""

from __future__ import annotations

import csv
from pathlib import Path

from null.contracts import NonEmptyStr, NullFloat, NullModel

__all__ = [
    "REQUIRED_COLUMNS",
    "RateRobustness",
    "RobustnessFormatError",
    "load_rate_robustness",
]

#: The columns a sweep CSV must carry. ``component``/``factor`` identify the case;
#: the three that follow are the finding itself. Anything else in the file is
#: ignored, so a sweep may report more than this without breaking the contract.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "component",
    "factor",
    "best_gross_net_sharpe",
    "best_gross_hash",
    "best_net_hash",
)

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


class RobustnessFormatError(ValueError):
    """The sweep file cannot be read as a robustness result. A stop, not a warning."""


class RateRobustness(NullModel):
    """One cost-rate sweep, summarised.

    ``inversion_holds_everywhere`` is the claim a report is allowed to make. It is
    the AND of two per-case facts, recomputed here from the raw columns rather than
    trusted from a summary column the writer could have got wrong:

      * the best-by-gross variant is net-negative -- selecting on gross picks a
        loser once costs are charged, and
      * the best-by-net variant is a different grid point from the best-by-gross.
    """

    source: NonEmptyStr
    n_cases: int
    components: tuple[NonEmptyStr, ...]
    factors: tuple[NullFloat, ...]
    inversion_holds_everywhere: bool
    broken_cases: tuple[NonEmptyStr, ...]
    #: The case NEAREST to breaking the finding -- the least-negative net Sharpe
    #: the naive gross pick achieved anywhere in the sweep. If this is below
    #: zero, it is below zero everywhere.
    naive_pick_net_sharpe_nearest_zero: NullFloat
    naive_pick_net_sharpe_most_negative: NullFloat

    #: Optional, and the reason this module is worth having. "The gross pick loses
    #: money" is a sign test; a sweep that also reports where the gross pick RANKS
    #: on the net-ranked grid, and how much net Sharpe selecting on gross gives up,
    #: is measuring the same phenomenon without balancing on a boundary. Present
    #: only when the sweep supplied both columns for every case.
    gross_pick_rank_range: tuple[int, int] | None = None
    net_sharpe_given_up_range: tuple[NullFloat, NullFloat] | None = None
    #: Grid size, for the rank's denominator. Supplied by the caller (a run's
    #: declared n_trials), not read from the sweep file.
    n_variants: int | None = None

    @property
    def sentence(self) -> str:
        """What the limitations band says. Derived, never hand-written."""
        span = (
            f"{min(self.factors):.2f}x to {max(self.factors):.2f}x"
            if self.factors
            else "no variation"
        )
        preamble = (
            f"Rate sensitivity WAS measured rather than assumed ({self.source}: "
            f"{self.n_cases} full re-runs, every charge component scaled "
            f"independently over {span}). "
        )

        if self.inversion_holds_everywhere:
            sign = (
                f"The gross-vs-net ranking inversion held in all {self.n_cases} "
                "cases: the best-by-gross variant stayed net-negative (nearest to "
                f"zero {self.naive_pick_net_sharpe_nearest_zero:+.3f}, most negative "
                f"{self.naive_pick_net_sharpe_most_negative:+.3f}) and the best-by-net "
                "variant stayed a different grid point throughout. "
            )
        else:
            listed = ", ".join(self.broken_cases[:6])
            more = (
                ""
                if len(self.broken_cases) <= 6
                else f" and {len(self.broken_cases) - 6} more"
            )
            sign = (
                "The claim that the best-by-gross variant is NET-NEGATIVE does NOT "
                f"survive it: that fails in {len(self.broken_cases)} of "
                f"{self.n_cases} cases ({listed}{more}), reaching "
                f"{self.naive_pick_net_sharpe_nearest_zero:+.3f} at its least "
                "negative. It is a sign test on a number close to zero, and this "
                "report does not establish it. "
            )

        if self.gross_pick_rank_range is None or self.net_sharpe_given_up_range is None:
            return preamble + sign + (
                "The sweep did not report where the gross pick ranks on the "
                "net-ranked grid, so the size of the selection error is unmeasured "
                "here even where its sign is not."
            )

        low, high = self.gross_pick_rank_range
        of_n = f" of {self.n_variants}" if self.n_variants else ""
        least, most = self.net_sharpe_given_up_range
        held = "What did survive every case" if not self.inversion_holds_everywhere else "Measured alongside it"
        rank = (
            f"{held}: across all {self.n_cases} cases the naive gross pick ranked "
            f"{low}-{high}{of_n} on the net-ranked grid, and selecting on gross gave "
            f"up {least:.3f} to {most:.3f} net Sharpe against selecting on net. That "
            "magnitude does not depend on which way the rates are wrong."
        )
        return preamble + sign + rank


def _parse_bool(raw: str, *, row: int, column: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise RobustnessFormatError(
        f"row {row}: {column}={raw!r} is not a boolean. A robustness verdict that "
        "cannot be read must not be guessed at."
    )


def _parse_float(raw: str, *, row: int, column: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise RobustnessFormatError(f"row {row}: {column}={raw!r} is not a number") from exc


def load_rate_robustness(path: Path, n_variants: int | None = None) -> RateRobustness:
    """Read a sweep CSV. Raises rather than returning a degraded result.

    ``n_variants`` is the grid size, used only as the denominator when reporting a
    rank. It comes from the caller (a run's declared ``n_trials``) rather than the
    sweep file, because the sweep reports what it measured and the run declares how
    many variants there were -- and those two agreeing is the caller's business to
    check, not something to infer from a CSV.
    """
    if not path.exists():
        raise RobustnessFormatError(
            f"No cost-rate robustness sweep at {path}. A report may only cite a "
            "sweep that exists."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RobustnessFormatError(f"{path} has a header but no cases.")

    missing = [c for c in REQUIRED_COLUMNS if c not in (rows[0].keys())]
    if missing:
        raise RobustnessFormatError(
            f"{path} is missing required column(s): {', '.join(missing)}. Expected "
            f"{', '.join(REQUIRED_COLUMNS)}."
        )

    components: list[str] = []
    factors: list[float] = []
    broken: list[str] = []
    naive_nets: list[float] = []
    ranks: list[int] = []
    given_up: list[float] = []

    for number, row in enumerate(rows, start=2):  # row 1 is the header
        component = (row["component"] or "").strip()
        if not component:
            raise RobustnessFormatError(f"row {number}: component is empty")
        factor = _parse_float(row["factor"], row=number, column="factor")
        naive_net = _parse_float(
            row["best_gross_net_sharpe"], row=number, column="best_gross_net_sharpe"
        )
        gross_hash = (row["best_gross_hash"] or "").strip()
        net_hash = (row["best_net_hash"] or "").strip()
        if not gross_hash or not net_hash:
            raise RobustnessFormatError(
                f"row {number}: best_gross_hash/best_net_hash must both name a variant; "
                "without them the two-different-variants half of the finding is "
                "unverifiable."
            )

        # Recomputed from the raw columns, not read from a summary column. A writer
        # that mislabels its own verdict must not be able to launder it through here.
        holds = naive_net < 0.0 and gross_hash != net_hash

        # If the writer also stated its own verdict, the two must agree. This is a
        # cross-check between two independent derivations of the same claim, not a
        # formality: a disagreement means either the sweep summarised itself wrongly
        # or this reader has drifted from the format, and both are reasons to stop
        # rather than to pick a winner.
        stated = row.get("inversion_holds")
        if stated is not None and str(stated).strip():
            if _parse_bool(str(stated), row=number, column="inversion_holds") != holds:
                raise RobustnessFormatError(
                    f"row {number} ({component}@{factor:.2f}): the file states "
                    f"inversion_holds={stated!r}, but recomputing it from "
                    f"best_gross_net_sharpe={naive_net:+.6f} and the two variant "
                    f"hashes gives {holds}. The sweep and this reader disagree about "
                    "what the sweep found."
                )

        if not holds:
            broken.append(f"{component}@{factor:.2f}")

        # Optional, and all-or-nothing: a rank range built from only the cases that
        # happened to report one would be a range over an unknown subset.
        raw_rank = row.get("gross_pick_rank_by_net")
        raw_given_up = row.get("net_sharpe_given_up_by_selecting_on_gross")
        if raw_rank is not None and str(raw_rank).strip():
            ranks.append(int(_parse_float(str(raw_rank), row=number, column="gross_pick_rank_by_net")))
        if raw_given_up is not None and str(raw_given_up).strip():
            given_up.append(
                _parse_float(
                    str(raw_given_up),
                    row=number,
                    column="net_sharpe_given_up_by_selecting_on_gross",
                )
            )

        if component not in components:
            components.append(component)
        if factor not in factors:
            factors.append(factor)
        naive_nets.append(naive_net)

    return RateRobustness(
        source=str(path.as_posix()),
        n_cases=len(rows),
        components=tuple(components),
        factors=tuple(sorted(factors)),
        inversion_holds_everywhere=not broken,
        broken_cases=tuple(broken),
        # max, not min: the case nearest to breaking the claim is the LEAST
        # negative one, not the most.
        naive_pick_net_sharpe_nearest_zero=max(naive_nets),
        naive_pick_net_sharpe_most_negative=min(naive_nets),
        gross_pick_rank_range=(
            (min(ranks), max(ranks)) if len(ranks) == len(rows) else None
        ),
        net_sharpe_given_up_range=(
            (min(given_up), max(given_up)) if len(given_up) == len(rows) else None
        ),
        n_variants=n_variants,
    )
