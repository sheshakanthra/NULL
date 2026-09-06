"""Corporate actions from the exchange. Ground truth, not inference.

Replaces the ratio-inference detector that preceded it. That approach had a
structural flaw rather than a calibration one: its two conditions were not
independent. Value traded staying roughly flat across a boundary is true of
**99.02% of ordinary trading days**, measured on this very dataset, so it could not
corroborate anything. The price ratio was doing all the work alone, and no
tolerance separates a 1:3 bonus (factor 1.333) from a 28% crash (implied 1.387).

With the real calendar there is nothing to separate. A move on a date matching an
announced action is adjusted by the announced ratio. A move matching nothing is a
market event, kept unadjusted. The old heuristic survives only as a cross-check.

What is adjustable, and what deliberately is not:

    SPLIT          adjusted -- face value divided, share count multiplied
    BONUS          adjusted -- free shares issued against existing holdings
    DEMERGER       NEVER adjusted. Value genuinely leaves the parent company, so
                   back-adjusting would inflate its history. Documented instead.
    DIVIDEND       not adjusted. Small relative to splits, and the benchmark is a
                   total-return index, so the comparison is already tilted against
                   the strategy. An auditor tilting against itself is the right
                   direction.
    RIGHTS         not adjusted. Dilution depends on subscription take-up, which
                   the calendar does not record.
    BUYBACK, AGM   not price-affecting in a way that needs adjustment.

Anything in an adjustable category whose ratio will not parse is FLAGGED, never
guessed at.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from null.contracts import NonEmptyStr, NonNegativeFloat, NullModel

__all__ = [
    "ActionKind",
    "CorporateAction",
    "CorporateActionsMissing",
    "DEFAULT_CACHE",
    "adjustment_factors",
    "load_corporate_actions",
    "parse_subject",
]

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "corporate_actions.parquet"
)

ActionKind = Literal[
    "split", "bonus", "bonus+split", "demerger", "dividend", "rights", "other",
    "unparseable",
]

#: Only these change the share count against existing holdings.
ADJUSTABLE: frozenset[str] = frozenset({"split", "bonus", "bonus+split"})


class CorporateActionsMissing(FileNotFoundError):
    """The calendar is absent. Adjustment falls back to nothing, ever."""


class CorporateAction(NullModel):
    symbol: NonEmptyStr
    ex_date: NonEmptyStr
    kind: NonEmptyStr
    factor: NonNegativeFloat
    """previous_close / close implied by the action. 1.0 means no price effect."""
    subject: NonEmptyStr

    @property
    def is_adjustable(self) -> bool:
        return self.kind in ADJUSTABLE and self.factor > 0.0 and self.factor != 1.0


# NSE writes these free-form and the shapes vary a lot across 15 years:
#   "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"
#   "Fv Splt Frm Rs 10 To Re 1"
#   "Bonus 1:1 / Face Value Split From Rs.10/- To Re.1/-"
#   "Annual General Meeting/ Dividend Rs 11/- Per Share/ Bonus 1:1"
#
# So the parser scans for EVERY component and multiplies, rather than returning on
# the first match. An earlier version returned on the first, which meant a combined
# "Bonus 1:1 / Face Value Split 10:1" applied 10 where 20 was due and left TITAN
# half-adjusted at -46.7%.
_SPLIT = re.compile(
    r"(?:face\s*value\s*spl|f\.?\s?v\.?\s*spl|sub-?division|stock\s*spl)\w*"
    r"[^A-Za-z0-9]*(?:\([^)]*\))?[^A-Za-z0-9]*"
    r"(?:from|frm)?\s*rs\.?\s*([\d.]+)"
    r"[^A-Za-z0-9]*(?:per\s+share)?[^A-Za-z0-9]*"
    r"to\s+(?:re|rs)\.?\s*([\d.]+)",
    re.I,
)
_SPLIT_ANY = re.compile(
    r"face\s*value\s*spl|f\.?\s?v\.?\s*spl|sub-?division|stock\s*spl", re.I
)
_BONUS = re.compile(r"bonus\s*[-:]?\s*(\d+)\s*:\s*(\d+)", re.I)
_BONUS_ANY = re.compile(r"bonus", re.I)
_DEMERGER = re.compile(r"demerger|de-merger|scheme\s+of\s+arrangement", re.I)
_DIVIDEND = re.compile(r"dividend", re.I)
_RIGHTS = re.compile(r"rights", re.I)
#: Bonus DEBENTURES are not equity bonuses and carry no share-count effect.
_BONUS_DEBENTURE = re.compile(r"bonus\s+debenture", re.I)


def parse_subject(subject: str) -> tuple[ActionKind, float]:
    """Classify one subject line and give its combined price factor.

    Every recognised component is multiplied together, because NSE routinely packs
    a bonus and a split into one line. Returns ``("unparseable", 0.0)`` when an
    action clearly IS a split or bonus but its ratio cannot be read, so it is
    flagged for a human rather than silently skipped.
    """
    text = " ".join(str(subject).split())

    # Demerger first and unconditionally: a scheme of arrangement can name a ratio,
    # and it must never be treated as one. Value leaves the company.
    if _DEMERGER.search(text):
        return "demerger", 1.0

    factor = 1.0
    found: list[str] = []

    for match in _SPLIT.finditer(text):
        old, new = float(match.group(1)), float(match.group(2))
        if new > 0.0 and old > new:
            factor *= old / new
            found.append("split")
    if not found and _SPLIT_ANY.search(text):
        return "unparseable", 0.0

    if not _BONUS_DEBENTURE.search(text):
        for match in _BONUS.finditer(text):
            new_shares, held = float(match.group(1)), float(match.group(2))
            if held > 0.0:
                # a:b means a free shares per b held; a holder of b ends with a+b.
                factor *= (new_shares + held) / held
                found.append("bonus")
        if "bonus" not in found and _BONUS_ANY.search(text):
            return "unparseable", 0.0

    if found:
        kind = "+".join(sorted(set(found)))
        return kind, factor  # type: ignore[return-value]

    if _DIVIDEND.search(text):
        return "dividend", 1.0
    if _RIGHTS.search(text):
        return "rights", 1.0
    return "other", 1.0


def load_corporate_actions(
    path: Path = DEFAULT_CACHE, *, symbols: tuple[str, ...] | None = None
) -> tuple[CorporateAction, ...]:
    """Read the committed calendar. Raises rather than falling back to inference."""
    if not path.exists():
        raise CorporateActionsMissing(
            f"No corporate action calendar at {path}. Run "
            "`python scripts/fetch_corporate_actions.py --refresh --symbols ...` and "
            "commit the result. NULL will not fall back to inferring splits from "
            "price ratios: that approach cannot separate a 1:3 bonus from a 28% "
            "crash, and guessing wrong either fabricates a crash or erases one."
        )
    import pandas as pd

    frame = pd.read_parquet(path)
    if symbols is not None:
        frame = frame[frame["symbol"].isin(set(symbols))]

    # Extract to plain Python at the boundary rather than iterating the frame:
    # itertuples leaves every field loosely typed, and these values go straight into
    # frozen contracts.
    symbols_out = [str(s).strip() for s in frame["symbol"]]
    ex_dates = [str(d.date()) for d in pd.to_datetime(frame["ex_date"])]
    subjects = [" ".join(str(s).split()) or "(blank)" for s in frame["subject"]]

    actions: list[CorporateAction] = []
    for symbol, ex_date, subject in zip(symbols_out, ex_dates, subjects):
        kind, factor = parse_subject(subject)
        actions.append(
            CorporateAction(
                symbol=symbol,
                ex_date=ex_date,
                kind=kind,
                factor=factor,
                subject=subject,
            )
        )
    return tuple(sorted(actions, key=lambda a: (a.symbol, a.ex_date)))


def adjustment_factors(
    actions: tuple[CorporateAction, ...],
) -> dict[tuple[str, str], float]:
    """Combined price factor per (symbol, ex_date).

    Actions on the same ex-date MULTIPLY. BAJFINANCE on 2025-06-16 carried a 1:2
    face-value split and a 4:1 bonus together, giving 2 x 5 = 10 -- which is why a
    single-ratio guess read it as a "1:10 split" and could never have named it.
    """
    combined: dict[tuple[str, str], float] = {}
    for action in actions:
        if not action.is_adjustable:
            continue
        key = (action.symbol, action.ex_date)
        combined[key] = combined.get(key, 1.0) * action.factor
    return combined
