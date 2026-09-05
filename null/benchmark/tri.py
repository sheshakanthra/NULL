"""Offline NIFTY 50 TRI loader. Reads the committed parquet. Never fetches.

The network stage is ``scripts/fetch_tri.py``, deliberately outside this package:
`null audit` must run with the network off (CLAUDE.md invariant 2), and the source
grep forbids network imports anywhere under ``null/``. This module only ever reads
a local file.

**There is no price-index fallback.** Not deprecated — absent. If the TRI cache is
missing, :func:`load_nifty50_tri` raises. It does not quietly substitute the price
index, because that would hand every strategy roughly 1.35%/yr of free alpha, which
is precisely the bias the benchmark harness exists to remove. A missing benchmark
is a stop, not a downgrade.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import numpy as np

from null.contracts import IST, NonEmptyStr, NullFloat, NullModel, Series

__all__ = [
    "PRI_TRI_ANNUALISED_GAP",
    "TRICacheMissing",
    "TRIValidation",
    "DEFAULT_CACHE",
    "load_nifty50_tri",
    "validate_tri_against_pri",
]

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "nifty50_tri.parquet"
)

#: NSE's own reported 20-year annualised figures to Feb 2026: 11.09% for the price
#: index against 12.44% for total return. The gap is dividends, and benchmarking
#: against the price index hands a strategy every basis point of it.
PRI_20Y_ANNUALISED = 0.1109
TRI_20Y_ANNUALISED = 0.1244
PRI_TRI_ANNUALISED_GAP = TRI_20Y_ANNUALISED - PRI_20Y_ANNUALISED  # 0.0135


class TRICacheMissing(FileNotFoundError):
    """The TRI cache is absent. This is a stop, not a reason to use the price index."""


class TRIValidation(NullModel):
    """Whether a candidate series actually behaves like a total-return index."""

    is_valid: bool
    tri_exceeds_pri_everywhere: bool
    windows_checked: int
    windows_violating: int
    tri_annualised: NullFloat
    pri_annualised: NullFloat
    observed_gap: NullFloat
    rationale: NonEmptyStr


def load_nifty50_tri(path: Path = DEFAULT_CACHE) -> Series:
    """Read the committed TRI parquet into a contract ``Series``.

    Raises rather than falling back. See the module docstring.
    """
    if not path.exists():
        raise TRICacheMissing(
            f"No TRI cache at {path}. Run `python scripts/fetch_tri.py --refresh` on a "
            "machine that can reach niftyindices.com and commit the result. NULL will "
            "NOT substitute the NIFTY price index: that omits roughly "
            f"{PRI_TRI_ANNUALISED_GAP:.2%}/yr of dividends and would hand every "
            "strategy that much free alpha, which is the exact bias the benchmark "
            "harness exists to remove."
        )

    import pandas as pd

    frame = pd.read_parquet(path)
    missing = {"date", "tri"} - set(frame.columns)
    if missing:
        raise ValueError(f"TRI cache is missing columns {sorted(missing)}")

    frame = frame.sort_values("date").reset_index(drop=True)
    stamps = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(IST)
    return Series(
        ts=tuple(t.to_pydatetime() for t in stamps),
        values=tuple(float(v) for v in frame["tri"]),
    )


def _annualised(values: np.ndarray, periods_per_year: int = 252) -> float:
    if values.size < 2 or values[0] <= 0.0 or values[-1] <= 0.0:
        return 0.0
    years = values.size / periods_per_year
    return float((values[-1] / values[0]) ** (1.0 / years) - 1.0)


def validate_tri_against_pri(
    tri: Series,
    pri: Series,
    *,
    window_years: int = 3,
    periods_per_year: int = 252,
    expected_gap: float = PRI_TRI_ANNUALISED_GAP,
    gap_tolerance: float = 0.004,
) -> TRIValidation:
    """Check a candidate TRI actually is one, before anything trusts it.

    Two independent checks, because either alone is passable by the wrong series:

    1. **Monotonic dominance.** Over any multi-year window, total return must exceed
       price return. Dividends are never negative, so a series that fails this is
       not a total-return index whatever it is labelled.
    2. **Magnitude.** The annualised gap should land near the published
       ``expected_gap``. A series that dominates by 0.1%/yr or by 5%/yr is not
       NIFTY 50 TRI even if it dominates monotonically.
    """
    t = tri.to_numpy()
    p = pri.to_numpy()
    n = int(min(t.size, p.size))
    if n < periods_per_year * window_years:
        raise ValueError(
            f"need at least {periods_per_year * window_years} aligned observations "
            f"to check {window_years}-year windows, got {n}"
        )
    t, p = t[:n], p[:n]

    step = periods_per_year * window_years
    windows = 0
    violating = 0
    for start in range(0, n - step, periods_per_year):
        end = start + step
        tri_growth = t[end] / t[start]
        pri_growth = p[end] / p[start]
        windows += 1
        if tri_growth <= pri_growth:
            violating += 1

    tri_ann = _annualised(t, periods_per_year)
    pri_ann = _annualised(p, periods_per_year)
    gap = tri_ann - pri_ann

    dominates = violating == 0
    gap_ok = abs(gap - expected_gap) <= gap_tolerance
    is_valid = dominates and gap_ok

    if not dominates:
        verdict = (
            f"{violating} of {windows} {window_years}-year windows show total return "
            "at or below price return. Dividends are never negative, so this series is "
            "not a total-return index."
        )
    elif not gap_ok:
        verdict = (
            f"Total return dominates price return everywhere, but the annualised gap "
            f"of {gap:.2%} is not close to the published {expected_gap:.2%}. This is "
            "some other series, or the wrong index."
        )
    else:
        verdict = (
            "Total return dominates price return in every window checked, and the "
            f"annualised gap of {gap:.2%} matches the published {expected_gap:.2%}."
        )

    return TRIValidation(
        is_valid=is_valid,
        tri_exceeds_pri_everywhere=dominates,
        windows_checked=windows,
        windows_violating=violating,
        tri_annualised=tri_ann,
        pri_annualised=pri_ann,
        observed_gap=gap,
        rationale=verdict,
    )
