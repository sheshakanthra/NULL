"""TRI loader and validator. BUILD.md §4 rule 1.

Note what these tests do and do not establish. They exercise the LOADER and the
VALIDATOR against constructed series. They do not validate real NSE data, because
the endpoint refuses this environment and no real series has been fetched --
see docs/data_sources.md. When the parquet is committed, the validator here is what
gets pointed at it.

Every constructed series below is explicitly synthetic. None of it is, or is
presented as, NIFTY data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from null.benchmark.tri import (
    PRI_TRI_ANNUALISED_GAP,
    DEFAULT_CACHE,
    TRICacheMissing,
    load_nifty50_tri,
    validate_tri_against_pri,
)
from null.contracts import Series

IST = timezone(timedelta(hours=5, minutes=30))
SEED = 20260905


def _levels(daily: np.ndarray, start: float = 1000.0) -> Series:
    base = datetime(2005, 1, 1, 15, 30, tzinfo=IST)
    values = start * np.cumprod(1.0 + daily)
    return Series(
        ts=tuple(base + timedelta(days=i) for i in range(values.size)),
        values=tuple(float(v) for v in values),
    )


def _pri_and_tri(n=252 * 20, dividend_yield=PRI_TRI_ANNUALISED_GAP, seed=SEED):
    """A synthetic price index and its total-return counterpart."""
    rng = np.random.default_rng(seed)
    price = rng.normal(0.0004, 0.0102, n)
    total = price + dividend_yield / 252.0
    return _levels(price), _levels(total)


# ---------------------------------------------------------------------------
# the loader refuses to fall back
# ---------------------------------------------------------------------------


def test_missing_cache_raises_and_never_substitutes_the_price_index() -> None:
    """The whole point of the module. A missing benchmark is a stop."""
    with pytest.raises(TRICacheMissing) as exc:
        load_nifty50_tri(Path("does/not/exist.parquet"))
    message = str(exc.value)
    assert "will NOT substitute" in message
    assert "1.35%" in message or "0.0135" in message
    assert "fetch_tri.py" in message


def test_there_is_no_price_index_fallback_anywhere_in_the_module() -> None:
    """Unreachable, not deprecated. Grep the source rather than trust the API."""
    source = (
        Path(__file__).resolve().parents[2] / "null" / "benchmark" / "tri.py"
    ).read_text(encoding="utf-8")
    for banned in ("fallback_to_pri", "use_price_index", "allow_price_index"):
        assert banned not in source


def test_the_loader_imports_nothing_that_can_reach_the_network() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "null" / "benchmark" / "tri.py"
    ).read_text(encoding="utf-8")
    for banned in ("import urllib", "import requests", "import httpx", "urlopen"):
        assert banned not in source


def test_the_parquet_round_trip_works_without_any_real_data(tmp_path) -> None:
    """CI coverage for the parquet path that does NOT depend on a fetched cache.

    This test exists because of a real failure. Every TRI test that touched parquet
    was skipped for missing data, so the commit that introduced the TRI loader went
    green while pyarrow -- which pandas needs to read or write parquet at all -- was
    undeclared in pyproject. The gap only surfaced two commits later when the OHLCV
    loader added tests that actually write a file.

    A synthetic frame round-tripping through to_parquet/read_parquet needs no
    external data, so this path is covered whether or not any cache exists.
    """
    import pandas as pd

    base = datetime(2020, 1, 1, 15, 30, tzinfo=IST)
    frame = pd.DataFrame(
        {
            "date": [base + timedelta(days=i) for i in range(10)],
            "tri": [20_000.0 * (1.0 + 0.001 * i) for i in range(10)],
        }
    )
    path = tmp_path / "tiny_tri.parquet"
    frame.to_parquet(path, index=False)

    series = load_nifty50_tri(path)
    assert len(series) == 10
    assert series.values[0] == pytest.approx(20_000.0)
    # The IST / ISO-8601 contract depends on timestamps surviving the round trip.
    assert series.ts[0].utcoffset() == timedelta(hours=5, minutes=30)
    assert series.ts[0].hour == 15 and series.ts[0].minute == 30
    assert list(series.ts) == sorted(series.ts)


def test_a_parquet_engine_is_actually_installed() -> None:
    """Fails loudly rather than as an ImportError inside an unrelated test."""
    import pyarrow  # noqa: F401

    import pandas as pd

    assert pd.io.parquet.get_engine("auto") is not None


@pytest.mark.skipif(
    not DEFAULT_CACHE.exists(),
    reason="TRI parquet not committed yet -- the endpoint refuses this environment",
)
def test_committed_cache_loads_and_validates() -> None:
    """Runs only once real data is committed. Skipped until then, deliberately."""
    series = load_nifty50_tri()
    assert len(series) > 5_000
    assert all(v > 0 for v in series.values)


# ---------------------------------------------------------------------------
# the validator
# ---------------------------------------------------------------------------


def test_a_correct_tri_validates() -> None:
    pri, tri = _pri_and_tri()
    result = validate_tri_against_pri(tri, pri)
    assert result.is_valid
    assert result.windows_violating == 0
    assert result.observed_gap == pytest.approx(PRI_TRI_ANNUALISED_GAP, abs=0.002)


def test_a_price_index_passed_as_tri_is_rejected() -> None:
    """The failure this exists to catch: someone hands us PRI and calls it TRI."""
    pri, _ = _pri_and_tri()
    result = validate_tri_against_pri(pri, pri)
    assert not result.is_valid
    assert "not a total-return index" in result.rationale


def test_a_series_that_dominates_by_the_wrong_margin_is_rejected() -> None:
    """The subtle case. Monotonic dominance alone is not enough.

    A series can exceed the price index in every window and still be the wrong
    index, or the right index with the wrong dividend treatment. Checking only
    dominance would wave that through.
    """
    pri, too_much = _pri_and_tri(dividend_yield=0.06)
    result = validate_tri_against_pri(too_much, pri)
    assert result.tri_exceeds_pri_everywhere is True
    assert not result.is_valid
    assert "not close to the published" in result.rationale

    pri2, too_little = _pri_and_tri(dividend_yield=0.001, seed=SEED + 1)
    assert not validate_tri_against_pri(too_little, pri2).is_valid


def test_an_occasional_violation_is_caught() -> None:
    """Dividends are never negative, so even one bad window disqualifies."""
    rng = np.random.default_rng(SEED)
    n = 252 * 20
    price = rng.normal(0.0004, 0.0102, n)
    total = price + PRI_TRI_ANNUALISED_GAP / 252.0
    total[252 * 5 : 252 * 7] -= 0.0015  # a stretch where "total" falls behind
    result = validate_tri_against_pri(_levels(total), _levels(price))
    assert result.windows_violating > 0
    assert not result.is_valid


def test_too_short_a_series_is_rejected_rather_than_guessed_at() -> None:
    pri, tri = _pri_and_tri(n=252)
    with pytest.raises(ValueError, match="aligned observations"):
        validate_tri_against_pri(tri, pri, window_years=3)


def test_published_gap_constant_matches_nse_reported_figures() -> None:
    assert PRI_TRI_ANNUALISED_GAP == pytest.approx(0.1244 - 0.1109)
