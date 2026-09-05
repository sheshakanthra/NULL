"""OHLCV loader, validators, and the bhavcopy format resolver.

The validators are exercised against constructed parquets, so they are tested
whether or not the real cache exists. Tests that need the committed data are
skipped until it lands and unskip themselves automatically.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from null.data.ohlcv import (
    ADV_WINDOW,
    DEFAULT_CACHE,
    MAX_SINGLE_DAY_MOVE,
    OHLCVCacheMissing,
    load_bars,
    validate_ohlcv,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fetch_bhavcopy as FB  # noqa: E402

SESSIONS_PER_YEAR = 248


def _write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _clean(
    tmp_path: Path, *, symbols=("AAA", "BBB"), years=(2021, 2022), seed=5
) -> Path:
    """A well-formed cache: sane moves, non-zero volume, plausible session counts."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        # Weekdays only, trimmed to a realistic session count.
        days = [
            d
            for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
        ][:SESSIONS_PER_YEAR]
        for symbol in symbols:
            price = 1000.0
            for day in days:
                price *= 1.0 + rng.normal(0.0004, 0.012)
                volume = float(rng.integers(500_000, 5_000_000))
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
                        "format_handler": "legacy_cm_bhavcopy",
                    }
                )
    return _write(tmp_path / "ohlcv.parquet", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# the format resolver -- the July 2024 change is the design constraint
# ---------------------------------------------------------------------------


def test_a_date_resolves_to_a_format_handler() -> None:
    assert FB.candidates(date(2015, 6, 1))[0].name == "legacy_cm_bhavcopy"
    assert FB.candidates(date(2025, 6, 1))[0].name == "udiff_2024"


def test_every_handler_is_a_fallback_for_every_date() -> None:
    """The cutover boundary is a guess, so a wrong guess costs a request, not data."""
    for day in (date(2015, 6, 1), date(2024, 7, 6), date(2025, 6, 1)):
        assert len(FB.candidates(day)) == len(FB.FORMATS)


def test_the_two_formats_use_different_urls_and_columns() -> None:
    legacy, udiff = FB.FORMATS
    assert legacy.url(date(2024, 3, 15)).endswith("cm15MAR2024bhav.csv.zip")
    assert udiff.url(date(2025, 3, 17)).endswith("20250317_F_0000.csv.zip")
    assert legacy.columns != udiff.columns
    assert legacy.series_column != udiff.series_column


def test_adding_a_third_format_needs_no_fetcher_change() -> None:
    """The point of the resolver. A new shape is a new entry, not a rewrite."""
    third = FB.BhavcopyFormat(
        name="hypothetical_2027",
        valid_from=date(2027, 1, 1),
        valid_to=None,
        url_template="https://example.invalid/%Y%m%d.zip",
        columns={"Sym": "symbol"},
        series_column="Series",
        date_column="Dt",
    )
    extended = (*FB.FORMATS, third)
    covering = [f for f in extended if f.covers(date(2027, 6, 1))]
    assert third in covering


# ---------------------------------------------------------------------------
# the loader refuses to fall back
# ---------------------------------------------------------------------------


def test_missing_cache_raises_and_names_no_substitute() -> None:
    with pytest.raises(OHLCVCacheMissing) as exc:
        load_bars(Path("nope/missing.parquet"))
    assert "will not substitute" in str(exc.value)
    assert "fetch_bhavcopy.py" in str(exc.value)


def test_loader_imports_nothing_that_can_reach_the_network() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "null" / "data" / "ohlcv.py"
    ).read_text(encoding="utf-8")
    for banned in ("import urllib", "import requests", "import httpx", "urlopen"):
        assert banned not in source


# ---------------------------------------------------------------------------
# adv_20 comes from value traded, not one day's price x volume
# ---------------------------------------------------------------------------


def test_adv_20_is_a_rolling_mean_of_value_traded(tmp_path) -> None:
    """The capacity gate reads this, so it has to be the real thing."""
    days = pd.date_range("2022-01-03", periods=30, freq="B")
    values = np.arange(1, 31, dtype=float) * 1e7
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": values / 100.0,
            "value_traded": values,
            "format_handler": "legacy_cm_bhavcopy",
        }
    )
    bars = load_bars(_write(tmp_path / "x.parquet", frame))
    last = bars[-1]
    expected = float(values[-ADV_WINDOW:].mean())
    assert last.adv_20 == pytest.approx(expected, rel=1e-9)
    # Not the single day's notional, which would be 3e8 here.
    assert last.adv_20 != pytest.approx(float(values[-1]), rel=1e-6)


def test_early_bars_use_what_exists_rather_than_dropping_out(tmp_path) -> None:
    bars = load_bars(_clean(tmp_path))
    assert all(b.adv_20 is not None and b.adv_20 > 0 for b in bars[:5])


# ---------------------------------------------------------------------------
# the four validators, each with a failing case
# ---------------------------------------------------------------------------


def test_a_clean_series_validates(tmp_path) -> None:
    result = validate_ohlcv(_clean(tmp_path))
    assert result.is_valid, result.rationale


def _unadjusted_split(frame: pd.DataFrame, symbol: str, position: int, factor=1.8):
    """What an unadjusted corporate action actually looks like.

    Every close from the event onward is scaled, producing exactly ONE anomalous
    transition. Bumping a single close would produce two -- a jump up and a jump
    straight back down -- which is not a split, it is a bad print.
    """
    mask = frame["symbol"] == symbol
    rows = frame.index[mask]
    event = rows[position]
    frame.loc[rows[position:], "close"] = frame.loc[rows[position:], "close"] * factor
    return str(pd.Timestamp(frame.loc[event, "date"]).date())


def test_an_unexplained_jump_is_flagged_not_accepted(tmp_path) -> None:
    frame = pd.read_parquet(_clean(tmp_path))
    _unadjusted_split(frame, "AAA", 100)
    path = _write(tmp_path / "jump.parquet", frame)
    result = validate_ohlcv(path)
    assert not result.is_valid
    assert result.suspicious_moves
    assert "corporate action" in result.rationale


def test_a_jump_matched_to_a_known_action_is_accepted(tmp_path) -> None:
    """The other half: a real, known action must not be reported as a defect."""
    frame = pd.read_parquet(_clean(tmp_path))
    day = _unadjusted_split(frame, "AAA", 100)
    path = _write(tmp_path / "known.parquet", frame)
    result = validate_ohlcv(path, corporate_action_dates={"AAA": {day}})
    assert not result.suspicious_moves, result.suspicious_moves


def test_zero_and_negative_volume_are_both_caught(tmp_path) -> None:
    frame = pd.read_parquet(_clean(tmp_path))
    frame.loc[frame.index[10], "volume"] = 0.0
    frame.loc[frame.index[11], "volume"] = -5.0
    result = validate_ohlcv(_write(tmp_path / "vol.parquet", frame))
    assert not result.is_valid
    assert result.zero_volume_days and result.negative_volume_days


def test_an_impossible_session_count_is_caught(tmp_path) -> None:
    """A year with 260 sessions means a non-trading date was picked up."""
    rng = np.random.default_rng(1)
    days = pd.date_range("2021-01-01", "2021-12-31", freq="D")[:260]
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + rng.normal(0, 0.5, len(days)),
            "volume": 1e6,
            "value_traded": 1e8,
            "format_handler": "legacy_cm_bhavcopy",
        }
    )
    result = validate_ohlcv(_write(tmp_path / "cal.parquet", frame))
    assert not result.is_valid
    assert result.years_outside_expected_day_count


def test_calendar_disagreement_with_the_benchmark_is_caught(tmp_path) -> None:
    path = _clean(tmp_path)
    ours = sorted({str(d.date()) for d in pd.read_parquet(path)["date"]})
    shifted = tuple(
        str((pd.Timestamp(d) + timedelta(days=1)).date()) for d in ours[:50]
    )
    result = validate_ohlcv(path, benchmark_dates=shifted)
    assert not result.is_valid
    assert result.calendar_mismatches


# ---------------------------------------------------------------------------
# the committed cache, once it exists
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DEFAULT_CACHE.exists(), reason="OHLCV parquet not committed yet")
def test_committed_cache_loads_and_validates() -> None:
    result = validate_ohlcv()
    assert result.is_valid, result.rationale
    bars = load_bars()
    assert len({b.symbol for b in bars}) >= 40
    assert all(b.adv_20 is not None for b in bars)
