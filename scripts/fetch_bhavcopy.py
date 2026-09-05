"""Fetch NSE daily bhavcopy. NETWORK STAGE — deliberately outside null/.

    python scripts/fetch_bhavcopy.py --refresh --start 2011-01-01 --end 2026-01-01

**The July 2024 format change is the design constraint, not a footnote.** NSE moved
the daily bhavcopy to a new URL and a new column layout mid-history, which is why
downstream libraries ended up with two separate functions. Here a date resolves to a
*format handler*, so a third format is a new entry in ``FORMATS`` rather than a
rewrite of the fetcher. Which handler produced each row is recorded, both as a
column in the parquet and as a summary in the provenance sidecar.

The exact cutover date is not treated as known. The resolver returns an ordered list
of candidate handlers per date and the fetcher tries them in order, recording which
one actually answered. That way a wrong guess about the boundary costs one extra
request rather than a silently empty history.

Politeness, same as the TRI fetcher and not configurable upward: one request at a
time, a sleep between them, two retries with a long backoff, never parallel.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "reference" / "nifty50_ohlcv.parquet"
SIDECAR = REPO / "data" / "reference" / "nifty50_ohlcv.provenance.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 20.0

#: Scope, deliberately tight. Not the full universe.
UNIVERSE_NOTE = "NIFTY 50 constituents as supplied by --symbols; daily; 15 years"


@dataclass(frozen=True)
class BhavcopyFormat:
    """One historical shape of the file. Add a handler; do not edit the fetcher."""

    name: str
    valid_from: date
    valid_to: date | None
    url_template: str
    columns: dict[str, str]
    series_column: str
    date_column: str

    def url(self, day: date) -> str:
        return day.strftime(self.url_template).replace(
            "{MON}", day.strftime("%b").upper()
        )

    def covers(self, day: date) -> bool:
        if day < self.valid_from:
            return False
        return self.valid_to is None or day <= self.valid_to


#: Ordered oldest-first. A third format goes here.
FORMATS: tuple[BhavcopyFormat, ...] = (
    BhavcopyFormat(
        name="legacy_cm_bhavcopy",
        valid_from=date(1996, 1, 1),
        valid_to=date(2024, 7, 5),
        url_template=(
            "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
            "%Y/{MON}/cm%d{MON}%Ybhav.csv.zip"
        ),
        columns={
            "SYMBOL": "symbol",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "TOTTRDQTY": "volume",
            "TOTTRDVAL": "value_traded",
            "TIMESTAMP": "date",
        },
        series_column="SERIES",
        date_column="TIMESTAMP",
    ),
    BhavcopyFormat(
        name="udiff_2024",
        valid_from=date(2024, 7, 8),
        valid_to=None,
        url_template=(
            "https://nsearchives.nseindia.com/content/cm/"
            "BhavCopy_NSE_CM_0_0_0_%Y%m%d_F_0000.csv.zip"
        ),
        columns={
            "TckrSymb": "symbol",
            "OpnPric": "open",
            "HghPric": "high",
            "LwPric": "low",
            "ClsPric": "close",
            "TtlTradgVol": "volume",
            "TtlTrfVal": "value_traded",
            "TradDt": "date",
        },
        series_column="SctySrs",
        date_column="TradDt",
    ),
)


def candidates(day: date) -> tuple[BhavcopyFormat, ...]:
    """Handlers to try for a date, best guess first.

    The cutover boundary is a guess, so every handler is a fallback for every date.
    A wrong guess costs one extra request, not a silently empty history.
    """
    primary = [f for f in FORMATS if f.covers(day)]
    rest = [f for f in FORMATS if f not in primary]
    return tuple(primary + rest)


def _fetch_zip(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,application/zip,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read()
    if "html" in content_type.lower():
        raise RuntimeError(
            f"got {content_type!r} and {len(raw)} bytes instead of a zip. An HTML body "
            "here is the site answering with a page rather than the archive, which "
            "means the URL shape has moved or the request never reached the file. "
            "Re-capture the live request; do not cycle headers."
        )
    return raw


def fetch_day(day: date, symbols: set[str]) -> tuple[list[dict[str, Any]], str]:
    """One trading day, filtered to the universe. Returns rows and the handler used."""
    import pandas as pd

    last_error: Exception | None = None
    for fmt in candidates(day):
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                time.sleep(RETRY_BACKOFF_SECONDS)
            try:
                blob = _fetch_zip(fmt.url(day))
                with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                    name = archive.namelist()[0]
                    frame = pd.read_csv(io.BytesIO(archive.read(name)))
                frame.columns = [c.strip() for c in frame.columns]
                frame = frame[frame[fmt.series_column].astype(str).str.strip() == "EQ"]
                frame = frame.rename(columns=fmt.columns)
                frame["symbol"] = frame["symbol"].astype(str).str.strip()
                frame = frame[frame["symbol"].isin(symbols)]
                frame["format_handler"] = fmt.name
                keep = [
                    "date", "symbol", "open", "high", "low", "close",
                    "volume", "value_traded", "format_handler",
                ]
                return frame[keep].to_dict("records"), fmt.name
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                last_error = exc
                break  # wrong format or holiday: try the next handler, not a retry
            except (RuntimeError, KeyError, ValueError, zipfile.BadZipFile) as exc:
                last_error = exc
    raise RuntimeError(f"no handler produced data for {day}: {last_error}")


def fetch_range(
    start: date, end: date, symbols: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    handler_counts: dict[str, int] = {}
    day = start
    while day <= end:
        if day.weekday() < 5:  # skip weekends without spending a request
            try:
                day_rows, handler = fetch_day(day, symbols)
                rows.extend(day_rows)
                handler_counts[handler] = handler_counts.get(handler, 0) + len(day_rows)
            except RuntimeError as exc:
                print(f"  {day}: skipped ({exc})", flush=True)
            time.sleep(REQUEST_DELAY_SECONDS)
        day += timedelta(days=1)
    return rows, handler_counts


def _chunk_path(year: int) -> Path:
    return CACHE.parent / "chunks" / f"nifty50_ohlcv_{year}.parquet"


def write_chunk(rows: list[dict[str, Any]], year: int) -> Path:
    """One year, checkpointed. A failure costs a year, not the history."""
    import pandas as pd

    path = _chunk_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def merge_chunks(*, fetched_on: datetime) -> None:
    """Combine year chunks into the committed cache plus its sidecar."""
    import pandas as pd

    paths = sorted((CACHE.parent / "chunks").glob("nifty50_ohlcv_*.parquet"))
    if not paths:
        raise RuntimeError("no year chunks to merge")
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    counts = frame["format_handler"].value_counts().to_dict()
    write_cache(frame.to_dict("records"), {k: int(v) for k, v in counts.items()},
                fetched_on=fetched_on)


def write_cache(
    rows: list[dict[str, Any]], handler_counts: dict[str, int], *, fetched_on: datetime
) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume", "value_traded"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values(["date", "symbol"])
    frame = frame.reset_index(drop=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE, index=False)

    SIDECAR.write_text(
        json.dumps(
            {
                "scope": UNIVERSE_NOTE,
                "fetched_on": fetched_on.date().isoformat(),
                "rows": int(len(frame)),
                "symbols": int(frame["symbol"].nunique()),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
                "rows_per_format_handler": handler_counts,
                "format_handlers": [
                    {
                        "name": f.name,
                        "valid_from": f.valid_from.isoformat(),
                        "valid_to": f.valid_to.isoformat() if f.valid_to else None,
                        "url_template": f.url_template,
                        "series_column": f.series_column,
                        "columns": f.columns,
                    }
                    for f in FORMATS
                ],
                "user_agent": USER_AGENT,
                "endpoint_shape_is_provenance": (
                    "NSE changed the bhavcopy URL and column layout in July 2024. The "
                    "handler definitions above are themselves provenance and may need "
                    "re-capture. An HTML body where a zip is expected means the URL "
                    "shape has moved: add a handler, do not cycle headers."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument(
        "--symbols",
        default="",
        help="comma-separated NIFTY 50 constituents. Required scope limit.",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_chunks(fetched_on=datetime.now(tz=timezone.utc))
        print(f"Merged existing chunks into {CACHE}")
        return 0
    if not args.refresh:
        print("No --refresh: refusing to touch the network. Nothing to do.")
        return 0
    if not args.symbols:
        print("--symbols is required. This fetcher does not pull the full universe.")
        return 2

    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    started = datetime.now(tz=timezone.utc)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    for year in range(start.year, end.year + 1):
        if _chunk_path(year).exists():
            print(f"{year}: chunk exists, skipping", flush=True)
            continue
        y0, y1 = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        print(f"{year}: {y0} .. {y1}", flush=True)
        rows, counts = fetch_range(y0, y1, symbols)
        if rows:
            write_chunk(rows, year)
            print(f"{year}: {len(rows)} rows, handlers {counts}", flush=True)

    merge_chunks(fetched_on=started)
    print(f"Merged chunks into {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
