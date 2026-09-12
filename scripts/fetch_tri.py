"""Fetch NIFTY 50 TRI from NSE Indices. NETWORK STAGE — deliberately outside null/.

`null audit` must run with the network off (CLAUDE.md invariant 2), and the source
grep in tests/unit/test_source_invariants.py forbids network imports anywhere under
null/. So the fetcher lives here, is run by hand, and writes a parquet that the
offline loader in null/benchmark/tri.py reads.

    python scripts/fetch_tri.py --refresh

**This is a community-documented endpoint, not a supported API.** It can change or
block without notice — exactly what happened to the bhavcopy format in July 2024.
That is why the resulting parquet is committed to the repository rather than
fetched on demand: roughly 6,800 daily rows is small, and committing it keeps the
M7 demo reproducible by anyone who clones, including after the endpoint dies.

Politeness rules, deliberate and not configurable upward:
  * one request at a time, never parallel
  * a sleep between requests
  * at most two retries, with a long backoff
  * chunked by year so a failure costs one chunk, not the whole history
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "reference" / "nifty50_tri.parquet"
SIDECAR = REPO / "data" / "reference" / "nifty50_tri.provenance.json"

#: Total-return route. Live-captured 12-Sep-2026 and confirmed against the JS the
#: historical-data page loads (liveindexsa.niftyindices.com/assets/js/IISLComponet.js,
#: function TotalReturnindexHistoricalData). Returns a TotalReturnsIndex column.
URL = "https://niftyindices.com/BackPage/getTotalReturnIndexString"

#: THE ROUTE HAS MOVED TWICE. The shape itself is provenance -- keep the trail.
#:   1. Backpage.aspx/getTotalReturnIndexString   (original ASP.NET ScriptService)
#:   2. www.niftyindices.com/BackPage/getTotalReturnIndexString  (case + path change)
#:   3. niftyindices.com/BackPage/getTotalReturnIndexString      (host dropped "www.")
LEGACY_URL_1 = "https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
LEGACY_URL_2 = "https://www.niftyindices.com/BackPage/getTotalReturnIndexString"

#: PRICE index, NOT total return. This is the route captured from the historical-data
#: page on 12-Sep-2026; probing it showed OPEN/HIGH/LOW/CLOSE columns and a level of
#: 23398.10 on 11-Sep-2026 against the TRI's 35553.30 for the same session. It is the
#: PRI table. It is fetched here for ONE purpose: to give validate_tri_against_pri a
#: price series to check the TRI against. It must never be written to the TRI cache.
PRI_URL = "https://niftyindices.com/BackPage/getHistoricaldatatabletoString"

INDEX_NAME = "NIFTY 50"
#: The TRI series begins 30-Jun-1999; requesting January returns nothing before that.
HISTORY_START = date(1999, 1, 1)
#: Server and page both refuse a window wider than a year (JS guard dated 28-08-2025).
MAX_WINDOW_DAYS = 365

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: Seconds between requests. Not a knob to turn down.
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 15.0


def _headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://niftyindices.com",
        "Referer": "https://niftyindices.com/reports/historical-data",
    }


def _payload(start: date, end: date) -> dict[str, str]:
    fmt = "%d-%b-%Y"
    return {
        "cinfo": (
            "{'name':'"
            + INDEX_NAME
            + "','startDate':'"
            + start.strftime(fmt)
            + "','endDate':'"
            + end.strftime(fmt)
            + "','indexName':'"
            + INDEX_NAME
            + "'}"
        )
    }


def _parse(raw: bytes, content_type: str) -> list[dict[str, Any]]:
    """Decode a web-method response.

    Do NOT gate on Content-Type. Both routes answer a *successful* call with
    ``text/html; charset=utf-8`` and a JSON body -- gating on the header, as this
    script used to, rejects good data. The real discriminator is whether the body
    parses as JSON: the refusal mode is the rendered page, which does not.

    Two body shapes are accepted because the routes disagree: a bare JSON array,
    and the classic ASP.NET ``{"d": "<json string>"}`` envelope.
    """
    text = raw.decode("utf-8", "replace").strip()
    try:
        decoded = json.loads(text)
    except ValueError:
        raise RuntimeError(
            f"response is not JSON: {content_type!r}, {len(raw)} bytes. That is the "
            "PAGE RENDERING, not a rejection -- the request never reached the "
            "web method, so the route or payload format has moved again. Re-capture "
            "the live request from a browser. Do NOT cycle headers, sessions or "
            "user agents; that is not what this failure means."
        ) from None
    if isinstance(decoded, dict) and "d" in decoded:
        inner = decoded["d"]
        decoded = json.loads(inner) if isinstance(inner, str) else inner
    if not isinstance(decoded, list):
        raise RuntimeError(f"expected a JSON array of rows, got {type(decoded).__name__}")
    return list(decoded)


def fetch_window(start: date, end: date, *, url: str = URL) -> list[dict[str, Any]]:
    """One request. Raises with the response shape when the endpoint refuses."""
    if (end - start).days > MAX_WINDOW_DAYS:
        raise ValueError(
            f"{start}..{end} spans {(end - start).days} days; the endpoint refuses "
            f"more than {MAX_WINDOW_DAYS}. Chunk it."
        )
    body = json.dumps(_payload(start, end)).encode()
    request = urllib.request.Request(url, method="POST", data=body, headers=_headers())

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            time.sleep(RETRY_BACKOFF_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read()
            return _parse(raw, content_type)
        except (urllib.error.URLError, RuntimeError, KeyError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"fetch failed after {MAX_RETRIES + 1} attempts: {last_error}")


def fetch_history(start: date, end: date, *, url: str = URL) -> list[dict[str, Any]]:
    """Year-by-year, sequentially. A failure costs one chunk, not the history."""
    rows: list[dict[str, Any]] = []
    year = start.year
    while year <= end.year:
        window_start = max(start, date(year, 1, 1))
        window_end = min(end, date(year, 12, 31))
        print(f"  {window_start} .. {window_end}", flush=True)
        rows.extend(fetch_window(window_start, window_end, url=url))
        time.sleep(REQUEST_DELAY_SECONDS)
        year += 1
    return rows


def to_tri_frame(rows: list[dict[str, Any]]) -> "Any":
    """TRI rows -> a sorted date/tri frame. Refuses anything lacking the TRI column."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    if "TotalReturnsIndex" not in frame.columns:
        raise RuntimeError(
            "response has no TotalReturnsIndex column, so it is NOT a total-return "
            f"series. Columns: {sorted(frame.columns)}. A response carrying "
            "OPEN/HIGH/LOW/CLOSE is the PRICE index from the historical-data table "
            "route -- writing it to the TRI cache would hand every strategy roughly "
            "1.35%/yr of free alpha. Refusing."
        )
    frame = frame.rename(columns={"Date": "date", "TotalReturnsIndex": "tri"})
    frame["date"] = pd.to_datetime(frame["date"], format="%d %b %Y")
    frame["tri"] = frame["tri"].astype(float)
    frame = frame[["date", "tri"]].drop_duplicates(subset="date")
    return frame.sort_values("date").reset_index(drop=True)


def to_pri_frame(rows: list[dict[str, Any]]) -> "Any":
    """Price rows -> a sorted date/pri frame. For validation only, never cached."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    if "CLOSE" not in frame.columns:
        raise RuntimeError(
            f"price response has no CLOSE column. Columns: {sorted(frame.columns)}"
        )
    frame = frame.rename(columns={"HistoricalDate": "date", "CLOSE": "pri"})
    frame["date"] = pd.to_datetime(frame["date"], format="%d %b %Y")
    frame["pri"] = frame["pri"].astype(float)
    frame = frame[["date", "pri"]].drop_duplicates(subset="date")
    return frame.sort_values("date").reset_index(drop=True)


def write_cache(
    rows: list[dict[str, Any]],
    *,
    fetched_on: datetime,
    validation: dict[str, Any],
) -> None:
    frame = to_tri_frame(rows)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE, index=False)

    SIDECAR.write_text(
        json.dumps(
            {
                "source": URL,
                "method": "POST",
                "index": INDEX_NAME,
                "payload_shape": _payload(HISTORY_START, date.today()),
                "response_columns": sorted(rows[0].keys()) if rows else [],
                "user_agent": USER_AGENT,
                "session": (
                    "None required. A cold POST answers in full; probing on "
                    "12-Sep-2026 showed a bootstrap GET of the historical-data page "
                    "sets no cookies at all and changes nothing about the response. "
                    "This is unlike the NSE corporate-actions API, which does need "
                    "a session cookie."
                ),
                "fetched_on": fetched_on.date().isoformat(),
                "rows": int(len(frame)),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
                "series_starts_late": (
                    "Requested from 1999-01-01 but the endpoint's TRI history begins "
                    "1999-06-30. The earlier months are absent upstream, not dropped."
                ),
                "tri_validation": validation,
                "route_history": {
                    "note": (
                        "The route has moved twice; the shape is itself provenance."
                    ),
                    "1_original": LEGACY_URL_1,
                    "2_superseded": LEGACY_URL_2,
                    "3_current": URL,
                },
                "price_route_is_not_this_one": (
                    f"{PRI_URL} is the historical-data TABLE route and returns the "
                    "PRICE index (OPEN/HIGH/LOW/CLOSE). It is NOT total return. It is "
                    "used by this script only to supply the price series that "
                    "validate_tri_against_pri checks the TRI against."
                ),
                "note": (
                    "Community-documented endpoint, not a supported API. It can change "
                    "or block without notice. This parquet is committed so the demo "
                    "stays reproducible if it does."
                ),
                "endpoint_shape_is_provenance": (
                    "The URL, method and cinfo payload format recorded here are "
                    "themselves provenance and may need re-capture. Do NOT gate on "
                    "Content-Type: a SUCCESSFUL call answers with text/html and a "
                    "JSON body. The refusal mode is a large body that does not parse "
                    "as JSON -- that is the PAGE RENDERING, meaning the request never "
                    "reached the web method, so the path or payload format has moved. "
                    "That is a signal to re-capture the live request from a browser, "
                    "NOT to try more headers, sessions or user agents."
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="actually hit the network. Without it this does nothing.",
    )
    parser.add_argument("--start", default=HISTORY_START.isoformat())
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()

    if not args.refresh:
        print("No --refresh: refusing to touch the network. Nothing to do.")
        return 0

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    started = datetime.now(tz=timezone.utc)

    print(f"Fetching {INDEX_NAME} TRI, {args.start} .. {args.end}")
    rows = fetch_history(start, end)

    print(f"\nFetching {INDEX_NAME} PRICE index for validation, same window")
    pri_rows = fetch_history(start, end, url=PRI_URL)

    # The validation is a gate, not a report: a series that cannot be shown to be a
    # total-return series is never written. See CLAUDE.md invariant 6, default REJECT.
    from null.benchmark.tri import validate_tri_against_pri
    from null.contracts import IST, Series

    import pandas as pd

    tri_frame, pri_frame = to_tri_frame(rows), to_pri_frame(pri_rows)
    merged = tri_frame.merge(pri_frame, on="date", how="inner")
    print(
        f"\nAligned {len(merged)} sessions "
        f"({merged['date'].min().date()} .. {merged['date'].max().date()})"
    )

    def _series(frame: "pd.DataFrame", column: str) -> Series:
        stamps = pd.to_datetime(frame["date"]).dt.tz_localize(IST)
        return Series(
            ts=tuple(t.to_pydatetime() for t in stamps),
            values=tuple(float(v) for v in frame[column]),
        )

    result = validate_tri_against_pri(
        _series(merged, "tri"), _series(merged, "pri")
    )
    print(f"\nTRI validation: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"  windows checked   : {result.windows_checked}")
    print(f"  windows violating : {result.windows_violating}")
    print(f"  TRI annualised    : {result.tri_annualised:.4%}")
    print(f"  PRI annualised    : {result.pri_annualised:.4%}")
    print(f"  observed gap      : {result.observed_gap:.4%}")
    print(f"  {result.rationale}")

    if not result.is_valid:
        print(
            "\nREFUSING TO WRITE THE CACHE. The fetched series did not validate as a "
            "total-return index. Committing it would mislabel the benchmark.",
        )
        return 1

    write_cache(rows, fetched_on=started, validation=result.model_dump(mode="json"))
    print(f"\nWrote {len(tri_frame)} rows to {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
