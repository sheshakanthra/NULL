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

URL = "https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
INDEX_NAME = "NIFTY 50"
HISTORY_START = date(1999, 1, 1)

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
        "Origin": "https://www.niftyindices.com",
        "Referer": "https://www.niftyindices.com/reports/historical-data",
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


def fetch_window(start: date, end: date) -> list[dict[str, Any]]:
    """One request. Raises with the response shape when the endpoint refuses."""
    body = json.dumps(_payload(start, end)).encode()
    request = urllib.request.Request(URL, method="POST", data=body, headers=_headers())

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            time.sleep(RETRY_BACKOFF_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read()
            if "application/json" not in content_type:
                raise RuntimeError(
                    f"expected JSON, got {content_type!r} and {len(raw)} bytes. The "
                    "endpoint served the HTML page instead of the web-method "
                    "response, which is how its bot protection refuses a client. "
                    "This is not a transient error and retrying will not fix it."
                )
            return list(json.loads(json.loads(raw.decode())["d"]))
        except (urllib.error.URLError, RuntimeError, KeyError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"fetch failed after {MAX_RETRIES + 1} attempts: {last_error}")


def fetch_history(start: date, end: date) -> list[dict[str, Any]]:
    """Year-by-year, sequentially. A failure costs one chunk, not the history."""
    rows: list[dict[str, Any]] = []
    year = start.year
    while year <= end.year:
        window_start = max(start, date(year, 1, 1))
        window_end = min(end, date(year, 12, 31))
        print(f"  {window_start} .. {window_end}", flush=True)
        rows.extend(fetch_window(window_start, window_end))
        time.sleep(REQUEST_DELAY_SECONDS)
        year += 1
    return rows


def write_cache(rows: list[dict[str, Any]], *, fetched_on: datetime) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "Date": "date",
            "TotalReturnsIndex": "tri",
            "Index Name": "index_name",
            "NTR_Value": "ntr",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%d %b %Y")
    frame["tri"] = frame["tri"].astype(float)
    frame = frame[["date", "tri"]].sort_values("date").reset_index(drop=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE, index=False)

    SIDECAR.write_text(
        json.dumps(
            {
                "source": URL,
                "method": "POST",
                "index": INDEX_NAME,
                "payload_shape": _payload(HISTORY_START, date.today()),
                "user_agent": USER_AGENT,
                "fetched_on": fetched_on.date().isoformat(),
                "rows": int(len(frame)),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
                "note": (
                    "Community-documented endpoint, not a supported API. It can change "
                    "or block without notice. This parquet is committed so the demo "
                    "stays reproducible if it does."
                ),
                "endpoint_shape_is_provenance": (
                    "The URL, method and cinfo payload format recorded here are "
                    "themselves provenance and may need re-capture. A response of "
                    "HTTP 200 with Content-Type text/html and a large body is the "
                    "PAGE RENDERING, not a rejection -- it means the request never "
                    "reached the ScriptService method, so the endpoint path or payload "
                    "format has moved. That is a signal to re-capture the live request "
                    "from a browser, NOT to try more headers, sessions or user agents."
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

    started = datetime.now(tz=timezone.utc)
    print(f"Fetching {INDEX_NAME} TRI, {args.start} .. {args.end}")
    rows = fetch_history(date.fromisoformat(args.start), date.fromisoformat(args.end))
    write_cache(rows, fetched_on=started)
    print(f"Wrote {len(rows)} rows to {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
