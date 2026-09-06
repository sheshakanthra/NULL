"""Fetch NSE corporate actions. NETWORK STAGE — deliberately outside null/.

    python scripts/fetch_corporate_actions.py --refresh

This is the ground truth that replaces ratio inference. NSE publishes every
corporate action with its symbol, series, ex-date, record date and a subject line
naming the action and its ratio. A move on a date matching a real announced action
is adjusted by the announced ratio, full stop -- no tolerance, no ambiguity.

Unlike point-in-time index membership, this HAS a clean authoritative source: the
exchange itself publishes it.

Fifteen requests, one per year, filtered locally to the NIFTY 50 constituents. The
API accepts a one-year window and reaches back past 2013, so chunking finer would
be needless load.

The endpoint needs a session cookie from the corporate-actions page before it will
answer -- a plain request returns an empty body. That bootstrap is the same shape
as any browser's first visit.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "reference" / "corporate_actions.parquet"
SIDECAR = REPO / "data" / "reference" / "corporate_actions.provenance.json"

PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
API = "https://www.nseindia.com/api/corporates-corporateActions"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
REQUEST_DELAY_SECONDS = 2.5
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 20.0


def _opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", USER_AGENT),
        ("Accept-Language", "en-US,en;q=0.9"),
    ]
    opener.open(PAGE, timeout=45)  # bootstrap: the API needs the session cookie
    return opener


def fetch_year(opener: urllib.request.OpenerDirector, year: int) -> list[dict[str, Any]]:
    url = f"{API}?index=equities&from_date=01-01-{year}&to_date=31-12-{year}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": PAGE,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    last: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            time.sleep(RETRY_BACKOFF_SECONDS)
        try:
            with opener.open(request, timeout=90) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read()
            if "json" not in content_type:
                raise RuntimeError(
                    f"expected JSON, got {content_type!r} and {len(raw)} bytes. The "
                    "session cookie was probably rejected; re-capture the request "
                    "shape rather than cycling headers."
                )
            payload = json.loads(raw.decode())
            return payload if isinstance(payload, list) else payload.get("data", [])
        except (urllib.error.URLError, RuntimeError, ValueError) as exc:
            last = exc
    raise RuntimeError(f"{year}: failed after {MAX_RETRIES + 1} attempts: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--symbols", default="")
    args = parser.parse_args()

    if not args.refresh:
        print("No --refresh: refusing to touch the network. Nothing to do.")
        return 0
    if not args.symbols:
        print("--symbols is required; this does not keep the whole market.")
        return 2

    import pandas as pd

    wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    started = datetime.now(tz=timezone.utc)
    opener = _opener()

    rows: list[dict[str, Any]] = []
    for year in range(args.start_year, args.end_year + 1):
        found = fetch_year(opener, year)
        kept = [r for r in found if str(r.get("symbol", "")).strip().upper() in wanted]
        rows.extend(kept)
        print(f"  {year}: {len(found)} actions, {len(kept)} for the universe", flush=True)
        time.sleep(REQUEST_DELAY_SECONDS)

    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "symbol": "symbol", "series": "series", "exDate": "ex_date",
            "recDate": "record_date", "subject": "subject", "comp": "company",
        }
    )
    keep = [c for c in ("symbol", "series", "ex_date", "record_date", "subject",
                        "company") if c in frame.columns]
    frame = frame[keep].drop_duplicates()
    frame["ex_date"] = pd.to_datetime(frame["ex_date"], format="%d-%b-%Y",
                                      errors="coerce")
    frame = frame.dropna(subset=["ex_date"]).sort_values(["symbol", "ex_date"])
    frame = frame.reset_index(drop=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE, index=False)

    SIDECAR.write_text(
        json.dumps(
            {
                "source": API,
                "method": "GET, after a session-cookie bootstrap from the page",
                "page": PAGE,
                "query_shape": "?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY",
                "scope": f"NIFTY 50 constituents, {args.start_year}-{args.end_year}",
                "fetched_on": started.date().isoformat(),
                "rows": int(len(frame)),
                "symbols": int(frame["symbol"].nunique()),
                "endpoint_shape_is_provenance": (
                    "The API needs a session cookie from the corporate-actions page. "
                    "A non-JSON body means the bootstrap was rejected or the endpoint "
                    "moved -- re-capture the live request, do not cycle headers."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(frame)} actions to {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
