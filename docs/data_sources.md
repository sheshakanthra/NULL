# STATUS: TRI route located and corrected; still not fetched from this environment

**Decision (Sheshakanth):** NIFTY 50 TRI comes from NSE Indices directly. The
ETF-NAV proxy option is dropped. The price index is not a fallback and is
unreachable in code, not merely deprecated.

## Update — the route had moved, found by reading the page's own JS

The original endpoint, `POST niftyindices.com/Backpage.aspx/getTotalReturnIndexString`,
returned HTTP 200 with `Content-Type: text/html` and 93,810 bytes on every attempt
across two sessions -- the page, not the web-method response. Four attempts (bare
POST, cookie-bootstrapped session, full browser headers with a valid
`ASP.NET_SessionId`, and a repeat of the cookie-bootstrap pattern that worked for
NSE's corporate-actions API) all produced the identical byte count, which is the
signature of a request that never reaches the ScriptService method at all.

Rather than try a fifth header variation, the historical-data page's own scripts
were read directly. `niftyindices.com/reports/historical-data` no longer contains
the strings `Backpage` or `getTotalReturnIndexString` anywhere -- the page has been
rebuilt onto a separate asset host, `liveindexsa.niftyindices.com`. Its
`IISLComponet.js` contains the live call:

```
url: "/BackPage/getTotalReturnIndexString"
```

Case changed (`Backpage.aspx` to `BackPage`, no `.aspx`) and the path segment
changed. The `cinfo` payload shape -- `{'name', 'startDate', 'endDate', 'indexName'}`
-- is byte-for-byte unchanged.

**Two shaped attempts against the corrected route, as instructed, no header
cycling:**

| # | request | result |
|---|---|---|
| 1 | Cookie bootstrap from `/reports/historical-data`, then POST to the corrected `www.niftyindices.com/BackPage/...` | Bootstrap sets **zero cookies** (`nseindia.com`'s corporate-actions bootstrap sets four: `nsit`, `AKA_A2`, `_abck`, `bm_sz`). With no session to carry, the cookie-bootstrap pattern that worked for corporate actions does not apply here -- there is nothing to carry. POST returned the same 93,810-byte page. |
| 2 | Direct POST to the corrected route, both hosts | `www.niftyindices.com/BackPage/...`: **200, 1,144 bytes** -- a materially different response from the 93,810-byte page, so the route correction changed something real. Reading the body on retry failed twice, first with a socket read timeout and then with `ConnectionResetError [WinError 10054]` -- the remote host is tearing the connection down before the body can be read a second time. `liveindexsa.niftyindices.com/BackPage/...`: **405, method not supported** -- that host's route doesn't accept POST at all. |

The 1,144-byte response on the `www` host is the most promising signal so far: it
is not the page, and it is a small enough body to plausibly be a JSON error or a
redirect stub rather than HTML. It has not been read, because reading it a second
time is what triggered the reset, and a third attempt would be cycling rather than
a new shaped attempt.

## Fallback checked: NSE Indices' downloadable index files

Per instruction, checked whether niftyindices exposes the same series as a static
download before concluding. `nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv`
is reachable and returns real data (confirmed: 200, `text/csv`, 108 index rows for
04-Jan-2024) -- but its "Nifty 50" row reads **21,658.6**, which is the **price
index** level for that date, not TRI (a TRI series compounding since a 1996 base
would sit far higher). No TRI-named variant appears among the 108 indices in that
file. **This fallback does not carry the series NULL needs.**

## Consequence

No parquet is committed. `null audit` has no benchmark series and the loader
raises rather than falling back, by design. `scripts/fetch_tri.py` is updated to
the corrected URL (`BackPage/getTotalReturnIndexString`, no `.aspx`), with the
superseded route kept in the file as `LEGACY_URL` for the record. Nothing has been
fabricated to fill the gap.

## To unblock

The 1,144-byte body is the next thing to read, from an environment where the
connection does not reset -- ordinary desktop browser DevTools, watching the
Network tab on the historical-data page, would show both the exact request the
browser sends and the response body directly. Compare that request against
`scripts/fetch_tri.py` line by line; whatever differs is the missing piece.

## What is built and tested

`scripts/fetch_tri.py` -- network stage, outside `null/` because invariant 2 forbids
network imports there. Sequential year-by-year chunks, 2s between requests, at most
two retries with 15s backoff, never parallel. Writes the parquet plus a provenance
sidecar recording the URL, method, exact payload shape, user agent, fetch date, row
count and date range.

`null/benchmark/tri.py` -- offline loader. Reads the committed parquet, imports
nothing that can reach a network (asserted by test), and **raises** when the cache
is absent rather than falling back. The exception message carries the 1.35%/yr
figure so the reason is unavoidable.

`validate_tri_against_pri` -- two independent checks, because either alone passes
the wrong series. Monotonic dominance over multi-year windows, since dividends are
never negative; and magnitude, since a series can dominate by 0.1%/yr or 6%/yr and
still not be NIFTY 50 TRI. Tested against a correct TRI, a price index passed off as
TRI, a series dominating by too much, one dominating by too little, and one with an
occasional violation.

## The figure that goes on every report

NSE reports **11.09%** annualised for the NIFTY 50 price index against **12.44%**
for total return over the 20 years to February 2026. The **1.35%/yr** gap is
dividends. Benchmarking against the price index hands a strategy every basis point
of it, which is why there is no fallback. If a caller supplies a non-TRI benchmark,
the limitations band on the report carries these numbers.

---

*The original investigation follows.*

# Data sources — open questions and what has actually been tested

Status as of M2 start. **The NIFTY 50 TRI question is NOT settled.** Nothing in
`null/benchmark/` should be written against a benchmark series until it is.

## Why this matters more than it looks

BUILD.md §4 rule 1: the benchmark is NIFTY 50 **TRI**, not the price index. The
price index excludes dividends, worth roughly 1.2–1.5%/yr. Benchmarking against
it hands every strategy that much free alpha — which is precisely the bias M2
exists to remove. Falling back to the price index "for now" would make NULL
commit the error it was built to catch.

## What was tested, and what happened

Target: `POST https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString`
— the official NSE Indices endpoint behind the public historical-data page. It is
documented to return daily TRI values back to 1999, unauthenticated.

| Attempt | Result |
|---|---|
| Bare POST, browser User-Agent | HTTP 200, `Content-Type: text/html`, 93,428 bytes — the page, not JSON |
| GET bootstrap first, then POST with cookie jar | HTTP 200, still `text/html`; bootstrap set no cookies |
| Full browser header set (`sec-ch-ua`, `Sec-Fetch-*`, `Origin`, `Accept-Language`), valid `ASP.NET_SessionId` from bootstrap | HTTP 200, still `text/html`, 93,810 bytes |

Control: `POST` with a JSON body to `httpbin.org/post` and `postman-echo.com/post`
both returned HTTP 200 with the body correctly echoed. **The sandbox is not the
problem.** NSE is serving the page instead of the web-method response for clients
that do not present a full browser fingerprint. Akamai sits in front of the site.

Conclusion: the endpoint is real and free, but not reachable from a plain HTTP
client. It needs either a real browser engine or a human with a browser.

## Options, with tradeoffs

None of these has been chosen. This is a decision for Sheshakanth.

### A. Manual one-time download, cached to parquet
Download the TRI series from the niftyindices historical-data page by hand, drop
the CSV in `data/`, and let NULL cache it to parquet.

- **For:** official series, correct values, zero scraping, no ToU problem for
  personal research use, unblocks M2 today. Fits the offline-first cache design
  exactly — the fetch stage is already meant to be separate and cached.
- **Against:** manual refresh whenever the series needs extending. Not automated.
- **Effort:** minutes.

### B. Headless browser (Playwright) driving the official page
Drive a real Chromium, let the page issue its own AJAX call, capture the response.

- **For:** official series, fully automated, extends without human involvement.
- **Against:** ~150MB browser dependency, slow, and fragile — it breaks whenever
  NSE changes the page. Must live strictly in the fetch stage, never importable
  from `null/` (invariant 2). niftyindices terms restrict redistribution, so the
  series must stay in gitignored `data/` and never be committed.
- **Effort:** half a day, plus ongoing maintenance.

### C. NIFTYBEES ETF adjusted close as a TRI proxy
- **For:** trivially available via yfinance.
- **Against:** **this reintroduces the bias in miniature.** The ETF charges an
  expense ratio and has tracking error, so its total return sits systematically
  *below* true index TRI — understating the benchmark and handing the strategy
  free alpha, the same direction of error as using the price index, just smaller
  (tens of bps rather than ~130). On top of that, BUILD.md §3 and §5 both warn
  that yfinance adjusted closes have documented errors on Indian tickers. Not
  acceptable without printing the bias and its direction on every report.
- **Effort:** an hour.

### D. Reconstruct TRI from the price index plus dividend yield
- **For:** no scraping; the price index is everywhere.
- **Against:** an approximation, not the series. Doing it properly needs
  point-in-time constituent dividends and ex-dates; using an aggregate published
  yield smears the timing and leaves an error plausibly in the 10–30bps/yr range,
  which is the same order as the alpha being tested for. Would need its own
  validation against a known-good TRI — which is the thing we do not have.
- **Effort:** a day, and it needs the answer to be able to check itself.

### E. Licensed data
NSE data vendor licence, broker feed, or a terminal.

- **For:** clean, supportable, redistributable within licence terms.
- **Against:** costs money and needs an account.

## Whichever is chosen

Per CLAUDE.md ("Known hard problem — do not hand-wave") and BUILD.md §5, the
choice and its bias direction must be printed on **every** report as a stated
limitation, alongside the cost-config `rates_are_verified` flag from M1.

## Also unresolved, carried from M1

`configs/costs_india_equity.yaml` has `_verified_on: "UNVERIFIED"`. The charge
rates have never been reconciled against a live broker charge list.
`CostConfig.rates_are_verified` returns `False` and must appear on every report
until someone does that reconciliation.
