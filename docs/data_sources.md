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
