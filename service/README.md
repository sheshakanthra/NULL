# NULL audit service

A FastAPI wrapper around the real `null audit` engine. `service/` imports
`null/` directly and never reimplements audit logic. W0 proved the wire
(`POST /audit/demo`, the committed example, no user input). W1 added real
input for one preset -- RSI(2) (`POST /audit/rsi2`) -- via the bounded
backtester in `service/backtest/rsi2.py`. W2 made that endpoint
asynchronous: submit -> poll -> result, run by the small in-process job
queue in `service/jobs.py`. W3 adds a live frontend for it --
`service/static/index.html`, served at `GET /app` -- reusing the showcase's
terminal visual language against a separate page that runs real audits.
Presets only; free-form strategy description is a later phase.

## Run locally

From the repo root:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e . -r service/requirements.txt
uvicorn service.app:app --reload --app-dir .
```

Both commands assume the repo root as the working directory. `-e .`
installs this repo itself as a package (`service/app.py` does `import null`
directly); it's a separate, explicit pip argument rather than a path written
inside `service/requirements.txt`, because that used to be `-e ..` inside
the file and it broke the Render build -- see `service/requirements.txt`'s
own comment and the Deploy section below for the full story. `--app-dir .`
matters separately: `service/app.py` locates the committed example via a
path relative to its own file, and the plain `uvicorn` console script
(unlike `python -m uvicorn`) doesn't put the current directory on
`sys.path` by default.

## Deploy (Render) -- W4

`render.yaml` at the repo root is a Render Blueprint: one free-tier web
service, `pip install -e . -r service/requirements.txt` as the build
command (same command as local dev, above -- both need the repo root
installed the same explicit way), `uvicorn service.app:app --host 0.0.0.0
--port $PORT --app-dir .` as the start (Render assigns `$PORT`; the service
must bind it, not a hardcoded port), `/health` as the health-check path.

**This step needs a Render account and cannot be done from inside this
repo or by an agent working in it** -- connecting a GitHub repo to Render is
an account-level action taken in Render's own dashboard:

1. In the Render dashboard: **New +** -> **Blueprint**, pick this repo
   (`sheshakanthra/NULL`) and branch (`main`). Render reads `render.yaml`
   and provisions the `null-audit-service` web service from it.
2. Once deployed, note the service's URL (`https://<name>.onrender.com` --
   the exact name depends on availability at creation time).
3. In the service's **Environment** tab, set `SHOWCASE_ORIGIN` to the
   showcase's real production origin (e.g. `https://<something>.vercel.app`,
   no trailing slash) -- `render.yaml` marks it `sync: false` deliberately,
   so it's set once in the dashboard rather than committed as a guess. Unset,
   the service still runs fine; only cross-origin `fetch` calls *from* the
   showcase would need it (see the CORS note below -- nothing calls it that
   way today).
4. Every push to `main` redeploys automatically, same as the showcase's
   Vercel project already does for `site/`.

**Free-tier reality, stated plainly because it will be the first thing a
visitor notices:** Render's free web services spin down after a period of
inactivity and cold-start on the next request, which can take 30-60s on top
of however long the request itself takes. A first-hit `/audit/rsi2` can
therefore take cold-start-plus-~80s before its job even starts running. The
live page's `checkWarmup()` (fires on load, pings `/health`) shows an amber
hint -- *"This is a free-tier instance -- it looks like it was asleep..."* --
when the health check itself is slow, so the wait reads as an explained cold
start rather than a broken page. This is an accepted tradeoff for a free
demo service, not something W5+ needs to fix.

**CORS.** `service/app.py` adds `CORSMiddleware` allow-listing exactly
`SHOWCASE_ORIGIN` (never `"*"` -- a wildcard would let any page on the
internet drive audits from a visitor's browser using this instance's
capacity) when that env var is set, and adds no CORS headers at all when
it's unset. Nothing in the current design actually needs this: the showcase
only links to `/app` (a plain `<a href>`, a full-page navigation, not a
`fetch`), and `/app` calls its own same-origin API. It's wired up per the
deploy brief so it's correct the moment anything cross-origin is added.

**Unlisted, not access-controlled.** The live tool (`/app` and everything
under it) is reachable by anyone with the direct URL but is not meant to be
found by search: `GET /robots.txt` disallows the whole origin, and `/app`
itself carries `<meta name="robots" content="noindex,nofollow">`. This is
discoverability, not authentication -- there is no login, and none is
planned for a free-tier demo. The showcase links to it, but only from its
footer, deliberately not from hero-level visible content -- the showcase is
the public portfolio piece; the live tool is the by-invitation demo behind
it.

## The live page

`http://127.0.0.1:8000/app` -- a strategy picker (RSI(2), the four grid
axes, pre-filled with the committed grid), submits to `POST /audit/rsi2`,
polls `GET /audit/jobs/{job_id}` every 2.5s (5-minute ceiling), and renders
the real result: queued/running state with an elapsed timer and all seven
gate names shown (no fake per-gate progress -- the backend only reports
overall job status, so the page doesn't pretend otherwise), then on done the
real gates (name, PASS/FAIL/NOT_COMPUTABLE, and the actual
`GateResult.rationale` sentence -- not a hand-tuned summary, since the grid
here isn't fixed), the REJECT/PASS stamp, the `deflated_sharpe` rationale in
the showcase's paper-coloured sentence box, and the full `evidence_hash`. A
429 (capacity) or a job `error` renders its own clean state, never a stack
trace. No framework -- vanilla JS, `fetch` + `setInterval`, same as
`site/index.html`.

It is a **separate page** from the fixed showcase (`site/index.html`) --
that one is a portfolio piece, its numbers locked to the committed artifact,
and this route never touches it. A nav link points from the live page back
to the showcase, visibly (that direction is safe); the showcase links back
only from its footer (see "Unlisted, not access-controlled" above). Both
links need the real production URLs, wired once this service is deployed
and the showcase's Vercel URL is known -- see the Deploy section above.

Verified with a real browser (Playwright, headless Chromium) against the
local backend: submitted the default (committed) grid, watched it through
queued -> running (elapsed timer, all seven gates shown RUNNING) -> done,
and confirmed the rendered verdict -- REJECT, the same four failing gates,
the real `deflated_sharpe` rationale, and the full `baff7b68...` hash --
byte-for-byte matches the committed artifact. Separately filled the
service's real capacity (three live submissions) and confirmed the page
renders the 429 "AT CAPACITY" state correctly against a genuinely busy
instance. No uncaught JS exceptions in either run (Chromium's console does
log a "Failed to load resource: 429" line for the capacity case -- that's
the browser's own network-panel logging of any non-2xx `fetch` response, not
an application error; the page's own error handling ran and rendered
correctly).

## Endpoints

- `GET /health` -- `{"status": "ok"}`
- `GET /app` -- the live audit page described above.
- `POST /audit/demo` -- runs `examples/rsi2_nifty/run.json` (plus its
  trials-parquet, cost-robustness sweep, and sensitivity surface) through the
  real `null audit` engine and returns `{"evidence_hash", "verdict",
  "evidence"}`. Before responding, it asserts the resulting `evidence_hash`
  matches the one committed at `examples/rsi2_nifty/audit_out/verdict.json`
  (`baff7b68...`). If it doesn't match, the endpoint returns a 500 rather than
  a verdict -- CLAUDE.md's invariant: this service must never emit a verdict
  that disagrees with the committed artifact.
- `POST /audit/rsi2` -- takes an RSI(2) grid (`periods`, `entries`, `exits`,
  `holding_caps`, each a list of ints). Validates it synchronously (sane
  per-parameter bands, capped at 200 variants -- a bad or oversized grid is a
  422, no job ever created) and enqueues a job; returns **202**
  `{"job_id", "status": "queued"}` immediately. It never runs the backtest
  inline: the real 108-variant grid takes ~80s end to end, well past
  Render's free-tier ~30s request timeout, so the work happens on a
  background thread and the response comes back at once.
- `GET /audit/jobs/{job_id}` -- poll for the result. `{"status":
  "queued"|"running"|"done"|"error", ...}`; `done` merges in `{"n_trials",
  "grid", "verdict", "evidence"}`, `error` adds a clean `"error"` message
  (never a stack trace). A 404 covers both "no such job" and "result
  expired" -- see the TTL note below.
- `GET /audit/rsi2/limits` -- the bounds `/audit/rsi2` enforces.

```
curl -X POST http://127.0.0.1:8000/audit/demo
curl -X POST http://127.0.0.1:8000/audit/rsi2 \
  -H "Content-Type: application/json" \
  -d '{"periods":[2,3,4],"entries":[5,10,15],"exits":[50,60,70],"holding_caps":[3,5,10,15]}'
# -> {"job_id": "...", "status": "queued"}
curl http://127.0.0.1:8000/audit/jobs/<job_id>
# -> poll until "status" is "done" or "error"
curl http://127.0.0.1:8000/health
```

Expect `"result": "REJECT"` with four failing gates (`beats_benchmark_net`,
`deflated_sharpe`, `reality_check`, `capacity`) -- the same verdict already on
disk at `examples/rsi2_nifty/audit_out/verdict.json`. `/audit/rsi2` with the
grid above reproduces that exact artifact; a different grid audits real,
different evidence and may reach a different verdict.

### The job layer (`service/jobs.py`)

In-process only -- no Celery, no Redis, no broker. Render's free tier is one
instance, and this is a demo service; a job queue that needs infrastructure
to run is infrastructure this project doesn't have anywhere to put.
**Tradeoff, stated rather than discovered in production: jobs live in this
process's memory only.** A restart -- a deploy, a crash, the free tier's own
idle-sleep -- loses every queued or in-flight job without a trace. Fine for
"submit it again"; not fine for anything that must not lose a request.

**Each job runs in a separate OS process, not a thread in this one.**
`JobManager`'s supervisor threads (below) submit the actual work to a
`ProcessPoolExecutor` and block on the result -- see the module's own
docstring, and `docs/findings.md` #9, for why: a CPU-bound, GIL-holding
computation on a thread in this process can starve this same process's
ability to answer `/health`, and on a resource-constrained free-tier
instance that's not theoretical -- it's the confirmed root cause of a real
production incident (below).

- At most `MAX_CONCURRENT_JOBS` (1) audits run at once -- picked
  conservatively for a free-tier instance with limited, likely shared,
  vCPU; more processes than cores buys queuing fairness, not real
  parallelism.
- A `MAX_QUEUE_SIZE` (2) wait queue sits in front of the worker(s). Total
  accepted capacity is `MAX_CONCURRENT_JOBS + MAX_QUEUE_SIZE` = 3 jobs; a
  submission past that is a **429**, immediately, not a growing backlog.
  This is the real security surface: a burst of requests can't melt a
  single free-tier instance, because the instance refuses to accept more
  work than it owns the capacity to run.
- Completed (`done`/`error`) jobs are evicted from the job table after
  `JOB_TTL_SECONDS` (30 minutes) so memory doesn't grow without bound,
  measured from whichever is later: when the job finished, or when it was
  last successfully fetched. A queued or running job is never a candidate
  for eviction, however long it runs relative to the TTL -- and a client
  that keeps polling a finished job resets that job's clock on every
  successful fetch, so it can't age out from under a poll gap shorter than
  the TTL.
- Submitted work must be a plain, module-level, picklable function plus
  picklable arguments -- never a lambda or a closure -- since it's pickled
  across the process boundary to actually run. See
  `service.backtest.rsi2.run_rsi2_audit` (what `POST /audit/rsi2` submits)
  and `tests/service/_support.py` (what the tests submit) for the pattern.

**Incident 1, Render free-tier production.** A submitted audit ran (gates
showed RUNNING, elapsed climbing) and then the client got 404 --
`GET /audit/jobs/{id}` reported the job gone while it was still supposed to
be running. The eviction logic above was never the cause (a running job's
`finished_at` is `None` for its entire run and was never a candidate --
`test_only_finished_jobs_are_evicted_by_ttl` already proved that, before
and after). Leading hypothesis at the time: the audit's
~20-million-iteration pure-Python cost loop ran on a *thread in this same
process*, plausibly denying the whole process -- including the thread
meant to answer `/health` -- enough wall-clock time on the free tier's
fraction of a CPU that Render's health check read the instance as
unresponsive and restarted it. Fixed by running jobs in a separate process
(above), so this process's own responsiveness never depends on the same
GIL an audit is spinning on --
`test_health_stays_responsive_while_a_job_is_running` in
`tests/service/test_app.py` pins exactly this. Full account in
`docs/findings.md` #9. A real, worthwhile fix on its own merits -- but
deployed, the very next production run OOM'd anyway.

**Incident 2, same deploy, one minute later -- the actual cause.** Render's
own event log named it precisely: "Ran out of memory (used over 512MB)."
Not the GIL: the worker process's own peak memory, measured directly
afterward at **1845MB**, over 3.5x the container's limit before the web
process's own ~163MB is even added. Root cause: `run_grid` (`examples/
rsi2_nifty/strategy.py`) held every one of 108 `VariantResult`s' full
weight-change list simultaneously -- for a high-turnover strategy,
thousands of `TargetWeight` objects per variant, times 108 -- when only the
*best* variant's weights are ever read by anything. Fixed by not carrying
weights for the other 107 (`run_variant`'s `include_weights` flag), cutting
peak RSS to **876MB** -- verified, not estimated (`psutil`, sampled every
50ms on a real `ProcessPoolExecutor` worker). `service/jobs.py`'s worker
pool also gained `max_tasks_per_child=1`, so nothing a job allocates and
doesn't clean up survives into the next job's baseline. Full account,
including a hoisting fix that helped speed but *not* memory (a useful
negative result), and the honest remaining gap against the original 350MB
target, in `docs/findings.md` #11.
`tests/examples/test_memory_budget.py` is the regression guard: fails if
the committed grid's real worker-process peak RSS regresses past 1100MB.

The earlier client-side mitigations (10-minute poll ceiling, tolerating a
few consecutive 404s, the TTL grace window) all stay -- good defence in
depth for a slow-but-not-crashed job -- but they were never the fix for
either incident; the process boundary and the memory fixes are.

### Before/after (the committed 108-variant grid)

| | time | worker peak RSS |
|---|---|---|
| Original (thread, scalar cost loop) | ~74-90s unloaded, ~200s under heavier load on the same machine, no code change in between -- the timing problem was never purely about Render, this machine's own variance made that visible | **1845MB** (measured after the fact, from the incident) |
| + process isolation | no meaningful runtime change (its effect is *survivability* under GIL contention, not speed or memory) | not separately measured -- superseded by the memory fixes below before a clean baseline was taken |
| + vectorised cost loop | **49.0s**, byte-identical `run.json` / `run.trials.parquet` / `sensitivity.json`, hash unchanged (`baff7b68...`) | unchanged by vectorisation alone |
| + weights-retention fix | no change | **876MB** (52% reduction) |
| + panel/timeline hoisting | faster (fewer redundant matrix rebuilds); not separately re-timed after the weights fix | ~880MB -- no measurable change; this fix was never about memory (see `docs/findings.md` #11's negative result) |

Production (Render, live URL) timing and memory, once verified against the
deployed instance, are in the session report.

## Tests

```
pytest tests/service/
```
