# NULL audit service

A FastAPI wrapper around the real `null audit` engine. `service/` imports
`null/` directly and never reimplements audit logic. W0 proved the wire
(`POST /audit/demo`, the committed example, no user input). W1 added real
input for one preset -- RSI(2) (`POST /audit/rsi2`) -- via the bounded
backtester in `service/backtest/rsi2.py`. W2 made that endpoint
asynchronous: submit -> poll -> result, run by the small in-process job
queue in `service/jobs.py`. Presets only; free-form strategy description is
a later phase.

## Run locally

From the repo root:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r service/requirements.txt
uvicorn service.app:app --reload --app-dir .
```

`--app-dir .` (repo root) matters: `service/app.py` locates the committed
example via a path relative to its own file, and the `-e ..` install in
`requirements.txt` assumes it's run from `service/` (so run the `pip install`
from the repo root as shown, where `service/requirements.txt`'s `..` resolves
to the repo root itself).

## Endpoints

- `GET /health` -- `{"status": "ok"}`
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

- At most `MAX_CONCURRENT_JOBS` (1) audits run at once -- each pins a CPU
  core for ~80s of mostly pure-Python cost arithmetic that doesn't release
  the GIL, so more workers than that buys queuing fairness, not real
  parallelism, on a free-tier instance's limited vCPU.
- A `MAX_QUEUE_SIZE` (2) wait queue sits in front of the worker(s). Total
  accepted capacity is `MAX_CONCURRENT_JOBS + MAX_QUEUE_SIZE` = 3 jobs; a
  submission past that is a **429**, immediately, not a growing backlog.
  This is the real security surface: a burst of requests can't melt a
  single free-tier instance, because the instance refuses to accept more
  work than it owns the capacity to run.
- Completed (`done`/`error`) jobs are evicted from the job table after
  `JOB_TTL_SECONDS` (30 minutes) so memory doesn't grow without bound. A
  queued or running job is never evicted out from under itself.

## Tests

```
pytest tests/service/
```
