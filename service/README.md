# NULL audit service

A FastAPI wrapper around the real `null audit` engine. `service/` imports
`null/` directly and never reimplements audit logic. W0 proved the wire
(`POST /audit/demo`, the committed example, no user input). W1 adds real
input for one preset -- RSI(2) (`POST /audit/rsi2`) -- via the bounded
backtester in `service/backtest/rsi2.py`. Presets only; free-form strategy
description is a later phase.

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
  `holding_caps`, each a list of ints), backtests it against the committed
  NIFTY 50 cache via `service/backtest/rsi2.py`, and audits the result with
  the real engine. Grid is validated first (sane per-parameter bands, capped
  at 200 variants) -- a bad or oversized grid is a 422, not a slow backtest.
  Returns `{"n_trials", "grid", "verdict", "evidence"}`.
- `GET /audit/rsi2/limits` -- the bounds `/audit/rsi2` enforces.

```
curl -X POST http://127.0.0.1:8000/audit/demo
curl -X POST http://127.0.0.1:8000/audit/rsi2 \
  -H "Content-Type: application/json" \
  -d '{"periods":[2,3,4],"entries":[5,10,15],"exits":[50,60,70],"holding_caps":[3,5,10,15]}'
curl http://127.0.0.1:8000/health
```

Expect `"result": "REJECT"` with four failing gates (`beats_benchmark_net`,
`deflated_sharpe`, `reality_check`, `capacity`) -- the same verdict already on
disk at `examples/rsi2_nifty/audit_out/verdict.json`. `/audit/rsi2` with the
grid above reproduces that exact artifact; a different grid audits real,
different evidence and may reach a different verdict.

The full 108-variant grid against the 50-name universe takes real wall-clock
time end to end (roughly a minute in this environment) -- fine for now, but
relevant when a later phase picks a request-timeout / async-job strategy for
deployment.

## Tests

```
pytest tests/service/
```
