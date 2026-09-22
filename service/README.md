# NULL audit service -- W0

A FastAPI wrapper around the real, committed `null audit` engine. This is the
**W0 skeleton**: it proves the web layer can run the actual audit and get the
actual committed answer back. There is no user input yet, no strategy
submission, and no backtester of its own -- `service/` imports `null/`
directly and never reimplements audit logic. See the live-audit phase notes
for what comes after W0.

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
  (`3f40aa2e...`). If it doesn't match, the endpoint returns a 500 rather than
  a verdict -- CLAUDE.md's invariant: this service must never emit a verdict
  that disagrees with the committed artifact.

```
curl -X POST http://127.0.0.1:8000/audit/demo
curl http://127.0.0.1:8000/health
```

Expect `"result": "REJECT"` with four failing gates (`beats_benchmark_net`,
`deflated_sharpe`, `reality_check`, `capacity`) -- the same verdict already on
disk at `examples/rsi2_nifty/audit_out/verdict.json`.

## Tests

```
pytest tests/service/
```
