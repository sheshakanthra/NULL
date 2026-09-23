"""NULL audit service. BUILD.md's live-audit phase.

W0 proved the wire: a web request can run the *real* ``null audit`` engine and
get back the *real* committed answer (``POST /audit/demo``). W1 added real
input for one preset -- RSI(2). W2 made that input asynchronous: Render's
free tier times out a request at ~30s, and the real 108-variant grid audit
takes ~80s end to end, so ``POST /audit/rsi2`` no longer blocks -- it
validates, enqueues, and returns a job_id immediately, and
``GET /audit/jobs/{job_id}`` polls for the result. The job layer itself
(``service/jobs.py``) is a small, bounded, in-process queue -- see that
module's docstring for why not Celery/Redis, and for the concurrency-cap
tradeoff. W3 added the live frontend (``service/static/index.html``, served
at ``/app``). W4 deploys this to Render (``render.yaml`` at the repo root) --
see that file and ``service/README.md`` for the free-tier cold-start
reality, and this module's CORS setup below.

Nothing here reimplements audit logic: every call below is the same code
path ``null/cli.py`` uses on the command line (``build_parser`` /
``run_audit_command``), imported from ``null`` directly. The RSI(2)
backtester that turns a caller's grid into a ``run.json`` lives in
``service/backtest/rsi2.py`` and is equally strict about not feeding the
auditor garbage -- see that module's docstring.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from null.cli import InputError, build_parser, run_audit_command
from service.backtest.rsi2 import (
    MAX_GRID_VARIANTS,
    MAX_HOLDING_CAP,
    MAX_PERIOD,
    MAX_THRESHOLD,
    MIN_HOLDING_CAP,
    MIN_PERIOD,
    MIN_THRESHOLD,
    GridSpec,
    GridSpecError,
    build_grid_spec,
    run_backtest,
)
from service.jobs import JobManager, JobQueueFullError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "rsi2_nifty"
COMMITTED_VERDICT = EXAMPLE_DIR / "audit_out" / "verdict.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# The exact invocation that reproduces the committed artifact -- see
# examples/rsi2_nifty/README.md. No --benchmark: it defaults to the committed
# NIFTY 50 TRI cache under data/reference/.
DEMO_ARGV = [
    "audit",
    str(EXAMPLE_DIR / "run.json"),
    "--trials-parquet",
    str(EXAMPLE_DIR / "run.trials.parquet"),
    "--cost-robustness",
    str(EXAMPLE_DIR / "cost_robustness.csv"),
    "--sensitivity",
    str(EXAMPLE_DIR / "sensitivity.json"),
]

app = FastAPI(title="NULL audit service")

#: One process-wide job manager. See service/jobs.py's docstring for the
#: in-memory-only, single-instance tradeoff this implies.
job_manager = JobManager()

def _normalize_origin(value: str) -> str:
    """Strip surrounding whitespace and a trailing slash from a configured
    origin. A trailing slash is the easy way to configure this wrong in
    Render's dashboard -- ``Access-Control-Allow-Origin`` must match the
    browser's ``Origin`` header exactly, and browsers never send one with a
    trailing slash, so a mismatched allow-list value would silently reject
    every real request rather than erroring loudly."""
    return value.strip().rstrip("/")


# CORS: the showcase (a separate Vercel origin, per render.yaml's comment)
# only links to this service today -- a plain <a href>, which needs no CORS
# at all -- but the deploy brief asks for this set up correctly in case that
# ever becomes a cross-origin fetch. Allow-listed by exact origin, never "*":
# a wildcard would let any page on the internet drive audits from a
# visitor's browser using this service's capacity. SHOWCASE_ORIGIN unset
# (local dev, or before it's configured in Render's dashboard) means no CORS
# middleware at all -- same-origin use of /app is unaffected either way.
SHOWCASE_ORIGIN = _normalize_origin(os.environ.get("SHOWCASE_ORIGIN", ""))
if SHOWCASE_ORIGIN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[SHOWCASE_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


def _committed_evidence_hash() -> str:
    """The evidence_hash the demo audit must reproduce.

    Read fresh on every call rather than cached at import time: if the
    committed artifact and this service ever disagree, that disagreement must
    show up immediately, not after a stale in-process value survives a
    redeploy of one but not the other.
    """
    data = json.loads(COMMITTED_VERDICT.read_text(encoding="utf-8"))
    hash_value = data.get("evidence_hash")
    if not isinstance(hash_value, str):
        raise RuntimeError(
            f"{COMMITTED_VERDICT} has no evidence_hash; the committed artifact "
            "itself is broken."
        )
    return hash_value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    """The live tool is unlisted, not access-controlled: reachable by anyone
    with the URL, but not meant to be crawled, indexed, or discovered by
    search. This disallows the whole origin -- /health and /audit/* are JSON
    endpoints, not pages, so there's nothing on this service worth a partial
    allow-list. See also the <meta name="robots"> tag on /app itself; the two
    are redundant on purpose (a crawler that ignores one may still respect
    the other)."""
    return "User-agent: *\nDisallow: /\n"


@app.get("/app", response_class=HTMLResponse)
def live_app() -> str:
    """The live audit tool -- W3. A separate page from the fixed showcase
    (``site/index.html``, portfolio piece, committed-artifact-locked,
    untouched by this route): this one calls ``POST /audit/rsi2`` and
    ``GET /audit/jobs/{job_id}`` for real, on whatever grid the visitor
    submits. Read from disk on every request rather than cached in memory --
    consistent with how ``_committed_evidence_hash`` above handles the same
    tradeoff, and cheap enough for a single small HTML file.
    """
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/audit/demo")
def audit_demo() -> dict[str, Any]:
    """Run the committed rsi2_nifty example through the real audit engine.

    Writes into a throwaway temp directory (never the repo's own
    examples/rsi2_nifty/audit_out/ -- that directory is the committed record,
    not a scratch pad this endpoint is allowed to overwrite), reads the
    verdict and evidence back, and asserts the evidence_hash matches the
    committed artifact before returning anything. A mismatch is a bug in NULL
    or in this wiring, and CLAUDE.md is explicit: the service must never emit
    a verdict that disagrees with the committed artifact, so a mismatch is a
    500, not a differently-labelled 200.
    """
    expected = _committed_evidence_hash()

    with tempfile.TemporaryDirectory(prefix="null-audit-demo-") as tmp:
        out = Path(tmp)
        argv = [*DEMO_ARGV, "--out", str(out)]
        args = build_parser().parse_args(argv)
        try:
            run_audit_command(args)
        except InputError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        verdict = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
        evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))

    observed = verdict.get("evidence_hash")
    if observed != expected:
        raise HTTPException(
            status_code=500,
            detail=(
                f"audit produced evidence_hash {observed!r}, which does not match "
                f"the committed artifact's {expected!r}. Refusing to return a "
                "verdict that disagrees with the committed audit."
            ),
        )

    return {"evidence_hash": observed, "verdict": verdict, "evidence": evidence}


class Rsi2GridRequest(BaseModel):
    """The four RSI(2) grid axes. Each is a list of distinct int values --
    e.g. the committed example is ``periods=[2,3,4]``, ``entries=[5,10,15]``,
    ``exits=[50,60,70]``, ``holding_caps=[3,5,10,15]`` (108 variants).

    Only the shape is validated here (FastAPI: non-empty lists of ints). The
    actual bounds -- sane parameter bands and the grid-size cap -- are
    enforced once, in ``service.backtest.rsi2.build_grid_spec``, so there is
    exactly one place that decides what grid this service will run.
    """

    periods: list[int] = Field(min_length=1)
    entries: list[int] = Field(min_length=1)
    exits: list[int] = Field(min_length=1)
    holding_caps: list[int] = Field(min_length=1)


def _run_rsi2_audit(spec: GridSpec) -> dict[str, Any]:
    """The actual backtest-then-audit work for one RSI(2) grid, run on a
    worker thread by :data:`job_manager`. Raises plain exceptions on
    failure -- never ``HTTPException``, which is a request-layer concept
    with no meaning on a background thread -- and ``JobManager`` turns
    whatever's raised into ``str(exc)`` on the job's ``error`` field: a
    clean message, never a stack trace, to whoever polls for the result.

    Two temp directories, not one: the backtester's own output (``run.json``
    and its siblings) is itself untrusted until the audit has run on it, so
    it is kept apart from the audit's output rather than the two being
    written into the same directory and risking a name collision silently
    shadowing one artifact with another.

    ``n_trials`` is asserted equal to the caller's own grid size before
    returning -- it must never be hardcoded, inferred, or allowed to drift
    from what was actually backtested (CLAUDE.md invariant 7).
    """
    with tempfile.TemporaryDirectory(prefix="null-rsi2-") as tmp:
        base = Path(tmp)
        backtest_dir = base / "backtest"
        audit_dir = base / "audit"

        artifacts = run_backtest(spec, backtest_dir)

        argv = [
            "audit",
            str(artifacts.run_path),
            "--trials-parquet",
            str(artifacts.trials_parquet_path),
            "--sensitivity",
            str(artifacts.sensitivity_path),
            "--out",
            str(audit_dir),
        ]
        args = build_parser().parse_args(argv)
        run_audit_command(args)

        verdict = json.loads((audit_dir / "verdict.json").read_text(encoding="utf-8"))
        evidence = json.loads((audit_dir / "evidence.json").read_text(encoding="utf-8"))

    observed_n_trials = verdict.get("generated_from", {}).get("n_trials")
    if observed_n_trials != spec.n_variants:
        raise RuntimeError(
            f"audited n_trials {observed_n_trials!r} does not match the "
            f"requested grid size {spec.n_variants}. Refusing to return a "
            "verdict whose declared trial count disagrees with the grid "
            "that was actually run."
        )

    return {
        "n_trials": spec.n_variants,
        "grid": {
            "periods": list(spec.periods),
            "entries": list(spec.entries),
            "exits": list(spec.exits),
            "holding_caps": list(spec.holding_caps),
        },
        "verdict": verdict,
        "evidence": evidence,
    }


@app.post("/audit/rsi2", status_code=202)
def audit_rsi2(request: Rsi2GridRequest) -> dict[str, Any]:
    """Validate a caller-chosen RSI(2) grid and enqueue it for backtest +
    audit against the committed NIFTY 50 cache. Returns immediately --
    poll ``GET /audit/jobs/{job_id}`` for the result.

    Validation (``build_grid_spec``, InputError-shaped errors -> 422) always
    runs before enqueueing, exactly as it did when this endpoint was
    synchronous: a bad grid is rejected on the spot, never queued to fail
    later on a worker thread. A full job queue (``JobQueueFullError`` -> 429)
    is the only other way this call can fail -- everything past that point
    happens on a background thread and is reported through the job, not this
    response.
    """
    try:
        spec = build_grid_spec(
            periods=tuple(request.periods),
            entries=tuple(request.entries),
            exits=tuple(request.exits),
            holding_caps=tuple(request.holding_caps),
        )
    except GridSpecError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        job_id = job_manager.submit(lambda: _run_rsi2_audit(spec))
    except JobQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return {"job_id": job_id, "status": "queued"}


@app.get("/audit/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Poll a job's status. ``done`` includes ``n_trials``/``grid``/
    ``verdict``/``evidence``; ``error`` includes a clean message, never a
    stack trace (see :func:`_run_rsi2_audit`). A 404 covers both "never
    existed" and "existed but its TTL expired" -- indistinguishable from the
    caller's side, and neither is this endpoint's problem to explain further.
    """
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"no job {job_id!r}: unknown, or its result has expired.",
        )

    payload: dict[str, Any] = {"job_id": job.job_id, "status": job.status}
    if job.status == "done":
        assert job.result is not None
        payload.update(job.result)
    elif job.status == "error":
        payload["error"] = job.error
    return payload


@app.get("/audit/rsi2/limits")
def audit_rsi2_limits() -> dict[str, Any]:
    """The bounds `/audit/rsi2` enforces, so a caller can build a valid
    request without first triggering a 422."""
    return {
        "period": {"min": MIN_PERIOD, "max": MAX_PERIOD},
        "entry": {"min": MIN_THRESHOLD, "max": MAX_THRESHOLD},
        "exit": {"min": MIN_THRESHOLD, "max": MAX_THRESHOLD},
        "holding_cap": {"min": MIN_HOLDING_CAP, "max": MAX_HOLDING_CAP},
        "max_grid_variants": MAX_GRID_VARIANTS,
    }
