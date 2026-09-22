"""NULL audit service. BUILD.md's live-audit phase.

W0 proved the wire: a web request can run the *real* ``null audit`` engine and
get back the *real* committed answer (``POST /audit/demo``). W1 adds real
input for one preset -- RSI(2) -- without touching that guarantee. Nothing
here reimplements audit logic: every call below is the same code path
``null/cli.py`` uses on the command line (``build_parser`` /
``run_audit_command``), imported from ``null`` directly. The RSI(2)
backtester that turns a caller's grid into a ``run.json`` lives in
``service/backtest/rsi2.py`` and is equally strict about not feeding the
auditor garbage -- see that module's docstring.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
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
    GridSpecError,
    build_grid_spec,
    run_backtest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "rsi2_nifty"
COMMITTED_VERDICT = EXAMPLE_DIR / "audit_out" / "verdict.json"

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


@app.post("/audit/rsi2")
def audit_rsi2(request: Rsi2GridRequest) -> dict[str, Any]:
    """Backtest a caller-chosen RSI(2) grid against the committed NIFTY 50
    cache and audit the result with the real engine.

    Two temp directories, not one: the backtester's own output (``run.json``
    and its siblings) is itself untrusted until the audit has run on it, so
    it is kept apart from the audit's output rather than the two being
    written into the same directory and risking a name collision silently
    shadowing one artifact with another.

    ``n_trials`` is asserted equal to the caller's own grid size before
    returning -- it must never be hardcoded, inferred, or allowed to drift
    from what was actually backtested (CLAUDE.md invariant 7).
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

    with tempfile.TemporaryDirectory(prefix="null-rsi2-") as tmp:
        base = Path(tmp)
        backtest_dir = base / "backtest"
        audit_dir = base / "audit"

        try:
            artifacts = run_backtest(spec, backtest_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
        try:
            run_audit_command(args)
        except InputError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        verdict = json.loads((audit_dir / "verdict.json").read_text(encoding="utf-8"))
        evidence = json.loads((audit_dir / "evidence.json").read_text(encoding="utf-8"))

    observed_n_trials = verdict.get("generated_from", {}).get("n_trials")
    if observed_n_trials != spec.n_variants:
        raise HTTPException(
            status_code=500,
            detail=(
                f"audited n_trials {observed_n_trials!r} does not match the "
                f"requested grid size {spec.n_variants}. Refusing to return a "
                "verdict whose declared trial count disagrees with the grid "
                "that was actually run."
            ),
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
