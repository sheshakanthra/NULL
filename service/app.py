"""NULL audit service -- W0 skeleton. BUILD.md's live-audit phase, W0.

Proves the wire: a web request can run the *real* ``null audit`` engine and get
back the *real* committed answer. Nothing here reimplements audit logic --
every call below is the same code path ``null/cli.py`` uses on the command
line (``build_parser`` / ``run_audit_command``), imported from ``null``
directly. This module only orchestrates: build an argv, run the CLI command
into a scratch directory, read back what it wrote, and refuse to answer if
that disagrees with the artifact already committed under
``examples/rsi2_nifty/audit_out/``.

No user input yet (BUILD.md W0 scope). ``POST /audit/demo`` takes nothing and
always audits the one committed example.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from null.cli import InputError, build_parser, run_audit_command

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
