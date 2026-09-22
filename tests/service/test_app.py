"""W0 acceptance test: the service must reproduce the committed audit exactly.

BUILD.md's live-audit-service W0: ``POST /audit/demo`` runs the real,
committed ``examples/rsi2_nifty`` example through the real ``null audit``
engine and must return the same verdict already committed at
``examples/rsi2_nifty/audit_out/verdict.json`` -- same ``evidence_hash``,
same REJECT, same four failing gates. A service that returns anything else
is wiring the web layer to something other than NULL's own judgement.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from service.app import app

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_VERDICT = (
    REPO_ROOT / "examples" / "rsi2_nifty" / "audit_out" / "verdict.json"
)

# Pinned per the phase spec: "REJECT, 4 gates fail, same evidence_hash
# 3f40aa2e…". A literal pin here catches the committed artifact itself
# drifting silently, on top of the dynamic comparison below.
PINNED_HASH_PREFIX = "3f40aa2e"

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_audit_demo_reproduces_the_committed_verdict() -> None:
    committed = json.loads(COMMITTED_VERDICT.read_text(encoding="utf-8"))
    expected_hash = committed["evidence_hash"]
    assert expected_hash.startswith(PINNED_HASH_PREFIX)

    response = client.post("/audit/demo")
    assert response.status_code == 200

    body = response.json()
    assert body["evidence_hash"] == expected_hash
    assert body["verdict"]["evidence_hash"] == expected_hash
    assert body["verdict"]["result"] == "REJECT"

    failed = {g["name"] for g in body["verdict"]["gates"] if not g["passed"]}
    assert failed == {
        "beats_benchmark_net",
        "deflated_sharpe",
        "reality_check",
        "capacity",
    }
    assert len(failed) == 4

    passed = {g["name"] for g in body["verdict"]["gates"] if g["passed"]}
    assert passed == {"leakage_clean", "walkforward_consistency", "sensitivity_plateau"}

    # evidence.json's own recorded hash must agree too -- the service returns
    # both files, and they describe the same audit.
    assert body["evidence"]["evidence_hash"] == expected_hash


# ---------------------------------------------------------------------------
# W1: POST /audit/rsi2 -- real input through a bounded backtester. THE
# acceptance test that matters, per the phase brief: given the exact grid
# examples/rsi2_nifty/build_run.py used, the backtester must reproduce a
# run.json close enough that auditing it yields the SAME verdict as the
# committed one. If it can't reproduce the known result, it isn't trustworthy
# on any new parameters. This is slow (the real 108-variant grid against the
# full 50-name universe, same cost as examples/rsi2_nifty/build_run.py) --
# that cost is the point; a fast test here would not be evidence of anything.
# ---------------------------------------------------------------------------

COMMITTED_GRID = {
    "periods": [2, 3, 4],
    "entries": [5, 10, 15],
    "exits": [50, 60, 70],
    "holding_caps": [3, 5, 10, 15],
}


def test_audit_rsi2_reproduces_the_committed_verdict_for_the_committed_grid() -> None:
    committed = json.loads(COMMITTED_VERDICT.read_text(encoding="utf-8"))
    expected_hash = committed["evidence_hash"]

    response = client.post("/audit/rsi2", json=COMMITTED_GRID)
    assert response.status_code == 200

    body = response.json()
    assert body["n_trials"] == 108
    assert body["grid"] == COMMITTED_GRID

    assert body["verdict"]["evidence_hash"] == expected_hash
    assert body["verdict"]["result"] == "REJECT"
    assert body["verdict"]["generated_from"]["n_trials"] == 108

    failed = {g["name"] for g in body["verdict"]["gates"] if not g["passed"]}
    assert failed == {
        "beats_benchmark_net",
        "deflated_sharpe",
        "reality_check",
        "capacity",
    }
    passed = {g["name"] for g in body["verdict"]["gates"] if g["passed"]}
    assert passed == {"leakage_clean", "walkforward_consistency", "sensitivity_plateau"}

    assert body["evidence"]["evidence_hash"] == expected_hash
    assert body["evidence"]["deflated_sharpe"]["n_trials"] == 108


def test_audit_rsi2_rejects_a_grid_over_the_variant_cap() -> None:
    response = client.post(
        "/audit/rsi2",
        json={
            "periods": [2, 3, 4, 5, 6],
            "entries": [5, 10, 15, 20, 25],
            "exits": [50, 60, 70, 80, 90],
            "holding_caps": [3, 5],
        },
    )
    assert response.status_code == 422
    assert "cap" in response.json()["detail"]


def test_audit_rsi2_rejects_a_period_outside_the_sane_band() -> None:
    response = client.post(
        "/audit/rsi2",
        json={"periods": [500], "entries": [5], "exits": [50], "holding_caps": [3]},
    )
    assert response.status_code == 422
    assert "band" in response.json()["detail"]


def test_audit_rsi2_rejects_an_empty_axis() -> None:
    response = client.post(
        "/audit/rsi2",
        json={"periods": [], "entries": [5], "exits": [50], "holding_caps": [3]},
    )
    # FastAPI's own Field(min_length=1) rejects this before it reaches
    # build_grid_spec -- still a 422, just from a different validator.
    assert response.status_code == 422


def test_audit_rsi2_limits_reports_the_enforced_bounds() -> None:
    response = client.get("/audit/rsi2/limits")
    assert response.status_code == 200
    body = response.json()
    assert body["max_grid_variants"] == 200
    assert body["period"]["min"] == 2
